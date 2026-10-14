import json
from typing import Any
import lancedb
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

model_name = "Qwen/Qwen3-Embedding-0.6B"
print(f"Loading embedding model: {model_name}...")

encoder = SentenceTransformer(model_name, trust_remote_code=True)
encoder.max_seq_length = 1024  # Token context window. Has the same effects as #trainable params.

# Configuration constants
MIN_TEXT_LENGTH = 200  # Filter by minimum characters per LanceDB entry
MAX_GROUP_SIZE = 10000  # Max size for grouped sections before splitting
MAX_PAGE_GAP = 2  # Max page gap for grouping #/body items across chunks


@dataclass
class Item:
    """Item object for fine-grained citation within a section."""
    label: str
    char_range: tuple[int, int] = (10 ** 9, -1)  # Character range in text (inclusive)
    page_range: tuple[int, int] = (10 ** 9, -1)  # Pages the text covers
    picture_ref: str = ""  # "" if not associated with any image


@dataclass
class ChunkItem:
    """Represents a doc_item with its source chunk reference for cross-chunk grouping."""
    parent_ref: str
    label: str
    text_parts: list[str]  # Extracted text from each prov
    char_ranges: list[tuple[int, int]]  # Character ranges within concatenated text
    page_range: tuple[int, int]
    page_ranges: list[tuple[int, int]]  # Individual page ranges from each prov
    picture_ref: str
    chunk_idx: int  # Source chunk index for tracking
    item_idx: int   # Original item index within chunk


@dataclass
class GroupedSection:
    """Represents a grouped collection of items sharing the same parent."""
    parent_ref: str
    heading: str
    text: str
    items: list[ChunkItem]
    page_range: tuple[int, int]
    char_range: tuple[int, int]
    chunk_indices: list[int]  # Which chunks contributed to this section


def get_heading_key(headings: list[str]) -> tuple:
    """Create a hashable key from headings for comparison."""
    return tuple(headings) if headings else ()


def can_merge_body_sections(
    prev_headings: list[str],
    prev_end_page: int,
    curr_headings: list[str],
    curr_start_page: int
) -> bool:
    """
    Determine if two #/body sections can be merged based on:
    1. Same heading path (same headings list)
    2. Page ranges are continuous or nearly so (gap <= MAX_PAGE_GAP)
    """
    # Check if headings match
    if get_heading_key(prev_headings) != get_heading_key(curr_headings):
        return False

    # Check page gap
    page_gap = curr_start_page - prev_end_page
    return page_gap <= MAX_PAGE_GAP


def group_doc_items_by_parent_multichunk(chunks_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Groups doc_items by their parent reference across multiple chunks, with special
    handling for #/body items.

    This approach:
    - Groups #/body items across chunks ONLY if they have the same headings and
      page ranges are continuous (gap <= MAX_PAGE_GAP)
    - Groups ALL other parent refs (#/groups/*, #/pictures/*) across ALL chunks
    - Splits large groups at chunk boundaries when exceeding MAX_GROUP_SIZE
    - Maintains original text from PDF

    Returns a list of grouped sections, each containing:
    - text: The heading and concatenated child texts (for embedding)
    - heading: The section heading (from chunk meta)
    - page_range: Combined page range of all items in group
    - char_range: Combined character range of all items in group
    - items: List of individual ChunkItem objects for fine-grained citation
    - parent_ref: The parent reference this group represents
    - chunk_indices: Which chunks contributed to this section
    """
    # First pass: collect all items by parent_ref, handling #/body specially
    # Structure: parent_ref -> list of (chunk_idx, item_list, headings, chunk_text)
    parent_to_chunks: dict[str, list[tuple[int, list, list[str], str]]] = {}

    for chunk_idx, chunk in enumerate(chunks_data):
        doc_items = chunk.get("meta", {}).get("doc_items", [])
        headings = chunk.get("meta", {}).get("headings", [])
        chunk_text = chunk.get("text", "")

        for item in doc_items:
            parent_ref = item["parent"]["$ref"]
            if parent_ref not in parent_to_chunks:
                parent_to_chunks[parent_ref] = []
            parent_to_chunks[parent_ref].append((chunk_idx, item, headings, chunk_text))

    grouped_sections = []

    for parent_ref, chunk_items in parent_to_chunks.items():
        if parent_ref == "#/body":
            # Special handling for #/body: only merge if headings match and pages are continuous
            sections = _group_body_items_by_headings(chunk_items)
            grouped_sections.extend(sections)
        else:
            # For all other parent refs, group across all chunks
            sections = _group_non_body_items(parent_ref, chunk_items)
            grouped_sections.extend(sections)

    return grouped_sections


def _group_body_items_by_headings(
    chunk_items: list[tuple[int, dict, list[str], str]]
) -> list[dict[str, Any]]:
    """
    Group #/body items by headings, merging only when headings match and pages are continuous.

    Deduplicates text extraction: when multiple items from the same chunk have overlapping
    charspans, we only include each character position once to avoid duplication.
    """
    # Sort by chunk index to maintain order
    chunk_items_sorted = sorted(chunk_items, key=lambda x: x[0])

    # Group by heading key
    heading_groups: dict[tuple, list[tuple[int, dict, list[str], str]]] = {}
    for chunk_idx, item, headings, chunk_text in chunk_items_sorted:
        heading_key = get_heading_key(headings)
        if heading_key not in heading_groups:
            heading_groups[heading_key] = []
        heading_groups[heading_key].append((chunk_idx, item, headings, chunk_text))

    # Merge consecutive groups within same heading if pages are continuous
    merged_sections = []

    for heading_key, items in heading_groups.items():
        if not items:
            continue

        current_section_items: list[ChunkItem] = []
        current_concat_text = ""
        current_full_char_range = (10 ** 9, -1)
        current_full_page_range = (10 ** 9, -1)
        current_char_offset = 0
        current_chunk_indices: list[int] = []
        current_page_ranges: list[tuple[int, int]] = []

        prev_end_page = None
        prev_headings = None

        # Track which char positions have been used for each chunk
        used_char_ranges: dict[int, list[tuple[int, int]]] = {}

        for chunk_idx, item, headings, chunk_text in items:
            # Check if we can merge with previous section
            can_merge = False
            if prev_headings is not None:
                can_merge = can_merge_body_sections(
                    prev_headings, prev_end_page, headings, item["prov"][0]["page_no"]
                )

            if not can_merge and current_section_items:
                # Finalize current section and start new one
                section = _finalize_section(
                    parent_ref="#/body",
                    heading=prev_headings[-1] if prev_headings else "",
                    items=current_section_items,
                    text=current_concat_text,
                    char_range=current_full_char_range,
                    page_range=current_full_page_range,
                    chunk_indices=current_chunk_indices
                )
                if section and len(section["text"]) >= MIN_TEXT_LENGTH:
                    merged_sections.append(section)

                # Reset for new section
                current_section_items = []
                current_concat_text = ""
                current_full_char_range = (10 ** 9, -1)
                current_full_page_range = (10 ** 9, -1)
                current_char_offset = 0
                current_chunk_indices = []
                current_page_ranges = []
                used_char_ranges = {}

            # Track charspan usage for this chunk to deduplicate
            if chunk_idx not in used_char_ranges:
                used_char_ranges[chunk_idx] = []

            # Check if this item overlaps with any already-used charspans
            # If so, skip the entire item to avoid partial extraction
            item_overlaps = False
            for prov in item["prov"]:
                prov_start = prov["charspan"][0]
                prov_end = prov["charspan"][1]
                for used_start, used_end in used_char_ranges[chunk_idx]:
                    # Check for ANY overlap
                    if not (prov_end <= used_start or prov_start >= used_end):
                        item_overlaps = True
                        break
                if item_overlaps:
                    break

            # Skip entire item if it overlaps with already-captured text
            if item_overlaps:
                continue

            # Process this item - extract all its provs
            new_item = ChunkItem(
                parent_ref="#/body",
                label=item["label"],
                text_parts=[],
                char_ranges=[],
                page_range=(10 ** 9, -1),
                page_ranges=[],
                picture_ref="",
                chunk_idx=chunk_idx,
                item_idx=chunk_idx
            )

            item_text_parts = []
            for prov in item["prov"]:
                prov_start = prov["charspan"][0]
                prov_end = prov["charspan"][1]

                # Check if this prov has already been used (fully or partially)
                is_duplicate = False
                for used_start, used_end in used_char_ranges[chunk_idx]:
                    # Check for overlap
                    if not (prov_end <= used_start or prov_start >= used_end):
                        is_duplicate = True
                        break

                if is_duplicate:
                    continue

                # Mark this charspan as used
                used_char_ranges[chunk_idx].append((prov_start, prov_end))

                char_start = current_char_offset + prov_start
                char_end = current_char_offset + prov_end

                current_full_char_range = (
                    min(current_full_char_range[0], char_start),
                    max(current_full_char_range[1], char_end)
                )
                current_full_page_range = (
                    min(current_full_page_range[0], prov["page_no"]),
                    max(current_full_page_range[1], prov["page_no"])
                )

                item_text = chunk_text[prov_start:prov_end]
                item_text_parts.append(item_text)

                new_item.char_ranges.append((char_start, char_end))
                new_item.page_ranges.append((prov["page_no"], prov["page_no"]))

                current_char_offset += prov_end - prov_start + 1

            new_item.text_parts = item_text_parts
            new_item.page_range = current_full_page_range

            if new_item.char_ranges:  # Only add if we have valid text parts
                current_section_items.append(new_item)
                current_chunk_indices.append(chunk_idx)

            if item_text_parts:
                grouped_text = "\n".join(item_text_parts)
                if current_concat_text:
                    current_concat_text += "\n\n" + grouped_text
                else:
                    current_concat_text = grouped_text

            prev_end_page = item["prov"][-1]["page_no"]
            prev_headings = headings

        # Finalize last section
        if current_section_items:
            section = _finalize_section(
                parent_ref="#/body",
                heading=prev_headings[-1] if prev_headings else "",
                items=current_section_items,
                text=current_concat_text,
                char_range=current_full_char_range,
                page_range=current_full_page_range,
                chunk_indices=current_chunk_indices
            )
            if section and len(section["text"]) >= MIN_TEXT_LENGTH:
                merged_sections.append(section)

    return merged_sections


def _group_non_body_items(
    parent_ref: str,
    chunk_items: list[tuple[int, dict, list[str], str]]
) -> list[dict[str, Any]]:
    """
    Group non-#/body items across all chunks, splitting if group exceeds MAX_GROUP_SIZE.

    Deduplicates text extraction: when multiple items from the same chunk have overlapping
    charspans, we only include each character position once to avoid duplication.
    """
    # Sort by chunk index to maintain order
    chunk_items_sorted = sorted(chunk_items, key=lambda x: x[0])

    all_sections = []
    current_section_items: list[ChunkItem] = []
    current_concat_text = ""
    current_full_char_range = (10 ** 9, -1)
    current_full_page_range = (10 ** 9, -1)
    current_char_offset = 0
    current_chunk_indices: list[int] = []

    # Track which char positions have been used for each chunk
    used_char_ranges: dict[int, list[tuple[int, int]]] = {}

    for chunk_idx, item, headings, chunk_text in chunk_items_sorted:
        # Initialize used char ranges for this chunk if needed
        if chunk_idx not in used_char_ranges:
            used_char_ranges[chunk_idx] = []

        # Check if this item overlaps with any already-used charspans
        # If so, skip the entire item to avoid partial extraction
        item_overlaps = False
        for prov in item["prov"]:
            prov_start = prov["charspan"][0]
            prov_end = prov["charspan"][1]
            for used_start, used_end in used_char_ranges[chunk_idx]:
                # Check for ANY overlap
                if not (prov_end <= used_start or prov_start >= used_end):
                    item_overlaps = True
                    break
            if item_overlaps:
                break

        # Skip entire item if it overlaps with already-captured text
        if item_overlaps:
            continue

        new_item = ChunkItem(
            parent_ref=parent_ref,
            label=item["label"],
            text_parts=[],
            char_ranges=[],
            page_range=(10 ** 9, -1),
            page_ranges=[],
            picture_ref="",
            chunk_idx=chunk_idx,
            item_idx=chunk_idx
        )

        # Extract picture reference if associated with a picture
        if parent_ref.startswith("#/pictures/"):
            pic_idx = parent_ref.split("/")[-1]
            new_item.picture_ref = f"pic_{pic_idx}.png"

        # Initialize used char ranges for this chunk if needed
        if chunk_idx not in used_char_ranges:
            used_char_ranges[chunk_idx] = []

        item_text_parts = []
        for prov in item["prov"]:
            prov_start = prov["charspan"][0]
            prov_end = prov["charspan"][1]

            # Mark this charspan as used
            used_char_ranges[chunk_idx].append((prov_start, prov_end))

            char_start = current_char_offset + prov_start
            char_end = current_char_offset + prov_end

            current_full_char_range = (
                min(current_full_char_range[0], char_start),
                max(current_full_char_range[1], char_end)
            )
            current_full_page_range = (
                min(current_full_page_range[0], prov["page_no"]),
                max(current_full_page_range[1], prov["page_no"])
            )

            item_text = chunk_text[prov_start:prov_end]
            item_text_parts.append(item_text)

            new_item.char_ranges.append((char_start, char_end))
            new_item.page_ranges.append((prov["page_no"], prov["page_no"]))

            current_char_offset += prov_end - prov_start + 1

        new_item.text_parts = item_text_parts
        new_item.page_range = current_full_page_range

        if new_item.char_ranges:  # Only add if we have valid text parts
            current_section_items.append(new_item)
            current_chunk_indices.append(chunk_idx)

        if item_text_parts:
            grouped_text = "\n".join(item_text_parts)
            if current_concat_text:
                current_concat_text += "\n\n" + grouped_text
            else:
                current_concat_text = grouped_text

        # Check if we need to split
        if len(current_concat_text) >= MAX_GROUP_SIZE and current_section_items:
            # Determine heading from items
            heading = ""
            if current_section_items:
                # Use the most common heading or first one
                heading = headings[-1] if headings else ""

            section = _finalize_section(
                parent_ref=parent_ref,
                heading=heading,
                items=current_section_items,
                text=current_concat_text,
                char_range=current_full_char_range,
                page_range=current_full_page_range,
                chunk_indices=current_chunk_indices
            )
            if section:
                all_sections.append(section)

            # Reset for new section
            current_section_items = []
            current_concat_text = ""
            current_full_char_range = (10 ** 9, -1)
            current_full_page_range = (10 ** 9, -1)
            current_char_offset = 0
            current_chunk_indices = []
            used_char_ranges = {}

    # Finalize remaining items
    if current_section_items:
        heading = ""
        if current_section_items:
            heading = headings[-1] if headings else ""

        section = _finalize_section(
            parent_ref=parent_ref,
            heading=heading,
            items=current_section_items,
            text=current_concat_text,
            char_range=current_full_char_range,
            page_range=current_full_page_range,
            chunk_indices=current_chunk_indices
        )
        if section:
            all_sections.append(section)

    return all_sections


def _finalize_section(
    parent_ref: str,
    heading: str,
    items: list[ChunkItem],
    text: str,
    char_range: tuple[int, int],
    page_range: tuple[int, int],
    chunk_indices: list[int]
) -> dict[str, Any] | None:
    """
    Finalize a grouped section, adjusting char_ranges for heading prefix.
    Returns None if text is too short.
    """
    if not text or len(text) < MIN_TEXT_LENGTH:
        return None

    # Prepend heading to the text for embedding context
    heading_prefix = f"{heading}\n\n" if heading else ""
    text_with_heading = heading_prefix + text

    # Adjust char_ranges to account for the heading prefix
    heading_len = len(heading_prefix)
    adjusted_items = []
    for item in items:
        adjusted_char_ranges = [
            [int(start + heading_len), int(end + heading_len)]
            for start, end in item.char_ranges
        ]
        adjusted_page_ranges = [
            [int(p[0]), int(p[1])]
            for p in item.page_ranges
        ]

        adjusted_item = {
            "parent_ref": item.parent_ref,
            "label": item.label,
            "text_parts": item.text_parts,
            "char_ranges": adjusted_char_ranges,
            "page_range": [int(item.page_range[0]), int(item.page_range[1])],
            "page_ranges": adjusted_page_ranges,
            "picture_ref": item.picture_ref,
            "chunk_idx": item.chunk_idx,
            "item_idx": item.item_idx
        }
        adjusted_items.append(adjusted_item)

    return {
        "parent_ref": parent_ref,
        "heading": heading,
        "text": text_with_heading,
        "items": adjusted_items,
        "page_range": page_range,
        "char_range": char_range,
        "chunk_indices": chunk_indices
    }


def enforce_minimum_length(groups: list[dict[str, Any]], min_length: int = MIN_TEXT_LENGTH) -> list[dict[str, Any]]:
    """
    Enforces minimum text length per group. Groups that are too small are filtered out.
    """
    return [g for g in groups if len(g["text"]) >= min_length]


def parseChunk(raw_chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Parse a raw chunk and return a list of grouped sections.
    Each section includes heading, concatenated text, and individual items for citation.

    NOTE: This function now calls the multi-chunk grouping function which processes
    ALL chunks together. When used in isolation (single chunk), it processes only that chunk.
    """
    # For backwards compatibility with single-chunk usage
    # If you want multi-chunk behavior, use group_doc_items_by_parent_multichunk directly
    # This function wraps the multi-chunk function for single-chunk usage
    groups = group_doc_items_by_parent_multichunk([raw_chunk])

    # Enforce minimum text length
    valid_groups = enforce_minimum_length(groups)

    return valid_groups


def populate_lancedb_from_jsonl(jsonl_path: str, db_uri: str, table_name: str):
    """
    Reads structured .JSONL chunks, generates embeddings,
    and inserts them into LanceDB with an FTS index for hybrid search.

    Uses multi-chunk grouping to combine related doc_items across chunks.
    """
    # Read chunks from the .JSONL file
    chunks_data: list[dict] = []
    print(f"Reading chunks from {jsonl_path}...")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks_data.append(json.loads(line))

    if not chunks_data:
        print("No chunks found in the provided JSONL file.")
        return

    # Process chunks using multi-chunk grouping
    print(f"Processing {len(chunks_data)} chunks with multi-chunk grouping...")
    all_groups = group_doc_items_by_parent_multichunk(chunks_data)

    # Enforce minimum text length
    valid_groups = enforce_minimum_length(all_groups, MIN_TEXT_LENGTH)

    print(f"Generated {len(valid_groups)} grouped sections from {len(chunks_data)} chunks.")

    # Extract text content (with heading) and generate dense vector embeddings
    texts = [g["text"] for g in valid_groups]
    print(f"Generating embeddings for {len(texts)} grouped sections...")
    embeddings = encoder.encode(texts, show_progress_bar=True)

    # Map groups and embeddings into records matching your schema
    records = []
    for i, group in enumerate(valid_groups):
        record = {
            "vector": embeddings[i].tolist(),
            "text": group["text"],
            "heading": group["heading"],
            "page_range": group["page_range"],
            "char_range": group["char_range"],
            "items": group["items"],
            "parent_ref": group["parent_ref"],
            "chunk_indices": group["chunk_indices"]
        }
        records.append(record)

    # Connect to local LanceDB and create/overwrite the table
    db = lancedb.connect(db_uri)
    print(f"Writing records to LanceDB table '{table_name}' at {db_uri}...")
    table = db.create_table(table_name, data=records, mode="overwrite")

    # Create a Full-Text Search (FTS) index on the text column
    print("Creating Full-Text Search (FTS) index on the 'text' column...")
    table.create_fts_index(field_names="text", replace=True)

    print("LanceDB successfully populated, indexed, and ready for hybrid queries!")


if __name__ == "__main__":
    populate_lancedb_from_jsonl(
        jsonl_path="lancedb_chunks/Skiena-The_Algorithm_Design_Manual/raw.jsonl",
        db_uri="./lancedb_data",
        table_name="textbook_chunks"
    )
