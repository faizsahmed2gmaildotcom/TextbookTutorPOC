import json, os
from typing import Any
import lancedb
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, asdict
from logging_config import get_logger

logger = get_logger(__name__)

model_name = "Qwen/Qwen3-Embedding-0.6B"
print(f"Loading embedding model: {model_name}...")
encoder = SentenceTransformer(model_name, trust_remote_code=True)
encoder.max_seq_length = 1024


@dataclass
class Item:
    label: str
    char_range: tuple[int, int] = (10 ** 9, -1)
    page_range: tuple[int, int] = (10 ** 9, -1)
    picture_ref: str = ""


def parseChunk(raw_chunk: dict[str, Any]) -> dict[str, Any]:
    import re

    parsed: dict[str, Any] = {
        "text": raw_chunk["text"],
        "items": []
    }

    char_offset = 0
    doc_items = raw_chunk["meta"]["doc_items"]
    for raw_item in doc_items:
        new_item = Item(raw_item["label"])

        if re.fullmatch(r"#/pictures/[0-9]*", raw_item["parent"]["$ref"]):
            new_item.picture_ref = f"pic_{raw_item['parent']['$ref'].split('/')[-1]}.png"

        for prov in raw_item["prov"]:
            new_item.char_range = (
                min(new_item.char_range[0], char_offset + prov["charspan"][0]),
                max(new_item.char_range[1], char_offset + prov["charspan"][1])
            )
            new_item.page_range = (
                min(new_item.page_range[0], prov["page_no"]),
                max(new_item.page_range[1], prov["page_no"])
            )
        char_offset += raw_item["prov"][-1]["charspan"][1] + 1

        parsed["items"].append(asdict(new_item))

    return parsed


def get_chunk_page_range(parsed_chunk: dict) -> tuple[int, int]:
    """Extract the min and max page numbers from a parsed chunk's items."""
    min_page = 10 ** 9
    max_page = -1
    for item in parsed_chunk.get("items", []):
        pr = item.get("page_range", (10 ** 9, -1))
        min_page = min(min_page, pr[0])
        max_page = max(max_page, pr[1])
    return (min_page, max_page) if min_page <= max_page else (1, 1)


def enrich_chunk_with_toc(parsed_chunk: dict, toc_structure: dict) -> dict[str, str | int | None]:
    """
    Determine which chapter/section a chunk belongs to based on its page range.

    Returns a dict with: chapter_id, chapter_name, section_id, section_name, page_start, page_end
    """
    page_start, page_end = get_chunk_page_range(parsed_chunk)

    enrichment = {
        "chapter_id": None,
        "chapter_name": None,
        "section_id": None,
        "section_name": None,
        "page_start": page_start,
        "page_end": page_end,
    }

    chapters = toc_structure.get("chapters", {})
    for c_id, chapter in chapters.items():
        c_start = chapter.get("start", 0)
        c_end = chapter.get("end", 0)
        if c_start <= page_start <= c_end or c_start <= page_end <= c_end:
            enrichment["chapter_id"] = c_id
            enrichment["chapter_name"] = chapter.get("name", "")

            # Check sections within this chapter
            for s_id, section in chapter.get("sections", {}).items():
                s_start = section.get("start", 0)
                s_end = section.get("end", 0)
                if s_start <= page_start <= s_end or s_start <= page_end <= s_end:
                    enrichment["section_id"] = s_id
                    enrichment["section_name"] = section.get("name", "")
                    break
            break

    return enrichment


def populate_lancedb_from_jsonl(
    jsonl_path: str, db_uri: str, table_name: str,
    toc_structure: dict | None = None
):
    """
    Reads structured .JSONL chunks, generates embeddings,
    and inserts them into LanceDB with an FTS index for hybrid search.

    If toc_structure is provided, chunks are enriched with chapter/section/page metadata.
    """
    chunks_data: list[dict] = []
    logger.info(f"Reading chunks from {jsonl_path}...")

    if not os.path.exists(jsonl_path):
        logger.error(f"JSONL file not found: {jsonl_path}")
        return None

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks_data.append(json.loads(line))

    if not chunks_data:
        logger.warning("No chunks found in the provided JSONL file.")
        return None

    texts = [chunk.get("text", "") for chunk in chunks_data]
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = encoder.encode(texts, show_progress_bar=True)

    records = []
    for i, chunk in enumerate(chunks_data):
        parsed_chunk = parseChunk(chunk)
        record = {
            "vector": embeddings[i].tolist(),
            **parsed_chunk
        }

        # Enrich with chapter/section metadata if ToC is available
        if toc_structure:
            enrichment = enrich_chunk_with_toc(parsed_chunk, toc_structure)
            record.update(enrichment)
            if enrichment["chapter_id"] is not None:
                logger.debug(f"Enriched chunk {i} with chapter={enrichment['chapter_id']}, section={enrichment['section_id']}")

        records.append(record)

    with open("temp.json", 'w') as fp:
        json.dump(records, fp)

    db = lancedb.connect(db_uri)
    logger.info(f"Writing {len(records)} records to LanceDB table '{table_name}' at {db_uri}...")
    table = db.create_table(table_name, data=records, mode="overwrite")

    logger.info("Creating Full-Text Search (FTS) index on the 'text' column...")
    table.create_fts_index(field_names="text", replace=True)

    logger.info("LanceDB successfully populated, indexed, and ready for hybrid queries!")

    return table


if __name__ == "__main__":
    from settings import settings
    import json

    # Load ToC structure for enrichment
    with open("chapters_sections.json", "r") as f:
        toc_data = json.load(f)

    # For now, use the first PDF's ToC structure
    pdf_name = "Skiena-The_Algorithm_Design_Manual.pdf"
    toc_structure = toc_data.get(pdf_name, None)

    populate_lancedb_from_jsonl(
        jsonl_path="lancedb_chunks/Skiena-The_Algorithm_Design_Manual/raw_hybrid.jsonl",
        db_uri=settings.LANCEDB_DATA_PATH,
        table_name="textbook_chunks",
        toc_structure=toc_structure
    )
