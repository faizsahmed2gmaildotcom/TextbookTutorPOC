"""
Table of Contents parsers using the Strategy pattern.

Three modes:
1. LLM (slow, general but random) — sends ToC text to an external LLM
2. docling (fast, heuristic but precise) — infers structure from docling SectionHeaderItem hierarchy
3. pymupdf (fastest, exact for PDFs with bookmarks) — uses native PDF ToC bookmarks
"""

from abc import ABC, abstractmethod
from enum import Enum
from logging_config import get_logger

logger = get_logger(__name__)


class TOCMode(Enum):
    LLM = "llm"
    DOCLING = "docling"
    PYMUPDF = "pymupdf"


class TOCParser(ABC):
    """Abstract base for ToC parsers."""

    @abstractmethod
    def parse(self, pdf_path: str) -> dict:
        """
        Parse the table of contents from a PDF.

        Returns:
            A dict with a "chapters" key, compatible with chapters_sections.json format:
            {"chapters": {"1": {"name": "...", "start": N, "end": N, "sections": {...}}}}
        """
        ...


def get_toc_parser(mode: TOCMode) -> TOCParser:
    """Factory to get the appropriate TOC parser for a given mode."""
    if mode == TOCMode.LLM:
        return LLMTOCParser()
    elif mode == TOCMode.DOCLING:
        return DoclingTOCParser()
    elif mode == TOCMode.PYMUPDF:
        return PymupdfTOCParser()
    else:
        raise ValueError(f"Unknown TOC mode: {mode}")


# ---------------------------------------------------------------------------
# Mode 1: LLM
# ---------------------------------------------------------------------------

class LLMTOCParser(TOCParser):
    """Parse ToC using an external LLM (current method, migrated to docling input)."""

    def parse(self, pdf_path: str) -> dict:
        from pdf_to_jsonl import convert_pdf, get_toc_pages_from_docling, get_toc_content_from_pymupdf
        from llm_parser import LLMParseTOC

        # Convert the full document to get ToC pages
        doc = convert_pdf(pdf_path, page_range=(1, 30))
        toc_pages = get_toc_pages_from_docling(doc)

        if not toc_pages:
            logger.error("LLM mode: Could not auto-detect ToC pages")
            raise RuntimeError("Cannot parse ToC with LLM mode: no document_index items found")

        # Get ToC content from pymupdf using the detected page numbers
        toc_content = get_toc_content_from_pymupdf(pdf_path, toc_pages)
        page_offset = toc_pages[-1]  # Final ToC page (because page ranges are 0-indexed, there should be no +1 here)

        logger.info(f"LLM mode: Parsing ToC with page_offset={page_offset}, pages={toc_pages}")
        return LLMParseTOC(toc_content, page_offset, pdf_path)


# ---------------------------------------------------------------------------
# Mode 2: docling (heuristic, based on SectionHeaderItem hierarchy)
# ---------------------------------------------------------------------------

class DoclingTOCParser(TOCParser):
    """Parse ToC by extracting SectionHeaderItem hierarchy from docling."""

    def parse(self, pdf_path: str) -> dict:
        import pymupdf
        from pdf_to_jsonl import convert_pdf
        from docling.datamodel.base_models import DocItemLabel

        doc = convert_pdf(pdf_path)
        logger.info("Docling mode: Extracting chapter/section structure from SectionHeaderItems")

        # Collect all SectionHeaderItems with their level and page
        headers = []
        for item, level in doc.iterate_items():
            if item.label == DocItemLabel.SECTION_HEADER:
                page_no = None
                if hasattr(item, 'prov') and item.prov:
                    page_no = min(prov.page_no for prov in item.prov)
                name = item.text if hasattr(item, 'text') else str(item)
                headers.append({"level": level, "name": name.strip(), "page": page_no})

        if not headers:
            logger.warning("Docling mode: No SectionHeaderItems found")
            return {"chapters": {}}

        # Determine chapter level (the minimum level among all headers)
        min_level = min(h["level"] for h in headers)
        # Chapter level = min_level, Section level = min_level + 1
        chapter_level = min_level
        section_level = min_level + 1

        logger.info(f"Docling mode: Found {len(headers)} headers, chapter_level={chapter_level}, section_level={section_level}")

        # Build chapter/section structure
        chapters = {}
        current_chapter = None
        chapter_counter = 0
        section_counter = 0

        for h in headers:
            if h["level"] == chapter_level:
                chapter_counter += 1
                section_counter = 0
                chapter_id = str(chapter_counter)
                current_chapter = {
                    "name": h["name"],
                    "start": h["page"],
                    "sections": {}
                }
                chapters[chapter_id] = current_chapter
            elif h["level"] == section_level and current_chapter is not None:
                section_counter += 1
                section_id = str(section_counter)
                current_chapter["sections"][section_id] = {
                    "name": h["name"],
                    "start": h["page"]
                }
            # Higher levels (subsections) are skipped

        # Compute end pages
        chapter_ids = sorted(chapters.keys(), key=lambda c: int(c))
        total_pages = pymupdf.open(pdf_path).page_count

        for i, c_id in enumerate(chapter_ids):
            chapter = chapters[c_id]
            if i < len(chapter_ids) - 1:
                next_chapter = chapters[chapter_ids[i + 1]]
                chapter["end"] = next_chapter["start"] - 1
            else:
                chapter["end"] = total_pages

            # Compute section end pages
            if chapter["sections"]:
                section_ids = sorted(chapter["sections"].keys(), key=lambda s: int(s))
                for j, s_id in enumerate(section_ids):
                    section = chapter["sections"][s_id]
                    if j < len(section_ids) - 1:
                        next_section = chapter["sections"][section_ids[j + 1]]
                        section["end"] = next_section["start"]
                    else:
                        section["end"] = chapter["end"]

        logger.info(f"Docling mode: Extracted {len(chapters)} chapters")
        return {"chapters": chapters}


# ---------------------------------------------------------------------------
# Mode 3: pymupdf (native PDF bookmarks/ToC)
# ---------------------------------------------------------------------------

class PymupdfTOCParser(TOCParser):
    """Parse ToC using pymupdf's native PDF bookmark extraction."""

    def parse(self, pdf_path: str) -> dict:
        import pymupdf
        import re

        logger.info("Pymupdf mode: Extracting native PDF ToC bookmarks")
        doc = pymupdf.open(pdf_path)
        toc = doc.get_toc()
        total_pages = doc.page_count
        doc.close()

        if not toc:
            logger.warning("Pymupdf mode: No native ToC bookmarks found")
            return {"chapters": {}}

        # Parse the flat ToC list into structured items
        items = []
        for level, title, page in toc:
            items.append({"level": level, "title": title.strip(), "page": page})

        min_level = min(item["level"] for item in items)
        max_level = max(item["level"] for item in items)

        # Count items per level
        level_counts = {}
        for item in items:
            level_counts[item["level"]] = level_counts.get(item["level"], 0) + 1

        logger.info(f"Pymupdf mode: {len(items)} bookmarks, levels {min_level}-{max_level}, counts={level_counts}")

        # Heuristic: the level with the MOST items is typically the section level.
        # The level immediately above it is the chapter level.
        # This works for:
        #   Skiena: level 3 has most → section, level 2 → chapter, level 1 → parts
        #   guide-t-cp: level 2 has most → section, level 1 → chapter

        if max_level > min_level:
            # Find the level with the maximum count (section level)
            section_level = max(level_counts, key=level_counts.get)
            if section_level > min_level:
                chapter_level = section_level - 1
            else:
                # section_level == min_level means min_level has most items
                chapter_level = min_level
                section_level = min_level + 1
        else:
            # Only one level — treat as chapters
            chapter_level = min_level
            section_level = min_level + 1  # won't match anything

        logger.info(f"Pymupdf mode: chapter_level={chapter_level}, section_level={section_level}")

        # Build chapter/section structure
        chapters = {}
        current_chapter = None
        chapter_counter = 0
        section_counter = 0

        # Skip items below chapter_level (they are frontmatter like "Preface", "Contents")
        # UNLESS chapter_level == min_level (then everything at min_level IS a chapter)
        skip_below = chapter_level if chapter_level > min_level else min_level - 1

        for item in items:
            if item["level"] < chapter_level:
                # Frontmatter / part headers — skip for now
                continue
            elif item["level"] == chapter_level:
                chapter_counter += 1
                section_counter = 0
                # Try to extract chapter number from title
                chapter_num_match = re.match(r'^(?:Chapter\s+)?(\d+)', item["title"])
                if chapter_num_match:
                    chapter_id = chapter_num_match.group(1)
                else:
                    chapter_id = str(chapter_counter)

                current_chapter = {
                    "name": item["title"],
                    "start": item["page"],
                    "sections": {}
                }
                chapters[chapter_id] = current_chapter
            elif item["level"] == section_level and current_chapter is not None:
                section_counter += 1
                # Try to extract section number
                section_num_match = re.match(r'^(\d+(?:\.\d+)?)', item["title"])
                if section_num_match:
                    parts = section_num_match.group(1).split('.')
                    section_id = parts[-1]
                else:
                    section_id = str(section_counter)
                current_chapter["sections"][section_id] = {
                    "name": item["title"],
                    "start": item["page"]
                }
            # Higher levels (subsections) are skipped

        # Compute end pages
        chapter_ids = sorted(chapters.keys(), key=lambda c: int(c) if c.isdigit() else c)

        for i, c_id in enumerate(chapter_ids):
            chapter = chapters[c_id]
            if i < len(chapter_ids) - 1:
                next_chapter = chapters[chapter_ids[i + 1]]
                chapter["end"] = next_chapter["start"] - 1
            else:
                chapter["end"] = total_pages

            if chapter["sections"]:
                section_ids = sorted(chapter["sections"].keys(), key=lambda s: int(s) if s.isdigit() else s)
                for j, s_id in enumerate(section_ids):
                    section = chapter["sections"][s_id]
                    if j < len(section_ids) - 1:
                        next_section = chapter["sections"][section_ids[j + 1]]
                        section["end"] = next_section["start"]
                    else:
                        section["end"] = chapter["end"]

        logger.info(f"Pymupdf mode: Extracted {len(chapters)} chapters")
        return {"chapters": chapters}
