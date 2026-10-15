import json
from typing import Any
from llm_client import client
import asyncio
from logging_config import get_logger

logger = get_logger(__name__)


def LLMParseTOC(content: str, page_offset: int, pdf_path: str):
    """
    Parse a table of contents using an LLM.

    Args:
        content: Concatenated text of the table of contents (from docling document_index items).
        page_offset: Offset to add to all page numbers (first page after the ToC).
        pdf_path: Path to the PDF file, for determining the last chapter's end page.

    Returns:
        Dict with chapters structure compatible with chapters_sections.json.
    """
    with open("temp.txt", 'w') as fp:
        fp.write(content)
    result: dict[str, dict[str, dict[str, Any]]] \
        = json.loads(asyncio.run(client.get_completion(task_type="parse_toc",
                                                       messages=[{
                                                           "role": "system",
                                                           "content":
                                                               """Extract the book chapter (a single number) and section (two numbers separated by a
                                                               separator, such as '.' or '-') page ranges. Each page number listed in the table of contents
                                                               is the starting page of that chapter/section.

                                                               CRITICAL: You must extract ALL chapters and sections from the table of contents.
                                                               Do not skip any chapters. Count the total number of chapters and ensure
                                                               you have extracted all of them before finishing. If the book has 23 chapters,
                                                               you MUST extract all 23 chapters with their page ranges."""
                                                       }, {
                                                           "role": "user",
                                                           "content": content
                                                       }],
                                                       )))

    chapters = sorted(list(result["chapters"].keys()), key=lambda c: int(c))

    # 1. Process all chapters and handle the last one
    for i, c_id in enumerate(chapters):
        cur_chapter = result["chapters"][c_id]

        cur_chapter["start"] += page_offset
        if c_id == chapters[-1]:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            cur_chapter.update({"end": doc.page_count})
            doc.close()
            logger.info(f"Last chapter '{cur_chapter.get('name', c_id)}' end set to PDF page count: {cur_chapter['end']}")
        else:
            next_chapter = result["chapters"][chapters[i + 1]]
            cur_chapter.update({"end": next_chapter["start"] - 1 + page_offset})

        # Use list() to iterate over a static copy of keys while modifying the dictionary
        for s_id in list(cur_chapter["sections"].keys()):
            s_split = s_id.split('.')
            if len(s_split) > 2:
                del cur_chapter["sections"][s_id]
            elif len(s_split) == 2:
                new_s_id = s_split[-1]
                cur_chapter["sections"][new_s_id] = cur_chapter["sections"].pop(s_id)

        if cur_chapter["sections"]:
            sections = sorted(list(cur_chapter["sections"].keys()), key=lambda s: int(s))
            for j, s_id in enumerate(sections[:-1]):
                cur_section = cur_chapter["sections"][s_id]
                next_section = cur_chapter["sections"][sections[j + 1]]

                cur_section["start"] += page_offset
                cur_section.update({"end": next_section["start"] + page_offset})

            # Last section ends at the chapter's end
            last_section = cur_chapter["sections"][sections[-1]]
            last_section["start"] += page_offset
            last_section.update({"end": cur_chapter["end"]})

    return result


def LLMParseVisuals(content: list[dict[str, Any]]):
    return json.loads(asyncio.run(client.get_completion(task_type="parse_visuals",
                                                        messages=[{
                                                            "role": "system",
                                                            "content":
                                                                """The user will upload page contents (text and diagrams) from a PDF. With as much detail as
                                                                possible, describe all images and diagrams on each page. Optimize your result to be
                                                                interpreted by a text-only LLM, whose job is to academically tutor the user (using your
                                                                image/diagram descriptions; all remaining text on the page will be parsed and uploaded
                                                                separately). The keys for the "pages" dict must be a 1-indexed page number as integers."""
                                                        }, {
                                                            "role": "user",
                                                            "content": content
                                                        }]
                                                        )))
