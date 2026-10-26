import json
from typing import Any
from llm_client import client
import asyncio
from logging_config import get_logger

logger = get_logger(__name__)


def LLMParseTOC(content: str, pdf_path: str, toc_offset: int):
    """
    Parse a table of contents using an LLM.

    Args:
        content: Concatenated text of the table of contents (from docling document_index items).
        pdf_path: Path to the PDF file, for determining the last chapter's end page.
        toc_offset: Offset to add to all page numbers as a fallback (first page after the ToC).

    Returns:
        Dict with structure compatible with chapters_sections.json.
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
                                                               you have extracted all of them before finishing."""
                                                       }, {
                                                           "role": "user",
                                                           "content": content
                                                       }],
                                                       )))

    # Map ToC pages to pages indexes
    from pypdf import PdfReader
    with PdfReader("PDFs/guide-t-cp.pdf") as reader:
        page_labels = reader.page_labels
    page_to_idx = {p: i + 1 for i, p in enumerate(page_labels)}

    # Fallback to offset by ToC page if page labels not provided (i.e. labels map 1:1 to pages)
    if all(label == str(idx) for label, idx in page_to_idx.items()):
        page_to_idx = {p: i + 1 + toc_offset for i, p in enumerate(page_labels)}

    chapters = sorted(list(result["chapters"].keys()), key=lambda c: int(c))

    # In the future, make this a recursive algorithm that offsets all "start" and "end" values regardless of depth, handling the last ones accordingly
    # 1. Process all chapters, handling the last one separately
    for i, c_id in enumerate(chapters):
        cur_chapter = result["chapters"][c_id]
        cur_chapter["start"] = page_to_idx[str(cur_chapter["start"])]

        if c_id == chapters[-1]:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            cur_chapter.update({"end": doc.page_count - 1})
            doc.close()
            logger.info(f"Last chapter '{cur_chapter.get('name', c_id)}' end set to PDF page count: {cur_chapter['end']}")
        else:
            next_chapter = result["chapters"][chapters[i + 1]]
            cur_chapter.update({"end": page_to_idx[str(next_chapter["start"])] - 1})
        # cur_chapter start and end have been mapped to indexes by this point

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

                cur_section["start"] = page_to_idx[str(cur_section["start"])]
                cur_section.update({"end": page_to_idx[str(next_section["start"])]})

            # Last section ends at the chapter's end
            last_section = cur_chapter["sections"][sections[-1]]
            last_section["start"] = page_to_idx[str(last_section["start"])]
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


if __name__ == "__main__":
    pass
