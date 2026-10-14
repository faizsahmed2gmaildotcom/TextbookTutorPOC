import json
from typing import Any
from llm_client import client
import asyncio


# This is a heuristic and has only been tested with table of contents in the form {chapter#}.{section#}.{subsection#}
def LLMParseTOC(content: list[dict[str, Any]], page_offset: int = 0):
    result: dict[str, dict[str, dict[str, Any]]] \
        = json.loads(asyncio.run(client.get_completion(task_type="parse_toc",
                                                       messages=[{
                                                           "role": "system",
                                                           "content":
                                                               """Extract the book chapter (X or a similar format) and section (X.Y or a similar format) 
                                                               page ranges. Each page number listed in the table of contents is the starting page of that 
                                                               chapter/section. Chapter number is X, and section number is Y."""
                                                       }, {
                                                           "role": "user",
                                                           "content": content
                                                       }],
                                                       )))

    chapters = sorted(list(result["chapters"].keys()), key=lambda c: int(c))

    # 1. Process all chapters except the last one
    for i, c_id in enumerate(chapters[:-1]):
        cur_chapter = result["chapters"][c_id]
        next_chapter = result["chapters"][chapters[i + 1]]  # Safer than str(int(c_id) + 1) in case of numbering gaps

        cur_chapter["start"] += page_offset
        cur_chapter.update({"end": next_chapter["start"] - 1 + page_offset})  # Chapters will always begin at the start of a page.

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
                cur_section.update({"end": next_section["start"] + page_offset})  # Sections may not begin at the start of a page!

            # Last section ends at the chapter's end
            last_section = cur_chapter["sections"][sections[-1]]
            last_section["start"] += page_offset
            last_section.update({"end": cur_chapter["end"]})

    # 2. Handle the very last chapter separately
    if chapters:
        last_chapter_id = chapters[-1]
        cur_chapter = result["chapters"][last_chapter_id]
        cur_chapter["start"] += page_offset
        # Note: Ensure cur_chapter["end"] is defined or handled if it's the absolute last page of the book
        if "end" in cur_chapter and cur_chapter["end"] is not None:
            cur_chapter["end"] = cur_chapter["end"] + page_offset

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
