from typing import Any

def LLMParseTOC(content: list[dict[str, Any]]):
    from llm_client import client
    import asyncio

    return asyncio.run(client.get_completion(task_type="parse_toc",
                                             messages=[{
                                                 "role": "system",
                                                 "content": """Extract the table of content's chapter and section page ranges. Each page number listed in the 
                                                 table of contents is the starting page of that chapter/section. Chapter numbers are written in the format 
                                                 "X", and section numbers are written in the format "X.Y". For sections,  the page before the next section's 
                                                 page is the current section's end, unless the section is the last section in the chapter in which case the 
                                                 section's end will be the same as the chapter's end. Do not consider subsections (subsections are in X.Y.Z 
                                                 or a similar format). The keys for the \"chapters\" dict are the chapter numbers, and must contain only the 
                                                 chapter number X; the keys for the \"section\" dict within each chapter are the section numbers, 
                                                 and must contain only the section number Y without the chapter number X."""
                                             }, {
                                                 "role": "user",
                                                 "content": content
                                             }],
                                             ))
