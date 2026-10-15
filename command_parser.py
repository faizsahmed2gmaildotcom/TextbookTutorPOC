"""
Command parser for textbook query interface.

Supports commands at the start of a query:
  /chapter N <query>     — Filter to chapter N
  /section N.M <query>   — Filter to section M in chapter N
  /pages from-to <query> — Filter to page range
"""

import re
from dataclasses import dataclass


@dataclass
class ParsedQuery:
    raw_query: str
    command: str | None        # "chapter", "section", "pages", or None
    chapter_id: str | None
    section_id: str | None
    page_from: int | None
    page_to: int | None
    search_text: str           # Query text after stripping command


def parse_query(user_input: str) -> ParsedQuery:
    """
    Parse /chapter, /section, /pages commands from the start of a query string.

    Returns a ParsedQuery with the command and extracted search text.
    """
    stripped = user_input.strip()

    # /chapter N <query>
    m = re.match(r'^/chapter\s+(\d+)(?:\s+(.+))?', stripped, re.IGNORECASE)
    if m:
        return ParsedQuery(
            raw_query=user_input,
            command="chapter",
            chapter_id=m.group(1),
            section_id=None,
            page_from=None,
            page_to=None,
            search_text=m.group(2).strip() if m.group(2) else ""
        )

    # /section N.M <query>
    m = re.match(r'^/section\s+(\d+)\.(\d+)(?:\s+(.+))?', stripped, re.IGNORECASE)
    if m:
        return ParsedQuery(
            raw_query=user_input,
            command="section",
            chapter_id=m.group(1),
            section_id=m.group(2),
            page_from=None,
            page_to=None,
            search_text=m.group(3).strip() if m.group(3) else ""
        )

    # /pages from-to <query>
    m = re.match(r'^/pages\s+(\d+)-(\d+)(?:\s+(.+))?', stripped, re.IGNORECASE)
    if m:
        return ParsedQuery(
            raw_query=user_input,
            command="pages",
            chapter_id=None,
            section_id=None,
            page_from=int(m.group(1)),
            page_to=int(m.group(2)),
            search_text=m.group(3).strip() if m.group(3) else ""
        )

    # No command → default
    return ParsedQuery(
        raw_query=user_input,
        command=None,
        chapter_id=None,
        section_id=None,
        page_from=None,
        page_to=None,
        search_text=stripped
    )
