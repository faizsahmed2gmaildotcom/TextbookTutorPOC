from typing import Literal, Any
import lancedb
from populate_lancedb import encoder
from logging_config import get_logger

logger = get_logger(__name__)


def LLMParseImg(img_path: str):
    import base64
    with open(img_path, "rb") as img_fp:
        return base64.b64encode(img_fp.read()).decode("utf-8")


def query_textbook_rag(
        db_uri: str,
        table_name: str,
        user_query: str,
        response_format: Literal["sequential", "formatted"],
        parsed=None
):
    """
    Performs a hybrid search in LanceDB and sends the top matching chunks
    with their page citations to the LLM via LiteLLM.

    If parsed (ParsedQuery) is provided with a command, applies a .where() filter
    to restrict results to the specified chapter/section/page range.
    """

    db = lancedb.connect(db_uri)
    table = db.open_table(table_name)

    logger.info(f"Executing hybrid search for query: '{user_query}'"
                + (f" [filter: {parsed.command}]" if parsed and parsed.command else ""))

    # Build the search with optional .where() filter
    if not user_query:  # Filter only
        search = table.search()
    else:  # Filter + search
        search = table.search(query_type="hybrid")
        query_vector = encoder.encode(user_query).tolist()
        search = search.vector(query_vector).text(user_query)

    if parsed and parsed.command == "chapter" and parsed.chapter_id:
        search = search.where(f"chapter_id = '{parsed.chapter_id}'")
        logger.info(f"Filtering to chapter {parsed.chapter_id}")
    elif parsed and parsed.command == "section" and parsed.chapter_id and parsed.section_id:
        search = search.where(
            f"chapter_id = '{parsed.chapter_id}' AND section_id = '{parsed.section_id}'"
        )
        logger.info(f"Filtering to section {parsed.chapter_id}.{parsed.section_id}")
    elif parsed and parsed.command == "pages" and parsed.page_from is not None and parsed.page_to is not None:
        # Check for range overlap: chunk overlaps with query range if chunk starts before or at to
        # AND chunk ends after or at from
        search = search.where(
            f"page_start <= {parsed.page_to} AND page_end >= {parsed.page_from}"
        )
        logger.info(f"Filtering to pages {parsed.page_from}-{parsed.page_to}")

    results = search.limit(20).to_pandas()

    if results.empty:
        return [{"type": "text", "text": "No relevant textbook passages found for this query."}]

    llm_input: list[dict[str, Any]] = []

    for _, row in results.iterrows():
        raw_text = row.get("text")
        items = row.get("items", [])
        if isinstance(items, str):
            import json
            items = json.loads(items)
        for item in items:
            if response_format == "sequential":
                pr = item.get("page_range", [0, 0])
                pages = str(pr[0]) + (f"-{pr[1]}" if pr[0] != pr[1] else "")

                pic_ref = item.get("picture_ref", "")
                chapter_name = row.get("chapter_name", "")
                section_name = row.get("section_name", "")

                context_parts = [f"Pages: {pages}"]
                if chapter_name:
                    context_parts.append(f"Chapter: {chapter_name}")
                if section_name:
                    context_parts.append(f"Section: {section_name}")
                if pic_ref:
                    context_parts.append(f"Associated picture: {pic_ref}")

                cr = item.get("char_range", [0, len(raw_text)])
                text_snippet = raw_text[cr[0]:cr[1]] if raw_text else ""

                llm_input.append({
                    "type": "text",
                    "text": " | ".join(context_parts) + "\n" + text_snippet
                })

                if pic_ref:
                    try:
                        # Determine picture path from db_uri
                        import os
                        pic_path = os.path.join("lancedb_chunks", "Skiena-The_Algorithm_Design_Manual", "pictures", pic_ref)
                        if os.path.exists(pic_path):
                            llm_input.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{LLMParseImg(pic_path)}"
                                }
                            })
                    except Exception as e:
                        logger.warning(f"Could not load picture {pic_ref}: {e}")

            elif response_format == "formatted":
                print(item)

    return llm_input


if __name__ == "__main__":
    from command_parser import parse_query, ParsedQuery

    DB_PATH = "lancedb_data"
    TABLE_NAME = "textbook_chunks"

    sample_query = "8.8 Exercises"
    parsed = ParsedQuery(
        raw_query="/chapter 8 8.8 Exercises",
        command="chapter",
        chapter_id="8",
        section_id=None,
        page_from=None,
        page_to=None,
        search_text="8.8 Exercises"
    )

    answer = query_textbook_rag(DB_PATH, TABLE_NAME, sample_query, "sequential", parsed)

    if answer:
        import json

        print(json.dumps(answer, indent=4))
