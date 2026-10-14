from typing import Literal, Any
import lancedb
from populate_lancedb import encoder


def LLMParseImg(img_path: str):
    import base64
    with open(img_path, "rb") as img_fp:
        return base64.b64encode(img_fp.read()).decode("utf-8")


def query_textbook_rag(db_uri: str, table_name: str, user_query: str, response_format: Literal["sequential", "formatted"]):
    """
    Performs a hybrid search in LanceDB and sends the top matching chunks
    with their page citations to the LLM via LiteLLM.
    """
    # Convert text query to a vector (which is what LanceDB understands)
    query_vector = encoder.encode(user_query).tolist()

    # 1. Connect to local LanceDB
    db = lancedb.connect(db_uri)
    table = db.open_table(table_name)

    print(f"Executing hybrid search for query: '{user_query}'... top items:\n")

    # 2. Perform Hybrid Search (Dense Vector + Full-Text Search)
    # LanceDB automatically handles vector embedding translation if an embedding function is configured,
    # or runs a hybrid match query combining semantic and keyword ranking.
    results = (
        table.search(query_type="hybrid")
        .vector(query_vector)
        .text(user_query)
        .limit(20)  # Fetch top N most relevant chunks
        .to_pandas()
    )

    if results.empty:
        return "No relevant textbook passages found for this query."

    # 3. Format retrieved chunks and build context with metadata citations
    llm_input: list[dict[str, Any]] = []

    for _, row in results.iterrows():
        raw_text = row.get("text")
        for item in row.get("items"):
            if response_format == "sequential":
                pages = str(item["page_range"][0]) + (f"-{item["page_range"][1]}" if item["page_range"][0] != item["page_range"][1] else "")

                llm_input.append({
                    "type": "text",
                    "text": f"Pages: {pages}{f" | Associated picture: {item["picture_ref"]}" if item["picture_ref"] else ""}\n"
                            f"{raw_text[item["char_range"][0]:item["char_range"][1]]}"
                })
                if item["picture_ref"]:
                    llm_input.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{LLMParseImg("lancedb_chunks/Skiena-The_Algorithm_Design_Manual/pictures/" + item["picture_ref"])}"
                        }
                    })

            elif response_format == "formatted":
                print(item)

    return llm_input


if __name__ == "__main__":
    # Configuration parameters
    DB_PATH = "lancedb_data"
    TABLE_NAME = "textbook_chunks"

    sample_query = "8.8 Exercises"

    # Run the RAG pipeline
    answer = query_textbook_rag(DB_PATH, TABLE_NAME, sample_query, "sequential")

    if answer:
        import json

        print(json.dumps(answer, indent=4))
