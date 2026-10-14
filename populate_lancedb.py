import json
from typing import Any
import lancedb
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, asdict

model_name = "Qwen/Qwen3-Embedding-0.6B"
print(f"Loading embedding model: {model_name}...")
encoder = SentenceTransformer(model_name, trust_remote_code=True)
encoder.max_seq_length = 1024  # Token context window. Has the same effects as #trainable params.


@dataclass
class Item:
    label: str
    char_range: tuple[int, int] = (10 ** 9, -1)  # Character range in text (inclusive)
    page_range: tuple[int, int] = (10 ** 9, -1)  # Pages the text covers
    picture_ref: str = ""  # "" if not associated with any image


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
            new_item.picture_ref = f"pic_{raw_item["parent"]["$ref"].split('/')[-1]}.png"

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


def populate_lancedb_from_jsonl(jsonl_path: str, db_uri: str, table_name: str):
    """
    Reads structured .JSONL chunks, generates embeddings,
    and inserts them into LanceDB with an FTS index for hybrid search.
    """
    # 2. Read chunks from the .JSONL file
    chunks_data: list[dict] = []
    print(f"Reading chunks from {jsonl_path}...")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks_data.append(json.loads(line))

    if not chunks_data:
        print("No chunks found in the provided JSONL file.")
        return

    # 3. Extract text content and generate dense vector embeddings in the batch
    texts = [chunk.get("text", "") for chunk in chunks_data]
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = encoder.encode(texts, show_progress_bar=True)

    # 4. Map chunks and embeddings into records matching your schema
    records = []
    for i, chunk in enumerate(chunks_data):
        # docling hybrid chunks store structural metadata; adjust keys based on your exact .JSONL structure
        parsed_chunk = parseChunk(chunk)
        record = {
            "vector": embeddings[i].tolist(),
            **parsed_chunk
        }
        records.append(record)

    # 5. Connect to local LanceDB and create/overwrite the table
    db = lancedb.connect(db_uri)
    print(f"Writing records to LanceDB table '{table_name}' at {db_uri}...")
    table = db.create_table(table_name, data=records, mode="overwrite")

    # 6. Create a Full-Text Search (FTS) index on the text column
    # LanceDB requires an FTS index on the queried text column for full-text and hybrid search.
    print("Creating Full-Text Search (FTS) index on the 'text' column...")
    table.create_fts_index(field_names="text", replace=True)

    print("LanceDB successfully populated, indexed, and ready for hybrid queries!")


if __name__ == "__main__":
    populate_lancedb_from_jsonl(
        jsonl_path="lancedb_chunks/Skiena-The_Algorithm_Design_Manual/raw_hybrid.jsonl",
        db_uri="./lancedb_data",
        table_name="textbook_chunks"
    )
