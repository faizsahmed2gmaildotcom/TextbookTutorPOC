from typing import Literal, Any
import lancedb
from populate_lancedb import encoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load Qwen3-Reranker-0.6B for re-ranking
print("Loading Qwen3-Reranker-0.6B...")
_reranker_tokenizer = None
_reranker_model = None


def _load_reranker():
    """Lazy load the reranker model and tokenizer."""
    global _reranker_tokenizer, _reranker_model
    if _reranker_tokenizer is None:
        _reranker_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Reranker-0.6B",
            trust_remote_code=True,
            use_fast=True
        )
        # Set padding token if not defined (Qwen3-Reranker uses pad_token_id)
        if _reranker_tokenizer.pad_token is None:
            _reranker_tokenizer.pad_token = _reranker_tokenizer.eos_token
    if _reranker_model is None:
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(
            "Qwen/Qwen3-Reranker-0.6B",
            trust_remote_code=True,
            dtype=torch.float16,
            device_map="auto"
        )
    return _reranker_tokenizer, _reranker_model


def rerank_results(query: str, documents: list[str], top_k: int = 10) -> list[int]:
    """
    Re-rank documents using Qwen3-Reranker-0.6B.

    Args:
        query: The user's search query
        documents: List of document texts to rank
        top_k: Number of top results to return after re-ranking

    Returns:
        List of indices corresponding to the top-k ranked documents
    """
    if not documents or len(documents) <= 1:
        return list(range(len(documents)))

    tokenizer, model = _load_reranker()

    # Qwen3-Reranker processes pairs one at a time for scoring
    scores = []
    with torch.no_grad():
        for doc in documents:
            # Tokenize single pair
            inputs = tokenizer(
                query,
                doc,
                truncation=True,
                max_length=4096,
                return_tensors="pt"
            )

            # Move inputs to the same device as the model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Get score for this pair
            output = model(**inputs)
            score = output.logits[0, 0].item()  # Get the first (and only) score
            scores.append(score)

    # Get indices sorted by score (highest first)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # Return top_k indices
    return ranked_indices[:top_k]


def LLMParseImg(img_path: str):
    import base64
    with open(img_path, "rb") as img_fp:
        return base64.b64encode(img_fp.read()).decode("utf-8")


def query_textbook_rag(db_uri: str, table_name: str, user_query: str, response_format: Literal["sequential", "formatted"], use_rerank: bool = True):
    """
    Performs a hybrid search in LanceDB and sends the top matching chunks
    with their page citations to the LLM via LiteLLM.

    Optionally re-ranks results using Qwen3-Reranker-0.6B for improved relevance.
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

    # Get raw texts for re-ranking
    raw_texts = results["text"].tolist()

    # 3. Re-rank results using Qwen3-Reranker-0.6B if enabled
    if use_rerank and len(raw_texts) > 1:
        print("Re-ranking results with Qwen3-Reranker-0.6B...")
        ranked_indices = rerank_results(user_query, raw_texts, top_k=20)
        # Reorder results based on reranker scores
        results = results.iloc[ranked_indices].reset_index(drop=True)

    # 4. Format retrieved chunks and build context with metadata citations
    llm_input: list[dict[str, Any]] = []

    for _, row in results.iterrows():
        raw_text = row.get("text")
        heading = row.get("heading", "")
        items = row.get("items", [])

        if response_format == "sequential":
            # The raw_text already contains the heading prepended in populate_lancedb.py
            # So we just use it directly without adding the header again
            # However, we can include the heading as a header for better context

            header_text = ""
            if heading:
                header_text = f"Section: {heading}\n\n"

            # Add character-based citations from items
            # Use the full raw_text directly since it contains all content
            # char_ranges are preserved for citation purposes but don't fragment the text
            extracted_text = raw_text

            llm_input.append({
                "type": "text",
                "text": header_text + extracted_text
            })

            # Add associated images
            for item in items:
                if item.get("picture_ref"):
                    img_path = f"lancedb_chunks/Skiena-The_Algorithm_Design_Manual/pictures/{item['picture_ref']}"
                    llm_input.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{LLMParseImg(img_path)}"
                        }
                    })

    return llm_input


if __name__ == "__main__":
    # Configuration parameters
    DB_PATH = "lancedb_data"
    TABLE_NAME = "textbook_chunks"

    sample_query = "Images"

    # Run the RAG pipeline with re-ranking enabled
    answer = query_textbook_rag(DB_PATH, TABLE_NAME, sample_query, "sequential", use_rerank=True)

    if answer:
        import json

        print(json.dumps(answer, indent=4))
