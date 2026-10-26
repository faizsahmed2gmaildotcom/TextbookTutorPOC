# TextbookTutorPOC - Agent Guide

## Project Overview

This is a **Textbook Tutor Proof of Concept** - an AI-powered educational assistant that helps students learn from textbook PDFs. The system uses LLMs to parse, understand, and tutor users on textbook content, with support for both text and visual (diagram/image) content.

### Key Capabilities

1. **PDF Processing**: Extracts text, images, and vector graphics from PDFs using `pymupdf`
2. **Content Parsing**: Uses LLMs to parse Table of Contents and identify chapter/section page ranges
3. **Hybrid RAG**: Uses LanceDB with both semantic (vector) and full-text search for retrieving textbook content
4. **Visual Understanding**: Parses diagrams and images using vision-capable LLMs (GLM-5.3-Flash)
5. **Structured Chunking**: Uses Docling's hybrid chunker to preserve structural relationships in textbook content

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Textbook Tutor POC                          │
├─────────────────────────────────────────────────────────────────────┤
│  Input: PDF Files (PDFs/)                                           │
│  Processing Pipeline:                                               │
│    PDF → Docling Parse → JSONL Chunks → LanceDB (RAG Index)        │
│  Query: User Query → Embedding → Hybrid Search → LLM Tutor         │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **PDF Ingestion**: Raw PDFs stored in `PDFs/` directory
2. **Document Parsing**: Docling converts PDFs to structured JSON/Markdown with layout analysis
3. **Chunking**: Hybrid chunker creates chunks with structural metadata
4. **Vector Indexing**: Embeddings generated via `sentence-transformers` and stored in LanceDB
5. **Query Processing**: User queries use hybrid search (vector + FTS) to retrieve relevant chunks
6. **LLM Tutoring**: Retrieved content sent to LLM with instructions for academic tutoring

---

## Core Modules

### `main.py`
- **Purpose**: Interactive CLI entry point for textbook tutoring sessions
- **Key Features**:
  - Selects PDF from `PDFs/` directory
  - Handles TOC parsing and page range mapping
  - Manages `chapters_sections.json` for persistent page range storage
- **User Workflow**: Interactive prompts guide user through PDF selection and page range configuration

### `pdf_skills.py`
- **Purpose**: Core PDF processing using `pymupdf` (fitz)
- **Key Classes/Methods**:
  - `PDF(file_path)`: Main PDF handler
  - `parsePageText(page_range)`: Extracts text from page ranges
  - `parsePageImgVec(page_range)`: Extracts images AND vector graphics
  - `parsePageContentsLLM1(page_range)`: Combined text + image extraction for LLM
  - `parsePagesAsImgs(page_range)`: Renders pages as images
  - `parsePagesAsPDF(page_range)`: Extracts pages as base64-encoded PDF
- **Vector Graphics Extraction**: The `__getVectorData()` method is notable - it reconstructs vector paths (lines, rectangles, bezier curves) from PDF drawings and renders them as high-DPI images with nearby text labels

### `llm_parser.py`
- **Purpose**: LLM-based content parsing with structured outputs
- **Key Functions**:
  - `LLMParseTOC(content, page_offset)`: Parses Table of Contents, outputs `TOCStructure` with chapters/sections and page ranges. Uses Qwen3-30B model.
  - `LLMParseVisuals(content)`: Describes images/diagrams for text-only LLM tutoring. Uses GLM-5.3-Flash (vision-capable).
- **Key Pattern**: Uses `llm_client.get_completion()` with `response_format` parameter for structured JSON output

### `llm_client.py`
- **Purpose**: LiteLLM wrapper for LLM completions
- **Key Features**:
  - Supports multiple models (Qwen3-30B for text, GLM-5.3-Flash for vision)
  - Configuration via `settings.py`
  - Task-type specific model selection (none, parse_toc, parse_visuals)
  - Supports both `chat.completions.create()` and `chat.completions.parse()` for structured outputs
- **Configuration**: Uses LiteLLM proxy at `http://lytellm.markhor-altair.ts.net:4000`

### `schemas.py`
- **Purpose**: Pydantic models for structured LLM outputs
- **Models**:
  - `TOCStructure`: Chapters with sections, each with name and start page
  - `Pages`: Dict of page numbers to visual element descriptions

### `db_query.py`
- **Purpose**: RAG query interface using LanceDB
- **Key Function**:
  - `query_textbook_rag(db_uri, table_name, user_query, response_format)`: Performs hybrid search (vector + FTS) and formats results for LLM
  - Supports `sequential` and `formatted` response formats
  - Includes picture references when relevant
- **Embeddings**: Uses `sentence-transformers` with `Qwen/Qwen3-Embedding-0.6B`

### `db_populate.py`
- **Purpose**: Builds LanceDB index from Docling JSONL chunks
- **Key Function**:
  - `populate_lancedb_from_jsonl(jsonl_path, db_uri, table_name)`: Reads chunks, generates embeddings, creates FTS index
  - Uses `parseChunk()` to extract item metadata (char ranges, page ranges, picture references)
- **Output Table**: `textbook_chunks` with vector column and FTS index on `text`

### `pdf_to_json.py`
- **Purpose**: Docling-based PDF analysis for layout understanding
- **Key Function**:
  - `process_textbook_with_docling(pdf_path, output_dir)`: Generates Markdown, structured JSON, and extracts figure metadata with bounding boxes

### `pdf_to_jsonl.py`
- **Purpose**: Docling-based hybrid chunking for RAG
- **Key Function**:
  - `export_chunks_to_jsonl(pdf_path)`: Converts PDF to structured chunks with Docling's `HybridChunker`
  - Exports to `lancedb_chunks/{pdf_name}/raw_hybrid.jsonl`
  - Extracts images to `lancedb_chunks/{pdf_name}/pictures/`
  - Uses `ResultPostprocessor` for hierarchical structure correction

### `minoslib.py`
- **Purpose**: Utility functions
- **Key Function**:
  - `validateInput(validation, user_prompt)`: Re-prompts user until input passes validation

### `settings.py`
- **Purpose**: Application configuration
- **Configuration**:
  - `LITELLM_BASE_URL`, `LITELLM_API_KEY`, timeout, max_retries
  - `LANCEDB_CHUNKS_PATH`, `LANCEDB_DATA_PATH`

---

## Data Structures

### TOC Structure (from `LLMParseTOC`)
```json
{
  "chapters": {
    "1": {
      "name": "Chapter Name",
      "start": 1,
      "end": 50,
      "sections": {
        "1": {"name": "Section 1", "start": 1, "end": 10},
        "2": {"name": "Section 2", "start": 11, "end": 25}
      }
    }
  }
}
```

### Chunk Structure (from Docling JSONL)
```json
{
  "text": "Chunk text content...",
  "meta": {
    "schema_name": "docling_core.transforms.chunker.DocMeta",
    "doc_items": [
      {
        "self_ref": "#/texts/5",
        "parent": {"$ref": "#/texts/4"},
        "children": [],
        "content_layer": "body",
        "label": "text",
        "prov": [{"page_no": 1, "bbox": {...}, "charspan": [0, 13]}]
      }
    ]
  }
}
```

### LanceDB Record
```json
{
  "vector": [0.1, -0.3, ...],
  "text": "Chunk text content...",
  "items": [
    {
      "label": "text",
      "char_range": [0, 100],
      "page_range": [1, 5],
      "picture_ref": "pic_12.png"
    }
  ]
}
```

---

## Workflow

### First-Time Setup

1. **Add PDFs**: Place PDF files in `PDFs/` directory
2. **Parse Document**: Run `pdf_to_jsonl.py` to generate chunks
3. **Build Index**: Run `db_populate.py` to create LanceDB index

### Interactive Usage (via `main.py`)

1. Run `python main.py`
2. Select PDF from `PDFs/` directory
3. If TOC not cached, provide page range for TOC
4. Page ranges are saved in `chapters_sections.json` for future use

### RAG Query Usage (via `db_query.py`)

```python
from db_query import query_textbook_rag

results = query_textbook_rag(
    db_uri="lancedb_data",
    table_name="textbook_chunks",
    user_query="What is dynamic programming?",
    response_format="sequential"
)
```

---

## Configuration

### Model Selection (in `llm_client.py`)

| Task Type | Model | Use Case |
|-----------|-------|----------|
| `none` | qwen3-30b-a3b-instruct-2507 | General completions |
| `parse_toc` | qwen3-30b-a3b-instruct-2507 | TOC parsing with schema |
| `parse_visuals` | glm-5.3-flash | Vision + text tasks |

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `PDFs/` | Source PDF files |
| `lancedb_chunks/{pdf_name}/` | Docling output (JSONL, images) |
| `lancedb_data/` | LanceDB vector index |
| `chapters_sections.json` | Cached TOC/page ranges |
| `temp/` | Temporary rendered images |

---

## Adding New Features

### For Future Agents

When adding new features or making changes, update this file with:

1. **Feature Name**: Clear, concise title
2. **Purpose**: What the feature does
3. **Implementation Details**:
   - Files modified/created
   - Key functions/classes added
   - Data structures used
4. **Usage Example**: How to use the new feature
5. **Configuration**: Any new settings or model changes

**Format Example**:
```markdown
### `feature_name.py` (New File)
- **Purpose**: Brief description
- **Key Functions**:
  - `new_function()`: What it does
- **Usage**: Example code
```

---

## Notes for Agents

1. **State Management**: The system uses persistent JSON files (`chapters_sections.json`) for cached results. Check for existing data before re-computing.

2. **Hybrid Search**: LanceDB uses both vector embeddings and FTS. This is crucial for textbook content where precise terminology matters.

3. **Visual Content**: The system has sophisticated image/vector extraction. Consider using `parsePageImgVec()` when images are relevant.

4. **Structured Outputs**: Prefer using `llm_client.get_completion()` with `response_format` parameter over raw text - it ensures consistent JSON structure.

5. **Page Indexing**: The code uses 1-indexed pages in user-facing APIs but 0-indexed internally (see `PageRange` class in `pdf_skills.py`).

6. **Embedding Model**: Uses `Qwen/Qwen3-Embedding-0.6B` - a small but effective embedding model loaded in `db_populate.py`.

7. **Document Structure**: Docling's hierarchical postprocessor (`ResultPostprocessor`) corrects heading levels and document structure. Always run this when processing PDFs.

---

## Testing Checklist

- [ ] PDF loads without errors (`pymupdf` install check)
- [ ] TOC parses correctly and saves to JSON
- [ ] Chunks generate successfully (`docling` install check)
- [ ] Embeddings compute without errors (`sentence-transformers` check)
- [ ] LanceDB index created and searchable
- [ ] Hybrid queries return relevant results
- [ ] LLM responses follow expected format

---

*This file documents the codebase as of 2026-10-14. Last updated by initial agent analysis.*
