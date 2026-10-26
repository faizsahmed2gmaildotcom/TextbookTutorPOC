import json, os
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat, DocItemLabel
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingSerializerProvider, ChunkingDocSerializer
from docling_core.transforms.serializer.markdown import MarkdownParams
from hierarchical.postprocessor import ResultPostprocessor
from logging_config import get_logger

logger = get_logger(__name__)


def get_docling_converter() -> DocumentConverter:
    """Factory for a DocumentConverter configured for PDF processing."""
    pipeline_options = PdfPipelineOptions(
        generate_page_images=True,
        images_scale=2.0,
        do_formula_enrichment=False  # Enable only on high VRAM machines
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def convert_pdf(pdf_path: str, page_range=None):
    """
    Convert a PDF to a DoclingDocument.

    Args:
        pdf_path: Path to the PDF file.
        page_range: Optional (start, end) tuple. If None, processes the full PDF.

    Returns:
        The DoclingDocument result.
    """
    converter = get_docling_converter()
    logger.info(f"Converting PDF: {pdf_path}" + (f" (pages {page_range[0]}-{page_range[1]})" if page_range else " (full document)"))
    if page_range:
        result = converter.convert(pdf_path, page_range=page_range)
    else:
        result = converter.convert(pdf_path)
    ResultPostprocessor(result).process()
    return result.document


def export_chunks_to_jsonl(doc, output_dir: str):
    """
    Export chunks and images from an already-converted DoclingDocument to JSONL.

    Args:
        doc: The DoclingDocument from convert_pdf().
        output_dir: Directory to write raw_hybrid.jsonl, raw_items.txt, and pictures.
    """
    os.makedirs(os.path.join(output_dir, "pictures"), exist_ok=True)

    class PictureTraversingSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc):
            # Enable traverse_pictures to process text/OCR content nested inside pictures
            params = MarkdownParams(traverse_pictures=True, compact_tables=True)
            return ChunkingDocSerializer(doc=doc, params=params)

    hybrid_chunker = HybridChunker(merge_peers=True)
    logger.info(f"Exporting chunks to {output_dir}...")

    ref_to_text = {elem.self_ref: elem.text for elem in doc.texts}

    with open(os.path.join(output_dir, "raw_hybrid.jsonl"), "w", encoding="utf-8") as f:
        # Text chunks
        for chunk in hybrid_chunker.chunk(doc):
            f.write(json.dumps(chunk.export_json_dict(), ensure_ascii=False) + "\n")

        # Picture chunks
        for pic in doc.pictures:
            pic_idx = pic.self_ref.split('/')[-1]
            pic_pil = pic.get_image(doc)
            pic_name = f"pic_{pic_idx}.png"
            if pic_pil:
                img_path = os.path.join(output_dir, "pictures", pic_name)
                pic_pil.save(img_path)

            pic_chunk = {
                "text": pic.caption_text(doc),
                "meta": {"doc_items": [pic.model_dump()], "headings": [ref_to_text[pic.parent.cref]]},
                "pic_id": pic_idx
            }
            # Extract parent, children, and captions as text (ts kinda sucks)
            # pic_chunk_texts: list[str] = ["- CAPTION -" + pic.caption_text(doc), "- OCR CONTENTS -"]
            # for child in pic.children:
            #     pic_chunk_texts.append(ref_to_text[child.cref])
            # pic_chunk["text"] = '\n'.join(pic_chunk_texts).strip()
            f.write(json.dumps(pic_chunk) + '\n')

    logger.info(f"Chunking and JSONL export complete for {output_dir}")


def detect_toc_pages(doc) -> tuple:
    """
    Auto-detect Table of Contents pages from a DoclingDocument.

    Finds all doc_items with label="document_index", collects their page numbers,
    and computes the page_offset (first page after the ToC).

    Returns:
        (toc_pages, page_offset) where toc_pages is a set of page numbers (1-indexed)
        or None if no ToC detected, and page_offset is the first page after the ToC.
    """
    toc_pages = set()
    for item, _level in doc.iterate_items():
        if item.label == DocItemLabel.DOCUMENT_INDEX:
            if hasattr(item, 'prov') and item.prov:
                for prov in item.prov:
                    toc_pages.add(prov.page_no)

    if not toc_pages:
        logger.warning("No document_index items found — cannot auto-detect ToC pages")
        return None, 0

    page_offset = max(toc_pages) + 1
    logger.info(f"Auto-detected ToC pages: {sorted(toc_pages)}, offset={page_offset}")
    return toc_pages, page_offset


def get_toc_pages_from_docling(doc) -> list[int]:
    """
    Auto-detect Table of Contents pages and return a sorted list.

    Finds all doc_items with label="document_index", collects their page numbers,
    and filters to keep only the first contiguous group of pages (excludes index).

    Returns:
        Sorted list of ToC page numbers (1-indexed).
    """
    toc_pages = set()
    for item, _level in doc.iterate_items():
        if item.label == DocItemLabel.DOCUMENT_INDEX:
            if hasattr(item, 'prov') and item.prov:
                for prov in item.prov:
                    toc_pages.add(prov.page_no)

    if not toc_pages:
        logger.warning("No document_index items found — cannot auto-detect ToC pages")
        return []

    # Filter to keep only the first contiguous group
    sorted_toc_pages = sorted(toc_pages)
    sorted_toc_pages = _separate_toc_from_index(sorted_toc_pages)

    logger.info(f"Auto-detected ToC pages: {sorted_toc_pages}")
    return sorted_toc_pages


def _separate_toc_from_index(sorted_toc_pages: list[int]) -> list[int]:
    """
    Keeps only the first set of adjacent ToC pages (excludes the index).

    Args:
        sorted_toc_pages: Sorted list of page numbers.

    Returns:
        List with only the first contiguous group of pages.
    """
    if not sorted_toc_pages:
        return []

    i = 0
    prev_page = sorted_toc_pages[0] - 1
    toc_tolerance = 1

    while i < len(sorted_toc_pages):
        if (sorted_toc_pages[i] - prev_page) > toc_tolerance:
            sorted_toc_pages.pop(i)
            continue
        prev_page = sorted_toc_pages[i]
        i += 1

    return sorted_toc_pages


def get_toc_content_from_pymupdf(pdf_path: str, toc_pages: list[int]) -> str:
    """
    Get raw page content for ToC pages using pymupdf.

    Args:
        pdf_path: Path to the PDF file.
        toc_pages: List of ToC page numbers (1-indexed).

    Returns:
        Concatenated text from ToC pages with clear separators.
    """
    import pymupdf

    logger.info(f"Getting ToC content from pymupdf for pages {toc_pages}")
    doc = pymupdf.open(pdf_path)

    toc_texts = []
    for page_num in toc_pages:
        page = doc[page_num - 1]  # pymupdf is 0-indexed
        text = page.get_text()
        toc_texts.append(text)

    doc.close()
    result = "\n".join(toc_texts)
    logger.info(f"Extracted ToC content: {len(result)} chars from {len(toc_pages)} pages")
    return result


if __name__ == "__main__":
    pdf_path = "PDFs/guide-t-cp.pdf"
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join("lancedb_chunks", pdf_name)
    doc = convert_pdf(pdf_path)
    export_chunks_to_jsonl(doc, output_dir)
