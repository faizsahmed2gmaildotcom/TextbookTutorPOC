import json, os
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from hierarchical.postprocessor import ResultPostprocessor


def export_chunks_to_jsonl(pdf_path: str):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join("lancedb_chunks", pdf_name)
    os.makedirs(os.path.join(output_dir, "pictures"), exist_ok=True)

    # 1. Convert the document
    pipeline_options = PdfPipelineOptions(
        generate_page_images=True,
        images_scale=2.0,  # Resolution
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    print(f"Parsing document for chunking: {pdf_path}")
    result = converter.convert(pdf_path, page_range=(1, 30))
    ResultPostprocessor(result).process()
    doc = result.document

    # 2. Initialize Docling's chunkers
    hybrid_chunker = HybridChunker()

    # 3. Serialize chunks and write to .JSONL
    print(f"Exporting chunks to JSONL")
    with open(os.path.join(output_dir, "raw_hybrid.jsonl"), "w", encoding="utf-8") as f:
        for chunk in hybrid_chunker.chunk(doc):
            # Export each chunk as a single JSON line
            f.write(json.dumps(chunk.export_json_dict(), ensure_ascii=False) + "\n")
    # Potentially swap to using doc.iterate_items() natively?
    with open(os.path.join(output_dir, "raw_items.txt"), "w", encoding="utf-8") as f:
        for item in doc.iterate_items():
            # Write each item as a single line
            f.write(json.dumps(str(item), ensure_ascii=False) + "\n")

    # 4. Extract images; send parent images along with text bodies
    for pic in doc.pictures:
        pic_idx = pic.self_ref.split('/')[-1]
        pic_pil = pic.get_image(doc)

        if pic_pil:
            img_path = os.path.join(output_dir, "pictures", f"pic_{pic_idx}.png")
            pic_pil.save(img_path)

    print("Chunking and JSONL export complete!")


if __name__ == "__main__":
    export_chunks_to_jsonl("PDFs/Skiena-The_Algorithm_Design_Manual.pdf")
