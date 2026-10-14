import os
from docling.document_converter import DocumentConverter


def process_textbook_with_docling(pdf_path: str, output_dir: str = "./output_corpus"):
    """
    Parses a PDF using Docling, exports canonical Markdown and JSON files,
    and extracts all layout figures with their bounding box coordinates.
    """
    os.makedirs(output_dir, exist_ok=True)
    converter = DocumentConverter()

    print(f"Converting and analyzing layout for: {pdf_path}...")
    result = converter.convert(pdf_path)
    doc = result.document
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Export canonical Markdown (for human inspection / indexing)
    md_path = os.path.join(output_dir, f"{base_name}.md")
    md_content = doc.export_to_markdown()
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved canonical Markdown to: {md_path}")

    # Export lossless structured JSON (durable parsed representation)
    json_path = os.path.join(output_dir, f"{base_name}.json")
    json_content = doc.export_to_dict()
    import json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_content, f, ensure_ascii=False, indent=2)
    print(f"Saved lossless structured JSON to: {json_path}")

    # Extract layout figures (images/diagrams) and their bounding boxes
    figures_dir = os.path.join(output_dir, "extracted_figures")
    os.makedirs(figures_dir, exist_ok=True)
    figure_metadata = []

    print(f"\nExtracting figures and bounding box references...")
    for idx, picture in enumerate(doc.pictures):
        img = picture.get_image(doc)
        img_filename = f"fig_{idx}.png"
        img_filepath = os.path.join(figures_dir, img_filename)

        if img:
            img.save(img_filepath)

        prov_records = []
        for prov in picture.prov:
            prov_records.append({
                "page": prov.page_no,
                "bbox": {
                    "x0": prov.bbox.l,
                    "y0": prov.bbox.t,
                    "x1": prov.bbox.r,
                    "y1": prov.bbox.b
                }
            })

        figure_metadata.append({
            "figure_index": idx,
            "image_path": img_filepath,
            "provenance": prov_records
        })
        print(f"• Captured Figure {idx} across {len(prov_records)} location(s)")

    meta_path = os.path.join(output_dir, "figures_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(figure_metadata, f, indent=2)
    print(f"Saved figure metadata manifest to: {meta_path}")


if __name__ == "__main__":
    process_textbook_with_docling("PDFs/Skiena-The_Algorithm_Design_Manual.pdf")
