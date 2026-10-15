import pymupdf


def extract_native_pdf_toc(pdf_path):
  doc = pymupdf.open(pdf_path)
  # doc.get_toc() returns a list of [level, title, page_number]
  toc = doc.get_toc()

  structured_toc = []
  for item in toc:
    level, title, page = item
    structured_toc.append(
        {"level": level, "title": title.strip(), "page": page}
    )
  return structured_toc


# Example usage:
print("Skiena:")
print(extract_native_pdf_toc("PDFs/Skiena-The_Algorithm_Design_Manual.pdf"))
print()

print("Guide to CP:")
print(extract_native_pdf_toc("PDFs/guide-t-cp.pdf"))
print()
