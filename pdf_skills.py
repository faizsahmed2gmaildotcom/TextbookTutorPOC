import os.path
from typing import Any, Iterable

import minoslib
from logging_config import get_logger

logger = get_logger(__name__)


class PageRange:
    """
    Creates a 1-indexed range of pages.
    __iter__ returns a 0-indexed range.
    """

    def __init__(self, page_idxes: Iterable[int]):
        self.__page_idxes = set(page_idxes)

    def __iter__(self):
        return iter(self.__page_idxes)

    def __len__(self) -> int:
        return len(self.__page_idxes)

    def __bool__(self) -> bool:
        return bool(self.__page_idxes)

    def __str__(self) -> str:
        return f"[{', '.join(sorted(map(lambda p: str(p), self.__page_idxes)))}]"


class PDF:
    def __init__(self, file_path: str):
        import pymupdf
        self.name = os.path.split(file_path)[1]
        self.pdf = pymupdf.open(file_path)
        self.toc: PageRange = PageRange(set())

    def __getTOCPageRange(self):
        import re
        input_range = minoslib.validateInput(lambda val: bool(re.fullmatch(r"[0-9]*-[0-9]*", val)),
                                             "Enter a page range 'from-to': ")
        start, end = (int(val) for val in input_range.split('-'))
        self.toc = PageRange(range(start - 1, end))

    def getTOCPageContents(self):
        if not self.toc:
            self.__getTOCPageRange()
        return self.parsePageText(self.toc)

    def __getImgData(self, img_list) -> list[dict[str, Any]]:
        import base64
        extracted_imgs = []
        for img_info in img_list:
            base_img = self.pdf.extract_image(xref=img_info[0])
            img_encoded = base64.b64encode(base_img["image"]).decode("utf-8")
            img_ext = base_img["ext"]
            extracted_imgs.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{img_ext};base64,{img_encoded}"}
            })
        return extracted_imgs

    def __getVectorData(self, page_range: PageRange) -> list[dict[str, Any]]:
        import pymupdf, base64
        proximity_margin = 20.0
        base64_imgs = []

        for page_idx in page_range:
            drawings = self.pdf[page_idx].get_drawings()
            if not drawings:
                continue
            text_dict = self.pdf[page_idx].get_text("dict")

            filtered_spans = []
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_rect = pymupdf.Rect(span["bbox"])
                        is_near = False
                        for path in drawings:
                            path_rect = path["rect"]
                            expanded_rect = path_rect + (-proximity_margin, -proximity_margin, proximity_margin, proximity_margin)
                            if expanded_rect.intersects(span_rect):
                                is_near = True
                                break
                        if is_near:
                            filtered_spans.append(span)

            out_doc = pymupdf.open()
            out_page = out_doc.new_page(width=self.pdf[page_idx].rect.width, height=self.pdf[page_idx].rect.height)
            shape = out_page.new_shape()

            # Reconstruct the vector paths onto the blank page canvas
            for path in drawings:
                for item in path["items"]:
                    kind = item[0]
                    if kind == "l":  # Line
                        shape.draw_line(item[1], item[2])
                    elif kind == "re":  # Rectangle
                        shape.draw_rect(item[1])
                    elif kind == "c":  # Bézier curve
                        shape.draw_bezier(item[1], item[2], item[3], item[4])
                    elif kind == "qu":  # Quad
                        shape.draw_quad(item[1])

                # Apply original styling (strokes, fills, line widths)
                shape.finish(
                    color=path.get("color"),
                    fill=path.get("fill"),
                    width=path.get("width", 1),
                    closePath=path.get("closePath", False)
                )

            # Reconstruct the nearby text boxes/labels onto the canvas
            for span in filtered_spans:
                flags = span.get("flags", 0)
                font = span.get("font", "").lower()
                is_bold = bool(flags & 16) or "bold" in font or "bx" in font
                is_italic = bool(flags & 2) or "italic" in font or "sl" in font or "ti" in font or "oblique" in font
                is_mono = bool(flags & 8) or "mono" in font or "courier" in font or "tt" in font
                is_serif = bool(flags & 4) or "times" in font or "roman" in font or "cmr" in font

                if is_mono:
                    fontname = "cobi" if is_bold and is_italic else "cobo" if is_bold else "coit" if is_italic else "cour"
                elif is_serif:
                    fontname = "tibi" if is_bold and is_italic else "tibo" if is_bold else "tiit" if is_italic else "tiro"
                else:
                    fontname = "hebi" if is_bold and is_italic else "hebo" if is_bold else "heit" if is_italic else "helv"

                color = pymupdf.sRGB_to_pdf(span["color"]) if "color" in span else (0, 0, 0)
                try:
                    shape.insert_text(
                        pymupdf.Point(span["origin"]),
                        span["text"],
                        fontsize=span["size"],
                        fontname=fontname,
                        color=color
                    )
                except Exception:
                    shape.insert_text(
                        pymupdf.Point(span["origin"]),
                        span["text"],
                        fontsize=span["size"],
                        fontname="helv",
                        color=color
                    )

            # Commit the synthesized shapes/text to the new page
            shape.commit(overlay=True)

            # 4. Render the synthesized isolation layer to a high-res pixmap image
            pix = out_page.get_pixmap(dpi=200)
            os.makedirs("temp", exist_ok=True)
            img_bytes = pix.tobytes("png")

            encoded_img = base64.b64encode(img_bytes).decode("utf-8")

            base64_imgs.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_img}"}
            })

            out_doc.close()

        return base64_imgs

    def parsePageImgVec(self, page_range: PageRange) -> list[dict[str, Any]]:
        extracted_imgs = []
        for page in page_range:
            img_list = self.pdf[page].get_images(full=True)
            extracted_imgs += self.__getImgData(img_list) + self.__getVectorData(PageRange([page]))

        return extracted_imgs

    def convertPageToText(self, page_idx: int):
        return f"--- Page {page_idx + 1} ---\n{self.pdf[page_idx].get_text()}"

    def parsePageAsImg(self, page: int) -> dict[str, Any]:
        import base64
        img = self.pdf[page].get_pixmap(dpi=200)
        encoded_img = base64.b64encode(img.tobytes()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded_img}"}
        }

    def parsePageText(self, page_range: PageRange) -> list[dict[str, Any]]:
        extracted_text = '\n\n'.join(self.convertPageToText(page_idx) for page_idx in page_range)
        extracted_text.split()
        return [
            {"type": "text", "text": extracted_text}
        ]

    def parsePageContentsLLM1(self, page_range: PageRange) -> list[dict[str, Any]]:
        extracted_text = self.parsePageText(page_range)
        extracted_imgs = self.parsePageImgVec(page_range)
        return [
            *extracted_text,
            *extracted_imgs
        ]

    def parsePagesAsImgs(self, page_range: PageRange) -> list[dict[str, Any]]:
        return [self.parsePageAsImg(page) for page in page_range]

    def parsePagesAsPDF(self, page_range: PageRange) -> list[dict[str, Any]]:
        import pymupdf, base64
        with pymupdf.open() as stripped_pdf:
            for page_idx in page_range:
                stripped_pdf.insert_pdf(self.pdf, from_page=page_idx, to_page=page_idx)
            encoded_pdf = base64.b64encode(stripped_pdf.tobytes()).decode("utf-8")

        return [{
            "type": "file",
            "filename": f"{self.name} | Pages: {page_range}",
            "file_data": f"data:application/pdf;base64,{encoded_pdf}"
        }]


if __name__ == '__main__':
    import asyncio
    from llm_client import client

    my_pdf = PDF('PDFs/Skiena-The_Algorithm_Design_Manual.pdf')
    pdf_content = my_pdf.parsePageContentsLLM1(PageRange((132, 132)))

    result = asyncio.run(client.get_completion(messages=[{
        "role": "system",
        "content": """The user will upload a slice of a PDF. With as much detail as possible, describe only all images and diagrams. Optimize your result to 
        be uploaded to a text-only LLM, whose job is to interpret and academically tutor the user (using your image/diagram descriptions; all remaining 
        text on the page will be parsed and uploaded separately)."""
    }, {
        "role": "user",
        "content": pdf_content
    }],
    ))

    with open('temp/latest.md', 'w') as log:
        log.write(result)
