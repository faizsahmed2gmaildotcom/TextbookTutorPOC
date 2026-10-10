import os.path
from typing import Any

import minoslib


class PageRange:
    """
    Creates a 1-indexed range of pages.
    __iter__ returns a 0-indexed range.
    """

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def __bool__(self):
        return self.end >= self.start

    def __iter__(self):
        if not bool(self):
            raise IndexError(f"{PageRange.__name__} instance: end cannot be before start")
        return iter(range(self.start - 1, self.end))

    def __str__(self) -> str:
        return f"[{self.start}, {self.end}]"


class PDF:
    def __init__(self, file_path: str):
        import pymupdf
        self.name = os.path.split(file_path)[1]
        self.pdf = pymupdf.open(file_path)
        self.toc: PageRange = PageRange(0, -1)

    def __getTOCPageRange(self):
        import re
        input_range = minoslib.validateInput(lambda val: bool(re.fullmatch(r"[0-9]*-[0-9]*", val)),
                                             "Enter a page range 'from-to': ")
        start, end = (int(val) for val in input_range.split('-'))
        self.toc = PageRange(start, end)

    def getTOCPageContents(self):
        if not self.toc:
            self.__getTOCPageRange()
        return self.parsePageContentsLLM(self.toc)

    def __getImgData(self, img_list) -> list:
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

    def getPageContents(self, page_range: PageRange) -> tuple[list[str], list[bytes]]:
        extracted_text = []
        extracted_imgs = []
        for page in page_range:
            extracted_text.append(f"--- Page {page} ---\n{self.pdf[page].get_text()}")
            img_list = self.pdf[page].get_images(full=True)
            extracted_imgs = self.__getImgData(img_list)

        return extracted_text, extracted_imgs

    def parsePageContentsLLM(self, page_range: PageRange) -> list[dict[str, Any]]:
        extracted_text, extracted_imgs = self.getPageContents(page_range)
        return [
            {"type": "text", "text": '\n\n'.join(extracted_text)},
            *extracted_imgs
        ]


if __name__ == '__main__':
    import asyncio
    from llm_client import client

    my_pdf = PDF('PDFs/Skiena-The_Algorithm_Design_Manual.pdf')
    print(my_pdf.parsePageContentsLLM(PageRange(147, 147)))

    print(asyncio.run(client.get_completion(task_type="none",
                                            messages=[{
                                                "role": "system",
                                                "content": """The user will upload the extracted contents of a slice of a PDF. In the slice,
                                                text is represented as raw text and all other objects (images, vectors, etc.) are represented as images.
                                                Based on the uploaded text and the images, return the visual contents of each image."""
                                            }, {
                                                "role": "user",
                                                "content": my_pdf.parsePageContentsLLM(PageRange(147, 147))
                                            }],
                                            )))
