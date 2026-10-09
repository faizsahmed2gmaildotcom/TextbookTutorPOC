import os.path
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
            raise IndexError(f"{PageRange.__name__} instance: end is earlier than start!")
        return iter(range(self.start - 1, self.end))

    def __str__(self) -> str:
        return f"[{self.start}, {self.end}]"


class PDF:
    def __init__(self, file_path: str):
        from pypdf import PdfReader
        self.name = os.path.split(file_path)[1]
        self.pypdf_reader = PdfReader(file_path)
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
        return self.getPageContents(self.toc)

    def getPageContents(self, page_range: PageRange) -> str:
        extracted_text = ""
        for page in page_range:
            extracted_text += self.pypdf_reader.pages[page].extract_text() + "\n\n"
        extracted_text.strip()

        return extracted_text
