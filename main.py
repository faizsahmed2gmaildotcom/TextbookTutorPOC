import json, os, logging
import pdf_skills, llm_parser
import minoslib

logger = logging.getLogger(__name__)


class JSONFile:
    def __init__(self, path: str):
        self.path = path
        self.data: dict = {}
        self.reload()

    def reload(self):
        with open(self.path, 'r+') as file:
            if not file.read():
                file.write("{}")
            file.seek(0)
            self.data = json.load(file)

    def save(self):
        with open(self.path, 'w') as file:
            json.dump(self.data, file)


if __name__ == '__main__':
    import re

    global_page_ranges = JSONFile('chapters_sections.json')

    pdf_names = set(os.listdir("PDFs"))
    pdf_name = minoslib.validateInput(lambda val: val in pdf_names,
                                      "Enter the name of the PDF to open in the PDFs dir: ")
    pdf = pdf_skills.PDF(os.path.join('PDFs', pdf_name))

    if not pdf.name in global_page_ranges.data:
        toc_query_content = pdf.getTOCPageContents()
        page_offset = int(minoslib.validateInput(lambda val: bool(re.fullmatch(r"[0-9]*", val)), "Enter a page offset: "))
        toc_parsed = llm_parser.LLMParseTOC(toc_query_content, page_offset)
        global_page_ranges.data.update({pdf.name: toc_parsed})
        global_page_ranges.save()

    book_page_ranges = global_page_ranges.data[pdf.name]
