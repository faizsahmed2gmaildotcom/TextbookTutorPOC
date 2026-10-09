import json, os, logging
import pdf_skills, toc_parser
import minoslib

logger = logging.getLogger(__name__)


class ChapterSectionRanges:
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
    global_page_ranges = ChapterSectionRanges('chapters_sections.json')

    pdf_names = set(os.listdir("PDFs"))
    pdf_name = minoslib.validateInput(lambda val: val in pdf_names,
                                      "Enter the name of the PDF to open in the PDFs dir: ")
    pdf = pdf_skills.PDF(os.path.join('PDFs', pdf_name))

    if not pdf.name in global_page_ranges.data:
        toc_raw = pdf.getTOCPageContents()
        toc_parsed = toc_parser.LLMParseTOC(toc_raw)
        global_page_ranges.data.update({pdf.name: json.loads(toc_parsed)})
        global_page_ranges.save()

    book_page_ranges = global_page_ranges.data[pdf.name]

    selected_chapter = minoslib.validateInput(lambda n: n in book_page_ranges["chapters"], "Select a chapter: ")
    print(f"Selected chapter: \"{selected_chapter} - {book_page_ranges["chapters"][selected_chapter]["name"]}\"")

    selected_section = minoslib.validateInput(lambda n: True if (n == '0') else (n in book_page_ranges["chapters"][selected_chapter]["sections"]),
                                              "Select a section, or enter 0 to select the entire chapter: ")
    if selected_section != '0':
        print(
            f"Selected section: \"{selected_chapter}.{selected_section} - {book_page_ranges["chapters"][selected_chapter]["sections"][selected_section]["name"]}\"")
        page_range = pdf_skills.PageRange(book_page_ranges["chapters"][selected_chapter]["sections"][selected_section]["start"],
                                          book_page_ranges["chapters"][selected_chapter]["sections"][selected_section]["end"])
    else:
        page_range = pdf_skills.PageRange(book_page_ranges["chapters"][selected_chapter]["start"],
                                          book_page_ranges["chapters"][selected_chapter]["end"])

    print(page_range)
