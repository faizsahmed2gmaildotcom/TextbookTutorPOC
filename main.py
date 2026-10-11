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
    visuals = JSONFile('visuals.json')

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

    if not pdf.name in visuals.data:
        visuals.data.update({pdf.name: {}})
        visuals.save()

    book_page_ranges = global_page_ranges.data[pdf.name]

    selected_chapter = minoslib.validateInput(lambda n: n in book_page_ranges["chapters"], "Select a chapter: ")
    print(f"Selected chapter: \"{selected_chapter} - {book_page_ranges["chapters"][selected_chapter]["name"]}\"")

    # Get page range to analyze from user
    selected_section = minoslib.validateInput(lambda n: True if (n == '0') else (n in book_page_ranges["chapters"][selected_chapter]["sections"]),
                                              "Select a section, or enter 0 to select the entire chapter: ")
    if selected_section != '0':
        print(
            f"Selected section: \"{selected_chapter}.{selected_section} - "
            f"{book_page_ranges["chapters"][selected_chapter]["sections"][f"{selected_chapter}.{selected_section}"]["name"]}\""
        )
        page_range = pdf_skills.PageRange(
            range(book_page_ranges["chapters"][selected_chapter]["sections"][f"{selected_chapter}.{selected_section}"]["start"] - 1,
                  book_page_ranges["chapters"][selected_chapter]["sections"][f"{selected_chapter}.{selected_section}"]["end"]))
    else:
        page_range = pdf_skills.PageRange(
            range(book_page_ranges["chapters"][selected_chapter]["start"] - 1,
                  book_page_ranges["chapters"][selected_chapter]["end"]))

    print(page_range)

    # Update visuals.json
    pages_missing_visuals = pdf_skills.PageRange(set(page_range) - set(int(p) - 1 for p in visuals.data[pdf.name]))
    if pages_missing_visuals:
        pdf_missing_visuals = pdf.parsePagesAsPDF(pages_missing_visuals)
        new_visuals_data = llm_parser.LLMParseVisuals(pdf_missing_visuals)
        visuals.data[pdf.name].update(new_visuals_data.get("pages", new_visuals_data))
        visuals.save()

    # Finally, we have all the data needed to start a chatbot session!
