import json, os
import pdf_skills, llm_parser
import minoslib
from logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


class JSONFile:
    def __init__(self, path: str):
        self.path = path
        self.data: dict = {}
        self.reload()

    def reload(self):
        if not os.path.exists(self.path):
            with open(self.path, 'w'):
                pass

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
    from settings import settings
    from toc_parsers import TOCMode, get_toc_parser
    from pdf_to_jsonl import convert_pdf, get_toc_pages_from_docling, get_toc_content_from_pymupdf, export_chunks_to_jsonl

    global_page_ranges = JSONFile('chapters_sections.json')

    pdf_names = set(os.listdir("PDFs"))
    pdf_name = minoslib.validateInput(lambda val: val in pdf_names,
                                      "Enter the name of the PDF to open in the PDFs dir: ")
    pdf_path = os.path.join('PDFs', pdf_name)
    pdf = pdf_skills.PDF(pdf_path)

    if pdf.name not in global_page_ranges.data:
        logger.info(f"Processing new PDF: {pdf.name}")
        toc_mode = TOCMode(settings.TOC_MODE)
        logger.info(f"Using TOC mode: {toc_mode.value}")

        # Try to parse ToC with the selected mode and fallback chain
        toc_parsed = None
        try:
            parser = get_toc_parser(toc_mode)
            toc_parsed = parser.parse(pdf_path)
            logger.info(f"Successfully parsed ToC with {toc_mode.value} mode")
        except Exception as e:
            logger.error(f"TOC parsing with {toc_mode.value} mode failed: {e}")
            logger.info("Falling back to LLM mode...")
            try:
                parser = get_toc_parser(TOCMode.LLM)
                toc_parsed = parser.parse(pdf_path)
                logger.info("Successfully parsed ToC with LLM mode")
            except Exception as e:
                logger.error(f"LLM mode also failed: {e}")
                # Fallback to manual user input
                logger.info("Falling back to manual ToC page range input...")
                toc_query_content = pdf.getTOCPageContents()
                page_offset = int(minoslib.validateInput(
                    lambda val: val.isdigit(),
                    "Enter the page offset (first page after ToC): "
                ))
                toc_parsed = llm_parser.LLMParseTOC(toc_query_content, page_offset, pdf_path)
                logger.info("Successfully parsed ToC with manual input")

        global_page_ranges.data.update({pdf.name: toc_parsed})
        global_page_ranges.save()
        logger.info(f"Saved ToC to chapters_sections.json for {pdf.name}")
    else:
        logger.info(f"PDF {pdf.name} already has ToC data in chapters_sections.json")

    book_page_ranges = global_page_ranges.data[pdf.name]

    # Export chunks to JSONL (full document, no page limit)
    pdf_name_noext = os.path.splitext(pdf_name)[0]
    output_dir = os.path.join(settings.LANCEDB_CHUNKS_PATH, pdf_name_noext)
    jsonl_path = os.path.join(output_dir, "raw_hybrid.jsonl")

    if not os.path.exists(jsonl_path):
        logger.info(f"Chunks not found at {jsonl_path}, generating...")
        doc = convert_pdf(pdf_path)
        export_chunks_to_jsonl(doc, output_dir)
    else:
        logger.info(f"Chunks already exist at {jsonl_path}")

    # Populate LanceDB with ToC structure for enrichment
    from populate_lancedb import populate_lancedb_from_jsonl
    populate_lancedb_from_jsonl(
        jsonl_path=jsonl_path,
        db_uri=settings.LANCEDB_DATA_PATH,
        table_name="textbook_chunks",
        toc_structure=book_page_ranges
    )
    logger.info("LanceDB populated successfully")

    # Generate chapter and section summaries
    from summary_generator import generate_summaries
    logger.info("Generating chapter and section summaries...")
    awaitable = generate_summaries(
        db_uri=settings.LANCEDB_DATA_PATH,
        table_name="textbook_chunks",
        chapters_sections_path="chapters_sections.json"
    )
    import asyncio
    global_page_ranges.data = asyncio.run(awaitable)
    global_page_ranges.save()
    logger.info("Chapter and section summaries generated and saved to chapters_sections.json")

    # Initialize the LangGraph orchestrator for agentic query processing
    from orchestrator import create_orchestrator, orchestrate_query

    logger.info("Initializing LangGraph orchestrator...")
    orchestrator = create_orchestrator()
    logger.info("LangGraph orchestrator initialized successfully")

    # Interactive query loop using the LangGraph orchestrator
    print("\n" + "=" * 60)
    print(f"Textbook Tutor ready! Book: {pdf.name}")
    print("Commands: /chapter N, /section N.M, /pages from-to")
    print("Type /quit to exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Query: ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip() == "/quit":
            break
        if not user_input.strip():
            continue

        try:
            # Use the LangGraph orchestrator to process the query
            response = asyncio.run(orchestrate_query(
                user_query=user_input,
                db_uri=settings.LANCEDB_DATA_PATH,
                table_name="textbook_chunks",
                summaries_path="chapters_sections.json"
            ))
            print(f"\n{response}\n")
        except Exception as e:
            logger.error(f"Error in orchestrator: {e}")
            print(f"Error: {e}")
