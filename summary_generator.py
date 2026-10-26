"""
Offline chapter/section summary generation.

After chunks are enriched with chapter/section metadata in LanceDB,
this module generates concise summaries for each chapter and section using an LLM.

The generation is hierarchical:
1. First, generate section summaries using ALL chunks in that section
2. Then, generate chapter summaries using ALL section summaries + chapter-level chunks

Summaries are stored directly in chapters_sections.json for runtime agentic orchestration.
"""

import json, os
from logging_config import get_logger
from orchestrator import Skills

logger = get_logger(__name__)


async def generate_summaries(
        skills: Skills,
        chapters_sections_path: str
) -> dict:
    """
    Generate summaries for all chapters and sections in chapters_sections.json.

    The generation is hierarchical:
    1. First generate section summaries using ALL chunks in that section
    2. Then generate chapter summaries using ALL section summaries + chapter-level chunks

    Summaries are stored directly in chapters_sections.json under each chapter/section.
    """
    from command_parser import parse_query
    from llm_client import client

    # Load chapters_sections.json
    if not os.path.exists(chapters_sections_path):
        logger.error(f"chapters_sections.json not found at {chapters_sections_path}")
        return {}

    with open(chapters_sections_path) as f:
        chapters_sections = json.load(f)

    # Process each PDF file in the chapters_sections
    for pdf_name, pdf_data in chapters_sections.items():
        logger.info(f"Generating summaries for {pdf_name}")

        chapters = pdf_data.get("chapters", {})
        chapter_ids: list[int] = sorted(chapters.keys(), key=lambda c: int(c) if c.isdigit() else c)

        # First pass: Generate section summaries for all sections
        for c_id in chapter_ids:
            chapter = chapters[c_id]
            sections = chapter.get("sections", {})

            if not sections:
                logger.info(f"  Chapter {c_id}: No sections found")
                continue

            logger.info(f"  Chapter {c_id}: Generating section summaries for {len(sections)} sections")

            for s_id, section in sections.items():
                sec_name = section.get("name", f"Section {s_id}")
                if section.setdefault("summary", ""):
                    logger.info(f"    Summary already present for Section {c_id}.{s_id}; skipping...")
                    continue

                logger.info(f"    Generating summary for Section {c_id}.{s_id}: {sec_name}")

                # Retrieve ALL chunks in this section using .where() filter
                parsed = parse_query(f"/section {c_id}.{s_id}")

                try:
                    section_contents = skills.getSectionRawContents([(c_id, s_id)])

                    response = await client.get_completion(
                        messages=[
                            {"role": "system",
                             "content": "Summarize this textbook section in 1-3 sentences. Include key concepts and important details. Be thorough but concise."},
                            {"role": "user", "content": section_contents}
                        ],
                        task_type="none"
                    )
                    sec_summary = response.strip()
                except Exception as e:
                    logger.error(f"Failed to generate summary for Section {c_id}.{s_id}: {e}")
                    sec_summary = f"Summary generation failed: {e}"

                # Store section summary directly in chapters_sections.json
                section["summary"] = sec_summary

        # Second pass: Generate chapter summaries using section summaries + chapter-level chunks
        for c_id in chapter_ids:
            chapter = chapters[c_id]
            ch_name = chapter.get("name", f"Chapter {c_id}")
            if chapter.setdefault("summary", ""):
                logger.info(f"Summary already present for Chapter {c_id}; skipping...")
                continue

            sections = chapter.get("sections", {})

            logger.info(f"Generating chapter summary for Chapter {c_id}: {ch_name}")

            # Gather all section summaries for this chapter
            section_summaries = []
            for s_id, section in sections.items():
                sec_name = section.get("name", f"Section {s_id}")
                sec_summary = section.get("summary", "")
                section_summaries.append(f"Section {s_id} ({sec_name}): {sec_summary}")

            # Retrieve chapter-level chunks (chunks not in any section)
            try:
                results = (
                    skills.lancedb_table.search()
                    .where(f"chapter_id = '{c_id}' AND section_id = ''")
                    .to_pandas()
                )
                chunk_texts = []

                for item in results:
                    if isinstance(item, dict) and item.get("type") == "text":
                        chunk_texts.append(item.get("text", ""))

                # Use ALL chapter-level chunks
                chapter_chunks = "\n\n".join(chunk_texts)
            except Exception as e:
                logger.error(f"Failed to retrieve chunks for Chapter {c_id}: {e}")
                chapter_chunks = ""

            # Combine section summaries and chapter chunks
            all_content = ""
            if section_summaries:
                all_content += "Section Summaries:\n" + "\n---\n".join(section_summaries) + "\n\n"
            if chapter_chunks:
                all_content += "Chapter Content:\n" + chapter_chunks

            if all_content.strip():
                response = await client.get_completion(
                    messages=[
                        {"role": "system",
                         "content": "Summarize this textbook chapter in 3-5 sentences. Include key topics, concepts, and their relationships. Be comprehensive."},
                        {"role": "user", "content": f"Chapter: {ch_name}\n\n{all_content}"}
                    ],
                    task_type="none"
                )
                summary = response.strip()
            else:
                summary = f"No content available for {ch_name}"

            # Store chapter summary directly in chapters_sections.json
            chapter["summary"] = summary

        # Save updated chapters_sections.json
        with open(chapters_sections_path, "w") as f:
            json.dump(chapters_sections, f, indent=2)

        logger.info(f"Saved summaries to {chapters_sections_path}")

    return chapters_sections


def load_summaries(summaries_path: str) -> dict:
    """Load chapter/section summaries from chapters_sections.json."""
    if not os.path.exists(summaries_path):
        logger.warning(f"chapters_sections.json not found: {summaries_path}")
        return {}
    with open(summaries_path) as f:
        return json.load(f)
