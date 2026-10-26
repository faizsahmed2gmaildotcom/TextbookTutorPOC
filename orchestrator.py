import os
from typing import Any
from langchain_openai import ChatOpenAI
from logging_config import get_logger
from settings import settings

logger = get_logger(__name__)


class Skills:
    from lancedb import Table
    def __init__(self, book_name: str, book_data: dict[str, Any], lancedb_table: Table):
        self.book_name = book_name
        self.book_data = book_data
        self.lancedb_table = lancedb_table

    def getChapterSummaries(self) -> str:
        logger.info(f"Getting chapter summaries for {self.book_name}")

        extracted_summaries: list[str] = []

        for chapter_num, chapter_meta in self.book_data["chapters"].items():
            extracted_summaries.append(f"--- \"{chapter_meta["name"]}\" : CHAPTER {chapter_num} : PAGES {chapter_meta["start"]}-{chapter_meta["end"]} ---\n,"
                                       f"{chapter_meta["summary"]}\n\n")

        return '\n'.join(extracted_summaries).strip()

    def getSectionSummaries(self, chapters: list[int]) -> str:
        logger.info(f"Getting section summaries for chapters {", ".join(map(lambda c: str(c), chapters))}")

        extracted_summaries: list[str] = []

        for chapter_num in (str(c) for c in chapters):
            if chapter_num not in self.book_data["chapters"]:
                logger.warning(f"Data for chapter {chapter_num} not found!")
                continue

            chapter_meta = self.book_data["chapters"][chapter_num]
            extracted_summaries.append(f"--- \"{chapter_meta["name"]}\" : CHAPTER {chapter_num} : PAGES {chapter_meta["start"]}-{chapter_meta["end"]} ---\n")
            for section_num, section_meta in chapter_meta["sections"].items():
                extracted_summaries.append(f"-- \"{section_meta["name"]}\" : SECTION {section_num} : PAGES {section_meta["start"]}-{section_meta["end"]} --\n"
                                           f"{section_meta["summary"]}\n")
            extracted_summaries.append("")

        return '\n'.join(extracted_summaries).strip()

    def getSectionRawContents(self, sections: list[tuple[int, int]]) -> list[dict[str, str]]:
        logger.info(f"Getting section contents for {", ".join(f"{s[0]}.{s[1]}" for s in sections)}")

        openai_parsed: list[dict[str, str]] = []

        for chapter, section in sections:
            # 1) Parse text per-page
            text_results = (
                self.lancedb_table.search()
                .where(f"chapter_id = '{chapter}' AND section_id = '{section}' AND pic_id = ''")
                .to_pandas()
                .sort_values("page_end")
            )

            section_start = section_end = -1
            section_name = ""
            page_text_contents: list[list[str]] = []
            last_header = ""
            last_page: int = -1

            for row in text_results.iterrows():
                if not section_name:
                    section_name = row[1]["section_name"]

                cur_page = row[1]["page_end"]
                if last_page != cur_page:
                    last_page = cur_page
                    page_text_contents.append([f"- PAGE {cur_page} -"])

                if row[0] == 0:
                    section_start = row[1]["page_start"]
                elif row[0] == (len(text_results) - 1):
                    section_end = row[1]["page_end"]

                if row[1]["header"].strip() and (row[1]["header"].strip() != last_header):
                    page_text_contents[-1].append("Header: " + row[1]["header"])
                    last_header = row[1]["header"].strip()

                page_text_contents[-1].append(row[1]["text"])

            if not section_name:
                raise IndexError(f"Section {chapter}.{section} not found!")

            # 2) Parse pictures per-page
            picture_results = (
                self.lancedb_table.search()
                .where(f"chapter_id = '{chapter}' AND section_id = '{section}' AND pic_id != ''")
                .to_pandas()
            )

            page_picture_contents: list[list[str]] = [[f"\n- PICTURES FOR PAGE {section_start + i} -"] for i in range(len(page_text_contents))]
            for row in picture_results.iterrows():
                cur_picture_page = page_picture_contents[row[1]["page_end"] - section_start]
                cur_picture_page.append(f"Picture ID: {row[1]["pic_id"]} | "
                                        f"Caption: {row[1]["text"] if row[1]["text"] else "N/A"}")

            for text, pic in zip(page_text_contents, page_picture_contents):
                if len(pic) > 1:
                    text += pic

            section_info = f"-- \"{section_name}\" : CHAPTER {chapter} : SECTION {section} : PAGES {section_start}-{section_end} --\n\n\n"
            openai_parsed.append({
                "type": "text",
                "text": section_info + "\n\n\n".join("\n".join(page).strip() for page in page_text_contents)
            })

        return openai_parsed

    def getPicturesByID(self, pic_ids: list[int]) -> list[dict[str, str]]:
        logger.info(f"Fetching pictures {', '.join(str(p_id) for p_id in pic_ids)}")

        import base64

        encoded_pics: list[str] = []
        for pic_id in pic_ids:
            with open(os.path.join(settings.LANCEDB_CHUNKS_PATH, os.path.splitext(self.book_name)[0], "pictures", f"pic_{pic_id}.png"), "rb") as f:
                encoded_pics.append(base64.b64encode(f.read()).decode("utf-8"))

        return [{"type": "image_url", "image_url": encoded_pic} for encoded_pic in encoded_pics]


def constructTools(skills: Skills):
    from langchain_core.tools import tool

    @tool
    def getChapterSummaries() -> str:
        """
        Fetches high-level summaries of all chapters in the textbook.
        Use this tool first to deduce which chapter(s) contain relevant context.
        """
        return skills.getChapterSummaries()

    @tool
    def getSectionSummaries(chapters: list[int]) -> str:
        """Fetches section-level summaries for the specified chapter numbers.

        Args:
            chapters: List of integer chapter numbers (e.g. [1, 2]).
        """
        # Enforce max_agent_chapters variable constraint
        capped_chapters = chapters[:settings.MAX_AGENT_CHAPTERS]
        return skills.getSectionSummaries(capped_chapters)

    @tool
    def getSectionRawContents(sections: list[tuple[int, int] | list[int]]) -> list[dict[str, str]]:
        """Fetches the full raw text content of specific textbook sections.

        Args:
            sections: List of (chapter_number, section_number) pairs, e.g. [[2, 2], [2, 3]].
        """
        # Convert JSON list pairs to tuples and enforce max_agent_sections constraint
        formatted_sections: list[tuple[int, int]] = [
            (int(item[0]), int(item[1])) for item in sections[:settings.MAX_AGENT_SECTIONS]
        ]
        return skills.getSectionRawContents(formatted_sections)

    @tool
    def getPicturesByID(picture_ids: list[int]) -> list[dict[str, str]]:
        """Fetches all pictures given their IDs, returning each picture in the same order in which they were requested.

        Args:
            picture_ids: List of pictures (by ID integers) to fetch. The returned list's order matches that of picture_ids', e.g. [0, 4] -> [picture 0 data, picture 4 data]
        """
        return skills.getPicturesByID(picture_ids)

    tools = [
        getChapterSummaries,
        getSectionSummaries,
        getSectionRawContents,
        getPicturesByID
    ]
    tools_by_name = {
        "getChapterSummaries": getChapterSummaries,
        "getSectionSummaries": getSectionSummaries,
        "getSectionRawContents": getSectionRawContents,
        "getPicturesByID": getPicturesByID
    }

    return tools, tools_by_name


def buildTextbookRagAgent(skills: Skills):
    """Constructs a tool-calling AgentExecutor configured with LiteLLM and constraints."""
    tools, tools_by_name = constructTools(skills)

    # Initialize ChatLiteLLM wrapper
    model_with_tools = ChatOpenAI(
        model="qwen3-coder-next:local",
        base_url=settings.LITELLM_BASE_URL,
        api_key=settings.LITELLM_API_KEY,
        temperature=0.0
    ).bind_tools(tools)

    # Construct the LangGraph agent
    from langchain.messages import AnyMessage
    from typing_extensions import TypedDict, Annotated
    import operator
    from langchain.messages import SystemMessage
    from langchain.messages import ToolMessage
    from typing import Literal
    from langgraph.graph import StateGraph, START, END

    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        llm_calls: int

    def llm_call(state: dict):
        return {
            "messages": [
                model_with_tools.invoke(
                    state["messages"][:-1] +
                    [
                        SystemMessage(
                            content="You are a textbook research assistant. Your task is to gather section-level textbook information "
                                    "to answer the user's question.\n\n"
                                    "Follow this multi-step procedure:\n"
                                    "1. Call `getChapterSummaries` to inspect all chapter summaries.\n"
                                    "2. Select the most relevant chapter numbers (at most {max_chapters} chapters) "
                                    "and request their section summaries using `getSectionSummaries`.\n"
                                    "3. Select the most relevant section numbers (at most {max_sections} sections) "
                                    "and request their raw text content using `getSectionRawContents`.\n"
                                    "   Optional: If and only if the user's query cannot be answered without analyzing a page's pictures, "
                                    "request that page's pictures using `getPicturesByID`.\n"
                                    "4. Answer the user's query thoroughly using ONLY the retrieved raw section contents."
                        ),
                        state["messages"][-1]
                    ]
                )
            ],
            "llm_calls": state.get('llm_calls', 0) + 1
        }

    def tool_node(state: dict):
        """Performs the tool call"""

        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}

    def should_continue(state: MessagesState) -> Literal["tool_node", END]:
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if last_message.tool_calls:
            return "tool_node"

        # Otherwise, we stop (reply to the user)
        return END

    # Build workflow
    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")

    # Compile the agent with context capabilities
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    local_storage = InMemorySaver()
    export_storage = InMemoryStore()
    agent = agent_builder.compile(checkpointer=local_storage, store=export_storage)

    # Show the agent
    # from IPython.display import Image, display
    # display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

    return agent


if __name__ == '__main__':
    from lancedb import connect
    from main import JSONFile

    chapters_sections = JSONFile("chapters_sections.json")
    book_data = chapters_sections.data["guide-t-cp.pdf"]
    table = connect(settings.LANCEDB_DATA_PATH).open_table("guide-t-cp.pdf")

    agent_skills = Skills("guide-t-cp.pdf", book_data, table)
    # print(agent_skills.getSectionRawContents([(4, 1), (4, 2), (4, 3)]))

    from langchain_core.messages import HumanMessage

    test_agent = buildTextbookRagAgent(agent_skills)
    result = test_agent.invoke(
        {"messages": [HumanMessage("Fig 4.4 can be found in section 4.2. What page is fig 4.4 on?")]},
        {"configurable": {"thread_id": "test"}}
    )

    print(result["messages"][-1].content)
