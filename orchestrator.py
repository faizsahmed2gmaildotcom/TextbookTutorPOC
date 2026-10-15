"""
LangGraph-based multi-agent orchestration for the textbook tutor.

Implements a supervisor-agent architecture using LangGraph's StateGraph:
- supervisor_agent: Main coordinator that analyzes queries and delegates to subagents
- chapter_agent: Handles chapter-wide queries
- section_agent: Handles section-specific queries (most precise)
- pages_agent: Handles page-range queries

The supervisor uses handoff tools to delegate to specialized subagents,
and aggregates results before composing the final response.
"""

import json
from typing import Literal, Sequence, Callable, TypedDict, Any

from langchain.agents import create_agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Local imports
from settings import settings
from llm_client import client as llm_client
from summary_generator import load_summaries
from command_parser import parse_query
from db_query import query_textbook_rag


# === LLM Configuration ===

def get_llm():
    """Get the configured LLM for agents."""
    return llm_client


# === AGENT STATES ===

class AgentState(TypedDict):
    """State for the multi-agent system."""
    # Original user query (preserved throughout)
    query: str
    # Current subagent in charge (for routing)
    next: Literal["__end__", "chapter_agent", "section_agent", "pages_agent", "supervisor"]
    # Summary context loaded from chapters_sections.json
    summary_context: dict
    # Final response after aggregation
    response: str
    # Messages for conversation history (managed by langgraph)
    messages: list
    # Parsed command query (for filtering to specific chapter/section/pages)
    parsed_query: Any  # ParsedQuery from command_parser.py


# === HELPER FUNCTIONS ===

def load_summary_context(chapters_sections_path: str = "chapters_sections.json") -> dict:
    """
    Load summary context from chapters_sections.json.

    Returns a flattened dict mapping chapter/section IDs to their summaries
    for quick lookup by the supervisor.
    """
    summaries = load_summaries(chapters_sections_path)

    # Flatten the structure for easier lookup
    context = {}

    for pdf_name, pdf_data in summaries.items():
        chapters = pdf_data.get("chapters", {})
        for c_id, chapter in chapters.items():
            context[f"chapter_{c_id}"] = {
                "id": c_id,
                "name": chapter.get("name", f"Chapter {c_id}"),
                "summary": chapter.get("summary", ""),
                "sections": {}
            }

            # Flatten sections
            for s_id, section in chapter.get("sections", {}).items():
                context[f"section_{c_id}_{s_id}"] = {
                    "id": s_id,
                    "chapter_id": c_id,
                    "name": section.get("name", f"Section {c_id}.{s_id}"),
                    "summary": section.get("summary", "")
                }
                context[f"chapter_{c_id}"]["sections"][s_id] = context[f"section_{c_id}_{s_id}"]

    return context


def determine_next_agent(query: str, default: str = "section_agent") -> str:
    """Determine which agent should handle the query based on content."""
    # Check for explicit section reference
    if "/section" in query or "section" in query.lower():
        return "section_agent"

    # Check for explicit chapter reference
    if "/chapter" in query:
        return "chapter_agent"

    # Check for page range reference
    if "/pages" in query:
        return "pages_agent"

    # Default to section agent for precision
    return default


# === DATABASE QUERY TOOLS ===

def create_database_query_tool(parsed_query: Any):
    """Create a database query tool bound to a specific parsed query.

    Args:
        parsed_query: The parsed command query with chapter/section/pages filter

    Returns:
        A tool function that can be invoked by agents to query the textbook database
    """
    def query_tool(user_query: str) -> str:
        """Query the textbook database for relevant passages with the filter from parsed_query."""
        results = query_textbook_rag(
            db_uri="lancedb_data",
            table_name="textbook_chunks",
            user_query=user_query,
            response_format="sequential",
            parsed=parsed_query
        )
        texts = [r.get("text", "") for r in results if r.get("type") == "text"]
        return "\n\n".join(texts) if texts else "No relevant passages found."

    query_tool.name = "query_textbook"
    query_tool.description = "Query the textbook database for relevant passages. Use this to retrieve specific content from the textbook based on the parsed command (chapter, section, or pages filter)."

    return query_tool


# === AGENT PROMPTS ===

def get_supervisor_prompt() -> str:
    """System prompt for the supervisor agent."""
    return """You are a textbook tutor supervisor. Your role is to coordinate multiple specialized agents to answer the user's question about a textbook.

Available subagents:
- chapter_agent: For general questions about an entire chapter
- section_agent: For focused questions about a specific section (most precise)
- pages_agent: For questions about content on specific page ranges

Your workflow:
1. Analyze the user's query to identify the target chapter/section/pages
2. Decide which agent should handle the query based on the scope
3. After receiving results from subagents, aggregate them into a coherent answer
4. Cite sources (chapter/section names and page ranges) in your response

Return your final answer after analyzing the query and deciding on the appropriate agent."""


def get_chapter_agent_prompt(parsed=None) -> str:
    """System prompt for the chapter agent.

    Args:
        parsed: Optional parsed command query with chapter_id if specified

    Returns:
        The prompt string with chapter-specific instructions
    """
    base = """You are a chapter-level textbook tutor. Handle queries about entire chapters.

Your task:
1. Use the chapter summary (from the conversation context) to understand the chapter scope
2. Retrieve relevant chunks from the chapter using LanceDB
3. Provide comprehensive coverage of the topic within this chapter
4. Cite the chapter name and relevant page ranges in your response

If the query is about a specific section within the chapter, return "delegate_to: section_agent"
with a brief explanation, and the supervisor will delegate appropriately."""

    if parsed and hasattr(parsed, 'command') and parsed.command == "chapter" and hasattr(parsed, 'chapter_id') and parsed.chapter_id:
        return base + f"\n\nIMPORTANT: User specifically requested chapter {parsed.chapter_id}. Filter your search to this chapter only using the query_textbook tool."

    return base


def get_section_agent_prompt(parsed=None) -> str:
    """System prompt for the section agent.

    Args:
        parsed: Optional parsed command query with chapter_id and section_id if specified

    Returns:
        The prompt string with section-specific instructions
    """
    base = """You are a section-level textbook tutor. Handle queries about specific sections with high precision.

Your task:
1. Focus on the exact topic the user asked about
2. Use the section summary to understand what this section covers
3. Retrieve relevant chunks from this specific section using LanceDB
4. Provide a focused, precise answer
5. Cite the section name and exact page range in your response

This is the most precise agent - use it when the query targets a specific section."""

    if parsed and hasattr(parsed, 'command') and parsed.command == "section" and hasattr(parsed, 'chapter_id') and parsed.chapter_id and hasattr(parsed, 'section_id') and parsed.section_id:
        return base + f"\n\nIMPORTANT: User specifically requested section {parsed.chapter_id}.{parsed.section_id}. Filter your search to this section only using the query_textbook tool."

    return base


def get_pages_agent_prompt(parsed=None) -> str:
    """System prompt for the pages agent.

    Args:
        parsed: Optional parsed command query with page_from and page_to if specified

    Returns:
        The prompt string with page-range-specific instructions
    """
    base = """You are a page-range textbook tutor. Handle queries about content on specific page ranges.

Your task:
1. Use the page range (from the conversation context or query) to filter retrieved chunks
2. Provide targeted answers based on content within the specified page range
3. Cite the exact page numbers in your response

Use this agent when the query specifies page numbers or page ranges."""

    if parsed and hasattr(parsed, 'command') and parsed.command == "pages" and hasattr(parsed, 'page_from') and parsed.page_from is not None and hasattr(parsed, 'page_to') and parsed.page_to is not None:
        return base + f"\n\nIMPORTANT: User specifically requested pages {parsed.page_from}-{parsed.page_to}. Filter your search to this page range only using the query_textbook tool."

    return base


# === AGENT CREATION ===

def create_supervisor_agent():
    """Create the main supervisor agent that coordinates all subagents."""
    llm = ChatOpenAI(
        model="qwen3-coder-next:local",
        base_url=settings.LITELLM_BASE_URL,
        api_key=settings.LITELLM_API_KEY,
        temperature=0.2
    )

    # Supervisor agent doesn't need tools - it makes routing decisions
    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=get_supervisor_prompt(),
        checkpointer=None
    )

    def supervisor_node(state: AgentState):
        """Node function that invokes the supervisor agent."""
        result = agent.invoke(state)
        # Determine next agent based on query
        result["next"] = determine_next_agent(state.get("query", ""))
        # Preserve parsed query from state for subagents
        result["parsed_query"] = state.get("parsed_query")
        return result

    return supervisor_node


def create_chapter_agent():
    """Create the chapter-level agent."""
    llm = ChatOpenAI(
        model="qwen3-coder-next:local",
        base_url=settings.LITELLM_BASE_URL,
        api_key=settings.LITELLM_API_KEY,
        temperature=0.2
    )

    def chapter_agent_node(state: AgentState):
        """Node function that invokes the chapter agent."""
        parsed = state.get("parsed_query")

        # Use parsed chapter if available
        chapter_id = None
        if parsed and hasattr(parsed, 'command') and parsed.command == "chapter" and hasattr(parsed, 'chapter_id') and parsed.chapter_id:
            chapter_id = parsed.chapter_id
        else:
            # Fallback: find first chapter
            for key, value in state.get("summary_context", {}).items():
                if key.startswith("chapter_") and isinstance(value, dict):
                    if value.get("summary"):
                        chapter_id = value.get("id")
                        break

        # Build context with chapter summary
        chapter_context = ""
        if chapter_id:
            chapter_info = state.get("summary_context", {}).get(f"chapter_{chapter_id}", {})
            chapter_context = f"\n\nChapter Context:\nID: {chapter_id}\nName: {chapter_info.get('name', '')}\nSummary: {chapter_info.get('summary', '')}"

        full_prompt = get_chapter_agent_prompt(parsed) + chapter_context

        # Create database query tool with parsed query for filtering
        tool = create_database_query_tool(parsed)

        agent = create_agent(
            llm,
            tools=[tool],
            system_prompt=full_prompt,
            checkpointer=None
        )

        result = agent.invoke(state)
        # Mark as complete - response comes directly from this agent
        result["next"] = "__end__"
        return result

    return chapter_agent_node


def create_section_agent():
    """Create the section-level agent."""
    llm = ChatOpenAI(
        model="qwen3-coder-next:local",
        base_url=settings.LITELLM_BASE_URL,
        api_key=settings.LITELLM_API_KEY,
        temperature=0.2
    )

    def section_agent_node(state: AgentState):
        """Node function that invokes the section agent."""
        parsed = state.get("parsed_query")

        # Use parsed section if available
        section_info = None
        if parsed and hasattr(parsed, 'command') and parsed.command == "section" and hasattr(parsed, 'chapter_id') and parsed.chapter_id and hasattr(parsed, 'section_id') and parsed.section_id:
            section_info = state.get("summary_context", {}).get(f"section_{parsed.chapter_id}_{parsed.section_id}")
        else:
            # Fallback: find first section
            for key, value in state.get("summary_context", {}).items():
                if key.startswith("section_") and isinstance(value, dict):
                    if value.get("summary"):
                        section_info = value
                        break

        # Build context with section summary
        section_context = ""
        if section_info:
            section_context = f"\n\nSection Context:\nID: {section_info.get('chapter_id')}.{section_info.get('id')}\nName: {section_info.get('name', '')}\nSummary: {section_info.get('summary', '')}"

        full_prompt = get_section_agent_prompt(parsed) + section_context

        # Create database query tool with parsed query for filtering
        tool = create_database_query_tool(parsed)

        agent = create_agent(
            llm,
            tools=[tool],
            system_prompt=full_prompt,
            checkpointer=None
        )

        result = agent.invoke(state)
        # Mark as complete - response comes directly from this agent
        result["next"] = "__end__"
        return result

    return section_agent_node


def create_pages_agent():
    """Create the page-range agent."""
    llm = ChatOpenAI(
        model="qwen3-coder-next:local",
        base_url=settings.LITELLM_BASE_URL,
        api_key=settings.LITELLM_API_KEY,
        temperature=0.2
    )

    def pages_agent_node(state: AgentState):
        """Node function that invokes the pages agent."""
        parsed = state.get("parsed_query")

        full_prompt = get_pages_agent_prompt(parsed)

        # Create database query tool with parsed query for filtering
        tool = create_database_query_tool(parsed)

        agent = create_agent(
            llm,
            tools=[tool],
            system_prompt=full_prompt,
            checkpointer=None
        )

        result = agent.invoke(state)
        # Mark as complete - response comes directly from this agent
        result["next"] = "__end__"
        return result

    return pages_agent_node


# === GRAPH DEFINITION ===

def build_orchestrator_graph():
    """
    Build the full orchestrator graph with supervisor and subagents.

    The graph workflow:
    1. Entry: supervisor node
    2. Supervisor analyzes query and decides which subagent to invoke
    3. Subagent retrieves from LanceDB and processes query
    4. Subagent returns to supervisor for aggregation
    5. Supervisor composes final response
    6. END: completion

    Returns:
        Compiled graph ready for invocation
    """
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("supervisor", create_supervisor_agent())
    workflow.add_node("chapter_agent", create_chapter_agent())
    workflow.add_node("section_agent", create_section_agent())
    workflow.add_node("pages_agent", create_pages_agent())

    # Conditional routing from supervisor
    def route_from_supervisor(state: AgentState) -> str:
        """Determine which agent should handle the query."""
        next_agent = state.get("next")

        if next_agent == "__end__":
            return END

        valid_agents = ["chapter_agent", "section_agent", "pages_agent", "supervisor"]
        if next_agent in valid_agents:
            return next_agent

        # Default routing based on query analysis
        query = state.get("query", "")
        return determine_next_agent(query)

    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "chapter_agent": "chapter_agent",
            "section_agent": "section_agent",
            "pages_agent": "pages_agent",
            END: END
        }
    )

    # Subagents complete the workflow (no routing back to supervisor)
    for agent in ["chapter_agent", "section_agent", "pages_agent"]:
        workflow.add_conditional_edges(
            agent,
            lambda state: state.get("next", "__end__"),
            {"__end__": "__end__"}
        )

    # Set entry point
    workflow.set_entry_point("supervisor")

    # Compile with checkpointing for persistence
    return workflow.compile(checkpointer=MemorySaver())


# === ORCHESTRATOR ENTRY POINT ===

def create_orchestrator():
    """
    Create and return a configured orchestrator instance.

    This is the main entry point for using the LangGraph orchestrator.
    """
    return build_orchestrator_graph()


async def orchestrate_query(
    user_query: str,
    db_uri: str,
    table_name: str,
    summaries_path: str = "chapters_sections.json"
) -> str:
    """
    Main orchestration entry point.

    1. Parse command from user query
    2. Retrieve initial chunks
    3. Analyze intent using supervisor agent
    4. Delegate to subagents as needed
    5. Aggregate results and return final answer

    Args:
        user_query: The user's input query
        db_uri: URI to the LanceDB database
        table_name: Name of the chunks table
        summaries_path: Path to chapters_sections.json

    Returns:
        Final answer string from the orchestrator
    """
    # Load summary context
    summary_context = load_summary_context(summaries_path)

    # Parse query to extract command information
    parsed = parse_query(user_query)

    # Initialize the orchestrator graph
    orchestrator = create_orchestrator()

    # Create initial state
    parsed = parse_query(user_query)
    initial_state = AgentState(
        query=user_query,
        next="supervisor",
        summary_context=summary_context,
        response="",
        messages=[HumanMessage(content=user_query)],
        parsed_query=parsed
    )

    # Config for persistence
    config = {"configurable": {"thread_id": "user-session-1"}}

    # Run the orchestrator
    result = orchestrator.invoke(initial_state, config=config)

    # Extract final response from messages
    if "messages" in result and result["messages"]:
        final_message = result["messages"][-1]
        if hasattr(final_message, "content"):
            return final_message.content
        return str(final_message)

    return "No response generated."
