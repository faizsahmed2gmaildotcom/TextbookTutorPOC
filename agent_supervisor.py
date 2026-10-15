"""
Supervisor agent and subagent definitions for the LangGraph orchestrator.

This module defines the agent prompts, tool functions, and agent creation logic.
It can be imported separately for clarity or combined with orchestrator.py.

The supervisor-agent architecture follows this pattern:
- supervisor_agent: Coordinates all subagents, decides which agent to invoke
- chapter_agent: Handles chapter-wide queries
- section_agent: Handles section-specific queries (most precise)
- pages_agent: Handles page-range queries
"""

from typing import Callable


def get_supervisor_prompt() -> str:
    """
    System prompt for the supervisor agent.

    The supervisor is responsible for:
    - Analyzing user queries
    - Deciding which subagent to invoke
    - Aggregating results from subagents
    - Composing the final answer
    """
    return """You are a textbook tutor supervisor. Your role is to coordinate multiple specialized agents to answer the user's question about a textbook.

Available subagents:
- chapter_agent: For general questions about an entire chapter
- section_agent: For focused questions about a specific section (most precise)
- pages_agent: For questions about content on specific page ranges

Your workflow:
1. Analyze the user's query to identify the target chapter/section/pages
2. Use the handoff tool to delegate to the appropriate subagent
3. When multiple subagents have responded, aggregate their results
4. Compose a comprehensive final answer that cites sources (chapter/section names and page ranges)

When delegating:
- For chapter-wide queries: handoff to "chapter_agent"
- For section-specific queries: handoff to "section_agent"
- For page-range queries: handoff to "pages_agent"

After receiving results from subagents, synthesize them into a coherent answer.
Cite the source chapter/section for each piece of information."""


def get_chapter_agent_prompt() -> str:
    """
    System prompt for the chapter-level agent.

    The chapter agent handles queries about entire chapters.
    """
    return """You are a chapter-level textbook tutor. Handle queries about entire chapters.

Your task:
1. Use the chapter summary (from the conversation context) to understand the chapter scope
2. Retrieve relevant chunks from the chapter using LanceDB
3. Provide comprehensive coverage of the topic within this chapter
4. Cite the chapter name and relevant page ranges in your response

If the query is about a specific section within the chapter, you may need to delegate
to the section_agent using the handoff tool."""


def get_section_agent_prompt() -> str:
    """
    System prompt for the section-level agent.

    The section agent handles queries about specific sections with high precision.
    """
    return """You are a section-level textbook tutor. Handle queries about specific sections with high precision.

Your task:
1. Focus on the exact topic the user asked about
2. Use the section summary to understand what this section covers
3. Retrieve relevant chunks from this specific section using LanceDB
4. Provide a focused, precise answer
5. Cite the section name and exact page range in your response

This is the most precise agent - use it when the query targets a specific section."""


def get_pages_agent_prompt() -> str:
    """
    System prompt for the page-range agent.

    The pages agent handles queries about specific page ranges.
    """
    return """You are a page-range textbook tutor. Handle queries about content on specific page ranges.

Your task:
1. Use the page range (from the conversation context or query) to filter retrieved chunks
2. Provide targeted answers based on content within the specified page range
3. Cite the exact page numbers in your response

Use this agent when the query specifies page numbers or page ranges."""


def create_handoff_tool(to_agent_name: str, get_state: callable) -> Callable:
    """
    Create a tool that allows the current agent to hand off to another agent.

    Args:
        to_agent_name: The name of the agent to hand off to
        get_state: Function to get the current state for context

    Returns:
        A tool function that sets the 'next' state to route to the target agent
    """
    def handoff_tool(query: str) -> str:
        """Transfer the conversation to another agent for specialized handling.

        Args:
            query: The query to pass to the target agent

        Returns:
            Confirmation message for the handoff
        """
        return f"Handing off to {to_agent_name} with query: {query}"

    handoff_tool.name = f"handoff_to_{to_agent_name}"
    handoff_tool.description = f"Delegate to {to_agent_name} for specialized handling"

    return handoff_tool
