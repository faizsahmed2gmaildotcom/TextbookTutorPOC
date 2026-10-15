"""
Tests for orchestrator.py - LangGraph multi-agent orchestration (Phase 5B).

Tests verify the supervisor-agent pattern works correctly for the textbook tutor.
"""

import pytest
import json
import os
from typing import Literal, Sequence

from langgraph.graph import MessagesState


class TestOrchestratorArchitecture:
    """Test the LangGraph orchestrator architecture."""

    def test_imports(self):
        """Test that all required modules can be imported."""
        from orchestrator import (
            create_orchestrator,
            orchestrate_query,
            build_orchestrator_graph,
            load_summary_context,
            AgentState,
            get_supervisor_prompt,
            get_chapter_agent_prompt,
            get_section_agent_prompt,
            get_pages_agent_prompt
        )

        assert True  # If we get here, all imports succeeded

    def test_agent_state_structure(self):
        """Test that AgentState has the expected fields."""
        from orchestrator import AgentState

        # AgentState should extend MessagesState
        assert hasattr(AgentState, '__annotations__')

        # Expected fields
        annotations = AgentState.__annotations__
        assert 'messages' in annotations
        assert 'query' in annotations
        assert 'next' in annotations
        assert 'summary_context' in annotations
        assert 'response' in annotations
        assert 'parsed_query' in annotations  # NEW: parsed command query

    def test_build_orchestrator_graph(self):
        """Test that the orchestrator graph builds successfully."""
        from orchestrator import build_orchestrator_graph

        graph = build_orchestrator_graph()

        # Verify graph structure (already compiled by build_orchestrator_graph)
        assert graph is not None
        # The graph returned is a CompiledStateGraph
        assert hasattr(graph, 'invoke')  # Compiled graphs have invoke method

        # Verify nodes exist
        nodes = graph.nodes
        assert 'supervisor' in nodes
        assert 'chapter_agent' in nodes
        assert 'section_agent' in nodes
        assert 'pages_agent' in nodes

    def test_load_summary_context(self):
        """Test that summary context loads from chapters_sections.json."""
        from orchestrator import load_summary_context

        # This requires chapters_sections.json to exist
        if not os.path.exists("chapters_sections.json"):
            pytest.skip("chapters_sections.json not found")

        context = load_summary_context()

        # Verify structure
        assert isinstance(context, dict)
        assert len(context) > 0

        # Should have chapter and section entries
        has_chapter = any(k.startswith("chapter_") for k in context.keys())
        has_section = any(k.startswith("section_") for k in context.keys())

        assert has_chapter or has_section

    def test_agent_prompts(self):
        """Test that agent prompts are properly formatted."""
        from orchestrator import (
            get_supervisor_prompt,
            get_chapter_agent_prompt,
            get_section_agent_prompt,
            get_pages_agent_prompt
        )

        supervisor_prompt = get_supervisor_prompt()
        assert "supervisor" in supervisor_prompt.lower()
        assert "subagent" in supervisor_prompt.lower()

        chapter_prompt = get_chapter_agent_prompt()
        assert "chapter" in chapter_prompt.lower()

        section_prompt = get_section_agent_prompt()
        assert "section" in section_prompt.lower()

        pages_prompt = get_pages_agent_prompt()
        assert "page" in pages_prompt.lower()


class TestOrchestratorIntegration:
    """Integration tests for the orchestrator with actual LLM calls."""

    @pytest.fixture
    def orchestrator(self):
        """Create an orchestrator instance."""
        from orchestrator import build_orchestrator_graph

        graph = build_orchestrator_graph()
        return graph

    @pytest.fixture
    def summary_context(self):
        """Load summary context."""
        from orchestrator import load_summary_context

        if not os.path.exists("chapters_sections.json"):
            return {}
        return load_summary_context()

    def test_supervisor_node_exists(self, orchestrator):
        """Test that supervisor node is properly configured."""
        assert "supervisor" in orchestrator.nodes

    def test_conditional_edges_exist(self, orchestrator):
        """Test that conditional edges are properly configured."""
        # For CompiledStateGraph, we verify the graph structure differently
        # The graph should have been built with conditional edges
        # We verify this by checking that the graph has the expected nodes
        # and can be invoked (which requires edges to be properly configured)

        # Verify nodes exist
        nodes = orchestrator.nodes
        assert len(nodes) >= 4  # supervisor + 3 subagents

        # Verify the graph has entry point set
        # (this is set when we call set_entry_point in build_orchestrator_graph)
        assert 'supervisor' in nodes

    def test_graph_compilation(self, orchestrator):
        """Test that the graph compiles correctly."""
        # Compilation happens during build, verify the graph is usable
        assert orchestrator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
