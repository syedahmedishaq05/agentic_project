"""
AMARA - LangGraph Orchestration Pipeline
Coordinates all four agents (Searcher → Summarizer → Critic → Writer)
through a structured state graph. Each node is a specialized agent.
"""
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END

from agents.searcher_agent   import SearcherAgent
from agents.summarizer_agent import SummarizerAgent
from agents.critic_agent     import CriticAgent
from agents.writer_agent     import WriterAgent
from rag.rag_pipeline        import RAGPipeline


# ── State Schema ──────────────────────────────────────────────────────────────
class AMARState(TypedDict):
    query:        str
    keywords:     List[str]
    papers:       List[Dict]
    critique:     str
    consensus:    str
    final_answer: str
    references:   str
    rag_context:  str
    error:        Optional[str]


# ── Node Functions ────────────────────────────────────────────────────────────
def searcher_node(state: AMARState) -> AMARState:
    """Node 1: Retrieve papers."""
    try:
        agent  = SearcherAgent()
        result = agent.run(state["query"])
        return {**state, **result, "error": None}
    except Exception as e:
        print(f"[Pipeline] Searcher error: {e}")
        return {**state, "papers": [], "keywords": [], "error": str(e)}


def rag_node(state: AMARState) -> AMARState:
    """Node 1b: Build RAG index from retrieved papers."""
    try:
        rag = RAGPipeline()
        if state.get("papers"):
            rag.build_index(state["papers"])
            rag_context = rag.get_context_string(state["query"])
        else:
            rag_context = ""
        return {**state, "rag_context": rag_context}
    except Exception as e:
        print(f"[Pipeline] RAG error: {e}")
        return {**state, "rag_context": "", "error": str(e)}


def summarizer_node(state: AMARState) -> AMARState:
    """Node 2: Summarize papers."""
    try:
        agent  = SummarizerAgent()
        result = agent.run(state)
        return {**state, **result, "error": None}
    except Exception as e:
        print(f"[Pipeline] Summarizer error: {e}")
        return {**state, "error": str(e)}


def critic_node(state: AMARState) -> AMARState:
    """Node 3: Critical analysis."""
    try:
        agent  = CriticAgent()
        result = agent.run(state)
        return {**state, **result, "error": None}
    except Exception as e:
        print(f"[Pipeline] Critic error: {e}")
        return {**state, "critique": "Error in critic agent.", "consensus": "", "error": str(e)}


def writer_node(state: AMARState) -> AMARState:
    """Node 4: Synthesize final answer."""
    try:
        agent  = WriterAgent()
        result = agent.run(state)
        return {**state, **result, "error": None}
    except Exception as e:
        print(f"[Pipeline] Writer error: {e}")
        return {**state, "final_answer": "Error generating final answer.", "error": str(e)}


# ── Graph Construction ────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """Build and compile the LangGraph state machine."""
    graph = StateGraph(AMARState)

    # Add nodes
    graph.add_node("searcher",   searcher_node)
    graph.add_node("rag",        rag_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("critic",     critic_node)
    graph.add_node("writer",     writer_node)

    # Define edges (sequential pipeline)
    graph.set_entry_point("searcher")
    graph.add_edge("searcher",   "rag")
    graph.add_edge("rag",        "summarizer")
    graph.add_edge("summarizer", "critic")
    graph.add_edge("critic",     "writer")
    graph.add_edge("writer",     END)

    return graph.compile()


# ── Main Entry Point ─────────────────────────────────────────────────────────
class AMARAPipeline:
    """High-level wrapper around the LangGraph pipeline."""

    def __init__(self):
        self.graph = build_graph()

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the full multi-agent pipeline for a research query.
        Returns the final state with answer, references, critique, etc.
        """
        print(f"\n{'='*60}")
        print(f"AMARA Pipeline Starting")
        print(f"Query: {query}")
        print(f"{'='*60}\n")

        initial_state: AMARState = {
            "query":        query,
            "keywords":     [],
            "papers":       [],
            "critique":     "",
            "consensus":    "",
            "final_answer": "",
            "references":   "",
            "rag_context":  "",
            "error":        None,
        }

        final_state = self.graph.invoke(initial_state)

        print(f"\n{'='*60}")
        print("AMARA Pipeline Complete")
        print(f"{'='*60}\n")

        return final_state
