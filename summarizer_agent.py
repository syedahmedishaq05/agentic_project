"""
AMARA - Summarizer Agent
Extracts and distills key ideas from retrieved academic papers
using extractive + abstractive summarization.
"""
from __future__ import annotations
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from config import get_llm


SYSTEM_PROMPT = """You are an expert academic paper summarizer.
Given a paper's title, authors, year, and abstract, produce a structured summary with:

1. **Core Contribution** (1-2 sentences): What is the main idea or finding?
2. **Methodology** (1-2 sentences): What approach, technique, or method is used?
3. **Key Results** (1-2 sentences): What were the main outcomes or findings?
4. **Relevance** (1 sentence): How is this relevant to the research question?

Be precise, academic, and avoid fluff. Use the paper's own terminology."""


class SummarizerAgent:
    """Agent responsible for summarizing retrieved papers."""

    def __init__(self):
        self.llm = get_llm(temperature=0.2)

    def summarize_paper(self, paper: Dict, query: str) -> Dict:
        """Generate a structured summary for a single paper."""
        content = f"""Research Question: {query}

Paper Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Year: {paper['year']}
Source: {paper['source']}

Abstract:
{paper['abstract']}

Please summarize this paper in relation to the research question."""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]
        try:
            response = self.llm.invoke(messages)
            summary = getattr(response, "text", None) or getattr(response, "content", "")
            summary = summary.strip()
        except Exception as e:
            summary = f"[Summarizer Error] Could not summarize: {e}"

        return {
            **paper,
            "summary": summary,
        }

    def run(self, searcher_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full summarizer pipeline:
        - Takes searcher output
        - Summarizes each paper individually
        - Returns papers with summaries attached
        """
        query  = searcher_output["query"]
        papers = searcher_output["papers"]
        print(f"[Summarizer] Summarizing {len(papers)} papers...")

        summarized = []
        for i, paper in enumerate(papers):
            print(f"[Summarizer] Paper {i+1}/{len(papers)}: {paper['title'][:60]}...")
            summarized_paper = self.summarize_paper(paper, query)
            summarized.append(summarized_paper)

        print("[Summarizer] Done.")
        return {
            **searcher_output,
            "papers": summarized,
        }
