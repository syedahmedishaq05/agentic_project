"""
AMARA - Writer Agent
Synthesizes all agent outputs into a clear, well-cited,
academically structured final answer.
"""
from __future__ import annotations
from typing import List, Dict, Any
from langchain.schema import HumanMessage, SystemMessage
from config import get_llm


SYSTEM_PROMPT = """You are an expert academic writer producing research synthesis reports.
You will receive:
- A research question
- Summaries of relevant papers (with titles, authors, years)
- A critical analysis of the literature (gaps, weaknesses, consensus)

Write a comprehensive, well-structured academic answer that includes:

## Introduction
Brief context and why this question matters.

## Literature Review
Synthesize findings from the retrieved papers. Cite papers inline as (Author et al., Year).

## Critical Analysis
Integrate the identified gaps, weaknesses, and consensus points.

## Conclusion
A direct, evidence-based answer to the research question. What does the literature say?

## References
List all cited papers in APA format.

Rules:
- Use inline citations like (Lewis et al., 2020) or (Author, Year)
- Be academic but readable
- Length: 600-900 words
- Grounded ONLY in the provided papers, do not hallucinate"""


class WriterAgent:
    """Agent responsible for synthesizing the final answer."""

    def __init__(self):
        self.llm = get_llm(temperature=0.4)

    def format_references(self, papers: List[Dict]) -> str:
        """Format papers as APA-style references."""
        refs = []
        for p in papers:
            authors = ", ".join(p["authors"]) if p["authors"] else "Unknown Authors"
            year    = p.get("year", "n.d.")
            title   = p.get("title", "Untitled")
            url     = p.get("url", "")
            ref = f"{authors} ({year}). {title}."
            if url:
                ref += f" Retrieved from {url}"
            refs.append(ref)
        return "\n".join(refs)

    def build_context(self, query: str, papers: List[Dict], critique: str, consensus: str) -> str:
        """Build the full context for the writer LLM call."""
        paper_context = []
        for i, p in enumerate(papers, 1):
            paper_context.append(
                f"[{i}] {p['title']} | {', '.join(p['authors'])} | {p['year']}\n"
                f"Summary: {p.get('summary', p.get('abstract', ''))[:500]}"
            )

        return f"""Research Question: {query}

===PAPER SUMMARIES===
{chr(10).join(paper_context)}

===CRITICAL ANALYSIS===
{critique}

===CONSENSUS & CONFLICTS===
{consensus}

Now write the comprehensive academic synthesis report."""

    def run(self, critic_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full writer pipeline:
        - Synthesizes all prior outputs into a final answer
        - Appends formatted references
        """
        query     = critic_output["query"]
        papers    = critic_output["papers"]
        critique  = critic_output.get("critique", "")
        consensus = critic_output.get("consensus", "")

        print("[Writer] Synthesizing final answer...")
        context = self.build_context(query, papers, critique, consensus)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
        response = self.llm.invoke(messages)
        final_answer = response.content.strip()

        print("[Writer] Done. Answer generated.")
        return {
            **critic_output,
            "final_answer": final_answer,
            "references":   self.format_references(papers),
        }
