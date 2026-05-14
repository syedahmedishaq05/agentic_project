"""
AMARA - Critic Agent
Identifies methodological weaknesses, research gaps, and missing
perspectives in the retrieved literature.
"""
from __future__ import annotations
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from config import get_llm


SYSTEM_PROMPT = """You are a rigorous academic peer reviewer and research critic.
You will be given a research question and summaries of relevant papers.
Your job is to critically analyze the body of literature and identify:

1. **Research Gaps**: Important questions the literature has NOT addressed
2. **Methodological Weaknesses**: Common flaws or limitations across the studies
3. **Missing Perspectives**: Viewpoints, populations, or contexts underrepresented
4. **Contradictions**: Where papers disagree or present conflicting findings
5. **Strengths**: What the literature has established well and reliably

Be constructively critical, specific, and grounded in the actual papers provided.
Format your response with clear headings for each section."""


CONSENSUS_PROMPT = """You are an expert research analyst.
Given the paper summaries below, identify:
- The 3 most agreed-upon findings across papers
- The 2 most contested or debated points
- The overall maturity level of this research area (emerging/developing/mature)

Be concise and specific."""


class CriticAgent:
    """Agent responsible for critical analysis of the literature."""

    def __init__(self):
        self.llm = get_llm(temperature=0.3)

    def build_literature_overview(self, papers: List[Dict]) -> str:
        """Build a structured text overview of all paper summaries."""
        lines = []
        for i, p in enumerate(papers, 1):
            lines.append(f"--- Paper {i} ---")
            lines.append(f"Title: {p['title']}")
            lines.append(f"Authors: {', '.join(p['authors'])}")
            lines.append(f"Year: {p['year']}")
            lines.append(f"Summary:\n{p.get('summary', p.get('abstract', ''))}")
            lines.append("")
        return "\n".join(lines)

    def critique_literature(self, query: str, papers: List[Dict]) -> str:
        """Generate a critical analysis of the literature."""
        overview = self.build_literature_overview(papers)
        content = f"""Research Question: {query}

Retrieved Literature:
{overview}

Please provide a critical analysis of this body of literature."""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]
        response = self.llm.invoke(messages)
        text = getattr(response, "text", None) or getattr(response, "content", "")
        return text.strip()

    def identify_consensus(self, query: str, papers: List[Dict]) -> str:
        """Identify consensus and contested points across papers."""
        overview = self.build_literature_overview(papers)
        content = f"""Research Question: {query}

Papers:
{overview}"""

        messages = [
            SystemMessage(content=CONSENSUS_PROMPT),
            HumanMessage(content=content),
        ]
        response = self.llm.invoke(messages)
        text = getattr(response, "text", None) or getattr(response, "content", "")
        return text.strip()

    def run(self, summarizer_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full critic pipeline:
        - Performs gap analysis
        - Identifies consensus/conflict
        - Returns critique appended to state
        """
        query  = summarizer_output["query"]
        papers = summarizer_output["papers"]
        print(f"[Critic] Analyzing {len(papers)} papers for gaps and weaknesses...")

        critique  = self.critique_literature(query, papers)
        consensus = self.identify_consensus(query, papers)

        print("[Critic] Analysis complete.")
        return {
            **summarizer_output,
            "critique":  critique,
            "consensus": consensus,
        }
