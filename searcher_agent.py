"""
AMARA - Searcher Agent
Retrieves relevant academic papers from ArXiv and Semantic Scholar
based on the user's research query.
"""
from __future__ import annotations
import time
import requests
import arxiv
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from config import get_llm, MAX_PAPERS, SEMANTIC_SCHOLAR_API_KEY


SYSTEM_PROMPT = """You are an expert academic literature searcher.
Given a research question, extract 3-5 concise search keywords/phrases
that will yield the most relevant academic papers.
Return ONLY a Python list of strings, e.g.: ["multi-agent systems LLM", "RAG academic QA"]
No explanation, no markdown, just the list."""


class SearcherAgent:
    """Agent responsible for retrieving academic papers."""

    def __init__(self):
        self.llm = get_llm(temperature=0.1)

    # ── Keyword Extraction ────────────────────────────────────────────────────
    def extract_keywords(self, query: str) -> List[str]:
        """Use LLM to extract optimal search keywords from the query."""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Research question: {query}"),
        ]
        response = self.llm.invoke(messages)
        raw_text = getattr(response, "text", None) or getattr(response, "content", "")
        try:
            keywords = eval(raw_text.strip())
            if isinstance(keywords, list):
                return keywords[:5]
        except Exception:
            pass
        # Fallback: use the first 5 words of the query
        return [" ".join(query.split()[:5])]

    # ── ArXiv Search ─────────────────────────────────────────────────────────
    def search_arxiv(self, keywords: List[str], max_results: int = None) -> List[Dict]:
        """Search ArXiv for papers matching keywords."""
        max_results = max_results or MAX_PAPERS
        papers = []
        seen_ids = set()

        for kw in keywords[:3]:  # limit API calls
            try:
                search = arxiv.Search(
                    query=kw,
                    max_results=max_results // len(keywords[:3]) + 2,
                    sort_by=arxiv.SortCriterion.Relevance,
                )
                for result in search.results():
                    if result.entry_id not in seen_ids:
                        seen_ids.add(result.entry_id)
                        papers.append({
                            "title":    result.title,
                            "authors":  [a.name for a in result.authors[:3]],
                            "abstract": result.summary[:1000],
                            "url":      result.pdf_url,
                            "year":     result.published.year,
                            "source":   "arxiv",
                            "id":       result.entry_id,
                        })
            except Exception as e:
                print(f"[Searcher] ArXiv error for '{kw}': {e}")

        return papers[:max_results]

    # ── Semantic Scholar Search ───────────────────────────────────────────────
    def search_semantic_scholar(self, keywords: List[str], max_results: int = None) -> List[Dict]:
        """Search Semantic Scholar for papers."""
        max_results = max_results or MAX_PAPERS
        papers = []
        seen_ids = set()
        headers = {}
        if SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

        for kw in keywords[:2]:
            try:
                url = "https://api.semanticscholar.org/graph/v1/paper/search"
                params = {
                    "query": kw,
                    "limit": max_results // 2 + 2,
                    "fields": "title,authors,abstract,year,externalIds,openAccessPdf",
                }
                resp = requests.get(url, params=params, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    for paper in data:
                        pid = paper.get("paperId", "")
                        if pid and pid not in seen_ids:
                            seen_ids.add(pid)
                            pdf_url = ""
                            if paper.get("openAccessPdf"):
                                pdf_url = paper["openAccessPdf"].get("url", "")
                            papers.append({
                                "title":    paper.get("title", ""),
                                "authors":  [a["name"] for a in paper.get("authors", [])[:3]],
                                "abstract": (paper.get("abstract") or "")[:1000],
                                "url":      pdf_url or f"https://www.semanticscholar.org/paper/{pid}",
                                "year":     paper.get("year", "N/A"),
                                "source":   "semantic_scholar",
                                "id":       pid,
                            })
                time.sleep(0.5)  # respect rate limits
            except Exception as e:
                print(f"[Searcher] Semantic Scholar error for '{kw}': {e}")

        return papers[:max_results]

    # ── Main Run ──────────────────────────────────────────────────────────────
    def run(self, query: str) -> Dict[str, Any]:
        """
        Full searcher pipeline:
        1. Extract keywords from query
        2. Search ArXiv + Semantic Scholar
        3. Return merged, deduplicated paper list
        """
        print(f"[Searcher] Processing query: {query[:80]}...")
        keywords = self.extract_keywords(query)
        print(f"[Searcher] Keywords extracted: {keywords}")

        arxiv_papers     = self.search_arxiv(keywords)
        ss_papers        = self.search_semantic_scholar(keywords)

        # Merge and deduplicate by title similarity
        all_papers = arxiv_papers + ss_papers
        seen_titles = set()
        unique_papers = []
        for p in all_papers:
            title_key = p["title"].lower()[:50]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(p)

        print(f"[Searcher] Found {len(unique_papers)} unique papers.")
        return {
            "query":    query,
            "keywords": keywords,
            "papers":   unique_papers[:MAX_PAPERS],
        }
