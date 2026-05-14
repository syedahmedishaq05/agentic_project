"""
AMARA - Evaluation Module
Computes ROUGE and BLEU scores comparing multi-agent output
against a single-LLM baseline and ground-truth answers.
"""
from __future__ import annotations
import json
from typing import Dict, List, Any, Optional

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ── Metric Computation ────────────────────────────────────────────────────────
def compute_rouge(hypothesis: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1_f": round(scores["rouge1"].fmeasure, 4),
        "rouge2_f": round(scores["rouge2"].fmeasure, 4),
        "rougeL_f": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bleu(hypothesis: str, reference: str) -> float:
    """Compute BLEU score with smoothing."""
    ref_tokens  = nltk.word_tokenize(reference.lower())
    hyp_tokens  = nltk.word_tokenize(hypothesis.lower())
    smoother    = SmoothingFunction().method4
    score       = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)
    return round(score, 4)


def evaluate_response(hypothesis: str, reference: str) -> Dict[str, float]:
    """Compute all metrics for a hypothesis against a reference."""
    rouge  = compute_rouge(hypothesis, reference)
    bleu   = compute_bleu(hypothesis, reference)
    length = len(hypothesis.split())
    return {**rouge, "bleu": bleu, "response_length_words": length}


# ── Baseline (Single LLM) ─────────────────────────────────────────────────────
def get_single_llm_baseline(query: str) -> str:
    """Get a direct single-LLM answer (no agent pipeline) for comparison."""
    from config import get_llm
    from langchain.schema import HumanMessage, SystemMessage

    llm = get_llm(temperature=0.3)
    messages = [
        SystemMessage(content=(
            "You are a helpful academic assistant. "
            "Answer the research question thoroughly based on your training knowledge."
        )),
        HumanMessage(content=query),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


# ── Benchmark Runner ──────────────────────────────────────────────────────────
class EvaluationRunner:
    """Runs the full comparative evaluation: multi-agent vs single-LLM."""

    def __init__(self):
        from pipeline import AMARAPipeline
        self.pipeline = AMARAPipeline()

    def evaluate_single_query(
        self,
        query: str,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate AMARA vs single-LLM on one query."""
        print(f"\n[Eval] Query: {query[:80]}...")

        # Multi-agent answer
        print("[Eval] Running AMARA pipeline...")
        amara_state  = self.pipeline.run(query)
        amara_answer = amara_state.get("final_answer", "")

        # Single-LLM baseline
        print("[Eval] Running single-LLM baseline...")
        baseline_answer = get_single_llm_baseline(query)

        result = {
            "query":           query,
            "amara_answer":    amara_answer,
            "baseline_answer": baseline_answer,
            "papers_retrieved": len(amara_state.get("papers", [])),
        }

        if ground_truth:
            result["ground_truth"]     = ground_truth
            result["amara_metrics"]    = evaluate_response(amara_answer,    ground_truth)
            result["baseline_metrics"] = evaluate_response(baseline_answer, ground_truth)
            result["amara_vs_baseline"] = {
                k: round(result["amara_metrics"][k] - result["baseline_metrics"][k], 4)
                for k in result["amara_metrics"]
            }
        return result

    def run_benchmark(self, benchmark_path: str, output_path: str = "evaluation/results.json"):
        """
        Run evaluation on a benchmark JSON file.
        Expected format: [{"query": "...", "ground_truth": "..."}, ...]
        """
        with open(benchmark_path) as f:
            benchmark = json.load(f)

        results = []
        for item in benchmark:
            res = self.evaluate_single_query(
                query        = item["query"],
                ground_truth = item.get("ground_truth"),
            )
            results.append(res)

        # Aggregate metrics
        if results and "amara_metrics" in results[0]:
            agg = {}
            for metric in results[0]["amara_metrics"]:
                agg[f"avg_amara_{metric}"]    = round(
                    sum(r["amara_metrics"][metric] for r in results) / len(results), 4
                )
                agg[f"avg_baseline_{metric}"] = round(
                    sum(r["baseline_metrics"][metric] for r in results) / len(results), 4
                )

            output = {"results": results, "aggregate": agg}
        else:
            output = {"results": results}

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n[Eval] Results saved to {output_path}")
        if "aggregate" in output:
            print("\nAggregate Metrics:")
            for k, v in output["aggregate"].items():
                print(f"  {k}: {v}")

        return output
