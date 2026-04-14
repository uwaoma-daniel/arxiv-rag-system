from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    precision_at_5:  float = 0.0
    recall_at_10:    float = 0.0
    mrr:             float = 0.0
    hit_rate_at_5:   float = 0.0
    num_queries:     int   = 0

    def to_dict(self) -> dict:
        return {
            "precision_at_5": round(self.precision_at_5, 4),
            "recall_at_10":   round(self.recall_at_10, 4),
            "mrr":            round(self.mrr, 4),
            "hit_rate_at_5":  round(self.hit_rate_at_5, 4),
            "num_queries":    self.num_queries,
        }


@dataclass
class GenerationMetrics:
    answer_accuracy:   float = 0.0
    faithfulness:      float = 0.0
    answer_relevancy:  float = 0.0
    citation_accuracy: float = 0.0
    rouge_l:           float = 0.0
    num_evaluated:     int   = 0

    def to_dict(self) -> dict:
        return {
            "answer_accuracy":   round(self.answer_accuracy, 4),
            "faithfulness":      round(self.faithfulness, 4),
            "answer_relevancy":  round(self.answer_relevancy, 4),
            "citation_accuracy": round(self.citation_accuracy, 4),
            "rouge_l":           round(self.rouge_l, 4),
            "num_evaluated":     self.num_evaluated,
        }


@dataclass
class LatencyMetrics:
    p50_ms:      float = 0.0
    p90_ms:      float = 0.0
    p95_ms:      float = 0.0
    p99_ms:      float = 0.0
    mean_ms:     float = 0.0
    num_queries: int   = 0

    def to_dict(self) -> dict:
        return {
            "p50_ms":      round(self.p50_ms,  2),
            "p90_ms":      round(self.p90_ms,  2),
            "p95_ms":      round(self.p95_ms,  2),
            "p99_ms":      round(self.p99_ms,  2),
            "mean_ms":     round(self.mean_ms, 2),
            "num_queries": self.num_queries,
        }


@dataclass
class EvaluationReport:
    retrieval:    RetrievalMetrics  = field(default_factory=RetrievalMetrics)
    generation:   GenerationMetrics = field(default_factory=GenerationMetrics)
    latency:      LatencyMetrics    = field(default_factory=LatencyMetrics)
    model_used:   str = ""
    vector_store: str = ""
    index_size:   int = 0
    timestamp:    str = ""
    version:      str = "1.0.0"

    def to_dict(self) -> dict:
        return {
            "version":      self.version,
            "timestamp":    self.timestamp,
            "model_used":   self.model_used,
            "vector_store": self.vector_store,
            "index_size":   self.index_size,
            "retrieval":    self.retrieval.to_dict(),
            "generation":   self.generation.to_dict(),
            "latency":      self.latency.to_dict(),
        }

    def save(self, path: str = "evaluation_report.json") -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
        logger.info(f"Evaluation report saved to {path}")

    def print_scorecard(self) -> None:
        r = self.retrieval
        g = self.generation
        la = self.latency

        targets = {
            "Retrieval Precision@5":  (r.precision_at_5,   0.80),
            "Retrieval Recall@10":    (r.recall_at_10,     0.75),
            "MRR":                    (r.mrr,              0.70),
            "Hit Rate@5":             (r.hit_rate_at_5,    0.90),
            "Answer Accuracy":        (g.answer_accuracy,  0.94),
            "RAGAS Faithfulness":     (g.faithfulness,     0.85),
            "RAGAS Answer Relevancy": (g.answer_relevancy, 0.80),
            "Citation Accuracy":      (g.citation_accuracy, 0.95),
            "P95 Latency (ms)":       (la.p95_ms,          2000),
        }

        print("\n" + "=" * 70)
        print("  RAG SYSTEM EVALUATION REPORT — v1.0.0")
        print("=" * 70)
        print(f"  Model:        {self.model_used}")
        print(f"  Vector Store: {self.vector_store}")
        print(f"  Index Size:   {self.index_size:,} documents")
        print(f"  Timestamp:    {self.timestamp}")
        print("=" * 70)
        print(f"  {'METRIC':<30} {'TARGET':>8} {'ACHIEVED':>10} {'STATUS':>8}")
        print("-" * 70)

        for metric, (achieved, target) in targets.items():
            if "Latency" in metric:
                passed = achieved < target
            else:
                passed = achieved >= target
            status = "✅" if passed else "❌"
            print(f"  {metric:<30} {target:>8.2f} {achieved:>10.4f} {status:>8}")

        print("=" * 70)


# ── Retrieval Evaluator ───────────────────────────────────────────────

class RetrievalEvaluator:
    """Evaluates retrieval quality given queries and known relevant paper_ids."""

    def evaluate(
        self,
        retriever,
        test_queries: List[Dict],
        top_k: int = 5,
    ) -> RetrievalMetrics:
        """
        Args:
            retriever:    DocumentRetriever instance.
            test_queries: list of {query, relevant_paper_ids}.
            top_k:        number of documents to retrieve.

        Returns:
            RetrievalMetrics
        """
        precisions: List[float] = []
        recalls:    List[float] = []
        rr_scores:  List[float] = []
        hits:       List[int]   = []

        for item in test_queries:
            query    = item["query"]
            relevant = set(item["relevant_paper_ids"])

            try:
                result = retriever.retrieve(query, top_k=top_k)
            except Exception as exc:
                logger.warning(
                    "Retrieval failed for query '%s…': %s", query[:50], exc
                )
                continue

            retrieved_ids = [doc.metadata.get("paper_id", "") for doc in result.documents]

            # Precision@k
            hits_at_k = sum(1 for pid in retrieved_ids[:top_k] if pid in relevant)
            precisions.append(hits_at_k / top_k if top_k > 0 else 0.0)

            # Recall@10
            recalls.append(hits_at_k / len(relevant) if relevant else 0.0)

            # MRR
            rr = 0.0
            for rank, pid in enumerate(retrieved_ids, 1):
                if pid in relevant:
                    rr = 1.0 / rank
                    break
            rr_scores.append(rr)

            # Hit Rate@5
            hits.append(
                1 if any(pid in relevant for pid in retrieved_ids[:5]) else 0
            )

        n = len(precisions)
        return RetrievalMetrics(
            precision_at_5=float(np.mean(precisions)) if precisions else 0.0,
            recall_at_10=float(np.mean(recalls))       if recalls    else 0.0,
            mrr=float(np.mean(rr_scores))              if rr_scores  else 0.0,
            hit_rate_at_5=float(np.mean(hits))         if hits       else 0.0,
            num_queries=n,
        )


# ── Latency Evaluator ─────────────────────────────────────────────────

class LatencyEvaluator:
    """Times end-to-end RAG pipeline queries."""

    def evaluate(
        self,
        rag_pipeline,
        test_queries: List[str],
    ) -> LatencyMetrics:
        """
        Args:
            rag_pipeline: RAGPipeline instance.
            test_queries: list of query strings to time.

        Returns:
            LatencyMetrics
        """
        latencies: List[float] = []

        for query in test_queries:
            try:
                t0 = time.perf_counter()
                rag_pipeline.query(question=query)
                latencies.append((time.perf_counter() - t0) * 1000)
            except Exception as exc:
                logger.warning("Query failed: '%s…': %s", query[:50], exc)
                continue

        if not latencies:
            return LatencyMetrics()

        arr = np.array(latencies)
        return LatencyMetrics(
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            mean_ms=float(np.mean(arr)),
            num_queries=len(latencies),
        )


# ── Citation Evaluator ────────────────────────────────────────────────

class CitationEvaluator:
    """Checks that every citation in generated answers exists in retrieved sources."""

    def evaluate(self, responses: List[Dict]) -> float:
        """
        Args:
            responses: list of {answer, sources, citations}.

        Returns:
            citation_accuracy: float 0-1
        """
        if not responses:
            return 0.0

        valid_count = 0
        for resp in responses:
            citations = resp.get("citations", [])
            sources   = resp.get("sources",   [])

            if not citations:
                valid_count += 1
                continue

            source_citations = {s.get("citation_str", "") for s in sources}
            if all(c in source_citations for c in citations):
                valid_count += 1

        return valid_count / len(responses)
