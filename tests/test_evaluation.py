from __future__ import annotations

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.evaluation import (
    CitationEvaluator,
    EvaluationReport,
    GenerationMetrics,
    LatencyEvaluator,
    LatencyMetrics,
    RetrievalEvaluator,
    RetrievalMetrics,
)


# ── Fixtures ──────────────────────────────────────────────────────────

def make_mock_retriever(paper_ids=None):
    """Build a mock retriever that returns the given paper_ids."""
    paper_ids = paper_ids or ["p1", "p2", "p3"]

    docs = []
    for i, pid in enumerate(paper_ids):
        doc = MagicMock()
        doc.metadata = {
            "paper_id":    pid,
            "citation_str": f"Author{i}, 202{i}",
            "title":       f"Paper {pid}",
            "year":        2020 + i,
            "category":    "cs.LG",
        }
        doc.score = max(0.1, 0.9 - i * 0.1)
        doc.text  = f"Text about {pid}."
        docs.append(doc)

    result = MagicMock()
    result.documents = docs
    result.context_string      = "context"
    result.retrieval_latency_ms = 50.0
    result.all_scores          = [d.score for d in docs]
    result.low_confidence      = False
    result.query               = "test query"

    retriever = MagicMock()
    retriever.retrieve.return_value = result
    return retriever


def make_mock_rag_pipeline():
    """Build a mock RAG pipeline that returns instantly."""
    response = MagicMock()
    response.answer              = "Test answer (Author, 2020)."
    response.citations           = ["Author, 2020"]
    response.sources             = []
    response.confidence          = 0.85
    response.model_used          = "flan_t5_large"
    response.retrieval_latency_ms = 50.0
    response.generation_latency_ms = 100.0
    response.low_confidence      = False

    pipeline = MagicMock()
    pipeline.query.return_value = response
    return pipeline


# ── RetrievalMetrics tests ────────────────────────────────────────────

class TestRetrievalMetrics:
    def test_to_dict_has_all_keys(self):
        m = RetrievalMetrics(
            precision_at_5=0.8, recall_at_10=0.75,
            mrr=0.7, hit_rate_at_5=0.9, num_queries=10,
        )
        d = m.to_dict()
        for key in ["precision_at_5", "recall_at_10", "mrr",
                    "hit_rate_at_5", "num_queries"]:
            assert key in d

    def test_to_dict_values_rounded(self):
        m = RetrievalMetrics(precision_at_5=0.123456)
        assert m.to_dict()["precision_at_5"] == round(0.123456, 4)

    def test_default_values_are_zero(self):
        m = RetrievalMetrics()
        assert m.precision_at_5 == 0.0
        assert m.mrr == 0.0


# ── LatencyMetrics tests ──────────────────────────────────────────────

class TestLatencyMetrics:
    def test_to_dict_has_all_keys(self):
        m = LatencyMetrics(
            p50_ms=100, p90_ms=200, p95_ms=300,
            p99_ms=500, mean_ms=150, num_queries=50,
        )
        d = m.to_dict()
        for key in ["p50_ms", "p90_ms", "p95_ms",
                    "p99_ms", "mean_ms", "num_queries"]:
            assert key in d

    def test_default_values_are_zero(self):
        assert LatencyMetrics().p95_ms == 0.0


# ── EvaluationReport tests ────────────────────────────────────────────

class TestEvaluationReport:
    def test_to_dict_has_all_sections(self):
        report = EvaluationReport(
            model_used="flan_t5_large",
            vector_store="chroma",
            index_size=10247,
        )
        d = report.to_dict()
        for key in ["retrieval", "generation", "latency",
                    "model_used", "index_size", "version"]:
            assert key in d

    def test_save_creates_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/report.json"
            EvaluationReport(model_used="test").save(path)
            assert Path(path).exists()
            data = json.loads(Path(path).read_text())
            assert "version" in data

    def test_save_json_is_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/report.json"
            EvaluationReport(
                model_used="flan_t5_large",
                vector_store="chroma",
                index_size=10247,
            ).save(path)
            data = json.loads(Path(path).read_text())
            assert data["model_used"] == "flan_t5_large"
            assert data["index_size"] == 10247

    def test_print_scorecard_runs_without_error(self, capsys):
        EvaluationReport(
            model_used="flan_t5_large",
            vector_store="chroma",
            index_size=10247,
        ).print_scorecard()
        captured = capsys.readouterr()
        assert "EVALUATION REPORT" in captured.out
        assert "Precision" in captured.out


# ── RetrievalEvaluator tests ──────────────────────────────────────────

class TestRetrievalEvaluator:
    def test_returns_retrieval_metrics(self):
        metrics = RetrievalEvaluator().evaluate(
            make_mock_retriever(["p1", "p2", "p3"]),
            [{"query": "What is attention?", "relevant_paper_ids": ["p1"]},
             {"query": "Few-shot learning?", "relevant_paper_ids": ["p2"]}],
            top_k=5,
        )
        assert isinstance(metrics, RetrievalMetrics)

    def test_perfect_retrieval_hit_rate_is_one(self):
        metrics = RetrievalEvaluator().evaluate(
            make_mock_retriever(["p1"]),
            [{"query": "test", "relevant_paper_ids": ["p1"]}],
            top_k=5,
        )
        assert metrics.hit_rate_at_5 == 1.0

    def test_no_match_gives_zero_hit_rate(self):
        metrics = RetrievalEvaluator().evaluate(
            make_mock_retriever(["p1", "p2"]),
            [{"query": "test", "relevant_paper_ids": ["p999"]}],
            top_k=5,
        )
        assert metrics.hit_rate_at_5 == 0.0

    def test_mrr_first_result_relevant(self):
        metrics = RetrievalEvaluator().evaluate(
            make_mock_retriever(["p1", "p2", "p3"]),
            [{"query": "test", "relevant_paper_ids": ["p1"]}],
            top_k=5,
        )
        assert metrics.mrr == pytest.approx(1.0, abs=0.01)

    def test_num_queries_matches_input(self):
        metrics = RetrievalEvaluator().evaluate(
            make_mock_retriever(["p1"]),
            [{"query": f"q{i}", "relevant_paper_ids": ["p1"]} for i in range(5)],
            top_k=5,
        )
        assert metrics.num_queries == 5

    def test_handles_retriever_exception_gracefully(self):
        retriever = MagicMock()
        retriever.retrieve.side_effect = Exception("retrieval failed")
        metrics = RetrievalEvaluator().evaluate(
            retriever,
            [{"query": "test", "relevant_paper_ids": ["p1"]}],
            top_k=5,
        )
        assert metrics.num_queries == 0


# ── LatencyEvaluator tests ────────────────────────────────────────────

class TestLatencyEvaluator:
    def test_returns_latency_metrics(self):
        metrics = LatencyEvaluator().evaluate(
            make_mock_rag_pipeline(), ["What is attention?"] * 5
        )
        assert isinstance(metrics, LatencyMetrics)

    def test_num_queries_correct(self):
        metrics = LatencyEvaluator().evaluate(
            make_mock_rag_pipeline(), ["query"] * 10
        )
        assert metrics.num_queries == 10

    def test_p95_gte_p50(self):
        metrics = LatencyEvaluator().evaluate(
            make_mock_rag_pipeline(), ["query"] * 20
        )
        assert metrics.p95_ms >= metrics.p50_ms

    def test_empty_queries_returns_zero_metrics(self):
        metrics = LatencyEvaluator().evaluate(make_mock_rag_pipeline(), [])
        assert metrics.num_queries == 0
        assert metrics.p95_ms == 0.0

    def test_handles_pipeline_exception_gracefully(self):
        pipeline = MagicMock()
        pipeline.query.side_effect = Exception("pipeline failed")
        metrics = LatencyEvaluator().evaluate(pipeline, ["query"] * 3)
        assert metrics.num_queries == 0


# ── CitationEvaluator tests ───────────────────────────────────────────

class TestCitationEvaluator:
    def test_all_valid_citations_returns_one(self):
        responses = [{
            "answer":    "The model (Smith et al., 2023) shows...",
            "citations": ["Smith et al., 2023"],
            "sources":   [{"citation_str": "Smith et al., 2023"}],
        }]
        assert CitationEvaluator().evaluate(responses) == 1.0

    def test_hallucinated_citation_reduces_score(self):
        responses = [{
            "answer":    "The model (Fake et al., 1999) shows...",
            "citations": ["Fake et al., 1999"],
            "sources":   [{"citation_str": "Smith et al., 2023"}],
        }]
        assert CitationEvaluator().evaluate(responses) < 1.0

    def test_no_citations_counts_as_valid(self):
        responses = [{
            "answer":    "This answer has no citations.",
            "citations": [],
            "sources":   [],
        }]
        assert CitationEvaluator().evaluate(responses) == 1.0

    def test_empty_responses_returns_zero(self):
        assert CitationEvaluator().evaluate([]) == 0.0

    def test_mixed_valid_invalid_returns_partial(self):
        responses = [
            {
                "answer":    "Valid (Smith, 2023).",
                "citations": ["Smith, 2023"],
                "sources":   [{"citation_str": "Smith, 2023"}],
            },
            {
                "answer":    "Invalid (Ghost, 1900).",
                "citations": ["Ghost, 1900"],
                "sources":   [{"citation_str": "Smith, 2023"}],
            },
        ]
        score = CitationEvaluator().evaluate(responses)
        assert 0.0 < score < 1.0
