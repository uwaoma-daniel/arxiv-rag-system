from __future__ import annotations

import os
import sys
import time
import anyio

# ── Set TEST_MODE before importing api.main ───────────────────────────
# This disables the rate limiter for the entire test session
os.environ["TEST_MODE"] = "1"

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from unittest.mock import MagicMock
import pytest
import httpx


# ── Helpers ───────────────────────────────────────────────────────────

def _make_mock_rag_response(low_confidence=False):
    from src.retrieval import ScoredDocument
    from src.generation import RAGResponse
    doc = ScoredDocument(
        chunk_id="test_001",
        text="Attention mechanisms allow models to focus on relevant tokens.",
        metadata={
            "paper_id":     "2301.00001",
            "title":        "Attention is All You Need",
            "citation_str": "Vaswani et al., 2017",
            "year":         2017,
            "category":     "cs.CL",
            "authors_raw":  "Vaswani et al.",
        },
        score=0.85,
    )
    return RAGResponse(
        answer="The attention mechanism... (Vaswani et al., 2017)",
        citations=["Vaswani et al., 2017"],
        sources=[doc],
        confidence=0.85,
        model_used="flan_t5_large",
        retrieval_latency_ms=50.0,
        generation_latency_ms=200.0,
        low_confidence=low_confidence,
    )


MOCK_CONFIG = {
    "embedding":    {"model_name": "all-MiniLM-L6-v2", "device": "cpu"},
    "vector_store": {"chroma_persist_dir": "/tmp/chroma"},
    "llm":          {"mode": "fallback"},
    "retrieval":    {"score_threshold": 0.30, "mmr_lambda": 0.7, "top_k": 5},
    "api":          {"max_turns": 6},
}


# ── Sync wrapper around async httpx ──────────────────────────────────

class SyncASGIClient:
    def __init__(self, app):
        self._app = app

    def _call(self, coro):
        return anyio.run(lambda: coro)

    def get(self, url, **kwargs):
        async def _inner():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self._app),
                base_url="http://testserver",
            ) as c:
                return await c.get(url, **kwargs)
        return anyio.run(_inner)

    def post(self, url, **kwargs):
        async def _inner():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self._app),
                base_url="http://testserver",
            ) as c:
                return await c.post(url, **kwargs)
        return anyio.run(_inner)

    def delete(self, url, **kwargs):
        async def _inner():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self._app),
                base_url="http://testserver",
            ) as c:
                return await c.delete(url, **kwargs)
        return anyio.run(_inner)


def _get_client():
    import api.main as M

    mock_pipeline         = MagicMock()
    mock_pipeline.query.return_value = _make_mock_rag_response()
    mock_llm              = MagicMock()
    mock_llm.backend_name = "flan_t5_large"
    mock_llm.is_loaded    = True
    mock_store            = MagicMock()
    mock_store.get_count.return_value = 10247
    mock_embedder         = MagicMock()
    mock_embedder.dimensions = 384

    M._rag_pipeline = mock_pipeline
    M._llm          = mock_llm
    M._vector_store = mock_store
    M._embedder     = mock_embedder
    M._config       = MOCK_CONFIG
    M._start_time   = time.perf_counter()

    M.app.router.lifespan_context = None

    return SyncASGIClient(M.app), mock_pipeline


@pytest.fixture(scope="module")
def client():
    return _get_client()


# ── Health endpoint ───────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        c, _ = client
        assert c.get("/health").status_code == 200

    def test_health_response_has_status(self, client):
        c, _ = client
        assert "status" in c.get("/health").json()

    def test_health_response_has_index_size(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert "index_size" in data
        assert isinstance(data["index_size"], int)

    def test_health_response_has_version(self, client):
        c, _ = client
        assert c.get("/health").json()["version"] == "1.0.0"

    def test_health_response_has_uptime(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


# ── Query endpoint ────────────────────────────────────────────────────

class TestQueryEndpoint:
    def test_query_returns_200(self, client):
        c, _ = client
        assert c.post("/query", json={"query": "What is attention mechanism?"}).status_code == 200

    def test_query_response_has_answer(self, client):
        c, _ = client
        data = c.post("/query", json={"query": "What is attention mechanism?"}).json()
        assert "answer" in data and len(data["answer"]) > 0

    def test_query_response_has_session_id(self, client):
        c, _ = client
        data = c.post("/query", json={"query": "What is few-shot learning?"}).json()
        assert "session_id" in data and len(data["session_id"]) > 0

    def test_query_response_has_sources(self, client):
        c, _ = client
        data = c.post("/query", json={"query": "What is BERT?"}).json()
        assert isinstance(data.get("sources"), list)

    def test_query_response_has_latency(self, client):
        c, _ = client
        data = c.post("/query", json={"query": "What is GAN?"}).json()
        assert "latency_ms" in data and data["latency_ms"] >= 0

    def test_query_response_has_citations(self, client):
        c, _ = client
        data = c.post("/query", json={"query": "Explain transformers."}).json()
        assert isinstance(data.get("citations"), list)

    def test_query_too_short_returns_422(self, client):
        c, _ = client
        assert c.post("/query", json={"query": "hi"}).status_code == 422

    def test_query_too_long_returns_422(self, client):
        c, _ = client
        assert c.post("/query", json={"query": "x" * 501}).status_code == 422

    def test_query_empty_string_returns_422(self, client):
        c, _ = client
        assert c.post("/query", json={"query": ""}).status_code == 422

    def test_query_with_session_id_continues_conversation(self, client):
        c, _ = client
        r1  = c.post("/query", json={"query": "What is attention?"}).json()
        sid = r1["session_id"]
        r2  = c.post("/query", json={
            "query":      "How does it work?",
            "session_id": sid,
        }).json()
        assert r2["session_id"] == sid
        assert r2["turn_number"] == 2

    def test_query_top_k_respected(self, client):
        c, _ = client
        assert c.post("/query", json={
            "query": "What is deep learning?",
            "top_k": 3,
        }).status_code == 200

    def test_query_include_sources_false(self, client):
        c, _ = client
        data = c.post("/query", json={
            "query":           "What is reinforcement learning?",
            "include_sources": False,
        }).json()
        assert data["sources"] == []


# ── History endpoint ──────────────────────────────────────────────────

class TestHistoryEndpoint:
    def test_history_returns_turns(self, client):
        c, _ = client
        sid  = c.post("/query", json={"query": "What is CNN?"}).json()["session_id"]
        data = c.get(f"/history/{sid}").json()
        assert "turns" in data and len(data["turns"]) >= 2

    def test_history_unknown_session_returns_404(self, client):
        c, _ = client
        assert c.get("/history/nonexistent-session-id-xyz").status_code == 404

    def test_history_has_session_id(self, client):
        c, _ = client
        sid  = c.post("/query", json={"query": "What is NLP?"}).json()["session_id"]
        data = c.get(f"/history/{sid}").json()
        assert data["session_id"] == sid


# ── Clear history endpoint ────────────────────────────────────────────

class TestClearHistoryEndpoint:
    def test_clear_returns_200(self, client):
        c, _ = client
        sid  = c.post("/query", json={"query": "What is LSTM?"}).json()["session_id"]
        assert c.delete(f"/history/{sid}").status_code == 200

    def test_clear_empties_history(self, client):
        c, _ = client
        sid  = c.post("/query", json={"query": "What is GRU?"}).json()["session_id"]
        c.delete(f"/history/{sid}")
        assert c.get(f"/history/{sid}").json()["total_turns"] == 0

    def test_clear_unknown_session_returns_404(self, client):
        c, _ = client
        assert c.delete("/history/nonexistent-session-xyz").status_code == 404


# ── Schema validation ─────────────────────────────────────────────────

class TestSchemaValidation:
    def test_query_request_injection_blocked(self, client):
        c, _ = client
        assert c.post("/query", json={
            "query": "<script>alert(1)</script> test query"
        }).status_code == 422

    def test_query_request_top_k_min(self, client):
        c, _ = client
        assert c.post("/query", json={
            "query":  "What is transfer learning?",
            "top_k":  0,
        }).status_code == 422

    def test_query_request_top_k_max(self, client):
        c, _ = client
        assert c.post("/query", json={
            "query":  "What is transfer learning?",
            "top_k":  21,
        }).status_code == 422
