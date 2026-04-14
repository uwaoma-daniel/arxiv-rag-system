from __future__ import annotations

import sys
import types

# ── Mock streamlit BEFORE importing anything from frontend ────────────
# Streamlit is not installed in the test environment — we mock the
# entire module so test_frontend.py can import streamlit_app cleanly.

def _make_streamlit_mock():
    st = types.ModuleType("streamlit")
    st.session_state  = {}
    st.secrets        = {}
    st.set_page_config = lambda **kw: None
    st.markdown        = lambda *a, **kw: None
    st.title           = lambda *a, **kw: None
    st.caption         = lambda *a, **kw: None
    st.divider         = lambda *a, **kw: None
    st.header          = lambda *a, **kw: None
    st.metric          = lambda *a, **kw: None
    st.text_input      = lambda *a, **kw: ""
    st.slider          = lambda *a, **kw: 5
    st.columns         = lambda n: [types.SimpleNamespace(
                             __enter__=lambda s: s,
                             __exit__=lambda s, *a: False
                         ) for _ in (range(n) if isinstance(n, int) else n)]
    st.expander        = lambda *a, **kw: types.SimpleNamespace(
                             __enter__=lambda s: s,
                             __exit__=lambda s, *a: False
                         )
    st.spinner         = lambda *a, **kw: types.SimpleNamespace(
                             __enter__=lambda s: s,
                             __exit__=lambda s, *a: False
                         )
    st.button          = lambda *a, **kw: False
    st.error           = lambda *a, **kw: None
    st.warning         = lambda *a, **kw: None
    st.info            = lambda *a, **kw: None
    st.success         = lambda *a, **kw: None
    st.image           = lambda *a, **kw: None
    st.rerun           = lambda: None
    st.stop            = lambda: None
    return st

sys.modules.setdefault("streamlit", _make_streamlit_mock())

# Now safe to insert path and import
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from unittest.mock import MagicMock, patch
import pytest


# ── Helpers ───────────────────────────────────────────────────────────

def make_api_response(**overrides):
    base = {
        "answer":           "The attention mechanism (Vaswani et al., 2017)...",
        "sources":          [{
            "paper_id":        "2301.00001",
            "title":           "Attention is All You Need",
            "authors":         ["Vaswani et al."],
            "year":            2017,
            "category":        "cs.CL",
            "relevance_score": 0.87,
            "citation_str":    "Vaswani et al., 2017",
            "excerpt":         "Attention mechanisms allow models...",
        }],
        "citations":        ["Vaswani et al., 2017"],
        "latency_ms":       1234.5,
        "model_used":       "flan_t5_large",
        "retrieved_chunks": 5,
        "session_id":       "test-session-uuid",
        "turn_number":      1,
        "low_confidence":   False,
        "confidence_score": 0.87,
    }
    base.update(overrides)
    return base


def make_health_response(**overrides):
    base = {
        "status":          "healthy",
        "llm_backend":     "flan_t5_large",
        "vector_store":    "chroma",
        "index_size":      10247,
        "embedding_model": "all-MiniLM-L6-v2",
        "uptime_seconds":  3600.0,
        "version":         "1.0.0",
    }
    base.update(overrides)
    return base


# ── call_query_api tests ──────────────────────────────────────────────

class TestCallQueryAPI:
    def test_returns_dict_on_success(self):
        from frontend.streamlit_app import call_query_api
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = make_api_response()
        with patch("frontend.streamlit_app.requests.post", return_value=mock_resp):
            result = call_query_api("What is attention?", session_id=None)
        assert isinstance(result, dict)
        assert result["answer"] != ""

    def test_returns_none_on_connection_error(self):
        import requests as req
        from frontend.streamlit_app import call_query_api
        with patch("frontend.streamlit_app.requests.post",
                   side_effect=req.exceptions.ConnectionError()):
            result = call_query_api("What is attention?", session_id=None)
        assert result is None

    def test_returns_none_on_timeout(self):
        import requests as req
        from frontend.streamlit_app import call_query_api
        with patch("frontend.streamlit_app.requests.post",
                   side_effect=req.exceptions.Timeout()):
            result = call_query_api("What is attention?", session_id=None)
        assert result is None

    def test_returns_none_on_api_error_status(self):
        from frontend.streamlit_app import call_query_api
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service unavailable"
        with patch("frontend.streamlit_app.requests.post", return_value=mock_resp):
            result = call_query_api("What is attention?", session_id=None)
        assert result is None

    def test_passes_session_id_in_payload(self):
        from frontend.streamlit_app import call_query_api
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = make_api_response(session_id="my-session")
        with patch("frontend.streamlit_app.requests.post", return_value=mock_resp) as mock_post:
            call_query_api("test query", session_id="my-session", top_k=3)
        payload = mock_post.call_args[1]["json"]
        assert payload["session_id"] == "my-session"
        assert payload["top_k"] == 3

    def test_passes_top_k_in_payload(self):
        from frontend.streamlit_app import call_query_api
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = make_api_response()
        with patch("frontend.streamlit_app.requests.post", return_value=mock_resp) as mock_post:
            call_query_api("test query", session_id=None, top_k=7)
        payload = mock_post.call_args[1]["json"]
        assert payload["top_k"] == 7


# ── call_health_api tests ─────────────────────────────────────────────

class TestCallHealthAPI:
    def test_returns_dict_on_success(self):
        from frontend.streamlit_app import call_health_api
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = make_health_response()
        with patch("frontend.streamlit_app.requests.get", return_value=mock_resp):
            result = call_health_api()
        assert isinstance(result, dict)
        assert result["status"] == "healthy"

    def test_returns_none_on_connection_error(self):
        import requests as req
        from frontend.streamlit_app import call_health_api
        with patch("frontend.streamlit_app.requests.get",
                   side_effect=req.exceptions.ConnectionError()):
            result = call_health_api()
        assert result is None

    def test_sets_api_reachable_true_on_success(self):
        import streamlit as st
        from frontend.streamlit_app import call_health_api
        st.session_state = {}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = make_health_response()
        with patch("frontend.streamlit_app.requests.get", return_value=mock_resp):
            call_health_api()
        assert st.session_state.get("api_reachable") is True

    def test_sets_api_reachable_false_on_error(self):
        import requests as req
        import streamlit as st
        from frontend.streamlit_app import call_health_api
        st.session_state = {}
        with patch("frontend.streamlit_app.requests.get",
                   side_effect=req.exceptions.ConnectionError()):
            call_health_api()
        assert st.session_state.get("api_reachable") is False


# ── call_clear_api tests ──────────────────────────────────────────────

class TestCallClearAPI:
    def test_returns_true_on_success(self):
        from frontend.streamlit_app import call_clear_api
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("frontend.streamlit_app.requests.delete", return_value=mock_resp):
            assert call_clear_api("test-session") is True

    def test_returns_false_on_error(self):
        import requests as req
        from frontend.streamlit_app import call_clear_api
        with patch("frontend.streamlit_app.requests.delete",
                   side_effect=req.exceptions.ConnectionError()):
            assert call_clear_api("test-session") is False


# ── Session state tests ───────────────────────────────────────────────

class TestSessionState:
    def test_init_state_sets_defaults(self):
        import streamlit as st
        from frontend.streamlit_app import _init_state
        st.session_state = {}
        _init_state()
        assert "conversation"  in st.session_state
        assert "session_id"    in st.session_state
        assert "query_count"   in st.session_state
        assert "total_latency" in st.session_state
        assert "last_metrics"  in st.session_state

    def test_init_state_does_not_overwrite_existing(self):
        import streamlit as st
        from frontend.streamlit_app import _init_state
        st.session_state = {
            "conversation":  [{"role": "user", "content": "existing"}],
            "session_id":    "existing-session",
            "query_count":   3,
            "total_latency": 999.0,
            "last_metrics":  {"latency_ms": 100},
            "api_reachable": True,
            "top_k":         5,
        }
        _init_state()
        assert st.session_state["query_count"] == 3
        assert st.session_state["session_id"] == "existing-session"

    def test_conversation_list_starts_empty(self):
        import streamlit as st
        from frontend.streamlit_app import _init_state
        st.session_state = {}
        _init_state()
        assert st.session_state["conversation"] == []

    def test_query_count_starts_at_zero(self):
        import streamlit as st
        from frontend.streamlit_app import _init_state
        st.session_state = {}
        _init_state()
        assert st.session_state["query_count"] == 0
