from __future__ import annotations

import time
import uuid
from typing import Optional

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")
APP_TITLE    = "📚 ArXiv Research Assistant"
APP_SUBTITLE = "Powered by RAG + Mistral-7B + SentenceTransformers"

st.set_page_config(
    page_title="ArXiv Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
.user-bubble {
    background-color: #DCF8C6;
    border-radius: 12px;
    padding: 10px 14px;
    margin: 6px 0;
    text-align: right;
    font-size: 15px;
}
.assistant-bubble {
    background-color: #F0F0F0;
    border-radius: 12px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 15px;
}
.source-card {
    border: 1px solid #E0E0E0;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 13px;
    background-color: #FAFAFA;
}
.confidence-high { color: #2E7D32; font-weight: bold; }
.confidence-mid  { color: #F57F17; font-weight: bold; }
.confidence-low  { color: #C62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "conversation":   [],
        "session_id":     None,
        "query_count":    0,
        "total_latency":  0.0,
        "last_metrics":   {},
        "api_reachable":  None,
        "top_k":          5,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── API helpers ───────────────────────────────────────────────────────

def call_query_api(
    query: str,
    session_id: Optional[str],
    top_k: int = 5,
) -> Optional[dict]:
    try:
        payload = {
            "query":           query,
            "top_k":           top_k,
            "session_id":      session_id,
            "include_sources": True,
        }
        resp = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Is the backend running?")
        st.session_state["api_reachable"] = False
        return None
    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def call_health_api() -> Optional[dict]:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            st.session_state["api_reachable"] = True
            return resp.json()
        return None
    except Exception:
        st.session_state["api_reachable"] = False
        return None


def call_clear_api(session_id: str) -> bool:
    try:
        resp = requests.delete(
            f"{API_BASE_URL}/history/{session_id}",
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ── Render components ─────────────────────────────────────────────────

def render_header() -> None:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(APP_TITLE)
        st.caption(APP_SUBTITLE)
    with col2:
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)]"
            "(https://github.com/uwaoma-daniel/arxiv-rag-system)  "
            "[![API Docs](https://img.shields.io/badge/API-Docs-blue)]"
            f"({API_BASE_URL}/docs)",
        )
    st.divider()


def render_source_card(source: dict) -> None:
    score = source.get("relevance_score", 0)
    score_color = (
        "confidence-high" if score >= 0.7
        else "confidence-mid" if score >= 0.4
        else "confidence-low"
    )
    st.markdown(
        f'<div class="source-card">'
        f'<b>{source.get("title", "Unknown title")}</b><br>'
        f'{source.get("citation_str", "")} &nbsp;|&nbsp;'
        f'<code>{source.get("category", "")}</code> &nbsp;|&nbsp;'
        f'<span class="{score_color}">Score: {score:.3f}</span><br>'
        f'<small>{source.get("excerpt", "")[:180]}...</small>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_conversation() -> None:
    for turn in st.session_state["conversation"]:
        if turn["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 {turn["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble">🤖 {turn["content"]}</div>',
                unsafe_allow_html=True,
            )
            sources   = turn.get("sources", [])
            citations = turn.get("citations", [])
            latency   = turn.get("latency_ms", 0)

            cols = st.columns([2, 1])
            with cols[0]:
                if sources:
                    with st.expander(f"📄 Sources ({len(sources)} papers)", expanded=False):
                        for source in sources:
                            render_source_card(source)
            with cols[1]:
                if citations:
                    with st.expander("🔖 Citations", expanded=False):
                        for c in citations:
                            st.markdown(f"• {c}")
            st.caption(f"⏱️ {latency:.0f}ms")


def render_sidebar(health: Optional[dict]) -> int:
    with st.sidebar:
        st.header("⚙️ System Status")

        if health:
            status = health.get("status", "unknown")
            color  = "🟢" if status == "healthy" else "🟡"
            st.markdown(f"{color} **API:** {status.title()}")
            st.markdown(f"🤖 **Model:** {health.get('llm_backend', '—')}")
            st.markdown(f"🗄️ **Index:** {health.get('index_size', 0):,} docs")
            st.markdown(f"🔌 **Store:** {health.get('vector_store', '—')}")
            st.markdown(f"⏰ **Uptime:** {health.get('uptime_seconds', 0):.0f}s")
        else:
            st.markdown("🔴 **API:** Unreachable")

        st.divider()
        st.header("📊 Last Query Metrics")
        metrics = st.session_state.get("last_metrics", {})
        if metrics:
            st.metric("Latency",    f"{metrics.get('latency_ms', 0):.0f}ms")
            st.metric("Chunks",     metrics.get("retrieved_chunks", 0))
            st.metric("Confidence", f"{metrics.get('confidence_score', 0):.2f}")
            st.metric("Turn",       metrics.get("turn_number", 1))
        else:
            st.caption("No queries yet")

        st.divider()
        st.header("📈 Session Stats")
        qc = st.session_state["query_count"]
        tl = st.session_state["total_latency"]
        st.metric("Queries",     qc)
        st.metric("Avg Latency", f"{(tl / qc):.0f}ms" if qc > 0 else "—")

        st.divider()
        st.header("ℹ️ Settings")
        top_k = st.slider("Documents to retrieve (top_k)", 1, 10, 5)
        st.session_state["top_k"] = top_k

        st.divider()
        cost_note = "Built on Kaggle Free Tier — zero cost."
        st.caption(cost_note)
        st.caption("[GitHub](https://github.com/uwaoma-daniel/arxiv-rag-system)")

    return top_k


def render_input_area():
    st.divider()
    col1, col2, col3 = st.columns([6, 1, 1])
    with col1:
        query = st.text_input(
            "Ask a question about AI/ML research:",
            placeholder="e.g. What is the attention mechanism in transformers?",
            label_visibility="collapsed",
            key="query_input",
        )
    with col2:
        submitted = st.button("🔍 Ask", use_container_width=True, type="primary")
    with col3:
        cleared = st.button("🗑️ Clear", use_container_width=True)

    return query, submitted, cleared


# ── Main app ──────────────────────────────────────────────────────────

def main() -> None:
    _init_state()
    render_header()

    health = call_health_api()
    top_k  = render_sidebar(health)

    render_conversation()

    query, submitted, cleared = render_input_area()

    if cleared:
        sid = st.session_state.get("session_id")
        if sid:
            call_clear_api(sid)
        st.session_state["conversation"]  = []
        st.session_state["session_id"]    = None
        st.session_state["query_count"]   = 0
        st.session_state["total_latency"] = 0.0
        st.session_state["last_metrics"]  = {}
        st.rerun()

    if submitted and query and query.strip():
        with st.spinner("🔍 Retrieving relevant papers..."):
            result = call_query_api(
                query=query.strip(),
                session_id=st.session_state["session_id"],
                top_k=top_k,
            )

        if result:
            st.session_state["session_id"] = result.get("session_id")

            st.session_state["conversation"].append({
                "role":    "user",
                "content": query.strip(),
            })
            st.session_state["conversation"].append({
                "role":       "assistant",
                "content":    result.get("answer", ""),
                "sources":    result.get("sources", []),
                "citations":  result.get("citations", []),
                "latency_ms": result.get("latency_ms", 0),
            })

            st.session_state["query_count"]   += 1
            st.session_state["total_latency"] += result.get("latency_ms", 0)
            st.session_state["last_metrics"]   = {
                "latency_ms":       result.get("latency_ms", 0),
                "retrieved_chunks": result.get("retrieved_chunks", 0),
                "confidence_score": result.get("confidence_score", 0),
                "turn_number":      result.get("turn_number", 1),
            }
            st.rerun()


if __name__ == "__main__":
    main()
