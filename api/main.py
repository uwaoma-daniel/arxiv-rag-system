from __future__ import annotations

import logging
import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    ClearResponse,
    ConversationTurn,
    HealthResponse,
    HistoryResponse,
    QueryRequest,
    QueryResponse,
    Source,
)
from src.embedding_pipeline import EmbeddingModel
from src.generation import CitationFormatter, LLMBackend, PromptBuilder, RAGPipeline
from src.retrieval import DocumentRetriever
from src.vector_store import ChromaVectorStore, QdrantVectorStore, QdrantVectorStore

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("api")

# ── Constants ─────────────────────────────────────────────────────────
VERSION        = "1.0.0"
MAX_SESSIONS   = 100
MAX_TURNS      = 6
SESSION_TTL_S  = 1800
RATE_LIMIT_RPM = 10

# Set TEST_MODE=1 to disable rate limiting (used in pytest)
TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"

# ── Global state ──────────────────────────────────────────────────────
_start_time: float = 0.0
_rag_pipeline: Optional[RAGPipeline] = None
_embedder: Optional[EmbeddingModel] = None
_vector_store = None  # ChromaVectorStore or QdrantVectorStore
_llm: Optional[LLMBackend] = None
_config: dict = {}

_sessions: OrderedDict[str, List[dict]] = OrderedDict()
_rate_tracker: Dict[str, List[float]] = {}


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time, _rag_pipeline, _embedder, _vector_store, _llm, _config

    _start_time = time.perf_counter()
    logger.info("Starting ArXiv RAG API...")

    # Load base config
    with open("config.yaml", "r") as f:
        _config = yaml.safe_load(f)

    # Allow entrypoint.py to override config via env var (HF Spaces)
    config_override = os.environ.get("RAG_CONFIG_OVERRIDE", "")
    if config_override:
        import yaml as _yaml
        _config = _yaml.safe_load(config_override)
        logger.info("Loaded config override from RAG_CONFIG_OVERRIDE")

    _embedder = EmbeddingModel(
        model_name=_config["embedding"]["model_name"],
        device=_config["embedding"]["device"],
    )
    _embedder.load()

    # Select vector store based on config
    store_type = _config["vector_store"].get("type", "chroma")
    if store_type == "qdrant":
        qdrant_url = (
            os.environ.get("QDRANT_URL")
            or _config["vector_store"].get("qdrant_url", "")
        )
        qdrant_key = (
            os.environ.get("QDRANT_API_KEY")
            or _config["vector_store"].get("qdrant_api_key", "")
        )
        _vector_store = QdrantVectorStore(
            url=qdrant_url,
            api_key=qdrant_key,
        )
        logger.info(f"Using Qdrant vector store: {qdrant_url[:40]}...")
    else:
        _vector_store = ChromaVectorStore(
            persist_dir=_config["vector_store"]["chroma_persist_dir"],
        )
        logger.info("Using ChromaDB vector store")
    _vector_store.connect()

    _llm = LLMBackend(mode=_config["llm"]["mode"])
    _llm.load()

    retriever = DocumentRetriever(
        vector_store=_vector_store,
        embedder=_embedder,
        score_threshold=_config["retrieval"]["score_threshold"],
        mmr_lambda=_config["retrieval"]["mmr_lambda"],
    )
    _rag_pipeline = RAGPipeline(
        retriever=retriever,
        llm=_llm,
        prompt_builder=PromptBuilder(),
        citation_formatter=CitationFormatter(),
        top_k=_config["retrieval"]["top_k"],
        max_history_turns=_config["api"]["max_turns"],
    )
    logger.info("RAG pipeline ready")
    yield
    logger.info("Shutting down ArXiv RAG API")


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ArXiv RAG API",
    description="Retrieval-Augmented Generation over 10K+ arXiv abstracts",
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware ────────────────────────────────────────────────────────

@app.middleware("http")
async def latency_and_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} | {latency_ms:.1f}ms"
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = f"{latency_ms:.1f}"
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # ── Skip rate limiting in test mode ──────────────────────────────
    if TEST_MODE:
        return await call_next(request)

    if request.url.path in ("/health", "/docs", "/openapi.json"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = 60.0

    timestamps = _rate_tracker.get(client_ip, [])
    timestamps = [t for t in timestamps if now - t < window]

    if len(timestamps) >= RATE_LIMIT_RPM:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit: {RATE_LIMIT_RPM} requests/minute"},
        )

    timestamps.append(now)
    _rate_tracker[client_ip] = timestamps
    return await call_next(request)


# ── Session helpers ───────────────────────────────────────────────────

def _get_or_create_session(session_id: Optional[str]) -> str:
    if session_id and session_id in _sessions:
        _sessions.move_to_end(session_id)
        return session_id

    new_id = session_id or str(uuid.uuid4())
    if len(_sessions) >= MAX_SESSIONS:
        _sessions.popitem(last=False)
    _sessions[new_id] = []
    return new_id


def _get_history(session_id: str) -> List[dict]:
    return _sessions.get(session_id, [])


def _append_history(session_id: str, role: str, content: str) -> None:
    history = _sessions.setdefault(session_id, [])
    history.append({"role": role, "content": content})
    max_messages = MAX_TURNS * 2
    if len(history) > max_messages:
        _sessions[session_id] = history[-max_messages:]


def _build_sources(documents) -> List[Source]:
    sources = []
    for doc in documents:
        meta = doc.metadata
        authors_raw = meta.get("authors_raw", meta.get("citation_str", ""))
        sources.append(Source(
            paper_id=meta.get("paper_id", ""),
            title=meta.get("title", ""),
            authors=[authors_raw] if authors_raw else [],
            year=int(meta.get("year", 0)),
            category=meta.get("category", ""),
            relevance_score=round(float(doc.score), 4),
            citation_str=meta.get("citation_str", ""),
            excerpt=doc.text[:200],
        ))
    return sources


# ── Endpoints ─────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if _rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    t0         = time.perf_counter()
    session_id = _get_or_create_session(request.session_id)
    history    = _get_history(session_id)
    turn_number = (len(history) // 2) + 1

    logger.info(
        f"Query | session={session_id} | turn={turn_number} | "
        f"query={request.query[:60]!r}"
    )

    try:
        response = _rag_pipeline.query(
            question=request.query,
            conversation_history=history,
            top_k=request.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="RAG pipeline error")

    _append_history(session_id, "user",      request.query)
    _append_history(session_id, "assistant", response.answer)

    latency_ms = (time.perf_counter() - t0) * 1000
    sources    = _build_sources(response.sources) if request.include_sources else []

    return QueryResponse(
        answer=response.answer,
        sources=sources,
        citations=response.citations,
        latency_ms=round(latency_ms, 2),
        model_used=response.model_used,
        retrieved_chunks=len(response.sources),
        session_id=session_id,
        turn_number=turn_number,
        low_confidence=response.low_confidence,
        confidence_score=round(float(response.confidence), 4),
    )


@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    index_size = 0
    store_name = "unavailable"
    if _vector_store is not None:
        try:
            index_size = _vector_store.get_count()
            store_name = "chroma"
        except Exception:
            store_name = "error"

    uptime = time.perf_counter() - _start_time

    return HealthResponse(
        status="healthy" if _rag_pipeline is not None else "degraded",
        llm_backend=_llm.backend_name if _llm else "unloaded",
        vector_store=store_name,
        index_size=index_size,
        embedding_model=_config.get("embedding", {}).get("model_name", "unknown"),
        uptime_seconds=round(uptime, 2),
        version=VERSION,
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def history_endpoint(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    raw_history = _sessions[session_id]
    turns = [
        ConversationTurn(
            role=msg["role"],
            content=msg["content"],
            turn_number=(i // 2) + 1,
        )
        for i, msg in enumerate(raw_history)
    ]
    return HistoryResponse(
        session_id=session_id,
        turns=turns,
        total_turns=len(turns) // 2,
    )


@app.delete("/history/{session_id}", response_model=ClearResponse)
async def clear_history_endpoint(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    _sessions[session_id] = []
    logger.info(f"History cleared for session: {session_id}")

    return ClearResponse(
        message="History cleared",
        session_id=session_id,
    )
