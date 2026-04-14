from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language question about research papers",
        examples=["What approaches exist for few-shot learning?"],
    )
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="Number of documents to retrieve",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for multi-turn conversation. Omit to start a new session.",
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source metadata in response",
    )

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        dangerous_patterns = ["<script>", "IGNORE PREVIOUS", "SYSTEM:"]
        for pattern in dangerous_patterns:
            if pattern.lower() in v.lower():
                raise ValueError("Query contains invalid content")
        return v.strip()


class Source(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    year: int
    category: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    citation_str: str
    excerpt: str


class ConversationTurn(BaseModel):
    role: str
    content: str
    turn_number: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    citations: List[str]
    latency_ms: float
    model_used: str
    retrieved_chunks: int
    session_id: str
    turn_number: int
    low_confidence: bool
    confidence_score: float


class HealthResponse(BaseModel):
    status: str
    llm_backend: str
    vector_store: str
    index_size: int
    embedding_model: str
    uptime_seconds: float
    version: str = "1.0.0"


class HistoryResponse(BaseModel):
    session_id: str
    turns: List[ConversationTurn]
    total_turns: int


class ClearResponse(BaseModel):
    message: str
    session_id: str
