---
title: ArXiv RAG API
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
---

# ArXiv RAG System — FastAPI Backend

REST API for the zero-cost RAG system over 10,000+ arXiv abstracts.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Submit a question, get a cited answer |
| GET | `/health` | System status |
| GET | `/history/{session_id}` | Conversation history |
| DELETE | `/history/{session_id}` | Clear history |

Interactive API docs at `/docs`.
