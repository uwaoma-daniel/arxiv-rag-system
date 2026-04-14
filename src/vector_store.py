# ============================================================
# CELL 2: Write src/vector_store.py
# ============================================================

# Cell: Rewrite src/vector_store.py using line-list (no quote corruption)

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    metadata: dict
    score: float


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert(self, ids, embeddings, texts, metadatas): ...

    @abstractmethod
    def search(self, query_vector, top_k=10): ...

    @abstractmethod
    def get_count(self) -> int: ...

    @abstractmethod
    def health_check(self) -> bool: ...

    @abstractmethod
    def get_existing_ids(self) -> Set[str]: ...


class ChromaVectorStore(BaseVectorStore):
    COLLECTION_NAME = 'arxiv_abstracts'

    def __init__(self, persist_dir: str) -> None:
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None

    def connect(self) -> None:
        import chromadb
        from chromadb.config import Settings
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={'hnsw:space': 'cosine'},
        )
        logger.info(f'ChromaDB connected. Count: {self._collection.count():,}')

    def upsert(self, ids, embeddings, texts, metadatas) -> None:
        if self._collection is None:
            raise RuntimeError('Not connected. Call connect() first.')
        safe_metadatas = [self._sanitize_metadata(m) for m in metadatas]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=safe_metadatas,
        )

    def search(self, query_vector, top_k=10) -> List[SearchResult]:
        if self._collection is None:
            raise RuntimeError('Not connected.')
        count = self.get_count()
        if count == 0:
            return []
        results = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, count),
            include=['documents', 'metadatas', 'distances'],
        )
        search_results = []
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            score = float(np.clip(1.0 - (distance / 2.0), 0.0, 1.0))
            search_results.append(SearchResult(
                chunk_id=results['ids'][0][i],
                text=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                score=float(score),
            ))
        return search_results

    def get_count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()

    def health_check(self) -> bool:
        try:
            self.get_count()
            return True
        except Exception as e:
            logger.error(f'ChromaDB health check failed: {e}')
            return False

    def get_existing_ids(self) -> Set[str]:
        if self._collection is None:
            return set()
        all_ids = self._collection.get(include=[])['ids']
        return set(all_ids)

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ''
            elif isinstance(value, list):
                sanitized[key] = str(value)
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized


class QdrantVectorStore(BaseVectorStore):
    COLLECTION_NAME = 'arxiv_abstracts'
    VECTOR_SIZE = 384

    def __init__(self, url: str, api_key: str) -> None:
        self.url = url
        self.api_key = api_key
        self._client = None

    def connect(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        self._client = QdrantClient(url=self.url, api_key=self.api_key, timeout=30)
        existing = [c.name for c in self._client.get_collections().collections]
        if self.COLLECTION_NAME not in existing:
            self._client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=self.VECTOR_SIZE, distance=Distance.COSINE),
            )
        logger.info(f'Qdrant connected. Count: {self.get_count():,}')

    def upsert(self, ids, embeddings, texts, metadatas) -> None:
        from qdrant_client.models import PointStruct
        points = []
        for chunk_id, embedding, text, metadata in zip(ids, embeddings, texts, metadatas):
            int_id = int(hashlib.md5(chunk_id.encode()).hexdigest(), 16) % (2 ** 63)
            payload = {'chunk_id': chunk_id, 'text': text, **metadata}
            points.append(PointStruct(id=int_id, vector=embedding.tolist(), payload=payload))
        self._client.upsert(collection_name=self.COLLECTION_NAME, points=points)

    def search(self, query_vector, top_k=10) -> List[SearchResult]:
        results = self._client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchResult(
                chunk_id=r.payload.get('chunk_id', str(r.id)),
                text=r.payload.get('text', ''),
                metadata={k: v for k, v in r.payload.items() if k not in ('chunk_id', 'text')},
                score=float(r.score),
            )
            for r in results
        ]

    def get_count(self) -> int:
        if self._client is None:
            return 0
        info = self._client.get_collection(self.COLLECTION_NAME)
        return info.vectors_count or 0

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception as e:
            logger.error(f'Qdrant health check failed: {e}')
            return False

    def get_existing_ids(self) -> Set[str]:
        existing = set()
        next_page = None
        while True:
            results, next_page = self._client.scroll(
                collection_name=self.COLLECTION_NAME,
                offset=next_page,
                limit=1000,
                with_payload=['chunk_id'],
            )
            for point in results:
                if 'chunk_id' in point.payload:
                    existing.add(point.payload['chunk_id'])
            if next_page is None:
                break
        return existing


def create_vector_store(config: dict) -> BaseVectorStore:
    store_type = config.get('vector_store', {}).get('type', 'chroma')
    if store_type == 'chroma':
        persist_dir = config['vector_store'].get('chroma_persist_dir', './chroma_db')
        store = ChromaVectorStore(persist_dir=persist_dir)
        store.connect()
        return store
    elif store_type == 'qdrant':
        url = os.getenv('QDRANT_URL')
        key = os.getenv('QDRANT_API_KEY')
        if not url or not key:
            raise EnvironmentError('QDRANT_URL and QDRANT_API_KEY required.')
        store = QdrantVectorStore(url=url, api_key=key)
        store.connect()
        return store
    else:
        raise ValueError(f'Unknown store type: {store_type}')
