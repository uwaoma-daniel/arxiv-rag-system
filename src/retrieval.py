from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScoredDocument:
    chunk_id: str
    text: str
    metadata: dict
    score: float

    @property
    def paper_id(self) -> str:
        return self.metadata.get('paper_id', '')

    @property
    def citation_str(self) -> str:
        return self.metadata.get('citation_str', '')

    @property
    def title(self) -> str:
        return self.metadata.get('title', '')

    @property
    def year(self) -> int:
        return int(self.metadata.get('year', 0))

    @property
    def category(self) -> str:
        return self.metadata.get('category', '')


@dataclass
class RetrievalResult:
    documents: List[ScoredDocument]
    context_string: str
    retrieval_latency_ms: float
    all_scores: List[float]
    low_confidence: bool
    query: str = ''

    @property
    def top_score(self) -> float:
        return max(self.all_scores) if self.all_scores else 0.0

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.all_scores)) if self.all_scores else 0.0


# ---------------------------------------------------------------------------
# DocumentRetriever
# ---------------------------------------------------------------------------

class DocumentRetriever:
    """
    Retrieves semantically relevant documents from a vector store.

    Pipeline:
      1. Embed query -> 384-dim vector
      2. Cosine similarity search -> top_k * 2 candidates
      3. MMR reranking -> select top_k diverse results
      4. Deduplicate by paper_id (one chunk per paper)
      5. Low-confidence detection (all scores < threshold)
      6. Format context string for LLM prompt

    Args:
        vector_store: Connected BaseVectorStore instance.
        embedder: Loaded EmbeddingModel instance.
        score_threshold: Min score to consider a result relevant.
        mmr_lambda: MMR trade-off. 1.0 = pure relevance, 0.0 = pure diversity.
    """

    def __init__(
        self,
        vector_store,
        embedder,
        score_threshold: float = 0.30,
        mmr_lambda: float = 0.7,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.score_threshold = score_threshold
        self.mmr_lambda = mmr_lambda

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_mmr: bool = True,
    ) -> RetrievalResult:
        """
        Main retrieval entry point.

        Args:
            query: Natural language question.
            top_k: Number of documents to return.
            use_mmr: Whether to apply MMR reranking.

        Returns:
            RetrievalResult with documents and context string.

        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            raise ValueError('Query must not be empty.')

        t0 = time.perf_counter()

        # Step 1: embed query
        query_vec = self.embedder.embed_single(query.strip())

        # Step 2: retrieve more candidates than needed for MMR
        candidates_k = min(top_k * 2, self.vector_store.get_count())
        if candidates_k == 0:
            return self._empty_result(query, time.perf_counter() - t0)

        raw_results = self.vector_store.search(query_vec, top_k=candidates_k)

        if not raw_results:
            return self._empty_result(query, time.perf_counter() - t0)

        # Convert to ScoredDocument list
        candidates = [
            ScoredDocument(
                chunk_id=r.chunk_id,
                text=r.text,
                metadata=r.metadata,
                score=r.score,
            )
            for r in raw_results
        ]

        # Step 3: MMR reranking or simple top-k
        if use_mmr and len(candidates) > 1:
            selected = self._mmr_rerank(candidates, query_vec, top_k)
        else:
            selected = candidates[:top_k]

        # Step 4: deduplicate by paper_id
        selected = self._deduplicate_by_paper(selected)

        # Step 5: low-confidence detection
        all_scores = [d.score for d in selected]
        low_confidence = len(all_scores) == 0 or max(all_scores) < self.score_threshold

        if low_confidence:
            logger.warning(
                f'Low confidence retrieval for query: "{query[:60]}..." '
                f'(max_score={max(all_scores, default=0):.3f} < {self.score_threshold})'
            )

        # Step 6: format context string
        context = self._format_context(selected)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f'Retrieved {len(selected)} docs in {latency_ms:.1f}ms | '
            f'top_score={max(all_scores, default=0):.3f} | '
            f'low_confidence={low_confidence}'
        )

        return RetrievalResult(
            documents=selected,
            context_string=context,
            retrieval_latency_ms=latency_ms,
            all_scores=all_scores,
            low_confidence=low_confidence,
            query=query,
        )

    # ------------------------------------------------------------------
    # MMR reranking
    # ------------------------------------------------------------------

    def _mmr_rerank(
        self,
        candidates: List[ScoredDocument],
        query_vec: np.ndarray,
        top_k: int,
    ) -> List[ScoredDocument]:
        """
        Maximal Marginal Relevance reranking.

        Selects documents that are both relevant to the query AND
        diverse from already-selected documents.

        MMR score = lambda * relevance - (1 - lambda) * max_similarity_to_selected

        lambda=0.7: 70% relevance weight, 30% diversity weight.
        """
        if not candidates:
            return []

        # Embed all candidate texts for pairwise similarity
        candidate_texts = [d.text for d in candidates]
        candidate_vecs = self.embedder.embed(
            candidate_texts,
            batch_size=32,
            show_progress=False,
        )

        # Start with the highest-scoring candidate
        selected_indices: List[int] = [0]
        remaining_indices: List[int] = list(range(1, len(candidates)))

        while len(selected_indices) < min(top_k, len(candidates)) and remaining_indices:
            best_idx = None
            best_score = float('-inf')

            for idx in remaining_indices:
                # Relevance: original similarity score to query
                relevance = candidates[idx].score

                # Redundancy: max cosine similarity to any selected doc
                selected_vecs = candidate_vecs[selected_indices]
                sims_to_selected = cosine_similarity(
                    candidate_vecs[idx].reshape(1, -1),
                    selected_vecs,
                )[0]
                redundancy = float(np.max(sims_to_selected))

                mmr_score = (
                    self.mmr_lambda * relevance
                    - (1 - self.mmr_lambda) * redundancy
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _deduplicate_by_paper(
        self,
        docs: List[ScoredDocument],
    ) -> List[ScoredDocument]:
        """
        Keep only the highest-scoring chunk per paper_id.
        Prevents two chunks from the same paper dominating results.
        """
        seen_paper_ids: set = set()
        deduplicated: List[ScoredDocument] = []
        for doc in docs:
            pid = doc.paper_id
            if pid and pid not in seen_paper_ids:
                seen_paper_ids.add(pid)
                deduplicated.append(doc)
            elif not pid:
                # No paper_id in metadata — include anyway
                deduplicated.append(doc)
        return deduplicated

    def _format_context(self, docs: List[ScoredDocument]) -> str:
        """
        Format retrieved documents into a context string for the LLM prompt.

        Each document is prefixed with its metadata so the LLM can
        generate Author-Year citations from the context.
        """
        if not docs:
            return 'No relevant documents found.'

        parts = []
        for i, doc in enumerate(docs, 1):
            header = (
                f'[{i}] {doc.citation_str} | '
                f'{doc.title[:80]} | '
                f'Score: {doc.score:.3f}'
            )
            parts.append(f'{header}\n{doc.text}')

        return '\n\n'.join(parts)

    def _empty_result(self, query: str, elapsed: float) -> RetrievalResult:
        return RetrievalResult(
            documents=[],
            context_string='No relevant documents found.',
            retrieval_latency_ms=elapsed * 1000,
            all_scores=[],
            low_confidence=True,
            query=query,
        )