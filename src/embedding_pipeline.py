from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import List, Dict, Optional, Tuple, Set

import nltk
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    paper_id: str
    title: str
    authors_raw: str
    first_author_last: str
    author_count: int
    year: int
    category: str
    citation_str: str
    chunk_index: int
    chunk_total: int
    word_count: int
    abstract_length: int


@dataclass
class IndexingReport:
    total_documents: int = 0
    total_chunks: int = 0
    failed_documents: int = 0
    skipped_documents: int = 0
    avg_chunks_per_doc: float = 0.0
    total_time_seconds: float = 0.0
    embedding_time_seconds: float = 0.0
    upsert_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            'total_documents': self.total_documents,
            'total_chunks': self.total_chunks,
            'failed_documents': self.failed_documents,
            'skipped_documents': self.skipped_documents,
            'avg_chunks_per_doc': round(self.avg_chunks_per_doc, 2),
            'total_time_seconds': round(self.total_time_seconds, 2),
            'embedding_time_seconds': round(self.embedding_time_seconds, 2),
            'upsert_time_seconds': round(self.upsert_time_seconds, 2),
        }

    def print_summary(self) -> None:
        print('\n' + '=' * 60)
        print('INDEXING REPORT')
        print('=' * 60)
        print(f'  Documents indexed:    {self.total_documents:,}')
        print(f'  Documents skipped:    {self.skipped_documents:,}')
        print(f'  Documents failed:     {self.failed_documents:,}')
        print(f'  Total chunks:         {self.total_chunks:,}')
        print(f'  Avg chunks/doc:       {self.avg_chunks_per_doc:.2f}')
        print(f'  Embedding time:       {self.embedding_time_seconds:.1f}s')
        print(f'  Upsert time:          {self.upsert_time_seconds:.1f}s')
        print(f'  Total time:           {self.total_time_seconds:.1f}s')
        print('=' * 60)


class SemanticChunker:

    def __init__(
        self,
        model: SentenceTransformer,
        similarity_threshold: float = 0.65,
        min_chunk_tokens: int = 50,
        max_chunk_tokens: int = 200,
        overlap_sentences: int = 1,
    ) -> None:
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return [text] if text else ['']
        sentences = self._tokenize_sentences(text)
        if len(sentences) <= 2:
            return [' '.join(sentences)]
        sentence_embeddings = self.model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        breakpoints = self._find_breakpoints(sentence_embeddings)
        raw_chunks = self._group_sentences(sentences, breakpoints)
        constrained = self._apply_constraints(raw_chunks)
        return self._apply_overlap(constrained, sentences)

    def chunk_with_metadata(self, text: str, paper_metadata: dict) -> List[ChunkRecord]:
        chunks = self.chunk(text)
        abstract_word_count = len(text.split())
        records = []
        for i, chunk_text in enumerate(chunks):
            pid = paper_metadata['paper_id']
            records.append(ChunkRecord(
                chunk_id=f'{pid}_chunk_{i}',
                text=chunk_text,
                paper_id=pid,
                title=paper_metadata['title'],
                authors_raw=paper_metadata['authors_raw'],
                first_author_last=paper_metadata['first_author_last'],
                author_count=paper_metadata['author_count'],
                year=paper_metadata['year'],
                category=paper_metadata['category'],
                citation_str=paper_metadata['citation_str'],
                chunk_index=i,
                chunk_total=len(chunks),
                word_count=len(chunk_text.split()),
                abstract_length=abstract_word_count,
            ))
        return records

    def _tokenize_sentences(self, text: str) -> List[str]:
        raw = nltk.sent_tokenize(text)
        return [s.strip() for s in raw if len(s.split()) >= 3]

    def _find_breakpoints(self, embeddings: np.ndarray) -> List[int]:
        breakpoints = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1),
            )[0][0]
            if sim < self.similarity_threshold:
                breakpoints.append(i)
        return breakpoints

    def _group_sentences(self, sentences: List[str], breakpoints: List[int]) -> List[List[str]]:
        if not breakpoints:
            return [sentences]
        groups: List[List[str]] = []
        current: List[str] = []
        for i, s in enumerate(sentences):
            current.append(s)
            if i in breakpoints:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return groups

    def _apply_constraints(self, raw_chunks: List[List[str]]) -> List[List[str]]:
        after_split: List[List[str]] = []
        for group in raw_chunks:
            after_split.extend(self._split_if_too_long(group))
        return self._merge_if_too_short(after_split)

    def _split_if_too_long(self, group: List[str]) -> List[List[str]]:
        if len(' '.join(group).split()) <= self.max_chunk_tokens:
            return [group]
        mid = len(group) // 2
        result: List[List[str]] = []
        if group[:mid]:
            result.extend(self._split_if_too_long(group[:mid]))
        if group[mid:]:
            result.extend(self._split_if_too_long(group[mid:]))
        return result

    def _merge_if_too_short(self, groups: List[List[str]]) -> List[List[str]]:
        if len(groups) <= 1:
            return groups
        merged: List[List[str]] = []
        i = 0
        while i < len(groups):
            current = groups[i]
            if len(' '.join(current).split()) < self.min_chunk_tokens and i + 1 < len(groups):
                merged.append(current + groups[i + 1])
                i += 2
            else:
                merged.append(current)
                i += 1
        return merged

    def _apply_overlap(self, chunks: List[List[str]], original_sentences: List[str]) -> List[str]:
        if not chunks:
            return []
        result: List[str] = []
        for i, group in enumerate(chunks):
            if i == 0 or self.overlap_sentences == 0:
                result.append(' '.join(group))
            else:
                overlap = chunks[i - 1][-self.overlap_sentences:]
                result.append(' '.join(overlap + group))
        return result


class EmbeddingModel:

    MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
    EXPECTED_DIMENSIONS = 384

    def __init__(self, model_name: str = None, device: str = 'auto') -> None:
        self.model_name = model_name or self.MODEL_NAME
        self._device = self._resolve_device(device)
        self._model: Optional[SentenceTransformer] = None

    def load(self) -> SentenceTransformer:
        if self._model is None:
            start = time.perf_counter()
            self._model = SentenceTransformer(self.model_name, device=self._device)
            elapsed = time.perf_counter() - start
            logger.info(f'Model loaded in {elapsed:.2f}s on {self._device}')
            self._validate_dimensions()
        return self._model

    def embed(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        if self._model is None:
            raise RuntimeError('Model not loaded. Call EmbeddingModel.load() first.')
        if not texts:
            raise ValueError('Cannot embed empty text list.')
        clean = [t if t.strip() else 'empty' for t in texts]
        try:
            embeddings = self._model.encode(
                clean,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and batch_size > 16:
                return self.embed(texts, batch_size=batch_size // 2, show_progress=show_progress)
            raise
        assert embeddings.shape == (len(texts), self.EXPECTED_DIMENSIONS)
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text], batch_size=1, show_progress=False)[0]

    @property
    def dimensions(self) -> int:
        return self.EXPECTED_DIMENSIONS

    @property
    def device(self) -> str:
        return self._device

    def _resolve_device(self, device: str) -> str:
        if device != 'auto':
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
        except ImportError:
            pass
        return 'cpu'

    def _validate_dimensions(self) -> None:
        test = self._model.encode(['dim check'], normalize_embeddings=True, convert_to_numpy=True)
        actual = test.shape[1]
        assert actual == self.EXPECTED_DIMENSIONS, (
            f'Model produces {actual}-dim. Expected {self.EXPECTED_DIMENSIONS}.'
        )


class IndexingPipeline:

    def __init__(
        self,
        chunker: SemanticChunker,
        embedder: EmbeddingModel,
        vector_stores: list,
        checkpoint_dir: str = './checkpoints',
        doc_batch_size: int = 100,
        embed_batch_size: int = 64,
        checkpoint_interval: int = 1000,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_stores = vector_stores
        self.checkpoint_dir = _Path(checkpoint_dir)
        self.doc_batch_size = doc_batch_size
        self.embed_batch_size = embed_batch_size
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / 'indexed_ids.json'

    def run(self, df: pd.DataFrame) -> IndexingReport:
        report = IndexingReport()
        t0 = time.perf_counter()
        already = self._load_checkpoint()
        for store in self.vector_stores:
            already.update(sid.split('_chunk_')[0] for sid in store.get_existing_ids())
        to_process = df[~df['id'].isin(already)].copy()
        report.skipped_documents = len(df) - len(to_process)
        if len(to_process) == 0:
            report.total_time_seconds = time.perf_counter() - t0
            return report
        newly: List[str] = []
        emb_t = upsert_t = 0.0
        pbar = tqdm(total=len(to_process), desc='Indexing', unit='docs')
        for start in range(0, len(to_process), self.doc_batch_size):
            batch = to_process.iloc[start: start + self.doc_batch_size]
            try:
                chunks, et, ut = self._process_batch(batch)
                emb_t += et
                upsert_t += ut
                report.total_chunks += len(chunks)
                newly.extend(set(c.paper_id for c in chunks))
                report.total_documents += len(batch)
            except Exception as e:
                logger.error(f'Batch failed: {e}', exc_info=True)
                report.failed_documents += len(batch)
            pbar.update(len(batch))
            done = start + len(batch)
            if done % self.checkpoint_interval < self.doc_batch_size:
                self._save_checkpoint(already | set(newly))
        pbar.close()
        self._save_checkpoint(already | set(newly))
        report.embedding_time_seconds = emb_t
        report.upsert_time_seconds = upsert_t
        report.total_time_seconds = time.perf_counter() - t0
        if report.total_documents > 0:
            report.avg_chunks_per_doc = report.total_chunks / report.total_documents
        report.print_summary()
        return report

    def _process_batch(self, batch: pd.DataFrame) -> Tuple[List[ChunkRecord], float, float]:
        all_chunks: List[ChunkRecord] = []
        for _, row in batch.iterrows():
            try:
                meta = self._row_to_metadata(row)
                all_chunks.extend(self.chunker.chunk_with_metadata(str(row['abstract']), meta))
            except Exception as e:
                logger.warning(f"Chunk failed {row.get('id', '??')}: {e}")
        if not all_chunks:
            return [], 0.0, 0.0
        texts = [c.text for c in all_chunks]
        t0 = time.perf_counter()
        embeddings = self.embedder.embed(texts, batch_size=self.embed_batch_size, show_progress=False)
        et = time.perf_counter() - t0
        ids = [c.chunk_id for c in all_chunks]
        metas = [self._to_meta_dict(c) for c in all_chunks]
        t1 = time.perf_counter()
        for store in self.vector_stores:
            store.upsert(ids=ids, embeddings=embeddings, texts=texts, metadatas=metas)
        ut = time.perf_counter() - t1
        return all_chunks, et, ut

    def _row_to_metadata(self, row: pd.Series) -> dict:
        return {
            'paper_id': str(row['id']),
            'title': str(row.get('title', '')),
            'authors_raw': str(row.get('authors', '')),
            'first_author_last': str(row.get('first_author_last', 'Unknown')),
            'author_count': int(row.get('author_count', 1)),
            'year': int(row.get('year', 2023)),
            'category': str(row.get('categories', 'cs.AI')),
            'citation_str': str(row.get('citation_str', '')),
        }

    @staticmethod
    def _to_meta_dict(r: ChunkRecord) -> dict:
        return {
            'chunk_id': r.chunk_id, 'paper_id': r.paper_id,
            'title': r.title, 'authors_raw': r.authors_raw,
            'first_author_last': r.first_author_last,
            'author_count': r.author_count, 'year': r.year,
            'category': r.category, 'citation_str': r.citation_str,
            'chunk_index': r.chunk_index, 'chunk_total': r.chunk_total,
            'word_count': r.word_count, 'abstract_length': r.abstract_length,
        }

    def _save_checkpoint(self, ids: Set[str]) -> None:
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(ids), f)

    def _load_checkpoint(self) -> Set[str]:
        if not self.checkpoint_file.exists():
            return set()
        try:
            with open(self.checkpoint_file) as f:
                return set(json.load(f))
        except Exception:
            return set()
