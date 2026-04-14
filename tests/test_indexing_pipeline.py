# ============================================================
# CELL 6: Phase 4 Tests — IndexingPipeline
# ============================================================

import sys
import json
import tempfile
sys.path.insert(0, '/kaggle/working')

import numpy as np
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer
from src.embedding_pipeline import (
    EmbeddingModel, SemanticChunker, IndexingPipeline, IndexingReport,
)
from src.vector_store import ChromaVectorStore


# ── Shared fixtures (module-scoped: load once for all tests) ──────────

@pytest.fixture(scope='module')
def embedder():
    em = EmbeddingModel(device='auto')
    em.load()
    return em


@pytest.fixture(scope='module')
def chunker(embedder):
    return SemanticChunker(
        model=embedder._model,
        similarity_threshold=0.65,
        min_chunk_tokens=50,
        max_chunk_tokens=200,
        overlap_sentences=1,
    )


def make_test_df(n=5):
    abstracts = [
        ('We propose a novel attention mechanism for transformer models. '
         'The mechanism allows the model to focus on relevant tokens. '
         'We evaluate on multiple NLP benchmarks and show improvements.'),
        ('This paper presents a new approach to few-shot learning. '
         'We use meta-learning to train models that generalize quickly. '
         'Experiments on miniImageNet show state-of-the-art results.'),
        ('We study reward shaping in reinforcement learning. '
         'Our approach uses auxiliary tasks to provide dense feedback. '
         'The method converges faster than standard policy gradient methods.'),
        ('Graph neural networks have achieved success on many tasks. '
         'We propose a new aggregation function for node embeddings. '
         'Our method outperforms baselines on node classification.'),
        ('We introduce a new model for image segmentation. '
         'The model uses a hierarchical feature pyramid network. '
         'We demonstrate improvements on COCO and ADE20K datasets.'),
    ]
    records = []
    for i in range(n):
        records.append({
            'id': f'2301.{i:05d}',
            'title': f'Test Paper {i}: A Novel Approach',
            'authors': f'Smith{i}, J., Jones{i}, A.',
            'first_author_last': f'Smith{i}',
            'author_count': 2,
            'year': 2023,
            'categories': 'cs.LG',
            'citation_str': f'Smith{i} and Jones{i}, 2023',
            'abstract': abstracts[i % len(abstracts)],
        })
    return pd.DataFrame(records)


@pytest.fixture
def pipeline_and_store(embedder, chunker):
    tmpdir_db = tempfile.mkdtemp()
    tmpdir_ck = tempfile.mkdtemp()
    store = ChromaVectorStore(persist_dir=tmpdir_db)
    store.connect()
    pipeline = IndexingPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_stores=[store],
        checkpoint_dir=tmpdir_ck,
        doc_batch_size=3,
        embed_batch_size=16,
        checkpoint_interval=3,
    )
    return pipeline, store


# ── Tests ─────────────────────────────────────────────────────────────

class TestIndexingPipelineOutput:

    def test_returns_indexing_report(self, pipeline_and_store):
        pipeline, store = pipeline_and_store
        report = pipeline.run(make_test_df(5))
        assert isinstance(report, IndexingReport)

    def test_documents_indexed_count_gt_0(self, pipeline_and_store):
        pipeline, store = pipeline_and_store
        pipeline.run(make_test_df(5))
        assert store.get_count() > 0

    def test_chunk_count_matches_report(self, pipeline_and_store):
        pipeline, store = pipeline_and_store
        report = pipeline.run(make_test_df(5))
        assert store.get_count() == report.total_chunks

    def test_all_documents_processed(self, pipeline_and_store):
        pipeline, store = pipeline_and_store
        report = pipeline.run(make_test_df(5))
        total = report.total_documents + report.skipped_documents
        assert total == 5

    def test_avg_chunks_per_doc_positive(self, pipeline_and_store):
        pipeline, store = pipeline_and_store
        report = pipeline.run(make_test_df(5))
        assert report.avg_chunks_per_doc > 0

    def test_timing_fields_positive(self, pipeline_and_store):
        pipeline, store = pipeline_and_store
        report = pipeline.run(make_test_df(5))
        assert report.total_time_seconds > 0
        assert report.embedding_time_seconds > 0

    def test_report_to_dict_json_serializable(self, pipeline_and_store):
        pipeline, store = pipeline_and_store
        report = pipeline.run(make_test_df(3))
        d = report.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0


class TestIndexingPipelineIdempotency:

    def test_second_run_skips_all(self, pipeline_and_store):
        pipeline, store = pipeline_and_store
        pipeline.run(make_test_df(5))
        count_after_first = store.get_count()
        report2 = pipeline.run(make_test_df(5))
        assert store.get_count() == count_after_first
        assert report2.skipped_documents == 5
        assert report2.total_documents == 0


class TestIndexingPipelineCheckpoint:

    def test_checkpoint_file_created(self, pipeline_and_store):
        pipeline, store = pipeline_and_store
        pipeline.run(make_test_df(5))
        assert pipeline.checkpoint_file.exists()

    def test_checkpoint_contains_correct_ids(self, pipeline_and_store):
        pipeline, store = pipeline_and_store
        df = make_test_df(5)
        pipeline.run(df)
        with open(pipeline.checkpoint_file) as f:
            saved_ids = set(json.load(f))
        for eid in df['id'].tolist():
            assert eid in saved_ids, f'ID {eid} missing from checkpoint'


class TestIndexingPipelineSearch:

    def test_indexed_chunks_are_searchable(self, pipeline_and_store, embedder):
        pipeline, store = pipeline_and_store
        pipeline.run(make_test_df(5))
        query_vec = embedder.embed_single('attention mechanism transformer')
        results = store.search(query_vec, top_k=3)
        assert len(results) > 0
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_metadata_present_in_search_results(self, pipeline_and_store, embedder):
        pipeline, store = pipeline_and_store
        pipeline.run(make_test_df(5))
        query_vec = embedder.embed_single('few-shot learning meta-learning')
        results = store.search(query_vec, top_k=1)
        assert len(results) >= 1
        meta = results[0].metadata
        required_keys = ['paper_id', 'title', 'year', 'category', 'citation_str']
        for key in required_keys:
            assert key in meta, f'Missing metadata key: {key}'
