
import sys
import tempfile
sys.path.insert(0, '/kaggle/working')

import numpy as np
import pytest
from src.vector_store import ChromaVectorStore, SearchResult


def make_embeddings(n, dims=384, seed=42):
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, dims).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def make_metadata(n):
    return [
        {
            'paper_id': f'paper_{i:04d}',
            'title': f'Test Paper {i}',
            'year': 2023,
            'category': 'cs.LG',
            'citation_str': f'Author{i}, 2023',
            'first_author_last': f'Author{i}',
            'author_count': 1,
            'chunk_index': 0,
            'chunk_total': 1,
        }
        for i in range(n)
    ]


@pytest.fixture
def store():
    tmpdir = tempfile.mkdtemp()
    s = ChromaVectorStore(persist_dir=tmpdir)
    s.connect()
    return s


class TestChromaUpsertAndCount:

    def test_upsert_and_count(self, store):
        store.upsert(['c1', 'c2', 'c3'], make_embeddings(3), ['T1.', 'T2.', 'T3.'], make_metadata(3))
        assert store.get_count() == 3

    def test_upsert_is_idempotent(self, store):
        ids = ['dup']
        emb = make_embeddings(1)
        store.upsert(ids, emb, ['Dup.'], make_metadata(1))
        store.upsert(ids, emb, ['Dup.'], make_metadata(1))
        assert store.get_count() == 1

    def test_get_existing_ids(self, store):
        ids = ['e1', 'e2', 'e3']
        store.upsert(ids, make_embeddings(3), ['t1', 't2', 't3'], make_metadata(3))
        existing = store.get_existing_ids()
        for eid in ids:
            assert eid in existing
        assert 'nonexistent' not in existing


class TestChromaSearch:

    def test_search_returns_search_result_objects(self, store):
        store.upsert(['cA', 'cB'], make_embeddings(2), ['NLP.', 'Robotics.'], make_metadata(2))
        results = store.search(make_embeddings(1)[0], top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_scores_in_0_1_range(self, store):
        store.upsert(
            [f'c{i}' for i in range(5)],
            make_embeddings(5),
            [f'Paper {i}.' for i in range(5)],
            make_metadata(5),
        )
        results = store.search(make_embeddings(1)[0], top_k=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0, f'Score {r.score} out of [0,1] range'

    def test_exact_match_score_near_one(self, store):
        emb = make_embeddings(1)
        store.upsert(['exact'], emb, ['Exact.'], make_metadata(1))
        results = store.search(emb[0], top_k=1)
        assert results[0].score > 0.99, f'Exact match score: {results[0].score:.6f}'

    def test_top_k_limit_respected(self, store):
        store.upsert(
            [f'k{i}' for i in range(10)],
            make_embeddings(10),
            [f't{i}' for i in range(10)],
            make_metadata(10),
        )
        assert len(store.search(make_embeddings(1)[0], top_k=3)) == 3
        assert len(store.search(make_embeddings(1)[0], top_k=5)) == 5

    def test_search_empty_store_returns_empty_list(self, store):
        assert store.search(make_embeddings(1)[0], top_k=5) == []


class TestChromaMetadata:

    def test_metadata_stored_and_retrieved(self, store):
        metadata = {
            'paper_id': 'meta_paper', 'title': 'Meta Title',
            'year': 2023, 'category': 'cs.AI',
            'citation_str': 'Meta et al., 2023',
            'first_author_last': 'Meta', 'author_count': 3,
            'chunk_index': 0, 'chunk_total': 1,
        }
        store.upsert(['m0'], make_embeddings(1), ['Meta content.'], [metadata])
        results = store.search(make_embeddings(1)[0], top_k=1)
        stored = results[0].metadata
        assert stored.get('paper_id') == 'meta_paper'
        assert stored.get('year') == 2023


class TestChromaHealth:

    def test_health_check_passes(self, store):
        assert store.health_check() is True
