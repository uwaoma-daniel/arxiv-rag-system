import sys
import tempfile
sys.path.insert(0, '/kaggle/working')

import numpy as np
import pytest
from src.embedding_pipeline import EmbeddingModel
from src.vector_store import ChromaVectorStore
from src.retrieval import DocumentRetriever, ScoredDocument, RetrievalResult


# ── Shared fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope='module')
def embedder():
    em = EmbeddingModel(device='auto')
    em.load()
    return em


@pytest.fixture(scope='module')
def populated_store(embedder):
    """ChromaDB with 10 realistic arXiv-style documents."""
    tmpdir = tempfile.mkdtemp()
    store = ChromaVectorStore(persist_dir=tmpdir)
    store.connect()

    docs = [
        ('att_001', 'Attention mechanisms allow models to focus on relevant input tokens.',
         'attention_paper', 'Vaswani et al., 2017', 'Attention is All You Need', 2017, 'cs.CL'),
        ('att_002', 'Self-attention computes query key and value representations.',
         'attention_paper', 'Vaswani et al., 2017', 'Attention is All You Need', 2017, 'cs.CL'),
        ('few_001', 'Few-shot learning enables models to generalize from very few examples.',
         'few_shot_paper', 'Brown et al., 2020', 'Language Models are Few-Shot Learners', 2020, 'cs.LG'),
        ('few_002', 'Meta-learning trains models to adapt quickly to new tasks.',
         'meta_paper', 'Finn et al., 2017', 'Model-Agnostic Meta-Learning', 2017, 'cs.LG'),
        ('gnn_001', 'Graph neural networks aggregate information from neighboring nodes.',
         'gnn_paper', 'Kipf and Welling, 2017', 'Semi-Supervised Classification with GCNs', 2017, 'cs.LG'),
        ('rl_001', 'Reinforcement learning trains agents via reward signals from environment.',
         'rl_paper', 'Mnih et al., 2015', 'Human-level control through deep RL', 2015, 'cs.AI'),
        ('cv_001', 'Convolutional neural networks learn hierarchical visual features.',
         'cv_paper', 'LeCun et al., 2015', 'Deep Learning', 2015, 'cs.CV'),
        ('bert_001', 'BERT uses bidirectional transformers for language representation.',
         'bert_paper', 'Devlin et al., 2019', 'BERT Pre-training of Deep Bidirectional Transformers', 2019, 'cs.CL'),
        ('gan_001', 'Generative adversarial networks train a generator and discriminator.',
         'gan_paper', 'Goodfellow et al., 2014', 'Generative Adversarial Nets', 2014, 'cs.LG'),
        ('opt_001', 'Adam optimizer adapts learning rates using first and second moments.',
         'opt_paper', 'Kingma and Ba, 2015', 'Adam: A Method for Stochastic Optimization', 2015, 'cs.LG'),
    ]

    texts = [d[1] for d in docs]
    embeddings = embedder.embed(texts, show_progress=False)
    ids = [d[0] for d in docs]
    metadatas = [
        {
            'paper_id': d[2], 'citation_str': d[3],
            'title': d[4], 'year': d[5], 'category': d[6],
        }
        for d in docs
    ]
    store.upsert(ids=ids, embeddings=embeddings, texts=texts, metadatas=metadatas)
    return store


@pytest.fixture(scope='module')
def retriever(populated_store, embedder):
    return DocumentRetriever(
        vector_store=populated_store,
        embedder=embedder,
        score_threshold=0.30,
        mmr_lambda=0.7,
    )


# ── Test 1: Basic retrieval returns results ────────────────────────────

class TestBasicRetrieval:

    def test_returns_retrieval_result(self, retriever):
        result = retriever.retrieve('What is attention mechanism?')
        assert isinstance(result, RetrievalResult)

    def test_returns_correct_number_of_docs(self, retriever):
        result = retriever.retrieve('attention mechanism', top_k=3)
        assert len(result.documents) <= 3

    def test_documents_are_scored_documents(self, retriever):
        result = retriever.retrieve('few-shot learning')
        assert all(isinstance(d, ScoredDocument) for d in result.documents)

    def test_scores_in_valid_range(self, retriever):
        result = retriever.retrieve('neural network training')
        for doc in result.documents:
            assert 0.0 <= doc.score <= 1.0, f'Score out of range: {doc.score}'

    def test_empty_query_raises_value_error(self, retriever):
        with pytest.raises(ValueError):
            retriever.retrieve('')

    def test_whitespace_query_raises_value_error(self, retriever):
        with pytest.raises(ValueError):
            retriever.retrieve('   ')


# ── Test 2: Relevance ordering ────────────────────────────────────────

class TestRelevanceOrdering:

    def test_attention_query_returns_attention_docs(self, retriever):
        result = retriever.retrieve('attention mechanism transformers', top_k=3)
        texts = ' '.join(d.text.lower() for d in result.documents)
        assert 'attention' in texts, (
            f'Expected attention-related docs, got: {[d.text[:50] for d in result.documents]}'
        )

    def test_top_result_higher_score_than_last(self, retriever):
        result = retriever.retrieve('graph neural networks', top_k=5)
        if len(result.documents) >= 2:
            assert result.documents[0].score >= result.documents[-1].score, (
                'Documents not ordered by score'
            )


# ── Test 3: MMR diversity ─────────────────────────────────────────────

class TestMMRDiversity:

    def test_mmr_returns_diverse_results(self, retriever):
        # Attention paper has 2 chunks (att_001, att_002) with same paper_id
        # After deduplication, only 1 chunk per paper should appear
        result = retriever.retrieve('attention mechanism', top_k=5, use_mmr=True)
        paper_ids = [d.paper_id for d in result.documents]
        assert len(paper_ids) == len(set(paper_ids)), (
            f'Duplicate paper_ids found: {paper_ids}'
        )

    def test_mmr_result_count_leq_top_k(self, retriever):
        result = retriever.retrieve('machine learning', top_k=4, use_mmr=True)
        assert len(result.documents) <= 4


# ── Test 4: Deduplication ────────────────────────────────────────────

class TestDeduplication:

    def test_no_duplicate_paper_ids(self, retriever):
        result = retriever.retrieve('attention transformer self-attention', top_k=5)
        paper_ids = [d.paper_id for d in result.documents if d.paper_id]
        assert len(paper_ids) == len(set(paper_ids)), (
            f'Duplicate paper_ids after deduplication: {paper_ids}'
        )


# ── Test 5: Context string ────────────────────────────────────────────

class TestContextString:

    def test_context_string_is_not_empty(self, retriever):
        result = retriever.retrieve('deep learning optimization')
        assert len(result.context_string) > 0

    def test_context_string_contains_citation(self, retriever):
        result = retriever.retrieve('attention mechanism transformers')
        assert any(
            doc.citation_str in result.context_string
            for doc in result.documents
            if doc.citation_str
        )

    def test_context_string_contains_score(self, retriever):
        result = retriever.retrieve('few-shot learning')
        assert 'Score:' in result.context_string


# ── Test 6: Low confidence detection ─────────────────────────────────

class TestLowConfidence:

    def test_low_confidence_on_unrelated_query(self, retriever):
        # Query about cooking — completely unrelated to AI papers
        result = retriever.retrieve(
            'how to bake chocolate cake with butter and flour',
            top_k=5,
        )
        # Either low_confidence=True OR scores are all low
        if result.low_confidence:
            assert result.top_score < 0.60
        # At minimum, scores should be lower than for relevant queries
        relevant_result = retriever.retrieve('attention mechanism transformers')
        assert relevant_result.top_score >= result.top_score, (
            'Relevant query should score higher than unrelated query'
        )

    def test_empty_store_returns_low_confidence(self, embedder):
        tmpdir = tempfile.mkdtemp()
        empty_store = ChromaVectorStore(persist_dir=tmpdir)
        empty_store.connect()
        retriever_empty = DocumentRetriever(
            vector_store=empty_store,
            embedder=embedder,
            score_threshold=0.30,
        )
        result = retriever_empty.retrieve('attention mechanism')
        assert result.low_confidence is True
        assert result.documents == []


# ── Test 7: Latency tracking ──────────────────────────────────────────

class TestLatencyTracking:

    def test_latency_is_positive(self, retriever):
        result = retriever.retrieve('transformer architecture')
        assert result.retrieval_latency_ms > 0

    def test_all_scores_populated(self, retriever):
        result = retriever.retrieve('convolutional neural network features')
        assert len(result.all_scores) == len(result.documents)


# ── Test 8: ScoredDocument properties ────────────────────────────────

class TestScoredDocumentProperties:

    def test_scored_document_paper_id_property(self, retriever):
        result = retriever.retrieve('few-shot learning')
        for doc in result.documents:
            assert isinstance(doc.paper_id, str)

    def test_scored_document_citation_str_property(self, retriever):
        result = retriever.retrieve('attention mechanism')
        for doc in result.documents:
            assert isinstance(doc.citation_str, str)

    def test_retrieval_result_mean_score(self, retriever):
        result = retriever.retrieve('deep learning')
        if result.all_scores:
            assert 0.0 <= result.mean_score <= 1.0

    def test_retrieval_result_top_score(self, retriever):
        result = retriever.retrieve('neural network')
        if result.all_scores:
            assert result.top_score == max(result.all_scores)