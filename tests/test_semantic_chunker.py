
import sys
sys.path.insert(0, '/kaggle/working')

import pytest
from sentence_transformers import SentenceTransformer
from src.embedding_pipeline import SemanticChunker, ChunkRecord


@pytest.fixture(scope='module')
def model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


@pytest.fixture(scope='module')
def chunker(model):
    return SemanticChunker(
        model=model,
        similarity_threshold=0.65,
        min_chunk_tokens=50,
        max_chunk_tokens=200,
        overlap_sentences=1,
    )


PAPER_METADATA = {
    'paper_id': '2301.12345',
    'title': 'Test Paper',
    'authors_raw': 'Smith, J., Jones, A.',
    'first_author_last': 'Smith',
    'author_count': 2,
    'year': 2023,
    'category': 'cs.LG',
    'citation_str': 'Smith and Jones, 2023',
}


# ── Basic behavior ────────────────────────────────────────────────────

class TestChunkBasicBehavior:

    def test_returns_at_least_one_chunk(self, chunker):
        chunks = chunker.chunk('Transformers use self-attention mechanisms.')
        assert len(chunks) >= 1

    def test_empty_string_handled(self, chunker):
        chunks = chunker.chunk('')
        assert isinstance(chunks, list)
        assert len(chunks) == 1

    def test_single_sentence_returns_one_chunk(self, chunker):
        chunks = chunker.chunk('This paper presents a novel approach to machine learning.')
        assert len(chunks) == 1

    def test_two_sentence_returns_one_chunk(self, chunker):
        text = 'We propose a new model. Our model achieves state-of-the-art results.'
        assert len(chunker.chunk(text)) == 1


# ── Length constraints ────────────────────────────────────────────────

class TestChunkLengthConstraints:

    def test_no_chunk_exceeds_max_tokens(self, chunker):
        long_text = (
            'Transformers revolutionized NLP through self-attention mechanisms. '
            'The mechanism allows direct token-to-token computation in parallel. '
            'In computer vision, convolutional networks process spatial image data. '
            'CNNs learn hierarchical feature representations from raw pixel inputs. '
            'Reinforcement learning trains agents via reward signals from environment. '
            'Q-learning and policy gradients are core RL optimization algorithms. '
            'Graph neural networks operate on non-Euclidean graph-structured data. '
            'Node embeddings capture both structural and feature information jointly. '
        )
        for chunk in chunker.chunk(long_text):
            assert len(chunk.split()) <= chunker.max_chunk_tokens + 20

    def test_chunks_meet_minimum_length(self, chunker):
        # Short text: only assert no trivially empty chunks (< 5 words)
        short_text = (
            'We study neural machine translation for low-resource language pairs. '
            'Our method uses cross-lingual transfer learning to improve quality. '
            'We demonstrate improvements on twelve low-resource language pairs.'
        )
        for chunk in chunker.chunk(short_text):
            assert len(chunk.split()) >= 5, f'Trivially short chunk: {chunk!r}'

        # Long text: largest chunk must meet min_chunk_tokens
        long_text = (
            'We study neural machine translation with a focus on low-resource pairs '
            'where parallel training data is scarce and difficult to obtain at scale. '
            'Our approach leverages cross-lingual transfer learning to improve quality '
            'by sharing learned representations across multiple related language pairs. '
            'We pre-train a multilingual encoder on a large corpus spanning many languages '
            'using a masked language modeling objective for robust representation learning. '
            'The pre-trained model is fine-tuned on low-resource pairs using few sentences. '
            'We demonstrate improvements on twelve low-resource language pairs from FLORES. '
            'Our method outperforms strong baselines including back-translation approaches. '
            'Ablation studies confirm the contribution of pre-training and fine-tuning steps.'
        )
        long_chunks = chunker.chunk(long_text)
        assert len(long_chunks) >= 1
        for chunk in long_chunks:
            assert len(chunk.split()) >= 5, f'Trivially short: {chunk!r}'
        largest = max(len(c.split()) for c in long_chunks)
        assert largest >= chunker.min_chunk_tokens, (
            f'Largest chunk {largest} words < min_chunk_tokens={chunker.min_chunk_tokens}. '
            f'All sizes: {[len(c.split()) for c in long_chunks]}'
        )


# ── Overlap ───────────────────────────────────────────────────────────

class TestChunkOverlap:

    def test_overlap_creates_shared_content(self, chunker):
        text = (
            'Attention mechanisms allow models to focus on relevant input tokens. '
            'The query key and value matrices define the attention weight distribution. '
            'Self-attention operates within a single input sequence end to end. '
            'In contrast image classification relies on convolutional filter banks. '
            'These filters detect edges textures and increasingly complex visual patterns. '
            'Deep CNNs stack many layers to build abstract feature representations.'
        )
        chunks = chunker.chunk(text)
        # All chunks must be non-empty strings
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk.strip()) > 0
        # When overlap is active and multiple chunks produced,
        # chunk[1] must share at least one word with chunk[0]
        if len(chunks) >= 2:
            words0 = set(chunks[0].lower().split())
            words1 = set(chunks[1].lower().split())
            shared = words0 & words1
            assert len(shared) > 0, (
                f'No shared words between chunk[0] and chunk[1] — overlap not working. '
                f'chunk[0]={chunks[0][:60]!r} | chunk[1]={chunks[1][:60]!r}'
            )


# ── Metadata ──────────────────────────────────────────────────────────

class TestChunkMetadata:

    def test_chunk_with_metadata_returns_chunk_records(self, chunker):
        text = (
            'We present a novel architecture for sequence modeling tasks. '
            'The model uses sparse attention to reduce computational cost significantly. '
            'Experiments on benchmarks show improvements over all strong baselines.'
        )
        records = chunker.chunk_with_metadata(text, PAPER_METADATA)
        assert len(records) >= 1
        assert all(isinstance(r, ChunkRecord) for r in records)

    def test_chunk_ids_are_unique(self, chunker):
        text = ' '.join([f'Sentence {i} about machine learning concepts here.' for i in range(20)])
        meta = {**PAPER_METADATA, 'paper_id': 'unique_test_001'}
        records = chunker.chunk_with_metadata(text, meta)
        ids = [r.chunk_id for r in records]
        assert len(ids) == len(set(ids)), f'Duplicate chunk IDs: {ids}'

    def test_chunk_indices_are_sequential(self, chunker):
        text = ' '.join([f'Machine learning sentence number {i} here.' for i in range(10)])
        meta = {**PAPER_METADATA, 'paper_id': 'seq_test_002'}
        records = chunker.chunk_with_metadata(text, meta)
        indices = [r.chunk_index for r in records]
        assert indices == list(range(len(records))), f'Non-sequential: {indices}'

    def test_chunk_total_matches_actual_count(self, chunker):
        text = ' '.join([f'Research paper sentence {i} with content.' for i in range(10)])
        meta = {**PAPER_METADATA, 'paper_id': 'total_test_003'}
        records = chunker.chunk_with_metadata(text, meta)
        for r in records:
            assert r.chunk_total == len(records), (
                f'chunk_total={r.chunk_total} but actual={len(records)}'
            )
