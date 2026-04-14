
import sys
sys.path.insert(0, '/kaggle/working')

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from src.embedding_pipeline import EmbeddingModel


@pytest.fixture(scope='module')
def embedder():
    em = EmbeddingModel(device='auto')
    em.load()
    return em


class TestEmbeddingModelShape:

    def test_single_embed_shape(self, embedder):
        vec = embedder.embed_single('Transformers are powerful models.')
        assert vec.shape == (384,), f'Expected (384,), got {vec.shape}'

    def test_batch_embed_shape(self, embedder):
        texts = ['First paper.', 'Second paper.', 'Third paper.']
        vecs = embedder.embed(texts, show_progress=False)
        assert vecs.shape == (3, 384), f'Expected (3, 384), got {vecs.shape}'

    def test_single_item_batch_shape(self, embedder):
        vecs = embedder.embed(['One sentence.'], show_progress=False)
        assert vecs.shape == (1, 384)

    def test_dimensions_property(self, embedder):
        assert embedder.dimensions == 384


class TestEmbeddingModelNormalization:

    def test_single_vector_normalized(self, embedder):
        vec = embedder.embed_single('Test normalization.')
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f'L2 norm = {norm:.6f}, expected 1.0'

    def test_batch_vectors_normalized(self, embedder):
        texts = ['Apple.', 'Banana.', 'Cherry.']
        vecs = embedder.embed(texts, show_progress=False)
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f'Norms: {norms}'

    def test_output_dtype_float32(self, embedder):
        vecs = embedder.embed(['dtype test'], show_progress=False)
        assert vecs.dtype == np.float32, f'Expected float32, got {vecs.dtype}'


class TestEmbeddingModelSimilarity:

    def test_similar_texts_beat_dissimilar(self, embedder):
        v_sim_a = embedder.embed_single('Attention mechanism in transformers.')
        v_sim_b = embedder.embed_single('Self-attention in neural networks.')
        similar_score = cos_sim(v_sim_a.reshape(1, -1), v_sim_b.reshape(1, -1))[0][0]

        v_dis_a = embedder.embed_single('Neural network attention for language.')
        v_dis_b = embedder.embed_single('Robot arm trajectory planning.')
        dissimilar_score = cos_sim(v_dis_a.reshape(1, -1), v_dis_b.reshape(1, -1))[0][0]

        assert similar_score > dissimilar_score, (
            f'Ranking broken: similar={similar_score:.4f} <= dissimilar={dissimilar_score:.4f}'
        )

    def test_similar_score_above_floor(self, embedder):
        v1 = embedder.embed_single('Attention mechanism in transformers.')
        v2 = embedder.embed_single('Self-attention in neural networks.')
        score = cos_sim(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
        assert score > 0.30, f'Similar pair score too low: {score:.4f} (floor > 0.30)'

    def test_near_identical_score_above_threshold(self, embedder):
        v1 = embedder.embed_single('We propose a transformer model for machine translation.')
        v2 = embedder.embed_single('We introduce a transformer architecture for neural machine translation.')
        score = cos_sim(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
        assert score > 0.80, f'Near-identical score too low: {score:.4f} (expected > 0.80)'

    def test_dissimilar_score_below_ceiling(self, embedder):
        v1 = embedder.embed_single('Neural network attention mechanisms for language.')
        v2 = embedder.embed_single('Robot arm trajectory planning in manufacturing.')
        score = cos_sim(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
        assert score < 0.60, f'Dissimilar texts too similar: {score:.4f} (expected < 0.60)'


class TestEmbeddingModelErrorHandling:

    def test_empty_list_raises_value_error(self, embedder):
        with pytest.raises(ValueError):
            embedder.embed([])

    def test_unloaded_model_raises_runtime_error(self):
        fresh = EmbeddingModel()
        with pytest.raises(RuntimeError):
            fresh.embed(['test'])


class TestEmbeddingModelSuccessMetric:

    def test_384_dim_success_metric(self, embedder):
        """Section 1.3 success metric: Embedding Dimensions = 384."""
        vec = embedder.embed_single('Success metric dimension check.')
        assert vec.shape[0] == 384
        assert embedder.dimensions == 384
        batch = embedder.embed(['a', 'b', 'c'], show_progress=False)
        assert batch.shape[1] == 384
