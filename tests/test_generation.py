import sys
sys.path.insert(0, '/kaggle/working')

import pytest
from src.generation import PromptBuilder, CitationFormatter, LLMBackend, RAGResponse
from src.retrieval import ScoredDocument


# ── Helpers ───────────────────────────────────────────────────────────

def make_source(paper_id='p1', citation='Smith et al., 2023', title='Test Paper'):
    return ScoredDocument(
        chunk_id=f'{paper_id}_chunk_0',
        text='This paper presents a novel approach to machine learning.',
        metadata={
            'paper_id': paper_id,
            'citation_str': citation,
            'title': title,
            'year': 2023,
            'category': 'cs.LG',
        },
        score=0.85,
    )


# ── PromptBuilder tests ───────────────────────────────────────────────

class TestPromptBuilder:

    @pytest.fixture
    def builder(self):
        return PromptBuilder()

    def test_build_qa_prompt_returns_string(self, builder):
        prompt = builder.build_qa_prompt(
            query='What is attention?',
            context='[1] Vaswani et al., 2017\nAttention is a mechanism...',
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_contains_query(self, builder):
        query = 'What is attention mechanism?'
        prompt = builder.build_qa_prompt(query=query, context='Some context.')
        assert query in prompt

    def test_prompt_contains_context(self, builder):
        context = 'Attention allows models to focus on relevant tokens.'
        prompt = builder.build_qa_prompt(query='What is attention?', context=context)
        assert context in prompt

    def test_prompt_contains_system_instructions(self, builder):
        prompt = builder.build_qa_prompt(query='test', context='test context')
        assert 'citation' in prompt.lower() or 'cite' in prompt.lower()

    def test_prompt_includes_conversation_history(self, builder):
        history = [
            {'role': 'user', 'content': 'What is a transformer?'},
            {'role': 'assistant', 'content': 'A transformer is a model architecture.'},
        ]
        prompt = builder.build_qa_prompt(
            query='How does it use attention?',
            context='Context here.',
            conversation_history=history,
        )
        assert 'transformer' in prompt.lower()

    def test_condensation_prompt_returns_string(self, builder):
        history = [{'role': 'user', 'content': 'What is attention?'}]
        prompt = builder.build_condensation_prompt(
            query='How does it differ?',
            conversation_history=history,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_condensation_prompt_contains_follow_up(self, builder):
        history = [{'role': 'user', 'content': 'What is attention?'}]
        prompt = builder.build_condensation_prompt(
            query='How does it work exactly?',
            conversation_history=history,
        )
        assert 'How does it work exactly?' in prompt


# ── CitationFormatter tests ───────────────────────────────────────────

class TestCitationFormatter:

    @pytest.fixture
    def formatter(self):
        return CitationFormatter()

    def test_extract_citations_standard_format(self, formatter):
        answer = 'The attention mechanism (Vaswani et al., 2017) revolutionized NLP.'
        citations = formatter.extract_citations(answer)
        assert 'Vaswani et al., 2017' in citations

    def test_extract_citations_single_author(self, formatter):
        answer = 'This was shown by (Smith, 2021) in their work.'
        citations = formatter.extract_citations(answer)
        assert 'Smith, 2021' in citations

    def test_extract_citations_two_authors(self, formatter):
        answer = 'As shown by (Kipf and Welling, 2017), GCNs work well.'
        citations = formatter.extract_citations(answer)
        assert 'Kipf and Welling, 2017' in citations

    def test_extract_citations_none_present(self, formatter):
        answer = 'This answer has no citations at all.'
        citations = formatter.extract_citations(answer)
        assert citations == []

    def test_get_cited_sources_finds_match(self, formatter):
        source = make_source(citation='Smith et al., 2023')
        answer = 'The model (Smith et al., 2023) achieves high accuracy.'
        cited = formatter.get_cited_sources(answer, [source])
        assert 'Smith et al., 2023' in cited

    def test_get_cited_sources_no_match(self, formatter):
        source = make_source(citation='Jones et al., 2022')
        answer = 'This answer mentions no relevant papers.'
        cited = formatter.get_cited_sources(answer, [source])
        assert cited == []

    def test_validate_citations_valid(self, formatter):
        source = make_source(citation='Vaswani et al., 2017')
        answer = 'Attention was introduced by (Vaswani et al., 2017).'
        assert formatter.validate_citations(answer, [source]) is True

    def test_validate_citations_no_citations(self, formatter):
        answer = 'This answer has no citations.'
        assert formatter.validate_citations(answer, []) is True

    def test_enforce_length_short_text_unchanged(self, formatter):
        text = 'Short answer here.'
        result = formatter.enforce_length(text, min_tokens=5, max_tokens=800)
        assert result == text

    def test_enforce_length_truncates_long_text(self, formatter):
        long_text = ' '.join(['word'] * 900) + '.'
        result = formatter.enforce_length(long_text, min_tokens=50, max_tokens=800)
        assert len(result.split()) <= 820


# ── LLMBackend tests ─────────────────────────────────────────────────

class TestLLMBackend:

    def test_unloaded_backend_raises_on_generate(self):
        llm = LLMBackend(mode='fallback')
        with pytest.raises(RuntimeError):
            llm.generate('test prompt')

    def test_backend_name_before_load(self):
        llm = LLMBackend(mode='fallback')
        assert llm.backend_name == 'unloaded'

    def test_is_loaded_false_before_load(self):
        llm = LLMBackend(mode='fallback')
        assert llm.is_loaded is False

    def test_fallback_loads_successfully(self):
        llm = LLMBackend(mode='fallback')
        llm.load()
        assert llm.is_loaded is True
        assert llm.backend_name == 'flan_t5_large'

    def test_fallback_generates_non_empty_text(self):
        llm = LLMBackend(mode='fallback')
        llm.load()
        result = llm.generate('What is machine learning? Answer briefly.')
        assert isinstance(result, str)
        assert len(result.strip()) > 0


# ── RAGResponse tests ─────────────────────────────────────────────────

class TestRAGResponse:

    def test_total_latency_is_sum(self):
        response = RAGResponse(
            answer='Test answer.',
            citations=[],
            sources=[],
            confidence=0.8,
            model_used='test_model',
            retrieval_latency_ms=100.0,
            generation_latency_ms=200.0,
            low_confidence=False,
        )
        assert response.total_latency_ms == 300.0

    def test_response_fields_accessible(self):
        response = RAGResponse(
            answer='Test.',
            citations=['Smith et al., 2023'],
            sources=[],
            confidence=0.75,
            model_used='mistral_7b',
            retrieval_latency_ms=50.0,
            generation_latency_ms=150.0,
            low_confidence=False,
        )
        assert response.answer == 'Test.'
        assert response.citations == ['Smith et al., 2023']
        assert response.model_used == 'mistral_7b'
        assert response.low_confidence is False