from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    answer: str
    citations: List[str]
    sources: list
    confidence: float
    model_used: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    low_confidence: bool

    @property
    def total_latency_ms(self) -> float:
        return self.retrieval_latency_ms + self.generation_latency_ms


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Builds structured prompts for Mistral-7B-Instruct.

    Mistral uses [INST] / [/INST] tags for instruction formatting.
    System prompt is prepended inside the first [INST] block.
    Conversation history is interleaved as user/assistant turns.
    """

    SYSTEM_PROMPT = (
        'You are an expert research assistant specializing in AI and machine learning. '
        'Answer questions using ONLY the provided research paper excerpts. '
        'Every factual claim must be supported by an Author-Year citation '
        'from the provided sources, formatted as (Author et al., YYYY) or (Author, YYYY). '
        'If the context does not contain enough information to answer, '
        'say: "I do not have enough information in the provided papers to answer this question." '
        'Do not fabricate facts or citations. '
        'Write 3-5 substantive paragraphs.'
    )

    def build_qa_prompt(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Build the full QA prompt for Mistral-7B-Instruct.

        Format:
          <s>[INST] {system} \n\nContext:\n{context} \n\nQuestion: {query} [/INST]
        """
        history_str = ''
        if conversation_history:
            history_parts = []
            for turn in conversation_history[-6:]:
                role = turn.get('role', '')
                content = turn.get('content', '')
                if role == 'user':
                    history_parts.append(f'Previous question: {content}')
                elif role == 'assistant':
                    history_parts.append(f'Previous answer: {content[:200]}...')
            if history_parts:
                history_str = 'Conversation history:\n' + '\n'.join(history_parts) + '\n\n'

        prompt = (
            f'<s>[INST] {self.SYSTEM_PROMPT}\n\n'
            f'{history_str}'
            f'Research paper excerpts:\n{context}\n\n'
            f'Question: {query} [/INST]'
        )
        return prompt

    def build_condensation_prompt(
        self,
        query: str,
        conversation_history: List[Dict],
    ) -> str:
        """
        Build a prompt to condense a follow-up question into a standalone query.
        Used for multi-turn: rewrites 'How does it differ?' into
        'How does the attention mechanism differ from RNN architectures?'
        """
        history_parts = []
        for turn in conversation_history[-4:]:
            role = turn.get('role', '')
            content = turn.get('content', '')
            if role == 'user':
                history_parts.append(f'User: {content}')
            elif role == 'assistant':
                history_parts.append(f'Assistant: {content[:150]}...')

        history_str = '\n'.join(history_parts)

        return (
            f'<s>[INST] Given the following conversation history, '
            f'rewrite the follow-up question as a standalone question '
            f'that can be understood without the conversation context. '
            f'Return ONLY the rewritten question, nothing else.\n\n'
            f'Conversation history:\n{history_str}\n\n'
            f'Follow-up question: {query}\n\n'
            f'Standalone question: [/INST]'
        )


# ---------------------------------------------------------------------------
# LLMBackend
# ---------------------------------------------------------------------------

class LLMBackend:
    """
    LLM backend with automatic fallback chain.

    Priority:
      1. Local Mistral-7B-Instruct (4-bit quantized) — if GPU available
      2. Local flan-t5-large (CPU safe) — always available

    The backend is selected once at load time and reused for all queries.
    """

    PRIMARY_MODEL   = 'mistralai/Mistral-7B-Instruct-v0.2'
    FALLBACK_MODEL  = 'google/flan-t5-large'

    def __init__(self, mode: str = 'auto') -> None:
        """
        Args:
            mode: 'auto' | 'local' | 'fallback'
                  auto   = try Mistral first, fall back to flan-t5
                  local  = force Mistral-7B (fails if no GPU)
                  fallback = force flan-t5-large
        """
        self.mode = mode
        self._pipeline = None
        self._backend_name = 'unloaded'
        self._is_seq2seq = False

    def load(self) -> None:
        """Load the LLM pipeline. Call once at startup."""
        if self.mode == 'fallback':
            self._load_fallback()
            return

        if self.mode in ('auto', 'local'):
            try:
                self._load_mistral()
                return
            except Exception as e:
                logger.warning(f'Mistral-7B load failed: {e}')
                if self.mode == 'local':
                    raise

        self._load_fallback()

    def _load_mistral(self) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available — cannot load Mistral-7B')

        logger.info(f'Loading {self.PRIMARY_MODEL} with 4-bit quantization...')
        t0 = time.perf_counter()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.PRIMARY_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            self.PRIMARY_MODEL,
            quantization_config=bnb_config,
            device_map='auto',
            torch_dtype=torch.float16,
        )

        self._pipeline = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=800,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )

        self._backend_name = 'mistral_7b_4bit'
        self._is_seq2seq = False
        elapsed = time.perf_counter() - t0
        logger.info(f'Mistral-7B loaded in {elapsed:.1f}s')

    def _load_fallback(self) -> None:
        from transformers import pipeline

        logger.info(f'Loading fallback: {self.FALLBACK_MODEL}')
        t0 = time.perf_counter()

        self._pipeline = pipeline(
            'text2text-generation',
            model=self.FALLBACK_MODEL,
            max_new_tokens=512,
            device=-1,
        )

        self._backend_name = 'flan_t5_large'
        self._is_seq2seq = True
        elapsed = time.perf_counter() - t0
        logger.info(f'flan-t5-large loaded in {elapsed:.1f}s')

    def generate(self, prompt: str, max_new_tokens: int = 800) -> str:
        """
        Generate text from prompt.

        Returns raw generated text string.
        Raises RuntimeError if pipeline not loaded.
        """
        if self._pipeline is None:
            raise RuntimeError('LLM not loaded. Call LLMBackend.load() first.')

        t0 = time.perf_counter()

        if self._is_seq2seq:
            # flan-t5: text2text pipeline
            result = self._pipeline(
                prompt,
                max_new_tokens=min(max_new_tokens, 512),
            )
            text = result[0]['generated_text']
        else:
            # Mistral: causal LM pipeline
            result = self._pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
            )
            text = result[0]['generated_text']

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f'Generated {len(text.split())} words in {elapsed:.1f}ms')
        return text.strip()

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None


# ---------------------------------------------------------------------------
# CitationFormatter
# ---------------------------------------------------------------------------

class CitationFormatter:
    """
    Extracts and validates Author-Year citations in generated answers.

    The LLM is instructed to cite using the citation_str from metadata,
    e.g. 'Vaswani et al., 2017'. This class validates that every citation
    in the answer corresponds to an actual retrieved source.
    """

    # Matches: (Smith et al., 2023) or (Smith, 2023) or (Smith and Jones, 2023)
    CITATION_PATTERN = re.compile(
        r'\(([A-Z][a-zA-Z]+(?:\s+et\s+al\.?|\s+and\s+[A-Z][a-zA-Z]+)?,\s*\d{4})\)',
    )

    def extract_citations(self, answer: str) -> List[str]:
        """Extract all (Author, YYYY) citations from answer text."""
        return self.CITATION_PATTERN.findall(answer)

    def get_cited_sources(self, answer: str, sources: list) -> List[str]:
        """
        Return citation strings from sources that are actually
        referenced in the answer text.
        """
        cited = []
        for source in sources:
            citation_str = source.citation_str if hasattr(source, 'citation_str') else source.get('citation_str', '')
            if citation_str and citation_str in answer:
                cited.append(citation_str)
        return cited

    def validate_citations(
        self,
        answer: str,
        sources: list,
    ) -> bool:
        """
        Check that every citation in the answer exists in the source list.
        Returns True if all citations are valid (no hallucinated references).
        """
        extracted = self.extract_citations(answer)
        if not extracted:
            return True

        valid_citations = set(
            s.citation_str if hasattr(s, 'citation_str') else s.get('citation_str', '')
            for s in sources
        )

        for citation in extracted:
            if not any(citation in vc or vc in citation for vc in valid_citations):
                logger.warning(f'Hallucinated citation detected: {citation}')
                return False
        return True

    def enforce_length(
        self,
        text: str,
        min_tokens: int = 100,
        max_tokens: int = 800,
    ) -> str:
        """
        Truncate text at sentence boundary if over max_tokens.
        Returns text as-is if within bounds.
        """
        words = text.split()
        if len(words) <= max_tokens:
            return text

        # Truncate at max_tokens words, then find nearest sentence end
        truncated = ' '.join(words[:max_tokens])
        last_period = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?'),
        )
        if last_period > len(truncated) * 0.7:
            return truncated[:last_period + 1]
        return truncated + '.'


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Orchestrates: retrieve -> prompt -> generate -> format citations

    Supports multi-turn conversation via question condensation:
    Follow-up questions are rewritten into standalone queries before
    vector search, so retrieval is not confused by pronouns or
    implicit references to previous turns.
    """

    def __init__(
        self,
        retriever,
        llm: LLMBackend,
        prompt_builder: Optional[PromptBuilder] = None,
        citation_formatter: Optional[CitationFormatter] = None,
        top_k: int = 5,
        max_history_turns: int = 6,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.citation_formatter = citation_formatter or CitationFormatter()
        self.top_k = top_k
        self.max_history_turns = max_history_turns

    def query(
        self,
        question: str,
        conversation_history: Optional[List[Dict]] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Run end-to-end RAG for a single question.

        Args:
            question: User's natural language question.
            conversation_history: List of {role, content} dicts.
                                  None or [] for first turn.
            top_k: Override default top_k for this query.

        Returns:
            RAGResponse with answer, citations, sources, latency.
        """
        if not question or not question.strip():
            raise ValueError('Question must not be empty.')

        k = top_k or self.top_k
        history = conversation_history or []

        # Trim history to max_history_turns
        if len(history) > self.max_history_turns * 2:
            history = history[-(self.max_history_turns * 2):]

        # Step 1: condense follow-up question if history exists
        search_query = question
        if history:
            search_query = self._condense_question(question, history)
            logger.info(f'Condensed query: "{search_query}"')

        # Step 2: retrieve documents
        retrieval_result = self.retriever.retrieve(search_query, top_k=k)

        # Step 3: handle low-confidence
        if retrieval_result.low_confidence:
            return RAGResponse(
                answer=(
                    'I do not have enough information in the provided papers '
                    'to answer this question confidently. '
                    'Please try rephrasing or asking about a different topic.'
                ),
                citations=[],
                sources=retrieval_result.documents,
                confidence=retrieval_result.mean_score,
                model_used=self.llm.backend_name,
                retrieval_latency_ms=retrieval_result.retrieval_latency_ms,
                generation_latency_ms=0.0,
                low_confidence=True,
            )

        # Step 4: build prompt
        prompt = self.prompt_builder.build_qa_prompt(
            query=question,
            context=retrieval_result.context_string,
            conversation_history=history,
        )

        # Step 5: generate answer
        gen_start = time.perf_counter()
        raw_answer = self.llm.generate(prompt)
        generation_latency_ms = (time.perf_counter() - gen_start) * 1000

        # Step 6: enforce length
        answer = self.citation_formatter.enforce_length(
            raw_answer, min_tokens=50, max_tokens=800
        )

        # Step 7: extract citations
        citations = self.citation_formatter.get_cited_sources(
            answer, retrieval_result.documents
        )

        logger.info(
            f'RAG complete | gen={generation_latency_ms:.0f}ms | '
            f'answer_words={len(answer.split())} | citations={len(citations)}'
        )

        return RAGResponse(
            answer=answer,
            citations=citations,
            sources=retrieval_result.documents,
            confidence=retrieval_result.mean_score,
            model_used=self.llm.backend_name,
            retrieval_latency_ms=retrieval_result.retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            low_confidence=False,
        )

    def _condense_question(
        self,
        question: str,
        history: List[Dict],
    ) -> str:
        """
        Rewrite a follow-up question into a standalone query.
        Falls back to original question if LLM not available or fails.
        """
        try:
            prompt = self.prompt_builder.build_condensation_prompt(
                query=question,
                conversation_history=history,
            )
            condensed = self.llm.generate(prompt, max_new_tokens=100)
            condensed = condensed.strip().strip('"').strip()
            if condensed and len(condensed) > 5:
                return condensed
        except Exception as e:
            logger.warning(f'Question condensation failed: {e}. Using original.')
        return question