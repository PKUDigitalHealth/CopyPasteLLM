"""
CP-Link Pipeline - Cohesive responses with transition generation

This pipeline extracts core sentences from context and generates intelligent
transition sentences to connect them into a cohesive response.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from CopyPasteLLM.pipelines.base import PipelineBase
from CopyPasteLLM.utils.text import TextProcessor, create_standard_response_structure


@dataclass
class CoreSentence:
    """Core sentence data structure"""
    content: str
    relevance_score: float


@dataclass
class TransitionSentence:
    """Transition sentence data structure"""
    content: str
    position: int


class CPLinkPipeline(PipelineBase):
    """
    CP-Link Pipeline - Extractive RAG with intelligent transitions.

    Algorithm:
    1. Extract core sentences from context
    2. Generate transition sentences to connect them
    3. Organize final response with transitions

    Args:
        verbose: Whether to print verbose output
        llm_backend: Optional pre-initialized LLMBackend
        **llm_kwargs: Additional arguments for LLMBackend
    """

    def __init__(self, verbose: bool = False, llm_backend=None, **llm_kwargs):
        super().__init__(
            pipeline_name="CP-Link",
            verbose=verbose,
            llm_backend=llm_backend,
            **llm_kwargs
        )

        self.text_processor = TextProcessor()

    def _get_extraction_prompt(self, context: str, query: str) -> str:
        """Get prompt for core sentence extraction"""
        return f"""You are an expert at extracting relevant information from text. Your task is to extract complete sentences from the context that directly answer the user's question.

## Context
{context}

## Question
{query}

## Instructions
1. Extract COMPLETE sentences EXACTLY as they appear in the context
2. NO modifications, paraphrasing, or combining of sentences
3. Preserve all terminology, measurements, and symbols exactly as written
4. ONLY extract sentences that are directly relevant to answering the question
5. Maintain original wording and phrasing precisely
6. If no relevant sentences are found, output: NO_RELEVANT_SENTENCES_FOUND

## Output Format
For each relevant sentence, output on a new line:
EXTRACTED: [exact sentence from context]

Now extract the relevant sentences:"""

    def _extract_core_sentences(
        self,
        context: str,
        query: str
    ) -> List[CoreSentence]:
        """Extract core sentences from context"""
        self._print_verbose("Extracting core sentences...")

        extraction_prompt = self._get_extraction_prompt(context, query)

        try:
            extracted_text = self._call_llm(extraction_prompt, agent_name="Core Sentence Extractor")

            core_sentences = []

            # Check for no sentences found
            if "NO_RELEVANT_SENTENCES_FOUND" in extracted_text:
                self._print_verbose("LLM found no relevant sentences")
                return []

            # Parse extracted sentences
            lines = extracted_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('EXTRACTED:'):
                    sentence_content = line.replace('EXTRACTED:', '').strip()
                    if sentence_content and len(sentence_content) > 10:
                        # Validate sentence exists in context
                        if sentence_content in context or self._find_similar_sentence(sentence_content, context):
                            relevance_score = self._calculate_relevance(sentence_content, query)
                            core_sentences.append(CoreSentence(
                                content=sentence_content,
                                relevance_score=relevance_score
                            ))

            if not core_sentences:
                self._print_verbose("Primary extraction failed, trying fallback...", "WARNING")
                # Fallback: extract quoted content
                quoted_sentences = re.findall(r'"([^"]+)"', extracted_text)
                for sentence in quoted_sentences:
                    if len(sentence) > 10:
                        if sentence in context or self._find_similar_sentence(sentence, context):
                            relevance_score = self._calculate_relevance(sentence, query)
                            core_sentences.append(CoreSentence(
                                content=sentence,
                                relevance_score=relevance_score
                            ))

            # Sort by relevance
            core_sentences.sort(key=lambda x: x.relevance_score, reverse=True)

            self._print_verbose(f"Extracted {len(core_sentences)} core sentences")
            return core_sentences

        except Exception as e:
            self._print_verbose(f"Core sentence extraction failed: {str(e)}", "ERROR")
            return []

    def _find_similar_sentence(self, sentence: str, context: str) -> bool:
        """Check if sentence exists in context (fuzzy match)"""
        words = sentence.split()
        if len(words) < 3:
            return False

        key_words = words[:min(5, len(words))]
        key_phrase = ' '.join(key_words)
        return key_phrase in context

    def _calculate_relevance(self, sentence: str, query: str) -> float:
        """Calculate relevance score between sentence and query"""
        sentence_words = set(sentence.lower().split())
        query_words = set(query.lower().split())

        if len(query_words) == 0:
            return 0.0

        overlap = len(sentence_words.intersection(query_words))
        return overlap / len(query_words)

    def _generate_transitions(
        self,
        core_sentences: List[CoreSentence],
        query: str
    ) -> List[TransitionSentence]:
        """Generate transition sentences"""
        self._print_verbose("Generating transition sentences...")

        if len(core_sentences) <= 1:
            return []

        # Build core sentences summary
        core_summary = "\n".join([
            f"{i+1}. {cs.content}"
            for i, cs in enumerate(core_sentences)
        ])

        transition_prompt = f"""## Instruction
You are a professional text organization expert. Generate concise transition sentences to connect the core sentences and make the response flow naturally.

## Query
{query}

## Core Sentences
{core_summary}

## Requirements
1. Transition sentences should be concise (no more than 15 words)
2. They should logically connect adjacent core sentences
3. Focus on creating smooth flow between ideas
4. Common types: progression, contrast, addition, conclusion

## Output Format
[TRANSITION_1_2]transition sentence content[/TRANSITION_1_2]
[TRANSITION_2_3]transition sentence content[/TRANSITION_2_3]
...

Optionally add:
[INTRO]introduction sentence[/INTRO]
[CONCLUSION]conclusion sentence[/CONCLUSION]

Please generate transitions:"""

        try:
            transition_text = self._call_llm(transition_prompt, agent_name="Transition Generator")

            transition_sentences = []

            # Parse intro
            intro_pattern = r'\[INTRO\](.*?)\[/INTRO\]'
            intro_match = re.search(intro_pattern, transition_text, re.DOTALL)
            if intro_match:
                intro_content = intro_match.group(1).strip()
                transition_sentences.append(TransitionSentence(
                    content=intro_content,
                    position=0
                ))

            # Parse transitions
            transition_pattern = r'\[TRANSITION_(\d+)_(\d+)\](.*?)\[/TRANSITION_\d+_\d+\]'
            transition_matches = re.findall(transition_pattern, transition_text, re.DOTALL)

            for from_idx, to_idx, transition_content in transition_matches:
                transition_content = transition_content.strip()
                position = int(from_idx)

                transition_sentences.append(TransitionSentence(
                    content=transition_content,
                    position=position
                ))

            # Parse conclusion
            conclusion_pattern = r'\[CONCLUSION\](.*?)\[/CONCLUSION\]'
            conclusion_match = re.search(conclusion_pattern, transition_text, re.DOTALL)
            if conclusion_match:
                conclusion_content = conclusion_match.group(1).strip()
                transition_sentences.append(TransitionSentence(
                    content=conclusion_content,
                    position=len(core_sentences)
                ))

            self._print_verbose(f"Generated {len(transition_sentences)} transition sentences")
            return transition_sentences

        except Exception as e:
            self._print_verbose(f"Transition generation failed: {str(e)}", "ERROR")
            return []

    def _organize_response(
        self,
        core_sentences: List[CoreSentence],
        transition_sentences: List[TransitionSentence]
    ) -> str:
        """Organize final response"""
        self._print_verbose("Organizing final response...")

        if not core_sentences:
            return "Unable to answer based on the given passages."

        response_parts = []

        # Add intro if available
        for transition in transition_sentences:
            if transition.position == 0:
                response_parts.append(transition.content)

        # Add core sentences with transitions
        for i, core_sentence in enumerate(core_sentences):
            response_parts.append(core_sentence.content)

            # Add transition after this sentence (if not last)
            for transition in transition_sentences:
                if transition.position == i + 1 and transition.position < len(core_sentences):
                    response_parts.append(transition.content)

        # Add conclusion if available
        for transition in transition_sentences:
            if transition.position == len(core_sentences):
                response_parts.append(transition.content)

        final_response = " ".join(response_parts)

        self._print_verbose(
            f"Response organized: {len(final_response)} chars, "
            f"{len(core_sentences)} core sentences, {len(transition_sentences)} transitions"
        )

        return final_response

    def process(self, context: str, query: str) -> Dict[str, Any]:
        """
        Process a RAG request with CP-Link pipeline.

        Args:
            context: Context text
            query: User query

        Returns:
            Dictionary with response and metrics
        """
        self.printer.print_separator(f"Start {self.pipeline_name} Pipeline")

        # Step 1: Extract core sentences
        core_sentences = self._extract_core_sentences(context, query)

        if not core_sentences:
            return create_standard_response_structure(
                output="Unable to answer based on the given passages."
            )

        # Step 2: Generate transitions
        transition_sentences = self._generate_transitions(core_sentences, query)

        # Step 3: Organize response
        final_response = self._organize_response(core_sentences, transition_sentences)

        # Step 4: Calculate extractiveness
        metrics = self._calculate_extractiveness(context, final_response)

        return create_standard_response_structure(
            output=final_response,
            extractiveness_coverage=metrics['coverage'],
            extractiveness_density=metrics['density'],
            extractiveness_score=metrics['score'],
            extra={
                "num_core_sentences": len(core_sentences),
                "num_transitions": len(transition_sentences),
                "core_sentences": [cs.content for cs in core_sentences],
                "transitions": [t.content for t in transition_sentences]
            }
        )
