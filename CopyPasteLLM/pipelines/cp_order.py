"""
CP-Order Pipeline - Ultra-strict extraction using complete sentences from context

This pipeline extracts complete sentences from the context using LLM, validates
them rigorously, and intelligently orders them to create a coherent response.
"""

import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from CopyPasteLLM.pipelines.base import PipelineBase
from CopyPasteLLM.utils.text import TextProcessor, create_standard_response_structure


@dataclass
class ExtractedSentence:
    """Data structure for extracted sentences"""
    content: str  # Original sentence content (immutable)
    start_pos: int  # Position in context
    end_pos: int  # End position in context
    relevance_score: float  # Relevance score
    sentence_index: int  # Index in original text


class CPOrderPipeline(PipelineBase):
    """
    CP-Order Pipeline - Strict extractive RAG with intelligent sentence ordering.

    Features:
    - Extracts complete sentences from context using LLM
    - Validates each sentence to ensure it exists in context
    - Intelligently orders sentences for logical coherence
    - Zero hallucination - only uses text from context

    Args:
        verbose: Whether to print verbose output
        llm_backend: Optional pre-initialized LLMBackend
        **llm_kwargs: Additional arguments for LLMBackend
    """

    def __init__(self, verbose: bool = False, llm_backend=None, **llm_kwargs):
        super().__init__(
            pipeline_name="CP-Order",
            verbose=verbose,
            llm_backend=llm_backend,
            **llm_kwargs
        )

        self.text_processor = TextProcessor()
        self.similarity_threshold = 0.7

    def _get_extraction_prompt(self, context: str, query: str) -> str:
        """Get prompt for sentence extraction"""
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

Example:
EXTRACTED: One of Galileo's lesser-known achievements is his development of the Three Laws of Motion.
EXTRACTED: These laws articulate the principles of inertia, the relationship between force and motion, and the law of action and reaction.

Now extract the relevant sentences:"""

    def _extract_sentences_with_llm(
        self,
        context: str,
        query: str
    ) -> List[ExtractedSentence]:
        """Extract sentences using LLM"""
        extraction_prompt = self._get_extraction_prompt(context, query)

        try:
            response_content = self._call_llm(
                extraction_prompt,
                agent_name="Sentence Extractor"
            )

            self._print_verbose(f"LLM extraction response length: {len(response_content)} chars")

            extracted_sentences = []

            # Check for explicit no-sentences-found
            if "NO_RELEVANT_SENTENCES_FOUND" in response_content:
                self._print_verbose("LLM explicitly found no relevant sentences")
                return []

            # Parse extracted sentences
            lines = response_content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('EXTRACTED:'):
                    sentence_content = line.replace('EXTRACTED:', '').strip()
                    if sentence_content and len(sentence_content) > 10:
                        validated_sentence = self._validate_extracted_sentence(
                            sentence_content, context
                        )
                        if validated_sentence:
                            extracted_sentences.append(validated_sentence)

            if not extracted_sentences:
                self._print_verbose("Primary extraction failed, trying fallback...", "WARNING")
                # Fallback: extract quoted content
                import re
                quoted_sentences = re.findall(r'"([^"]+)"', response_content)
                for sentence in quoted_sentences:
                    if len(sentence) > 10:
                        validated_sentence = self._validate_extracted_sentence(sentence, context)
                        if validated_sentence:
                            extracted_sentences.append(validated_sentence)

            self._print_verbose(f"Successfully extracted and validated {len(extracted_sentences)} sentences")
            return extracted_sentences

        except Exception as e:
            self._print_verbose(f"LLM extraction failed: {str(e)}", "ERROR")
            return []

    def _validate_extracted_sentence(
        self,
        extracted_sentence: str,
        context: str
    ) -> Optional[ExtractedSentence]:
        """Validate that extracted sentence exists in context"""
        extracted_clean = self.text_processor.normalize_text(extracted_sentence)

        # Method 1: Direct string match
        if extracted_clean in self.text_processor.normalize_text(context):
            start_pos = context.find(extracted_sentence)
            if start_pos == -1:
                start_pos = self._find_similar_text_position(extracted_sentence, context)

            if start_pos != -1:
                end_pos = start_pos + len(extracted_sentence)
                return ExtractedSentence(
                    content=extracted_sentence,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    relevance_score=1.0,
                    sentence_index=0
                )

        # Method 2: Fuzzy match
        best_match = self._find_best_fuzzy_match(extracted_sentence, context)
        if best_match and best_match['similarity'] >= self.similarity_threshold:
            return ExtractedSentence(
                content=best_match['matched_text'],
                start_pos=best_match['start_pos'],
                end_pos=best_match['end_pos'],
                relevance_score=best_match['similarity'],
                sentence_index=0
            )

        # Method 3: Substring match
        substring_match = self._find_substring_match(extracted_sentence, context)
        if substring_match and substring_match['similarity'] >= self.similarity_threshold:
            return ExtractedSentence(
                content=substring_match['matched_text'],
                start_pos=substring_match['start_pos'],
                end_pos=substring_match['end_pos'],
                relevance_score=substring_match['similarity'],
                sentence_index=0
            )

        self._print_verbose(f"Sentence validation failed: {extracted_sentence[:50]}...")
        return None

    def _find_similar_text_position(self, target: str, context: str) -> int:
        """Find similar text position in context"""
        target_normalized = self.text_processor.normalize_text(target)
        context_normalized = self.text_processor.normalize_text(context)

        pos = context_normalized.find(target_normalized)
        if pos != -1:
            return self._map_position_to_original(pos, context_normalized, context)
        return -1

    def _map_position_to_original(
        self,
        normalized_pos: int,
        normalized_text: str,
        original_text: str
    ) -> int:
        """Map normalized position to original text position"""
        if normalized_pos == 0:
            return 0

        prefix = normalized_text[:normalized_pos]
        for i in range(len(original_text)):
            if self.text_processor.normalize_text(original_text[:i]).endswith(prefix):
                return i
        return -1

    def _find_best_fuzzy_match(self, target: str, context: str) -> Optional[Dict]:
        """Find best fuzzy match using sliding window"""
        try:
            from difflib import SequenceMatcher
        except ImportError:
            return None

        target_len = len(target)
        best_match = None
        best_similarity = 0

        # Sliding window search
        for i in range(len(context) - target_len + 1):
            window = context[i:i + target_len]
            similarity = SequenceMatcher(None, target, window).ratio()

            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = {
                    'matched_text': window,
                    'start_pos': i,
                    'end_pos': i + target_len,
                    'similarity': similarity
                }

        # Try different window sizes
        for window_size in [target_len - 10, target_len + 10, target_len - 20, target_len + 20]:
            if window_size <= 0 or window_size > len(context):
                continue

            for i in range(len(context) - window_size + 1):
                window = context[i:i + window_size]
                similarity = SequenceMatcher(None, target, window).ratio()

                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = {
                        'matched_text': window,
                        'start_pos': i,
                        'end_pos': i + window_size,
                        'similarity': similarity
                    }

        return best_match

    def _find_substring_match(self, target: str, context: str) -> Optional[Dict]:
        """Check if target is a substring of context"""
        target_normalized = self.text_processor.normalize_text(target)
        words = target_normalized.split()

        if len(words) < 3:
            return None

        # Find key phrase
        for start_word in range(len(words) - 2):
            for end_word in range(start_word + 3, len(words) + 1):
                key_phrase = ' '.join(words[start_word:end_word])

                if key_phrase in self.text_processor.normalize_text(context):
                    context_pos = context.find(key_phrase)
                    if context_pos != -1:
                        # Extend to sentence boundaries
                        start_pos = self._find_sentence_start(context, context_pos)
                        end_pos = self._find_sentence_end(context, context_pos + len(key_phrase))

                        matched_text = context[start_pos:end_pos].strip()
                        from difflib import SequenceMatcher
                        similarity = SequenceMatcher(None, target, matched_text).ratio()

                        if similarity >= self.similarity_threshold:
                            return {
                                'matched_text': matched_text,
                                'start_pos': start_pos,
                                'end_pos': end_pos,
                                'similarity': similarity
                            }
        return None

    def _find_sentence_start(self, text: str, pos: int) -> int:
        """Find sentence start position"""
        sentence_starters = ['.', '!', '?', '。', '！', '？', '\n', '\r']

        for i in range(pos, -1, -1):
            if i == 0 or text[i-1] in sentence_starters:
                while i < len(text) and text[i].isspace():
                    i += 1
                return i
        return 0

    def _find_sentence_end(self, text: str, pos: int) -> int:
        """Find sentence end position"""
        sentence_enders = ['.', '!', '?', '。', '！', '？']

        for i in range(pos, len(text)):
            if text[i] in sentence_enders:
                return i + 1
        return len(text)

    def _llm_sort_sentences(
        self,
        sentences: List[ExtractedSentence],
        query: str
    ) -> List[ExtractedSentence]:
        """Use LLM to intelligently sort sentences"""
        if not sentences or len(sentences) <= 1:
            return sentences

        try:
            # Assign IDs to sentences
            sentence_info = []
            for i, sent in enumerate(sentences):
                sentence_info.append({
                    'id': f"SENT_{i+1}",
                    'content': sent.content,
                    'original_index': sent.sentence_index,
                    'relevance_score': sent.relevance_score
                })

            # Build LLM prompt
            sentences_text = "\n".join([
                f"{info['id']}: {info['content']}"
                for info in sentence_info
            ])

            prompt = f"""You are an expert at organizing extracted sentences to create coherent and logical responses.

Given the user query and a list of extracted sentences, determine the optimal order for these sentences to create the most logical, coherent, and helpful response.

User Query: "{query}"

Extracted Sentences:
{sentences_text}

Analyze the logical relationships and determine the best order considering:
1. Query relevance and directness
2. Logical flow and coherence
3. Information hierarchy (general to specific, or cause to effect)
4. Natural reading progression

Provide your reasoning and then output the optimal order.

Format your response as:
REASONING: [Your detailed reasoning]
ORDER: [comma-separated list of sentence IDs, e.g., SENT_2,SENT_1,SENT_3]

Important:
- Only use the sentence IDs provided above
- Include ALL sentences in your ordering
- Consider the query context"""

            self._print_verbose("Requesting LLM for sentence sorting...")

            llm_response = self._call_llm(prompt, agent_name="Sentence Sorter")
            self._print_verbose(f"LLM sorting response: {llm_response[:300]}...")

            # Parse response
            order_line = None
            lines = llm_response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('ORDER:'):
                    order_line = line.replace('ORDER:', '').strip()

            if not order_line:
                import re
                order_match = re.search(r'SENT_\d+(?:,\s*SENT_\d+)*', llm_response)
                if order_match:
                    order_line = order_match.group()

            if order_line:
                # Parse sentence order
                ordered_ids = [sid.strip() for sid in order_line.split(',')]

                # Validate all IDs are present
                expected_ids = {info['id'] for info in sentence_info}
                provided_ids = set(ordered_ids)

                if expected_ids == provided_ids:
                    # Reorder sentences according to LLM
                    id_to_sentence = {
                        f"SENT_{i+1}": sent
                        for i, sent in enumerate(sentences)
                    }

                    sorted_sentences = []
                    for sid in ordered_ids:
                        if sid in id_to_sentence:
                            sorted_sentences.append(id_to_sentence[sid])

                    self._print_verbose(f"LLM sorting successful: {' -> '.join(ordered_ids)}")
                    return sorted_sentences
                else:
                    self._print_verbose(f"LLM returned incomplete IDs, missing: {expected_ids - provided_ids}")
            else:
                self._print_verbose("Could not parse sentence order from LLM response")

        except Exception as e:
            self._print_verbose(f"LLM sorting failed: {str(e)}", "ERROR")

        # Fallback: sort by relevance
        self._print_verbose("LLM sorting failed, sorting by relevance score")
        return sorted(sentences, key=lambda x: x.relevance_score, reverse=True)

    def _build_response(self, extracted_sentences: List[ExtractedSentence]) -> str:
        """Build final response from extracted sentences"""
        self._print_verbose("Building final response...")

        if not extracted_sentences:
            return "Unable to answer based on the given passages."

        # Concatenate sentences with spaces
        response_parts = [sentence.content.strip() for sentence in extracted_sentences]
        final_response = " ".join(response_parts)

        self._print_verbose(
            f"Response built: {len(final_response)} chars, {len(extracted_sentences)} sentences"
        )

        return final_response

    def process(self, context: str, query: str) -> Dict[str, Any]:
        """
        Process a RAG request with CP-Order pipeline.

        Args:
            context: Context text
            query: User query

        Returns:
            Dictionary with response and metrics
        """
        self.printer.print_separator(f"Start {self.pipeline_name} Pipeline")

        # Step 1: Extract sentences with LLM
        extracted_sentences = self._extract_sentences_with_llm(context, query)

        if not extracted_sentences:
            return create_standard_response_structure(
                output="Unable to answer based on the given passages."
            )

        # Step 2: Build unsorted response
        unsorted_response = self._build_response(extracted_sentences)

        # Step 3: Extract raw sentences array
        raw_sentences = [sentence.content for sentence in extracted_sentences]

        # Step 4: Shuffle for LLM sorting
        random.shuffle(extracted_sentences)

        # Step 5: Sort with LLM
        sorted_sentences = self._llm_sort_sentences(extracted_sentences, query)

        # Step 6: Extract sorted sentences array
        sorted_sentences_array = [sentence.content for sentence in sorted_sentences]

        # Step 7: Build final response
        final_response = self._build_response(sorted_sentences)

        # Step 8: Calculate extractiveness
        metrics = self._calculate_extractiveness(context, final_response)

        return create_standard_response_structure(
            output=final_response,
            extractiveness_coverage=metrics['coverage'],
            extractiveness_density=metrics['density'],
            extractiveness_score=metrics['score'],
            extra={
                "unsorted_response": unsorted_response,
                "raw_sentences": raw_sentences,
                "sorted_sentences": sorted_sentences_array,
                "num_extracted_sentences": len(extracted_sentences)
            }
        )
