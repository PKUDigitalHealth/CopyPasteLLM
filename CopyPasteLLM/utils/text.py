"""
Text processing utilities for CopyPasteLLM pipelines
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple


class TextProcessor:
    """Text processing utility class"""

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for comparison.

        Args:
            text: Input text

        Returns:
            Normalized text with unified whitespace and trimmed quotes
        """
        # Unify whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove quotes
        text = text.strip('"\'""''')
        return text

    @staticmethod
    def find_sentence_boundaries(text: str, pos: int) -> Tuple[int, int]:
        """
        Find sentence boundaries around a position.

        Args:
            text: Text content
            pos: Current position in text

        Returns:
            Tuple of (start_pos, end_pos) for sentence boundaries
        """
        sentence_starters = ['.', '!', '?', '。', '！', '？', '\n', '\r']
        sentence_enders = ['.', '!', '?', '。', '！', '？']

        # Find sentence start position
        start_pos = 0
        for i in range(pos, -1, -1):
            if i == 0 or text[i-1] in sentence_starters:
                # Skip whitespace
                while i < len(text) and text[i].isspace():
                    i += 1
                start_pos = i
                break

        # Find sentence end position
        end_pos = len(text)
        for i in range(pos, len(text)):
            if text[i] in sentence_enders:
                end_pos = i + 1
                break

        return start_pos, end_pos

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, text1, text2).ratio()
        except ImportError:
            # Simple word overlap similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 and not words2:
                return 1.0
            if not words1 or not words2:
                return 0.0

            intersection = words1.intersection(words2)
            union = words1.union(words2)

            return len(intersection) / len(union)


class JSONParser:
    """Robust JSON parser for LLM outputs with multiple fallback strategies"""

    @staticmethod
    def parse_llm_output(llm_output: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM output with multiple fallback strategies.

        Args:
            llm_output: Raw LLM output string

        Returns:
            Parsed dictionary or None if all strategies fail
        """
        parsing_strategies = [
            JSONParser._parse_direct,
            JSONParser._parse_clean_markdown,
            JSONParser._parse_extract_json,
            JSONParser._parse_fix_common_errors,
            JSONParser._parse_regex_extraction
        ]

        for strategy in parsing_strategies:
            try:
                result = strategy(llm_output)
                if result is not None:
                    return result
            except Exception:
                continue

        return None

    @staticmethod
    def _parse_direct(llm_output: str) -> Optional[Dict[str, Any]]:
        """Strategy 1: Direct JSON parsing"""
        return json.loads(llm_output)

    @staticmethod
    def _parse_clean_markdown(llm_output: str) -> Optional[Dict[str, Any]]:
        """Strategy 2: Clean markdown code blocks and parse"""
        cleaned = re.sub(r'```json\s*|\s*```', '', llm_output)
        cleaned = cleaned.strip()
        return json.loads(cleaned)

    @staticmethod
    def _parse_extract_json(llm_output: str) -> Optional[Dict[str, Any]]:
        """Strategy 3: Extract JSON object using regex"""
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        return None

    @staticmethod
    def _parse_fix_common_errors(llm_output: str) -> Optional[Dict[str, Any]]:
        """Strategy 4: Fix common JSON formatting errors"""
        fixed = llm_output

        # Fix single quotes
        fixed = re.sub(r"'([^']*)':", r'"\1":', fixed)
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)

        # Fix unquoted keys
        fixed = re.sub(r'(\w+):', r'"\1":', fixed)

        # Remove trailing commas
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)

        return json.loads(fixed)

    @staticmethod
    def _parse_regex_extraction(llm_output: str) -> Optional[Dict[str, Any]]:
        """Strategy 5: Extract key fields using regex"""
        result = {}

        # Extract winner field
        winner_match = re.search(r'"?winner"?\s*:\s*"?([^",\}]+)"?', llm_output, re.IGNORECASE)
        if winner_match:
            result['winner'] = winner_match.group(1).strip()

        # Extract confidence field
        conf_match = re.search(r'"?confidence"?\s*:\s*([0-9.]+)', llm_output, re.IGNORECASE)
        if conf_match:
            result['confidence'] = float(conf_match.group(1))

        # Extract reason field
        reason_match = re.search(r'"?reason"?\s*:\s*"([^"]+)"', llm_output, re.IGNORECASE)
        if reason_match:
            result['reason'] = reason_match.group(1)

        return result if result else None


def create_standard_response_structure(
    output: str,
    extractiveness_coverage: float = 0.0,
    extractiveness_density: float = 0.0,
    extractiveness_score: float = 0.0,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create standardized response structure for all pipelines.

    Args:
        output: Generated response text
        extractiveness_coverage: Coverage metric (0-1)
        extractiveness_density: Density metric (0-1)
        extractiveness_score: Overall extractiveness score (0-1)
        extra: Optional additional pipeline-specific data

    Returns:
        Standardized response dictionary
    """
    response = {
        "output": output,
        "extractiveness_coverage": extractiveness_coverage,
        "extractiveness_density": extractiveness_density,
        "extractiveness_score": extractiveness_score,
    }

    if extra:
        response.update(extra)

    return response
