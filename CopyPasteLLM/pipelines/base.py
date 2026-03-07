"""
Base pipeline class for CopyPasteLLM RAG pipelines
"""

from abc import abstractmethod
from typing import Dict, Any, Optional
from CopyPasteLLM.utils.llm import LLMBackend
from CopyPasteLLM.metrics.extractiveness import ExtractivenessMetrics


class VerbosePrinter:
    """Simple verbose printing utility"""

    def __init__(self, verbose: bool = False, component_name: str = ""):
        self.verbose = verbose
        self.component_name = component_name

    def print_verbose(self, message: str, stage: str = "", level: str = "INFO"):
        """Print verbose message if verbose mode is enabled"""
        if not self.verbose:
            return

        if stage:
            prefix = f"[{self.component_name}] [{stage}] {level}: {message}"
        else:
            prefix = f"[{self.component_name}] {level}: {message}"

        print(prefix)

    def print_stage(self, message: str, stage: str = ""):
        """Print stage message"""
        self.print_verbose(message, stage, "INFO")

    def print_separator(self, title: str = ""):
        """Print separator line"""
        if self.verbose:
            if title:
                print(f"\n{'='*20} {title} {'='*20}\n")
            else:
                print(f"{'='*60}")


class PipelineBase:
    """
    Base class for CopyPasteLLM pipelines.

    Provides common functionality for all RAG pipelines including LLM calling,
    extractiveness tracking, and standardized response formatting.

    Args:
        pipeline_name: Name of the pipeline for logging
        verbose: Whether to print verbose output
        llm_backend: Optional pre-initialized LLMBackend. If None, creates new one.
        **llm_kwargs: Additional arguments passed to LLMBackend if not provided
    """

    def __init__(
        self,
        pipeline_name: str,
        verbose: bool = False,
        llm_backend: Optional[LLMBackend] = None,
        **llm_kwargs
    ):
        self.pipeline_name = pipeline_name
        self.verbose = verbose
        self.printer = VerbosePrinter(verbose=verbose, component_name=pipeline_name)

        # Initialize LLM backend
        if llm_backend is not None:
            self.llm_backend = llm_backend
        else:
            self.llm_backend = LLMBackend(**llm_kwargs)

        # Initialize extractiveness metrics
        self.extractiveness_metrics = ExtractivenessMetrics(verbose=False)

    def _print_verbose(self, message: str, stage: str = "", level: str = "INFO"):
        """Print verbose message"""
        self.printer.print_verbose(message, stage, level)

    def _call_llm(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        agent_name: str = "",
        max_retries: int = 3
    ) -> str:
        """
        Call LLM with a prompt.

        Args:
            prompt: The prompt to send to the LLM
            temperature: Override default temperature
            agent_name: Optional name for logging
            max_retries: Maximum number of retries

        Returns:
            The LLM response text
        """
        return self.llm_backend.call(
            prompt=prompt,
            temperature=temperature,
            agent_name=agent_name or self.pipeline_name,
            max_retries=max_retries
        )

    def _calculate_extractiveness(
        self,
        context: str,
        response: str
    ) -> Dict[str, float]:
        """
        Calculate extractiveness metrics for a response.

        Args:
            context: Original context text
            response: Generated response

        Returns:
            Dictionary with coverage, density, and score
        """
        coverage, density, score = self.extractiveness_metrics.calculate_extractiveness_score(
            context, response
        )
        return {
            "coverage": coverage,
            "density": density,
            "score": score
        }

    @abstractmethod
    def process(self, context: str, query: str) -> Dict[str, Any]:
        """
        Process a RAG request.

        Args:
            context: Context text
            query: User query/question

        Returns:
            Dictionary with response and metrics
        """
        raise NotImplementedError("Subclasses must implement process method")
