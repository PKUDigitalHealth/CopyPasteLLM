"""
OpenAI-style client for CopyPasteLLM pipelines

This module provides a familiar OpenAI-style interface for using CopyPasteLLM
pipelines to generate extractive RAG responses with hallucination mitigation.
"""

import os
import time
from typing import Literal, List, Optional, Dict, Any
from dataclasses import dataclass, field

from CopyPasteLLM.utils.llm import LLMBackend, create_llm_backend
from CopyPasteLLM.pipelines import CPOrderPipeline, CPLinkPipeline, CPRefinePipeline
from CopyPasteLLM.metrics.extractiveness import ExtractivenessMetrics
from CopyPasteLLM.config import get_config


@dataclass
class CopyPasteResponse:
    """
    Standardized response format for CopyPasteLLM pipelines.

    Attributes:
        content: Generated response text
        extractiveness_score: Overall extractiveness (0-1)
        extractiveness_coverage: Coverage metric (0-1)
        extractiveness_density: Density metric (0-1)
        pipeline: Which pipeline was used
        processing_time: Time taken in seconds
        extra: Additional pipeline-specific data
    """
    content: str
    extractiveness_score: float
    extractiveness_coverage: float
    extractiveness_density: float
    pipeline: str
    processing_time: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def render_heatmap(self, context: str, show_legend: bool = True) -> str:
        """
        Render extractiveness heatmap for terminal visualization.

        Args:
            context: Original context text
            show_legend: Whether to show color legend

        Returns:
            String with ANSI color codes
        """
        metrics = ExtractivenessMetrics()
        return metrics.render_extractiveness_heatmap(context, self.content, show_legend)

    def get_fragments(
        self,
        context: str,
        min_length: int = 2,
        sort_by_length: bool = True
    ) -> List[str]:
        """
        Get extracted fragments from response.

        Args:
            context: Original context text
            min_length: Minimum fragment length
            sort_by_length: Whether to sort by length

        Returns:
            List of extracted fragment strings
        """
        metrics = ExtractivenessMetrics()
        fragments, _ = metrics.get_extractive_fragments(
            context,
            self.content,
            descending=sort_by_length,
            include_min_len=min_length
        )
        return fragments


class CopyPasteClient:
    """
    OpenAI-style client for CopyPasteLLM pipelines.

    This client provides a familiar interface similar to the OpenAI Python SDK,
    making it easy to adopt for users already familiar with OpenAI's API.

    Example:
        ```python
        from CopyPasteLLM import CopyPasteClient

        client = CopyPasteClient(
            api_key="sk-...",
            model="gpt-4o-mini",
            default_pipeline="cp-order"
        )

        response = client.responses.create(
            context="Your context document...",
            query="Your question here?",
            pipeline="cp-order"
        )

        print(response.content)
        print(f"Extractiveness: {response.extractiveness_score:.3f}")
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = None,
        default_pipeline: Literal["cp-order", "cp-link", "cp-refine"] = None,
        temperature: float = None,
        timeout: int = None,
        verbose: bool = None,
        enable_thinking: bool = False,
        thinking_budget: int = 4096,
        config_file: Optional[str] = None
    ):
        """
        Initialize CopyPasteLLM client.

        Args:
            api_key: OpenAI API key. If None, reads from .env or OPENAI_API_KEY env variable.
            base_url: Custom base URL for API requests.
            model: Model name to use. If None, uses DEFAULT_MODEL from .env.
            default_pipeline: Default pipeline to use. If None, uses DEFAULT_PIPELINE from .env.
            temperature: Sampling temperature. If None, uses DEFAULT_TEMPERATURE from .env.
            timeout: Request timeout in seconds. If None, uses DEFAULT_TIMEOUT from .env.
            verbose: Whether to print verbose output. If None, uses VERBOSE from .env.
            enable_thinking: Enable thinking mode for supported models (default: False).
            thinking_budget: Max tokens for thinking output (default: 4096, range: 128-32768).
            config_file: Optional path to .env configuration file.
        """
        # Load configuration from .env file
        config = get_config(config_file)

        # Use config values as defaults if not explicitly provided
        if api_key is None:
            api_key = config.openai_api_key
        if base_url is None:
            base_url = config.openai_base_url
        if model is None:
            model = config.default_model
        if default_pipeline is None:
            default_pipeline = config.default_pipeline
        if temperature is None:
            temperature = config.default_temperature
        if timeout is None:
            timeout = config.default_timeout
        if verbose is None:
            verbose = config.verbose
        # Note: enable_thinking and thinking_budget default to False/4096 unless explicitly set
        # We don't read from .env for these to keep thinking mode opt-in only

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.default_pipeline = default_pipeline
        self.temperature = temperature
        self.timeout = timeout
        self.verbose = verbose
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self._config_file = config_file

        # Initialize shared LLM backend
        self.llm_backend = LLMBackend(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout=timeout,
            verbose=verbose,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget
        )

        # Initialize extractiveness metrics
        self.extractiveness_metrics = ExtractivenessMetrics()

        if verbose:
            print(f"CopyPasteLLM client initialized:")
            print(f"  Model: {model}")
            print(f"  Default pipeline: {default_pipeline}")
            print(f"  Temperature: {temperature}")

    class responses:
        """Namespace for response generation (mimicking openai.responses)"""

        @staticmethod
        def create(
            context: str,
            query: str,
            pipeline: Literal["cp-order", "cp-link", "cp-refine"] = None,
            model: str = None,
            extractiveness: bool = True,
            show_heatmap: bool = False,
            **pipeline_kwargs
        ) -> CopyPasteResponse:
            """
            Generate an extractive response.

            This is a static method that should be called from a client instance.
            Use `client.responses.create()` instead.

            Args:
                context: Context document text
                query: User question/query
                pipeline: Pipeline to use ("cp-order", "cp-link", "cp-refine")
                model: Override default model
                extractiveness: Whether to calculate extractiveness metrics
                show_heatmap: Whether to show extractiveness heatmap
                **pipeline_kwargs: Additional arguments for pipeline

            Returns:
                CopyPasteResponse object
            """
            raise NotImplementedError(
                "Use client.responses.create() from a CopyPasteClient instance"
            )

    def responses_create(
        self,
        context: str,
        query: str,
        pipeline: Literal["cp-order", "cp-link", "cp-refine"] = None,
        model: str = None,
        extractiveness: bool = True,
        show_heatmap: bool = False,
        **pipeline_kwargs
    ) -> CopyPasteResponse:
        """
        Generate an extractive response.

        Args:
            context: Context document text
            query: User question/query
            pipeline: Pipeline to use ("cp-order", "cp-link", "cp-refine")
            model: Override default model
            extractiveness: Whether to calculate extractiveness metrics
            show_heatmap: Whether to show extractiveness heatmap
            **pipeline_kwargs: Additional arguments for pipeline

        Returns:
            CopyPasteResponse object with generated response and metrics

        Example:
            ```python
            response = client.responses_create(
                context="Galileo was responsible for...",
                query="Which law did Galileo describe?",
                pipeline="cp-order"
            )
            print(response.content)
            ```
        """
        # Use default pipeline if not specified
        if pipeline is None:
            pipeline = self.default_pipeline

        # Override model if specified
        if model is not None:
            llm_backend = LLMBackend(
                api_key=self.api_key,
                base_url=self.base_url,
                model=model,
                temperature=self.temperature,
                timeout=self.timeout,
                verbose=self.verbose
            )
        else:
            llm_backend = self.llm_backend

        # Start timer
        start_time = time.time()

        # Create pipeline instance
        pipeline_instance = self._create_pipeline(pipeline, llm_backend)

        # Process request
        try:
            result = pipeline_instance.process(context, query)

            # Extract results
            content = result.get("output", "")
            coverage = result.get("extractiveness_coverage", 0.0)
            density = result.get("extractiveness_density", 0.0)
            score = result.get("extractiveness_score", 0.0)

            # Extract extra data
            extra = {k: v for k, v in result.items() if k not in
                    ["output", "extractiveness_coverage", "extractiveness_density", "extractiveness_score"]}

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create response object
            response = CopyPasteResponse(
                content=content,
                extractiveness_score=score,
                extractiveness_coverage=coverage,
                extractiveness_density=density,
                pipeline=pipeline,
                processing_time=processing_time,
                extra=extra
            )

            # Show heatmap if requested
            if show_heatmap:
                print("\n" + "="*60)
                print("Extractiveness Heatmap")
                print("="*60)
                print(response.render_heatmap(context, show_legend=True))

            return response

        except Exception as e:
            if self.verbose:
                print(f"Response generation failed: {str(e)}")
            raise

    def _create_pipeline(self, pipeline_name: str, llm_backend: LLMBackend):
        """Create pipeline instance"""
        pipeline_map = {
            "cp-order": CPOrderPipeline,
            "cp-link": CPLinkPipeline,
            "cp-refine": CPRefinePipeline,
        }

        if pipeline_name not in pipeline_map:
            raise ValueError(
                f"Unknown pipeline: {pipeline_name}. "
                f"Choose from: {list(pipeline_map.keys())}"
            )

        pipeline_class = pipeline_map[pipeline_name]

        # CP-Refine has special parameters
        if pipeline_name == "cp-refine":
            return pipeline_class(
                verbose=self.verbose,
                llm_backend=llm_backend,
                max_iterations=5,
                min_extractiveness=0.99
            )
        else:
            return pipeline_class(
                verbose=self.verbose,
                llm_backend=llm_backend
            )

    # Add responses namespace with bound method
    @property
    def responses(self) -> Any:
        """Responses namespace for OpenAI-style API"""
        class _ResponsesNamespace:
            def __init__(self, client):
                self.client = client

            def create(
                self,
                context: str,
                query: str,
                pipeline: Literal["cp-order", "cp-link", "cp-refine"] = None,
                model: str = None,
                extractiveness: bool = True,
                show_heatmap: bool = False,
                **pipeline_kwargs
            ) -> CopyPasteResponse:
                return self.client.responses_create(
                    context=context,
                    query=query,
                    pipeline=pipeline,
                    model=model,
                    extractiveness=extractiveness,
                    show_heatmap=show_heatmap,
                    **pipeline_kwargs
                )

        return _ResponsesNamespace(self)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current configuration"""
        return {
            "model": self.model,
            "base_url": self.base_url or "default (OpenAI)",
            "default_pipeline": self.default_pipeline,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "api_key_prefix": self.api_key[:7] + "..." if self.api_key else None
        }
