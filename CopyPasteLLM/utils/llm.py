"""
OpenAI LLM backend for CopyPasteLLM pipelines
"""

import os
from typing import Optional, Dict, Any
from openai import OpenAI


class LLMBackend:
    """
    Unified LLM backend supporting OpenAI API.

    This class provides a simple interface for calling LLMs using the OpenAI API
    or compatible endpoints (like Azure OpenAI, SiliconFlow, etc.).

    Args:
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY environment variable.
        base_url: Custom base URL for API requests. If None, uses default OpenAI endpoint.
        model: Model name to use (default: "gpt-4o-mini").
        temperature: Sampling temperature (default: 0.1).
        timeout: Request timeout in seconds (default: 180).
        verbose: Whether to print verbose output (default: False).
        enable_thinking: Enable thinking mode for supported models (default: False).
                        Supported models include GLM-5, GLM-4.7, DeepSeek-V3.2, etc.
        thinking_budget: Max tokens for chain-of-thought output (default: 4096).
                         Only used when enable_thinking=True. Range: 128-32768.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        timeout: int = 180,
        verbose: bool = False,
        enable_thinking: bool = False,
        thinking_budget: int = 4096
    ):
        # Auto-detect API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key not found. Please provide api_key parameter or "
                    "set OPENAI_API_KEY environment variable."
                )

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.verbose = verbose
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget

        # Initialize OpenAI client directly
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

        if self.verbose:
            print(f"Initialized LLM backend:")
            print(f"  Model: {model}")
            print(f"  Base URL: {base_url or 'default (OpenAI)'}")
            print(f"  Temperature: {temperature}")
            if enable_thinking:
                print(f"  Thinking mode: enabled (budget={thinking_budget})")
            else:
                print(f"  Thinking mode: disabled")

    def call(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        agent_name: str = "",
        max_retries: int = 3
    ) -> str:
        """
        Call LLM with a prompt.

        Args:
            prompt: The prompt to send to the LLM.
            temperature: Override default temperature for this call.
            agent_name: Optional name for logging (e.g., "Extractor", "Reviewer").
            max_retries: Maximum number of retries on failure.

        Returns:
            The LLM response text.
        """
        if temperature is None:
            temperature = self.temperature

        if self.verbose and agent_name:
            print(f"\n{'='*60}")
            print(f"LLM Call: {agent_name}")
            print(f"{'='*60}")
            print(f"Model: {self.model}")
            print(f"Temperature: {temperature}")
            print(f"Prompt length: {len(prompt)} characters")

        # Prepare extra body for thinking mode (only if enabled)
        # Some APIs reject enable_thinking parameter even when set to False
        extra_body = {}
        if self.enable_thinking:
            extra_body["enable_thinking"] = True
            extra_body["thinking_budget"] = self.thinking_budget

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                extra_body=extra_body if extra_body else None,
            )
            response_text = response.choices[0].message.content

            if self.verbose:
                print(f"Response length: {len(response_text)} characters")
                if agent_name:
                    print(f"✓ {agent_name} completed")

            return response_text

        except Exception as e:
            if self.verbose:
                print(f"✗ LLM call failed: {str(e)}")
            raise

    def call_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        agent_name: str = ""
    ) -> str:
        """
        Call LLM with both system and user prompts.

        Args:
            system_prompt: System message for setting behavior.
            user_prompt: User message/question.
            temperature: Override default temperature.
            agent_name: Optional name for logging.

        Returns:
            The LLM response text.
        """
        if temperature is None:
            temperature = self.temperature

        if self.verbose and agent_name:
            print(f"\n{'='*60}")
            print(f"LLM Call: {agent_name}")
            print(f"{'='*60}")

        # Prepare extra body for thinking mode (only if enabled)
        # Some APIs reject enable_thinking parameter even when set to False
        extra_body = {}
        if self.enable_thinking:
            extra_body["enable_thinking"] = True
            extra_body["thinking_budget"] = self.thinking_budget

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                extra_body=extra_body if extra_body else None,
            )
            response_text = response.choices[0].message.content

            if self.verbose:
                print(f"Response length: {len(response_text)} characters")

            return response_text

        except Exception as e:
            if self.verbose:
                print(f"✗ LLM call failed: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current LLM configuration"""
        return {
            "model": self.model,
            "base_url": self.base_url or "default (OpenAI)",
            "temperature": self.temperature,
            "timeout": self.timeout,
            "api_key_prefix": self.api_key[:7] + "..." if self.api_key else None
        }


def create_llm_backend(
    provider: str = "openai",
    **kwargs
) -> LLMBackend:
    """
    Factory function to create LLM backend.

    Args:
        provider: Provider name (currently only "openai" supported).
        **kwargs: Additional arguments passed to LLMBackend constructor.

    Returns:
        Initialized LLMBackend instance.
    """
    if provider.lower() != "openai":
        raise ValueError(
            f"Provider '{provider}' not supported. "
            "Currently only 'openai' is supported."
        )

    return LLMBackend(**kwargs)
