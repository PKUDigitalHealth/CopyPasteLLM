"""
Configuration management for CopyPasteLLM

Loads configuration from environment variables and .env file
"""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class Config:
    """Configuration class for CopyPasteLLM"""

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration

        Args:
            env_file: Path to .env file. If None, searches for .env in default locations
        """
        # Load .env file if available
        if DOTENV_AVAILABLE:
            if env_file:
                load_dotenv(env_file)
            else:
                # Search for .env file
                env_paths = [
                    Path.cwd() / ".env",
                    Path(__file__).parent / ".env",
                    Path(__file__).parent.parent / ".env",
                ]
                for env_path in env_paths:
                    if env_path.exists():
                        load_dotenv(env_path)
                        break

    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables or .env file. "
                "Please set it in your .env file or environment."
            )
        return api_key

    @property
    def openai_base_url(self) -> Optional[str]:
        """Get OpenAI base URL"""
        return os.getenv("OPENAI_BASE_URL")

    @property
    def default_model(self) -> str:
        """Get default model name"""
        return os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

    @property
    def default_temperature(self) -> float:
        """Get default temperature"""
        return float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))

    @property
    def default_timeout(self) -> int:
        """Get default timeout in seconds"""
        return int(os.getenv("DEFAULT_TIMEOUT", "180"))

    @property
    def default_pipeline(self) -> str:
        """Get default pipeline"""
        pipeline = os.getenv("DEFAULT_PIPELINE", "cp-order")
        if pipeline not in ["cp-order", "cp-link", "cp-refine"]:
            raise ValueError(
                f"Invalid DEFAULT_PIPELINE: {pipeline}. "
                "Must be one of: cp-order, cp-link, cp-refine"
            )
        return pipeline

    @property
    def verbose(self) -> bool:
        """Get verbose logging setting"""
        return os.getenv("VERBOSE", "false").lower() == "true"

    @property
    def enable_thinking(self) -> bool:
        """Get enable_thinking setting for SiliconFlow reasoning models"""
        return os.getenv("ENABLE_THINKING", "false").lower() == "true"

    @property
    def thinking_budget(self) -> int:
        """Get thinking_budget for SiliconFlow reasoning models"""
        try:
            budget = int(os.getenv("THINKING_BUDGET", "4096"))
            # Clamp to valid range
            return max(128, min(32768, budget))
        except ValueError:
            return 4096

    # Alternative providers
    @property
    def siliconflow_api_key(self) -> Optional[str]:
        """Get SiliconFlow API key (if using SiliconFlow)"""
        return os.getenv("SILICONFLOW_API_KEY")

    @property
    def azure_openai_api_key(self) -> Optional[str]:
        """Get Azure OpenAI API key (if using Azure)"""
        return os.getenv("AZURE_OPENAI_API_KEY")

    @property
    def azure_openai_endpoint(self) -> Optional[str]:
        """Get Azure OpenAI endpoint (if using Azure)"""
        return os.getenv("AZURE_OPENAI_ENDPOINT")


# Global config instance
_config = None


def get_config(env_file: Optional[str] = None) -> Config:
    """
    Get global configuration instance

    Args:
        env_file: Optional path to .env file

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(env_file)
    return _config


def reset_config():
    """Reset global configuration (mainly for testing)"""
    global _config
    _config = None
