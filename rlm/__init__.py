"""
Recursive Language Model (RLM) Package

A framework for processing extremely long contexts using recursive decomposition
and sub-calls to language models.
"""

from .core import RecursiveLanguageModel
from .metrics import RLMMetrics
from .cache import RLMCache
from .providers import (
    LLMProvider,
    OpenAIProvider,
    XAIProvider,
    AnthropicProvider,
    create_provider,
    LLMResponse,
    TokenUsageDetails
)

__version__ = "0.3.0"
__all__ = [
    "RecursiveLanguageModel",
    "RLMMetrics",
    "RLMCache",
    "LLMProvider",
    "OpenAIProvider",
    "XAIProvider",
    "AnthropicProvider",
    "create_provider",
    "LLMResponse",
    "TokenUsageDetails"
]
