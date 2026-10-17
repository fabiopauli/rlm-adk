"""
Recursive Language Model (RLM) Package

A framework for processing extremely long contexts using recursive decomposition
and sub-calls to language models.
"""

from .core import RecursiveLanguageModel
from .metrics import RLMMetrics
from .cache import RLMCache
from .files import (
    FileLoader,
    FileSource,
    FileParser,
    LocalFileSource,
    GCSFileSource,
    LazyFile,
    LazyFileCollection,
)
from .providers import (
    LLMProvider,
    OpenAIProvider,
    XAIProvider,
    AnthropicProvider,
    RunInfraProvider,
    create_provider,
    LLMResponse,
    TokenUsageDetails
)

__version__ = "0.3.0"
__all__ = [
    "RecursiveLanguageModel",
    "RLMMetrics",
    "RLMCache",
    "FileLoader",
    "FileSource",
    "FileParser",
    "LocalFileSource",
    "GCSFileSource",
    "LazyFile",
    "LazyFileCollection",
    "LLMProvider",
    "OpenAIProvider",
    "XAIProvider",
    "AnthropicProvider",
    "RunInfraProvider",
    "create_provider",
    "LLMResponse",
    "TokenUsageDetails"
]
