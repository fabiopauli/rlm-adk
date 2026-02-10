"""
LLM Provider abstraction for multi-provider support.

Supports OpenAI, xAI Grok, Anthropic Claude, and other providers with unified interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
import time
import random

logger = logging.getLogger(__name__)


@dataclass
class TokenUsageDetails:
    """Detailed token usage information."""

    # Main counts
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Prompt details
    prompt_text_tokens: int = 0
    prompt_audio_tokens: int = 0
    prompt_image_tokens: int = 0
    prompt_cached_tokens: int = 0

    # Completion details
    completion_reasoning_tokens: int = 0
    completion_audio_tokens: int = 0
    completion_accepted_prediction_tokens: int = 0
    completion_rejected_prediction_tokens: int = 0

    # Additional metadata
    num_sources_used: int = 0


@dataclass
class LLMResponse:
    """Unified LLM response format."""

    content: str
    usage: TokenUsageDetails
    model: str
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Provides unified interface for different LLM APIs.
    """

    def __init__(
        self,
        api_key: str,
        default_model: str,
        context_window: int = 128_000,
        timeout: int = 600,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0
    ):
        """
        Initialize provider.

        Args:
            api_key: API key for the provider
            default_model: Default model to use
            context_window: Maximum context window size (default 128k)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on transient failures (0 to disable)
            retry_base_delay: Base delay in seconds for exponential backoff
            retry_max_delay: Maximum delay in seconds between retries
        """
        self.api_key = api_key
        self.default_model = default_model
        self.context_window = context_window
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is transient and should be retried.

        Checks for rate limits (429), timeouts, connection errors, and
        server errors (500/502/503/504). Subclasses can override for
        provider-specific logic.

        Args:
            error: The exception to check

        Returns:
            True if the error is retryable
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Rate limit indicators
        if any(term in error_type for term in ['ratelimit', 'rate_limit']):
            return True
        if '429' in error_str or 'rate limit' in error_str or 'rate_limit' in error_str:
            return True

        # Timeout indicators
        if any(term in error_type for term in ['timeout', 'timed_out']):
            return True
        if 'timeout' in error_str or 'timed out' in error_str:
            return True

        # Connection errors
        if any(term in error_type for term in ['connection', 'connect']):
            return True
        if 'connection' in error_str:
            return True

        # Server errors (5xx)
        if any(term in error_type for term in ['internalserver', 'internal_server', 'server_error']):
            return True
        for code in ['500', '502', '503', '504']:
            if code in error_str:
                return True

        # Overloaded
        if 'overloaded' in error_str or '529' in error_str:
            return True

        return False

    def _call_with_retry(self, fn, *args, **kwargs):
        """
        Call a function with exponential backoff retry on transient errors.

        Args:
            fn: Function to call
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn

        Returns:
            Result of fn(*args, **kwargs)

        Raises:
            The last exception if all retries are exhausted
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e

                if attempt >= self.max_retries or not self._is_retryable_error(e):
                    raise

                # Exponential backoff with jitter
                delay = min(
                    self.retry_base_delay * (2 ** attempt) + random.uniform(0, 1),
                    self.retry_max_delay
                )
                logger.warning(
                    f"Retryable error (attempt {attempt + 1}/{self.max_retries}): "
                    f"{type(e).__name__}: {e}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        raise last_error

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> LLMResponse:
        """
        Create a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to self.default_model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    def get_context_window(self, model: str) -> int:
        """Get context window size for a model."""
        pass

    @abstractmethod
    def get_pricing(self, model: str) -> Dict[str, float]:
        """
        Get pricing for a model.

        Returns:
            Dict with 'prompt' and 'completion' prices per 1K tokens
        """
        pass

    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        on_token: Optional[Any] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Create a streaming chat completion, calling on_token for each chunk.

        Default implementation falls back to non-streaming chat_completion.
        Providers should override this for true streaming support.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to self.default_model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            on_token: Callback function(str) called with each text chunk
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse object (complete response)
        """
        # Default: fall back to non-streaming
        response = self.chat_completion(messages, model, max_tokens, temperature, **kwargs)
        if on_token and response.content:
            on_token(response.content)
        return response


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    # Context windows for OpenAI models
    CONTEXT_WINDOWS = {
        "gpt-5-mini": 128_000,
        "gpt-5-nano": 128_000,
        "gpt-4.1": 128_000,
        "gpt-4.1-mini": 128_000,
        "gpt-4.1-nano": 128_000,
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "gpt-4-turbo": 128_000,
        "gpt-4": 8_192,
        "gpt-3.5-turbo": 16_385,
    }

    # Pricing (USD per 1K tokens)
    # Source: https://openai.com/api/pricing
    PRICING = {
        "gpt-5-mini": {"prompt": 0.00025, "completion": 0.002},
        "gpt-5-nano": {"prompt": 0.00005, "completion": 0.0004},
        "gpt-4.1": {"prompt": 0.002, "completion": 0.008},
        "gpt-4.1-mini": {"prompt": 0.0004, "completion": 0.0016},
        "gpt-4.1-nano": {"prompt": 0.0001, "completion": 0.0004},
        "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    }

    def __init__(
        self,
        api_key: str,
        default_model: str = "gpt-4o",
        context_window: Optional[int] = None,
        timeout: int = 600,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0
    ):
        """Initialize OpenAI provider."""
        import openai

        # Auto-detect context window if not provided
        if context_window is None:
            context_window = self.CONTEXT_WINDOWS.get(default_model, 128_000)

        super().__init__(api_key, default_model, context_window, timeout,
                         max_retries, retry_base_delay, retry_max_delay)
        self.client = openai.OpenAI(api_key=api_key, timeout=timeout)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> LLMResponse:
        """Create OpenAI chat completion with retry on transient failures."""
        model = model or self.default_model

        # GPT-5 models use max_completion_tokens instead of max_tokens
        api_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        # Add max tokens parameter with correct name for model
        if max_tokens is not None:
            if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3"):
                api_kwargs["max_completion_tokens"] = max_tokens
            else:
                api_kwargs["max_tokens"] = max_tokens

        # Add any other kwargs
        api_kwargs.update(kwargs)

        try:
            response = self._call_with_retry(
                self.client.chat.completions.create, **api_kwargs
            )

            # Parse usage details
            usage_data = response.usage
            usage = TokenUsageDetails(
                prompt_tokens=usage_data.prompt_tokens,
                completion_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens,
            )

            # Add prompt details if available
            if hasattr(usage_data, 'prompt_tokens_details') and usage_data.prompt_tokens_details:
                details = usage_data.prompt_tokens_details
                usage.prompt_cached_tokens = getattr(details, 'cached_tokens', 0)

            # Add completion details if available
            if hasattr(usage_data, 'completion_tokens_details') and usage_data.completion_tokens_details:
                details = usage_data.completion_tokens_details
                usage.completion_reasoning_tokens = getattr(details, 'reasoning_tokens', 0)

            return LLMResponse(
                content=response.choices[0].message.content.strip(),
                usage=usage,
                model=model,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    def get_context_window(self, model: str) -> int:
        """Get context window for OpenAI model."""
        return self.CONTEXT_WINDOWS.get(model, self.context_window)

    def get_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for OpenAI model."""
        return self.PRICING.get(model, self.PRICING["gpt-4o"])

    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        on_token: Optional[Any] = None,
        **kwargs
    ) -> LLMResponse:
        """Create streaming OpenAI chat completion."""
        model = model or self.default_model

        api_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if max_tokens is not None:
            if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3"):
                api_kwargs["max_completion_tokens"] = max_tokens
            else:
                api_kwargs["max_tokens"] = max_tokens

        api_kwargs.update(kwargs)

        try:
            stream = self._call_with_retry(
                self.client.chat.completions.create, **api_kwargs
            )

            content_parts = []
            finish_reason = None
            usage = TokenUsageDetails()

            for chunk in stream:
                # Extract token content
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    content_parts.append(token)
                    if on_token:
                        on_token(token)

                # Capture finish reason
                if chunk.choices and chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # Capture usage from final chunk (when stream_options.include_usage=True)
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_data = chunk.usage
                    usage = TokenUsageDetails(
                        prompt_tokens=usage_data.prompt_tokens,
                        completion_tokens=usage_data.completion_tokens,
                        total_tokens=usage_data.total_tokens,
                    )
                    if hasattr(usage_data, 'prompt_tokens_details') and usage_data.prompt_tokens_details:
                        details = usage_data.prompt_tokens_details
                        usage.prompt_cached_tokens = getattr(details, 'cached_tokens', 0)
                    if hasattr(usage_data, 'completion_tokens_details') and usage_data.completion_tokens_details:
                        details = usage_data.completion_tokens_details
                        usage.completion_reasoning_tokens = getattr(details, 'reasoning_tokens', 0)

            return LLMResponse(
                content="".join(content_parts).strip(),
                usage=usage,
                model=model,
                finish_reason=finish_reason,
                raw_response=None
            )

        except Exception as e:
            logger.error(f"OpenAI streaming error: {str(e)}")
            raise


class XAIProvider(LLMProvider):
    """xAI Grok API provider."""

    # Context windows for xAI models
    CONTEXT_WINDOWS = {
        "grok-4": 128_000,
        "grok-4-1-fast-reasoning": 128_000,
        "grok-4-1-fast-non-reasoning": 128_000,
        "grok-4-fast-reasoning": 128_000,
        "grok-4-fast-non-reasoning": 128_000,
        "grok-beta": 128_000,
    }

    # Pricing for xAI models (USD per 1K tokens)
    # Source: https://x.ai/api pricing page
    PRICING = {
        "grok-4": {"prompt": 0.002, "completion": 0.010},
        "grok-4-1-fast-reasoning": {"prompt": 0.0002, "completion": 0.0005},
        "grok-4-1-fast-non-reasoning": {"prompt": 0.0002, "completion": 0.0005},
        "grok-4-fast-reasoning": {"prompt": 0.002, "completion": 0.010},
        "grok-4-fast-non-reasoning": {"prompt": 0.002, "completion": 0.010},
        "grok-beta": {"prompt": 0.005, "completion": 0.015},
    }

    def __init__(
        self,
        api_key: str,
        default_model: str = "grok-4",
        context_window: Optional[int] = None,
        timeout: int = 3600,  # Longer timeout for reasoning models
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0
    ):
        """Initialize xAI provider."""
        try:
            from xai_sdk import Client
            from xai_sdk.chat import user, system, assistant
        except ImportError:
            raise ImportError(
                "xAI SDK not installed. Install with: pip install xai-sdk"
            )

        # Auto-detect context window if not provided
        if context_window is None:
            context_window = self.CONTEXT_WINDOWS.get(default_model, 128_000)

        super().__init__(api_key, default_model, context_window, timeout,
                         max_retries, retry_base_delay, retry_max_delay)
        self.client = Client(api_key=api_key, timeout=timeout)

        # Store message builder functions
        self._user = user
        self._system = system
        self._assistant = assistant

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> LLMResponse:
        """Create xAI chat completion with retry on transient failures."""
        model = model or self.default_model

        # Sample kwargs
        sample_kwargs = {}
        if temperature != 1.0:
            sample_kwargs['temperature'] = temperature
        # Add any other kwargs except max_tokens
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'max_tokens'}
        sample_kwargs.update(filtered_kwargs)

        def _do_xai_call():
            # Create chat session
            chat = self.client.chat.create(model=model)

            # Convert messages to xAI format
            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                if role == "system":
                    chat.append(self._system(content))
                elif role == "user":
                    chat.append(self._user(content))
                elif role == "assistant":
                    chat.append(self._assistant(content))

            # Note: xAI SDK's sample() doesn't support max_tokens parameter
            return chat.sample(**sample_kwargs)

        try:
            response = self._call_with_retry(_do_xai_call)

            # Parse usage details
            usage_data = response.usage
            usage = TokenUsageDetails(
                prompt_tokens=getattr(usage_data, 'prompt_tokens', 0),
                completion_tokens=getattr(usage_data, 'completion_tokens', 0),
                total_tokens=getattr(usage_data, 'total_tokens', 0),
            )

            # Parse prompt details
            if hasattr(usage_data, 'prompt_tokens_details') and usage_data.prompt_tokens_details:
                details = usage_data.prompt_tokens_details
                usage.prompt_text_tokens = getattr(details, 'text_tokens', 0)
                usage.prompt_audio_tokens = getattr(details, 'audio_tokens', 0)
                usage.prompt_image_tokens = getattr(details, 'image_tokens', 0)
                usage.prompt_cached_tokens = getattr(details, 'cached_tokens', 0)

            # Parse completion details
            if hasattr(usage_data, 'completion_tokens_details') and usage_data.completion_tokens_details:
                details = usage_data.completion_tokens_details
                usage.completion_reasoning_tokens = getattr(details, 'reasoning_tokens', 0)
                usage.completion_audio_tokens = getattr(details, 'audio_tokens', 0)
                usage.completion_accepted_prediction_tokens = getattr(details, 'accepted_prediction_tokens', 0)
                usage.completion_rejected_prediction_tokens = getattr(details, 'rejected_prediction_tokens', 0)

            # Additional metadata
            usage.num_sources_used = getattr(usage_data, 'num_sources_used', 0)

            return LLMResponse(
                content=response.content,
                usage=usage,
                model=model,
                finish_reason=getattr(response, 'finish_reason', None),
                raw_response=response
            )

        except Exception as e:
            logger.error(f"xAI API error: {str(e)}")
            raise

    def get_context_window(self, model: str) -> int:
        """Get context window for xAI model."""
        return self.CONTEXT_WINDOWS.get(model, self.context_window)

    def get_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for xAI model."""
        return self.PRICING.get(model, self.PRICING["grok-4"])


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    # Context windows for Anthropic models
    CONTEXT_WINDOWS = {
        "claude-opus-4-6": 200_000,
        "claude-sonnet-4-5-20250929": 200_000,
        "claude-sonnet-4-5-20250514": 200_000,
        "claude-haiku-4-5-20251001": 200_000,
    }

    # Pricing (USD per 1K tokens)
    PRICING = {
        "claude-opus-4-6": {"prompt": 0.015, "completion": 0.075},
        "claude-sonnet-4-5-20250929": {"prompt": 0.003, "completion": 0.015},
        "claude-sonnet-4-5-20250514": {"prompt": 0.003, "completion": 0.015},
        "claude-haiku-4-5-20251001": {"prompt": 0.0008, "completion": 0.004},
    }

    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-sonnet-4-5-20250514",
        context_window: Optional[int] = None,
        timeout: int = 600,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 60.0
    ):
        """Initialize Anthropic provider."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic SDK not installed. Install with: pip install anthropic"
            )

        # Auto-detect context window if not provided
        if context_window is None:
            context_window = self.CONTEXT_WINDOWS.get(default_model, 200_000)

        super().__init__(api_key, default_model, context_window, timeout,
                         max_retries, retry_base_delay, retry_max_delay)
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> LLMResponse:
        """Create Anthropic chat completion with retry on transient failures."""
        model = model or self.default_model

        # Separate system messages from conversation messages
        system_content = ""
        conversation_messages = []

        for msg in messages:
            if msg["role"] == "system":
                # Anthropic uses a separate system parameter
                if system_content:
                    system_content += "\n\n"
                system_content += msg["content"]
            else:
                conversation_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Build API kwargs
        api_kwargs = {
            "model": model,
            "messages": conversation_messages,
            "max_tokens": max_tokens or 4096,
        }

        if system_content:
            api_kwargs["system"] = system_content

        if temperature != 1.0:
            api_kwargs["temperature"] = temperature

        # Add any other kwargs (excluding those handled above)
        for k, v in kwargs.items():
            if k not in ("system", "messages", "model", "max_tokens", "temperature"):
                api_kwargs[k] = v

        try:
            response = self._call_with_retry(
                self.client.messages.create, **api_kwargs
            )

            # Parse usage details
            usage = TokenUsageDetails(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )

            # Check for cache usage if available
            if hasattr(response.usage, 'cache_read_input_tokens'):
                usage.prompt_cached_tokens = getattr(response.usage, 'cache_read_input_tokens', 0)

            # Extract text content from response
            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text

            return LLMResponse(
                content=content.strip(),
                usage=usage,
                model=model,
                finish_reason=response.stop_reason,
                raw_response=response
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise

    def get_context_window(self, model: str) -> int:
        """Get context window for Anthropic model."""
        return self.CONTEXT_WINDOWS.get(model, self.context_window)

    def get_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for Anthropic model."""
        return self.PRICING.get(model, self.PRICING["claude-sonnet-4-5-20250514"])

    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        on_token: Optional[Any] = None,
        **kwargs
    ) -> LLMResponse:
        """Create streaming Anthropic chat completion."""
        model = model or self.default_model

        # Separate system messages from conversation messages
        system_content = ""
        conversation_messages = []

        for msg in messages:
            if msg["role"] == "system":
                if system_content:
                    system_content += "\n\n"
                system_content += msg["content"]
            else:
                conversation_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        api_kwargs = {
            "model": model,
            "messages": conversation_messages,
            "max_tokens": max_tokens or 4096,
        }

        if system_content:
            api_kwargs["system"] = system_content

        if temperature != 1.0:
            api_kwargs["temperature"] = temperature

        for k, v in kwargs.items():
            if k not in ("system", "messages", "model", "max_tokens", "temperature"):
                api_kwargs[k] = v

        try:
            content_parts = []
            usage = TokenUsageDetails()
            finish_reason = None

            with self._call_with_retry(
                self.client.messages.stream, **api_kwargs
            ) as stream:
                for event in stream:
                    # Handle text delta events
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_delta' and hasattr(event, 'delta'):
                            if hasattr(event.delta, 'text'):
                                token = event.delta.text
                                content_parts.append(token)
                                if on_token:
                                    on_token(token)
                        elif event.type == 'message_stop':
                            pass

                # Get the final message for usage info
                final_message = stream.get_final_message()
                usage = TokenUsageDetails(
                    prompt_tokens=final_message.usage.input_tokens,
                    completion_tokens=final_message.usage.output_tokens,
                    total_tokens=final_message.usage.input_tokens + final_message.usage.output_tokens,
                )
                if hasattr(final_message.usage, 'cache_read_input_tokens'):
                    usage.prompt_cached_tokens = getattr(
                        final_message.usage, 'cache_read_input_tokens', 0
                    )
                finish_reason = final_message.stop_reason

            return LLMResponse(
                content="".join(content_parts).strip(),
                usage=usage,
                model=model,
                finish_reason=finish_reason,
                raw_response=None
            )

        except Exception as e:
            logger.error(f"Anthropic streaming error: {str(e)}")
            raise


def create_provider(
    provider_name: str,
    api_key: str,
    default_model: Optional[str] = None,
    context_window: Optional[int] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    retry_base_delay: Optional[float] = None,
    retry_max_delay: Optional[float] = None
) -> LLMProvider:
    """
    Factory function to create LLM provider.

    Args:
        provider_name: Provider name ('openai', 'xai', or 'anthropic')
        api_key: API key
        default_model: Default model to use
        context_window: Context window size
        timeout: Request timeout
        max_retries: Maximum number of retries on transient failures (0 to disable)
        retry_base_delay: Base delay in seconds for exponential backoff
        retry_max_delay: Maximum delay in seconds between retries

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_name = provider_name.lower()

    kwargs = {"api_key": api_key}
    if default_model:
        kwargs["default_model"] = default_model
    if context_window:
        kwargs["context_window"] = context_window
    if timeout:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    if retry_base_delay is not None:
        kwargs["retry_base_delay"] = retry_base_delay
    if retry_max_delay is not None:
        kwargs["retry_max_delay"] = retry_max_delay

    if provider_name == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_name in ["xai", "grok"]:
        return XAIProvider(**kwargs)
    elif provider_name in ["anthropic", "claude"]:
        return AnthropicProvider(**kwargs)
    else:
        raise ValueError(
            f"Unsupported provider: {provider_name}. "
            f"Supported providers: 'openai', 'xai', 'anthropic'"
        )
