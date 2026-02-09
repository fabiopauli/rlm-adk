"""
Unit tests for Anthropic provider and multi-model support.

Run with: pytest tests/test_anthropic_provider.py
"""

import pytest
from unittest.mock import MagicMock, patch
from rlm.providers import (
    AnthropicProvider,
    create_provider,
    LLMResponse,
    TokenUsageDetails,
)


class TestAnthropicProviderConfig:
    """Test AnthropicProvider configuration and metadata."""

    def test_context_windows(self):
        assert AnthropicProvider.CONTEXT_WINDOWS["claude-opus-4-6"] == 200_000
        assert AnthropicProvider.CONTEXT_WINDOWS["claude-sonnet-4-5-20250514"] == 200_000
        assert AnthropicProvider.CONTEXT_WINDOWS["claude-haiku-4-5-20250514"] == 200_000

    def test_pricing(self):
        pricing = AnthropicProvider.PRICING
        # Opus is most expensive
        assert pricing["claude-opus-4-6"]["prompt"] > pricing["claude-sonnet-4-5-20250514"]["prompt"]
        assert pricing["claude-opus-4-6"]["completion"] > pricing["claude-sonnet-4-5-20250514"]["completion"]
        # Sonnet is more expensive than Haiku
        assert pricing["claude-sonnet-4-5-20250514"]["prompt"] > pricing["claude-haiku-4-5-20250514"]["prompt"]
        assert pricing["claude-sonnet-4-5-20250514"]["completion"] > pricing["claude-haiku-4-5-20250514"]["completion"]

    @patch("anthropic.Anthropic")
    def test_init_default_model(self, mock_anthropic_cls):
        provider = AnthropicProvider(api_key="test-key")
        assert provider.default_model == "claude-sonnet-4-5-20250514"
        assert provider.context_window == 200_000

    @patch("anthropic.Anthropic")
    def test_init_custom_model(self, mock_anthropic_cls):
        provider = AnthropicProvider(
            api_key="test-key",
            default_model="claude-opus-4-6",
        )
        assert provider.default_model == "claude-opus-4-6"
        assert provider.context_window == 200_000

    @patch("anthropic.Anthropic")
    def test_get_context_window(self, mock_anthropic_cls):
        provider = AnthropicProvider(api_key="test-key")
        assert provider.get_context_window("claude-opus-4-6") == 200_000
        assert provider.get_context_window("claude-haiku-4-5-20250514") == 200_000
        # Unknown model falls back to provider default
        assert provider.get_context_window("unknown-model") == 200_000

    @patch("anthropic.Anthropic")
    def test_get_pricing(self, mock_anthropic_cls):
        provider = AnthropicProvider(api_key="test-key")
        pricing = provider.get_pricing("claude-opus-4-6")
        assert "prompt" in pricing
        assert "completion" in pricing
        assert pricing["prompt"] == 0.015
        assert pricing["completion"] == 0.075

    @patch("anthropic.Anthropic")
    def test_get_pricing_fallback(self, mock_anthropic_cls):
        provider = AnthropicProvider(api_key="test-key")
        # Unknown model falls back to Sonnet pricing
        pricing = provider.get_pricing("unknown-model")
        assert pricing == AnthropicProvider.PRICING["claude-sonnet-4-5-20250514"]


class TestAnthropicProviderChatCompletion:
    """Test AnthropicProvider chat_completion with mocked API calls."""

    def _make_mock_response(self, text="Hello", input_tokens=10, output_tokens=5):
        """Create a mock Anthropic API response."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = text

        mock_usage = MagicMock()
        mock_usage.input_tokens = input_tokens
        mock_usage.output_tokens = output_tokens
        mock_usage.cache_read_input_tokens = 0

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.usage = mock_usage
        mock_response.stop_reason = "end_turn"

        return mock_response

    @patch("anthropic.Anthropic")
    def test_basic_completion(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response("Test response")

        provider = AnthropicProvider(api_key="test-key")
        response = provider.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-sonnet-4-5-20250514",
        )

        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.model == "claude-sonnet-4-5-20250514"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15
        assert response.finish_reason == "end_turn"

    @patch("anthropic.Anthropic")
    def test_system_message_separation(self, mock_anthropic_cls):
        """System messages should be passed as 'system' param, not in messages."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response()

        provider = AnthropicProvider(api_key="test-key")
        provider.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a helpful assistant."
        # Only user message should be in messages list
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    @patch("anthropic.Anthropic")
    def test_multiple_system_messages_concatenated(self, mock_anthropic_cls):
        """Multiple system messages should be concatenated."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response()

        provider = AnthropicProvider(api_key="test-key")
        provider.chat_completion(
            messages=[
                {"role": "system", "content": "First system msg."},
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Second system msg."},
            ],
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "First system msg." in call_kwargs["system"]
        assert "Second system msg." in call_kwargs["system"]

    @patch("anthropic.Anthropic")
    def test_no_system_message(self, mock_anthropic_cls):
        """When no system message, 'system' param should not be set."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response()

        provider = AnthropicProvider(api_key="test-key")
        provider.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    @patch("anthropic.Anthropic")
    def test_default_max_tokens(self, mock_anthropic_cls):
        """max_tokens defaults to 4096 when not specified."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response()

        provider = AnthropicProvider(api_key="test-key")
        provider.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 4096

    @patch("anthropic.Anthropic")
    def test_custom_max_tokens(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response()

        provider = AnthropicProvider(api_key="test-key")
        provider.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=2048,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 2048

    @patch("anthropic.Anthropic")
    def test_temperature_passed(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response()

        provider = AnthropicProvider(api_key="test-key")
        provider.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.5,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    @patch("anthropic.Anthropic")
    def test_default_temperature_not_passed(self, mock_anthropic_cls):
        """Temperature=1.0 (default) should not be explicitly passed."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response()

        provider = AnthropicProvider(api_key="test-key")
        provider.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=1.0,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "temperature" not in call_kwargs

    @patch("anthropic.Anthropic")
    def test_cache_tokens_parsed(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_response = self._make_mock_response(input_tokens=100, output_tokens=50)
        mock_response.usage.cache_read_input_tokens = 30
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(api_key="test-key")
        response = provider.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response.usage.prompt_cached_tokens == 30

    @patch("anthropic.Anthropic")
    def test_uses_default_model(self, mock_anthropic_cls):
        """When no model specified, uses the default."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = self._make_mock_response()

        provider = AnthropicProvider(
            api_key="test-key",
            default_model="claude-haiku-4-5-20250514",
        )
        response = provider.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5-20250514"
        assert response.model == "claude-haiku-4-5-20250514"


class TestCreateProviderAnthropic:
    """Test create_provider factory with Anthropic."""

    @patch("anthropic.Anthropic")
    def test_create_anthropic(self, mock_anthropic_cls):
        provider = create_provider(
            provider_name="anthropic",
            api_key="test-key",
            default_model="claude-opus-4-6",
        )
        assert isinstance(provider, AnthropicProvider)
        assert provider.default_model == "claude-opus-4-6"

    @patch("anthropic.Anthropic")
    def test_create_claude_alias(self, mock_anthropic_cls):
        provider = create_provider(
            provider_name="claude",
            api_key="test-key",
        )
        assert isinstance(provider, AnthropicProvider)

    def test_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_provider(provider_name="invalid", api_key="test-key")


class TestMultiModelCore:
    """Test multi-model (orchestrator/smart/simple) routing in RecursiveLanguageModel."""

    @patch("anthropic.Anthropic")
    def test_three_tier_init(self, mock_anthropic_cls):
        from rlm.core import RecursiveLanguageModel

        rlm = RecursiveLanguageModel(
            api_key="test-key",
            model="claude-opus-4-6",
            sub_model="claude-sonnet-4-5-20250514",
            simple_model="claude-haiku-4-5-20250514",
            provider="anthropic",
        )

        assert rlm.model == "claude-opus-4-6"
        assert rlm.sub_model == "claude-sonnet-4-5-20250514"
        assert rlm.simple_model == "claude-haiku-4-5-20250514"
        assert rlm.simple_provider is not None

    @patch("anthropic.Anthropic")
    def test_no_simple_model_fallback(self, mock_anthropic_cls):
        """Without simple_model, simple_provider should be None."""
        from rlm.core import RecursiveLanguageModel

        rlm = RecursiveLanguageModel(
            api_key="test-key",
            model="claude-opus-4-6",
            sub_model="claude-sonnet-4-5-20250514",
            provider="anthropic",
        )

        assert rlm.simple_model is None
        assert rlm.simple_provider is None

    @patch("anthropic.Anthropic")
    def test_llm_query_fast_falls_back_without_simple_model(self, mock_anthropic_cls):
        """llm_query_fast should fall back to llm_query when no simple model."""
        from rlm.core import RecursiveLanguageModel

        rlm = RecursiveLanguageModel(
            api_key="test-key",
            model="claude-opus-4-6",
            provider="anthropic",
        )

        # Mock _llm_query to verify fallback
        rlm._llm_query = MagicMock(return_value="fallback response")
        result = rlm._llm_query_fast("test prompt")

        assert result == "fallback response"
        rlm._llm_query.assert_called_once_with("test prompt", parent_call_id=None)

    @patch("anthropic.Anthropic")
    def test_detect_provider_from_model(self, mock_anthropic_cls):
        from rlm.core import RecursiveLanguageModel

        assert RecursiveLanguageModel._detect_provider("claude-opus-4-6") == "anthropic"
        assert RecursiveLanguageModel._detect_provider("claude-sonnet-4-5-20250514") == "anthropic"
        assert RecursiveLanguageModel._detect_provider("claude-haiku-4-5-20250514") == "anthropic"
        assert RecursiveLanguageModel._detect_provider("gpt-4o") == "openai"
        assert RecursiveLanguageModel._detect_provider("grok-4") == "xai"

    @patch("anthropic.Anthropic")
    def test_repl_globals_include_llm_query_fast(self, mock_anthropic_cls):
        from rlm.core import RecursiveLanguageModel

        rlm = RecursiveLanguageModel(
            api_key="test-key",
            model="claude-opus-4-6",
            simple_model="claude-haiku-4-5-20250514",
            provider="anthropic",
        )

        globals_dict = rlm._setup_repl_globals("test context")

        assert "llm_query" in globals_dict
        assert "llm_query_fast" in globals_dict
        assert globals_dict["context"] == "test context"

    @patch("anthropic.Anthropic")
    def test_metrics_summary_includes_simple_model(self, mock_anthropic_cls):
        from rlm.core import RecursiveLanguageModel

        rlm = RecursiveLanguageModel(
            api_key="test-key",
            model="claude-opus-4-6",
            sub_model="claude-sonnet-4-5-20250514",
            simple_model="claude-haiku-4-5-20250514",
            provider="anthropic",
        )

        summary = rlm.get_metrics_summary()
        assert summary["provider"]["root_model"] == "claude-opus-4-6"
        assert summary["provider"]["sub_model"] == "claude-sonnet-4-5-20250514"
        assert summary["provider"]["simple_model"] == "claude-haiku-4-5-20250514"

    @patch("anthropic.Anthropic")
    def test_metrics_summary_without_simple_model(self, mock_anthropic_cls):
        from rlm.core import RecursiveLanguageModel

        rlm = RecursiveLanguageModel(
            api_key="test-key",
            model="claude-opus-4-6",
            provider="anthropic",
        )

        summary = rlm.get_metrics_summary()
        assert "simple_model" not in summary["provider"]


class TestAnthropicMetricsPricing:
    """Test that Anthropic model pricing works in metrics."""

    def test_anthropic_pricing_in_fallback_table(self):
        from rlm.metrics import RLMMetrics

        metrics = RLMMetrics()

        # Record calls with Anthropic models
        metrics.record_call(
            model="claude-opus-4-6",
            prompt_tokens=1000,
            completion_tokens=1000,
        )

        # Opus: $0.015/1K prompt + $0.075/1K completion = $0.09
        expected_cost = (1000 / 1000 * 0.015) + (1000 / 1000 * 0.075)
        assert metrics.total_cost == pytest.approx(expected_cost, rel=1e-5)

    def test_sonnet_pricing(self):
        from rlm.metrics import RLMMetrics

        metrics = RLMMetrics()
        metrics.record_call(
            model="claude-sonnet-4-5-20250514",
            prompt_tokens=1000,
            completion_tokens=1000,
        )

        expected_cost = (1000 / 1000 * 0.003) + (1000 / 1000 * 0.015)
        assert metrics.total_cost == pytest.approx(expected_cost, rel=1e-5)

    def test_haiku_pricing(self):
        from rlm.metrics import RLMMetrics

        metrics = RLMMetrics()
        metrics.record_call(
            model="claude-haiku-4-5-20250514",
            prompt_tokens=1000,
            completion_tokens=1000,
        )

        expected_cost = (1000 / 1000 * 0.0008) + (1000 / 1000 * 0.004)
        assert metrics.total_cost == pytest.approx(expected_cost, rel=1e-5)

    def test_cost_by_model_multi_provider(self):
        from rlm.metrics import RLMMetrics

        metrics = RLMMetrics()
        metrics.record_call(model="claude-opus-4-6", prompt_tokens=100, completion_tokens=50)
        metrics.record_call(model="claude-sonnet-4-5-20250514", prompt_tokens=100, completion_tokens=50)
        metrics.record_call(model="claude-haiku-4-5-20250514", prompt_tokens=100, completion_tokens=50)

        assert "claude-opus-4-6" in metrics.cost_by_model
        assert "claude-sonnet-4-5-20250514" in metrics.cost_by_model
        assert "claude-haiku-4-5-20250514" in metrics.cost_by_model
        # Opus should be most expensive
        assert metrics.cost_by_model["claude-opus-4-6"] > metrics.cost_by_model["claude-sonnet-4-5-20250514"]
        assert metrics.cost_by_model["claude-sonnet-4-5-20250514"] > metrics.cost_by_model["claude-haiku-4-5-20250514"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
