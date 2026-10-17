"""
Unit tests for the RunInfra provider (OpenAI-compatible gateway).

Run with: pytest tests/test_runinfra_provider.py
"""

import pytest
from unittest.mock import MagicMock, patch
from rlm.providers import (
    RunInfraProvider,
    create_provider,
    LLMResponse,
)


def _mock_completion(content="Hello", prompt_tokens=10, completion_tokens=5):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].finish_reason = "stop"
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens
    response.usage.prompt_tokens_details = None
    response.usage.completion_tokens_details = None
    return response


def _mock_model(model_id, context_window, input_price, output_price):
    m = MagicMock()
    m.id = model_id
    m.model_extra = {
        "context_window": context_window,
        "pricing": {"input": input_price, "output": output_price},
    }
    return m


class TestRunInfraProviderConfig:
    """Test RunInfraProvider configuration and metadata."""

    @patch("openai.OpenAI")
    def test_init_uses_gateway_base_url(self, mock_openai_cls):
        provider = RunInfraProvider(api_key="test-key")
        assert provider.default_model == "deepseek-v4-1-flash-1017"
        assert provider.context_window == 1_048_576
        _, kwargs = mock_openai_cls.call_args
        assert kwargs["base_url"] == "https://api.runinfra.ai/v1"
        assert kwargs["api_key"] == "test-key"

    def test_handles_model(self):
        assert RunInfraProvider.handles_model("deepseek-v4-1-flash-1017")
        assert RunInfraProvider.handles_model("glm-5-3-flash")
        assert RunInfraProvider.handles_model("qwen3-8-27b")
        assert not RunInfraProvider.handles_model("gpt-4o")
        assert not RunInfraProvider.handles_model("claude-opus-4-6")

    @patch("openai.OpenAI")
    def test_catalog_overrides_static_tables(self, mock_openai_cls):
        client = mock_openai_cls.return_value
        client.models.list.return_value.data = [
            _mock_model("deepseek-v4-1-flash-1017", 1_048_576, 0.3, 1.2),
            _mock_model("new-model-x", 65_536, 0.1, 0.2),
        ]
        provider = RunInfraProvider(api_key="test-key")
        assert provider.get_pricing("deepseek-v4-1-flash-1017") == {"prompt": 0.0003, "completion": 0.0012}
        assert provider.get_context_window("new-model-x") == 65_536
        # Catalog is fetched once
        provider.get_pricing("new-model-x")
        assert client.models.list.call_count == 1
        # Class tables are not modified
        assert "new-model-x" not in RunInfraProvider.CONTEXT_WINDOWS

    @patch("openai.OpenAI")
    def test_catalog_failure_falls_back(self, mock_openai_cls):
        mock_openai_cls.return_value.models.list.side_effect = RuntimeError("offline")
        provider = RunInfraProvider(api_key="test-key")
        assert provider.get_pricing("glm-5-3-flash") == RunInfraProvider.PRICING["glm-5-3-flash"]
        assert provider.get_pricing("unknown") == RunInfraProvider.PRICING["deepseek-v4-1-flash-1017"]


class TestRunInfraChatCompletion:
    """Test RunInfraProvider.chat_completion request shaping."""

    @patch("openai.OpenAI")
    def test_basic_completion(self, mock_openai_cls):
        client = mock_openai_cls.return_value
        client.chat.completions.create.return_value = _mock_completion("Hi there")
        provider = RunInfraProvider(api_key="test-key")

        result = provider.chat_completion([{"role": "user", "content": "Hello"}], max_tokens=100)

        assert isinstance(result, LLMResponse)
        assert result.content == "Hi there"
        assert result.usage.prompt_tokens == 10
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["model"] == "deepseek-v4-1-flash-1017"
        assert kwargs["max_tokens"] == 100
        assert "X-Client-Request-Id" in kwargs["extra_headers"]
        assert "reasoning_effort" not in kwargs

    @patch("openai.OpenAI")
    def test_request_id_unique_per_call(self, mock_openai_cls):
        client = mock_openai_cls.return_value
        client.chat.completions.create.return_value = _mock_completion()
        provider = RunInfraProvider(api_key="test-key")
        provider.chat_completion([{"role": "user", "content": "a"}])
        provider.chat_completion([{"role": "user", "content": "b"}])
        ids = [c.kwargs["extra_headers"]["X-Client-Request-Id"]
               for c in client.chat.completions.create.call_args_list]
        assert ids[0] != ids[1]

    @patch("openai.OpenAI")
    def test_reasoning_effort_passed(self, mock_openai_cls):
        client = mock_openai_cls.return_value
        client.chat.completions.create.return_value = _mock_completion()
        provider = RunInfraProvider(api_key="test-key", reasoning_effort="low")
        provider.chat_completion([{"role": "user", "content": "Hello"}])
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["reasoning_effort"] == "low"

    @patch("openai.OpenAI")
    def test_none_content_becomes_empty(self, mock_openai_cls):
        client = mock_openai_cls.return_value
        client.chat.completions.create.return_value = _mock_completion(content=None)
        provider = RunInfraProvider(api_key="test-key")
        result = provider.chat_completion([{"role": "user", "content": "Hello"}])
        assert result.content == ""


class TestCreateProviderRunInfra:
    """Test the factory and core auto-detection."""

    @patch("openai.OpenAI")
    def test_create_runinfra(self, mock_openai_cls):
        provider = create_provider("runinfra", api_key="k", reasoning_effort="high")
        assert isinstance(provider, RunInfraProvider)
        assert provider.reasoning_effort == "high"

    @patch("openai.OpenAI")
    def test_core_detects_runinfra(self, mock_openai_cls):
        from rlm import RecursiveLanguageModel
        rlm = RecursiveLanguageModel(
            model="deepseek-v4-1-flash-1017",
            runinfra_api_key="runinfra-key",
            reasoning_effort="low",
        )
        assert isinstance(rlm.root_provider, RunInfraProvider)
        assert rlm.root_provider.reasoning_effort == "low"
        assert RecursiveLanguageModel._detect_provider("glm-5-3-flash") == "runinfra"
        _, kwargs = mock_openai_cls.call_args
        assert kwargs["api_key"] == "runinfra-key"

    def test_unsupported_provider_mentions_runinfra(self):
        with pytest.raises(ValueError, match="runinfra"):
            create_provider("nope", api_key="k")
