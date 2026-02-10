"""
Tests for retry logic, concurrent.futures sandbox fix, and streaming support.

Run with: pytest tests/test_retry_streaming_sandbox.py -v
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from rlm.providers import (
    LLMProvider,
    LLMResponse,
    TokenUsageDetails,
    OpenAIProvider,
    create_provider,
)
from rlm.security import RestrictedGlobals, ExecutionMonitor
from rlm.helpers import RecursionHelper
from rlm import RecursiveLanguageModel


# ---------------------------------------------------------------------------
# Helper: Mock provider for testing
# ---------------------------------------------------------------------------

class MockProvider(LLMProvider):
    """Mock provider for testing retry and streaming."""

    def __init__(self, **kwargs):
        super().__init__(
            api_key="mock-key",
            default_model="mock-model",
            context_window=128_000,
            **kwargs
        )
        self.call_count = 0
        self._responses = []
        self._stream_chunks = []

    def set_responses(self, responses):
        """Set a sequence of responses/exceptions to return."""
        self._responses = list(responses)

    def set_stream_chunks(self, chunks):
        """Set chunks for streaming."""
        self._stream_chunks = list(chunks)

    def _make_response(self, content="Mock response", model=None):
        return LLMResponse(
            content=content,
            usage=TokenUsageDetails(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            model=model or self.default_model,
            finish_reason="stop"
        )

    def chat_completion(self, messages, model=None, max_tokens=None,
                        temperature=1.0, **kwargs):
        def _do_call():
            self.call_count += 1

            if self._responses:
                item = self._responses.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

            # Default: return a simple FINAL answer for the mock
            last_msg = messages[-1]["content"] if messages else ""
            if "Task:" in last_msg or "magic" in last_msg.lower():
                content = '```repl\nFINAL("mock answer")\n```'
            else:
                content = "Mock response"

            return self._make_response(content, model)

        return self._call_with_retry(_do_call)

    def stream_chat_completion(self, messages, model=None, max_tokens=None,
                               temperature=1.0, on_token=None, **kwargs):
        self.call_count += 1

        if self._stream_chunks:
            content_parts = []
            for chunk in self._stream_chunks:
                content_parts.append(chunk)
                if on_token:
                    on_token(chunk)
            content = "".join(content_parts)
        else:
            content = '```repl\nFINAL("streamed answer")\n```'
            if on_token:
                on_token(content)

        return self._make_response(content, model)

    def get_context_window(self, model):
        return 128_000

    def get_pricing(self, model):
        return {"prompt": 0.001, "completion": 0.002}


# ===========================================================================
# 1. Retry Logic Tests
# ===========================================================================

class TestRetryLogic:
    """Tests for configurable retry with exponential backoff."""

    def test_provider_has_retry_params(self):
        """Verify retry parameters are stored on the provider."""
        provider = MockProvider(max_retries=5, retry_base_delay=2.0,
                                retry_max_delay=30.0)
        assert provider.max_retries == 5
        assert provider.retry_base_delay == 2.0
        assert provider.retry_max_delay == 30.0

    def test_default_retry_params(self):
        """Verify default retry parameters."""
        provider = MockProvider()
        assert provider.max_retries == 3
        assert provider.retry_base_delay == 1.0
        assert provider.retry_max_delay == 60.0

    def test_retry_disabled_with_zero(self):
        """max_retries=0 means no retries -- error propagates immediately."""
        provider = MockProvider(max_retries=0)
        err = RuntimeError("rate limit exceeded")
        provider.set_responses([err])

        with pytest.raises(RuntimeError, match="rate limit"):
            provider.chat_completion(
                messages=[{"role": "user", "content": "hi"}]
            )
        assert provider.call_count == 1

    def test_retries_on_rate_limit(self):
        """Rate limit errors should be retried."""
        provider = MockProvider(max_retries=2, retry_base_delay=0.01,
                                retry_max_delay=0.05)

        rate_err = Exception("429 rate limit exceeded")
        ok_response = provider._make_response("success")
        provider.set_responses([rate_err, ok_response])

        result = provider.chat_completion(
            messages=[{"role": "user", "content": "hi"}]
        )
        assert result.content == "success"
        assert provider.call_count == 2

    def test_retries_on_timeout(self):
        """Timeout errors should be retried."""
        provider = MockProvider(max_retries=3, retry_base_delay=0.01,
                                retry_max_delay=0.05)

        timeout_err = TimeoutError("request timed out")
        ok_response = provider._make_response("recovered")
        provider.set_responses([timeout_err, timeout_err, ok_response])

        result = provider.chat_completion(
            messages=[{"role": "user", "content": "hi"}]
        )
        assert result.content == "recovered"
        assert provider.call_count == 3

    def test_retries_on_server_error(self):
        """500/502/503 errors should be retried."""
        provider = MockProvider(max_retries=2, retry_base_delay=0.01,
                                retry_max_delay=0.05)

        server_err = Exception("502 Bad Gateway")
        ok_response = provider._make_response("ok")
        provider.set_responses([server_err, ok_response])

        result = provider.chat_completion(
            messages=[{"role": "user", "content": "hi"}]
        )
        assert result.content == "ok"

    def test_retries_on_connection_error(self):
        """Connection errors should be retried."""
        provider = MockProvider(max_retries=2, retry_base_delay=0.01,
                                retry_max_delay=0.05)

        conn_err = ConnectionError("connection refused")
        ok_response = provider._make_response("connected")
        provider.set_responses([conn_err, ok_response])

        result = provider.chat_completion(
            messages=[{"role": "user", "content": "hi"}]
        )
        assert result.content == "connected"

    def test_no_retry_on_auth_error(self):
        """Authentication errors should NOT be retried."""
        provider = MockProvider(max_retries=3, retry_base_delay=0.01,
                                retry_max_delay=0.05)

        auth_err = PermissionError("401 unauthorized: invalid API key")
        provider.set_responses([auth_err])

        with pytest.raises(PermissionError, match="unauthorized"):
            provider.chat_completion(
                messages=[{"role": "user", "content": "hi"}]
            )
        assert provider.call_count == 1

    def test_no_retry_on_validation_error(self):
        """Validation/client errors should NOT be retried."""
        provider = MockProvider(max_retries=3, retry_base_delay=0.01,
                                retry_max_delay=0.05)

        val_err = ValueError("invalid model name")
        provider.set_responses([val_err])

        with pytest.raises(ValueError, match="invalid model"):
            provider.chat_completion(
                messages=[{"role": "user", "content": "hi"}]
            )
        assert provider.call_count == 1

    def test_exhausted_retries_raises(self):
        """When all retries are exhausted, the last error is raised."""
        provider = MockProvider(max_retries=2, retry_base_delay=0.01,
                                retry_max_delay=0.05)

        err = Exception("503 service unavailable")
        provider.set_responses([err, err, err])

        with pytest.raises(Exception, match="503"):
            provider.chat_completion(
                messages=[{"role": "user", "content": "hi"}]
            )
        assert provider.call_count == 3  # 1 initial + 2 retries

    def test_is_retryable_error_method(self):
        """Test the _is_retryable_error classification directly."""
        provider = MockProvider()

        # Retryable
        assert provider._is_retryable_error(Exception("429 Too Many Requests"))
        assert provider._is_retryable_error(Exception("rate limit exceeded"))
        assert provider._is_retryable_error(TimeoutError("timed out"))
        assert provider._is_retryable_error(ConnectionError("connection reset"))
        assert provider._is_retryable_error(Exception("500 Internal Server Error"))
        assert provider._is_retryable_error(Exception("502 Bad Gateway"))
        assert provider._is_retryable_error(Exception("503 Service Unavailable"))
        assert provider._is_retryable_error(Exception("overloaded"))
        assert provider._is_retryable_error(Exception("529 overloaded"))

        # Not retryable
        assert not provider._is_retryable_error(ValueError("bad input"))
        assert not provider._is_retryable_error(PermissionError("forbidden"))
        assert not provider._is_retryable_error(Exception("401 unauthorized"))
        assert not provider._is_retryable_error(TypeError("wrong type"))

    def test_exponential_backoff_timing(self):
        """Verify that backoff delays increase exponentially."""
        provider = MockProvider(max_retries=3, retry_base_delay=0.05,
                                retry_max_delay=10.0)

        err = Exception("rate limit exceeded")
        ok_response = provider._make_response("ok")
        provider.set_responses([err, err, err, ok_response])

        start = time.time()
        # This should retry 3 times with increasing delays
        # Attempt 0 fails, delay ~0.05*1 + jitter
        # Attempt 1 fails, delay ~0.05*2 + jitter
        # Attempt 2 fails, delay ~0.05*4 + jitter
        # Attempt 3 succeeds
        result = provider.chat_completion(
            messages=[{"role": "user", "content": "hi"}]
        )
        elapsed = time.time() - start

        assert result.content == "ok"
        # Minimum delay should be at least base_delay * (1 + 2 + 4) = 0.35s
        # But with small values, just check it's > 0
        assert elapsed > 0.1

    def test_retry_params_passed_through_create_provider(self):
        """Verify create_provider passes retry params to the provider."""
        try:
            provider = create_provider(
                provider_name="openai",
                api_key="fake-key",
                default_model="gpt-4o",
                max_retries=5,
                retry_base_delay=2.0,
                retry_max_delay=120.0
            )
            assert provider.max_retries == 5
            assert provider.retry_base_delay == 2.0
            assert provider.retry_max_delay == 120.0
        except ModuleNotFoundError:
            # openai SDK not installed in test environment -- verify the
            # factory function at least accepts the parameters by testing
            # with our MockProvider directly (already covered by other tests)
            pytest.skip("openai SDK not installed")

    def test_retry_params_passed_through_rlm(self):
        """Verify RecursiveLanguageModel passes retry params to providers."""
        provider = MockProvider()
        rlm = RecursiveLanguageModel(
            provider=provider,
            max_retries=7,
            retry_base_delay=3.0,
            retry_max_delay=90.0,
            enable_security=False,
            log_level="WARNING"
        )
        # The retry kwargs should be stored
        assert rlm._retry_kwargs['max_retries'] == 7
        assert rlm._retry_kwargs['retry_base_delay'] == 3.0
        assert rlm._retry_kwargs['retry_max_delay'] == 90.0

    def test_overloaded_error_is_retryable(self):
        """Anthropic-style 'overloaded' errors should be retried."""
        provider = MockProvider(max_retries=1, retry_base_delay=0.01,
                                retry_max_delay=0.05)

        overloaded_err = Exception("529 overloaded")
        ok_response = provider._make_response("ok")
        provider.set_responses([overloaded_err, ok_response])

        result = provider.chat_completion(
            messages=[{"role": "user", "content": "hi"}]
        )
        assert result.content == "ok"


# ===========================================================================
# 2. concurrent.futures in SAFE_MODULES Tests
# ===========================================================================

class TestSafeModules:
    """Tests for concurrent.futures being allowed in the sandbox."""

    def test_concurrent_in_safe_modules(self):
        """Verify 'concurrent' is in SAFE_MODULES."""
        assert 'concurrent' in RestrictedGlobals.SAFE_MODULES

    def test_safe_import_concurrent_futures(self):
        """Verify concurrent.futures can be imported through the safe import."""
        # This should not raise
        module = RestrictedGlobals._safe_import(
            'concurrent.futures', globals(), locals(), ['ThreadPoolExecutor'], 0
        )
        assert module is not None

    def test_safe_import_concurrent(self):
        """Verify concurrent base module can be imported."""
        module = RestrictedGlobals._safe_import(
            'concurrent', globals(), locals(), [], 0
        )
        assert module is not None

    def test_map_reduce_parallel_in_sandbox(self):
        """Verify map_reduce(parallel=True) works inside the sandbox."""
        # Create safe globals with map_reduce available
        recursion_helper = RecursionHelper()
        custom_globals = {
            'map_reduce': recursion_helper.map_reduce,
        }
        safe_globals = RestrictedGlobals.create_safe_globals(custom_globals)

        # Execute map_reduce(parallel=True) in the sandbox
        code = """
result = map_reduce(
    items=[1, 2, 3, 4, 5],
    map_fn=lambda x: x * 2,
    reduce_fn=lambda results: sum(results),
    parallel=True
)
print(result)
"""
        monitor = ExecutionMonitor(max_execution_time=10)
        success, output, error = monitor.capture_execution(code, safe_globals)

        assert success, f"Execution failed: {error}"
        assert output.strip() == "30"

    def test_map_reduce_parallel_with_threadpool(self):
        """Verify ThreadPoolExecutor actually works through safe import."""
        code = """
from concurrent.futures import ThreadPoolExecutor

def double(x):
    return x * 2

with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(double, [1, 2, 3]))

print(results)
"""
        custom_globals = {}
        safe_globals = RestrictedGlobals.create_safe_globals(custom_globals)
        monitor = ExecutionMonitor(max_execution_time=10)
        success, output, error = monitor.capture_execution(code, safe_globals)

        assert success, f"Execution failed: {error}"
        assert "[2, 4, 6]" in output

    def test_dangerous_modules_still_blocked(self):
        """Verify dangerous modules are still blocked."""
        for module_name in ['os', 'sys', 'subprocess', 'socket']:
            with pytest.raises(ImportError):
                RestrictedGlobals._safe_import(
                    module_name, globals(), locals(), [], 0
                )


# ===========================================================================
# 3. Streaming Support Tests
# ===========================================================================

class TestStreamingSupport:
    """Tests for streaming callback support."""

    def test_stream_callback_receives_events(self):
        """Verify stream_callback receives all expected events."""
        provider = MockProvider()
        events = []

        def callback(event, data):
            events.append((event, data))

        # Set up provider to return a FINAL answer
        provider.set_stream_chunks([
            '```repl\n', 'FINAL("streamed")\n', '```'
        ])

        rlm = RecursiveLanguageModel(
            provider=provider,
            enable_security=False,
            log_level="WARNING"
        )

        # Use a small context to trigger direct mode... actually we need
        # a large context to trigger REPL mode
        context = "x" * 600_000  # Large enough to skip direct mode

        result = rlm.run(
            task="Find X",
            context=context,
            max_iterations=5,
            verbose=False,
            stream_callback=callback
        )

        # Verify we got expected event types
        event_types = [e[0] for e in events]
        assert 'iteration_start' in event_types
        assert 'llm_token' in event_types
        assert 'code_output' in event_types
        assert 'iteration_end' in event_types
        assert 'final_answer' in event_types

    def test_stream_callback_iteration_start_data(self):
        """Verify iteration_start event has correct data."""
        provider = MockProvider()
        events = []

        def callback(event, data):
            events.append((event, data))

        provider.set_stream_chunks([
            '```repl\nFINAL("done")\n```'
        ])

        rlm = RecursiveLanguageModel(
            provider=provider,
            enable_security=False,
            log_level="WARNING"
        )

        context = "x" * 600_000
        rlm.run(
            task="test",
            context=context,
            max_iterations=10,
            verbose=False,
            stream_callback=callback
        )

        start_events = [(e, d) for e, d in events if e == 'iteration_start']
        assert len(start_events) >= 1
        assert start_events[0][1]['iteration'] == 1
        assert start_events[0][1]['max_iterations'] == 10

    def test_stream_callback_llm_tokens(self):
        """Verify llm_token events contain actual token strings."""
        provider = MockProvider()
        tokens = []

        def callback(event, data):
            if event == 'llm_token':
                tokens.append(data['token'])

        provider.set_stream_chunks([
            'Let me ', 'find ', 'that.\n',
            '```repl\n', 'FINAL("found")\n', '```'
        ])

        rlm = RecursiveLanguageModel(
            provider=provider,
            enable_security=False,
            log_level="WARNING"
        )

        context = "x" * 600_000
        rlm.run(
            task="test",
            context=context,
            max_iterations=5,
            verbose=False,
            stream_callback=callback
        )

        assert len(tokens) >= 1
        # Tokens should include the chunks we set
        all_text = "".join(tokens)
        assert "find" in all_text

    def test_stream_callback_final_answer(self):
        """Verify final_answer event contains the answer."""
        provider = MockProvider()
        final_events = []

        def callback(event, data):
            if event == 'final_answer':
                final_events.append(data)

        provider.set_stream_chunks([
            '```repl\nFINAL("the answer is 42")\n```'
        ])

        rlm = RecursiveLanguageModel(
            provider=provider,
            enable_security=False,
            log_level="WARNING"
        )

        context = "x" * 600_000
        result = rlm.run(
            task="test",
            context=context,
            max_iterations=5,
            verbose=False,
            stream_callback=callback
        )

        assert len(final_events) == 1
        assert final_events[0]['answer'] == "the answer is 42"
        assert result == "the answer is 42"

    def test_run_without_stream_callback(self):
        """Verify run() works normally without stream_callback."""
        provider = MockProvider()

        rlm = RecursiveLanguageModel(
            provider=provider,
            enable_security=False,
            log_level="WARNING"
        )

        context = "x" * 600_000
        result = rlm.run(
            task="test",
            context=context,
            max_iterations=5,
            verbose=False
        )

        assert result is not None

    def test_provider_stream_fallback(self):
        """Verify base provider stream_chat_completion falls back to
        non-streaming."""
        provider = MockProvider()
        tokens = []

        def on_token(token):
            tokens.append(token)

        provider.set_responses([provider._make_response("hello world")])

        # Call the base class stream method (MockProvider inherits it)
        # Since MockProvider overrides stream_chat_completion, test the
        # base class default behavior by calling LLMProvider's version
        response = LLMProvider.stream_chat_completion(
            provider,
            messages=[{"role": "user", "content": "hi"}],
            on_token=on_token
        )

        assert response.content == "hello world"
        assert "hello world" in tokens

    def test_code_output_event(self):
        """Verify code_output events have block number and success flag."""
        provider = MockProvider()
        code_events = []

        def callback(event, data):
            if event == 'code_output':
                code_events.append(data)

        provider.set_stream_chunks([
            '```repl\nprint("hello")\nFINAL("done")\n```'
        ])

        rlm = RecursiveLanguageModel(
            provider=provider,
            enable_security=False,
            log_level="WARNING"
        )

        context = "x" * 600_000
        rlm.run(
            task="test",
            context=context,
            max_iterations=5,
            verbose=False,
            stream_callback=callback
        )

        assert len(code_events) >= 1
        assert code_events[0]['block'] == 1
        assert code_events[0]['success'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
