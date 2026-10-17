"""
Tests for real recursion (rlm_query) and parallel sub-calls (llm_query_batched).

Uses a scripted fake provider, no API keys needed.
Run with: pytest tests/test_recursion.py
"""

import re
import threading
import time

import pytest

from rlm import RecursiveLanguageModel
from rlm.providers import LLMProvider, LLMResponse, TokenUsageDetails


def _usage():
    return TokenUsageDetails(prompt_tokens=10, completion_tokens=5, total_tokens=15)


class ScriptedProvider(LLMProvider):
    """
    Fake provider. REPL turns (system prompt present) get code from `scripts`,
    keyed by the RLM depth parsed from the system prompt; plain sub-calls are
    answered by `sub_call_fn`.
    """

    def __init__(self, scripts, sub_call_fn=None, sub_call_delay=0.0, context_window=100):
        super().__init__(api_key="k", default_model="fake", context_window=context_window)
        self.scripts = scripts
        self.sub_call_fn = sub_call_fn or (lambda prompt: f"ok:{prompt}")
        self.sub_call_delay = sub_call_delay
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.sub_calls = 0

    def chat_completion(self, messages, model=None, max_tokens=None, **kwargs):
        if messages[0]["role"] == "system":
            depth = int(re.search(r"you are at depth (\d+)", messages[0]["content"]).group(1))
            text = f"```repl\n{self.scripts[depth]}\n```"
        else:
            with self._lock:
                self.active += 1
                self.sub_calls += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(self.sub_call_delay)
                text = self.sub_call_fn(messages[-1]["content"])
            finally:
                with self._lock:
                    self.active -= 1
        return LLMResponse(content=text, usage=_usage(), model=model or "fake")

    def get_context_window(self, model):
        return self.context_window

    def get_pricing(self, model):
        return {"prompt": 0.001, "completion": 0.001}


def _rlm(provider, **kwargs):
    kwargs.setdefault("enable_cache", False)
    return RecursiveLanguageModel(provider=provider, log_level="ERROR", **kwargs)


class TestLLMQueryBatched:

    def test_results_in_order_and_parallel(self):
        provider = ScriptedProvider(
            scripts={0: "res = llm_query_batched([f'p{i}' for i in range(6)])\nFINAL(res)"},
            sub_call_delay=0.2,
        )
        rlm = _rlm(provider, max_concurrency=4)
        start = time.time()
        result = rlm.run("t", "x" * 1000, force_repl=True, verbose=False)
        elapsed = time.time() - start

        assert result == [f"ok:p{i}" for i in range(6)]
        assert provider.max_active > 1          # ran concurrently
        assert provider.max_active <= 4         # bounded by max_concurrency
        assert elapsed < 6 * 0.2                # faster than sequential

    def test_failed_prompt_returns_error_string(self):
        def sub(prompt):
            if prompt == "bad":
                raise ValueError("boom")
            return "good"

        provider = ScriptedProvider(
            scripts={0: "FINAL(llm_query_batched(['a', 'bad', 'b']))"},
            sub_call_fn=sub,
        )
        rlm = _rlm(provider, max_retries=0)
        result = rlm.run("t", "x" * 1000, force_repl=True, verbose=False)
        assert result[0] == "good" and result[2] == "good"
        assert result[1].startswith("[ERROR]")


class TestRLMQuery:

    def test_children_run_their_own_repl(self):
        # Root splits the context and delegates each half to a child RLM;
        # each child counts 'x' in its own context with code.
        provider = ScriptedProvider(scripts={
            0: ("half = len(context) // 2\n"
                "parts = [context[:half], context[half:]]\n"
                "res = rlm_query_batched(['count x'] * 2, contexts=parts)\n"
                "print('root saw', res)\n"
                "FINAL(sum(int(r) for r in res))"),
            1: "print('child output')\nFINAL(str(context.count('x')))",
        })
        rlm = _rlm(provider, max_depth=2)
        context = "x" * 300 + "y" * 300 + "x" * 150 + "y" * 150

        result = rlm.run("count x", context, force_repl=True, verbose=False)

        assert result == 450
        summary = rlm.get_metrics_summary()
        assert summary["max_recursion_depth"] == 1
        # Child prints stay in the child's REPL feedback, not the root's
        root_feedback = "\n".join(m["content"] for m in rlm.history if m["role"] == "user")
        assert "root saw" in root_feedback
        assert "child output" not in root_feedback

    def test_nested_recursion_two_levels(self):
        provider = ScriptedProvider(scripts={
            0: "FINAL(rlm_query('go', context=context))",
            1: "FINAL('d1>' + str(rlm_query('go', context=context)))",
            2: "FINAL('d2:' + str(len(context)))",
        })
        rlm = _rlm(provider, max_depth=2)
        result = rlm.run("go", "z" * 800, force_repl=True, verbose=False)
        assert result == "d1>d2:800"
        assert rlm.get_metrics_summary()["max_recursion_depth"] == 2

    def test_max_depth_falls_back_to_llm_query(self):
        provider = ScriptedProvider(
            scripts={0: "FINAL(rlm_query('summarize', context='abc'))"},
            sub_call_fn=lambda prompt: "plain:" + prompt.replace("\n", "|"),
        )
        rlm = _rlm(provider, max_depth=0)
        result = rlm.run("t", "x" * 1000, force_repl=True, verbose=False)
        assert result == "plain:summarize||Context:|abc"

    def test_small_child_context_uses_direct_mode(self):
        provider = ScriptedProvider(
            scripts={0: "FINAL(rlm_query('what is it?', context='tiny'))"},
            sub_call_fn=lambda prompt: "direct-answer",
            context_window=100_000,
        )
        rlm = _rlm(provider, max_depth=2)
        result = rlm.run("t", "x" * 1000, force_repl=True, verbose=False)
        assert result == "direct-answer"

    def test_child_failure_returns_error_string(self):
        provider = ScriptedProvider(scripts={
            0: "FINAL(rlm_query('loop', context=context))",
            1: "x = 1",  # never calls FINAL
        })
        rlm = _rlm(provider, max_depth=1, child_max_iterations=2)
        result = rlm.run("t", "x" * 1000, force_repl=True, verbose=False)
        assert str(result).startswith("[ERROR] child RLM failed")

    def test_budget_exceeded_propagates(self):
        provider = ScriptedProvider(scripts={
            0: "FINAL(llm_query_batched(['p'] * 50))",
        })
        rlm = _rlm(provider, max_cost=0.0001)
        with pytest.raises(RuntimeError, match="Budget exceeded|consecutive REPL errors"):
            rlm.run("t", "x" * 1000, force_repl=True, verbose=False, max_iterations=3)


class TestTimeout:

    def test_slow_llm_calls_do_not_count_against_timeout(self):
        provider = ScriptedProvider(
            scripts={0: "a = llm_query('one')\nb = llm_query('two')\nFINAL(a + b)"},
            sub_call_delay=0.8,
        )
        rlm = _rlm(provider)
        rlm.execution_monitor.max_execution_time = 1
        result = rlm.run("t", "x" * 1000, force_repl=True, verbose=False)
        assert result == "ok:oneok:two"

    def test_busy_code_still_times_out(self):
        from rlm.security import ExecutionMonitor
        monitor = ExecutionMonitor(max_execution_time=1)
        success, _, error = monitor.capture_execution("while 1:\n    pass", {})
        assert not success
        assert "Timeout" in error
