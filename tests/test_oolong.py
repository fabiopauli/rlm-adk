"""
Oolong Benchmark for RLM-ADK.

Evaluates long-context aggregation reasoning using the Oolong benchmark
(https://arxiv.org/abs/2511.02817). Tests classification, counting, and
comparison tasks across large volumes of text.

Usage:
    # Smoke test (~$0.25, 5 min)
    python -m tests.test_oolong --dataset synth --subset 5 --max-cost 1.0

    # Quick benchmark (~$2, 20 min)
    python -m tests.test_oolong --dataset synth --subset 25 --max-cost 5.0

    # Moderate benchmark (~$5, 30 min) -- DEFAULT
    python -m tests.test_oolong --dataset synth --subset 50 --max-cost 5.0

    # Standard validation run (~$35-65, 8-22 hours)
    python -m tests.test_oolong --dataset synth --split validation --max-cost 100.0 --checkpoint results_checkpoint.json

    # Full test run (~$275-530, multi-day)
    python -m tests.test_oolong --dataset synth --split test --max-cost 600.0 --checkpoint results_checkpoint.json
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from rlm.core import RecursiveLanguageModel


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class OolongScorer:
    """Scoring functions matching the official Oolong evaluation."""

    @staticmethod
    def parse_gold(gold_raw: Any) -> Any:
        """Parse gold answer from dataset's list-string format.

        The oolong-synth `answer` field stores values as string representations
        of Python lists, e.g. "['spam']", "[4]", "['less common than']".
        """
        if gold_raw is None:
            return None

        text = str(gold_raw).strip()

        # Try parsing as Python literal (handles "['spam']", "[4]", etc.)
        try:
            import ast
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list) and len(parsed) == 1:
                return parsed[0]
            if isinstance(parsed, list):
                return parsed
            return parsed
        except (ValueError, SyntaxError):
            pass

        return text

    @staticmethod
    def score_numeric(gold: float, pred: float) -> float:
        """Partial credit for numeric answers: 0.75^|gold - pred|."""
        return 0.75 ** abs(gold - pred)

    @staticmethod
    def score_exact_match(gold: str, pred: str) -> float:
        """Case-insensitive exact match for labels/dates/comparisons."""
        return 1.0 if gold.strip().lower() == pred.strip().lower() else 0.0

    @staticmethod
    def score_set_overlap(gold_list: List[str], pred_list: List[str]) -> float:
        """Set overlap: len(intersection) / len(gold) for list answers."""
        if not gold_list:
            return 1.0 if not pred_list else 0.0
        gold_set = {s.strip().lower() for s in gold_list}
        pred_set = {s.strip().lower() for s in pred_list}
        return len(gold_set & pred_set) / len(gold_set)

    @staticmethod
    def parse_synth_answer(raw: str, answer_type: str) -> Any:
        """Extract clean answer from RLM output for synth dataset.

        Handles \\boxed{}, "Answer:/Label:/User:", numeric extraction, etc.
        """
        if raw is None:
            return None

        text = str(raw).strip()

        # Extract from \boxed{...}
        boxed = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            text = boxed.group(1).strip()

        # Extract from "Answer: ...", "Label: ...", "User: ...", or "answer is ..."
        answer_match = re.search(
            r'(?:(?:answer|label|user)\s*(?:is|:)\s*)(.+?)(?:\.|$)',
            text, re.IGNORECASE
        )
        if answer_match:
            text = answer_match.group(1).strip()

        # Strip trailing period/whitespace
        text = text.rstrip('. ')

        answer_type_upper = answer_type.upper() if answer_type else ""

        if answer_type_upper == "NUMERIC":
            # Extract numeric value
            num_match = re.search(r'-?\d+(?:\.\d+)?', text)
            if num_match:
                val = float(num_match.group())
                return int(val) if val == int(val) else val
            return None

        if answer_type_upper == "COMPARISON":
            # Gold is just the comparison phrase (e.g. "less common than").
            # Model may answer "ham is less common than spam" — extract the phrase.
            comp_match = re.search(
                r'(more common than|less common than|same frequency as|'
                r'more common|less common|the same frequency)',
                text, re.IGNORECASE,
            )
            if comp_match:
                return comp_match.group(1).strip().lower()

        # LABEL, USER, DATE -- return as string
        return text

    @staticmethod
    def parse_real_answer(raw: str) -> Any:
        """Parse answer for real dataset. Detect type (int/str/list)."""
        if raw is None:
            return None

        text = str(raw).strip()

        # Extract from \boxed{...}
        boxed = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            text = boxed.group(1).strip()

        # Extract from "Answer: ..."
        answer_match = re.search(
            r'(?:answer\s*(?:is|:)\s*)(.+?)(?:\.|$)',
            text, re.IGNORECASE
        )
        if answer_match:
            text = answer_match.group(1).strip()

        text = text.rstrip('. ')

        # Try list (comma-separated)
        if ',' in text:
            items = [item.strip() for item in text.split(',') if item.strip()]
            if len(items) > 1:
                return items

        # Try integer
        num_match = re.fullmatch(r'-?\d+', text)
        if num_match:
            return int(num_match.group())

        # Try float
        num_match = re.fullmatch(r'-?\d+\.\d+', text)
        if num_match:
            return float(num_match.group())

        return text

    @classmethod
    def score_synth(cls, gold_raw: Any, pred_raw: str, answer_type: str) -> float:
        """Score a synth-dataset prediction."""
        gold = cls.parse_gold(gold_raw)
        pred = cls.parse_synth_answer(pred_raw, answer_type)
        if pred is None:
            return 0.0

        answer_type_upper = answer_type.upper() if answer_type else ""

        if answer_type_upper == "NUMERIC":
            try:
                return cls.score_numeric(float(gold), float(pred))
            except (ValueError, TypeError):
                return 0.0

        # LABEL, COMPARISON, USER, DATE
        return cls.score_exact_match(str(gold), str(pred))

    @classmethod
    def score_real(cls, gold: Any, pred_raw: str) -> float:
        """Score a real-dataset prediction."""
        pred = cls.parse_real_answer(pred_raw)
        if pred is None:
            return 0.0

        # List answer
        if isinstance(gold, list):
            pred_list = pred if isinstance(pred, list) else [str(pred)]
            return cls.score_set_overlap(
                [str(g) for g in gold],
                [str(p) for p in pred_list]
            )

        # Integer / float -- numeric partial credit
        if isinstance(gold, (int, float)):
            try:
                return cls.score_numeric(float(gold), float(pred))
            except (ValueError, TypeError):
                return 0.0

        # String exact match
        return cls.score_exact_match(str(gold), str(pred))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OolongResult:
    """Per-question result."""
    id: str = ""
    question: str = ""
    expected: Any = None
    predicted: Any = None
    score: float = 0.0
    answer_type: str = ""
    task: str = ""
    context_window_id: str = ""
    context_len: int = 0
    cost: float = 0.0
    tokens: int = 0
    duration: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Truncate long strings for readability
        if isinstance(d.get("question"), str) and len(d["question"]) > 200:
            d["question"] = d["question"][:200] + "..."
        return d


@dataclass
class OolongBenchmarkSummary:
    """Aggregated benchmark results."""
    overall_score: float = 0.0
    num_questions: int = 0
    num_scored: int = 0
    score_by_answer_type: Dict[str, float] = field(default_factory=dict)
    score_by_task: Dict[str, float] = field(default_factory=dict)
    score_by_context_len: Dict[str, float] = field(default_factory=dict)
    total_cost: float = 0.0
    total_tokens: int = 0
    total_duration: float = 0.0
    num_errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_results(cls, results: List[OolongResult]) -> "OolongBenchmarkSummary":
        """Compute summary from a list of results."""
        if not results:
            return cls()

        scored = [r for r in results if r.error is None]
        overall = sum(r.score for r in scored) / len(scored) if scored else 0.0

        # By answer type
        by_type: Dict[str, List[float]] = defaultdict(list)
        for r in scored:
            if r.answer_type:
                by_type[r.answer_type].append(r.score)
        score_by_type = {
            k: sum(v) / len(v) for k, v in by_type.items()
        }

        # By task
        by_task: Dict[str, List[float]] = defaultdict(list)
        for r in scored:
            if r.task:
                by_task[r.task].append(r.score)
        score_by_task = {
            k: sum(v) / len(v) for k, v in by_task.items()
        }

        # By context length bucket
        by_ctx: Dict[str, List[float]] = defaultdict(list)
        for r in scored:
            if r.context_len > 0:
                bucket = _context_len_bucket(r.context_len)
                by_ctx[bucket].append(r.score)
        score_by_ctx = {
            k: sum(v) / len(v) for k, v in sorted(by_ctx.items())
        }

        return cls(
            overall_score=overall,
            num_questions=len(results),
            num_scored=len(scored),
            score_by_answer_type=score_by_type,
            score_by_task=score_by_task,
            score_by_context_len=score_by_ctx,
            total_cost=sum(r.cost for r in results),
            total_tokens=sum(r.tokens for r in results),
            total_duration=sum(r.duration for r in results),
            num_errors=sum(1 for r in results if r.error is not None),
        )


def _context_len_bucket(n_tokens: int) -> str:
    """Bucket context length into human-readable ranges."""
    if n_tokens <= 4_000:
        return "0-4K"
    elif n_tokens <= 16_000:
        return "4-16K"
    elif n_tokens <= 32_000:
        return "16-32K"
    elif n_tokens <= 64_000:
        return "64K"
    elif n_tokens <= 128_000:
        return "128K"
    else:
        return ">128K"


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class OolongBenchmark:
    """Main benchmark runner for the Oolong long-context evaluation."""

    def __init__(
        self,
        api_key: str,
        model: str = "grok-4-1-fast-reasoning",
        provider: str = "xai",
        verbose: bool = True,
        max_cost_per_question: float = 0.50,
        max_cost_total: float = 5.0,
    ):
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.verbose = verbose
        self.max_cost_per_question = max_cost_per_question
        self.max_cost_total = max_cost_total
        self.results: List[OolongResult] = []
        self._cumulative_cost = 0.0
        self._cumulative_tokens = 0
        self._start_time = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_synth(
        self,
        split: str = "validation",
        max_samples: Optional[int] = 50,
        max_context_windows: Optional[int] = None,
        max_context_len: Optional[int] = None,
    ) -> Dict[str, List[Dict]]:
        """Load oolong-synth from HuggingFace, grouped by context_window_id.

        Returns:
            Dict mapping context_window_id -> list of question dicts.
        """
        from datasets import load_dataset

        if self.verbose:
            print(f"Loading oolong-synth ({split}) with streaming...")

        ds = load_dataset("oolongbench/oolong-synth", split=split, streaming=True)

        windows: Dict[str, List[Dict]] = defaultdict(list)
        total_loaded = 0

        for row in ds:
            cw_id = str(row.get("context_window_id", ""))
            ctx_len = row.get("context_len", 0)

            if max_context_len and ctx_len > max_context_len:
                continue

            if max_context_windows and len(windows) >= max_context_windows and cw_id not in windows:
                # Already have enough distinct windows
                if max_samples and total_loaded >= max_samples:
                    break
                continue

            windows[cw_id].append(row)
            total_loaded += 1

            if max_samples and total_loaded >= max_samples:
                break

        if self.verbose:
            print(f"  Loaded {total_loaded} questions across {len(windows)} context windows")

        # Sort windows by context length (shortest first) for fast early signal
        sorted_windows = dict(
            sorted(
                windows.items(),
                key=lambda kv: kv[1][0].get("context_len", 0) if kv[1] else 0,
            )
        )
        return sorted_windows

    def load_real(
        self,
        config: str = "toy_dnd",
        split: str = "test",
        max_samples: Optional[int] = 50,
        max_context_windows: Optional[int] = None,
    ) -> Dict[str, List[Dict]]:
        """Load oolong-real from HuggingFace, grouped by context_window_id.

        Args:
            config: 'toy_dnd' or 'dnd'
        """
        from datasets import load_dataset

        if self.verbose:
            print(f"Loading oolong-real/{config} ({split}) with streaming...")

        ds = load_dataset(
            "oolongbench/oolong-real", config, split=split, streaming=True
        )

        windows: Dict[str, List[Dict]] = defaultdict(list)
        total_loaded = 0

        for row in ds:
            cw_id = str(row.get("context_window_id", ""))

            if max_context_windows and len(windows) >= max_context_windows and cw_id not in windows:
                if max_samples and total_loaded >= max_samples:
                    break
                continue

            windows[cw_id].append(row)
            total_loaded += 1

            if max_samples and total_loaded >= max_samples:
                break

        if self.verbose:
            print(f"  Loaded {total_loaded} questions across {len(windows)} context windows")

        sorted_windows = dict(
            sorted(
                windows.items(),
                key=lambda kv: len(kv[1][0].get("context_window_text", "")) // 4 if kv[1] else 0,
            )
        )
        return sorted_windows

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _create_rlm(self) -> RecursiveLanguageModel:
        """Create a fresh RLM instance."""
        api_key_kwarg = {}
        if self.provider == "xai":
            api_key_kwarg["xai_api_key"] = self.api_key
        elif self.provider == "openai":
            api_key_kwarg["openai_api_key"] = self.api_key
        elif self.provider == "anthropic":
            api_key_kwarg["anthropic_api_key"] = self.api_key

        return RecursiveLanguageModel(
            model=self.model,
            provider=self.provider,
            max_cost=self.max_cost_per_question,
            enable_cache=True,
            cache_ttl=None,  # No TTL within a context window
            log_level="WARNING" if not self.verbose else "INFO",
            **api_key_kwarg,
        )

    def _process_context_window(
        self,
        cw_id: str,
        questions: List[Dict],
        dataset_type: str,
    ) -> List[OolongResult]:
        """Process all questions sharing a context window.

        One RLM instance per window. Cache is preserved between questions
        (since sub-calls on the same context may overlap).
        """
        results = []
        rlm = self._create_rlm()

        for i, q in enumerate(questions):
            # Budget check
            if self._cumulative_cost >= self.max_cost_total:
                if self.verbose:
                    print(f"  [BUDGET] Cumulative cost ${self._cumulative_cost:.2f} >= ${self.max_cost_total:.2f}. Stopping.")
                break

            q_id = str(q.get("id", f"{cw_id}_{i}"))
            question_text = q.get("question", "")
            context = q.get("context_window_text_with_labels", q.get("context_window_text", ""))
            context_len = q.get("context_len", len(context) // 4)

            if dataset_type == "synth":
                # answer_type stored as "ANSWER_TYPE.LABEL" — strip prefix
                raw_answer_type = q.get("answer_type", "")
                answer_type = str(raw_answer_type).replace("ANSWER_TYPE.", "")
                # task stored as "TASK_TYPE.MOST_FREQ" — strip prefix
                raw_task = q.get("task", "")
                task = str(raw_task).replace("TASK_TYPE.", "")
                gold = q.get("answer", q.get("gold_answer", ""))
            else:
                answer_type = ""
                task = q.get("question_type", "")
                gold = q.get("answer", q.get("gold_answer", ""))

            result = OolongResult(
                id=q_id,
                question=question_text,
                expected=gold,
                answer_type=answer_type,
                task=task,
                context_window_id=cw_id,
                context_len=context_len,
            )

            if self.verbose:
                type_label = f" [{answer_type}]" if answer_type else ""
                print(f"  Q{i+1}/{len(questions)}{type_label}: {question_text[:80]}...")

            t0 = time.time()
            try:
                # Save cache before reset (preserve across questions in same window)
                if i > 0:
                    saved_cache = rlm.cache._cache.copy() if rlm.cache else None
                    rlm.reset()
                    if saved_cache and rlm.cache:
                        rlm.cache._cache.update(saved_cache)

                answer = rlm.run(
                    task=question_text,
                    context=context,
                    verbose=self.verbose,
                    max_iterations=30,
                    force_repl=True,
                )
                result.predicted = answer
                result.duration = time.time() - t0

                # Gather metrics from this question
                metrics = rlm.metrics.get_summary()
                result.cost = metrics["cost"]["total_usd"]
                result.tokens = metrics["tokens"]["total"]

                # Score
                if dataset_type == "synth":
                    result.score = OolongScorer.score_synth(gold, str(answer), answer_type)
                else:
                    result.score = OolongScorer.score_real(gold, str(answer))

            except Exception as e:
                result.duration = time.time() - t0
                result.error = str(e)
                if self.verbose:
                    print(f"    ERROR: {e}")

            self._cumulative_cost += result.cost
            self._cumulative_tokens += result.tokens
            results.append(result)

            if self.verbose:
                score_str = f"{result.score:.2f}" if result.error is None else "ERR"
                parsed = None
                if result.error is None and dataset_type == "synth":
                    parsed = OolongScorer.parse_synth_answer(str(result.predicted), answer_type)
                elif result.error is None:
                    parsed = OolongScorer.parse_real_answer(str(result.predicted))
                print(f"    Gold: {repr(gold)} | Predicted: {repr(result.predicted)} | Parsed: {repr(parsed)}")
                print(f"    Score: {score_str} | Cost: ${result.cost:.4f} | Tokens: {result.tokens}")

        return results

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def run_synth(
        self,
        split: str = "validation",
        max_samples: Optional[int] = 50,
        max_context_windows: Optional[int] = None,
        max_context_len: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        resume_path: Optional[str] = None,
    ) -> OolongBenchmarkSummary:
        """Run benchmark on oolong-synth dataset."""
        self._start_time = time.time()
        self.results = []

        # Resume from checkpoint if provided
        completed_ids = set()
        if resume_path and os.path.exists(resume_path):
            self.results, completed_ids = self._load_checkpoint(resume_path)
            self._cumulative_cost = sum(r.cost for r in self.results)
            self._cumulative_tokens = sum(r.tokens for r in self.results)
            if self.verbose:
                print(f"Resumed {len(self.results)} results from checkpoint (cost so far: ${self._cumulative_cost:.2f})")
        elif checkpoint_path and os.path.exists(checkpoint_path):
            self.results, completed_ids = self._load_checkpoint(checkpoint_path)
            self._cumulative_cost = sum(r.cost for r in self.results)
            self._cumulative_tokens = sum(r.tokens for r in self.results)
            if self.verbose:
                print(f"Resumed {len(self.results)} results from checkpoint (cost so far: ${self._cumulative_cost:.2f})")

        windows = self.load_synth(
            split=split,
            max_samples=max_samples,
            max_context_windows=max_context_windows,
            max_context_len=max_context_len,
        )

        total_windows = len(windows)
        for wi, (cw_id, questions) in enumerate(windows.items()):
            # Skip already-completed windows
            if cw_id in completed_ids:
                if self.verbose:
                    print(f"\n[Window {wi+1}/{total_windows}] {cw_id} -- SKIPPED (already completed)")
                continue

            if self._cumulative_cost >= self.max_cost_total:
                if self.verbose:
                    print(f"\n[BUDGET EXCEEDED] ${self._cumulative_cost:.2f} >= ${self.max_cost_total:.2f}")
                break

            ctx_len = questions[0].get("context_len", 0) if questions else 0
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"[Window {wi+1}/{total_windows}] {cw_id} | {len(questions)} questions | ~{ctx_len:,} tokens")
                print(f"{'='*60}")

            window_results = self._process_context_window(cw_id, questions, "synth")
            self.results.extend(window_results)

            # Checkpoint after each window
            if checkpoint_path:
                self._save_checkpoint(checkpoint_path)

            # Progress report
            if self.verbose:
                self._print_progress(wi + 1, total_windows)

        summary = OolongBenchmarkSummary.from_results(self.results)
        summary.total_duration = time.time() - self._start_time
        return summary

    def run_real(
        self,
        config: str = "toy_dnd",
        split: str = "test",
        max_samples: Optional[int] = 50,
        max_context_windows: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        resume_path: Optional[str] = None,
    ) -> OolongBenchmarkSummary:
        """Run benchmark on oolong-real dataset."""
        self._start_time = time.time()
        self.results = []

        completed_ids = set()
        if resume_path and os.path.exists(resume_path):
            self.results, completed_ids = self._load_checkpoint(resume_path)
            self._cumulative_cost = sum(r.cost for r in self.results)
            self._cumulative_tokens = sum(r.tokens for r in self.results)
            if self.verbose:
                print(f"Resumed {len(self.results)} results from checkpoint (cost so far: ${self._cumulative_cost:.2f})")
        elif checkpoint_path and os.path.exists(checkpoint_path):
            self.results, completed_ids = self._load_checkpoint(checkpoint_path)
            self._cumulative_cost = sum(r.cost for r in self.results)
            self._cumulative_tokens = sum(r.tokens for r in self.results)
            if self.verbose:
                print(f"Resumed {len(self.results)} results from checkpoint (cost so far: ${self._cumulative_cost:.2f})")

        windows = self.load_real(
            config=config,
            split=split,
            max_samples=max_samples,
            max_context_windows=max_context_windows,
        )

        total_windows = len(windows)
        for wi, (cw_id, questions) in enumerate(windows.items()):
            if cw_id in completed_ids:
                if self.verbose:
                    print(f"\n[Window {wi+1}/{total_windows}] {cw_id} -- SKIPPED (already completed)")
                continue

            if self._cumulative_cost >= self.max_cost_total:
                if self.verbose:
                    print(f"\n[BUDGET EXCEEDED] ${self._cumulative_cost:.2f} >= ${self.max_cost_total:.2f}")
                break

            ctx_len = questions[0].get("context_len", 0) if questions else 0
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"[Window {wi+1}/{total_windows}] {cw_id} | {len(questions)} questions | ~{ctx_len:,} tokens")
                print(f"{'='*60}")

            window_results = self._process_context_window(cw_id, questions, "real")
            self.results.extend(window_results)

            if checkpoint_path:
                self._save_checkpoint(checkpoint_path)

            if self.verbose:
                self._print_progress(wi + 1, total_windows)

        summary = OolongBenchmarkSummary.from_results(self.results)
        summary.total_duration = time.time() - self._start_time
        return summary

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: str):
        """Save current results to a checkpoint file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "results": [r.to_dict() for r in self.results],
            "summary": OolongBenchmarkSummary.from_results(self.results).to_dict(),
        }
        # Write atomically via temp file
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)

    def _load_checkpoint(self, path: str) -> Tuple[List[OolongResult], set]:
        """Load results from checkpoint. Returns (results, completed_window_ids)."""
        with open(path) as f:
            data = json.load(f)

        results = []
        completed_windows = set()
        for rd in data.get("results", []):
            r = OolongResult(
                id=rd.get("id", ""),
                question=rd.get("question", ""),
                expected=rd.get("expected"),
                predicted=rd.get("predicted"),
                score=rd.get("score", 0.0),
                answer_type=rd.get("answer_type", ""),
                task=rd.get("task", ""),
                context_window_id=rd.get("context_window_id", ""),
                context_len=rd.get("context_len", 0),
                cost=rd.get("cost", 0.0),
                tokens=rd.get("tokens", 0),
                duration=rd.get("duration", 0.0),
                error=rd.get("error"),
            )
            results.append(r)
            if r.context_window_id:
                completed_windows.add(r.context_window_id)

        return results, completed_windows

    # ------------------------------------------------------------------
    # Progress / output
    # ------------------------------------------------------------------

    def _print_progress(self, windows_done: int, windows_total: int):
        """Print running progress after a context window completes."""
        scored = [r for r in self.results if r.error is None]
        avg_score = sum(r.score for r in scored) / len(scored) if scored else 0.0
        elapsed = time.time() - self._start_time if self._start_time else 0

        print(f"\n--- Progress: {windows_done}/{windows_total} windows ---")
        print(f"  Questions: {len(self.results)} | Scored: {len(scored)} | Errors: {len(self.results) - len(scored)}")
        print(f"  Avg score: {avg_score:.3f}")
        print(f"  Cost: ${self._cumulative_cost:.2f} / ${self.max_cost_total:.2f}")
        print(f"  Tokens: {self._cumulative_tokens:,}")
        print(f"  Elapsed: {elapsed/60:.1f} min")

        if windows_done > 0 and windows_done < windows_total:
            eta_seconds = (elapsed / windows_done) * (windows_total - windows_done)
            print(f"  ETA: ~{eta_seconds/60:.0f} min")

    def print_summary(self, summary: OolongBenchmarkSummary):
        """Print a formatted summary table."""
        print(f"\n{'='*60}")
        print("OOLONG BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Model: {self.model}")
        print(f"Questions: {summary.num_questions} (scored: {summary.num_scored}, errors: {summary.num_errors})")
        print(f"Overall Score: {summary.overall_score:.3f}")
        print(f"Total Cost: ${summary.total_cost:.2f}")
        print(f"Total Tokens: {summary.total_tokens:,}")
        print(f"Duration: {summary.total_duration/60:.1f} min")

        if summary.score_by_answer_type:
            print(f"\n{'─'*40}")
            print("Score by Answer Type:")
            for atype, score in sorted(summary.score_by_answer_type.items()):
                print(f"  {atype:20s}: {score:.3f}")

        if summary.score_by_task:
            print(f"\n{'─'*40}")
            print("Score by Task:")
            for task, score in sorted(summary.score_by_task.items()):
                print(f"  {task:20s}: {score:.3f}")

        if summary.score_by_context_len:
            print(f"\n{'─'*40}")
            print("Score by Context Length:")
            for bucket, score in summary.score_by_context_len.items():
                print(f"  {bucket:20s}: {score:.3f}")

        print(f"{'='*60}\n")

    def save_results(self, path: str):
        """Export full results to JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "provider": self.provider,
            "config": {
                "max_cost_per_question": self.max_cost_per_question,
                "max_cost_total": self.max_cost_total,
            },
            "summary": OolongBenchmarkSummary.from_results(self.results).to_dict(),
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        if self.verbose:
            print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Oolong Benchmark for RLM-ADK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run profiles:
  Smoke test:           --dataset synth --subset 5 --max-cost 1.0
  Quick benchmark:      --dataset synth --subset 25 --max-cost 5.0
  Moderate (default):   --dataset synth --subset 50 --max-cost 5.0
  Standard validation:  --dataset synth --split validation --max-cost 100.0 --checkpoint ckpt.json
  Full test:            --dataset synth --split test --max-cost 600.0 --checkpoint ckpt.json
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["synth", "real"],
        default="synth",
        help="Dataset to benchmark (default: synth)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split (default: validation)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=50,
        help="Max number of questions to evaluate (default: 50). Use 0 for all.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Max number of context windows to process",
    )
    parser.add_argument(
        "--max-context-len",
        type=int,
        default=None,
        help="Max context length in tokens (synth only)",
    )
    parser.add_argument(
        "--real-config",
        type=str,
        choices=["toy_dnd", "dnd"],
        default="toy_dnd",
        help="Config for real dataset (default: toy_dnd)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="grok-4-1-fast-reasoning",
        help="Model to use (default: grok-4-1-fast-reasoning)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "xai", "anthropic"],
        help="Provider (auto-detected from model if not specified)",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=5.0,
        help="Maximum total cost in USD (default: 5.0)",
    )
    parser.add_argument(
        "--max-cost-per-question",
        type=float,
        default=0.50,
        help="Maximum cost per question in USD (default: 0.50)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file for saving/resuming progress",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Explicitly resume from a checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save full results to JSON file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    load_dotenv()

    # Auto-detect provider
    provider = args.provider
    if not provider:
        if args.model.startswith("grok"):
            provider = "xai"
        elif args.model.startswith("claude"):
            provider = "anthropic"
        else:
            provider = "openai"

    # Get API key
    if provider == "xai":
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            print("ERROR: Please set XAI_API_KEY environment variable")
            sys.exit(1)
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: Please set ANTHROPIC_API_KEY environment variable")
            sys.exit(1)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: Please set OPENAI_API_KEY environment variable")
            sys.exit(1)

    max_samples = args.subset if args.subset > 0 else None

    bench = OolongBenchmark(
        api_key=api_key,
        model=args.model,
        provider=provider,
        verbose=not args.quiet,
        max_cost_per_question=args.max_cost_per_question,
        max_cost_total=args.max_cost,
    )

    if args.dataset == "synth":
        summary = bench.run_synth(
            split=args.split,
            max_samples=max_samples,
            max_context_windows=args.max_windows,
            max_context_len=args.max_context_len,
            checkpoint_path=args.checkpoint,
            resume_path=args.resume,
        )
    else:
        summary = bench.run_real(
            config=args.real_config,
            split=args.split,
            max_samples=max_samples,
            max_context_windows=args.max_windows,
            checkpoint_path=args.checkpoint,
            resume_path=args.resume,
        )

    bench.print_summary(summary)

    # Save results
    if args.output:
        bench.save_results(args.output)
    elif args.checkpoint:
        # Final save to checkpoint
        bench._save_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
