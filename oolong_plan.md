# Plan: Run Oolong Benchmark on RLM-ADK with Grok 4.1

## Context

The [Oolong benchmark](https://arxiv.org/abs/2511.02817) evaluates long-context **aggregation reasoning** -- tasks where models must classify, count, and compare across large volumes of text. No frontier model exceeds 50% accuracy at 128K context. This is a perfect fit for RLM-ADK's recursive decomposition approach, which processes long contexts via a REPL loop with sub-calls rather than loading everything into the neural context window.

We'll create a test script + Colab notebook to evaluate RLM with `grok-4-1-fast-reasoning` on both Oolong datasets.

---

## Deliverables

### 1. `tests/test_oolong.py` -- Benchmark test script

**Key classes:**

- **`OolongScorer`** -- Scoring functions matching the official Oolong evaluation:
  - `score_numeric(gold, pred)` -- partial credit: `0.75^|gold-pred|`
  - `score_exact_match(gold, pred)` -- case-insensitive exact match for labels/dates/comparisons
  - `score_set_overlap(gold_list, pred_list)` -- `len(intersection)/len(gold)` for list answers
  - `parse_synth_answer(raw, answer_type)` -- extract clean answer from RLM output (handles `\boxed{}`, "Answer:", numeric extraction, etc.)
  - `parse_real_answer(raw)` -- detect type (int/str/list) and parse accordingly

- **`OolongResult`** dataclass -- per-question result (id, question, expected, predicted, score, answer_type, cost, tokens, duration, error)

- **`OolongBenchmarkSummary`** dataclass -- aggregated results (overall_score, score_by_answer_type, score_by_task, score_by_context_len, total_cost, total_tokens)

- **`OolongBenchmark`** -- Main runner:
  - `load_synth(split, max_samples, max_context_windows, max_context_len)` -- loads from HuggingFace with streaming, groups by `context_window_id`
  - `load_real(config, split, max_samples, max_context_windows)` -- loads `toy_dnd` or `dnd` config
  - `_process_context_window(cw_id, questions, dataset_type)` -- processes all questions sharing a context, one RLM instance per window
  - `run_synth()` / `run_real()` -- entry points
  - Budget enforcement: per-question cap + cumulative cap, stops early if exceeded
  - **Checkpoint/resume support:** saves results after each context window to allow resuming interrupted runs

**Context reuse strategy:** Questions grouped by `context_window_id` (25 questions per window in synth). One `RecursiveLanguageModel` instance per window. Between questions, call `rlm.reset()` but save/restore `rlm.cache` to preserve sub-call cache (since `reset()` calls `cache.clear()` -- see `rlm/core.py:980-992`).

**Run profiles (CLI presets):**
```bash
# Smoke test (~$0.25, 5 min)
python -m tests.test_oolong --dataset synth --subset 5 --max-cost 1.0

# Quick benchmark (~$2, 20 min)
python -m tests.test_oolong --dataset synth --subset 25 --max-cost 5.0

# Moderate benchmark (~$5, 30 min) -- DEFAULT
python -m tests.test_oolong --dataset synth --subset 50 --max-cost 5.0

# Standard validation run (~$35-65, 8-22 hours) -- FULL VALIDATION SPLIT
python -m tests.test_oolong --dataset synth --split validation --max-cost 100.0 --checkpoint results_checkpoint.json

# Full test run (~$275-530, multi-day)
python -m tests.test_oolong --dataset synth --split test --max-cost 600.0 --checkpoint results_checkpoint.json
```

**Standard validation run features:**
- Processes ALL 1,300 questions across ALL context windows in the validation split
- `--checkpoint <file>` flag: saves progress after each context window; if the run is interrupted, re-running with the same checkpoint file resumes from where it left off
- `--resume <file>` flag: explicitly resume from a previous checkpoint
- Intermediate progress logging: prints running average score, cost, and ETA after each context window
- Context length stratification: processes windows sorted by context length (shortest first) so early results are fast and you get signal quickly
- Results saved incrementally to JSON, so partial runs still produce usable data

### 2. `notebooks/oolong_colab.ipynb` -- Colab notebook

**Cells:**
1. **Install + Clone** -- `pip install xai-sdk tiktoken python-dotenv datasets` + `git clone https://github.com/fabiopauli/rlm-adk.git`
2. **API Key** -- Load `XAI_API_KEY` from Colab secrets via `google.colab.userdata`
3. **Configuration** -- Editable params with preset profiles:
   - `PROFILE = "moderate"` (default: 50 questions, $5)
   - Can switch to `"quick"` (25q, $2), `"standard_validation"` (1300q, $100), or `"custom"`
4. **Load Data** -- Stream from HuggingFace with `streaming=True` (avoids downloading 24GB), group by context_window_id
5. **Run Benchmark** -- Import and run `OolongBenchmark` from `tests/test_oolong`
6. **Results** -- Print summary table: overall score, score by answer type, score by task, cost
7. **Save** -- Export JSON + `files.download()`
8. **Debug cell** -- Run a single question with verbose output for inspection
9. **Standard validation run cell** -- Dedicated cell for the full validation run with checkpoint support and Google Drive save for persistence across Colab reconnects

### 3. `pyproject.toml` update

Add `datasets>=2.0.0` to dependencies.

---

## Scoring Summary

| Answer Type | Method | Formula |
|---|---|---|
| NUMERIC | Partial credit | `0.75^abs(gold-pred)` |
| LABEL | Exact match (case-insensitive) | 1.0 or 0.0 |
| COMPARISON | Exact match | 1.0 or 0.0 |
| USER / DATE | Exact match | 1.0 or 0.0 |
| Integer (real) | Partial credit | `0.75^abs(gold-pred)` |
| String (real) | Exact match | 1.0 or 0.0 |
| List (real) | Set overlap | `len(intersection)/len(gold)` |

---

## Cost Estimates (Grok 4.1: $0.0002/$0.0005 per 1K input/output)

| Tier | Questions | Est. Cost | Est. Time |
|---|---|---|---|
| Smoke test | 5 | ~$0.25 | ~5 min |
| Quick benchmark | 25 | ~$1-2 | ~15-25 min |
| Moderate (default) | 50 | ~$2-5 | ~30 min |
| **Standard validation** | **1,300** | **~$35-65** | **~8-22 hours** |
| Full test | 10,600 | ~$275-530 | ~3-7 days |

Default Colab config: **Moderate benchmark (50 questions on oolong-synth, ~$5, ~30 min)**.
Standard validation: **Full oolong-synth validation split (1,300 questions, ~$50, ~12 hours)** with checkpoint/resume.

---

## Files to Create/Modify

| File | Action |
|---|---|
| `tests/test_oolong.py` | **Create** -- benchmark script (~500 lines) |
| `notebooks/oolong_colab.ipynb` | **Create** -- Colab notebook (9 cells) |
| `pyproject.toml` | **Edit** -- add `datasets>=2.0.0` dependency |

**Existing files reused (not modified):**
- `rlm/core.py` -- `RecursiveLanguageModel.run()`, `.reset()`, `.print_metrics()`
- `rlm/providers.py` -- `XAIProvider` with `grok-4-1-fast-reasoning` support
- `rlm/metrics.py` -- `RLMMetrics` for cost/token tracking
- `rlm/cache.py` -- `RLMCache` for sub-call caching
- `tests/test_rlm_comprehensive.py` -- pattern reference for `TestResult`, CLI, JSON export

---

## Verification

1. **Unit test scoring:** Run scorer tests (exact match, numeric partial credit, set overlap, answer parsing)
2. **Smoke test:** `python -m tests.test_oolong --dataset synth --subset 5 --max-cost 1.0` -- verify 5 questions complete with scores
3. **Colab test:** Upload notebook, set XAI_API_KEY in Colab secrets, run all cells with moderate profile (50 questions)
4. **Check results JSON:** Verify output has scores broken down by answer_type and task
5. **Checkpoint test:** Start a run with `--checkpoint`, interrupt it (Ctrl+C), resume with same checkpoint file, verify it continues from where it stopped
