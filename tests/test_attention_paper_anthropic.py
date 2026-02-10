"""
Test RLM agentic (REPL) processing on the "Attention Is All You Need" PDF paper
using Anthropic Claude.

Forces REPL mode by setting context_window=1 so the orchestrator decomposes the
task into sub-calls via llm_query / llm_query_fast instead of answering directly.

Uses a three-tier model setup:
  - Sonnet 4.5 as orchestrator (writes Python code to decompose the task)
  - Sonnet 4.5 for smart sub-tasks (via llm_query)
  - Haiku 4.5 for simple sub-tasks (via llm_query_fast)

Run with: ANTHROPIC_API_KEY=your-key python3 tests/test_attention_paper_anthropic.py
"""

import os
import json
import time
from pathlib import Path
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from rlm import RecursiveLanguageModel

load_dotenv()

SCRIPT_DIR = Path(__file__).parent

# ---------- Load PDF ----------
pdf_path = SCRIPT_DIR / "Attention_is_all_you_need.pdf"
if not pdf_path.exists():
    print(f"ERROR: PDF not found at {pdf_path}")
    exit(1)

reader = PdfReader(str(pdf_path))
pages_text = [page.extract_text() or "" for page in reader.pages]
paper_text = "\n\n".join(pages_text)

print("=" * 70)
print("RLM ATTENTION PAPER ANALYSIS — ANTHROPIC CLAUDE (AGENTIC MODE)")
print("=" * 70)
print()
print(f"PDF pages:         {len(reader.pages)}")
print(f"Extracted length:  {len(paper_text):,} characters")
print(f"Word count:        {len(paper_text.split()):,} words")
print(f"Estimated tokens:  ~{len(paper_text) // 4:,}")
print()

# ---------- Task ----------
task = """Analyze this research paper and provide:
1) A concise summary of the paper's main contribution (the Transformer architecture)
2) The key components of the architecture (multi-head attention, positional encoding, etc.)
3) The main results and benchmarks reported (BLEU scores, training cost comparisons)
4) Why this paper was significant for the field of deep learning and NLP"""

print("TASK:")
print("-" * 70)
print(task)
print()

# ---------- API key ----------
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("ERROR: ANTHROPIC_API_KEY not set.")
    print("Run: export ANTHROPIC_API_KEY='your-key-here'")
    exit(1)

print("=" * 70)
print("RUNNING RLM WITH ANTHROPIC — AGENTIC/REPL MODE")
print("(Sonnet 4.5 orchestrator + Haiku 4.5 sub-tasks)")
print("=" * 70)
print()

try:
    rlm = RecursiveLanguageModel(
        api_key=api_key,
        model="claude-sonnet-4-5-20250929",
        simple_model="claude-haiku-4-5-20251001",
        provider="anthropic",
        context_window=1,           # Force REPL mode (bypass direct mode)
        max_cost=5.0,
        enable_cache=True,
    )

    start = time.time()
    result = rlm.run(
        task=task,
        context=paper_text,
        verbose=True,
        max_iterations=50,
    )
    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print("RLM RESULT:")
    print("=" * 70)
    print(result)
    print("=" * 70)

    # Print metrics
    rlm.print_metrics()

    print(f"\nWall time: {elapsed:.2f}s")

    # ---------- Save results ----------
    output_data = {
        "task": task,
        "result": result,
        "model": "claude-sonnet-4-5-20250929",
        "mode": "direct" if rlm.metrics.iterations == 1 else "repl",
        "wall_time_seconds": round(elapsed, 2),
        "pdf_pages": len(reader.pages),
        "context_chars": len(paper_text),
        "metrics": {
            "total_cost": rlm.metrics.total_cost,
            "total_tokens": rlm.metrics.total_tokens,
            "prompt_tokens": rlm.metrics.total_prompt_tokens,
            "completion_tokens": rlm.metrics.total_completion_tokens,
            "sub_calls": rlm.metrics.sub_calls,
            "iterations": rlm.metrics.iterations,
        },
    }
    output_path = SCRIPT_DIR / "attention_rlm_results_anthropic_repl.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    print(traceback.format_exc())
    rlm.print_metrics()
