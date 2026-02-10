"""
Demonstration of RLM processing on Huberman Lab podcast transcript
using Anthropic Claude with the new direct-mode optimization.

Uses a three-tier model setup:
  - Sonnet 4.5 as orchestrator
  - Sonnet 4.5 for smart sub-tasks
  - Haiku 4.5 for simple sub-tasks

Run with: ANTHROPIC_API_KEY=your-key python3 tests/test_huberman_demo_anthropic.py
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from rlm import RecursiveLanguageModel

load_dotenv()

SCRIPT_DIR = Path(__file__).parent

# Read the transcript
transcript_path = SCRIPT_DIR / "huberman_transcript.txt"
with open(transcript_path, "r") as f:
    transcript = f.read()

print("=" * 70)
print("RLM HUBERMAN TRANSCRIPT ANALYSIS — ANTHROPIC CLAUDE")
print("=" * 70)
print()
print(f"Transcript length: {len(transcript):,} characters")
print(f"Word count:        {len(transcript.split()):,} words")
print(f"Estimated tokens:  ~{len(transcript) // 4:,}")
print()

task = """Extract and summarize:
1) The main concepts about the nervous system discussed in this podcast
2) Any practical tools or protocols mentioned that listeners can apply
3) Key scientific insights about neuroplasticity and learning"""

print("TASK:")
print("-" * 70)
print(task)
print()

# Check API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("ERROR: ANTHROPIC_API_KEY not set.")
    print("Run: export ANTHROPIC_API_KEY='your-key-here'")
    exit(1)

print("=" * 70)
print("RUNNING RLM WITH ANTHROPIC (Sonnet 4.5 + Haiku 4.5)")
print("=" * 70)
print()

try:
    rlm = RecursiveLanguageModel(
        api_key=api_key,
        model="claude-sonnet-4-5-20250929",
        simple_model="claude-haiku-4-5-20251001",
        provider="anthropic",
        max_cost=5.0,
        enable_cache=True,
    )

    start = time.time()
    result = rlm.run(
        task=task,
        context=transcript,
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

    # Compare with old Grok results if available
    old_results_path = SCRIPT_DIR / "huberman_rlm_results.json"
    if old_results_path.exists():
        with open(old_results_path) as f:
            old = json.load(f)
        old_metrics = old["metrics"]

        print("\n" + "=" * 70)
        print("COMPARISON: Old (Grok REPL) vs New (Anthropic Direct)")
        print("=" * 70)
        print(f"  {'Metric':<25} {'Old (Grok)':>15} {'New (Anthropic)':>15} {'Reduction':>12}")
        print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")

        new_tokens = rlm.metrics.total_tokens
        new_cost = rlm.metrics.total_cost
        new_iters = rlm.metrics.iterations
        new_subcalls = rlm.metrics.sub_calls

        old_tokens = old_metrics["total_tokens"]
        old_cost = old_metrics["total_cost"]
        old_iters = old_metrics["iterations"]
        old_subcalls = old_metrics["sub_calls"]

        def ratio(old_v, new_v):
            if new_v == 0:
                return "N/A"
            return f"{old_v / new_v:.1f}x"

        print(f"  {'Tokens':<25} {old_tokens:>15,} {new_tokens:>15,} {ratio(old_tokens, new_tokens):>12}")
        print(f"  {'Cost ($)':<25} {old_cost:>15.4f} {new_cost:>15.4f} {ratio(old_cost, new_cost):>12}")
        print(f"  {'Iterations':<25} {old_iters:>15} {new_iters:>15} {ratio(old_iters, new_iters):>12}")
        print(f"  {'Sub-calls':<25} {old_subcalls:>15} {new_subcalls:>15} {'—':>12}")
        print()

    # Save results
    output_data = {
        "task": task,
        "result": result,
        "model": "claude-sonnet-4-5-20250929",
        "mode": "direct" if rlm.metrics.iterations == 1 else "repl",
        "wall_time_seconds": round(elapsed, 2),
        "metrics": {
            "total_cost": rlm.metrics.total_cost,
            "total_tokens": rlm.metrics.total_tokens,
            "prompt_tokens": rlm.metrics.total_prompt_tokens,
            "completion_tokens": rlm.metrics.total_completion_tokens,
            "sub_calls": rlm.metrics.sub_calls,
            "iterations": rlm.metrics.iterations,
        },
    }
    output_path = SCRIPT_DIR / "huberman_rlm_results_anthropic.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to: {output_path}")

except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    print(traceback.format_exc())
    rlm.print_metrics()
