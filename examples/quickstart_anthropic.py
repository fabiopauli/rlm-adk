"""
Quick Start Example with Anthropic Claude - Multi-model strategy

Uses a three-tier model setup:
  - Opus 4.6 as orchestrator (generates code, plans decomposition)
  - Sonnet 4.5 for smart sub-tasks (analysis, reasoning, classification)
  - Haiku for simple sub-tasks (extraction, yes/no, short answers)

Run this after setting your ANTHROPIC_API_KEY environment variable:
    export ANTHROPIC_API_KEY="your-key-here"
    python quickstart_anthropic.py
"""

import os
from dotenv import load_dotenv
from rlm import RecursiveLanguageModel


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Please set ANTHROPIC_API_KEY environment variable")
        print("Example: export ANTHROPIC_API_KEY='your-key-here'")
        return

    # Sample context with mixed complexity tasks
    context = """
    Document 1: The weather today is sunny with a high of 75°F.
    Document 2: Python was created by Guido van Rossum and first released in 1991.
    Document 3: The secret activation code is ALPHA-7749-BRAVO.
    Document 4: Machine learning is a subset of artificial intelligence that enables
    systems to learn and improve from experience without being explicitly programmed.
    It focuses on the development of computer programs that can access data and use
    it to learn for themselves.
    Document 5: Claude is an AI assistant made by Anthropic.
    """

    # Create RLM with Anthropic model setup
    # Sonnet 4.5 as orchestrator, Haiku for simple sub-tasks
    rlm = RecursiveLanguageModel(
        api_key=api_key,
        model="claude-sonnet-4-5-20250929",          # Orchestrator
        simple_model="claude-haiku-4-5-20251001",    # Simple sub-tasks
        provider="anthropic",
        max_cost=5.0
    )

    # Run task
    try:
        result = rlm.run(
            task="Find the secret activation code in the documents",
            context=context
        )

        # Print result
        print("\n" + "=" * 60)
        print("RESULT:", result)
        print("=" * 60)

        # Show metrics with per-model breakdown
        rlm.print_metrics()

    except Exception as e:
        print(f"\nError: {str(e)}")
        rlm.print_metrics()


if __name__ == "__main__":
    main()
