# Project Structure

```
rlm-adk/
│
├── rlm/                          # Main package
│   ├── __init__.py               # Package initialization and exports
│   ├── core.py                   # Enhanced RecursiveLanguageModel
│   ├── core_legacy.py            # Legacy implementation (reference)
│   ├── providers.py              # Multi-provider support (OpenAI, xAI, Anthropic)
│   ├── metrics.py                # Metrics tracking system
│   ├── cache.py                  # LRU caching system
│   ├── security.py               # Sandboxing and security
│   └── helpers.py                # Advanced helper functions
│
├── examples/                     # Usage examples
│   ├── quickstart_anthropic.py   # Minimal Anthropic Claude example
│   ├── quickstart_grok.py        # Minimal Grok example
│   ├── quickstart_gpt5.py        # Minimal GPT-5 example
│   ├── basic_usage.py            # Needle-in-haystack pattern
│   ├── classification_example.py # Classification & aggregation
│   ├── verification_example.py   # Verification pattern
│   ├── long_output_example.py    # Long output generation
│   ├── advanced_patterns.py      # Map-reduce pattern
│   ├── grok_basic_example.py     # Grok integration example
│   ├── grok_reasoning_example.py # Grok reasoning metrics
│   └── multi_provider_example.py # Cross-provider comparison
│
├── tests/                        # Test suite
│   ├── test_anthropic_provider.py       # Anthropic provider unit tests (mocked)
│   ├── test_attention_paper_anthropic.py # PDF paper analysis integration test
│   ├── test_huberman_demo_anthropic.py  # Anthropic transcript demo
│   ├── test_huberman_demo.py            # Grok transcript demo
│   ├── test_rlm_comprehensive.py        # Long-context integration tests
│   ├── test_retry_streaming_sandbox.py  # Retry, streaming, and sandbox tests
│   ├── test_helpers.py                  # Helper function tests
│   ├── test_cache.py                    # Caching tests
│   ├── test_metrics.py                  # Metrics tests
│   ├── test_mock.py                     # Mock/stub tests
│   ├── test_data_generator.py           # Test data generation
│   ├── Attention_is_all_you_need.pdf    # Test PDF (Transformer paper)
│   ├── huberman_transcript.txt          # Test transcript data
│   ├── huberman_rlm_results.json        # Grok baseline results
│   └── huberman_rlm_results_anthropic.json # Anthropic baseline results
│
├── main.py                       # CLI entry point
├── Makefile                      # Development shortcuts (uv-based)
├── pyproject.toml                # Package metadata
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment configuration template
├── README.md                     # Main documentation
├── PROJECT_STRUCTURE.md          # This file
└── .gitignore                    # Git ignore rules
```

## Component Descriptions

### Core Package (`rlm/`)

#### `core.py` (~992 lines)
Main RLM implementation:
- `RecursiveLanguageModel` class
- **Direct mode**: Single-call fast-path when context fits in window
- **REPL mode**: Agentic loop with code generation and sub-calls
- Three-tier model support (orchestrator / smart sub-model / fast sub-model)
- REPL environment management with helper globals
- Iteration loop with budget controls
- Code execution and `FINAL()` answer handling

#### `providers.py` (~870 lines)
Multi-provider abstraction:
- `LLMProvider` abstract base class
- `AnthropicProvider` - Claude models (Opus 4.6, Sonnet 4.5, Haiku 4.5, 200k context)
- `OpenAIProvider` - GPT models (GPT-4o, GPT-5)
- `XAIProvider` - Grok models (Grok-4, 128k context)
- `create_provider()` factory function
- Per-provider context windows, pricing, retry logic
- Streaming and non-streaming completions
- Cached token tracking (Anthropic prompt caching)

#### `metrics.py` (~310 lines)
Comprehensive metrics tracking:
- `RLMMetrics` class
- `CallMetrics` dataclass
- Token counting (prompt, completion, cached, reasoning)
- Cost calculation per model using provider pricing
- Recursion depth tracking
- Budget limit enforcement
- JSON export

#### `cache.py` (~154 lines)
LRU caching for sub-calls:
- `RLMCache` class
- Hash-based lookup by `(prompt, model)` key
- TTL support
- Hit/miss tracking
- Size management and statistics

#### `security.py` (~421 lines)
Sandboxing for safe code execution:
- `RestrictedGlobals` - Safe globals dictionary
- `ExecutionMonitor` - Code safety checks
- `SafeFileAccess` - Restricted file operations
- Blocks dangerous imports, file writes, system calls

#### `helpers.py` (~714 lines)
Advanced helper utilities available in the REPL:
- `TextProcessor` - Chunking (char/token-based), splitting, truncation
- `SearchHelper` - Regex search, keyword filter, section extraction
- `AggregationHelper` - join, sum, count, list, dict strategies
- `VerificationHelper` - Answer validation, consensus checking
- `RecursionHelper` - Recursive split, map-reduce, BFS decomposition

### Examples (`examples/`)

Each example demonstrates a specific RLM pattern:

1. **quickstart_anthropic.py**: Three-tier Claude setup (Sonnet + Haiku)
2. **quickstart_grok.py**: Minimal Grok example
3. **quickstart_gpt5.py**: Minimal GPT-5 example
4. **basic_usage.py**: Needle-in-haystack with filtering + probing
5. **classification_example.py**: Chunking + classification + aggregation
6. **verification_example.py**: Extraction + verification loop
7. **long_output_example.py**: Section extraction + summarization
8. **advanced_patterns.py**: Map-reduce with sentiment analysis
9. **multi_provider_example.py**: Cross-provider comparison

### Tests (`tests/`)

#### Unit Tests (mocked, no API calls)
- **test_anthropic_provider.py** (~629 lines): Anthropic provider config, chat completion, system message handling, multi-model routing, pricing, direct mode, history management
- **test_helpers.py** (~173 lines): 30+ tests for TextProcessor, SearchHelper, AggregationHelper
- **test_cache.py** (~117 lines): LRU eviction, TTL, stats
- **test_metrics.py** (~168 lines): Token/cost tracking
- **test_mock.py** (~225 lines): Mock/stub functionality
- **test_retry_streaming_sandbox.py** (~641 lines): Retry logic, streaming, security sandbox
- **test_data_generator.py** (~659 lines): Test data generation utilities

#### Integration Tests (require API keys)
- **test_attention_paper_anthropic.py**: Processes the "Attention Is All You Need" PDF with Anthropic Claude in agentic REPL mode
- **test_huberman_demo_anthropic.py**: Huberman transcript analysis with Anthropic Claude
- **test_huberman_demo.py**: Huberman transcript analysis with Grok
- **test_rlm_comprehensive.py** (~730 lines): Needle-in-haystack (256k), multi-needle, reasoning, summarization, edge cases

Run unit tests: `pytest tests/test_helpers.py tests/test_cache.py tests/test_metrics.py tests/test_mock.py tests/test_anthropic_provider.py`

Run integration tests: `ANTHROPIC_API_KEY=... python3 tests/test_attention_paper_anthropic.py`

## File Statistics

```
Lines of Code:
  - rlm/ package:  ~3,900 lines
  - examples/:     ~500 lines
  - tests/:        ~3,800 lines
  - main.py:       ~16,500 lines
  - Total:         ~24,700 lines
```

## Dependencies

### Required
- `openai>=1.0.0` - OpenAI API client
- `anthropic>=0.79.0` - Anthropic API client
- `xai-sdk>=1.5.0` - xAI API client
- `tiktoken>=0.5.0` - Token counting
- `python-dotenv>=1.0.0` - Environment variable loading

### Optional (Dev)
- `pytest>=7.0.0` - Testing
- `PyPDF2>=3.0.0` - PDF text extraction (for PDF integration tests)

## Key Design Decisions

1. **Multi-Provider Architecture**: Abstract `LLMProvider` base class with OpenAI, xAI, and Anthropic implementations
2. **Dual Processing Modes**: Direct mode (small context, single call) and REPL mode (large context, agentic decomposition)
3. **Three-Tier Model Strategy**: Different models for orchestration, smart sub-tasks, and fast sub-tasks
4. **Modular Components**: Separate concerns (core, providers, metrics, cache, security, helpers)
5. **Safety**: Sandboxing enabled by default with restricted globals and code pattern checking
6. **Observability**: Full metrics tracking with per-model cost breakdown and JSON export
7. **Caching**: LRU cache with TTL to avoid redundant sub-calls
8. **Budget Controls**: Max cost and max token limits with automatic enforcement
