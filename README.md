# Transform Evals

A comprehensive evaluation framework for testing code generation capabilities of LLMs on data transformation tasks.

## Overview

This project evaluates how well different LLM models can generate Python code for data transformation tasks. It supports multiple LLM providers (Anthropic, Google, OpenAI) and includes two main datasets:

- **ETL Code Tasks**: Standard data transformation tasks
- **LLM Embedding Tasks**: Tasks involving API calls to embedding and LLM services

## Features

- **Multi-Provider Support**: Evaluate models from Anthropic, Google, and OpenAI
- **Web Search Integration**: Optional web search tool for API documentation lookup
- **Comprehensive Evaluation**: Uses Claude Sonnet as a judge to evaluate code quality, correctness, efficiency, and robustness
- **Concurrent Execution**: Parallel evaluation of multiple models and test cases
- **Detailed Results**: Generates pass/fail matrices, score tables, and latency metrics

## Setup

### Prerequisites

- Python 3.8+
- API keys for at least one LLM provider (Anthropic, Google, OpenAI)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd transform-evals
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp env.example .env
# Edit .env and add your API keys
```

Required environment variables:
- `ANTHROPIC_API_KEY` - Required for judge model and Anthropic generator models
- `GOOGLE_API_KEY` - Required for Google generator models
- `OPENAI_API_KEY` - Required for OpenAI generator models

Optional (for specific test cases):
- `JINA_API_KEY` - For Jina AI embedding tasks
- `VOYAGE_API_KEY` - For Voyage AI embedding tasks
- `COHERE_API_KEY` - For Cohere API tasks

## Usage

### Basic Usage

Run evaluations on all datasets with default models:

```bash
python src/evaluate_transforms.py
```

### Running Specific Datasets

Run only the ETL dataset:
```bash
python src/evaluate_transforms.py --dataset etl
```

Run only the LLM embedding dataset:
```bash
python src/evaluate_transforms.py --dataset llm_embedding
```

Run multiple datasets:
```bash
python src/evaluate_transforms.py --dataset etl --dataset llm_embedding
```

### Using Custom Models

Specify custom models for each provider:

```bash
# Use a specific OpenAI model
python src/evaluate_transforms.py --openai-model gpt-5.1-codex-mini --dataset etl

# Use a specific Google model
python src/evaluate_transforms.py --google-model gemini-3-flash-preview --dataset llm_embedding

# Use a specific Anthropic model
python src/evaluate_transforms.py --anthropic-model claude-haiku-4-5 --dataset etl

# Combine multiple models
python src/evaluate_transforms.py --openai-model gpt-5.1-codex-mini --google-model gemini-3-flash-preview --dataset etl --dataset llm_embedding
```

### Web Search Configuration

Enable web search (default):
```bash
python src/evaluate_transforms.py --web-search yes
```

Disable web search:
```bash
python src/evaluate_transforms.py --web-search no
```

Or use the legacy flag:
```bash
python src/evaluate_transforms.py --no-web-search
```

**Note**: Web search is automatically enabled for `llm_embedding` dataset tasks that require API documentation lookup, even if disabled globally.

### Example Commands

```bash
# Run both datasets with no web search for a specific OpenAI model
python src/evaluate_transforms.py --openai-model gpt-5.1-codex-mini --dataset etl --dataset llm_embedding --web-search no

# Run llm_embedding dataset with web search for a specific OpenAI model
python src/evaluate_transforms.py --openai-model gpt-5.2-codex-mini --dataset llm_embedding --web-search yes

# Run with all three providers using custom models
python src/evaluate_transforms.py \
  --openai-model gpt-5.1-codex-mini \
  --google-model gemini-3-flash-preview \
  --anthropic-model claude-haiku-4-5 \
  --dataset etl \
  --dataset llm_embedding
```

## Project Structure

```
transform-evals/
├── src/
│   ├── config.py                    # Configuration constants
│   ├── prompts.py                    # Prompt templates
│   ├── evaluate_transforms.py        # Main entry point
│   ├── tools/                        # Tools package
│   │   ├── __init__.py
│   │   └── web_search_tool.py        # Web search tool definitions
│   ├── evaluation/                   # Evaluation modules
│   │   ├── __init__.py
│   │   ├── judge.py                  # Code judging functions (using Claude Sonnet)
│   │   ├── test_case.py               # Test case evaluation logic
│   │   └── dataset.py                 # Dataset-level evaluation
│   ├── generation/                   # Code generation modules
│   │   ├── __init__.py
│   │   ├── code_generation.py        # Code generation functions
│   │   └── code_execution.py         # Code execution utilities
│   └── utils/                        # Utility modules
│       ├── __init__.py
│       ├── clients.py                 # LLM client initialization
│       └── results.py                 # Results processing and display
├── data/
│   ├── elt_code_eval_dataset.json      # ETL dataset
│   └── etl_llm_embedding_dataset.json  # LLM embedding dataset
├── results/                            # Evaluation results (JSON files)
├── env.example                         # Environment variables template
└── README.md                           # This file
```

## Evaluation Process

1. **Code Generation**: Each model generates Python code for a given transformation task
2. **Code Execution**: The generated code is executed with test input data
3. **Judging**: Claude Sonnet evaluates the code on multiple dimensions:
   - Correctness (0-10)
   - Code Quality (0-10)
   - Efficiency (0-10)
   - Robustness (0-10)
   - Similarity to Ground Truth (0-10)
   - Overall Score (average of all scores)
4. **Results**: Results are saved as JSON files and displayed in tables

## Output

Results are saved in the `results/` directory with timestamps:

- Individual dataset results: `code_gen_eval_{dataset_name}_{timestamp}.json`
- Combined results: `code_gen_eval_combined_{timestamp}.json`

Each result file includes:
- Metadata (models used, timestamp, configuration)
- Test case results with generated code, execution results, and evaluations
- Summary statistics (average scores, pass rates, latency metrics)

## Evaluation Metrics

- **Pass Rate**: Percentage of test cases that pass evaluation
- **Average Score**: Mean overall score across all test cases
- **Latency Metrics**: 
  - Code generation latency
  - Execution latency
  - Judging latency
  - Total latency

## Notes

- The judge model (Claude Sonnet) requires an Anthropic API key
- Web search is particularly useful for LLM/API embedding tasks that require current API documentation
- Some test cases may require additional API keys (Jina, Voyage, Cohere) - these are optional and failures due to missing keys are evaluated leniently
- The evaluation framework supports concurrent execution for faster evaluation

## Troubleshooting

### Missing API Keys

If you see errors about missing API keys, ensure your `.env` file is properly configured with all required keys.

### Rate Limiting

If you encounter rate limit errors, the framework includes automatic retry logic with exponential backoff for Google API calls.

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## License

See LICENSE file for details.
