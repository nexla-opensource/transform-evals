"""Main entry point for code generation evaluation."""
import os
import sys
import argparse
import logging
import json
from datetime import datetime

from config import (
    DATASETS, DEFAULT_GENERATOR_MODELS, JUDGE_MODEL,
    ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY
)
from evaluation.dataset import evaluate_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

_script_dir = os.path.dirname(os.path.abspath(__file__))
_base_dir = os.path.dirname(_script_dir)

# Global variable for generator models (can be modified by command line args)
GENERATOR_MODELS = DEFAULT_GENERATOR_MODELS.copy()


def build_generator_models(google_model=None, openai_model=None, anthropic_model=None):
    """Build GENERATOR_MODELS list based on provided model names.
    
    Args:
        google_model: Model name for Google (e.g., "gemini-3-flash-preview")
        openai_model: Model name for OpenAI (e.g., "gpt-5.1-codex-mini")
        anthropic_model: Model name for Anthropic (e.g., "claude-haiku-4-5")
    
    Returns:
        List of model config dictionaries
    """
    models = []
    
    if google_model:
        models.append({
            "name": google_model,
            "provider": "google",
            "model_id": google_model
        })
    
    if openai_model:
        models.append({
            "name": openai_model,
            "provider": "openai",
            "model_id": openai_model
        })
    
    if anthropic_model:
        models.append({
            "name": anthropic_model,
            "provider": "anthropic",
            "model_id": anthropic_model
        })
    
    if not models:
        return DEFAULT_GENERATOR_MODELS.copy()
    
    return models


def main():
    """Main entry point."""
    missing_keys = []
    if not ANTHROPIC_API_KEY:
        missing_keys.append("ANTHROPIC_API_KEY")
    if not GOOGLE_API_KEY:
        missing_keys.append("GOOGLE_API_KEY")
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    
    if missing_keys:
        logger.error("Error: Missing API keys:")
        for key in missing_keys:
            logger.error(f"  - {key}")
        logger.error("\nPlease set these as environment variables.")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="Code generation evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Run both datasets with no web search for gpt-5.1-codex-mini
  python {sys.argv[0]} --openai-model gpt-5.1-codex-mini --dataset etl --dataset llm_embedding --web-search no

  # Run llm_embedding dataset with web search for gpt-5.2-codex-mini
  python {sys.argv[0]} --openai-model gpt-5.2-codex-mini --dataset llm_embedding --web-search yes

Available datasets: {', '.join(DATASETS.keys())}
        """
    )
    
    parser.add_argument(
        "--google-model",
        type=str,
        help="Custom model name for Google provider (e.g., gemini-3-flash-preview)"
    )
    
    parser.add_argument(
        "--openai-model",
        type=str,
        help="Custom model name for OpenAI provider (e.g., gpt-5.1-codex-mini)"
    )
    
    parser.add_argument(
        "--anthropic-model",
        type=str,
        help="Custom model name for Anthropic provider (e.g., claude-haiku-4-5)"
    )
    
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(DATASETS.keys()),
        help="Dataset(s) to evaluate (can be specified multiple times). If not specified, all datasets are used."
    )
    
    parser.add_argument(
        "--web-search",
        type=str,
        choices=["yes", "no", "true", "false", "y", "n"],
        default="yes",
        help="Enable web search tool (yes/no). Default: yes"
    )
    
    # Support legacy arguments for backward compatibility
    parser.add_argument(
        "--no-web-search",
        action="store_true",
        help="Disable web search (legacy flag, use --web-search no instead)"
    )
    
    args = parser.parse_args()
    
    global GENERATOR_MODELS
    GENERATOR_MODELS = build_generator_models(
        google_model=args.google_model,
        openai_model=args.openai_model,
        anthropic_model=args.anthropic_model
    )
    
    # Update config module's GENERATOR_MODELS so other modules can use it
    import config
    config.GENERATOR_MODELS = GENERATOR_MODELS
    
    if args.no_web_search:
        enable_web_search = False
    else:
        enable_web_search = args.web_search.lower() in ["yes", "true", "y"]
    
    datasets_to_run = args.dataset if args.dataset else list(DATASETS.keys())
    
    logger.info("="*100)
    logger.info("CONFIGURATION")
    logger.info("="*100)
    logger.info(f"Models: {[m['name'] for m in GENERATOR_MODELS]}")
    logger.info(f"Datasets: {datasets_to_run}")
    logger.info(f"Web Search: {enable_web_search}")
    logger.info("="*100)
    
    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "generator_models": GENERATOR_MODELS,
            "judge_model": JUDGE_MODEL,
            "web_search_enabled": enable_web_search
        },
        "dataset_results": {}
    }
    
    for dataset_name in datasets_to_run:
        dataset_file = DATASETS[dataset_name]
        if not os.path.exists(dataset_file):
            logger.warning(f"Dataset file '{dataset_file}' not found, skipping {dataset_name}")
            continue
        
        dataset_web_search = enable_web_search and (dataset_name == "llm_embedding")
        
        logger.info(f"\n{'='*100}")
        logger.info(f"Evaluating dataset: {dataset_name.upper()}")
        logger.info(f"{'='*100}")
        
        results = evaluate_dataset(dataset_name, dataset_file, enable_web_search=dataset_web_search)
        
        all_results["dataset_results"][dataset_name] = {
            "dataset_name": dataset_name,
            "dataset_file": dataset_file,
            "web_search_enabled": dataset_web_search,
            "results": results
        }
        
        output_file = os.path.join(_base_dir, "results", f"code_gen_eval_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        logger.info(f"\nSaving results to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "dataset_name": dataset_name,
                    "dataset_file": dataset_file,
                    "generator_models": GENERATOR_MODELS,
                    "judge_model": JUDGE_MODEL,
                    "web_search_enabled": dataset_web_search
                },
                "results": results
            }, f, indent=2, default=str)
        logger.info("Results saved successfully")
    
    if len(datasets_to_run) > 1:
        combined_output_file = os.path.join(_base_dir, "results", f"code_gen_eval_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        logger.info(f"\nSaving combined results to {combined_output_file}")
        os.makedirs(os.path.dirname(combined_output_file), exist_ok=True)
        with open(combined_output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Combined results saved successfully")


if __name__ == "__main__":
    main()
