"""Configuration constants and settings for evaluation."""
import os
from dotenv import load_dotenv

load_dotenv()

_script_dir = os.path.dirname(os.path.abspath(__file__))
_base_dir = os.path.dirname(_script_dir)

DATASETS = {
    "etl": os.path.join(_base_dir, "data", "elt_code_eval_dataset.json"),
    "llm_embedding": os.path.join(_base_dir, "data", "etl_llm_embedding_dataset.json")
}

DEFAULT_GENERATOR_MODELS = [
    {"name": "claude-haiku-4.5", "provider": "anthropic", "model_id": "claude-haiku-4-5"},
    {"name": "gemini-flash-3", "provider": "google", "model_id": "gemini-3-flash-preview"},
    {"name": "gpt-4o-mini", "provider": "openai", "model_id": "gpt-4o-mini"}
]

# Global variable that can be modified by command line arguments
GENERATOR_MODELS = DEFAULT_GENERATOR_MODELS.copy()

JUDGE_MODEL = "claude-sonnet-4.5"
JUDGE_MODEL_ID = "claude-sonnet-4-5"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

MAX_CONCURRENT_REQUESTS = 10
BATCH_SIZE = 5

