"""Client initialization for different LLM providers."""
import anthropic
import google.genai as genai
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

from config import ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY


def get_anthropic_client():
    """Get Anthropic client instance."""
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def get_openai_client():
    """Get OpenAI client instance."""
    return OpenAI(api_key=OPENAI_API_KEY)


def get_google_client():
    """Get Google GenAI client instance."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    try:
        return genai.Client(api_key=GOOGLE_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Google GenAI client: {e}")
        raise

