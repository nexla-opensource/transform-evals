"""
Native web search tool definitions for each provider.
Uses built-in web search capabilities from Anthropic, Google, and OpenAI.
"""


def get_anthropic_tool_definition():
    """Get Anthropic-compatible web search tool definition."""
    return {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5
    }


def get_google_tool_definition():
    """Get Google Gemini-compatible web search tool definition."""
    from google.genai import types
    return types.Tool(
        google_search=types.GoogleSearch()
    )


def get_openai_tool_definition():
    """Get OpenAI-compatible web search tool definition."""
    return {
        "type": "web_search"
    }
