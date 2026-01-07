"""Code generation functions for different LLM providers."""
import json
import asyncio
import re
import logging
from typing import Dict, Any

from utils.clients import get_anthropic_client, get_openai_client, get_google_client
from config import GOOGLE_API_KEY
from prompts import code_generation_prompt, code_generation_prompt_with_web_search
from tools.web_search_tool import (
    get_anthropic_tool_definition,
    get_openai_tool_definition,
    get_google_tool_definition
)
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


async def generate_code_async(task: str, input_data: Any, output_data: Any, model_config: Dict) -> str:
    """Generate code using the specified model asynchronously."""
    prompt = code_generation_prompt.format(
        task=task,
        input_data=json.dumps(input_data, indent=2),
        output_data=json.dumps(output_data, indent=2)
    )
    
    provider = model_config["provider"]
    model_id = model_config["model_id"]
    
    try:
        if provider == "anthropic":
            client = get_anthropic_client()
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.messages.create(
                    model=model_id,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            return response.content[0].text
        
        elif provider == "google":
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is not set. Please set the GOOGLE_API_KEY environment variable.")
            
            client = get_google_client()
            loop = asyncio.get_event_loop()
            
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    response = await loop.run_in_executor(
                        None,
                        lambda: client.models.generate_content(
                            model=model_id,
                            contents=[prompt]
                        )
                    )
                    break
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                        if attempt < max_retries - 1:
                            retry_match = re.search(r'retry.*?(\d+(?:\.\d+)?)s', error_str, re.IGNORECASE)
                            if retry_match:
                                retry_delay = float(retry_match.group(1)) + 1
                            else:
                                retry_delay = min(retry_delay * 2, 60)
                            
                            logger.warning(f"Rate limit exceeded (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay:.1f}s...")
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            logger.error(f"Rate limit exceeded after {max_retries} attempts. Please check your quota: https://ai.dev/usage?tab=rate-limit")
                            raise ValueError(f"Google GenAI rate limit exceeded. {error_str}")
                    else:
                        raise
            
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                if hasattr(response.candidates[0], 'content'):
                    if hasattr(response.candidates[0].content, 'parts'):
                        return response.candidates[0].content.parts[0].text
                    elif hasattr(response.candidates[0].content, 'text'):
                        return response.candidates[0].content.text
            elif hasattr(response, 'content'):
                if hasattr(response.content, 'parts'):
                    return response.content.parts[0].text
                elif hasattr(response.content, 'text'):
                    return response.content.text
            
            raise ValueError(f"Unexpected response format from Google GenAI: {type(response)}, attributes: {dir(response)}")
        
        elif provider == "openai":
            client = get_openai_client()
            loop = asyncio.get_event_loop()
            
            use_responses_api = "codex" in model_id.lower() or "gpt-5" in model_id.lower()
            
            if use_responses_api:
                try:
                    response = await loop.run_in_executor(
                        None,
                        lambda: client.responses.create(
                            model=model_id,
                            input=prompt
                        )
                    )
                    if hasattr(response, 'output_text'):
                        return response.output_text
                    elif hasattr(response, 'output'):
                        return response.output
                    else:
                        return str(response)
                except Exception as e:
                    logger.warning(f"Responses API failed for {model_id}, trying chat.completions: {e}")
            
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000
                )
            )
            return response.choices[0].message.content
        
        raise ValueError(f"Unknown provider: {provider}")
    except Exception as e:
        logger.error(f"Error generating code with {provider}/{model_id}: {str(e)}")
        raise


async def generate_code_with_tools_async(
    task: str, 
    input_data: Any, 
    output_data: Any, 
    model_config: Dict,
    enable_web_search: bool = False
) -> str:
    """
    Generate code using function calling with optional web search tool.
    
    Args:
        task: Task description
        input_data: Input data example
        output_data: Expected output data
        model_config: Model configuration dict
        enable_web_search: Whether to enable web search tool
    
    Returns:
        Generated code string
    """
    if not enable_web_search:
        # Fall back to regular generation
        return await generate_code_async(task, input_data, output_data, model_config)
    
    prompt = code_generation_prompt_with_web_search.format(
        task=task,
        input_data=json.dumps(input_data, indent=2),
        output_data=json.dumps(output_data, indent=2)
    )
    
    provider = model_config["provider"]
    model_id = model_config["model_id"]
    
    messages = [{"role": "user", "content": prompt}]
    max_iterations = 5  # Prevent infinite loops
    
    try:
        for iteration in range(max_iterations):
            if provider == "anthropic":
                # Anthropic handles web search automatically - just pass the tool definition
                tools = [get_anthropic_tool_definition()]
                client = get_anthropic_client()
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.messages.create(
                        model=model_id,
                        max_tokens=2000,
                        messages=messages,
                        tools=tools
                    )
                )
                
                if response.stop_reason == "tool_use":
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    messages.append({
                        "role": "user",
                        "content": "Now that you have the API documentation, please provide ONLY the Python code for the transform function. Do not include any explanations or planning statements. Begin directly with 'def transform(data):'"
                    })
                    continue
                else:
                    if response.content:
                        text_blocks = [block.text for block in response.content if hasattr(block, 'text')]
                        if text_blocks:
                            code_text = text_blocks[0]
                            if code_text.strip().startswith(('def transform', 'import ', 'from ')) or 'def transform' in code_text:
                                return code_text
                            elif any(phrase in code_text.lower() for phrase in ["i'll", "i will", "now let me", "let me search"]):
                                logger.warning(f"Got planning text instead of code, prompting for actual code (iteration {iteration + 1})")
                                messages.append({
                                    "role": "assistant",
                                    "content": response.content
                                })
                                messages.append({
                                    "role": "user",
                                    "content": "Please provide ONLY the Python code for the transform function. Do not include explanations or planning statements. Begin directly with 'def transform(data):'"
                                })
                                continue
                            else:
                                return code_text
                        return str(response.content[0]) if response.content else ""
                    return ""
            
            elif provider == "openai":
                client = get_openai_client()
                loop = asyncio.get_event_loop()
                
                input_text = prompt
                if len(messages) > 1:
                    input_text = "\n\n".join([
                        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                        for msg in messages
                    ])
                
                tools = [get_openai_tool_definition()]
                
                try:
                    response = await loop.run_in_executor(
                        None,
                        lambda: client.responses.create(
                            model=model_id,
                            tools=tools,
                            input=input_text
                        )
                    )
                    
                    if hasattr(response, 'output_text'):
                        return response.output_text
                    elif hasattr(response, 'output'):
                        return response.output
                    else:
                        return str(response)
                except AttributeError:
                    logger.warning("OpenAI responses API not available, falling back to chat.completions")
                    response = await loop.run_in_executor(
                        None,
                        lambda: client.chat.completions.create(
                            model=model_id,
                            messages=messages,
                            max_tokens=2000
                        )
                    )
                    return response.choices[0].message.content
            
            elif provider == "google":
                client = get_google_client()
                loop = asyncio.get_event_loop()
                
                grounding_tool = get_google_tool_definition()
                
                gemini_contents = []
                for msg in messages:
                    if msg["role"] == "user":
                        gemini_contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
                    elif msg["role"] == "assistant":
                        gemini_contents.append({"role": "model", "parts": [{"text": msg["content"]}]})
                
                config = genai_types.GenerateContentConfig(
                    tools=[grounding_tool]
                )
                
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=model_id,
                        contents=gemini_contents,
                        config=config
                    )
                )
                
                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        text_parts = [p.text for p in candidate.content.parts if hasattr(p, 'text')]
                        if text_parts:
                            return "\n".join(text_parts)
                
                return str(response)
        
        logger.warning(f"Reached max iterations ({max_iterations}) for tool calling")
        if messages:
            return messages[-1].get("content", "")
        return ""
    
    except Exception as e:
        logger.error(f"Error generating code with tools ({provider}/{model_id}): {str(e)}")
        # Fall back to regular generation
        return await generate_code_async(task, input_data, output_data, model_config)

