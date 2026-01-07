"""Code judging functions using Claude Sonnet as judge."""
import json
import re
import asyncio
import logging
from typing import Dict, Any

from utils.clients import get_anthropic_client
from config import JUDGE_MODEL_ID
from prompts import judge_prompt
from generation.code_execution import extract_code_from_response
from tools.web_search_tool import get_anthropic_tool_definition

logger = logging.getLogger(__name__)


async def judge_code_async(task: str, input_data: Any, expected_output: Any, ground_truth_code: str,
                          generated_code: str, actual_output: Any, execution_success: bool, 
                          error_message: str, enable_web_search: bool = False) -> Dict:
    """Evaluate generated code using Claude Sonnet 4.5 as judge asynchronously."""
    execution_status = "Success" if execution_success else "Failed"
    error_msg_section = f"\n**Error Message:**\n{error_message}" if error_message else ""
    
    prompt = judge_prompt.format(
        task=task,
        input_data=json.dumps(input_data, indent=2),
        expected_output=json.dumps(expected_output, indent=2),
        ground_truth_code=ground_truth_code,
        generated_code=extract_code_from_response(generated_code),
        actual_output=json.dumps(actual_output, indent=2) if actual_output is not None else "N/A",
        execution_status=execution_status,
        error_message=error_msg_section
    )
    
    try:
        client = get_anthropic_client()
        loop = asyncio.get_event_loop()
        
        messages = [{"role": "user", "content": prompt}]
        api_params = {
            "model": JUDGE_MODEL_ID,
            "max_tokens": 2000,
            "messages": messages
        }
        
        if enable_web_search:
            tools = [get_anthropic_tool_definition()]
            api_params["tools"] = tools
        
        max_iterations = 3
        response = None
        
        for iteration in range(max_iterations):
            response = await loop.run_in_executor(
                None,
                lambda: client.messages.create(**api_params)
            )
            
            if enable_web_search and response.stop_reason == "tool_use":
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                api_params["messages"] = messages
                continue
            else:
                break
        
        if response and response.content:
            text_blocks = [block.text for block in response.content if hasattr(block, 'text')]
            if text_blocks:
                response_text = text_blocks[0]
            else:
                response_text = str(response.content[0]) if response.content else str(response)
        elif response:
            response_text = str(response)
        else:
            response_text = ""
        
        json_text = None
        
        if "```json" in response_text:
            parts = response_text.split("```json")
            if len(parts) > 1:
                json_text = parts[1].split("```")[0].strip()
        elif "```" in response_text:
            parts = response_text.split("```")
            if len(parts) > 1:
                json_text = parts[1].split("```")[0].strip()
        
        if not json_text:
            first_brace = response_text.find('{')
            last_brace = response_text.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_text = response_text[first_brace:last_brace+1].strip()
        
        if not json_text:
            json_text = response_text.strip()
            if not json_text.startswith('{'):
                first_brace = json_text.find('{')
                if first_brace != -1:
                    json_text = json_text[first_brace:]
            if not json_text.endswith('}'):
                last_brace = json_text.rfind('}')
                if last_brace != -1:
                    json_text = json_text[:last_brace+1]
        
        parsed_result = None
        parse_errors = []
        
        try:
            parsed_result = json.loads(json_text)
        except json.JSONDecodeError as e:
            parse_errors.append(f"Direct parse failed: {str(e)}")
            
            try:
                fixed_text = re.sub(r',\s*}', '}', json_text)
                fixed_text = re.sub(r',\s*]', ']', fixed_text)
                fixed_text = re.sub(r'(?<!\\)"(?=.*":)', '\\"', fixed_text)
                parsed_result = json.loads(fixed_text)
            except json.JSONDecodeError as e2:
                parse_errors.append(f"Fixed parse failed: {str(e2)}")
                
                try:
                    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_text, re.DOTALL)
                    if json_match:
                        parsed_result = json.loads(json_match.group(0))
                except json.JSONDecodeError as e3:
                    parse_errors.append(f"Regex extract failed: {str(e3)}")
        
        if parsed_result:
            required_fields = ["correctness", "code_quality", "efficiency", "robustness", 
                             "similarity_to_ground_truth", "overall_score", "passed", "feedback"]
            missing_fields = [f for f in required_fields if f not in parsed_result]
            if missing_fields:
                logger.warning(f"Judge response missing fields: {missing_fields}, using defaults")
                for field in missing_fields:
                    if field == "passed":
                        parsed_result[field] = False
                    elif field == "feedback":
                        parsed_result[field] = "Judge response incomplete"
                    elif field == "error_analysis":
                        parsed_result[field] = ""
                    else:
                        parsed_result[field] = 0
            
            return parsed_result
        else:
            logger.error(f"Judge JSON parsing failed after all strategies. Errors: {parse_errors}")
            logger.error(f"Response text (first 500 chars): {response_text[:500]}")
            return {
                "correctness": 0, "code_quality": 0, "efficiency": 0,
                "robustness": 0, "similarity_to_ground_truth": 0, "overall_score": 0,
                "passed": False, 
                "feedback": f"Judge evaluation failed: Could not parse JSON response. Errors: {'; '.join(parse_errors)}",
                "error_analysis": f"Judge response was not valid JSON. Response preview: {response_text[:200]}..."
            }
        
    except Exception as e:
        logger.warning(f"Judge evaluation failed: {e}")
        logger.exception("Full traceback:")
        return {
            "correctness": 0, "code_quality": 0, "efficiency": 0,
            "robustness": 0, "similarity_to_ground_truth": 0, "overall_score": 0,
            "passed": False, "feedback": f"Judge evaluation failed: {str(e)}",
            "error_analysis": f"Judge could not complete evaluation: {str(e)}"
        }

