"""Test case evaluation functions."""
import time
import asyncio
import logging
from typing import Dict

from config import GENERATOR_MODELS
from generation.code_generation import generate_code_async, generate_code_with_tools_async
from generation.code_execution import execute_transform_code
from evaluation.judge import judge_code_async

logger = logging.getLogger(__name__)


async def evaluate_model_on_test_case(
    test_case: Dict, 
    model_config: Dict, 
    test_case_id: int,
    enable_web_search: bool = False
) -> Dict:
    """Evaluate a single model on a single test case asynchronously.
    
    Args:
        test_case: Test case dictionary
        model_config: Model configuration
        test_case_id: Test case ID
        enable_web_search: Whether to enable web search tool for API lookups
    
    Returns:
        Dictionary containing evaluation results with latency metrics
    """
    model_name = model_config["name"]
    start_time = time.perf_counter()
    
    try:
        logger.info(f"Test {test_case_id} - {model_name}: Generating code...")
        
        task_lower = test_case["task"].lower()
        
        is_api_task = any(keyword in task_lower for keyword in [
            "openai", "cohere", "huggingface", "sentence_transformers", 
            "anthropic claude", "voyage ai", "jina ai",
            "gpt-4", "claude", "gemini", "embedding model", "embedding api"
        ])
        
        use_web_search = enable_web_search and is_api_task
        
        if use_web_search:
            logger.info(f"Test {test_case_id} - {model_name}: Using web search tool for API lookup")
        elif enable_web_search and not is_api_task:
            logger.debug(f"Test {test_case_id} - {model_name}: Web search enabled but task doesn't require APIs, using regular prompt")
        
        code_gen_start = time.perf_counter()
        if use_web_search:
            generated_code = await generate_code_with_tools_async(
                test_case["task"],
                test_case["input"],
                test_case["output"],
                model_config,
                enable_web_search=True
            )
        else:
            generated_code = await generate_code_async(
                test_case["task"],
                test_case["input"],
                test_case["output"],
                model_config
            )
        code_gen_latency = time.perf_counter() - code_gen_start
        
        logger.info(f"Test {test_case_id} - {model_name}: Executing code...")
        
        execution_start = time.perf_counter()
        success, output, error_msg, is_api_key_error = execute_transform_code(generated_code, test_case["input"])
        execution_latency = time.perf_counter() - execution_start
        
        task_lower = test_case["task"].lower()
        is_summary_task = any(keyword in task_lower for keyword in [
            "summary", "summarize", "generate summary", "text generation"
        ])
        
        logger.info(f"Test {test_case_id} - {model_name}: Judging code...")
        
        is_api_task = any(keyword in task_lower for keyword in [
            "openai", "cohere", "huggingface", "anthropic", "embedding", 
            "gpt-4", "claude", "gemini", "api", "sentence_transformers"
        ])
        judge_web_search = enable_web_search and is_api_task
        
        judge_context = ""
        if is_api_key_error:
            judge_context = "\n\n**IMPORTANT**: Execution failed due to missing API keys (Jina/Voyage/Cohere). Evaluate the CODE STRUCTURE and LOGIC, not execution success. If the code structure is correct and would work with proper API keys, give high correctness score."
        if is_summary_task:
            judge_context += "\n\n**IMPORTANT**: This is a summary/text generation task. Evaluate semantic similarity, NOT exact string matching. Different phrasings that capture the same information should be considered correct."
        
        judge_start = time.perf_counter()
        evaluation = await judge_code_async(
            test_case["task"] + judge_context,
            test_case["input"],
            test_case["output"],
            test_case.get("ground_truth_code", ""),
            generated_code,
            output,
            success,
            error_msg,
            enable_web_search=judge_web_search
        )
        judge_latency = time.perf_counter() - judge_start
        
        total_latency = time.perf_counter() - start_time
        
        status = "PASS" if evaluation.get("passed") else "FAIL"
        score = evaluation.get("overall_score", 0)
        
        logger.info(f"Test {test_case_id} - {model_name}: {status} (score: {score:.1f}, latency: {total_latency:.2f}s)")
        
        return {
            "model": model_name,
            "test_case_id": test_case_id,
            "generated_code": generated_code,
            "execution_success": success,
            "actual_output": output,
            "error_message": error_msg,
            "evaluation": evaluation,
            "status": status,
            "score": score,
            "latency": {
                "code_generation": round(code_gen_latency, 3),
                "execution": round(execution_latency, 3),
                "judging": round(judge_latency, 3),
                "total": round(total_latency, 3)
            }
        }
        
    except Exception as e:
        total_latency = time.perf_counter() - start_time
        logger.error(f"Test {test_case_id} - {model_name}: ERROR - {str(e)}")
        return {
            "model": model_name,
            "test_case_id": test_case_id,
            "generated_code": "",
            "execution_success": False,
            "actual_output": None,
            "error_message": str(e),
            "evaluation": {
                "correctness": 0, "code_quality": 0, "efficiency": 0,
                "robustness": 0, "similarity_to_ground_truth": 0,
                "overall_score": 0, "passed": False,
                "feedback": f"Model testing failed: {str(e)}",
                "error_analysis": f"Exception during evaluation: {str(e)}"
            },
            "status": "ERROR",
            "score": 0,
            "latency": {
                "code_generation": 0.0,
                "execution": 0.0,
                "judging": 0.0,
                "total": round(total_latency, 3)
            }
        }


async def evaluate_test_case_all_models(
    test_case: Dict, 
    test_case_id: int,
    enable_web_search: bool = False
) -> Dict:
    """Evaluate all models on a single test case concurrently.
    
    Args:
        test_case: Test case dictionary
        test_case_id: Test case ID
        enable_web_search: Whether to enable web search tool
    """
    task_short = test_case['task'][:70] + "..." if len(test_case['task']) > 70 else test_case['task']
    logger.info(f"[Test {test_case_id}] Starting evaluation: {task_short}")
    
    tasks = [
        evaluate_model_on_test_case(test_case, model_config, test_case_id, enable_web_search)
        for model_config in GENERATOR_MODELS
    ]
    
    model_results = await asyncio.gather(*tasks)
    
    logger.info(f"[Test {test_case_id}] Completed:")
    for result in model_results:
        latency = result.get('latency', {}).get('total', 0.0)
        logger.info(f"  {result['model']:20} {result['status']:6} (score: {result['score']:.1f}, latency: {latency:.2f}s)")
    
    return {
        "test_case_id": test_case_id,
        "task": test_case["task"],
        "input": test_case["input"],
        "expected_output": test_case["output"],
        "ground_truth_code": test_case.get("ground_truth_code", ""),
        "model_results": [{k: v for k, v in r.items() if k not in ['test_case_id', 'status', 'score']} 
                         for r in model_results]
    }

