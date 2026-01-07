"""Dataset-level evaluation functions."""
import json
import os
import asyncio
import random
import logging
from typing import Dict
from datetime import datetime

from config import DATASETS, GENERATOR_MODELS, BATCH_SIZE, JUDGE_MODEL, MAX_CONCURRENT_REQUESTS
from evaluation.test_case import evaluate_test_case_all_models
from utils.results import calculate_summary, print_pass_fail_table, print_scores_table, print_comparative_tables

logger = logging.getLogger(__name__)

_script_dir = os.path.dirname(os.path.abspath(__file__))
_base_dir = os.path.dirname(_script_dir)


async def evaluate_dataset_async(dataset_name: str, dataset_file: str, enable_web_search: bool = False) -> Dict:
    """Evaluate all models on a single dataset with concurrent execution.
    
    Args:
        dataset_name: Name of the dataset
        dataset_file: Path to dataset file
        enable_web_search: Whether to enable web search tool (useful for LLM/API tasks)
    """
    logger.info("="*100)
    logger.info(f"EVALUATING DATASET: {dataset_name.upper()}")
    logger.info(f"File: {dataset_file}")
    logger.info(f"Models: {', '.join([m['name'] for m in GENERATOR_MODELS])}")
    logger.info(f"Web Search Enabled: {enable_web_search}")
    logger.info(f"Concurrency: {MAX_CONCURRENT_REQUESTS} concurrent requests, batch size: {BATCH_SIZE}")
    logger.info("="*100)
    
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} test cases")
    
    results = {
        "dataset_name": dataset_name,
        "dataset_file": dataset_file,
        "test_cases": []
    }
    
    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i:i+BATCH_SIZE]
        batch_start_idx = i + 1
        
        logger.info(f"\n--- Processing batch {i//BATCH_SIZE + 1} (tests {batch_start_idx}-{batch_start_idx+len(batch)-1}) ---")
        
        tasks = [
            evaluate_test_case_all_models(test_case, batch_start_idx + j, enable_web_search)
            for j, test_case in enumerate(batch)
        ]
        
        batch_results = await asyncio.gather(*tasks)
        results["test_cases"].extend(batch_results)
        
        logger.info(f"Batch {i//BATCH_SIZE + 1} completed")
    
    summary = calculate_summary(results)
    results["summary"] = summary
    
    logger.info("\n" + "="*100)
    logger.info(f"SUMMARY - {dataset_name.upper()}")
    logger.info("="*100)
    for model_name, stats in summary.items():
        avg_latency = stats.get('avg_total_latency', 0.0)
        logger.info(f"{model_name:20} | Avg Score: {stats['average_score']:5.2f}/10 | "
                   f"Pass Rate: {stats['pass_rate']:5.1f}% ({stats['passed']}/{stats['total']}) | "
                   f"Avg Latency: {avg_latency:.2f}s")
    
    print_pass_fail_table(results, dataset_name)
    print_scores_table(results, dataset_name, summary)
    
    return results


def evaluate_dataset(dataset_name: str, dataset_file: str, enable_web_search: bool = False) -> Dict:
    """Synchronous wrapper for evaluate_dataset_async."""
    return asyncio.run(evaluate_dataset_async(dataset_name, dataset_file, enable_web_search))


async def evaluate_all_datasets_mixed_async(enable_web_search: bool = True) -> Dict:
    """Evaluate all models on all datasets with mixed test cases and concurrent execution."""
    logger.info("\n" + "="*100)
    logger.info("CODE GENERATION EVALUATION - MIXED DATASETS")
    logger.info(f"Datasets: {', '.join(DATASETS.keys())}")
    logger.info(f"Models: {', '.join([m['name'] for m in GENERATOR_MODELS])}")
    logger.info(f"Judge: {JUDGE_MODEL}")
    logger.info(f"Concurrency: {MAX_CONCURRENT_REQUESTS} concurrent requests, batch size: {BATCH_SIZE}")
    logger.info("="*100)
    
    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "datasets": DATASETS,
            "generator_models": GENERATOR_MODELS,
            "judge_model": JUDGE_MODEL,
            "execution_mode": "mixed_concurrent",
            "web_search_enabled": enable_web_search
        },
        "dataset_results": {}
    }
    
    all_test_cases = []
    
    for dataset_name, dataset_file in DATASETS.items():
        if not os.path.exists(dataset_file):
            logger.warning(f"Dataset file {dataset_file} not found, skipping...")
            continue
        
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded {len(dataset)} test cases from {dataset_name}")
        
        for test_case in dataset:
            test_case["_dataset"] = dataset_name
            all_test_cases.append(test_case)
            
        all_results["dataset_results"][dataset_name] = {
            "dataset_name": dataset_name,
            "dataset_file": dataset_file,
            "test_cases": []
        }
    
    if not all_test_cases:
        logger.error("Error: No test cases loaded")
        return all_results
    
    random.shuffle(all_test_cases)
    logger.info(f"\nTotal test cases (mixed): {len(all_test_cases)}")
    logger.info("="*100)
    
    for i in range(0, len(all_test_cases), BATCH_SIZE):
        batch = all_test_cases[i:i+BATCH_SIZE]
        batch_start_idx = i + 1
        
        logger.info(f"\n--- Processing batch {i//BATCH_SIZE + 1} (tests {batch_start_idx}-{batch_start_idx+len(batch)-1}) ---")
        
        tasks = [
            evaluate_test_case_all_models(
                test_case, 
                batch_start_idx + j,
                enable_web_search=(enable_web_search and test_case.get("_dataset") == "llm_embedding")
            )
            for j, test_case in enumerate(batch)
        ]
        
        batch_results = await asyncio.gather(*tasks)
        
        for result, test_case in zip(batch_results, batch):
            dataset_name = test_case["_dataset"]
            all_results["dataset_results"][dataset_name]["test_cases"].append(result)
        
        logger.info(f"Batch {i//BATCH_SIZE + 1} completed")
    
    for dataset_name in all_results["dataset_results"]:
        if all_results["dataset_results"][dataset_name]["test_cases"]:
            summary = calculate_summary(all_results["dataset_results"][dataset_name])
            all_results["dataset_results"][dataset_name]["summary"] = summary
    
    for dataset_name, dataset_results in all_results["dataset_results"].items():
        if "summary" in dataset_results:
            logger.info("\n" + "="*100)
            logger.info(f"SUMMARY - {dataset_name.upper()}")
            logger.info("="*100)
            for model_name, stats in dataset_results["summary"].items():
                avg_latency = stats.get('avg_total_latency', 0.0)
                logger.info(f"{model_name:20} | Avg Score: {stats['average_score']:5.2f}/10 | "
                           f"Pass Rate: {stats['pass_rate']:5.1f}% ({stats['passed']}/{stats['total']}) | "
                           f"Avg Latency: {avg_latency:.2f}s")
            
            print_pass_fail_table(dataset_results, dataset_name)
            print_scores_table(dataset_results, dataset_name, dataset_results["summary"])
    
    # Comparative Analysis
    comparative_summary = {}
    for model_config in GENERATOR_MODELS:
        model_name = model_config["name"]
        comparative_summary[model_name] = {}
        
        for dataset_name in DATASETS.keys():
            if dataset_name in all_results["dataset_results"]:
                if "summary" in all_results["dataset_results"][dataset_name]:
                    summary = all_results["dataset_results"][dataset_name]["summary"]
                    if model_name in summary:
                        comparative_summary[model_name][dataset_name] = summary[model_name]
    
    all_results["comparative_summary"] = comparative_summary
    
    best_models = {}
    for dataset_name in DATASETS.keys():
        if dataset_name in all_results["dataset_results"]:
            if "summary" in all_results["dataset_results"][dataset_name]:
                summary = all_results["dataset_results"][dataset_name]["summary"]
                if summary:
                    best_model = max(summary.items(), key=lambda x: x[1]["average_score"])
                    best_models[dataset_name] = {
                        "model": best_model[0],
                        "average_score": best_model[1]["average_score"],
                        "pass_rate": best_model[1]["pass_rate"]
                    }
    
    all_results["best_models_per_dataset"] = best_models
    
    overall_scores = {}
    for model_name in comparative_summary:
        scores = [stats["average_score"] for stats in comparative_summary[model_name].values()]
        pass_rates = [stats["pass_rate"] for stats in comparative_summary[model_name].values()]
        latencies = [stats.get("avg_total_latency", 0.0) for stats in comparative_summary[model_name].values()]
        
        overall_avg = sum(scores) / len(scores) if scores else 0
        overall_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0
        overall_avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        overall_scores[model_name] = {
            "overall_average_score": round(overall_avg, 2),
            "overall_pass_rate": round(overall_pass_rate, 2),
            "overall_avg_latency": round(overall_avg_latency, 3)
        }
    
    all_results["overall_performance"] = overall_scores
    
    logger.info("\n" + "="*100)
    logger.info("COMPARATIVE ANALYSIS")
    logger.info("="*100)
    print_comparative_tables(all_results, overall_scores)
    
    if best_models:
        logger.info("\n" + "="*100)
        logger.info("BEST MODELS PER DATASET")
        logger.info("="*100)
        for dataset_name, info in best_models.items():
            logger.info(f"{dataset_name.upper():20} | {info['model']:20} | "
                       f"Score: {info['average_score']:.2f}/10 | Pass Rate: {info['pass_rate']:.1f}%")
    
    logger.info("\n" + "="*100)
    logger.info("OVERALL PERFORMANCE")
    logger.info("="*100)
    for model_name, stats in overall_scores.items():
        avg_latency = stats.get('overall_avg_latency', 0.0)
        logger.info(f"{model_name:20} | Avg Score: {stats['overall_average_score']:.2f}/10 | "
                   f"Avg Pass Rate: {stats['overall_pass_rate']:.1f}% | "
                   f"Avg Latency: {avg_latency:.2f}s")
    
    output_file = os.path.join(_base_dir, "results", f"code_gen_eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    logger.info(f"\nSaving results to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Results saved successfully")
    
    return all_results


def evaluate_all_datasets(enable_web_search: bool = True) -> Dict:
    """Synchronous wrapper for evaluate_all_datasets_mixed_async."""
    return asyncio.run(evaluate_all_datasets_mixed_async(enable_web_search=enable_web_search))

