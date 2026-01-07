"""Results processing, calculation, and display functions."""
import time
import logging
from typing import Dict, List

from config import GENERATOR_MODELS

logger = logging.getLogger(__name__)


def calculate_summary(results: Dict) -> Dict:
    """Calculate summary statistics for a dataset, including latency metrics."""
    summary = {}
    
    for model_config in GENERATOR_MODELS:
        model_name = model_config["name"]
        scores = []
        pass_count = 0
        total_count = 0
        latencies = {
            "code_generation": [],
            "execution": [],
            "judging": [],
            "total": []
        }
        
        for test_case in results["test_cases"]:
            for model_result in test_case["model_results"]:
                if model_result["model"] == model_name:
                    total_count += 1
                    eval_data = model_result["evaluation"]
                    scores.append(eval_data.get("overall_score", 0))
                    if eval_data.get("passed", False):
                        pass_count += 1
                    
                    if "latency" in model_result:
                        latencies["code_generation"].append(model_result["latency"]["code_generation"])
                        latencies["execution"].append(model_result["latency"]["execution"])
                        latencies["judging"].append(model_result["latency"]["judging"])
                        latencies["total"].append(model_result["latency"]["total"])
        
        avg_score = sum(scores) / len(scores) if scores else 0
        pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0
        
        avg_latencies = {}
        for key, values in latencies.items():
            avg_latencies[f"avg_{key}_latency"] = round(sum(values) / len(values), 3) if values else 0.0
        
        summary[model_name] = {
            "average_score": round(avg_score, 2),
            "pass_rate": round(pass_rate, 2),
            "passed": pass_count,
            "total": total_count,
            **avg_latencies
        }
    
    return summary


def print_table(headers: List[str], rows: List[List[str]], title: str = None):
    """Print a formatted table."""
    if title:
        print(f"\n{title}")
        print("=" * 100)
    
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_row)
    print("-" * len(header_row))
    
    for row in rows:
        print(" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)))


def print_pass_fail_table(results: Dict, dataset_name: str):
    """Print binary pass/fail table for all test cases."""
    model_names = [m["name"] for m in GENERATOR_MODELS]
    headers = ["Test Case"] + model_names
    rows = []
    
    for test_case in results["test_cases"]:
        task_short = test_case["task"][:45] + "..." if len(test_case["task"]) > 45 else test_case["task"]
        row = [f"#{test_case['test_case_id']}: {task_short}"]
        
        for model_name in model_names:
            for model_result in test_case["model_results"]:
                if model_result["model"] == model_name:
                    passed = model_result["evaluation"].get("passed", False)
                    row.append("PASS" if passed else "FAIL")
                    break
        rows.append(row)
    
    print_table(headers, rows, f"PASS/FAIL MATRIX - {dataset_name.upper()}")


def print_scores_table(results: Dict, dataset_name: str, summary: Dict):
    """Print overall scores table for all test cases."""
    model_names = [m["name"] for m in GENERATOR_MODELS]
    headers = ["Test Case"] + model_names
    rows = []
    
    for test_case in results["test_cases"]:
        task_short = test_case["task"][:45] + "..." if len(test_case["task"]) > 45 else test_case["task"]
        row = [f"#{test_case['test_case_id']}: {task_short}"]
        
        for model_name in model_names:
            for model_result in test_case["model_results"]:
                if model_result["model"] == model_name:
                    score = model_result["evaluation"].get("overall_score", 0)
                    row.append(f"{score:.1f}")
                    break
        rows.append(row)
    
    rows.append(["=" * 50] + ["=" * 15] * len(model_names))
    
    avg_row = ["AVERAGE"]
    for model_name in model_names:
        avg_row.append(f"{summary[model_name]['average_score']:.2f}")
    rows.append(avg_row)
    
    print_table(headers, rows, f"OVERALL SCORES (0-10) - {dataset_name.upper()}")


def print_comparative_tables(all_results: Dict, overall_scores: Dict):
    """Print comparative tables across all datasets."""
    model_names = [m["name"] for m in GENERATOR_MODELS]
    
    # Pass Rate Comparison
    headers = ["Dataset"] + model_names
    rows = []
    
    from config import DATASETS
    for dataset_name in DATASETS.keys():
        if dataset_name in all_results["dataset_results"]:
            summary = all_results["dataset_results"][dataset_name]["summary"]
            row = [dataset_name.upper()]
            for model_name in model_names:
                if model_name in summary:
                    stats = summary[model_name]
                    row.append(f"{stats['pass_rate']:.1f}% ({stats['passed']}/{stats['total']})")
                else:
                    row.append("N/A")
            rows.append(row)
    
    print_table(headers, rows, "PASS RATE COMPARISON")
    
    # Score Comparison
    rows = []
    for dataset_name in DATASETS.keys():
        if dataset_name in all_results["dataset_results"]:
            summary = all_results["dataset_results"][dataset_name]["summary"]
            row = [dataset_name.upper()]
            for model_name in model_names:
                if model_name in summary:
                    row.append(f"{summary[model_name]['average_score']:.2f}")
                else:
                    row.append("N/A")
            rows.append(row)
    
    rows.append(["=" * 20] + ["=" * 15] * len(model_names))
    
    overall_row = ["OVERALL AVERAGE"]
    for model_name in model_names:
        if model_name in overall_scores:
            overall_row.append(f"{overall_scores[model_name]['overall_average_score']:.2f}")
        else:
            overall_row.append("N/A")
    rows.append(overall_row)
    
    print_table(headers, rows, "AVERAGE SCORE COMPARISON (0-10)")

