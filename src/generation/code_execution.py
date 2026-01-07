"""Code execution and extraction utilities."""
import traceback
from typing import Tuple, Any


def extract_code_from_response(response: str) -> str:
    """Extract Python code from markdown code blocks, filtering out planning statements."""
    code = ""
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0].strip()
    elif "```" in response:
        code = response.split("```")[1].split("```")[0].strip()
    else:
        code = response.strip()
    
    lines = code.split('\n')
    filtered_lines = []
    found_code_start = False
    
    for line in lines:
        if any(phrase in line.lower() for phrase in ["i'll", "i will", "now let me", "let me search", 
                                                      "based on my research", "here is the"]):
            if not found_code_start:
                continue
        
        if line.strip().startswith(('def ', 'import ', 'from ', 'class ')) or found_code_start:
            found_code_start = True
            filtered_lines.append(line)
        elif found_code_start:
            filtered_lines.append(line)
        elif line.strip() and not line.strip().startswith('#'):
            found_code_start = True
            filtered_lines.append(line)
    
    if filtered_lines:
        return '\n'.join(filtered_lines).strip()
    
    return code.strip()


def execute_transform_code(code: str, input_data: Any) -> Tuple[bool, Any, str, bool]:
    """Execute the generated transform code. Returns (success, output, error_message, is_api_key_error).
    
    is_api_key_error indicates if failure was due to missing API keys (Jina, Voyage, Cohere),
    which should be evaluated leniently.
    """
    try:
        clean_code = extract_code_from_response(code)
        namespace = {}
        exec(clean_code, namespace)
        
        if "transform" not in namespace:
            return False, None, "Error: 'transform' function not found in generated code", False
        
        result = namespace["transform"](input_data)
        return True, result, "", False
        
    except Exception as e:
        error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
        error_str = str(e).lower()
        
        is_api_key_error = any(keyword in error_str for keyword in [
            "jina_api_key", "voyage_api_key", "cohere_api_key",
            "jinaai_api_key", "voyageai_api_key",
            "401", "unauthorized", "authentication",
            "keyerror", "environment variable",
            "no module named 'voyageai'",
        ])
        
        if "api" in error_str and ("key" in error_str or "token" in error_str):
            if any(provider in error_str for provider in ["jina", "voyage", "cohere"]):
                is_api_key_error = True
        
        return False, None, error_msg, is_api_key_error

