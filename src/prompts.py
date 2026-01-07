"""Prompt templates for code generation evaluation."""

CODE_GENERATION_PROMPT = """You are an expert Python programmer specializing in data transformation tasks.

Given the following ETL transformation task:

**Task:** {task}

**Input Data:**
```json
{input_data}
```

**Expected Output:**
```json
{output_data}
```

Write a Python function called `transform(data)` that takes the input data and produces the expected output. 

Requirements:
- The function MUST be named `transform` and take a single parameter `data`
- Return the transformed result
- Handle edge cases appropriately
- Keep the code clean and efficient
- Only output the Python code, no explanations

Important constraints:
- DO NOT use external services - write direct, programmatic transformation logic
- Use only standard Python libraries (json, re, datetime, etc.)
- The solution must be deterministic and self-contained
- Write the transformation logic directly in Python code

Your response should contain ONLY the Python code for the transform function."""

CODE_GENERATION_PROMPT_WITH_WEB_SEARCH = """You are an expert Python programmer specializing in data transformation tasks involving external API integrations.

# TASK SPECIFICATION

**Task Description:** {task}

**Input Data:**
```json
{input_data}
```

**Expected Output:**
```json
{output_data}
```

# OBJECTIVE

Write a production-ready Python function named `transform(data)` that processes the input data and produces the expected output by making real API calls to external services.

---

# CRITICAL REQUIREMENTS

## Function Signature
- Function MUST be named `transform`
- Must accept exactly one parameter: `data` (dict)
- Must return the transformed result (dict)
- Include proper type hints where appropriate

## API Integration Requirements
This task REQUIRES making ACTUAL API calls to external services. You MUST:

**DO:**
- Make real API calls to the specified services (OpenAI, Anthropic, Cohere, HuggingFace, Voyage AI, Jina AI, etc.)
- Use official client libraries or HTTP requests with correct endpoints
- Handle API responses according to their documented structure
- Include necessary imports at the function level
- Preserve all original fields from input data in the output
- Add only the new fields specified in the expected output

**DO NOT:**
- Create fake, simulated, or deterministic outputs
- Generate embeddings using hashlib, random, numpy.random, or mathematical functions
- Use mock/placeholder values like "[vector of 1536 dimensions]"
- Hardcode responses or use pre-computed values
- Skip API calls and return template data

## CRITICAL: OpenAI API Version Requirement
**If using OpenAI APIs, you MUST use OpenAI Python SDK v1.x (current version).**
- Use `from openai import OpenAI` and `client = OpenAI()`
- Use `client.embeddings.create()` NOT `openai.Embedding.create()` (deprecated)
- Use `client.chat.completions.create()` NOT `openai.Completion.create()` (deprecated)
- Access responses as attributes: `response.data[0].embedding` NOT `response['data'][0]['embedding']`
- Do NOT use `openai.api_key = ...` - use environment variables or `OpenAI(api_key=...)`

---

# WEB SEARCH WORKFLOW

You have access to the `web_search` tool. Follow this systematic approach:

## Step 1: Identify Required APIs
Analyze the task to determine which API(s) are needed (e.g., OpenAI embeddings, Claude summarization, Cohere classification).

## Step 2: Search for Current Documentation
For EACH required API, search using queries like:
- "OpenAI embeddings API Python documentation 2024 v1.x"
- "Anthropic Claude Python SDK messages.create example"
- "Cohere embed API Python current version"
- "HuggingFace sentence-transformers encode method"
- "Voyage AI Python client embeddings"
- "Jina AI embeddings API endpoint 2024"

## Step 3: Verify API Patterns
Search for:
- Correct import statements
- Current model names and identifiers
- Exact method signatures and parameters
- Response structure and data access patterns
- Authentication methods (API keys, headers)

## Step 4: Check for Recent Changes
Look for:
- API version updates or deprecations
- Model name changes (e.g., current Claude or GPT model versions)
- SDK version differences (e.g., Cohere v1 vs v2)
- Breaking changes in response formats

---

# IMPLEMENTATION GUIDELINES

## Code Structure Requirements
- Import statements must be placed inside the function
- Create a copy of input data to avoid mutation
- Validate that required fields exist before processing
- Initialize API clients with appropriate credentials
- Make actual API calls using verified syntax from documentation
- Extract results from API responses correctly
- Add new fields to the result dictionary
- Return the complete transformed result

## Error Handling
Include basic error handling for API calls to prevent crashes on network issues or invalid responses.

## Data Handling Best Practices
1. Preserve input fields: All fields from input should appear in output
2. Add new fields: Only add the new fields specified in expected output
3. Handle missing fields: Check if required fields exist before processing
4. Type conversions: Convert numpy arrays to lists for JSON serialization
5. String formatting: Use f-strings for clarity

---

# QUALITY CHECKLIST

Before finalizing your code, verify:

- Function is named `transform` with correct signature
- All imports are included (inside the function)
- Real API calls are made (no simulation)
- API syntax matches current documentation from web search
- Response data is accessed correctly based on API structure
- All input fields are preserved in output
- New fields match expected output structure
- Code handles edge cases (missing fields, empty strings)
- No hardcoded fake data or mock responses
- Clean, readable code with appropriate comments

---

# OUTPUT FORMAT

**CRITICAL: Output ONLY the Python code for the transform function. Do NOT include:**
- Explanations before or after the code
- Test cases or example usage
- Comments about what you searched for or planning statements
- Installation instructions
- Natural language descriptions of what you will do

**DO NOT output statements like "I'll search for..." or "Now let me...". Output ONLY the actual Python code.**

Begin your response directly with:
```python
def transform(data):
```

End with the function's return statement."""

JUDGE_PROMPT = """You are an expert code reviewer evaluating the quality of generated Python transformation code.

**Task Description:**
{task}

**Input Data:**
```json
{input_data}
```

**Expected Output:**
```json
{expected_output}
```

**Ground Truth Code:**
```python
{ground_truth_code}
```

**Generated Code:**
```python
{generated_code}
```

**Actual Output from Generated Code:**
```json
{actual_output}
```

**Execution Status:** {execution_status}
{error_message}

Please evaluate the generated code on the following dimensions (score each from 0-10):

1. **Correctness**: Does the code produce the correct output? For summary/text generation tasks, evaluate semantic similarity, not exact string matching. For other tasks, check if output structure and values match expected.
2. **Code Quality**: Is the code well-structured, readable, and maintainable?
3. **Efficiency**: Is the solution efficient in terms of time and space complexity?
4. **Robustness**: Does it handle edge cases and potential errors?
5. **Similarity to Ground Truth**: How closely does the approach match the ground truth solution? Alternative approaches (e.g., single API call vs multiple) are acceptable if they produce correct results.

**Critical evaluation guidelines:**

1. **For Summary/Text Generation Tasks**: If the task involves generating summaries, text, or other LLM-produced content:
   - DO NOT require exact string matching
   - Evaluate semantic correctness: Does the output capture the same key information?
   - Consider different phrasings, word choices, and detail levels as valid if semantically equivalent
   - Pass the test if the output is semantically correct, even if wording differs

2. **For API Key/Environment Errors**: If execution failed due to missing API keys (JINA_API_KEY, VOYAGE_API_KEY, COHERE_API_KEY, etc.):
   - Evaluate the CODE STRUCTURE and LOGIC, not execution success
   - Check if the code would work correctly with proper API keys
   - If code structure is correct, give high correctness score
   - Note in feedback that failure was due to missing credentials, not code errors

3. **For Alternative Approaches**: If the code uses a different approach than ground truth (e.g., single API call instead of multiple):
   - Accept it if it produces correct or semantically equivalent results
   - Do not penalize for different implementation patterns
   - Focus on correctness of output, not exact method matching

4. **For Execution Failures**: If the code failed to execute:
   - Analyze WHY it failed (syntax error, logic error, missing dependency, API key issue)
   - If failure is due to missing API keys or dependencies (not code errors), evaluate code structure
   - Provide detailed analysis in error_analysis field

**Important**: If the code failed to execute or produced incorrect output, you MUST provide detailed analysis in the feedback explaining:
- What specific part of the code is incorrect (if any)
- Why the output doesn't match the expected result
- What the code should have done differently
- Line-by-line analysis if the error is subtle

Respond in the following JSON format (must be valid JSON):
{{
  "correctness": <score 0-10>,
  "code_quality": <score 0-10>,
  "efficiency": <score 0-10>,
  "robustness": <score 0-10>,
  "similarity_to_ground_truth": <score 0-10>,
  "overall_score": <average of all scores>,
  "passed": <true if output matches expected semantically (for text tasks) or exactly (for structured tasks), false otherwise>,
  "feedback": "<detailed explanation of the evaluation, with specific analysis of errors for failing cases>",
  "error_analysis": "<for failing cases only: detailed breakdown of where and why the code is incorrect>"
}}

**Critical: Output ONLY the JSON object. Do not include any text before or after the JSON object. Ensure all strings are properly escaped. The JSON must be parseable without errors.**"""

