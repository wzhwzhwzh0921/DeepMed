#!/usr/bin/env python3
"""LLM-based medical answer extraction and evaluation with model pool support."""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import signal
from typing import Any, Dict, List, Optional, Tuple

# Fix tmux environment issues for volcengine SDK
os.environ['NO_PROXY'] = os.environ.get('NO_PROXY', '') + ',.volces.com,volces.com'
if 'TMUX' in os.environ:
    # Clear problematic proxy settings in tmux
    for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
        if proxy_var in os.environ:
            del os.environ[proxy_var]

# Add utils path - Replace with your utils directory path
# sys.path.append('/path/to/your/workspace')
# from utils.api_wrapper import APIClient

# If no api_wrapper, use simple requests implementation
import requests

class APIClient:
    """Simple API client implementation, replaces original api_wrapper"""
    def __init__(self, models, api_type="openai"):
        self.models = models if isinstance(models, list) else [models]
        self.api_type = api_type
        self.current_model_idx = 0
        # Configure your API info here
        self.api_base = "add api base here for LLM judger"
        self.api_key = "add api key here for LLM judger"

    def __call__(self, messages, retry_time=2, sleep_duration=1.0, temperature=0.2, thinking=None):
        model = self.models[self.current_model_idx % len(self.models)]
        self.current_model_idx += 1

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }

        for attempt in range(retry_time + 1):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt < retry_time:
                    import time
                    time.sleep(sleep_duration)
                else:
                    raise e
        return None


class LLMEvaluator:
    """LLM-based evaluator for medical questions using GPT-style endpoints."""

    def __init__(self, model_name: str = "gpt-4o-2024-11-20", timeout: int = 30, enable_thinking: bool = False):
        self.model_name = model_name
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        # Decide api type based on model name
        if any(tag in model_name.lower() for tag in ["gpt", "chatgpt", "o1"]):
            try:
                self.client = APIClient([model_name], api_type="chatgpt")
                self.api_type = "chatgpt"
            except Exception:
                print(f"Warning: ChatGPT API failed for {model_name}, falling back to default model")
                self.client = APIClient(["add endpoint here for fallback model"], api_type="openai")
                self.api_type = "openai"
                self.model_name = "fallback-model"
        else:
            # Use standard OpenAI-compatible API
            self.client = APIClient([model_name], api_type="openai")
            self.api_type = "openai"
        print(f"Initialized LLM Evaluator with model: {self.model_name} (API: {self.api_type}, Thinking: {self.enable_thinking})")

    def _call_with_timeout(self, messages, timeout=None):
        """Call API with timeout control"""
        if timeout is None:
            timeout = self.timeout
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            if self.enable_thinking:
                future = executor.submit(self.client, messages, retry_time=2, sleep_duration=1.0, temperature=0.2, thinking="enabled")
            else:
                future = executor.submit(self.client, messages, retry_time=2, sleep_duration=1.0, temperature=0.2)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                print(f"Warning: API call timed out after {timeout} seconds")
                return None
            except Exception as e:
                print(f"Error in API call: {e}")
                return None

    def extract_answer_from_tools(self, record: Dict[str, Any]) -> Optional[str]:
        """Extract answer from finish tool call if available."""
        try:
            # Look for conversation_history in the record
            conversation_history = record.get("conversation_history", [])
            if not conversation_history:
                return None
                
            # Find the last assistant message with tool_calls
            for message in reversed(conversation_history):
                if message.get("role") == "assistant" and "tool_calls" in message:
                    tool_calls = message["tool_calls"]
                    # Look for finish tool call
                    for tool_call in tool_calls:
                        if (tool_call.get("type") == "function" and 
                            tool_call.get("function", {}).get("name") == "finish"):
                            # Extract answer from arguments
                            arguments = tool_call["function"].get("arguments", "{}")
                            if isinstance(arguments, str):
                                args_data = json.loads(arguments)
                            else:
                                args_data = arguments
                            return args_data.get("answer", "")
            return None
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return None

    def extract_answer_from_tags(self, prediction: str) -> Optional[str]:
        """Extract answer from <answer></answer> tags."""
        # Look for <answer></answer> tags
        pattern = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
        match = pattern.search(prediction)
        if match:
            return match.group(1).strip()
        return None

    def extract_answer(self, prediction: str, question: str, input_filename: str = "", record: Optional[Dict[str, Any]] = None) -> str:
        if not prediction or not prediction.strip():
            return ""

        # Determine extraction method based on filename
        is_tools_model = "_tools" in input_filename.lower()
        
        if is_tools_model:
            # Try to extract from finish tool call
            extracted_answer = self.extract_answer_from_tools(record) if record else None
            if extracted_answer:
                answer = extracted_answer.strip()
                if answer and answer.upper() != "NO_ANSWER":
                    return answer
            
            # If tool extraction failed, return empty for tools models
            return ""
        else:
            # Try to extract from <answer></answer> tags first
            extracted_answer = self.extract_answer_from_tags(prediction)
            if extracted_answer:
                answer = extracted_answer.strip()
                if answer and answer.upper() != "NO_ANSWER":
                    return answer

        # Fallback to LLM-based extraction for non-tools models
        if not is_tools_model:
            extract_prompt = """You are an expert at extracting final answers from model predictions.

                            Given a model's response to a question, extract ONLY the final answer. The answer should be:
                            - Concise and direct
                            - Remove any reasoning, explanation, or additional text
                            - For multiple choice questions, return only the option letter (A, B, C, D, etc.)
                            - For numerical answers, return only the number
                            - For short answers, return only the essential answer phrase

                            If no clear answer can be found, return "NO_ANSWER".
                            Question:
                            {question}
                            Model Prediction:
                            {prediction}

                            Final Answer:"""

            messages = [
                {"role": "user", "content": extract_prompt.format(prediction=prediction,question=question)}
            ]

            try:
                response = self._call_with_timeout(messages)
                if response:
                    return response.strip()
                else:
                    print(f"Warning: No response from LLM for answer extraction")
                    return ""
            except Exception as e:
                print(f"Error extracting answer: {e}")
                return ""
        
        return ""

    def judge_correctness(self, model_answer: str, reference_answer: str, question: str = "") -> bool:
        if not model_answer or not reference_answer:
            return False

        judge_prompt = """You are an expert medical knowledge evaluator. Your task is to determine if a model's answer is correct.
                Guidelines:
                Don't try to solve the problems; your task is to determine whether the model's answer matches the correct answer.
                1. Focus on the semantic meaning, not exact wording
                2. Consider medical synonyms and equivalent terms
                3. For multiple choice questions, exact letter match is required
                4. For numerical answers, consider reasonable approximations
                5. Ignore formatting differences
                6. Consider partial credit for partially correct answers

                Question (for context):
                {question}

                Reference Answer (Ground Truth):
                {reference_answer}

                Model Answer:
                {model_answer}

                Is the model answer correct? Respond with ONLY "True" or "False"."""

        messages = [
            {"role": "user", "content": judge_prompt.format(
                question=question,
                reference_answer=reference_answer,
                model_answer=model_answer
            )}
        ]

        try:
            response = self._call_with_timeout(messages)
            if response:
                return response.strip().upper() == "TRUE"
            else:
                print(f"Warning: No response from LLM for correctness judgment")
                return False
        except Exception as e:
            print(f"Error judging correctness: {e}")
            return False


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: str, mode: str = "w") -> None:
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_records(path: str) -> Tuple[List[Dict[str, Any]], Dict[Any, Dict[str, Any]]]:
    if not os.path.exists(path):
        return [], {}

    existing = load_jsonl(path)
    index = {}
    for item in existing:
        record_id = item.get("id")
        if record_id is not None:
            index[record_id] = item
    return existing, index


def process_single_record(record: Dict[str, Any], evaluator: LLMEvaluator, input_filename: str = "") -> Dict[str, Any]:
    start_time = time.time()
    record_id = record.get("id", "unknown")
    
    enriched = dict(record)
    enriched["judger_model"] = evaluator.model_name

    if enriched.get("end_flag") is False:
        enriched["extracted_answer"] = ""
        enriched["is_correct"] = False
        elapsed_time = time.time() - start_time
        print(f"Record {record_id}: EP={evaluator.model_name}, Time={elapsed_time:.2f}s, Status=SKIPPED (end_flag=False)")
        return enriched

    prediction = enriched.get("prediction", "")
    reference_answer = enriched.get("answer", "")
    question = enriched.get("question", "")
    
    # Extract answer
    extract_start = time.time()
    extracted_answer = evaluator.extract_answer(prediction, question, input_filename, record)
    extract_time = time.time() - extract_start
    enriched["extracted_answer"] = extracted_answer

    # Judge correctness
    judge_start = time.time()
    if extracted_answer and reference_answer:
        is_correct = evaluator.judge_correctness(extracted_answer, reference_answer, question)
        enriched["is_correct"] = is_correct
    else:
        is_correct = False
        enriched["is_correct"] = False
    judge_time = time.time() - judge_start

    total_time = time.time() - start_time
    print(f"Record {record_id}: EP={evaluator.model_name}, Time={total_time:.2f}s (extract={extract_time:.2f}s, judge={judge_time:.2f}s), Result={is_correct}")
    
    return enriched


def process_records_parallel(records: List[Dict[str, Any]], evaluators: List[LLMEvaluator], max_workers: int = 10, input_filename: str = "") -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not records:
        return results
    pool_size = len(evaluators)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_record = {}
        for idx, record in enumerate(records):
            evaluator = evaluators[idx % pool_size]
            future = executor.submit(process_single_record, record, evaluator, input_filename)
            future_to_record[future] = record

        for future in as_completed(future_to_record):
            try:
                result = future.result()
                results.append(result)
                if len(results) % 10 == 0:
                    print(f"Processed {len(results)}/{len(records)} records...")
            except Exception as e:
                record = future_to_record[future]
                print(f"Error processing record {record.get('id', 'unknown')}: {e}")
                enriched = dict(record)
                enriched["extracted_answer"] = ""
                enriched["is_correct"] = False
                results.append(enriched)

    return results


def parse_model_list(raw: Optional[str], fallback: str) -> List[str]:
    if raw is None or not raw.strip():
        return [fallback]
    models = [item.strip() for item in raw.split(",") if item.strip()]
    return models if models else [fallback]


def main():
    parser = argparse.ArgumentParser(description="LLM-based medical answer evaluation")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--model",
        default="add endpoint here for default judger model",
        help="Default model name/endpoint when --models is not provided"
    )
    parser.add_argument("--models", default=None, help="Comma separated list of model names to rotate")
    parser.add_argument("--max-workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds for each API call")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")

    args = parser.parse_args()

    model_names = parse_model_list(args.models, args.model)

    print(f"=== LLM Medical Answer Evaluation ===")
    print(f"Model pool: {', '.join(model_names)}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max workers: {args.max_workers}")
    print(f"Timeout: {args.timeout}s")
    print(f"Resume: {args.resume}")
    print("=" * 50)

    evaluators = [LLMEvaluator(name, timeout=args.timeout) for name in model_names]

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from input file")

    existing_records, existing_index = load_existing_records(args.output) if args.resume else ([], {})

    if args.resume:
        print(f"Resume enabled: {len(existing_records)} records already processed")

    to_process = []
    for record in records:
        record_id = record.get("id")
        if args.resume and record_id in existing_index:
            continue
        to_process.append(record)

    print(f"Records to process this run: {len(to_process)}")

    processed_new = []
    if to_process:
        print("Starting evaluation...")
        start_time = time.time()
        processed_new = process_records_parallel(to_process, evaluators, args.max_workers, args.input)
        save_jsonl(processed_new, args.output, mode="a" if args.resume else "w")
        elapsed_time = time.time() - start_time
        print(f"Evaluation completed in {elapsed_time:.2f} seconds")
    elif not existing_records:
        open(args.output, "a", encoding="utf-8").close()

    total_processed = len(existing_records) + len(processed_new)
    all_records = existing_records + processed_new

    with_answers = sum(1 for item in all_records if item.get("extracted_answer"))
    correct = sum(1 for item in all_records if item.get("is_correct"))

    usage: Dict[str, int] = {}
    for item in all_records:
        model = item.get("judger_model")
        if model:
            usage[model] = usage.get(model, 0) + 1

    print(f"\n=== Evaluation Results ===")
    print(f"Total processed records: {total_processed}")
    print(f"Records with model answers: {with_answers}")
    print(f"Correct answers: {correct}")

    if with_answers > 0:
        accuracy = correct / with_answers * 100
        print(f"Accuracy: {accuracy:.2f}%")

    if usage:
        print("Model usage:")
        for model, count in sorted(usage.items(), key=lambda x: x[0]):
            print(f"  {model}: {count}")

    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
