#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate questions from multi-hop path data
- Input: multi_hop_results.jsonl (multi-hop search results)
- Output: generated questions and answers
- Feature: path reversal, answer points to first entity (original start)
"""

import os
import json
import re
import time
import random
import argparse
import requests
import threading
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Prompt template: forward-last mode (follow chain, answer is last node)
PROMPT_FORWARD_LAST = r"""
You are an expert medical puzzle creator.

INPUT
- Path: %(path)s
- Node Introductions: %(intro)s
- Target (Answer): %(target)s (the LAST node in the path)

TASK
Write a single, challenging multi-step clinical riddle that forces the solver to follow:
Node1 → Node2 → ... → Target (last node).

HARD RULES (must follow)
1) Path coverage: Mention EVERY node (1–2 key attributes each). Explicitly state the link between each consecutive pair.
2) Clinical depth: Use concrete clinical/pathologic/diagnostic/mechanistic details when relevant.
3) Obfuscation: Do NOT name any entity in the path. Paraphrase; avoid keyword-copying from the node text.
4) Answer derivation: Do NOT over-describe the Target (max 1–2 identifying features). Make the penultimate node do most of the "pointing" to the answer.
5) Form: 150–250 words. Do NOT start with "Starting with/Begin with/Consider". MUST end with a clear question.

OUTPUT (exactly)
<question>: ...
<answer>: %(target)s
"""

# Prompt template: forward-first mode (follow chain, answer is first node)
PROMPT_FORWARD_FIRST = r"""
You are an expert medical puzzle creator.

INPUT
- Path: %(path)s
- Node Introductions: %(intro)s
- Target (Answer): %(target)s (the FIRST node in the path)

TASK
Write a single, challenging multi-step clinical riddle. The question describes a chain of reasoning starting from an UNKNOWN entity (the answer) and progressing through subsequent nodes. The solver must work BACKWARDS from the clues to identify what the first entity was.

HARD RULES (must follow)
1) **CRITICAL - Minimal exposure of Target**: Give only 1 VAGUE attribute of the Target (first node). Do NOT describe it in detail - this is the answer! Most clues should come from nodes 2, 3, etc.
2) Path coverage: Mention nodes 2, 3, ... with 1–2 key attributes each. Show how each leads to the next.
3) Clinical depth: Use concrete clinical/pathologic/diagnostic/mechanistic details for intermediate nodes.
4) Obfuscation: Do NOT name any entity in the path. Paraphrase; avoid keyword-copying from the node text.
5) Backward reasoning: Structure the question so it says "A certain [vague category] leads to X, which causes Y, which is treated by Z. What is this [vague category]?"
6) Form: 150–250 words. MUST end with a clear question asking for the FIRST entity.

OUTPUT (exactly)
<question>: ...
<answer>: %(target)s
"""

# Prompt template: numerical/dosage version (extra requirements)
NUMERICAL_REQUIREMENT = r"""
### IMPORTANT: Numerical & Dosage Focus (Only if data available)
For THIS question, IF the node introductions contain numerical or quantitative information, you should incorporate them.

**CRITICAL: Only include numerical clues if they are explicitly mentioned or can be reliably inferred from the provided node introductions. Do NOT fabricate or guess numbers that are not supported by the input data.**

If numerical data IS available, consider including:
- **Dosage clues**: "typically initiated at a dose measured in single-digit milligrams twice daily", "maximum daily dose approaches a three-digit number in milligrams"
- **Numerical thresholds**: "blood pressure exceeding 180/120 mmHg", "heart rate below 60 bpm", "half-life of approximately 3-7 hours"
- **Quantitative features**: "bioavailability around 50%%", "protein binding exceeds 90%%", "onset of action within 1-2 hours"
- **Obfuscated numbers**: Instead of "500mg", say "a dose in the mid-triple digits in milligrams"; instead of "12 hours", say "a half-life spanning roughly half a day"
- **Mathematical hints**: "the standard dose is a power of 2 multiplied by 25", "duration measured in single-digit hours"
- **Percentages and ratios**: "accounts for roughly one-third of cases", "occurs in approximately 15-20%% of patients"

If NO numerical data is available in the node introductions, simply generate a standard question without forcing numerical content.

"""


class QuestionGenerator:
    """Question generator"""

    def __init__(self, api_key: str = None, model: str = None,
                 max_retries: int = 5, base_sleep: float = 2.0):
        # API configuration
        self.api_key = api_key or "add-your-api-key-here"
        self.model = model or "gpt-4o"
        self.base_url = "add-your-api-base-here"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.max_retries = max_retries
        self.base_sleep = base_sleep

    def _call_with_retry(self, payload: dict, timeout: int = 500) -> Optional[dict]:
        """API call with retry"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout
                )

                # Handle QPM limit (429)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", self.base_sleep * (2 ** attempt)))
                    sleep_time = retry_after + random.uniform(0, 1)
                    print(f"    QPM limit, waiting {sleep_time:.1f}s before retry ({attempt + 1}/{self.max_retries})")
                    time.sleep(sleep_time)
                    continue

                # Handle server error (5xx)
                if response.status_code >= 500:
                    sleep_time = self.base_sleep * (2 ** attempt) + random.uniform(0, 1)
                    print(f"    Server error {response.status_code}, waiting {sleep_time:.1f}s before retry ({attempt + 1}/{self.max_retries})")
                    time.sleep(sleep_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                sleep_time = self.base_sleep * (attempt + 1)
                print(f"    Request timeout, waiting {sleep_time:.1f}s before retry ({attempt + 1}/{self.max_retries})")
                time.sleep(sleep_time)
                last_error = "Timeout"

            except requests.exceptions.ConnectionError as e:
                sleep_time = self.base_sleep * (2 ** attempt) + random.uniform(0, 1)
                print(f"    Connection error, waiting {sleep_time:.1f}s before retry ({attempt + 1}/{self.max_retries})")
                time.sleep(sleep_time)
                last_error = str(e)

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    sleep_time = self.base_sleep * (attempt + 1)
                    print(f"    API call failed: {e}, waiting {sleep_time:.1f}s before retry ({attempt + 1}/{self.max_retries})")
                    time.sleep(sleep_time)

        print(f"    Retry {self.max_retries} times still failed: {last_error}")
        return None

    def generate(self, path: List[str], summaries: Dict[str, Any], target_entity: str,
                 numerical_prob: float = 0.3, answer_mode: str = "forward-last") -> Optional[Dict[str, str]]:
        """Generate question

        Args:
            path: Entity path (order, A->B->C->D)
            summaries: Summary info for each entity
            target_entity: Target entity (answer)
            numerical_prob: Probability of adding numerical/dosage requirement (default 0.3, i.e. 30%)
            answer_mode: Answer mode
                - "forward-last": Follow chain, answer is last node
                - "forward-first": Follow chain, answer is first node (less description for first node)

        Returns:
            Dictionary containing question and answer
        """
        # Build node introductions
        intro_dict = {}
        for entity in path:
            if entity in summaries:
                summary_info = summaries[entity]
                if isinstance(summary_info, dict):
                    intro_dict[entity] = summary_info.get("final_description", "")[:2000]
                else:
                    intro_dict[entity] = str(summary_info)[:2000]

        # Decide by probability whether to add numerical/dosage requirement
        include_numerical = random.random() < numerical_prob

        # Select prompt template based on mode
        if answer_mode == "forward-first":
            prompt_template = PROMPT_FORWARD_FIRST
        else:  # forward-last
            prompt_template = PROMPT_FORWARD_LAST

        # Build prompt
        base_prompt = prompt_template % {
            "path": str(path),
            "intro": json.dumps(intro_dict, ensure_ascii=False, indent=2),
            "target": target_entity
        }

        if include_numerical:
            # Insert numerical requirement after "3) Obfuscation"
            prompt = base_prompt.replace(
                "3) Obfuscation",
                NUMERICAL_REQUIREMENT + "\n3) Obfuscation"
            )
        else:
            prompt = base_prompt

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            "reasoning": True
        }

        result = self._call_with_retry(payload)
        if not result:
            return None

        try:
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse <question> and <answer>, compatible with multiple formats
            # Format 1: <question>: xxx <answer>: xxx
            # Format 2: <question>\nxxx\n<answer>\nxxx
            # Format 3: <question> xxx </question> <answer> xxx </answer>

            # Try to match <question>...</question> format
            question_match = re.search(
                r"<question>:?\s*(.*?)(?:</question>|<answer>)", content, re.DOTALL | re.IGNORECASE)
            if not question_match:
                # Try to match to end of file
                question_match = re.search(
                    r"<question>:?\s*(.*?)$", content, re.DOTALL | re.IGNORECASE)

            answer_match = re.search(
                r"<answer>:?\s*(.*?)(?:</answer>|$)", content, re.DOTALL | re.IGNORECASE)

            question = question_match.group(1).strip() if question_match else None
            answer = answer_match.group(1).strip() if answer_match else None

            # Clean up extra content in answer
            if answer:
                answer = answer.split('\n')[0].strip()  # Only take first line

            if question and answer:
                return {
                    "question": question,
                    "answer": answer,
                    "numerical_focus": include_numerical  # Record whether numerical requirement was used
                }
            else:
                print(f"    Parse failed, raw response: {content[:200]}...")
                return None

        except Exception as e:
            print(f"    Parse response failed: {e}")
            return None


def process_single_record(args) -> Optional[Dict[str, Any]]:
    """Process single record"""
    record, generator, numerical_prob, answer_mode = args

    try:
        record_id = record.get("id", "unknown")
        entity_chain = record.get("entity_chain", [])
        summaries = record.get("summaries", {})

        # Check path length
        if len(entity_chain) < 2:
            print(f"  ID={record_id}: Path too short, skip")
            return None

        # Determine path and target based on mode
        # Both use sequential path A->B->C->D
        path = entity_chain
        if answer_mode == "forward-last":
            # Answer is last node
            target_entity = entity_chain[-1]
        else:  # forward-first
            # Answer is first node
            target_entity = entity_chain[0]

        # Generate question
        result = generator.generate(
            path=path,
            summaries=summaries,
            target_entity=target_entity,
            numerical_prob=numerical_prob,
            answer_mode=answer_mode
        )

        if result:
            # Build node introductions (for output)
            node_introductions = {}
            for entity in entity_chain:
                if entity in summaries:
                    summary_info = summaries[entity]
                    if isinstance(summary_info, dict):
                        node_introductions[entity] = summary_info.get("final_description", "")
                    else:
                        node_introductions[entity] = str(summary_info)

            return {
                "id": record_id,
                "question": result["question"],
                "answer": result["answer"],
                "target_entity": target_entity,
                "original_path": entity_chain,
                "path_length": len(entity_chain),
                "answer_mode": answer_mode,
                "numerical_focus": result.get("numerical_focus", False),
                "node_introductions": node_introductions  # Node context information
            }
        else:
            return None

    except Exception as e:
        print(f"  Processing failed: {e}")
        return None


def get_processed_paths(output_path: str) -> set:
    """Get processed original_paths (for checkpoint resume)"""
    processed_paths = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "original_path" in data:
                        # Convert list to tuple for hash
                        path_key = tuple(data["original_path"])
                        processed_paths.add(path_key)
                except:
                    continue
    return processed_paths


def main():
    parser = argparse.ArgumentParser(description="Generate questions from multi-hop paths")
    parser.add_argument("--input", "-i", type=str,
                        default="./multi_hop_results.jsonl",
                        help="Input file (multi-hop search results)")
    parser.add_argument("--output", "-o", type=str,
                        default="./generated_questions.jsonl",
                        help="Output file")
    parser.add_argument("--workers", "-w", type=int, default=16,
                        help="Concurrency (default16)")
    parser.add_argument("--sample", "-s", type=int, default=None,
                        help="Random sample count")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Limit processing count")
    parser.add_argument("--min-path-length", type=int, default=3,
                        help="Minimum path length (default3)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model to use (defaultgpt-4o)")
    parser.add_argument("--numerical-prob", type=float, default=0.3,
                        help="Probability of numerical/dosage related questions (default 0.3, i.e. 30%%)")
    parser.add_argument("--answer-mode", type=str, default="forward-last",
                        choices=["forward-last", "forward-first"],
                        help="Answer mode: forward-last=Answer is last node, forward-first=Answer is first node (defaultforward-last)")

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Multi-hop path question generator")
    print(f"{'='*60}")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Concurrency: {args.workers}")
    print(f"Minimum path length: {args.min_path_length}")
    print(f"Model: {args.model}")
    print(f"Numerical/dosage probability: {args.numerical_prob*100:.0f}%")
    print(f"Answer mode: {args.answer_mode}")
    if args.sample:
        print(f"Random sampling: {args.sample}")
    if args.limit:
        print(f"Sequential limit: {args.limit}")
    print(f"{'='*60}")

    # Get processed original_paths (for checkpoint resume)
    processed_paths = get_processed_paths(args.output)
    print(f"\nProcessed (based on original_path): {len(processed_paths)} records")

    # Create generator
    generator = QuestionGenerator(model=args.model)

    # First pass: only count line numbers meeting criteria (not loading full data to memory)
    print(f"Scanning input file...")
    valid_line_numbers = []
    line_num = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                line_num += 1
                continue
            try:
                # Only parse necessary fields to determine if processing needed
                record = json.loads(line)

                # Skip processed (based on original_path)
                entity_chain = record.get("entity_chain", [])
                path_key = tuple(entity_chain)
                if path_key in processed_paths:
                    line_num += 1
                    continue

                # Skip failed records
                if not record.get("success", False):
                    line_num += 1
                    continue

                # Check path length
                if len(entity_chain) < args.min_path_length:
                    line_num += 1
                    continue

                valid_line_numbers.append(line_num)
            except:
                pass
            line_num += 1

    total_available = len(valid_line_numbers)
    print(f"Records meeting criteria: {total_available} records")

    # Random sampling or limit count
    if args.sample:
        if args.sample < total_available:
            valid_line_numbers = random.sample(valid_line_numbers, args.sample)
            print(f"Random sampling: from {total_available} records select {args.sample} records")
        else:
            print(f"Random sampling: requested {args.sample} records, but only have {total_available} records, using all")
    elif args.limit:
        valid_line_numbers = valid_line_numbers[:args.limit]
        print(f"Sequential limit: take first {args.limit} records")

    print(f"To process: {len(valid_line_numbers)} records")

    if not valid_line_numbers:
        print("No records to process")
        return

    # Convert to set for fast lookup
    valid_line_set = set(valid_line_numbers)

    # Define batch size
    BATCH_SIZE = 500  # Process per batch500records

    # Concurrent processing
    write_lock = threading.Lock()
    success_count = 0
    error_count = 0

    # Use generator to read data in batches
    def read_records_by_batch(file_path, valid_lines, batch_size):
        """Read records meeting criteria in batches"""
        batch = []
        line_num = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line_num in valid_lines:
                    try:
                        record = json.loads(line)
                        batch.append(record)
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                    except:
                        pass
                line_num += 1
        if batch:
            yield batch

    # Generate simplified output file path (for checking)
    output_base, output_ext = os.path.splitext(args.output)
    simple_output = f"{output_base}_simple{output_ext}"
    print(f"Simplified output: {simple_output}")

    with open(args.output, "a", encoding="utf-8") as fout, \
         open(simple_output, "a", encoding="utf-8") as fout_simple:
        total_to_process = len(valid_line_numbers)
        with tqdm(total=total_to_process, desc="Generate question") as pbar:
            for batch_records in read_records_by_batch(args.input, valid_line_set, BATCH_SIZE):
                # Create thread pool for current batch
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {
                        executor.submit(process_single_record, (record, generator, args.numerical_prob, args.answer_mode)): record.get("id", "")
                        for record in batch_records
                    }

                    for future in as_completed(futures):
                        record_id = futures[future]
                        try:
                            result = future.result(timeout=600)

                            if result:
                                with write_lock:
                                    # Write full version
                                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                                    fout.flush()
                                    # Write simplified version (only keep question, answer, path)
                                    simple_result = {
                                        "question": result["question"],
                                        "answer": result["answer"],
                                        "path": result["original_path"]
                                    }
                                    fout_simple.write(json.dumps(simple_result, ensure_ascii=False) + "\n")
                                    fout_simple.flush()
                                success_count += 1
                            else:
                                error_count += 1

                        except Exception as e:
                            print(f"\nProcessing ID={record_id} Failed: {e}")
                            error_count += 1

                        pbar.update(1)
                        pbar.set_postfix({"Success": success_count, "Failed": error_count})

    # Statistics
    print(f"\n{'='*60}")
    print(f"Processing completed!")
    print(f"{'='*60}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"  Full output: {args.output}")
    print(f"  Simplified output: {simple_output}")


if __name__ == "__main__":
    main()

