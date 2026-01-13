#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Question obfuscation Agent
Functions:
Use model to replace keywords in medical questions with descriptive statements, 
To increase question difficulty, obfuscate keywords.
"""

import os
import json
import time
import random
import requests
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm


class GeminiClient:
    """Gemini API client (via proxy)"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("BYTEDANCE_GPT_API_KEY", "add-your-api-key-here")
        self.model = model or "gpt-4o"
        self.base_url = "add-your-api-base-here"
        self.headers = {
            "Content-Type": "application/json",
            "Api-Key": self.api_key,  # Use correct header format
            "X-TT-LOGID": ""
        }

    def chat(self, messages: List[Dict], timeout: int = 120, max_retries: int = 5) -> str:
        """Call Gemini API with retry support"""
        payload = {
            "model": self.model,
            "messages": messages
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout
                )

                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                    print(f"API rate limited, waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    print(f"API returned error status code: {response.status_code}, Response: {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return ""

                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content:
                    print(f"API returned empty content, Response: {result}")
                return content

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                print(f"Gemini API timeout")
                return ""
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                print(f"Gemini API call failed: {e}")
                return ""

        return ""


class QuestionObfuscator:
    """Question obfuscation Agent"""

    OBFUSCATE_PROMPT = """You are a medical question obfuscator. Your task is to:
1. Replace disease names with placeholders (Disease Z, Disease Y, etc. - using reverse alphabet)
2. Replace drug names with placeholders (Drug Z, Drug Y, etc. - using reverse alphabet)
3. Generate indirect descriptions for each entity

IMPORTANT: Use reverse alphabet (Z, Y, X, W...) for placeholders to avoid confusion with answer options (A, B, C, D).

Original Question (with Options):
{question}

Entities to replace:
- Diseases: {diseases}
- Drugs: {drugs}

Reference Information (for your understanding only, DO NOT copy directly):
{descriptions}

Instructions:
1. Replace each disease/drug in BOTH the question AND the options with its placeholder
2. Keep the option labels (A, B, C, D) unchanged - only replace entity names within the option text
3. For each entity, write an indirect description:
   - For diseases: describe using symptoms, clinical signs, lab findings, pathophysiology, epidemiology, risk factors, complications, etc. Do NOT mention the disease name.
   - For drugs: describe using mechanism of action, pharmacokinetics, side effects, contraindications, drug class, therapeutic use, drug interactions, etc. Do NOT mention the drug name.
4. The description should be informative enough to identify the entity but NOT directly reveal its name
5. Make the description comprehensive and medically accurate

Output Format (follow exactly):
[Modified Question]
<the question AND options with entities replaced by placeholders>

[Entity Descriptions]
Disease Z: <indirect description>
Drug Z: <indirect description>
...

Example:
Original: "A patient with diabetes mellitus is prescribed metformin. What is the mechanism?
Options:
A. Inhibits DPP-4
B. Decreases hepatic glucose production
C. Stimulates insulin release
D. Blocks glucose absorption"

Output:
[Modified Question]
A patient with Disease Z is prescribed Drug Z. What is the mechanism?
Options:
A. Inhibits DPP-4
B. Decreases hepatic glucose production
C. Stimulates insulin release
D. Blocks glucose absorption

[Entity Descriptions]
Disease Z: A chronic metabolic condition characterized by elevated fasting blood glucose levels, often associated with polyuria, polydipsia, and HbA1c above 6.5%.
Drug Z: A first-line oral antihyperglycemic agent that works by decreasing hepatic glucose production and improving insulin sensitivity, commonly causing gastrointestinal side effects.

Now process the given question:"""

    def __init__(self, api_key: str = None, model: str = None):
        self.client = GeminiClient(api_key=api_key, model=model)

    def obfuscate(self, question: str, options: str, diseases: List[str], drugs: List[str],
                  descriptions: Dict[str, str]) -> Dict[str, Any]:
        """Obfuscate single question (with options)"""
        # Merge question and options
        full_question = question
        if options and options.strip():
            full_question = f"{question}\n\nOptions:\n{options}"

        # Prepare entity mapping (use reverse alphabet Z, Y, X... to avoid confusion with options A, B, C, D)
        entity_mapping = {}
        disease_labels = []
        drug_labels = []

        for i, d in enumerate(diseases):
            label = f"Disease {chr(90 - i)}"  # Disease Z, Y, X...
            entity_mapping[d] = label
            disease_labels.append(f"{d} -> {label}")

        for i, drug in enumerate(drugs):
            # Drugs continue reverse order after diseases, avoid duplicates
            label = f"Drug {chr(90 - i)}"  # Drug Z, Y, X...
            entity_mapping[drug] = label
            drug_labels.append(f"{drug} -> {label}")

        # Prepare reference descriptions (for model understanding)
        desc_lines = []
        for entity, label in entity_mapping.items():
            desc = descriptions.get(entity, "")
            if desc:
                desc_lines.append(f"  - {label} ({entity}): {desc[:500]}")

        descriptions_text = "\n".join(desc_lines) if desc_lines else "(No reference available)"

        prompt = self.OBFUSCATE_PROMPT.format(
            question=full_question,
            diseases=", ".join(disease_labels) if disease_labels else "(none)",
            drugs=", ".join(drug_labels) if drug_labels else "(none)",
            descriptions=descriptions_text
        )

        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat(messages)

        # Parse response
        obfuscated_question = ""
        entity_descriptions = {}

        if response:
            lines = response.strip().split('\n')
            in_question = False
            in_descriptions = False
            question_lines = []

            for line in lines:
                line_stripped = line.strip()
                if '[Modified Question]' in line_stripped:
                    in_question = True
                    in_descriptions = False
                    continue
                elif '[Entity Descriptions]' in line_stripped:
                    in_question = False
                    in_descriptions = True
                    continue

                if in_question and line_stripped:
                    question_lines.append(line)
                elif in_descriptions and line_stripped:
                    # Parse "Disease A: description" format
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            entity_label = parts[0].strip()
                            entity_desc = parts[1].strip()
                            entity_descriptions[entity_label] = entity_desc

            obfuscated_question = '\n'.join(question_lines).strip()

        # If parsing failed, return original question (with options)
        if not obfuscated_question:
            obfuscated_question = full_question

        # Check if success: question must be modified and have entity descriptions
        is_modified = obfuscated_question != full_question
        success = is_modified and bool(entity_descriptions)

        return {
            "original_question": question,
            "original_options": options,
            "original_full_question": full_question,
            "diseases": diseases,
            "drugs": drugs,
            "entity_mapping": entity_mapping,
            "obfuscated_question": obfuscated_question,
            "entity_descriptions": entity_descriptions,
            "raw_response": response[:2000] if response and not success else "",  # Save raw response for debugging on failure
            "success": success
        }


def load_keyword_descriptions(filepath: str) -> Dict[str, str]:
    """Load keyword description file"""
    descriptions = {}
    if not os.path.exists(filepath):
        return descriptions

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    keyword = record.get("keyword", "")
                    desc = record.get("final_description", "")
                    if keyword and desc:
                        descriptions[keyword] = desc
                except json.JSONDecodeError:
                    continue
    return descriptions


def process_single_question(args):
    """Process single question (for concurrency)"""
    record, obfuscator, descriptions = args
    question = record.get("question", "")
    record_id = record.get("id", "")
    diseases = record.get("extracted_diseases", [])
    drugs = record.get("extracted_drugs", [])

    # Get options (support multiple formats)
    options = record.get("options", "")
    if isinstance(options, list):
        # If list format, convert to string
        options = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    elif isinstance(options, dict):
        # If dict format {"A": "xxx", "B": "yyy"}
        options = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])

    # Filter empty strings
    diseases = [d for d in diseases if d and d.strip()]
    drugs = [d for d in drugs if d and d.strip()]

    try:
        result = obfuscator.obfuscate(question, options, diseases, drugs, descriptions)
        result["id"] = record_id
        return result
    except Exception as e:
        return {
            "id": record_id,
            "original_question": question,
            "original_options": options,
            "obfuscated_question": question,
            "diseases": diseases,
            "drugs": drugs,
            "success": False,
            "error": str(e)
        }


def get_processed_ids(output_file: str) -> set:
    """Get processed record IDs"""
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "id" in record:
                            processed_ids.add(record["id"])
                    except json.JSONDecodeError:
                        continue
    return processed_ids


def has_entities(record: Dict) -> bool:
    """Check if record has entities"""
    diseases = record.get("extracted_diseases", [])
    drugs = record.get("extracted_drugs", [])
    diseases = [d for d in diseases if d and d.strip()]
    drugs = [d for d in drugs if d and d.strip()]
    return bool(diseases) or bool(drugs)


def main():
    # ==================== Configuration parameters (modify here directly) ====================
    INPUT_FILE = "./extracted_entities_all.jsonl"  # Contains option entities
    DESCRIPTIONS_FILE = "./entity_cache.jsonl"
    OUTPUT_FILE = "./obfuscated_questions_v2.jsonl"  # New version output

    # API configuration
    GEMINI_API_KEY = "add-your-api-key-here"
    GEMINI_MODEL = "gpt-4o"

    MAX_WORKERS = 8  # Concurrent processing count
    LIMIT = None  # Limit number of records to process, None means process all
    # ================================================================

    print(f"Input file: {INPUT_FILE}")
    print(f"Description file: {DESCRIPTIONS_FILE}")
    print(f"Output file: {OUTPUT_FILE}")

    # Load keyword descriptions
    print(f"\nLoad keyword descriptions...")
    descriptions = load_keyword_descriptions(DESCRIPTIONS_FILE)
    print(f"Loaded {len(descriptions)} keyword descriptions")

    # Get processed record IDs
    processed_ids = get_processed_ids(OUTPUT_FILE)
    print(f"Processed records: {len(processed_ids)}")

    # ReadingInput file, filteringNo entitiesand processed records
    print(f"\nReading records to process...")
    pending_records = []
    total_records = 0
    skipped_no_entities = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                total_records += 1
                record_id = record.get("id")

                # Skip processed records
                if record_id in processed_ids:
                    continue

                # Skip records without entities
                if not has_entities(record):
                    skipped_no_entities += 1
                    continue

                pending_records.append(record)
            except json.JSONDecodeError:
                continue

    if LIMIT:
        pending_records = pending_records[:LIMIT]

    print(f"Total records in input file: {total_records}")
    print(f"Skipped records without entities: {skipped_no_entities}")
    print(f"Records to process: {len(pending_records)}")

    if not pending_records:
        print("No records to process, exiting")
        return

    # Create obfuscator
    obfuscator = QuestionObfuscator(
        api_key=GEMINI_API_KEY,
        model=GEMINI_MODEL
    )

    # Statistics variables
    total_processed = 0
    total_success = 0
    output_lock = Lock()

    # Concurrent processing
    print(f"\nStarting concurrent processing (max_workers={MAX_WORKERS})...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [(record, obfuscator, descriptions) for record in pending_records]
        futures = {executor.submit(process_single_question, task): task[0].get("id", "") for task in tasks}

        with tqdm(total=len(futures), desc="Obfuscate questions") as pbar:
            for future in as_completed(futures):
                record_id = futures[future]
                try:
                    result = future.result()

                    # Thread-safe write to output file
                    with output_lock:
                        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
                            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

                    total_processed += 1
                    if result.get("success", False):
                        total_success += 1

                    pbar.update(1)
                    pbar.set_postfix({"Success": total_success, "Processed": total_processed})

                except Exception as e:
                    print(f"\nProcess record ID={record_id} Failed: {e}")
                    pbar.update(1)

    # Statistics info
    print(f"\n{'='*60}")
    print("Processing completed! Statistics:")
    print(f"{'='*60}")
    print(f"  Processed records: {total_processed}")
    print(f"  Successfully obfuscated: {total_success}")
    if total_processed > 0:
        print(f"  Success rate: {total_success/total_processed*100:.1f}%")


if __name__ == "__main__":
    main()
