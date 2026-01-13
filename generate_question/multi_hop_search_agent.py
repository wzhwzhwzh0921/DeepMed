#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-hop entity search Agent
Functions:
1. Start from initial entity, search and generate summary
2. Extract related entities from summary
3. Select most logically related entity as next hop
4. Repeat until reaching specified hop count
5. Output entity chain and all summaries
"""

import os
import json
import argparse
import requests
import time
import random
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

# Reuse existing classes
from keyword_search_agent import (
    EntityCache, URLCache, SerperClient, JinaReader, DoubaoSummarizer
)


class CandidateEntityPool:
    """Candidate entity pool manager

    Save related entities extracted during multi-hop walk, for later use as starting points
    """

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.lock = Lock()
        self.candidates = set()  # Use set for deduplication
        self._load()

    def _load(self):
        """Load candidate entities from file"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.candidates = set(data.get("candidates", []))
            except Exception as e:
                print(f"Failed to load candidate entity pool: {e}")
                self.candidates = set()

    def _save(self):
        """Save candidate entities to file"""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "candidates": list(self.candidates),
                    "count": len(self.candidates)
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save candidate entity pool: {e}")

    def add_entities(self, entities: List[str]):
        """Add candidate entities"""
        with self.lock:
            before_count = len(self.candidates)
            self.candidates.update(entities)
            added = len(self.candidates) - before_count
            if added > 0:
                self._save()
            return added

    def remove_entity(self, entity: str) -> bool:
        """Remove a candidate entity (after selected as start)"""
        with self.lock:
            # Case-insensitive matching
            to_remove = None
            for c in self.candidates:
                if c.lower() == entity.lower():
                    to_remove = c
                    break
            if to_remove:
                self.candidates.discard(to_remove)
                self._save()
                return True
            return False

    def sample_one(self) -> Optional[str]:
        """Randomly sample one candidate entity"""
        with self.lock:
            if not self.candidates:
                return None
            return random.choice(list(self.candidates))

    def sample_and_remove(self) -> Optional[str]:
        """Randomly sample and remove one candidate entity"""
        with self.lock:
            if not self.candidates:
                return None
            entity = random.choice(list(self.candidates))
            self.candidates.discard(entity)
            self._save()
            return entity

    def contains(self, entity: str) -> bool:
        """Check if entity is in pool"""
        with self.lock:
            return any(c.lower() == entity.lower() for c in self.candidates)

    def size(self) -> int:
        """Return candidate pool size"""
        return len(self.candidates)

    def get_all(self) -> List[str]:
        """Get all candidate entities"""
        with self.lock:
            return list(self.candidates)


class RelatedEntityExtractor:
    """Agent1: Extract related entities from page summary"""

    EXTRACT_PROMPT = """You are a medical entity extractor. Your task is to extract related medical entities that are EXPLICITLY mentioned in the given text.

Current Entity: {current_entity}

Text (Page Summary):
{summary}

Instructions:
1. Extract ONLY entities that are EXPLICITLY mentioned in the text above
2. Focus on: diseases, drugs, symptoms, treatments, mechanisms, proteins, genes, pathways
3. DO NOT include the current entity "{current_entity}" itself
4. DO NOT make up or infer entities that are not in the text
5. Return 5-10 most relevant entities

Return your answer in JSON format ONLY:
{{
    "related_entities": ["entity1", "entity2", "entity3", ...]
}}
"""

    def __init__(self, api_key: str = None, model: str = None,
                 max_retries: int = 5, base_sleep: float = 2.0):
        self.api_key = api_key or os.getenv("ARK_API_KEY", "add-your-api-key-here")
        self.model = model or "gpt-4o"
        self.base_url = "add-your-api-base-here"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.max_retries = max_retries
        self.base_sleep = base_sleep

    def _call_with_retry(self, payload: dict, timeout: int = 60) -> Optional[dict]:
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
                # For other errors, wait briefly before retry
                if attempt < self.max_retries - 1:
                    sleep_time = self.base_sleep * (attempt + 1)
                    print(f"    API call failed: {e}, waiting {sleep_time:.1f}s before retry ({attempt + 1}/{self.max_retries})")
                    time.sleep(sleep_time)

        print(f"    Retry {self.max_retries} times still failed: {last_error}")
        return None

    def extract(self, current_entity: str, summary: str, timeout: int = 60) -> List[str]:
        """Extract related entities from summary"""
        prompt = self.EXTRACT_PROMPT.format(
            current_entity=current_entity,
            summary=summary[:8000]  # Limit length
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"}
        }

        result = self._call_with_retry(payload, timeout)
        if not result:
            return []

        try:
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            parsed = json.loads(content.strip())
            entities = parsed.get("related_entities", [])
            # Filter out current entity
            entities = [e for e in entities if e.lower() != current_entity.lower()]
            return entities

        except Exception as e:
            print(f"    Parse response failed: {e}")
            return []


class EntitySelector:
    """Agent2: Select most logically related entity as next hop"""

    SELECT_PROMPT = """You are a medical knowledge expert. Your task is to select the most logically related entity for knowledge exploration.

Current Entity: {current_entity}
Current Entity Summary: {current_summary}

Candidate Entities (extracted from the summary):
{candidates}

Already Visited Entities (DO NOT select these):
{visited}

Instructions:
1. Select ONE entity from the candidates that has the STRONGEST logical relationship with the current entity
2. Consider relationships like: mechanism of action, treatment targets, disease pathways, drug interactions, clinical associations
3. DO NOT select any entity that is already in the visited list
4. Prefer entities that would provide NEW and VALUABLE medical knowledge
5. If no good candidate exists, return "NONE"

Return your answer in JSON format ONLY:
{{
    "selected_entity": "entity_name",
    "reason": "brief explanation of why this entity is most relevant"
}}
"""

    def __init__(self, api_key: str = None, model: str = None,
                 max_retries: int = 5, base_sleep: float = 2.0):
        self.api_key = api_key or os.getenv("ARK_API_KEY", "add-your-api-key-here")
        self.model = model or "gpt-4o"
        self.base_url = "add-your-api-base-here"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.max_retries = max_retries
        self.base_sleep = base_sleep

    def _call_with_retry(self, payload: dict, timeout: int = 60) -> Optional[dict]:
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
                # For other errors, wait briefly before retry
                if attempt < self.max_retries - 1:
                    sleep_time = self.base_sleep * (attempt + 1)
                    print(f"    API call failed: {e}, waiting {sleep_time:.1f}s before retry ({attempt + 1}/{self.max_retries})")
                    time.sleep(sleep_time)

        print(f"    Retry {self.max_retries} times still failed: {last_error}")
        return None

    def select(self, current_entity: str, current_summary: str,
               candidates: List[str], visited: List[str], timeout: int = 60) -> Dict[str, str]:
        """Select most relevant next hop entity"""
        # Filter visited entities
        filtered_candidates = [c for c in candidates if c.lower() not in [v.lower() for v in visited]]

        if not filtered_candidates:
            return {"selected_entity": "NONE", "reason": "No valid candidates available"}

        candidates_text = "\n".join([f"- {c}" for c in filtered_candidates])
        visited_text = "\n".join([f"- {v}" for v in visited]) if visited else "(none)"

        prompt = self.SELECT_PROMPT.format(
            current_entity=current_entity,
            current_summary=current_summary[:3000],
            candidates=candidates_text,
            visited=visited_text
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"}
        }

        result = self._call_with_retry(payload, timeout)
        if not result:
            return {"selected_entity": "NONE", "reason": "API call failed after retries"}

        try:
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            parsed = json.loads(content.strip())
            return {
                "selected_entity": parsed.get("selected_entity", "NONE"),
                "reason": parsed.get("reason", "")
            }

        except Exception as e:
            print(f"    Failed to parse selection response: {e}")
            return {"selected_entity": "NONE", "reason": str(e)}


class MultiHopSearchAgent:
    """Multi-hop search Agent"""

    def __init__(self, serper_api_key: str = None, jina_api_key: str = None,
                 doubao_api_key: str = None, doubao_model: str = None,
                 entity_cache: EntityCache = None, url_cache: URLCache = None,
                 candidate_pool: CandidateEntityPool = None,
                 num_pages: int = 5):
        # Search and summary components
        self.searcher = SerperClient(api_key=serper_api_key)
        self.reader = JinaReader(api_key=jina_api_key, url_cache=url_cache)
        self.summarizer = DoubaoSummarizer(api_key=doubao_api_key, model=doubao_model)

        # Agent components
        self.entity_extractor = RelatedEntityExtractor(api_key=doubao_api_key, model=doubao_model)
        self.entity_selector = EntitySelector(api_key=doubao_api_key, model=doubao_model)

        # Cache
        self.entity_cache = entity_cache
        self.url_cache = url_cache
        self.candidate_pool = candidate_pool  # Candidate entity pool
        self.num_pages = num_pages

    def search_and_summarize(self, entity: str) -> Dict[str, Any]:
        """Search entity and generate summary"""
        # First check entity cache
        if self.entity_cache:
            cached = self.entity_cache.get(entity)
            if cached and cached.get("success"):
                return {
                    "entity": entity,
                    "final_description": cached.get("final_description", ""),
                    "page_summaries": cached.get("page_summaries", []),
                    "from_cache": True,
                    "success": True
                }

        result = {
            "entity": entity,
            "search_results": [],
            "page_summaries": [],
            "final_description": "",
            "from_cache": False,
            "success": False
        }

        # Step 1: Search
        search_results = self.searcher.search(entity, num=self.num_pages)
        result["search_results"] = search_results

        if not search_results:
            return result

        # Step 2: Get webpage content
        page_contents = []
        for sr in search_results:
            url = sr.get("url", "")
            if url:
                content = self.reader.read(url)
                page_contents.append(content)

        # Step 3: Summarize each webpage
        summaries = []
        for pc in page_contents:
            if pc["content"] and "Access failed" not in pc["content"]:
                summary = self.summarizer.summarize(
                    keyword=entity,
                    title=pc["title"],
                    content=pc["content"]
                )
                if summary and "NO_RELEVANT_INFO" not in summary:
                    summaries.append(summary)
                    result["page_summaries"].append({
                        "url": pc["url"],
                        "title": pc["title"],
                        "summary": summary
                    })

        # Step 4: Combined summary
        if summaries:
            final_description = self.summarizer.combine_summaries(entity, summaries)
            result["final_description"] = final_description
            result["success"] = bool(final_description)

            # Save to cache
            if self.entity_cache and result["success"]:
                self.entity_cache.set(entity, {
                    "keyword": entity,
                    "final_description": final_description,
                    "page_summaries": result["page_summaries"],
                    "success": True
                })

        return result

    def multi_hop_search(self, start_entity: str, min_hops: int = 2, max_hops: int = 3) -> Dict[str, Any]:
        """Execute multi-hop search

        Args:
            start_entity: Start entity
            min_hops: Target hop count lower bound (min when randomly selecting)
            max_hops: Target hop count upper bound (max when randomly selecting)
        """
        # Randomly select target hop count for this search
        target_hops = random.randint(min_hops, max_hops)

        result = {
            "start_entity": start_entity,
            "min_hops": min_hops,
            "max_hops": max_hops,
            "target_hops": target_hops,  # Randomly selected target hop count for this time
            "entity_chain": [start_entity],
            "hop_details": [],
            "summaries": {},
            "actual_hops": 0,
            "success": True
        }

        print(f"\n{'#'*50}")
        print(f"# Target hop count: {target_hops} (Range: {min_hops}-{max_hops})")
        print(f"{'#'*50}")

        current_entity = start_entity
        visited = [start_entity]

        for hop in range(target_hops):
            print(f"\n{'='*50}")
            print(f"Hop {hop + 1}/{target_hops}: {current_entity}")
            print(f"{'='*50}")

            # Step 1: Search current entity and generate summary
            search_result = self.search_and_summarize(current_entity)

            if not search_result["success"]:
                print(f"  Search failed, terminating multi-hop")
                result["success"] = False  # Did not complete target hop count
                break

            # Update actual hop count
            result["actual_hops"] = hop + 1

            # Save summary
            result["summaries"][current_entity] = {
                "final_description": search_result["final_description"],
                "page_summaries": search_result["page_summaries"],
                "from_cache": search_result.get("from_cache", False)
            }

            print(f"  Summary length: {len(search_result['final_description'])} characters")
            print(f"  From cache: {search_result.get('from_cache', False)}")
            print(f"  Current hop count: {hop + 1}, Target hop count: {target_hops}")

            # If last hop, no need to select next entity
            if hop == target_hops - 1:
                result["hop_details"].append({
                    "hop": hop + 1,
                    "entity": current_entity,
                    "related_entities": [],
                    "selected_next": None,
                    "selection_reason": "Reached target hops"
                })
                result["success"] = True  # Successfully completed target hop count
                break

            # Step 2: Extract related entities from summary
            # Combine all page summaries
            all_summaries = "\n\n".join([
                ps["summary"] for ps in search_result["page_summaries"]
            ])

            related_entities = self.entity_extractor.extract(current_entity, all_summaries)
            print(f"  Extracted {len(related_entities)} related entities: {related_entities[:5]}...")

            # Add extracted entities to candidate pool (excluding visited)
            if self.candidate_pool and related_entities:
                entities_to_add = [e for e in related_entities if e.lower() not in [v.lower() for v in visited]]
                if entities_to_add:
                    added = self.candidate_pool.add_entities(entities_to_add)
                    print(f"  Added {added} new entities to candidate pool (Pool size: {self.candidate_pool.size()})")

            if not related_entities:
                print(f"  No related entities found, completed {hop + 1}/{target_hops} hops")
                result["hop_details"].append({
                    "hop": hop + 1,
                    "entity": current_entity,
                    "related_entities": [],
                    "selected_next": None,
                    "selection_reason": "No related entities found"
                })
                result["success"] = False  # Did not complete target hop count
                break

            # Step 3: Select next hop entity
            selection = self.entity_selector.select(
                current_entity=current_entity,
                current_summary=search_result["final_description"],
                candidates=related_entities,
                visited=visited
            )

            next_entity = selection["selected_entity"]
            selection_reason = selection["reason"]

            print(f"  Selecting next hop: {next_entity}")
            print(f"  Selection reason: {selection_reason}")

            result["hop_details"].append({
                "hop": hop + 1,
                "entity": current_entity,
                "related_entities": related_entities,
                "selected_next": next_entity,
                "selection_reason": selection_reason
            })

            if next_entity == "NONE" or not next_entity:
                print(f"  No suitable next hop entity, completed {hop + 1}/{target_hops} hops")
                result["success"] = False  # Did not complete target hop count
                break

            # Update current entity
            current_entity = next_entity
            visited.append(current_entity)
            result["entity_chain"].append(current_entity)

        print(f"\n{'='*50}")
        print(f"Multi-hop search completed!")
        print(f"Target hop count: {target_hops}, Actual hop count: {result['actual_hops']}")
        print(f"Entity chain: {' -> '.join(result['entity_chain'])}")
        print(f"Success: {result['success']}")
        print(f"{'='*50}")

        return result


def process_single_entity(args) -> Dict[str, Any]:
    """Process multi-hop search for single entity"""
    entity, agent, min_hops, max_hops, record_id = args

    try:
        result = agent.multi_hop_search(entity, min_hops=min_hops, max_hops=max_hops)
        result["id"] = record_id
        return result
    except Exception as e:
        return {
            "id": record_id,
            "start_entity": entity,
            "error": str(e),
            "success": False
        }


def main():
    parser = argparse.ArgumentParser(description="Multi-hop entity search Agent")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input file (JSONL format, containing entity or keyword field)")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output file (JSONL format)")
    parser.add_argument("--min-hops", type=int, default=3,
                        help="Hop count lower bound, randomly selected each time (1-15, default 3)")
    parser.add_argument("--max-hops", type=int, default=10,
                        help="Hop count upper bound, randomly selected each time (1-15, default 10)")
    parser.add_argument("--workers", "-w", type=int, default=8,
                        help="Concurrency (default8)")
    parser.add_argument("--sample", "-s", type=int, default=None,
                        help="Random sample count, randomly select specified number of entities from input file as starting points")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Limit processing count (take first N sequentially, different from --sample)")
    parser.add_argument("--entity-cache", type=str,
                        default="./entity_cache.jsonl",
                        help="Entity cache file")
    parser.add_argument("--url-cache", type=str,
                        default="./url_cache.jsonl",
                        help="URL cache file")
    parser.add_argument("--candidate-pool", type=str,
                        default="./candidate_entity.json",
                        help="Candidate entity pool file")
    parser.add_argument("--candidate-prob", type=float, default=0.3,
                        help="Probability of selecting start point from candidate pool (0-1, default0.3)")

    args = parser.parse_args()

    # Validate hop count
    if args.min_hops < 1 or args.min_hops > 15:
        print(f"Error: Hop count lower bound must be in 1-15 range, current: {args.min_hops}")
        return

    if args.max_hops < 1 or args.max_hops > 15:
        print(f"Error: Hop count upper bound must be in 1-15 range, current: {args.max_hops}")
        return

    if args.min_hops > args.max_hops:
        print(f"Error: Hop count lower bound cannot exceed upper bound")
        return

    print(f"{'='*60}")
    print(f"Multi-hop entity search Agent")
    print(f"{'='*60}")
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Hop count range: {args.min_hops} - {args.max_hops} (Randomly selected each time)")
    print(f"Concurrency: {args.workers}")
    if args.sample:
        print(f"Random sampling: {args.sample} starting points")
    if args.limit:
        print(f"Sequential limit: {args.limit}")
    print(f"Entity cache: {args.entity_cache}")
    print(f"URL cache: {args.url_cache}")
    print(f"Candidate entity pool: {args.candidate_pool}")
    print(f"Candidate pool selection probability: {args.candidate_prob*100:.0f}%")

    # Initialize cache
    entity_cache = EntityCache(args.entity_cache)
    url_cache = URLCache(args.url_cache)
    candidate_pool = CandidateEntityPool(args.candidate_pool)

    print(f"\nEntity cache: {entity_cache.get_stats()['total']} records")
    print(f"URL cache: {url_cache.get_stats()['total']} records")
    print(f"Candidate entity pool: {candidate_pool.size()} records")

    # API configuration
    SERPER_API_KEY = os.getenv("SERPER_API_KEY", "add-your-serper-api-key-here")
    JINA_API_KEY = os.getenv("JINA_API_KEY", "add-your-jina-api-key-here")
    DOUBAO_API_KEY = os.getenv("ARK_API_KEY", "add-your-api-key-here")
    DOUBAO_MODEL = os.getenv("ARK_MODEL", "gpt-4o")

    # Create Agent
    agent = MultiHopSearchAgent(
        serper_api_key=SERPER_API_KEY,
        jina_api_key=JINA_API_KEY,
        doubao_api_key=DOUBAO_API_KEY,
        doubao_model=DOUBAO_MODEL,
        entity_cache=entity_cache,
        url_cache=url_cache,
        candidate_pool=candidate_pool,
        num_pages=5
    )

    # Read all entities from input file
    print(f"\nReading input file...")
    input_entities = []  # entity list
    with open(args.input, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                # Get entity (supports multiple field names)
                entity = record.get("entity") or record.get("keyword") or record.get("start_entity")
                if entity:
                    input_entities.append(entity)
            except:
                pass

    print(f"Entities in input file: {len(input_entities)} records")
    print(f"Entities in candidate pool: {candidate_pool.size()} records")

    # Determine total count to process
    if args.sample:
        target_count = args.sample
    elif args.limit:
        target_count = args.limit
    else:
        target_count = len(input_entities)

    # Calculate proportionally how many from each source
    from_candidate_target = int(target_count * args.candidate_prob)
    from_input_target = target_count - from_candidate_target

    # Actual count available (cannot exceed respective pool size)
    from_candidate_actual = min(from_candidate_target, candidate_pool.size())
    from_input_actual = min(from_input_target, len(input_entities))

    # If one side not enough, try to supplement from other side
    shortfall_candidate = from_candidate_target - from_candidate_actual
    shortfall_input = from_input_target - from_input_actual

    if shortfall_candidate > 0 and len(input_entities) > from_input_actual:
        extra_from_input = min(shortfall_candidate, len(input_entities) - from_input_actual)
        from_input_actual += extra_from_input

    if shortfall_input > 0 and candidate_pool.size() > from_candidate_actual:
        extra_from_candidate = min(shortfall_input, candidate_pool.size() - from_candidate_actual)
        from_candidate_actual += extra_from_candidate

    print(f"\nTarget processing count: {target_count} records")
    print(f"  Randomly selected from input file: {from_input_actual} records")
    print(f"  Randomly select from candidate pool: {from_candidate_actual} records")

    # Build task list
    pending_tasks = []
    next_id = 1

    # Randomly selected from input file
    if from_input_actual > 0:
        selected_from_input = random.sample(input_entities, from_input_actual)
        for entity in selected_from_input:
            record_id = f"input_{next_id}"
            next_id += 1
            pending_tasks.append((entity, agent, args.min_hops, args.max_hops, record_id))

    # Randomly select from candidate pool and remove
    for _ in range(from_candidate_actual):
        entity = candidate_pool.sample_and_remove()
        if entity:
            record_id = f"candidate_{next_id}"
            next_id += 1
            pending_tasks.append((entity, agent, args.min_hops, args.max_hops, record_id))

    # Shuffle order
    random.shuffle(pending_tasks)

    print(f"To process: {len(pending_tasks)} records")

    if not pending_tasks:
        print("No records to process")
        return

    # Concurrent processing
    output_lock = Lock()
    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_entity, task): task[4] for task in pending_tasks}

        with tqdm(total=len(futures), desc="Multi-hop search") as pbar:
            for future in as_completed(futures):
                record_id = futures[future]
                try:
                    result = future.result()

                    # Write results
                    with output_lock:
                        with open(args.output, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')

                    if result.get("success", False):
                        success_count += 1
                    else:
                        error_count += 1

                    pbar.update(1)
                    pbar.set_postfix({"Success": success_count, "Failed": error_count})

                except Exception as e:
                    print(f"\nProcessing ID={record_id} Failed: {e}")
                    error_count += 1
                    pbar.update(1)

    # Statistics
    print(f"\n{'='*60}")
    print(f"Processing completed!")
    print(f"{'='*60}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {error_count}")
    print(f"  Output file: {args.output}")


if __name__ == "__main__":
    main()
