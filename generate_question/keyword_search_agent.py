#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyword search and description generation Agent
Functions:
1. Use Serper to search keywords
2. Use Jina to visit top 5 webpages
3. Use API to summarize each webpage content
4. Combine 5 summaries to generate final keyword description
5. Process input file line by line, supports checkpoint resume
6. Maintain entity cache to avoid duplicate processing of same keywords
"""

import os
import json
import time
import random
import requests
import fcntl
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threading import Lock


class EntityCache:
    """Entity cache manager for storing and reusing processed keyword descriptions

    Optimized: only load key -> file offset index, read full content on demand, greatly reduce memory usage
    """

    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.index: Dict[str, int] = {}  # key -> File offset
        self.memory_cache: Dict[str, Dict] = {}  # Runtime added cache (not yet written to file)
        self.lock = Lock()
        self._build_index()

    def _build_index(self):
        """Build key -> file offset index (without loading full content)"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        try:
                            # Only parse to get keyword, not saving full record
                            record = json.loads(line)
                            keyword = record.get("keyword", "")
                            if keyword and record.get("success", False):
                                self.index[keyword.lower()] = offset
                        except json.JSONDecodeError:
                            continue
            print(f"Loaded from cache {len(self.index)} processed keywords")

    def get(self, keyword: str) -> Optional[Dict]:
        """Get cached keyword description (read from file on demand)"""
        key = keyword.lower()
        with self.lock:
            # First check memory cache (newly added)
            if key in self.memory_cache:
                return self.memory_cache[key]

            # Read from file on demand
            if key in self.index:
                try:
                    with open(self.cache_file, 'r', encoding='utf-8') as f:
                        f.seek(self.index[key])
                        line = f.readline()
                        if line.strip():
                            return json.loads(line)
                except Exception:
                    pass
            return None

    def set(self, keyword: str, result: Dict):
        """Set cache and append write to file"""
        key = keyword.lower()
        with self.lock:
            # Append write to cache file and record offset
            with open(self.cache_file, 'a', encoding='utf-8') as f:
                offset = f.tell()
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                self.index[key] = offset

    def has(self, keyword: str) -> bool:
        """Check if keyword is cached"""
        key = keyword.lower()
        with self.lock:
            return key in self.index or key in self.memory_cache

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics info"""
        with self.lock:
            total = len(self.index) + len(self.memory_cache)
            return {"total": total, "success": total}


class URLCache:
    """URL-level cache manager for storing and reusing webpage raw content

    Optimized version: only load url -> file offset index, read full content on demand, significantly reducing memory usage
    """

    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.index: Dict[str, int] = {}  # url -> File offset
        self.lock = Lock()
        self._build_index()

    def _build_index(self):
        """Build url -> file offset index (without loading full content)"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        try:
                            # Only parse to get url, not saving full record
                            record = json.loads(line)
                            url = record.get("url", "")
                            if url:
                                self.index[url] = offset
                        except json.JSONDecodeError:
                            continue
            print(f"Loaded from URL cache {len(self.index)} webpage contents")

    def get(self, url: str) -> Optional[Dict]:
        """Get cached webpage content (read from file on demand)"""
        with self.lock:
            if url in self.index:
                try:
                    with open(self.cache_file, 'r', encoding='utf-8') as f:
                        f.seek(self.index[url])
                        line = f.readline()
                        if line.strip():
                            return json.loads(line)
                except Exception:
                    pass
            return None

    def set(self, url: str, result: Dict):
        """Set cache and append write to file"""
        with self.lock:
            if url not in self.index:  # Avoid duplicate writes
                with open(self.cache_file, 'a', encoding='utf-8') as f:
                    offset = f.tell()
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    self.index[url] = offset

    def has(self, url: str) -> bool:
        """Check if URL is cached"""
        with self.lock:
            return url in self.index

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics info"""
        with self.lock:
            return {"total": len(self.index)}


class SerperClient:
    """Serper search client"""

    def __init__(self, api_key: str = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("SERPER_API_KEY", "")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        })

    def search(self, query: str, num: int = 5, gl: str = "us", hl: str = "en") -> List[Dict]:
        """Search and return top num results"""
        url = "https://google.serper.dev/search"
        payload = {"q": query, "gl": gl, "hl": hl, "num": num}

        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            organic = data.get("organic", [])
            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                }
                for item in organic[:num]
            ]
        except Exception as e:
            print(f"SerperSearchFailed: {e}")
            return []


class JinaReader:
    """Jina Reader webpage content extraction"""

    def __init__(self, api_key: str = None, timeout: int = 30, url_cache: URLCache = None):
        self.api_key = api_key or "add-your-jina-api-key-here"
        self.timeout = timeout
        self.url_cache = url_cache
        self.cache_hits = 0
        self.api_calls = 0

    def read(self, url: str, max_chars: int = 30000) -> Dict[str, str]:
        # First check cache
        if self.url_cache:
            cached = self.url_cache.get(url)
            if cached:
                self.cache_hits += 1
                return {
                    "url": cached.get("url", url),
                    "title": cached.get("title", ""),
                    "content": cached.get("content", "")[:max_chars] if cached.get("content") else ""
                }
        # Call Jina API
        self.api_calls += 1
        try:
            jina_url = f"https://r.jina.ai/{url}"
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            response = requests.get(jina_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            payload = data.get("data", data) if isinstance(data, dict) else data
            title = payload.get("title", "").strip() if isinstance(payload, dict) else ""
            content = payload.get("content", "").strip() if isinstance(payload, dict) else ""

            # Save to cache (save original full content)
            if self.url_cache and content and "Access failed" not in content:
                self.url_cache.set(url, {
                    "url": url,
                    "title": title or "(No title)",
                    "content": content  # Save full content
                })

            # Truncate when returning
            if content and len(content) > max_chars:
                content = content[:max_chars] + "\n...[Content truncated]"

            return {
                "url": url,
                "title": title or "(No title)",
                "content": content or "(No content)"
            }
        except Exception as e:
            return {
                "url": url,
                "title": "(Cannot access)",
                "content": f"Access failed: {str(e)}"
            }


class DoubaoSummarizer:
    """API Summarizer"""

    # Token threshold: below this value keep original text
    TOKEN_THRESHOLD = 10000

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("ARK_API_KEY", "add-your-api-key-here")
        self.model = model or "gpt-4o"
        self.base_url = "add-your-api-base-here"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Initialize tokenizer
        self.tokenizer = None
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "add-your-tokenizer-path-here",
                trust_remote_code=True
            )
            print(f"DoubaoSummarizer: Tokenizer loaded successfully")
        except Exception as e:
            print(f"DoubaoSummarizer: Tokenizer loading failed, will use character estimation: {e}")

    def count_tokens(self, text: str) -> int:
        """Calculate token count of text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Simple estimation: about 1 token per 4 characters on average
            return len(text) // 4

    def summarize(self, keyword: str, title: str, content: str, timeout: int = 120) -> str:
        """Summarize webpage content, extract description about keyword

        If content less than 10k tokens, return original text directly
        If content more than 10k tokens, use LLM for summarization, keep more entity relationship info
        """
        # Calculate token count
        content_tokens = self.count_tokens(content)

        # If content below threshold, return original text directly
        if content_tokens < self.TOKEN_THRESHOLD:
            print(f"    Content {content_tokens} tokens < {self.TOKEN_THRESHOLD}, keeping original text")
            return content

        print(f"    Content {content_tokens} tokens >= {self.TOKEN_THRESHOLD}, performing summarization")

        # When above threshold, use LLM summarization with more detailed prompt
        prompt = f"""You are a medical knowledge extractor. Your task is to extract and preserve important information about a specific medical term from a webpage.

Keyword to focus on: {keyword}

Webpage Title: {title}

Webpage Content:
{content}

Instructions:
1. Extract ALL information that is relevant to the keyword "{keyword}"
2. IMPORTANT: Preserve information about RELATIONSHIPS between entities, including:
   - Drug-disease relationships (what drugs treat what diseases)
   - Drug-drug interactions
   - Disease-symptom relationships
   - Mechanism of action and pathways
   - Contraindications and side effects
   - Clinical associations and comorbidities
3. Keep as much original text as possible, especially:
   - Specific names of drugs, diseases, proteins, genes, pathways
   - Numerical data (dosages, percentages, statistics)
   - Causal relationships and mechanisms
4. If the content doesn't contain useful information about the keyword, respond with "NO_RELEVANT_INFO"
5. The summary can be LONG (up to 2000 words) - do NOT over-compress
6. Use professional medical terminology and preserve technical details

Detailed extraction about "{keyword}":"""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"}
        }

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            print(f"API call failed: {e}")
            return ""

    def combine_summaries(self, keyword: str, summaries: List[str], timeout: int = 120) -> str:
        """Combine multiple summaries into one complete keyword description"""
        valid_summaries = [s for s in summaries if s and "NO_RELEVANT_INFO" not in s]

        if not valid_summaries:
            return f"Unable to find relevant information about {keyword}."

        summaries_text = "\n\n".join([f"Source {i+1}: {s}" for i, s in enumerate(valid_summaries)])

        prompt = f"""You are a medical knowledge synthesizer. Your task is to combine multiple information sources about a medical term into a coherent, comprehensive description.

Keyword: {keyword}

Information from multiple sources:
{summaries_text}

Instructions:
1. Synthesize ALL the relevant information into ONE comprehensive description
2. IMPORTANT: Preserve ALL relationships between entities:
   - Drug-disease relationships
   - Drug-drug interactions
   - Disease-symptom relationships
   - Mechanism of action and pathways
   - Contraindications and side effects
   - Clinical associations
3. Keep specific entity names (drugs, diseases, proteins, genes, pathways)
4. Preserve numerical data and statistics
5. Remove only truly redundant/duplicate information
6. The final description can be LONG (up to 1500 words) - thoroughness is more important than brevity
7. Structure logically but do NOT over-compress
8. Do NOT mention the sources or use phrases like "according to source 1"

Comprehensive description of "{keyword}":"""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"}
        }

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            print(f"Combined summaryFailed: {e}")
            return "\n".join(valid_summaries) if valid_summaries else ""


class KeywordSearchAgent:
    """Keyword search and description generation Agent"""

    def __init__(self, serper_api_key: str = None, jina_api_key: str = None,
                 doubao_api_key: str = None, doubao_model: str = None,
                 num_pages: int = 5, entity_cache: EntityCache = None,
                 url_cache: URLCache = None):
        self.searcher = SerperClient(api_key=serper_api_key)
        self.reader = JinaReader(api_key=jina_api_key, url_cache=url_cache)
        self.summarizer = DoubaoSummarizer(api_key=doubao_api_key, model=doubao_model)
        self.num_pages = num_pages
        self.entity_cache = entity_cache
        self.url_cache = url_cache

    def process(self, keyword: str, use_cache: bool = True) -> Dict[str, Any]:
        """Process single keyword, return combined description"""
        # First check cache
        if use_cache and self.entity_cache:
            cached = self.entity_cache.get(keyword)
            if cached:
                return cached

        result = {
            "keyword": keyword,
            "search_results": [],
            "page_summaries": [],
            "final_description": "",
            "success": False,
            "from_cache": False
        }

        # Step 1: Search keywords
        search_results = self.searcher.search(keyword, num=self.num_pages)
        result["search_results"] = search_results

        if not search_results:
            result["error"] = "No search results found"
            return result

        # Step 2: Visit each webpage and getContent
        page_contents = []
        for sr in search_results:
            url = sr.get("url", "")
            if url:
                content = self.reader.read(url)
                page_contents.append({
                    "url": url,
                    "title": content.get("title", ""),
                    "content": content.get("content", "")
                })

        # Step 3: Summarize each webpage content
        summaries = []
        for pc in page_contents:
            if pc["content"] and "Access failed" not in pc["content"]:
                summary = self.summarizer.summarize(
                    keyword=keyword,
                    title=pc["title"],
                    content=pc["content"]
                )
                if summary:
                    summaries.append(summary)
                    result["page_summaries"].append({
                        "url": pc["url"],
                        "title": pc["title"],
                        "summary": summary
                    })

        # Step 4: Generate final description from all summaries
        if summaries:
            final_description = self.summarizer.combine_summaries(keyword, summaries)
            result["final_description"] = final_description
            result["success"] = bool(final_description)
        else:
            result["error"] = "No valid summaries generated"

        # Update cache
        if self.entity_cache and result["success"]:
            self.entity_cache.set(keyword, result)

        return result


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


def process_single_record(record: Dict, agent: KeywordSearchAgent, entity_cache: EntityCache) -> Dict:
    """Process single record, get descriptions for all entities"""
    diseases = record.get("extracted_diseases", [])
    drugs = record.get("extracted_drugs", [])

    # Collect all keywords to process
    keywords = []
    keywords.extend([d for d in diseases if d and d.strip()])
    keywords.extend([d for d in drugs if d and d.strip()])

    # Deduplicate
    keywords = list(set(keywords))

    # Get description for each keyword
    keyword_descriptions = {}
    cache_hits = 0
    api_calls = 0

    for kw in keywords:
        # First check cache
        cached = entity_cache.get(kw)
        if cached:
            keyword_descriptions[kw] = cached.get("final_description", "")
            cache_hits += 1
        else:
            # Call API to process (agent.process internally writes to cache, no need to repeat)
            result = agent.process(kw, use_cache=False)
            if result.get("success"):
                keyword_descriptions[kw] = result.get("final_description", "")
                # Note: no need to call entity_cache.set() again, agent.process() already wrote internally
            else:
                keyword_descriptions[kw] = ""
            api_calls += 1

    # Build output record
    output_record = {
        "id": record.get("id"),
        "question": record.get("question", ""),
        "extracted_diseases": diseases,
        "extracted_drugs": drugs,
        "disease_descriptions": {d: keyword_descriptions.get(d, "") for d in diseases if d},
        "drug_descriptions": {d: keyword_descriptions.get(d, "") for d in drugs if d},
        "cache_hits": cache_hits,
        "api_calls": api_calls
    }

    return output_record


def process_record_wrapper(args):
    """Wrapper function for processing single record (for concurrency)"""
    record, agent, entity_cache, skip_empty = args
    record_id = record.get("id")

    try:
        # Check if has entities
        diseases = record.get("extracted_diseases", [])
        drugs = record.get("extracted_drugs", [])
        has_entities = bool([d for d in diseases if d]) or bool([d for d in drugs if d])

        if skip_empty and not has_entities:
            # For records without entities, return empty description directly
            return {
                "id": record_id,
                "question": record.get("question", ""),
                "extracted_diseases": diseases,
                "extracted_drugs": drugs,
                "disease_descriptions": {},
                "drug_descriptions": {},
                "cache_hits": 0,
                "api_calls": 0,
                "skipped": True
            }

        # Process record
        output_record = process_single_record(record, agent, entity_cache)
        return output_record

    except Exception as e:
        return {
            "id": record_id,
            "question": record.get("question", ""),
            "extracted_diseases": record.get("extracted_diseases", []),
            "extracted_drugs": record.get("extracted_drugs", []),
            "disease_descriptions": {},
            "drug_descriptions": {},
            "cache_hits": 0,
            "api_calls": 0,
            "error": str(e)
        }


def main():
    # ==================== Configuration parameters (modify here directly) ====================
    INPUT_FILE = "./extracted_entities_all.jsonl"
    OUTPUT_FILE = "./keyword_descriptions.jsonl"
    CACHE_FILE = "./entity_cache.jsonl"
    URL_CACHE_FILE = "./url_cache.jsonl"  # URL level cache

    # API configuration
    SERPER_API_KEY = "add-your-serper-api-key-here"
    JINA_API_KEY = "add-your-jina-api-key-here"
    DOUBAO_API_KEY = "add-your-api-key-here"
    DOUBAO_MODEL = "gpt-4o"

    NUM_PAGES = 5  # Number of webpages to visit for each keyword search
    MAX_WORKERS = 32  # Concurrent processing count (recommend not too high to avoid API rate limiting)
    LIMIT = None  # Limit record count to process, None means process all
    SKIP_EMPTY = True  # Whether to skip records without entities
    # ================================================================

    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Entity cache: {CACHE_FILE}")
    print(f"URL cache: {URL_CACHE_FILE}")

    # Initialize entity cache
    entity_cache = EntityCache(CACHE_FILE)
    cache_stats = entity_cache.get_stats()
    print(f"Entity cache status: {cache_stats['total']} keywords cached")

    # Initialize URL cache
    url_cache = URLCache(URL_CACHE_FILE)
    url_cache_stats = url_cache.get_stats()
    print(f"URL cache status: {url_cache_stats['total']} webpages cached")

    # Get processed record IDs
    processed_ids = get_processed_ids(OUTPUT_FILE)
    print(f"Processed records: {len(processed_ids)}")

    # Create Agent
    agent = KeywordSearchAgent(
        serper_api_key=SERPER_API_KEY,
        jina_api_key=JINA_API_KEY,
        doubao_api_key=DOUBAO_API_KEY,
        doubao_model=DOUBAO_MODEL,
        num_pages=NUM_PAGES,
        entity_cache=entity_cache,
        url_cache=url_cache
    )

    # Read all records to process
    print("\nReading records to process...")
    pending_records = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_id = record.get("id")
                # Skip processed records
                if record_id not in processed_ids:
                    pending_records.append(record)
            except json.JSONDecodeError:
                continue

    if LIMIT:
        pending_records = pending_records[:LIMIT]

    print(f"Records to process: {len(pending_records)}")

    if not pending_records:
        print("No records to process, exiting")
        return

    # Statistics variables
    total_processed = 0
    total_skipped = 0
    total_cache_hits = 0
    total_api_calls = 0
    output_lock = Lock()

    # Concurrent process records
    print(f"\nStarting concurrent processing (max_workers={MAX_WORKERS})...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [(record, agent, entity_cache, SKIP_EMPTY) for record in pending_records]
        futures = {executor.submit(process_record_wrapper, task): task[0].get("id") for task in tasks}

        with tqdm(total=len(futures), desc="Process record") as pbar:
            for future in as_completed(futures):
                record_id = futures[future]
                try:
                    output_record = future.result()

                    # Thread-safe write to output file
                    with output_lock:
                        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
                            f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')

                    # Update statistics
                    if output_record.get("skipped"):
                        total_skipped += 1
                    else:
                        total_processed += 1
                        total_cache_hits += output_record.get("cache_hits", 0)
                        total_api_calls += output_record.get("api_calls", 0)

                    pbar.update(1)
                    pbar.set_postfix({
                        "Processed": total_processed,
                        "Cache hits": total_cache_hits,
                        "API calls": total_api_calls
                    })

                except Exception as e:
                    print(f"\nProcess record ID={record_id} Failed: {e}")
                    pbar.update(1)

    # Final statistics
    final_cache_stats = entity_cache.get_stats()
    final_url_cache_stats = url_cache.get_stats()

    # Get JinaReader statistics (from agent)
    jina_cache_hits = agent.reader.cache_hits
    jina_api_calls = agent.reader.api_calls

    print(f"\n{'='*60}")
    print("Processing completed! Statistics:")
    print(f"{'='*60}")
    print(f"  Processed records: {total_processed}")
    print(f"  Skipped record count: {total_skipped} (No entities)")
    print(f"\n  [Entity cache]")
    print(f"  Entity cache hits: {total_cache_hits}")
    print(f"  Entity API calls: {total_api_calls}")
    print(f"  Entity cache total: {final_cache_stats['total']}")
    if total_cache_hits + total_api_calls > 0:
        hit_rate = total_cache_hits / (total_cache_hits + total_api_calls) * 100
        print(f"  Entity cache hitsrate: {hit_rate:.1f}%")
    print(f"\n  [URL cache]")
    print(f"  URL cache hits: {jina_cache_hits}")
    print(f"  Jina API calls: {jina_api_calls}")
    print(f"  URL cache total: {final_url_cache_stats['total']}")
    if jina_cache_hits + jina_api_calls > 0:
        url_hit_rate = jina_cache_hits / (jina_cache_hits + jina_api_calls) * 100
        print(f"  URL cache hitsrate: {url_hit_rate:.1f}%")


if __name__ == "__main__":
    main()
