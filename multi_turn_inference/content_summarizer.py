# -*- coding: utf-8 -*-
import requests
import json
import re
import time
import os
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional
from transformers import AutoTokenizer

class ContentSummarizer:
    """
    Content summarizer that extracts relevant paragraphs based on user goal
    """
    
    def __init__(self, api_base: str, api_key: str, model_name: str):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.max_chunk_tokens = 95000  # Max tokens per chunk
        self.max_total_tokens = 270000  # Return unable to fetch if exceeds this

        # Check if special authentication is needed
        # Default to standard OpenAI Bearer Token authentication
        # If you need other authentication methods, modify this logic
        self.use_custom_auth = False  # Set to True to use custom auth header

        # Initialize tokenizer - replace with your tokenizer path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "add tokenizer path here for content length estimation",
                trust_remote_code=True
            )
            print(f"✅ Tokenizer loaded successfully")
        except Exception as e:
            print(f"❌ Tokenizer load failed: {str(e)}, will use character length estimation")
            self.tokenizer = None

        print(f"✅ ContentSummarizerInitialize: model={model_name}")

        # Error log file - relative to script directory
        import os
        self.error_log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summarizer_errors.log")
        self.log_lock = Lock()

    def _log_error(self, error_msg: str):
        """Log error to file"""
        try:
            with self.log_lock:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.error_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{timestamp}] {error_msg}\n")
        except Exception:
            pass  # Log write failure does not affect main flow

    def summarize_content(self, title: str, content: str, goal: str) -> str:
        """
        Extract relevant paragraphs from page content based on goal, supports chunking for long content by token count
        
        Args:
            title: Page title
            content: Full page content (clean text extracted by Jina etc)
            goal: User information goal
            
        Returns:
            Extracted relevant paragraphs
        """
        # If content is too short, return directly
        if len(content) < 500:
            return content
        
        # If tokenizer not loaded, use original logic
        if not self.tokenizer:
            return self._summarize_single_chunk(title, content, goal)
            
        # Check content token count
        content_tokens = len(self.tokenizer.encode(content))
        print(f"📊 Original content tokens: {content_tokens}")

        # If exceeds max total tokens, return unable to fetch
        if content_tokens > self.max_total_tokens:
            print(f"⚠️ Content exceeds {self.max_total_tokens} tokens，unable to process")
            return f"[Content too long: {content_tokens} tokens exceeds limit of {self.max_total_tokens}. Unable to access information.]"

        # If content does not exceed max chunk size, process directly
        if content_tokens <= self.max_chunk_tokens:
            return self._summarize_single_chunk(title, content, goal)
        
        # Need to chunk
        chunks = self._split_content_by_tokens(content, self.max_chunk_tokens)
        print(f"📊 Content split into {len(chunks)} chunks")
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            chunk_tokens = len(self.tokenizer.encode(chunk))
            print(f"🔄 Processing chunk {i+1}/{len(chunks)} chunks (tokens: {chunk_tokens})")
            
            try:
                chunk_summary = self._summarize_single_chunk(
                    title=f"{title} (Part {i+1}/{len(chunks)})",
                    content=chunk,
                    goal=goal
                )
                chunk_summaries.append(chunk_summary)
            except Exception as e:
                print(f"❌ Chunk {i+1} chunksSummary failed: {str(e)}")
                # On failure use fallback
                fallback_summary = self._fallback_extract(chunk, goal)
                chunk_summaries.append(fallback_summary)
        
        # Merge all chunk summaries
        combined_summary = "\n\n=== Content Part Separator ===\n\n".join(chunk_summaries)
        
        # Check if merged summary is too long, may need further compression
        final_tokens = len(self.tokenizer.encode(combined_summary))
        print(f"📊 Final summary tokens: {final_tokens}")
        
        return combined_summary
    
    def _split_content_by_tokens(self, content: str, max_tokens: int) -> List[str]:
        """
        Split content into chunks by token count
        
        Args:
            content: Content to split
            max_tokens: Max tokens per chunk
            
        Returns:
            List[str]: List of content chunks
        """
        # First split by paragraphs
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = len(self.tokenizer.encode(paragraph))
            
            # If single paragraph exceeds limit, need to split by sentences
            if paragraph_tokens > max_tokens:
                # Save current chunk first (if any)
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                
                # Process long paragraph
                long_chunks = self._split_long_paragraph(paragraph, max_tokens)
                chunks.extend(long_chunks)
                continue
            
            # Check if adding this paragraph would exceed limit
            if current_tokens + paragraph_tokens > max_tokens:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk
                current_chunk = paragraph
                current_tokens = paragraph_tokens
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += paragraph_tokens
        
        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_long_paragraph(self, paragraph: str, max_tokens: int) -> List[str]:
        """
        Split long paragraph
        """
        # Split by sentences
        sentences = re.split(r'[.!?]\s+', paragraph)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            # If single sentence exceeds limit, need hard split by tokens
            if sentence_tokens > max_tokens:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                
                # Hard split long sentence
                hard_chunks = self._split_by_token_limit(sentence, max_tokens)
                chunks.extend(hard_chunks)
                continue
            
            # Check if adding this sentence would exceed limit
            if current_tokens + sentence_tokens > max_tokens:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
                current_tokens += sentence_tokens
        
        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_token_limit(self, text: str, max_tokens: int) -> List[str]:
        """
        Hard split text by token limit
        """
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word
            test_tokens = len(self.tokenizer.encode(test_chunk))
            
            if test_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk = test_chunk
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _summarize_single_chunk(self, title: str, content: str, goal: str) -> str:
        """
        Summarize single chunk (original logic)
        """
        # New prompt - output JSON format
        user_prompt = f"""Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rationale", "evidence", "summary" fields**"""

        messages = [
            {"role": "user", "content": user_prompt}
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.2
        }

        # Use standard OpenAI authentication
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Uncomment below to use custom authentication
        # if self.use_custom_auth:
        #     headers = {
        #         "Content-Type": "application/json",
        #         "Api-Key": self.api_key,
        #         "X-TT-LOGID": ""
        #     }
        
        # Add retry mechanism, following original file pattern
        max_retries = 6
        base_sleep_time = 1
        
        for retry_count in range(max_retries + 1):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    extracted_content = result['choices'][0]['message']['content'].strip()

                    # If summary is too short and original is not long, return original
                    if len(extracted_content) < 100 and len(content) < 3000:
                        return content

                    # Parse JSON, only extract evidence and summary fields
                    parsed_result = self._parse_json_output(extracted_content)
                    return parsed_result
                else:
                    error_msg = f"[EP: {self.model_name}] Summary API request failed (attempt {retry_count + 1}/{max_retries + 1}): {response.status_code} - {response.text[:200]}"
                    print(error_msg)
                    self._log_error(error_msg)

                    if retry_count < max_retries:
                        sleep_time = base_sleep_time * (2 ** retry_count)
                        print(f"Waiting {sleep_time} seconds before retry...")
                        time.sleep(sleep_time)
                    else:
                        print(f"[EP: {self.model_name}] Summary max retries reached，using fallback")
                        break

            except requests.exceptions.Timeout as e:
                error_msg = f"[EP: {self.model_name}] Summary request timeout (attempt {retry_count + 1}/{max_retries + 1}): {str(e)}"
                print(error_msg)
                self._log_error(error_msg)
                if retry_count < max_retries:
                    sleep_time = base_sleep_time * (2 ** retry_count)
                    print(f"Waiting {sleep_time} seconds before retry...")
                    time.sleep(sleep_time)
                else:
                    print(f"[EP: {self.model_name}] Summary max retries reached，using fallback")
                    break

            except Exception as e:
                error_msg = f"[EP: {self.model_name}] Summary request exception (attempt {retry_count + 1}/{max_retries + 1}): {str(e)}"
                print(error_msg)
                self._log_error(error_msg)
                if retry_count < max_retries:
                    sleep_time = base_sleep_time * (2 ** retry_count)
                    print(f"Waiting {sleep_time} seconds before retry...")
                    time.sleep(sleep_time)
                else:
                    print(f"[EP: {self.model_name}] Summary max retries reached，using fallback")
                    break
        
        # If all retries fail, use simple keyword matching as fallback
        return self._fallback_extract(content, goal)

    def _parse_json_output(self, content: str) -> str:
        """
        Parse JSON output, extract evidence and summary fields, return as string
        """
        try:
            # Try to parse JSON directly
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    return content  # Parse failed, return original content
            else:
                # Try to extract brace content
                brace_match = re.search(r'\{[^{}]*"evidence"[^{}]*\}', content, re.DOTALL)
                if brace_match:
                    try:
                        data = json.loads(brace_match.group(0))
                    except json.JSONDecodeError:
                        return content
                else:
                    return content

        # Extract evidence and summary fields
        evidence = data.get("evidence", "")
        summary = data.get("summary", "")

        # Concatenate into string
        if evidence and summary:
            return f'"evidence": "{evidence}", "summary": "{summary}"'
        elif evidence:
            return f'"evidence": "{evidence}"'
        elif summary:
            return f'"summary": "{summary}"'
        else:
            return content  # If neither exists, return original content

    def _fallback_extract(self, content: str, goal: str) -> str:
        """
        Fallback simple keyword matching extraction
        """
        try:
            # Split goal into keywords
            keywords = []
            # Simple tokenization (consider more complex NLP methods)
            words = re.findall(r'\b\w+\b', goal.lower())
            keywords.extend([w for w in words if len(w) > 2])
            
            if not keywords:
                # If no keywords, return first 2000 chars
                return content[:2000] + ("..." if len(content) > 2000 else "")
            
            # Split content by paragraphs
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
            relevant_paragraphs = []
            
            for paragraph in paragraphs:
                paragraph_lower = paragraph.lower()
                # Check if paragraph contains keywords
                if any(keyword in paragraph_lower for keyword in keywords):
                    relevant_paragraphs.append(paragraph)
            
            if relevant_paragraphs:
                result = '\n\n'.join(relevant_paragraphs)
                # Limit length to avoid too long
                if len(result) > 5000:
                    result = result[:5000] + "..."
                return result
            else:
                # If no relevant paragraphs found, return first 2000 chars
                return content[:2000] + ("..." if len(content) > 2000 else "")
                
        except Exception as e:
            print(f"Fallback extraction failed: {str(e)}")
            return content[:1000] + ("..." if len(content) > 1000 else "")