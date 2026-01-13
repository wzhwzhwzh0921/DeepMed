# serper_jina_client.py
import os
import time
import random
from typing import Dict, List, Optional, Union

import requests
from bs4 import BeautifulSoup


class SerperClientError(Exception):
    pass


class Serper_client:
    """
    Simplified Serper + Jina client:
    - search(query): Use Serper /search, output numbered format, store {index: URL} mapping
    - click(index): Use Jina Reader to get clean page content (fallback to requests+BS4)
    """
    def __init__(
        self,
        serper_api_key: Optional[str] = None,
        jina_api_key: Optional[str] = None,
        gl: str = "sg",
        hl: str = "zh-CN",
        symbol: str = "§",
        timeout: int = 30,
        max_retries: int = 2,
        retry_delay: float = 0.8,
        user_agent: str = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
    ):
        # Prefer passed API key, then environment variable
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY") or ""
        if not self.serper_api_key:
            raise SerperClientError("Missing SERPER_API_KEY (parameter or environment variable).")
        self.jina_api_key = jina_api_key or os.getenv("JINA_API_KEY")  # optional

        self.gl = gl
        self.hl = hl
        self.symbol = symbol
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.user_agent = user_agent

        self.index_map: Dict[int, str] = {}
        
        # Add search and visit count statistics
        self.search_count = 0  # search call count
        self.visit_count = 0   # visit call count (including click)

        self._session = requests.Session()
        self._session.headers.update({
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        })

    # ---------- Default blocked sites ----------
    # Controlled by BLOCK_HUGGINGFACE env var (true/false), default blocked
    DEFAULT_BLOCKED_SITES = ["huggingface.co"] if os.getenv("BLOCK_HUGGINGFACE", "true").lower() == "true" else []

    # ---------- 1) Search ----------

    def search(self, query: str, num: int = 10, tbs: Optional[str] = None,
               blocked_sites: Optional[List[str]] = None) -> str:
        """
        Call Serper /search, return formatted results and refresh self.index_map
        Target format:
        Search Results: [§1] Title: xxx Abstract: xxx. url: xxx,
        [§2] Title: xxx Abstract: xxx. url: xxx,

        Args:
            blocked_sites: Sites to block, default blocks huggingface.co
                          Pass empty list [] to cancel default blocking
        """
        # Increment search count
        self.search_count += 1

        # Process blocked sites
        sites_to_block = blocked_sites if blocked_sites is not None else self.DEFAULT_BLOCKED_SITES
        if sites_to_block:
            block_query = " ".join([f"-site:{site}" for site in sites_to_block])
            query = f"{query} {block_query}"

        url = "https://google.serper.dev/search"
        payload = {"q": query, "gl": self.gl, "hl": self.hl, "num": num}
        if tbs:
            payload["tbs"] = tbs

        data = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self._session.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt == self.max_retries:
                    raise SerperClientError(f"Serper search failed (retried {self.max_retries} times):{e}")
                time.sleep(self.retry_delay * (2 ** attempt) + random.uniform(0, 0.5))

        organic = (data or {}).get("organic", []) or []
        if not organic:
            # Keep existing index_map
            return "No relevant results found."

        entries: List[str] = []
        
        # Get current max index，New search results start from next number
        start_index = max(self.index_map.keys()) + 1 if self.index_map else 1

        for i, item in enumerate(organic[:num], start=start_index):
            title = (item.get("title") or "(No title)").strip()
            abstract = (item.get("snippet") or item.get("description") or "").strip()
            link = (item.get("link") or item.get("url") or "").strip()
            self.index_map[i] = link  # Add to existing index_map
            # Output compact one-line format with numbering
            entries.append(f"【{self.symbol}{i}】Title：{title} Abstract：{abstract} 。 url：{link}，")

        # First entry follows "Search Results:", rest on new lines
        result = "\n" + (entries[0] if entries else "")
        if len(entries) > 1:
            result += "\n" + "\n".join(entries[1:])
        return result

    # ---------- 2) Click (via Jina Reader) ----------

    def click(self, index: int, max_chars: Union[int, float] = 50_000) -> Dict[str, str]:
        """
        Click by index and get content:
        - Prefer Jina Reader (r.jina.ai/<URL>, returns JSON: url/title/content/...)
        - If fails/empty, fallback to requests + BeautifulSoup
        """
        # Increment visit count
        self.visit_count += 1
        
        if index not in self.index_map:
            raise SerperClientError(f"Index does not exist. Please call search first or check index range.")
        target_url = self.index_map[index]
        if not target_url:
            raise SerperClientError(f"Index {index} URL is empty.")

        # 1) Jina Reader
        try:
            jurl = f"https://r.jina.ai/{target_url}"
            jheaders = {"Accept": "application/json"}
            if self.jina_api_key:
                jheaders["Authorization"] = f"Bearer {self.jina_api_key}"
            jr = requests.get(jurl, headers=jheaders, timeout=self.timeout)
            jr.raise_for_status()
            j = jr.json()
            payload = j.get("data") if isinstance(j, dict) and "data" in j else j
            title = (payload.get("title") or "").strip() if isinstance(payload, dict) else ""
            content = (payload.get("content") or "").strip() if isinstance(payload, dict) else ""
            if content:
                # If max_chars is infinity, do not truncate
                if max_chars == float('inf'):
                    text = content
                else:
                    text = content[:max_chars] + ("\n...[Content truncated]" if len(content) > max_chars else "")
                final_url = payload.get("url") or target_url
                return {"url": final_url, "title": title or "(No title)", "text": text}
        except Exception as e:
            return {"url": target_url, "title": "Unable to fetch page content", "text": f"Error:{e}"}

    # ---------- 3) Direct URL visit ----------

    def visit(self, url: str, max_chars: Union[int, float] = 50_000, goal: str = None, summarizer=None) -> Dict[str, str]:
        """
        Visit specified URL directly and get content:
        - Prefer Jina Reader (r.jina.ai/<URL>, returns JSON: url/title/content/...)
        - If fails/empty, fallback to requests + BeautifulSoup
        """
        # Increment visit count
        self.visit_count += 1
        
        if not url:
            raise SerperClientError("URL cannot be empty")
        
        # Ensure URL format is correct
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # 1) Jina Reader
        try:
            jurl = f"https://r.jina.ai/{url}"
            jheaders = {"Accept": "application/json"}
            if self.jina_api_key:
                jheaders["Authorization"] = f"Bearer {self.jina_api_key}"
            jr = requests.get(jurl, headers=jheaders, timeout=self.timeout)
            jr.raise_for_status()
            j = jr.json()
            payload = j.get("data") if isinstance(j, dict) and "data" in j else j
            title = (payload.get("title") or "").strip() if isinstance(payload, dict) else ""
            content = (payload.get("content") or "").strip() if isinstance(payload, dict) else ""
            if content:
                # If max_chars is infinity, do not truncate
                if max_chars == float('inf'):
                    text = content
                else:
                    text = content[:max_chars] + ("\n...[Content truncated]" if len(content) > max_chars else "")
                
                # If goal and summarizer provided, summarize content
                if goal and summarizer:
                    try:
                        print(f"🔍 Summarizing page content: {url}")
                        summarized_text = summarizer.summarize_content(
                            title=title or "(No title)",
                            content=text,
                            goal=goal
                        )
                        text = summarized_text
                        print(f"📄 Summary completed")
                    except Exception as e:
                        print(f"⚠️ Summary failed, using original content: {str(e)}")
                
                final_url = payload.get("url") or url
                return {"url": final_url, "title": title or "(No title)", "text": text}
        except Exception as e:
            # 2) Fallback: use requests directly + BeautifulSoup
            try:
                headers = {"User-Agent": self.user_agent}
                response = requests.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title = soup.title.string.strip() if soup.title else "(No title)"
                
                # Extract main content
                # Prioritize common content tags
                content_tags = soup.find_all(['p', 'article', 'main', 'div'], class_=['content', 'article', 'main'])
                if not content_tags:
                    content_tags = soup.find_all(['p', 'div'])
                
                text_parts = []
                for tag in content_tags[:10]:  # Limit tag extraction count
                    text = tag.get_text().strip()
                    if text and len(text) > 20:  # Filter too short text
                        text_parts.append(text)
                
                content = '\n\n'.join(text_parts)
                if content:
                    # If max_chars is infinity, do not truncate
                    if max_chars == float('inf'):
                        text = content
                    else:
                        text = content[:max_chars] + ("\n...[Content truncated]" if len(content) > max_chars else "")
                    
                    # If goal and summarizer provided, summarize content
                    if goal and summarizer:
                        try:
                            print(f"🔍 Summarizing page content: {url} (fallback)")
                            summarized_text = summarizer.summarize_content(
                                title=title,
                                content=text,
                                goal=goal
                            )
                            text = summarized_text
                            print(f"📄 Summary completed")
                        except Exception as e:
                            print(f"⚠️ Summary failed, using original content: {str(e)}")
                    
                    return {"url": url, "title": title, "text": text}
                else:
                    return {"url": url, "title": title, "text": "Unable to extract page content"}
                    
            except Exception as e2:
                return {"url": url, "title": "Unable to fetch page content", "text": f"JinaError:{e}, DirectVisitError:{e2}"}

    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get search and visit count statistics
        """
        return {
            "search_count": self.search_count,
            "visit_count": self.visit_count
        }
    
    def reset_usage_stats(self):
        """
        Reset search and visit count statistics
        """
        self.search_count = 0
        self.visit_count = 0


# ---------- Usage example ----------
if __name__ == "__main__":
    # export SERPER_API_KEY="your-serper-key"
    # export JINA_API_KEY="your-jina-key"  # optional
    client = Serper_client(symbol="§")

    # 1) Search (output formatted results)
    print("=== Search test ===")
    search_result = client.search("what is deepseek r1", num=3, tbs="qdr:w")
    print(search_result)
    print("Index map:", client.index_map)

    # 2) Click (by index)
    if client.index_map:
        print("\n=== Click test ===")
        page = client.click(1)
        print("Title:", page["title"])
        print("URL:", page["url"])
        print("Preview:", page["text"][:200] + "..." if len(page["text"]) > 200 else page["text"])

    # 3) Direct URL visit
    print("\n=== Direct URL visit test ===")
    test_url = "https://www.wikipedia.org"
    try:
        page = client.visit(test_url)
        print("Title:", page["title"])
        print("URL:", page["url"])
        print("Preview:", page["text"][:200] + "..." if len(page["text"]) > 200 else page["text"])
    except Exception as e:
        print(f"Visit failed: {e}")
