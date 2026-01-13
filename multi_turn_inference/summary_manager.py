# -*- coding: utf-8 -*-
import os
import itertools
import threading
from typing import Optional, List
from content_summarizer import ContentSummarizer

class SummaryManager:
    """
    Summary manager that manages and provides content summarization services
    Supports multiple endpoint rotation for load balancing
    """

    def __init__(self):
        self.summarizers: List[ContentSummarizer] = []
        self.summarizer_cycle = None
        self.cycle_lock = threading.Lock()
        self.enabled = False
        # Backward compatibility
        self.summarizer: Optional[ContentSummarizer] = None

    def initialize(self, api_base: Optional[str], api_key: Optional[str], model_name: Optional[str]):
        """
        Initialize summarizer with multiple model_name and weight distribution support

        Args:
            api_base: Summary model API address
            api_key: Summary model API key
            model_name: Summary model name, supports formats:
                - Single: "model-name"
                - Multiple equal: "model-1,model-2,model-3"
                - With weights: "model-1:3,model-2:1,model-3:2" (indicates call frequency)
        """
        if api_base and api_key and model_name:
            try:
                # Parse model_name with weight distribution
                # Format: "model-1:3,model-2:1" or "model-1,model-2"
                weighted_list = []  # List expanded by weight
                model_weights = []  # List for display

                for item in model_name.split(','):
                    item = item.strip()
                    if not item:
                        continue

                    if ':' in item:
                        # With weights format: model-name:3
                        name, weight_str = item.rsplit(':', 1)
                        name = name.strip()
                        try:
                            weight = int(weight_str.strip())
                        except ValueError:
                            weight = 1
                    else:
                        # Without weight, default to 1
                        name = item
                        weight = 1

                    weight = max(1, weight)  # At least 1
                    model_weights.append((name, weight))

                    # Create summarizer and add to list by weight
                    summarizer = ContentSummarizer(
                        api_base=api_base,
                        api_key=api_key,
                        model_name=name
                    )
                    # Add repeatedly by weight
                    for _ in range(weight):
                        weighted_list.append(summarizer)

                if weighted_list:
                    self.summarizers = list(set(weighted_list))  # Dedupe and save unique instances
                    self.summarizer_cycle = itertools.cycle(weighted_list)  # Rotate by weight
                    self.summarizer = self.summarizers[0]
                    self.enabled = True

                    total_weight = sum(w for _, w in model_weights)
                    print(f"✅ Summarizer initialized successfully: {len(model_weights)} endpoints, total weight {total_weight}")
                    for name, weight in model_weights:
                        ratio = weight / total_weight * 100
                        print(f"   - {name} (weight: {weight}, ratio: {ratio:.1f}%)")
                else:
                    self.enabled = False

            except Exception as e:
                print(f"❌ Summarizer initialization failed: {str(e)}")
                self.enabled = False
        else:
            print("💡 Summarizer params incomplete, content summarization disabled")
            self.enabled = False

    def get_next_summarizer(self) -> Optional[ContentSummarizer]:
        """
        Get next available summarizer (rotation)

        Returns:
            ContentSummarizer: Next summarizer instance
        """
        if not self.summarizer_cycle:
            return self.summarizer

        with self.cycle_lock:
            return next(self.summarizer_cycle)

    def is_enabled(self) -> bool:
        """
        Check if summarizer is available

        Returns:
            bool: Whether summarizer is enabled and available
        """
        return self.enabled and len(self.summarizers) > 0

    def summarize_if_needed(self, title: str, content: str, goal: str) -> str:
        """
        Summarize if enabled and goal provided; otherwise return original

        Args:
            title: Page title
            content: Full page content (clean text extracted by Jina etc)
            goal: User information goal

        Returns:
            str: Summarized or original content
        """
        if self.is_enabled() and goal and goal.strip():
            try:
                # Get next summarizer (round-robin)
                summarizer = self.get_next_summarizer()
                if not summarizer:
                    return content

                print(f"🔍 Summarizing page content (Goal: {goal[:50]}...) [model: {summarizer.model_name}]")
                summarized_content = summarizer.summarize_content(
                    title=title,
                    content=content,
                    goal=goal
                )
                print(f"📄 Summary done, original length: {len(content)}, summary length: {len(summarized_content)}")
                return summarized_content
            except Exception as e:
                print(f"⚠️ Summary failed, using original: {str(e)}")
                return content
        else:
            # No summarizer or goal, return original
            return content

    def format_result(self, url: str, title: str, content: str, goal: Optional[str] = None) -> str:
        """
        Format result output

        Args:
            url: Page URL
            title: Page title
            content: Content (possibly summarized)
            goal: User goal (optional)

        Returns:
            str: Formatted result
        """
        if goal and goal.strip() and self.is_enabled():
            return f"URL: {url}\nTitle: {title}\nRelevant Content (Goal: {goal}):\n{content}"
        else:
            return f"URL: {url}\nTitle: {title}\nContent: {content}"

# Global summary manager instance
summary_manager = SummaryManager()
