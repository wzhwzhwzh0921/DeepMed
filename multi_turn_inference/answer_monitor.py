# -*- coding: utf-8 -*-
"""
Dynamic answer monitoring module
Detects model answer stability using sliding window detection
"""

import re
import requests
import json

class AnswerMonitor:
    """
    Sliding window answer stability detector
    - Records model answer each turn
    - Triggers stable signal if answers match for window_size turns
    """

    def __init__(self, enabled=False, window_size=6, use_llm_extract=False,
                 llm_api_base=None, llm_api_key=None, llm_model_name=None):
        """
        Initialize answer monitor

        Args:
            enabled: Whether monitoring is enabled
            window_size: Sliding window size (consecutive turns with same answer for stability)
            use_llm_extract: Whether to use LLM for answer extraction (more accurate but slower)
            llm_api_base: LLM API address
            llm_api_key: LLM API key
            llm_model_name: LLM model name (supports weight format, auto-strips weight suffix)
        """
        self.enabled = enabled
        self.window_size = window_size
        self.use_llm_extract = use_llm_extract
        self.llm_api_base = llm_api_base
        self.llm_api_key = llm_api_key
        # Parse model name, strip possible weight suffix (e.g. "model-name:1" -> "model-name")
        if llm_model_name and ':' in llm_model_name:
            # Take first model before comma, then strip weight
            first_model = llm_model_name.split(',')[0].strip()
            if ':' in first_model:
                self.llm_model_name = first_model.rsplit(':', 1)[0].strip()
            else:
                self.llm_model_name = first_model
        else:
            self.llm_model_name = llm_model_name

        # Answer history
        self.answer_history = []
        self.current_answer = None
        # Valid options list
        self.valid_options = None
        self.num_options = 0
        # Question and options content (for LLM extraction)
        self.question = None
        self.options_content = None

    def reset(self):
        """Reset monitor state (called for each new question)"""
        self.answer_history = []
        self.current_answer = None
        self.valid_options = None
        self.num_options = 0
        self.question = None
        self.options_content = None

    def set_options(self, options, question=None):
        """
        Set valid options for current question

        Args:
            options: Options list，e.g. ['option A text', 'option B text', ...]
            question: Question text
        """
        self.question = question
        if options and isinstance(options, list):
            self.num_options = len(options)
            # Generate valid option letter list，e.g. ['A', 'B', 'C', 'D', 'E']
            self.valid_options = [chr(ord('A') + i) for i in range(self.num_options)]
            # Save option content, formatted as "A: xxx\nB: xxx\n..."
            self.options_content = '\n'.join([f"{chr(ord('A') + i)}: {opt}" for i, opt in enumerate(options)])
        else:
            self.valid_options = None
            self.num_options = 0
            self.options_content = None

    def extract_answer_regex(self, text):
        """
        Extract answer from text using regex
        Supports A-E options
        """
        if not text:
            return None

        patterns = [
            # Explicit answer declaration (supports A-Z))
            r'(?:Thus|So|Therefore)[,\s]+(?:the\s+)?answer[:\s]+([A-Z])',
            r'(?:The\s+)?answer\s+is[:\s]+([A-Z])',
            r'(?:The\s+)?correct\s+answer\s+is[:\s]+([A-Z])',
            r'answer[:\s]+([A-Z])\b',
            r'answer\s+([A-Z])\.',
            # Choice declaration
            r'(?:I\s+)?(?:would\s+)?choose[:\s]+([A-Z])',
            r'(?:The\s+)?best\s+answer\s+is[:\s]+([A-Z])',
            # Bold format
            r'\*\*([A-Z])\*\*',
            # answer tag
            r'<answer>\s*([A-Z])',
            # yes/no/maybe (for PubMedQA style)
            r'(?:The\s+)?answer\s+is[:\s]+(yes|no|maybe)',
            r'answer[:\s]+(yes|no|maybe)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                ans = matches[-1].upper()  # Take last match
                # Convert yes/no/maybe
                if ans == 'YES':
                    return 'A'
                elif ans == 'NO':
                    return 'B'
                elif ans == 'MAYBE':
                    return 'C'
                return ans

        return None

    def extract_answer_llm(self, text, question=None):
        """
        Extract answer using LLM (more accurate but slower)
        """
        if not self.llm_api_base or not self.llm_api_key or not self.llm_model_name:
            return None  # No fallback to regex

        try:
            # Use actual valid options，Default if not setA-J
            if self.valid_options:
                valid_choices = self.valid_options
                options_str = ', '.join(valid_choices)
                last_option = valid_choices[-1]
            else:
                valid_choices = [chr(ord('A') + i) for i in range(10)]  # default A-J
                options_str = 'A, B, C, D, E, F, G, H, I, J'
                last_option = 'J'

            # Build context with question and options
            context_parts = []
            if self.question:
                context_parts.append(f"Question: {self.question}")
            if self.options_content:
                context_parts.append(f"Options:\n{self.options_content}")
            question_context = '\n\n'.join(context_parts) if context_parts else ''

            prompt = f"""Extract the answer choice from the model's response text.

{question_context}

IMPORTANT: The ONLY valid answer choices are: {options_str}

Rules:
1. Look for explicit answer statements like "the answer is X", "I choose X", "correct answer: X"
2. If the text mentions an option by its CONTENT (e.g., "the answer is Islam" when option E is "Islam"), map it to the correct letter (E)
3. The answer MUST be one of: {options_str}
4. Any letter beyond '{last_option}' is INVALID
5. If no clear answer is found, respond with "NONE"

Model's response:
{text[-1500:]}

Answer (must be one of {options_str}, or NONE):"""

            # Use standard OpenAI authentication
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.llm_api_key}'
            }

            payload = {
                'model': self.llm_model_name,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 10,
                'temperature': 0
            }

            response = requests.post(
                f"{self.llm_api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip().upper()
                # Take first character only，Avoid patterns like
                if content and content[0] in valid_choices:
                    return content[0]

            # LLM extraction failed, return None
            return None

        except Exception as e:
            print(f"[AnswerMonitor] LLM answer extraction failed: {e}")
            return None

    def update(self, assistant_content):
        """
        Update answer state

        Args:
            assistant_content: Model response content this turn

        Returns:
            (is_stable, stable_answer): Whether stable, stable answer
        """
        if not self.enabled:
            return False, None

        # Extract answer this turn
        if self.use_llm_extract:
            extracted = self.extract_answer_llm(assistant_content)
        else:
            extracted = self.extract_answer_regex(assistant_content)

        # Update current answer if extracted
        if extracted:
            self.current_answer = extracted

        # Record current answer state this turn
        self.answer_history.append(self.current_answer)

        # Check sliding window
        if len(self.answer_history) >= self.window_size:
            window = self.answer_history[-self.window_size:]
            # If all answers in window same and not None
            if window[0] is not None and all(a == window[0] for a in window):
                print(f"[AnswerMonitor] Answer stability detected: {window[0]} (consecutive turns)")
                return True, window[0]

        return False, None

    def get_status(self):
        """Get current monitor status"""
        return {
            'enabled': self.enabled,
            'window_size': self.window_size,
            'current_answer': self.current_answer,
            'history_length': len(self.answer_history),
            'answer_history': self.answer_history[-10:] if self.answer_history else []  # Return only last 10 turns
        }


# Global answer monitor management
class AnswerMonitorManager:
    """Answer monitor manager class (global config)"""

    def __init__(self):
        self.enabled = False
        self.window_size = 6
        self.use_llm_extract = False
        self.llm_api_base = None
        self.llm_api_key = None
        self.llm_model_name = None

    def initialize(self, enabled=False, window_size=6, use_llm_extract=False,
                   llm_api_base=None, llm_api_key=None, llm_model_name=None):
        """Initialize global config"""
        self.enabled = enabled
        self.window_size = window_size
        self.use_llm_extract = use_llm_extract
        self.llm_api_base = llm_api_base
        self.llm_api_key = llm_api_key
        self.llm_model_name = llm_model_name

        if enabled:
            print(f"[AnswerMonitor] Dynamic answer monitoring enabled")
            print(f"  - Sliding window size: {window_size}")
            print(f"  - Using LLM extraction: {use_llm_extract}")
            if use_llm_extract and llm_model_name:
                print(f"  - Monitor model: {llm_model_name}")

    def create_monitor(self):
        """Create independent monitor instance for each question"""
        return AnswerMonitor(
            enabled=self.enabled,
            window_size=self.window_size,
            use_llm_extract=self.use_llm_extract,
            llm_api_base=self.llm_api_base,
            llm_api_key=self.llm_api_key,
            llm_model_name=self.llm_model_name
        )

    def is_enabled(self):
        return self.enabled


# Global instance
answer_monitor_manager = AnswerMonitorManager()
