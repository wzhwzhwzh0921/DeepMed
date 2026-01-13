# -*- coding: utf-8 -*-
import requests
import json
import os
from datetime import datetime
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import itertools
from serper_client import Serper_client
from summary_manager import summary_manager
from python_client import Python_client
from answer_monitor import answer_monitor_manager

# Set basic params - passed as arguments
api_key = "sk-anything"
model_name = "tongyi-30b-a"

# System prompt - Consistent with training
SYSTEM_PROMPT_FINISH = "You are a Medical deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must call finish tool to give your final answer."

SYSTEM_PROMPT_ANSWER = "You are a Medical deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags."

# Default uses finish mode
SYSTEM_PROMPT = SYSTEM_PROMPT_FINISH

# Tool definitions - Consistent with training
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of query strings. Include multiple complementary search queries in a single call."
                    }
                },
                "required": ["query"]
            },
            "extra_config": {
                "domain_filter": ["huggingface.co/datasets"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Visit webpage(s) and return the summary of the content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
                    },
                    "goal": {
                        "type": "string",
                        "description": "The specific information goal for visiting webpage(s)."
                    }
                },
                "required": ["url", "goal"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call this function to finish the question and give final answer",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "Final answer for the question"}
                },
                "required": ["answer"]
            }
        }
    }
]

# Tool definitions without finish - for answer mode
TOOLS_NO_FINISH = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of query strings. Include multiple complementary search queries in a single call."
                    }
                },
                "required": ["query"]
            },
            "extra_config": {
                "domain_filter": ["huggingface.co/datasets"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Visit webpage(s) and return the summary of the content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
                    },
                    "goal": {
                        "type": "string",
                        "description": "The specific information goal for visiting webpage(s)."
                    }
                },
                "required": ["url", "goal"]
            }
        }
    }
]

# Serper client config (independent instance per thread)
SERPER_CONFIG = {
    "serper_api_key": None,  # Read from env var
    "symbol": "§"
}

# Python client config
PYTHON_CONFIG = {
    "timeout": 30,
    "max_retries": 1,
    "max_output_length": 10000
}

# Create thread lock for file writing
write_lock = threading.Lock()

# Global URL dispatcher
url_cycle = None
url_cycle_lock = threading.Lock()

# Global API key dispatcher
serper_key_cycle = None
serper_key_cycle_lock = threading.Lock()
jina_key_cycle = None
jina_key_cycle_lock = threading.Lock()

def init_url_cycle(base_urls):
    """Initialize URL cycle dispatcher"""
    global url_cycle
    if isinstance(base_urls, str):
        # If string, try split by comma
        if ',' in base_urls:
            url_list = [url.strip() for url in base_urls.split(',')]
        else:
            url_list = [base_urls]
    else:
        url_list = base_urls
    
    url_cycle = itertools.cycle(url_list)
    print(f"Initialized load balancing with model addresses: {url_list}")

def get_next_url():
    """Get next available URL"""
    global url_cycle
    with url_cycle_lock:
        return next(url_cycle) if url_cycle else "http://0.0.0.0:8000/v1"

def init_serper_key_cycle(serper_keys):
    """Initialize Serper API key cycle dispatcher"""
    global serper_key_cycle
    if not serper_keys:
        return
        
    if isinstance(serper_keys, str):
        if ',' in serper_keys:
            key_list = [key.strip() for key in serper_keys.split(',')]
        else:
            key_list = [serper_keys]
    else:
        key_list = serper_keys
    
    serper_key_cycle = itertools.cycle(key_list)
    print(f"Initialized Serper API key load balancing with keys")

def get_next_serper_key():
    """Get next available Serper API key"""
    global serper_key_cycle
    with serper_key_cycle_lock:
        return next(serper_key_cycle) if serper_key_cycle else None

def init_jina_key_cycle(jina_keys):
    """Initialize Jina API key cycle dispatcher"""
    global jina_key_cycle
    if not jina_keys:
        return
        
    if isinstance(jina_keys, str):
        if ',' in jina_keys:
            key_list = [key.strip() for key in jina_keys.split(',')]
        else:
            key_list = [jina_keys]
    else:
        key_list = jina_keys
    
    jina_key_cycle = itertools.cycle(key_list)
    print(f"Initialized Jina API key load balancing with keys")

def get_next_jina_key():
    """Get next available Jina API key"""
    global jina_key_cycle
    with jina_key_cycle_lock:
        return next(jina_key_cycle) if jina_key_cycle else None


def mock_search_tool(tool_name, tool_arguments, serper_client, python_client=None):
    # Track actual search and visit counts，recorded in serper_client
    # Convert to lowercase for model compatibility（e.g. "Visit" -> "visit"）
    tool_name = tool_name.lower()
    if tool_name == "search":
        # Process search list parameter
        query_list = tool_arguments.get('query', [])
        if isinstance(query_list, str):
            query_list = [query_list]  # Compatible with single query
        
        all_results = []
        # Each query calls search once，Stats auto-increment in serper_client.search
        for query in query_list:
            try:
                result = serper_client.search(query)
                all_results.append(f"Query: {query}\n{result}")
            except Exception as e:
                all_results.append(f"Query: {query}\nError: {str(e)}")
        
        return "\n\n=== Next Query ===\n\n".join(all_results)
        
    elif tool_name == "visit":
        # Process visit list parameter
        url_list = tool_arguments.get('url', [])
        goal = tool_arguments.get('goal', '')  # Get user goal
        
        if isinstance(url_list, str):
            url_list = [url_list]  # Compatible with single URL
        
        all_results = []
        # Each URL calls visit once, so stats auto-accumulate in serper_client.visit
        for url in url_list:
            try:
                # Pass goal and summarizer directly to visit，Let it summarize immediately after getting content
                summarizer = summary_manager.summarizer if summary_manager.is_enabled() else None
                page_content = serper_client.visit(
                    url=url, 
                    max_chars=float('inf'),
                    goal=goal,
                    summarizer=summarizer
                )
                
                # Format result
                if goal and summary_manager.is_enabled():
                    result = f"URL: {url}\nTitle: {page_content['title']}\nRelevant Content (Goal: {goal}):\n{page_content['text']}"
                else:
                    result = f"URL: {url}\nTitle: {page_content['title']}\nContent: {page_content['text']}"
                
                all_results.append(result)
            except Exception as e:
                all_results.append(f"URL: {url}\nError: {str(e)}")
        
        return "\n\n=== Next URL ===\n\n".join(all_results)
        
    elif tool_name == "click":
        # Backward compatible, recommend using visit
        try:
            # No truncation, return full content，click also increments visit_count
            page_content = serper_client.click(tool_arguments['index'], max_chars=float('inf'))
            result = f"Title: {page_content['title']}\nURL: {page_content['url']}\nContent: {page_content['text']}"
            return result
        except Exception as e:
            return f"Error clicking on result {tool_arguments['index']}: {str(e)}"
            
    elif tool_name == "python":
        # Process python code execution
        if python_client is None:
            return "Error: Python client not available"
        
        code = tool_arguments.get('code', '')
        if not code:
            return "Error: No Python code provided"
        
        try:
            # Execute Python code
            result = python_client.execute(code, safe_mode=True)
            # Use python_client formatting method
            return python_client.format_result(result)
        except Exception as e:
            return f"Python execution error: {str(e)}"
            
    else:
        # Return simple message
        return f"Tool {tool_name} is not available in this simplified version."

def multi_turn_chat(question=None, options=None, question_type=None, max_tokens=128000, max_turns=10, base_url=None, temperature=1.0, top_p=None, use_tool=True, repetition_penalty=None, frequency_penalty=None, presence_penalty=None, prompt_mode='finish'):
    """
    Multi-turn conversation processing questions and options
    Returns: (prediction, conversation_history, total_turns, end_flag, usage_stats)
    """
    # Select system prompt and tools based on prompt_mode
    if prompt_mode == 'answer':
        system_prompt = SYSTEM_PROMPT_ANSWER
        current_tools = TOOLS_NO_FINISH
    else:
        system_prompt = SYSTEM_PROMPT_FINISH
        current_tools = TOOLS

    # Create answer monitor instance
    answer_monitor = answer_monitor_manager.create_monitor()
    # Set valid options and question content
    if options:
        answer_monitor.set_options(options, question=question)
    # Get current thread API key
    current_serper_key = get_next_serper_key()
    current_jina_key = get_next_jina_key()
    
    # Create independent serper_client for current conversation，Using API key from rotation
    thread_serper_config = SERPER_CONFIG.copy()
    if current_serper_key:
        thread_serper_config["serper_api_key"] = current_serper_key
    if current_jina_key:
        thread_serper_config["jina_api_key"] = current_jina_key
        
    thread_serper_client = Serper_client(**thread_serper_config)
    
    # Create independent python_client for current conversation
    thread_python_client = Python_client(**PYTHON_CONFIG)
    
    # Get actual URL (with load balancing)
    actual_base_url = base_url if base_url else get_next_url()
    
    # Modified to standard OpenAI format headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    url = f"{actual_base_url}/chat/completions"
    print(f"Using model address: {actual_base_url}")
    
    # Initialize conversation messages
    current_question = question 
    
    # Build user message with question and options
    user_message_content = current_question

    if options:
        if isinstance(options, dict):
            option_lines = []
            for k, v in options.items():
                option_lines.append("{}. {}".format(k, v))
            options_text = "Options:\n" + "\n".join(option_lines)
        else:
            options_text = "Options:{}".format(options)
        user_message_content += "\n" + options_text

    # Set prompt based on use_tool - add system prompt
    if use_tool:
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message_content
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message_content
            }
        ]
    
    turn = 0
    prediction = None
    conversation_history = messages.copy()  # Save full conversation history
    end_flag = False  # Record if ended correctly
    last_assistant_content = None
    # Answer source: model(model output), monitor_early_stop(early stop), monitor_fallback(max turns fallback)
    answer_source = None

    while turn < max_turns:
        print(f"\n=== Turn {turn} ===")
        if use_tool:
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                # If server has auto tool calling, enable below
                "tools": current_tools,
                "tool_choice": "auto",
            }
        else:
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

        # Add anti-repetition params (if set)
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if top_p is not None:
            payload["top_p"] = top_p

        # -------- Request with retry --------
        max_retries = 3
        base_sleep_time = 2
        response = None
        for retry_count in range(max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=600)
                if response.status_code == 200:
                    break
                else:
                    print(f"API request failed (attempt {retry_count + 1}/{max_retries + 1}): {response.status_code} - {response.text}")
            except Exception as e:
                print(f"Request exception (attempt {retry_count + 1}/{max_retries + 1}): {str(e)}")

            if retry_count < max_retries:
                sleep_time = base_sleep_time * (2 ** retry_count)
                print(f"Waiting {sleep_time} seconds before retry...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached, exiting this turn")
                break

        if not response or response.status_code != 200:
            break

        # -------- Parse model response (standard tool_calls only)--------
        try:
            data = response.json()
            assistant_message = data["choices"][0]["message"]
            content = assistant_message.get("content") or ""
            tool_calls = assistant_message.get("tool_calls") or []

            print(f"\n💭 --- ASSISTANT RESPONSE ---\n{content}")

            # Dynamic answer monitoring: check answer stability
            is_stable, stable_answer = answer_monitor.update(content)
            if is_stable:
                print(f"🎯 [AnswerMonitor] Answer stability detected, stable answer: {stable_answer}")
                prediction = stable_answer  # Use answer detected by monitor directly
                end_flag = True
                answer_source = "monitor_early_stop"
                break

            # Simple fuse: end if no tool_calls and repeated content
            if content and content == last_assistant_content and not tool_calls:
                print("⚠️ Repeated content detected, stopping to avoid loop")
                prediction = content
                end_flag = True
                break
            last_assistant_content = content

            # Record assistant message (with tool_calls)
            msg_to_append = {"role": "assistant", "content": content}
            if tool_calls:
                msg_to_append["tool_calls"] = tool_calls
            messages.append(msg_to_append)
            conversation_history.append(msg_to_append.copy())

            # ====== Case 1: Standard tool_calls (OpenAI tool calling)======
            if tool_calls:
                finish_called = False
                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = json.loads(tc["function"]["arguments"] or "{}")
                    except Exception:
                        fn_args = {}

                    print(f"\n🛠️ --- TOOL USE (tool_calls): {fn_name} ---\nInput: {fn_args}")

                    # Allow ending with finish tool
                    if fn_name == "finish":
                        prediction = fn_args.get("answer", "")
                        print(f"\n✅ --- FINAL ANSWER ---\n{prediction}")
                        end_flag = True
                        answer_source = "model"
                        finish_called = True
                        break

                    # Execute tool
                    tool_result = mock_search_tool(
                        fn_name, fn_args,
                        serper_client=thread_serper_client,
                        python_client=thread_python_client
                    )
                    print(f"\n📋 --- TOOL RESULT ---\n{tool_result}")

                    # Return tool result (must have same tool_call_id)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "name": fn_name,
                        "content": tool_result,
                    })
                    conversation_history.append(messages[-1].copy())

                if finish_called:
                    break

                # Tool result returned, next turn for model integration
                turn += 1
                continue

            # ====== Case 2: No tool calls -> treat as final answer, end ======
            prediction = content
            end_flag = True
            answer_source = "model"
            print(f"\n✅ --- Model ended correctly (no tool calls)---")
            break

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Request failed: {str(e)}")
            break

    # If not ended correctly (max turns or early exit), try using monitor recorded answer
    if not end_flag:
        print(f"\n⚠️ --- Not ended correctly (turns={turn}, max={max_turns}) ---")
        # Use monitor recorded answer if available
        if answer_monitor_manager.is_enabled():
            monitor_status = answer_monitor.get_status()
            if monitor_status['current_answer']:
                print(f"📝 Using monitor recorded answer: {monitor_status['current_answer']}")
                prediction = monitor_status['current_answer']  # Use answer detected by monitor directly
                end_flag = True  # Mark as ended (via monitor)
                answer_source = "monitor_fallback"

    # If model ended normally without <answer> tag, fallback to monitor answer
    if answer_source == "model" and prediction and "<answer>" not in prediction.lower():
        if answer_monitor_manager.is_enabled():
            monitor_status = answer_monitor.get_status()
            if monitor_status['current_answer']:
                print(f"📝 Model did not use <answer> tag, using monitor detected answer: {monitor_status['current_answer']}")
                prediction = monitor_status['current_answer']
                answer_source = "monitor_fallback"

    # Get search and visit statistics
    usage_stats = thread_serper_client.get_usage_stats()
    
    # Get Python execution statistics
    python_stats = thread_python_client.get_usage_stats()
    
    # Merge statistics
    usage_stats.update({
        'python_execution_count': python_stats['execution_count'],
        'python_success_count': python_stats['success_count'],
        'python_error_count': python_stats['error_count'],
        'python_success_rate': python_stats['success_rate']
    })

    # Add answer source
    usage_stats['answer_source'] = answer_source  # model / monitor_early_stop / monitor_fallback / None

    # Add answer monitor status
    if answer_monitor_manager.is_enabled():
        monitor_status = answer_monitor.get_status()
        usage_stats['answer_monitor'] = {
            'enabled': True,
            'window_size': monitor_status['window_size'],
            'detected_answer': monitor_status['current_answer'],
            'history_length': monitor_status['history_length']
        }
    print(f"\n📊 --- Usage statistics ---")
    print(f"Search count: {usage_stats['search_count']}")
    print(f"Visit count: {usage_stats['visit_count']}")
    print(f"Python execution count: {usage_stats['python_execution_count']}")
    print(f"Python success count: {usage_stats['python_success_count']}")
    print(f"Python success rate: {usage_stats['python_success_rate']}%")
    print(f"Answer source: {answer_source}")
    
    # Auto-save to conversations.jsonl (disabled))
    # save_conversation_to_file(question, options, prediction, conversation_history, turn, usage_stats)

    return prediction, conversation_history, turn, end_flag, usage_stats

# def save_conversation_to_file(question, options, prediction, conversation_history, total_turns, usage_stats=None, file_path="conversations.jsonl"):
#     """
#     Save conversation to file (disabled))
#     """
#     try:
#         # Get current script directory
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         full_path = os.path.join(script_dir, file_path)
#
#         # Construct data to save
#         conversation_data = {
#             "id": str(uuid.uuid4()),
#             "timestamp": datetime.now().isoformat(),
#             "question": question,
#             "options": options,
#             "prediction": prediction,
#             "total_turns": total_turns,
#             "conversation_history": conversation_history,
#             "status": "completed"
#         }
#
#         # Add usage statistics info
#         if usage_stats:
#             conversation_data["usage_stats"] = usage_stats
#
#         # Thread-safe file write
#         with write_lock:
#             with open(full_path, 'a', encoding='utf-8') as f:
#                 json_line = json.dumps(conversation_data, ensure_ascii=False, separators=(',', ':'))
#                 f.write(json_line + '\n')
#
#         print(f"✅ Conversation saved to: {full_path}")
#         return True
#
#     except Exception as e:
#         print(f"⚠️ Error saving conversation: {str(e)}")
#         return False

def write_to_jsonl(data, output_file):
    """
    Thread-safe JSONL write, one JSON object per line
    """
    try:
        with write_lock:  # Ensure thread safety
            with open(output_file, 'a', encoding='utf-8') as f:
                # Ensure single-line JSON write
                json_line = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
                f.write(json_line + '\n')
        return True
    except Exception as e:
        print(f"Error writing to JSONL file: {str(e)}")
        return False

def get_processed_lines(output_jsonl_file):
    """
    Get processed line numbers from output JSONL file
    """
    processed_lines = set()
    if os.path.exists(output_jsonl_file):
        try:
            with open(output_jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'line_number' in data:
                            processed_lines.add(data['line_number'])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading processed line info: {str(e)}")
    return processed_lines

def process_single_question(line_number, data, output_jsonl_file, max_tokens=128000, max_turns=10, base_url=None, temperature=1.0, top_p=None, use_tool=True, repetition_penalty=None, frequency_penalty=None, presence_penalty=None, prompt_mode='finish'):
    """
    Process single question for concurrent execution
    """
    try:
        # Extract question part
        if 'question' not in data:
            print(f"Line has no question field")
            return None
        
        question = data['question']
        # Extract options or choices part
        options = data.get('options', None)
        if options is None:
            options = data.get('choices', None)
        question_type = data.get('question_type', None)
        # Extract answer field (if exists)
        answer = data.get('answer', None)
        # Extract id field (if exists)
        input_id = data.get('id', None)
        
        print(f"\n========= Processing line =========")
        print(f"Question: {question}")
        if options:
            print(f"Options/Choices: {options}")
        if answer:
            print(f"answer: {answer}")
        if input_id:
            print(f"Input ID: {input_id}")
        
        # Call multi_turn_chat to process question and options
        prediction, conversation_history, total_turns, end_flag, usage_stats = multi_turn_chat(question, options, question_type, max_tokens, max_turns, base_url, temperature, top_p, use_tool, repetition_penalty, frequency_penalty, presence_penalty, prompt_mode)
        
        # Extract last turn model response as final_prediction from history
        final_prediction = prediction  # Default uses original prediction

        # Check Answer source: if answer from monitor, use prediction directly, do not override
        answer_source = usage_stats.get('answer_source', None)
        should_extract_from_history = (prediction is None or prediction == "") and answer_source not in ['monitor_early_stop', 'monitor_fallback']

        if should_extract_from_history and conversation_history:
            # Find last assistant message from end
            for message in reversed(conversation_history):
                if isinstance(message, dict) and message.get('role') == 'assistant':
                    # Check if has content field
                    if 'content' in message and message['content']:
                        final_prediction = message['content']
                        break
                    # Check if has tool call but no clear answer
                    elif '<tool_call>' in message.get('content', ''):
                        # If no clear prediction but has tool call
                        final_prediction = "Model used tool but provided no clear answer"
                        break
        
        # Prepare JSONL data, use id field from input file
        result_entry = {
            'id': input_id,  # Prefer id field from input file
            'line_number': line_number,
            'question': question,
            'options': options,
            'prediction': final_prediction,  # Use extracted last turn model response
            'total_turns': total_turns,
            'end_flag': end_flag,  # Add correct end flag
            'answer_source': answer_source,  # Answer source: model / monitor_early_stop / monitor_fallback / None
            'conversation_history': conversation_history,
            'usage_stats': usage_stats,  # Add search and visit statistics
            'timestamp': datetime.now().isoformat()
        }

        # Add answer field to result if exists
        if answer is not None:
            result_entry['answer'] = answer
        
        # Write to JSONL file
        if write_to_jsonl(result_entry, output_jsonl_file):
            print(f"Line result written to: {output_jsonl_file}")
        
        return result_entry
        
    except json.JSONDecodeError:
        print(f"Line is not valid JSON")
        return None
    except Exception as e:
        print(f"Processing lineerror during: {str(e)}")
        return None

def process_jsonl_file(jsonl_file_path, output_jsonl_file=None, max_workers=1, max_tokens=128000, max_turns=10, base_urls="http://0.0.0.0:8000/v1", summary_api_base=None, summary_api_key=None, summary_model_name=None, temperature=1.0, top_p=None, use_tool=True, serper_keys=None, jina_keys=None, repetition_penalty=None, frequency_penalty=None, presence_penalty=None, prompt_mode='finish', answer_monitor_enabled=False, answer_monitor_window=8, answer_monitor_llm=False, answer_monitor_model=None):
    """
    Concurrently process JSONL file, extract question and options per line
    Write result to output_jsonl_file after each question
    Supports resume, ensures written data contains id field
    Uses multi-threading for efficiency
    """
    if not os.path.exists(jsonl_file_path):
        print(f"File does not exist: {jsonl_file_path}")
        return []
    
    # Initialize URL cycle dispatcher
    init_url_cycle(base_urls)
    
    # Initialize API key cycle dispatcher
    init_serper_key_cycle(serper_keys)
    init_jina_key_cycle(jina_keys)
    
    # Initialize content summarizer
    summary_manager.initialize(summary_api_base, summary_api_key, summary_model_name)

    # Initialize answer monitor (using summary model API config)
    answer_monitor_manager.initialize(
        enabled=answer_monitor_enabled,
        window_size=answer_monitor_window,
        use_llm_extract=answer_monitor_llm,
        llm_api_base=summary_api_base,
        llm_api_key=summary_api_key,
        llm_model_name=answer_monitor_model if answer_monitor_model else summary_model_name
    )

    # Generate timestamp-based filename if output not specified
    if not output_jsonl_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_jsonl_file = f"conversation_results_local_{timestamp}.jsonl"
        
    # Check output file, add id field to lines without id if exists
    if os.path.exists(output_jsonl_file):
        print(f"Check and process existing output file: {output_jsonl_file}")
    else:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_jsonl_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Create output directory: {output_dir}")
        
        # Create output file
        open(output_jsonl_file, 'w').close()
        print(f"Create output file: {output_jsonl_file}")
    
    # Get processed line numbers
    processed_lines = get_processed_lines(output_jsonl_file)
    if processed_lines:
        print(f"Detected processed line numbers: {processed_lines}")
        print(f"Will skip these, start from next unprocessed...")
    
    # Collect all pending tasks
    tasks = []
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                # Skip if already processed
                if line_number in processed_lines:
                    print(f"Skipping processed line")
                    continue
                
                try:
                    # Parse JSON line
                    data = json.loads(line.strip())
                    if 'question' in data:
                        tasks.append((line_number, data, output_jsonl_file))
                    else:
                        print(f"Line has no question field")
                except json.JSONDecodeError:
                    print(f"Line is not valid JSON")
                except Exception as e:
                    print(f"Error parsing line: {str(e)}")
                
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return []
    
    print(f"Preparing concurrent processing tasks with threads")
    
    results = []
    completed_count = 0
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks (no base_url, use load balancing)
        future_to_task = {
            executor.submit(process_single_question, line_number, data, output_jsonl_file, max_tokens, max_turns, None, temperature, top_p, use_tool, repetition_penalty, frequency_penalty, presence_penalty, prompt_mode): (line_number, data)
            for line_number, data, output_jsonl_file in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            line_number, data = future_to_task[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                completed_count += 1
                print(f"Progress: completed")
                
                # Add delay to avoid too fast API calls
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Processing lineexception occurred during: {str(e)}")
                completed_count += 1
    
    print(f"All tasks done, processed successful results")
    return results


if __name__ == "__main__":
    import argparse
    # Preset dataset paths - add your own dataset paths as needed
    PRESET_DATASETS = {
        # Example format:
        # 'dataset_name': '/path/to/your/dataset.jsonl',
    }
    
    parser = argparse.ArgumentParser(description='Multi-turn tool call JSONL file processor')
    parser.add_argument('input_jsonl', help='Input JSONL file path or preset dataset name (hle_medical, medbrowsecomp, medxpertqa, pubmedqa, supergpqa)')
    parser.add_argument('output_jsonl', help='Output JSONL file path')
    parser.add_argument('--workers', '-w', type=int, default=1, help='Concurrent worker threads (default: 1)')
    parser.add_argument('--max-tokens', '-t', type=int, default=128000, help='Max model tokens (default: 128000)')
    parser.add_argument('--max-turns', '-r', type=int, default=10, help='Max conversation turns (default: 10)')
    parser.add_argument('--base-urls', '-u', type=str, default="http://0.0.0.0:8000/v1", help='Model API address, comma separated (default: http://0.0.0.0:8000/v1)')
    parser.add_argument('--summary-api-base', type=str, help='Summary model API address')
    parser.add_argument('--summary-api-key', type=str, help='Summary model API key')
    parser.add_argument('--summary-model-name', type=str, help='Summary model name')
    parser.add_argument('--temperature', '-temp', type=float, default=1.0, help='Model temperature for output randomness (default: 1.0)')
    parser.add_argument('--top-p', type=float, default=None, help='Top-p sampling for diversity (recommended: 0.9-0.99)')
    parser.add_argument('--use-tool', action='store_true', default=True, help='Whether to use tools (default: True)')
    parser.add_argument('--no-use-tool', dest='use_tool', action='store_false', help='Disable tool usage')
    parser.add_argument('--serper-keys', type=str, help='Serper API key list, comma separated')
    parser.add_argument('--jina-keys', type=str, help='Jina API key list, comma separated')
    # Anti-repetition parameters
    parser.add_argument('--repetition-penalty', type=float, default=None, help='Repetition penalty, reduces repetition (recommended: 1.05-1.15)')
    parser.add_argument('--frequency-penalty', type=float, default=None, help='Frequency penalty based on token count (range: -2to2)')
    parser.add_argument('--presence-penalty', type=float, default=None, help='Presence penalty for any token occurrence (range: -2to2)')
    # Prompt mode parameters
    parser.add_argument('--prompt-mode', type=str, default='finish', choices=['finish', 'answer'], help='System prompt mode: finish uses finish tool, answer uses <answer> tag (default: finish)')

    # Answer monitoring parameters
    parser.add_argument('--answer-monitor', action='store_true', default=False, help='Enable dynamic answer monitoring (sliding window stability detection)')
    parser.add_argument('--answer-monitor-window', type=int, default=8, help='Answer monitor window size, consecutive turns for stability (default: 8)')
    parser.add_argument('--answer-monitor-llm', action='store_true', default=False, help='Use LLM for answer extraction (more accurate but slower, default uses regex)')
    parser.add_argument('--answer-monitor-model', type=str, default=None, help='LLM model name for answer monitoring (requires --answer-monitor-llm)')

    args = parser.parse_args()
    
    # Check if preset dataset name
    input_file = args.input_jsonl
    if input_file in PRESET_DATASETS:
        input_file = PRESET_DATASETS[input_file]
        print(f"Using preset dataset '{args.input_jsonl}': {input_file}")
    elif not os.path.exists(input_file):
        print(f"error: File does not exist '{input_file}'")
        print(f"Available preset datasets: {list(PRESET_DATASETS.keys())}")
        exit(1)
    
    print(f"=== Processing JSONL file: {input_file} ===")
    print(f"Prompt mode: {args.prompt_mode}")
    if args.repetition_penalty or args.frequency_penalty or args.presence_penalty:
        print(f"Anti-repetition parameters: repetition_penalty={args.repetition_penalty}, frequency_penalty={args.frequency_penalty}, presence_penalty={args.presence_penalty}")
    if args.answer_monitor:
        print(f"Answer monitoring: Enabled")
        print(f"  - Sliding window size: {args.answer_monitor_window}")
        print(f"  - Use LLM extraction: {args.answer_monitor_llm}")
        if args.answer_monitor_model:
            print(f"  - Monitor model: {args.answer_monitor_model}")
    results = process_jsonl_file(
        input_file,
        args.output_jsonl,
        args.workers,
        args.max_tokens,
        args.max_turns,
        args.base_urls,
        args.summary_api_base,
        args.summary_api_key,
        args.summary_model_name,
        args.temperature,
        args.top_p,
        args.use_tool,
        args.serper_keys,
        args.jina_keys,
        args.repetition_penalty,
        args.frequency_penalty,
        args.presence_penalty,
        args.prompt_mode,
        args.answer_monitor,
        args.answer_monitor_window,
        args.answer_monitor_llm,
        args.answer_monitor_model
    )
    print(f"Processing complete, total results")