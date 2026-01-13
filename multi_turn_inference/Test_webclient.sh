#!/bin/bash

# Configure multiple API keys for rotation (comma separated)
export SERPER_API_KEYS="add api key here for Serper search model"

export JINA_API_KEYS="add api key here for Jina reader model"

# Backward compatibility - set single key as fallback
export SERPER_API_KEY="add api key here for Serper search model"
export JINA_API_KEY="add api key here for Jina reader model"

# Whether to block huggingface search results (true/false)
export BLOCK_HUGGINGFACE="false"

# Default parameters
# Support running multiple datasets: add to array separated by spaces
# Can use preset dataset names or full file paths
INPUT_JSONLS=(
    # Add your datasets here, examples:
    # "/path/to/your/dataset.jsonl"
    # "preset_dataset_name"
)

# Output path template, {dataset} will be replaced with current dataset identifier
OUTPUT_TEMPLATE="./output/{dataset}_result.jsonl"
MAX_WORKERS=64
MAX_TOKENS=12800
MAX_TURNS=30
BASE_URLS="http://0.0.0.0:8000/v1,http://0.0.0.0:8001/v1,http://0.0.0.0:8002/v1,http://0.0.0.0:8003/v1"
TEMPERATURE=1.0
TOP_P=1.0
USE_TOOL="--use-tool"  # Enable tools by default, use --no-use-tool to disable

# ========== Anti-repetition Parameters ==========
# Set to true to enable anti-repetition, false to disable
ENABLE_ANTI_REPEAT="false"
# repetition_penalty: Penalize all previously appeared tokens, >1 reduces repetition (recommended: 1.05-1.15)
REPETITION_PENALTY=1.1
# frequency_penalty: Penalize based on token occurrence count, more occurrences = more penalty (range: -2 to 2, recommended: 0.3-0.7)
FREQUENCY_PENALTY=0
# presence_penalty: Penalize if token appeared at all, regardless of count (range: -2 to 2, recommended: 0.2-0.5)
PRESENCE_PENALTY=0

# ========== Prompt Mode Parameters ==========
# finish: Use finish tool to end conversation
# answer: Use <answer></answer> tags to end conversation
PROMPT_MODE="answer"

# ========== Dynamic Answer Monitoring Parameters ==========
# When enabled, conversation ends early if model answer stabilizes within sliding window
# Set to true to enable answer monitoring, false to disable
ENABLE_ANSWER_MONITOR="true"
# Sliding window size: consecutive turns with same answer to consider stable (recommended: 8)
ANSWER_MONITOR_WINDOW=20
# Whether to use LLM for answer extraction (more accurate but slower, default uses regex)
ANSWER_MONITOR_USE_LLM="true"
# Model for answer monitoring (leave empty to use summary model)
ANSWER_MONITOR_MODEL=""

# Summary model parameters - Using GPT-4o API
# Support multiple endpoint rotation with weight distribution:
#   - Single: "model-name"
#   - Multiple equal: "model1,model2"
#   - With weights: "model1:3,model2:1" (model1 called 3 times, model2 called 1 time, i.e. 75%:25%)
SUMMARY_API_BASE="add api base here for GPT-4o summary model"
SUMMARY_API_KEY="add api key here for GPT-4o summary model"
SUMMARY_MODEL_NAME="gpt-4o"
# Run each dataset sequentially
for INPUT_JSONL in "${INPUT_JSONLS[@]}"; do
    # Derive output path
    DATASET_TAG="$INPUT_JSONL"
    if [[ -f "$INPUT_JSONL" ]]; then
        DATASET_TAG=$(basename "$INPUT_JSONL")
        DATASET_TAG=${DATASET_TAG%.jsonl}
    fi
    OUTPUT_JSONL=${OUTPUT_TEMPLATE//\{dataset\}/$DATASET_TAG}

    # Display current parameters
    echo "==================== Run Parameters ===================="
    echo "Input file:     $INPUT_JSONL"
    echo "Output file:    $OUTPUT_JSONL"
    echo "Workers:        $MAX_WORKERS"
    echo "Max tokens:     $MAX_TOKENS"
    echo "Max turns:      $MAX_TURNS"
    echo "API address:    $BASE_URLS"
    echo "Temperature:    $TEMPERATURE"
    echo "Top-p:          $TOP_P"
    echo "Anti-repeat:    $ENABLE_ANTI_REPEAT"
    if [[ "$ENABLE_ANTI_REPEAT" == "true" ]]; then
        echo "  - repetition_penalty: $REPETITION_PENALTY"
        echo "  - frequency_penalty:  $FREQUENCY_PENALTY"
        echo "  - presence_penalty:   $PRESENCE_PENALTY"
    fi
    echo "Prompt mode:    $PROMPT_MODE"
    echo "Answer monitor: $ENABLE_ANSWER_MONITOR"
    if [[ "$ENABLE_ANSWER_MONITOR" == "true" ]]; then
        echo "  - Window size:    $ANSWER_MONITOR_WINDOW"
        echo "  - Use LLM:        $ANSWER_MONITOR_USE_LLM"
        if [[ -n "$ANSWER_MONITOR_MODEL" ]]; then
            echo "  - Monitor model:  $ANSWER_MONITOR_MODEL"
        fi
    fi
    echo "========================================================"

    # Build Python command
    # Get script directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CMD="python3 \"$SCRIPT_DIR/multi_turn_tool_call_local.py\" \"$INPUT_JSONL\" \"$OUTPUT_JSONL\" --workers \"$MAX_WORKERS\" --max-tokens \"$MAX_TOKENS\" --max-turns \"$MAX_TURNS\" --base-urls \"$BASE_URLS\" --temperature \"$TEMPERATURE\" --top-p \"$TOP_P\""

    # Add API key rotation parameters
    if [[ -n "$SERPER_API_KEYS" ]]; then
        CMD="$CMD --serper-keys \"$SERPER_API_KEYS\""
    fi

    if [[ -n "$JINA_API_KEYS" ]]; then
        CMD="$CMD --jina-keys \"$JINA_API_KEYS\""
    fi

    # Add summary parameters if all exist
    if [[ -n "$SUMMARY_API_BASE" && -n "$SUMMARY_API_KEY" && -n "$SUMMARY_MODEL_NAME" ]]; then
        CMD="$CMD --summary-api-base \"$SUMMARY_API_BASE\" --summary-api-key \"$SUMMARY_API_KEY\" --summary-model-name \"$SUMMARY_MODEL_NAME\""
    fi

    # Add anti-repetition parameters
    if [[ "$ENABLE_ANTI_REPEAT" == "true" ]]; then
        CMD="$CMD --repetition-penalty $REPETITION_PENALTY --frequency-penalty $FREQUENCY_PENALTY --presence-penalty $PRESENCE_PENALTY"
    fi

    # Add prompt mode parameter
    CMD="$CMD --prompt-mode $PROMPT_MODE"

    # Add answer monitoring parameters
    if [[ "$ENABLE_ANSWER_MONITOR" == "true" ]]; then
        CMD="$CMD --answer-monitor --answer-monitor-window $ANSWER_MONITOR_WINDOW"
        if [[ "$ANSWER_MONITOR_USE_LLM" == "true" ]]; then
            CMD="$CMD --answer-monitor-llm"
        fi
        if [[ -n "$ANSWER_MONITOR_MODEL" ]]; then
            CMD="$CMD --answer-monitor-model \"$ANSWER_MONITOR_MODEL\""
        fi
    fi

    echo "Executing: $CMD"
    eval $CMD
done
