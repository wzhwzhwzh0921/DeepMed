#!/bin/bash

# ============================================
# Multi-hop Path Question Generation Script
# ============================================

# Default parameters
WORKERS=128
SAMPLE=""
LIMIT=""
MIN_PATH_LENGTH=3
MODEL="gpt-4o"
NUMERICAL_PROB="0.7"
ANSWER_MODE="forward-last"  # forward-last or forward-first

# File paths
INPUT_FILE="./multi_hop_results.jsonl"
OUTPUT_FILE="./generated_questions.jsonl"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -s|--sample)
            SAMPLE="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        --min-path-length)
            MIN_PATH_LENGTH="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --numerical-prob)
            NUMERICAL_PROB="$2"
            shift 2
            ;;
        --answer-mode)
            ANSWER_MODE="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -w, --workers NUM        Concurrency (default: 16)"
            echo "  -s, --sample NUM         Random sample count"
            echo "  -l, --limit NUM          Sequential limit count"
            echo "  --min-path-length NUM    Minimum path length (default: 3)"
            echo "  --model MODEL            Model to use (default: gpt-4o)"
            echo "  --numerical-prob PROB    Numerical/dosage question probability (default: 0.3)"
            echo "  --answer-mode MODE       Answer mode (default: forward-last)"
            echo "                           forward-last: Follow chain, answer is last node"
            echo "                           forward-first: Follow chain, answer is first node"
            echo "  -i, --input FILE         Input file (multi-hop search results)"
            echo "  -o, --output FILE        Output file"
            echo "  -h, --help               Show help"
            echo ""
            echo "Answer mode description:"
            echo "  forward-last:  Chain A->B->C->D, question follows order, answer is D"
            echo "  forward-first: Chain A->B->C->D, question follows order, answer is A (first node has less description)"
            echo ""
            echo "Examples:"
            echo "  $0 --sample 100 -w 32"
            echo "  $0 --answer-mode forward-first --limit 50"
            echo "  $0 --answer-mode forward-last --min-path-length 5"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Multi-hop Path Question Generation"
echo "============================================"
echo "Input file:     $INPUT_FILE"
echo "Output file:    $OUTPUT_FILE"
echo "Workers:        $WORKERS"
echo "Min path length: $MIN_PATH_LENGTH"
echo "Model: $MODEL"
echo "Numerical/dosage prob: $NUMERICAL_PROB"
echo "Answer mode: $ANSWER_MODE"
if [ -n "$SAMPLE" ]; then
    echo "Random sample: $SAMPLE"
fi
if [ -n "$LIMIT" ]; then
    echo "Sequential limit: $LIMIT"
fi
echo "============================================"
echo ""

# Build command
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="python3 \"$SCRIPT_DIR/generate_questions_multihop.py\""
CMD="$CMD --input $INPUT_FILE"
CMD="$CMD --output $OUTPUT_FILE"
CMD="$CMD --workers $WORKERS"
CMD="$CMD --min-path-length $MIN_PATH_LENGTH"
CMD="$CMD --model $MODEL"
CMD="$CMD --numerical-prob $NUMERICAL_PROB"
CMD="$CMD --answer-mode $ANSWER_MODE"

if [ -n "$SAMPLE" ]; then
    CMD="$CMD --sample $SAMPLE"
fi

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Run
echo "Running command: $CMD"
echo ""

$CMD

echo ""
echo "============================================"
echo "Done!"
echo "Output file: $OUTPUT_FILE"
echo "============================================"
