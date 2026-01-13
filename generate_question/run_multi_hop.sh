#!/bin/bash

# ============================================
# Multi-hop Entity Search Script
# ============================================

# Default parameters
MIN_HOPS=5          # Hop count lower bound (min value when randomly selecting)
MAX_HOPS=10         # Hop count upper bound (max value when randomly selecting)
WORKERS=32          # Concurrency
SAMPLE="10000"      # Random sample count (empty=no sampling)
LIMIT=""            # Processing limit (empty=all)

# File paths
INPUT_FILE="./entity_cache.jsonl"
OUTPUT_FILE="./multi_hop_results.jsonl"
ENTITY_CACHE="./entity_cache.jsonl"
URL_CACHE="./url_cache.jsonl"
CANDIDATE_POOL="./candidate_entity.json"
CANDIDATE_PROB="0.5"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --min-hops)
            MIN_HOPS="$2"
            shift 2
            ;;
        --max-hops)
            MAX_HOPS="$2"
            shift 2
            ;;
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
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --candidate-pool)
            CANDIDATE_POOL="$2"
            shift 2
            ;;
        --candidate-prob)
            CANDIDATE_PROB="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --min-hops NUM      Hop count lower bound (1-15, default: 3)"
            echo "  --max-hops NUM      Hop count upper bound (1-15, default: 10)"
            echo "  -w, --workers NUM   Concurrency (default: 32)"
            echo "  -s, --sample NUM    Random sample count, randomly select N starting points from input"
            echo "  -l, --limit NUM     Sequential limit, take first N"
            echo "  -i, --input FILE    Input file"
            echo "  -o, --output FILE   Output file"
            echo "  --candidate-pool FILE  Candidate entity pool file (default: candidate_entity.json)"
            echo "  --candidate-prob NUM   Probability of selecting from candidate pool (0-1, default: 0.3)"
            echo "  -h, --help          Show help"
            echo ""
            echo "Note:"
            echo "  - Each search randomly selects target hop count within [min-hops, max-hops] range"
            echo "  - --sample is random sampling, --limit takes first N sequentially"
            echo ""
            echo "Examples:"
            echo "  $0 --sample 100 --min-hops 3 --max-hops 10"
            echo "  $0 -s 50 --min-hops 5 --max-hops 8 -w 16"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate hop count
if [ "$MIN_HOPS" -lt 1 ] || [ "$MIN_HOPS" -gt 15 ]; then
    echo "Error: Hop count lower bound must be between 1-15"
    exit 1
fi

if [ "$MAX_HOPS" -lt 1 ] || [ "$MAX_HOPS" -gt 15 ]; then
    echo "Error: Hop count upper bound must be between 1-15"
    exit 1
fi

if [ "$MIN_HOPS" -gt "$MAX_HOPS" ]; then
    echo "Error: Min hops cannot be greater than max hops"
    exit 1
fi

echo "============================================"
echo "Multi-hop Entity Search"
echo "============================================"
echo "Hop range: $MIN_HOPS - $MAX_HOPS (randomly selected each time)"
echo "Workers:   $WORKERS"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Candidate entity pool: $CANDIDATE_POOL"
echo "Candidate pool selection prob: $CANDIDATE_PROB"
if [ -n "$SAMPLE" ]; then
    echo "Random sample: $SAMPLE starting points"
fi
if [ -n "$LIMIT" ]; then
    echo "Sequential limit: $LIMIT"
fi
echo "============================================"
echo ""

# Build command
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="python3 \"$SCRIPT_DIR/multi_hop_search_agent.py\""
CMD="$CMD --input $INPUT_FILE"
CMD="$CMD --output $OUTPUT_FILE"
CMD="$CMD --min-hops $MIN_HOPS"
CMD="$CMD --max-hops $MAX_HOPS"
CMD="$CMD --workers $WORKERS"
CMD="$CMD --entity-cache $ENTITY_CACHE"
CMD="$CMD --url-cache $URL_CACHE"
CMD="$CMD --candidate-pool $CANDIDATE_POOL"
CMD="$CMD --candidate-prob $CANDIDATE_PROB"

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
