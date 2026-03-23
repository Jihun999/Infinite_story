#!/bin/bash

# Evaluation Script for Infinite-Story
# Computes DreamSim, CLIP-T, CLIP-I, DINO metrics

EVAL_DIR=${1:-"./output/story_generation"}
GPU_ID=${2:-0}

python3 evaluate.py \
    --dir "$EVAL_DIR" \
    --gpu "$GPU_ID" \
    --remove_background \
    --output "./eval_results.txt"
