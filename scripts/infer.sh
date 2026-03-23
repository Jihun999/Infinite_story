#!/bin/bash

# Consistent Story Generation Inference Script
# Uses weight=0.85 (paper default)

CUDA_VISIBLE_DEVICES=0 python3 story_generation.py \
    --gpu_idx 0 \
    --exp_name "./output/story_generation" \
    --infer_type "story" \
    --cfg 3.0 \
    --tau 0.5 \
    --pn 1M \
    --model_path "weights/infinity_2b_reg.pth" \
    --vae_type 32 \
    --vae_path "weights/infinity_vae_d32reg.pth" \
    --add_lvl_embeding_only_first_block 1 \
    --use_bit_label 1 \
    --model_type infinity_2b \
    --rope2d_each_sa_layer 1 \
    --rope2d_normalized_by_hw 2 \
    --use_scale_schedule_embedding 0 \
    --checkpoint_type torch \
    --text_encoder_ckpt google/flan-t5-xl \
    --text_channels 2048 \
    --apply_spatial_patchify 0 \
    --seed 42 \
    --attn_control True \
    --cfg_control True \
    --text_replace True \
    --text_scaling True \
    --cross_scale 1 \
    --weight 0.85 \
    --prompt_path "./prompt/consistory_plus.yaml"
