import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
from typing import List
import time
import hashlib
import argparse
import shutil
import re

import cv2
import numpy as np
import torch
torch._dynamo.config.cache_size_limit = 64
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
from PIL import Image, ImageEnhance
import torch.nn.functional as F
from torch.cuda.amp import autocast

import PIL.Image as PImage
from torchvision.transforms.functional import to_tensor
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates


def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False):
    if enable_positive_prompt:
        prompt = aug_with_positive_prompt(prompt)
    captions = [prompt]

    tokens = text_tokenizer(text=captions, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)
    kv_compact = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)
    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple


def cal_scale_difference(common_feature, common_len):
    """
    Compute scale ratio between anchor and variant text features.
    common_feature: [N, L, D] where N = batch size (1 anchor + variants)
    common_len: number of common prefix tokens
    """
    anchor_feature = common_feature[0, :common_len]
    diff_feature = common_feature[1:, :common_len]

    anchor_feature = anchor_feature.unsqueeze(0).expand_as(diff_feature)

    anchor_norm = anchor_feature.norm(dim=-1)
    diff_norm = diff_feature.norm(dim=-1)

    scale_ratio = anchor_norm / diff_norm
    mean_scale_ratio = scale_ratio.mean(dim=1, keepdim=True)

    return mean_scale_ratio



def encode_prompt_batch(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False, infer_args=None):
    if enable_positive_prompt:
        prompt = [aug_with_positive_prompt(p) for p in prompt]
    if not isinstance(prompt, list):
        prompt = [prompt]

    tokens = text_tokenizer(text=prompt, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    diff_pos = (input_ids != input_ids[0]).int().cumsum(dim=1)
    match_counts = (diff_pos == 0).sum(dim=1)
    common_len = match_counts[1]

    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
    if infer_args['text_replace']:
        common_feature = text_features[:, :common_len]
        scale = cal_scale_difference(common_feature, common_len)
        text_features[1:, :common_len] = common_feature[0].expand_as(text_features[1:, :common_len])
        if infer_args['text_scaling']:
            text_features[1:, common_len+1:] = text_features[1:, common_len+1:] * scale.unsqueeze(-1).expand(-1, 1, 2048)

    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)
    kv_compact = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)
    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    if infer_args is not None:
        infer_args['prompt_tokens'] = []
        for idx in range(len(prompt)):
            token_list = text_tokenizer.convert_ids_to_tokens(input_ids[idx])
            used_tokens = [token for token in token_list if token != "<pad>"]
            infer_args['prompt_tokens'].append(used_tokens)
        return text_cond_tuple, infer_args, common_len
    else:
        return text_cond_tuple


def aug_with_positive_prompt(prompt):
    for key in ['man', 'woman', 'men', 'women', 'boy', 'girl', 'child', 'person', 'human', 'adult', 'teenager', 'employee',
                'employer', 'worker', 'mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather', 'son', 'daughter']:
        if key in prompt:
            prompt = prompt + '. very smooth faces, good looking faces, face to the camera, perfect facial features'
            break
    return prompt


def gen_batch_img_story(
    infinity_test,
    vae,
    text_tokenizer,
    text_encoder,
    prompt,
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    infer_args=None
):
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    text_cond_tuple, infer_args, common_len = encode_prompt_batch(
        text_tokenizer, text_encoder, prompt, enable_positive_prompt, infer_args=infer_args
    )
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt_batch(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        stt = time.time()
        _, _, img_list = infinity_test.autoregressive_infer_cfg_batch_story(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=len(prompt), negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            infer_args=infer_args, common_len=common_len
        )
    gen_time = time.time() - stt
    print(f"Total cost: {time.time() - sstt:.2f}s, Inference cost: {gen_time:.2f}s")
    return img_list


def save_slim_model(infinity_model_path, save_file=None, device='cpu', key='gpt_fsdp'):
    print('[Save slim model]')
    full_ckpt = torch.load(infinity_model_path, map_location=device)
    infinity_slim = full_ckpt['trainer'][key]
    if not save_file:
        save_file = osp.splitext(infinity_model_path)[0] + '-slim.pth'
    print(f'Save to {save_file}')
    torch.save(infinity_slim, save_file)
    print('[Save slim model] done')
    return save_file


def load_tokenizer(t5_path=''):
    print(f'[Loading tokenizer and text encoder]')
    text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(t5_path, revision=None, legacy=True)
    text_tokenizer.model_max_length = 512
    text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.float16)
    text_encoder.to('cuda')
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return text_tokenizer, text_encoder


def load_infinity(
    rope2d_each_sa_layer,
    rope2d_normalized_by_hw,
    use_scale_schedule_embedding,
    pn,
    use_bit_label,
    add_lvl_embeding_only_first_block,
    model_path='',
    scale_schedule=None,
    vae=None,
    device='cuda',
    model_kwargs=None,
    text_channels=2048,
    apply_spatial_patchify=0,
    use_flex_attn=False,
    bf16=False,
    args=None
):
    print(f'[Loading Infinity]')
    text_maxlen = 512

    if args.infer_type == 'story':
        from infinity.models.infinity_batch_story_generate import Infinity
    else:
        from infinity.models.infinity import Infinity

    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        infinity_test: Infinity = Infinity(
            vae_local=vae, text_channels=text_channels, text_maxlen=text_maxlen,
            shared_aln=True, raw_scale_schedule=scale_schedule,
            checkpointing='full-block',
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=use_flex_attn,
            add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
            use_bit_label=use_bit_label,
            rope2d_each_sa_layer=rope2d_each_sa_layer,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pn=pn,
            apply_spatial_patchify=apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            **model_kwargs,
        ).to(device=device)
        print(f'[Infinity loaded] model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={bf16}')

        if bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()

        infinity_test.eval()
        infinity_test.requires_grad_(False)

        infinity_test.cuda()
        torch.cuda.empty_cache()

        print(f'[Loading weights from {model_path}]')
        state_dict = torch.load(model_path, map_location=device)
        print(infinity_test.load_state_dict(state_dict))
        infinity_test.rng = torch.Generator(device=device)
        return infinity_test


def load_visual_tokenizer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.vae_type in [16, 18, 20, 24, 32, 64]:
        from infinity.models.bsq_vae.vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type
        codebook_size = 2 ** codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult = [1, 2, 4, 4]
            decoder_ch_mult = [1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult = [1, 2, 4, 4, 4]
            decoder_ch_mult = [1, 2, 4, 4, 4]
        vae = vae_model(args.vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size,
                        encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
    else:
        raise ValueError(f'vae_type={args.vae_type} not supported')
    return vae


def load_transformer(vae, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    if args.checkpoint_type == 'torch':
        if osp.exists(args.cache_dir):
            local_model_path = osp.join(args.cache_dir, 'tmp', model_path.replace('/', '_'))
        else:
            local_model_path = model_path
        if args.enable_model_cache:
            slim_model_path = model_path.replace('ar-', 'slim-')
            local_slim_model_path = local_model_path.replace('ar-', 'slim-')
            os.makedirs(osp.dirname(local_slim_model_path), exist_ok=True)
            if not osp.exists(local_slim_model_path):
                if osp.exists(slim_model_path):
                    shutil.copyfile(slim_model_path, local_slim_model_path)
                else:
                    if not osp.exists(local_model_path):
                        shutil.copyfile(model_path, local_model_path)
                    save_slim_model(local_model_path, save_file=local_slim_model_path, device=device)
                    if not osp.exists(slim_model_path):
                        shutil.copyfile(local_slim_model_path, slim_model_path)
                        os.remove(local_model_path)
                        os.remove(model_path)
            slim_model_path = local_slim_model_path
        else:
            slim_model_path = model_path
        print(f'[Loading checkpoint from {slim_model_path}]')

    if args.model_type == 'infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    elif args.model_type == 'infinity_layer12':
        kwargs_model = dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer16':
        kwargs_model = dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer24':
        kwargs_model = dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer32':
        kwargs_model = dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer40':
        kwargs_model = dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer48':
        kwargs_model = dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)

    infinity = load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer,
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        use_scale_schedule_embedding=args.use_scale_schedule_embedding,
        pn=args.pn,
        use_bit_label=args.use_bit_label,
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
        model_path=slim_model_path,
        scale_schedule=None,
        vae=vae,
        device=device,
        model_kwargs=kwargs_model,
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=args.use_flex_attn,
        bf16=args.bf16,
        args=args
    )
    return infinity
