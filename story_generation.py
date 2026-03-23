import torch
import numpy as np
import os
import os.path as osp
import argparse
from tools.run_infinity import *
import torchvision
import yaml


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def process_prompt(prompt_path, exp_name):
    with open(os.path.expanduser(prompt_path), 'r') as f:
        data = yaml.safe_load(f)

    instances = []
    for subject_domain, subject_domain_instances in data.items():
        for index, instance in enumerate(subject_domain_instances):
            id_prompt = f'{instance["style"]} {instance["subject"]}'
            frame_prompt_list = instance["settings"]
            save_dir = os.path.join(exp_name, f"{subject_domain}_{index}")
            instances.append((id_prompt, frame_prompt_list, save_dir))
    return instances


def get_args():
    parser = argparse.ArgumentParser(description='Infinite-Story: Consistent Story Generation')
    parser.add_argument('--cfg', type=str, default='3.0')
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--pn', type=str, default='1M', choices=['0.06M', '0.25M', '1M'])
    parser.add_argument('--model_path', type=str, default='weights/infinity_2b_reg.pth')
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_type', type=int, default=32)
    parser.add_argument('--vae_path', type=str, default='weights/infinity_vae_d32reg.pth')
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=1, choices=[0,1])
    parser.add_argument('--use_bit_label', type=int, default=1, choices=[0,1])
    parser.add_argument('--model_type', type=str, default='infinity_2b')
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0,1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0,1])
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1,2,4,8,16])
    parser.add_argument('--text_encoder_ckpt', type=str, default='google/flan-t5-xl')
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0,1])
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0,1])
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--cache_dir', type=str, default='/dev/shm')
    parser.add_argument('--enable_model_cache', type=int, default=0, choices=[0,1])
    parser.add_argument('--checkpoint_type', type=str, default='torch')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--bf16', type=int, default=1, choices=[0,1])
    parser.add_argument('--infer_type', type=str, default='story')
    parser.add_argument('--exp_name', type=str, default='./output/story_generation')
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--prompt_path', type=str, default='./prompt/consistory_plus.yaml')
    parser.add_argument('--cross_scale', type=float, default=1, help='Cross-attention scaling factor')
    parser.add_argument('--weight', type=float, default=0.85, help='Adaptive Style Injection weight (paper default: 0.85)')
    parser.add_argument('--attn_control', type=str2bool, default=True, help='Enable Adaptive Style Injection')
    parser.add_argument('--cfg_control', type=str2bool, default=True, help='Enable Synchronized Guidance Adaptation')
    parser.add_argument('--text_replace', type=str2bool, default=True, help='Enable Identity Prompt Replacement')
    parser.add_argument('--text_scaling', type=str2bool, default=True, help='Enable text feature scaling')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(args.gpu_idx)
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    infer_args = {
        'attn': [args.attn_control, 4, args.weight, args.cfg_control, args.cross_scale, args.attn_control],
        'text_replace': args.text_replace,
        'text_scaling': args.text_scaling,
    }

    print(f'[Config] weight={args.weight}, attn_control={args.attn_control}, '
          f'cfg_control={args.cfg_control}, text_replace={args.text_replace}, '
          f'text_scaling={args.text_scaling}')

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    h_div_w = 1.0
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    seed = args.seed
    print(f"[Seed] {seed}")
    instances = process_prompt(args.prompt_path, args.exp_name)

    for id_prompt, frame_prompt_list, save_dir in instances:
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n[Generating] {id_prompt}")

        batch_size = 4
        first_prompt = f"{id_prompt} {frame_prompt_list[0]}"
        other_prompts = frame_prompt_list[1:]

        for i in range(0, len(other_prompts), batch_size - 1):
            batch_settings = other_prompts[i:i + (batch_size - 1)]
            batch_prompts = [first_prompt] + [f"{id_prompt} {setting}" for setting in batch_settings]

            generated_images = gen_batch_img_story(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                prompt=batch_prompts,
                g_seed=seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=0,
                infer_args=infer_args
            )

            for j, (img_tensor, prompt_text) in enumerate(zip(generated_images, batch_prompts)):
                save_path = osp.join(save_dir, f"{prompt_text}.png")
                torchvision.utils.save_image(img_tensor.squeeze(0), save_path)
                print(f"  Saved: {save_path}")
