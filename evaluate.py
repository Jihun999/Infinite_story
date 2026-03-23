"""
Evaluation script for Infinite-Story.
Computes style consistency (DreamSim, CLIP-I, DINO) and text alignment (CLIP-T) metrics.
"""

import os
import argparse
import itertools
from pathlib import Path
from itertools import combinations
from statistics import harmonic_mean

import numpy as np
import torch
import torch.nn.functional as F
import clip
import sklearn.preprocessing
from PIL import Image
from tqdm import tqdm
from dreamsim import dreamsim
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from carvekit.api.high import HiInterface
    CARVEKIT_AVAILABLE = True
except ImportError:
    CARVEKIT_AVAILABLE = False


def get_device(gpu_id=0):
    return f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'


def load_models(device):
    dreamsim_model, dreamsim_processor = dreamsim(pretrained=True, device=device)
    clip_text_model, _ = clip.load('ViT-B/32', device=device, jit=False)
    clip_text_model.eval()
    clip_image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    clip_image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return dreamsim_model, dreamsim_processor, clip_text_model, clip_image_model, clip_image_processor


def load_bg_remover(device):
    if not CARVEKIT_AVAILABLE:
        print("[Warning] carvekit not installed. Background removal disabled.")
        return None
    return HiInterface(
        object_type="object",
        batch_size_seg=5,
        batch_size_matting=1,
        device=device,
        seg_mask_size=640,
        matting_mask_size=2048,
        trimap_prob_threshold=231,
        trimap_dilation=30,
        trimap_erosion_iters=5,
        fp16=False
    )


def replace_bg_with_noise(image_path, interface):
    image = interface([image_path])[0].convert('RGB')
    image_array = np.array(image)
    black_pixels = np.all(image_array == [130, 130, 130], axis=-1)
    noise = np.random.randint(0, 256, size=image_array.shape, dtype=np.uint8)
    image_array[black_pixels] = noise[black_pixels]
    return Image.fromarray(image_array)


def get_clip_text_score(image, caption, model, device):
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    image = preprocess(image).unsqueeze(0).to(device)
    caption = 'A photo depicts ' + caption if not caption.startswith('A photo depicts') else caption
    caption_tokens = clip.tokenize(caption, truncate=True).to(device)

    image_features = model.encode_image(image)
    text_features = model.encode_text(caption_tokens)

    image_features = sklearn.preprocessing.normalize(image_features.cpu().detach().numpy(), axis=1)
    text_features = sklearn.preprocessing.normalize(text_features.cpu().detach().numpy(), axis=1)

    return 2.5 * np.sum(image_features * text_features)


def compute_dreamsim_distance(image1, image2, model, processor, device):
    features1 = processor(image1).to(device)
    features2 = processor(image2).to(device)
    return model(features1, features2)


def compute_clip_image_distance(image1, image2, model, processor, device):
    inputs1 = processor(images=image1, return_tensors="pt").to(device)
    inputs2 = processor(images=image2, return_tensors="pt").to(device)
    with torch.no_grad():
        features1 = model.get_image_features(**inputs1)
        features2 = model.get_image_features(**inputs2)
    return F.cosine_similarity(features1, features2)


def calculate_clip_text_scores(folder_path, model, device):
    subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    avg_distances = {}
    for subfolder_name in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder_name)
        images, text_prompts = [], []
        for f in Path(subfolder_path).rglob('*'):
            if f.suffix.lower() in {'.png', '.jpg'}:
                images.append(Image.open(f))
                text_prompts.append(f.stem)

        if not images:
            continue

        total = sum(get_clip_text_score(img, txt, model, device).item() for img, txt in zip(images, text_prompts))
        avg_distances[subfolder_name] = total / len(images)
    return avg_distances


def calculate_pairwise_distances(folder_path, mode, model, processor, device, remove_bg=False, bg_interface=None):
    subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    avg_distances = {}
    for subfolder_name in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder_name)
        images = []
        for f in Path(subfolder_path).rglob('*'):
            if f.suffix.lower() in {'.png', '.jpg'}:
                if remove_bg and bg_interface is not None:
                    images.append(replace_bg_with_noise(f, bg_interface))
                else:
                    images.append(Image.open(f))

        if len(images) < 2:
            continue

        total = 0
        for combo in combinations(images, 2):
            if mode == "dreamsim":
                total += compute_dreamsim_distance(combo[0], combo[1], model, processor, device).item()
            elif mode == "clip_image":
                total += compute_clip_image_distance(combo[0], combo[1], model, processor, device).item()
        avg_distances[subfolder_name] = total / (len(images) * (len(images) - 1) / 2)
    return avg_distances


def dino_score(folder_path, device):
    processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8')
    model_dino = ViTModel.from_pretrained('facebook/dino-vitb8').to(device)
    model_dino.eval()

    folder_paths = sorted([p for p in Path(folder_path).iterdir() if p.is_dir()])
    scores = []

    for folder in tqdm(folder_paths, desc="DINO evaluation"):
        image_paths = list(folder.glob('*.png')) + list(folder.glob('*.jpg'))
        if len(image_paths) < 2:
            continue

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model_dino(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]

        pairwise_sims = [
            F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
            for i, j in itertools.combinations(range(len(image_paths)), 2)
        ]
        scores.append(sum(pairwise_sims) / len(pairwise_sims))

    return np.mean(scores) if scores else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Infinite-Story generation quality.")
    parser.add_argument('--dir', type=str, required=True, help='Path to the folder containing generated image subfolders')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--remove_background', action='store_true', help='Remove background before computing style consistency')
    parser.add_argument('--output', type=str, default='./eval_results.txt', help='Path to save evaluation results')
    args = parser.parse_args()

    device = get_device(args.gpu)
    print(f"[Evaluation] Using device: {device}")
    print(f"[Evaluation] Evaluating: {args.dir}")

    dreamsim_model, dreamsim_processor, clip_text_model, clip_image_model, clip_image_processor = load_models(device)
    bg_interface = load_bg_remover(device) if args.remove_background else None

    # DreamSim (style consistency - lower is more consistent)
    print("\n[1/4] Computing DreamSim scores...")
    dreamsim_scores = calculate_pairwise_distances(
        args.dir, 'dreamsim', dreamsim_model, dreamsim_processor, device,
        remove_bg=args.remove_background, bg_interface=bg_interface
    )
    avg_dreamsim = np.mean(list(dreamsim_scores.values())) if dreamsim_scores else 0

    # CLIP-T (text alignment)
    print("[2/4] Computing CLIP-T scores...")
    clipt_scores = calculate_clip_text_scores(args.dir, clip_text_model, device)
    avg_clipt = np.mean(list(clipt_scores.values())) if clipt_scores else 0

    # CLIP-I (image consistency)
    print("[3/4] Computing CLIP-I scores...")
    clipi_scores = calculate_pairwise_distances(
        args.dir, 'clip_image', clip_image_model, clip_image_processor, device,
        remove_bg=args.remove_background, bg_interface=bg_interface
    )
    avg_clipi = np.mean(list(clipi_scores.values())) if clipi_scores else 0

    # DINO (style consistency)
    print("[4/4] Computing DINO scores...")
    avg_dino = dino_score(args.dir, device)

    # Harmonic mean
    hm = harmonic_mean([avg_clipt, avg_clipi, 1 - avg_dreamsim, avg_dino]) if all([avg_clipt, avg_clipi, avg_dreamsim, avg_dino]) else 0

    # Print results
    results = f"""
=== Evaluation Results ===
Directory: {args.dir}
Background removal: {args.remove_background}

Style Consistency:
  DreamSim (lower=better): {avg_dreamsim:.5f}
  CLIP-I (higher=better):  {avg_clipi:.5f}
  DINO (higher=better):    {avg_dino:.5f}

Text Alignment:
  CLIP-T (higher=better):  {avg_clipt:.5f}

Overall:
  Harmonic Mean:            {hm:.5f}
===========================
"""
    print(results)

    with open(args.output, "a") as f:
        f.write(results)
    print(f"[Saved] Results written to {args.output}")


if __name__ == "__main__":
    main()
