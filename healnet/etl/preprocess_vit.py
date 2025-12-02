'''
Preprocess WSI images using image patch sizes of 256 x 256 with no overlap.
Each patch is processed through a pre-trained ViT-Large model to obtain a 1024-dimensional feature vector.

Improvement: Use ViT-Large to convert each patch into a 1024-dim feature vector instead of ResNet50.

Based on the description in the HealNet paper:
- Tissue segmentation is performed to identify relevant regions.
- Image patches of size 256x256 pixels are extracted without overlap at the 20x pyramid level.
'''

import os
import torch
import openslide
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import ViTModel, ViTImageProcessor
from healnet.utils import Config


def load_config(path):
    return Config(path).read()


def build_vit_extractor(device):
    model_name = "microsoft/uni-vit-large-patch16"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name).to(device).eval()
    feat_dim = model.config.hidden_size  # should be 1024
    return model, processor, feat_dim


def extract_vit_patches(slide_path, save_path, patch_size, level,
                        model, processor, device, batch_size):

    slide = openslide.OpenSlide(slide_path)
    W, H = slide.level_dimensions[level]

    xs = np.arange(0, W, patch_size)
    ys = np.arange(0, H, patch_size)

    patches = []

    # collect patches
    for y in ys:
        for x in xs:
            region = slide.read_region(
                (int(x * slide.level_downsamples[level]),
                 int(y * slide.level_downsamples[level])),
                level,
                (patch_size, patch_size)
            ).convert("RGB")
            patches.append(region)

    # embed patches
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size),
                      desc=f"VIT extracting: {os.path.basename(slide_path)}"):

            batch_imgs = patches[i:i + batch_size]
            inputs = processor(images=batch_imgs, return_tensors="pt")["pixel_values"].to(device)

            out = model(inputs).last_hidden_state[:, 0, :]  # CLS token (B,1024)
            feats.append(out.cpu())

    feats = torch.cat(feats, dim=0)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, save_path)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    tcga_path = cfg.tcga_path
    dataset = args.dataset
    patch_size = cfg.data.patch_size
    level = cfg.data.wsi_level
    batch_size = cfg["train_loop.batch_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    slide_dir = Path(tcga_path) / "wsi" / dataset
    out_dir = Path(tcga_path) / "wsi" / f"{dataset}_preprocessed_level{level}" / "patch_features"

    model, processor, feat_dim = build_vit_extractor(device)
    slide_paths = list(slide_dir.glob("*.svs"))

    for slide_path in slide_paths:
        save_path = out_dir / f"{slide_path.stem}.pt"
        if save_path.exists():
            print(f"[skip] {save_path.name}")
            continue

        extract_vit_patches(
            slide_path,
            save_path,
            patch_size,
            level,
            model,
            processor,
            device,
            batch_size
        )
