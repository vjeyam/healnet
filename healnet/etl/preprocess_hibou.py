"""
Preprocess WSI images using CellViT-Hibou-L patch embeddings.
Each 256x256 patch is passed into the CellViT-DINOv2-based Hibou transformer
to extract CLS embeddings (1280 dim for 'cellvit-hibou-l').

Usage:
    python preprocess_hibou.py --config config/main_gpu.yml --dataset blca
"""

from huggingface_hub import login
login("Insert HF token") # hugging face token

import os
import torch
import openslide
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from healnet.utils import Config


def load_config(path):
    return Config(path).read()


def build_hibou_extractor(device, model_name="histai/cellvit-hibou-l"):
    """
    Build CellViT-Hibou-L extractor using HuggingFace.

    Returns:
        model: DINOv2-based ViT backbone with CellViT modifications
        processor: image preprocessing pipeline
        feat_dim: CLS token feature size (1280 for L)
    """

    processor = AutoImageProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True
    ).to(device).eval()

    # Hidden size (1280 for Hibou-L; confirmed from HF config)
    feat_dim = model.config.hidden_size

    return model, processor, feat_dim


def extract_hibou_patches(
    slide_path, save_path, patch_size, level,
    model, processor, device, batch_size
):
    slide = openslide.OpenSlide(slide_path)
    W, H = slide.level_dimensions[level]

    xs = np.arange(0, W, patch_size)
    ys = np.arange(0, H, patch_size)

    patches = []

    # extract patches
    for y in ys:
        for x in xs:
            region = slide.read_region(
                (int(x * slide.level_downsamples[level]),
                 int(y * slide.level_downsamples[level])),
                level,
                (patch_size, patch_size)
            ).convert("RGB")
            patches.append(region)

    # embed patches with DINOv2+CellViT Hibou
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size),
                      desc=f"CELLViT-HIBOU extracting: {os.path.basename(slide_path)}"):

            batch_imgs = patches[i:i + batch_size]

            inputs = processor(
                images=batch_imgs,
                return_tensors="pt"
            )["pixel_values"].to(device)

            outputs = model(inputs)

            # CLS token representation
            cls = outputs.last_hidden_state[:, 0, :]   # (B, 1280)

            feats.append(cls.cpu())

    feats = torch.cat(feats, dim=0)   # (N_patches, feat_dim)

    # save patch embeddings
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, save_path)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="histai/cellvit-hibou-l",
        help="Use CellViT-Hibou-L (default) or CellViT-Hibou-B"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    tcga_path = cfg.tcga_path
    dataset = args.dataset
    patch_size = cfg.data.patch_size
    level = cfg.data.wsi_level
    batch_size = cfg["train_loop.batch_size"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    slide_dir = Path(tcga_path) / "wsi" / dataset
    out_dir = Path(tcga_path) / "wsi" / f"{dataset}_preprocessed_level{level}" / "patch_features_cellvit_hibou"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor, feat_dim = build_hibou_extractor(device, model_name=args.model)

    slide_paths = list(slide_dir.glob("*.svs"))

    for slide_path in slide_paths:
        save_path = out_dir / f"{slide_path.stem}.pt"

        if save_path.exists():
            print(f"[skip] {save_path.name}")
            continue

        extract_hibou_patches(
            slide_path,
            save_path,
            patch_size,
            level,
            model,
            processor,
            device,
            batch_size
        )