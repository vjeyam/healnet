'''
Preprocess WSI images by downsampling to a specified level and extracting patches.
Each patch is processed through a pre-trained ResNet50 model to obtain a 1024-dimensional feature vector.

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
from torchvision import models, transforms
from healnet.utils import Config


def load_config(config_path):
    cfg = Config(config_path).read()
    return cfg


def build_resnet_extractor(device):
    """
    Return ResNet50 pretrained on Kather100k (via RetCCL weights),
    output 2048-dim -> reduced to 1024-dim.
    """

    # Load pretrained Kather100k ResNet50 model
    model = torch.hub.load(
        'pococklab/Kather100k-models',
        'resnet50_kather',
        pretrained=True,
        trust_repo=True
    )
    
    # NOTE: backup in case Kather100k model is unavailable
    # model = torch.hub.load(
    #     'MahmoodLab/RetCCL',         # repo
    #     'retccl_resnet50',           # model name
    #     pretrained=True,
    #     trust_repo=True
    # )

    # The model outputs 2048-dim global pooled features
    model = model.to(device).eval()

    # Reduce 2048 → 1024 to match HealNet
    reducer = torch.nn.Linear(2048, 1024).to(device).eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, reducer, preprocess


def extract_resnet_patches(slide_path, save_path, patch_size, level,
                           model, reducer, preprocess, device, batch_size):

    slide = openslide.OpenSlide(slide_path)
    W, H = slide.level_dimensions[level]

    xs = np.arange(0, W, patch_size)
    ys = np.arange(0, H, patch_size)

    patch_tensors = []

    # patch reading loop
    for y in ys:
        for x in xs:
            region = slide.read_region(
                (int(x * slide.level_downsamples[level]),
                 int(y * slide.level_downsamples[level])),
                level,
                (patch_size, patch_size)
            ).convert("RGB")

            patch_tensors.append(preprocess(region))

    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(patch_tensors), batch_size),
                      desc=f"Extracting: {os.path.basename(slide_path)}"):

            batch = torch.stack(patch_tensors[i:i + batch_size]).to(device)

            feat = model(batch).squeeze(-1).squeeze(-1)  # (B,2048)
            feat = reducer(feat)  # (B,1024)
            features.append(feat.cpu())

    features = torch.cat(features, dim=0)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, save_path)
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

    model, reducer, preprocess = build_resnet_extractor(device)

    slide_paths = list(slide_dir.glob("*.svs"))

    for slide_path in slide_paths:
        save_path = out_dir / f"{slide_path.stem}.pt"
        if save_path.exists():
            print(f"[skip] {save_path.name}")
            continue

        extract_resnet_patches(
            slide_path,
            save_path,
            patch_size,
            level,
            model,
            reducer,
            preprocess,
            device,
            batch_size
        )
