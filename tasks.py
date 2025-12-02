from invoke import task
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from healnet.utils import Config
from torchvision import transforms
from openslide import OpenSlide
import pandas as pd
import torch
from tqdm import tqdm
import os
import h5py
import torchvision.models as models

from healnet.encoders import build_patch_encoder


@task
def install(c, system: str):
    assert system in ["linux", "mac"], "Invalid OS specified, must be one of 'linux' or 'mac'"

    print(f"Installing gdc-client for {system}...")
    if system == "linux":
        c.run("curl -0 https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip "
              "--output gdc-client.zip")
        c.run("unzip gdc-client.zip")
    if system == "mac":
        c.run("curl -0 https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_OSX_x64.zip "
              "--output gdc-client.zip")
        c.run("unzip gdc-client.zip")
    print(f"Installed gdc-client at {os.getcwd()}")
    # cleanup
    os.remove("gdc-client.zip")

@task
def download(c, dataset:str, config:str="config/main_gpu.yml", samples: int = None):
    valid_datasets = ["brca", "blca", "kirp", "ucec", "hnsc", "paad", "luad", "lusc"]
    conf = Config(config).read()
    download_dir = Path(conf.tcga_path).joinpath(f"wsi/{dataset}")

    # create download dir if doesn't exist (first time running)
    if not download_dir.exists():
        download_dir.mkdir(parents=True)

    assert dataset in valid_datasets, f"Invalid dataset arg, must be one of {valid_datasets}"

    manifest_path = Path(f"./data/tcga/gdc_manifests/filtered/{dataset}_wsi_manifest_filtered.txt")
    # manifest_path = Path(conf.tcga_path).joinpath(f"gdc_manifests/filtered/{dataset}_wsi_manifest_filtered.txt")
    # Download entire manifest unless specified otherwise
    if samples is not None:
        manifest = pd.read_csv(manifest_path, sep="\t")
        manifest = manifest.sample(n=int(samples), random_state=42)
        tmp_path = manifest_path.parent.joinpath(f"{dataset}_tmp.txt")
        manifest.to_csv(tmp_path, sep="\t", index=False)
        print(f"Downloading {manifest.shape[0]} files from {dataset} dataset...")
        c.run(f"{conf.gdc_client} download -m {tmp_path} -d {download_dir}")
        # cleanup
        os.remove(tmp_path)

    else:
        command = f"{conf.gdc_client} download -m {manifest_path} -d {download_dir}"
        try:
            c.run(command)
        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"Command: {command}")

    # flatten directory structure (required to easily run CLAM preprocessing)
    flatten(c, dataset, config)

@task
def flatten(c, dataset: str, config: str):
    """
    Flattens directory structure for WSI images after download using the GDC client from
     `data_dir/*.svs` instead of `data_dir/hash_subdir/*.svs`.
    Args:
        c:
        dataset:
        config:

    Returns:
    """
    conf = Config(config).read()
    download_dir = Path(conf.tcga_path).joinpath(f"wsi/{dataset}")
    # flatten directory structure
    c.run(f"find {download_dir} -type f -name '*.svs' -exec mv {{}} {download_dir} \;")
    # remove everything that's not a .svs file
    c.run(f"find {download_dir} ! -name '*.svs' -delete")

@task
def preprocess(
    c,
    dataset: str,
    level: int,
    config: str="config/main_gpu.yml",
    step:str= "patch",
    encoder: str = None,
    encoder_checkpoint: str = None,
):
    """
    Preprocesses WSI images for downstream tasks.
    Args:
        c:
        dataset:
        config:

    Returns:

    """
    conf = Config(config).read()
    raw_path = Path(conf.tcga_path).joinpath(f"wsi/{dataset}")
    prep_path = Path(conf.tcga_path).joinpath(f"wsi/{dataset}_preprocessed_level{level}")
    # create prep dir
    prep_path.mkdir(parents=True, exist_ok=True)

    assert os.path.exists(raw_path), f"Raw data path not found: {raw_path}"
    valid_steps = ["patch", "features"]
    assert step in valid_steps, f"Invalid step arg, must be one of {valid_steps}"

    # clone CLAM repo if doesn't exist
    if not os.path.exists("CLAM/"):
        c.run("git clone git@github.com:mahmoodlab/CLAM.git")


    if not os.path.exists(prep_path.joinpath("valid_prep_ids.csv")):
        # check which slides have specified level available (only pass those to preprocessing)
        valid_ids = []
        for slide_id in os.listdir(raw_path):
            # check whether specified level is available in slide
            slide = OpenSlide(raw_path.joinpath(f"{slide_id}"))
            try:
                slide.level_dimensions[int(level)]
                valid_ids.append(slide_id)
            except IndexError as e:
                print(f"Level {level} not available for slide {slide_id}")
                continue

        # write temp csv file with valid slide ids to pass to CLAM
        valid_slide_df = pd.DataFrame({"slide_id": valid_ids})
        valid_slide_df.to_csv(prep_path.joinpath("valid_prep_ids.csv"), index=False)

    if step == "patch":
        c.run(f"python CLAM/create_patches_fp.py --source {raw_path} --save_dir {prep_path} --process_list valid_prep_ids.csv "
          f"--patch_size {int(conf.data.patch_size)} --patch_level {int(level)} --seg --patch --stitch")

    if step == "features":
        encoder_name = (encoder or conf.get("patch_encoder", "resnet50")).lower()
        encoder_ckpt = encoder_checkpoint or conf.get("patch_encoder_checkpoint")
        # only take preprocessed slides
        patch_dir = prep_path.joinpath("patches")
        patch_files = [filename for filename in os.listdir(patch_dir) if filename.endswith(".h5")]
        slide_ids = [os.path.splitext(filename)[0] for filename in patch_files]
        # load patch coords
        coords = {}
        for slide_id in slide_ids:
            patch_path = prep_path.joinpath(f"patches/{slide_id}.h5")
            try:
                h5_file = h5py.File(patch_path, "r")
                patch_coords = h5_file["coords"][:]
                if patch_coords.dtype != int:
                    patch_coords = patch_coords.astype(int)
                coords[slide_id] = patch_coords
            except FileNotFoundError as e:
                print(f"No patches available for file {patch_path}")
                pass
        max_patches = max([coords.get(key).shape[0] for key in coords.keys()])
        print(f"Max patches: {max_patches}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        patch_encoder, feat_dim, region_transform = build_patch_encoder(
            encoder_name,
            device=device,
            checkpoint=encoder_ckpt,
        )

        if encoder_name == "resnet50":
            feat_path = prep_path.joinpath("patch_features")
        else:
            feat_path = prep_path.joinpath(f"patch_features_{encoder_name}")
        feat_path.mkdir(parents=True, exist_ok=True)

        num_slides = len(slide_ids)
        patch_tensors = torch.zeros(max_patches, feat_dim)
        # extract features
        for slide_count, slide_id in enumerate(coords.keys()):
            save_path = feat_path.joinpath(f"{slide_id}.pt")
            # check if features already extracted
            if os.path.exists(save_path):
                print(f"[{encoder_name}] Features already extracted for slide {slide_id}, skipping...")
                continue

            slide = OpenSlide(raw_path.joinpath(f"{slide_id}.svs"))
            print(f"[{encoder_name}] slide {slide_count+1}/{num_slides}")

            patch_tensors.zero_()

            for idx, coord in enumerate(tqdm(coords[slide_id])):
                # print(f"{dataset.upper()} Level {level}: Processing patch {idx} of {len(coords[slide_id])} for "
                #       f"slide {slide_count+1}/{num_slides}")
                x, y = coord
                patch_image = slide.read_region((x, y), level=int(level), size=(256, 256)).convert("RGB")
                patch_region = region_transform(patch_image)
                patch_region = patch_region.to(device).unsqueeze(0)
                outputs = patch_encoder(patch_region)
                if hasattr(outputs, "last_hidden_state"):
                    patch_features = outputs.last_hidden_state[:, 0, :]
                else:
                    patch_features = outputs
                patch_tensors[idx] = patch_features.cpu().detach().view(-1)

            # save features (clone to avoid in-place modification on next slide)
            torch.save(patch_tensors.clone(), save_path)
        print(f"[{encoder_name}] Finished feature extraction for dataset {dataset}.")
