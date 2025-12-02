# healnet/encoders/patch.py

from invoke import task
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # go up to repo root
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from healnet.utils import Config
from openslide import OpenSlide
import pandas as pd
import os


@task
def install(c, system: str):
    """
    Download and unpack the GDC client for Linux or macOS.
    """
    assert system in ["linux", "mac"], "Invalid OS specified, must be one of 'linux' or 'mac'"

    print(f"Installing gdc-client for {system}...")
    if system == "linux":
        c.run(
            "curl -0 "
            "https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip "
            "--output gdc-client.zip"
        )
        c.run("unzip gdc-client.zip")
    elif system == "mac":
        c.run(
            "curl -0 "
            "https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_OSX_x64.zip "
            "--output gdc-client.zip"
        )
        c.run("unzip gdc-client.zip")

    print(f"Installed gdc-client at {os.getcwd()}")
    os.remove("gdc-client.zip")


@task
def download(c, dataset: str, config: str = "config/main_gpu.yml", samples: int = None):
    """
    Download WSI data for a TCGA dataset using the GDC client and flatten the directory.
    """
    valid_datasets = ["brca", "blca", "kirp", "ucec", "hnsc", "paad", "luad", "lusc"]
    assert dataset in valid_datasets, f"Invalid dataset arg, must be one of {valid_datasets}"

    conf = Config(config).read()
    download_dir = Path(conf.tcga_path).joinpath(f"wsi/{dataset}")

    # create download dir if doesn't exist (first time running)
    if not download_dir.exists():
        download_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(f"./data/tcga/gdc_manifests/filtered/{dataset}_wsi_manifest_filtered.txt")

    if samples is not None:
        manifest = pd.read_csv(manifest_path, sep="\t")
        manifest = manifest.sample(n=int(samples), random_state=42)
        tmp_path = manifest_path.parent.joinpath(f"{dataset}_tmp.txt")
        manifest.to_csv(tmp_path, sep="\t", index=False)
        print(f"Downloading {manifest.shape[0]} files from {dataset} dataset...")
        c.run(f"{conf.gdc_client} download -m {tmp_path} -d {download_dir}")
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
def flatten(c, dataset: str, config: str = "config/main_gpu.yml"):
    """
    Flatten GDC download structure so WSIs live directly under wsi/{dataset} as *.svs.
    """
    conf = Config(config).read()
    download_dir = Path(conf.tcga_path).joinpath(f"wsi/{dataset}")

    # move all .svs files up
    c.run(f"find {download_dir} -type f -name '*.svs' -exec mv {{}} {download_dir} \\;")
    # delete everything that's not a .svs file
    c.run(f"find {download_dir} ! -name '*.svs' -delete")


@task
def preprocess(
    c,
    dataset: str,
    level: int,
    config: str = "config/main_gpu.yml",
    step: str = "patch",
):
    """
    Preprocess WSIs with CLAM to produce patch .h5 files.

    This task **only**:
      - Checks which slides have the requested level available.
      - Writes valid_prep_ids.csv.
      - Runs CLAM create_patches_fp.py for patch extraction.

    Feature extraction (ResNet/VIT/Hibou) is handled by separate scripts:
      - preprocess_resnet.py
      - preprocess_vit.py
      - preprocess_hibou.py
    """
    assert step in ["patch"], "Only step='patch' is supported here (features are handled separately)."

    conf = Config(config).read()
    raw_path = Path(conf.tcga_path).joinpath(f"wsi/{dataset}")
    prep_path = Path(conf.tcga_path).joinpath(f"wsi/{dataset}_preprocessed_level{level}")
    prep_path.mkdir(parents=True, exist_ok=True)

    assert os.path.exists(raw_path), f"Raw data path not found: {raw_path}"

    # clone CLAM repo if doesn't exist
    if not os.path.exists("CLAM/"):
        c.run("git clone https://github.com/mahmoodlab/CLAM.git")

    valid_csv = prep_path.joinpath("valid_prep_ids.csv")
    if not valid_csv.exists():
        # check which slides have specified level available (only pass those to preprocessing)
        valid_ids = []
        for slide_id in os.listdir(raw_path):
            if not slide_id.endswith(".svs"):
                continue
            slide_fp = raw_path.joinpath(slide_id)
            try:
                slide = OpenSlide(slide_fp)
                slide.level_dimensions[int(level)]  # will raise IndexError if level not present
                valid_ids.append(slide_id)
            except Exception as e:
                print(f"Level {level} not available for slide {slide_id}: {e}")
                continue

        valid_slide_df = pd.DataFrame({"slide_id": valid_ids})
        valid_slide_df.to_csv(valid_csv, index=False)
        print(f"Wrote valid slide IDs to {valid_csv}")

    if step == "patch":
        cmd = (
            f"python CLAM/create_patches_fp.py "
            f"--source {raw_path} "
            f"--save_dir {prep_path} "
            f"--process_list valid_prep_ids.csv "
            f"--patch_size {int(conf.data.patch_size)} "
            f"--patch_level {int(level)} "
            f"--seg --patch --stitch"
        )
        print(f"Running CLAM patch extraction:\n{cmd}")
        c.run(cmd)
        print(f"Finished CLAM patch extraction for {dataset} at level {level}.")
