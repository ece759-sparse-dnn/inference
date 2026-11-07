"""Download datasets for HPEC Graph Challenge."""

import argparse as ap
import gzip
import shutil
import tarfile

import requests

from . import constants, layout

BASE_URL: str = "https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019"
TIMEOUT = (10, 60)


def layers_url(*, size: int) -> str:
    """Download URL for layers."""
    return f"{BASE_URL}/dnn/neuron{size}.tar.gz"


def categories_url(*, size: int, depth: int) -> str:
    """Download URL for categories."""
    return f"{BASE_URL}/dnn/neuron{size}-l{depth}-categories.tsv"


def images_url(*, size: int) -> str:
    """Download URL for MNIST images."""
    return f"{BASE_URL}/mnist/sparse-images-{size}.tsv.gz"


def fetch(url: str) -> requests.Response:
    """Fetch response for a URL."""
    response = requests.get(url, stream=True, timeout=TIMEOUT)
    response.raise_for_status()
    return response


def get_layers(*, size: int) -> None:
    """Download layers for a particular size."""
    out = layout.raw_dir()
    done = layout.raw_layers_marker(size)
    if done.exists():
        print(f"  layers {size}: already present")
        return
    print(f"  layers {size}: downloading and extracting...")
    out.parent.mkdir(parents=True, exist_ok=True)
    with (
        fetch(layers_url(size=size)) as response,
        tarfile.open(fileobj=response.raw, mode="r|gz") as tar,
    ):
        tar.extractall(path=out, filter="data")
    done.touch()


def get_categories(*, size: int) -> None:
    """Download all categories for a particular size."""
    for depth in constants.DEPTHS:
        dest = layout.raw_categories_tsv(size, depth)
        if dest.exists():
            print(f"  categories {size}-{depth}: already present")
            continue
        print(f"  categories {size}-{depth}: downloading...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        with (
            fetch(categories_url(size=size, depth=depth)) as response,
            open(tmp, "wb") as dst,
        ):
            shutil.copyfileobj(response.raw, dst)
        tmp.rename(dest)
    print(f"  categories {size}: done")


def get_images(*, size: int) -> None:
    """Downloads images for a particular size."""
    dest = layout.raw_images_tsv(size)
    if dest.exists():
        print(f"  images {size}: already present")
        return
    print(f"  images {size}: downloading...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with (
        fetch(images_url(size=size)) as response,
        gzip.GzipFile(fileobj=response.raw) as src,
        open(tmp, "wb") as dst,
    ):
        shutil.copyfileobj(src, dst)
    tmp.rename(dest)


def download(config: ap.Namespace):
    """Downloads datasets for desired size(s)."""
    for size in config.s:
        get_images(size=size)
        get_layers(size=size)
        get_categories(size=size)
