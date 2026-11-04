# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests>=2.34.2",
# ]
# ///
"""Download Sparse DNN Graph Challenge data:
    - layer weights
    - input images
    - truth categories
Usage:
    uv run scripts/get_data.py data/            # 1024 neurons - default
    uv run scripts/get_data.py data/ 1024 4096  # specific size(s)
"""

import argparse
import gzip
import shutil
import sys
import tarfile
from pathlib import Path

import requests

BASE_URL = "https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019"
SIZES = [1024, 4096, 16384, 65536]
LAYERS = [120, 480, 1920]
TIMEOUT = (10, 60)


def fetch(url: str) -> requests.Response:
    response = requests.get(url, stream=True, timeout=TIMEOUT)
    response.raise_for_status()
    return response


def get_layers(n: int, out: Path) -> None:
    """downloads neuronN.tar.gz -> out/neuronN/nN-l1.tsv"""
    done = out / f".neuron{n}.done"
    if done.exists():
        print(f"  layers {n}: already present")
        return
    print(f"  layers {n}: downloading and extracting...")
    with (
        fetch(f"{BASE_URL}/dnn/neuron{n}.tar.gz") as response,
        tarfile.open(fileobj=response.raw, mode="r|gz") as tar,
    ):
        tar.extractall(path=out, filter="data")
    done.touch()


def get_images(n: int, out: Path) -> None:
    """downloads sparse-images-N.tsv.gz -> out/mnist/sparse-images-N.tsv"""
    dest = out / "mnist" / f"sparse-images-{n}.tsv"
    if dest.exists():
        print(f"  images {n}: already present")
        return
    print(f"  images {n}: downloading...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")  # marks partial until completed
    with (
        fetch(f"{BASE_URL}/mnist/sparse-images-{n}.tsv.gz") as response,
        gzip.GzipFile(fileobj=response.raw) as src,
        open(tmp, "wb") as dst,
    ):
        shutil.copyfileobj(src, dst)
    tmp.rename(dest)  # completed


def get_categories(n: int, out: Path) -> None:
    """downloads truth files for every depth -> out/categories/neuronN-lL-categories.tsv"""
    for layers in LAYERS:
        name = f"neuron{n}-l{layers}-categories.tsv"
        dest = out / "categories" / name
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        with fetch(f"{BASE_URL}/dnn/{name}") as response, open(tmp, "wb") as dst:
            shutil.copyfileobj(response.raw, dst)
        tmp.rename(dest)
    print(f"  categories {n}: ok")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("out", type=Path, help="data directory, e.g. data/")
    parser.add_argument(
        "sizes",
        type=int,
        nargs="*",
        default=[1024],
        choices=SIZES,
        help="neurons per layer (default: 1024)",
    )
    args = parser.parse_args()

    if 65536 in args.sizes:
        print("Note: the 65536 network is ~16 GB compressed and much larger unpacked.")

    args.out.mkdir(parents=True, exist_ok=True)
    try:
        for n in args.sizes:
            print(f"Neurons {n}:")
            get_categories(n, args.out)
            get_images(n, args.out)
            get_layers(n, args.out)
    except (requests.RequestException, OSError, tarfile.TarError) as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1

    print(f"Done: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
