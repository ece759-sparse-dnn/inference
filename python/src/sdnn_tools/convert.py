import argparse as ap
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp

from . import constants, layout


def read_tsv(tsv_file: Path, *, names: list[str], types) -> pd.DataFrame:
    """Create DataFrame from TSV."""
    return pd.read_csv(
        tsv_file, sep="\t", header=None, names=names, dtype=types, engine="pyarrow"
    )


def read_triplets(tsv_file: Path) -> pd.DataFrame:
    """Create DataFrame from COO-style TSV."""
    return read_tsv(
        tsv_file,
        names=["src", "dst", "val"],
        types={"src": np.int32, "dst": np.int32, "val": np.float32},
    )


def layer_tsvs(size: int) -> list[Path]:
    """List of layer TSVs, sorted by layer number."""
    pattern = re.compile(rf"n{size}-l(\d+)\.tsv")
    numbered = []
    for path in layout.raw_layers_dir(size).iterdir():
        if match := pattern.fullmatch(path.name):
            numbered.append((int(match.group(1)), path))
    numbered.sort()

    numbers = [k for k, _ in numbered]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"layers for size {size} are not contiguous from 1")
    return [path for _, path in numbered]


def layer_to_csr(
    df: pd.DataFrame, size: int, nnz_per_row: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a layer's DataFrame to CSR format."""
    coo = sp.sparse.coo_array(
        (df["val"].to_numpy(), (df["src"].to_numpy() - 1, df["dst"].to_numpy() - 1)),
        shape=(size, size),
    )
    assert np.all(coo.count_nonzero(axis=1) == nnz_per_row)
    csr = coo.tocsr()
    csr.sort_indices()

    return (
        csr.indptr.astype(np.int32),
        csr.indices.astype(np.int32),
        csr.data.astype(np.float32),
    )


def convert_layers(size: int, nnz_per_row: int) -> int:
    """Serialize layer TSVs into CSR format .npy files."""
    tsvs = layer_tsvs(size)
    num_layers = len(tsvs)
    nnz = size * nnz_per_row
    dest = layout.bin_network_dir(size)

    open_memmap = np.lib.format.open_memmap
    indptr = open_memmap(
        dest / "weights_indptr.npy", "w+", np.int32, (num_layers, size + 1)
    )
    indices = open_memmap(
        dest / "weights_indices.npy", "w+", np.int32, (num_layers, nnz)
    )
    values = open_memmap(
        dest / "weights_values.npy", "w+", np.float32, (num_layers, nnz)
    )

    for i, tsv in enumerate(tsvs):
        indptr[i], indices[i], values[i] = layer_to_csr(
            read_triplets(tsv), size, nnz_per_row
        )
        if (i + 1) % 100 == 0 or i + 1 == num_layers:
            print(f"  layers {size}: {i + 1}/{num_layers}", end="\r", flush=True)
    print()

    for mapped_array in (indptr, indices, values):
        mapped_array.flush()
    return num_layers


def convert_images(size: int) -> int:
    """Serialize MNIST TSV into CSR format .npy files."""
    df = read_triplets(layout.raw_images_tsv(size))
    num_images = int(df["src"].max())
    coo = sp.sparse.coo_array(
        (df["val"].to_numpy(), (df["src"].to_numpy() - 1, df["dst"].to_numpy() - 1)),
        shape=(num_images, size),
    )
    csr = coo.tocsr()
    csr.sort_indices()

    dest = layout.bin_network_dir(size)
    np.save(dest / "images_indptr.npy", csr.indptr.astype(np.int64))
    np.save(dest / "images_indices.npy", csr.indices.astype(np.int32))
    np.save(dest / "images_values.npy", csr.data.astype(np.float32))
    print(f"  images {size}: {num_images} images, {csr.nnz} nonzeros")
    return num_images


def convert_categories(size: int) -> list[int]:
    """Serialize category TSVs into .npy files."""
    pattern = re.compile(rf"neuron{size}-l(\d+)-categories\.tsv")
    dest = layout.bin_network_dir(size)
    depths = []
    for path in sorted(layout.raw_categories_dir().iterdir()):
        if not (match := pattern.fullmatch(path.name)):
            continue
        depth = int(match.group(1))
        rows = read_tsv(path, names=["row"], types={"row": np.int32})["row"]
        np.save(dest / f"categories_l{depth}.npy", rows.to_numpy() - 1)
        depths.append(depth)
    depths.sort()
    print(f"  categories {size}: depths {depths}")
    return depths


def convert_dataset(
    size: int,
    *,
    bias: float,
    nnz_per_row: int = constants.NNZ_PER_ROW,
    ymax: float = constants.YMAX,
    force: bool = False,
) -> None:
    """Serialize one dataset (layers, images, categories) into <data>/bin/neuron{size}."""
    if not layout.raw_layers_marker(size).exists():
        print(f"  dataset {size}: not downloaded")
        return
    marker = layout.bin_marker(size)
    if marker.exists() and not force:
        print(f"  dataset {size}: already converted")
        return
    marker.unlink(missing_ok=True)

    num_layers = convert_layers(size, nnz_per_row)
    num_images = convert_images(size)
    depths = convert_categories(size)

    meta = {
        "version": constants.BIN_VERSION,
        "format": "csr",
        "neurons": size,
        "layers": num_layers,
        "nnz_per_layer": size * nnz_per_row,
        "images": num_images,
        "category_depths": depths,
        "bias": bias,
        "ymax": ymax,
    }
    (layout.bin_network_dir(size) / "meta.json").write_text(json.dumps(meta, indent=2))
    marker.touch()


def convert(config: ap.Namespace) -> None:
    """Serialize datasets into .npy formats."""
    for size in config.s:
        convert_dataset(size, bias=constants.BIAS[size], force=config.force)
