from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from sdnn_tools import layout

TINY_SIZE = 4
TINY_NNZ_PER_ROW = 2


def write_triplets(path: Path, dense: np.ndarray, *, seed: int = 0) -> None:
    rows, cols = np.nonzero(dense)
    order = np.random.default_rng(seed).permutation(len(rows))
    with path.open("w") as f:
        for r, c in zip(rows[order], cols[order]):
            f.write(f"{r + 1}\t{c + 1}\t{dense[r, c]:.6e}\n")


def write_rows(path: Path, rows: np.ndarray) -> None:
    path.write_text("".join(f"{r + 1}\n" for r in rows))


@dataclass
class TinyRaw:
    size: int
    nnz_per_row: int
    weights: list[np.ndarray]
    images: np.ndarray
    categories: dict[int, np.ndarray]


@pytest.fixture
def data_root(tmp_path: Path):
    layout.set_data_root(tmp_path)
    yield tmp_path
    layout.set_data_root(None)


@pytest.fixture
def tiny_raw(data_root: Path) -> TinyRaw:
    n, k = TINY_SIZE, TINY_NNZ_PER_ROW
    ring = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        ring[i, i] = ring[i, (i + 1) % n] = 1.0
    weights = [0.5 * ring, 0.25 * ring, ring.T.copy()]
    images = np.array(
        [[1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    categories = {3: np.array([0, 2])}

    for layer, w in enumerate(weights, start=1):
        write_triplets(layout.raw_layer_tsv(n, layer), w, seed=layer)
    write_triplets(layout.raw_images_tsv(n), images)
    for depth, rows in categories.items():
        write_rows(layout.raw_categories_tsv(n, depth), rows)
    layout.raw_layers_marker(n).touch()

    return TinyRaw(n, k, weights, images, categories)
