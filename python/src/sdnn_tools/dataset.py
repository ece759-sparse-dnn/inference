import json
from dataclasses import dataclass, field

import numpy as np
import scipy as sp

from . import constants, layout


@dataclass(frozen=True)
class Dataset:
    neurons: int
    bias: float
    ymax: float
    images: sp.sparse.csr_array
    weights_indptr: np.ndarray
    weights_indices: np.ndarray
    weights_values: np.ndarray
    categories: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        return self.weights_indptr.shape[0]

    def layer(self, i: int) -> sp.sparse.csr_array:
        return sp.sparse.csr_array(
            (self.weights_values[i], self.weights_indices[i], self.weights_indptr[i]),
            shape=(self.neurons, self.neurons),
        )


def load(size: int) -> Dataset:
    root = layout.bin_network_dir(size)
    meta_path = root / "meta.json"
    if not layout.bin_marker(size).exists() or not meta_path.exists():
        raise FileNotFoundError(f"no converted dataset at {root}")
    meta = json.loads(meta_path.read_text())
    if meta["version"] != constants.BIN_VERSION:
        raise ValueError(
            f"{meta_path}: bin version {meta['version']}, "
            f"expected {constants.BIN_VERSION}"
        )

    images = sp.sparse.csr_array(
        (
            np.load(root / "images_values.npy"),
            np.load(root / "images_indices.npy"),
            np.load(root / "images_indptr.npy"),
        ),
        shape=(meta["images"], size),
    )
    return Dataset(
        neurons=size,
        bias=meta["bias"],
        ymax=meta["ymax"],
        images=images,
        weights_indptr=np.load(root / "weights_indptr.npy", mmap_mode="r"),
        weights_indices=np.load(root / "weights_indices.npy", mmap_mode="r"),
        weights_values=np.load(root / "weights_values.npy", mmap_mode="r"),
        categories={
            depth: np.load(root / f"categories_l{depth}.npy")
            for depth in meta["category_depths"]
        },
    )
