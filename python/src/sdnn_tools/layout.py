"""Data directory layout."""

import os
from pathlib import Path

_data_root: Path | None = None


def repo_root() -> Path:
    """First parent of this file containing flake.nix or .git."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "flake.nix").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError(f"could not locate repository root from {current}")


def set_data_root(path: Path | None) -> None:
    """Override the data directory root."""
    global _data_root
    _data_root = path.resolve() if path is not None else None


def data_root() -> Path:
    if _data_root is not None:
        return _data_root
    if env := os.environ.get("SDNN_DATA_DIR"):
        return Path(env).resolve()
    return repo_root() / "data"


def _mkdir(path: Path) -> Path:
    """Create directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir() -> Path:
    """Subdirectory where unserialized datasets are stored."""
    return _mkdir(data_root() / "raw")


def raw_layers_dir(size: int) -> Path:
    """Subdirectory where unserialized NN layers are stored."""
    return _mkdir(raw_dir() / f"neuron{size}")


def raw_layers_marker(size: int) -> Path:
    """File indicator for whether NN layers are downloaded."""
    return raw_dir() / f".neuron{size}.done"


def raw_layer_tsv(size: int, layer: int) -> Path:
    """File containing a layer's (source, dest, weight) triples."""
    return raw_layers_dir(size) / f"n{size}-l{layer}.tsv"


def raw_images_dir() -> Path:
    """Subdirectory where unserialized MNIST images are stored."""
    return _mkdir(raw_dir() / "mnist")


def raw_images_tsv(size: int) -> Path:
    """File containing MNIST (image, neuron, weight) triples."""
    return raw_images_dir() / f"sparse-images-{size}.tsv"


def raw_categories_dir() -> Path:
    """Subdirectory where unserialized categories are stored."""
    return _mkdir(raw_dir() / "categories")


def raw_categories_tsv(size: int, depth: int) -> Path:
    """File containing a category's truth values."""
    return raw_categories_dir() / f"neuron{size}-l{depth}-categories.tsv"


def bin_dir() -> Path:
    """Subdirectory where serialized datasets are stored."""
    return _mkdir(data_root() / "bin")


def bin_network_dir(size: int) -> Path:
    """Subdirectory where serialized NN layers are stored."""
    return _mkdir(bin_dir() / f"neuron{size}")


def bin_marker(size: int) -> Path:
    """File indicator for whether NN layers are serialized."""
    return bin_network_dir(size) / ".done"
