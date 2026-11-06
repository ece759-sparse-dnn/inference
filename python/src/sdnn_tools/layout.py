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
    """Override the data root."""
    global _data_root
    _data_root = path.resolve() if path is not None else None


def data_root() -> Path:
    if _data_root is not None:
        return _data_root
    if env := os.environ.get("SDNN_DATA_DIR"):
        return Path(env).resolve()
    return repo_root() / "data"


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir() -> Path:
    return _mkdir(data_root() / "raw")


def raw_layers_dir(size: int) -> Path:
    return _mkdir(raw_dir() / f"neuron{size}")


def raw_layers_marker(size: int) -> Path:
    return raw_dir() / f".neuron{size}.done"


def raw_layer_tsv(size: int, layer: int) -> Path:
    """Layer TSV, 1-indexed layer number as in the challenge files."""
    return raw_layers_dir(size) / f"n{size}-l{layer}.tsv"


def raw_images_dir() -> Path:
    return _mkdir(raw_dir() / "mnist")


def raw_images_tsv(size: int) -> Path:
    return raw_images_dir() / f"sparse-images-{size}.tsv"


def raw_categories_dir() -> Path:
    return _mkdir(raw_dir() / "categories")


def raw_categories_tsv(size: int, layers: int) -> Path:
    return raw_categories_dir() / f"neuron{size}-l{layers}-categories.tsv"


def bin_dir() -> Path:
    return _mkdir(data_root() / "bin")


def bin_network_dir(size: int) -> Path:
    return _mkdir(bin_dir() / f"neuron{size}")


def bin_marker(size: int) -> Path:
    return bin_network_dir(size) / ".done"
