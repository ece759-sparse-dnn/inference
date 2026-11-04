# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=3.1.1",
#     "scipy>=1.18.1",
# ]
# ///

import sys
from pathlib import Path
import argparse

import pandas as pd
import scipy as sp
import numpy as np


def convert_layer(tsv_file: Path) -> None:
    assert tsv_file.suffix == ".tsv"

    df = pd.read_csv(
        tsv_file,
        sep="\t",
        header=None,
        names=["src", "dst", "val"],
        dtype={"src": np.int32, "dst": np.int32, "val": np.float64},
        engine="pyarrow",
    )

    src = df["src"].values
    dst = df["dst"].values
    val = df["val"].values

    # tsv are already 1-indexed
    num_rows = src.max()
    num_cols = dst.max()

    csr = sp.sparse.coo_array((val, (src, dst)), shape=(num_rows, num_cols)).tocsr()
    sp.sparse.save_npz(tsv_file.with_suffix(".npz"), csr)


def main():
    pass


if __name__ == "__main__":
    sys.exit(main())
