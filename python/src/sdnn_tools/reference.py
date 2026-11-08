import argparse as ap
import time

import numpy as np
import scipy as sp

from . import dataset as ds


def layer_step(
    y: sp.sparse.csr_array, w: sp.sparse.csr_array, bias: float, ymax: float
) -> sp.sparse.csr_array:
    # Y_next = min(ReLU(y @ w + bias), ymax)
    y_next = y @ w
    y_next.eliminate_zeros()
    y_next.data += bias
    np.clip(y_next.data, min=0, max=ymax, out=y_next.data)
    y_next.eliminate_zeros()
    print(
        f"  the next tensor has {y_next.nnz} nnz entries and {len(active_rows(y))} active rows"
    )
    return y_next


def active_rows(y: sp.sparse.csr_array) -> np.ndarray:
    return np.unique(y.nonzero()[0])


def infer(dataset: ds.Dataset, num_layers: int) -> np.ndarray:
    y = dataset.images
    for i in range(num_layers):
        y = layer_step(y, dataset.layer(i), dataset.bias, dataset.ymax)
    return active_rows(y)


def verify(config: ap.Namespace) -> None:
    for size in config.s:
        data = ds.load(size)
        truth = data.categories[config.layers]
        start = time.perf_counter()
        rows = infer(data, config.layers)
        elapsed = time.perf_counter() - start

        edges = data.images.shape[0] * data.weights_indices.shape[1] * config.layers
        ok = np.array_equal(rows, truth)
        print(
            f"  {size}-l{config.layers}: {'PASS' if ok else 'FAIL'} "
            f"({len(rows)} active, truth {len(truth)}) "
            f"{elapsed:.2f}s, {edges / elapsed:.3e} edges/s"
        )
        if not ok:
            missing = np.setdiff1d(truth, rows)
            extra = np.setdiff1d(rows, truth)
            print(f"    missing {missing[:10]}..., extra {extra[:10]}...")
