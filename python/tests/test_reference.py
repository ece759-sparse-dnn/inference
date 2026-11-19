import numpy as np
import pytest
import scipy as sp

from sdnn_tools import dataset as ds
from sdnn_tools import layout
from sdnn_tools.reference import active_rows, infer, layer_step


def csr(rows) -> sp.sparse.csr_array:
    return sp.sparse.csr_array(np.asarray(rows, dtype=np.float32))


def dense_layer(y, w, bias, ymax):
    z = y @ w
    z = np.where(z != 0, z + bias, 0.0)
    return np.clip(z, 0.0, ymax)


def test_matches_dense_formula():
    rng = np.random.default_rng(0)
    y = (rng.random((8, 6)) < 0.4).astype(np.float32)
    w = np.where(rng.random((6, 6)) < 0.5, rng.normal(size=(6, 6)), 0.0)
    w = w.astype(np.float32)

    out = layer_step(csr(y), csr(w), bias=-0.3, ymax=32.0)

    np.testing.assert_allclose(out.toarray(), dense_layer(y, w, -0.3, 32.0), rtol=1e-6)


def test_returns_csr_with_same_shape():
    y = csr([[1, 0, 1], [0, 0, 0]])
    w = csr(np.eye(3))
    out = layer_step(y, w, bias=-0.1, ymax=32.0)
    assert out.format == "csr"
    assert out.shape == (2, 3)


def test_clamps_at_ymax():
    y = csr([[1, 1, 1]])
    w = csr([[20, 0], [20, 0], [20, 1]])
    out = layer_step(y, w, bias=0.0, ymax=32.0)
    np.testing.assert_allclose(out.toarray(), [[32.0, 1.0]])


def test_bias_skips_structural_zeros():
    y = csr([[1, 0], [0, 0]])
    w = csr([[2, 0], [0, 3]])
    out = layer_step(y, w, bias=0.5, ymax=32.0)
    np.testing.assert_allclose(out.toarray(), [[2.5, 0.0], [0.0, 0.0]])


def test_bias_skips_cancelled_sums():
    y = csr([[1, 1]])
    w = csr([[1, 2], [-1, 0]])
    out = layer_step(y, w, bias=0.5, ymax=32.0)
    np.testing.assert_allclose(out.toarray(), [[0.0, 2.5]])


def test_stores_no_explicit_zeros():
    y = csr([[1, 1, 0], [0, 1, 1]])
    w = csr([[0.1, 1, 0], [0.1, 1, 0], [0, 1, 0.2]])
    out = layer_step(y, w, bias=-0.3, ymax=32.0)
    assert np.all(out.data != 0)
    assert out.nnz == np.count_nonzero(out.toarray())


def test_empty_input_stays_empty():
    y = sp.sparse.csr_array((4, 3), dtype=np.float32)
    w = csr(np.eye(3))
    out = layer_step(y, w, bias=-0.3, ymax=32.0)
    assert out.nnz == 0
    assert out.shape == (4, 3)


def test_active_rows_basic():
    y = csr([[0, 0], [0, 3], [0, 0], [1, 1]])
    rows = active_rows(y)
    np.testing.assert_array_equal(rows, [1, 3])
    assert np.issubdtype(rows.dtype, np.integer)


def test_active_rows_none_active():
    rows = active_rows(sp.sparse.csr_array((3, 5), dtype=np.float32))
    assert rows.shape == (0,)
    assert np.issubdtype(rows.dtype, np.integer)


def test_active_rows_ignores_stored_zeros():
    y = sp.sparse.csr_array(
        (
            np.array([0.0, 4.0], dtype=np.float32),
            np.array([1, 0], dtype=np.int32),
            np.array([0, 1, 1, 2], dtype=np.int32),
        ),
        shape=(3, 2),
    )
    np.testing.assert_array_equal(active_rows(y), [2])


def tiny_dataset() -> ds.Dataset:
    """4 neurons, 2 layers (2 nnz per row, like the real data), 3 images."""
    n = 4
    w0 = csr([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]])
    w1 = csr(0.25 * w0.toarray())
    layers = [w0, w1]
    return ds.Dataset(
        neurons=n,
        bias=-0.6,
        ymax=32.0,
        images=csr([[1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]),
        weights_indptr=np.stack([w.indptr for w in layers]),
        weights_indices=np.stack([w.indices for w in layers]),
        weights_values=np.stack([w.data for w in layers]),
    )


def test_infer_tiny_network():
    np.testing.assert_array_equal(infer(tiny_dataset(), 2), [2])


def test_infer_zero_layers_returns_input_rows():
    np.testing.assert_array_equal(infer(tiny_dataset(), 0), [0, 2])


@pytest.mark.data
def test_neuron1024_l120_matches_truth():
    if not layout.bin_marker(1024).exists():
        pytest.skip("neuron1024 not converted")
    data = ds.load(1024)
    np.testing.assert_array_equal(infer(data, 120), data.categories[120])
