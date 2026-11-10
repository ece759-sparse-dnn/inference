# sdnn-tools

Dataset management and a reference inference for the Sparse DNN Graph
Challenge.

## Usage

From the project root, run
```shell
uv run --project python sdnn-tools -h
```
or
```shell
uv run sdnn-tools -h
```
from the `python/` subdirectory.

| Subcommand | Description |
| ---------- | ----------- |
| `download -s SIZE...` | Download datasets into `<data>/raw` |
| `convert -s SIZE... [--force]` | Serialize `<data>/raw` TSVs into `<data>/bin` |
| `verify -s SIZE... [--layers L]` | Run the SciPy reference inference and compare with truth vals |

`SIZE` is one of `1024`, `4096`, `16384`, `65536` (default `1024`). 
`L` is one of `120`, `480`, `1920` (default `120`).

## Data Root

The data root directory can be specified the following ways:
- command-line argument `--data-dir DATA_DIR`
- environment variable `$SDNN_DATA_DIR`

By default, it is `inference/data`.

The directory layout for downloaded / serialized datsets looks like:
```text
data/
  raw/
    .neuron1024.done
    neuron1024/
      n1024-l1.tsv
      ...
    neuron4096/
      ...
    mnist/
      sparse-images-1024.tsv
      ...
    categories/
      neuron1024-l120-categories.tsv
      ...
  bin/
    neuron1024/
      .done
      meta.json
      weights_indptr.npy
      weights_indices.npy
      weights_values.npy
      images_indptr.npy
      images_indices.npy
      images_values.npy
      categories_l120.npy
      categories_l480.npy
      categories_l1920.npy
    neuron4096/
      ...
```

See [`src/sdnn_tools/README.md`](src/sdnn_tools/README.md) for the
serialized format.
