# sdnn-tools

## Dataset Management

From the project root, run
```shell
uv run --project python sdnn-tools -h
```
or
```shell
uv run sdnn-tools -h
```
from the `python/` subdirectory for information about using `sdnn-tools`.

The data root directory can be specified with environment variable `$SDNN_DATA_DIR`.

Otherwise, it is `inference/data`.

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
