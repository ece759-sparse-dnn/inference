## Data Serialization

Each `<data>/bin/neuron{N}/` directory holds one dataset. All arrays are
little-endian, C order, and 0-indexed. `L` is the number of layers, `I` the
number of images, and `nnz` the number of nonzeros per layer.

| filename | datatype | shape |
| -------- | -------- | ----- |
| `weights_indptr.npy` | `i32` | `(L, N + 1)` |
| `weights_indices.npy` | `i32` | `(L, nnz)` |
| `weights_values.npy` | `f32` | `(L, nnz)` |
| `images_indptr.npy` | `i32` | `(I + 1,)` |
| `images_indices.npy` | `i32` | `(images_indptr[-1],)` |
| `images_values.npy` | `f32` | `(images_indptr[-1],)` |
| `categories_l{depth}.npy` | `i32` | `(active rows,)` |

Each weights row `i` is the CSR matrix for layer `i`, with column indices
sorted within each row. Images form one `(I, N)` CSR matrix. Categories are
the sorted rows still active after `depth` layers.

### `meta.json`

| key | description |
| --- | ----------- |
| `version` | serialization format version (`BIN_VERSION`) |
| `neurons` | `N` |
| `layers` | `L` |
| `nnz_per_layer` | `nnz` |
| `images` | `I` |
| `category_depths` | depths with a `categories_l{depth}.npy` |
| `bias` | bias added to nonzero entries each layer |
| `ymax` | upper clamp of the activation |

`.done` is written last as a marker. Its absence means the conversion is missing / corrupted. 

Note:
C++/CUDA consumers are responsible for casting
serialized data into larger types to avoid overflow.
The Python conversions only guarantee that the serialized
data is correct as-is, and does not allocate space for
larger types than necessary.
