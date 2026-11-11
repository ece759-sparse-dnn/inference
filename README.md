# inference

Serial, OpenMP, and CUDA implementations of the
[Sparse Deep Neural Network Graph Challenge](https://graphchallenge.mit.edu/challenges/)
inference, with benchmarks.

## Status

| Component | State |
| --------- | ----- |
| Dataset download / conversion (`sdnn-tools`) | working |
| Python reference inference (`sdnn-tools verify`) | working |
| C++ dataset loader (`sdnn::load_dataset`) | working |
| Benchmark driver (`sdnn_run`) | working |
| Serial backend | not implemented |
| OpenMP backend | not implemented |
| CUDA backend | not implemented |

## Dependencies

- GCC (15.2)
- CMake (Minimum 3.25)
- Ninja (1.13)
- CUDA (13.0.0)
- Python (3.12)
- UV (0.13)

### Slurm Module Setup
```bash
module purge
module load gnu15/15.2.0 nvidia/cuda/13.0.0
```

## Quickstart + Usage

For help / info,
```bash
uv run --project python sdnn-tools -h
```
To download datasets,
```bash
uv run --project python sdnn-tools download -h
uv run --project python sdnn-tools download -s 1024
```
To serialize downloaded datasets,
```bash
uv run --project python sdnn-tools convert -h
uv run --project python sdnn-tools convert -s 1024
```
To verify that inference on datasets gives correct results,
```bash
uv run --project python sdnn-tools verify -h
uv run --project python sdnn-tools verify -s 1024 --layers 120
```

For building the C++ applications,
```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

To run the serial inference backend,
```bash
./build/debug/apps/sdnn_run --backend serial --size 1024 --layers 120
```

To run C++ testing,
```bash
ctest --preset debug
```

## Datasets

> [!WARNING]
> The data sets for 4096 Neurons onwards are quite large.

Datasets are downloaded and serialized by the `sdnn-tools` Python package.
See [`python/README.md`](python/README.md) for info about subcommands, the data root and
 directory layout, and
[`python/src/sdnn_tools/README.md`](python/src/sdnn_tools/README.md) for the
current serialized format.

## Building w/ CMake

| Preset | Description |
| ------ | ----------- |
| `debug` | Debug, CPU only |
| `asan` | Debug + ASan/UBSan (Linux) |
| `release` | Release, CPU only, `-march=native` |
| `cuda-euler` | Release + CUDA, sm_89 |
| `cuda-chtc` | Release + CUDA, sm_80/89/90/120 |

```bash
cmake --preset <preset>
cmake --build --preset <preset>
ctest --preset <preset>    # debug, asan, release only
```

> [!WARNING]
> `-march=native` targets the CPU the build itself runs on.

## Running

```bash
./build/<preset>/apps/sdnn_run \
  [--backend NAME] [--size N] [--layers L] [--data-dir DIR] [--csv FILE]
```

| Option | Default |
| ------ | ------- |
| `--backend` | `serial` |
| `--size` | `1024` |
| `--layers` | `120` |
| `--data-dir` | `$SDNN_DATA_DIR`, else `inference/data` |
| `--csv` | none |

The program loads `<data>/bin/neuron<size>`, benchmarks the backend's
inference, and checks the active rows against the truth categories. One CSV
row is always printed to stdout with the format

```text
backend,neurons,layers,images,threads,seconds,edges_per_second,pass
```

`OMP_NUM_THREADS` should be set for OpenMP runs.

## Adding Backends

A backend is a function with signature

```cpp
std::vector<std::int32_t> fn(const sdnn::Dataset &dataset, int layers);
```

that runs `layers` layers over `dataset.images` and returns the sorted,
0-indexed rows that are still active. 

For each layer, `Y = min(max(Y W + bias, 0), ymax)`, where `bias` is added only to entries
that are nonzero in `Y W`.

To add it to the project,
1. Declare it in `include/sdnn/backend.hpp`.
2. Register it in `backends()` in `src/backends.cpp`.
3. Add its source to `src/CMakeLists.txt`.
