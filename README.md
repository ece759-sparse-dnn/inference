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

## Running on Euler (Slurm)

Timed or heavy runs go through Slurm; the login node is for editing, building
and short tests only. All scripts live in `scripts/slurm/` and are run from the
repo root.

### One-time setup

Point `SDNN_DATA_DIR` at the team's shared, already-converted dataset so nobody
needs their own copy (add it to your `~/.bashrc`):

```bash
export SDNN_DATA_DIR=/path/to/shared/sdnn-data    # must contain bin/neuron<size>/
```

`sbatch` passes your environment through, so jobs see it too.

### Build

On the login node (configuring downloads GoogleTest):

```bash
scripts/slurm/build.sh
```

This loads the modules from [Slurm Module Setup](#slurm-module-setup) and builds
the `cuda-euler` preset into `build/cuda-euler/`.

### Run one backend

```bash
sbatch [-c CPUS] [--gres=gpu:1] scripts/slurm/run.sbatch BACKEND [SIZE] [LAYERS] [THREADS] [CSV]
```

| Argument | Default |
| -------- | ------- |
| `BACKEND` | required: `serial`, `openmp` or `cuda` |
| `SIZE` | `1024` |
| `LAYERS` | `120` |
| `THREADS` | the CPUs allocated with `-c` (must not exceed them) |
| `CSV` | `results/euler.csv` |

```bash
sbatch              scripts/slurm/run.sbatch serial 1024 120
sbatch -c 16        scripts/slurm/run.sbatch openmp 1024 120 16
sbatch --gres=gpu:1 scripts/slurm/run.sbatch cuda   1024 120
```

Each job appends one row to the CSV (with a header if the file is new) and
writes its log to `sdnn-run-<jobid>.out`. The job fails if the result does not
match the truth. For OpenMP runs, set `OMP_SCHEDULE` (e.g. `dynamic,64`) before
`sbatch`. The CSV does not record it, but the job log does.

Check on jobs with `squeue -u $USER`.

### CPU benchmark and schedule sweep

```bash
sbatch scripts/slurm/bench.sbatch
```

Runs the unit tests for the CPU backends, checks `serial` and `openmp` against
the truth, then sweeps OpenMP schedules and thread counts. Results go to
`results/<jobid>/sweep.csv` and `summary.txt`. Override with `SIZE`, `DEPTHS`,
`THREADS`, `SCHEDULES` and `REPS` exported before `sbatch`; see the header of
`scripts/slurm/bench.sbatch`.

> [!WARNING]
> `cuda-euler` uses `-march=native`, so CPU code is tuned for the machine that
> runs `build.sh`. If the login and compute nodes have different CPUs, CPU
> timings may be off, and newer instructions may even crash on older nodes.

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
