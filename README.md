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
| Serial backend | working |
| OpenMP backend | working |
| CUDA backend | working |

## Dependencies

- GCC (15.2)
- CMake (Minimum 3.25)
- Ninja (1.13)
- CUDA (13.0.0)
- Python (3.12)
- UV (0.13)

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

To run the OpenMP inference backend,
```bash
./build/debug/apps/sdnn_run --backend openmp --size 1024 --layers 120
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

## CUDA active row compaction

Setting `cuda.compact_after=K` packs the active images after layer K (1 ≤ K < L);
the `cuda_compact8` preset uses K = 8.
Subsequent layers run on the packed rows, and the backend returns the original
image IDs. Timing includes detection, packing and ID recovery. Both activation
buffers still have their original capacity.

```bash
./build/cuda-euler/apps/sdnn_run --backend cuda --size 1024 --layers 120
./build/cuda-euler/apps/sdnn_run --backend cuda_compact8 --size 1024 --layers 120
./build/cuda-euler/apps/sdnn_run --backend cuda --set cuda.compact_after=7
```

To repeat the layer sweep, run the following inside a single GPU allocation
with `SDNN_DATA_DIR` pointing to the converted official data.

```bash
python3 scripts/sweep_compaction.py ./build/cuda-euler/apps/sdnn_run timings.csv
```

The sweep measures every candidate in three shuffled rounds. It retests the
fastest five alongside layers 8 and 10 and the uncompressed baseline in balanced
order. In Euler job 128922 on one RTX 4000 SFF Ada, confirmation medians were
324.0 ms at layer 7, 324.4 ms at layer 8 and 1,440.9 ms without compaction.

## Running

```bash
./build/<preset>/apps/sdnn_run \
  [--backend NAME] [--size N] [--layers L] [--data-dir DIR] [--csv FILE] \
  [--set KEY=VALUE]...
```

| Option | Default |
| ------ | ------- |
| `--backend` | `serial` |
| `--size` | `1024` |
| `--layers` | `120` |
| `--data-dir` | `$SDNN_DATA_DIR`, else `inference/data` |
| `--csv` | none |
| `--set` | the backend preset's options |

`--backend` takes a backend or one of its presets (`sdnn_run` with no
arguments lists them). `--set` overrides one option of that preset and may be
repeated; the CSV `backend` column then reads e.g. `cuda+cuda.compact_after=7`.

| Key | Meaning |
| --- | ------- |
| `cuda.compact_after` | compact active rows after layer K; 0 disables |

The program loads `<data>/bin/neuron<size>`, benchmarks the backend's
inference, and checks the active rows against the truth categories. One CSV
row is always printed to stdout with the format

```text
backend,neurons,layers,images,threads,seconds,edges_per_second,pass
```

`OMP_NUM_THREADS` should be set for OpenMP runs.

## Running on Euler (Slurm)

Euler is for building and testing on small datasets (1024 neurons). 
Larger runs should use CHTC (files are too big for euler). 
Practically speaking, shared dataset directories aren't an option (max size 1000MB)

Use Slurm. All scripts live in `scripts/slurm/` and are run from the repo root.

### Datasets

From the login node, run
```bash
scripts/slurm/data.sh
```

This installs uv into `~/.local/bin` if it isn't on your `PATH`, then
runs `sdnn-tools download` and `convert` into `data/` in the repo, where
`sdnn_run` and the Slurm scripts look by default. Sizes that are already
converted are skipped. Set `PRUNE_RAW=1` to delete the raw TSVs after
conversion.

To keep the data elsewhere, export `SDNN_DATA_DIR` (in `~/.zshrc`, or
`~/.bashrc` for bash).

> [!WARNING]
> Data counts against your home quota. Converted sizes are roughly 0.5 GB
> (1024), 2 GB (4096), 9 GB (16384) and 35 GB (65536), and the raw downloads
> are larger.

### Build

```bash
scripts/slurm/build.sh
```

This configures the `cuda-euler` preset on the login node (a few seconds; it
downloads GoogleTest, and compute nodes may not have internet access), then
submits `scripts/slurm/build.sbatch` to compile on a compute node. Compiling
there keeps long builds off the login node and makes `-march=native` target
the compute nodes' CPU. Extra arguments go to `sbatch`, e.g.
`scripts/slurm/build.sh -c 16`.

The build job's ID is printed; queue runs behind it so they start only if the
build succeeds:

```bash
sbatch --dependency=afterok:<jobid> --gres=gpu:1 scripts/slurm/run.sbatch cuda 1024 120
```

The build log goes to `sdnn-build-<jobid>.out`. To rebuild after code changes
without reconfiguring, `sbatch scripts/slurm/build.sbatch`.

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
> `cuda-euler` uses `-march=native`, so CPU code is tuned for the node that
> compiled it. The build job records that node's CPU model in
> `build/cuda-euler/build-cpu.txt`, and `run.sbatch` and `bench.sbatch` warn
> when they land on a different one. If they do, check whether the partition
> mixes CPU types (`sinfo -N -o "%N %P %c"`, `scontrol show node <name>`) and
> pin the build and runs to one type with `--constraint` or `--nodelist`.

## Adding Backends

A backend is a function with signature

```cpp
std::vector<std::int32_t> fn(const sdnn::Dataset &dataset, int layers,
                             const sdnn::InferOptions &options);
```

that runs `layers` layers over `dataset.images` and returns the sorted,
0-indexed rows that are still active. 

For each layer, `Y = min(max(Y W + bias, 0), ymax)`, where `bias` is added only to entries
that are nonzero in `Y W`.

To add it to the project,
1. Declare it in `include/sdnn/backend.hpp`.
2. Register it in `backends()` in `src/backends.cpp`.
3. Add its source to `src/CMakeLists.txt`.

## Adding an optimization

Optimizations are options of an existing backend, not new backends, so they
share its code, tests and benchmark path.

1. Add a field to the backend's options struct (e.g. `CudaOptions`) in
   `include/sdnn/backend.hpp`, defaulting to the unoptimized behavior.
2. Give it a key in `set_option` in `src/backends.cpp`, so `--set` and sweeps
   can reach it.
3. Optionally add a named preset to `backends()`. Every preset is checked
   against the truth data by `BackendTest`.
4. Branch on the option in host code. If it changes what a kernel does, make
   the variant a template parameter of the kernel in `src/cuda/kernels.cu` and
   pick the instantiation on the host, so inner loops carry no runtime
   branches.
