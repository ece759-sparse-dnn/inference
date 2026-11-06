# inference

https://graphchallenge.mit.edu/challenges/

## Dependencies

- GCC (15.2)
- CMake (Minimum 3.24)
- CUDA (13.0.0)
- Python (3.12)
- UV (0.13)

## Slurm Module Setup
```bash
module purge
module load gnu15/15.2.0 nvidia/cuda/13.0.0
```

## Downloading Datasets
> [!WARNING]
> The data sets for 4096 Neurons onwards are quite large.

Info:
```bash
uv run scripts/get_data.py --help
```
Fetching the 1024 Neuron data (recommended):
```bash
uv run scripts/get_data.py data/
```
Fetching datasets for specific size:
```bash
uv run scripts/get_data.py data/ 4096
```
