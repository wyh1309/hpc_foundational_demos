# MPI Distributed Memory and Scaling Experiment

This module introduces MPI collective communication and distributed-memory
parallelism. It contains a basic `MPI_Reduce` example and a row-blocked dense
matrix multiplication benchmark with strong- and weak-scaling modes.

## Project Overview

The `mpi_basic` program demonstrates a reduction in which every process sends
its local value to rank 0. The root process prints the global maximum and the
time spent in `MPI_Reduce`.

The `mpi_block_matmul` program computes

```text
C = A * B
```

with the following communication pattern:

1. Rank 0 initializes the global matrices.
2. `MPI_Scatter` distributes contiguous rows of `A` to all ranks.
3. `MPI_Bcast` replicates the complete `B` matrix on every rank.
4. Each rank computes its local rows of `C` using the cache-friendly `i-k-j`
   loop order.
5. `MPI_Gather` assembles the local results on rank 0.

The implementation is intentionally simple and is designed to expose the
trade-off between computation, collective communication, and network traffic.

## Experimental Environment

| Parameter | Value |
|-----------|-------|
| Programming model | MPI distributed memory |
| MPI implementation | Open MPI (`mpirun`) |
| Language | C++17 |
| Build system | CMake |
| Matrix data type | `double` |
| Single-node process counts | 1, 2, 4 |
| Strong-scaling matrix size | `GLOBAL_N = 4096` |
| Weak-scaling local rows | `1024` rows per process |
| Multi-node layout | 2 nodes, up to 2 processes per node |

## Build

From this directory:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The build produces:

```text
build/src/mpi_basic/mpi_basic
build/src/mpi_block_matmul/mpi_block_matmul
```

The build requires a C++17 compiler, CMake, and an MPI implementation with
`MPI::MPI_CXX` support.

## Run the Basic Reduction Example

```bash
mpirun -np 4 ./build/src/mpi_basic/mpi_basic
```

Each rank contributes `rank * 10`; with four processes, rank 0 reports a global
maximum of `30`.

## Matrix Multiplication Command Line

The matrix multiplication executable supports two modes:

```text
Strong scaling: ./mpi_block_matmul strong GLOBAL_N [BASELINE_TIME]
Weak scaling:   ./mpi_block_matmul weak LOCAL_ROWS_PER_PROC
```

For strong scaling, `GLOBAL_N` must be divisible by the MPI process count and
the global problem size stays fixed. For weak scaling, each rank owns the same
number of local rows, so `GLOBAL_N` grows with the number of processes.

The program prints CSV fields containing the mode, process count, global matrix
size, local row count, wall time, strong-scaling speedup, and efficiency. The
timed region includes the barrier, scatter, broadcast, local multiplication,
gather, and final barrier. Matrix initialization is outside the timed region.

## Single-Node Scaling

Run the complete single-node experiment with:

```bash
cd build/src/mpi_block_matmul
../../../../module_03_mpi_basic/src/mpi_block_matmul/run_all_scaling.sh single
```

The script uses `GLOBAL_N=4096` for strong scaling and `1024` local rows per
process for weak scaling. It records `strong_single.csv` and `weak_single.csv`
in the current directory. The script can also be run directly from the source
directory after changing `EXE` to the built executable path.

Recorded single-node results:

### Strong Scaling

| Processes | Global N | Local rows | Wall time (s) | Speedup | Efficiency |
|----------:|---------:|-----------:|--------------:|--------:|-----------:|
| 1 | 4096 | 4096 | 44.5704 | 1.000 | 100.0% |
| 2 | 4096 | 2048 | 24.4734 | 1.821 | 91.1% |
| 4 | 4096 | 1024 | 12.8179 | 3.477 | 86.9% |

The four-process run reaches a 3.48x speedup and 86.9% parallel efficiency.
The decrease in efficiency is caused by collective communication and
synchronization overhead, while the local matrix multiplication remains the
dominant computation.

### Weak Scaling

| Processes | Global N | Local rows | Wall time (s) |
|----------:|---------:|-----------:|--------------:|
| 1 | 1024 | 1024 | 0.406914 |
| 2 | 2048 | 1024 | 2.62954 |
| 4 | 4096 | 1024 | 13.0075 |

Weak-scaling speedup is not reported because the problem size grows with the
number of processes. The wall time increases substantially because every rank
receives the full `B` matrix and the amount of communication grows with the
global matrix size.

## Two-Node Scaling

The script's multi-node mode expects a hostfile at `/root/mpi-hosts` and uses
the private `192.168.1.0/24` interface for TCP communication:

```bash
cd build/src/mpi_block_matmul
../../../../module_03_mpi_basic/src/mpi_block_matmul/run_all_scaling.sh multi
```

The hostfile should provide two nodes. The script places one process per node
for two-process runs and two processes per node for four-process runs.

Recorded two-node results:

### Strong Scaling

| Processes | Global N | Local rows | Wall time (s) | Speedup | Efficiency |
|----------:|---------:|-----------:|--------------:|--------:|-----------:|
| 1 | 4096 | 4096 | 44.5444 | 1.000 | 100.0% |
| 2 | 4096 | 2048 | 47.1697 | 0.944 | 47.2% |
| 4 | 4096 | 1024 | 34.9113 | 1.276 | 31.9% |

Compared with the single-node results, the two-node runs are slower because
`MPI_Scatter`, `MPI_Bcast`, and `MPI_Gather` must transfer data over the
network. In particular, replicating the full `B` matrix on every process makes
communication a significant part of the runtime.

### Weak Scaling

| Processes | Global N | Local rows | Wall time (s) |
|----------:|---------:|-----------:|--------------:|
| 1 | 1024 | 1024 | 0.407114 |
| 2 | 2048 | 1024 | 8.27128 |
| 4 | 4096 | 1024 | 34.7854 |

These results demonstrate that the current row-blocked algorithm is a useful
MPI teaching example, but not a communication-optimal large-scale matmul
design. A 2D decomposition such as SUMMA would distribute both `A` and `B`
and reduce the replicated-data bottleneck.

## Core Technical Points

| Technique | Role |
|-----------|------|
| `MPI_Scatter` | Distributes contiguous row blocks of `A` |
| `MPI_Bcast` | Replicates `B` on every rank |
| `MPI_Gather` | Collects local `C` blocks on rank 0 |
| `MPI_Reduce` | Computes a global maximum in the basic example |
| Strong scaling | Keeps global work fixed while increasing process count |
| Weak scaling | Keeps local work fixed while increasing global problem size |
| `i-k-j` loop order | Improves local matrix access locality |

## Skills Demonstrated

- C++17 MPI programming with Open MPI
- Collective communication and rank/process management
- Row-wise block decomposition for dense matrix multiplication
- Strong- and weak-scaling benchmark design
- Single-node and multi-node process placement
- Interpreting communication overhead and parallel efficiency

  > Implemented CPU-based distributed matrix multiplication with MPI using row-wise
  > decomposition and collective communication (MPI_Scatter, MPI_Bcast, MPI_Gather).
  > Evaluated single-node and two-node scaling, achieving 3.48x speedup and 86.9%
  > parallel efficiency on 4 processes, and analyzed cross-node communication
  > overhead.
