# MPI Distributed Block Matrix Multiplication

This module introduces distributed-memory parallelism with MPI. It includes a
small `MPI_Reduce` example and a block matrix multiplication program that
partitions the rows of matrix **A** across MPI processes.

## Project Overview

The `mpi_block_matmul` program performs dense matrix multiplication

\[
    C = AB
\]

using the following communication pattern:

1. MPI rank 0 initializes the input matrices.
2. `MPI_Scatter` distributes a contiguous block of rows of **A** to every rank.
3. `MPI_Bcast` makes the complete **B** matrix available on every rank.
4. Each rank computes its local rows of **C** with an `i \rightarrow k \rightarrow j`
   loop order.
5. `MPI_Gather` collects the local result blocks on rank 0.

The matrix dimension `N` must be divisible by the number of MPI processes,
because every rank receives `N / world_size` rows.
Replace `N` in the commands below with the benchmark dimension you want to use.
The current `main.cpp` requires this argument even though it is not visible in
the shortened command shown in the recorded run log.

## Experimental Environment

| Parameter | Value |
|-----------|-------|
| Programming model | MPI distributed memory |
| MPI implementation | Open MPI (`mpirun`) |
| Language | C++17 |
| Build system | CMake |
| Matrix data type | `double` |
| Matrix size | Runtime argument `N` |
| MPI process count | 4 |

The benchmark timer starts after rank 0 has initialized the matrices and covers
the scatter, broadcast, local multiplication, and gather phases.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The executable is generated under `build/src/mpi_block_matmul/`. The current
CMake target is named `mpi_block_matual`, so the generated file is
`build/src/mpi_block_matmul/mpi_block_matual`. The benchmark commands below use
the intended `mpi_block_matmul` name from the run log; adjust the executable
path/name when running the CMake-built file directly.

## Run on One Node

For a four-process run on one machine:

```bash
/usr/bin/mpirun \
  --allow-run-as-root \
  -np 4 \
  --bind-to none \
  --tag-output \
  ./mpi_block_matmul N
```

Recorded output:

```text
[1,0]<stdout>:time: 0.252688 s
```

## Run on Two Nodes with JuChiYun Cloud

I implemented and validated a two-node MPI run on the JuChiYun cloud platform.
The hostfile lists the two compute nodes, with two MPI processes placed on each
node. The TCP interface options select the private `192.168.1.0/24` network for
inter-node MPI communication.

```bash
/usr/bin/mpirun \
  --allow-run-as-root \
  --hostfile /root/mpi-hosts \
  -np 4 \
  --map-by ppr:2:node \
  --bind-to none \
  --mca oob_tcp_if_include 192.168.1.0/24 \
  --mca btl_tcp_if_include 192.168.1.0/24 \
  --tag-output \
  ./mpi_block_matmul N
```

Recorded output from the two-node experiment:

```text
[1,0]<stdout>:time: 1.64053 s
```

The two-node run took approximately 6.49 times longer than the single-node
sample. This is expected for this implementation: every process needs the full
**B** matrix, and the scatter, broadcast, and gather operations cross the
network when ranks are distributed across nodes. The two measurements are
useful as a communication-overhead demonstration; they are not a controlled
scaling comparison unless the hardware, matrix size, and process placement are
kept identical.

## Result Analysis

### Distributed data decomposition

- **A** is partitioned by rows, so each rank stores only `N / P` rows locally.
- **B** is replicated on every rank by `MPI_Bcast`.
- **C** is produced locally and assembled on rank 0 with `MPI_Gather`.

This design is straightforward and minimizes the amount of code needed to
demonstrate MPI collectives. Its main limitation is the replicated **B** matrix:
memory use and broadcast cost grow with `N^2` on every rank.

### Communication and computation

The local computation uses the cache-friendly `i-k-j` loop order introduced in
the earlier CPU and OpenMP modules. MPI adds distributed parallelism, but it
also introduces synchronization and network transfer costs. On a single node,
collective operations use shared-memory transport; across the two cloud nodes,
they use the configured TCP network.

## MPI Basic Reduction Example

The `mpi_basic` executable demonstrates `MPI_Reduce`: every rank contributes a
local value and rank 0 prints the global maximum and the reduction time.

```bash
mpirun -np 4 ./mpi_basic
```

## Skills Demonstrated

- C++17 MPI programming with Open MPI
- Row-wise block decomposition for dense matrix multiplication
- Collective communication with `MPI_Scatter`, `MPI_Bcast`, `MPI_Gather`, and
  `MPI_Reduce`
- Multi-node process placement with hostfiles and `ppr:2:node`
- Inter-node MPI networking and benchmark interpretation
