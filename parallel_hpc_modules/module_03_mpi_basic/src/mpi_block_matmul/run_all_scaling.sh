#!/bin/bash

set -e

# ====================== Configuration; edit as needed ======================
EXE="./mpi_matmul_scaling"
HOSTFILE="/root/mpi-hosts"
STRONG_N=4096
WEAK_LOCAL_ROWS=1024

# Common MPI network parameters
MPI_COMMON_OPTS="\
--bind-to none \
--mca oob_tcp_if_include 192.168.1.0/24 \
--mca btl_tcp_if_include 192.168.1.0/24 \
--tag-output"
# ============================================================

function run_single_node {
    echo "======================================"
    echo "[MODE] Single‑node scaling"
    rm -f strong_single.csv weak_single.csv

    # strong scaling: np 1 2 4
    for p in 1 2 4; do
        echo "  run strong np=$p"
        mpirun -np ${p} ${EXE} strong ${STRONG_N} >> strong_single.csv
    done

    # weak scaling: np 1 2 4
    for p in 1 2 4; do
        echo "  run weak np=$p"
        mpirun -np ${p} ${EXE} weak ${WEAK_LOCAL_ROWS} >> weak_single.csv
    done
    echo "[Single‑node done] output: strong_single.csv weak_single.csv"
}

function run_multi_node {
    echo "======================================"
    echo "[MODE] Multi‑node (2‑node) scaling"
    rm -f strong_multi.csv weak_multi.csv

    MPIRUN_BASE="mpirun --allow-run-as-root --hostfile ${HOSTFILE} ${MPI_COMMON_OPTS}"

    # strong scaling
    # np=1: node‑0 only
    echo "  run strong np=1"
    ${MPIRUN_BASE} -np 1 --map-by ppr:1:node ${EXE} strong ${STRONG_N} >> strong_multi.csv

    # np=2: 1 proc per node
    echo "  run strong np=2"
    ${MPIRUN_BASE} -np 2 --map-by ppr:1:node ${EXE} strong ${STRONG_N} >> strong_multi.csv

    # np=4: 2 proc per node, cross‑node
    echo "  run strong np=4"
    ${MPIRUN_BASE} -np 4 --map-by ppr:2:node ${EXE} strong ${STRONG_N} >> strong_multi.csv

    # weak scaling
    echo "  run weak np=1"
    ${MPIRUN_BASE} -np 1 --map-by ppr:1:node ${EXE} weak ${WEAK_LOCAL_ROWS} >> weak_multi.csv

    echo "  run weak np=2"
    ${MPIRUN_BASE} -np 2 --map-by ppr:1:node ${EXE} weak ${WEAK_LOCAL_ROWS} >> weak_multi.csv

    echo "  run weak np=4"
    ${MPIRUN_BASE} -np 4 --map-by ppr:2:node ${EXE} weak ${WEAK_LOCAL_ROWS} >> weak_multi.csv

    echo "[Multi‑node done] output: strong_multi.csv weak_multi.csv"
}

# -------- entry point --------
if [ $# -ne 1 ]; then
    echo "Usage:"
    echo "  ./run_all_scaling.sh single      # only single‑node scaling"
    echo "  ./run_all_scaling.sh multi       # only two‑node scaling"
    echo "  ./run_all_scaling.sh all         # run both single + multi node"
    exit 1
fi

case "$1" in
single)
    run_single_node
    ;;
multi)
    run_multi_node
    ;;
all)
    run_single_node
    run_multi_node
    ;;
*)
    echo "unknown argument, use single / multi / all"
    exit 1
;;
esac

echo "===== All experiments finished ====="
ls -l *.csv
