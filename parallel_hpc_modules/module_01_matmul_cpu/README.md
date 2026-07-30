# Matrix Multiplication Loop Reordering Experiment
This experiment verifies the importance of **CPU cache spatial locality** via loop reordering on dense matrix multiplication.

## Experiment Configuration
- Matrix dimension: $N = 1024$ ($1024 \times 1024$, `double` precision)
- Compiler: GCC
- Build mode: **Release (-O3)**
- Platform: Ubuntu Linux

## Benchmark Raw Results
- ijk (naive loop order): 5.32859 s
- ikj (optimized loop order): 0.48581 s

**Calculated Speedup**: $5.32859 \div 0.48581 \approx \boldsymbol{10.97\times}$

## Result Analysis
1. **ijk implementation**
    Loop nesting: `i → j → k`
    Inner variable `k`. Memory access for matrix $B$ is $B[k\cdot N + j]$.
    $i,j$ are fixed, $k$ keeps changing → **column-wise access on B**.
    Memory addresses jump discontinuously, resulting in severe cache misses and low execution efficiency.

2. **ikj implementation**
    Loop nesting: `i → k → j`
    Inner variable `j`. Memory access for matrix $B$ is $B[k\cdot N + j]$.
    $i,k$ are fixed, $j$ increases sequentially → **row-wise continuous access on B**.
    Continuous memory access fully exploits CPU cache line spatial locality, drastically reducing cache misses.

## Key Conclusion
Loop reordering does **not change the mathematical computation result**, only reorganizes the memory access sequence. Proper loop transformation can yield over **10× performance gain** purely by improving cache utilization, without any algorithmic optimization to reduce floating-point operations.

> Note: Under Debug build (-O0), the performance gap between two versions is negligible. Compiler optimization must be enabled to observe cache locality effects.