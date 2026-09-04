# 下一步工作

## 当前状态

CPU 测试已经通过，CUDA forward demo 和 benchmark 也已经实现并在 GPU 上运行成功。

当前基线结果：

- 配置：`4096 x 256`，`ds = 0.01`；
- kernel：约 `0.158 ms`；
- end-to-end：约 `0.828 ms`；
- CPU serial：约 `12.028 ms`；
- 端到端加速比：约 `14.5x`；
- H2D：约 `0.662 ms`，约占 end-to-end 的 80%；
- RMSE 和最大绝对误差：`0`。

因此，当前首要问题不是继续增加功能，而是用 profiler 验证数据传输瓶颈，并完成一次有证据的优化。

## 现在应该做什么

### 1. 保存可复现的基线

在 GPU 节点上保存以下信息：

```bash
nvidia-smi
nvcc --version
cmake --build build-cuda --parallel
./build-cuda/src/forward_demo 4096 256
./build-cuda/src/benchmark 5 20 > results/baseline.csv
```

同时记录 GPU 型号、驱动版本、CUDA 版本、编译选项和运行日期。

benchmark 的 CSV 表头已修正为与实际数据列顺序一致。输出顺序是：

```text
...,cpu_ms,speedup_end_to_end,samples_per_second_millions,
   h2d_gbps,d2h_gbps,device_memory_mib,max_abs_error
```

可以直接保存为 CSV，用于后续绘图和报告。

### 2. 运行 Compute Sanitizer

先确认没有越界、非法访问或同步问题：

```bash
compute-sanitizer --tool memcheck \
  ./build-cuda/src/forward_demo 1024 64
```

一般尺寸和线程边界可以直接用 `forward_demo` 快速检查：

```text
num_rays = 1、非 32 的倍数、大规模
num_samples = 1、64、256
```

`sigma = 0` 需要单独作为 correctness test，因为它会触发零吸收的特殊数学分支。当前 CPU 测试已经覆盖该情况；后续应补充 CUDA 版本，并同时验证 CPU/CUDA 结果与解析结果一致。

`num_rays = 0` 或 `num_samples = 0` 属于 API 边界行为，不是正常 demo 输入。可以在接口测试中验证返回值和错误处理，不必为它们单独写 demo。

### 3. 运行 Nsight Systems

Nsight Systems 用来确认 H2D、kernel、D2H、同步和 CPU/GPU 空闲间隙：

```bash
nsys profile --stats=true -o results/forward_demo_nsys \
  ./build-cuda/src/forward_demo 4096 256
```

benchmark 的 profiling 版本应减少迭代次数：

```bash
nsys profile --stats=true -o results/benchmark_nsys \
  ./build-cuda/src/benchmark 2 5
```

重点查看：

- H2D 是否占端到端时间的主要部分；
- H2D、kernel 和 D2H 是否存在空隙；
- 每次迭代是否有不必要的同步；
- CUDA context 初始化是否已经排除在正式测量之外。

### 4. 运行 Nsight Compute

Nsight Compute 用来分析 kernel 本身。先采集少量指标：

```bash
ncu --section MemoryWorkloadAnalysis \
    --section Occupancy \
    -o results/forward_kernel_ncu \
    ./build-cuda/src/forward_demo 4096 256
```

重点查看 global memory 合并访问、memory throughput、occupancy、register 使用和 warp 效率。

只有 profiler 证明当前布局是瓶颈时，才进行 `[ray][sample]` 到 `[sample][ray]` 的布局实验。

## Profiling 工具分别做什么

### Compute Sanitizer

它是 CUDA 的运行时错误检查工具，重点发现越界访问、非法显存访问、未初始化内存使用，以及部分同步和竞态问题。它回答的是“程序是否安全地访问 GPU 资源”，不是“程序运行得多快”。

### Nsight Systems

它观察整个应用的时间线，包括 CPU、CUDA API、H2D、kernel、D2H、同步和多个 stream 之间的关系。它适合判断 H2D 是否占主要时间、CPU/GPU 是否互相等待、GPU 是否有空闲间隙，以及拷贝和 kernel 是否能够重叠。

它回答的是“整个应用的时间花在哪里”。

### Nsight Compute

它深入分析单个 CUDA kernel 的硬件执行效率，包括 global memory 合并访问、memory/compute throughput、occupancy、register 和 shared memory 使用、warp 效率及分支发散。

它回答的是“这个 kernel 内部为什么快或慢”。

推荐顺序是：

```text
Compute Sanitizer -> Nsight Systems -> Nsight Compute -> 针对性优化
```

## 后续优化顺序

### 第一项：pinned memory 和异步拷贝

当前 benchmark 每次测量都会执行同步 H2D、kernel 和 D2H。优先评估：

```text
cudaMallocHost
cudaMemcpyAsync
cudaStreamSynchronize
```

比较优化前后的 H2D、kernel、D2H、end-to-end、带宽和正确性误差。

### 第二项：复用 device buffer

当前每个 benchmark case 都会分配和释放 device buffer。可以保留一次分配，在多次 forward 调用间复用，避免把内存管理开销混入工作流。

### 第三项：减少重复 H2D

如果多个 forward 调用使用同一份 `sigma`、`source` 和初始强度，应只传输一次输入，再重复执行 kernel。需要同时报告：

- kernel-only；
- 首次 end-to-end；
- 输入已驻留 GPU 时的 end-to-end。

### 第四项：ray batching

对于超过显存容量的问题，将 rays 分批处理：

```text
for each ray_batch:
    H2D
    kernel
    D2H
```

记录 batch size、显存峰值、总时间和结果误差。

### 第五项：数据布局实验

当前布局是 `[ray][sample]`。可以实验性地改为 `[sample][ray]`，但必须同步修改 CPU、CUDA、输入生成和测试，并用 Nsight Compute 对比访存指标和实际性能。

## 暂不做的工作

当前不建议优先实现 MPI、多 GPU 强弱缩放、GPU 逆问题或复杂 NeRF 网络。Module 07 的主题应保持为：

```text
物理传输模型
+ CPU 数值参考实现
+ CUDA forward kernel
+ 正确性验证
+ GPU profiling
+ 内存传输优化
```

## 最低完成标准

项目完成前至少应具备：

1. CPU/CUDA 结果验证；
2. RMSE 和最大绝对误差；
3. H2D、kernel、D2H、end-to-end 时间；
4. 多问题规模 benchmark；
5. 一次 Compute Sanitizer 检查；
6. 一次 Nsight Systems 和 Nsight Compute 分析；
7. 一项有数据支撑的优化；
8. 优化前后结果、命令和环境信息；
9. README 中对瓶颈和优化效果的解释。
