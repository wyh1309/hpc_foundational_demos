# 下一步工作建议

## 1. `ctest` 是什么？

`ctest` 是 CMake 提供的测试运行工具。它负责发现并执行 CMake 配置中通过 `add_test()` 注册的测试程序，并汇总测试结果。

它本身不是 GoogleTest、Catch2 等测试框架。当前项目的测试程序是普通 C++ 可执行文件，测试逻辑由程序自行返回成功或失败；CMake 再通过 `add_test()` 把它们交给 `ctest` 管理。

当前项目的测试配置位于 `tests/CMakeLists.txt`，测试源码位于 `tests/`。已经实现的 `test_forward` 会检查：

- 多条射线的 CPU forward 结果；
- 常系数数值结果与解析解的一致性；
- `sigma = 0` 时的特殊处理；
- 变系数输入随采样数增加的收敛性。

运行方式：

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

`--output-on-failure` 表示只有测试失败时才显示更详细的输出，适合脚本和持续集成环境。

当前测试只覆盖 CPU 路径。CUDA forward 还需要增加 GPU 正确性测试，包括 CPU/CUDA 结果误差、边界尺寸、空输入和不同数据规模。

## 2. 当前项目状态判断

项目已经具备 CPU forward、CUDA kernel、常数参数逆问题求解器和基础 CMake 测试结构，但还没有形成完整的性能实验闭环。

目前的主要缺口是：

- `BUILD_CUDA_FORWARD` 默认关闭；
- `src/demos/benchmark.cu`、`forward_demo.cu` 和 `inverse_demo.cu` 目前为空，因此 demo 和 benchmark 不会生成；
- CUDA 接口只接收 device pointer，没有统一的 device buffer 生命周期管理；
- 没有记录 kernel、H2D、D2H 和端到端时间；
- 没有显存峰值、吞吐率和可复现实验配置；
- 还没有多 GPU、MPI 或强弱缩放实现。

因此，README 中的 CUDA 加速比和 benchmark 结果暂时不能视为已验证结论。

## 3. 推荐实施顺序

### 阶段一：完成 CUDA forward 验证

先实现 `forward_demo` 和 `benchmark`：

1. 生成可控规模的 `num_rays x num_samples` 输入；
2. 分配 device memory，并完成 H2D、kernel、D2H 流程；
3. 使用 CUDA events 分别记录数据传输、kernel 和端到端耗时；
4. 与 CPU 结果比较 RMSE 和最大绝对误差；
5. 增加 warmup，避免首次 CUDA 初始化影响测量；
6. 输出 ray 数、sample 数、耗时、吞吐率、显存使用和加速比。

### 阶段二：单 GPU 尺度实验

单 GPU 上应先做问题规模实验：

- 固定 `num_samples`，增加 ray 数；
- 固定 ray 数，增加 sample 数；
- 比较 kernel-only 和 end-to-end 两种时间；
- 判断瓶颈是算力、显存带宽、数据传输还是 kernel launch。

这类实验通常称为 throughput 或 problem-size scaling。只有在 GPU 数量变化时，才是严格意义上的强缩放或弱缩放。

### 阶段三：多 GPU 强弱缩放

强缩放：固定总问题规模，增加 GPU 数量。

```text
total_rays 和 num_samples 固定
GPU 数量：1, 2, 4, ...
```

弱缩放：固定每张 GPU 的工作量，增加 GPU 数量。

```text
rays_per_gpu 和 num_samples 固定
总 rays 随 GPU 数量增加
```

当前 forward kernel 可以作为每张 GPU 的本地计算核心。多 GPU 层需要补充设备选择、ray 分块、进程间数据分发和结果汇总。若使用 MPI，还应记录通信时间和计算时间，避免只报告 kernel 时间。

## 4. CUDA 显存优化方案

### 4.1 优先优化数据布局

当前输入布局是：

```text
[ray][sample]
```

一个 warp 中的线程处理不同 ray 时，同一个 sample 的访问间隔约为 `num_samples`，不利于合并访存。建议评估改为：

```text
[sample][ray]
```

这样同一个 sample 下相邻线程访问相邻地址，通常更有利于显存带宽利用。布局改变后，CPU、CUDA、数据生成和测试必须统一约定。

### 4.2 对 ray 做分块处理

不要要求完整输入一次性驻留 GPU。可以将 rays 分成多个 batch：

```text
for each ray_batch:
    H2D(sigma, source, initial_intensity)
    launch kernel
    D2H(output)
```

这样显存峰值由完整数据规模降低到单个 batch 规模，允许处理远大于显存容量的数据集。

### 4.3 复用和异步管理内存

建议建立一个 forward runner，负责：

- 初始化和复用 `sigma`、`source`、`initial_intensity`、`output` buffer；
- 使用 pinned host memory；
- 使用 `cudaMemcpyAsync`；
- 使用 2 到 4 个 CUDA stream 重叠拷贝和计算；
- 评估 `cudaMallocAsync` 与 memory pool，减少频繁分配释放。

### 4.4 利用常数参数问题的特点

当前逆问题是全局常数 `sigma` 和 `q`。如果 forward 输入确实是常数，不必为每条 ray、每个 sample 保存完整的 `sigma` 和 `q` 数组，只需传递标量参数，显存可以从 `O(num_rays * num_samples)` 降到 `O(num_rays)`，同时减少内存读取。

如果未来使用空间变化参数，则继续使用分块输入，并考虑 checkpointing；当前 forward 只输出终点强度，不需要保存每个 sample 的中间状态。

### 4.5 正确性和边界检查

优化前后都应保留：

- CPU/CUDA RMSE 检查；
- `sigma` 接近零的测试；
- `num_rays = 0` 和 `num_samples = 0` 的测试；
- 大尺寸下索引溢出检查；
- CUDA kernel 错误和同步错误检查。

建议将 `ray * num_samples` 这类索引改为 `size_t` 计算，避免大问题规模下的整数溢出。

## 5. 推荐的 benchmark 配置

### Strong scaling

固定总工作量，扫描 GPU 数量：

```text
num_rays = 固定
num_samples = 固定
gpu_count = 1, 2, 4, ...
```

报告总时间、加速比、并行效率、通信时间和计算时间。

### Weak scaling

固定每张 GPU 的工作量：

```text
rays_per_gpu = 固定
num_samples = 固定
gpu_count = 1, 2, 4, ...
total_rays = rays_per_gpu * gpu_count
```

报告每张 GPU 的计算时间、总时间、通信开销和并行效率。

### Memory scaling

逐渐增加输入规模，记录：

- device allocated bytes；
- free memory before and after allocation；
- kernel 时间；
- H2D/D2H 时间；
- 端到端时间；
- batch size 和是否发生分块。

## 6. 最优先的三个提交

1. 实现可运行的 `benchmark.cu` 和 `forward_demo.cu`；
2. 增加 CUDA correctness test 和显存/耗时统计；
3. 实现 ray batching，再评估 `[sample][ray]` 数据布局优化。

多 GPU 强弱缩放应放在这三项之后。否则即使得到 scaling 曲线，也难以区分计算、数据传输、内存布局和初始化开销的影响。
