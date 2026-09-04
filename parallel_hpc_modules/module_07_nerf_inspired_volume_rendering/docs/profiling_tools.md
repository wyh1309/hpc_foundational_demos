# CUDA Profiling 工具说明

本项目使用手动计时和 NVIDIA CUDA 工具分析 forward solver。它们关注的问题不同，应该配合使用。

## 工具与问题对应关系

| 工具 | 主要回答的问题 |
| --- | --- |
| CUDA events | 各阶段花了多少时间？ |
| Compute Sanitizer | CUDA 内存和同步是否正确？ |
| Nsight Systems | 整个应用的时间花在哪里？ |
| Nsight Compute | 单个 kernel 为什么快或慢？ |

## 安装与检查

这三个工具通常不是分别安装的独立软件包，而是随 NVIDIA CUDA Toolkit 一起提供：

- Compute Sanitizer：`compute-sanitizer`；
- Nsight Systems：`nsys`；
- Nsight Compute：`ncu`。

因此，优先安装与项目编译所使用版本匹配的 CUDA Toolkit。安装完成后，还需要把 Toolkit 的 `bin` 目录加入 `PATH`。

### 检查是否已安装

可以逐个检查命令：

```bash
command -v compute-sanitizer
command -v nsys
command -v ncu
```

再检查版本：

```bash
compute-sanitizer --version
nsys --version
ncu --version
```

如果命令存在但无法运行 GPU 程序，再检查 NVIDIA 驱动：

```bash
nvidia-smi
```

这里要区分两种情况：

- 找不到 `compute-sanitizer`、`nsys` 或 `ncu`：通常是 Toolkit 未安装，或 `PATH` 未配置；
- 工具可以显示版本，但运行程序时报驱动错误：工具已安装，问题在 NVIDIA 驱动、GPU 节点或 CUDA/驱动版本兼容性。

也可以直接检查常见安装目录：

```bash
ls /usr/local/cuda/bin/{compute-sanitizer,nsys,ncu}
ls /usr/local/cuda-*/bin/{compute-sanitizer,nsys,ncu}
```

### Ubuntu 安装

推荐从 NVIDIA 官方 CUDA 下载页面选择与 Ubuntu 版本、GPU 驱动和项目需求匹配的 CUDA Toolkit 安装方式：

<https://developer.nvidia.com/cuda-downloads>

安装完成后，如果工具已经存在但命令找不到，可以临时设置：

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

如果使用具体版本目录：

```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
```

确认三个工具：

```bash
command -v compute-sanitizer
command -v nsys
command -v ncu
```

使用 NVIDIA 的 Debian 仓库安装时，先按照 CUDA 官方安装页面配置对应的仓库和 keyring，再安装 Toolkit。不要直接假设系统仓库中的 `nvidia-cuda-toolkit` 包包含所需版本的 Nsight 工具；不同 Ubuntu 版本中的包版本可能较旧或组件不完整。

### HPC 集群环境

集群通常由管理员安装 NVIDIA 驱动和多个 CUDA Toolkit 版本，用户不应自行安装驱动。先查看可用模块：

```bash
module avail cuda
```

加载项目需要的 CUDA 版本，例如：

```bash
module load cuda/12.4
```

然后检查：

```bash
command -v compute-sanitizer
command -v nsys
command -v ncu
nvidia-smi
```

如果模块系统没有提供这些工具，应联系集群管理员安装对应 CUDA Toolkit，或者加载包含 Nsight Systems 和 Nsight Compute 的完整 Toolkit 模块。仅安装 CUDA runtime 通常不足以提供这些分析工具。

### 工具安装成功的判断标准

至少应满足：

```bash
compute-sanitizer --version
nsys --version
ncu --version
nvidia-smi
```

前三条能够输出版本，说明工具已安装；最后一条能够识别 GPU，才说明当前节点具备实际运行和 profiling CUDA 程序的条件。

## CUDA events

当前 demo 和 benchmark 使用 CUDA events 分别测量：

- H2D；
- kernel；
- D2H；
- end-to-end。

它适合回答：

> 哪个阶段耗时最多？

例如，如果 H2D 为 `0.66 ms`、kernel 为 `0.16 ms`，可以初步判断端到端性能主要受数据传输影响。

CUDA events 能提供准确的 GPU 时间间隔，但只给出选定区间的数值，不能解释区间内部发生了哪些等待、同步或资源利用问题。

## Compute Sanitizer

Compute Sanitizer 是 CUDA 运行时错误检查工具，不是性能分析工具。它主要用于发现：

- 数组越界；
- 非法显存访问；
- 未初始化内存使用；
- 部分同步和竞态问题。

示例：

```bash
compute-sanitizer --tool memcheck \
  ./build-cuda/src/forward_demo 1024 64
```

它回答的是：

> CUDA 程序是否安全、正确地访问 GPU 资源？

性能优化前应先运行它，避免在存在非法访问时比较性能结果。

## Nsight Systems

Nsight Systems 用于查看整个应用的时间线，能够显示：

- CPU 调用；
- CUDA API；
- H2D、kernel 和 D2H；
- stream 与同步；
- CPU/GPU 空闲时间。

示例：

```bash
nsys profile --stats=true \
  -o results/forward_demo_nsys \
  ./build-cuda/src/forward_demo 4096 256
```

它适合判断：

- H2D 是否确实占据端到端时间的主要部分；
- CPU 是否在等待 GPU；
- GPU 是否存在空闲间隙；
- 拷贝和 kernel 是否能够重叠；
- 是否有不必要的同步或 kernel launch 开销。

它回答的是：

> 整个应用的时间花在哪里？

本项目中，Nsight Systems 用于验证“H2D 是主要瓶颈”这一假设。

## Nsight Compute

Nsight Compute 用于深入分析单个 CUDA kernel 的硬件执行效率。它可以观察：

- global memory 是否合并访问；
- memory throughput 和 compute throughput；
- occupancy；
- register 和 shared memory 使用；
- warp 执行效率；
- 分支发散；
- kernel launch 配置。

示例：

```bash
ncu --section MemoryWorkloadAnalysis \
    --section Occupancy \
    -o results/forward_kernel_ncu \
    ./build-cuda/src/forward_demo 4096 256
```

它回答的是：

> 单个 kernel 为什么快或慢？

例如，当前输入布局为 `[ray][sample]`。如果 Nsight Compute 显示 global memory 访问不理想，才有必要实验性地改为 `[sample][ray]`，并比较修改前后的硬件指标和实际运行时间。

## 推荐使用顺序

```text
手动计时
    -> Compute Sanitizer
    -> Nsight Systems
    -> Nsight Compute
    -> 针对性优化
    -> 重新验证正确性和性能
```

在本项目中，各阶段的含义是：

1. 手动计时发现 H2D 可能是主要瓶颈；
2. Compute Sanitizer 确认没有明显 CUDA 内存错误；
3. Nsight Systems 确认端到端时间线和同步关系；
4. Nsight Compute 分析 kernel 的访存和硬件利用率；
5. 根据证据选择 pinned memory、异步拷贝、buffer 复用或数据布局优化；
6. 优化后重新检查 RMSE、最大绝对误差和各阶段耗时。

## 手动计时与 profiling 的关系

手动计时并不会被 profiling 工具取代。它适合快速获得稳定的阶段耗时和 benchmark 表格；profiling 工具则用于解释这些数字背后的原因。

```text
手动计时：发现 H2D 可能是瓶颈
Nsight Systems：确认应用时间线和等待关系
Nsight Compute：分析 kernel 内部效率
Compute Sanitizer：保证 CUDA 访问正确
```
