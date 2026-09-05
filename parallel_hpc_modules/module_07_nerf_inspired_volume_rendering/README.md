# NeRF-Inspired Volume Rendering: CUDA-Accelerated 1D RTE Forward Solver

**一个诚实的HPC项目：正向求解已实现，逆问题规划中**

---

## 项目定位

这是一个**工程严谨、范围明确**的CUDA加速科学计算项目。当前版本实现：

✅ **当前已实现**：
- 一维辐射传输方程（RTE）正向求解（CPU + CUDA）
- CPU reference 与 CUDA kernel 的数值一致性验证
- CUDA Events 分阶段计时、多问题规模 benchmark 和传输瓶颈分析
- CUDA 内存安全检查的可复现实验入口

🔮 **下一阶段规划（当前尚未实现）**：
- 常数参数逆问题：从观测数据反推全局常数 σ, q
- 网格搜索与牛顿法优化
- 空间变化参数 σ(s), q(s) 逆问题
- 伴随变量法梯度计算
- Adam/SGD优化器
- 正则化框架

详见 [FUTURE_WORK.md](./FUTURE_WORK.md)

---

## 核心方程

一维线性吸收-发射传输方程：

```
dI(s)/ds = -σ(s)·I(s) + q(s)
```

- **正问题（当前实现）**：给定 (σ, q) → 求解 I(L)
- **逆问题（尚未实现）**：给定 I_obs → 反推 (σ, q)

**常系数解析解**（验证用）：
```
I(L) = I₀·exp(-σL) + (q/σ)·(1 - exp(-σL))
```

---

## 当前范围与逆问题规划

| 方面 | 常数参数（下一阶段） | 空间变化参数（后续展望） |
|------|------------------|--------------------------|
| **参数维度** | 2个（σ, q） | N个（N个采样点） |
| **解析导数** | ✅ 可推导闭式解 | ❌ 需数值反向传播 |
| **优化算法** | 网格搜索 + 牛顿法 | 梯度下降系列 |
| **计算复杂度** | O(1)参数更新 | O(N)梯度计算 |
| **工程量** | 单人2-3周 | 团队数月 |
| **科研价值** | ✅ 展示逆问题核心思想 | 完整科研项目 |

**当前边界**：本版本先完成正向模型、CUDA 并行化、正确性验证和性能实验。常数参数逆问题作为下一阶段的**最小可验证单元**（MVP），适合：
- 本科/硕士课程项目
- HPC编程能力展示
- 留学申请Portfolio（展示工程诚实性）
- 后续扩展的坚实基础

---

## 项目结构

```
module_07_nerf_inspired_volume_rendering/
├── include/                  # 公共接口
│   ├── types.hpp            # 公共数据结构
│   ├── forward_solver.hpp   # 正向求解器
│   └── inverse_solver.hpp   # 逆问题预留接口（当前未实现）
│
├── src/
│   ├── forward/             # 正向求解（CPU + CUDA）
│   ├── inverse/             # 逆问题规划（当前未实现）
│   │   ├── grid_search.*    # 网格搜索
│   │   └── newton_method.*  # 牛顿法
│   ├── utils/               # 工具函数
│   └── demos/               # 演示程序
│
├── tests/                   # 单元测试
├── docs/                    # 详细文档
├── scripts/                 # 实验脚本
└── results/                 # 实验结果
```

完整架构说明见 [PROJECT_ARCHITECTURE.md](./PROJECT_ARCHITECTURE.md)

---

## 快速开始

### 构建

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DBUILD_CUDA_FORWARD=ON \
    -DBUILD_DEMOS=ON
cmake --build build -j4
```

### 运行演示

**1. 正向求解验证**
```bash
./build/src/forward_demo

# 输出：
# - CPU vs CUDA 结果对比
# - 解析解误差（RMSE < 1e-5）
# - 性能benchmark（GFLOPS，加速比）
```

**1.1. pinned-memory H2D 对照实验**
```bash
./build/src/improved_demo 4096 256

# 使用 cudaMallocHost 分配 page-locked host buffers，
# 输出 H2D、kernel、D2H、端到端耗时和数值校验结果。
```

**2. 逆问题优化（规划中，当前不可运行）**
```bash
# inverse_demo 尚未实现

# 计划输出：参数恢复、损失收敛和噪声鲁棒性分析
```

**3. 性能测试**
```bash
./build/src/benchmark
```

---

## 实现路线图

### Phase 1: 正向求解（1周）

**目标**：验证CUDA实现正确性

- [x] CPU参考实现（欧拉法 + 精确指数积分）
- [ ] 解析解验证（常系数情况）
- [x] CUDA kernel实现（一线程一射线）
- [x] CPU-CUDA误差对齐（RMSE < 1e-5）
- [x] 性能benchmark（加速比分析）

**关键文件**：
- `src/forward/cpu_forward.cpp`
- `src/forward/cuda_forward.cu`
- `src/demos/forward_demo.cu`

### Phase 2: 逆问题Baseline（规划中）

**目标**：在正向求解稳定后，实现网格搜索优化。当前尚未开始编码。

- [ ] 合成数据生成器（ground truth + 噪声）
- [ ] 网格搜索算法（暴力枚举参数空间）
- [ ] 损失函数：L2 loss = ||I_computed - I_obs||²
- [ ] 参数恢复实验（可视化参数空间）

**关键文件**：
- `src/utils/data_generator.cpp`
- `src/inverse/grid_search.cpp`
- `src/demos/inverse_demo.cu`

### Phase 3: 牛顿法优化（规划中）

**目标**：利用解析导数加速收敛。当前尚未开始编码。

- [ ] 推导常数参数的解析导数（见 docs/）
- [ ] 实现牛顿法/准牛顿法
- [ ] 收敛速度对比（网格搜索 vs 牛顿法）
- [ ] 初值敏感性分析

**关键文件**：
- `docs/mathematical_derivation.md`
- `src/inverse/newton_method.cpp`

### Phase 4: 逆问题验证与实验（规划中）

- [ ] 噪声鲁棒性实验（不同噪声水平）
- [ ] 收敛性分析（迭代次数 vs 精度）
- [ ] 参数非负约束（投影法）
- [ ] 完整实验报告 + 可视化

---

## 技术文档

### 实现指南
- **[IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)**：实现范围、模块大纲、开发顺序与验收标准
  - 正向求解算法（CPU/CUDA）
  - 常数参数解析导数推导
  - 网格搜索 vs 牛顿法对比
  - 数值稳定性处理

### 理论展望
- **[FUTURE_WORK.md](./FUTURE_WORK.md)**：方案B完整理论框架
  - 空间变化参数逆问题
  - 伴随变量法数学推导
  - Adam优化器原理
  - 正则化与病态问题

### 架构说明
- **[PROJECT_ARCHITECTURE.md](./PROJECT_ARCHITECTURE.md)**：项目组织原则

---

## 实验结果与性能分析

本项目将正向求解包装成一个可复现的 CUDA/HPC 性能实验：CPU 实现作为数值参考，CUDA Events 分别测量 H2D、kernel、D2H 和端到端延迟，实验前执行 warmup 以排除 CUDA context 初始化影响。当前结果来自 NVIDIA A30 24 GB（驱动 580.65.06，`nvcc` 12.1）环境；完整 CSV 保存在 [results/forward/baseline.csv](./results/forward/baseline.csv)。

### 正确性

`forward_demo` 的 `4096 x 256` 独立运行记录见 [results/forward/results.txt](./results/forward/results.txt)：CPU 与 CUDA 输出的 `RMSE = 0`、最大绝对误差为 `0`，验证通过。benchmark 中的全部六组问题规模也得到 `max_abs_error = 0`。因此，下面的性能比较是在相同输入和一致数值结果的前提下进行的。

### 多规模 benchmark

`baseline.csv` 使用 5 次 warmup 和 20 次测量，时间单位为 ms，吞吐量单位为 Msamples/s。

| rays × samples | kernel | end-to-end | throughput | CPU/GPU speedup |
|---:|---:|---:|---:|---:|
| 1024 × 64 | 0.045 | 0.135 | 1,449.6 | 5.68× |
| 4096 × 64 | 0.046 | 0.327 | 5,688.9 | 9.26× |
| 16384 × 64 | 0.084 | 0.784 | 12,549.0 | 15.39× |
| 1024 × 256 | 0.156 | 0.434 | 1,675.9 | 6.93× |
| 4096 × 256 | 0.157 | 0.836 | 6,688.4 | 14.43× |
| 16384 × 256 | 0.310 | 2.776 | 13,522.6 | 17.28× |

结果体现出三个工程结论：

- 总采样数增大时 kernel 时间和端到端时间总体上升，说明工作量扩展符合预期；中等及大规模问题的 GPU 并行度更高，吞吐量明显优于小规模问题。
- `4096 × 256` 时 H2D 约 `0.673 ms`，占端到端时间约 81%，而 kernel 仅约 `0.157 ms`；`16384 × 256` 时 H2D 占比进一步接近 89%。当前端到端瓶颈主要是主机到设备的数据传输，而不是计算 kernel 本身。
- 因此，下一步优化优先级是 pinned memory、异步拷贝、device buffer 复用和减少重复 H2D；只有在时间线或硬件指标证明必要时，才进一步尝试数据布局或 kernel 内部优化。

### 工具边界与可复现性

性能和正确性使用不同工具，避免混淆工具职责：

```bash
# 性能计时：CUDA Events 已集成在 demo 和 benchmark 中
./build/src/forward_demo 4096 256
./build/src/benchmark 5 20 > results/forward/baseline.csv

# 内存安全/竞态检查：这是 correctness/debugging，不是性能计时
compute-sanitizer --tool memcheck ./build/src/forward_demo 1024 64
```

当前云 GPU 运行在受限 Docker 容器中，无法访问 NVIDIA 硬件 performance counters，因此没有将 `ncu`/`nsys` 的结果冒充为已完成的硬件 profiling。相关工具的使用边界和后续实验记录见 [docs/profiling_tools.md](./docs/profiling_tools.md)。

---

## 常见问题

### Q1: 为什么当前没有逆问题实现？

**A**: 当前版本有意将范围限定为正向求解，先建立可验证的 CPU reference、CUDA kernel 和性能基线。逆问题尚未实现；常数参数版本将作为下一阶段的最小可验证单元，计划包括：
- 正向模型构建
- 损失函数设计
- 优化算法选择
- 噪声鲁棒性

空间变化参数会进一步引入梯度计算和正则化等工程复杂度。见 [FUTURE_WORK.md](./FUTURE_WORK.md) 了解扩展路径。

### Q2: 这个项目适合什么场景？

**A**:
- ✅ 本科/硕士课程项目
- ✅ HPC/CUDA编程练习
- ✅ 科研入门（传输方程与GPU并行计算）
- ✅ 留学申请Portfolio（展示工程诚实性和性能分析）
- ❌ 直接发表论文（需扩展为空间变化参数）

### Q3: 后续如何加入逆问题？

**A**: 先完成常数参数逆问题，再扩展到空间变化参数。参考 [FUTURE_WORK.md](./FUTURE_WORK.md)，关键步骤：
1. 实现反向传播（伴随变量法）
2. 实现Adam优化器
3. 添加正则化（L2/TV）
4. CUDA梯度kernel优化

预计工作量：**3-6个月**（全职）

### Q4: 与NeRF的关系？

**A**:
- **相同**：都求解传输方程，都用体积渲染公式
- **不同**：NeRF用神经网络表示σ(s), q(s)；本项目当前只实现给定参数的正向数值求解
- **后续规划**：逆问题阶段再引入传统数值优化方法，而不是神经网络

---

## 参考文献

### 传输方程理论
1. Chandrasekhar, "Radiative Transfer", Dover Publications, 1960
2. Modest, "Radiative Heat Transfer", Academic Press, 2013

### NeRF相关
3. Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields", ECCV 2020

### 逆问题优化
4. Nocedal & Wright, "Numerical Optimization", Springer, 2006
5. Arridge, "Optical tomography in medical imaging", Inverse Problems, 1999

### CUDA编程
6. NVIDIA, "CUDA C++ Programming Guide"
7. Sanders & Kandrot, "CUDA by Example", 2010

---

## 许可证

与上级项目保持一致。

---

## 致谢

感谢项目指导与技术讨论。

---

**项目原则：工程诚实、边界清晰、科研视野完整。**
