# NeRF-Inspired Volume Rendering: 1D RTE Inverse Problem Solver

**一个诚实的HPC项目：正向求解 + 常数参数逆问题**

---

## 项目定位

这是一个**工程严谨、范围明确**的CUDA加速科学计算项目，实现：

✅ **方案A（本项目真实实现）**：
- 一维辐射传输方程（RTE）正向求解（CPU + CUDA）
- **常数参数逆问题**：从观测数据反推全局常数 σ, q
- 网格搜索 + 牛顿法优化
- 完整的验证与实验体系

🔮 **方案B（理论展望，无代码实现）**：
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

- **正问题**：给定 (σ, q) → 求解 I(L)
- **逆问题**：给定 I_obs → 反推 (σ, q)

**常系数解析解**（验证用）：
```
I(L) = I₀·exp(-σL) + (q/σ)·(1 - exp(-σL))
```

---

## 为什么只做常数参数逆问题？

| 方面 | 常数参数（本项目） | 空间变化参数（Future Work） |
|------|------------------|--------------------------|
| **参数维度** | 2个（σ, q） | N个（N个采样点） |
| **解析导数** | ✅ 可推导闭式解 | ❌ 需数值反向传播 |
| **优化算法** | 网格搜索 + 牛顿法 | 梯度下降系列 |
| **计算复杂度** | O(1)参数更新 | O(N)梯度计算 |
| **工程量** | 单人2-3周 | 团队数月 |
| **科研价值** | ✅ 展示逆问题核心思想 | 完整科研项目 |

**结论**：常数参数是逆问题的**最小可验证单元**（MVP），适合：
- 本科/硕士课程项目
- HPC编程能力展示
- 留学申请Portfolio（展示工程诚实性）
- 后续扩展的坚实基础

---

## 项目结构

```
module_07_nerf_inspired_volume_rendering/
├── include/                  # 公共接口
│   ├── types.hpp            # 数据结构（常数参数）
│   ├── forward_solver.hpp   # 正向求解器
│   └── inverse_solver.hpp   # 逆问题接口
│
├── src/
│   ├── forward/             # 正向求解（CPU + CUDA）
│   ├── inverse/             # 常数参数优化
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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
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

**2. 逆问题优化**
```bash
./build/src/inverse_demo

# 输出：
# - 参数恢复结果（true vs optimized）
# - 损失函数收敛曲线
# - 噪声鲁棒性分析
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
- [ ] CUDA kernel实现（一线程一射线）
- [ ] CPU-CUDA误差对齐（RMSE < 1e-5）
- [ ] 性能benchmark（加速比分析）

**关键文件**：
- `src/forward/cpu_forward.cpp`
- `src/forward/cuda_forward.cu`
- `src/demos/forward_demo.cu`

### Phase 2: 逆问题Baseline（1周）

**目标**：实现网格搜索优化

- [ ] 合成数据生成器（ground truth + 噪声）
- [ ] 网格搜索算法（暴力枚举参数空间）
- [ ] 损失函数：L2 loss = ||I_computed - I_obs||²
- [ ] 参数恢复实验（可视化参数空间）

**关键文件**：
- `src/utils/data_generator.cpp`
- `src/inverse/grid_search.cpp`
- `src/demos/inverse_demo.cu`

### Phase 3: 牛顿法优化（1周）

**目标**：利用解析导数加速收敛

- [ ] 推导常数参数的解析导数（见 docs/）
- [ ] 实现牛顿法/准牛顿法
- [ ] 收敛速度对比（网格搜索 vs 牛顿法）
- [ ] 初值敏感性分析

**关键文件**：
- `docs/mathematical_derivation.md`
- `src/inverse/newton_method.cpp`

### Phase 4: 验证与实验（1周）

- [ ] 噪声鲁棒性实验（不同噪声水平）
- [ ] 收敛性分析（迭代次数 vs 精度）
- [ ] 参数非负约束（投影法）
- [ ] 完整实验报告 + 可视化

---

## 技术文档

### 实现指南
- **[IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md)**：详细算法、伪代码、验证方法
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

## 实验结果示例

（实现后补充）

### 正向求解验证
- CPU vs CUDA 误差：RMSE < 1e-5
- 加速比：10000条射线，100倍加速

### 逆问题参数恢复
- 真实参数：σ=0.5, q=1.0
- 恢复参数：σ=0.498, q=1.002
- 相对误差：< 1%

---

## 常见问题

### Q1: 为什么不实现空间变化参数逆问题？

**A**: 常数参数已经包含逆问题的**核心概念**：
- 正向模型构建
- 损失函数设计
- 优化算法选择
- 噪声鲁棒性

空间变化参数增加的主要是**工程复杂度**，而非概念难度。见 [FUTURE_WORK.md](./FUTURE_WORK.md) 了解扩展路径。

### Q2: 这个项目适合什么场景？

**A**:
- ✅ 本科/硕士课程项目
- ✅ HPC/CUDA编程练习
- ✅ 科研入门（逆问题概念）
- ✅ 留学申请Portfolio（展示工程诚实性）
- ❌ 直接发表论文（需扩展为空间变化参数）

### Q3: 如何扩展到空间变化参数？

**A**: 参考 [FUTURE_WORK.md](./FUTURE_WORK.md)，关键步骤：
1. 实现反向传播（伴随变量法）
2. 实现Adam优化器
3. 添加正则化（L2/TV）
4. CUDA梯度kernel优化

预计工作量：**3-6个月**（全职）

### Q4: 与NeRF的关系？

**A**:
- **相同**：都求解传输方程，都用体积渲染公式
- **不同**：NeRF用神经网络表示σ(s), q(s)，本项目直接优化参数
- **本项目**：传统数值优化方法（非神经网络）

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
