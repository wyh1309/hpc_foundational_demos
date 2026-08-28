# 项目目录结构（方案A实现 + 方案B展望）

```
module_07_nerf_inspired_volume_rendering/
│
├── README.md                          # 主文档（严格区分A/B）
├── IMPLEMENTATION_GUIDE.md            # 技术实现大纲（A方案）
├── FUTURE_WORK.md                     # 理论展望（B方案纯理论）
├── CMakeLists.txt                     # 根CMake
│
├── include/                           # 公共接口
│   ├── types.hpp                     # 数据结构（仅常数参数）
│   ├── forward_solver.hpp            # 正向求解器接口
│   └── inverse_solver.hpp            # 逆问题接口（常数参数）
│
├── src/
│   ├── CMakeLists.txt
│   │
│   ├── forward/                      # 【A方案】正向求解
│   │   ├── cpu_forward.cpp
│   │   ├── cpu_forward.hpp
│   │   ├── cuda_forward.cu
│   │   └── cuda_forward.cuh
│   │
│   ├── inverse/                      # 【A方案】常数参数逆问题
│   │   ├── grid_search.cpp          # 网格搜索优化
│   │   ├── grid_search.hpp
│   │   ├── newton_method.cpp        # 牛顿法（解析导数）
│   │   └── newton_method.hpp
│   │
│   ├── utils/                        # 工具函数
│   │   ├── data_generator.cpp       # 合成数据生成
│   │   ├── data_generator.hpp
│   │   ├── validation.cpp           # 误差计算
│   │   ├── validation.hpp
│   │   ├── plotting.cpp             # 结果输出接口
│   │   ├── plotting.hpp
│   │   └── cuda_utils.cuh           # CUDA辅助
│   │
│   └── demos/                        # 演示程序
│       ├── forward_demo.cu          # 正向求解演示
│       ├── inverse_demo.cu          # 逆问题演示
│       └── benchmark.cu             # 性能测试
│
├── tests/                            # 单元测试
│   ├── test_forward.cpp             # 正向求解测试
│   ├── test_inverse.cpp             # 逆问题测试
│   └── test_convergence.cpp         # 收敛性测试
│
├── scripts/                          # 实验脚本
│   ├── run_forward_validation.sh
│   ├── run_inverse_experiments.sh
│   └── plot_results.py              # Python绘图
│
├── results/                          # 实验结果
│   ├── forward/
│   ├── inverse/
│   └── plots/
│
├── docs/                             # 详细文档
│   ├── mathematical_derivation.md   # 数学推导
│   ├── implementation_details.md    # 实现细节
│   └── experiment_protocol.md       # 实验规范
│
└── assets/                           # 资源文件
    └── equations/                    # LaTeX方程图片

```

## 方案A（本项目真实实现）vs 方案B（理论展望）对照

| 功能模块 | 方案A（已实现/将实现） | 方案B（Future Work） |
|---------|---------------------|---------------------|
| **正向求解** | ✅ 1D常/变系数RTE求解<br>✅ CPU高精度 + CUDA并行<br>✅ 解析解验证 | 🔮 2D/3D扩展<br>🔮 自适应步长 |
| **参数类型** | ✅ **仅全局常数σ、q** | 🔮 空间变化σ(s), q(s) |
| **逆问题算法** | ✅ 网格搜索<br>✅ 牛顿法（解析导数） | 🔮 梯度下降系列<br>🔮 Adam/SGD优化器<br>🔮 伴随变量法 |
| **梯度计算** | ✅ 常数参数解析导数 | 🔮 反向传播<br>🔮 CUDA梯度kernel |
| **正则化** | ✅ 参数非负投影 | 🔮 L2正则<br>🔮 空间平滑正则 |
| **验证体系** | ✅ 合成数据实验<br>✅ 噪声鲁棒性<br>✅ 初值敏感性 | 🔮 真实数据应用 |

## 关键边界说明

### ✅ 方案A实现范围（可交付、可验证、可展示）

**正向问题**：
- 输入：常数或空间变化的 σ(s), q(s)
- 输出：终点辐射强度 I(L)
- 实现：CPU参考 + CUDA加速

**逆问题（核心创新）**：
- **仅反演2个全局常数参数**：σ_const, q_const
- 观测数据：带噪声的 I_obs
- 优化目标：min ||I_computed(σ, q) - I_obs||²
- 算法：
  1. 网格搜索（暴力baseline）
  2. 牛顿法（利用解析导数）

**为什么只做常数参数？**
- 解析导数可推导（见 IMPLEMENTATION_GUIDE.md）
- 参数空间低维（2D），易可视化
- 收敛快，单机可完成
- 足以展示逆问题核心思想
- 工程量适合单人完成

### 🔮 方案B理论展望（纯文档、无代码）

所有涉及以下内容的，**只出现在 FUTURE_WORK.md**：
- 空间变化参数 σ(s), q(s) 的逐点反演
- 伴随变量法反向传播
- CUDA梯度kernel与显存优化
- Adam/SGD迭代优化器
- 正则化框架（L2、TV正则）
- 大规模参数优化

这些内容会在文档中详细介绍**数学原理、算法框架、潜在挑战**，作为科研视野展示，但**明确标注为未来工作**。

---

**总结**：
- 代码诚实：只实现常数参数逆问题
- 文档完整：覆盖空间变化参数的完整理论
- 边界清晰：README明确标注 A/B 分界
- 适配申请：展示工程严谨性 + 科研前瞻性
