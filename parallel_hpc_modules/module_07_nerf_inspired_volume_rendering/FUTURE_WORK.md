# Future Work: Spatially-Varying Parameter Inverse Problem

**本文档为方案B（理论展望），所有内容均未实现代码，仅作为科研方向参考。**

---

## 概述

方案A（本项目）实现了**常数参数逆问题**，这是逆问题的最小可验证单元。

方案B扩展到**空间变化参数 σ(s), q(s) 的逐点反演**，需要：
1. 反向传播（伴随变量法）计算梯度
2. 迭代优化器（Adam/SGD）
3. 正则化框架（L2/TV）
4. 大规模并行优化

**工作量估计**：3-6个月（全职科研项目）

---

## 1. 问题定义

### 1.1 从常数到空间变化

| 问题类型 | 参数形式 | 参数维度 | 优化算法 |
|---------|---------|---------|---------|
| **方案A（已实现）** | σ, q 为常数 | 2个参数 | 网格搜索 + 牛顿法 |
| **方案B（展望）** | σ(s), q(s) 逐点变化 | 2N个参数（N个采样点） | 梯度下降系列 |

### 1.2 数学形式

**给定**：
- 观测数据 `I_obs`（单条或多条射线）
- 初始强度 `I₀`

**求解**：
- 空间变化参数 `{σ_i, q_i}_{i=0}^{N-1}` 使得损失函数最小：

```
L(σ, q) = (1/2M)·Σ_rays ||I_computed - I_obs||² + R(σ, q)
```

其中 `R(σ, q)` 是正则化项（防止病态问题）。

---

## 2. 伴随变量法（Adjoint Method）

### 2.1 为什么需要伴随变量法？

**问题**：空间变化参数有N个采样点，朴素梯度计算需要：
- 对每个参数扰动一次正向求解 → O(N) 次正向传播
- 总复杂度：O(N²) 对于N个参数

**解决方案**：伴随变量法
- 一次正向传播 + 一次反向传播 → 得到所有参数的梯度
- 总复杂度：O(N)
- 这是深度学习反向传播的数学基础

### 2.2 数学推导

定义**伴随变量**（拉格朗日乘子）：
```
λ_i = ∂L/∂I_i
```

**反向递推关系**（从离散化的传输方程推导）：

对于欧拉前向法：
```
I_{i+1} = I_i + Δs·(-σ_i·I_i + q_i)
```

应用链式法则：
```
λ_i = λ_{i+1}·∂I_{i+1}/∂I_i = λ_{i+1}·(1 - σ_i·Δs)
```

边界条件：
```
λ_N = ∂L/∂I_N = I_N - I_obs
```

**参数梯度**：
```
∂L/∂σ_i = λ_{i+1}·∂I_{i+1}/∂σ_i = -λ_{i+1}·I_i·Δs
∂L/∂q_i = λ_{i+1}·∂I_{i+1}/∂q_i = λ_{i+1}·Δs
```

### 2.3 算法流程

```python
# 前向传播（保存所有中间状态）
I = [I0]
for i in range(N):
    I_new = I[-1] + Δs * (-σ[i] * I[-1] + q[i])
    I.append(I_new)

# 计算损失
loss = 0.5 * (I[N] - I_obs)**2

# 反向传播
λ = [0] * (N+1)
λ[N] = I[N] - I_obs

grad_σ = [0] * N
grad_q = [0] * N

for i in range(N-1, -1, -1):
    grad_σ[i] = -λ[i+1] * I[i] * Δs
    grad_q[i] = λ[i+1] * Δs
    λ[i] = λ[i+1] * (1 - σ[i] * Δs)

return grad_σ, grad_q
```

### 2.4 CUDA实现策略

**挑战**：
- 前向传播需要保存所有 `I_i`（显存消耗 O(num_rays × N)）
- 反向传播从后向前遍历（非coalesced访问）

**优化方案**：
1. **双buffer**：ping-pong缓冲区
2. **Checkpointing**：只保存关键点，需要时重新计算
3. **共享内存**：缓存常用的 `λ` 和 `I` 值
4. **Warp-level并行**：多线程协作处理单条射线

---

## 3. 梯度下降优化器

### 3.1 随机梯度下降（SGD）

**基础版本**：
```python
for iteration in range(max_iter):
    grad_σ, grad_q = compute_gradients(...)
    σ -= learning_rate * grad_σ
    q -= learning_rate * grad_q

    # 投影到可行域
    σ = np.maximum(σ, 0)
    q = np.maximum(q, 0)
```

**问题**：
- 收敛慢
- 需要精心调节学习率
- 容易陷入局部最优

### 3.2 Momentum SGD

**改进**：利用历史梯度加速收敛
```python
v_σ = 0
v_q = 0
β = 0.9

for iteration in range(max_iter):
    grad_σ, grad_q = compute_gradients(...)

    # 动量更新
    v_σ = β * v_σ + (1 - β) * grad_σ
    v_q = β * v_q + (1 - β) * grad_q

    σ -= learning_rate * v_σ
    q -= learning_rate * v_q
```

**优点**：
- 加速收敛（特别是在峡谷地形）
- 减少震荡

### 3.3 Adam优化器（推荐）

**自适应学习率**：每个参数独立调整步长

```python
# 初始化
m_σ, v_σ = np.zeros(N), np.zeros(N)
m_q, v_q = np.zeros(N), np.zeros(N)
β1, β2 = 0.9, 0.999
ε = 1e-8

for t in range(1, max_iter + 1):
    grad_σ, grad_q = compute_gradients(...)

    # 一阶矩估计（梯度均值）
    m_σ = β1 * m_σ + (1 - β1) * grad_σ
    m_q = β1 * m_q + (1 - β1) * grad_q

    # 二阶矩估计（梯度方差）
    v_σ = β2 * v_σ + (1 - β2) * grad_σ**2
    v_q = β2 * v_q + (1 - β2) * grad_q**2

    # 偏差修正
    m_σ_hat = m_σ / (1 - β1**t)
    m_q_hat = m_q / (1 - β1**t)
    v_σ_hat = v_σ / (1 - β2**t)
    v_q_hat = v_q / (1 - β2**t)

    # 参数更新
    σ -= learning_rate * m_σ_hat / (np.sqrt(v_σ_hat) + ε)
    q -= learning_rate * m_q_hat / (np.sqrt(v_q_hat) + ε)
```

**优点**：
- 自动调节学习率
- 鲁棒性强
- 是深度学习的标准选择

### 3.4 超参数调优

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `learning_rate` | 0.001 - 0.01 | 太大发散，太小收敛慢 |
| `β1` | 0.9 | 一阶矩衰减率 |
| `β2` | 0.999 | 二阶矩衰减率 |
| `ε` | 1e-8 | 数值稳定性 |
| `max_iter` | 1000 - 10000 | 根据收敛曲线调整 |

---

## 4. 正则化（Regularization）

### 4.1 为什么需要正则化？

**问题**：空间变化参数逆问题是**病态问题**（ill-posed）：
- 解不唯一
- 小的观测噪声导致大的参数扰动
- 过拟合噪声

**解决方案**：正则化约束参数的复杂度

### 4.2 L2正则化（Tikhonov正则化）

**动机**：偏好小参数值

```
R_L2(σ, q) = (λ/2)·(||σ||² + ||q||²)
```

**梯度贡献**：
```
∂R/∂σ_i = λ·σ_i
∂R/∂q_i = λ·q_i
```

**效果**：
- 防止参数爆炸
- 平滑参数分布

### 4.3 TV正则化（Total Variation）

**动机**：偏好空间平滑但允许跳变

```
R_TV(σ) = λ·Σ_i |σ_{i+1} - σ_i|
```

**梯度**（次梯度，因为不可微）：
```
∂R/∂σ_i = λ·(sign(σ_i - σ_{i-1}) - sign(σ_{i+1} - σ_i))
```

**效果**：
- 保持边缘清晰
- 去除高频噪声
- 适合分段常数参数

### 4.4 物理约束

**非负约束**：σ ≥ 0, q ≥ 0

**投影法**（Projected Gradient Descent）：
```python
σ -= learning_rate * grad_σ
σ = np.maximum(σ, 0)  # 投影到可行域
```

**Penalty法**（软约束）：
```python
penalty = np.sum(np.maximum(-σ, 0)**2)  # 惩罚负值
loss += penalty_weight * penalty
```

---

## 5. 实现挑战与解决方案

### 5.1 显存管理

**问题**：前向传播需保存所有中间状态
- N=1000 采样点，M=10000 射线 → 40MB（单精度）
- 可行，但需要优化

**解决方案**：
1. **Gradient Checkpointing**：只保存部分checkpoint，需要时重新计算
2. **混合精度**：使用FP16存储，FP32计算
3. **逐批处理**：分批处理射线，减少峰值显存

### 5.2 数值稳定性

**问题**：
- `σ → 0` 时除零
- `σ·Δs` 很大时指数溢出

**解决方案**：
- 泰勒展开处理小σ
- 限制σ的上界（如 σ_max = 10）
- 使用log-scale参数化：`σ = exp(θ)`

### 5.3 初始化策略

**问题**：糟糕的初始化导致收敛到局部最优

**策略**：
1. **常数初始化**：用方案A的结果
2. **随机扰动**：在常数基础上加小噪声
3. **分层初始化**：先粗网格，再插值到细网格
4. **预训练**：用少量数据先训练一个粗略模型

### 5.4 多射线联合优化

**单射线**：
```
L = (1/2)·(I_computed - I_obs)²
```

**多射线**（提高约束）：
```
L = (1/2M)·Σ_{m=1}^M (I_m - I_obs,m)²
```

**好处**：
- 增加观测数据，减少不适定性
- 不同射线的约束相互制约
- 更鲁棒，更准确

---

## 6. 实验设计

### 6.1 合成数据实验

**步骤**：
1. 设计ground truth参数（如正弦波、分段常数）
2. 正向求解得到清洁观测
3. 添加不同水平的噪声
4. 运行逆问题优化
5. 评估恢复精度

**指标**：
- **参数误差**：`RMSE(σ_recovered, σ_true)`
- **相对误差**：`||σ_recovered - σ_true|| / ||σ_true||`
- **前向误差**：`||I_recovered - I_obs||`

### 6.2 消融实验

**测试不同组件的贡献**：
1. SGD vs Momentum vs Adam
2. 无正则化 vs L2 vs TV
3. 单射线 vs 多射线
4. 不同噪声水平
5. 不同参数空间分辨率（N = 100, 500, 1000）

### 6.3 收敛性分析

**可视化**：
1. Loss曲线（vs iteration）
2. 参数误差曲线
3. 梯度范数（监控vanishing/exploding gradient）
4. 参数恢复动画（GIF）

---

## 7. 与NeRF的联系

### 7.1 NeRF是什么？

**NeRF（Neural Radiance Fields）**：
- 用**多层感知机（MLP）**表示 `σ(x, y, z)` 和 `c(x, y, z, θ, φ)`
- 通过多视角图像优化MLP参数
- 本质是**高维参数逆问题**

### 7.2 本项目 vs NeRF

| 方面 | 本项目（方案B） | NeRF |
|------|----------------|------|
| **参数表示** | 离散采样点 `{σ_i}` | 神经网络 MLP(x) → σ |
| **参数维度** | O(N)，N个采样点 | O(W)，W个神经网络权重 |
| **优化算法** | Adam + 反向传播 | Adam + 反向传播（相同！） |
| **正则化** | L2/TV | Weight decay |
| **输入** | 1D射线，单通道 | 3D场景，多视角RGB |

**核心思想相同**：都是可微渲染 + 梯度优化

### 7.3 从本项目到NeRF的路径

1. **本项目**：1D传输方程，离散参数
2. **扩展到3D**：3D体积，离散体素
3. **引入神经网络**：用MLP替代离散采样
4. **多视角约束**：多个相机的观测
5. **完整NeRF**：位置编码、视角相关颜色

**结论**：本项目是NeRF的**数学简化版本**，展示了核心概念。

---

## 8. 实现路线图（预估）

### Phase 1: 反向传播（1-2个月）

- [ ] 推导伴随变量法方程
- [ ] 实现CPU反向传播
- [ ] 数值梯度验证（关键！）
- [ ] 实现CUDA反向传播
- [ ] 显存优化（checkpointing）

### Phase 2: 优化器（1个月）

- [ ] 实现SGD
- [ ] 实现Momentum SGD
- [ ] 实现Adam（重点）
- [ ] 超参数调优实验

### Phase 3: 正则化（1个月）

- [ ] L2正则化
- [ ] TV正则化
- [ ] 参数约束（投影法）
- [ ] 消融实验

### Phase 4: 大规模实验（1-2个月）

- [ ] 合成数据生成器（多种ground truth）
- [ ] 多射线联合优化
- [ ] 不同噪声水平实验
- [ ] 收敛性分析
- [ ] 完整论文撰写

**总工作量**：3-6个月（全职科研）

---

## 9. 参考文献（方案B专属）

### 伴随变量法
1. Lions, J. L., "Optimal Control of Systems Governed by Partial Differential Equations", Springer, 1971
2. Plessix, R. E., "A review of the adjoint-state method for computing the gradient of a functional with geophysical applications", Geophysical Journal International, 2006

### 逆问题理论
3. Engl, H. W., Hanke, M., & Neubauer, A., "Regularization of Inverse Problems", Kluwer, 1996
4. Vogel, C. R., "Computational Methods for Inverse Problems", SIAM, 2002

### 深度学习优化
5. Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015
6. Ruder, S., "An overview of gradient descent optimization algorithms", arXiv:1609.04747, 2016

### NeRF系列
7. Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields", ECCV 2020
8. Barron et al., "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields", ICCV 2021
9. Müller et al., "Instant Neural Graphics Primitives", SIGGRAPH 2022

### 可微渲染
10. Kato et al., "Differentiable Rendering: A Survey", arXiv:2006.12057, 2020
11. Nimier-David et al., "Mitsuba 2: A Retargetable Forward and Inverse Renderer", SIGGRAPH Asia 2019

---

## 10. 结语

方案B（空间变化参数逆问题）是一个**完整的科研项目**，包含：
- 高级数学（变分法、最优控制）
- 高性能计算（CUDA优化、显存管理）
- 机器学习（优化器、正则化）
- 数值分析（病态问题、收敛性）

**适合作为**：
- 硕士/博士论文课题
- 发表在Scientific Computing或Computer Graphics会议
- 开源库（如PyTorch的inverse-transport扩展）

**不适合作为**：
- 本科课程项目（工作量过大）
- 快速原型验证（可先用PyTorch自动微分）

**从方案A到方案B的建议**：
1. 先完成方案A，确保理解逆问题核心概念
2. 学习PyTorch，用自动微分快速原型化
3. 阅读NeRF源代码，理解工程实践
4. 再用CUDA手动实现优化性能

---

**这就是完整的科研视野！**
