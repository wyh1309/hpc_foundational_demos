# 实现指南：常数参数逆问题求解器

**本指南仅聚焦方案A（真实实现范围）**

---

## 目录

1. [数学基础](#1-数学基础)
2. [正向求解实现](#2-正向求解实现)
3. [常数参数逆问题](#3-常数参数逆问题)
4. [实现检查清单](#4-实现检查清单)
5. [常见问题](#5-常见问题)

---

## 1. 数学基础

### 1.1 传输方程

一维线性吸收-发射传输方程：

```
dI(s)/ds = -σ(s)·I(s) + q(s)
```

**物理意义**：
- `I(s)`: 辐射强度（radiance）
- `σ(s)`: 吸收系数（absorption coefficient）
- `q(s)`: 发射源项（emission source）
- 第一项：介质吸收导致强度衰减
- 第二项：介质自发光增加强度

### 1.2 常系数解析解

当 `σ, q` 为常数时，积分因子法得到：

```
I(L) = I₀·exp(-σL) + (q/σ)·(1 - exp(-σL))
```

**特殊情况**（σ → 0）：
```
I(L) = I₀ + q·L
```

**用途**：验证数值求解器的正确性。

### 1.3 NeRF风格离散化

对于变系数情况，使用欧拉前向法：

```
I_{i+1} = I_i + Δs·(-σ_i·I_i + q_i)
       = I_i·(1 - σ_i·Δs) + q_i·Δs
```

**数值稳定性**：当 `σ·Δs` 较大时，使用精确指数积分：

```
I_{i+1} = I_i·exp(-σ_i·Δs) + (q_i/σ_i)·(1 - exp(-σ_i·Δs))
```

---

## 2. 正向求解实现

### 2.1 CPU参考实现

**算法伪代码**：

```cpp
float solve_transport_cpu(
    const float* sigma,     // [N] 吸收系数
    const float* q,         // [N] 发射源
    int N,                  // 采样点数
    float ds,               // 步长
    float I0)               // 初始强度
{
    float I = I0;
    for (int i = 0; i < N; i++) {
        // 精确指数积分（数值稳定）
        float sigma_ds = sigma[i] * ds;
        if (sigma_ds < 1e-5f) {
            // 泰勒展开避免除零
            I = I + ds * (-sigma[i] * I + q[i]);
        } else {
            float exp_term = expf(-sigma_ds);
            I = I * exp_term + (q[i] / sigma[i]) * (1.0f - exp_term);
        }
    }
    return I;
}
```

**关键点**：
- 使用`expf`（单精度）而非`exp`
- 小σ情况的泰勒展开
- 循环内避免分支（性能优化）

### 2.2 CUDA并行实现

**并行策略**：一个线程处理一条射线

```cuda
__global__ void volume_transport_kernel(
    const float* sigma,          // [num_rays * N]
    const float* q,              // [num_rays * N]
    float* output,               // [num_rays]
    int num_rays,
    int N,
    float ds,
    float I0)
{
    int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= num_rays) return;

    float I = I0;
    int offset = ray_id * N;

    for (int i = 0; i < N; i++) {
        float sigma_val = sigma[offset + i];
        float q_val = q[offset + i];
        float sigma_ds = sigma_val * ds;

        // 精确积分
        if (sigma_ds < 1e-5f) {
            I += ds * (-sigma_val * I + q_val);
        } else {
            float exp_term = expf(-sigma_ds);
            I = I * exp_term + (q_val / sigma_val) * (1.0f - exp_term);
        }
    }

    output[ray_id] = I;
}
```

**启动配置**：
```cpp
int threads = 256;
int blocks = (num_rays + threads - 1) / threads;
volume_transport_kernel<<<blocks, threads>>>(sigma, q, output, num_rays, N, ds, I0);
```

### 2.3 验证方法

**步骤1**：常系数解析解验证
```cpp
// 设置常数参数
float sigma_const = 0.5f;
float q_const = 1.0f;
float L = 10.0f;
int N = 1000;
float ds = L / N;

// 数值解
float I_numerical = solve_transport_cpu(sigma_arr, q_arr, N, ds, I0);

// 解析解
float I_analytical = I0 * expf(-sigma_const * L)
                    + (q_const / sigma_const) * (1.0f - expf(-sigma_const * L));

// 相对误差
float rel_error = fabsf(I_numerical - I_analytical) / fabsf(I_analytical);
assert(rel_error < 1e-4f);  // 单精度阈值
```

**步骤2**：CPU vs CUDA一致性
```cpp
// RMSE计算
float rmse = 0.0f;
for (int i = 0; i < num_rays; i++) {
    float diff = cpu_output[i] - gpu_output[i];
    rmse += diff * diff;
}
rmse = sqrtf(rmse / num_rays);
assert(rmse < 1e-5f);
```

---

## 3. 常数参数逆问题

### 3.1 问题定义

**给定**：
- 观测数据 `I_obs`（可能含噪声）
- 初始强度 `I₀`
- 射线长度 `L`

**求解**：
- 常数参数 `σ, q` 使得 `I_computed(σ, q)` 最接近 `I_obs`

**损失函数**：
```
L(σ, q) = (1/2)·(I_computed(σ, q) - I_obs)²
```

**约束**：
- `σ ≥ 0`（吸收系数非负）
- `q ≥ 0`（发射源非负）

### 3.2 合成数据生成

```cpp
struct SyntheticData {
    float sigma_true;
    float q_true;
    float I_obs;
    float noise_level;
};

SyntheticData generate_data(float sigma_gt, float q_gt, float noise_std) {
    // 正向求解得到清洁观测
    float I_clean = solve_transport_analytical(sigma_gt, q_gt, L, I0);

    // 添加高斯噪声
    std::normal_distribution<float> dist(0.0f, noise_std);
    float noise = dist(rng);
    float I_obs = I_clean + noise;

    return {sigma_gt, q_gt, I_obs, noise_std};
}
```

### 3.3 网格搜索优化

**算法**：暴力枚举参数空间，找到最小损失

```cpp
struct GridSearchResult {
    float sigma_opt;
    float q_opt;
    float loss_min;
};

GridSearchResult grid_search(
    float I_obs,
    float sigma_min, float sigma_max, int sigma_steps,
    float q_min, float q_max, int q_steps)
{
    float loss_min = INFINITY;
    float sigma_opt = 0.0f, q_opt = 0.0f;

    for (int i = 0; i < sigma_steps; i++) {
        float sigma = sigma_min + i * (sigma_max - sigma_min) / sigma_steps;

        for (int j = 0; j < q_steps; j++) {
            float q = q_min + j * (q_max - q_min) / q_steps;

            // 正向求解
            float I_pred = solve_transport_analytical(sigma, q, L, I0);

            // 损失计算
            float loss = 0.5f * (I_pred - I_obs) * (I_pred - I_obs);

            if (loss < loss_min) {
                loss_min = loss;
                sigma_opt = sigma;
                q_opt = q;
            }
        }
    }

    return {sigma_opt, q_opt, loss_min};
}
```

**复杂度**：O(N_σ × N_q × N_samples)

**优点**：
- 实现简单
- 全局搜索，不会陷入局部最优
- 可可视化损失地形

**缺点**：
- 计算量大（参数空间指数增长）
- 精度受网格分辨率限制

### 3.4 牛顿法优化

**解析导数推导**：

对于常系数解析解：
```
I(σ, q) = I₀·exp(-σL) + (q/σ)·(1 - exp(-σL))
```

损失函数：
```
L(σ, q) = (1/2)·(I(σ, q) - I_obs)²
```

**一阶导数**（链式法则）：
```
∂L/∂σ = (I - I_obs) · ∂I/∂σ
∂L/∂q = (I - I_obs) · ∂I/∂q
```

其中：
```
∂I/∂σ = -I₀·L·exp(-σL) - (q/σ²)·(1 - exp(-σL)) + (q/σ)·L·exp(-σL)
∂I/∂q = (1/σ)·(1 - exp(-σL))
```

**二阶导数**（海森矩阵）：
```
∂²L/∂σ² = (∂I/∂σ)² + (I - I_obs)·∂²I/∂σ²
∂²L/∂q² = (∂I/∂q)²
∂²L/∂σ∂q = ∂I/∂σ · ∂I/∂q + (I - I_obs)·∂²I/∂σ∂q
```

**牛顿法更新**：
```
[σ_{k+1}]   [σ_k]       [∂²L/∂σ²    ∂²L/∂σ∂q]^{-1}  [∂L/∂σ]
[q_{k+1}] = [q_k] - H^{-1}·g = [              ] · [ ]
                                 [∂²L/∂σ∂q  ∂²L/∂q²]     [∂L/∂q]
```

**算法伪代码**：

```cpp
struct NewtonResult {
    float sigma;
    float q;
    int iterations;
    bool converged;
};

NewtonResult newton_method(
    float I_obs,
    float sigma_init, float q_init,
    int max_iter = 100,
    float tol = 1e-6f)
{
    float sigma = sigma_init;
    float q = q_init;

    for (int iter = 0; iter < max_iter; iter++) {
        // 正向求解
        float I = solve_transport_analytical(sigma, q, L, I0);
        float residual = I - I_obs;

        // 一阶导数
        float dI_dsigma = compute_dI_dsigma(sigma, q, L, I0);
        float dI_dq = compute_dI_dq(sigma, q, L, I0);
        float grad_sigma = residual * dI_dsigma;
        float grad_q = residual * dI_dq;

        // 收敛检查
        float grad_norm = sqrtf(grad_sigma*grad_sigma + grad_q*grad_q);
        if (grad_norm < tol) {
            return {sigma, q, iter, true};
        }

        // 二阶导数（海森矩阵）
        float H_ss = dI_dsigma * dI_dsigma;  // 高斯-牛顿近似
        float H_qq = dI_dq * dI_dq;
        float H_sq = dI_dsigma * dI_dq;

        // 求解线性系统 H·delta = -grad
        float det = H_ss * H_qq - H_sq * H_sq;
        float delta_sigma = (-grad_sigma * H_qq + grad_q * H_sq) / det;
        float delta_q = (grad_sigma * H_sq - grad_q * H_ss) / det;

        // 更新参数（带步长控制）
        float alpha = 1.0f;  // 可用line search优化
        sigma += alpha * delta_sigma;
        q += alpha * delta_q;

        // 投影到可行域
        sigma = fmaxf(sigma, 0.0f);
        q = fmaxf(q, 0.0f);
    }

    return {sigma, q, max_iter, false};  // 未收敛
}
```

**优点**：
- 收敛快（二次收敛）
- 精度高
- 适合实时应用

**缺点**：
- 需要初值（可用网格搜索粗搜）
- 可能陷入局部最优
- 海森矩阵计算复杂

---

## 4. 实现检查清单

### Phase 1: 正向求解 ✓

- [ ] CPU实现：欧拉法 + 精确指数积分
- [ ] 常系数解析解验证（误差 < 1e-4）
- [ ] 变系数数值求解
- [ ] CUDA kernel实现
- [ ] CPU vs CUDA误差对齐（RMSE < 1e-5）
- [ ] 性能benchmark（加速比 > 10x）
- [ ] 步长收敛性测试

### Phase 2: 网格搜索

- [ ] 合成数据生成器
- [ ] 网格搜索实现
- [ ] 损失地形可视化（2D热图）
- [ ] 参数恢复实验（无噪声）
- [ ] 噪声鲁棒性测试（不同SNR）

### Phase 3: 牛顿法

- [ ] 解析导数推导（纸笔验证）
- [ ] 一阶导数实现
- [ ] 数值导数验证（有限差分对比）
- [ ] 二阶导数（海森矩阵）
- [ ] 牛顿法迭代
- [ ] 收敛速度对比（vs 网格搜索）
- [ ] 初值敏感性分析

### Phase 4: 验证与实验

- [ ] 多组ground truth测试
- [ ] 噪声水平扫描（0%, 1%, 5%, 10%）
- [ ] 初值扫描（不同起点）
- [ ] 参数非负约束验证
- [ ] 完整实验报告
- [ ] 结果可视化脚本

---

## 5. 常见问题

### Q1: 如何验证解析导数正确性？

**A**: 使用有限差分法：

```cpp
float numerical_gradient_sigma(float sigma, float q, float epsilon = 1e-5f) {
    float I_plus = solve_transport_analytical(sigma + epsilon, q, L, I0);
    float I_minus = solve_transport_analytical(sigma - epsilon, q, L, I0);
    return (I_plus - I_minus) / (2.0f * epsilon);
}

// 验证
float analytical_grad = compute_dI_dsigma(sigma, q, L, I0);
float numerical_grad = numerical_gradient_sigma(sigma, q);
float relative_error = fabs(analytical_grad - numerical_grad) / fabs(analytical_grad);
assert(relative_error < 1e-3f);  // 数值导数精度较低
```

### Q2: 牛顿法不收敛怎么办？

**A**:
1. 检查初值是否合理（用网格搜索找粗解）
2. 添加步长控制（backtracking line search）
3. 使用阻尼牛顿法：`param += damping * delta`
4. 检查海森矩阵是否正定（可用BFGS近似）

### Q3: 如何选择网格搜索范围？

**A**:
- 物理先验：σ ∈ [0, 10]，q ∈ [0, 10]
- 如果完全未知，先粗网格探索，再细化
- 可视化损失地形找合理范围

### Q4: 常数参数能恢复到什么精度？

**A**:
- 无噪声：相对误差 < 0.1%
- 1% 噪声：相对误差 ~ 1-2%
- 10% 噪声：相对误差 ~ 10-20%
- 精度主要受噪声限制，不是算法限制

---

## 附录：核心公式速查

**正向求解（常系数）**：
```
I(L) = I₀·exp(-σL) + (q/σ)·(1 - exp(-σL))
```

**损失函数**：
```
L(σ, q) = (1/2)·(I(σ, q) - I_obs)²
```

**解析梯度**：
```
∂I/∂σ = -I₀·L·exp(-σL) - (q/σ²)·(1 - exp(-σL)) + (q/σ)·L·exp(-σL)
∂I/∂q = (1/σ)·(1 - exp(-σL))
```

**牛顿更新**：
```
θ_{k+1} = θ_k - H^{-1}·∇L
```

---

**下一步**：开始实现 `src/forward/cpu_forward.cpp`！
