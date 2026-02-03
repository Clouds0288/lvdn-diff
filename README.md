# README: Physics-Guided Graph Diffusion for LVDN State Estimation

## 1. 项目背景与动机 (Motivation)

**课题名称：** 基于物理引导图扩散模型的不确定性低压配电网状态估计
**核心问题：**
传统的低压配电网（LVDN）状态估计面临三大挑战：

1. **非高斯分布：** 高比例光伏与EV接入导致电压分布呈现双峰、重尾特性，传统高斯过程（GP）或回归模型失效。
2. **量测极度稀缺：** 低可观测性导致传统 WLS 不收敛。
3. **拓扑未知变化：** 缺乏实时开关状态信息，导致物理模型失配。

**解决方案：**
提出一种 **GNN-Diffusion-Guidance** 框架。利用 GNN 捕获拓扑特征，Diffusion 学习复杂状态流形，Cross-Attention 注入稀疏量测，Inference-time Guidance 实现物理约束与拓扑自适应。

---

## 2. 方法论架构 (Methodology Overview)

* **Backbone (骨干网络):**
* **Denoising Network:** 基于 **GNN** (如 GraphSAGE/GAT) 的去噪网络。
* **Conditioning:** 使用 **Topology-Guided Cross-Attention** 机制，让状态  动态关注稀疏量测 。


* **Inference Strategy (推理策略):**
* **Prior:** 使用训练好的 Diffusion Model 作为先验分布 。
* **Guidance:** 使用 **Manifold Constrained Gradient (MCG)**，在采样步中注入物理梯度 。
* **Adaptation:** 引入 **Hypothesis Testing (假设检验)** 机制，在推理时动态选择最优拓扑 。



---

## 3. 对比基准方法 (Baselines)

为了体现学术严谨性，对比方法分为四大类，覆盖了从传统到前沿的完整光谱。

### A. 物理驱动基准 (Physics-Based / Model-Based)

*用于确立物理模型的精度上限和鲁棒性下限。*

1. **WLS (Weighted Least Squares):** 行业金标准。使用牛顿-拉夫逊法求解。
* *作用：* 衡量在理想参数下的精度上限。


2. **WLS-LAV (Least Absolute Value):** 鲁棒状态估计（基于 L1 范数）。
* *作用：* **补充对比点。** 证明在坏数据（Bad Data）下，你的 Diffusion 比传统的抗差估计更强。


3. **LinDistFlow:** 线性化状态估计。
* *作用：* 作为推理速度的基准，以及验证你的“线性化引导”策略的有效性。



### B. 确定性深度学习基准 (Deterministic DL)

*用于证明“生成式”和“图结构”的必要性。*

1. **GAT (Graph Attention Network):** 纯监督学习回归模型。
* *作用：* **Ablation 对比。** 证明只有 Attention 是不够的，必须要有 Diffusion 的生成过程才能处理不确定性。


2. **LSTM / GRU:** 时序回归模型。
* *作用：* **补充对比点。** 证明空间拓扑相关性（GNN）比单纯的时间相关性更重要。



### C. 概率/小样本学习基准 (Reference Paper & SOTA)

*用于对标直接竞争对手。*

1. 
GP-Regularized SE (Hu & Xu, 2025):


* *作用：* **核心竞品。** 证明在“非高斯分布”下，GP 的单峰假设是错误的，而 Diffusion 是正确的。


2. 
**GCN-II / PAWNN:**  参考文章中的 SOTA 方法。



### D. 生成式 AI 基准 (Generative AI)

*用于证明选择 Diffusion 的正确性。*

1. **C-VAE (Conditional Variational Autoencoder):**
* *作用：* 证明 Diffusion 生成的样本更清晰（High Fidelity），没有 VAE 的模糊问题。


2. **Normalizing Flows (NF, e.g., MAF/RealNVP):**
* *作用：* **补充对比点。** NF 速度快但表达能力弱。证明对于复杂的电网流形，Diffusion 的拟合能力更强。


3. **Standard Diffusion (Concat):**
* *作用：* **Ablation 对比。** 简单的拼接输入，证明你的 **Cross-Attention** 注入机制更有效。



---

## 4. 实验设置 (Experimental Setup)

### 数据集 (Test Systems)

* **IEEE 33-bus:** 机理验证、可视化分析。
* **IEEE 119-bus:** 扩展性测试。
* **IEEE 342-bus (Unbalanced):** 大规模、三相不平衡测试。

### 关键数据生成策略 (Data Generation)

* **光伏 (PV):** 使用 **Beta 分布** 生成出力，，模拟双峰特性。
* **云层遮挡:** 叠加随机的**阶跃信号 (Step Function)**，制造电压突变。
* **负荷 (Load):** 使用真实智能电表数据，叠加高斯混合噪声。
* **量测噪声:** 并不是简单的白噪声，而是包含**野值 (Outliers)** 和 **丢失 (Missing)**。

---

## 5. 核心实验清单 (Key Experiments Checklist)

### 🧪 Exp 1: 非高斯分布下的“绝杀”图 (Distribution Fidelity)

* **目标:** 证明 Diffusion 能拟合双峰分布，而 GP/WLS 不能。
* **操作:** 在光伏波动剧烈时段，画出某节点电压的 **PDF (概率密度函数)** 曲线。
* **指标:** KL Divergence, Wasserstein Distance。
* **预期:** Diffusion 曲线完美贴合真值（双峰），GP 曲线为宽扁单峰。

### 🧪 Exp 2: 零样本拓扑自适应 (Zero-Shot Topology Adaptation)

* **目标:** 证明无需重训即可适应未知拓扑。
* **操作:** 训练集仅含 T1。测试集包含 T2, T3, T4 (未知)。开启 **Hypothesis Testing Guidance**。
* **对比:** WLS (错误拓扑), Hu's Method (无特征图修正), Ours.
* **指标:** RMSE, 物理残差 (Physical Violation)。

### 🧪 Exp 3: 稀疏与坏数据鲁棒性 (Robustness)

* **目标:** 证明 Cross-Attention 的物理检索能力。
* **操作:** 随机移除 80% 量测，或污染 20% 量测。
* **可视化:** 绘制 **Attention Map**，展示权重如何从坏传感器转移到好传感器。
* **对比:** WLS-LAV (传统抗差), GAT, Ours.

### 🧪 Exp 4: 工程权衡分析 (Engineering Trade-off)

* **目标:** 验证 LinDistFlow 引导的实用性。
* **操作:** 对比 **AC-Guidance** vs **LinDistFlow-Guidance**。
* **图表:** 横轴=推理时间/步数，纵轴=精度 (RMSE)。
* **预期:** LinDistFlow 引导速度快 10 倍，精度损失 < 5%。

---

## 6. 评价指标 (Metrics)

1. **准确性:** RMSE, MAE.
2. **不确定性质量:**
* **PICP / MPIW:** (区间覆盖率/宽度)
* **CRPS:** (连续分级概率评分 - 综合概率指标)


3. **物理一致性:**
* **PVR (Physical Violation Rate):** 


4. **计算效率:** Average Inference Time (s).

---

## 7. 实施路线图 (Implementation Roadmap)

* [ ] **Phase 1: 基础搭建**
* 基于 PyTorch Geometric (PyG) 搭建 GNN-Diffusion 骨架。
* 在 IEEE 33 节点上实现前向扩散和反向采样。


* [ ] **Phase 2: 核心模块**
* 实现 **Topology-Guided Cross-Attention**。
* 实现 **Inference-time Guidance** (先用 AC 方程)。


* [ ] **Phase 3: 对比实验**
* 复现 GP-Regularized 方法 (Hu & Xu)。
* 跑通 WLS 和 WLS-LAV 基准。


* [ ] **Phase 4: 进阶功能**
* 实现 **LinDistFlow 近似引导**。
* 实现 **Hypothesis Testing** 拓扑选择逻辑。



