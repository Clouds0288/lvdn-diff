
# 物理引导扩散模型用于低压配电网可观测性增强

# Physics-Guided Diffusion for Observability Enhancement in Low Voltage Distribution Networks

## 1. 项目概述 (Project Overview)

**研究目标：** 针对欧洲低压配电网（Low Voltage Distribution Networks, LVDNs）中普遍存在的**稀疏可观测性 (Sparse Observability)** 挑战，仅利用变压器首端（Head-end）测量数据 () 及历史统计数据，重构节点级的电压和功率状态。

**核心方法论：** 采用一种**受物理引导 (Physics-Guided)** 的**条件扩散模型 (Conditional Diffusion Model / Score-based Generative Model)**。该模型旨在解决电力系统逆向潮流运算中典型的一对多、**病态反演 (Ill-posed Inverse Problem)** 问题。

---

## 2. 系统描述：CIGRE 欧洲低压基准模型 (System Description: CIGRE European LV Benchmark)

本研究利用 **CIGRE 欧洲低压配电网（住宅馈线变体）** 作为仿真测试床：

* **电压等级：** 0.4 kV（线间电压 Line-to-Line）。
* **拓扑结构：** 辐射型结构 (Radial structure)，采用典型的欧洲地下电缆布置。
* **关键特性：**
* **三相不平衡 (Three-phase Unbalance)：** 存在非对称负荷及单相光伏 (PV) 接入。
* **高 R/X 比：** 电阻特性显著，导致传统的线性潮流模型（如 LinDistFlow）精度不足。
* **产消者渗透率 (Prosumer Penetration)：** 住宅节点具有高比例的屋顶光伏接入。



---

## 3. 实验设计 (Experimental Design)

### 3.1 数据生成/数字孪生 (Data Generation / Digital Twin)

由于缺乏现实中的高频同步测量数据，我们通过 **Pandapower** 构建了合成数据流水线：

* **输入分布 (Input Distributions)：**
* **负荷 (Load)：** 标准负荷特性曲线 (H0) + 高斯噪声 + 用户特定缩放因子。
* **光伏 (PV)：** 慕尼黑历史辐照度数据 + 空间相关性噪声（模拟云层移动）。


* **仿真模拟：** 执行时序潮流计算 (Time-series Power Flow / Quasi-static)，生成 50,000 个样本（约等效于 1 年的 15 分钟分辨率数据）。
* **数据集结构：**
* **X (条件/输入)：** 变压器端测量值 、时间嵌入 (Time Embedding)、历史统计均值 。
* **Y (目标/真值)：** 节点电压幅值 、节点有功功率注入 。



### 3.2 模型架构 (Model Architecture)

**混合架构：图扩散模型 (Hybrid Architecture: Graph-Diffusion)**

1. **骨干网络 (Backbone)：** 采用 **图神经网络 (GNN)** 捕捉配电网的物理拓扑依赖和空间相关性。
2. **生成机制：** 采用 **去噪扩散概率模型 (DDPM)** 学习条件概率分布 。
3. **物理引导 (Physics Guidance)：**
* **训练阶段：** 通过损失函数引入基于物理方程的**软约束 (Soft Constraint)**。
* **采样阶段：** 利用潮流方程（或基尔霍夫定律 KCL/KVL 的线性近似）的梯度进行**流形引导 (Manifold Guidance)**。



### 3.3 评估指标 (Evaluation Metrics)

本研究重点在于**风险评估**，而非单纯的点估计准确度：

* **CRPS (Continuous Ranked Probability Score)：** 用于评估模型生成的概率分布质量。
* **违规检测得分 (VDS)：** 评估模型捕捉“尾部风险”（如电压越限、过载）的能力，这些风险通常会被确定性回归模型抹平。
* **物理一致性 (Physical Consistency)：** 计算生成数据的功率平衡残差 。

---

## 4. 基准模型对比 (Baselines for Comparison)

为了验证扩散模型的优越性，将与以下方法进行对比：

1. **WLS 状态估计 (Weighted Least Squares)：** 使用伪测量值 (Pseudo-measurements) 的传统工业界方法。
2. **确定性 GNN：** 标准的图回归模型，旨在最小化均方误差 (MSE)。
3. **高斯过程 (Gaussian Process)：** 假设数据符合高斯分布的经典概率基准。

