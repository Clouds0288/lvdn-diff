
# 📊 Experimental Design: Generative AI for Robust Distribution System State Estimation

**Project:** Physics-Guided Diffusion for Power Grids
**Focus:** Comparative analysis of Generative AI vs. Traditional/Discriminative methods under uncertainty.

---

## 1. 🎯 研究背景与核心目标 (Background & Objectives)

在配电网实际工程中，我们面临着**“低可观测性”**与**“高不确定性”**的双重挑战。传统的状态估计方法（如 WLS）严重依赖完整的数据和准确的物理参数，而在实际场景中往往难以满足。

本实验旨在系统性地验证 **生成式 AI (Physics-Guided Diffusion)** 相对于 **传统物理方法** 和 **判别式 AI** 在处理数据缺失、噪声干扰及拓扑错误时的优劣性。

**核心假设：**

1. 在数据完整且准确时，传统方法与判别式 AI 效率更高。
2. 在数据缺失、含噪声或物理参数（如相位连接）错误时，生成式 AI 具备更强的鲁棒性和纠错能力。

---

## 2. ⚔️ 对比模型 (Competing Models)

我们将构建三类模型进行平行对比实验：

| ID | 模型类型 | 具体实现 | 输入数据 | 输出形式 | 优势/劣势 |
| --- | --- | --- | --- | --- | --- |
| **M1** | **Baseline (Physics)** | **WLS State Estimation** (via `pandapower`) | 拓扑 + 已知节点  | 确定性状态值 | ✅ 物理严谨❌ 不可观测时报错，对坏数据敏感 | 
<br>
| **M2** | **Discriminative AI** | **GNN Regressor** (Backbone only) | 拓扑 + 已知节点 + Mask | 确定性预测值 | ✅ 推理速度快❌ 倾向于均值回归，无置信区间 |
<br>
| **M3** | **Generative AI** | **Physics-Guided Diffusion** (+ Inpainting/CQR) | 拓扑 + 已知节点 + Mask | **概率分布** (均值 + 区间) | ✅ 鲁棒性强，可量化不确定性❌ 计算开销较大 |

---

## 3. 🛠️ 训练策略 (Training Strategy for AI Models)

为了让 AI 模型（M2 和 M3）适应各种不确定性场景，我们不针对特定缺失率训练多个模型，而是训练一个**“通用补全模型” (Universal Inpainting Model)**。

### 3.1 动态掩码训练 (Dynamic Masking)

在训练过程中，对每个 Batch 实时随机生成掩码（Mask）：

* **输入特征**: ，包含 `[P_meas, Q_meas, V_meas, Mask_P, Mask_Q, Mask_V, Time_Emb]`。
* **随机策略**: 随机选择  的节点作为“已知量测”，其余节点作为“未知待求”（填 0 或均值）。
* **目的**: 强迫模型学会从任意稀疏的观测中推断全网状态。

### 3.2 噪声注入 (Noise Injection)

* **训练时**: 给已知节点的量测值叠加  的高斯噪声。
* **目的**: 防止模型过拟合于完美的 Ground Truth，教导模型“输入仅供参考”。

---

## 4. 🧪 实验场景设计 (Scenarios & Sensitivity Analysis)

我们将通过三组实验来全方位评估模型性能。

### 📌 Experiment A: 可观测性灵敏度 (Observability Sensitivity)

**目标**: 验证在传感器数量极其有限的情况下，谁能猜得更准。

* **变量**: 节点观测率 。
* : 仅知变电站总表 ()，难度最高。
* : 几乎全网可视。


* **数据设置**: 无额外坏数据，拓扑正确。
* **预期**: M3 (Diffusion) 在  的低观测区应显著优于 M2；M1 在低观测区可能无法计算。

### 📌 Experiment B: 抗噪与坏数据测试 (Robustness & Bad Data)

**目标**: 验证模型在数据质量差时的表现。

* **固定条件**: 观测率  (模拟典型工程场景)。
* **变量 1 (噪声)**: 量测噪声 。
* **变量 2 (坏数据)**: 随机将 5% 的已知量测值篡改为异常值（如归零或翻倍）。
* **预期**: M1 误差飙升；M2 受影响较大；M3 应能利用物理约束“过滤”掉坏数据。

### 📌 Experiment C: 拓扑/相位不确定性 (Topology Uncertainty)

**目标**: **(核心难点)** 验证当物理先验（相位连接）错误时，AI 能否纠错。

* **场景描述**: 实际用户接 A 相，但输入给模型的 `edge_index` 错误地标记为 B 相。
* **变量**: 相位错误率 。
* **预期**:
* M1 (WLS) 必然计算错误。
* M3 (Diffusion) 可能通过生成的 **Likelihood (似然度)** 或 **Residual (残差)** 暗示出拓扑错误（即：按 B 相算出来的电压分布概率极低，从而推断可能接错了）。



---

## 5. 📏 评价指标 (Evaluation Metrics)

### 5.1 准确性 (Accuracy) —— 适用于所有模型

* **RMSE (Root Mean Square Error)**: 电压/功率预测误差。
* **MAE (Mean Absolute Error)**: 平均绝对误差。

### 5.2 可靠性 (Reliability) —— 仅适用于生成式 AI (M3)

* **CRPS (Continuous Ranked Probability Score)**: 衡量概率分布与真实值的吻合度（综合指标）。
* **Coverage Rate (覆盖率)**: 真实值落在 95% 置信区间内的比例（目标：）。
* **Interval Width (区间宽度)**: 在满足覆盖率前提下，区间越窄越好（体现敏锐度）。

### 5.3 工程实用性 (Application)

* **越限检测 (Violation Detection)**:
* **Recall (召回率)**: 真实越限的节点，模型抓住了多少？
* **Precision (精确率)**: 模型报警的节点，有多少是真的越限了？



---

## 6. 📅 执行路线 (Roadmap)

### Phase 1: Infrastructure (基础改造)

* [ ] **Data Prep**: 修改数据预处理脚本，保留原始  矩阵，支持动态 Mask 生成。
* [ ] **Model Arch**: 升级模型输入层，维度从 2 扩充为 7 (支持 Mask 和 Condition 输入)。
* [ ] **Training Loop**: 实现 `DynamicMaskBatch` 采样器，加入训练时噪声注入。

### Phase 2: Training (模型训练)

* [ ] 训练 **Generative Model (Diffusion)** (Target: Inpainting capable).
* [ ] 训练 **Discriminative Model (GNN Regressor)** (Baseline).

### Phase 3: Evaluation (评测与绘图)

* [ ] 实现 **WLS Baseline** (Pandapower 接口)。
* [ ] 编写自动化评测脚本，遍历 Exp A/B/C 的所有参数组合。
* [ ] **Visualization**:
* [ ] 绘制 "RMSE vs. Observability Ratio" 折线图。
* [ ] 绘制 "Coverage vs. Noise Level" 柱状图。
* [ ] 绘制 "Phase Error Correction" 案例图 (展示 AI 如何修正错误相位导致的电压偏差)。



---

## 7. 📂 目录结构 (Project Structure)

```
├── notebooks/
│   ├── 01_data_generation.ipynb       # 生成 CIGRE 基础数据
│   ├── 02_data_preprocessing.ipynb    # [需修改] 增加 Mask 支持
│   ├── 03_model_definition.ipynb      # [需修改] 升级输入维度
│   ├── 04_training_loop.ipynb         # [需修改] 动态掩码训练
│   ├── 05_inference_cqr.ipynb         # Conformal Prediction 验证
│   └── 06_comprehensive_eval.ipynb    # [新建] 运行 Exp A/B/C 并画图
├── src/
│   ├── model.py                       # 模型类定义
│   ├── trainer.py                     # 训练逻辑
│   └── baselines.py                   # WLS 和 GNN-Reg 实现
└── README.md                          # 本文档

```