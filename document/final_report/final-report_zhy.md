# 赵会洋负责章节：Baseline、评估与复现说明

> 本文件为最终报告可直接合并版本，用于并入 `document/final_report/核心贡献主线_xwb.md` 或最终报告正文。  
> 数据口径采用组长统一测试结果，参考 `document/presentation/实验报告.md`、`document/presentation/ChiikaDD.pptx` 与 `document/final_report/实验记录表.md`。  
> 代码口径统一指向 `./src`，实验结论不使用个人本地单独运行结果。

---

## 4.3 技术栈

本项目整体采用 Python 作为主要开发语言，围绕表格型回归预测任务构建数据处理、基线建模、核心模型训练、评估与可视化流程。核心代码位于 `src/` 目录，其中 baseline 相关代码集中在 `src/baseline/`，最终模型与统一实验入口位于 `src/generate_final.py` 和 `src/core_model.py`。

| 模块 | 使用技术 | 作用 |
|:---|:---|:---|
| 数据读取与处理 | pandas, NumPy | 读取 `train.csv`、`test.csv`、`sample_submission.csv`，完成目标列清洗、缺失值处理、特征矩阵构造 |
| 特征工程 | pandas, scikit-learn | 时间字段拆分、类别特征 One-hot 编码、数值特征类型转换与缺失填充 |
| 基线模型 | NumPy, scikit-learn, LightGBM, XGBoost | 构建 Simple Linear、Random Forest、LightGBM、XGBoost 四类基线 |
| 核心模型 | LightGBM, scikit-learn | 构建调参后的 LightGBM 与 Random Forest 融合模型 |
| 评估指标 | NumPy, scikit-learn | 计算 RMSE、MAE、R² 等指标 |
| 实验复现 | argparse, JSON, CSV | 支持命令行参数、输出预测文件、保存实验配置与指标 |

baseline 部分采用统一的预处理逻辑，保证四类模型输入特征一致，避免由于数据处理差异影响模型对比。最终核心方法在相同数据口径下进一步进行 LightGBM 超参数适配与 LightGBM/Random Forest 异质模型融合，从而形成从简单基线到进阶模型的完整对照。

---

## 5.1 评测数据集与指标

本项目使用纽约市出租车与豪华轿车委员会（NYC TLC）公开的 2026 年 3 月高运量网约车（FHVHV）行程数据。实验阶段采用统一抽样后的训练集与测试集进行评测：训练集规模为 100,000 行，清洗后约 99,995 行；测试集规模为 10,000 行，其中 9,996 行可用于有效评估。预测目标为乘客基础费用 `base_passenger_fare`。

模型输入特征采用统一的 baseline 编码方案，共形成 61 维特征。该特征集主要包括时间特征、类别 One-hot 特征和数值型行程特征。时间字段不会直接以字符串形式输入模型，而是转换为小时、日期、月份、星期等结构化时间特征；类别字段进行 One-hot 编码；数值字段进行类型转换，并对缺失或无法转换的值填充为 0。

实验主指标为 RMSE，辅助指标为 MAE 和 R²。

| 指标 | 含义 | 作用 |
|:---|:---|:---|
| RMSE | 均方根误差 | 主评估指标，对大误差更敏感，适合衡量费用预测中高价订单的偏差 |
| MAE | 平均绝对误差 | 衡量模型平均每单预测误差，便于从业务角度解释 |
| R² | 决定系数 | 衡量模型对费用波动的解释能力 |

RMSE 的计算公式如下：

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2}
$$

其中，$\hat{y}_i$ 表示模型对第 $i$ 条订单的预测费用，$y_i$ 表示真实基础乘客费用。由于 RMSE 与费用单位一致，因此可以直接解释为美元尺度下的预测误差。

---

## 5.2 基线对比实验

为了建立可靠的性能参照系，本项目构建了四类 baseline 方法：Simple Linear、Random Forest、LightGBM 和 XGBoost。四类模型均使用 `src/baseline/` 中统一的预处理逻辑，包括目标列清洗、时间特征提取、类别特征 One-hot 编码和数值缺失填充，从而保证模型对比主要反映算法差异，而不是数据处理差异。

### 5.2.1 基线方法说明

**Simple Linear** 使用 NumPy 最小二乘法求解线性回归权重。实现时在特征矩阵后拼接一列常数项作为截距，然后通过最小二乘求解模型参数。该方法训练速度快、可解释性较强，但只能拟合线性关系，难以捕捉出租车费用与时间、距离、区域、平台等因素之间的复杂非线性关系。

**Random Forest** 使用 `RandomForestRegressor` 训练多棵决策树，并对多棵树的预测结果取平均。该方法能够捕捉非线性关系，对特征尺度不敏感，并且对异常值相对稳健。由于每棵树基于不同采样和特征划分训练，Random Forest 具有较好的方差控制能力，是本项目中表现最好的 baseline。

**LightGBM** 使用 GBDT 梯度提升框架进行回归训练。模型将训练数据转换为 LightGBM 的 `Dataset` 格式，通过逐轮拟合残差不断提升预测能力。LightGBM 适合结构化表格数据，训练效率高，能够捕捉复杂非线性关系，因此在 baseline 中表现接近 Random Forest。

**XGBoost** 使用 `DMatrix` 数据结构和平方误差回归目标进行训练。XGBoost 是工业界常用的强基线模型，但在本项目统一测试中表现不佳，说明在当前特征编码、参数设置和数据规模下，XGBoost 未能充分适配本任务。

### 5.2.2 基线结果

统一测试结果如下：

| 排名 | 模型 | RMSE | MAE | R² | 说明 |
|:---:|:---|---:|---:|---:|:---|
| 1 | Random Forest | 2.0778 | 1.1671 | 0.9935 | 最优 baseline |
| 2 | LightGBM | 2.0798 | 1.1596 | 0.9935 | 与 Random Forest 非常接近 |
| 3 | Simple Linear | 2.6077 | 1.5344 | 0.9897 | 线性假设限制较明显 |
| 4 | XGBoost | 3.9751 | 1.3879 | 0.9762 | 当前参数与特征设置下表现最差 |

实验结果表明，树模型整体优于简单线性模型，说明费用预测任务中存在明显的非线性关系。其中 Random Forest 取得最优 baseline 结果，RMSE 为 2.0778；LightGBM baseline 与其非常接近，RMSE 为 2.0798。Simple Linear 的 RMSE 为 2.6077，说明单纯线性关系难以充分表达费用变化。XGBoost 在当前统一实验设置下 RMSE 为 3.9751，表现显著弱于其他 baseline，后续核心模型没有将其作为主要融合对象。

基于该对比，后续核心方法选择 Random Forest 和 LightGBM 作为主要优化对象：LightGBM 通过超参数适配降低偏差，Random Forest 通过 Bagging 机制控制方差，两者在预测层进行异质融合。

### 5.2.3 Baseline 与核心方法对比

在最优 baseline 为 Random Forest（RMSE=2.0778）的基础上，核心方法通过 LightGBM 超参数适配和 LightGBM/Random Forest 融合进一步提升性能。最终 LGB+RF 60/40 融合模型 RMSE 为 1.9596，相比最优 baseline 提升 5.69%。

| 方法 | RMSE | MAE | R² | 相对最优 baseline |
|:---|---:|---:|---:|:---:|
| Baseline - Random Forest | 2.0778 | 1.1671 | 0.9935 | 基准线 |
| Core Model - LGB Optimized | 2.0229 | 1.1378 | 0.9938 | +2.64% |
| Core Model - LGB+RF Blend 60/40 | 1.9596 | 1.1169 | 0.9942 | +5.69% |

该结果说明，单一强 baseline 已经能够取得较好效果，但针对数据规模进行超参数适配，并利用不同集成学习范式的互补性进行融合，仍然可以带来稳定提升。

---

## 5.4 一键复现命令

本项目核心实验入口为 `src/generate_final.py`。该脚本支持读取统一处理后的数据，训练优化后的 LightGBM 与 Random Forest，并输出融合预测结果、指标文件和运行配置。

推荐复现命令如下：

```bash
python src/generate_final.py --nrows 100000 --input-dir data/processed --output-dir results/final_100k
```

该命令会在 `results/final_100k` 下输出核心模型预测文件、综合对比指标和运行配置。统一测试结果中使用的训练规模为 100,000 行，对应有效评估样本数为 9,996 行。

若需要分别运行四类 baseline，可执行：

```bash
python src/baseline/simple_linear_model.py
python src/baseline/Random_Forest.py
python src/baseline/LightGBM_model.py
python src/baseline/XGBoost_model.py
```

---

## 5.5 随机种子与可复现性说明

为了提高实验可复现性，本项目在 baseline 与核心模型训练中尽量固定随机种子。Random Forest、LightGBM 和 XGBoost 等涉及随机采样或随机特征选择的模型均使用固定随机种子 42，除种子消融实验外，其余主实验保持一致。

核心模型采用统一的数据划分、统一的 61 维 baseline 特征编码和统一的评估脚本。最终融合模型的权重通过统一实验中的权重搜索确定，最佳结果为 LightGBM 0.60 + Random Forest 0.40，RMSE=1.9596。同时，实验记录表中也保留了 0.50/0.50、0.55/0.45、0.65/0.35 等邻近权重的结果，用于验证融合策略的稳定性。

项目还保存了实验配置文件和指标文件，便于追踪每次运行时的数据规模、模型参数、融合权重和输出目录。最终报告中的实验结论均以统一测试结果为准，避免由于不同成员本地环境、依赖版本或运行参数不同导致指标不一致。

---

## 7. 代码仓库审计

本项目代码主要位于 `src/` 目录，围绕数据处理、baseline 建模、核心模型训练、结果评估和可视化展开。代码结构如下：

| 路径 | 作用 |
|:---|:---|
| `src/baseline/model_utils.py` | baseline 统一预处理工具，包括数据读取、目标列清洗、时间特征提取、类别编码、数值填充和提交文件保存 |
| `src/baseline/simple_linear_model.py` | Simple Linear baseline，使用最小二乘法训练线性回归模型 |
| `src/baseline/Random_Forest.py` | Random Forest baseline，使用多棵决策树平均预测 |
| `src/baseline/LightGBM_model.py` | LightGBM baseline，使用 GBDT 方式训练回归模型 |
| `src/baseline/XGBoost_model.py` | XGBoost baseline，使用平方误差目标训练回归模型 |
| `src/generate_final.py` | 最终统一实验入口，训练优化 LightGBM、Random Forest，并进行模型融合和结果保存 |
| `src/core_model.py` | 核心 LightGBM 模型实现，包含更完整的参数化训练与指标保存逻辑 |
| `src/data_audit.py` | 数据审计脚本，用于识别缺失、异常和数据质量问题 |
| `src/governance_v2.py` | 数据治理流程，包含修复、标记和稳健化处理 |
| `src/preprocess.py` | 数据预处理逻辑 |
| `src/plot_feature_importance.py` | 特征重要性分析与可视化 |
| `src/plot_results.py` | 实验结果可视化 |

baseline 代码经过拆分后，每类模型拥有独立入口文件，同时共享 `model_utils.py` 中的统一预处理逻辑。这样既保证了四类 baseline 的可独立运行，也避免了重复实现数据处理流程。核心模型代码与 baseline 代码在目录层面分离，便于报告中清晰区分“基线方法”和“进阶方法”。

从可复现性角度看，仓库中保留了统一训练入口、固定随机种子、实验记录表和输出配置文件。最终报告中的指标采用统一测试结果，避免将个人本地单独运行结果混入正式结论。

---

## 供核心贡献主线并入的最终段落

以下内容可直接并入 `document/final_report/核心贡献主线_xwb.md` 中已有的对应章节。

### 可放入 2.2 任务形式化的补充

实验使用统一抽样后的 FHVHV 2026 年 3 月数据，训练规模为 100,000 行，清洗后约 99,995 行；测试规模为 10,000 行，其中 9,996 行用于有效评估。输入特征采用 61 维 baseline 编码，预测目标为 `base_passenger_fare`。主指标为 RMSE，辅助指标为 MAE 和 R²。所有模型预测结果均与测试集真实费用逐行对齐后计算指标。

### 可放入 5.3 消融实验的补充

基线实验表明，Random Forest 是最优 baseline，RMSE=2.0778；LightGBM baseline 与其接近，RMSE=2.0798。核心方法首先通过 LightGBM 超参数适配将单模型 RMSE 降至 2.0229，再通过 LGB+RF 60/40 融合将 RMSE 进一步降至 1.9596。该结果说明，最终性能提升不仅来自单模型调参，也来自 Boosting 与 Bagging 两种集成范式的互补。

### 可放入 6.1 失败案例分析的补充

XGBoost baseline 在统一测试中 RMSE=3.9751，显著弱于 Random Forest 和 LightGBM。这说明强模型并不一定在所有参数与特征设置下都能取得优势。对于本任务，模型是否适配当前特征编码、样本规模和目标分布，比模型本身的复杂度更重要。因此最终融合没有纳入 XGBoost，而是选择表现稳定且误差互补的 LightGBM 与 Random Forest。
