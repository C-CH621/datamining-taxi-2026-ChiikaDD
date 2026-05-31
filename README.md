# datamining-taxi-2026-ChiikaDD
基于 NYC TLC 黄色出租车数据的 Data-Centric 数据挖掘项目，聚焦费用预测与数据质量审查。

## 项目内容
- 任务：出租车行程费用预测与异常样本识别
- 方法：数据初步审查、清洗、特征工程与建模对比
- 数据：`yellow_tripdata_2026-01.parquet`

## 当前进展
- 已完成 2026-01 数据的首轮量化审查（缺失值、异常值、类别分布、重复率、时间一致性）
- 已形成项目方案文档：`ChiikaDD.md`
- 已补齐中期报告所需的代码仓库结构，包括数据审计、预处理、基线模型、进阶模型与评测入口

## 目录
- `proposal/ChiikaDD.md`：项目 proposal 与方法设计
- `data/raw/`：原始 TLC 数据文件目录（大文件不纳入 Git）
- `data/processed/`：清洗与特征工程后的中间数据
- `src/audit.py`：数据质量审计入口
- `src/preprocess.py`：数据清洗与特征工程入口
- `src/baseline.py`：Ridge 线性基线模型
- `src/core_model.py`：HistGradientBoosting 进阶模型
- `src/evaluate.py`：回归指标计算工具
- `results/`：实验指标与预测结果输出目录
- `reports/figures/`：报告图表输出目录
- `tests/`：最小单元测试

## 快速开始
1. 安装依赖：
   ```bash
   pip install pandas pyarrow scikit-learn pytest
   ```
2. 放置原始数据：
   ```text
   data/raw/yellow_tripdata_2026-01.parquet
   ```
3. 运行数据审计：
   ```bash
   python -m src.audit --input data/raw/yellow_tripdata_2026-01.parquet --output results/audit_summary.json
   ```
4. 生成清洗特征：
   ```bash
   python -m src.preprocess --input data/raw/yellow_tripdata_2026-01.parquet --output data/processed/yellow_tripdata_2026-01_features.parquet
   ```
5. 运行基线模型：
   ```bash
   python -m src.baseline --input data/raw/yellow_tripdata_2026-01.parquet --output results/baseline_metrics.json
   ```
6. 运行进阶模型：
   ```bash
   python -m src.core_model --input data/raw/yellow_tripdata_2026-01.parquet --output results/core_model_metrics.json
   ```

## AI 工具声明
本项目使用了 Codex、Gemini 辅助代码与文档工作，所有结果均由团队人工审查。
