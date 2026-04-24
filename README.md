# datamining-taxi-2026-ChiikaDD
基于 NYC TLC 黄色出租车数据的 Data-Centric 数据挖掘项目，聚焦费用预测与数据质量审查。

## 项目内容
- 任务：出租车行程费用预测与异常样本识别
- 方法：数据初步审查、清洗、特征工程与建模对比
- 数据：`yellow_tripdata_2026-01.parquet`

## 当前进展
- 已完成 2026-01 数据的首轮量化审查（缺失值、异常值、类别分布、重复率、时间一致性）
- 已形成项目方案文档：`ChiikaDD.md`

## 目录
- `ChiikaDD.md`：项目 proposal 与方法设计
- `audit_yellow_2026_01.py`：Parquet 审查脚本
- `yellow_tripdata_2026-01.parquet`：样例数据

## 快速开始
1. 安装依赖：`pip install pandas pyarrow`
2. 运行审查脚本：`python audit_yellow_2026_01.py`

## AI 工具声明
本项目使用了 Codex、Gemini 辅助代码与文档工作，所有结果均由团队人工审查。

