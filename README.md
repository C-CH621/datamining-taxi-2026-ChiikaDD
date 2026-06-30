# datamining-taxi-2026-ChiikaDD

NYC TLC 高容量网约车（FHVHV）费用预测与 Data-Centric 数据治理项目。项目最终口径使用 `fhvhv_tripdata_2026-03`，预测目标为 `base_passenger_fare`，围绕全量数据审计、治理诊断、baseline 对比、LightGBM/Random Forest 融合、消融实验和 SHAP 可解释性展开。

## 项目结构

```text
.
├── data/
│   ├── raw/                         # 原始大文件目录，不纳入 Git
│   └── processed/                   # 处理后数据说明与小型元数据
├── document/
│   ├── proposal/                    # 开题报告
│   ├── midterm/                     # 中期报告
│   ├── presentation/                # 展示材料
│   └── final_report/                # 最终报告、成员稿与 PDF
├── reports/                         # 数据治理报告
├── results/                         # 实验结果、预测文件、图表
├── src/
│   ├── data_audit.py                # 原始数据流式审计
│   ├── governance_v2.py             # 缺失/噪声/漂移/子群体治理诊断
│   ├── preprocess.py                # 审计、治理与切分入口
│   ├── baseline/                    # Simple Linear / RF / LGB / XGB baseline
│   ├── core_model.py                # 核心 LightGBM 模型
│   ├── generate_final.py            # 最终 LGB+RF 融合实验入口
│   ├── plot_feature_importance.py   # LightGBM importance 与 SHAP 图
│   ├── plot_results.py              # 实验图表生成
│   └── evaluate.py                  # 通用回归指标评估工具
└── tests/
```

## 主要结果

主实验使用 100,000 行训练口径和 9,996 条有效测试样本。最终 LGB+RF 60/40 融合模型达到：

| 模型 | RMSE | MAE | R² |
|:---|---:|---:|---:|
| Random Forest baseline | 2.0778 | 1.1671 | 0.9935 |
| Optimized LightGBM | 2.0229 | 1.1378 | 0.9938 |
| LGB+RF Blend 60/40 | 1.9596 | 1.1155 | 0.9942 |

最终报告见：

- `document/final_report/final-report.md`
- `document/final_report/final-report.pdf`

## 快速复现

安装依赖：

```bash
pip install -r requirements.txt
```

运行最终融合实验：

```bash
python src/generate_final.py \
  --nrows 100000 \
  --input-dir data/processed/fhvhv_tripdata_2026-03_prediction_format \
  --output-dir results/final_100k
```

运行 SHAP 与特征重要性图：

```bash
python src/plot_feature_importance.py \
  --nrows 100000 \
  --input-dir data/processed/fhvhv_tripdata_2026-03_prediction_format \
  --output-dir results/figures
```

评估单个提交文件：

```bash
python src/evaluate.py \
  --truth data/processed/fhvhv_tripdata_2026-03_prediction_format/sample_submission.csv \
  --prediction results/final_100k/submission_core_model_blended.csv
```

运行测试：

```bash
pytest
```

## 数据说明

原始 TLC 数据文件较大，不提交到 Git。当前仓库保留了处理后预测格式数据、实验结果和报告所需图表。若从原始数据重建，可参考：

```bash
python src/preprocess.py audit \
  --input data/raw/fhvhv_tripdata_2026-03.csv \
  --output results/raw_audit_2026_03.json

python src/governance_v2.py

python src/preprocess.py split \
  --input data/processed/fhvhv_tripdata_2026-03_governed_v2_full.csv \
  --output-dir data/processed/fhvhv_tripdata_2026-03_prediction_format
```

## AI 工具声明

本项目使用 Codex、Gemini 辅助代码、调试和文档整理。所有报告结论均以仓库中的脚本、实验记录和结果文件为依据，并由团队人工审查。
