# Baseline 模型说明

本目录中的脚本已经适配 `data/processed` 下的新数据：

```text
data/processed/train.csv
data/processed/test.csv
data/processed/sample_submission.csv
```

新数据的预测目标为 `base_passenger_fare`，提交文件包含两列：

```text
pickup_datetime,base_passenger_fare
```

## 文件说明

| 文件 | 模型 | 说明 |
| --- | --- | --- |
| `simple_linear_model.py` | 简单线性模型 | 使用公共预处理后的特征，通过 NumPy 最小二乘法拟合线性模型。 |
| `Random_Forest.py` | Random Forest | 使用公共预处理后的特征，训练随机森林回归模型。 |
| `LightGBM_model.py` | LightGBM | 使用公共预处理后的特征，训练 LightGBM 回归模型。 |
| `XGBoost_model.py` | XGBoost | 使用公共预处理后的特征，训练 XGBoost 回归模型。 |
| `model_utils.py` | 公共工具模块 | 负责读取数据、清洗目标列、构造时间特征、编码类别特征和输出 submission。 |

## 运行方式

从项目根目录运行：

```powershell
python src\baseline\simple_linear_model.py --output results\baseline\submission_simple_linear_model.csv
python src\baseline\Random_Forest.py --output results\baseline\submission_Random_Forest.csv
python src\baseline\LightGBM_model.py --output results\baseline\submission_LightGBM.csv
python src\baseline\XGBoost_model.py --output results\baseline\submission_XGBoost.csv
```

默认读取 `data/processed`。如果需要指定其他输入目录，可以使用：

```powershell
--input-dir path\to\processed
```

由于 `train.csv` 较大，脚本默认只读取前 `1000000` 行训练数据。可以通过 `--nrows` 调整：

```powershell
--nrows 500000
```
