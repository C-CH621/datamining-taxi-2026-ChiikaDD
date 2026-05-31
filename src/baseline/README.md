# Baseline 模型说明

本目录包含 4 个新版 baseline 模型和 1 个 RMSE 评估脚本，均已适配 `data/processed` 中的处理后数据。

## 数据说明

默认读取以下文件：

```text
data/processed/train.csv
data/processed/test.csv
data/processed/sample_submission.csv
```

预测目标为：

```text
base_passenger_fare
```

生成的结果文件包含两列：

```text
pickup_datetime,base_passenger_fare
```

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `model_utils.py` | 公共工具模块，负责读取数据、清洗训练集、构造时间特征、编码类别特征、输出 submission。 |
| `simple_linear_model.py` | 简单线性模型，使用公共预处理后的特征，通过 NumPy 最小二乘法训练。 |
| `Random_Forest.py` | 随机森林回归模型。 |
| `LightGBM_model.py` | LightGBM 回归模型。 |
| `XGBoost_model.py` | XGBoost 回归模型。 |
| `evaluate_rmse.py` | 将 `results/baseline` 中的预测结果与 `data/processed/sample_submission.csv` 对比，计算 RMSE。 |

## 使用的特征

代码会读取 `train.csv` 和 `test.csv` 中的全部列，然后进行统一预处理。

目标列：

```text
base_passenger_fare
```

该列只作为训练标签，不作为模型输入特征。

原始时间列：

```text
request_datetime
on_scene_datetime
pickup_datetime
dropoff_datetime
```

这些原始时间字符串不会直接进入模型，而是派生出 `hour`、`dayofweek`、`day`、`month` 等时间特征。同时额外构造：

```text
request_to_pickup_seconds
request_to_scene_seconds
```

类别特征会进行 one-hot 编码：

```text
hvfhs_license_num
dispatching_base_num
originating_base_num
shared_request_flag
shared_match_flag
access_a_ride_flag
wav_request_flag
wav_match_flag
```

布尔特征会转为 `0/1`。其余可转为数值的列会作为数值特征进入模型。

## 生成预测结果

从项目根目录运行：

```powershell
python src\baseline\simple_linear_model.py --output results\baseline\submission_simple_linear_model.csv
python src\baseline\LightGBM_model.py --output results\baseline\submission_LightGBM.csv
python src\baseline\XGBoost_model.py --output results\baseline\submission_XGBoost.csv
python src\baseline\Random_Forest.py --output results\baseline\submission_Random_Forest.csv
```

默认输入目录为 `data/processed`。如需指定其他目录：

```powershell
--input-dir path\to\processed
```

默认读取前 `1000000` 行训练数据。可以通过 `--nrows` 调整：

```powershell
--nrows 500000
```

Random Forest 额外支持：

```powershell
--n-estimators 30
--n-jobs 1
```

本次实际生成结果时，Simple Linear Model、LightGBM、XGBoost 使用默认 `1000000` 行训练数据；Random Forest 使用 `200000` 行训练数据和 `30` 棵树，因为默认 `1000000` 行、`100` 棵树在当前环境中运行时间过长。

## 计算 RMSE

运行：

```powershell
python src\baseline\evaluate_rmse.py --truth data\processed\sample_submission.csv --results-dir results\baseline
```

RMSE 计算公式为：

```text
RMSE = sqrt(mean((prediction - ground_truth)^2))
```

`sample_submission.csv` 中有 4 行 `base_passenger_fare` 缺失，因此评估脚本会跳过这 4 行，在剩余 `9996` 行上计算 RMSE。

## 当前结果

| 排名 | 模型 | 结果文件 | 训练设置 | RMSE |
| --- | --- | --- | --- | --- |
| 1 | LightGBM | `submission_LightGBM.csv` | `nrows=1000000` | 2.025860 |
| 2 | Random Forest | `submission_Random_Forest.csv` | `nrows=200000`, `n_estimators=30` | 2.129185 |
| 3 | XGBoost | `submission_XGBoost.csv` | `nrows=1000000` | 3.569263 |
| 4 | Simple Linear Model | `submission_simple_linear_model.csv` | `nrows=1000000` | 7.435211 |

从当前结果看，LightGBM 的 RMSE 最低，是这组 baseline 中表现最好的模型；Random Forest 排名第二，但本次使用的训练规模小于其他模型；XGBoost 排名第三；简单线性模型作为基础 baseline，误差最大。
