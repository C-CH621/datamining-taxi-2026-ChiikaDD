# Baseline 模型说明

本目录包含 4 个可以独立运行的基线模型脚本，用于纽约出租车车费预测任务。

## 文件说明

| 文件 | 模型 | 说明 |
| --- | --- | --- |
| `simple_linear_model.py` | 简单线性模型 | 使用上车点和下车点经纬度差值作为特征，通过 NumPy 最小二乘法拟合线性模型。 |
| `Random_Forest.py` | Random Forest | 使用清洗后的行程数据、Haversine 距离和时间特征，训练随机森林回归模型。 |
| `LightGBM_model.py` | LightGBM | 使用与随机森林相同的清洗和特征工程流程，训练 LightGBM 回归模型。 |
| `XGBoost_model.py` | XGBoost | 使用与随机森林相同的清洗和特征工程流程，训练 XGBoost 回归模型。 |
| `model_utils.py` | 公共工具模块 | 为树模型提供数据读取、特征工程、数据清洗和 submission 输出函数。 |

## 输入数据

每个脚本默认使用 Kaggle 竞赛格式的数据目录，目录中需要包含：

```text
train.csv
test.csv
sample_submission.csv
```

默认输入目录是 `../input`。如果数据不在该目录，可以通过 `--input-dir` 指定。

## 使用方法

建议从项目根目录运行以下命令；如果在其他目录运行，请相应调整路径。

### 简单线性模型

```powershell
python src\baseline\simple_linear_model.py --input-dir path\to\input --output results\baseline\submission_simple-linear-model.csv
```

可选参数：

```powershell
--nrows 10000000
```

`--nrows` 表示读取多少行训练数据，默认值为 `10000000`。

### Random Forest

```powershell
python src\baseline\Random_Forest.py --input-dir path\to\input --output results\baseline\submission_Random_Forest.csv
```

可选参数：

```powershell
--nrows 1000000
--random-state 42
```

`--nrows` 表示读取多少行训练数据，默认值为 `1000000`。  
`--random-state` 用于控制随机森林的随机种子，便于复现实验结果。

### LightGBM

```powershell
python src\baseline\LightGBM_model.py --input-dir path\to\input --output results\baseline\submission_LightGBM.csv
```

可选参数：

```powershell
--nrows 1000000
```

`--nrows` 表示读取多少行训练数据，默认值为 `1000000`。

### XGBoost

```powershell
python src\baseline\XGBoost_model.py --input-dir path\to\input --output results\baseline\submission_XGBoost.csv
```

可选参数：

```powershell
--nrows 1000000
```

`--nrows` 表示读取多少行训练数据，默认值为 `1000000`。

## 输出结果

每个脚本都会生成一个 submission CSV 文件，包含以下两列：

```text
key,fare_amount
```

输出路径由 `--output` 参数控制。如果不指定 `--output`，脚本会在当前运行目录下生成默认名称的 submission 文件。

## 模型对比说明

各 baseline 结果文件的 RMSE 和排名如下。RMSE 越低，模型效果越好。

| 排名 | 结果文件 | 模型 | RMSE |
| --- | --- | --- | --- |
| 1 | `submission_LightGBM.csv` | LightGBM | 3.37 |
| 2 | `submission_Random_Forest.csv` | Random Forest | 3.39 |
| 3 | `submission_XGBoost.csv` | XGBoost | 3.61 |
| 4 | `submission_simple-linear-model.csv` | 简单线性模型 | 5.74 |

从结果看，LightGBM 的 RMSE 最低，是当前 baseline 中表现最好的模型；Random Forest 与 LightGBM 非常接近，XGBoost 次之，简单线性模型作为最基础的 baseline 表现最弱。
