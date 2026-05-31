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

- `simple_linear_model.py` 是最轻量的基线模型，特征较少，运行速度快，但预测能力较弱。
- `Random_Forest.py`、`LightGBM_model.py` 和 `XGBoost_model.py` 共用 `model_utils.py` 中的数据清洗和特征工程逻辑。
- 在已有 baseline 结果中，LightGBM 的记录分数最好，Random Forest 与其非常接近，XGBoost 次之，简单线性模型最弱。
