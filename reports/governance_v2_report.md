# 数据治理V2：诊断与对照实验

## 1) 数据问题诊断（问题强度 + 证据 + 业务影响）

### 1.1 缺失机制（MCAR/MAR/MNAR）

| 字段 | 缺失率 | 机制判定 | 证据 | 业务影响 |
|---|---:|---|---|---|
| originating_base_num | 27.5933% | MAR_likely | AUC=0.999, KS-p=2.1e-08, effect=0.062 | high |
| speed_mph | 0.0020% | MCAR_likely | missing count too small; no stable mechanism inference | low |

### 1.2 标签噪音

- 规则冲突率：`3.0263%`
- 弱监督不一致率：`17.1704%`
- 疑似标签噪音率：`18.5087%`
- 人工抽检样本：`120`，样本中明显异常占比：`0.00%`
- 抽检文件：`C:\Users\ASUS\Desktop\数据挖掘\datamining-taxi-2026-ChiikaDD\results\manual_audit_sample_noise.csv`

### 1.3 时间漂移（周/月）

| 特征 | PSI(Week1 vs Week4) | KS-p | Wasserstein |
|---|---:|---:|---:|
| trip_miles | 0.0015 | 0.006623 | 0.1357 |
| trip_time | 0.0006 | 0.3317 | 14.7849 |
| base_passenger_fare | 0.0002 | 0.001669 | 0.3640 |
| driver_pay | 0.0003 | 0.0003531 | 0.1736 |

- 月漂移说明：second month file not found

### 1.4 子群体差异（区域/时段/平台）

- 详见 `governance_v2_diagnosis.json` 中 `subgroup` 部分。

## 2) 对照实验设计与结果

- 策略A：只删异常（Delete-only）
- 策略B：标记+修复+缩尾（Governed）
- 固定变量：同切分、同模型、同特征、同随机种子

| 策略 | RMSE | MAE | 子群体误差标准差 | 公平性差距 | 样本损失率 |
|---|---:|---:|---:|---:|---:|
| A_delete | 8.6570 | 4.7044 | 0.7185 | 1.4369 | 0.0087% |
| B_govern | 8.6181 | 4.6993 | 0.7139 | 1.4279 | 0.0000% |

- 显著性检验（配对置换检验）：p-value=`0.1417`
- A-B 平均绝对误差差值（A减B，>0表示B更优）：`0.005146`

## 3) 结论边界

1. 本结论基于 2026-03 单月原始数据；月级漂移需补充至少两个月数据。
2. 人工抽检为规则引导抽检，不等同于双人盲审；最终噪音率可进一步复核。
3. 策略收益与当前目标字段（`base_passenger_fare`）相关，迁移到其他目标需重评估。