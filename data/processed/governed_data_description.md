# 治理后数据说明（fhvhv_tripdata_2026-03_governed_v2_full）

## 1. 数据概览
- 原始数据：`fhvhv_tripdata_2026-03.csv`
- 治理后数据：`fhvhv_tripdata_2026-03_governed_v2_full.parquet / .csv`
- 数据规模：`22,058,358` 行（保留全量，不做大规模删样本）
- 目标用途：出租车费用相关建模与分析（以 `base_passenger_fare` 为核心目标）

## 2. 治理目标
1. 保留业务信息：避免“异常即删除”的粗暴清洗。
2. 提升训练可用性：对影响预测的缺失与异常进行修复或稳健变换。
3. 可审计可追踪：每类治理动作都有对应标记字段（`flag_*`）。

## 3. 核心治理策略

### 3.1 语义约束治理（规则层）
- 对不应为负的关键数值字段（如 `base_passenger_fare`, `trip_miles`, `trip_time`, `driver_pay`）：
  - 负值不直接删行；
  - 转为缺失并打标记：
    - `flag_fare_negative`
    - `flag_driver_pay_negative`
    - `flag_trip_miles_negative`
    - `flag_trip_time_negative`

### 3.2 一致性治理（跨字段层）
- 新增派生字段：
  - `duration_seconds = dropoff_datetime - pickup_datetime`
  - `speed_mph = trip_miles / (trip_time/3600)`
- 新增一致性异常标记：
  - `flag_duration_inconsistent`：`|duration_seconds - trip_time| > 300s`
  - `flag_speed_outlier`：`speed_mph < 1` 或 `> 80`

### 3.3 缺失值治理（影响导向）
- 原则：
  - 对预测影响小且可证明低风险的情况可删除（本版本基本保留全量）；
  - 对影响预测的关键特征优先补全。
- 本版补全方式（训练友好）：
  - 对 `trip_miles`, `trip_time`, `driver_pay`, `tips`
  - 先按 `dispatching_base_num + pickup_hour` 分组中位数补全；
  - 若组内仍缺失，再用全局中位数补全。

### 3.4 稳健化治理（统计层）
- 对长尾数值特征增加缩尾列（保留原始列）：
  - `trip_miles_capped`
  - `trip_time_capped`
  - `driver_pay_capped`
  - `tips_capped`
- 缩尾区间：约 `0.1% ~ 99.9%` 分位，降低极端值对模型参数的杠杆效应。

### 3.5 类别标准化
- `shared_request_flag`, `shared_match_flag`, `access_a_ride_flag`, `wav_request_flag`, `wav_match_flag`
- 标准化为 `Y / N / UNK`，避免脏类别导致编码问题。

## 4. 新增字段说明（治理痕迹）
- 标记字段：
  - `flag_fare_negative`
  - `flag_driver_pay_negative`
  - `flag_trip_miles_negative`
  - `flag_trip_time_negative`
  - `flag_duration_inconsistent`
  - `flag_speed_outlier`
- 派生字段：
  - `duration_seconds`
  - `speed_mph`
  - `pickup_hour`
- 稳健特征：
  - `trip_miles_capped`
  - `trip_time_capped`
  - `driver_pay_capped`
  - `tips_capped`
- 质量汇总：
  - `quality_issue_count`（每行命中治理问题的数量）

## 5. 字段字典（含义说明）

### 5.1 原始业务字段

| 字段名 | 含义 |
|---|---|
| `hvfhs_license_num` | 平台牌照编号（如 Uber/Lyft 对应 TLC 牌照） |
| `dispatching_base_num` | 派单基础站点编号 |
| `originating_base_num` | 起始基础站点编号 |
| `request_datetime` | 乘客发起叫车请求时间 |
| `on_scene_datetime` | 车辆到达上车点时间 |
| `pickup_datetime` | 实际上车时间（行程开始） |
| `dropoff_datetime` | 实际下车时间（行程结束） |
| `PULocationID` | 上车区域 ID（TLC Taxi Zone） |
| `DOLocationID` | 下车区域 ID（TLC Taxi Zone） |
| `trip_miles` | 行程里程（英里） |
| `trip_time` | 行程时长（通常为秒） |
| `base_passenger_fare` | 乘客基础车费（不含附加费/税） |
| `tolls` | 过路费 |
| `bcf` | Black Car Fund 附加费 |
| `sales_tax` | 销售税 |
| `congestion_surcharge` | 拥堵附加费 |
| `airport_fee` | 机场附加费 |
| `tips` | 小费 |
| `driver_pay` | 司机收入（平台支付给司机） |
| `shared_request_flag` | 是否为拼车请求 |
| `shared_match_flag` | 是否成功拼车匹配 |
| `access_a_ride_flag` | 是否为 Access-A-Ride 相关行程 |
| `wav_request_flag` | 是否请求无障碍车辆（WAV） |
| `wav_match_flag` | 是否成功匹配无障碍车辆（WAV） |
| `cbd_congestion_fee` | CBD（中央商务区）拥堵费 |

### 5.2 治理新增字段

| 字段名 | 含义 |
|---|---|
| `flag_fare_negative` | `base_passenger_fare` 是否出现负值 |
| `flag_driver_pay_negative` | `driver_pay` 是否出现负值 |
| `flag_trip_miles_negative` | `trip_miles` 是否出现负值 |
| `flag_trip_time_negative` | `trip_time` 是否出现负值 |
| `duration_seconds` | 由 `dropoff_datetime - pickup_datetime` 计算的时长（秒） |
| `speed_mph` | 速度（英里/小时）=`trip_miles / (trip_time/3600)` |
| `flag_duration_inconsistent` | `duration_seconds` 与 `trip_time` 是否冲突（阈值 300 秒） |
| `flag_speed_outlier` | 速度是否异常（<1 或 >80 mph） |
| `pickup_hour` | 上车小时（0-23） |
| `trip_miles_capped` | 里程缩尾后字段（稳健版） |
| `trip_time_capped` | 时长缩尾后字段（稳健版） |
| `driver_pay_capped` | 司机收入缩尾后字段（稳健版） |
| `tips_capped` | 小费缩尾后字段（稳健版） |
| `quality_issue_count` | 每条样本命中的治理问题数量（质量问题计数） |

## 6. 文件位置
- Parquet：`data/processed/fhvhv_tripdata_2026-03_governed_v2_full.parquet`
- CSV：`data/processed/fhvhv_tripdata_2026-03_governed_v2_full.csv`
- 配套诊断：
  - `data/processed/governance_v2_diagnosis.json`
  - `data/processed/governance_v2_experiment_table.csv`
  - `data/processed/governance_v2_report.md`
