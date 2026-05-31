# Raw Data README

## 数据文件

- 文件名：`fhvhv_tripdata_2026-03.parquet`
- 数据来源：`https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2026-03.parquet`
- 数据时间范围：`2026-03`
- 总记录数：`22,058,358`
- 字段数量：`25`

## 字段说明

| 字段名 | 含义 |
|---|---|
| `hvfhs_license_num` | 平台牌照编号（如 Uber/Lyft 对应的 TLC 牌照） |
| `dispatching_base_num` | 派单基础站点编号（负责派单的 base） |
| `originating_base_num` | 起始基础站点编号（行程所属/发起 base） |
| `request_datetime` | 乘客发起叫车请求时间 |
| `on_scene_datetime` | 车辆到达上车点时间 |
| `pickup_datetime` | 实际上车时间（行程开始） |
| `dropoff_datetime` | 实际下车时间（行程结束） |
| `PULocationID` | 上车区域 ID（对应 TLC Taxi Zone） |
| `DOLocationID` | 下车区域 ID（对应 TLC Taxi Zone） |
| `trip_miles` | 行程里程（英里） |
| `trip_time` | 行程时长（通常为秒） |
| `base_passenger_fare` | 乘客基础车费（不含附加费/税） |
| `tolls` | 过路费 |
| `bcf` | Black Car Fund 费用（纽约相关法定附加费） |
| `sales_tax` | 销售税 |
| `congestion_surcharge` | 拥堵附加费 |
| `airport_fee` | 机场附加费 |
| `tips` | 小费 |
| `driver_pay` | 司机收入（平台支付给司机） |
| `shared_request_flag` | 是否为拼车请求标记 |
| `shared_match_flag` | 是否成功匹配为拼车标记 |
| `access_a_ride_flag` | 是否为 Access-A-Ride 相关行程标记 |
| `wav_request_flag` | 是否请求无障碍车辆（WAV）标记 |
| `wav_match_flag` | 是否成功匹配无障碍车辆（WAV）标记 |
| `cbd_congestion_fee` | CBD（中央商务区）拥堵费 |

## 使用建议

- 时间分析建议使用：`pickup_datetime`、`dropoff_datetime`。
- 区域分析建议联动 TLC Taxi Zone 映射表解析 `PULocationID`/`DOLocationID`。
- 金额字段建模前建议做异常值治理（如负值处理、分位数缩尾）。
