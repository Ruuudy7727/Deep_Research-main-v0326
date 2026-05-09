## DATABASE 链路设计与流程

本项目的 DATABASE 子链路采用“**LLM 负责理解与规划，程序负责约束与执行**”的混合设计。它的目标不是让模型自由生成 SQL，而是把“自然语言查库”拆成一条可控、可观测、可降级的执行链路：**场景路由 -> 查询规划 -> 计划清洗 -> SQL 构建 -> 接口执行 -> 证据总结 -> 最终回答**。

---

## 一句话总结

这条链路的本质是：**让 LLM 做业务理解和查询规划，让程序掌控最终 SQL 的合法性与执行细节，从而实现既灵活又稳定的数据库查询。**

---

## 1. 整体设计思路

### 1.1 为什么要这样设计

如果直接让 LLM 自由输出 SQL，容易出现以下问题：

- 表名或字段名幻觉
- 时间字段选错
- route 与业务表不匹配
- limit、排序、过滤条件越界
- 生成 SQL 虽“像样”，但并不一定能被真实接口执行

因此本链路采用“**LLM 给候选计划，程序做强约束收敛**”的方式。

### 1.2 设计原则

- **职责分层**
  - LLM 负责“理解问题、判断场景、生成查询计划”
  - 程序层负责“校验、清洗、拼 SQL、执行、容错、标准化”
- **安全优先**
  - 所有表/字段都受 `TABLE_SCHEMAS` 白名单约束
  - route 与可查表之间用 allowlist 限制
  - `limit`、`time_filter`、`order_by` 都会做类型与范围修正
- **稳定优先**
  - 不让 LLM 直接决定最终 SQL
  - 对非法字段、别名字段、历史字段名进行自动修正或剔除
- **可观测优先**
  - 保留完整 DB 链路日志（LLM 三段、sanitizer、SQL、结果预览、错误事件）
  - 单步失败允许降级，不让整条链路直接中断

---

## 2. 分场景设计：每类问题查哪些表

本链路按业务意图分为三类 route：`alerting`、`troubleshooting`、`station_device_td`。三类 route 不是简单分表，而是把“用户问题的目标”映射成“固定可查表集合 + 对应时间语义 + 推荐字段模式”。

### 2.1 `alerting`：异常总览 / 告警摘要

**目标**：先给出风险总览、高频异常关键词、重复模式，再决定是否继续下钻。

**主要表**：

- `alarm_event`：主表，适合做异常摘要、风险等级、触发时段、summary 分析
- `volt_temp_abnormal_result`：补充电压/温度异常细节

**适合回答的问题**：

- 某站点/某 pack 最近有哪些告警？
- 某时间段高频异常词是什么？
- 风险等级高不高？是否存在重复触发？

**推荐过滤条件**：

- 设备定位：`station_code`、`bmu_code`，必要时 `cell_id`
- 时间范围：`time_filter.start_time/end_time`

**时间语义**：

- `alarm_event`：双字段 `start_end`
- `volt_temp_abnormal_result`：单字段，通常按 `time BETWEEN start AND end`

**典型 SQL（alarm_event）**：

```sql
SELECT start_time, end_time, station_code, bmu_code, summary_cn
FROM alarm_event
WHERE station_code = '00256'
  AND bmu_code = 'pack-7'
  AND start_time >= '2026-03-22 00:00:00'
  AND end_time <= '2026-03-22 23:59:59'
ORDER BY start_time DESC
LIMIT 100;
```

### 2.2 `troubleshooting`：根因排查 / 诊断线索

**目标**：基于诊断表给出“容量异常 / 内阻异常 / 微短路 / 自放电”等根因线索与风险排序。

**主要表**：

- `capacity_inconsistent_cells`
- `dcr_abnormal_cells`
- `isc_score_result`
- `volt_temp_abnormal_result`（辅助查看异常伴随现象）

**适合回答的问题**：

- 某 BMU 下哪些电芯疑似微短路？
- 哪些电芯容量不一致或存在自放电风险？
- 某电芯的内阻异常是否持续、严重程度如何？

**推荐过滤条件**：

- 主定位：`bmu_code` + `cell_index`（或 `cell`）
- 可附带：`station_code`
- 常见排序字段：`microshort_score`、`risk_warning_count`、`abnormal_days`

**时间语义**：

- `capacity_inconsistent_cells`：单字段（默认 `first_occurrence`）
- `dcr_abnormal_cells`：单字段（默认 `first_abnormal_time`）
- `isc_score_result`：双字段 `window_start_end`

**典型 SQL（isc_score_result）**：

```sql
SELECT bmu_code, window_id, cell, microshort_score, microshort_score_pct, diagnosis_result
FROM isc_score_result
WHERE bmu_code = 'pack-7'
  AND window_start >= '2026-03-15 00:00:00'
  AND window_end <= '2026-03-22 23:59:59'
ORDER BY microshort_score DESC
LIMIT 50;
```

### 2.3 `station_device_td`：设备时序数据 / 趋势分析

**目标**：提供设备运行过程中的时序数值，用于趋势分析、异常下钻、对照分析或后续画图。

**主要表**：

- `box_data`：箱/柜体级时序数据
- `cluster_data`：簇级时序数据，适合查 `current`、`power`、`soc` 等
- `bmu_data`：BMU/pack 级更细粒度运行数据

**适合回答的问题**：

- 某 cluster 最近 24h 电流/功率波动趋势如何？
- 某 BMU 的 `vmax/vmin`、`tmax/tmin` 如何变化？
- 异常发生前后设备运行曲线怎样？

**推荐过滤条件**：

- 分层定位：`station_code`、`box_code`、`cluster_code`、`bmu_code`
- 时间范围：`time_filter.start_time/end_time`，若未给定通常默认近 24h
- 若要用于画图，`select_fields` 通常至少要包含时间列 `ts` 和一个数值列

**时间语义**：

- 三表均为单字段模式，默认按 `ts BETWEEN start AND end`

**典型 SQL（cluster_data）**：

```sql
SELECT ts, cluster_code, station_code, soc, voltage, current, power
FROM cluster_data
WHERE station_code = '00256'
  AND cluster_code = 'cluster-01'
  AND ts BETWEEN '2026-03-22 00:00:00' AND '2026-03-22 23:59:59'
ORDER BY ts ASC
LIMIT 500;
```

### 2.4 三类场景的统一约束

- route 决定了**允许访问的表范围**
- 表的 schema 决定了**允许访问的字段范围**
- 时间模式决定了**时间条件的写法**
- 最终 SQL 不是 LLM 直接输出，而是程序基于 plan 构建

---

## 3. 提取 SQL 并查询的流程设计思路

### 3.1 端到端流程

1. **Supervisor 路由**
   - `single_agent_supervisor` 判断问题是否要走 DATABASE
   - 同时抽取设备参数，如 `station_code`、`bmu_code`

2. **LLM #1：意图路由**
   - 判断用户问题属于 `alerting`、`troubleshooting`、`station_device_td`
   - 同时补出最小可执行范围：设备范围、时间范围、目标表方向
   - 如果信息不够，则输出 `clarification_needed`

3. **LLM #2：查询规划**
   - 不直接写 SQL，而是输出结构化 `plans[]`
   - 每个 plan 包含：
     - `table`
     - `select_fields`
     - `filters`
     - `time_filter`
     - `order_by`
     - `limit`

4. **程序层：计划清洗（sanitize）**
   - 关键函数：`_sanitize_plan`
   - 负责：
     - 表名别名归一（`_resolve_table_name`）
     - route-table allowlist 校验
     - 字段别名映射（`DB_FIELD_ALIASES`）
     - 删除不在 schema 中的字段
     - 修正时间字段
     - 安全夹紧 `limit`

5. **程序层：SQL 构建（build）**
   - 关键函数：`_build_sql_from_plan`
   - 拼接顺序：
     - `SELECT ... FROM ...`
     - `WHERE filters`
     - 时间条件
     - `ORDER BY`
     - `LIMIT`

6. **执行层：调用真实查询接口**
   - `station_device_td` 优先走 `getTdSqlData`
   - 其他表通常走 `getMySqlData`
   - `sql_parser` / `sql_parser_td` 负责把 SQL 转成真实接口请求参数

7. **结果标准化**
   - 列式响应转行式
   - 去重
   - 按表聚合
   - 保留样本结果用于后续总结

8. **LLM #3：证据融合**
   - 输入：结构化查询结果 JSON（`per_table_rows + sample_rows`）
   - 输出：`db_evidence_bundle`
   - 包含风险等级、重复模式、诊断结论、下一步建议等

9. **最终回答生成**
   - `solve_simple_task` 汇总结果摘要与 evidence bundle
   - 输出用户可读的中文结论

---

## 4. 三个 LLM 各自的作用

### 4.1 LLM #1：路由器

作用：**理解用户到底在问什么数据**。

- 判断场景属于哪一类 route
- 判断是否信息不足需要反问
- 尽量补出最小可执行设备范围和时间范围

一句话概括：**把自然语言问题映射成业务查询场景。**

### 4.2 LLM #2：查询规划器

作用：**把“我要查什么”变成结构化查询计划**。

- 选表
- 选字段
- 生成过滤条件
- 生成时间范围
- 生成排序与 limit

一句话概括：**把业务问题变成 plan，而不是直接变成 SQL。**

### 4.3 LLM #3：证据总结器

作用：**把查询结果变成诊断结论**。

- 汇总风险
- 识别高频模式
- 提炼异常线索
- 给出结论和建议

一句话概括：**把结构化结果变成用户能理解的证据表达。**

---

## 5. 为什么最终 SQL 能稳定执行

虽然 LLM 可能输出不存在的字段、历史别名或不够准确的时间列，但最终 SQL 成功率高，关键原因在于程序层做了“二次收敛”：

- schema 白名单过滤
- 字段别名映射
- 非法项剔除
- 时间字段纠偏
- route 与表的 allowlist 对齐
- `limit` 安全夹紧
- 执行层重试与结果标准化

因此，这条链路并不是“LLM 写 SQL”，而是：

> **LLM 给候选计划，程序生成可执行 SQL。**

---

## 6. 时间语义设计（重点）

系统通过 `TABLE_SCHEMAS` 中的 `time_fields` 与 `time_range_mode` 区分不同表的时间条件写法。

### 6.1 单时间字段模式

- 条件：`<time_field> BETWEEN start AND end`
- 适用表：
  - `volt_temp_abnormal_result`
  - `capacity_inconsistent_cells`
  - `dcr_abnormal_cells`
  - `box_data`
  - `cluster_data`
  - `bmu_data`

### 6.2 双时间字段模式

- `start_end`（如 `alarm_event`）
  - 条件：`start_time >= start AND end_time <= end`
- `window_start_end`（如 `isc_score_result`）
  - 条件：`window_start >= start AND window_end <= end`

> 说明：当前双字段采用“区间完整包含”语义，而不是“区间相交”语义。

### 6.3 设计含义

这意味着不同表的时间过滤并不是统一模板，而是由表 schema 驱动。这样可以确保查询条件与线上真实数据模型保持一致。

---

## 7. 执行层设计：SQL 如何落到真实接口

本链路中生成的是 SQL 风格字符串，但真正执行时，会通过 parser 转成内部接口请求。

- `getMySqlData`
  - 用于 MySQL 风格查询路径
  - `sql_parser` 负责请求参数转换、接口调用、结果 DataFrame 化
- `getTdSqlData`
  - 用于 TD 时序查询路径
  - `sql_parser_td` 负责从 SQL 中解析 `SELECT / FROM / WHERE / ORDER / LIMIT`

执行层还负责：

- 列式响应转行式
- 非法 `selects` 自动移除重试
- 结果结构统一，方便后续 evidence summarizer 使用

---

## 8. 日志、调试与可观测性

### 8.1 核心链路日志

- 文件：`db_chain.jsonl`
- 关键事件：
  - `db_chain_start`
  - `llm_intent_router`
  - `llm_query_planner`
  - `plan_sanitized`
  - `sql_built`
  - `sql_result`
  - `llm_evidence_summarizer`
  - `llm_evidence_summarizer_error`
  - `db_chain_end`

### 8.2 清洗日志（可选）

- 文件：`db_plan_sanitizer.jsonl`
- 记录 plan 清洗详情与执行错误

### 8.3 档位建议

- `LOG_PROFILE=lite`：日常运行，日志较简洁
- `LOG_PROFILE=full`：问题排查，日志更完整

---

## 9. 典型失败与降级策略

### 9.1 LLM JSON 解析失败

- 记录 `*_error` 事件
- 返回错误信息
- 若发生在 evidence 阶段，则使用 fallback `evidence_bundle`，不阻断最终回答

### 9.2 字段不存在或字段名不兼容

- 通过 sanitize 剔除或映射
- 记录 warnings，如 `select_fields_dropped`

### 9.3 接口返回列式结构

- parser 自动转为行式结构
- 保证下游 DataFrame / 记录处理一致

### 9.4 单条 plan 执行失败

- 记录 `plan_execution_error`
- 不影响其他 plan 继续执行

---

## 10. 汇报版总结

如果用一句更适合汇报的话来概括：

> **DATABASE 链路把自然语言查库拆成“场景路由 -> 查询规划 -> 程序清洗 -> SQL 执行 -> 证据总结”五段，让 LLM 负责理解与归纳，让程序负责约束与落地，从而实现既灵活又稳定的数据库查询。**

### 补充记忆点

- **按场景分表**：告警、诊断、时序三类 route 各查各的表
- **按阶段分工**：LLM 做“理解与规划”，程序做“执行与纠偏”
- **按 schema 控制**：表、字段、时间语义全部由代码白名单约束
- **按日志可追踪**：每一步都可落日志，便于排查和复盘