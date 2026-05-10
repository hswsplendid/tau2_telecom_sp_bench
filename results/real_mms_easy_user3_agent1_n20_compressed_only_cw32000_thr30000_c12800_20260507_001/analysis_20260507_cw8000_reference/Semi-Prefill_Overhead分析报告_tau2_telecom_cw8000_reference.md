# Semi-Prefill Overhead 分析报告（tau2_telecom，Prefix Caching 场景）

## 1 实验概述

| 参数 | 值 |
|------|-----|
| 模型 | Llama-3.3-70B-Instruct |
| 上下文窗口 (cw) | 32,000 tokens |
| 压缩阈值 (threshold) | 30,000 tokens |
| 保留最近 (keep_recent) | 2,800 tokens |
| 摘要上限 (summary_max) | 1,024 tokens |
| 预留 (reserve) | 2,000 tokens |
| 有效样本数 | 3 / 4 |
| 配置来源 | run_config.json |

**硬件参数（理论模型，与 cw8000 参考报告保持一致）**

| 符号 | 含义 | 值 | 来源 |
|------|------|-----|------|
| $r_{pf}$ | Prefill 速率 | 0.15 ms/tok | cw8000 参考报告 |
| $r_{dec}$ | Decode 速率 | 12.0 ms/tok | cw8000 参考报告 |
| $c_{fix}$ | 请求固定开销 | 20 ms | cw8000 参考报告 |

**异步压缩 vs 同步压缩**

| 模式 | 含义 | 关键路径上的压缩成本 |
|------|------|-------------------|
| **Async（异步摘要）** | 摘要在后台/空闲时预生成，不阻塞用户请求 | 仅 Semi-Prefill：$(B_2+C_1) \times r_{pf} + c_{fix}$ |
| **Sync（同步摘要）** | 摘要在压缩触发时同步生成，用户必须等待 | Summary Prefill $B_1 \times r_{pf}$ + Summary Decode $B_2 \times r_{dec}$ + Semi-Prefill |

> 目录名声明 n20，但当前目录中只有 4 个 timing/ABC 文件，其中 3 个含有效 timing。

---

## 2 Agent 工作负载统计

### 2.1 基本统计

| 指标 | 值 |
|------|-----|
| 总轮数 | 17 |
| LLM calls | 26 |
| 压缩次数 | 3 |
| 压缩率 | 17.6% |
| 平均压缩/请求 | 1.00 |
| 平均 rounds/请求 | 5.67 |
| 平均累计上下文 | 190,983 tokens |
| 累计/首轮范围 | 5.4×–7.2× |

| 样本 | 总轮数 | 压缩次数 | 压缩率 | δ 中位数 | ctx max | 累计 ctx | 累计/首轮 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25efad | 9 | 1 | 11.1% | 122 | 31,132 | 193,859 | 6.43 |
| 37fd19 | 2 | 1 | 50.0% | 0 | 30,685 | 215,874 | 7.17 |
| 81b878 | 6 | 1 | 16.7% | 243 | 31,132 | 163,215 | 5.42 |


### 2.2 每轮新增 tokens（δ）

跨样本 δ 中位数约 **182 tokens/轮**。该值用于估计普通增量 prefill 基线：

$$\delta_{med} \times r_{pf} + c_{fix} \approx 182 \times 0.15 + 20 = 47\text{ ms/轮}$$

### 2.3 上下文长度分布

![Context Length Over Turns](charts/context_length_over_turns.png)

**特征**：图中蓝色为首轮 Full Prefill，绿色为 Prefix Cache 命中的增量轮，橙色为压缩后的 Semi-Prefill 轮。多轮 agentic request 平均跨越 **5.67 rounds**，平均累计上下文为 **191.0K tokens**，是单轮请求的 **5.4×–7.2×**。

### 2.4 Context Churn Ratio

![Context Churn Ratio](charts/context_churn_ratio.png)

增量轮 churn 由新增 token 占当前上下文比例估计；压缩轮 churn 由 $(B_2+C_1)/L_{post}$ 估计。压缩轮通常出现明显尖峰，因为 B₂ 重写导致 C₁ 也需要重新 prefill。

---

## 3 Semi-Prefill 触发统计

### 3.1 每次压缩的 token 段分布

![Semi-Prefill Composition](charts/semi_prefill_composition.png)

| 样本 | 事件 | Turn | A | B2 | C1 | B2+C1 | Async SP ms | Sync event ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25efad | 1 | 6 | 6,848 | 270 | 2,879 | 3,149 | 492 | 6,957 |
| 37fd19 | 1 | 1 | 6,845 | 270 | 3,259 | 3,529 | 549 | 6,885 |
| 81b878 | 1 | 6 | 6,848 | 270 | 2,892 | 3,162 | 494 | 6,959 |


**统计汇总**：

| 指标 | 值 |
|------|-----|
| B₂+C₁ 均值 | 3,280 tokens |
| B₂+C₁ 范围 | 3,149 – 3,529 tokens |
| C₁ 均值 | 3,010 tokens |
| B₂ 均值 | 270 tokens |
| Semi-prefill 均值 | 512 ms/event |
| Sync 单事件均值 | 6,934 ms/event |

### 3.2 Semi-Prefill 尖峰 vs 增量基线

![Semi-Prefill Spikes](charts/semi_prefill_spikes.png)

Async 下每次压缩事件的 semi-prefill 平均为 **512 ms**，约为普通增量 prefill 基线的 **10.8×**。

---

## 4 执行时间分解（Prefix Caching 理论模型）

### 4.1 公式

| 轮类型 | Prefill 成本 | Decode 成本 | 备注 |
|--------|-------------|-------------|------|
| T0（Full Prefill） | $L_0 \times r_{pf} + c_{fix}$ | $O \times r_{dec}$ | 冷启动 |
| 增量轮（Cached） | $\delta \times r_{pf} + c_{fix}$ | $O \times r_{dec}$ | Prefix Cache 命中 |
| 压缩轮（Async） | $(B_2 + C_1) \times r_{pf} + c_{fix}$ | $O \times r_{dec}$ | 仅 Semi-Prefill |
| 压缩轮（Sync） | $B_1 \times r_{pf} + B_2 \times r_{dec} + (B_2+C_1) \times r_{pf} + c_{fix}$ | $O \times r_{dec}$ | 含摘要生成 |

### 4.2 总时间分解

![Phase Breakdown](charts/phase_breakdown_stacked.png)

| 样本 | Async s | Sync s | Full ms | Incr ms | SP ms | Decode ms | SumDec ms | Sync overhead |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25efad | 14.48 | 20.94 | 4,540 | 337 | 492 | 9,108 | 3,240 | 49.7% |
| 37fd19 | 8.69 | 15.03 | 4,537 | 20 | 549 | 3,588 | 3,240 | 84.5% |
| 81b878 | 12.60 | 19.06 | 4,540 | 230 | 494 | 7,332 | 3,240 | 57.5% |


![Phase Pie Charts](charts/phase_pie_charts.png)

### 4.3 Async vs Sync 对比

![Phase Breakdown Sync vs Async](charts/phase_breakdown_sync_vs_async.png)

同步摘要使理论总延迟从 **35.8s** 增加到 **55.0s**，增幅 **53.9%**。新增成本主要来自 Summary Decode。

### 4.4 Prefill-Only 分解（排除 Decode）

![Async Prefill-Only Bar](charts/async_prefill_only_bar.png)

![Async Prefill-Only Pie](charts/async_prefill_only_pie.png)

**关键发现（Async）**：Semi-Prefill 只发生在 **17.6%** 的 rounds，却消耗了 **9.8%** 的 Prefill-only 计算量；Incremental Prefill 占 **3.7%**。

![Prefill-Only Sync vs Async](charts/prefill_only_sync_vs_async_bar.png)

Sync 下 Prefill-only 总量从 **15.7s** 增至 **35.0s**，倍率 **2.2×**，其中 Summary Decode 占 Sync Prefill-only 的 **27.8%**。

### 4.5 Per-Turn Latency Breakdown

前 3 个有效样本的逐轮图如下，文件命名与参考报告保持一致：

![Per-Turn Latency 25efad](charts/per_turn_latency_25efad.png)
![Per-Turn Latency 37fd19](charts/per_turn_latency_37fd19.png)
![Per-Turn Latency 81b878](charts/per_turn_latency_81b878.png)

![Per-Turn Sync vs Async 25efad](charts/per_turn_latency_sync_async_25efad.png)
![Per-Turn Sync vs Async 37fd19](charts/per_turn_latency_sync_async_37fd19.png)
![Per-Turn Sync vs Async 81b878](charts/per_turn_latency_sync_async_81b878.png)

---

## 5 同步 vs 异步摘要：时间占比对比

### 5.1 摘要生成的成本分解

![Per-Event Sync Breakdown](charts/per_event_sync_breakdown.png)

![Per-Event Sync vs Async](charts/per_event_sync_vs_async.png)

Async 下单次压缩事件平均只承担 **0.51s** 的 Semi-Prefill；Sync 下平均膨胀至 **6.93s**。

### 5.2 总延迟与占比

![Sync vs Async Stacked](charts/sync_vs_async_stacked.png)

![Sync vs Async Pie](charts/sync_vs_async_pie.png)

![Prefill-Only Breakdown](charts/sync_vs_async_prefill_only.png)

### 5.3 Overhead 对比

![Overhead Async vs Sync](charts/overhead_async_vs_sync.png)

| 场景 | 压缩 Overhead |
|------|-------------|
| Async | **4.49%** |
| Sync | **60.76%** |

---

## 6 核心结论

| 结论 | 数据支撑 |
|------|----------|
| 单个 agentic request 平均触发 1.00 次压缩 | 3 events / 3 active samples |
| 平均跨越 5.67 rounds | timing/summary 统计 |
| 平均累计上下文 191.0K tokens | sum(prompt_tokens) |
| 累计上下文是单轮的 5.4×–7.2× | accumulated / first prompt |
| Async 压缩 overhead 为 4.49% | 仅 Semi-Prefill 进关键路径 |
| Sync 压缩 overhead 为 60.76% | Summary Decode 进入关键路径 |

## 附录

- 生成脚本：`code/generate_cw8000_reference_analysis.py`
- JSON 结果：`analysis_results.json`
- 表格目录：`tables/`
- 图表目录：`charts/`
