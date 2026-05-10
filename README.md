# tau2 telecom semi-prefill compression benchmark

这个目录是针对 `tau2-bench` 的 `telecom` 类别新增的压缩与 semi-prefill 测试方案。它现在采用 **zero-rewrite 数据集筛选 + reference 初始上下文塑形** 的方式：不改 tau2 原始 `tasks.json`，只生成不同实验子集的 task id 文件；需要长上下文时，只在 agent 内部历史中注入中性 telecom reference context，用来制造可压缩的 B1 段。

它参考了 `/root/agentbench_semi_prefill_bench` 的 prompt/ABC/timing 保存形式，以及 `/root/bfcl_long_context_sp_bench` 的 run-name 隔离、样本筛选和分析指标。

## 当前目标

- 原始数据集：`/root/tau2-bench/data/tau2/domains/telecom/tasks.json`
- 数据集改造方式：`zero_rewrite_id_filter_only`，只筛选 task IDs，不修改 task JSON
- 当前优先子集：`data/telecom_mms_easy_none_user3_agent1_actions_4_6_ids.json`
- 当前推荐运行模式：`--task-split full --initial-context-mode reference --target-initial-tokens 29800`
- 默认压缩参数：`cw=32000`，`reserve=2000`，阈值 `thr=30000`
- C1 最近保留段：默认 `2800` tokens，目标范围 `2000-3000`，允许诊断中看到 `>3000`
- 目标触发率：同时报告 `compressions / agent LLM steps`、`compressions / dialogue messages`、`compressions / tau2 messages`。真实 tau2 turn 口径优先看后两者，目标 `P <= 1/5` 或 `P <= 1/6`。
- 输出：完整 prompt、trace、timing、checkpoint、压缩前后 ABC 段和逐 message token 记录

当前已经验证的代表 run：

```text
results/real_mms_easy_user3_agent1_n20_compressed_only_cw32000_thr30000_c12800_20260507_001
```

该目录名中的 `n20` 表示样本 id 文件声明 20 个候选样本；实际已完成/可读样本数需要以 `timing/`、`abc_segments/` 和 `tau2_results/results.json` 为准。

## 目录结构

```text
tau2_telecom_sp_bench/
  sp_config.py          # 路径、模型、压缩参数、输出目录
  dataset_rewrite.py    # 多条件 zero-rewrite task-id 筛选
  agent.py              # tau2 agent 注册、压缩、ABC/prompt/timing/checkpoint 记录
  run.py                # prepare-data / run / analyze 入口
  analyze.py            # workload、semi-prefill、score 分析
  data/                 # 样本 ID 和数据筛选摘要
  results/              # 新实验输出目录，不覆盖旧结果
```

## 数据集改造策略

本项目不直接改写 tau2 原始任务。所有子集均由 `dataset_rewrite.py` 根据 `evaluation_criteria.actions`、task family、persona、requestor action 数、ticket 是否包含 phone number、ID 关键词等条件筛选生成。

基础全集统计：

```text
telecom total tasks: 2285
actions >= 6: 1275
```

基础筛选命令：

```bash
cd /root/tau2_telecom_sp_bench
./run_tau2.sh prepare-data --min-actions 6
```

这会写入：

- `data/telecom_actions_ge6_ids.json`
- `data/telecom_actions_ge6_summary.json`

### 最新推荐子集

当前更推荐用更稳定、动作数适中、容易控制 user/agent 行为的 telecom 子集，而不是直接跑全量 `actions>=6`。已有子集如下：

| ID 文件 | 样本数 | 主要筛选条件 | 用途 |
|---|---:|---|---|
| `data/telecom_actions_ge6_ids.json` | 1275 | actions >= 6 | 最大 zero-rewrite 候选池 |
| `data/telecom_long_actions_ge10_ids.json` | 57 | actions >= 10，user actions >= 8，ticket 有 phone，按 actions 降序 | 长动作链压力测试 |
| `data/telecom_medium_actions_6_8_ids.json` | 20 | actions 6-8，user actions >= 4，assistant actions >= 1，mms/mobile，各 family 最多 10 | 中等长度混合子集 |
| `data/telecom_medium_mobile_easy_none_actions_6_8_ids.json` | 10 | mobile_data_issue，persona Easy/None，actions 6-8 | mobile data 专项 |
| `data/telecom_mms_easy_none_actions_4_6_ids.json` | 20 | mms_issue，persona Easy/None，actions 4-6，user actions >= 3 | MMS 简化稳定子集 |
| `data/telecom_mms_easy_none_user3_agent1_actions_4_6_ids.json` | 20 | mms_issue，persona Easy/None，actions 4-6，user actions >= 3，assistant actions >= 1 | 当前优先分析子集 |
| `data/telecom_service_overdue_easy_none_actions_4_ids.json` | 1 | service_issue，overdue_bill_suspension，actions=4 | service overdue 单样本 smoke |
| `data/telecom_stable_easy_none_agent1_actions_3_5_ids.json` | 10 | mms/mobile/service，persona Easy/None，actions 3-5，排除不稳定故障关键词 | stable easy baseline/compressed 对比 |

这些文件对应的 summary 都在 `data/*_summary.json`。如果重新生成同名文件，会覆盖筛选摘要；需要保护旧筛选记录时，请指定新的 `--output` 和 `--summary-out`。

### 推荐生成命令

当前优先 MMS 子集：

```bash
cd /root/tau2_telecom_sp_bench
./run_tau2.sh prepare-data \
  --min-actions 4 \
  --max-actions 6 \
  --min-user-actions 3 \
  --min-assistant-actions 1 \
  --families mms_issue \
  --personas Easy None \
  --require-ticket-phone \
  --sort-by-actions-asc \
  --limit-per-family 20 \
  --output data/telecom_mms_easy_none_user3_agent1_actions_4_6_ids.json \
  --summary-out data/telecom_mms_easy_none_user3_agent1_actions_4_6_summary.json
```

stable easy 对比子集：

```bash
./run_tau2.sh prepare-data \
  --min-actions 3 \
  --max-actions 5 \
  --min-user-actions 2 \
  --min-assistant-actions 1 \
  --families mms_issue mobile_data_issue service_issue \
  --personas Easy None \
  --exclude-id-terms \
    bad_vpn bad_wifi_calling break_apn_mms_setting break_apn_settings \
    break_app_both_permissions break_app_sms_permission break_app_storage_permission \
    data_saver_mode_on data_usage_exceeded unseat_sim_card \
    user_abroad_roaming_disabled_on user_abroad_roaming_enabled_off \
  --require-ticket-phone \
  --sort-by-actions-asc \
  --limit-per-family 20 \
  --output data/telecom_stable_easy_none_agent1_actions_3_5_ids.json \
  --summary-out data/telecom_stable_easy_none_agent1_actions_3_5_summary.json
```

如果 zero-rewrite 子集在 `cw32000/thr30000` 下触发率太低，可以使用 runner 的 `--initial-context-mode reference`。它只向 agent 内部历史注入中性的 telecom 背景参考消息，不改 task JSON，也不暴露给 user simulator。这个模式用于制造接近真实长上下文 agent workload 的可压缩 B1 段。

注意 tau2 默认 `telecom` 的 `base` split 只有 114 个任务；actions>=6 筛选文件来自完整 `tasks.json`。运行筛选样本时通常需要加 `--task-split full`，否则不在 `base` split 的任务会被 tau2 runner 拒绝。

## Python 运行环境

本目录不要替换系统 `/usr/bin/python3`。tau2 的 `pyproject.toml` 声明需要 Python `>=3.12,<3.14`，所以这里提供本地 wrapper：

```bash
cd /root/tau2_telecom_sp_bench
./run_tau2.sh --help
```

`run_tau2.sh` 默认使用已有的 `/root/tau2-bench/.venv/bin/python`，该解释器当前是 Python 3.12.12。需要临时切换时可以设置：

```bash
TAU2_PYTHON=/path/to/python3.12 ./run_tau2.sh run --help
```

这只影响 `/root/tau2_telecom_sp_bench` 的运行方式，不修改 `/usr/bin/python3`，也不改变 `/root/bfcl_long_context_sp_bench` 和 `/root/agentbench_semi_prefill_bench` 的现有命令。

## 启动模型服务

按现有 split serve 方式启动 vLLM：

```bash
cd /root/vllm
./split_serve.sh -all
```

启动工具代理：

```bash
/usr/bin/python3 -m vllm_tool_proxy.vllm_tool_proxy.server \
  --backend-url http://10.10.111.43:8005 \
  --port 6003 \
  --tool-parser auto \
  --native-template
```

本机已验证可用的代理入口也可以写成：

```bash
/usr/bin/python3 -m vllm_tool_proxy.server \
  --backend-url http://127.0.0.1:8005 \
  --port 6003 \
  --tool-parser auto \
  --native-template
```

支持的模型键：

- `GLM-4-9B-0414`
- `Llama-3.3-70B-Instruct`
- `Qwen3-235B-A22B`

对应路径在 `sp_config.py` 中：

```text
/root/share/models/GLM-4-9B-0414
/root/share/models/Llama-3.3-70B-Instruct
/root/share/models/Qwen3-235B-A22B
```

## 运行示例

轻量 smoke test，先验证 zero-rewrite 子集和代理是否可用：

```bash
cd /root/tau2_telecom_sp_bench
./run_tau2.sh prepare-data --min-actions 6
./run_tau2.sh run \
  --mode compressed \
  --model Llama-3.3-70B-Instruct \
  --limit 1 \
  --initial-context-mode zero-rewrite \
  --auto-resume \
  --health-check
```

当前推荐的 MMS reference 长上下文运行方式：

```bash
./run_tau2.sh run \
  --mode compressed \
  --model Llama-3.3-70B-Instruct \
  --task-split full \
  --sample-ids-file data/telecom_mms_easy_none_user3_agent1_actions_4_6_ids.json \
  --limit 20 \
  --initial-context-mode reference \
  --target-initial-tokens 29800 \
  --context-window 32000 \
  --reserve-tokens 2000 \
  --keep-recent-tokens 2800 \
  --summary-max-tokens 1024 \
  --max-steps 40 \
  --auto-resume \
  --health-check \
  --run-name real_mms_easy_user3_agent1_n20_compressed_only_cw32000_thr30000_c12800_20260507_001
```

同一 run-name 下再次运行并带 `--auto-resume` 时，tau2 的 `results.json` 会跳过已完成样本。agent 侧还会在每个 step 写 `checkpoints/<sample>.json`，用于定位样本内部中断点和恢复调查。

如果只想做短验证，不要复用正式 run-name；另建一个新的结果目录名，例如：

```bash
./run_tau2.sh run \
  --mode compressed \
  --model Llama-3.3-70B-Instruct \
  --task-split full \
  --sample-ids-file data/telecom_mms_easy_none_user3_agent1_actions_4_6_ids.json \
  --limit 3 \
  --max-steps 12 \
  --initial-context-mode reference \
  --target-initial-tokens 29800 \
  --context-window 32000 \
  --reserve-tokens 2000 \
  --keep-recent-tokens 2800 \
  --auto-resume \
  --run-name real_mms_easy_user3_agent1_n3_short_cw32000_thr30000_c12800_$(date +%Y%m%d)_001
```

## 输出格式

每个 run 写到：

```text
results/<run_name>/
  tau2_results/results.json       # tau2 官方结果，可用于 benchmark 分数
  prompt_logs/<sample>.jsonl      # 每次 agent LLM 请求的完整 messages 和 message token
  traces/<sample>.json            # 每步摘要、tool calls、压缩次数
  abc_segments/<sample>.json      # 完整 ABC 段
  timing/<sample>.json            # full_prefill / semi_prefill / incremental 延迟记录
  checkpoints/<sample>.json       # agent 内部 step checkpoint
  run_config.json
  sample_ids.json
  analysis.json
  scores.json
```

ABC 保存字段：

```text
压缩前: A + B1 + C1
压缩后: A + B2 + C1
```

其中每段都保存：

- 完整 message 内容
- 段 token 数
- 逐 message token 数
- `B2_to_B1_ratio`
- `C1_tokens_after`
- `B2_plus_C1_tokens`
- 摘要生成耗时

## 分析

单独分析一个 run：

```bash
cd /root/tau2_telecom_sp_bench
./run_tau2.sh analyze results/telecom_llama_reference_turn40_smoke
```

当前推荐 run 的分析命令：

```bash
./run_tau2.sh analyze \
  results/real_mms_easy_user3_agent1_n20_compressed_only_cw32000_thr30000_c12800_20260507_001
```

分析输出包括：

- 每个任务轮数与 agent LLM 调用步数
- 每轮新增 token 数
- 工具调用次数
- 总上下文长度分布
- context churn ratio
- 压缩触发次数与 `P_compress_per_step = compressions / agent LLM steps`
- `P_compress_per_dialogue_message = compressions / (assistant + user messages)`
- `P_compress_per_tau2_message = compressions / all tau2 messages`
- C1 长度分布和各 P 口径下的 `P<=1/5` / `P<=1/6` 计数
- semi-prefill 次数和每次 token 长度
- tau2 benchmark `avg_reward` 与 `pass_hat_1`

时延拆分说明：当前 agent 使用 tau2/litellm 非流式调用，脚本能可靠记录每次请求总耗时、摘要生成耗时和分类标签。严格的 TTFT/decode 拆分需要 vLLM streaming 或 server 侧 telemetry；`analyze.py` 提供 `--decode-ms-per-token` 参数，可以把服务器测得的 decode 单 token 延迟代入，估算：

```text
prefill / semi-prefill / incremental / decode_est / summary_generation
```

## 断点续传

样本之间：tau2 官方 `results.json` checkpoint 已接入，使用 `--auto-resume` 可跳过已完成的 task/trial。

样本内部：agent 每次 LLM 调用后都会写 checkpoint，包含当前压缩后的 messages、turn、step、compression_count、previous_summary。tau2 官方 orchestrator 不提供完整 environment/user state 的 step-level resume，所以这里的样本内 checkpoint 主要用于定位中断点、保留 prompt/ABC 证据和决定是否重跑该样本；不删除旧输出。

## 推荐验证顺序

1. `prepare-data`，确认目标子集的 `selected_tasks`、family/persona/action histogram。
2. `py_compile` 静态检查。
3. `--limit 1 --initial-context-mode zero-rewrite` smoke test，确认 tau2 runner、proxy、tool path 可用。
4. 如果 zero-rewrite 不触发压缩，改用 `--initial-context-mode reference --target-initial-tokens 29800`。
5. `--limit 3 --max-steps 12/16` 短验证，检查 `[COMPRESS]`、ABC、timing、checkpoint 是否写入。
6. 对优先子集跑 `--limit 20 --max-steps 40` 或继续 `--auto-resume`。
7. 使用 `analyze` 和独立 analysis 目录检查 P、C1、semi-prefill、context accumulation 和 score。

## 已有分析产物

针对当前 MMS 子集，已有两套派生分析目录：

```text
results/real_mms_easy_user3_agent1_n20_compressed_only_cw32000_thr30000_c12800_20260507_001/analysis_20260507_unified
results/real_mms_easy_user3_agent1_n20_compressed_only_cw32000_thr30000_c12800_20260507_001/analysis_20260507_cw8000_reference
```

- `analysis_20260507_unified`：侧重三数据集统一口径，包含 per-sample workload、compression events、context accumulation、ABC composition。
- `analysis_20260507_cw8000_reference`：参考 `/root/semi_prefill_bench/results/cw8000_cw8000_rs500/analysis` 的报告结构，包含 `async_prefill_only_pie.png`、`context_length_over_turns.png`、sync/async 理论模型和 prefill-only 分解。

注意：这些目录是已有结果的离线派生分析，不是新的 benchmark run。不要删除旧结果目录；需要重跑时使用新的 run-name。
