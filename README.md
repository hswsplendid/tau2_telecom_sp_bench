# tau2 telecom semi-prefill compression benchmark

这个目录是针对 `tau2-bench` 的 `telecom` 类别新增的压缩与 semi-prefill 测试方案。它参考了 `/root/agentbench_semi_prefill_bench` 的 prompt/ABC/timing 保存形式，以及 `/root/bfcl_long_context_sp_bench` 的 run-name 隔离、样本筛选和分析指标。

## 目标

- 数据集：`/root/tau2-bench/data/tau2/domains/telecom/tasks.json`
- 默认样本选择：`evaluation_criteria.actions >= 6`
- 默认压缩参数：`cw=32000`，`reserve=2000`，阈值 `thr=30000`
- C1 最近保留段：默认 `2800` tokens，目标范围 `2000-3000`，允许诊断中看到 `>3000`
- 目标触发率：单样本内部压缩触发次数 / agent LLM 调用步数 `P <= 1/5` 或 `P <= 1/6`
- 输出：完整 prompt、trace、timing、checkpoint、压缩前后 ABC 段和逐 message token 记录

## 目录结构

```text
tau2_telecom_sp_bench/
  sp_config.py          # 路径、模型、压缩参数、输出目录
  dataset_rewrite.py    # actions>=6 的 zero-rewrite 样本 ID 选择
  agent.py              # tau2 agent 注册、压缩、ABC/prompt/timing/checkpoint 记录
  run.py                # prepare-data / run / analyze 入口
  analyze.py            # workload、semi-prefill、score 分析
  data/                 # 样本 ID 和数据筛选摘要
  results/              # 新实验输出目录，不覆盖旧结果
```

## 数据集改造策略

第一阶段采用 **0 改造**：不修改 tau2 原始任务 JSON，只筛选 actions 数量足够多的 telecom 样本。当前统计结果是：

```text
telecom total tasks: 2285
actions >= 6: 1275
```

生成筛选文件：

```bash
cd /root/tau2_telecom_sp_bench
/usr/bin/python3 dataset_rewrite.py --min-actions 6
```

这会写入：

- `data/telecom_actions_ge6_ids.json`
- `data/telecom_actions_ge6_summary.json`

如果 0 改造样本在 `cw32000/thr30000` 下触发率太低，可以使用 runner 的 `--initial-context-mode reference`。它只向 agent 内部历史注入中性的 telecom 背景参考消息，不改 task JSON，也不暴露给 user simulator。这个模式用于制造接近真实长上下文 agent workload 的可压缩 B1 段。

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

轻量 smoke test，先验证 0 改造 actions>=6 子集：

```bash
cd /root/tau2_telecom_sp_bench
/usr/bin/python3 run.py prepare-data --min-actions 6
/usr/bin/python3 run.py run \
  --mode compressed \
  --model Llama-3.3-70B-Instruct \
  --limit 1 \
  --initial-context-mode zero-rewrite \
  --auto-resume \
  --health-check
```

如果需要让样本更稳定地进入 `thr=30000` 附近，使用 reference 初始上下文：

```bash
/usr/bin/python3 run.py run \
  --mode compressed \
  --model Llama-3.3-70B-Instruct \
  --limit 3 \
  --initial-context-mode reference \
  --target-initial-tokens 26000 \
  --context-window 32000 \
  --reserve-tokens 2000 \
  --keep-recent-tokens 2800 \
  --auto-resume \
  --run-name telecom_llama_reference_turn40_smoke
```

同一 run-name 下再次运行并带 `--auto-resume` 时，tau2 的 `results.json` 会跳过已完成样本。agent 侧还会在每个 step 写 `checkpoints/<sample>.json`，用于定位样本内部中断点和恢复调查。

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
/usr/bin/python3 run.py analyze results/telecom_llama_reference_turn40_smoke
```

分析输出包括：

- 每个任务轮数与 agent LLM 调用步数
- 每轮新增 token 数
- 工具调用次数
- 总上下文长度分布
- context churn ratio
- 压缩触发次数与 `P = compressions / steps`
- C1 长度分布和 `P<=1/5` / `P<=1/6` 计数
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

1. `prepare-data`，确认 actions>=6 样本数量。
2. `py_compile` 静态检查。
3. `--limit 1 --initial-context-mode zero-rewrite` smoke test。
4. 如果 0 改造不触发压缩，改用 `--initial-context-mode reference --target-initial-tokens 26000`。
5. `--limit 3` 验证 P、C1 和分数文件。
6. 再扩大到完整 actions>=6 子集。
