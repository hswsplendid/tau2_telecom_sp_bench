"""Analyze tau2 telecom compression workload and benchmark scores."""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from pathlib import Path

import sp_config as CFG

if str(CFG.TAU2_SRC) not in sys.path:
    sys.path.insert(0, str(CFG.TAU2_SRC))


def load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def describe(values: list[float | int | None]) -> dict:
    clean = [value for value in values if value is not None]
    if not clean:
        return {}
    report = {
        "min": min(clean),
        "max": max(clean),
        "mean": round(stats.mean(clean), 4),
        "median": round(stats.median(clean), 4),
    }
    if len(clean) >= 4:
        quartiles = stats.quantiles(clean, n=4)
        report["p25"] = round(quartiles[0], 4)
        report["p75"] = round(quartiles[2], 4)
    return report


def list_sample_ids(run_dir: Path) -> list[str]:
    timing_dir = run_dir / "timing"
    if not timing_dir.exists():
        return []
    return sorted(path.stem for path in timing_dir.glob("*.json"))


def load_samples(run_dir: Path) -> dict:
    samples = {}
    for sample_id in list_sample_ids(run_dir):
        samples[sample_id] = {
            "timing": load_json(run_dir / "timing" / f"{sample_id}.json", []),
            "trace": load_json(run_dir / "traces" / f"{sample_id}.json", {}),
            "abc": load_json(run_dir / "abc_segments" / f"{sample_id}.json", []),
        }
    return samples


def extract_scores(run_dir: Path) -> dict:
    results_path = run_dir / "tau2_results" / "results.json"
    data = load_json(results_path, {})
    simulations = data.get("simulations", [])
    rows = []
    for simulation in simulations:
        reward_info = simulation.get("reward_info") or {}
        reward = reward_info.get("reward")
        rows.append(
            {
                "simulation_id": simulation.get("id"),
                "task_id": simulation.get("task_id"),
                "reward": reward,
                "success": reward is not None and reward >= 1.0 - 1e-6,
                "termination_reason": simulation.get("termination_reason"),
                "duration_s": simulation.get("duration"),
            }
        )
    rewards = [row["reward"] for row in rows if row["reward"] is not None]
    return {
        "results_path": str(results_path),
        "num_simulations": len(rows),
        "avg_reward": round(sum(rewards) / len(rewards), 6) if rewards else None,
        "pass_hat_1": round(sum(1 for row in rows if row["success"]) / len(rows), 6) if rows else None,
        "termination_reasons": _count_values(row["termination_reason"] for row in rows),
        "per_simulation": rows,
    }


def _count_values(values) -> dict:
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def analyze_workload(samples: dict) -> dict:
    per_sample = []
    all_prompt_tokens = []
    all_output_tokens = []
    all_churn = []
    all_context_deltas = []
    semi_prefill_lengths = []

    for sample_id, payload in samples.items():
        timing = payload["timing"]
        trace = payload["trace"]
        abc_events = payload["abc"]
        if not timing:
            continue

        prompt_tokens = [row.get("prompt_tokens", 0) for row in timing]
        output_tokens = [row.get("output_tokens", 0) for row in timing]
        deltas = [prompt_tokens[index] - prompt_tokens[index - 1] for index in range(1, len(prompt_tokens))]
        all_prompt_tokens.extend(prompt_tokens)
        all_output_tokens.extend(output_tokens)
        all_context_deltas.extend(deltas)

        churn = []
        for event in abc_events:
            before = event.get("pre_prompt_tokens") or 0
            after = event.get("post_prompt_tokens") or 0
            if before > 0:
                churn.append((before - after) / before)
        all_churn.extend(churn)

        semi_events = [row for row in timing if row.get("classification") == "semi_prefill"]
        lengths = [row.get("semi_prefill_tokens", 0) for row in semi_events]
        semi_prefill_lengths.extend(lengths)
        compressions = len(abc_events)
        steps = len(timing)
        compression_rate = compressions / max(steps, 1)

        per_sample.append(
            {
                "sample_id": sample_id,
                "task_id": trace.get("task_id"),
                "turns": trace.get("total_turns", 0),
                "steps": steps,
                "tool_calls": trace.get("tool_calls", 0),
                "compressions": compressions,
                "P_compress_per_step": round(compression_rate, 6),
                "P_le_1_5": 0 < compression_rate <= CFG.P_TARGET_1_5,
                "P_le_1_6": 0 < compression_rate <= CFG.P_TARGET_1_6,
                "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else 0,
                "mean_new_tokens_per_step": round(stats.mean(output_tokens), 4) if output_tokens else 0,
                "context_churn_ratio_mean": round(stats.mean(churn), 6) if churn else None,
                "semi_prefill_count": len(semi_events),
                "semi_prefill_lengths": lengths,
                "min_c1": min((event.get("C1_tokens_after", 0) for event in abc_events), default=None),
                "max_c1": max((event.get("C1_tokens_after", 0) for event in abc_events), default=None),
                "reference_context_tokens": trace.get("reference_context_tokens", 0),
            }
        )

    return {
        "summary": {
            "n_samples_with_timing": len(per_sample),
            "turns_per_task": describe([row["turns"] for row in per_sample]),
            "steps_per_task": describe([row["steps"] for row in per_sample]),
            "tool_calls_per_task": describe([row["tool_calls"] for row in per_sample]),
            "compressions_per_task": describe([row["compressions"] for row in per_sample]),
            "compression_rate_P": describe([row["P_compress_per_step"] for row in per_sample]),
            "samples_P_le_1_5": sum(1 for row in per_sample if row["P_le_1_5"]),
            "samples_P_le_1_6": sum(1 for row in per_sample if row["P_le_1_6"]),
            "samples_P_zero": sum(1 for row in per_sample if row["P_compress_per_step"] == 0),
            "context_length_distribution": describe(all_prompt_tokens),
            "context_delta_distribution": describe(all_context_deltas),
            "new_tokens_per_step": describe(all_output_tokens),
            "context_churn_ratio": describe(all_churn),
            "semi_prefill_lengths": describe(semi_prefill_lengths),
            "semi_prefill_events": len(semi_prefill_lengths),
            "samples_C1_lte_2000": sum(1 for row in per_sample if row["min_c1"] is not None and row["min_c1"] <= CFG.C1_MIN_TOKENS),
            "samples_C1_gt_3000_diagnostic": sum(1 for row in per_sample if row["max_c1"] is not None and row["max_c1"] > CFG.C1_DIAGNOSTIC_MAX_TOKENS),
        },
        "per_sample": per_sample,
    }


def analyze_time(samples: dict, decode_ms_per_token: float = 0.0) -> dict:
    aggregate = {
        "prefill_ms": 0.0,
        "semi_prefill_ms": 0.0,
        "summary_generation_ms": 0.0,
        "incremental_ms": 0.0,
        "decode_ms_est": 0.0,
    }
    per_sample = []
    for sample_id, payload in samples.items():
        timing = payload["timing"]
        abc_events = payload["abc"]
        prefill = semi_prefill = incremental = decode = 0.0
        summary_ms = sum((event.get("summary_generation_time_s") or 0.0) * 1000 for event in abc_events)
        for row in timing:
            decode_est = (row.get("output_tokens") or 0) * decode_ms_per_token
            request_ms = max((row.get("total_ms") or 0.0) - decode_est, 0.0)
            decode += decode_est
            if row.get("classification") == "full_prefill":
                prefill += request_ms
            elif row.get("classification") == "semi_prefill":
                semi_prefill += request_ms
            else:
                incremental += request_ms
        total = prefill + semi_prefill + incremental + decode + summary_ms
        row = {
            "sample_id": sample_id,
            "prefill_ms": round(prefill, 3),
            "semi_prefill_ms": round(semi_prefill, 3),
            "summary_generation_ms": round(summary_ms, 3),
            "incremental_ms": round(incremental, 3),
            "decode_ms_est": round(decode, 3),
            "total_ms": round(total, 3),
            "pct": _pct_breakdown(prefill, semi_prefill, summary_ms, incremental, decode),
        }
        per_sample.append(row)
        aggregate["prefill_ms"] += prefill
        aggregate["semi_prefill_ms"] += semi_prefill
        aggregate["summary_generation_ms"] += summary_ms
        aggregate["incremental_ms"] += incremental
        aggregate["decode_ms_est"] += decode
    total_ms = sum(aggregate.values())
    aggregate["total_ms"] = round(total_ms, 3)
    aggregate["pct"] = _pct_breakdown(
        aggregate["prefill_ms"],
        aggregate["semi_prefill_ms"],
        aggregate["summary_generation_ms"],
        aggregate["incremental_ms"],
        aggregate["decode_ms_est"],
    )
    aggregate["timing_method"] = (
        "Agent request latency is measured by non-streaming litellm total time. "
        "decode_ms_est uses --decode-ms-per-token; set it from server-side telemetry for paper-grade split."
    )
    return {"aggregate": aggregate, "per_sample": per_sample}


def _pct_breakdown(prefill: float, semi_prefill: float, summary_ms: float, incremental: float, decode: float) -> dict:
    total = prefill + semi_prefill + summary_ms + incremental + decode
    return {
        "prefill": round(prefill / total * 100, 4) if total else 0.0,
        "semi_prefill": round((semi_prefill + summary_ms) / total * 100, 4) if total else 0.0,
        "semi_prefill_request": round(semi_prefill / total * 100, 4) if total else 0.0,
        "summary_generation": round(summary_ms / total * 100, 4) if total else 0.0,
        "incremental": round(incremental / total * 100, 4) if total else 0.0,
        "decode_est": round(decode / total * 100, 4) if total else 0.0,
    }


def analyze_run(run_dir: Path, decode_ms_per_token: float = 0.0) -> dict:
    samples = load_samples(run_dir)
    report = {
        "run_dir": str(run_dir),
        "scores": extract_scores(run_dir),
        "workload": analyze_workload(samples),
        "time_breakdown": analyze_time(samples, decode_ms_per_token=decode_ms_per_token),
    }
    output = run_dir / "analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    scores_path = run_dir / "scores.json"
    with open(scores_path, "w", encoding="utf-8") as handle:
        json.dump(report["scores"], handle, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tau2 telecom semi-prefill benchmark run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--decode-ms-per-token", type=float, default=0.0)
    args = parser.parse_args()

    report = analyze_run(args.run_dir, decode_ms_per_token=args.decode_ms_per_token)
    workload = report["workload"]["summary"]
    scores = report["scores"]
    timing = report["time_breakdown"]["aggregate"]
    print(f"run_dir: {report['run_dir']}")
    print(f"simulations: {scores['num_simulations']}")
    print(f"avg_reward: {scores['avg_reward']}")
    print(f"pass_hat_1: {scores['pass_hat_1']}")
    print(f"samples with timing: {workload['n_samples_with_timing']}")
    print(f"P<=1/5: {workload['samples_P_le_1_5']}")
    print(f"P<=1/6: {workload['samples_P_le_1_6']}")
    print(f"P=0: {workload['samples_P_zero']}")
    print(f"semi-prefill pct: {timing['pct']['semi_prefill']}%")
    print(f"saved: {args.run_dir / 'analysis.json'}")


if __name__ == "__main__":
    main()
