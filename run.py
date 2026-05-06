"""Runner for tau2 telecom semi-prefill compression experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import sp_config as CFG

for import_path in (CFG.TAU2_SRC, CFG.COMPRESSOR_ROOT, CFG.BENCH_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from dataset_rewrite import write_selection  # noqa: E402


REQUIRED_RUNTIME_MODULES = {
    "rich": "rich",
    "tabulate": "tabulate",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pandas": "pandas",
    "psutil": "psutil",
    "loguru": "loguru>=0.7.3",
    "docstring_parser": "docstring-parser>=0.16",
    "litellm": "litellm>=1.80.15,<1.82.7",
    "tenacity": "tenacity>=9.0.0",
    "deepdiff": "deepdiff>=8.4.2",
    "addict": "addict>=2.4.0",
    "yaml": "PyYAML>=6.0.2",
    "toml": "toml>=0.10.2",
    "dotenv": "python-dotenv>=1.0.0",
    "typer": "typer>=0.12.5",
    "requests": "requests>=2.31.0",
    "numpy": "numpy>=1.24.0",
    "httpx": "httpx>=0.24.0",
}


def model_path(model_key: str) -> str:
    return CFG.MODEL_REGISTRY[model_key]["model_path"]


def agent_llm_name(model_key: str) -> str:
    return f"openai/{model_path(model_key)}"


def load_sample_ids(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def check_python_runtime() -> dict:
    missing = [package for module, package in REQUIRED_RUNTIME_MODULES.items() if importlib.util.find_spec(module) is None]
    version_info = {
        "executable": sys.executable,
        "version": platform.python_version(),
        "ok": not missing,
        "missing_packages": missing,
    }
    if sys.version_info < (3, 12):
        version_info["version_warning"] = "tau2 pyproject declares requires-python >=3.12,<3.14; imports were validated here, but full benchmark runs should watch for syntax/runtime drift."
    return version_info


def check_health() -> dict:
    report = {"proxy": False, "swap": None, "python": check_python_runtime()}
    try:
        import urllib.request

        with urllib.request.urlopen(f"{CFG.PROXY_URL.rstrip('/')}/models", timeout=15) as response:
            payload = json.loads(response.read())
            report["proxy"] = True
            report["models"] = [item.get("id") for item in payload.get("data", [])]
    except Exception as exc:
        report["proxy_error"] = str(exc)
    try:
        result = subprocess.run(["free", "-m"], capture_output=True, text=True, check=False)
        for line in result.stdout.splitlines():
            if line.startswith("Swap:"):
                parts = line.split()
                report["swap"] = {"total_mb": int(parts[1]), "used_mb": int(parts[2]), "free_mb": int(parts[3])}
    except Exception as exc:
        report["swap_error"] = str(exc)
    return report


def write_run_meta(run_name: str, args, sample_ids: list[str]) -> None:
    dirs = CFG.run_dirs(run_name)
    CFG.ensure_run_dirs(dirs)
    payload = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "domain": "telecom",
        "mode": args.mode,
        "model": args.model,
        "model_path": model_path(args.model),
        "proxy_url": CFG.PROXY_URL,
        "vllm_backend_url": CFG.VLLM_BACKEND_URL,
        "sample_ids_file": str(args.sample_ids_file),
        "num_sample_ids": len(sample_ids),
        "context": {
            "context_window": args.context_window,
            "reserve_tokens": args.reserve_tokens,
            "threshold_tokens": args.context_window - args.reserve_tokens,
            "keep_recent_tokens": args.keep_recent_tokens,
            "summary_max_tokens": args.summary_max_tokens,
            "initial_context_mode": args.initial_context_mode,
            "target_initial_tokens": args.target_initial_tokens,
        },
        "targets": {
            "P_max_1_5": CFG.P_TARGET_1_5,
            "P_max_1_6": CFG.P_TARGET_1_6,
            "C1_min_tokens": CFG.C1_MIN_TOKENS,
            "C1_diagnostic_max_tokens": CFG.C1_DIAGNOSTIC_MAX_TOKENS,
        },
        "health_at_start": check_health() if args.health_check else None,
    }
    with open(dirs["config_file"], "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    with open(dirs["samples_file"], "w", encoding="utf-8") as handle:
        json.dump(sample_ids, handle, indent=2, ensure_ascii=False)


def run_mode(run_name: str, mode: str, args, sample_ids: list[str]):
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.batch import run_domain

    dirs = CFG.run_dirs(run_name)
    agent_name = "telecom_sp_baseline_agent" if mode == "baseline" else "telecom_sp_compressed_agent"
    llm_args_agent = {
        "temperature": args.temperature,
        "api_base": CFG.PROXY_URL,
        "api_key": CFG.API_KEY,
        "model_key": args.model,
        "run_root": str(dirs["base"]),
        "initial_context_mode": args.initial_context_mode,
        "target_initial_tokens": args.target_initial_tokens,
        "context_window": args.context_window,
        "reserve_tokens": args.reserve_tokens,
        "keep_recent_tokens": args.keep_recent_tokens,
        "summary_max_tokens": args.summary_max_tokens,
    }
    run_config = TextRunConfig(
        domain="telecom",
        agent=agent_name,
        llm_agent=agent_llm_name(args.model),
        llm_args_agent=llm_args_agent,
        llm_user=args.user_llm,
        llm_args_user=CFG.USER_LLM_ARGS,
        num_tasks=None,
        task_ids=sample_ids[: args.limit] if args.limit else sample_ids,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_concurrency=args.concurrency,
        seed=args.seed,
        save_to=str(dirs["tau2_results"]),
        auto_resume=args.auto_resume,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        timeout=args.timeout,
        log_level=args.log_level,
        verbose_logs=args.verbose_logs,
        hallucination_retries=0,
    )
    print(f"[run] {run_name} mode={mode} samples={len(run_config.task_ids)} out={dirs['base']}")
    return run_domain(run_config)


def cmd_prepare(args) -> None:
    report = write_selection(
        output=args.output,
        summary_out=args.summary_out,
        min_actions=args.min_actions,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_run(args) -> None:
    runtime = check_python_runtime()
    if runtime["missing_packages"]:
        install_hint = f" {sys.executable} -m pip install " + " ".join(f"'{package}'" for package in runtime["missing_packages"])
        raise RuntimeError(f"Missing Python packages for the active interpreter: {runtime['missing_packages']}. Install with:{install_hint}")

    from agent import register_agents
    from analyze import analyze_run

    register_agents()
    if not args.sample_ids_file.exists():
        print(f"[prepare] sample id file missing, creating {args.sample_ids_file}")
        write_selection(output=args.sample_ids_file, summary_out=CFG.DATASET_SUMMARY_FILE, min_actions=args.min_actions)
    sample_ids = load_sample_ids(args.sample_ids_file)
    if args.ids:
        requested = set(args.ids)
        sample_ids = [sample_id for sample_id in sample_ids if sample_id in requested]
    if not sample_ids:
        raise ValueError("No sample IDs selected")

    base_run_name = args.run_name or f"telecom_{args.model}_{args.mode}_cw{args.context_window}_thr{args.context_window - args.reserve_tokens}_{args.initial_context_mode}"
    modes = ["baseline", "compressed"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_name = base_run_name if len(modes) == 1 else f"{base_run_name}_{mode}"
        write_run_meta(run_name, args, sample_ids[: args.limit] if args.limit else sample_ids)
        run_mode(run_name, mode, args, sample_ids)
        if args.analyze:
            report = analyze_run(CFG.run_dirs(run_name)["base"], decode_ms_per_token=args.decode_ms_per_token)
            print(f"[analysis] saved {CFG.run_dirs(run_name)['analysis_file']}")
            print(f"[score] avg_reward={report['scores']['avg_reward']} pass_hat_1={report['scores']['pass_hat_1']}")


def cmd_analyze(args) -> None:
    from analyze import analyze_run

    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = (CFG.BENCH_ROOT / run_dir) if run_dir.parts and run_dir.parts[0] == "results" else (CFG.RESULTS_DIR / run_dir)
    report = analyze_run(run_dir, decode_ms_per_token=args.decode_ms_per_token)
    print(json.dumps({"scores": report["scores"], "workload_summary": report["workload"]["summary"]}, indent=2, ensure_ascii=False))


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["baseline", "compressed", "both"], default="compressed")
    parser.add_argument("--model", choices=list(CFG.MODEL_REGISTRY.keys()), default=CFG.DEFAULT_MODEL)
    parser.add_argument("--sample-ids-file", type=Path, default=CFG.DEFAULT_SAMPLE_IDS_FILE)
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-actions", type=int, default=CFG.MIN_ACTIONS)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-steps", type=int, default=CFG.DEFAULT_MAX_STEPS)
    parser.add_argument("--num-trials", type=int, default=CFG.DEFAULT_NUM_TRIALS)
    parser.add_argument("--concurrency", type=int, default=CFG.DEFAULT_CONCURRENCY)
    parser.add_argument("--seed", type=int, default=CFG.DEFAULT_SEED)
    parser.add_argument("--user-llm", default=CFG.USER_LLM)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--context-window", type=int, default=CFG.CONTEXT_WINDOW)
    parser.add_argument("--reserve-tokens", type=int, default=CFG.RESERVE_TOKENS)
    parser.add_argument("--keep-recent-tokens", type=int, default=CFG.KEEP_RECENT_TOKENS)
    parser.add_argument("--summary-max-tokens", type=int, default=CFG.SUMMARY_MAX_TOKENS)
    parser.add_argument("--initial-context-mode", choices=["zero-rewrite", "reference"], default=CFG.INITIAL_CONTEXT_MODE)
    parser.add_argument("--target-initial-tokens", type=int, default=CFG.TARGET_INITIAL_TOKENS)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=CFG.DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-delay", type=float, default=CFG.DEFAULT_RETRY_DELAY)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--health-check", action="store_true")
    parser.add_argument("--verbose-logs", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--analyze", action="store_true", default=True)
    parser.add_argument("--no-analyze", dest="analyze", action="store_false")
    parser.add_argument("--decode-ms-per-token", type=float, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="tau2 telecom compression and semi-prefill benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="write actions>=N zero-rewrite sample IDs")
    prepare.add_argument("--min-actions", type=int, default=CFG.MIN_ACTIONS)
    prepare.add_argument("--limit", type=int, default=None)
    prepare.add_argument("--output", type=Path, default=CFG.DEFAULT_SAMPLE_IDS_FILE)
    prepare.add_argument("--summary-out", type=Path, default=CFG.DATASET_SUMMARY_FILE)
    prepare.set_defaults(func=cmd_prepare)

    run_parser = subparsers.add_parser("run", help="run tau2-bench telecom samples")
    add_run_args(run_parser)
    run_parser.set_defaults(func=cmd_run)

    analyze_parser = subparsers.add_parser("analyze", help="analyze a completed or partial run")
    analyze_parser.add_argument("run_dir", type=Path)
    analyze_parser.add_argument("--decode-ms-per-token", type=float, default=0.0)
    analyze_parser.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    os.environ.setdefault("OPENAI_API_KEY", CFG.API_KEY)
    args.func(args)


if __name__ == "__main__":
    main()
