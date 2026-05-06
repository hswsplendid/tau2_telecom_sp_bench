"""Configuration for tau2 telecom semi-prefill compression benchmark."""

from __future__ import annotations

import os
from pathlib import Path


BENCH_ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = BENCH_ROOT / "results"
DATA_DIR = BENCH_ROOT / "data"

TAU2_ROOT = Path("/root/tau2-bench")
TAU2_SRC = TAU2_ROOT / "src"
TAU2_TELECOM_DIR = TAU2_ROOT / "data" / "tau2" / "domains" / "telecom"
TELECOM_TASKS_FILE = TAU2_TELECOM_DIR / "tasks.json"
COMPRESSOR_ROOT = Path("/root/bfcl_compression_bench")


MODEL_REGISTRY = {
    "GLM-4-9B-0414": {
        "model_path": "/root/share/models/GLM-4-9B-0414",
        "tokenizer_path": "/root/share/models/GLM-4-9B-0414",
    },
    "Llama-3.3-70B-Instruct": {
        "model_path": "/root/share/models/Llama-3.3-70B-Instruct",
        "tokenizer_path": "/root/share/models/Llama-3.3-70B-Instruct",
    },
    "Qwen3-235B-A22B": {
        "model_path": "/root/share/models/Qwen3-235B-A22B",
        "tokenizer_path": "/root/share/models/Qwen3-235B-A22B",
    },
}

DEFAULT_MODEL = "Llama-3.3-70B-Instruct"


# vLLM is expected to be started by split_serve.sh and exposed through
# vllm_tool_proxy so tau2 tool calls use the native template path.
VLLM_BACKEND_URL = os.environ.get("VLLM_BACKEND_URL", "http://10.10.111.43:8005")
PROXY_URL = os.environ.get("PROXY_URL", "http://localhost:6003/v1")
API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")


# User simulator and optional NL evaluator. tau2-bench can use hosted models,
# or another OpenAI-compatible model if the caller supplies --user-llm.
USER_LLM = os.environ.get("USER_LLM", "gpt-4.1")
USER_LLM_ARGS = {"temperature": 0.0}
EVAL_LLM = os.environ.get("EVAL_LLM", "gpt-4.1")


# Compression preset requested by the experiment.
CONTEXT_WINDOW = 32000
RESERVE_TOKENS = 2000
THRESHOLD_TOKENS = CONTEXT_WINDOW - RESERVE_TOKENS
KEEP_RECENT_TOKENS = 2800
SUMMARY_MAX_TOKENS = 1024
QUALITY_GUARD_ENABLED = False
QUALITY_GUARD_MAX_RETRIES = 0
USE_STRUCTURED_INSTRUCTIONS = True
PRESERVED_RECENT_TURNS = 1
KEEP_RECENT_TURNS = 1


# Dataset selection. This is a zero-rewrite subset: task JSON is unchanged;
# only task IDs are selected for tau2's built-in task loader.
MIN_ACTIONS = 6
DEFAULT_SAMPLE_IDS_FILE = DATA_DIR / "telecom_actions_ge6_ids.json"
DATASET_SUMMARY_FILE = DATA_DIR / "telecom_actions_ge6_summary.json"


# Optional workload shaping. "zero-rewrite" leaves task data and initial agent
# history untouched. "reference" injects neutral, compressible reference context
# into the agent history only; it does not modify task JSON or the user simulator.
INITIAL_CONTEXT_MODE = os.environ.get("TAU2_INITIAL_CONTEXT_MODE", "zero-rewrite")
TARGET_INITIAL_TOKENS = int(os.environ.get("TAU2_TARGET_INITIAL_TOKENS", "26000"))
REFERENCE_CHUNK_TOKENS = 900
INCLUDE_TASK_TICKET = os.environ.get("TAU2_INCLUDE_TASK_TICKET", "1") not in {"0", "false", "False"}
STEPWISE_TECH_SUPPORT = os.environ.get("TAU2_STEPWISE_TECH_SUPPORT", "1") not in {"0", "false", "False"}


DEFAULT_MAX_STEPS = 40
DEFAULT_NUM_TRIALS = 1
DEFAULT_CONCURRENCY = 1
DEFAULT_SEED = 300
DEFAULT_MAX_RETRIES = 0
DEFAULT_RETRY_DELAY = 5.0


P_TARGET_1_5 = 1 / 5
P_TARGET_1_6 = 1 / 6
C1_MIN_TOKENS = 2000
C1_DIAGNOSTIC_MAX_TOKENS = 3000


def run_dirs(run_name: str) -> dict[str, Path]:
    base = RESULTS_DIR / run_name
    return {
        "base": base,
        "tau2_results": base / "tau2_results",
        "prompt_logs": base / "prompt_logs",
        "traces": base / "traces",
        "abc": base / "abc_segments",
        "timing": base / "timing",
        "checkpoints": base / "checkpoints",
        "config_file": base / "run_config.json",
        "samples_file": base / "sample_ids.json",
        "summary_file": base / "summary.json",
        "analysis_file": base / "analysis.json",
        "scores_file": base / "scores.json",
    }


def ensure_run_dirs(dirs: dict[str, Path]) -> None:
    for key, path in dirs.items():
        if key.endswith("_file"):
            continue
        path.mkdir(parents=True, exist_ok=True)
