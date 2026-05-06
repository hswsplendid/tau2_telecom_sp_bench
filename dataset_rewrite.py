"""Build a zero-rewrite telecom task subset for compression experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sp_config as CFG


def action_count(task: dict) -> int:
    criteria = task.get("evaluation_criteria") or {}
    return len(criteria.get("actions") or [])


def load_tasks(path: Path = CFG.TELECOM_TASKS_FILE) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def select_tasks(tasks: list[dict], min_actions: int = CFG.MIN_ACTIONS) -> list[dict]:
    return [task for task in tasks if action_count(task) >= min_actions]


def summarize(tasks: list[dict], selected: list[dict], min_actions: int) -> dict:
    histogram = Counter(action_count(task) for task in tasks)
    selected_histogram = Counter(action_count(task) for task in selected)
    return {
        "source_tasks_file": str(CFG.TELECOM_TASKS_FILE),
        "transformation": "zero_rewrite_id_filter_only",
        "min_actions": min_actions,
        "total_tasks": len(tasks),
        "selected_tasks": len(selected),
        "action_count_histogram": dict(sorted(histogram.items())),
        "selected_action_count_histogram": dict(sorted(selected_histogram.items())),
        "first_selected_ids": [task["id"] for task in selected[:20]],
        "note": (
            "The task JSON is not modified. tau2-bench still loads the original "
            "telecom dataset; this file only records task IDs with enough expected actions."
        ),
    }


def write_selection(
    output: Path = CFG.DEFAULT_SAMPLE_IDS_FILE,
    summary_out: Path = CFG.DATASET_SUMMARY_FILE,
    min_actions: int = CFG.MIN_ACTIONS,
    limit: int | None = None,
) -> dict:
    tasks = load_tasks()
    selected = select_tasks(tasks, min_actions=min_actions)
    if limit is not None:
        selected = selected[:limit]

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as handle:
        json.dump([task["id"] for task in selected], handle, indent=2, ensure_ascii=False)

    report = summarize(tasks, selected, min_actions=min_actions)
    with open(summary_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select tau2 telecom tasks by expected action count.")
    parser.add_argument("--min-actions", type=int, default=CFG.MIN_ACTIONS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=CFG.DEFAULT_SAMPLE_IDS_FILE)
    parser.add_argument("--summary-out", type=Path, default=CFG.DATASET_SUMMARY_FILE)
    args = parser.parse_args()

    report = write_selection(
        output=args.output,
        summary_out=args.summary_out,
        min_actions=args.min_actions,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
