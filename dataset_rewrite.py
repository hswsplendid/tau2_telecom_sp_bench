"""Build a zero-rewrite telecom task subset for compression experiments."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import sp_config as CFG


def action_count(task: dict) -> int:
    criteria = task.get("evaluation_criteria") or {}
    return len(criteria.get("actions") or [])


def action_requestor_count(task: dict, requestor: str) -> int:
    criteria = task.get("evaluation_criteria") or {}
    return sum(1 for action in criteria.get("actions") or [] if action.get("requestor", "assistant") == requestor)


def action_names(task: dict) -> set[str]:
    criteria = task.get("evaluation_criteria") or {}
    return {action.get("name") for action in criteria.get("actions") or [] if action.get("name")}


def task_family(task: dict) -> str:
    match = re.match(r"\[([^\]]+)\]", task.get("id", ""))
    return match.group(1) if match else task.get("id", "").split("|", 1)[0]


def task_persona(task: dict) -> str:
    match = re.search(r"\[PERSONA:([^\]]+)\]$", task.get("id", ""))
    return match.group(1) if match else "unknown"


def has_phone_in_ticket(task: dict) -> bool:
    return "phone number" in str(task.get("ticket") or "").lower()


def load_tasks(path: Path = CFG.TELECOM_TASKS_FILE) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def select_tasks(
    tasks: list[dict],
    min_actions: int = CFG.MIN_ACTIONS,
    *,
    max_actions: int | None = None,
    min_user_actions: int = 0,
    min_assistant_actions: int = 0,
    families: set[str] | None = None,
    exclude_families: set[str] | None = None,
    personas: set[str] | None = None,
    exclude_personas: set[str] | None = None,
    include_actions: set[str] | None = None,
    exclude_actions: set[str] | None = None,
    include_id_terms: set[str] | None = None,
    exclude_id_terms: set[str] | None = None,
    require_ticket_phone: bool = False,
    sort_by_actions_desc: bool = False,
    sort_by_actions_asc: bool = False,
    limit_per_family: int | None = None,
) -> list[dict]:
    selected = []
    per_family = Counter()
    candidates = list(tasks)
    if sort_by_actions_asc and sort_by_actions_desc:
        raise ValueError("Only one of sort_by_actions_asc or sort_by_actions_desc can be set")
    if sort_by_actions_asc:
        candidates.sort(key=lambda task: (action_count(task), task_family(task), task.get("id", "")))
    if sort_by_actions_desc:
        candidates.sort(key=lambda task: (-action_count(task), task_family(task), task.get("id", "")))
    for task in candidates:
        family = task_family(task)
        persona = task_persona(task)
        if families is not None and family not in families:
            continue
        if exclude_families is not None and family in exclude_families:
            continue
        if personas is not None and persona not in personas:
            continue
        if exclude_personas is not None and persona in exclude_personas:
            continue
        names = action_names(task)
        if include_actions is not None and not include_actions.issubset(names):
            continue
        if exclude_actions is not None and names.intersection(exclude_actions):
            continue
        task_id = task.get("id", "")
        if include_id_terms is not None and not all(term in task_id for term in include_id_terms):
            continue
        if exclude_id_terms is not None and any(term in task_id for term in exclude_id_terms):
            continue
        if action_count(task) < min_actions:
            continue
        if max_actions is not None and action_count(task) > max_actions:
            continue
        if action_requestor_count(task, "user") < min_user_actions:
            continue
        if action_requestor_count(task, "assistant") < min_assistant_actions:
            continue
        if require_ticket_phone and not has_phone_in_ticket(task):
            continue
        if limit_per_family is not None and per_family[family] >= limit_per_family:
            continue
        selected.append(task)
        per_family[family] += 1
    return selected


def summarize(tasks: list[dict], selected: list[dict], min_actions: int, filters: dict | None = None) -> dict:
    histogram = Counter(action_count(task) for task in tasks)
    selected_histogram = Counter(action_count(task) for task in selected)
    family_histogram = Counter(task_family(task) for task in selected)
    persona_histogram = Counter(task_persona(task) for task in selected)
    return {
        "source_tasks_file": str(CFG.TELECOM_TASKS_FILE),
        "transformation": "zero_rewrite_id_filter_only",
        "min_actions": min_actions,
        "filters": filters or {},
        "total_tasks": len(tasks),
        "selected_tasks": len(selected),
        "action_count_histogram": dict(sorted(histogram.items())),
        "selected_action_count_histogram": dict(sorted(selected_histogram.items())),
        "selected_family_histogram": dict(sorted(family_histogram.items())),
        "selected_persona_histogram": dict(sorted(persona_histogram.items())),
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
    max_actions: int | None = None,
    min_user_actions: int = 0,
    min_assistant_actions: int = 0,
    families: set[str] | None = None,
    exclude_families: set[str] | None = None,
    personas: set[str] | None = None,
    exclude_personas: set[str] | None = None,
    include_actions: set[str] | None = None,
    exclude_actions: set[str] | None = None,
    include_id_terms: set[str] | None = None,
    exclude_id_terms: set[str] | None = None,
    require_ticket_phone: bool = False,
    sort_by_actions_desc: bool = False,
    sort_by_actions_asc: bool = False,
    limit_per_family: int | None = None,
) -> dict:
    tasks = load_tasks()
    selected = select_tasks(
        tasks,
        min_actions=min_actions,
        max_actions=max_actions,
        min_user_actions=min_user_actions,
        min_assistant_actions=min_assistant_actions,
        families=families,
        exclude_families=exclude_families,
        personas=personas,
        exclude_personas=exclude_personas,
        include_actions=include_actions,
        exclude_actions=exclude_actions,
        include_id_terms=include_id_terms,
        exclude_id_terms=exclude_id_terms,
        require_ticket_phone=require_ticket_phone,
        sort_by_actions_desc=sort_by_actions_desc,
        sort_by_actions_asc=sort_by_actions_asc,
        limit_per_family=limit_per_family,
    )
    if limit is not None:
        selected = selected[:limit]

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as handle:
        json.dump([task["id"] for task in selected], handle, indent=2, ensure_ascii=False)

    filters = {
        "max_actions": max_actions,
        "min_user_actions": min_user_actions,
        "min_assistant_actions": min_assistant_actions,
        "families": sorted(families) if families else None,
        "exclude_families": sorted(exclude_families) if exclude_families else None,
        "personas": sorted(personas) if personas else None,
        "exclude_personas": sorted(exclude_personas) if exclude_personas else None,
        "include_actions": sorted(include_actions) if include_actions else None,
        "exclude_actions": sorted(exclude_actions) if exclude_actions else None,
        "include_id_terms": sorted(include_id_terms) if include_id_terms else None,
        "exclude_id_terms": sorted(exclude_id_terms) if exclude_id_terms else None,
        "require_ticket_phone": require_ticket_phone,
        "sort_by_actions_desc": sort_by_actions_desc,
        "sort_by_actions_asc": sort_by_actions_asc,
        "limit_per_family": limit_per_family,
        "limit": limit,
    }
    report = summarize(tasks, selected, min_actions=min_actions, filters=filters)
    with open(summary_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select tau2 telecom tasks by expected action count.")
    parser.add_argument("--min-actions", type=int, default=CFG.MIN_ACTIONS)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-user-actions", type=int, default=0)
    parser.add_argument("--min-assistant-actions", type=int, default=0)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--exclude-families", nargs="+", default=None)
    parser.add_argument("--personas", nargs="+", default=None)
    parser.add_argument("--exclude-personas", nargs="+", default=None)
    parser.add_argument("--include-actions", nargs="+", default=None)
    parser.add_argument("--exclude-actions", nargs="+", default=None)
    parser.add_argument("--include-id-terms", nargs="+", default=None)
    parser.add_argument("--exclude-id-terms", nargs="+", default=None)
    parser.add_argument("--require-ticket-phone", action="store_true")
    parser.add_argument("--sort-by-actions-desc", action="store_true")
    parser.add_argument("--sort-by-actions-asc", action="store_true")
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--output", type=Path, default=CFG.DEFAULT_SAMPLE_IDS_FILE)
    parser.add_argument("--summary-out", type=Path, default=CFG.DATASET_SUMMARY_FILE)
    args = parser.parse_args()

    report = write_selection(
        output=args.output,
        summary_out=args.summary_out,
        min_actions=args.min_actions,
        max_actions=args.max_actions,
        limit=args.limit,
        min_user_actions=args.min_user_actions,
        min_assistant_actions=args.min_assistant_actions,
        families=set(args.families) if args.families else None,
        exclude_families=set(args.exclude_families) if args.exclude_families else None,
        personas=set(args.personas) if args.personas else None,
        exclude_personas=set(args.exclude_personas) if args.exclude_personas else None,
        include_actions=set(args.include_actions) if args.include_actions else None,
        exclude_actions=set(args.exclude_actions) if args.exclude_actions else None,
        include_id_terms=set(args.include_id_terms) if args.include_id_terms else None,
        exclude_id_terms=set(args.exclude_id_terms) if args.exclude_id_terms else None,
        require_ticket_phone=args.require_ticket_phone,
        sort_by_actions_desc=args.sort_by_actions_desc,
        sort_by_actions_asc=args.sort_by_actions_asc,
        limit_per_family=args.limit_per_family,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
