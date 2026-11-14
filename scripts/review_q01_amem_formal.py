"""Review serial Q01 A-MEM run directories.

The reviewer consumes only durable checkpoint JSONL and cost summaries.  It
does not rerun the builder, infer missing histories, or treat a controlled
scripted run as a real-model result.  The output is therefore suitable for a
post-run gate after the formal LoCoMo and LongMemEval-S runners finish.  Arm
totals include the paired baseline setup for M1-M3, while preserving
maintenance-only and setup-only components for audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARMS = (
    "M0_full_rebuild",
    "M1_tombstone_filter",
    "M2_replay_repair_all",
    "M3_exact_read_set",
)

# BuildCost fields that can be added across the maintenance execution and the
# paired baseline setup.  The setup is deliberately recorded separately by
# ``run_q01_amem_paired`` because every non-M0 arm receives a fresh baseline
# adapter.  A reviewer must still expose both components and their full
# execution total; otherwise M1-M3 appear to be cheaper than the work that
# actually ran.
_COST_FIELDS = (
    "wall_time_ms",
    "input_events",
    "replayed_events",
    "prefix_replayed_events",
    "suffix_replayed_events",
    "checkpoint_restore_events",
    "checkpoint_restore_bytes",
    "retrieved_candidates",
    "generated_objects",
    "state_records",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "embedding_calls",
    "embedding_items",
    "index_updates",
    "memory_reads",
    "bytes_read",
    "bytes_written",
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sum_cost(costs: list[dict[str, Any]], field: str) -> int | float:
    """Add one numeric cost field while tolerating older/missing receipts."""

    values = [cost.get(field) for cost in costs]
    present = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not present:
        return 0
    total = sum(present)
    # Keep integer counters as integers in JSON while retaining fractional wall
    # times.  This also makes exact regression assertions straightforward.
    return int(total) if all(isinstance(value, int) for value in present) else total


def _cost_totals(costs: list[dict[str, Any]]) -> dict[str, int | float]:
    return {field: _sum_cost(costs, field) for field in _COST_FIELDS}


def _load_run(name: str, run: Path, expected: int | None) -> dict[str, Any]:
    instance_dir = run / "checkpoints" / "instances"
    rows: list[dict[str, Any]] = []
    if instance_dir.is_dir():
        for path in sorted(instance_dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                line = next((item for item in handle if item.strip()), "")
            if line:
                rows.append(json.loads(line))
    arm_summary: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row["arms"][arm] for row in rows if arm in row.get("arms", {})]
        # ``cost`` is the post-deletion maintenance execution.  In paired
        # real-model runs, M1-M3 also carry a fresh baseline build in
        # ``metadata.baseline_setup_cost``.  Preserve both components and
        # make the historical top-level fields represent the full work so the
        # review table cannot under-report those arms.
        costs = [row["cost"] for row in arm_rows]
        setup_costs = [
            row.get("metadata", {}).get("baseline_setup_cost", {})
            for row in arm_rows
        ]
        maintenance = _cost_totals(costs)
        baseline_setup = _cost_totals(
            [cost for cost in setup_costs if isinstance(cost, dict)]
        )
        full = {
            field: maintenance[field] + baseline_setup[field]
            for field in _COST_FIELDS
        }
        arm_summary[arm] = {
            "instances": len(arm_rows),
            "exact_count": sum(bool(row.get("exact_to_m0")) for row in arm_rows),
            "mismatched_instances": [
                rows[index].get("instance_id")
                for index, row in enumerate([row["arms"][arm] for row in rows if arm in row.get("arms", {})])
                if not row.get("exact_to_m0")
            ],
            # Backward-compatible top-level fields now report full paired
            # execution cost, including the setup build for M1-M3.
            **full,
            # Explicit component fields make the accounting auditable and
            # retain the incremental maintenance view used by older analyses.
            **{f"maintenance_{field}": value for field, value in maintenance.items()},
            **{f"baseline_setup_{field}": value for field, value in baseline_setup.items()},
            **{f"full_{field}": value for field, value in full.items()},
        }

    probes = []
    for row in rows:
        for arm in ARMS:
            probe = row.get("arms", {}).get(arm, {}).get("query_probe")
            if probe is None:
                continue
            probes.append(
                {
                    "instance_id": row.get("instance_id"),
                    "arm": arm,
                    "evidence_complete": bool(probe.get("evidence_complete", False)),
                    "deleted_source_exposed": bool(probe.get("deleted_source_exposed", False)),
                    "read_count": len(probe.get("read_set", [])),
                    "candidate_count": len(probe.get("candidate_ids", [])),
                    "llm_calls": int(probe.get("llm_calls", 0) or 0),
                    "input_tokens": int(probe.get("input_tokens", 0) or 0),
                    "output_tokens": int(probe.get("output_tokens", 0) or 0),
                    "query_cost_observed": bool(probe.get("query_cost_observed", False)),
                }
            )
    cost_summary_path = run / "cost_summary.json"
    cost_summary = json.loads(cost_summary_path.read_text(encoding="utf-8")) if cost_summary_path.exists() else None
    fingerprint_path = run / "checkpoints" / "run_fingerprint.json"
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8")) if fingerprint_path.exists() else None
    return {
        "run_path": str(run.resolve()),
        "expected_instances": expected,
        "instances": len(rows),
        "complete": expected is not None and len(rows) == expected,
        "instance_ids": [row.get("instance_id") for row in rows],
        "arm_summary": arm_summary,
        "query_probes": {
            "count": len(probes),
            "evidence_complete": sum(item["evidence_complete"] for item in probes),
            "deleted_source_exposed": sum(item["deleted_source_exposed"] for item in probes),
            "llm_calls": sum(item["llm_calls"] for item in probes),
            "input_tokens": sum(item["input_tokens"] for item in probes),
            "output_tokens": sum(item["output_tokens"] for item in probes),
            "query_cost_complete": bool(probes) and all(item["query_cost_observed"] for item in probes),
            "records": probes,
        },
        "fingerprint": fingerprint,
        "cost_summary": cost_summary,
        "cost_summary_sha256": _hash(cost_summary_path) if cost_summary_path.exists() else None,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Q01 formal A-MEM run review",
        "",
        f"- Evidence level: `{report['evidence_level']}`",
        f"- Decision gate: `{report['decision']['status']}`",
        "",
    ]
    for name, workload in report["workloads"].items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Instances: {workload['instances']}/{workload['expected_instances']} (`complete={workload['complete']}`)",
                f"- Query probes: {workload['query_probes']['evidence_complete']}/{workload['query_probes']['count']} evidence-complete; {workload['query_probes']['deleted_source_exposed']} deleted-source exposures; query cost complete={workload['query_probes']['query_cost_complete']}",
                "- Arm cost totals include paired baseline setup for M1-M3; JSON also retains maintenance and baseline_setup components.",
                "",
                "| Arm | Exact to M0 | Full replayed events | Full LLM calls | Full embedding calls | Full wall time (ms) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for arm in ARMS:
            value = workload["arm_summary"][arm]
            lines.append(
                f"| {arm} | {value['exact_count']}/{value['instances']} | {value['replayed_events']} | {value['llm_calls']} | {value['embedding_calls']} | {value['wall_time_ms']:.1f} |"
            )
        lines.append("")
        if workload.get("cost_summary"):
            lines.append("Cost scopes are preserved in the run's `cost_summary.json`; provider money cost must be supplied separately when applicable.")
            lines.append("")
    lines.extend(
        [
            "## Decision interpretation",
            "",
            "This review is a gate on execution evidence. Exact M0 equality does not establish natural occurrence, novelty or superiority. Missing histories, incomplete evidence, or a controlled scripted fingerprint keep the scientific decision unverified.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo-run", type=Path, required=True)
    parser.add_argument("--longmemeval-run", type=Path, required=True)
    parser.add_argument("--locomo-expected", type=int, default=10)
    parser.add_argument("--longmemeval-expected", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence-level", default="real_model_formal")
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "evidence_level": args.evidence_level,
        "workloads": {
            "locomo": _load_run("locomo", args.locomo_run, args.locomo_expected),
            "longmemeval_s": _load_run("longmemeval_s", args.longmemeval_run, args.longmemeval_expected),
        },
    }
    complete = all(value["complete"] for value in report["workloads"].values())
    report["decision"] = {
        "status": "REAL_WORKLOAD_COMPLETE" if complete and args.evidence_level.startswith("real_model_") else "INCOMPLETE_OR_CONTROLLED",
        "scientific_claim_supported": False,
        "reason": "review evidence is incomplete or controlled; apply Q01 keep/narrow/close gates after full real-model runs",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(args.out)
    print(args.out.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
