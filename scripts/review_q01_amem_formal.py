"""Review serial Q01 A-MEM run directories.

The reviewer consumes only durable checkpoint JSONL and cost summaries.  It
does not rerun the builder, infer missing histories, or treat a controlled
scripted run as a real-model result.  The output is therefore suitable for a
post-run gate after the formal LoCoMo and LongMemEval-S runners finish. Current
paired rows report arm-local work; shared baseline and M0-repeat components are
read once from the lifecycle cost ledger. Legacy per-arm setup receipts remain
accepted for backwards-compatible audit.
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
# paired baseline setup. Current runs record one shared baseline component;
# legacy rows may still carry a per-arm setup receipt.
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


def _sum_cost(costs: list[dict[str, Any]], field: str) -> int | float | None:
    """Add one numeric cost field while tolerating older/missing receipts."""

    values = [cost.get(field) for cost in costs]
    present = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not present:
        return None
    total = sum(present)
    # Keep integer counters as integers in JSON while retaining fractional wall
    # times.  This also makes exact regression assertions straightforward.
    return int(total) if all(isinstance(value, int) for value in present) else total


def _cost_totals(costs: list[dict[str, Any]]) -> dict[str, int | float | None]:
    return {field: _sum_cost(costs, field) for field in _COST_FIELDS}


def _optional_int(value: Any) -> int | None:
    """Preserve an unobserved numeric receipt as unknown instead of zero."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


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
        # ``cost`` is the arm-local execution.  Shared baseline and reference
        # repeat builds are recorded once in the lifecycle cost ledger rather
        # than duplicated into every M1-M3 row.
        costs = [row["cost"] for row in arm_rows]
        setup_costs = [
            row.get("metadata", {}).get("baseline_setup_cost", {})
            for row in arm_rows
        ]
        maintenance = _cost_totals(costs)
        # Empty metadata objects are common on M0 and mean that no paired
        # setup component was recorded.  Do not treat ``{}`` as an observed
        # all-unknown setup receipt: a missing setup is known zero here,
        # whereas a non-empty receipt may legitimately contain unknown fields.
        valid_setup_costs = [cost for cost in setup_costs if isinstance(cost, dict) and cost]
        baseline_setup = (
            _cost_totals(valid_setup_costs)
            if valid_setup_costs
            else {field: 0 for field in _COST_FIELDS}
        )
        full = {
            field: (
                maintenance[field] + baseline_setup[field]
                if maintenance[field] is not None and baseline_setup[field] is not None
                else None
            )
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
            # Top-level fields report arm-local execution.  Shared lifecycle
            # components are exposed separately below.
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
            records = probe.get("records") if isinstance(probe, dict) else None
            if isinstance(records, list):
                for record in records:
                    probes.append(
                        {
                            "instance_id": row.get("instance_id"),
                            "arm": arm,
                            "question_id": record.get("question_id"),
                            "evidence_complete": bool(record.get("evidence_complete", False)),
                            "deleted_source_exposed": bool(record.get("deleted_source_exposed", False)),
                            "read_count": len(record.get("read_set", [])),
                            "candidate_count": len(record.get("candidate_ids", [])),
                            "llm_calls": _optional_int(record.get("llm_calls")) or 0,
                            "input_tokens": _optional_int(record.get("input_tokens")),
                            "output_tokens": _optional_int(record.get("output_tokens")),
                            "query_cost_observed": bool(record.get("query_cost_observed", False)),
                            "query_isolated": bool(record.get("query_isolated", False)),
                            "continuation_checks_passed": bool(record.get("continuation_check_passed", False)),
                        }
                    )
                continue
            else:
                input_total = _optional_int(probe.get("input_tokens"))
                output_total = _optional_int(probe.get("output_tokens"))
                read_count = len(probe.get("read_set", []))
                candidate_count = len(probe.get("candidate_ids", []))
                llm_calls = _optional_int(probe.get("llm_calls")) or 0
                evidence_complete = bool(probe.get("evidence_complete", False))
                deleted_exposed = bool(probe.get("deleted_source_exposed", False))
                cost_observed = bool(probe.get("query_cost_observed", False))
                isolated = bool(probe.get("query_isolated", False))
            probes.append(
                {
                    "instance_id": row.get("instance_id"),
                    "arm": arm,
                    "evidence_complete": evidence_complete,
                    "deleted_source_exposed": deleted_exposed,
                    "read_count": read_count,
                    "candidate_count": candidate_count,
                    "llm_calls": llm_calls,
                    "input_tokens": input_total,
                    "output_tokens": output_total,
                    "query_cost_observed": cost_observed,
                    "query_isolated": isolated,
                    "continuation_checks_passed": bool(
                        probe.get("continuation_checks_passed", False)
                    ),
                }
            )
    cost_summary_path = run / "cost_summary.json"
    cost_summary = json.loads(cost_summary_path.read_text(encoding="utf-8")) if cost_summary_path.exists() else None
    fingerprint_path = run / "checkpoints" / "run_fingerprint.json"
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8")) if fingerprint_path.exists() else None
    shared_components = {}
    if isinstance(cost_summary, dict):
        scopes = cost_summary.get("scopes", {})
        for scope_name in ("common_baseline", "reference_repeat", "query_probe"):
            bucket = scopes.get(scope_name)
            if isinstance(bucket, dict):
                shared_components[scope_name] = bucket
    return {
        "run_path": str(run.resolve()),
        "expected_instances": expected,
        "instances": len(rows),
        "complete": (
            expected is not None
            and len(rows) == expected
            and len({row.get("instance_id") for row in rows}) == len(rows)
            and all(
                all(
                    arm in row.get("arms", {})
                    and isinstance(row["arms"].get(arm), dict)
                    and row["arms"][arm].get("state_hash")
                    and isinstance(row["arms"][arm].get("cost"), dict)
                    for arm in ARMS
                )
                for row in rows
            )
        ),
        "instance_ids": [row.get("instance_id") for row in rows],
        "arm_summary": arm_summary,
        "query_probes": {
            "count": len(probes),
            "evidence_complete": sum(item["evidence_complete"] for item in probes),
            "deleted_source_exposed": sum(item["deleted_source_exposed"] for item in probes),
            "llm_calls": sum(item["llm_calls"] for item in probes),
            "input_tokens": (
                sum(item["input_tokens"] for item in probes)
                if probes and all(item["input_tokens"] is not None for item in probes)
                else None
            ),
            "output_tokens": (
                sum(item["output_tokens"] for item in probes)
                if probes and all(item["output_tokens"] is not None for item in probes)
                else None
            ),
            "query_cost_complete": bool(probes) and all(item["query_cost_observed"] for item in probes),
            "query_isolation_complete": bool(probes) and all(item["query_isolated"] for item in probes),
            "continuation_checks_passed": sum(
                item["continuation_checks_passed"] for item in probes
            ),
            "records": probes,
        },
        "query_gate": {
            "all_arms_have_probe": bool(rows) and all(
                isinstance(row.get("arms", {}).get(arm, {}).get("query_probe"), dict)
                for row in rows for arm in ARMS
            ),
            "all_probes_isolated": bool(probes) and all(item["query_isolated"] for item in probes),
            "all_probes_evidence_complete": bool(probes) and all(item["evidence_complete"] for item in probes),
            "all_continuation_checks_passed": bool(probes) and all(
                item["continuation_checks_passed"] for item in probes
            ),
        },
        "fingerprint": fingerprint,
        "cost_summary": cost_summary,
        "cost_summary_sha256": _hash(cost_summary_path) if cost_summary_path.exists() else None,
        "shared_components": shared_components,
    }


def _markdown(report: dict[str, Any]) -> str:
    def display(value: Any) -> str:
        return "unknown" if value is None else str(value)

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
                f"- Query probes: {workload['query_probes']['evidence_complete']}/{workload['query_probes']['count']} evidence-complete; {workload['query_probes']['deleted_source_exposed']} deleted-source exposures; continuation checks passed={workload['query_probes']['continuation_checks_passed']}; query cost complete={workload['query_probes']['query_cost_complete']}; query isolation complete={workload['query_probes']['query_isolation_complete']}",
                "- Arm rows report arm-local work; shared common-baseline and reference-repeat work are recorded once under `shared_components`.",
                "",
                "| Arm | Exact to M0 | Arm-local replayed events | Arm-local LLM calls | Arm-local embedding calls | Arm-local wall time (ms) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for arm in ARMS:
            value = workload["arm_summary"][arm]
            lines.append(
                f"| {arm} | {value['exact_count']}/{value['instances']} | {display(value['replayed_events'])} | {display(value['llm_calls'])} | {display(value['embedding_calls'])} | {display(value['wall_time_ms'])} |"
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
