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
OPTIONAL_ARMS = ("M_DAG_materialized_dependency_repair_all",)


def _arms_for_rows(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    """Return core arms plus any explicitly requested optional baselines."""

    names = list(ARMS)
    for arm in OPTIONAL_ARMS:
        if any(arm in row.get("arms", {}) for row in rows):
            names.append(arm)
    return tuple(names)

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
    "llm_cache_hits",
    "input_tokens",
    "output_tokens",
    "embedding_calls",
    "embedding_items",
    "index_updates",
    "memory_reads",
    "bytes_read",
    "bytes_written",
)

_INSTRUMENTATION_FIELDS = (
    "score_encode_calls",
    "score_matrix_rows",
    "score_matrix_values",
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sum_cost(costs: list[dict[str, Any]], field: str) -> int | float | None:
    """Add one numeric cost field without hiding unknown components.

    A mixed list such as ``[11, None]`` is not an observed total of ``11``:
    the missing component is part of the cost ledger and must remain unknown.
    This conservative rule prevents lifecycle comparisons from silently
    understating one arm when a provider or checkpoint backend omitted a
    counter for only some instances.
    """

    if not costs:
        return None
    values = [cost.get(field) for cost in costs]
    if any(
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        for value in values
    ):
        return None
    total = sum(values)
    # Keep integer counters as integers in JSON while retaining fractional wall
    # times.  This also makes exact regression assertions straightforward.
    return int(total) if all(isinstance(value, int) for value in values) else total


def _cost_totals(costs: list[dict[str, Any]]) -> dict[str, int | float | None]:
    return {field: _sum_cost(costs, field) for field in _COST_FIELDS}


def _cost_coverage(costs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Report whether every row supplied a numeric receipt for each field.

    A lifecycle total is only comparable when its component coverage is
    explicit.  ``unknown`` is intentionally retained for any missing or
    non-numeric receipt instead of turning the missing work into zero.
    """

    coverage: dict[str, dict[str, Any]] = {}
    expected = len(costs)
    for field in _COST_FIELDS:
        observed = sum(
            isinstance(cost.get(field), (int, float))
            and not isinstance(cost.get(field), bool)
            for cost in costs
        )
        coverage[field] = {
            "observed_count": observed,
            "expected_count": expected,
            "coverage": (observed / expected) if expected else 0.0,
            "status": "complete" if expected and observed == expected else (
                "unknown" if observed < expected else "empty"
            ),
        }
    return coverage


def _optional_int(value: Any) -> int | None:
    """Preserve an unobserved numeric receipt as unknown instead of zero."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _sum_optional(values: list[int | None]) -> int | None:
    """Sum observed counters while keeping any missing value unknown."""

    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _load_run(name: str, run: Path, expected: int | None) -> dict[str, Any]:
    instance_dir = run / "checkpoints" / "instances"
    rows: list[dict[str, Any]] = []
    if instance_dir.is_dir():
        for path in sorted(instance_dir.glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                line = next((item for item in handle if item.strip()), "")
            if line:
                rows.append(json.loads(line))
    run_arms = _arms_for_rows(rows)
    arm_summary: dict[str, Any] = {}
    for arm in run_arms:
        arm_row_pairs = [
            (row.get("instance_id"), row["arms"][arm])
            for row in rows
            if arm in row.get("arms", {})
        ]
        arm_rows = [pair[1] for pair in arm_row_pairs]
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
                instance_id
                for instance_id, row in arm_row_pairs
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
            "cost_coverage": {
                "maintenance": _cost_coverage(costs),
                "baseline_setup": _cost_coverage(valid_setup_costs),
                "full": _cost_coverage(
                    [
                        {
                            field: full[field]
                            for field in _COST_FIELDS
                        }
                    ]
                    if arm_rows
                    else []
                ),
            },
        }

    probes = []
    for row in rows:
        for arm in run_arms:
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
                            "context_segments_complete": bool(
                                record.get("context_segments_complete", False)
                            ),
                            "deleted_source_exposed": bool(record.get("deleted_source_exposed", False)),
                            "read_count": len(record.get("read_set", [])),
                            "candidate_count": len(record.get("candidate_ids", [])),
                            # A missing query trace is unknown, not zero.  The
                            # runner normally records this as an observable
                            # event count; legacy rows may omit it.
                            "llm_calls": _optional_int(record.get("llm_calls")),
                            "input_tokens": _optional_int(record.get("input_tokens")),
                            "output_tokens": _optional_int(record.get("output_tokens")),
                            "query_cost_observed": bool(record.get("query_cost_observed", False)),
                            "query_isolated": bool(record.get("query_isolated", False)),
                            "continuation_checks_passed": bool(record.get("continuation_check_passed", False)),
                            "instrumentation_cost": dict(record.get("instrumentation_cost", {}) or {}),
                        }
                    )
                continue
            else:
                input_total = _optional_int(probe.get("input_tokens"))
                output_total = _optional_int(probe.get("output_tokens"))
                read_count = len(probe.get("read_set", []))
                candidate_count = len(probe.get("candidate_ids", []))
                llm_calls = _optional_int(probe.get("llm_calls"))
                evidence_complete = bool(probe.get("evidence_complete", False))
                deleted_exposed = bool(probe.get("deleted_source_exposed", False))
                cost_observed = bool(probe.get("query_cost_observed", False))
                isolated = bool(probe.get("query_isolated", False))
            probes.append(
                {
                    "instance_id": row.get("instance_id"),
                    "arm": arm,
                    "evidence_complete": evidence_complete,
                    "context_segments_complete": bool(
                        probe.get("context_segments_complete", False)
                    ),
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
                    "instrumentation_cost": dict(probe.get("instrumentation_cost", {}) or {}),
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
    row_contracts = []
    for row in rows:
        row_arms = row.get("arms", {})
        row_arm_map = row.get("arms", {})
        m0_metadata = row_arm_map.get("M0_full_rebuild", {}).get("metadata", {})
        m1_metadata = row_arm_map.get("M1_tombstone_filter", {}).get("metadata", {})
        m2_metadata = row_arm_map.get("M2_replay_repair_all", {}).get("metadata", {})
        m3_metadata = row_arm_map.get("M3_exact_read_set", {}).get("metadata", {})
        mdag_metadata = row_arm_map.get(OPTIONAL_ARMS[0], {}).get("metadata", {})
        mdag_present = OPTIONAL_ARMS[0] in row_arm_map
        query_present = all(
            isinstance(row_arm_map.get(arm, {}).get("query_probe"), dict)
            for arm in run_arms
        )
        row_contracts.append(
            {
                "m0_repeat": m0_metadata.get("reference_repeat_equal") is True,
                "m0_repeat_provider_replay": (
                    m0_metadata.get("reference_repeat_provider_replay_policy")
                    == "m0_repeat_request_key_cassette"
                    and m0_metadata.get("reference_repeat_cassette_backed") is True
                    and m0_metadata.get("reference_repeat_provider_replay_complete") is True
                    and int(
                        m0_metadata.get(
                            "reference_repeat_cache_misses",
                            m0_metadata.get("reference_repeat_provider_cache_misses", 0),
                        )
                        or 0
                    )
                    == 0
                ),
                "shared_baseline": all(
                    metadata.get("shared_baseline") is True
                    for metadata in (m1_metadata, m2_metadata, m3_metadata)
                ),
                "m_dag": (
                    not mdag_present
                    or (
                        mdag_metadata.get("dependency_baseline")
                        == "materialized_dependency_dag"
                        and mdag_metadata.get("dag_complete") is True
                        and mdag_metadata.get("repair_policy")
                        == "repair_all_downstream_events"
                    )
                ),
                "checkpoint_replay": (
                    m2_metadata.get("checkpoint_restored") is True
                    and m3_metadata.get("checkpoint_restored") is True
                ),
                "m3_strong_policy": (
                    (
                        m3_metadata.get("m3_mode") == "conservative_full"
                        and m3_metadata.get("m3_status")
                        == "strong_conservative_full_history_checkpoint_replay"
                        and m3_metadata.get("dependency_baseline")
                        == "complete_conservative_barrier"
                    )
                    or (
                        m3_metadata.get("m3_mode") == "recheck"
                        and m3_metadata.get("m3_status")
                        == "generic_recheck_request_key_cache"
                        and m3_metadata.get("dependency_baseline")
                        == "generic_recheck_request_key_cache"
                    )
                ),
                "query_probe_present": query_present,
                "context_lineage_complete": all(
                    bool(record.get("context_segments_complete", False))
                    for arm_name in run_arms
                    for record in (
                        row_arm_map.get(arm_name, {}).get("query_probe", {}).get("records", [])
                        if isinstance(row_arm_map.get(arm_name, {}).get("query_probe"), dict)
                        else []
                    )
                ),
            }
        )
    execution_contract = {
        "all_m0_repeats_verified": bool(row_contracts) and all(item["m0_repeat"] for item in row_contracts),
        "all_m0_repeat_provider_replays_verified": bool(row_contracts) and all(
            item["m0_repeat_provider_replay"] for item in row_contracts
        ),
        "all_shared_baselines_verified": bool(row_contracts) and all(item["shared_baseline"] for item in row_contracts),
        "all_checkpoint_replays_verified": bool(row_contracts) and all(item["checkpoint_replay"] for item in row_contracts),
        "all_m3_strong_policies_verified": bool(row_contracts) and all(item["m3_strong_policy"] for item in row_contracts),
        "all_optional_m_dag_contracts_verified": bool(row_contracts) and all(
            item["m_dag"] for item in row_contracts
        ),
        "all_query_probes_present": bool(row_contracts) and all(item["query_probe_present"] for item in row_contracts),
        "all_context_lineage_verified": bool(row_contracts) and all(
            item["context_lineage_complete"] for item in row_contracts
        ),
        "lifecycle_scopes_present": bool(shared_components.get("common_baseline")) and bool(shared_components.get("reference_repeat")),
    }
    execution_contract["passed"] = all(execution_contract.values())
    return {
        "run_path": str(run.resolve()),
        "expected_instances": expected,
        "instances": len(rows),
        "complete": (
            expected is not None
            and len(rows) == expected
            and len({row.get("instance_id") for row in rows}) == len(rows)
            and execution_contract["passed"]
            and all(
                all(
                    arm in row.get("arms", {})
                    and isinstance(row["arms"].get(arm), dict)
                    and row["arms"][arm].get("state_hash")
                    and isinstance(row["arms"][arm].get("cost"), dict)
                    for arm in run_arms
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
            "llm_calls": _sum_optional([item["llm_calls"] for item in probes]),
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
            # Score reconstruction is extra audit work, separate from the
            # adapter's retrieval/embedding counters.  Keep these totals in
            # the formal review so the instrumentation overhead cannot be
            # mistaken for free observation.
            "instrumentation_cost": {
                field: _sum_optional(
                    [
                        _optional_int(item.get("instrumentation_cost", {}).get(field))
                        for item in probes
                    ]
                )
                for field in _INSTRUMENTATION_FIELDS
            },
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
                for row in rows for arm in run_arms
            ),
            "all_probes_isolated": bool(probes) and all(item["query_isolated"] for item in probes),
            "all_probes_evidence_complete": bool(probes) and all(item["evidence_complete"] for item in probes),
            "all_continuation_checks_passed": bool(probes) and all(
                item["continuation_checks_passed"] for item in probes
            ),
        },
        "execution_contract": execution_contract,
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
        for arm in workload.get("arm_summary", {}):
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

