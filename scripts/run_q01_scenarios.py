"""Run deterministic deletion scenarios and paired M0-M3 reports offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bemula.research import (
    DeterministicMemoryBuilder,
    SourceEvent,
    generate_deletion_scenarios,
    paired_observation,
    paired_bootstrap_mean_difference,
    run_all_arms,
    summarize_paired_observations,
    write_failure_report,
)


def _fixture_events() -> list[SourceEvent]:
    return [
        SourceEvent("e0", "src-a", "approval limit is ten thousand", timestamp="t0"),
        SourceEvent("e1", "src-b", "approval limit is ten thousand", timestamp="t1"),
        SourceEvent("e2", "src-c", "office is in Shanghai", timestamp="t2"),
        SourceEvent("e3", "src-d", "travel next week", timestamp="t3"),
        SourceEvent("e4", "src-e", "approval policy changed", timestamp="t4"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    events = _fixture_events()
    baseline = DeterministicMemoryBuilder().build(events)
    manifest = generate_deletion_scenarios(
        events,
        dataset="controlled",
        history_id="q01-fixture",
        seed=0,
        trace=baseline.trace,
    )
    manifest.write_json(args.out / "deletion_manifest.json")

    observations = []
    reference_replayed_events: dict[str, float] = {}
    for scenario in manifest.scenarios:
        outputs = run_all_arms(events, scenario.deleted_source_ids)
        reference_replayed_events[scenario.scenario_id] = float(
            outputs[next(iter(outputs))][0].cost.replayed_events
        )
        for arm, (result, comparison) in outputs.items():
            observations.append(
                paired_observation(
                    scenario_id=scenario.scenario_id,
                    arm=arm.value,
                    comparison=comparison,
                    cost=result.cost,
                    metadata={
                        "scenario_kind": scenario.kind.value,
                        "deleted_source_ids": list(scenario.deleted_source_ids),
                    },
                )
            )
    summaries = summarize_paired_observations(
        observations,
        reference_replayed_events=reference_replayed_events,
    )
    write_failure_report(args.out / "failure_report.json", observations, summaries)
    bootstrap: dict[str, dict[str, float | int]] = {}
    for arm in sorted({observation.arm for observation in observations}):
        items = [observation for observation in observations if observation.arm == arm]
        left = [reference_replayed_events[item.scenario_id] for item in items]
        right = [float(item.cost.replayed_events) for item in items]
        bootstrap[arm] = paired_bootstrap_mean_difference(left, right, seed=0)
    summary = {
        "offline": True,
        "api_calls": 0,
        "history_events": len(events),
        "scenario_count": len(manifest.scenarios),
        "scenario_kinds": sorted({scenario.kind.value for scenario in manifest.scenarios}),
        "arms": {key: value.to_dict() for key, value in summaries.items()},
        "paired_bootstrap_replay_savings": bootstrap,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.out / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
