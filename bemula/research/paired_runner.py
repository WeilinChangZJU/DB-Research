"""Dataset-neutral paired M0--M3 runner for Q01 validation.

The original ``run_research_eval.py`` is an answer-level Bemula runner.  This
module is the explicit Q01 path: it converts one stable source history into a
controlled ``SourceEvent`` stream, generates deletion masks from an auditable
manifest, runs every maintenance arm against the same history and mask, and
keeps the per-history/per-scenario records needed for paired statistics.

The default builder is deterministic and model-free.  Its output is therefore
``controlled_replay`` evidence, not a claim about an LLM memory system.  A
builder-specific implementation can be injected once it satisfies the same
M0--M3 contracts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .maintenance import DeterministicMemoryBuilder, MaintenanceArm, SourceEvent, run_all_arms
from .scenarios import (
    DeletionScenario,
    DeletionScenarioManifest,
    generate_deletion_scenarios,
    source_events_from_instance,
)
from .stats import (
    PairedObservation,
    paired_bootstrap_mean_difference,
    paired_observation,
    summarize_paired_observations,
)


def _item_source_ids(item: Any) -> tuple[str, ...]:
    values = getattr(item, "source_ids", None)
    if values is None:
        values = getattr(item, "related_source_ids", ())
    return tuple(str(value) for value in (values or ()))


@dataclass
class PairedHistoryResult:
    """All paired maintenance evidence for one independent history."""

    dataset: str
    instance_id: str
    source_hash: str
    manifest: DeletionScenarioManifest
    executed_scenarios: tuple[DeletionScenario, ...] = ()
    observations: list[PairedObservation] = field(default_factory=list)
    reference_replayed_events: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        summaries = summarize_paired_observations(
            self.observations,
            reference_replayed_events=self.reference_replayed_events,
        )
        return {
            "dataset": self.dataset,
            "instance_id": self.instance_id,
            "source_hash": self.source_hash,
            "metadata": dict(self.metadata),
            "aggregation_unit": "scenario_with_history_cluster",
            "manifest": self.manifest.to_dict(),
            "manifest_scenario_count": len(self.manifest.scenarios),
            "executed_scenario_count": len(self.executed_scenarios),
            "executed_scenario_ids": [item.scenario_id for item in self.executed_scenarios],
            "observations": [item.to_dict() for item in self.observations],
            "summaries": {key: value.to_dict() for key, value in sorted(summaries.items())},
            "reference_replayed_events": dict(sorted(self.reference_replayed_events.items())),
        }


def run_paired_instance(
    instance: Any,
    *,
    builder: DeterministicMemoryBuilder | None = None,
    seed: int = 0,
    max_scenarios: int | None = None,
    max_events: int | None = None,
) -> PairedHistoryResult:
    """Run a complete paired maintenance matrix for one source history."""

    all_events = source_events_from_instance(instance)
    events = all_events if max_events is None else all_events[: max(0, max_events)]
    if not events:
        raise ValueError(f"instance {getattr(instance, 'instance_id', '<unknown>')} has no source events")
    active_builder = builder or DeterministicMemoryBuilder()
    baseline = active_builder.build(events, arm=MaintenanceArm.M0_FULL_REBUILD)
    used_source_ids = {event.source_id for event in events}
    query_source_ids = {
        str(getattr(item, "item_id", f"item-{index}")): _item_source_ids(item)
        for index, item in enumerate(getattr(instance, "items", ()) or ())
    }
    query_source_ids = {
        key: tuple(source_id for source_id in values if source_id in used_source_ids)
        for key, values in query_source_ids.items()
    }
    manifest = generate_deletion_scenarios(
        events,
        dataset=str(getattr(instance, "dataset", "unknown")),
        history_id=str(getattr(instance, "instance_id", "unknown")),
        seed=seed,
        trace=baseline.trace,
        query_source_ids=query_source_ids,
    )
    scenarios: Sequence[DeletionScenario] = manifest.scenarios
    if max_scenarios is not None:
        scenarios = scenarios[: max(0, max_scenarios)]
        manifest = replace(
            manifest,
            scenarios=tuple(scenarios),
            metadata={**dict(manifest.metadata), "scenario_limit": max(0, max_scenarios)},
        )

    observations: list[PairedObservation] = []
    reference_costs: dict[str, float] = {}
    for scenario in scenarios:
        outputs = run_all_arms(
            events,
            scenario.deleted_source_ids,
            builder=active_builder,
        )
        reference_costs[scenario.scenario_id] = float(
            outputs[MaintenanceArm.M0_FULL_REBUILD][0].cost.replayed_events
        )
        for arm, (result, comparison) in outputs.items():
            observations.append(
                paired_observation(
                    scenario_id=scenario.scenario_id,
                    arm=arm.value,
                    comparison=comparison,
                    cost=result.cost,
                    metadata={
                        "dataset": getattr(instance, "dataset", "unknown"),
                        "instance_id": getattr(instance, "instance_id", "unknown"),
                        "scenario_kind": scenario.kind.value,
                        "deleted_source_ids": list(scenario.deleted_source_ids),
                        "builder": active_builder.name,
                        "evidence_level": "controlled_replay",
                        "reference_replayed_events": reference_costs[scenario.scenario_id],
                    },
                )
            )
    return PairedHistoryResult(
        dataset=str(getattr(instance, "dataset", "unknown")),
        instance_id=str(getattr(instance, "instance_id", "unknown")),
        source_hash=manifest.source_hash,
        manifest=manifest,
        executed_scenarios=tuple(scenarios),
        observations=observations,
        reference_replayed_events=reference_costs,
        metadata={
            "source_event_count_total": len(all_events),
            "source_event_count_used": len(events),
            "scaled_source_prefix": max_events is not None and len(events) < len(all_events),
        },
    )


def run_paired_histories(
    instances: Iterable[Any],
    *,
    builder: DeterministicMemoryBuilder | None = None,
    seed: int = 0,
    max_scenarios: int | None = None,
    max_events: int | None = None,
) -> dict[str, Any]:
    """Run histories independently and return a reviewable aggregate.

    Scenarios within one history remain paired and are not counted as
    independent histories.  The aggregate includes both history-level results
    and scenario-level observations so callers can choose a cluster bootstrap
    instead of treating questions/masks as independent samples.
    """

    histories = [
        run_paired_instance(
            instance,
            builder=builder,
            seed=seed,
            max_scenarios=max_scenarios,
            max_events=max_events,
        )
        for instance in instances
    ]
    observations = [item for history in histories for item in history.observations]
    reference = {
        f"{history.instance_id}:{scenario_id}": value
        for history in histories
        for scenario_id, value in history.reference_replayed_events.items()
    }
    # Scenario IDs are derived from the frozen source IDs.  They are normally
    # history-qualified by dataset adapters; retain a defensive first-seen
    # mapping here while the per-history reports keep the exact paired lookup.
    reference_by_scenario: dict[str, float] = {}
    for item in observations:
        value = item.metadata.get("reference_replayed_events")
        if isinstance(value, (int, float)):
            reference_by_scenario.setdefault(item.scenario_id, float(value))
    summaries = summarize_paired_observations(
        observations,
        reference_replayed_events=reference_by_scenario,
    )
    bootstrap: dict[str, dict[str, float | int]] = {}
    for arm in sorted({item.arm for item in observations}):
        arm_items = [item for item in observations if item.arm == arm]
        left = [
            history.reference_replayed_events.get(item.scenario_id, 0.0)
            for history in histories
            for item in history.observations
            if item.arm == arm
        ]
        right = [float(item.cost.replayed_events) for item in arm_items]
        bootstrap[arm] = paired_bootstrap_mean_difference(left, right, seed=seed)
    return {
        "schema_version": 1,
        "evidence_level": "controlled_replay",
        "aggregation_unit": "scenario_with_history_cluster",
        "independent_history_count": len(histories),
        "scenario_count": sum(len(history.executed_scenarios) for history in histories),
        "manifest_scenario_count": sum(len(history.manifest.scenarios) for history in histories),
        "histories": [history.to_dict() for history in histories],
        "summaries": {key: value.to_dict() for key, value in sorted(summaries.items())},
        "paired_bootstrap_replay_difference": bootstrap,
        "reference_replayed_events": reference,
    }


def write_paired_results(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write an aggregate paired report as deterministic UTF-8 JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = [
    "PairedHistoryResult",
    "run_paired_instance",
    "run_paired_histories",
    "write_paired_results",
]
