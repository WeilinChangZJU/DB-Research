"""Q01 orchestration for deterministic maintenance-arm experiments.

This module is deliberately separate from Bemula's answer-level runner.  It
joins the reference maintenance arms, canonical state comparator, and cost
accounting into one replayable result object.  The default entry point uses
the offline deterministic builder; real A-MEM/LightMem builders must implement
the same state/trace/cost contracts before they are eligible for Q01 claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .cost import CostRecorder
from .maintenance import (
    BuildCost,
    BuildResult,
    DeterministicMemoryBuilder,
    MaintenanceArm,
    SourceEvent,
    StateComparison,
    run_all_arms,
)
from .state import CanonicalState, CanonicalStateComparator, CanonicalStateDiff


@dataclass
class Q01ArmResult:
    """One maintenance arm and its comparison against the M0 reference."""

    arm: MaintenanceArm
    build: BuildResult
    state: CanonicalState
    state_diff: CanonicalStateDiff | None = None
    snapshot_diff: StateComparison | None = None

    @property
    def exact_to_reference(self) -> bool:
        return bool(self.state_diff and self.state_diff.equivalent)


@dataclass
class Q01RunResult:
    """Serializable result for one fixed history and deletion mask."""

    dataset: str
    instance_id: str
    deleted_ids: tuple[str, ...]
    reference: Q01ArmResult
    arms: dict[MaintenanceArm, Q01ArmResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def exact_arms(self) -> tuple[MaintenanceArm, ...]:
        return tuple(arm for arm, result in self.arms.items() if result.exact_to_reference)

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset": self.dataset,
            "instance_id": self.instance_id,
            "deleted_ids": list(self.deleted_ids),
            "reference_arm": self.reference.arm.value,
            "arms": {
                arm.value: {
                    "exact_to_reference": result.exact_to_reference,
                    "canonical_state_hash": result.state.canonical_hash,
                    "visible_state_hash": result.state.visible_state_hash,
                    "runtime_state_hash": result.state.runtime_state_hash,
                    "audit_trace_hash": result.state.audit_trace_hash,
                    "snapshot_state_hash": result.build.snapshot.state_hash,
                    "snapshot_equal": result.snapshot_diff.equal if result.snapshot_diff else None,
                    "mismatch_counts": result.state_diff.mismatch_counts if result.state_diff else {},
                    "cost": {
                        "input_events": result.build.cost.input_events,
                        "replayed_events": result.build.cost.replayed_events,
                        "prefix_replayed_events": result.build.cost.prefix_replayed_events,
                        "suffix_replayed_events": result.build.cost.suffix_replayed_events,
                        "checkpoint_restore_events": result.build.cost.checkpoint_restore_events,
                        "cost_scope": result.build.cost.cost_scope,
                        "retrieved_candidates": result.build.cost.retrieved_candidates,
                        "generated_objects": result.build.cost.generated_objects,
                        "state_records": result.build.cost.state_records,
                    },
                    "maintenance_metadata": dict(result.build.metadata),
                }
                for arm, result in self.arms.items()
            },
            "metadata": dict(self.metadata),
        }


def _as_state(build: BuildResult, *, dataset: str, instance_id: str) -> CanonicalState:
    return CanonicalState.from_snapshot(
        build.snapshot,
        dataset=dataset,
        instance_id=instance_id,
    )


def run_toy_experiment(
    events: Sequence[SourceEvent],
    deleted_ids: Iterable[str],
    *,
    dataset: str = "fixture",
    instance_id: str = "fixture-0",
    cost: CostRecorder | None = None,
) -> Q01RunResult:
    """Run M0-M3 on the deterministic reference builder.

    This function is an offline architecture test and not an LLM experiment.
    It uses the same deletion mask for every arm and compares every result to
    the M0 full-rebuild state with the canonical comparator.
    """

    deleted = tuple(dict.fromkeys(str(item) for item in deleted_ids))
    raw = run_all_arms(events, deleted)
    reference_build = raw[MaintenanceArm.M0_FULL_REBUILD][0]
    reference_state = _as_state(reference_build, dataset=dataset, instance_id=instance_id)
    comparator = CanonicalStateComparator()

    results: dict[MaintenanceArm, Q01ArmResult] = {}
    for arm, (build, snapshot_diff) in raw.items():
        state = _as_state(build, dataset=dataset, instance_id=instance_id)
        state_diff = comparator.compare(state, reference_state)
        result = Q01ArmResult(
            arm=arm,
            build=build,
            state=state,
            state_diff=state_diff,
            snapshot_diff=snapshot_diff,
        )
        results[arm] = result
        if cost is not None:
            cost.record_arm_cost(
                arm=arm.value,
                cost=build.cost,
                phase="reference_rebuild" if arm is MaintenanceArm.M0_FULL_REBUILD else "maintenance",
                exact_to_reference=state_diff.equivalent,
            )

    return Q01RunResult(
        dataset=dataset,
        instance_id=instance_id,
        deleted_ids=deleted,
        reference=results[MaintenanceArm.M0_FULL_REBUILD],
        arms=results,
        metadata={"builder": DeterministicMemoryBuilder.name, "offline_reference": True},
    )


__all__ = ["Q01ArmResult", "Q01RunResult", "run_toy_experiment"]
