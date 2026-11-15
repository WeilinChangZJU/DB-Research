"""Run a deterministic, source-level A-MEM M0--M3 preflight.

This runner executes the pinned official A-MEM robust implementation through
the same Bemula source-aware wrapper as the real-model runner.  It uses a
deterministic local LLM and hash embedder so the code path can be audited when
no API key or local serving endpoint is available.  Its output is explicitly
labelled ``controlled_scripted_official`` and must not be reported as a
real-model result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from bemula.baselines.amem import AmemAdapter
from bemula.llm import BaseLLM
from bemula.research import (
    CheckpointStore,
    CostRecorder,
    MaintenanceArm,
    SourceEvent,
    TraceRecorder,
    run_amem_all_arms,
    sha256_file,
    sha256_tree,
    write_manifest,
    build_manifest,
)


OFFICIAL_REPO = Path(
    os.environ.get(
        "BEMULA_AMEM_PATH",
        Path(__file__).resolve().parents[3]
        / "_amem_extract"
        / "A-mem-f9a173dec325d84fcc97e7828775c7b241c33466",
    )
).resolve()


class ScriptedOfficialLLM(BaseLLM):
    """Deterministic parser-compatible substitute for A-MEM's LLM calls."""

    model = "controlled-scripted-official-v1"

    def __init__(self, *, policy: str = "context_sensitive_v1") -> None:
        self.last_usage: dict[str, int] = {}
        self.calls = 0
        self.policy = policy

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        task: str = "generic",
    ) -> str:
        prompt = str(messages[-1].get("content", "")) if messages else ""
        upper = prompt.upper()
        self.calls += 1
        output: str
        if task == "amem_answer":
            output = "controlled scripted answer"
        elif "ANALYZE THE NEW MEMORY NOTE" in upper:
            if self.policy == "fixed_noop":
                decision = "NO_EVOLUTION"
            else:
                branch = hashlib.sha256(prompt.encode("utf-8")).digest()[0] % 4
                decision = (
                    "NO_EVOLUTION", "STRENGTHEN", "UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE"
                )[branch]
            output = f"DECISION: {decision}\nREASON: deterministic context-sensitive policy"
        elif "CONNECTIONS:" in upper:
            output = "CONNECTIONS:\nTAGS: controlled, scripted, memory"
        elif "NEIGHBOR 0:" in upper:
            # This branch is defensive; the deterministic policy normally
            # returns NO_EVOLUTION before neighbour updates are requested.
            count = max(1, len(re.findall(r"NEIGHBOR\s+\d+:", upper)))
            blocks = []
            for index in range(count):
                blocks.append(
                    f"NEIGHBOR {index}:\nCONTEXT: controlled context\n"
                    "TAGS: controlled, scripted, memory"
                )
            output = "\n\n".join(blocks)
        elif "KEYWORDS:" in upper or "ANALYZE CONTENT" in upper:
            output = (
                "KEYWORDS: controlled, scripted, memory\n"
                "CONTEXT: Deterministic controlled preflight content.\n"
                "TAGS: controlled, scripted, preflight"
            )
        elif "LIST EXACTLY 5 KEYWORDS" in upper:
            output = "controlled, scripted, memory, preflight, note"
        else:
            output = "controlled scripted response"
        self.last_usage = {
            "prompt_tokens": max(1, len(prompt) // 4),
            "completion_tokens": max(1, len(output) // 4),
            "total_tokens": max(2, len(prompt) // 4 + len(output) // 4),
        }
        return output


class HashEmbedder:
    """Small deterministic lexical embedder with no downloaded model."""

    dimension = 64

    def __init__(self) -> None:
        self.calls = 0
        self.items = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.items += len(texts)
        rows: list[np.ndarray] = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for token in re.findall(r"\w+", str(text).lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign
            norm = float(np.linalg.norm(vector))
            if norm:
                vector /= norm
            rows.append(vector)
        return np.vstack(rows) if rows else np.empty((0, self.dimension), dtype=np.float32)


def _load_histories(path: Path) -> list[dict[str, Any]]:
    histories: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            events = [
                SourceEvent(
                    event_id=str(raw.get("event_id", f"e{index}")),
                    source_id=str(raw["source_id"]),
                    content=str(raw["content"]),
                    timestamp=str(raw.get("timestamp", "")),
                    metadata=dict(raw.get("metadata", {})),
                    speaker=(
                        str(raw.get("speaker"))
                        if raw.get("speaker") is not None
                        else None
                    ),
                    session_id=(
                        str(raw.get("session_id"))
                        if raw.get("session_id") is not None
                        else None
                    ),
                )
                for index, raw in enumerate(item.get("events", []))
            ]
            deleted = tuple(dict.fromkeys(str(value) for value in item.get("deleted_source_ids", [])))
            event_sources = {event.source_id for event in events}
            unknown = sorted(set(deleted) - event_sources)
            if unknown:
                raise ValueError(f"{path}:{line_no} unknown deleted source IDs: {unknown}")
            histories.append(
                {
                    "instance_id": str(item["instance_id"]),
                    "dataset": str(item.get("dataset", "q01")),
                    "events": events,
                    "deleted_source_ids": deleted,
                    "items": list(item.get("items", [])),
                }
            )
    return histories


def _summary(
    instance: dict[str, Any],
    outputs: dict[Any, tuple[Any, Any]],
    *,
    canonical_snapshots: dict[str, Any],
    query_probes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm, (result, comparison) in outputs.items():
        canonical = canonical_snapshots[arm.value]
        arms[arm.value] = {
            "exact_to_m0": comparison.equal,
            "state_hash": canonical.canonical_hash,
            "visible_state_hash": canonical.visible_state_hash,
            "runtime_state_hash": canonical.runtime_state_hash,
            "audit_trace_hash": canonical.audit_trace_hash,
            "cost": result.cost.__dict__,
            "metadata": dict(result.metadata),
            "mismatches": {
                "changed_ids": list(comparison.changed_ids),
                "reason": comparison.reason,
            },
            "query_probe": query_probes.get(arm.value),
        }
    return {
        "schema_version": 1,
        "evidence_level": "controlled_scripted_official",
        "official_amem_commit": "f9a173dec325d84fcc97e7828775c7b241c33466",
        "instance_id": instance["instance_id"],
        "dataset": instance["dataset"],
        "deleted_source_ids": list(instance["deleted_source_ids"]),
        "arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument(
        "--m3-mode",
        choices=("materialized", "recheck", "conservative_full"),
        default="materialized",
        help="M3 dependency policy; recheck enables request-key reuse over a generic re-evaluation.",
    )
    parser.add_argument(
        "--include-m-dag",
        action="store_true",
        help=(
            "Also run the explicit materialized dependency-DAG repair-all "
            "baseline as a fifth opt-in arm."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--scripted-policy",
        choices=("context_sensitive_v1", "fixed_noop"),
        default="context_sensitive_v1",
        help="Deterministic controlled LLM policy; context_sensitive_v1 exercises all evolution branches.",
    )
    args = parser.parse_args()
    if not OFFICIAL_REPO.is_dir():
        raise SystemExit(f"official A-MEM source is missing: {OFFICIAL_REPO}")
    args.out.mkdir(parents=True, exist_ok=True)
    histories = _load_histories(args.histories)
    if args.limit is not None:
        histories = histories[: args.limit]

    root = Path.cwd().resolve()
    store = CheckpointStore(args.out)
    fingerprint = {
        "schema_version": 1,
        "runner": "run_q01_amem_controlled",
        "evidence_level": "controlled_scripted_official",
        "official_amem_commit": "f9a173dec325d84fcc97e7828775c7b241c33466",
        "official_amem_source_tree_sha256": sha256_tree(OFFICIAL_REPO),
        "official_amem_robust_sha256": sha256_file(OFFICIAL_REPO / "memory_layer_robust.py"),
        "histories_sha256": sha256_file(args.histories),
        "retrieve_k": args.retrieve_k,
        "m3_mode": args.m3_mode,
        "include_m_dag": bool(args.include_m_dag),
        "repository_tree_sha256": sha256_tree(root, exclude=(".venv", ".git", "results")),
        "scripted_model": ScriptedOfficialLLM.model,
        "embedding": {"backend": "hash", "dimension": HashEmbedder.dimension},
        "scripted_policy": args.scripted_policy,
    }
    store.ensure_fingerprint(fingerprint)
    done = store.completed_ids(validate=True) if args.resume else set()
    cost = CostRecorder(args.out / "cost.jsonl")
    if args.resume:
        cost.load_existing()
    trace = TraceRecorder(args.out / "trace.jsonl")
    embedder = HashEmbedder()

    for instance in histories:
        instance_id = instance["instance_id"]
        if instance_id in done:
            continue

        def factory() -> AmemAdapter:
            return AmemAdapter(
                ScriptedOfficialLLM(policy=args.scripted_policy),
                repo_path=OFFICIAL_REPO,
                embedder=embedder,
                retrieve_k=args.retrieve_k,
                progress=False,
            )

        outputs = run_amem_all_arms(
            instance["events"],
            instance["deleted_source_ids"],
            factory,
            retrieve_k=args.retrieve_k,
            trace=trace,
            trace_context={
                "instance_id": instance["instance_id"],
                "base_instance_id": instance.get("base_instance_id", instance["instance_id"]),
                "scenario_id": instance["instance_id"],
                "selection": instance.get("selection", {}),
            },
            m3_mode=args.m3_mode,
            include_m_dag=args.include_m_dag,
        )
        # Probe the actual serving path after maintenance.  The question is a
        # hash-stable snippet of the deleted event, so the probe can detect
        # residual source exposure without persisting source text in results.
        deleted = set(instance["deleted_source_ids"])
        deleted_events = [event for event in instance["events"] if event.source_id in deleted]
        item_questions = [
            str(item.get("question", ""))
            for item in instance.get("items", [])
            if isinstance(item, dict) and str(item.get("question", "")).strip()
        ]
        question = (
            item_questions[0]
            if item_questions
            else (deleted_events[0].content if deleted_events else instance["events"][0].content)
        )[:512]
        canonical_snapshots = {
            arm.value: result.wrapper.canonical_snapshot(
                dataset=instance["dataset"], instance_id=instance["instance_id"]
            )
            for arm, (result, _comparison) in outputs.items()
        }
        query_probes: dict[str, dict[str, Any]] = {}
        for arm, (result, _comparison) in outputs.items():
            query_isolated = False
            try:
                probe_wrapper = result.wrapper.fork(trace=trace)
                probe_wrapper.set_trace_context(phase="query_probe")
                query_isolated = True
            except Exception:
                probe_wrapper = result.wrapper
            adapter_llm = getattr(probe_wrapper.adapter, "llm", None)
            before_llm = int(getattr(adapter_llm, "calls", 0) or 0)
            adapter_embedder = getattr(probe_wrapper.adapter, "embedder", None)
            before_embed_calls = int(getattr(adapter_embedder, "calls", 0) or 0)
            before_embed_items = int(getattr(adapter_embedder, "items", 0) or 0)
            started = time.perf_counter()
            probe = probe_wrapper.query(question)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            after_llm = int(getattr(adapter_llm, "calls", 0) or 0)
            usage = dict(getattr(adapter_llm, "last_usage", {}) or {})
            after_embed_calls = int(getattr(adapter_embedder, "calls", 0) or 0)
            after_embed_items = int(getattr(adapter_embedder, "items", 0) or 0)
            evidence = probe.evidence
            details = dict(evidence.details)
            source_ids = sorted(set(evidence.source_ids) | set(evidence.read_set))
            deleted_exposed = bool(deleted.intersection(source_ids))
            query_probes[arm.value] = {
                "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "source_ids": source_ids,
                "read_set": sorted(set(evidence.read_set)),
                "candidate_ids": list(details.get("candidate_ids", [])),
                "selected_ids": list(details.get("selected_ids", [])),
                "context_object_ids": list(details.get("read_object_ids", [])),
                "context_source_ids": list(details.get("context_source_ids", [])),
                "context_segments": list(details.get("context_segments", [])),
                "context_segments_complete": bool(details.get("context_segments_complete", False)),
                "context_mapping_complete": bool(details.get("context_mapping_complete", False)),
                "lineage_coverage": bool(details.get("lineage_coverage", False)),
                "context_coverage": bool(details.get("context_coverage", False)),
                "raw_context_hash": details.get("raw_context_hash"),
                "evidence_complete": bool(details.get("evidence_complete", False)),
                "deleted_source_exposed": deleted_exposed,
                "query_isolated": query_isolated,
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "query_cost_observed": bool(usage) or max(0, after_llm - before_llm) == 0,
                "answer_sha256": hashlib.sha256(str(probe.answer).encode("utf-8")).hexdigest(),
            }
            cost.record(
                "query_probe",
                scope="query_probe",
                duration_ms=elapsed_ms,
                counters={
                    "llm_calls": max(0, after_llm - before_llm),
                    "provider_attempts": max(0, after_llm - before_llm),
                    "provider_successes": max(0, after_llm - before_llm),
                    "input_tokens": usage.get("prompt_tokens", 0) if usage else 0,
                    "output_tokens": usage.get("completion_tokens", 0) if usage else 0,
                    "embedding_calls": max(0, after_embed_calls - before_embed_calls),
                    "embedding_items": max(0, after_embed_items - before_embed_items),
                    "retrieved_items": len(details.get("candidate_ids", [])),
                    "memory_reads": len(evidence.read_set),
                },
                arm=arm.value,
                query_sha256=query_probes[arm.value]["question_sha256"],
                deleted_source_exposed=deleted_exposed,
                evidence_complete=bool(details.get("evidence_complete", False)),
                query_isolated=query_isolated,
            )
        summary = _summary(
            instance,
            outputs,
            canonical_snapshots=canonical_snapshots,
            query_probes=query_probes,
        )
        for arm, (result, comparison) in outputs.items():
            cost.record_arm_cost(
                arm=arm.value,
                cost=result.cost,
                exact_to_reference=comparison.equal,
                phase="reference_rebuild" if arm is MaintenanceArm.M0_FULL_REBUILD else "maintenance",
            )
            setup = result.metadata.get("baseline_setup_cost")
            if setup:
                cost.record_arm_cost(
                    arm=f"{arm.value}:baseline_setup",
                    cost=SimpleNamespace(**setup),
                    exact_to_reference=None,
                    phase="baseline_setup",
                )
        shared_baseline = next(
            (
                result.metadata.get("shared_baseline_cost")
                for result, _comparison in outputs.values()
                if isinstance(result.metadata.get("shared_baseline_cost"), dict)
            ),
            None,
        )
        if shared_baseline is not None:
            cost.record_arm_cost(
                arm="shared_baseline",
                cost=SimpleNamespace(**shared_baseline),
                exact_to_reference=None,
                phase="common_baseline",
            )
        reference_repeat = outputs[MaintenanceArm.M0_FULL_REBUILD][0].metadata.get("reference_repeat_cost")
        if isinstance(reference_repeat, dict):
            cost.record_arm_cost(
                arm="reference_repeat",
                cost=SimpleNamespace(**reference_repeat),
                exact_to_reference=None,
                phase="reference_repeat",
            )
        trace.record(
            "q01_instance_complete",
            instance_id=instance_id,
            deleted_source_ids=list(instance["deleted_source_ids"]),
            arm_count=len(outputs),
        )
        store.commit(instance_id, [summary])

    cost.write_summary(args.out / "cost_summary.json")
    write_manifest(
        args.out / "manifest.json",
        build_manifest(
            root=root,
            data_paths=[args.histories, OFFICIAL_REPO],
            execution={
                "runner": "run_q01_amem_controlled",
                "status": "completed",
                "evidence_level": "controlled_scripted_official",
                "num_instances": len(store.completed_ids(validate=True)),
                "resume": args.resume,
                "retrieve_k": args.retrieve_k,
                "m3_mode": args.m3_mode,
                "include_m_dag": bool(args.include_m_dag),
            },
            extra={"run_fingerprint_sha256": store.load_fingerprint()["sha256"]},
        ),
    )
    print(f"Wrote controlled official A-MEM results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
