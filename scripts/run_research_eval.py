"""Opt-in provenance-aware Bemula evaluation runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from bemula.config import load_config
from bemula.eval.judge import LLMJudge
from bemula.eval.metrics import compute_metrics
from bemula.eval.runner import ItemRecord, run_instance
from bemula.llm import get_llm
from bemula.research import CheckpointStore, TraceRecorder, build_manifest, sha256_tree, write_manifest
from bemula.schemas import DialogueInstance, TestItem, load_jsonl
from scripts.run_eval import _build_baseline, write_report


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--baseline", default="fullcontext")
    ap.add_argument("--dialogues", default="data/quick_check/dialogues.jsonl")
    ap.add_argument("--questions", default="data/quick_check/questions.jsonl")
    ap.add_argument("--out", default="results/research_smoke")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prompt-path", action="append", default=[])
    args = ap.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    root = Path.cwd().resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    load_dotenv(override=True)
    config = load_config(args.config)
    started_at = utc_now()
    store = CheckpointStore(out)
    trace = TraceRecorder(out / "trace.jsonl")

    instances = load_jsonl(args.dialogues, DialogueInstance)
    if args.limit is not None:
        instances = instances[:args.limit]
    items_by_instance: dict[str, list[TestItem]] = {}
    for item in load_jsonl(args.questions, TestItem):
        items_by_instance.setdefault(item.instance_id, []).append(item)

    embed_backend = os.environ.get("BEMULA_EMBED_BACKEND", "api").strip().lower()
    embed_provider = {"api": "openai", "local": "huggingface"}.get(embed_backend, embed_backend)

    # Resume is only safe when all semantic inputs and the implementation tree
    # are unchanged.  This receipt is checked before reading completed IDs.
    tree_digest = sha256_tree(root, exclude=(".venv", ".git", "results"))
    config_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    fingerprint_payload = {
        "schema_version": 1,
        "baseline": args.baseline,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.is_file() else None,
        "dialogues": str(Path(args.dialogues).resolve()),
        "questions": str(Path(args.questions).resolve()),
        "prompt_paths": [str(Path(item).resolve()) for item in args.prompt_path],
        "seed": config.get("seed"),
        "repository_tree_sha256": tree_digest,
    }
    store.ensure_fingerprint(fingerprint_payload)
    done = store.completed_ids(validate=True) if args.resume else set()
    todo = [instance for instance in instances if instance.instance_id not in done]
    if done:
        print(f"Resuming: skipping {len(done)} complete instance(s)")

    judge = LLMJudge(
        get_llm(model=os.environ.get("BEMULA_JUDGE_MODEL")),
        temperature=config.get("evaluation", {}).get("judge_temperature", 0.0),
    )
    agent_llm = get_llm(model=os.environ.get("BEMULA_AGENT_MODEL"))
    # A single adapter is stateful (reset/observe/query mutate it).  Use
    # thread-local ownership so concurrent tasks never interleave operations on
    # the same memory instance.  The first adapter is reused only in the
    # explicitly serial path.
    adapter_local = threading.local()

    def adapter_for_current_worker():
        adapter = getattr(adapter_local, "adapter", None)
        if adapter is None:
            worker_llm = agent_llm if args.workers == 1 else get_llm(model=os.environ.get("BEMULA_AGENT_MODEL"))
            adapter = _build_baseline(args.baseline, worker_llm, embed_provider)
            adapter_local.adapter = adapter
        return adapter

    def write_run_manifest(status: str) -> None:
        manifest = build_manifest(
            root=root,
            config_paths=[args.config],
            data_paths=[args.dialogues, args.questions],
            prompt_paths=args.prompt_path,
            seed=config.get("seed"),
            execution={
                "status": status,
                "started_at": started_at,
                "finished_at": utc_now() if status != "running" else None,
                "baseline": args.baseline,
                "workers": args.workers,
                "limit": args.limit,
                "resume": args.resume,
                "num_instances": len(store.completed_ids(validate=True)),
                "num_items": len(store.load_records(validate=True)),
                "judge_failures": judge.failures,
            },
            extra={
                "runner": "scripts/run_research_eval.py",
                "repository_tree_sha256": sha256_tree(root, exclude=(".venv", ".git", "results")),
                "run_fingerprint_sha256": store.load_fingerprint()["sha256"],
            },
        )
        write_manifest(out / "manifest.json", manifest)

    write_run_manifest("running")

    def run_one(instance: DialogueInstance) -> None:
        trace.record("instance_start", instance_id=instance.instance_id)
        records = run_instance(
            adapter_for_current_worker(), instance, items_by_instance.get(instance.instance_id, []), judge
        )
        store.commit(instance.instance_id, records)
        trace.record("instance_complete", instance_id=instance.instance_id, record_count=len(records))

    try:
        if args.workers == 1:
            for instance in tqdm(todo, desc="research-eval", unit="inst"):
                run_one(instance)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(run_one, instance): instance for instance in todo
                }
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="research-eval", unit="inst"):
                    future.result()
    except KeyboardInterrupt:
        trace.record("run_interrupted")
        write_run_manifest("interrupted")
        raise
    except Exception as exc:
        trace.record("run_failed", error_type=type(exc).__name__, error=str(exc))
        write_run_manifest("failed")
        raise

    records = [ItemRecord.model_validate(item) for item in store.load_records(validate=True)]
    metrics = compute_metrics(records)
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(metrics, args.baseline, out)
    store.export_items(out / "items.jsonl")
    write_run_manifest("completed")
    trace.record("run_complete", num_instances=metrics["num_instances"], num_items=metrics["num_items"])
    print(f"Wrote {metrics['num_items']} item records for {metrics['num_instances']} instances to {out}")


if __name__ == "__main__":
    main()

