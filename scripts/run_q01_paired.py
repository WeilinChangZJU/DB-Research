"""Run the deterministic Q01 paired matrix on a real dataset adapter.

This runner consumes the source/evidence-preserving adapters and executes the
model-free deterministic builder.  It is a controlled replay and provenance
screen; it does not claim A-MEM or any other agent memory method was evaluated.
Use ``run_q01_amem_paired.py`` only after an A-MEM history JSONL and model
configuration have been frozen.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bemula.research import (
    build_manifest,
    run_paired_histories,
    sha256_tree,
    write_manifest,
    write_paired_results,
)
from bemula.research.datasets import load_locomo, load_longmemeval_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("locomo", "longmemeval_s"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-events", type=int, default=None,
                        help="optional scaled source prefix for a cheap pilot")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.max_scenarios is not None and args.max_scenarios < 1:
        raise SystemExit("--max-scenarios must be >= 1")
    if args.max_events is not None and args.max_events < 1:
        raise SystemExit("--max-events must be >= 1")

    if args.dataset == "locomo":
        instances = load_locomo(args.source, limit=args.limit)
    else:
        instances = load_longmemeval_s(args.source, limit=args.limit)
    payload = run_paired_histories(
        instances,
        seed=args.seed,
        max_scenarios=args.max_scenarios,
        max_events=args.max_events,
    )
    payload["dataset"] = args.dataset
    payload["source_path"] = str(args.source.resolve())
    write_paired_results(args.out / "summary.json", payload)
    write_manifest(
        args.out / "manifest.json",
        build_manifest(
            root=Path.cwd().resolve(),
            data_paths=[args.source],
            seed=args.seed,
            execution={
                "runner": "scripts/run_q01_paired.py",
                "dataset": args.dataset,
                "limit": args.limit,
                "max_scenarios": args.max_scenarios,
                "max_events": args.max_events,
                "evidence_level": "controlled_replay",
                "independent_history_count": payload["independent_history_count"],
                "scenario_count": payload["scenario_count"],
            },
            extra={
                "repository_tree_sha256": sha256_tree(
                    Path.cwd().resolve(), exclude=(".venv", ".git", "results")
                ),
            },
        ),
    )
    print(args.out / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
