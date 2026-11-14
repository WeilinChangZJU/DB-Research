"""Run the offline Q01 platform fixture without any model/API calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bemula.research import CostRecorder, SourceEvent, run_toy_experiment


def fixture_events() -> list[SourceEvent]:
    return [
        SourceEvent("e0", "src-a", "approval limit is ten thousand", metadata={"session": "s0"}),
        SourceEvent("e1", "src-b", "approval limit is ten thousand", metadata={"session": "s1"}),
        SourceEvent("e2", "src-c", "office is in Shanghai", metadata={"session": "s2"}),
        SourceEvent("e3", "src-d", "approval limit is ten thousand", metadata={"session": "s3"}),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/q01_fixture")
    parser.add_argument("--delete", action="append", default=None)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cost = CostRecorder(out / "cost.jsonl")
    deleted = args.delete if args.delete else ["src-a"]
    result = run_toy_experiment(fixture_events(), deleted, cost=cost)
    (out / "summary.json").write_text(
        json.dumps(result.summary(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cost.write_summary(out / "cost_summary.json")
    print(f"Wrote offline Q01 fixture to {out}")


if __name__ == "__main__":
    main()
