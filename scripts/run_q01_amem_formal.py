"""Run the frozen serial real-model Q01 A-MEM workloads in order.

The command is intentionally strict: it verifies the frozen input hashes and
official A-MEM file hashes before starting, refuses to run without a model
credential, and invokes the two workload runners one at a time with resume.
When both finish, it invokes ``review_q01_amem_formal.py`` to produce the
state/source/cost gate report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_files_only(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_blocker(path: Path, *, message: str, freeze: Path) -> None:
    payload = {
        "schema_version": 1,
        "status": "blocked_missing_external_model_state",
        "message": message,
        "freeze_path": str(freeze.resolve()),
        "api_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "model": os.environ.get("BEMULA_AGENT_MODEL", "deepseek-flash"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _verify_freeze(
    freeze_path: Path,
    *,
    agent_model: str,
    embedding_backend: str,
    embedding_model: str,
    locomo: Path,
    longmemeval: Path,
    amem_path: Path,
) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    model = freeze["model_config"]["model"]
    if agent_model != model:
        raise RuntimeError(f"agent model {agent_model!r} differs from frozen model {model!r}")
    if embedding_backend == "local":
        local = freeze["model_config"].get("local_embedding_variant", {})
        if embedding_model != local.get("model"):
            raise RuntimeError(
                f"local embedding model {embedding_model!r} differs from frozen local variant {local.get('model')!r}"
            )
    elif embedding_backend == "api":
        if embedding_model != freeze["model_config"].get("embedding_model"):
            raise RuntimeError(
                f"API embedding model {embedding_model!r} differs from frozen model {freeze['model_config'].get('embedding_model')!r}"
            )
    else:
        raise RuntimeError(f"unsupported embedding backend: {embedding_backend!r}")
    for key, path in (("locomo", locomo), ("longmemeval_s", longmemeval)):
        expected = freeze["inputs"][key]["sha256"]
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"{key} input hash mismatch: expected {expected}, got {actual}")
    official = freeze["official_amem"]
    robust = amem_path / "memory_layer_robust.py"
    requirements = amem_path / "requirements.txt"
    if _sha256(robust) != official["memory_layer_robust_sha256"]:
        raise RuntimeError("official A-MEM memory_layer_robust.py hash mismatch")
    if _sha256(requirements) != official["requirements_sha256"]:
        raise RuntimeError("official A-MEM requirements.txt hash mismatch")
    if _source_tree_files_only(amem_path) != official["source_tree_sha256_files_only"]:
        raise RuntimeError("official A-MEM source tree hash mismatch")
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=Path("experiment_results_20261114/q01_amem_freeze/freeze.json"))
    parser.add_argument("--locomo", type=Path, default=Path("experiment_results_20261114/q01_inputs/locomo_explicit_histories.jsonl"))
    parser.add_argument("--longmemeval", type=Path, default=Path("experiment_results_20261114/q01_inputs/longmemeval_s_explicit_histories.jsonl"))
    parser.add_argument("--out-root", type=Path, default=Path("experiment_results_20261114/q01_runs/formal_real"))
    parser.add_argument("--review-out", type=Path, default=Path("experiment_results_20261114/q01_amem_freeze/formal_real_review.json"))
    parser.add_argument("--agent-model", default=os.environ.get("BEMULA_AGENT_MODEL", "deepseek-flash"))
    parser.add_argument("--embedding-backend", default=os.environ.get("BEMULA_EMBED_BACKEND", "api"))
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("BEMULA_HF_EMBED_MODEL", os.environ.get("BEMULA_EMBED_MODEL", "text-embedding-3-small")),
    )
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument("--locomo-limit", type=int, default=None)
    parser.add_argument("--longmemeval-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evidence-level", default="real_model_formal")
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    amem_path = Path(os.environ.get("BEMULA_AMEM_PATH", freeze_path.parents[2] / "_amem_extract" / "A-mem-f9a173dec325d84fcc97e7828775c7b241c33466")).resolve()
    blocker_path = args.out_root / "formal_run_blocker.json"
    try:
        _verify_freeze(
            freeze_path,
            agent_model=args.agent_model,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            locomo=args.locomo.resolve(),
            longmemeval=args.longmemeval.resolve(),
            amem_path=amem_path,
        )
    except Exception as exc:
        _write_blocker(blocker_path, message=f"freeze verification failed: {type(exc).__name__}: {exc}", freeze=freeze_path)
        print(blocker_path)
        return 2
    if not os.environ.get("OPENAI_API_KEY"):
        _write_blocker(blocker_path, message="OPENAI_API_KEY is not configured; no real-model call was started", freeze=freeze_path)
        print(blocker_path)
        return 2

    args.out_root.mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ)
    child_env["BEMULA_EMBED_BACKEND"] = args.embedding_backend
    if args.embedding_backend == "local":
        child_env["BEMULA_HF_EMBED_MODEL"] = args.embedding_model
    else:
        child_env["BEMULA_EMBED_MODEL"] = args.embedding_model
    runner = Path(__file__).with_name("run_q01_amem_paired.py")
    workloads = (
        ("locomo", args.locomo.resolve(), args.locomo_limit),
        ("longmemeval_s", args.longmemeval.resolve(), args.longmemeval_limit),
    )
    for name, input_path, limit in workloads:
        out = args.out_root / name
        command = [
            sys.executable,
            str(runner),
            "--histories", str(input_path),
            "--out", str(out),
            "--retrieve-k", str(args.retrieve_k),
            "--agent-model", args.agent_model,
        ]
        if limit is not None:
            command.extend(["--limit", str(limit)])
        if args.resume:
            command.append("--resume")
        subprocess.run(command, check=True, env=child_env)

    reviewer = Path(__file__).with_name("review_q01_amem_formal.py")
    subprocess.run(
        [
            sys.executable,
            str(reviewer),
            "--locomo-run", str(args.out_root / "locomo"),
            "--longmemeval-run", str(args.out_root / "longmemeval_s"),
            "--locomo-expected", str(args.locomo_limit if args.locomo_limit is not None else 10),
            "--longmemeval-expected", str(args.longmemeval_limit if args.longmemeval_limit is not None else 500),
            "--out", str(args.review_out),
            "--evidence-level", args.evidence_level,
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
