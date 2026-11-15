"""Create a checked-in-schema freeze receipt for a formal Q01 A-MEM run.

The receipt contains only paths, hashes, and non-secret model settings.  It
does not copy datasets or the external A-MEM source into this repository.  A
formal run verifies the resulting receipt before creating any provider client.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from bemula.research.manifest import sha256_file, sha256_tree
from scripts.run_q01_amem_formal import PINNED_AMEM_COMMIT


# The provider replay policy is part of the correctness oracle.  A repeated
# M0 build is a fresh runtime/build, but it is backed by the first build's
# request-key response cassette so provider sampling cannot masquerade as a
# state-maintenance mismatch.  Keep the value stable across freeze receipts,
# runner fingerprints, and result summaries.
M0_REPEAT_PROVIDER_REPLAY_POLICY = "m0_repeat_request_key_cassette"
# Keep the receipt version stable for existing formal-run consumers; the
# added model-config fields are optional to readers that already accept
# unknown keys, while new receipts carry them explicitly.
FREEZE_SCHEMA_VERSION = 1


def _commit(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"A-MEM path is not a readable git checkout: {root}") from exc
    value = completed.stdout.strip()
    if not value:
        raise ValueError(f"A-MEM checkout has no HEAD: {root}")
    return value


def _input_entry(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def build_freeze(
    *,
    locomo: Path,
    longmemeval: Path,
    amem: Path,
    model: str,
    embedding_backend: str,
    embedding_model: str,
    thinking: str,
    base_url: str | None,
    retrieve_k: int,
    expected_commit: str,
    amem_temperature: float = 0.0,
    provider_replay_policy: str = "m0_repeat_request_key_cassette",
    random_seed: int | None = None,
) -> dict[str, Any]:
    if provider_replay_policy != M0_REPEAT_PROVIDER_REPLAY_POLICY:
        raise ValueError(
            "unsupported provider_replay_policy: "
            f"{provider_replay_policy!r}; expected {M0_REPEAT_PROVIDER_REPLAY_POLICY!r}"
        )
    try:
        amem_temperature = float(amem_temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("amem_temperature must be a finite non-negative number") from exc
    if not (amem_temperature >= 0.0 and amem_temperature != float("inf")):
        raise ValueError("amem_temperature must be a finite non-negative number")
    if random_seed is not None and not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer or null")
    amem = amem.resolve()
    if not amem.is_dir():
        raise ValueError(f"A-MEM directory not found: {amem}")
    commit = _commit(amem)
    if commit != expected_commit:
        raise ValueError(f"A-MEM commit mismatch: expected {expected_commit}, got {commit}")
    robust = amem / "memory_layer_robust.py"
    requirements = amem / "requirements.txt"
    if not robust.is_file() or not requirements.is_file():
        raise ValueError("A-MEM checkout must contain memory_layer_robust.py and requirements.txt")
    for path in (locomo, longmemeval):
        if not path.is_file():
            raise ValueError(f"dataset input not found: {path}")
    model_config: dict[str, Any] = {
        "model": model,
        "thinking": thinking,
        "embedding_backend": embedding_backend,
        "retrieve_k": retrieve_k,
        "amem_temperature": amem_temperature,
        "provider_replay_policy": provider_replay_policy,
        "random_seed": random_seed,
    }
    if base_url:
        model_config["base_url"] = base_url
    if embedding_backend == "local":
        model_config["local_embedding_variant"] = {"model": embedding_model}
    else:
        model_config["embedding_model"] = embedding_model
    return {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "official_amem": {
            "path": str(amem),
            "commit": commit,
            "memory_layer_robust_sha256": sha256_file(robust),
            "requirements_sha256": sha256_file(requirements),
            "source_tree_sha256_files_only": sha256_tree(amem),
        },
        "model_config": model_config,
        "inputs": {
            "locomo": _input_entry(locomo),
            "longmemeval_s": _input_entry(longmemeval),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--longmemeval", type=Path, required=True)
    parser.add_argument("--amem", type=Path, default=os.environ.get("BEMULA_AMEM_PATH", "A-mem"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("BEMULA_AGENT_MODEL", "deepseek-flash"))
    parser.add_argument("--embedding-backend", choices=("api", "local"), default=os.environ.get("BEMULA_EMBED_BACKEND", "api"))
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("BEMULA_HF_EMBED_MODEL", os.environ.get("BEMULA_EMBED_MODEL", "text-embedding-3-small")),
    )
    parser.add_argument("--thinking", default=os.environ.get("BEMULA_THINKING", "disabled"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--retrieve-k", type=int, default=10)
    parser.add_argument(
        "--amem-temperature",
        type=float,
        default=float(os.environ.get("BEMULA_AMEM_TEMPERATURE", "0.0")),
        help="Frozen default temperature for official A-MEM metadata/evolution calls.",
    )
    parser.add_argument(
        "--provider-replay-policy",
        default=os.environ.get(
            "BEMULA_PROVIDER_REPLAY_POLICY", "m0_repeat_request_key_cassette"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=(
            int(os.environ["BEMULA_RANDOM_SEED"])
            if os.environ.get("BEMULA_RANDOM_SEED") not in (None, "")
            else None
        ),
    )
    parser.add_argument("--expected-commit", default=PINNED_AMEM_COMMIT)
    args = parser.parse_args()
    if args.retrieve_k < 1:
        raise SystemExit("--retrieve-k must be >= 1")
    freeze = build_freeze(
        locomo=args.locomo.expanduser().resolve(),
        longmemeval=args.longmemeval.expanduser().resolve(),
        amem=args.amem.expanduser().resolve(),
        model=args.model,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        thinking=args.thinking,
        base_url=args.base_url,
        retrieve_k=args.retrieve_k,
        expected_commit=args.expected_commit,
        amem_temperature=args.amem_temperature,
        provider_replay_policy=args.provider_replay_policy,
        random_seed=args.random_seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(freeze, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

