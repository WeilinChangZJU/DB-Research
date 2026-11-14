"""Provenance manifest helpers for auditable Bemula research runs.

The manifest intentionally records metadata and hashes, never secret values.
All functions are best-effort for optional runtime details so a manifest can
also be produced in a minimal offline test environment.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: str | Path, *, exclude: Iterable[str] = ()) -> str:
    """Hash a directory deterministically, including relative paths.

    ``exclude`` contains path names relative to ``root``.  The relative path
    is included before the file bytes so renaming a file changes the digest.
    """
    root = Path(root)
    excluded = {Path(item).as_posix() for item in exclude}
    digest = hashlib.sha256()
    if root.is_file():
        return sha256_file(root)
    if not root.exists():
        return "missing"
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in excluded or any(rel.startswith(prefix.rstrip("/") + "/") for prefix in excluded):
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\n")
    return digest.hexdigest()


def _git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _package_versions() -> dict[str, str]:
    """Return installed package versions without requiring pip."""
    names = ("bemula", "openai", "pydantic", "python-dotenv", "torch", "transformers",
             "sentence-transformers", "scikit-learn", "numpy", "pytest")
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def _safe_env(names: Iterable[str]) -> dict[str, Any]:
    """Record configured environment variables without exposing credentials."""
    out: dict[str, Any] = {}
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        if any(token in name.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            out[name] = {"configured": bool(value), "sha256": hashlib.sha256(value.encode()).hexdigest()[:12]}
        else:
            out[name] = value
    return out


def build_manifest(
    *,
    root: str | Path,
    config_paths: Iterable[str | Path] = (),
    data_paths: Iterable[str | Path] = (),
    prompt_paths: Iterable[str | Path] = (),
    seed: int | None = None,
    execution: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable run manifest.

    Paths are recorded as supplied plus a SHA-256 digest.  A missing path is
    represented explicitly rather than silently omitted.
    """
    root_path = Path(root).resolve()

    def digest_entries(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for raw in paths:
            path = Path(raw)
            if not path.is_absolute():
                path = root_path / path
            path = path.resolve()
            entry: dict[str, Any] = {"path": str(path)}
            if path.is_file():
                entry.update({"kind": "file", "sha256": sha256_file(path), "bytes": path.stat().st_size})
            elif path.is_dir():
                entry.update({"kind": "directory", "sha256": sha256_tree(path)})
            else:
                entry.update({"kind": "missing", "sha256": None})
            entries.append(entry)
        return entries

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "root": str(root_path),
            "git_commit": _git_value(root_path, "rev-parse", "HEAD"),
            "git_dirty": bool(_git_value(root_path, "status", "--porcelain")),
        },
        "inputs": {
            "config": digest_entries(config_paths),
            "data": digest_entries(data_paths),
            "prompts": digest_entries(prompt_paths),
        },
        "execution": {"seed": seed, **(execution or {})},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _package_versions(),
        },
        "environment": _safe_env((
            "OPENAI_BASE_URL", "BEMULA_GEN_MODEL", "BEMULA_AGENT_MODEL",
            "BEMULA_JUDGE_MODEL", "BEMULA_EMBED_BACKEND", "BEMULA_EMBED_MODEL",
            "BEMULA_HF_EMBED_MODEL", "BEMULA_THINKING", "CUDA_VISIBLE_DEVICES",
            "OPENAI_API_KEY", "EMBED_PROXY_API_KEY",
        )),
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Atomically write a manifest as UTF-8 JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    # Windows requires a writable descriptor for fsync; use r+b after the
    # atomic payload has been written so the same code works cross-platform.
    with temp.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)
