"""Crash-safe per-instance checkpoints for long Bemula evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class CheckpointValidationError(RuntimeError):
    """Raised when a checkpoint manifest or instance file is inconsistent."""


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _safe_name(instance_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance_id).strip("._")[:80] or "instance"
    digest = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}.jsonl"


class CheckpointStore:
    """Store completed instances atomically and derive a consolidated JSONL.

    A record is considered resumable only after the per-instance JSONL file is
    durable and a matching entry is committed to ``completed_instances.jsonl``.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.instance_dir = self.root / "checkpoints" / "instances"
        self.manifest_path = self.root / "checkpoints" / "completed_instances.jsonl"
        self.state_dir = self.root / "checkpoints" / "states"
        self.state_manifest_path = self.root / "checkpoints" / "states.jsonl"
        self.fingerprint_path = self.root / "checkpoints" / "run_fingerprint.json"
        self._lock = threading.Lock()

    def ensure_fingerprint(self, fingerprint: str | dict[str, Any]) -> dict[str, Any]:
        """Pin the immutable run configuration used by resumable checkpoints.

        A resume is valid only when the caller presents the same fingerprint.
        The first call initializes the receipt; later mismatches fail loudly so
        results from another model, prompt, mask, or code tree cannot be mixed
        into the run.
        """
        if isinstance(fingerprint, dict):
            canonical = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            payload = {"sha256": digest, "value": fingerprint}
        else:
            digest = str(fingerprint)
            payload = {"sha256": digest}
        with self._lock:
            if self.fingerprint_path.exists():
                try:
                    existing = json.loads(self.fingerprint_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise CheckpointValidationError("invalid run fingerprint receipt") from exc
                if existing.get("sha256") != payload["sha256"]:
                    raise CheckpointValidationError(
                        "checkpoint run fingerprint mismatch; refusing unsafe resume"
                    )
                return existing
            data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
            _atomic_write(self.fingerprint_path, data)
        return payload

    def load_fingerprint(self) -> dict[str, Any] | None:
        """Return the pinned run fingerprint, if this store has one."""
        if not self.fingerprint_path.exists():
            return None
        try:
            value = json.loads(self.fingerprint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CheckpointValidationError("invalid run fingerprint receipt") from exc
        if not isinstance(value, dict) or not isinstance(value.get("sha256"), str):
            raise CheckpointValidationError("malformed run fingerprint receipt")
        return value

    @staticmethod
    def _encode(records: Iterable[dict[str, Any]]) -> bytes:
        return b"".join(
            (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            for record in records
        )

    def _read_manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CheckpointValidationError(f"invalid manifest line {line_no}: {exc}") from exc
                if not isinstance(item, dict) or not isinstance(item.get("instance_id"), str):
                    raise CheckpointValidationError(f"invalid manifest entry at line {line_no}")
                entries.append(item)
        return entries

    def completed_ids(self, *, validate: bool = True) -> set[str]:
        ids: set[str] = set()
        for entry in self._read_manifest():
            path = self.instance_dir / entry["file"]
            if not path.is_file():
                continue
            if validate:
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
                    continue
                if data.count(b"\n") != int(entry.get("record_count", -1)):
                    continue
            ids.add(entry["instance_id"])
        return ids

    def commit(self, instance_id: str, records: Iterable[dict[str, Any] | Any]) -> dict[str, Any]:
        """Commit one complete instance and return its manifest entry."""
        normalized: list[dict[str, Any]] = []
        for record in records:
            if hasattr(record, "model_dump"):
                record = record.model_dump(mode="json")
            if not isinstance(record, dict):
                raise TypeError("checkpoint records must be mappings or Pydantic models")
            normalized.append(record)
        data = self._encode(normalized)
        file_name = _safe_name(instance_id)
        instance_path = self.instance_dir / file_name
        with self._lock:
            _atomic_write(instance_path, data)
            entry = {
                "instance_id": instance_id,
                "file": file_name,
                "record_count": len(normalized),
                "sha256": hashlib.sha256(data).hexdigest(),
                "committed_at": datetime.now(timezone.utc).isoformat(),
            }
            entries = [e for e in self._read_manifest() if e.get("instance_id") != instance_id]
            entries.append(entry)
            manifest_data = self._encode(entries)
            _atomic_write(self.manifest_path, manifest_data)
        return entry

    def load_records(self, *, validate: bool = True) -> list[dict[str, Any]]:
        """Load records only from committed, valid instance checkpoints."""
        out: list[dict[str, Any]] = []
        for entry in self._read_manifest():
            path = self.instance_dir / entry["file"]
            if not path.is_file():
                if validate:
                    raise CheckpointValidationError(f"missing checkpoint file: {path}")
                continue
            data = path.read_bytes()
            if validate and (
                hashlib.sha256(data).hexdigest() != entry.get("sha256")
                or data.count(b"\n") != int(entry.get("record_count", -1))
            ):
                raise CheckpointValidationError(f"checkpoint integrity failure: {path}")
            for line_no, line in enumerate(data.decode("utf-8").splitlines(), 1):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise CheckpointValidationError(f"invalid record {path}:{line_no}") from exc
        return out

    def export_items(self, path: str | Path) -> int:
        """Atomically materialize committed records as the legacy items.jsonl."""
        records = self.load_records(validate=True)
        data = self._encode(records)
        _atomic_write(Path(path), data)
        return len(records)

    def validate(self) -> list[str]:
        """Return validation errors without throwing."""
        errors: list[str] = []
        try:
            entries = self._read_manifest()
        except CheckpointValidationError as exc:
            return [str(exc)]
        for entry in entries:
            path = self.instance_dir / str(entry.get("file", ""))
            if not path.is_file():
                errors.append(f"missing file for {entry.get('instance_id')}")
                continue
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
                errors.append(f"hash mismatch for {entry.get('instance_id')}")
            if data.count(b"\n") != int(entry.get("record_count", -1)):
                errors.append(f"record count mismatch for {entry.get('instance_id')}")
        return errors

    def commit_state(self, state_id: str, state: Any, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Persist a semantic state snapshot with an integrity receipt.

        This is separate from per-instance answer checkpoints because a Q01
        maintenance arm must be able to restore the complete state oracle,
        including source provenance, artifacts, relations, queries and trace.
        """

        if hasattr(state, "model_dump"):
            payload = state.model_dump(mode="json")
        elif isinstance(state, dict):
            payload = state
        else:
            raise TypeError("state must be a Pydantic model or mapping")
        envelope = {
            "schema_version": 1,
            "state_id": state_id,
            "metadata": metadata or {},
            "state": payload,
        }
        data = (json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        file_name = _safe_name(state_id).replace(".jsonl", ".json")
        path = self.state_dir / file_name
        with self._lock:
            _atomic_write(path, data)
            entry = {
                "state_id": state_id,
                "file": file_name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "byte_count": len(data),
                "committed_at": datetime.now(timezone.utc).isoformat(),
            }
            entries: list[dict[str, Any]] = []
            if self.state_manifest_path.exists():
                for line in self.state_manifest_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        entries.append(json.loads(line))
            entries = [item for item in entries if item.get("state_id") != state_id]
            entries.append(entry)
            _atomic_write(self.state_manifest_path, self._encode(entries))
        return entry

    def load_state(self, state_id: str, *, validate: bool = True) -> dict[str, Any]:
        """Load a committed semantic state envelope and verify its hash."""

        if not self.state_manifest_path.exists():
            raise CheckpointValidationError(f"state manifest does not exist: {self.state_manifest_path}")
        entry = None
        for line in self.state_manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("state_id") == state_id:
                entry = json.loads(line)
        if entry is None:
            raise CheckpointValidationError(f"unknown state checkpoint: {state_id}")
        path = self.state_dir / str(entry["file"])
        if not path.is_file():
            raise CheckpointValidationError(f"missing state checkpoint: {path}")
        data = path.read_bytes()
        if validate and hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            raise CheckpointValidationError(f"state checkpoint integrity failure: {path}")
        try:
            envelope = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise CheckpointValidationError(f"invalid state checkpoint: {path}") from exc
        if not isinstance(envelope, dict) or envelope.get("state_id") != state_id:
            raise CheckpointValidationError(f"state checkpoint identity mismatch: {path}")
        return envelope
