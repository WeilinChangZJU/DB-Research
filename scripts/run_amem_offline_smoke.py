"""Exercise the official robust A-MEM logic with deterministic local doubles.

This is an instrumentation smoke test, not a quality experiment. It performs
no network request and never reads the DeepSeek configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from bemula.baselines.amem import AmemAdapter
from bemula.research import AmemResearchWrapper
from bemula.schemas import Turn


class ScriptedLLM:
    def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "EVOLUTION" in prompt.upper():
            return "NO_EVOLUTION"
        if kwargs.get("task") == "amem_answer":
            return "fixture-answer"
        # A-MEM's robust parser falls back to deterministic heuristics for
        # content analysis when the scripted response is empty.
        return ""


class LexicalEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.zeros(32, dtype=np.float32)
            for token in text.lower().split():
                slot = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % 32
                vector[slot] += 1.0
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.vstack(vectors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo or Path(os.environ.get("BEMULA_AMEM_PATH", "A-mem"))
    if not repo.is_dir():
        raise SystemExit(f"official A-MEM repository not found: {repo}")

    adapter = AmemAdapter(
        ScriptedLLM(),
        repo_path=repo,
        embedder=LexicalEmbedder(),
        progress=False,
    )
    wrapper = AmemResearchWrapper(adapter, retrieve_k=2)
    wrapper.observe(Turn(speaker="user", text="office is in Shanghai", timestamp="t0"), source_id="src-a")
    wrapper.observe(Turn(speaker="user", text="approval limit is ten thousand", timestamp="t1"), source_id="src-b")
    result = wrapper.query("Where is the office?")
    payload = {
        "offline": True,
        "api_calls": 0,
        "official_repo": str(repo),
        "answer": result.answer,
        "llm_trace_count": len(wrapper.introspect()["llm_trace"]),
        "event_types": [event.event_type.value for event in wrapper.decision_events],
        "canonical_hash": wrapper.canonical_hash(dataset="offline", instance_id="smoke"),
        "supports": wrapper.introspect()["supports"],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(args.out)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
