# Bemula / Q01 source-level memory-maintenance platform

This repository is the latest source-only snapshot for the Q01 validation
platform. It contains the benchmark adapters, source-aware A-MEM wrapper,
serial M0–M3 maintenance runner, provenance/state/cost instrumentation, tests,
and reproduction documentation. It intentionally excludes datasets, generated
histories, API keys, raw traces, experiment results, and external repository
clones.

Start with [REPRODUCTION.md](REPRODUCTION.md). It documents the pinned A-MEM
commit, environment variables, external dataset boundary, explicit deletion
history generation, focused real-model pilot, serial runner, output contract,
full-cost accounting, and interpretation limits.

Current research status: the focused real-provider pilot completed execution on
one 32-event window per workload, but Q01 remains `PROMISING / UNVERIFIED`.
The pilot is evidence that the measurement path runs; it is not a claim of
general correctness, novelty, or superiority.

## Setup

```bash
uv sync                   # core dependencies (Python >= 3.10)
cp .env.example .env      # fill in OPENAI_BASE_URL / OPENAI_API_KEY / BEMULA_*_MODEL
```

All LLM calls (generation, judging, baselines) go through the single
OpenAI-compatible endpoint configured in `.env`.

## Run evaluation

Basic form:

```bash
uv run python -m scripts.run_eval \
    --baseline simplemem \
    --dialogues data/core/dialogues.jsonl \
    --questions data/core/questions.jsonl \
    --out results/simplemem_core \
    --workers 8 \
    --resume
```

Available baselines: `rag`, `mem0`, `amem`, `simplemem`, `lightmem`, `langmem`, `fullcontext`.

Key options:

- `--limit N`: evaluate only the first N instances (smoke runs).
- `--workers N`: evaluate N instances in parallel (one adapter per worker).
- `--resume`: skip instances already recorded in the out dir's `items.jsonl`.

Outputs under `--out`: `items.jsonl` (per-item records), `metrics.json`,
`report.md`, `run_meta.json`.

### Baseline-specific requirements

- `amem` / `simplemem` / `memoryos` / `lightmem` load from their official
  repository clones; set `BEMULA_AMEM_PATH` / `BEMULA_SIMPLEMEM_PATH` /
  `BEMULA_MEMORYOS_PATH` / `BEMULA_LIGHTMEM_PATH` in `.env` (fallback: a
  sibling directory like `../SimpleMem`).
- `simplemem` needs extra packages passed via `--with` (see the production
  example below); `lancedb` must stay below 0.37.
- Alternatively install everything once: `uv sync --extra baselines`.

### Production run (background, logged, resumable)

```bash
mkdir -p logs results/simplemem_core_deepseek

BEMULA_THINKING=disabled \
PYTHONUNBUFFERED=1 \
setsid nohup uv run --frozen \
  --with "lancedb==0.25.3" \
  --with "pyarrow==17.0.0" \
  --with pylance \
  --with dateparser \
  --with tantivy \
  python -m scripts.run_eval \
  --config configs/default.yaml \
  --baseline simplemem \
  --dialogues data/core/dialogues.jsonl \
  --questions data/core/questions.jsonl \
  --out results/simplemem_core_deepseek \
  --workers 8 \
  --resume \
  > logs/simplemem_core_deepseek.log 2>&1 < /dev/null &

echo "pid=$!"
```

## Data generation

Only needed when regenerating the benchmark data (ready-made splits live in
`data/core`, `data/full`, `data/quickcheck`, `data/smallcheck`):

```bash
# 1. User profiles
uv run python -m scripts.generate_profiles --config configs/default.yaml \
    --out data/profiles.jsonl

# 2. Dialogues with embedded forget requests (+ paired control variants)
uv run python -m scripts.generate_dialogues --config configs/default.yaml \
    --profiles data/profiles.jsonl --out data/dialogues.jsonl

# 3. Test items
uv run python -m scripts.generate_questions --config configs/default.yaml \
    --profiles data/profiles.jsonl --dialogues data/dialogues.jsonl \
    --out data/questions.jsonl
```

Every generation script accepts `--limit N` for small trial runs.

## ResearchProtocol adaptation layer

The P0 research layer is opt-in and leaves the original evaluator unchanged. See [RESEARCH_PROTOCOL_PLAN.md](RESEARCH_PROTOCOL_PLAN.md), [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md), and [docs/RESEARCH_DATASETS.md](docs/RESEARCH_DATASETS.md).

Run an audited smoke/evaluation job with per-instance checkpoints and a provenance manifest:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python -m scripts.run_research_eval \
  --config configs/deepseek.yaml \
  --baseline fullcontext \
  --dialogues data/quick_check/dialogues.jsonl \
  --questions data/quick_check/questions.jsonl \
  --out results/research_smoke \
  --limit 1 --workers 1
```

## Q01 offline platform fixture

Run the deterministic platform contract without DeepSeek or any external model:

```bash
.venv/bin/python -m scripts.run_q01_fixture \
  --out results/q01_fixture \
  --delete src-a
```

The fixture writes `summary.json`, `cost.jsonl`, and `cost_summary.json`. It exercises M0 full rebuild, M1 tombstone/filter, M2 replay, M3 read-set replay, canonical state comparison, and cost counters. It is a platform test, not a memory quality result.

The full Q01 layering and current verification receipt are documented in [docs/Q01_PLATFORM_ARCHITECTURE.md](docs/Q01_PLATFORM_ARCHITECTURE.md) and [Q01_PLATFORM_RECEIPT_20261114.md](Q01_PLATFORM_RECEIPT_20261114.md).

## A-MEM source-aware offline fixture

The A-MEM research wrapper and its single-history M0–M3 check are available
without importing the official model or calling DeepSeek:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_amem_fixture.py \
  --out results/amem_fixture/summary.json
```

See [docs/AMEM_RESEARCH_WRAPPER.md](docs/AMEM_RESEARCH_WRAPPER.md) and
[AMEM_WRAPPER_RECEIPT_20261114.md](AMEM_WRAPPER_RECEIPT_20261114.md) for the
provenance, snapshot, decision-trace, dependency-introspection, and
unsupported-capability boundaries.

## Q01 deletion scenarios and paired reports

Generate deterministic deletion masks and M0–M3 failure/cost reports without
an external model:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_q01_scenarios.py \
  --out results/q01_scenarios
```

The implementation roadmap is recorded in
[Q01_IMPLEMENTATION_ROADMAP_20261114.md](Q01_IMPLEMENTATION_ROADMAP_20261114.md).

To exercise the official robust A-MEM code without a network client, use the
instrumentation smoke:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_amem_offline_smoke.py \
  --repo /path/to/A-mem --out results/amem_offline_smoke/summary.json
```

The current implementation receipt is in
[PLATFORM_IMPLEMENTATION_RECEIPT_20261114.md](PLATFORM_IMPLEMENTATION_RECEIPT_20261114.md).

For a no-API check of the external workload boundary, load one instance from
each dataset and emit source/evidence-preserving deletion manifests:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_dataset_adapters_smoke.py \
  --locomo /path/to/locomo10.json \
  --longmemeval /path/to/longmemeval_s_cleaned.json \
  --out results/q01_dataset_adapter_smoke --limit 1
```

This validates the LoCoMo and LongMemEval-S adapters and scenario generation;
it does not run a memory builder.

For a model-free paired maintenance pilot over one or more adapter histories:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_q01_paired.py \
  --dataset locomo --source /path/to/locomo10.json \
  --out results/q01_locomo_paired --limit 1 --max-scenarios 3
```

The output labels itself `controlled_replay`, preserves independent-history
clustering, and reports when a scaled source prefix is used. For a source-level
A-MEM run with explicit deletion masks, use the serial
`scripts/run_q01_amem_paired.py` runner described in
[docs/Q01_PLATFORM_ARCHITECTURE.md](docs/Q01_PLATFORM_ARCHITECTURE.md). The
answer-level `scripts/run_research_eval.py` remains a separate behavior runner
and is not a substitute for the Q01 M0--M3 matrix.
