# Q01 source-level memory-maintenance reproduction guide

This repository contains the latest source code for the Q01 validation platform. It separates source-level deletion maintenance from answer-level memory unlearning. The main experiment compares four serial maintenance arms on the same event history and deletion set:

- **M0**: independent rebuild from `H \ D`; this is the correctness reference.
- **M1**: live source tombstone/filter with retriever rebuild and no repair.
- **M2**: conservative suffix replay after the earliest affected step.
- **M3**: materialized read/control-closure replay baseline.

The repository intentionally does not contain benchmark datasets, API keys, large experiment results, or provider logs. Obtain the datasets separately and keep them outside Git. Never commit `.env`, API keys, raw traces, or generated JSONL histories that contain source text.

## 1. Environment

The supported runtime is Python 3.10 or newer. `uv` is recommended because the lock file records the tested dependency set.

```bash
git clone https://github.com/WeilinChangZJU/DB-Research.git
cd DB-Research
uv sync --extra dev
```

For the official A-MEM adapter, install the pinned source separately and set its path. The validation cycle used commit `f9a173dec325d84fcc97e7828775c7b241c33466`.

```bash
git clone https://github.com/WujiangXu/A-mem.git ../A-mem
cd ../A-mem
git checkout f9a173dec325d84fcc97e7828775c7b241c33466
cd ../DB-Research
```

Copy an example environment file locally; do not add the resulting file to Git:

```bash
cp .env.deepseek.example .env
# edit .env locally, or export the variables in the shell
set -a; . ./.env; set +a
export BEMULA_AMEM_PATH="$PWD/../A-mem"
```

The minimum real-provider variables are:

```text
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=<local secret>
BEMULA_AGENT_MODEL=deepseek-flash
BEMULA_THINKING=disabled
```

DeepSeek chat is compatible with the agent path used here, but its public API does not provide the declared `text-embedding-3-small` endpoint. For a cost-bounded reproduction, stage the cached local model and use:

```text
BEMULA_EMBED_BACKEND=local
BEMULA_HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

If an OpenAI-compatible embedding service is available, use `BEMULA_EMBED_BACKEND=api` and set `BEMULA_EMBED_MODEL` to the model actually served by that endpoint. Record the backend and model in the run fingerprint.

## 2. Verify the source tree before any model call

Run the deterministic tests first:

```bash
PYTHONPATH=. uv run pytest -q tests
PYTHONPATH=. uv run python -m compileall -q .
```

The Q01 platform tests cover canonical state comparison, source masks, typed provenance, dependency invalidation, checkpoint fingerprints, cost scopes, query evidence, and trace redaction. A real run should not begin if this gate fails.

## 3. Prepare external datasets

Place the externally downloaded files outside the repository, for example:

```text
../datasets/locomo10.json
../datasets/longmemeval_s_cleaned.json
```

The LoCoMo adapter treats each conversation turn as a source event. The LongMemEval-S adapter preserves session-level answer overlays; those overlays are not an exact turn-level evidence oracle. Keep the dataset version and source SHA-256 in the run receipt.

Generate explicit source-level histories. The output rows contain stable source IDs, ordered events, an explicit deletion set, question overlays without gold answers, and manifest hashes:

```bash
PYTHONPATH=. uv run python scripts/build_q01_amem_histories.py \
  --dataset locomo \
  --source ../datasets/locomo10.json \
  --out ../q01_inputs/locomo_explicit_histories.jsonl \
  --mask-policy ordinal_quartile_single_v1

PYTHONPATH=. uv run python scripts/build_q01_amem_histories.py \
  --dataset longmemeval_s \
  --source ../datasets/longmemeval_s_cleaned.json \
  --out ../q01_inputs/longmemeval_s_explicit_histories.jsonl \
  --mask-policy ordinal_quartile_single_v1
```

For a cost-bounded mechanism pilot, derive one short window around a frozen deletion source. This is a pilot and must not be reported as a full-dataset result:

```bash
PYTHONPATH=. uv run python scripts/build_q01_focused_window_pilot.py \
  --locomo ../q01_inputs/locomo_explicit_histories.jsonl \
  --longmemeval ../q01_inputs/longmemeval_s_explicit_histories.jsonl \
  --out-dir ../q01_inputs/focused_window_pilot \
  --window-events 32 \
  --locomo-start 96 \
  --longmemeval-start 129
```

The helper refuses to create a row unless the frozen deleted source lies in the selected window. Do not infer `deleted_source_ids` from model answers or retrieval results.

## 4. Freeze configuration and run serial M0–M3

The formal orchestrator verifies the agent model, embedding backend/model, input hashes, official A-MEM source hashes, and the source-tree digest before starting. Prepare a local freeze JSON from the repository template or a reviewed run receipt; paths and hashes must describe the files on the current machine.

Run LoCoMo first and LongMemEval-S second. `--limit 1` means one history, not one event; use the focused-window inputs above to keep a real pilot affordable:

```bash
PYTHONPATH=. uv run python scripts/run_q01_amem_formal.py \
  --freeze ../q01_freeze/freeze_focused_window.json \
  --locomo ../q01_inputs/focused_window_pilot/locomo_explicit_histories_focused_pilot.jsonl \
  --longmemeval ../q01_inputs/focused_window_pilot/longmemeval_s_explicit_histories_focused_pilot.jsonl \
  --locomo-limit 1 \
  --longmemeval-limit 1 \
  --embedding-backend local \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --out-root ../q01_runs/formal_real_focused_window \
  --review-out ../q01_freeze/formal_real_focused_window_review.json \
  --agent-model deepseek-flash \
  --retrieve-k 10 \
  --evidence-level real_model_focused_window \
  --resume
```

The command is intentionally serial. Each arm receives an isolated adapter; M0 is rebuilt independently, while M1–M3 record their separately executed baseline setup and maintenance costs. Resume is safe only when the run fingerprint matches the input, model, retrieval setting and source tree.

## 5. Review outputs

Each workload directory should contain:

```text
checkpoints/run_fingerprint.json
checkpoints/completed_instances.jsonl
checkpoints/instances/*.jsonl
trace.jsonl
cost.jsonl
cost_summary.json
manifest.json
```

The reviewer emits JSON and Markdown. Check all of the following before interpreting a result:

1. Both workloads have the expected number of completed histories.
2. M0 is exact to its own reference and all mutable-arm setup costs are included in the full totals.
3. `visible_state_hash`/canonical state mismatches are inspected by object and relationship, not reduced to an answer score.
4. Query probes are evidence-complete and their read/context source sets do not contain the deleted source. A missing evidence mapping is an unknown, not proof of zero exposure.
5. Query answer-model tokens are present in the query cost scope. Older runs made before the query-chat hook was added have unknown query cost and must not report it as zero.
6. M2/M3 replay mode and status are read from per-arm metadata. A label alone is not evidence that a general exact dependency baseline has been proved.

For a focused pilot, the correct scientific interpretation is execution evidence only. It does not estimate natural occurrence rates, establish generality, prove novelty, or justify a superiority claim. Calibrate repeated M0 variability before attributing real-model mismatches to maintenance.

## 6. Offline checks and boundaries

These commands do not require a provider key and are useful for collaborator setup checks:

```bash
PYTHONPATH=. uv run python scripts/run_q01_fixture.py \
  --out ../q01_runs/q01_fixture --delete src-a

PYTHONPATH=. uv run python scripts/run_amem_fixture.py \
  --out ../q01_runs/amem_fixture/summary.json

PYTHONPATH=. uv run python scripts/run_q01_scenarios.py \
  --out ../q01_runs/q01_scenarios
```

These verify platform contracts only. They are not evidence that a memory builder has superior forgetting quality. The answer-level `scripts/run_research_eval.py` path is a separate behavior benchmark and is not a substitute for source-level paired M0–M3 maintenance.

## 7. Reproducibility and cleanup rules

- Keep datasets, API keys, `.env`, raw traces, generated histories, and result directories outside Git.
- Preserve the exact A-MEM commit, model name, embedding backend/model, retrieval `k`, input hashes, and source-tree digest with every run.
- Do not compare runs made with different prompts, temperature, provider endpoints, retrieval implementations, or deletion masks.
- Report provider token usage and measured wall time separately; do not infer provider money cost without a dated pricing record.
- Preserve failed or blocked runs as small sanitized receipts when they explain a gate decision, but remove raw logs and temporary extraction trees before publishing the source repository.
