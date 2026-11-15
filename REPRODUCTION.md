# Q01 source-level memory-maintenance reproduction guide

This repository contains the latest source code for the Q01 validation platform. It separates source-level deletion maintenance from answer-level memory unlearning. The main experiment compares four serial maintenance arms on the same event history and deletion set:

- **M0**: independent rebuild from `H \ D`; this is the correctness reference.
- **M1**: live source tombstone/filter with retriever rebuild and no repair.
- **M2**: conservative suffix replay from a runtime checkpoint immediately
  before the earliest deleted source.
- **M3**: explicit dependency-baseline policy. The formal runner defaults to
  `m3_mode=conservative_full`, which restores the empty checkpoint and replays
  the complete retained history as a conservative correctness baseline. The
  optional `m3_mode=materialized` policy restores the earliest observed
  read/control boundary and is labelled limited because query invalidation,
  NOOP predicate scope, output-stability propagation, and complete input
  memoization are not claimed.

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

The formal orchestrator verifies the agent model, embedding backend/model, input hashes, official A-MEM commit and source hashes, and the source-tree digest before starting. Generate the freeze JSON from the actual files on the current machine; do not hand-edit hashes:

```bash
PYTHONPATH=. uv run python scripts/build_q01_amem_freeze.py \
  --amem "$BEMULA_AMEM_PATH" \
  --locomo ../q01_inputs/locomo_explicit_histories.jsonl \
  --longmemeval ../q01_inputs/longmemeval_s_explicit_histories.jsonl \
  --embedding-backend local \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --out ../q01_freeze/freeze.json
```

The helper refuses a non-git A-MEM directory or a checkout other than the
pinned commit.  The receipt records the endpoint URL and thinking mode when
provided, and the formal orchestrator checks those values together with the
embedding backend/model and retrieval `k`; it never records an API key.  A
freeze receipt is local run state and must remain outside Git.

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
  --retrieve-k 5 \
  --query-limit 1 \
  --evidence-level real_model_focused_window \
  --resume
```

Before retrying a run that stopped with `formal_run_blocker.json` status
`blocked_external_model_run`, repair the provider condition and pass `--resume`
explicitly. The interrupted history may have paid requests without a committed
instance checkpoint, so the runner refuses an implicit replay; a successful
end-to-end review removes the active blocker receipt.

The wrapper records provider exceptions separately from model decisions. The
official A-MEM implementation can catch an upstream 401/402/429 and store an
unevolved note; the Q01 wrapper now refuses that fallback after the operation
and the formal run must be treated as blocked. Do not interpret a stderr-only
partial trace as a completed history or as a zero-cost/zero-exposure result.

The command is intentionally serial. Each paired history executes two
independent M0 rebuilds (the second is a repeatability check) and one common
baseline build. A single request-keyed counterfactual response cassette is
shared for correctness across those builds, while provider attempts, method
memo hits, and oracle cassette hits are counted separately. M1--M3 fork that
baseline; they do not re-ingest a retained prefix. In `m3_mode=recheck`, the
restored checkpoint also imports the authorized full old-baseline response
memo for unchanged suffix requests, while current writes are re-executed.
Checkpoint restore bytes/events, suffix execution, query probes, and shared
lifecycle components are recorded separately. Resume is safe only when
the run fingerprint matches the input, model, retrieval setting and source
tree.

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

The paired runner defaults to one declared item per history (`--query-limit 1`)
for cost-bounded pilots. Use a larger explicit value when a workload review
requires more items; the selected item IDs and count are stored in every row's
`query_probe` record, so a one-item pilot cannot be confused with a complete
question evaluation.

The reviewer emits JSON and Markdown. Check all of the following before interpreting a result:

1. Both workloads have the expected number of completed histories.
2. M0 is exact to its independent repeat; shared baseline and repeat costs are
   present once in lifecycle scopes rather than duplicated into every arm.
3. `visible_state_hash`/canonical state mismatches are inspected by object and relationship, not reduced to an answer score.
4. Query probes are evidence-complete and their read/context source sets do not contain the deleted source. A missing evidence mapping is an unknown, not proof of zero exposure.
5. Query answer-model tokens are present in the query cost scope. Older runs made before the query-chat hook was added have unknown query cost and must not report it as zero.
6. Continuation answer/evidence hashes are compared against the isolated M0
   probe. A failed isolation or incomplete context mapping is an execution
   limitation, not evidence that the deletion method is wrong or successful.
7. M2/M3 replay mode, checkpoint restoration, and status are read from
per-arm metadata. Formal review requires the conservative M3 status
`strong_conservative_full_history_checkpoint_replay` and the
`complete_conservative_barrier` dependency-baseline marker. A label alone is
not evidence that a new algorithm is superior; the materialized policy remains
diagnostic and is labelled
`limited_materialized_read_control_checkpoint_replay`.

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
