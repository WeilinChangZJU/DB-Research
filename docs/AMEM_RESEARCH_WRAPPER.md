# A-MEM research wrapper

`bemula.research.amem.AmemResearchWrapper` is the source-aware boundary around
the existing Bemula `AmemAdapter`. It leaves the adapter's benchmark behavior
unchanged and adds the evidence needed by the Q01 maintenance protocol.

The wrapper accepts an external `source_id` for every observed turn and keeps
the source-to-note mapping. A snapshot contains the source records and the
current A-MEM notes, including note metadata, support sets, links, and a stable
content hash. The wrapper records `ingest`, `candidate`, `write`, `update`,
`noop`, `read`, `select`, and `llm_call` events. When the stock adapter exposes
its internal `llm_controller`, the wrapper records prompt/response hashes,
call-kind and conservative branch labels such as `NO_EVOLUTION` without
persisting raw prompt or response text. Retrieval candidates are captured at
the `find_related_memories`/retriever boundary and are mapped back from
A-MEM's integer positions to note IDs. `canonical_snapshot()` normalizes
physical note UUIDs into semantic IDs and preserves query candidate/selected
order. `introspect()` exposes the forward and reverse dependency maps and the
trace quality.

The trace is explicitly labelled `wrapper_delta`. The stock adapter does not
expose its parsed branch as a stable public API, so the wrapper's branch labels
are conservative response parsing rather than a formal causal proof. The
wrapper has semantic checkpoint save/load; physical A-MEM index rehydration
still requires a builder-specific restore routine. Native source delete is
unsupported, so M0--M3 use fresh instances plus replay. The A-MEM M1 arm is
explicitly a conservative cascade: any materialized note whose observed
support intersects the deletion mask is removed, including a note with
surviving secondary support. It is a weak isolation baseline and must not be
described as exact source-only deletion.

The snapshot is serialisable, but stock A-MEM note UUIDs are physical
identifiers. Cross-run semantic equality must therefore pass through the Q01
canonical state normalizer; the wrapper reports `stable_snapshot_ids=false`.
Retriever scores are computed when the upstream retriever exposes its
embedding matrix and model encoder; if it returns indices only, the wrapper
preserves candidate order and reports `retriever_scores=false`. Raw context
hashes are recorded when `find_related_memories_raw` is available.

## Offline single-history check

Run the deterministic fixture from the repository root:

```text
python scripts/run_amem_fixture.py --out <output-dir>/summary.json
```

The fixture has three source events and removes `src-a`. It imports neither
the official A-MEM repository nor a model client. The expected contract is:

| Arm | Meaning | Expected state result |
|---|---|---|
| M0 | full rebuild from retained source events | reference |
| M1 | tombstone/filter the existing state | intentionally not exact for this shared-support history |
| M2 | replay from the earliest affected event | exact to M0 |
| M3 | replay from the observed dependency boundary | exact to M0 |

M2 and M3 currently re-ingest their unaffected prefix before executing the
suffix. That prefix work is included in `maintenance_full_execution` cost and
`checkpoint_restored=false`; current results do not claim checkpoint-based cost
savings. M3 is a distinct materialized read/control invalidation baseline, but
not yet a memoizing self-adjusting execution engine.

The result is an engineering contract check. It is not a DeepSeek quality
result and does not establish an A-MEM paper reproduction.
