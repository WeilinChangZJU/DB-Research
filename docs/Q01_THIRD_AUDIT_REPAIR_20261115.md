# Q01 third-audit repair and pilot receipt

Date: 2026-11-15

This document records the implementation work performed after the third
independent audit. The audit package was treated as an engineering
constraint set; its assertions were re-run against the current source and
were not copied into a scientific result claim.

## Repair receipt

The source-level changes cover the following contracts.

- Canonical artifact identity now retains source, time and origin information
  so same-text events from different sources do not collapse. Ordered
  retrieval parent links are preserved and compared as ordered links.
- Query evidence signatures exclude physical UUIDs and audit-only fields.
  Continuation checks require evidence-complete records on both sides and
  prefer an independent M0-repeat query when it is available.
- Ambiguous text-only lineage is reported as incomplete. Coverage counts
  non-whitespace characters inside mapped intervals, so duplicate text cannot
  create coverage ratios above one.
- Forks rebind the copied retriever and consolidation closure to the fork's
  provider/embedder. Read events no longer create a pseudo decision-output
  cycle. Provider configuration is loaded once without a late overriding
  `.env` read.
- Checkpoint restore bytes are UTF-8 bytes of the semantic snapshot estimate,
  and the scope is recorded explicitly. This remains an auditable estimate,
  not a claim that the semantic export is a complete runtime serialization.
- Mixed or missing cost components remain `unknown`. Formal review JSON now
  carries per-field observed count, expected count, coverage and status.
- M3 has an explicit `recheck` mode. It restores the materialized
  invalidation boundary, re-evaluates retrieval/control/write logic and reuses
  only exact request-key responses from the paired baseline. It is a strong
  generic pilot baseline, not a proof of a complete self-adjusting runtime.

## Verification

The Windows source snapshot passed 123 tests, including 18 third-audit
contract tests. The deterministic official A-MEM preflight also completed two
LoCoMo and two LongMemEval-S histories in serial M0--M3 mode with the pinned
official commit `f9a173dec325d84fcc97e7828775c7b241c33466`. Those controlled
runs had stable M0 repeats and M2/M3 state equality on the selected histories.
Their query lineage remained incomplete by design, so they are mechanism and
execution evidence only.

## Real-model pilot

One history from each workload was run with the same freeze receipt, local
MiniLM embeddings, `deepseek-flash`, thinking disabled, `retrieve_k=5`, one
declared question, serial execution and `m3_mode=recheck`.

| Workload | Histories | M0 repeat | M1 exact | M2 exact | M3 exact | Evidence-complete probes | Deleted-source exposures |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 1/4 | 0 |
| LongMemEval-S | 1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/4 | 0 |

Both workloads satisfied the operational checks for shared baseline,
checkpoint replay, isolated query probes and the explicit M3 recheck policy.
Both failed the M0-repeat gate. The state differences therefore cannot be
attributed to M1--M3 maintenance. The absence of an observed deleted-source
exposure is a lower bound because the final rendered context-to-source map was
incomplete on most probes.

The lifecycle ledger measured model calls and provider token usage. It did
not expose provider money pricing, so monetary cost is not inferred. Local
embedding call/item counters were not complete in this pilot and remain
`unknown` in the formal review; they must not be treated as zero. Query score
reconstruction is recorded separately as instrumentation work.

## Q01 decision

This round does not establish H1 (a natural non-materialized decision
dependency), H2 (meaningful natural occurrence), H3 (avoidable cost after a
correct generic baseline), or H4 (research novelty). It also does not
disprove them: the decisive oracle and source evidence gates are still open.

The appropriate status remains **PROMISING / UNVERIFIED**. The next gate is to
make the frozen real-model M0 repeat stable, then rerun the paired pilot with
complete final-context lineage. Until those gates pass, no maintenance arm
should be declared superior and no new U method should be presented as a
research contribution.
