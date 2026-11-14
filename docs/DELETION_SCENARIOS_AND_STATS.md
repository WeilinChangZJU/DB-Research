# Deletion scenarios and paired failure reports

`bemula.research.scenarios` creates a deterministic deletion manifest over one
frozen `SourceEvent` history. It records the source hash, scenario kind,
deleted source IDs, event IDs, and optional trace-derived metadata. The
generator includes single, isolated, shared-support, early/middle/late,
batch, top-k boundary, NOOP, retrieval-only, linked-neighbor, and reinsert
conditions when the history or trace contains the corresponding evidence.

The manifest is an input condition, not a post-deletion truth label. The M0
full rebuild remains the state oracle for each scenario.

`bemula.research.stats` classifies state differences into machine-readable
failure records and summarizes paired observations by history-level arm. It
reports exact rate, replayed-event mean and p95, candidate counts, failure
counts, and replay cost savings relative to a supplied M0 reference. Reports
are JSON and can be checked into a run receipt or minimized into a replayable
fixture.

Cost savings are reported only for observations that are exact to the declared
M0 state. A low-cost incorrect state is retained as a failure and excluded
from the efficiency claim. `affected_set_precision_recall` is available as a
separate diagnostic for conservative over-invalidation versus missed objects.

```python
from bemula.research import (
    generate_deletion_scenarios,
    paired_observation,
    summarize_paired_observations,
)

manifest = generate_deletion_scenarios(events, dataset="locomo", history_id="sample-0")
summary = summarize_paired_observations(observations)
```

Statistics must use independent histories as the unit of analysis. Multiple
queries or deletion masks from one history are paired observations, not
independent samples. `paired_bootstrap_mean_difference` provides a seeded
history-level bootstrap interval for paired cost or replay differences.
