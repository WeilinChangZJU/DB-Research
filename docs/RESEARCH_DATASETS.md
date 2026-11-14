# External research dataset adapters

The adapters under `bemula.research.datasets` read the original files in place;
large datasets are not copied into the Bemula repository. Every converted
instance carries the source path and SHA-256 digest, and every source turn has
a stable `source_id`, content hash, speaker, timestamp, and raw metadata.

## LoCoMo

```python
from bemula.research.datasets import load_locomo
instances = load_locomo("/path/to/locomo10.json", limit=1)
```

The official `dia_id` values (`D1:3`, etc.) are retained in each source ID.
LoCoMo QA evidence is mapped to `ResearchItem.source_ids`; unresolved evidence
is marked by `metadata["evidence_resolved"] = false` instead of being silently
dropped.

## LongMemEval-S / LongMenEval-S

```python
from bemula.research.datasets import load_longmemeval_s
instances = load_longmemeval_s("/path/to/longmemeval_s_cleaned.json", limit=1)
```

The adapter accepts JSON and JSONL. It creates one evaluation instance per
question, keeps the haystack session IDs and dates, and maps
`answer_session_ids` to source turn IDs. The canonical dataset name is
`longmemeval_s`; aliases include `LongMemEval-S` and the spelling
`LongMenEval-S` used in some project notes.

These adapters provide retention/long-context inputs and provenance. They do
not invent Bemula F1-F6 forget requests or attack labels for external datasets;
a separate ResearchProtocol suite must define those semantics before using
unlearning metrics.
