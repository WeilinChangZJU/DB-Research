"""External dataset adapters for the Bemula ResearchProtocol layer."""

from .common import ResearchInstance, ResearchItem, SourceRecord, content_hash, file_hash, dump_jsonl
from .locomo import iter_locomo, load_locomo
from .longmemeval import (
    iter_longmemeval,
    iter_longmemeval_s,
    iter_longmeneval_s,
    load_longmemeval,
    load_longmemeval_s,
    load_longmeneval_s,
)

__all__ = [
    "ResearchInstance",
    "ResearchItem",
    "SourceRecord",
    "content_hash",
    "file_hash",
    "dump_jsonl",
    "iter_locomo",
    "load_locomo",
    "iter_longmemeval",
    "iter_longmemeval_s",
    "iter_longmeneval_s",
    "load_longmemeval",
    "load_longmemeval_s",
    "load_longmeneval_s",
]
