from collections import OrderedDict

from bemula.baselines.base import MemoryAdapter
from bemula.research import AmemResearchWrapper, CanonicalArtifact, CanonicalStateComparator
from bemula.schemas import Turn


class _Note:
    def __init__(self, note_id: str, content: str, timestamp: str):
        self.id = note_id
        self.content = content
        self.timestamp = timestamp
        self.context = "fixture"
        self.keywords = []
        self.tags = []
        self.links: list[int] = []


class _System:
    def __init__(self):
        self.memories = OrderedDict()
        self.counter = 0

    def add_note(self, content: str, time: str = "") -> str:
        note_id = f"n{self.counter}"
        self.counter += 1
        self.memories[note_id] = _Note(note_id, content, time)
        return note_id

    def find_related_memories(self, query: str, k: int = 5):
        return "", []


class _Adapter(MemoryAdapter):
    name = "identity-order-fixture"

    def reset(self):
        self._system = _System()

    def __init__(self):
        self.reset()

    def observe(self, turn: Turn) -> None:
        self._system.add_note(turn.text, time=turn.timestamp)

    def query(self, question: str) -> str:
        return ""


def _wrapper(texts: tuple[str, ...]) -> AmemResearchWrapper:
    wrapper = AmemResearchWrapper(_Adapter())
    for index, text in enumerate(texts):
        wrapper.observe(Turn(speaker="user", text=text, timestamp=f"t{index}"), source_id=f"s{index}")
    return wrapper


def test_same_content_from_distinct_sources_keeps_distinct_identity():
    state = _wrapper(("same payload", "same payload")).canonical_snapshot()
    ids = [artifact.canonical_id for artifact in state.artifacts]
    assert len(set(ids)) == 2


def test_retargeting_duplicate_content_link_is_detected():
    wrapper = _wrapper(("same payload", "same payload", "hub"))
    wrapper._memories()["n2"].links = [0]
    left = wrapper.canonical_snapshot()
    wrapper._memories()["n2"].links = [1]
    right = wrapper.canonical_snapshot()
    assert not CanonicalStateComparator().compare(left, right).equivalent


def test_ordered_retrieval_links_are_visible_state():
    wrapper = _wrapper(("first neighbor", "second neighbor", "hub"))
    wrapper._memories()["n2"].links = [0, 1]
    left = wrapper.canonical_snapshot()
    wrapper._memories()["n2"].links = [1, 0]
    right = wrapper.canonical_snapshot()
    assert not CanonicalStateComparator().compare(left, right).equivalent


def test_equal_artifact_multiset_has_order_invariant_hash():
    first = CanonicalArtifact(object_id="a", canonical_id="same", content="x", metadata={"tag": "A"})
    second = CanonicalArtifact(object_id="b", canonical_id="same", content="x", metadata={"tag": "B"})
    from bemula.research import CanonicalState

    left = CanonicalState(artifacts=[first, second])
    right = CanonicalState(artifacts=[second, first])
    assert CanonicalStateComparator().compare(left, right).equivalent
    assert left.visible_state_hash == right.visible_state_hash


class _MultiOutputSystem(_System):
    def __init__(self, first_ids: tuple[str, str]):
        super().__init__()
        self.first_ids = first_ids

    def add_fixed(self, note_id: str, content: str) -> None:
        self.memories[note_id] = _Note(note_id, content, "")


class _MultiOutputAdapter(MemoryAdapter):
    name = "identity-order-multi-output-fixture"

    def __init__(self, first_ids: tuple[str, str]):
        self.first_ids = first_ids
        self.reset()

    def reset(self):
        self._system = _MultiOutputSystem(self.first_ids)
        self._step = 0

    def observe(self, turn: Turn) -> None:
        if self._step == 0:
            for note_id in self.first_ids:
                self._system.add_fixed(note_id, "duplicate")
        else:
            first = next(iter(self._system.memories.values()))
            first.content = "updated"
            self._system.add_fixed("new", "new")
        self._step += 1

    def query(self, question: str) -> str:
        return ""


def test_source_lineage_order_ignores_physical_uuid_order():
    """Creation-order lineage remains stable when physical IDs are reversed."""

    def build(first_ids: tuple[str, str]):
        wrapper = AmemResearchWrapper(_MultiOutputAdapter(first_ids))
        wrapper.observe(Turn(speaker="user", text="first", timestamp="t0"), source_id="s0")
        wrapper.observe(Turn(speaker="user", text="second", timestamp="t1"), source_id="s1")
        return wrapper.canonical_snapshot()

    left = build(("z-physical", "a-physical"))
    right = build(("a-physical", "z-physical"))
    comparison = CanonicalStateComparator().compare(left, right)
    assert comparison.equivalent
