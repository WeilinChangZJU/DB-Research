"""Independent audit regressions for report 1114.zip (2026-11-14).

These assert intended contracts, not the uploaded implementation's current
behaviour. Failures are expected on the reviewed snapshot. No model, provider,
or network access is used. Run with the platform directory on PYTHONPATH.
"""
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
from bemula.baselines.base import MemoryAdapter
from bemula.schemas import Turn
from bemula.research.amem import AmemResearchWrapper, AmemHistoryBuilder, run_amem_all_arms
from bemula.research.maintenance import SourceEvent, MaintenanceArm, _token_set
from bemula.research.state import CanonicalState, CanonicalArtifact, CanonicalQuery, CanonicalStateComparator
from bemula.research.evaluation import compare_query_results
from bemula.research.backend import QueryResult, QueryEvidence, InMemoryBackend
from bemula.research.protocol import ResearchProtocol, ResearchInstance, ResearchTurn
from bemula.research.scenarios import generate_deletion_scenarios, DeletionScenarioKind
from bemula.research.datasets.common import ResearchInstance as DatasetInstance, SourceRecord as DatasetSource

class Note:
    def __init__(self, nid, content, timestamp):
        self.id, self.content, self.timestamp = nid, content, timestamp
        self.context, self.keywords, self.tags, self.links = 'fixture', [], [], []

class System:
    def __init__(self):
        self.memories = OrderedDict()
        self.counter = 0
    def add_note(self, content, time=''):
        nid = f'n{self.counter}'
        self.counter += 1
        self.memories[nid] = Note(nid, content, time)
        return nid
    def find_related_memories(self, query, k=5):
        ranked = [(i,n) for i,n in enumerate(self.memories.values()) if set(query.lower().split()) & set(n.content.lower().split())]
        return '', [i for i,n in ranked[:k]]

class Adapter(MemoryAdapter):
    name = 'independent-audit-fixture'
    def __init__(self): self.reset()
    def reset(self): self._system = System()
    def observe(self, turn):
        self._system.find_related_memories(turn.text, k=5)
        self._system.add_note(turn.text, turn.timestamp)
    def query(self, question):
        _, indices = self._system.find_related_memories(question, k=5)
        notes = list(self._system.memories.values())
        return '\n'.join(notes[i].content for i in indices)

class CopyingAdapter(Adapter):
    """Explicitly copies previously read note payload into the new note."""
    def observe(self, turn):
        _, ids = self._system.find_related_memories(turn.text, k=5)
        notes = list(self._system.memories.values())
        content = turn.text + ''.join(' | COPIED: '+notes[i].content for i in ids)
        self._system.add_note(content, turn.timestamp)

class NoopAdapter(Adapter):
    def observe(self, turn):
        if not self._system.memories:
            self._system.add_note(turn.text, turn.timestamp)
        # Subsequent input is ignored; no reads or writes occur.

EVENTS = [SourceEvent('e0','a','secret cobalt',timestamp='t0'), SourceEvent('e1','b','public garden',timestamp='t1')]

def test_negative_control_m0_m2_m3_match_on_fixture():
    results = run_amem_all_arms(EVENTS, ['a'], Adapter)
    assert all(results[a][1].equal for a in [MaintenanceArm.M0_FULL_REBUILD,MaintenanceArm.M2_REPLAY,MaintenanceArm.M3_READ_SET])

def test_tokens_are_words_not_literal_backslash_or_w():
    assert _token_set('office Shanghai') == {'office','shanghai'}

def test_m1_empty_delete_is_identity():
    results = run_amem_all_arms(EVENTS, [], Adapter)
    assert results[MaintenanceArm.M1_TOMBSTONE][1].equal

def test_m1_live_query_does_not_read_deleted_source():
    builder = AmemHistoryBuilder(Adapter)
    baseline = builder.build(EVENTS, arm=MaintenanceArm.M0_FULL_REBUILD)
    result = builder.run_arm(EVENTS, ['a'], MaintenanceArm.M1_TOMBSTONE, baseline=baseline)
    assert 'a' not in [r.source_id for r in result.snapshot.records]
    assert 'cobalt' not in result.wrapper.query('secret').answer

def test_read_dependency_reaches_downstream_write():
    wrapper = AmemResearchWrapper(CopyingAdapter())
    wrapper.observe(Turn(speaker='user',text='shared cobalt',timestamp='t0'),source_id='a')
    wrapper.observe(Turn(speaker='user',text='shared downstream',timestamp='t1'),source_id='b')
    written = set(wrapper._source_to_objects['b'])
    assert written <= wrapper.dependency_graph().affected_closure(['a'])

def test_copied_payload_inherits_source_dependency():
    wrapper = AmemResearchWrapper(CopyingAdapter())
    wrapper.observe(Turn(speaker='user',text='shared cobalt',timestamp='t0'),source_id='a')
    wrapper.observe(Turn(speaker='user',text='shared downstream',timestamp='t1'),source_id='b')
    target = next(iter(wrapper._source_to_objects['b']))
    assert 'cobalt' in wrapper._memories()[target].content
    assert 'a' in wrapper._object_support[target]

def test_ignored_input_does_not_gain_fabricated_materialized_support():
    wrapper = AmemResearchWrapper(NoopAdapter())
    wrapper.observe(Turn(speaker='user',text='original',timestamp='t0'),source_id='a')
    wrapper.observe(Turn(speaker='user',text='ignored',timestamp='t1'),source_id='b')
    assert all('b' not in support for support in wrapper._object_support.values())

def test_query_score_keys_are_physical_id_invariant():
    def state(nid):
        return CanonicalState(artifacts=[CanonicalArtifact(object_id=nid,canonical_id='stable',content='same',support_set=['a'])],
            queries=[CanonicalQuery(query_id='q',candidate_ids=[nid],selected_ids=[nid],scores={nid:0.5})])
    assert CanonicalStateComparator().compare(state('uuid-left'),state('uuid-right')).equivalent

def test_comparator_preserves_artifact_multiplicity():
    a = CanonicalArtifact(object_id='x',content='same',support_set=['s'])
    b = a.model_copy(update={'object_id':'y'})
    diff = CanonicalStateComparator().compare(CanonicalState(artifacts=[a]),CanonicalState(artifacts=[a,b]))
    assert not diff.equivalent

def test_snapshot_detects_embedding_value_change():
    wrapper = AmemResearchWrapper(Adapter())
    wrapper.observe(Turn(speaker='user',text='same',timestamp='t0'),source_id='a')
    wrapper._system().retriever = SimpleNamespace(corpus=['same'], embeddings=np.array([[1.,0.]]),document_ids={'same':0})
    before = wrapper.canonical_snapshot()
    wrapper._system().retriever.embeddings[:] = np.array([[0.,1.]])
    after = wrapper.canonical_snapshot()
    assert not CanonicalStateComparator().compare(before,after).equivalent

def test_snapshot_detects_consolidation_counter_change():
    wrapper = AmemResearchWrapper(Adapter())
    wrapper.observe(Turn(speaker='user',text='same',timestamp='t0'),source_id='a')
    wrapper._system().evo_cnt = 0
    before = wrapper.canonical_snapshot()
    wrapper._system().evo_cnt = 99
    after = wrapper.canonical_snapshot()
    assert not CanonicalStateComparator().compare(before,after).equivalent

def test_old_source_exposure_includes_read_set():
    left=QueryResult(answer='same',evidence=QueryEvidence(source_ids=[],read_set=['deleted']))
    right=QueryResult(answer='same',evidence=QueryEvidence(source_ids=[],read_set=[]))
    assert compare_query_results(left,right,deleted_source_ids=['deleted']).old_source_exposure == ('deleted',)

def test_topk_boundary_is_not_asserted_without_a_retrieval_trace():
    manifest=generate_deletion_scenarios(EVENTS,top_k=1)
    assert not any(s.kind is DeletionScenarioKind.TOP_K_BOUNDARY for s in manifest.scenarios)

def test_explicit_empty_control_history_is_respected():
    turn=ResearchTurn(dataset='d',instance_id='i',source_id='a',text='secret')
    result=ResearchProtocol(InMemoryBackend()).run_instance(ResearchInstance(dataset='d',instance_id='i',turns=[turn],control_turns=[],items=[]))
    assert result.control_snapshot.records == []

def test_dataset_protocol_conversion_preserves_delete_mask():
    instance=DatasetInstance(dataset='d',instance_id='i',sessions=[[DatasetSource('d','i','a','x','user')]],items=[],source_path='fixture',source_hash='fixture',metadata={'forget_source_ids':['a']})
    assert instance.to_protocol_instance().forget_source_ids == ['a']
