"""Desired contracts, not claims about stochastic model quality.
Run with reviewed source on PYTHONPATH. Failing assertions demonstrate residual gaps.
"""
from __future__ import annotations
import copy,json,hashlib,os
from pathlib import Path
from types import SimpleNamespace
import pytest
from audit_helpers_third import *

# ---------- New deterministic counterexamples ----------
def test_equal_content_nodes_keep_distinct_source_bound_identity():
    w=wrapper(('same payload','same payload','hub'))
    ids=[x.canonical_id for x in w.canonical_snapshot().artifacts[:2]]
    assert len(set(ids))==2, f'different source/time objects collapse to {ids}'

def test_retargeting_link_between_duplicate_content_nodes_is_detected():
    w=wrapper(('same payload','same payload','hub'))
    w._memories()['n2'].links=[0];a=w.canonical_snapshot()
    w._memories()['n2'].links=[1];b=w.canonical_snapshot()
    assert not CanonicalStateComparator().compare(a,b).equivalent

def test_link_order_affecting_rendered_context_is_not_erased():
    w=wrapper(('first neighbor','second neighbor','hub'))
    w._system().retriever.only_seed=2
    w._memories()['n2'].links=[0,1];a=w.canonical_snapshot();qa=w.adapter.query('q')
    w._memories()['n2'].links=[1,0];b=w.canonical_snapshot();qb=w.adapter.query('q')
    assert qa!=qb
    assert not CanonicalStateComparator().compare(a,b).equivalent, {'answer_a':qa,'answer_b':qb}

def test_equal_multiset_produces_equal_visible_hash():
    a=CanonicalArtifact(object_id='a',canonical_id='same',content='x',metadata={'tag':'A'})
    b=CanonicalArtifact(object_id='b',canonical_id='same',content='x',metadata={'tag':'B'})
    x=CanonicalState(artifacts=[a,b]);y=CanonicalState(artifacts=[b,a])
    assert CanonicalStateComparator().compare(x,y).equivalent
    assert x.visible_state_hash==y.visible_state_hash

def test_probe_evidence_hash_is_uuid_invariant():
    wa,wb=wrapper(prefix='left-'),wrapper(prefix='right-')
    a,_,_=probe_query(SimpleNamespace(wrapper=wa),arm=MaintenanceArm.M0_FULL_REBUILD,question='q',question_id='q',deleted=set(),trace=TraceRecorder())
    b,_,_=probe_query(SimpleNamespace(wrapper=wb),arm=MaintenanceArm.M2_REPLAY,question='q',question_id='q',deleted=set(),trace=TraceRecorder())
    assert a['source_ids']==b['source_ids'] and a['answer_sha256']==b['answer_sha256']
    assert a['evidence_sha256']==b['evidence_sha256'], {'a':a,'b':b}

def test_incomplete_evidence_cannot_pass_continuation_check():
    r={'question_id':'q','answer_sha256':'same-answer','evidence_sha256':'same-evidence',
       'query_isolated':True,'deleted_source_exposed':False,'evidence_complete':False}
    probes={MaintenanceArm.M0_FULL_REBUILD.value:{'records':[copy.deepcopy(r)]},
            MaintenanceArm.M2_REPLAY.value:{'records':[copy.deepcopy(r)]}}
    continuation_checks(probes)
    assert not probes[MaintenanceArm.M2_REPLAY.value]['continuation_checks_passed']

def test_ambiguous_duplicate_text_not_certified_complete_lineage():
    w=wrapper(('duplicate','duplicate','hub'))
    w._system().retriever.only_seed=2;w._memories()['n2'].links=[1]
    q=w.query('q')
    # Actual expanded payload is n1/source s1, not n0/source s0.
    assert set(q.evidence.source_ids)=={'s1','s2'} or not q.evidence.details['evidence_complete'], q.evidence.model_dump()

def test_context_coverage_ratio_is_bounded():
    w=wrapper(('alpha beta',));q=w.query('q')
    assert 0<=q.evidence.details['context_coverage_ratio']<=1, q.evidence.details['context_coverage_ratio']

def test_pinned_amem_structured_context_segments_certify_lineage():
    w=wrapper(('seed content','first neighbor','second neighbor'),prefix='seg-')
    system=w._system()
    system.retriever.only_seed=2
    system.memories['seg-2'].links=[0,1]

    def official_renderer(query, k=10):
        notes=list(system.memories.values())
        parts=[]
        for i in system.retriever.search(query,k):
            note=notes[i]
            parts.append(
                'talk start time:'+str(note.timestamp)
                +'memory content: '+str(note.content)
                +'memory context: '+str(note.context)
                +'memory keywords: '+str(note.keywords)
                +'memory tags: '+str(note.tags)+'\n'
            )
            for j,neighbor in enumerate(note.links):
                n=notes[int(neighbor)]
                parts.append(
                    'talk start time:'+str(n.timestamp)
                    +'memory content: '+str(n.content)
                    +'memory context: '+str(n.context)
                    +'memory keywords: '+str(n.keywords)
                    +'memory tags: '+str(n.tags)+'\n'
                )
                if j>=k: break
        return ''.join(parts)

    system.find_related_memories_raw=official_renderer
    q=w.query('q')
    assert q.evidence.details['context_segments_complete'] is True
    assert q.evidence.details['evidence_complete'] is True
    assert [segment['role'] for segment in q.evidence.details['context_segments']]==['seed','neighbor','neighbor']
    assert set(q.evidence.source_ids)=={'s0','s1','s2'}

def test_empty_final_context_is_complete_when_raw_renderer_is_observed():
    w=wrapper(('unused',))
    w._system().retriever.search=lambda query,k=10: []
    q=w.query('no-match')
    assert q.evidence.details['context_segments_complete'] is True
    assert q.evidence.details['context_mapping_complete'] is True
    assert q.evidence.details['evidence_complete'] is True

def test_production_fork_keeps_embedder_provider_identity():
    p=production_adapter();cl=p.fork()
    assert cl._system.retriever.model._embedder is cl.embedder, 'retriever uses a deepcopy, adapter counters read the original'

def test_production_fork_embedding_work_reaches_cost_meter():
    p=production_adapter();cl=p.fork()
    w=object.__new__(AmemResearchWrapper);w.adapter=cl
    before=AmemHistoryBuilder._embedding_counters(w)
    cl._system.retriever.model.encode(['x','y'])
    after=AmemHistoryBuilder._embedding_counters(w)
    assert (after[0]-before[0],after[1]-before[1])==(1,2), {'before':before,'after':after,'actual':vars(cl._system.retriever.model._embedder)}

def test_fork_consolidation_does_not_capture_original_adapter():
    p=production_adapter();cl=p.fork()
    cl.embedder=CountingEmbedder();cl.embedder.mutable_marker='clone'
    cl._system.consolidate_memories()
    assert cl._system.retriever.model._embedder is cl.embedder, 'consolidation closure is bound to the original adapter'

def test_read_event_is_not_a_write_to_its_read_object():
    e=ProvenanceEvent(event_id='read-1',event_type=ProvenanceEventType.READ,step=1,object_ids=['n0'],selected_ids=['n0'])
    g=DependencyGraph.from_events([e])
    pairs={(x.upstream,x.downstream) for x in g.edges}
    assert not any((b,a) in pairs for a,b in pairs if a!=b), pairs

def test_partial_unknown_cost_not_presented_as_full_total():
    from scripts.review_q01_amem_formal import _sum_cost
    assert _sum_cost([{'input_tokens':11},{'input_tokens':None}],'input_tokens') is None

# ---------- Diagnostics verifying improvements/boundaries (expected passes) ----------
def test_m2_really_avoids_prefix_reobserve_diagnostic():
    outputs=run_amem_all_arms(events(),['s3'],Adapter,m3_mode='conservative_full')
    m2=outputs[MaintenanceArm.M2_REPLAY][0];m3=outputs[MaintenanceArm.M3_READ_SET][0]
    assert m2.cost.replayed_events==0 and m2.metadata['checkpoint_restored']
    assert m3.cost.replayed_events==3

def test_m3_materialized_replay_is_not_more_local_than_m2_diagnostic():
    for d in ('s0','s1','s2','s3'):
        outputs=run_amem_all_arms(events(),[d],Adapter,m3_mode='materialized')
        m2=outputs[MaintenanceArm.M2_REPLAY][0];m3=outputs[MaintenanceArm.M3_READ_SET][0]
        assert m3.metadata['replay_start']<=m2.metadata['replay_start']
        assert m3.cost.replayed_events>=m2.cost.replayed_events

def test_checkpoint_memory_copies_all_prefixes_diagnostic():
    b=AmemHistoryBuilder(Adapter).build(events(8),arm=MaintenanceArm.M0_FULL_REBUILD)
    sizes=[len(w._memories()) for w in b.wrapper._checkpoints.values()]
    assert sizes==list(range(9)) and sum(sizes)==36

def test_visible_exact_can_coexist_with_wrong_index_diagnostic():
    w=wrapper();a=w.canonical_snapshot();w._system().retriever.embeddings[:]=42;b=w.canonical_snapshot()
    d=CanonicalStateComparator().compare(a,b)
    assert d.visible_equivalent and not d.runtime_equivalent and d.equivalent

def test_frozen_endpoint_is_not_overridden_by_late_dotenv(tmp_path, monkeypatch):
    """Exact resolve_endpoint body; real dotenv, a temporary local .env, no network.

    This models the environment after _verify_freeze has accepted the endpoint.
    The source body is copied unchanged solely to give dotenv an isolated file
    location, not to modify the reviewed archive or load actual credentials.
    """
    import ast
    from dotenv import load_dotenv
    source=(ROOT/'bemula/llm.py').read_text(encoding='utf-8')
    isolated=tmp_path/'llm_under_review.py';isolated.write_text(source,encoding='utf-8')
    tree=ast.parse(source,filename=str(isolated))
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='resolve_endpoint')
    ns={'os':os,'load_dotenv':load_dotenv}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(isolated),'exec'),ns)
    monkeypatch.setenv('OPENAI_BASE_URL','https://frozen.invalid/v1')
    monkeypatch.setenv('OPENAI_API_KEY','synthetic-test-key')
    monkeypatch.setenv('BEMULA_THINKING','disabled')
    (tmp_path/'.env').write_text('OPENAI_BASE_URL=https://changed.invalid/v1\nBEMULA_THINKING=enabled\n',encoding='utf-8')
    actual,_=ns['resolve_endpoint']('frozen-model')
    assert actual=='https://frozen.invalid/v1' and os.environ['BEMULA_THINKING']=='disabled', (actual,os.environ['BEMULA_THINKING'])
