"""Third review: deterministic fixtures plus exact AST loading of selected source definitions.
No source under review is patched; no live providers or model downloads are used.
"""
from __future__ import annotations
import ast, copy, hashlib, json, os, time
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
import numpy as np
import bemula
from bemula.baselines.base import MemoryAdapter, TurnProgress
from bemula.schemas import Turn
from bemula.research.amem import AmemResearchWrapper, AmemHistoryBuilder, run_amem_all_arms
from bemula.research.maintenance import SourceEvent, MaintenanceArm
from bemula.research.state import CanonicalState, CanonicalArtifact, CanonicalStateComparator, ProvenanceEvent, ProvenanceEventType
from bemula.research.dependency import DependencyGraph
from bemula.research.trace import TraceRecorder
ROOT = Path(bemula.__file__).resolve().parent.parent

def load_defs(relative, names, namespace=None):
    path = ROOT/relative
    tree=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    nodes=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    nodes.extend(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names)
    ns = {} if namespace is None else namespace
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
    return ns

class Note:
    def __init__(self, nid, content, timestamp='t0'):
        self.id,self.content,self.timestamp=nid,content,timestamp
        self.context='fixture';self.keywords=[];self.tags=[];self.links=[]
class CountingEmbedder:
    def __init__(self): self.calls=0; self.items=0;self.mutable_marker='original'
    def embed(self,texts):
        self.calls+=1;self.items+=len(texts)
        return np.ones((len(texts),2))
class EncodeModel:
    def encode(self,texts):return np.ones((len(texts),2))
class Retriever:
    def __init__(self):
        self.model=EncodeModel(); self.corpus=[];self.document_ids={};self.embeddings=np.empty((0,2));self.only_seed=None
    def add_documents(self,texts):
        start=len(self.corpus)
        for i,text in enumerate(texts):self.document_ids[text]=start+i;self.corpus.append(text)
        new=self.model.encode(texts)
        self.embeddings=new if self.embeddings is None or not len(self.embeddings) else np.vstack((self.embeddings,new))
    def search(self,q,k=10):
        if self.only_seed is not None:return [self.only_seed]
        return list(range(min(k,len(self.corpus))))
class System:
    def __init__(self,prefix='n'):
        self.prefix=prefix;self.counter=0;self.memories=OrderedDict();self.retriever=Retriever();self.unknown_tail='';self.evo_cnt=0;self.evo_threshold=100
    def add_note(self,content,time='t0'):
        nid=f'{self.prefix}{self.counter}';self.counter+=1
        self.memories[nid]=Note(nid,content,time);self.retriever.add_documents([content]);return nid
    def find_related_memories(self,query,k=5):return '',[]
    def find_related_memories_raw(self,query,k=10):
        notes=list(self.memories.values());parts=[]
        for i in self.retriever.search(query,k):
            parts.append(notes[i].content)
            for j in notes[i].links:parts.append(notes[int(j)].content)
        return '\n'.join(parts)+self.unknown_tail
class Adapter(MemoryAdapter):
    name='audit3-fixture'
    def __init__(self,prefix='n'):self.prefix=prefix;self.retrieve_k=10;self.reset()
    def reset(self):self._system=System(self.prefix);self.observed=[]
    def observe(self,turn):self.observed.append(turn.text);self._system.add_note(turn.text,time=turn.timestamp)
    def query(self,q):return self._system.find_related_memories_raw(q,self.retrieve_k)
    def _new_retriever(self):return Retriever()

def wrapper(texts=('alpha',),prefix='n'):
    w=AmemResearchWrapper(Adapter(prefix))
    for i,text in enumerate(texts):w.observe(Turn(speaker='user',text=text,timestamp=f't{i}'),source_id=f's{i}')
    return w

def events(n=4):return [SourceEvent(f'e{i}',f's{i}',f'payload-{i}',timestamp=f't{i}') for i in range(n)]

class FakeLLM:
    def chat(self,*a,**kw):return 'NO_EVOLUTION'

# Original production AmemAdapter class; only external providers are replaced.
production_ns={'copy':copy,'os':os,'sys':__import__('sys'),'Path':Path,'SimpleNamespace':SimpleNamespace,
               'MemoryAdapter':MemoryAdapter,'TurnProgress':TurnProgress,'Turn':Turn}
load_defs('bemula/baselines/amem.py',{'_AmemLLM','_APIEmbeddingModel','AmemAdapter'},production_ns)
ProductionAdapter=production_ns['AmemAdapter']

def production_adapter():
    p=object.__new__(ProductionAdapter)
    p.llm=FakeLLM();p.embedder=CountingEmbedder();p.retrieve_k=5;p._progress=TurnProgress('audit',enabled=False)
    p._system_class=System;p._retriever_class=Retriever;p.provider_fork_mode='fresh_provider'
    p._system=p._create_system()
    return p

probe_ns={'hashlib':hashlib,'json':json,'time':time,'TraceRecorder':TraceRecorder,'MaintenanceArm':MaintenanceArm}
probe_query=load_defs('scripts/run_q01_amem_paired.py',{'_probe_query'},probe_ns)['_probe_query']

def continuation_checks(query_probes):
    # Execute the three exact AST statements from main implementing the probe check.
    path=ROOT/'scripts/run_q01_amem_paired.py';tree=ast.parse(path.read_text(),filename=str(path))
    main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
    loop=next(x for x in main.body if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='instance')
    start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='m0_records' for t in n.targets))
    ns={'query_probes':query_probes,'MaintenanceArm':MaintenanceArm}
    code=ast.Module(body=loop.body[start:start+3],type_ignores=[])
    exec(compile(ast.fix_missing_locations(code),str(path),'exec'),ns)
    return query_probes
