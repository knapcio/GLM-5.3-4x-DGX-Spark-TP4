#!/usr/bin/env python3
"""Request-bound vLLM /metrics receipts, stdlib only. See README.md for scope.

Use RequestProbe around EACH streaming request, not around a benchmark suite.
Counters are attributable only under exclusive c1 traffic on one engine.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time
import urllib.request

COUNTERS = {
 'accepted_tokens': 'vllm:spec_decode_num_accepted_tokens',
 'draft_tokens': 'vllm:spec_decode_num_draft_tokens',
 'num_drafts': 'vllm:spec_decode_num_drafts',
 'completed_requests': 'vllm:request_success',
 'generation_tokens': 'vllm:generation_tokens',
}
EXACT = {'running': 'vllm:num_requests_running', 'waiting': 'vllm:num_requests_waiting',
         'decode_seconds': 'vllm:request_decode_time_seconds_sum',
         'decode_count': 'vllm:request_decode_time_seconds_count',
         'process_start': 'process_start_time_seconds'}
SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z_0-9:]*)(\{.*\})?\s+([^\s]+)(?:\s+\S+)?$')
LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*(?:,|$)')

class MetricsError(ValueError):
    pass

def parse_labels(raw):
    if not raw:
        return ()
    text, pos, values = raw[1:-1], 0, {}
    while pos < len(text):
        m = LABEL.match(text,pos)
        if not m or m[1] in values:
            raise MetricsError('invalid or duplicate Prometheus label')
        values[m[1]] = re.sub(r'\\([\\"n])', lambda x: '\n' if x[1]=='n' else x[1], m[2])
        pos = m.end()
    return tuple(sorted(values.items()))

def parse_metrics(text, labels=None):
    """Keep per-series values; never sum histogram buckets/positions/created."""
    selection = labels or {}
    series = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # OpenMetrics exemplars follow '#'; not part of the sample value.
        line = line.split(' # ',1)[0]
        m = SAMPLE.fullmatch(line)
        if not m:
            raise MetricsError('invalid metrics sample')
        pairs = parse_labels(m[2]);lab = dict(pairs)
        if m[1]!='process_start_time_seconds' and any(lab.get(k)!=v for k,v in selection.items()):
            continue
        key = (m[1],pairs)
        if key in series:
            raise MetricsError('duplicate metric series')
        value = float(m[3])
        # Unrelated exporter NaN/Inf is allowed; selected metrics are checked.
        series[key] = value
    return Snapshot(series)

class Snapshot:
    def __init__(self, series):
        self.series = series

    def select(self, key, required=True):
        names = [COUNTERS[key]+'_total', COUNTERS[key]] if key in COUNTERS else [EXACT[key]]
        for name in names:
            values = {lab:v for (n,lab),v in self.series.items() if n==name}
            if values:
                if any(not math.isfinite(v) or v<0 for v in values.values()):
                    raise MetricsError(f'invalid metric {name}')
                engines = {dict(lab).get('engine',dict(lab).get('engine_id')) for lab in values}
                if len(engines)>1:
                    raise MetricsError('select exactly one engine with labels')
                models = {dict(lab).get('model_name') for lab in values}
                if len(models)>1:
                    raise MetricsError('select exactly one model with labels')
                return name,values
        if required:
            raise MetricsError(f'missing metric {key}')
        return None,{}

    def value(self,key,required=True):
        _,vs = self.select(key,required)
        return sum(vs.values()) if vs else None

    def drained(self):
        return self.value('running')==0 and self.value('waiting')==0

def delta(before,after,key,required=True):
    bn,b = before.select(key,required);an,a = after.select(key,required)
    if not b and not a:
        return None
    if bn!=an or b.keys()!=a.keys():
        raise MetricsError(f'metric identity/series changed: {key}')
    if any(a[k]<b[k] for k in b):
        raise MetricsError(f'counter reset: {key}')
    return sum(a[k]-b[k] for k in b)

def summarize(before,after,client_decode_wall_s=None,delivered_tokens=None):
    if delivered_tokens is not None and (type(delivered_tokens) is not int or delivered_tokens<0):
        raise MetricsError('delivered_tokens must be a nonnegative integer')
    if not before.drained() or not after.drained():
        raise MetricsError('request boundary not drained')
    bp,ap = before.value('process_start',False),after.value('process_start',False)
    if bp!=ap:
        raise MetricsError('exporter restarted or process identity changed')
    d = {k:delta(before,after,k) for k in ('accepted_tokens','draft_tokens','num_drafts','completed_requests')}
    d['generation_tokens'] = delta(before,after,'generation_tokens',False)
    count = delta(before,after,'decode_count')
    if d['completed_requests']!=1 or count!=1:
        raise MetricsError('need exactly one published completed request/decode histogram sample')
    if not all(float(v).is_integer() for k,v in d.items() if v is not None):
        raise MetricsError('noninteger counter delta')
    if d['accepted_tokens']>d['draft_tokens'] or d['num_drafts']<=0:
        raise MetricsError('invalid speculative counter deltas')
    seconds = delta(before,after,'decode_seconds')
    source = 'server_request_decode_time_sum'
    if client_decode_wall_s is not None:
        if not math.isfinite(client_decode_wall_s) or client_decode_wall_s<=0:
            raise MetricsError('invalid client decode wall time')
        seconds,source = client_decode_wall_s,'client_first_to_last_output_token'
    if seconds<=0:
        raise MetricsError('zero decode wall time')
    cycles = d['num_drafts']
    committed = d['accepted_tokens']+cycles
    return dict(deltas=d,decode_wall_s=seconds,server_decode_wall_s=delta(before,after,'decode_seconds'),
                time_source=source,cycle_ms=1000*seconds/cycles,
                cycle_ms_kind='request wall / draft opportunities; NOT GPU-event cycle median',
                committed_per_cycle=committed/cycles,mean_K=d['draft_tokens']/cycles,
                acceptance=d['accepted_tokens']/d['draft_tokens'] if d['draft_tokens'] else None,
                reconstructed_tps=committed/seconds,delivered_tokens=delivered_tokens,
                delivered_per_cycle=delivered_tokens/cycles if delivered_tokens is not None else None,
                scope='exclusive c1; terminal clipping/first-token cycle can differ from delivered count')

def scrape(url,timeout=5):
    t = time.monotonic_ns()
    with urllib.request.urlopen(url,timeout=timeout) as response:
        text = response.read().decode('utf-8')
    return text,dict(start_monotonic_ns=t,end_monotonic_ns=time.monotonic_ns(),wall_time_ns=time.time_ns())

class GateMetrics:
    def __init__(self,url,out_dir,*,labels=None,metadata=None,fetch=None,settle_timeout_s=10):
        self.url,self.out_dir = url,Path(out_dir)
        self.labels,self.metadata = labels or {},metadata or {}
        self.fetch = fetch or (lambda: scrape(url))
        self.settle_timeout_s=settle_timeout_s
        self.out_dir.mkdir(parents=True,exist_ok=True)

    def request(self,request_id,*,exclusive=False,metadata=None):
        if not exclusive:
            raise MetricsError('explicit exclusive c1 attribution required')
        return RequestProbe(self,request_id,{**self.metadata,**(metadata or {})})

class RequestProbe:
    def __init__(self,gate,request_id,metadata):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',request_id):
            raise MetricsError('use filesystem-safe unique request_id')
        self.gate,self.id,self.metadata = gate,request_id,metadata
        self.first_ns,self.last_ns,self.delivered = None,None,None
        self.client_wall = None
        self.record = None

    def sample(self,name):
        text,timing = self.gate.fetch()
        path = self.gate.out_dir/f'{self.id}.{name}.prom'
        with path.open('x') as f:
            f.write(text)
        self.receipts[name] = dict(path=str(path),sha256=hashlib.sha256(text.encode()).hexdigest(),**timing)
        return parse_metrics(text,self.gate.labels)

    def __enter__(self):
        self.receipts = {}
        self.before = self.sample('before')  # Last operation before issuing request.
        if not self.before.drained():
            raise MetricsError('before scrape contains active/queued requests')
        return self

    def token(self):
        """Call only for content/reasoning output, not role-only or DONE chunks.
        For batched SSE tokens this measures first-to-last output CHUNK wall.
        """
        now = time.monotonic_ns()
        if self.first_ns is None:
            self.first_ns = now
        self.last_ns = now

    def set_decode_wall(self,seconds,delivered_tokens=None):
        """Prefer exact first-to-last timestamps already recorded by the client."""
        self.client_wall,self.delivered = seconds,delivered_tokens

    def __exit__(self,exc_type,exc,tb):
        row = dict(request_id=self.id,metadata=self.metadata,labels=self.gate.labels,
                   exclusive_c1_claim=True,status='INVALID',receipts=self.receipts)
        try:
            after = self.sample('after')  # Immediate, including on client failure.
            if exc is not None:
                raise MetricsError('request raised '+exc_type.__name__)
            immediate_count=delta(self.before,after,'completed_requests')
            deadline=time.monotonic()+self.gate.settle_timeout_s
            attempt=0
            # Keep the immediate after scrape. Poll only publication/drain lag,
            # never reset, identity, or evidence of multiple requests.
            while delta(self.before,after,'completed_requests')==0 or (
                    delta(self.before,after,'completed_requests')==1 and not after.drained()):
                if time.monotonic()>=deadline:
                    raise MetricsError('metrics publication/drain deadline; retain immediate scrape')
                time.sleep(.1)
                attempt+=1
                after=self.sample(f'after_settled_{attempt:03d}')
            row['immediate_completed_delta']=immediate_count
            row['publication_poll_count']=attempt
            seconds = self.client_wall
            if seconds is None and self.first_ns is not None and self.last_ns>self.first_ns:
                seconds = (self.last_ns-self.first_ns)/1e9
            row.update(summarize(self.before,after,seconds,self.delivered),status='OK')
        except Exception as error:
            row['error'] = str(error)
        row['hardware_cycle_gate_eligible']=False
        self.record = row
        with (self.gate.out_dir/'requests.jsonl').open('a') as f:
            f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        # Metrics failures are visible to gate callers; retain original client
        # exceptions and never turn a failed request into a qualified sample.
        if exc is None and row['status']!='OK':
            raise MetricsError(row['error'])
        return False

def paired(a,b):
    """One matched prompt/seed/repetition pair, with metadata consistency."""
    for key in ('status',):
        if a.get(key)!='OK' or b.get(key)!='OK':
            raise MetricsError('cannot pair invalid requests')
    keys = ('pair_key','prompt_sha256','seed','k_mode','target_M')
    for k in keys:
        if k not in a['metadata'] or k not in b['metadata'] or a['metadata'][k]!=b['metadata'][k]:
            raise MetricsError('unmatched metadata: '+k)
    if a['time_source']!=b['time_source'] or a.get('labels')!=b.get('labels'):
        raise MetricsError('time source or selected metric labels differ')
    return dict(pair_key=a['metadata']['pair_key'],A_request=a['request_id'],B_request=b['request_id'],
                A_cycle_ms=a['cycle_ms'],B_cycle_ms=b['cycle_ms'],
                cycle_delta_ms=b['cycle_ms']-a['cycle_ms'],cycle_ratio=b['cycle_ms']/a['cycle_ms'],
                A_committed_per_cycle=a['committed_per_cycle'],B_committed_per_cycle=b['committed_per_cycle'],
                committed_delta=b['committed_per_cycle']-a['committed_per_cycle'],
                committed_ratio=b['committed_per_cycle']/a['committed_per_cycle'],
                metric_kind=a['cycle_ms_kind'])

def self_test():
    root = Path(__file__).resolve().parent
    # This file is a real recorded, flattened /metrics scrape. No invented
    # baseline counters: the original provenance is in metrics_sample.source.json.
    raw = json.loads((root/'metrics_sample.recorded.json').read_text())
    text = '\n'.join(f'{k} {v}' for k,v in raw.items())+'\n'
    s = parse_metrics(text)
    assert s.value('accepted_tokens')==5594 and s.value('draft_tokens')==7996
    assert s.value('num_drafts')==3998 and s.drained()
    changed = dict(raw)
    increments = {'vllm:spec_decode_num_accepted_tokens_total':300,
                  'vllm:spec_decode_num_draft_tokens_total':400,'vllm:spec_decode_num_drafts_total':200,
                  'vllm:request_success_total':1,'vllm:request_decode_time_seconds_count':1,
                  'vllm:request_decode_time_seconds_sum':16,'vllm:generation_tokens_total':500}
    for k,v in increments.items():
        changed[k]+=v
    after = parse_metrics('\n'.join(f'{k} {v}' for k,v in changed.items()))
    r = summarize(s,after,16,500)
    assert r['cycle_ms']==80 and r['committed_per_cycle']==2.5 and r['mean_K']==2
    # Counter reset, missing counter, active boundary, identity changes.
    for field,value in [('vllm:spec_decode_num_drafts_total',1),('vllm:num_requests_running',1),
                        ('process_start_time_seconds',raw['process_start_time_seconds']+1)]:
        bad = dict(changed);bad[field]=value
        try:
            summarize(s,parse_metrics('\n'.join(f'{k} {v}' for k,v in bad.items())),16)
        except MetricsError:
            pass
        else:
            raise AssertionError(field)
    missing = dict(changed);del missing['vllm:spec_decode_num_drafts_total']
    try:
        summarize(s,parse_metrics('\n'.join(f'{k} {v}' for k,v in missing.items())),16)
    except MetricsError:
        pass
    else:
        raise AssertionError('missing counter')
    labels = parse_metrics('vllm:spec_decode_num_drafts_total{model_name="GLM, \\"test\\"",engine="0"} 12\n'
                           'vllm:spec_decode_num_drafts_created{model_name="GLM"} 999\n'
                           'vllm:spec_decode_num_drafts_total{model_name="other",engine="1"} 88\n',
                           {'engine':'0'})
    assert labels.value('num_drafts')==12
    metadata = dict(pair_key='p0-seed53-r0',prompt_sha256='test',seed=53,k_mode='k2',target_M=3)
    a=dict(r,status='OK',request_id='a',metadata=metadata)
    b=dict(a,request_id='b',cycle_ms=76,committed_per_cycle=2.6)
    pair=paired(a,b)
    assert pair['cycle_delta_ms']==-4 and abs(pair['cycle_ratio']-.95)<1e-12
    import tempfile
    with tempfile.TemporaryDirectory(prefix='metrics-selftest-',dir=root) as tmp:
        replies=iter([(text,{}),(text,{}),('\n'.join(f'{k} {v}' for k,v in changed.items()),{})])
        gate=GateMetrics('unused',tmp,fetch=lambda: next(replies),settle_timeout_s=1)
        with gate.request('test',exclusive=True,metadata=metadata) as request:
            request.set_decode_wall(16,500)
        assert request.record['status']=='OK' and request.record['publication_poll_count']==1
        assert (Path(tmp)/'test.before.prom').exists() and (Path(tmp)/'test.after.prom').exists()
    print(json.dumps(dict(self_test='PASS',recorded_baseline=dict(accepted=5594,draft=7996,drafts=3998),
                          synthetic_delta_test=dict(cycle_ms=80,committed_per_cycle=2.5),paired=pair),sort_keys=True))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--pair',nargs=2,type=Path,metavar=('A_JSONL','B_JSONL'))
    a=p.parse_args()
    if a.self_test:
        self_test()
    elif a.pair:
        banks=[]
        for path in a.pair:
            records=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            rows={}
            for row in records:
                key=row['metadata']['pair_key']
                if key in rows:
                    raise MetricsError('duplicate pair_key')
                rows[key]=row
            banks.append(rows)
        if banks[0].keys()!=banks[1].keys():
            raise MetricsError('missing request pairs')
        pairs=[paired(banks[0][k],banks[1][k]) for k in sorted(banks[0])]
        if not pairs:
            raise MetricsError('empty pair set')
        for row in pairs:
            print(json.dumps(row,sort_keys=True))
        print(json.dumps(dict(type='paired_summary',n=len(pairs),
                             median_cycle_ratio=statistics.median(x['cycle_ratio'] for x in pairs),
                             median_committed_ratio=statistics.median(x['committed_ratio'] for x in pairs)),sort_keys=True))
    else:
        p.error('use --self-test or --pair A.jsonl B.jsonl; import GateMetrics for request hooks')

if __name__=='__main__':
    main()
