import collections,json
from pathlib import Path
root=Path(__file__).resolve().parent
raw=root.parent/'15-workload-timeline-retry-02'
summary=json.loads((raw/'timeline-summary-with-layout.json').read_text())
window=summary['phase_boundaries']['request.decode']
lo,hi=window['start_us'],window['end_us']
counts=collections.defaultdict(lambda:[0,0.0])
intervals=[]
selected=0
with (root/'prior-decode-gpu.jsonl').open('w') as out:
 for line in (raw/'gpu-correlation.jsonl').open():
  v=json.loads(line)
  if v['end_us'] < lo or v['start_us'] > hi: continue
  out.write(line)
  selected+=1
  start,end=max(lo,v['start_us']),min(hi,v['end_us'])
  counts[v['kernel']][0]+=1
  counts[v['kernel']][1]+=end-start
  intervals.append((start,end))
union=[]
for a,b in sorted(intervals):
 if union and a<=union[-1][1]: union[-1][1]=max(b,union[-1][1])
 else: union.append([a,b])
scopes=collections.defaultdict(lambda:[0,0.0])
Ts=collections.Counter()
with (root/'prior-decode-source.jsonl').open('w') as out:
 for line in (raw/'source-events-request.jsonl').open():
  v=json.loads(line)
  if v['ts']+v.get('dur',0)<lo or v['ts']>hi: continue
  out.write(line)
  if v['ph']=='X':
   scopes[v['name']][0]+=1
   scopes[v['name']][1]+=v.get('dur',0)
   if v['name']=='decode.verify': Ts[v['args']['T']]+=1
r={'class':'reused instrumented resident decode structure; not unprofiled performance or hardware roof ratio','source_profile_binary_sha256':'1fcff5f003688a33f3c43ff98cd8ae7cd081f9fa04ae3fc1e0b7f96366d304da','decode_window':window,'gpu_events':selected,'gpu_active_union_ms':sum(b-a for a,b in union)/1000,'instrumented_wall_ms':(hi-lo)/1000,'scopes':{k:{'calls':v[0],'inclusive_cpu_ms':v[1]/1000} for k,v in scopes.items()},'verify_T_distribution':dict(Ts),'kernels':[{'name':k,'calls':v[0],'device_ms':v[1]/1000} for k,v in sorted(counts.items(),key=lambda x:-x[1][1])],'limits':['Kernel names/scoped request phases do not yet recover exact MMVQ/expert dimensions and residency.','Prior round04 modifies only prefill queue profiling; decode kernels are unchanged.','Do not divide these perturbed timings by unprofiled request time.']}
(root/'prior-decode-summary.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:r[k] for k in ['gpu_events','gpu_active_union_ms','instrumented_wall_ms','scopes','verify_T_distribution']},indent=2),flush=True)
print(json.dumps(r['kernels'][:18],indent=2),flush=True)
