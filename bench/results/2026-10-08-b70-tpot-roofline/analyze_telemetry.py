import bisect,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
arm=root/'02-normal-telemetry'
data=[json.loads(x) for x in (arm/'telemetry.jsonl').read_text().splitlines()]
data=[x for x in data if x['kind']=='sample']
times=[(x['begin_raw_ns']+x['end_raw_ns'])/2 for x in data]
# Sysman cumulative counters are bytes. Each structure's own timestamp is used
# only for its own deltas; external RAW clock aligns counter reads to IPC ROI.
def counters(x):
 if x['pci']['result'] or any(m['result'] for m in x['memory']): raise RuntimeError('counter read failure')
 return {'pcie_rx':x['pci']['rx_bytes'],'pcie_tx':x['pci']['tx_bytes'],'dram_read':sum(m['read_bytes'] for m in x['memory']),'dram_write':sum(m['write_bytes'] for m in x['memory'])}
values=[counters(x) for x in data]
def interpolate(t):
 i=bisect.bisect_right(times,t)-1
 if i<0 or i+1>=len(times):raise RuntimeError('ROI not bracketed')
 span=times[i+1]-times[i]
 if span<=0:raise RuntimeError('clock regression')
 f=(t-times[i])/span
 return {k:values[i][k]+f*(values[i+1][k]-values[i][k]) for k in values[i]},span/1e6
out=[]
for row in json.loads((arm/'results.json').read_text()):
 if not row['measured']:continue
 raw=json.loads((arm/(row['label']+'-raw.json')).read_text())
 tokens=[x for x in raw['ipc']['events'] if x['line'].startswith('T ')]
 lo,hi=tokens[0]['raw_ns'],tokens[-1]['raw_ns']
 before,s0=interpolate(lo);after,s1=interpolate(hi)
 seconds=(hi-lo)/1e9
 traffic={k:after[k]-before[k] for k in before}
 if any(v<0 for v in traffic.values()):raise RuntimeError('counter regression')
 out.append({'label':row['label'],'engine_generated':len(tokens),'ipc_first_last_s':seconds,'client_tpot_ms':row['latency']['client_tpot_ms'],'boundary_sample_spacing_ms':[s0,s1],'physical_bytes':traffic,'average_GB_per_s':{k:v/seconds/1e9 for k,v in traffic.items()},'physical_bytes_per_token_interval':{k:v/(len(tokens)-1) for k,v in traffic.items()}})
result={'evidence_class':'capacity_diagnostic','performance_claim':False,'source':'02-normal-telemetry/telemetry.jsonl','clock_alignment':'External CLOCK_MONOTONIC_RAW read midpoints; piecewise linear cumulative-counter interpolation at IPC first/last token arrivals.','samples':out,'limits':['Whole-device byte counters; no kernel-by-kernel attribution.','First/last IPC arrival differ from engine-producer boundaries by reader scheduling.','100ms sampling and 1-3ms reads introduce endpoint interpolation error.','Observed physical traffic differs from API compulsory useful traffic; do not mix the two.','No inferred absence of host-mapped loads from CPU expert counters.']}
(root/'normal-telemetry-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(out,indent=2))
