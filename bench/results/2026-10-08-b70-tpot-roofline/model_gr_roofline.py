import collections,json,re
from pathlib import Path
root=Path(__file__).resolve().parent
trace=root/'03-graph-contract-trace'
summary=json.loads((trace/'graph-trace-summary.json').read_text())
contracts=[x for x in summary['capture_contracts'] if x['scope']=='decode.gr_read_api']
shapes={(x['N'],x['K'],x['ldx'],x['ldy'],x['quant_type']) for x in contracts}
if shapes!={(10240,320,4,2560,30)}:raise RuntimeError('GR capture contract mismatch')
Ts={x['T'] for x in contracts}
roofs={x['name']:x for x in map(json.loads,(root/'roofs-retry-01.jsonl').read_text().splitlines()) if x['kind']=='roof'}
rows=sorted((json.loads(x) for x in (trace/'decode-correlation.jsonl').open()),key=lambda x:x['start_us'])
def grid(name):
 m=re.search(r'\[SIMD32 \{(\d+); (\d+); (\d+)\}',name)
 return tuple(map(int,m.groups())) if m else None
state=None;bound=[];rejected=collections.Counter();all_up=0
for row in rows:
 name=row['kernel'];g=grid(name)
 if '::gr_norm_split_k>' in name:
  if not g or g[0]%4 or g[1:]!=(1,1):state=None;continue
  T=g[0]//4
  state={'T':T,'events':[row],'stage':'norm'} if T in Ts else None
 elif '::gr_down_sliced_k>' in name:
  if state and state['stage']=='norm' and g==(80,1,1):state['events'].append(row);state['stage']='down'
  else:state=None
 elif '::gr_down_reduce_k>' in name:
  if state and state['stage']=='down' and g==((state['T']*324+127)//128,1,1):state['events'].append(row);state['stage']='reduce'
  else:state=None
 elif '::gr_up_multi_kernel_' in name:
  all_up+=1
  if state and state['stage']=='reduce' and g==(160,1,1):
   events=state['events']+[row]
   if any(a['start_us']+a['duration_us']>b['start_us']+0.02 for a,b in zip(events,events[1:])):rejected['overlapping producer chain']+=1
   else:bound.append({'T':state['T'],'up_event':row,'chain_start_us':events[0]['start_us'],'chain_device_us':sum(x['duration_us'] for x in events),'contract':{'N':10240,'K':320,'HC':4,'n_embd':2560,'weight_dtype':'BF16','activation_accumulation':'FP32','weight_layout':'row-major [10240,320]'},'binding_method':'Captured API invariants plus verified norm/down/reduce/up dispatcher geometry and ordered nonoverlapping producer chain'})
  else:rejected['incomplete/ambiguous producer chain']+=1
  state=None
byT=collections.defaultdict(lambda:[0,0.0,0.0])
for b in bound:
 v=byT[b['T']];v[0]+=1;v[1]+=b['up_event']['duration_us'];v[2]+=b['chain_device_us']
modeled=[]
for T,v in sorted(byT.items()):
 weights=2*10240*320
 operations=2*T*10240*320
 lower=weights/(roofs['device_copy']['peak']*1e3)
 compute=operations/(roofs['fp32_simd_fma']['peak']*1e3)
 up_us=v[1]/v[0]
 modeled.append({'T':T,'calls':v[0],'instrumented_up_mean_us':up_us,'instrumented_up_device_ms':v[1]/1000,'instrumented_chain_device_ms':v[2]/1000,'modeled_matmul_operations':operations,'weight_only_compulsory_bytes':weights,'weight_only_arithmetic_intensity':operations/weights,'weight_only_memory_lower_bound_us':lower,'matmul_only_compute_lower_bound_us':compute,'weight_only_ratio_to_peak_copy_roof':lower/up_us,'weight_only_ratio_to_median_copy_roof':weights/(roofs['device_copy']['median']*1e3)/up_us,'current_local_bytes':8*(320+4*16)*4,'candidate_local_bytes':(1 if T==1 else 2 if T==2 else 4 if T<=4 else 8)*(320+4*16)*4})
result={'evidence_class':'capacity_diagnostic','performance_claim':False,'family':'fused_gr_read_multi up stage','source_revision':'aeea0ab7334bca3db4ef1f814ce4275d4d10c991 plus CPU annotation overlay','all_up_events':all_up,'exact_bound_up_events':len(bound),'coverage_pct':100*len(bound)/all_up,'rejected_bindings':dict(rejected),'shapes':modeled,'binding_proof':['Real captured GR API arguments uniquely establish N=10240,K=320,HC=4,n_embd=2560,BF16 weights.','Default split norm grid is T*HC; down is D/128=80; reduce is ceil(T*(LR+HC)/128); up is n_embd/16=160.','Each event chain must match all four dispatch geometries and preserve nonoverlapping producer order; no nearby arbitrary CPU operator is used.'],'removable_cost':'gr_up_multi_kernel reserves 8-token SLM arrays and carries runtime token-loop bounds even for T1/2/3/4; specialize capacity without changing actual T, data layout, accumulation order, barriers or route defaults.','limits':['Ratios use instrumented GPU event time; calibrate selected shapes with unprofiled event micro before projecting TPOT.','Weight-only compulsory bytes and matmul-only operations omit epilogue/activation traffic, sigmoid and reduction work; roof ratios are diagnostic lower-bound models.','No automatic attribution to other kernel families or physical per-kernel traffic.','A theoretical gap is not an achieved speedup or proof that SLM is the limiting resource.']}
(root/'gr-roofline.json').write_text(json.dumps(result,indent=2)+'\n')
with (root/'gr-up-exact-bindings.jsonl').open('w') as out:
 for v in bound:out.write(json.dumps(v)+'\n')
plan={'candidate':'GR up token-capacity specialization','hypothesis':result['removable_cost'],'trace':'gr-up-exact-bindings.jsonl','roofline':'gr-roofline.json','frozen_shape_domain':{'N':10240,'K':320,'HC':4,'n_embd':2560,'actual_T':[1,2,3,4,6],'candidate_capacities':[1,2,4,8]},'preserve':['FP32 operation/reduction order','BF16 weight layout','actual T and output bytes','barriers and in-order queue','default route','model/KV/MTP/cache fixtures'],'correctness':'baseline/candidate byte equality on all visible GR outputs and scratch, T1..8, pending write on/off, real weights plus adversarial inputs; guards and finite outputs','micro':'8 weight/input sets; same-process alternating B/C/B continuous chunks; >=1000 calls and2s warm, >=2000 calls and3s timed; same target; output reset outside timed call; reject nonrepeatable or unstable direction','E2E':'existing32768/256 fixture; 3 warmups +3 measured requests per fresh B/C/B block; TPOT primary, exact outputs, MTP counts and baseline drift; no GPU profilers','evidence_class':'development_performance_comparison','remote_status':'local iteration only; no new push authorization assumed'}
(root/'round-01-gr-static-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'all_up_events':all_up,'bound':len(bound),'coverage_pct':result['coverage_pct'],'T_models':modeled},indent=2))
