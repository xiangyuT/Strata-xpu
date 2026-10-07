import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
roofs={x['name']:x for x in map(json.loads,(root/'roofs-retry-01.jsonl').read_text().splitlines()) if x['kind']=='roof'}
gr=json.loads((root/'gr-roofline.json').read_text())
ai=np.logspace(-1,2.5,400)
compute=roofs['fp32_simd_fma']['peak']/1000
memory=roofs['device_copy']['peak']*ai/1000
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.2))
ax.loglog(ai,np.minimum(compute,memory),label='FP32 SIMD / device streaming reference',color='#2456a6')
ax.loglog(ai,roofs['mapped_host_read']['peak']*ai/1000,'--',label='Mapped-host useful-read reference',color='#ab5b19')
for s in gr['shapes']:
 x=s['weight_only_arithmetic_intensity'];y=s['modeled_matmul_operations']/s['instrumented_up_mean_us']/1e6
 ax.scatter(x,y,s=45,color='#258552');ax.annotate('T='+str(s['T']),(x,y),xytext={1:(8,-14),2:(-12,-20),3:(0,12),4:(12,-8)}[s['T']],textcoords='offset points')
ax.set(xlabel='Weight-only arithmetic intensity (modeled operations/byte)',ylabel='Modeled matmul throughput (TFLOP/s)',title='GR up: exact captured shapes')
ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=8,loc='lower right')
agg=json.loads((root/'aggregate-roofline.json').read_text())
measured=np.mean([s['measured_client_TPOT_ms'] for s in agg['samples']])
dram=np.mean([s['physical_DRAM_reference_floor_ms_per_interval'] for s in agg['samples']])
pcie=np.mean([s['physical_PCIe_RX_reference_floor_ms_per_interval'] for s in agg['samples']])
bx.bar(['Observed TPOT','DRAM reference','PCIe RX reference'],[measured,dram,pcie],color=['#2456a6','#258552','#ab5b19'])
for i,v in enumerate([measured,dram,pcie]):bx.text(i,v+.15,f'{v:.2f} ms',ha='center')
bx.set(ylabel='Milliseconds per token interval',title='Whole-device bandwidth relaxation',ylim=(0,17))
bx.tick_params(axis='x',labelsize=9);bx.grid(axis='y',alpha=.2)
fig.text(.5,.01,'Left: instrumented timings, tensor-byte streaming model. Right: physical counter streaming references; not achievable TPOT predictions.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.05,1,1))
for extension in ['png','svg','pdf']:fig.savefig(root/('roofline-model.'+extension),dpi=180)
print('Standalone roofline artifacts created')
