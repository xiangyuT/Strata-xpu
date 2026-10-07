# B70 decode trace, roofs and TPOT tuning, 2026-10-07–08

This follows the [prefill campaign](../2026-10-07-b70-32k-tuning/current-summary.json) from
`aeea0ab7334bca3db4ef1f814ce4275d4d10c991`, on
`xiangyuT/Strata:dev/b70-iq2xs-sycl-20261007`. The primary metric is client TPOT.
All builds, GPU checks and inference run in the existing omix container. Model
weights remain on the host and are mounted read-only. No versions, driver settings,
model/KV formats or MTP policy were changed.

## Workload and measurement boundary

One Arc Pro B70 (`0xE223`, PCI `0000:cc:00.0`), oneAPI 2026.1.1, AOT `bmg-g31`,
Level Zero V2, compute runtime 26.31.39395.13, kernel 6.17.0-1007-intel. The container
has a 16-CPU quota and 96 GiB RAM limit. Its full container/image identities and
engine arguments are retained in [campaign-plan.json](campaign-plan.json).

Original Flash-Next IQ2_XS at
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:ed59f92082b1e93c0e96d60a8b11aab089b52f09`,
existing packed MTP at `de4b8e4d43b917e7706784d8bb445c9af86a3540`;
32,768 fresh input tokens and 256 output tokens, temperature 0, seed 42,
thinking off, no prefix reuse, context 131,072, INT8 KV with 32,768 resident cells,
`--spec 4 --spec-min-p 0.5`, automatic 2,048-token prefill chunks,
`--no-prefill-borrow --pcie-frac 0.30 --vram-reserve-mib 1536`.
Requested expert-cache slots are 17,772; actual capacity is 18,609 (25 GiB),
with the remaining 5,967 experts mirrored in 8.02 GiB of host USM.

The three recorded request bodies and model artifacts retain the aligned community
fixture from the earlier campaign. Each development B/C/B block has three excluded
32K warmups and three measured requests; cold loading and profiled requests are
excluded. This is development evidence, not formal acceptance or a quality study.

Client TTFT is request start to first nonempty SSE text delta. Client TPOT is
first-to-last nonempty text-delta arrival divided by 255 engine-token intervals.
Text deltas may coalesce tokens. IPC TPOT separately measures the frontend reader's
256 token-line arrivals; engine `decode_ms/256` is a different metric. Raw delta
and token gaps are preserved, and are not interchangeable.

## Decode structure and attribution

The complete measured decode window contains 292,808 device events. Their sum and
active union are both 3,694.910 ms; the instrumented wall window is 5,246.617 ms.
These are diagnostic trace times, not unprofiled utilization or client latency.
The request executes 102 verify windows (T=1/2/3/4/6: 21/14/13/53/1), interleaved
with draft and commit/emit. Normal CPU stage timings put verify near 85–87% of decode.
The frozen MTP policy means TPOT is an accepted-output-token metric, not one
kernel invocation or one verify step.

| Family | Device event sum | Share of this same capture |
|---|---:|---:|
| Dense MMVQ | 1,075.895 ms | 29.12% |
| Gated residual (GR) | 595.255 ms | 16.11% |
| Expert gate/up | 581.340 ms | 15.73% |
| Expert down | 414.382 ms | 11.21% |
| QSA | 309.793 ms | 8.38% |
| GDN | 216.248 ms | 5.85% |

[prior-decode-summary.json](prior-decode-summary.json) retains every kernel name,
count, geometry and event sum. The small ring-wait/fetch/scatter/empty PCIe-plan
launches were counted; their timing share does not justify prioritizing them.
Zero CPU expert calls do **not** mean zero PCIe traffic: GPU kernels read host USM.

Compiled CPU annotations identify exact API shapes and layers. Cached command-graph
replays do not reexecute the kernel wrapper on the CPU; deferred Level Zero
finalization prevents assigning all device events to the nearest CPU scope.
The first position-filtered attempt missed startup graph captures and was
superseded by a decode-only capture. Missing bindings remain missing, rather than
being filled with guessed dimensions.

In the separate shape-annotated capture (binary SHA-256
`4ce155dfaada92ce4bdc79f06cbed485384dd22ac1ed461450ef23543939d7f0`),
all 10,295 GR up events are uniquely bound through the ordered norm/down/reduce/up
producer chain and all four launch geometries. Actual dimensions are D=10,240,
LR=320, HC=4, embedding=2,560, BF16 weights; up has 160 work-groups. This supports
the exact GR model. Its instrumented timing source is the shape capture, while
the family-share table above uses the earlier whole-window trace. These captures
are not interchangeable performance samples. Dense MMVQ/QSA/GDN family inventories
are complete, but their
full per-event shape and physical-traffic roofs are **not** proved here.

The separate routing audit copies each plan into persistent host USM before its
scratch is reused and reads it after the existing window wait. It adds diagnostic
copies and CPU output, so its timings cannot be used for performance claims.
The measured request validates 4,896 plans and 145,440 routed entries, including
group starts, output permutation, token bounds, format and host/device allocation.
There are no explicit PCIe-plan groups in this fixture, while host-mirrored expert
groups remain present. [Routing summary](09-expert-routing-audit/summary.json).

## Measured roofs and quantitative limits

The roof bench uses initialized pseudorandom 256 MiB buffers, two-second warmups
and seven event-timed samples. It uses the same DPCT in-order queue as the engine.
The initial plain-SYCL-queue attempt stalled in UR V2 event waiting and yielded no
measurement; its failure receipt is retained. No backend or driver workaround
was applied to the measured engine.

| Standalone kernel | Peak | Median |
|---|---:|---:|
| Device copy, read+write useful bytes | 530.209 GB/s | 529.980 GB/s |
| Device read | 560.336 GB/s | 560.258 GB/s |
| Mapped-host useful read | 17.985 GB/s | 17.979 GB/s |
| FP32 SIMD FMA | 22.785 TFLOP/s | see raw samples |
| INT8 SIMD dp4a | 45.612 TOP/s | see raw samples |

These decode paths use SIMD instructions; advertised XMX throughput is not their
compute roof. [Raw roof samples](roofs-retry-01.jsonl).

Read-only 100 ms Sysman counters observe decode DRAM reads at 164–168 GB/s,
writes at 26.5–26.9 GB/s, and PCIe RX at 2.46–2.78 GB/s. Endpoint interpolation
and whole-device scope are explicit limitations. Standalone calibration validates
counter units and shows physical PCIe RX bytes are about 1.423 times the useful
read payload; physical counter bytes are kept separate from compulsory/API bytes.
VTune Metrics Discovery was unavailable; Sysman byte counters were available.

The whole-device bandwidth relaxation gives a 4.927–4.968 ms/interval reference,
against observed TPOT around 14.57 ms. **This does not predict attainable TPOT or
prove a removable 66% gap.** It assumes transferable streaming rates and overlap,
and omits instructions, codebook decoding, dependencies, cache behavior, launches
and host costs. [Aggregate model](aggregate-roofline.json).

GR up has 6,553,600 weight tensor bytes per invocation. Assuming these useful
bytes stream from device memory, its copy-roof reference is 12.360 us, versus
instrumented means of 20.893/22.678/24.801/
26.854 us for T=1/2/3/4. The 126.214 ms aggregate bookkeeping gap is not measured
removable work. Activation/epilogue traffic, nonlinear instructions and reductions
are omitted from that lower-bound model. SLM-limited occupancy was not measured.
[Exact shape model](gr-roofline.json).
[Standalone roofline figure](roofline-model.svg) ([PDF](roofline-model.pdf)).
The historical JSON labels these tensor bytes `compulsory`; that is not a
measurement of per-kernel compulsory DRAM traffic. Cache residency can change the
physical traffic and applicable bandwidth tier.

## Iterations and pretrial cost analysis

1. **Static T capacity:** reduces reserved SLM from 12 KiB to 1.5/3/6 KiB.
   Byte checks passed, but direct-call micro gains were marginal. Occupancy/stall
   causality was not quantified before trial. Removed from production; no E2E gain claim.
2. **Prepare GR lo once:** the old up stage repeats T×320 SiLU preparation and
   writes in 160 work-groups. The optional route prepares it in the existing down
   reduction, preserving signed zero, arithmetic and reduction order. Real-weight
   checks cover T=1..8, pending writes on/off, all visible outputs and guards.
   The isolated B/C/B observed TPOT 14.579545 → 14.529435 ms (-0.3438%), 3/3 wins,
   baseline TPOT drift 0.133%. TTFT drift was 2.873%; no TTFT/total gain is claimed.
3. **SG16 paired lanes:** preserves the old reduction order and passes correctness,
   but native-graph GR micro regresses 1.6–5.3%. Shuffle cost was not quantified
   before trial. Rejected before E2E and removed from production.
4. **IQ2_XXS shared preparation:** the actual IQ2_XS artifact is mixed-format.
   Sixteen-type GU groups have sizes 1/2/3/4 in counts 17,038/4,665/1,626/521.
   Sharing preparation in two/four-entry chunks changes 33,330 preparation passes
   to 25,476, a **23.564% work-count reduction**. This format represents only
   140.453 ms, or **3.801%** of the same capture's device event sum. Preparation's
   cost fraction is unknown: the optimistic proportional screening bound is
   **33.097 ms**, with 8.274/16.548/24.823 ms at assumed fractions 25/50/75%.
   These are sensitivity scenarios, not measured savings or endpoint predictions.
   The routing counts and timing shares come from separate diagnostic captures
   of the same fixed workload, with slightly different verify-window mixes.
   The [provenance notice](cost-model-provenance-notice.json) makes this assumption
   explicit; the screening estimate is not an exact recoverable workflow bound.

Round 4 freezes this cost model before implementation. It hoists weight grid,
sign and scale preparation while retaining each activation's integer dot,
integer multiply/divide truncation, FP32 scale and lane reduction. Main IQ2_S
groups already share preparation, and down remains unchanged. Forty-eight actual
and boundary fixtures × five finite input patterns pass GU/final byte comparison,
candidate GU graph replay and guards. Actual grouping/placement is reused, but
expert IDs are representative real GGUF selections and activations are synthetic.

The GU-only graph micro rotates eight plans per T, uses two-second/1,000-call
warmups and three-second/2,000-call measurements. B/C/B changes for T=1/2/3/4 are
+4.175% / -10.892% / -10.249% / -22.431%; baseline drift is below 0.16%.
The T=1 regression is retained. [Micro evidence](10-iq2xxs-graph-micro/summary.json)
and [frozen workflow plan](iq2xxs-bcb-plan.json).

The isolated round-4 workflow B/C/B observes client TPOT **14.559464 → 14.523083
ms/token (-0.2502%)**, 3/3 wins and baseline drift -0.0312%. All replies match,
with 18,609 actual cache slots in every block. IPC TPOT changes -0.2493%; engine
decode changes -0.2464%. TTFT is 31.960 → 31.965 seconds (+0.0172%), effectively
unchanged. [Paired workflow evidence](iq2xxs-bcb-summary.json). The representative
graph screening estimate was 18.674 ms per captured call mix; the measured engine
decode change is 9.200 ms. The former is not an endpoint prediction: selected
weights, activations, cache history and graph batching differ. The workflow
measurement defines the improvement.

A subsequent native-graph GR sidecar also shows about 19–23% improvement at T≥2,
which remains much larger than the measured model improvement. The harness used
packed per-token output structs (stride 2,888 floats), whereas the engine uses
separate lo/rs/inject/mixed arrays with strides 320/4/4/2,560. Thus switching to
graph replay alone does not explain the projection gap. The original source and
measurement remain preserved; the harness now matches those source
strides. Its new 640-case graph correctness check passes, and the native-graph
B/C/B component gains are 3.76/4.10/5.60/6.12/6.66/6.45% for T=1/2/3/4/6/8,
with baseline drift below 0.4%. Matching the layout removes much of the earlier
inflation, while remaining differences in actual activations, cache history,
full graph composition and executable remain unmodeled. No endpoint projection
is made. [Layout audit](gr-layout-audit-notice.json),
[corrected micro](18-gr-stride-micro/summary.json).

The earlier weighted **direct API** projection of 115.92 ms for GR lo was invalid:
it uses a different submission mode and output layout. The new native-graph
measurement shows submission overhead alone cannot explain the discrepancy. The
original artifacts remain unchanged and a [dated supersession notice](direct-api-projection-superseded.json)
withdraws that projection. Only complete workflow measurements establish TPOT.
GR shape/profile capture and prepared-Lo route reprofile also use different
annotation/filter configurations; their 253.46 → 246.06 ms up totals are diagnostic
context, not a controlled 7.4 ms saving. Only the paired workflow results establish
the retained improvement.

The [candidate audit](candidate-cost-audit-v2.json) preserves the missing causal
evidence of the earlier trials; it is historical and predates the routing audit.

## Final workflow comparison

The final engine removes the two rejected production variants and keeps both
successful changes opt-in. A new normal-binary B/C/B measures the combination
directly, using three measured samples and three excluded warmups per block.

| Metric | Paired baseline mean | Candidate mean | Paired mean change |
|---|---:|---:|---:|
| Client TPOT | 14.559223 ms/token | 14.486293 ms/token | -0.5007% |
| IPC-arrival TPOT | 14.559010 ms/token | 14.485626 ms/token | -0.5038% |
| Engine decode | 3,738.150 ms | 3,719.767 ms | -0.4917% |
| Client TTFT | 31.965413 s | 31.963660 s | -0.0055% |
| Total request | 35.678384 s | 35.657874 s | -0.0575% |

All three paired TPOT samples improve (-0.5457/-0.4696/-0.4867%), with baseline
drift -0.0591%. Outputs match both baselines, actual cache is 18,609 in every
block, and args/model/KV/MTP/sampling are identical. Draft offered/accepted counts also
match sample-by-sample (201/155, 203/159, 204/156) in all three blocks. TTFT remains effectively
unchanged. This is a small local decode improvement; it does not close the
aggregate reference gap or establish general throughput.
[Full paired results](final-bcb-summary.json), [frozen plan](final-bcb-plan.json).

Final normal engine SHA-256:
`1b260dc85a0c14349f41a1f3e037356cef7f63a572197c8d8d63da76e7fd6032`.
The final real-weight graph checks pass 640 GR cases and 240 expert cases,
including GU/final graph replay, empty-group no-write, signed zero, finite outputs
and buffer guards. `gr_parity` and `native_grouped_parity` also pass with both
candidate flags enabled. The later harness-layout correction changes only the
benchmark executable, not this measured engine or its source.

The separate actual-model route capture records 9,792 GU events in the measured
request, including 2,244 new IQ2_XXS launches (1,122 active-plan and 1,122 empty
stride-four calls), and zero legacy format-16 GU launches. The profiled response
matches the normal one. These timings are excluded from the workflow comparison
and not compared to the differently instrumented earlier trace.
[Route receipt](17-final-gu-route-reprofile/route-summary.json).

## Reproduction and evidence storage

Normal builds keep `STRATA_SYCL_WORKLOAD_TRACE=OFF`. The new opt-in flags are
`STRATA_SYCL_GR_LO_ONCE=1` and `STRATA_SYCL_EXPERT_MULTI16=1`; each defaults off.
The latter applies only to grouped IQ2_XXS GU with eight lanes per row. Existing
CUDA/HIP sources and legacy SYCL defaults remain intact.

Build `strata`, `gr_lo_once_bench`, `iq2xxs_reuse_bench`, `decode_roof_bench`
and optional Linux `decode_telemetry` inside the recorded omix runtime. Run the
GR bench against the recorded pack directory with `--graph`. Run the expert bench
against GGUF shard 1 and `10-iq2xxs-graph-micro/boundary-plans.txt`; use the original
`plans.txt --micro` for the 32-fixture component comparison. Run normal inference
through the recorded `run_latency_block.py`, three warmups and three samples per
block, using the frozen configuration and request hashes.

Profiling builds enable `STRATA_SYCL_WORKLOAD_TRACE=ON`; CPU scopes use
`STRATA_WORKLOAD_TRACE_FILE` and `STRATA_WORKLOAD_TRACE_DECODE_ONLY=1`.
`STRATA_SYCL_ROUTING_AUDIT_FILE` additionally enables graph plan copies and must
be excluded from performance arms. Primitive scopes primarily describe graph
capture; they are not a per-replay operator trace.

[artifact-index.json](artifact-index.json) hashes compact evidence and points to
the immutable raw directories. Multi-GB traces, model files and executables remain
outside Git. Rejected variants and their original harness are preserved under
`rejected-gr-candidates-source/` for audit, without adding rejected runtime flags
to the final engine. No formal acceptance, other shape/model/GPU claim, or matched
cross-platform TTFT/TPOT comparison is made.

The archived `.patch` retains exact Git context whitespace for reproducibility;
source/document whitespace checks exclude that raw patch. Generated SVG trailing
whitespace is normalized without changing plotted data.
