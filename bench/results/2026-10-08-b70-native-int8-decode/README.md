# B70 INT8 decode iteration, 2026-10-08

The original IQ2_XS model's client TPOT improves **6.68%** on one Intel Arc Pro
B70: **14.568688 → 13.595418 ms**. Each of the three paired changes exceeds 5%.
This meets the user's requested local development threshold; it is a
three-sample development comparison, not formal acceptance or a result for other
GPUs/models. The [previous 0.501% observation](../2026-10-08-b70-tpot-roofline/README.md)
remains historical evidence and is insufficient for delivery.

## Measurement

| Metric | Paired baseline mean | Candidate mean | Paired mean change |
|---|---:|---:|---:|
| Client TPOT | 14.568688 ms | 13.595418 ms | -6.680834% |
| Frontend IPC-arrival TPOT | 14.568146 ms | 13.594924 ms | -6.680736% |
| Engine decode / 256 | 14.611523 ms | 13.635156 ms | -6.682393% |
| Client TTFT | 31.958464 s | 31.978961 s | +0.064136% |

| Pair | Baseline before | Candidate | Baseline after | Client TPOT change |
|---|---:|---:|---:|---:|
| 1 | 14.682056 ms | 13.676656 ms | 14.668008 ms | -6.803230% |
| 2 | 14.308577 ms | 13.354890 ms | 14.318508 ms | -6.697517% |
| 3 | 14.713942 ms | 13.754708 ms | 14.721039 ms | -6.541754% |

Wins: **3/3**; before/after baseline mean drift: **+0.006816%**. Each pair's
comparator is the mean of its two baseline values; the reported percentage is
the mean of the three paired percentage changes. TTFT has no measured benefit.
Output text and MTP offered/accepted counts match all three arms, including
warmups. Actual expert-cache capacity is 18,609 slots (25 GiB); all 5,967 missing
experts remain mirrored in 8.02 GiB host USM; draft vocabulary remains 106,299.

[Frozen plan](model-bcb-plan.json), [derived comparison](model-bcb-summary.json)
and the three arms' structured request results are retained. Each fresh server
executes one short warmup and three excluded 32K warmups, then three measured
requests with seed 42. Prompt reuse stays zero; every measured request has
32,768 input and 256 output tokens. Client TPOT uses the first-to-last nonempty
SSE text arrival divided by 255. IPC arrivals and engine decode/256 have separate
boundaries; raw SSE/IPC logs remain under the host artifact pointer.

Both previous small-gain flags, `STRATA_SYCL_GR_LO_ONCE` and
`STRATA_SYCL_EXPERT_MULTI16`, are **0 in all three arms**. No external GPU profiler or Sysman
sampler is active in this comparison. Context 131,072, INT8 KV/resident 32,768,
spec 4/min-p 0.5, requested cache 17,772, mirror cap 16,384 MiB, PCIe fraction
0.30, reserve 1,536 MiB, adaptation interval 1,000,000 and no prefill borrowing
remain unchanged. The retained prefill configuration is SG8, static type/row on
and copy profiling off. Model/pack/MTP weights stay on the host, mounted read-only.

## What changed

The IQ2_XS GGUF contains mixed quantization formats. The selected hotspots are
IQ4_XS dense projections and Q2_0 expert down projections within those unchanged
weights. Both already use Q8_1 activations and hardware signed INT8 `dp4a`.

* `STRATA_SYCL_IQ4_LOOKUP=1`: replace the IQ4_XS nibble lookup's compare/select
  chain with an exact packed bit-select tree. The native wide geometry, scales,
  dot products and FP32 expression/reduction are retained. Its ordinary wide
  guards and fallback remain; the flag defaults off.
* `STRATA_SYCL_Q2_WORD_EXPAND=1`: expand packed Q2_0 codes directly to signed INT8
  bytes {-1,0,1,2}, using two bit-spread stages and exact add/xor bias conversion.
  This removes generic byte-permute table extraction. It selects the native
  down route only for format 42, eight lanes per row, old/split modes off.
  Other routes retain their fallback; the flag defaults off. The internal
  template key **10042 is not a new GGUF format**.

No activation/weight/KV format, accumulation order, model parameters or project
version changes. No CUDA/upstream source is removed. Both optimizations are
independently selectable; the **combined** client result is measured directly.
There is no model-level ablation claim for either flag alone.

## Selection and checks

The original complete decode capture has 3,694.910 ms device time. IQ4_XS wide
projections account for 19.584%; Q2_0 down accounts for 414.382 ms, or 11.215%.
These shares use the same complete capture denominator, not the filtered
integration trace. Captured native contracts establish the exact dense shapes;
expert contracts establish n_embd 2560, n_ff 640 and down row size 180 bytes.
They establish the operator family, without assigning arbitrary tensor/layer
owners to cached-graph replay events.

The original IQ4_XS code emitted 101 compare and 96 select instructions in the
NC4 instance. Packed bit-select passes 65,536 exhaustive combinations and
1,040 real-weight matrix cases (T1–T8, five patterns, guards and graph replay),
including the actual 106,299-row MTP subset. Its native-graph component screen
improves about 7–20% depending on the exact cell. Weighted screening alone was
insufficient for the requested 5% endpoint target.

Q2_0's affine codebook permits direct unpacking: per-byte values 0–3 plus 0x7f
cannot carry into a neighboring byte; xor 0x80 yields {-1,0,1,2}. The diagnostic
checks every packed 16-bit word against the old GPU permutes and an independent
codebook. Real-weight grouped checks compare original complete expert output,
the isolated baseline and candidate graph replay, byte-for-byte.

Q2_0 T1/T2/T3/T4 component changes are **-44.071%, -41.230%, -42.555%, -45.302%**;
maximum baseline drift is 0.031%. Each cell rotates eight representative real
weight/grouping fixtures, uses at least 2 s/1,000 excluded warmup calls and
3 s/2,000 measured calls per B/C/B arm. Inputs/groupings are representative and
adversarial, rather than captured full-model activation histories. The emitted
down kernel retains 56 `dp4a` and 53 send instructions; static MOV drops from
675 to 222 and SHR from 268 to 16. Counts explain removed work, not cycle bounds.

The integrated build passes the IQ4_XS checks again, Q2_0's 64 real-weight
fixtures × five patterns (including T5–T8 and host USM), and
`native_grouped_parity` with both new flags off/on. Empty groups leave outputs
untouched; guards and finite checks pass. No tolerance is relaxed.

[Executable instruction comparison](integration-executable-instruction-equivalence.json)
confirms the micro and integrated candidate instructions through EOT match.
Full ELF/text hashes differ in fingerprint immediates after EOT; this difference
is explicitly retained rather than described as whole-binary equality.

## Hardware and reuse

[The requested omni tuning reference](https://github.com/xiangyuT/omni-xpu-kernel-tuning)
was inspected read-only at `78e2318e7c969184b5c2477dd56c9920ae272772`.
Its quantization/matmul, DPAS packing/resource and correctness experience informs
the investigation. The local reference worktree stays at
`ce0bb4ce39ff280a13ff016040c1df11e63fbb88`; no source changes or remote writes
are made there.

Actual decode ISA contains `dp4a` and no `dpas` in the inspected dominant
IQ4_XS/Q2_0 instances. The improvement optimizes preparation for existing INT8
SIMD arithmetic. Strata's FP16 joint-matrix helper is a standalone path and
rejects several current formats, including IQ4_XS; prompt-attention XMX is off
in this configuration. Prefill calls oneMKL FP16/BF16 GEMM, but actual selected
prefill DPAS instructions have not been proven here.

[Intel documents XMX support for INT8 and other types](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/xmx.html).
Directly swapping the reference's row/channel-scaled oneDNN INT8 kernel would
violate the current along-K group scales and, for IQ2 formats, integer scaling/
truncation boundaries. An INT8 XMX adapter would need to preserve these and
measure packing, padding and reduction costs. [Screening](int8-xmx-screening.json)
uses about 4.0 TOP/s logical dot throughput for the large T4 head versus the
measured 45.612 TOP/s SIMD roof; this ratio does not prove recoverable XMX gain.

## Integrated route trace and negative results

[Route proof](route-proof-summary.json) isolates the entire fresh 32K/256 request
after excluded startup and warmups: **18,609 IQ4_XS bit-select calls and 9,792
Q2_0 word-expand calls**, with zero old selected routes. Output and MTP match the
unprofiled candidate. The two-family filtered trace proves dispatch coverage;
it is neither a complete-device denominator nor endpoint performance evidence.
It does not identify individual tensor/layer owners or an exact decode-only
interval. The original full trace remains the workload attribution reference.

The first paused route capture produced memory/barrier events but zero compute
kernel events. Its successful request receipt remains unchanged; a separate
notice marks the profiling result inconclusive. Profiling from graph creation
with plain name substrings resolves kernel capture in the next immutable attempt.

Rejected experiments are retained: the first hand-copied IQ4 harness was invalid;
ESIMD API and accumulator-argument mistakes failed their gates; FP contraction
produced a 1 ULP mismatch, and preventing contraction restored exact checks but
still regressed performance. Subgroup-register lookup passed correctness but
regressed a main cell 162%; corrected ESIMD regressed major cells roughly
169–193%. Add/xor masks remove extra MUL instructions but lose to the selected
bit-select implementation. None of these routes is enabled by the production
flags. Plans, identities, summaries and local raw pointers preserve the failures.

## Runtime and reproduction

Source base: `dev/b70-iq2xs-sycl-20261007` at
`0214cecaf4b27252a52e9e1706a2ed3f60076cee`, plus the frozen source overlay in
[integration-01/source-identity.json](integration-01/source-identity.json).
Normal binary SHA256:
`2d223006b2af5574ed83df3137bf02f454c0e827ff42f8e1fc56772d8658e2de`.
All builds, GPU checks and model runs use the existing omix container
`a4fcf657c89a4d5536b5aad26f7dfd53f68975cd4d26f785ea36ea101a76ebe8`;
image `sha256:065778dc5445d3ce2a099d7e7f019aea6cbf18e24702e0870a25d12c06ebc931`,
based on `intel/omix:0.4.0-devel-ubuntu24.04`.
Target `0xE223`, PCI `0000:cc:00.0`, renderD131; oneAPI 2026.1.1,
compute 26.31.39395.13, AOT bmg-g31, precise FP and rounded divide/sqrt,
trace OFF for all performance data. No host installation or inference tests.

Use the checked-in benchmark targets and unchanged canonical
[latency driver](../2026-10-08-b70-tpot-roofline/drivers/run_latency_block.py).
For the measured configuration set both new flags to 1 while retaining the
plan's other parameters. For B/C/B, set them to 0/1/0 respectively. The model
paths reference the existing host weights; do not put weights in the checkout.
Raw logs, ISA dumps, binaries and large Chrome traces remain outside Git at
the [artifact index](artifact-index.json) pointer. Protected branches, project
versions and container/image identity are unchanged.
