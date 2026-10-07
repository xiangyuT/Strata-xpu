# Intel Arc (experimental)

> **Fork update, 2026-10-07:** this fork aligns the SYCL port with the shared engine headers at
> `82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`. The current build and B70 experiment are recorded in
> [Fork build check](#fork-build-check-2026-10-07). Earlier versions and community measurements below retain
> their original identities and conditions.

Strata 0.1.39 includes an **experimental Intel Arc engine**: Strata's own engine ported to SYCL (Intel oneAPI).
maxfridbe wrote it in [#423](https://github.com/Niko1221/Strata/pull/423), with fixes from the people testing it.
The code is in `sycl/`, and the port's own notes, measurements and maintenance procedure are in
[docs/INTEL.md](INTEL.md). This page covers what you need, how to build it, and what has been tested.

**Experimental means:** the Strata maintainers have no Intel GPU. We compile the port and run its kernel tests, but
we have not run it on an Arc. Every result on Arc hardware below comes from the community. The NVIDIA and AMD
engines are unchanged: the Intel build is a separate CMake target, off by default.

There is **no ready-made Intel engine** in the release zips. You build it from source on Linux.

## What has been run, and by whom

| | Hardware | Result | Reported in |
|---|---|---|---|
| Port author | Arc Pro B70 32 GB, Ubuntu 24.04 | Coder IQ1_M: 70-78 tok/s decode, ~790 tok/s prompt; IQ2_XS: 51-64 tok/s; 256K context measured | #423, [INTEL.md](INTEL.md) |
| Community | 2x Arc Pro B70, `--layer-split` | Flash-Next IQ3_XXS 66 tok/s decode, 394 tok/s prompt (with the `stage_room` fix that is now in 0.1.39) | #423 |
| Community | Arc Pro B50 16 GB | Coder IQ1_M ~23 tok/s, IQ2_XS ~25-27 tok/s, up to 128K | #423 |
| Community | Arc B580 12 GB, WSL2 | IQ2_XS ~21 tok/s, Coder ~15 tok/s; **device loss also seen** | #423 |
| Community | 2x Arc Pro B60 24 GB, `--layer-split` | Coder IQ1_M: 56.6 tok/s decode, 428 tok/s prompt (2,129 tokens); Flash-Next IQ2_XS: 58.6-61.3 tok/s decode, 436 tok/s prompt; all experts in VRAM; 8K context | [bench/results/2026-10-04-community-2x-arc-pro-b60](../bench/results/2026-10-04-community-2x-arc-pro-b60/README.md) |
| Community | one Arc Pro B60 24 GB | Coder IQ1_M with 4,042 of 12,288 experts mirrored in RAM: 11.9 tok/s decode (needs the ring-wait fix from the same report) | same |
| Strata maintainers | no Arc | compile check and kernel tests on a CPU device only (below) | this release |

The 0.1.39 port re-migrates 0.1.38's port onto the 0.1.39 engine sources (the #606 NaN fix, the #649 verify
trace, the new prompt paths). 0.1.39's `sycl/` did not compile against 0.1.39's own engine sources (#784: the
`ThreadAffinity` type of #626 and the layer range of `NativeDense::load` from #559), and its ring waits never saw a
slow layer as still running (#866, #867). Both are fixed in 0.1.40, and the two B60 reports below were measured with
exactly those two fixes on top of 0.1.39. **No one has run an unpatched 0.1.39 port on an Arc**, and the other rows were
measured on earlier versions.

The two B60 rows ran `6f32ec0` plus two small `sycl/` fixes (the compile fix and the ring-wait fix), AOT `bmg-g21`, on Ubuntu 24.04 with
`xe`, Level Zero V2, NEO 26.09.37435.12 and oneAPI 2026.1.1, without Docker. Host: Ryzen 5 5600, 64 GB RAM, PCIe 3.0 x8 per card.
Full flags, per-request timings and engine logs are in the report linked in the table.

## What was tested here (0.1.39)

- **Compiles:** Ubuntu 24.04 (WSL2), Intel oneAPI DPC++ 2026.1.1 + oneMKL 2026.1. The whole `sycl/` project
  builds with 0 errors: the `strata` engine plus all 157 targets (the kernel parity tests and benches). The build is
  SPIR-V (JIT). The AOT build (`STRATA_SYCL_AOT`) was not built here, because it needs `ocloc`.
- **Kernel parity tests on a CPU** (`ONEAPI_DEVICE_SELECTOR=opencl:cpu`, Intel's OpenCL CPU runtime, AMD Ryzen 5 7600):
  14 of 25 pass. That is the same set that PR #423's own 0.1.38 port passes on that device: the failures are tight
  float tolerances on the CPU's math (rel 3e-6 against a 1e-6 limit), model fixtures that are not present, and two
  tests that time out on a CPU. This is a check that the kernels compile and compute, not a test of an Arc.
- **Setup:** `--backend sycl` warns, then hands over to `sycl/setup_intel.py` on Linux (unit tests:
  `tools/test_setup_sycl.py`).
- **Unchanged:** the CUDA engine's greedy output, checked byte-identical against the gated 0.1.39 build (Q2_0 and Coder).
  The HIP build is not affected (the option is off by default).

**Not tested by anyone yet:** Windows (no native build path; see below), Arc on WSL2 for the 0.1.39 port, the Alchemist
A-series, integrated Arc GPUs (the B390 / Panther Lake in #515; the shared-memory planning does not exist yet), and
images (not wired on Intel).

## What you need

- **Linux**, e.g. Ubuntu 24.04, with Intel's GPU driver (the `xe` or `i915` kernel driver plus the compute runtime /
  Level Zero; on Ubuntu, `intel-opencl-icd libze1 libze-intel-gpu1`, or Intel's
  [client GPU guide](https://dgpu-docs.intel.com/driver/client/overview.html)).
- **Intel oneAPI**: the DPC++ compiler (`icpx`, 2025.3 or newer; 2026.1 is what was built here) and **oneMKL**. About 5 GB.
- `cmake` 3.24+, `ninja`, `git` (the build fetches ggml unless you point `STRATA_GGML_DIR` at a llama.cpp checkout),
  Python 3.
- For `setup --backend sycl` today: **Docker**. `sycl/setup_intel.py` runs the engine in the `strata-sycl-dev`
  image built from `sycl/tools/Dockerfile`. Note that the Dockerfile starts from a community llama.cpp SYCL image
  (`ghcr.io/snailium/...`), not an Intel or Strata image.
- VRAM: the port keeps the experts on the card (`--stream-experts`). A 32 GB card holds the Coder IQ1_M or IQ2_XS.
  Smaller cards mirror part of the experts in RAM and are slower.

## Build (Linux)

Install oneAPI from Intel's apt repository (this is what was used here):

```sh
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel ninja-build cmake git
```

Then build the engine. Either command works: the first goes through the top-level CMake option, the second
configures the `sycl/` project directly.

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_ENABLE_SYCL=ON
#   or: cmake -S sycl -B build-sycl -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl --target strata
```

Options:

- `-DSTRATA_SYCL_AOT=bmg-g31` (Arc Pro B70) or `bmg-g21` (B580 / B570 / Pro B60) compiles the GPU code ahead of
  time. This needs `ocloc` (Intel's `intel-ocloc` package). Without it, the first start JIT-compiles every kernel,
  which takes about 47 s.
- `-DSTRATA_SYCL_PARITY=OFF` skips the kernel tests (on by default). Run them with
  `ctest --test-dir build-sycl` on the card.

`setup_intel.py` looks for `build-sycl-aot/strata` or `build-sycl/strata` in the checkout. With the
top-level option the engine is at `build-sycl/sycl/strata`, so either use `-S sycl` or set `STRATA_SYCL_BIN`.

## Setup and running

```sh
./setup.sh --backend sycl [setup's usual options, e.g. --model IQ2_XS --context 32768]
```

This prints the experimental warning and continues with `sycl/setup_intel.py`. That script finds the Arc in sysfs,
uses the SYCL engine you built, and writes the config and `run-<model>.sh`. It still downloads and packs the model
the usual way. To run the engine by hand (no Docker), see "How to run it by hand" in [INTEL.md](INTEL.md).

Things that matter on an Arc (details in INTEL.md):

- **Do not ask for more VRAM than the card has.** On the `xe` driver, an allocation past VRAM can push buffers into
  RAM until the machine runs out of memory and stalls. Leave about 1.5 GB free.
- `SYCL_CACHE_PERSISTENT=0`: the persistent JIT cache crashed on Xe2 during the first compile.
- Two cards: `ONEAPI_DEVICE_SELECTOR=level_zero:*` (the image pins `level_zero:0`; `strata-sycl.sh` now passes the
  variable through) and `--layer-split`.
- **Arc Pro B60:** the PCI id is `8086:e211` (`lspci -nn`, the kernel's `xe` id list files it with the BMG-G21 cards), `sycl-ls` prints
  `Intel(R) Arc(TM) Pro B60 Graphics 20.1.0`, and `ocloc ids bmg-g21` prints 20.1.0, so `-DSTRATA_SYCL_AOT=bmg-g21` is the right target.
  `setup_intel.py` knows both B60 ids, `e211` and `e221` (both seen on B60 cards). On a `--layer-split` the startup line "N experts are neither in VRAM nor mirrored"
  counts the other card's layers; the lines `100% of the experts resident` that follow are the ones to read. Set `STRATA_MIRROR_MIB=0` there,
  or the pinned mirror is allocated and not used.
- **`setvars.sh` and `set -u`:** `source /opt/intel/oneapi/setvars.sh` in a shell with `set -u` stops at `OCL_ICD_FILENAMES: unbound variable`
  (oneAPI 2026.1.1). Source it before `set -u`, or run `set +u` around it.
- **Without Docker:** `sycl/serve/strata-sycl.sh` needs Docker. Natively, `source setvars.sh`, export `SYCL_CACHE_PERSISTENT=0`,
  `ZES_ENABLE_SYSMAN=1` and `STRATA_VERIFY_DEVICE_PLAN=1`, then run `build-sycl-aot/strata` or `sycl/serve/server_intel.py` with an `exe` that does this.
  A normal user is not in the `render` group on Ubuntu, and then `sycl-ls` shows only the CPU device.
- `STRATA_VERIFY_NO_HOST=1` (set by `strata-sycl.sh`) is only valid when every expert is in VRAM. On smaller cards
  that path is the one that has hung, and #667 found the likely reason: the GPU does not see the CPU's flag
  stores without a system fence.

## Windows

There is no Windows path yet. `setup --backend sycl` on Windows stops and points here. oneAPI exists for Windows,
but `sycl/CMakeLists.txt` uses GCC-style flags (`-mavx512f`, `-fp-model=precise`, `-qmkl`) and the runner is a
bash/Docker script, so a native Windows build would need work. Nobody has tried it. WSL2 with an Arc has been
used by one tester (the B580 row above), but setup cannot detect the card there, because it reads `/sys/class/drm`,
which WSL2 does not have.

## Reporting a problem

Open an issue with: the card, the driver version, `sycl-ls` output, the oneAPI version, the model and flags, and the
engine's stderr. Results from real cards are what move this from experimental to supported.

## Fork build check (2026-10-07)

> Update 2026-10-07: this section retains the first 8K experiment. The later
> [aligned workload and GPU-count results](#aligned-workloads-and-gpu-count-2026-10-07) follow below.

Built with oneAPI DPC++ 2026.1.1 and oneMKL in `intel/omix:0.4.0-devel-ubuntu24.04`, AOT `bmg-g31`, and run on one
Arc Pro B70 (`0xE223`), Linux kernel 6.17.0-1007-intel, compute runtime 26.31.39395.13. The port's version label is
still `0.1.39-sycl`; no project version was changed.

The compatibility changes keep the existing default SYCL paths. They add the shared headers' registration-ready
arguments, support input row strides in BF16 and native GEMM (including dequantization in row slices), and keep CUDA
driver VMM unavailable. The multi-token gated-residual read returns false because it does not write fused q8_1
images. Optional fused GDN history commits, fused GDN q8_1 output, resident-plan error buffers, shared-expert fused
gate modes, and padded prefill output layouts are not implemented here: asking for them throws a named error.

Six selected GPU checks pass: `gemm_stride_parity`, `gr_parity`, `gdn_parity`, `quantize_act_parity`, `kv_q8_parity`,
and `iq_multi_parity`. `native_expert_parity` passes on the actual IQ2_XS GGUF's layers 0, 23 and 47. Through the
HTTP server, arithmetic, a Chinese capital question, and a Python addition function pass smoke checks.

IQ2_XS came from `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`; both shards match their published SHA-256. All 31 local BF16 MTP tensors
match the hashes at `de4b8e4d43b917e7706784d8bb445c9af86a3540`, and were packed as q2_0. At an 8,192-token context
with INT8 KV, 18,140 experts are in VRAM, 6,436 are mirrored in 8.66 GiB of host memory, and startup reports 2,107 MiB
of free VRAM.

Three measured requests per workload, after a separate warmup, temperature 0, seed 42, thinking off, 256 generated
tokens, MTP `--spec 4 --spec-min-p 0.5`, `--prefill auto --no-prefill-borrow --vram-reserve-mib 1536`,
`--adapt-every 1000000 --prompt-cache 0`. Every measured request reports zero reused prompt tokens.

| Workload | Prompt tokens | Decode, three runs (tok/s) | Mean decode (tok/s) | Prompt, three runs (tok/s) |
|---|---:|---|---:|---|
| English explanation with a code example | 40 | 69.5 / 69.5 / 69.5 | 69.5 | 136.8 / 136.8 / 136.8 |
| Chinese explanation | 39 | 60.1 / 59.5 / 60.1 | 59.9 | 119.1 / 119.0 / 119.0 |
| Technical records, then an explanation | 3,946 | 66.9 / 66.9 / 67.4 | 67.1 | 863.5 / 862.7 / 862.8 |

Prompts, responses, timings, settings, hashes and limitations:
[results.json](../bench/results/2026-10-07-b70-iq2xs/results.json). This is a development experiment with no baseline
comparison; larger contexts, vision, multi-GPU, and a complete quality evaluation were not run. The inherited CUDA
header/runtime startup warning is misleading for this SYCL build. GPU monitor readings from the Intel wrapper
were not used, because its sysfs reader currently selects the first Intel card.


## Aligned workloads and GPU count (2026-10-07)

Measured with the same SYCL binary as the first experiment, SHA-256
`8fbececa8b72cc758c9b5c9970bad761a51eb410317ed77ee89a3b59a69cd48c`,
from fork commit `35c54d714cae62115261acec4c8f34193df278e6`. All inference ran in an omix container.
The prepared multi-card runtime is a filesystem snapshot of that container, with no dependency reinstall or
project version change. Existing host weights were mounted read-only for this phase.

The measured request bodies at 4,096 and 32,768 tokens match the six request SHA-256 hashes in the
[RTX 5090 community report](../bench/results/2026-09-30-community-rtx-5090/README.md).
The GGUF revision, packed dense weights, native-expert metadata, tokenizer, expert profile and MTP runtime
also match that report's artifact hashes. The public harness was adapted only to add an excluded warmup at
each length and reject streamed errors. Its measured request generator and payload fields were preserved.

Primary settings: context 131,072; INT8 KV with 32,768 resident cells; `--spec 4 --spec-min-p 0.5`;
temperature 0; thinking off; server-default seed 42; 256 generated tokens; one request at a time.
Every count uses `--prefill auto --no-prefill-borrow` (effective 2,048-token chunks),
`--pcie-frac 0.30 --vram-reserve-mib 1536 --adapt-every 1000000 --prompt-cache 0`.
The CPU quota is 16, the container RAM limit 96 GiB. Source build and parity checks were reused because
the binary did not change; the new device configurations were checked through real inference.

One B70: `cc:00.0`. Two B70s: `18:00.0` and `cc:00.0`, split at layer 24.
Four B70s: `18:00.0`, `36:00.0`, `54:00.0`, `cc:00.0`, split at layers 12, 24 and 36.
Two and four cards passed the arithmetic and Chinese smoke checks. Their startup logs confirm all 24,576
experts resident across the stages; the unused host mirror is disabled there (`STRATA_MIRROR_MIB=0`).
The single-card baseline retains a 16,384 MiB mirror cap. This is layer splitting, not tensor parallelism.

Each cell is the median of three measured requests. Loading, short warmup and per-length warmups are excluded.
All 18 primary requests had zero reused prompt tokens, generated 256 tokens and reached the output limit.

| B70 count | Prompt tokens | Prompt tok/s | Decode tok/s | TTFT seconds | Total seconds |
|---:|---:|---:|---:|---:|---:|
| 1 | 4,096 | 772.4 | 67.4 | 5.337 | 9.112 |
| 2 | 4,096 | 762.1 | 71.6 | 5.408 | 8.959 |
| 4 | 4,096 | 654.1 | 68.3 | 6.296 | 10.029 |
| 1 | 32,768 | 820.0 | 66.9 | 40.036 | 43.847 |
| 2 | 32,768 | 831.6 | 69.3 | 39.476 | 43.021 |
| 4 | 32,768 | 663.6 | 67.1 | 49.449 | 53.126 |

Two cards increased median decode throughput by 6.3% at 4K and 3.5% at 32K, while total latency fell only
1.7% and 1.9%. Four cards did not improve decode materially and increased 32K total latency by about 21%.
These are three-request observations, not a claim of stable scaling or general speed on other workloads.

[results.json](../bench/results/2026-10-07-b70-iq2xs-aligned/results.json) retains all 36 measured requests,
including the earlier automatic-probe baseline and two borrow-setting arms, with responses, counters,
configuration, median/range summaries, request and artifact identities, and community references.
The primary GPU-count table uses the later explicitly pinned single-card baseline in the same prepared
runtime as the multi-card arms. The earlier automatic-probe prefill results differ by about 7%, so small
improvements should not be treated as a stable speedup. The borrow records are exploratory.

NVIDIA references use different engine revisions and host hardware, so they remain whole-system references.
A configured 128K limit is not a measured 128K workload: only 4K and 32K inputs were tested here.
Vision, concurrent requests, general answer quality and formal acceptance were not evaluated.


## Single-card focus and community references (2026-10-07)

**Follow-up, 2026-10-07:** the table in this section is the preserved aligned hardware reference.
The fixed-cache 32K tuning campaign below has its own paired baseline and local component profiles.
Do not derive code speedups by comparing its new timings with this historical table.

The user's current priority is single-card B70 optimization. The completed two- and four-card results above
remain recorded as capacity and scaling observations. The matching fixed-parameter `1gpu` arm is the current
single-card reference; the earlier automatic-probe arm is retained separately.

The closest community comparisons use original Flash-Next IQ2_XS, 4,096 / 32,768 prompt tokens, greedy decoding,
reasoning off, 256 generated tokens, zero prefix reuse and medians of three runs. Values below are tok/s:

| Single-card platform | 4K prompt / decode | 32K prompt / decode |
|---|---:|---:|
| This fork, B70 | 772.4 / 67.4 | 820.0 / 66.9 |
| [RTX 5070 Ti, 0.1.39](https://github.com/obiscr/Strata/blob/36ddda197895b09895b5bcbb956813b35fce219b/bench/results/2026-10-06-community-rtx5070ti-5900x/README.md) | 1,992 / 102.0 | 2,817 / 101.3 |
| [RTX 5090, 0.1.29](../bench/results/2026-09-30-community-rtx-5090/README.md) | 4,269.8 / 179.4 | 5,543.2 / 175.7 |

The six request body hashes and shared model artifacts were checked against the 5090 report. That report also
uses a 131,072-token context and 32,768 resident INT8 KV cells. The 5070 Ti report uses the same public harness,
but a 65,536-token context. NVIDIA's prompt chunks, borrowing policy, CPU, PCIe link, engine revision and other
startup choices differ, so these are whole-system references rather than a controlled comparison of GPU speed.
The relative gap is larger for prefill than decode. This suggests investigating single-card prefill first,
but no local profile has yet identified the component responsible.

[The Intel port author's B70 record](INTEL.md) reports original IQ2_XS decode at 58.6 / 64.2 tok/s after
19 / 2,184 input tokens. It is useful context for this fork's roughly 67 tok/s, but uses different prompts and
port revisions. The often-cited B70 70-78 tok/s figures are Coder IQ1_M, so they are excluded from this IQ2_XS
comparison. [single-card-reference.json](../bench/results/2026-10-07-b70-iq2xs-aligned/single-card-reference.json)
records the user direction, measurement source digest, community sources, comparability classes and limits.


## Single-card 32K kernel tuning (2026-10-07)

The campaign uses the same recorded 32,768-input / 256-output requests, seed 42, greedy decoding,
reasoning off, zero prompt reuse, 131,072 context and INT8 KV. Its requested expert-cache budget is
fixed at 17,772; the variable-size native expert profile actually fits 18,609 slots (25.00 GiB).
The earlier automatic-cache hardware-reference table remains separate. Loading, three 32K warmups
per block and all profiled requests are excluded from performance comparisons.

Round 1 groups eight independent 256-value GU dequant blocks into a 256-thread work-group.
The quantization formulas, interleaved FP16 output and root-sync launch property are preserved.
All 135 format/shape cases and three real expert-weight cases passed byte-exact comparison with
guarded output buffers. In the development baseline/candidate/baseline comparison, three samples
per block, mean prefill fell from 36.187 to 35.229 seconds (-2.65%) and mean total latency from
39.974 to 39.017 seconds (-2.40%). All three paired samples improved; prefill baseline drift was
0.011%, and generated text matched both baseline blocks exactly. Decode time was essentially unchanged.

Reprofiling confirms the same 302,213 GU calls took 6.404 seconds of device execution rather than
7.347 seconds. Other leading kernels remained near unchanged. These trace times explain the
direction of the unprofiled comparison; they are not independent workflow speedup estimates.
VTune software CPU sampling completed. GPU counters could not be collected because Metrics Discovery
failed to initialize; no host package, driver or profiling-permission changes were made.

This remains a target-local development experiment in an omix container, with host weights mounted
read-only. The legacy route remains the default; round 1 is selected with
`STRATA_SYCL_DEQUANT_GU_SG=8`. Later format and indexing experiments are still being evaluated.
[iteration-01.json](../bench/results/2026-10-07-b70-32k-tuning/iteration-01.json) retains the exact
identities, paired deltas, outputs, profiling limits and raw artifact pointer. Project versions are unchanged.

**Latency follow-up, 2026-10-07:** earlier `decode_ms / generated_tokens` is the engine's average
decode time, not a measured client token interval. A separate unprofiled observation of the typed/fixed-row
candidate recorded both raw IPC token lines and client text-delta arrival times. Three full 32K requests
had mean client TTFT 35.197 seconds, first-to-last-text TPOT 14.602 ms/token, IPC-arrival TPOT 14.602 ms/token,
and engine average decode time 14.645 ms/token. Client TPOT uses 255 intervals for 256 engine-generated tokens;
text chunking can coalesce the initial tokens. Each response had only 240-245 nonempty text deltas, so delta
intervals are retained separately from the 256 raw-token arrivals. The IPC clock measures the frontend reader,
not GPU compute or the engine producer. [latency-14.json](../bench/results/2026-10-07-b70-32k-tuning/latency-14.json)
records the definitions and each sample. Historical records cannot recover a token-gap distribution.

The third development comparison completed using a compatible retry for its after block; the first after
attempt timed out during model startup and produced no measurements. Prefill fell 0.31% and TTFT 0.31%
relative to SG8. No decode improvement is claimed, because the decode baseline drift exceeds its paired change.
**Trace follow-up, 2026-10-07:** the complete request timeline is now captured, with source CPU scopes and
GPU kernels/copies/submission flows on a shared monotonic-raw clock. All 617,912 GEMM executions in the
measured request correlate to their actual source shape, stride and API input dtype, including 302,213
gate/up and 302,213 down products. These are actual per-expert row counts, not dimensions inferred from
generic kernel names. Loading and warmup records remain in the raw capture and are excluded by the request
window. The profiled reply matches the unprofiled one.

A second device-only capture removes full API logging and compiled CPU scopes. Both captures report all
explicit copies on Compute Engine `<0,0>` and near-zero compute/copy overlap. This warrants inspecting copy
queue routing, but instrumentation still roughly doubles prefill time; observed gaps are not an unprofiled
utilization estimate. GPU reads from the mapped host mirror are not explicit memcpy events, and hardware
bandwidth/occupancy counters remain unavailable. Thus the timeline resolves execution structure and GEMM
attribution, while some hardware bottleneck attribution remains open.

The optional CPU annotations use `-DSTRATA_SYCL_WORKLOAD_TRACE=ON` and
`STRATA_WORKLOAD_TRACE_FILE`; normal builds default to OFF and add no GPU barrier or wait. The latest
normal build passed three 32K requests with matching replies: mean client TTFT 35.198 seconds and TPOT
14.599 ms/token. [current-summary.json](../bench/results/2026-10-07-b70-32k-tuning/current-summary.json)
links the exact binary identities, measured definitions, full and lighter trace summaries, and raw data.
No TPOT improvement is claimed. The next untested question is copy-queue backend routing.

**Routing qualification, 2026-10-07:** Unitrace's engine name is derived from the original command-list
queue group. Level Zero V2 can ask the driver to offload copies internally, so `<0,0>` does not prove
that the physical copy engine is unused. The device exposes a separate copy-only group, but its maximum
fill pattern is one byte; the stager's eight-byte sequence marker would need special handling there.
The raw trace observations remain valid. No forced copy engine or driver setting was introduced.

## Transfer event profiling experiment (2026-10-07)

The migrated DPCT helper enables profiling for every queue created under `DPCT_PROFILING_ENABLED`.
The prefill transfer queue inherits this property, although the phase timers query only compute/peer
events. Round 4 adds an optional in-order transfer queue with the same device, context and async error
handler, without profiling. Queue ownership is explicit; the legacy queue remains the default.

The queue protocol test passed for all three actual expert blob sizes (1,510,400 / 1,305,600 / 1,177,600
bytes), with eight rotating slots, cross-queue dependencies, full byte/guard checks and 64-bit host
sequence markers. Its aggregate wall spans are protocol diagnostics, not pure copy or LLM throughput.

The no-profiler development comparison uses three excluded 32K warmups and three measured requests in
each baseline/candidate/baseline block. All measured responses match both baselines exactly; actual
expert-cache capacity remains 18,609 in all blocks. Results below use paired baseline means:

| Metric | Baseline | Candidate | Paired mean change |
|---|---:|---:|---:|
| Client TTFT | 35.212 s | 31.968 s | -9.21% |
| Client first-to-last-text TPOT | 14.604 ms/token | 14.563 ms/token | -0.28% |
| IPC token-arrival TPOT | 14.604 ms/token | 14.563 ms/token | -0.28% |
| Total latency | 38.936 s | 35.682 s | -8.36% |

All three paired TTFT samples improved; before/after TTFT drift was 0.010%. The TPOT change is small,
so this is principally a prefill/TTFT improvement on this workload. These three-sample development
results do not establish general decode speed or formal acceptance.

Select the tested candidate with `STRATA_SYCL_PREFILL_COPY_PROFILING=0`, together with the retained
GU settings `STRATA_SYCL_DEQUANT_GU_SG=8`, `STRATA_SYCL_DEQUANT_GU_STATIC_TYPE=1` and
`STRATA_SYCL_DEQUANT_GU_STATIC_ROW=1`. Use a normal build (`STRATA_SYCL_WORKLOAD_TRACE=OFF`).
The candidate binary SHA-256 is `bfe419763ad8a94b4f0d81740127d9ace98c2570504716d7738e6f08d6a2879d`.
Weights, requests, sampling, project versions and legacy default behavior are unchanged.
[latency-bcb-summary.json](../bench/results/2026-10-07-b70-32k-tuning/round-04-copy-routing/latency-bcb-summary.json)
retains per-pair deltas, drift, token clocks and response hashes. The earlier current-summary snapshot is
preserved as `current-summary-before-round04.json`; this section supersedes its pending copy-queue status.

VTune software CPU sampling of the retained candidate completed after three excluded 32K warmups.
The measured reply remains identical and prefill took 31.898 seconds. Driver functions still dominate
CPU samples, with hidden symbols preventing unique leaf attribution; their time can overlap GPU work.
This is a separate profile, not another performance sample or proof of a CPU-bound kernel.
The result folder includes the captured request bodies and timestamp/correlation drivers. The drivers
use the same container mounts (`/src/Strata`, read-only `/models/Strata`, writable `/artifacts`) and
per-arm configuration files. Failed profile/startup attempts remain preserved in the raw artifact root.
