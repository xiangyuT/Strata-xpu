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
