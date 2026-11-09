# DeepSeek V4.1 Flash on Intel XPU

This branch preserves the single-GPU SYCL implementation used for
DeepSeek-V4.1-Flash-1017 UD-IQ1_M with its matching DSpark Q8_0 draft.
It contains the adopted tuning mode 17, including the earlier model-format,
Engram, CPU expert, expert-cache, attention and speculative-decoding work.
The engine version remains 0.1.45.3.

## Source identity and validation scope

The snapshot starts at `d6418a14325d1ebbc178aadd8d978a95664dc01e`, based on
upstream `f57c0adcb31c00f3f216550c004dac4e713a9f70`, and includes the previously
uncommitted SYCL implementation. Its mode-17 production sources match the
retained measured build; one header retains the previously recorded final
newline normalization. The historical executable SHA-256 is
`9dc4244dae02f67925c00ee160cc07799de109a3326e157d9750043d4689e66e`.
This is provenance for the historical executable, not a reproducible-build hash.

The retained checks passed 129,280 initial logits, 33,095,680 subsequent
logits across 255 positions, 74 rollback windows, and a 256-token DSpark
generation. The adopted run used one Arc Pro B70, 12 CPU participants,
128 GiB container RAM, F16 KV, context 2048, prefill batch 128 and an expert
cache budget of 15360 MiB (1872 slots, 16105144320 bytes).
Historical endpoint validation used 1024 input tokens and 256 output tokens,
with three warmups and three measured requests per arm. It does not validate
other lengths, devices or model-quality benchmarks.

Publication preserves that implementation without a new GPU run. The later
CPU work-queue candidate and FP16/INT8 diagnostic variants were not adopted
and are absent from this snapshot. Raw evidence and binaries stay outside Git.

## Build

Use the Linux Intel oneAPI environment described in [Intel Arc setup](INTEL_ARC.md).
The historical compiler was icpx 2026.1.1 with the `bmg-g31` AOT target.
The cold-expert CPU path requires AVX-512; check the available host resources
before attempting the large model.

```sh
cmake -S sycl -B build-sycl -G Ninja \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_SYCL_AOT=bmg-g31 \
  -DSTRATA_DEEPSEEK41=ON
cmake --build build-sycl --target strata-deepseek41 -j8
```

This target is separate from the Qwen `strata` executable and the CUDA
DeepSeek configuration. No model files or generated tokenizer are included.

## Serve with mode 17

Keep all target GGUF shards together. Extract the target tokenizer:

```sh
python tools/strata_tokenizer.py \
  --gguf /models/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf \
  --out pack/deepseek41/tokenizer --check
cp examples/deepseek41-sycl-server.json strata-deepseek41-sycl.json
mkdir -p logs
```

Edit the model and draft paths in the local JSON. Select the intended Intel
GPU through the runtime's device selection before launch. Set CPU affinity
for the available host; the original 12-participant placement is not portable.
The example explicitly sets `STRATA_DS41_SYCL_DSPARK_TUNING=17` because the
preserved source still defaults to mode 8 on BMG G31 and mode 0 elsewhere.

```sh
python serve/server.py --engine strata --config strata-deepseek41-sycl.json \
  --host 127.0.0.1 --port 8082
```

Check the engine log for `DSpark tuning=17`. The example uses greedy sampling,
draft depth 5 and confidence 0.75. Keep prompt caching disabled for comparisons
with the historical workload. Model files, local configurations, tokenizer
exports and logs remain local.

This fork snapshot was requested for
`xiangyuT/Strata-xpu:feat/deepseek41-flash-dspark-sycl`. The inherited CUDA
publication instructions for another repository do not authorize further writes.
