// Same-runtime diagnostic roofs for decode. No model or performance candidate.
// Random initialized buffers avoid zero-page/compression bandwidth artifacts.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

using Clock = std::chrono::steady_clock;
using U4 = sycl::vec<uint32_t, 4>;
template<int Mode> class DecodeMemoryRoof;
class DecodeFp32Roof;
class DecodeDot4Roof;
static int64_t raw_ns() {
#ifdef __linux__
    timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return int64_t(ts.tv_sec) * 1000000000 + ts.tv_nsec;
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
#endif
}

template<class F>
void measure(const char* name, double work, const char* unit, F launch) {
    const auto warm_start = Clock::now();
    int warm_calls = 0;
    do { launch().wait_and_throw(); ++warm_calls; }
    while (warm_calls < 10 || std::chrono::duration<double>(Clock::now() - warm_start).count() < 2);
    std::array<double, 7> rates{}, durations{};
    std::array<int, 7> calls{};
    const int64_t timed_start = raw_ns();
    for (int sample = 0; sample < 7; ++sample) {
        const auto begin = Clock::now();
        double ns = 0;
        do {
            auto event = launch();
            event.wait_and_throw();
            ns += event.template get_profiling_info<sycl::info::event_profiling::command_end>() -
                  event.template get_profiling_info<sycl::info::event_profiling::command_start>();
            ++calls[sample];
        } while (calls[sample] < 10 || std::chrono::duration<double>(Clock::now() - begin).count() < 0.5);
        durations[sample] = ns / calls[sample];
        rates[sample] = work / durations[sample]; // bytes/ns = GB/s; operations/ns = GOP/s
    }
    auto ordered = rates;
    std::sort(ordered.begin(), ordered.end());
    std::printf("{\"kind\":\"roof\",\"name\":\"%s\",\"unit\":\"%s\",\"work_per_call\":%.0f,"
                "\"warm_calls\":%d,\"samples\":[", name, unit, work, warm_calls);
    for (int i = 0; i < 7; ++i)
        std::printf("%s{\"calls\":%d,\"event_ns_per_call\":%.3f,\"rate\":%.6f}", i ? "," : "", calls[i], durations[i], rates[i]);
    std::printf("],\"peak\":%.6f,\"median\":%.6f,\"timed_start_raw_ns\":%lld,\"timed_end_raw_ns\":%lld}\n",
                ordered.back(), ordered[3], (long long)timed_start, (long long)raw_ns());
    std::fflush(stdout);
}

template<int Mode>
sycl::event memory(sycl::queue& q, const uint32_t* input, uint32_t* output, size_t elements) {
    // Copy: one 16-byte read and write per item. Read: 256-byte read + 4-byte sink.
    constexpr size_t words = Mode == 0 ? 4 : 64;
    const size_t items = elements / words;
    return q.parallel_for<DecodeMemoryRoof<Mode>>(sycl::nd_range<1>(items, 256),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const size_t id = item.get_global_linear_id();
            if constexpr (Mode == 0) {
                U4 value; value.load(id, input); value.store(id, output);
            } else {
                U4 sum(0);
                for (int j = 0; j < 16; ++j) { U4 value; value.load(id * 16 + j, input); sum ^= value; }
                output[id] = sum[0] ^ sum[1] ^ sum[2] ^ sum[3];
            }
        });
}

int main(int argc, char** argv) try {
    sycl::queue& q = dpct::get_in_order_queue();
    const auto device = q.get_device();
    if (device.get_info<sycl::info::device::vendor_id>() != 0x8086) throw std::runtime_error("Intel GPU required");
    const auto name = device.get_info<sycl::info::device::name>();
    std::printf("{\"kind\":\"environment\",\"device\":\"%s\",\"timing\":\"SYCL event execution time\","
                "\"memory_bytes\":268435456,\"random_seed\":42,\"compute_note\":\"FP32 SIMD FMA and INT8 SIMD dp4a; XMX TOPS are not the roof of these decode paths\"}\n", name.c_str());
    std::fflush(stdout);
    constexpr size_t elements = (256ull << 20) / 4;
    std::vector<uint32_t> random(elements);
    uint32_t seed = 42;
    for (auto& x : random) { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; x = seed; }
    auto* input = sycl::malloc_device<uint32_t>(elements, q);
    auto* output = sycl::malloc_device<uint32_t>(elements, q);
    auto* host = sycl::malloc_host<uint32_t>(elements, q);
    if (!input || !output || !host) throw std::bad_alloc();
    std::copy(random.begin(), random.end(), host);
    std::fprintf(stderr, "stage: initialized random buffers; upload\n");
    q.memcpy(input, random.data(), elements * 4).wait_and_throw();
    std::fprintf(stderr, "stage: uploaded; device copy correctness\n");
    memory<0>(q, input, output, elements).wait_and_throw();
    std::vector<uint32_t> copied(elements);
    q.memcpy(copied.data(), output, elements * 4).wait_and_throw();
    if (copied != random) throw std::runtime_error("device copy mismatch");
    std::fprintf(stderr, "stage: correctness passed; roofs\n");
    measure("device_copy", elements * 8.0, "GB/s", [&] { return memory<0>(q, input, output, elements); });
    measure("device_read", elements * 4.0 + elements / 64.0 * 4.0, "GB/s", [&] { return memory<1>(q, input, output, elements); });
    measure("mapped_host_read", elements * 4.0, "GB/s", [&] { return memory<2>(q, host, output, elements); });
    std::vector<uint32_t> sinks(elements / 64);
    q.memcpy(sinks.data(), output, sinks.size() * 4).wait_and_throw();
    for (size_t i = 0; i < sinks.size(); ++i) {
        uint32_t expected = 0;
        for (int j = 0; j < 64; ++j) expected ^= random[i * 64 + j];
        if (sinks[i] != expected) throw std::runtime_error("mapped read checksum mismatch");
    }
    if (argc > 1 && std::string(argv[1]) == "--memory-only") {
        std::printf("{\"kind\":\"correctness\",\"device_copy\":\"byte-exact\",\"mapped_read\":\"all checksums exact\"}\n");
        sycl::free(input, q); sycl::free(output, q); sycl::free(host, q);
        return 0;
    }
    constexpr int threads = 1024 * 128, iterations = 4096, accumulators = 8;
    auto* foutput = reinterpret_cast<float*>(output);
    auto fp32 = [&] {
        return q.parallel_for<DecodeFp32Roof>(sycl::nd_range<1>(threads, 128),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                const int id = item.get_global_linear_id();
                const float seed_value = float(input[id] & 65535) / 65536.f;
                float a[accumulators];
                for (int j = 0; j < accumulators; ++j) a[j] = seed_value + j * 0.01f;
                for (int k = 0; k < iterations; ++k) {
#pragma unroll
                    for (int j = 0; j < accumulators; ++j) a[j] = sycl::fma(a[j], 0.999f + j * 0.00001f, 0.001f * (j + 1));
                }
                float sum = 0; for (auto x : a) sum += x;
                foutput[id] = sum;
            });
    };
    measure("fp32_simd_fma", double(threads) * iterations * accumulators * 2, "GOP/s", fp32);
    auto dot4 = [&] {
        return q.parallel_for<DecodeDot4Roof>(sycl::nd_range<1>(threads, 128),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                const int id = item.get_global_linear_id();
                const int a = input[id] & 0x0f0f0f0f, b = input[id + threads] & 0x0f0f0f0f;
                int c[accumulators]; for (int j = 0; j < accumulators; ++j) c[j] = j;
                for (int k = 0; k < iterations; ++k) {
#pragma unroll
                    for (int j = 0; j < accumulators; ++j) c[j] = dpct::dp4a(a, b, c[j]);
                }
                int sum = 0; for (auto x : c) sum += x;
                output[id] = sum;
            });
    };
    measure("int8_simd_dot4", double(threads) * iterations * accumulators * 8, "GOP/s", dot4);
    std::printf("{\"kind\":\"correctness\",\"device_copy\":\"byte-exact\",\"mapped_read\":\"all checksums exact\"}\n");
    sycl::free(input, q); sycl::free(output, q); sycl::free(host, q);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "decode roofs: %s\n", e.what()); return 1; }
