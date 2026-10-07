#define DPCT_PROFILING_ENABLED
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_tuning.hpp"
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

using namespace strata::kernels;
using Clock = std::chrono::steady_clock;
constexpr int H = 2560, FF = 640, G = 16;
struct Fixture {
    NativeExpertLayout layout;
    int T, layer, entries, groups;
    std::vector<void*> blobs;
    unsigned long long* pointers;
    int32_t *starts, *count, *dst, *tok;
    float *x, *gatebase, *upbase, *outbase;
    void *xq, *scratch;
};
template<class T> T* upload(sycl::queue& q, const std::vector<T>& data) {
    auto* p = sycl::malloc_device<T>(data.size(), q);
    if (!p) throw std::bad_alloc();
    q.memcpy(p, data.data(), data.size() * sizeof(T)).wait_and_throw(); return p;
}
void input(Fixture& f, sycl::queue& q, int pattern) {
    std::vector<float> x((size_t)f.T * H);
    for (size_t i = 0; i < x.size(); ++i) {
        const float v = float(int((i * 7919 + f.layer * 17) % 2001) - 1000) / 1000.f;
        x[i] = pattern == 1 ? 0.f : pattern == 2 ? v * 1e-20f : pattern == 3 ? v * 1000.f : pattern == 4 ? std::copysign(0.f, v) : v;
    }
    q.memcpy(f.x, x.data(), x.size() * 4);
    quantize_q8_1_rows(f.x, f.T, H, f.xq, &q); q.wait_and_throw();
}
void clear(Fixture& f, sycl::queue& q) {
    q.fill(f.gatebase, 13.75f, (size_t)f.entries * FF + 2 * G);
    q.fill(f.upbase, 13.75f, (size_t)f.entries * FF + 2 * G);
    q.fill(f.outbase, 13.75f, (size_t)f.entries * H + 2 * G);
    q.wait_and_throw();
}
void gu(Fixture& f, sycl::queue& q) {
    native_expert_gu_probe(f.layout, f.pointers, f.starts, f.count, f.tok, f.xq,
                           f.gatebase + G, f.upbase + G, f.entries, &q);
}
std::vector<float> snapshot(Fixture& f, sycl::queue& q, bool final) {
    const size_t size = final ? (size_t)f.entries * H + 2 * G : (size_t)f.entries * FF + 2 * G;
    std::vector<float> data(size * (final ? 1 : 2));
    q.memcpy(data.data(), final ? f.outbase : f.gatebase, size * 4);
    if (!final) q.memcpy(data.data() + size, f.upbase, size * 4);
    q.wait_and_throw();
    for (size_t base = 0; base < data.size(); base += size) for (size_t i = 0; i < size; ++i) {
        if (!std::isfinite(data[base + i])) throw std::runtime_error("nonfinite expert output");
        if ((i < G || i >= size - G) && data[base + i] != 13.75f) throw std::runtime_error("expert output guard changed");
    }
    return data;
}
void parity(Fixture& f, sycl::queue& q) {
    for (int pattern = 0; pattern < 5; ++pattern) {
        input(f, q, pattern); clear(f, q); native_expert_set_iq2xxs_reuse(false);
        gu(f, q); const auto baseline = snapshot(f, q, false);
        clear(f, q); native_expert_set_iq2xxs_reuse(true);
        dpct::experimental::command_graph_ptr graph = nullptr;
        dpct::experimental::begin_recording(&q); gu(f, q);
        dpct::experimental::end_recording(&q, &graph);
        auto executable = graph->finalize(); q.ext_oneapi_graph(executable); q.wait_and_throw();
        const auto candidate = snapshot(f, q, false); delete graph;
        if (std::memcmp(baseline.data(), candidate.data(), baseline.size() * 4)) throw std::runtime_error("GU byte mismatch");
        auto full = [&] { native_expert_grouped(f.layout, f.pointers, f.starts, f.count, f.dst, f.tok,
                                               f.entries, f.entries, f.xq, f.scratch, f.outbase + G, &q); };
        clear(f, q); native_expert_set_iq2xxs_reuse(false); full(); const auto old = snapshot(f, q, true);
        clear(f, q); native_expert_set_iq2xxs_reuse(true);
        graph = nullptr;
        dpct::experimental::begin_recording(&q); full();
        dpct::experimental::end_recording(&q, &graph);
        auto full_executable = graph->finalize(); q.ext_oneapi_graph(full_executable); q.wait_and_throw();
        const auto now = snapshot(f, q, true); delete graph;
        if (std::memcmp(old.data(), now.data(), old.size() * 4)) throw std::runtime_error("expert final byte mismatch");
    }
    // The native engine also launches a stride-four GU for an empty PCIe plan.
    // Its runtime group count must leave every output byte untouched.
    q.fill(f.count, 0, 1); q.wait_and_throw();
    auto empty_gu = [&] {
        native_expert_gu_probe(f.layout, f.pointers, f.starts, f.count, f.tok, f.xq,
                               f.gatebase + G, f.upbase + G, 4, &q);
    };
    clear(f, q); native_expert_set_iq2xxs_reuse(false); empty_gu();
    const auto empty_baseline = snapshot(f, q, false);
    clear(f, q); native_expert_set_iq2xxs_reuse(true); empty_gu();
    const auto empty_candidate = snapshot(f, q, false);
    if (std::memcmp(empty_baseline.data(), empty_candidate.data(), empty_baseline.size() * 4) ||
        std::any_of(empty_candidate.begin(), empty_candidate.end(), [](float value) { return value != 13.75f; }))
        throw std::runtime_error("empty GU wrote output");
    q.fill(f.count, f.groups, 1); q.wait_and_throw();
}
void measure(std::vector<Fixture*>& fixtures, sycl::queue& q, bool candidate, const char* arm) {
    q.wait_and_throw(); native_expert_set_iq2xxs_reuse(candidate);
    dpct::experimental::command_graph_ptr graph = nullptr;
    dpct::experimental::begin_recording(&q);
    for (auto* f : fixtures) gu(*f, q);
    dpct::experimental::end_recording(&q, &graph);
    auto executable = graph->finalize();
    const int per = fixtures.size();
    const auto warm = Clock::now(); int calls = 0;
    do { for (int i = 0; i < 32; ++i) { q.ext_oneapi_graph(executable); calls += per; } q.wait_and_throw(); }
    while (calls < 1000 || std::chrono::duration<double>(Clock::now() - warm).count() < 2);
    const int batch = std::max(per * 16, int(0.15 * calls / std::chrono::duration<double>(Clock::now() - warm).count() / per) * per);
    const auto start = Clock::now(); calls = 0; int chunk = 0;
    do {
        const auto before = Clock::now(); auto begin = q.ext_oneapi_submit_barrier();
        for (int i = 0; i < batch / per; ++i) q.ext_oneapi_graph(executable);
        auto end = q.ext_oneapi_submit_barrier(); end.wait_and_throw();
        const double wall = std::chrono::duration<double, std::micro>(Clock::now() - before).count() / batch;
        const double span = (end.get_profiling_info<sycl::info::event_profiling::command_start>() -
                             begin.get_profiling_info<sycl::info::event_profiling::command_end>()) / 1000.0 / batch;
        std::printf("{\"kind\":\"graph_chunk\",\"arm\":\"%s\",\"T\":%d,\"calls\":%d,\"chunk\":%d,\"wall_us\":%.6f,\"event_span_us\":%.6f}\n",
                    arm, fixtures.front()->T, batch, chunk++, wall, span);
        calls += batch;
    } while (calls < 2000 || std::chrono::duration<double>(Clock::now() - start).count() < 3);
    std::printf("{\"kind\":\"block_complete\",\"arm\":\"%s\",\"T\":%d,\"calls\":%d}\n", arm, fixtures.front()->T, calls);
    std::fflush(stdout); delete graph;
}
int main(int argc, char** argv) try {
    if (argc < 3 || argc > 4) throw std::runtime_error("usage: iq2xxs_reuse_bench <GGUF> <plans.txt> [--micro]");
    if (argc == 4 && std::string(argv[3]) != "--micro") throw std::runtime_error("unexpected argument");
    const bool micro = argc == 4 && std::string(argv[3]) == "--micro";
    auto& q = dpct::get_in_order_queue(); strata::GgufFile gguf(argv[1]);
    std::ifstream plans(argv[2]); if (!plans) throw std::runtime_error("plans unavailable");
    std::vector<Fixture> fixtures;
    while (true) {
        Fixture f{};
        if (!(plans >> f.T >> f.layer >> f.entries >> f.groups)) { if (plans.eof()) break; throw std::runtime_error("bad plan header"); }
        if (f.T < 1 || f.T > 8 || f.entries != f.T * 10 || f.groups < 1 || f.groups > f.entries) throw std::runtime_error("plan bounds");
        const std::string prefix = "blk." + std::to_string(f.layer) + ".ffn_";
        const auto* gate = gguf.find(prefix + "gate_exps.weight"), *up = gguf.find(prefix + "up_exps.weight"), *down = gguf.find(prefix + "down_exps.weight");
        if (!gate || !up || !down || gate->type != 16 || up->type != 16 || down->type != 42) throw std::runtime_error("real weight format mismatch");
        if (gate->shape != std::vector<uint64_t>{H, FF, 512} || up->shape != gate->shape ||
            down->shape != std::vector<uint64_t>{FF, H, 512}) throw std::runtime_error("real weight shape mismatch");
        f.layout = native_expert_layout(16, 42, H, FF);
        std::vector<unsigned long long> ptrs; std::vector<int32_t> starts{0}, dst, tok;
        for (int g = 0; g < f.groups; ++g) {
            int count, host, expert;
            if (!(plans >> count >> host >> expert) || count < 1 || count > f.T || expert < 0 || expert >= 512) throw std::runtime_error("group bounds");
            std::vector<uint8_t> blob(f.layout.bytes);
            std::memcpy(blob.data(), gguf.tensor_data(*gate) + expert * f.layout.up_off, f.layout.up_off);
            std::memcpy(blob.data() + f.layout.up_off, gguf.tensor_data(*up) + expert * f.layout.up_off, f.layout.up_off);
            std::memcpy(blob.data() + f.layout.down_off, gguf.tensor_data(*down) + expert * (f.layout.bytes - f.layout.down_off), f.layout.bytes - f.layout.down_off);
            void* memory = host ? sycl::malloc_host(blob.size(), q) : sycl::malloc_device(blob.size(), q);
            if (!memory) throw std::bad_alloc();
            if (host) std::memcpy(memory, blob.data(), blob.size()); else q.memcpy(memory, blob.data(), blob.size()).wait_and_throw();
            f.blobs.push_back(memory); ptrs.push_back((unsigned long long)memory);
            for (int i = 0; i < count; ++i) { int d,t; if (!(plans >> d >> t) || d < 0 || d >= f.entries || t < 0 || t >= f.T) throw std::runtime_error("mapping bounds"); dst.push_back(d); tok.push_back(t); }
            starts.push_back(dst.size());
        }
        if (dst.size() != (size_t)f.entries) throw std::runtime_error("entry total mismatch");
        auto sorted = dst; std::sort(sorted.begin(), sorted.end()); for (int i = 0; i < f.entries; ++i) if (sorted[i] != i) throw std::runtime_error("dst permutation");
        f.pointers = upload(q, ptrs); f.starts = upload(q, starts); f.dst = upload(q, dst); f.tok = upload(q, tok); f.count = upload(q, std::vector<int32_t>{f.groups});
        f.x = sycl::malloc_device<float>((size_t)f.T * H, q); f.xq = sycl::malloc_device((size_t)f.T * H / 32 * 36, q);
        f.gatebase = sycl::malloc_device<float>((size_t)f.entries * FF + 2 * G, q); f.upbase = sycl::malloc_device<float>((size_t)f.entries * FF + 2 * G, q);
        f.outbase = sycl::malloc_device<float>((size_t)f.entries * H + 2 * G, q); f.scratch = sycl::malloc_device(native_expert_scratch_bytes(f.entries, FF), q);
        fixtures.push_back(std::move(f));
    }
    if (fixtures.size() < 32) throw std::runtime_error("at least 32 grouping fixtures required");
    for (auto& f : fixtures) parity(f, q);
    std::printf("{\"kind\":\"correctness\",\"fixtures\":%zu,\"patterns\":5,\"GU_and_final\":\"byte-exact\",\"GU_and_final_candidate_graph_replay\":true,\"empty_groups_no_write\":true,\"guards\":true,\"finite\":true}\n", fixtures.size()); std::fflush(stdout);
    if (micro) for (int T : {1,2,3,4}) {
        std::vector<Fixture*> selected; for (auto& f : fixtures) if (f.T == T) { input(f,q,0); selected.push_back(&f); }
        if (selected.size()!=8) throw std::runtime_error("8 rotating fixtures required per T");
        measure(selected,q,false,"baseline-before"); measure(selected,q,true,"candidate"); measure(selected,q,false,"baseline-after");
    }
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr,"IQ2_XXS reuse: %s\n",e.what()); return 1; }
