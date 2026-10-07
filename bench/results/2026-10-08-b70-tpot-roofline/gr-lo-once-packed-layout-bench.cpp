// Byte-exact GR route comparison on the actual dense pack, then steady B/C/B.
#define DPCT_PROFILING_ENABLED
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr_tuning.hpp"
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::kernels;
using Clock = std::chrono::steady_clock;
constexpr int N = 2560, HC = 4, D = HC * N, LR = 320, TM = kFusedGrMaxT, O = LR + 2 * HC + N, GUARD = 16;
static bool graph_mode = false;
void select_route(bool candidate) {
    fused_gr_set_lo_once(candidate);
}
struct Tensor { int part, kind; uint64_t offset, bytes, codes, codebytes; int64_t n0, n1; };
struct Fixture {
    int layer;
    float *Rbase, *R, *norm, *bo, *inj, *xnbase, *xn, *outbase, *out;
    uint16_t *down, *up, *inject;
    std::vector<float> hR, hbo, hinj;
    FusedGrArgs args[TM];
};
template<class T> T* upload(sycl::queue& q, const std::vector<T>& h) {
    auto* p = sycl::malloc_device<T>(h.size(), q);
    if (!p) throw std::bad_alloc();
    q.memcpy(p, h.data(), h.size() * sizeof(T)).wait_and_throw(); return p;
}
template<class T> std::vector<T> read_tensor(std::ifstream& file, const Tensor& t, int kind, size_t size, int64_t n0, int64_t n1) {
    if (t.part != 0 || t.kind != kind || t.bytes != size * sizeof(T) || t.codes != t.offset || t.codebytes != t.bytes || t.n0 != n0 || t.n1 != n1)
        throw std::runtime_error("unexpected dense-pack tensor contract");
    std::vector<T> values(size); file.seekg(t.offset); file.read((char*)values.data(), t.bytes);
    if (!file) throw std::runtime_error("short tensor read"); return values;
}
void reset(Fixture& f, sycl::queue& q, int T, bool apply, int pattern) {
    for (int t = 0; t < TM; ++t) for (int i = 0; i < D; ++i) {
        const float x = float((i * 7919 + t * 97 + f.layer * 11) % 2001 - 1000) / 1000.f;
        f.hR[GUARD + t * D + i] = pattern == 5 ? std::copysign(0.f, x) : pattern == 1 ? 0.f : pattern == 2 ? x * 1e-20f : pattern == 3 ? x * 1000.f : x * float(1 << (2 * (i / N)));
    }
    std::fill(f.hR.begin(), f.hR.begin() + GUARD, 13.75f);
    std::fill(f.hR.end() - GUARD, f.hR.end(), 13.75f);
    for (size_t i = 0; i < f.hbo.size(); ++i) f.hbo[i] = pattern == 4 ? 0.f : float(int(i % 31) - 15) * 0.01f;
    for (size_t i = 0; i < f.hinj.size(); ++i) f.hinj[i] = float(int(i % 9) - 4);
    q.memcpy(f.Rbase, f.hR.data(), f.hR.size() * 4);
    q.memcpy(f.bo, f.hbo.data(), f.hbo.size() * 4);
    q.memcpy(f.inj, f.hinj.data(), f.hinj.size() * 4);
    q.fill(f.outbase, 13.75f, TM * O + 2 * GUARD);
    q.fill(f.xnbase, 13.75f, TM * D + 2 * GUARD);
    for (int t = 0; t < TM; ++t) f.args[t].apply = apply;
    q.wait_and_throw();
}
std::vector<float> snapshot(Fixture& f, sycl::queue& q) {
    std::vector<float> h(TM * D + 2 * GUARD + TM * O + 2 * GUARD + TM * D + 2 * GUARD);
    q.memcpy(h.data(), f.Rbase, (TM * D + 2 * GUARD) * 4);
    q.memcpy(h.data() + TM * D + 2 * GUARD, f.outbase, (TM * O + 2 * GUARD) * 4);
    q.memcpy(h.data() + TM * D + TM * O + 4 * GUARD, f.xnbase, (TM * D + 2 * GUARD) * 4);
    q.wait_and_throw(); return h;
}
void check_guards(const std::vector<float>& h, int T, const std::vector<float>& initial_R) {
    size_t start = 0;
    for (int width : {D, O, D}) {
        for (size_t i = 0; i < TM * width + 2 * GUARD; ++i) {
            const float v = h[start + i];
            if (!std::isfinite(v)) throw std::runtime_error("nonfinite output");
            if (i < GUARD || i >= GUARD + T * width) {
                const float expected = start == 0 ? initial_R[i] : 13.75f;
                if (std::memcmp(&v, &expected, sizeof(float)) != 0)
                    throw std::runtime_error("guard/inactive row changed");
            }
        }
        start += TM * width + 2 * GUARD;
    }
}
void micro(std::vector<Fixture>& fixtures, sycl::queue& q, int T, bool candidate, const char* arm) {
    q.wait_and_throw(); select_route(candidate);
    for (auto& f : fixtures) reset(f, q, T, true, 4); // zero pending output keeps repeated activations fixed
    size_t cursor = 0;
    dpct::experimental::command_graph_ptr graph = nullptr;
    dpct::experimental::command_graph_exec_ptr executable = nullptr;
    if (graph_mode) {
        dpct::experimental::begin_recording(&q);
        for (auto& f : fixtures) fused_gr_read_multi(f.args, T, f.xn, &q);
        dpct::experimental::end_recording(&q, &graph);
        executable = new sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>(graph->finalize());
    }
    auto launch = [&] {
        if (graph_mode) q.ext_oneapi_graph(*executable);
        else { auto& f = fixtures[cursor++ % fixtures.size()]; fused_gr_read_multi(f.args, T, f.xn, &q); }
    };
    const int calls_per_launch = graph_mode ? int(fixtures.size()) : 1;
    const auto warm = Clock::now(); int warm_calls = 0;
    do { for (int i = 0; i < 64; ++i) { launch(); warm_calls += calls_per_launch; } q.wait_and_throw(); }
    while (warm_calls < 1000 || std::chrono::duration<double>(Clock::now() - warm).count() < 2);
    const double warm_seconds = std::chrono::duration<double>(Clock::now() - warm).count();
    const int batch = std::max(128, int(0.15 * warm_calls / warm_seconds / 8) * 8);
    const auto timed = Clock::now(); int calls = 0, chunks = 0;
    double total_wall = 0, total_event = 0;
    do {
        const auto before = Clock::now();
        auto first = q.ext_oneapi_submit_barrier();
        for (int i = 0; i < batch / calls_per_launch; ++i) launch();
        auto last = q.ext_oneapi_submit_barrier(); last.wait_and_throw();
        const double wall = std::chrono::duration<double, std::micro>(Clock::now() - before).count();
        const double event = (last.get_profiling_info<sycl::info::event_profiling::command_start>() -
                              first.get_profiling_info<sycl::info::event_profiling::command_end>()) / 1000.0;
        std::printf("{\"kind\":\"micro_chunk\",\"arm\":\"%s\",\"T\":%d,\"chunk\":%d,\"calls\":%d,\"wall_us_per_call\":%.6f,\"event_span_us_per_call\":%.6f}\n",
                    arm, T, chunks++, batch, wall / batch, event / batch);
        total_wall += wall; total_event += event; calls += batch;
    } while (calls < 2000 || std::chrono::duration<double>(Clock::now() - timed).count() < 3);
    std::printf("{\"kind\":\"micro_summary\",\"arm\":\"%s\",\"T\":%d,\"calls\":%d,\"warm_calls\":%d,\"wall_us_per_call\":%.6f,\"event_span_us_per_call\":%.6f}\n",
                arm, T, calls, warm_calls, total_wall / calls, total_event / calls);
    std::fflush(stdout);
    delete executable; delete graph;
}
int main(int argc, char** argv) try {
    if (argc < 2) throw std::runtime_error("usage: gr_lo_once_bench <pack-directory> [--micro] [--graph]");
    bool run_micro = false;
    for (int i = 2; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--micro") run_micro = true;
        else if (option == "--graph") graph_mode = true;
        else throw std::runtime_error("unexpected arguments");
    }
    sycl::queue& q = dpct::get_in_order_queue();
    std::map<std::string, Tensor> index;
    std::ifstream table(std::string(argv[1]) + "/index.txt"); std::string line;
    while (std::getline(table, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream input(line); std::string name; Tensor t;
        if (!(input >> name >> t.part >> t.kind >> t.offset >> t.bytes >> t.codes >> t.codebytes >> t.n0 >> t.n1))
            throw std::runtime_error("malformed pack index");
        index.emplace(name, t);
    }
    std::ifstream weights(std::string(argv[1]) + "/dense.bin", std::ios::binary);
    if (!table.eof() || !weights) throw std::runtime_error("pack unavailable");
    std::vector<Fixture> fixtures;
    for (int layer : {0, 1, 2, 3, 12, 23, 36, 47}) {
        const std::string prefix = "blk." + std::to_string(layer) + ".hc_attn_";
        Fixture f{}; f.layer = layer;
        f.down = upload(q, read_tensor<uint16_t>(weights, index.at(prefix + "down.weight"), 4, LR * D, D, LR));
        f.up = upload(q, read_tensor<uint16_t>(weights, index.at(prefix + "up.weight"), 4, D * LR, LR, D));
        f.inject = upload(q, read_tensor<uint16_t>(weights, index.at(prefix + "inject.weight"), 4, HC * D, D, HC));
        f.norm = upload(q, read_tensor<float>(weights, index.at(prefix + "norm.weight"), 2, D, D, 0));
        f.hR.resize(TM * D + 2 * GUARD); f.hbo.resize(TM * N); f.hinj.resize(TM * HC);
        f.Rbase = sycl::malloc_device<float>(f.hR.size(), q); f.R = f.Rbase + GUARD;
        f.bo = sycl::malloc_device<float>(f.hbo.size(), q); f.inj = sycl::malloc_device<float>(f.hinj.size(), q);
        f.xnbase = sycl::malloc_device<float>(TM * D + 2 * GUARD, q); f.xn = f.xnbase + GUARD;
        f.outbase = sycl::malloc_device<float>(TM * O + 2 * GUARD, q); f.out = f.outbase + GUARD;
        for (int t = 0; t < TM; ++t) {
            auto& a = f.args[t]; a.R = f.R + t * D; a.R_out = f.R + t * D;
            a.w_down = f.down; a.w_up = f.up; a.w_inject = f.inject; a.w_norm = f.norm;
            a.bo_prev = f.bo + t * N; a.inj_prev = f.inj + t * HC;
            a.lo = f.out + t * O; a.rs = a.lo + LR; a.inject_out = a.rs + HC; a.mixed = a.inject_out + HC;
        }
        fixtures.push_back(std::move(f));
    }
    int cases = 0;
    for (auto& f : fixtures) for (int T = 1; T <= TM; ++T) for (bool apply : {false, true}) for (int pattern : {0, 1, 2, 3, 5}) {
        reset(f, q, T, apply, pattern); select_route(false);
        fused_gr_read_multi(f.args, T, f.xn, &q); auto baseline = snapshot(f, q);
        reset(f, q, T, apply, pattern); select_route(true);
        if (graph_mode) {
            dpct::experimental::command_graph_ptr graph = nullptr;
            dpct::experimental::begin_recording(&q);
            fused_gr_read_multi(f.args, T, f.xn, &q);
            dpct::experimental::end_recording(&q, &graph);
            auto executable = graph->finalize(); q.ext_oneapi_graph(executable);
            q.wait_and_throw(); delete graph;
        } else fused_gr_read_multi(f.args, T, f.xn, &q);
        auto candidate = snapshot(f, q);
        check_guards(baseline, T, f.hR); check_guards(candidate, T, f.hR);
        if (std::memcmp(baseline.data(), candidate.data(), baseline.size() * 4) != 0)
            throw std::runtime_error("GR baseline/candidate byte mismatch at layer=" + std::to_string(f.layer) + " T=" + std::to_string(T));
        ++cases;
    }
    std::printf("{\"kind\":\"correctness\",\"variant\":\"%s\",\"mode\":\"%s\",\"cases\":%d,\"real_weight_sets\":8,\"T_domain\":\"1..8\",\"pending_write\":\"on/off\",\"patterns\":5,\"outputs\":\"R,lo,rs,inject,mixed,xn\",\"result\":\"byte-exact,finite,guards passed\"}\n", "lo-once", graph_mode ? "native-graph" : "direct", cases);
    std::fflush(stdout);
    if (run_micro) for (int T : {1, 2, 3, 4, 6, 8}) {
        micro(fixtures, q, T, false, "baseline-before");
        micro(fixtures, q, T, true, "candidate");
        micro(fixtures, q, T, false, "baseline-after");
    }
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "GR Lo once: %s\n", error.what()); return 1; }
