#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_dequant.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
bool static_type_micro = false;
bool static_row_micro = false;

void check_case(int type, int64_t rows, int64_t cols,
                const std::vector<uint8_t>& gate, const std::vector<uint8_t>& up,
                bool benchmark) {
    auto& q = dpct::get_in_order_queue();
    const size_t count = (size_t) rows * cols * 2;
    const size_t guarded = count + 64;
    auto* dg = sycl::malloc_device<uint8_t>(gate.size(), q);
    auto* du = sycl::malloc_device<uint8_t>(up.size(), q);
    auto* output = sycl::malloc_device<uint16_t>(guarded, q);
    q.memcpy(dg, gate.data(), gate.size());
    q.memcpy(du, up.data(), up.size());
    q.memset(output, 0x5a, guarded * 2).wait();
    strata::kernels::iq_dequant_gu_f16(type, dg, du, rows, cols, output + 32, &q);
    std::vector<uint16_t> reference(guarded), actual(guarded);
    q.memcpy(reference.data(), output, guarded * 2).wait();
    for (int sg : {2, 4, 8}) for (bool typed : {false, true}) for (bool row_constant : {false, true}) {
        q.memset(output, 0x5a, guarded * 2).wait();
        strata::kernels::iq_dequant_gu_f16_tiled(type, dg, du, rows, cols, output + 32, &q, sg, typed, row_constant);
        q.memcpy(actual.data(), output, guarded * 2).wait();
        if (actual != reference || !std::all_of(actual.begin(), actual.begin() + 32, [](auto v) { return v == 0x5a5a; }) ||
            !std::all_of(actual.end() - 32, actual.end(), [](auto v) { return v == 0x5a5a; })) {
            std::cerr << "dequant mismatch/guard corruption type=" << type << " rows=" << rows << " cols=" << cols << " sg=" << sg << " static_type=" << typed << " static_row=" << row_constant << '\n';
            std::exit(1);
        }
    }
    std::cout << "{\"kind\":\"correctness\",\"type\":" << type << ",\"rows\":" << rows
              << ",\"cols\":" << cols << ",\"byte_exact\":true}" << std::endl;
    if (benchmark) {
        for (size_t i = 32; i < 32 + count; ++i) {
            if (!std::isfinite((float)sycl::bit_cast<sycl::half>(reference[i]))) {
                std::cerr << "nonfinite real model weight" << std::endl;
                std::exit(1);
            }
        }
        std::vector<uint16_t*> rotating;
        for (int i = 0; i < 8; ++i) rotating.push_back(sycl::malloc_device<uint16_t>(count, q));
        const std::vector<int> variants = static_row_micro ? std::vector<int>{9, 10} :
                                         static_type_micro ? std::vector<int>{8, 9} : std::vector<int>{1, 2, 4, 8};
        for (int variant : variants) {
            const int sg = variant >= 9 ? 8 : variant;
            auto call = [&](int i) {
                if (sg == 1) strata::kernels::iq_dequant_gu_f16(type, dg, du, rows, cols, rotating[i % 8], &q);
                else strata::kernels::iq_dequant_gu_f16_tiled(type, dg, du, rows, cols, rotating[i % 8], &q, sg, variant >= 9, variant == 10);
            };
            size_t calls = 0;
            auto warm = Clock::now();
            do {
                for (int i = 0; i < 256; ++i) call(i);
                q.wait_and_throw();
                calls += 256;
            } while (calls < 1000 || std::chrono::duration<double>(Clock::now() - warm).count() < 2.0);
            std::vector<double> device_times, wall_times;
            auto start = Clock::now();
            calls = 0;
            do {
                auto begin = q.ext_oneapi_submit_barrier();
                auto wall = Clock::now();
                for (int i = 0; i < 1024; ++i) call(i);
                auto end = q.ext_oneapi_submit_barrier();
                end.wait_and_throw();
                wall_times.push_back(std::chrono::duration<double, std::micro>(Clock::now() - wall).count() / 1024);
                device_times.push_back((end.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                        begin.get_profiling_info<sycl::info::event_profiling::command_end>()) / 1024000.0);
                calls += 1024;
            } while (calls < 2000 || std::chrono::duration<double>(Clock::now() - start).count() < 3.0);
            std::cout << "{\"kind\":\"micro\",\"type\":" << type << ",\"subgroups\":" << sg
                      << ",\"static_type\":" << (variant >= 9 ? "true" : "false")
                      << ",\"static_row\":" << (variant == 10 ? "true" : "false") << ",\"calls\":" << calls << ",\"device_us\":[";
            for (size_t i = 0; i < device_times.size(); ++i) std::cout << (i ? "," : "") << device_times[i];
            std::cout << "],\"wall_us\":[";
            for (size_t i = 0; i < wall_times.size(); ++i) std::cout << (i ? "," : "") << wall_times[i];
            std::cout << "]}" << std::endl;
        }
        for (auto* ptr : rotating) sycl::free(ptr, q);
    }
    sycl::free(dg, q); sycl::free(du, q); sycl::free(output, q);
}

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3 && !(argc == 4 && (std::string(argv[3]) == "--static-type" || std::string(argv[3]) == "--static-row"))) {
        std::cerr << "usage: iq_dequant_gu_bench [model.gguf native_experts.txt [--static-type|--static-row]]\n";
        return 2;
    }
    unsetenv("STRATA_SYCL_DEQUANT_GU_SG");
    unsetenv("STRATA_SYCL_DEQUANT_GU_STATIC_TYPE");
    unsetenv("STRATA_SYCL_DEQUANT_GU_STATIC_ROW");
    static_type_micro = argc == 4 && std::string(argv[3]) == "--static-type";
    static_row_micro = argc == 4 && std::string(argv[3]) == "--static-row";
    std::mt19937 rng(42);
    for (int type : {16, 17, 18, 20, 21, 22, 23, 29, 42, 11, 12, 13, 7, 6, 8}) {
        for (int rows : {1, 3, 7}) for (int cols : {256, 768, 2560}) {
            const size_t bytes = strata::kernels::iq_row_bytes(type, cols) * rows;
            std::vector<uint8_t> g(bytes), u(bytes);
            for (auto& value : g) value = rng() & 255;
            for (auto& value : u) value = rng() & 255;
            check_case(type, rows, cols, g, u, false);
        }
    }
    if (argc == 3 || static_type_micro || static_row_micro) {
        std::ifstream meta(argv[2]);
        std::string line;
        while (std::getline(meta, line)) {
            if (line.empty() || line[0] == '#') continue;
            int layer, gt, dt; int64_t base, blob, go, uo, down;
            std::istringstream row(line);
            row >> layer >> gt >> dt >> base >> blob >> go >> uo >> down;
            if (layer != 0 && layer != 1 && layer != 8) continue;
            const size_t bytes = strata::kernels::iq_row_bytes(gt, 2560) * 640;
            std::vector<uint8_t> g(bytes), u(bytes);
            std::ifstream weights(argv[1], std::ios::binary);
            weights.seekg(go); weights.read((char*)g.data(), bytes);
            weights.seekg(uo); weights.read((char*)u.data(), bytes);
            if (!weights) return 2;
            check_case(gt, 640, 2560, g, u, true);
        }
    }
}
