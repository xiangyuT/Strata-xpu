#pragma once
#include "strata/kernels/iq_kernels.hpp"
namespace strata::kernels {
// Bench only: change with no queue work pending. Production reads the opt-in
// STRATA_SYCL_EXPERT_MULTI16 flag before graph capture.
void native_expert_set_iq2xxs_reuse(bool enabled);
// Exact production GU launch boundary, isolated from SwiGLU/down for graph micro.
void native_expert_gu_probe(const NativeExpertLayout& layout, const unsigned long long* pointers,
                            const int32_t* starts, const int32_t* groups, const int32_t* tokens,
                            const void* x_q8_1, float* gate, float* up, int64_t capacity, void* stream);
// Diagnostic only, original LN8 down boundary; production dispatch unchanged.
void native_expert_down_q2_probe(bool candidate, const NativeExpertLayout& layout,
                               const unsigned long long* pointers, const int32_t* starts,
                               const int32_t* groups, const int32_t* destinations,
                               const void* h_q8_1, float* output, int64_t capacity, void* stream);
}
