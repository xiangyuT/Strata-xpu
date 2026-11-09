#pragma once
#include <cstdint>

namespace strata::deepseek41::cuda {
// SYCL prefill score grid with four-head value reuse and one row list/token.
// Scratch bytes: align256(heads*tokens*(window+max_selected)*4)
//                + tokens*(window+max_selected)*8.
// False retains caller dispatch for small batches or unsupported geometry.
// Original score-grid ABI and CUDA/HIP headers are unchanged.
bool attention_positions_prefill(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* stream);
// Internal two-head QK controls; original entry points retain one-head QK.
bool attention_positions_prefill(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* stream,bool paired_qk);
bool attention_positions_score_grid_paired(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* stream,bool pair_heads);

}
