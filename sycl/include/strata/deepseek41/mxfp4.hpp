#pragma once
#include <cstdint>

namespace strata::deepseek41::cuda {
inline constexpr int max_prefill_batch=128;
// SYCL grouped expert chain: read each activation from its original token row
// and write each result to its original route slot. The host destinations are
// copied into kernel arguments before return; no device gather/scatter copies.
// One expert appears at most once per token, up to max_prefill_batch rows.
void mxfp4_expert_group(const void* gate,const void* up,const void* down,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,
    const int32_t* destinations,int count,float limit,void* stream);
// Mixed IQ expert formats retain original bytes, FP32 activations and the
// same lane-wise accumulation/reduction contract as MXFP4.
void expert_group(int gate_type,int up_type,int down_type,
    const void* gate,const void* up,const void* down,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,
    const int32_t* destinations,int count,float limit,void* stream);
// Explicit row tiles for the measured BMG prefill path. The original entry
// keeps its dispatch; supported tiles are4/8/16, with format fallbacks intact.
void expert_group_tiled(int gate_type,int up_type,int down_type,
    const void* gate,const void* up,const void* down,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,
    const int32_t* destinations,int count,float limit,void* stream,int gate_tile,int down_tile);
// Internal metadata control. Old overloads retain original decoder kernels.
void expert_group_tiled(int gate_type,int up_type,int down_type,
    const void* gate,const void* up,const void* down,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,
    const int32_t* destinations,int count,float limit,void* stream,int gate_tile,int down_tile,bool packed_metadata);
// IQ1 GU lookahead keeps narrow metadata reads and the original FMA order.
void expert_group_tiled(int gate_type,int up_type,int down_type,
    const void* gate,const void* up,const void* down,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,
    const int32_t* destinations,int count,float limit,void* stream,int gate_tile,int down_tile,bool packed_metadata,bool iq1_gu_narrow);

// Internal small-row control; existing overloads retain the original dispatch.
void expert_group_tiled(int gate_type,int up_type,int down_type,
    const void* gate,const void* up,const void* down,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,
    const int32_t* destinations,int count,float limit,void* stream,int gate_tile,int down_tile,
    bool packed_metadata,bool iq1_gu_narrow,bool small_rows);

// Independent single-row experts share two launches and disjoint scratch rows.
inline constexpr int max_expert_batch=4;
struct ExpertBatchJob { const uint8_t *gate,*up,*down; int entry,count; };
void expert_group_batch(int gate_type,int down_type,const ExpertBatchJob* jobs,int count,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,float limit,void* stream);
// Internal one-row contract control; the old overload keeps original dispatch.
void expert_group_batch(int gate_type,int down_type,const ExpertBatchJob* jobs,int count,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,float limit,void* stream,bool one_row);

void expert_group_batch(int gate_type,int down_type,const ExpertBatchJob* jobs,int count,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,float limit,void* stream,bool one_row,bool narrow_down);
} // namespace strata::deepseek41::cuda
