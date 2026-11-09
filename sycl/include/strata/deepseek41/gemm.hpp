#pragma once

namespace strata::deepseek41::cuda {
// Q6_K weights and Q8_1 activations. Sixteen-column reuse with native T8
// accumulation; <=8 columns retain caller dispatch, tails retain native MMVQ.
bool q6_k_gemm(const void* weights,const void* q8_1,float* output,
              int input,int rows,int tokens,void* stream);

// Q8_0 weights and existing Q8_1 activations. INT8 XMX with the same dot16
// scale/accumulation and SG32 reduction as native_mmvq's default Wide32Q8.
// Caller retains its existing dispatch when the function returns false.
bool q8_0_gemm(const void* weights,const void* q8_1,float* output,
              int input,int rows,int tokens,void* stream);
}
