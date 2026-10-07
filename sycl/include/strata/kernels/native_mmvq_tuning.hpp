#pragma once
namespace strata::kernels {
// Diagnostic only. 0 is the existing scalar lookup, 1 an exact bit-select
// lookup, 2 a non-equivalent no-codebook cost control, 3 a rejected ESIMD
// prototype, 4 an exact subgroup-register lookup, 5 bit-select with add/xor
// masks. No production dispatch flag selects these alternatives. Modes 1/2/4/5
// retain the native float kernel body.
void native_iq4_xs_decode_probe(int mode, const void* w, const void* x_q8_1,
                               float* y, int K, int N, int T, void* stream);
void native_iq4_xs_esimd_probe(const void* w, const void* x_q8_1, float* y,
                              int K, int N, int T, void* stream);
void native_iq4_esimd_lut_probe(unsigned int* output, void* stream);
void native_iq4_group_lut_probe(unsigned int* output, void* stream);
}
