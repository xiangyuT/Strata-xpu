// Native Engram rows used by the DeepSeek V4.1 artifact family.
// Unsloth revision e5679175f75caf12694b664630f2640d64a8f8a0 stores type 67
// as [UE8M0 scale, 32 E4M3FN values] blocks. Checked byte-for-byte against
// the original checkpoint's separate F8_E4M3 weight / F8_E8M0 scale tensors.
// This producer-specific id is deliberately NOT registered as a global ggml type.
#pragma once

#include "strata/artifact/dequant.hpp"
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::deepseek41 {
constexpr uint32_t fp8_engram_type = 67;
// New Unsloth/llama.cpp GGUF schema; keep interpretation local to Engram.
constexpr uint32_t mxfp8_engram_type = 43;

inline uint64_t engram_row_bytes(uint32_t type, uint64_t width) noexcept {
    const uint64_t block = (type == fp8_engram_type || type == mxfp8_engram_type || type == 20) ? 32 : type == 12 ? 256 : 0;
    const uint64_t bytes = (type == fp8_engram_type || type == mxfp8_engram_type) ? 33 : type == 20 ? 18 : 144;
    if (!block || !width || width % block || width / block > UINT64_MAX / bytes) return 0;
    return width / block * bytes;
}

inline float e4m3fn_value(uint8_t code) noexcept {
    const uint32_t sign = uint32_t(code & 0x80) << 24;
    const uint32_t exponent = (code >> 3) & 15, mantissa = code & 7;
    if (exponent == 15 && mantissa == 7)
        return std::bit_cast<float>(sign | 0x7fc00000u);
    if (!exponent) {
        const float subnormal = float(mantissa) * (1.0f / 512.0f);
        return std::bit_cast<float>(std::bit_cast<uint32_t>(subnormal) | sign);
    }
    return std::bit_cast<float>(sign | ((exponent + 120) << 23) | (mantissa << 20));
}

inline float e8m0_value(uint8_t code) noexcept {
    // E8M0 has no zero/infinity: code 0 is 2^-127 and code 255 is NaN.
    if (code == 255) return std::numeric_limits<float>::quiet_NaN();
    return std::bit_cast<float>(code ? uint32_t(code) << 23 : 0x00400000u);
}

inline void dequantize_engram_row(uint32_t type, const uint8_t* src, uint64_t width, float* dst) {
    if (!src || !dst || !engram_row_bytes(type,width) || width > SIZE_MAX / sizeof(float))
        throw std::invalid_argument("unsupported Engram row layout");
    if (type == 12) {
        for (uint64_t i = 0; i < width; i += 256) dequantize_q4_K(src+(i/256)*144,dst+i);
    } else if (type == 20) {
        for (uint64_t i = 0; i < width; i += 32) dequantize_iq4_nl(src+(i/32)*18,dst+i);
    } else {
        for (uint64_t i = 0; i < width; i += 32) {
            const auto* block = src+(i/32)*33;
            const float scale = e8m0_value(block[0]);
            for (int j = 0; j < 32; ++j) dst[i+j] = e4m3fn_value(block[j+1])*scale;
        }
    }
}
} // namespace strata::deepseek41
