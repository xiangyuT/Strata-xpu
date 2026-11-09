#pragma once
// Original GGML codebooks and block layouts; shared FP32 decoding for the
// DeepSeek GPU and CPU expert paths. No activation requantization is applied.
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/sycl.hpp>
#endif
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

namespace strata::deepseek41::iq {
inline uint16_t u16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1])<<8); }
inline uint32_t u32(const uint8_t* p) { return uint32_t(u16(p)) | (uint32_t(u16(p+2))<<16); }
inline float half(uint16_t h) {
#ifdef __SYCL_DEVICE_ONLY__
    // GPU hardware converts every finite half exactly. Keep nonfinite payloads
    // on the portable path so their bit representation is unchanged too.
    if ((h&0x7c00)!=0x7c00) return float(sycl::bit_cast<sycl::half>(h));
#endif
    uint32_t bits=uint32_t(h&0x8000)<<16, e=(h>>10)&31, m=h&1023;
    if (!e) {
        if (m) { e=113; while (!(m&1024)) { m<<=1; --e; } bits|=(e<<23)|((m&1023)<<13); }
    } else bits|=((e==31?255:e+112)<<23)|(m<<13);
    return std::bit_cast<float>(bits);
}
inline bool supported(int type) { return type==16 || type==17 || type==19 || type==22 || type==29 || type==39; }
inline size_t row_bytes(int type,int width) {
    if (width<1 || !supported(type) || width%(type==39?32:256))
        throw std::invalid_argument("unsupported DeepSeek expert format/row");
    return type==39 ? size_t(width/32)*17 : size_t(width/256)*(type==16?66:type==17?74:type==19?50:type==29?56:82);
}
template<int Type> inline constexpr int block_bytes=Type==16?66:Type==17?74:Type==19?50:Type==29?56:82;
inline uint8_t iq2_signs(uint32_t value) {
    value&=127;
    // The eighth sign is the parity of the seven stored sign bits.
    return uint8_t(value^(uint32_t(std::popcount(value)&1)<<7));
}
inline uint8_t codebook_byte(const uint64_t* table,int index,int lane) {
    // The existing SYCL IQ dequantizers also read each codebook's byte view.
    // Avoid a 64-bit load and lane-variable extraction for every scalar weight.
    if constexpr(std::endian::native==std::endian::little)
        return reinterpret_cast<const uint8_t*>(table)[size_t(index)*8+lane];
    else return uint8_t(table[index]>>(8*lane));
}
template<int Type> inline float block_scale(const uint8_t* b) {
    static_assert(Type==16 || Type==17 || Type==19 || Type==22 || Type==29);
    if constexpr(Type==29) {
        // IQ1_M packs the global FP16 scale in the four scale words' high nibbles.
        const uint16_t h=(u16(b+48)>>12) | ((u16(b+50)>>8)&0xf0) |
            ((u16(b+52)>>4)&0xf00) | (u16(b+54)&0xf000);
        return half(h);
    } else return half(u16(b));
}
// at is a channel within one original 256-value block. The caller can reuse d
// without changing the per-weight rounding or the dot's lane/FMA order.
template<int Type> inline float block_weight(const uint8_t* b,int at,float d) {
    static_assert(Type==16 || Type==17 || Type==19 || Type==22 || Type==29);
    const int ib=at/32, l=(at%32)/8, j=at%8;
    if constexpr(Type==19) {
        const uint16_t h=u16(b+34+ib*2);
        const int index=b[2+ib*4+l] | (((h>>(3*l))&7)<<8);
        const int grid=int(int8_t(codebook_byte(iq1s_grid,index,j)));
        const float dl=d*float(2*((h>>12)&7)+1);
        return dl*(float(grid)+((h&0x8000)?-0.125f:0.125f));
    } else if constexpr(Type==29) {
        const int qh=(b[32+ib*2+l/2]>>(4*(l%2)))&15;
        const int index=b[ib*4+l] | ((qh&7)<<8);
        const int grid=int(int8_t(codebook_byte(iq1s_grid,index,j)));
        const int scale=(u16(b+48+(ib/2)*2)>>(6*(ib%2)+3*(l/2)))&7;
        const float dl=d*float(2*scale+1);
        return dl*(float(grid)+((qh&8)?-0.125f:0.125f));
    } else {
        int index, signs, scale;
        const uint64_t* grid;
        if constexpr(Type==16) {
            const uint32_t meta=u32(b+2+ib*8+4);
            index=b[2+ib*8+l]; signs=iq2_signs(meta>>(7*l)); scale=meta>>28;
            grid=iq2xxs_grid;
        } else if constexpr(Type==17) {
            const uint16_t q=u16(b+2+(ib*4+l)*2);
            index=q&511; signs=iq2_signs(q>>9); scale=(b[66+ib]>>(4*(l/2)))&15;
            grid=iq2xs_grid;
        } else {
            index=b[2+ib*4+l] | ((int(b[66+ib])<<(8-2*l))&768);
            signs=b[34+ib*4+l]; scale=(b[74+ib]>>(4*(l/2)))&15;
            grid=iq2s_grid;
        }
        const float db=d*(0.5f+float(scale))*0.25f;
        const float magnitude=db*float(codebook_byte(grid,index,j));
        return (signs&(1<<j)) ? -magnitude : magnitude;
    }
}
// Scalar access remains useful for parity checks and general callers. Explicit
// byte reads support every original GGUF block alignment.
template<int Type> inline float weight(const uint8_t* row,int i) {
    const uint8_t* b=row+size_t(i/256)*block_bytes<Type>;
    return block_weight<Type>(b,i%256,block_scale<Type>(b));
}
} // namespace strata::deepseek41::iq
