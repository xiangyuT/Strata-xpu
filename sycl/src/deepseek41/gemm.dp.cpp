// Q8 block layout and SG32 accumulation follow native_mmvq.dp.cpp (ggml MIT).
// Copyright (c) 2023-2026 The ggml authors. See the repository's MIT license.
#include "strata/deepseek41/gemm.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/sycl_math.hpp"
#include <algorithm>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace strata::deepseek41::cuda {
namespace {
namespace es = sycl::ext::intel::esimd;
namespace xmx = es::xmx;
bool available(const sycl::device& device) {
    struct Features { sycl::device device; bool supported; };
    static thread_local std::optional<Features> cached;
    if (cached && cached->device == device) return cached->supported;
    bool supported = false;
    if (device.has(sycl::aspect::ext_intel_esimd)) {
        namespace mx = sycl::ext::oneapi::experimental::matrix;
        for (const auto& c : device.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>())
            if (c.atype == mx::matrix_type::sint8 && c.btype == mx::matrix_type::sint8 &&
                c.nsize == 16 && c.ksize == 32) supported = true;
    }
    cached = Features{device, supported};
    return supported;
}

// Q6_K raw GGUF blocks and the existing Q8_1 activation format.
namespace K = strata::kernels;
struct Q6Block { uint8_t ql[128], qh[64]; int8_t scales[16]; sycl::half d; };
struct Q81Block { sycl::half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q6Block)==210 && sizeof(Q81Block)==36);
// Immediate column consumption bounds activation live ranges. Sixteen query
// columns share each decoded weight block; the native32lane accumulator order,
// rounded second term, fused first term and XOR reduction remain unchanged.
inline sycl::int4 q6_load16(const void* p) {
    const uintptr_t address=reinterpret_cast<uintptr_t>(p);
    const auto* chunks=reinterpret_cast<const sycl::int4*>(address&~uintptr_t(15));
    const int index=int(address>>2)&3;
    const bool half=(address&2)!=0;
    const sycl::int4 lo=chunks[0],hi=(address&15)?chunks[1]:lo;
    const uint32_t words[8]={uint32_t(lo.x()),uint32_t(lo.y()),uint32_t(lo.z()),uint32_t(lo.w()),
                             uint32_t(hi.x()),uint32_t(hi.y()),uint32_t(hi.z()),uint32_t(hi.w())};
    uint32_t shifted[5],result[4];
#pragma unroll
    for(int i=0;i<5;++i)shifted[i]=index==0?words[i]:index==1?words[i+1]:index==2?words[i+2]:words[(i+3)&7];
#pragma unroll
    for(int i=0;i<4;++i)result[i]=half?(shifted[i]>>16)|(shifted[i+1]<<16):shifted[i];
    return sycl::int4(int(result[0]),int(result[1]),int(result[2]),int(result[3]));
}
inline sycl::int4 q6_act16(const int8_t* p) {
    sycl::int4 value;
    value.load(0,sycl::address_space_cast<sycl::access::address_space::global_space,sycl::access::decorated::no>(reinterpret_cast<const int*>(p)));
    return value;
}
template<int Start,int Width,int Columns>
inline void q6_columns(const Q81Block* x,int stride,int block,int group,int offset,
                           const int (&vi0)[4],const int (&vi1)[4],float dsc0,float dsc1,float (&acc)[Columns]) {
#pragma clang fp contract(off)
    sycl::int4 u0[Width],u1[Width];float dx0[Width],dx1[Width];
#pragma unroll
    for(int j=0;j<Width;++j) {
        const Q81Block* xb=x+size_t(Start+j)*stride+block*8+offset;
        u0[j]=q6_act16(xb[0].qs+16*(group&1));
        u1[j]=q6_act16(xb[2].qs+16*(group&1));
        dx0[j]=xb[0].ds[0];dx1[j]=xb[2].ds[0];
    }
#pragma unroll
    for(int j=0;j<Width;++j) {
        int s0=0,s1=0;
#pragma unroll
        for(int p=0;p<4;++p){s0=strata::dp4a(vi0[p],u0[j][p],s0);s1=strata::dp4a(vi1[p],u1[j][p],s1);}
        const float scale0=dx0[j]*dsc0,scale1=dx1[j]*dsc1;
        const float term1=scale1*float(s1);
        const float pair=sycl::fma(scale0,float(s0),term1);
        acc[Start+j]=acc[Start+j]+pair;
    }
    if constexpr(Start+Width<Columns)q6_columns<Start+Width,Width,Columns>(x,stride,block,group,offset,vi0,vi1,dsc0,dsc1,acc);
}
template<int Columns,int Width>
void q6_stream(sycl::queue& q,const Q6Block* w,const Q81Block* x,float* y,int input,int rows,int requested) {
    if(requested<=8){K::native_q6_k_mmvq(w,x,y,input,rows,requested,&q);return;}
    const int tokens=requested/Columns*Columns,blocks=input/256,stride=input/32;
    if(tokens)q.parallel_for(sycl::nd_range<2>({size_t(tokens/Columns),((size_t(rows)+3)/4)*128},{1,128}),
      [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
#pragma clang fp contract(off)
        const int lane=int(item.get_local_id(1))%32,row=int(item.get_group(1))*4+int(item.get_local_id(1))/32;
        if(row>=rows)return;
        const int g=lane&7,sub=lane>>3,shift=2*((g&3)>>1),qh_index=2*(g>>2)+(g&1);
        const int scale_index=8*(g>>2)+(g&3),xoffset=4*(g>>2)+((g&3)>>1);
        const auto* acts=x+size_t(item.get_group(0))*Columns*stride;
        float acc[Columns]={};
        for(int kb=sub;kb<blocks;kb+=4) {
            const Q6Block* block=w+size_t(row)*blocks+kb;
            const auto ql=q6_load16(block->ql+16*g),qh=q6_load16(block->qh+16*qh_index);
            const float d=block->d,dsc0=d*float(block->scales[scale_index]),dsc1=d*float(block->scales[scale_index+4]);
            int vi0[4],vi1[4];
#pragma unroll
            for(int p=0;p<4;++p) {
                const uint32_t low=uint32_t(ql[p]),high=uint32_t(qh[p]>>shift);
                const uint32_t a=(low&0x0f0f0f0fu)|((high<<4)&0x30303030u);
                const uint32_t b=((low>>4)&0x0f0f0f0fu)|(((high>>4)<<4)&0x30303030u);
                vi0[p]=int(((a|0x80808080u)-0x20202020u)^(~a&0x80808080u));
                vi1[p]=int(((b|0x80808080u)-0x20202020u)^(~b&0x80808080u));
            }
            q6_columns<0,Width,Columns>(acts,stride,kb,g,xoffset,vi0,vi1,dsc0,dsc1,acc);
        }
        const auto sg=item.get_sub_group();
#pragma unroll
        for(int j=0;j<Columns;++j) {
            float sum=acc[j];
#pragma unroll
            for(int offset=16;offset;offset/=2)sum+=sycl::permute_group_by_xor(sg,sum,offset);
            if(!lane)y[(size_t(item.get_group(0))*Columns+j)*rows+row]=sum;
        }
    });
    for(int first=tokens;first<requested;first+=8)
        K::native_q6_k_mmvq(w,x+size_t(first)*stride,y+size_t(first)*rows,input,rows,std::min(8,requested-first),&q);
}


// Wide32Q8 accumulates dot16 halves separately, at block residues modulo16.
// Preserve its scale expression boundary and final XOR reduction tree.
SYCL_ESIMD_FUNCTION inline es::simd<float,16> accumulate(es::simd<float,16> dw,
        float dx, es::simd<int,16> dot, es::simd<float,16> sum) {
#pragma clang fp contract(off)
    const es::simd<float,16> scale = dw * dx, values = es::convert<float>(dot);
    es::simd<float,16> term;
#ifdef __SYCL_DEVICE_ONLY__
    // Round the product before adding, as native Wide32Q8 does. ESIMD's
    // backend fused it even with contraction disabled and an arithmetic fence.
    __asm__("mul (M1, 16) %0 %1 %2" : "=r"(term.data_ref())
            : "r"(scale.data()), "r"(values.data()));
#else
    term = scale * values;
#endif
    return term + sum;
}
}

bool q6_k_gemm(const void* weights,const void* acts,float* output,int input,int rows,int tokens,void* stream) {
    if(tokens<=8)return false;
    if(!weights || !acts || !output || !stream || input<256 || input%256 || rows<1 || tokens>128)
        throw std::invalid_argument("DeepSeek Q6 GEMM geometry");
    auto& q=*static_cast<sycl::queue*>(stream);
    q6_stream<16,1>(q,static_cast<const Q6Block*>(weights),static_cast<const Q81Block*>(acts),output,input,rows,tokens);
    return true;
}

bool q8_0_gemm(const void* weights,const void* acts,float* output,int input,int rows,int tokens,void* stream) {
    static const bool wide=[] {const char* e=std::getenv("STRATA_MMVQ_WIDE_32");return !e || std::atoi(e)!=0;}();
    if (!wide || tokens <= 8) return false;
    if (!weights || !acts || !output || !stream || input < 32 || input%32 || rows < 1 || tokens > 128)
        throw std::invalid_argument("DeepSeek Q8 GEMM geometry");
    auto& q = *static_cast<sycl::queue*>(stream);
    const int blocks = input/32;
    // ESIMD gather byte offsets are uint32; larger matrices retain native MMVQ.
    if (uint64_t(blocks)*rows*34 > UINT32_MAX || uint64_t(blocks)*tokens*36 > UINT32_MAX ||
        uint64_t(rows)*tokens*4 > UINT32_MAX ||
        !available(q.get_device())) return false;
    const auto* w = static_cast<const uint8_t*>(weights);
    const auto* x = static_cast<const uint8_t*>(acts);
    const uint32_t ws=uint32_t(blocks)*34u, xs=uint32_t(blocks)*36u;
    // Two-dimensional grids avoid runtime integer division in ESIMD dispatch.
    q.parallel_for(sycl::nd_range<2>({size_t((tokens+7)/8),size_t((rows+15)/16)*16},{1,16}),
      [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
#pragma clang fp contract(off)
        es::slm_init<16384>();
        const int part = int(item.get_local_id(1));
        const int first_row = int(item.get_group(1))*16;
        const int first_token = int(item.get_group(0))*8;
        es::simd<uint32_t,128> bi(0,1);
        es::simd<uint32_t,64> ai(0,1);
        es::simd<uint32_t,16> ri(0,1);
        es::simd<uint32_t,8> ti(0,1);
        const es::simd<uint32_t,128> bo = ((bi%16)+first_row)*ws+(bi/16)*4+2;
        const es::simd<uint32_t,64> ao = ((ai/8)+first_token)*xs+(ai%8)*4+4;
        const es::simd_mask<128> bm = (bi%16)+first_row < uint32_t(rows);
        const es::simd_mask<64> am = (ai/8)+first_token < uint32_t(tokens);
        es::simd<float,128> low = 0, high = 0;
        for (int kb=part;kb<blocks;kb+=16) {
            const auto offsets = bo+uint32_t(kb)*34u;
            es::simd<uint32_t,128> b0 = es::gather<uint16_t,128>(reinterpret_cast<const uint16_t*>(w),offsets,bm);
            es::simd<uint32_t,128> b1 = es::gather<uint16_t,128>(reinterpret_cast<const uint16_t*>(w),offsets+2,bm);
            es::simd<uint32_t,128> b = b0 | (b1<<16);
            b.merge(0,!bm);
            es::simd<uint32_t,128> bl = b, bh = b;
            bl.merge(0,bi/16>=4); bh.merge(0,bi/16<4);
            es::simd<uint32_t,64> a = es::gather<uint32_t,64>(reinterpret_cast<const uint32_t*>(x),ao+uint32_t(kb)*36u,am);
            a.merge(0,!am);
            es::simd<int,128> blv=bl.bit_cast_view<int>(),bhv=bh.bit_cast_view<int>();
            es::simd<int,64> av=a.bit_cast_view<int>();
            es::simd<int,128> dl = xmx::dpas<8,8,int,int,int,int,xmx::dpas_argument_type::s8,xmx::dpas_argument_type::s8>(es::simd<int,128>(0),blv,av);
            es::simd<int,128> dh = xmx::dpas<8,8,int,int,int,int,xmx::dpas_argument_type::s8,xmx::dpas_argument_type::s8>(es::simd<int,128>(0),bhv,av);
            es::simd<float,16> dw = es::gather<sycl::half,16>(reinterpret_cast<const sycl::half*>(w),
                (ri+first_row)*ws+uint32_t(kb)*34u,ri+first_row<uint32_t(rows));
            es::simd<float,8> dx = es::gather<sycl::half,8>(reinterpret_cast<const sycl::half*>(x),
                (ti+first_token)*xs+uint32_t(kb)*36u,ti+first_token<uint32_t(tokens));
#pragma unroll
            for (int m=0;m<8;++m) {
                low.select<16,1>(m*16) = accumulate(dw,dx[m],dl.select<16,1>(m*16),low.select<16,1>(m*16));
                high.select<16,1>(m*16) = accumulate(dw,dx[m],dh.select<16,1>(m*16),high.select<16,1>(m*16));
            }
        }
        es::slm_block_store<float,128>(part*1024,low);
        es::slm_block_store<float,128>(part*1024+512,high);
        es::barrier();
        for (int offset=8;offset;offset/=2) {
            if (part<offset) {
                low += es::slm_block_load<float,128>((part+offset)*1024);
                high += es::slm_block_load<float,128>((part+offset)*1024+512);
                es::slm_block_store<float,128>(part*1024,low);
                es::slm_block_store<float,128>(part*1024+512,high);
            }
            es::barrier();
        }
        if (!part) {
            low += high;
#pragma unroll
            for (int m=0;m<8;++m) if (first_token+m<tokens)
                es::scatter<float,16>(output,(ri+first_row+uint32_t(first_token+m)*rows)*4,
                    low.select<16,1>(m*16),ri+first_row<uint32_t(rows));
        }
    });
    return true;
}
}
