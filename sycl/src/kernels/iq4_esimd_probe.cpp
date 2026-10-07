// Diagnostic ESIMD route: one explicit vector holds a native 32-lane row.
// Integer decode and the exact native FP32 lane/loop/reduction order are kept.
#include "strata/kernels/native_mmvq_tuning.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace e = sycl::ext::intel::esimd;
namespace {
SYCL_ESIMD_FUNCTION inline e::simd<uint32_t,32> lookup(e::simd<uint32_t,32> q) {
    e::simd<uint32_t,16> table;
    constexpr uint32_t values[] = {129,152,173,191,207,221,234,246,1,13,25,38,53,69,89,113};
#pragma unroll
    for (int i=0;i<16;++i) table[i]=values[i];
    e::simd<uint16_t,32> i0=q&15, i1=(q>>8)&15, i2=(q>>16)&15, i3=(q>>24)&15;
    return table.iselect(i0)|(table.iselect(i1)<<8)|(table.iselect(i2)<<16)|(table.iselect(i3)<<24);
}
template<int T> class IQ4EsimdProbe;
template<int T> void launch(const void* weights,const void* acts,float* y,int K,int N,sycl::queue& q) {
    const auto* w=static_cast<const uint8_t*>(weights);const auto* x=static_cast<const uint8_t*>(acts);
    q.parallel_for<IQ4EsimdProbe<T>>(sycl::nd_range<1>((N+15)/16*16,16),
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            const int row=it.get_global_linear_id();if(row>=N)return;
            e::simd<uint32_t,32> lane(0,1), g=lane&7, sub=lane>>3;
            const int nb=K/256, xs=K/32;
            e::simd<float,32> acc[T];
#pragma unroll
            for(int j=0;j<T;++j)acc[j]=0.f;
            for(int k0=0;k0<nb;k0+=4){
                e::simd<uint32_t,32> kb=sub+k0;const auto valid=kb<(uint32_t)nb;kb.merge(0,!valid);
                e::simd<uint32_t,32> off=(uint32_t)row*nb*136+kb*136;
                auto dh=e::gather<uint16_t,32>(reinterpret_cast<const uint16_t*>(w),off);
                auto sh=e::gather<uint16_t,32>(reinterpret_cast<const uint16_t*>(w),off+2);
                auto sl=e::gather<uint8_t,32>(w,off+4+(g>>1));
                e::simd<uint32_t,32> h=sh,l=sl;
                e::simd<int32_t,32> ls=((l>>(4*(g&1)))&15)|(((h>>(2*g))&3)<<4);
                e::simd<float,32> df=dh.template bit_cast_view<sycl::half>().read(), sf=ls-32;
                const e::simd<float,32> dw=df*sf;
                e::simd<int32_t,32> lo[4],hi[4];
#pragma unroll
                for(int i=0;i<4;++i){auto v=e::gather<uint32_t,32>(reinterpret_cast<const uint32_t*>(w),off+8+g*16+4*i);
                    auto vl=lookup(v&0x0f0f0f0f),vh=lookup((v>>4)&0x0f0f0f0f);
                    lo[i]=vl.template bit_cast_view<int32_t>().read();hi[i]=vh.template bit_cast_view<int32_t>().read();}
#pragma unroll
                for(int j=0;j<T;++j){const e::simd<uint32_t,32> ao=((uint32_t)j*xs+kb*8+g)*36;
                    auto ds=e::gather<uint16_t,32>(reinterpret_cast<const uint16_t*>(x),ao);
                    e::simd<float,32> dy=ds.template bit_cast_view<sycl::half>().read();
                    e::simd<int32_t,32> dot=0;
#pragma unroll
                    for(int i=0;i<8;++i){auto u=e::gather<int32_t,32>(reinterpret_cast<const int32_t*>(x),ao+4+4*i);
                        // ESIMD uses (accumulator, weight, activation); the
                        // CUDA/DPCT wrapper uses (weight, activation, accumulator).
                        dot=e::dp4a<int32_t>(dot,i<4?lo[i]:hi[i-4],u);}
                    e::simd<float,32> di=dot;
                    const e::simd<float,32> scale=dw*dy;
                    e::simd<float,32> term=scale*di;
                    // The VC backend otherwise fuses the final multiply/add
                    // despite the frontend's contraction flag. This integer
                    // mask leaves every valid term unchanged and keeps that
                    // product separate; invalid lanes still skip the update.
                    term.merge(0.f,!valid);
                    const e::simd<float,32> next=acc[j]+term;
                    acc[j].merge(next,valid);
                }
            }
#pragma unroll
            for(int j=0;j<T;++j){
                e::simd<float,16> a=acc[j].template select<16,1>(0)+acc[j].template select<16,1>(16);
                e::simd<float,8> b=a.template select<8,1>(0)+a.template select<8,1>(8);
                e::simd<float,4> c=b.template select<4,1>(0)+b.template select<4,1>(4);
                e::simd<float,2> d=c.template select<2,1>(0)+c.template select<2,1>(2);
                e::simd<float,1> z=d.template select<1,1>(0)+d.template select<1,1>(1);
                y[(size_t)j*N+row]=(float)z[0];
            }
        });
}
}
void native_iq4_xs_esimd_probe(const void* w,const void* x,float* y,int K,int N,int T,void* stream) {
    if(!w||!x||!y||!stream||K<=0||K%256||N<=0||T<1||T>8)throw std::invalid_argument("IQ4 ESIMD probe contract");
    auto& q=*strata::q_of(stream);
    switch(T){
#define GO(TT) case TT:launch<TT>(w,x,y,K,N,q);break;
        GO(1) GO(2) GO(3) GO(4) GO(5) GO(6) GO(7) GO(8)
#undef GO
    }
}
void native_iq4_esimd_lut_probe(uint32_t* output,void* stream) {
    auto& q=*strata::q_of(stream);
    q.parallel_for(sycl::nd_range<1>(2048,16),[=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
        e::simd<uint32_t,32> id(it.get_global_linear_id()*32,1);
        e::simd<uint32_t,32> v=(id&15)|((id&240)<<4)|((id&3840)<<8)|((id&61440)<<12)|((id*7919)&0xf0f0f0f0);
        auto out=lookup(v);e::block_store(output+it.get_global_linear_id()*32,out);
    });
}
} // namespace strata::kernels
