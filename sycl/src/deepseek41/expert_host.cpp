#include "strata/deepseek41/expert_host.hpp"
#include <sycl/sycl.hpp>
#include "strata/deepseek41/iq_weights.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <cstdio>
#include <climits>
#include <stdexcept>
#if defined(__linux__) && defined(__x86_64__)
#include "strata/deepseek41/flow_pool.hpp"
#include "strata/platform/aux_cpus.hpp"
#include <immintrin.h>
#include <bit>
#include <set>
namespace strata::deepseek41 {
namespace {
std::vector<int> cores(int requested){
 cpu_set_t allowed;CPU_ZERO(&allowed);
 if(sched_getaffinity(0,sizeof(allowed),&allowed))throw std::runtime_error("cold expert CPU affinity");
 std::vector<int> all,chosen;for(int i=0;i<CPU_SETSIZE;++i)if(CPU_ISSET(i,&allowed))all.push_back(i);
 std::set<std::pair<long,long>> seen;
 for(auto c:strata::aux_cpus::read_topology(all))if(c.pkg>=0 && c.core>=0 && seen.emplace(c.pkg,c.core).second)chosen.push_back(c.cpu);
 const int count=requested?requested:std::min(12,int(chosen.size()));
 if(count<1 || count>int(chosen.size()) || count>48)throw std::invalid_argument("cold expert physical CPU count");
 chosen.resize(count);return chosen;
}
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma")))
float dot(const uint8_t* row,const float* x,int n){
 const auto lut=_mm512_setr_epi32(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
 __m512 lo=_mm512_setzero_ps(),hi=lo;
 for(int k=0;k<n;k+=32,row+=17){
  const uint8_t code=row[0];const uint32_t bits=code<2?0x00200000u<<code:uint32_t(code-1)<<23;
  const auto scale=_mm512_set1_ps(std::bit_cast<float>(bits));
  const auto packed=_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row+1)));
  const auto l=_mm512_cvtepi32_ps(_mm512_permutexvar_epi32(_mm512_and_si512(packed,_mm512_set1_epi32(15)),lut));
  const auto h=_mm512_cvtepi32_ps(_mm512_permutexvar_epi32(_mm512_srli_epi32(packed,4),lut));
  lo=_mm512_fmadd_ps(_mm512_mul_ps(l,scale),_mm512_loadu_ps(x+k),lo);
  hi=_mm512_fmadd_ps(_mm512_mul_ps(h,scale),_mm512_loadu_ps(x+k+16),hi);
 }
 const auto s=_mm512_add_ps(lo,hi);const auto a=_mm256_add_ps(_mm512_castps512_ps256(s),_mm512_extractf32x8_ps(s,1));
 const auto b=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
 const auto c=_mm_add_ps(b,_mm_movehl_ps(b,b));return _mm_cvtss_f32(_mm_add_ss(c,_mm_shuffle_ps(c,c,_MM_SHUFFLE(1,1,1,1))));
}
// Decode each 32-value group once. Accumulators still follow the GPU's
// 32 lanes and reduction order; weights stay in their original GGUF encoding.
template<int Type>
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma")))
float dot_iq(const uint8_t* row,const float* x,int n){
 __m512 lo=_mm512_setzero_ps(),hi=lo;
 constexpr int Bytes=iq::block_bytes<Type>;
 for(int block=0;block<n;block+=256,row+=Bytes){
  const float d=iq::block_scale<Type>(row);
  for(int ib=0;ib<8;++ib){
   uint64_t grids[4];uint32_t signs=0;float scales[2];
   if constexpr(Type==19 || Type==29){
    uint32_t negative_delta=0;
    if constexpr(Type==19){
     const uint16_t h=iq::u16(row+34+ib*2);
     scales[0]=scales[1]=d*float(2*((h>>12)&7)+1);
     for(int l=0;l<4;++l)grids[l]=iq1s_grid[row[2+ib*4+l] | (((h>>(3*l))&7)<<8)];
     negative_delta=(h&0x8000)?UINT32_MAX:0;
    }else{
     const uint16_t sc=iq::u16(row+48+(ib/2)*2);
     for(int half=0;half<2;++half)scales[half]=d*float(2*((sc>>(6*(ib%2)+3*half))&7)+1);
     for(int l=0;l<4;++l){
      const int qh=(row[32+ib*2+l/2]>>(4*(l%2)))&15;
      grids[l]=iq1s_grid[row[ib*4+l] | ((qh&7)<<8)];
      if(qh&8)negative_delta|=255u<<(8*l);
     }
    }
    for(int half=0;half<2;++half){
     const auto bytes=_mm_set_epi64x(std::bit_cast<int64_t>(grids[half*2+1]),std::bit_cast<int64_t>(grids[half*2]));
     const auto delta=_mm512_mask_mov_ps(_mm512_set1_ps(0.125f),__mmask16(negative_delta>>(half*16)),_mm512_set1_ps(-0.125f));
     const auto values=_mm512_add_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes)),delta);
     const auto weights=_mm512_mul_ps(_mm512_set1_ps(scales[half]),values);
     const int k=block+ib*32+half*16;
     if(!half)lo=_mm512_fmadd_ps(weights,_mm512_loadu_ps(x+k),lo);
     else hi=_mm512_fmadd_ps(weights,_mm512_loadu_ps(x+k),hi);
    }
   }else{
    for(int l=0;l<4;++l){
     uint32_t sign;int scale;
     if constexpr(Type==16){
      const uint32_t meta=iq::u32(row+2+ib*8+4);
      grids[l]=iq2xxs_grid[row[2+ib*8+l]];sign=ksigns_iq2xs[(meta>>(7*l))&127];scale=meta>>28;
     }else if constexpr(Type==17){
      const uint16_t q=iq::u16(row+2+(ib*4+l)*2);
      grids[l]=iq2xs_grid[q&511];sign=ksigns_iq2xs[q>>9];scale=(row[66+ib]>>(4*(l/2)))&15;
     }else{
      const int index=row[2+ib*4+l] | ((int(row[66+ib])<<(8-2*l))&768);
      grids[l]=iq2s_grid[index];sign=row[34+ib*4+l];scale=(row[74+ib]>>(4*(l/2)))&15;
     }
     signs|=sign<<(l*8);scales[l/2]=d*(0.5f+float(scale))*0.25f;
    }
    for(int half=0;half<2;++half){
     const auto bytes=_mm_set_epi64x(std::bit_cast<int64_t>(grids[half*2+1]),std::bit_cast<int64_t>(grids[half*2]));
     const auto values=_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes));
     auto weights=_mm512_mul_ps(_mm512_set1_ps(scales[half]),values);
     const auto sign_bits=_mm512_maskz_set1_epi32(__mmask16(signs>>(half*16)),int32_t(0x80000000u));
     weights=_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(weights),sign_bits));
     const int k=block+ib*32+half*16;
     if(!half)lo=_mm512_fmadd_ps(weights,_mm512_loadu_ps(x+k),lo);
     else hi=_mm512_fmadd_ps(weights,_mm512_loadu_ps(x+k),hi);
    }
   }
  }
 }
 const auto s=_mm512_add_ps(lo,hi);const auto a=_mm256_add_ps(_mm512_castps512_ps256(s),_mm512_extractf32x8_ps(s,1));
 const auto b=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
 const auto c=_mm_add_ps(b,_mm_movehl_ps(b,b));return _mm_cvtss_f32(_mm_add_ss(c,_mm_shuffle_ps(c,c,_MM_SHUFFLE(1,1,1,1))));
}
// IQ2_XXS magnitudes8/25/43 have distinct low nibbles. Finite F16 has at
// most11 significant bits and these coefficients at most11, so all original
// intermediates and the combined product are exact in FP32. IQ1_S needs at
// most19 product bits. Nonfinite scales retain the original decoder.
struct alignas(64) IQ2Coefficients {
 std::array<std::array<float,16>,16> rows{};
 constexpr IQ2Coefficients(){for(int s=0;s<16;++s){
  rows[s][8]=float(2*s+1);rows[s][9]=float((2*s+1)*25)/8;
  rows[s][11]=float((2*s+1)*43)/8;
 }}
};
struct alignas(64) IQ1Coefficients {
 std::array<std::array<std::array<float,16>,2>,8> rows{};
 constexpr IQ1Coefficients(){for(int s=0;s<8;++s)for(int delta=0;delta<2;++delta)
  for(int i:{0,1,15})rows[s][delta][i]=float(2*s+1)*(float(i==15?-1:i)+(delta?-.125f:.125f));
 }
};
static constexpr IQ2Coefficients iq2_coefficients;
static constexpr IQ1Coefficients iq1_coefficients;
bool coefficient_available(){
#ifndef __SYCL_DEVICE_ONLY__
 __builtin_cpu_init();
 if(!__builtin_cpu_supports("f16c") || !__builtin_cpu_supports("bmi2"))return false;
 static const bool tables_match=[] {
  for(uint64_t word:iq2xxs_grid)for(int k=0;k<8;++k){
   const auto v=uint8_t(word>>(8*k));if(v!=8 && v!=25 && v!=43)return false;
  }
  for(uint64_t word:iq1s_grid)for(int k=0;k<8;++k){
   const auto v=int8_t(word>>(8*k));if(v!=-1 && v!=0 && v!=1)return false;
  }
  return true;
 }();
 return tables_match;
#else
 return false;
#endif
}
__attribute__((target("bmi2")))
uint32_t coefficient_signs(uint32_t packed){
 const uint32_t bytes=_pdep_u32(packed,0x7f7f7f7f);
 uint32_t p=bytes^(bytes>>4);p^=p>>2;p^=p>>1;
 return bytes^((p&0x01010101)<<7);
}
inline uint64_t coefficient_grid(const uint64_t* table,unsigned index){
 // Retain compressed scalar grid reads; auto-vectorized gathers regressed.
 return *reinterpret_cast<const volatile uint64_t*>(table+index);
}
template<int Type>
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma,f16c,bmi2")))
float dot_coefficient(const uint8_t* row,const float* x,int n){
 static_assert(Type==16 || Type==19);
 const auto* original=row;__m512 lo=_mm512_setzero_ps(),hi=lo;
 for(int block=0;block<n;block+=256,row+=iq::block_bytes<Type>){
  const uint16_t h=iq::u16(row);
  if((h&0x7c00)==0x7c00){
   // Call the same out-of-line decoder as the original dispatch. Inlining
   // another copy can change which NaN sign survives the reduction.
   auto volatile fallback=&dot_iq<Type>;
   return fallback(original,x,n);
  }
  const auto d=_mm512_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))));
  for(int ib=0;ib<8;++ib){
   uint64_t grids[4];uint32_t signs=0;int scale,delta=0;
   if constexpr(Type==19){
    const uint16_t meta=iq::u16(row+34+ib*2);scale=(meta>>12)&7;delta=bool(meta&0x8000);
    for(int l=0;l<4;++l)grids[l]=coefficient_grid(iq1s_grid,row[2+ib*4+l]|(((meta>>(3*l))&7)<<8));
   }else{
    const uint32_t meta=iq::u32(row+2+ib*8+4);scale=meta>>28;signs=coefficient_signs(meta);
    for(int l=0;l<4;++l)grids[l]=coefficient_grid(iq2xxs_grid,row[2+ib*8+l]);
   }
   for(int half=0;half<2;++half){
    const auto bytes=_mm_set_epi64x(std::bit_cast<int64_t>(grids[half*2+1]),std::bit_cast<int64_t>(grids[half*2]));
    const auto indices=_mm512_and_si512(_mm512_cvtepu8_epi32(bytes),_mm512_set1_epi32(15));
    const float* table;
    if constexpr(Type==19)table=iq1_coefficients.rows[scale][delta].data();
    else table=iq2_coefficients.rows[scale].data();
    auto weights=_mm512_mul_ps(d,_mm512_permutexvar_ps(indices,_mm512_load_ps(table)));
    if constexpr(Type==16){
     const auto mask=_mm512_maskz_set1_epi32(__mmask16(signs>>(half*16)),int32_t(0x80000000u));
     weights=_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(weights),mask));
    }
    const auto input=_mm512_loadu_ps(x+block+ib*32+half*16);
    if(!half)lo=_mm512_fmadd_ps(weights,input,lo);else hi=_mm512_fmadd_ps(weights,input,hi);
   }
  }
 }
 const auto s=_mm512_add_ps(lo,hi);const auto a=_mm256_add_ps(_mm512_castps512_ps256(s),_mm512_extractf32x8_ps(s,1));
 const auto b=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
 const auto c=_mm_add_ps(b,_mm_movehl_ps(b,b));return _mm_cvtss_f32(_mm_add_ss(c,_mm_shuffle_ps(c,c,_MM_SHUFFLE(1,1,1,1))));
}
// Two independent coefficient dots share one activation load. Each keeps
// the original32lane FMA/reduction; nonfinite rows use canonical decoding.
__attribute__((always_inline,target("avx512f,avx512bw,avx512dq,avx512vl,fma,f16c,bmi2")))
inline float pair_reduce(__m512 lo,__m512 hi) {
    const auto s=_mm512_add_ps(lo,hi);
    const auto a=_mm256_add_ps(_mm512_castps512_ps256(s),_mm512_extractf32x8_ps(s,1));
    const auto b=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
    const auto c=_mm_add_ps(b,_mm_movehl_ps(b,b));
    return _mm_cvtss_f32(_mm_add_ss(c,_mm_shuffle_ps(c,c,_MM_SHUFFLE(1,1,1,1))));
}

template<int Type>
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma,f16c,bmi2")))
void dot_pair_coefficient(const uint8_t* gate,const uint8_t* up,const float* x,int n,
                        float* gate_out,float* up_out) {
    static_assert(Type==16 || Type==19);
    const auto* original_gate=gate;
    const auto* original_up=up;
    __m512 g0=_mm512_setzero_ps(),g1=g0,u0=g0,u1=g0;
    for(int block=0;block<n;block+=256,
        gate+=iq::block_bytes<Type>,up+=iq::block_bytes<Type>) {
        const uint16_t h[2]={iq::u16(gate),iq::u16(up)};
        if((h[0]&0x7c00)==0x7c00 || (h[1]&0x7c00)==0x7c00) {
            // Preserve the canonical independent dots, including NaN signs.
            auto volatile fallback=&dot_iq<Type>;
            *gate_out=fallback(original_gate,x,n);
            *up_out=fallback(original_up,x,n);
            return;
        }
        const __m512 scales[2]={
            _mm512_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h[0])))),
            _mm512_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h[1]))))};
        for(int ib=0;ib<8;++ib) {
            const auto x0=_mm512_loadu_ps(x+block+ib*32);
            const auto x1=_mm512_loadu_ps(x+block+ib*32+16);
            __m512 decoded[2][2];
            #pragma unroll
            for(int matrix=0;matrix<2;++matrix) {
                const auto* row=matrix?up:gate;
                uint64_t grids[4];
                uint32_t signs=0;
                int scale,delta=0;
                if constexpr(Type==19) {
                    const uint16_t meta=iq::u16(row+34+ib*2);
                    scale=(meta>>12)&7;delta=bool(meta&0x8000);
                    for(int l=0;l<4;++l)
                        grids[l]=coefficient_grid(iq1s_grid,
                            row[2+ib*4+l]|(((meta>>(3*l))&7)<<8));
                }else{
                    const uint32_t meta=iq::u32(row+2+ib*8+4);
                    scale=meta>>28;signs=coefficient_signs(meta);
                    for(int l=0;l<4;++l)
                        grids[l]=coefficient_grid(iq2xxs_grid,row[2+ib*8+l]);
                }
                #pragma unroll
                for(int half=0;half<2;++half) {
                    const auto bytes=_mm_set_epi64x(std::bit_cast<int64_t>(grids[half*2+1]),
                                                   std::bit_cast<int64_t>(grids[half*2]));
                    const auto indices=_mm512_and_si512(_mm512_cvtepu8_epi32(bytes),
                                                       _mm512_set1_epi32(15));
                    const float* table;
                    if constexpr(Type==19)table=iq1_coefficients.rows[scale][delta].data();
                    else table=iq2_coefficients.rows[scale].data();
                    auto value=_mm512_mul_ps(scales[matrix],
                        _mm512_permutexvar_ps(indices,_mm512_load_ps(table)));
                    if constexpr(Type==16) {
                        const auto mask=_mm512_maskz_set1_epi32(
                            __mmask16(signs>>(half*16)),int32_t(0x80000000u));
                        value=_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(value),mask));
                    }
                    decoded[matrix][half]=value;
                }
            }
            g0=_mm512_fmadd_ps(decoded[0][0],x0,g0);
            g1=_mm512_fmadd_ps(decoded[0][1],x1,g1);
            u0=_mm512_fmadd_ps(decoded[1][0],x0,u0);
            u1=_mm512_fmadd_ps(decoded[1][1],x1,u1);
        }
    }
    *gate_out=pair_reduce(g0,g1);
    *up_out=pair_reduce(u0,u1);
}

using PairDot=void(*)(const uint8_t*,const uint8_t*,const float*,int,float*,float*);
// Original IQ2_XS/IQ2_S decode arithmetic for two independent rows.
// Share activation loads without changing either row's decode or dot order.
__attribute__((always_inline,target("avx512f,avx512bw,avx512dq,avx512vl,fma")))
inline float raw_pair_reduce(__m512 low,__m512 high) {
    const auto s=_mm512_add_ps(low,high);const auto a=_mm256_add_ps(_mm512_castps512_ps256(s),_mm512_extractf32x8_ps(s,1));
    const auto b=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
    const auto c=_mm_add_ps(b,_mm_movehl_ps(b,b));return _mm_cvtss_f32(_mm_add_ss(c,_mm_shuffle_ps(c,c,_MM_SHUFFLE(1,1,1,1))));
}

template<int Type>
__attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,fma")))
void dot_pair_raw(const uint8_t* first,const uint8_t* second,const float* x,int n,float* a,float* b) {
    static_assert(Type==17 || Type==22);
    const uint8_t* rows[2]={first,second};
    __m512 lo[2]={_mm512_setzero_ps(),_mm512_setzero_ps()};
    __m512 hi[2]={_mm512_setzero_ps(),_mm512_setzero_ps()};
    for(int block=0;block<n;block+=256) {
        float d[2];
        for(int r=0;r<2;++r) {
            const auto* row=rows[r]+size_t(block/256)*iq::block_bytes<Type>;
            if((iq::u16(row)&0x7c00)==0x7c00) {
                // Preserve the original whole-dot NaN payload/order, including
                // the case where the other row contains a different NaN.
                auto volatile original=&dot_iq<Type>;
                *a=original(first,x,n);*b=original(second,x,n);return;
            }
            d[r]=iq::block_scale<Type>(row);
        }
        for(int ib=0;ib<8;++ib) {
            const int k=block+ib*32;
            const auto xl=_mm512_loadu_ps(x+k),xh=_mm512_loadu_ps(x+k+16);
            for(int r=0;r<2;++r) {
                const auto* row=rows[r]+size_t(block/256)*iq::block_bytes<Type>;
                uint64_t grids[4];uint32_t signs=0;float scales[2];
                for(int l=0;l<4;++l) {
                    uint32_t sign;int scale;
                    if constexpr(Type==17) {
                        const uint16_t q=iq::u16(row+2+(ib*4+l)*2);
                        grids[l]=iq2xs_grid[q&511];sign=ksigns_iq2xs[q>>9];
                        scale=(row[66+ib]>>(4*(l/2)))&15;
                    }else {
                        const int index=row[2+ib*4+l] | ((int(row[66+ib])<<(8-2*l))&768);
                        grids[l]=iq2s_grid[index];sign=row[34+ib*4+l];
                        scale=(row[74+ib]>>(4*(l/2)))&15;
                    }
                    signs|=sign<<(l*8);scales[l/2]=d[r]*(0.5f+float(scale))*0.25f;
                }
                for(int half=0;half<2;++half) {
                    const auto bytes=_mm_set_epi64x(std::bit_cast<int64_t>(grids[half*2+1]),std::bit_cast<int64_t>(grids[half*2]));
                    const auto values=_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes));
                    auto weights=_mm512_mul_ps(_mm512_set1_ps(scales[half]),values);
                    const auto sign_bits=_mm512_maskz_set1_epi32(__mmask16(signs>>(half*16)),int32_t(0x80000000u));
                    weights=_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(weights),sign_bits));
                    if(!half)lo[r]=_mm512_fmadd_ps(weights,xl,lo[r]);
                    else hi[r]=_mm512_fmadd_ps(weights,xh,hi[r]);
                }
            }
        }
    }
    *a=raw_pair_reduce(lo[0],hi[0]);*b=raw_pair_reduce(lo[1],hi[1]);
}

PairDot raw_pair(int type) {
    if(type==17)return dot_pair_raw<17>;
    if(type==22)return dot_pair_raw<22>;
    throw std::invalid_argument("raw CPU pair format");
}

PairDot pair_decoder(int type) {
 if(type==16)return dot_pair_coefficient<16>;
 if(type==19)return dot_pair_coefficient<19>;
 return nullptr;
}

using Dot=float(*)(const uint8_t*,const float*,int);
Dot decoder(int type,bool coefficient=false){
 if(coefficient){if(type==16)return dot_coefficient<16>;if(type==19)return dot_coefficient<19>;}
 switch(type){case 39:return dot;case 16:return dot_iq<16>;case 17:return dot_iq<17>;
  case 19:return dot_iq<19>;case 22:return dot_iq<22>;case 29:return dot_iq<29>;
  default:throw std::invalid_argument("unsupported CPU expert format");}
}

}
bool ExpertHost::available(){
#ifndef __SYCL_DEVICE_ONLY__
 __builtin_cpu_init();return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("fma");
#else
 return false;
#endif
}
struct ExpertHost::Impl {
 sycl::queue& q;int input,ff;FlowPool pool;
 cpu_set_t original_affinity;pthread_t caller=pthread_self();
 float *x=nullptr,*gu=nullptr,*hidden=nullptr,*out=nullptr,*device_gu=nullptr,*device_hidden=nullptr;
 uint64_t calls=0,input_copies=0;sycl::event input_ready;bool prepared=false;
 const int capacity,input_rows;int prepared_rows=0;bool coefficient,paired,raw_down;
 Impl(void* stream,int h,int f,int threads,int max_jobs,int max_rows,bool enable_coefficient,bool enable_paired,bool enable_raw_down):q(*static_cast<sycl::queue*>(stream)),input(h),ff(f),pool(cores(threads)),capacity(max_jobs),input_rows(max_rows),coefficient(enable_coefficient && coefficient_available()),paired(enable_paired && coefficient),raw_down(enable_raw_down){
  if(!q.has_property<sycl::property::queue::in_order>() || h<32 || f<32 || h%32 || f%32 || max_jobs<1 || max_rows<1 || h>INT_MAX/max_jobs || f>INT_MAX/max_jobs || h>INT_MAX/max_rows)throw std::invalid_argument("cold expert geometry/queue");
  if(pthread_getaffinity_np(caller,sizeof(original_affinity),&original_affinity))throw std::runtime_error("cold expert caller affinity");
  try{
   x=sycl::malloc_host<float>(size_t(h)*input_rows,q);gu=sycl::malloc_host<float>(size_t(f)*2*capacity,q);hidden=sycl::malloc_host<float>(size_t(f)*capacity,q);out=sycl::malloc_host<float>(size_t(h)*capacity,q);
   device_gu=sycl::malloc_device<float>(size_t(f)*2*capacity,q);device_hidden=sycl::malloc_device<float>(size_t(f)*capacity,q);
   if(!x || !gu || !hidden || !out || !device_gu || !device_hidden)throw std::bad_alloc();
   if(coefficient)std::fprintf(stderr,"strata-deepseek41-sycl: CPU coefficient decoder enabled for IQ2_XXS/IQ1_S; original other formats\n");
  }catch(...){release();throw;}
 }
 void release(){try{q.wait_and_throw();}catch(...){}for(auto p:{x,gu,hidden,out,device_gu,device_hidden})if(p)sycl::free(p,q);}
 ~Impl(){release();pthread_setaffinity_np(caller,sizeof(original_affinity),&original_affinity);}
 void prepare_input(const float* in){prepare_inputs(in,1);}
 void prepare_inputs(const float* in,int rows){
  if(!in || rows<1 || rows>input_rows)throw std::invalid_argument("cold expert input");
  input_ready=q.memcpy(x,in,size_t(input)*rows*4);prepared=true;prepared_rows=rows;input_copies+=rows;
 }
 void copy_expert(uint8_t* destination,const uint8_t* gate,size_t gate_bytes,
                  const uint8_t* up,size_t up_bytes,const uint8_t* down,size_t down_bytes){
  constexpr size_t Max=std::numeric_limits<size_t>::max();
  if(gate_bytes>Max-up_bytes || gate_bytes+up_bytes>Max-down_bytes)
   throw std::invalid_argument("expert staging byte overflow");
  const size_t total=gate_bytes+up_bytes+down_bytes;
  if(!total)return;
  if(!destination || (gate_bytes && !gate) || (up_bytes && !up) || (down_bytes && !down))
   throw std::invalid_argument("expert staging pointers");
  const std::array<const uint8_t*,3> sources{gate,up,down};
  const std::array<size_t,3> sizes{gate_bytes,up_bytes,down_bytes};
  const int workers=std::min(4,pool.size());
  // Bound publication overhead for small blobs; no format/shape whitelist.
  if(workers==1 || total<(1u<<20)){
   size_t at=0;for(int i=0;i<3;++i){if(sizes[i])std::memcpy(destination+at,sources[i],sizes[i]);at+=sizes[i];}
   return;
  }
  const size_t chunk=total/size_t(workers)+(total%size_t(workers)!=0);
  if(chunk>Max-63)throw std::invalid_argument("expert staging stripe overflow");
  const size_t stripe=(chunk+63)&~size_t(63);
  // run_batch's activation callback runs after its GU pool epoch completed
  // and before the down epoch begins, so the pool is also idle there.
  pool.run([&](int id){
   if(id>=workers)return;
   const size_t first=size_t(id)*stripe;
   if(first>=total)return;
   const size_t last=first+std::min(stripe,total-first);size_t segment=0;
   for(int i=0;i<3;++i){
    const size_t begin=std::max(first,segment),end=std::min(last,segment+sizes[i]);
    if(begin<end)std::memcpy(destination+begin,sources[i]+begin-segment,end-begin);
    segment+=sizes[i];
   }
  });
 }
 void run_batch(int gt,int ut,int dt,const ExpertHost::Job* jobs,int count,float limit,const std::function<void()>& after_activation,bool contiguous_rows){
  if(!prepared || !jobs || count<1 || count>capacity)throw std::invalid_argument("cold expert batch/input");
  for(int j=0;j<count;++j)if(!jobs[j].gate || !jobs[j].up || !jobs[j].down || !jobs[j].output || jobs[j].input_row<0 || jobs[j].input_row>=prepared_rows)
   throw std::invalid_argument("cold expert batch pointers");
  const auto gate_dot=decoder(gt,coefficient),up_dot=decoder(ut,coefficient),down_dot=decoder(dt,coefficient);
  const size_t gs=iq::row_bytes(gt,input),us=iq::row_bytes(ut,input),ds=iq::row_bytes(dt,ff);
  // The staged input is read-only across consecutive cold batches of this FFN.
  input_ready.wait_and_throw();
  if(const auto pair=(paired && gt==ut)?pair_decoder(gt):nullptr) {
if(contiguous_rows){
   const int tiles=ff/16+(ff%16!=0);
   pool.run([&](int id){for(int at=id;at<count*tiles;at+=pool.size()){
    const int j=at/tiles,first=16*(at%tiles),last=first+std::min(16,ff-first);
    for(int row=first;row<last;row+=1){pair(jobs[j].gate+size_t(row)*gs,jobs[j].up+size_t(row)*us,
         x+size_t(jobs[j].input_row)*input,input,gu+size_t(j)*2*ff+row,gu+size_t(j)*2*ff+ff+row);}
   }});
  }else {   pool.run([&](int id){for(int at=id;at<count*ff;at+=pool.size()){
    const int j=at/ff,row=at%ff;
    pair(jobs[j].gate+size_t(row)*gs,jobs[j].up+size_t(row)*us,
         x+size_t(jobs[j].input_row)*input,input,gu+size_t(j)*2*ff+row,gu+size_t(j)*2*ff+ff+row);
   }});}

  }else {
if(contiguous_rows){
   const int tiles=ff/16+(ff%16!=0);
   pool.run([&](int id){for(int at=id;at<count*tiles;at+=pool.size()){
    const int j=at/tiles,first=16*(at%tiles),last=first+std::min(16,ff-first);
    for(int row=first;row<last;row+=1){gu[size_t(j)*2*ff+row]=gate_dot(jobs[j].gate+size_t(row)*gs,x+size_t(jobs[j].input_row)*input,input);
   gu[size_t(j)*2*ff+ff+row]=up_dot(jobs[j].up+size_t(row)*us,x+size_t(jobs[j].input_row)*input,input);}
   }});
  }else {  pool.run([&](int id){for(int at=id;at<count*ff;at+=pool.size()){
   const int j=at/ff,row=at%ff;gu[size_t(j)*2*ff+row]=gate_dot(jobs[j].gate+size_t(row)*gs,x+size_t(jobs[j].input_row)*input,input);
   gu[size_t(j)*2*ff+ff+row]=up_dot(jobs[j].up+size_t(row)*us,x+size_t(jobs[j].input_row)*input,input);
  }});}

  }
  q.memcpy(device_gu,gu,size_t(count)*ff*8);
  auto* dg=device_gu;auto* dh=device_hidden;const int n=ff;
  q.parallel_for(sycl::range<1>(size_t(count)*ff),[=](sycl::id<1> at){
   const int i=at[0],j=i/n,row=i%n;
   const float g=sycl::fmin(dg[size_t(j)*2*n+row],limit),u=sycl::fmax(-limit,sycl::fmin(dg[size_t(j)*2*n+n+row],limit));
   dh[i]=(g/(1+sycl::exp(-g)))*u;
  });
  auto hidden_ready=q.memcpy(hidden,device_hidden,size_t(count)*ff*4);
  if(after_activation)after_activation();
  hidden_ready.wait_and_throw();
  if(const auto pair=(raw_down && (dt==17 || dt==22))?raw_pair(dt):(paired?pair_decoder(dt):nullptr)) {
   const int pairs=(input+1)/2;
if(contiguous_rows){
   const int tiles=input/16+(input%16!=0);
   pool.run([&](int id){for(int at=id;at<count*tiles;at+=pool.size()){
    const int j=at/tiles,first=16*(at%tiles),last=first+std::min(16,input-first);
    for(int row=first;row<last;row+=2){const auto* w=jobs[j].down+size_t(row)*ds;
    if(row+1<input)pair(w,w+ds,hidden+size_t(j)*ff,ff,out+size_t(j)*input+row,out+size_t(j)*input+row+1);
    else out[size_t(j)*input+row]=down_dot(w,hidden+size_t(j)*ff,ff);}
   }});
  }else {   pool.run([&](int id){for(int at=id;at<count*pairs;at+=pool.size()){
    const int j=at/pairs,row=2*(at%pairs);const auto* w=jobs[j].down+size_t(row)*ds;
    if(row+1<input)pair(w,w+ds,hidden+size_t(j)*ff,ff,out+size_t(j)*input+row,out+size_t(j)*input+row+1);
    else out[size_t(j)*input+row]=down_dot(w,hidden+size_t(j)*ff,ff);
   }});}

  }else {
if(contiguous_rows){
   const int tiles=input/16+(input%16!=0);
   pool.run([&](int id){for(int at=id;at<count*tiles;at+=pool.size()){
    const int j=at/tiles,first=16*(at%tiles),last=first+std::min(16,input-first);
    for(int row=first;row<last;row+=1){out[size_t(j)*input+row]=down_dot(jobs[j].down+size_t(row)*ds,hidden+size_t(j)*ff,ff);}
   }});
  }else {  pool.run([&](int id){for(int at=id;at<count*input;at+=pool.size()){
   const int j=at/input,row=at%input;out[at]=down_dot(jobs[j].down+size_t(row)*ds,hidden+size_t(j)*ff,ff);
  }});}

  }
  for(int j=0;j<count;++j)q.memcpy(jobs[j].output,out+size_t(j)*input,size_t(input)*4);
  calls+=count;
 }

 void run(int gt,int ut,int dt,const uint8_t* gate,const uint8_t* up,const uint8_t* down,const float* in,float* result,float limit){
  if(!gate || !up || !down || !in || !result)throw std::invalid_argument("cold expert pointers");
  const auto gate_dot=decoder(gt,coefficient),up_dot=decoder(ut,coefficient),down_dot=decoder(dt,coefficient);
  const size_t gs=iq::row_bytes(gt,input),us=iq::row_bytes(ut,input),ds=iq::row_bytes(dt,ff);
  prepared=false;q.memcpy(x,in,size_t(input)*4).wait_and_throw();++input_copies;
  if(const auto pair=(paired && gt==ut)?pair_decoder(gt):nullptr) {
   pool.run([&](int id){for(int row=id;row<ff;row+=pool.size())
    pair(gate+size_t(row)*gs,up+size_t(row)*us,x,input,gu+row,gu+ff+row);
   });
  }else {
  pool.run([&](int id){for(int row=id;row<ff;row+=pool.size()){
   gu[row]=gate_dot(gate+size_t(row)*gs,x,input);gu[ff+row]=up_dot(up+size_t(row)*us,x,input);
  }});
  }
  q.memcpy(device_gu,gu,size_t(ff)*8);
  auto* dg=device_gu;auto* dh=device_hidden;const int n=ff;
  q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> at){const int i=at[0];const float g=sycl::fmin(dg[i],limit),u=sycl::fmax(-limit,sycl::fmin(dg[n+i],limit));dh[i]=(g/(1+sycl::exp(-g)))*u;});
  q.memcpy(hidden,device_hidden,size_t(ff)*4).wait_and_throw();
  if(const auto pair=(raw_down && (dt==17 || dt==22))?raw_pair(dt):(paired?pair_decoder(dt):nullptr)) {
   pool.run([&](int id){for(int at=id;at<(input+1)/2;at+=pool.size()){
    const int row=2*at;const auto* w=down+size_t(row)*ds;
    if(row+1<input)pair(w,w+ds,hidden,ff,out+row,out+row+1);
    else out[row]=down_dot(w,hidden,ff);
   }});
  }else {
  pool.run([&](int id){for(int row=id;row<input;row+=pool.size())out[row]=down_dot(down+size_t(row)*ds,hidden,ff);});
  }
  q.memcpy(result,out,size_t(input)*4);++calls;
 }
};
ExpertHost::ExpertHost(void* q,int input,int ff,int threads,int max_jobs,int max_rows):ExpertHost(q,input,ff,threads,max_jobs,max_rows,false){}
ExpertHost::ExpertHost(void* q,int input,int ff,int threads,int max_jobs,int max_rows,bool coefficient):ExpertHost(q,input,ff,threads,max_jobs,max_rows,coefficient,false){}
ExpertHost::ExpertHost(void* q,int input,int ff,int threads,int max_jobs,int max_rows,bool coefficient,bool paired):ExpertHost(q,input,ff,threads,max_jobs,max_rows,coefficient,paired,false){}
ExpertHost::ExpertHost(void* q,int input,int ff,int threads,int max_jobs,int max_rows,bool coefficient,bool paired,bool raw_down){if(!q || !available())throw std::invalid_argument("cold expert requires AVX512");p_=std::make_unique<Impl>(q,input,ff,threads,max_jobs,max_rows,coefficient,paired,raw_down);}
ExpertHost::~ExpertHost()=default;
void ExpertHost::run(int gt,int ut,int dt,const uint8_t* g,const uint8_t* u,const uint8_t* d,const float* x,float* y,float limit){p_->run(gt,ut,dt,g,u,d,x,y,limit);}
void ExpertHost::prepare_input(const float* x){p_->prepare_input(x);}
void ExpertHost::prepare_inputs(const float* x,int rows){p_->prepare_inputs(x,rows);}
void ExpertHost::run_batch(int gt,int ut,int dt,const Job* jobs,int count,float limit,const std::function<void()>& after_activation,bool contiguous_rows){p_->run_batch(gt,ut,dt,jobs,count,limit,after_activation,contiguous_rows);}
void ExpertHost::copy_expert(uint8_t* dst,const uint8_t* g,size_t gb,const uint8_t* u,size_t ub,const uint8_t* d,size_t db){p_->copy_expert(dst,g,gb,u,ub,d,db);}
void ExpertHost::report()const{std::fprintf(stderr,"strata-deepseek41-sycl: CPU cold workers=%d groups=%llu activation_H2D_bytes=%llu activation_D2H_bytes=%llu\n",p_->pool.size(),(unsigned long long)p_->calls,(unsigned long long)(p_->calls*(size_t(p_->ff)*8+size_t(p_->input)*4)),(unsigned long long)(p_->calls*size_t(p_->ff)*4+p_->input_copies*size_t(p_->input)*4));}
}
#else
namespace strata::deepseek41 {
struct ExpertHost::Impl {};
bool ExpertHost::available(){return false;}
ExpertHost::ExpertHost(void*,int,int,int,int,int){throw std::runtime_error("cold expert CPU unavailable");}
ExpertHost::ExpertHost(void*,int,int,int,int,int,bool){throw std::runtime_error("cold expert CPU unavailable");}
ExpertHost::ExpertHost(void*,int,int,int,int,int,bool,bool){throw std::runtime_error("cold expert CPU unavailable");}
ExpertHost::ExpertHost(void*,int,int,int,int,int,bool,bool,bool){throw std::runtime_error("cold expert CPU unavailable");}
ExpertHost::~ExpertHost()=default;
void ExpertHost::run(int,int,int,const uint8_t*,const uint8_t*,const uint8_t*,const float*,float*,float){throw std::runtime_error("cold expert CPU unavailable");}
void ExpertHost::prepare_inputs(const float*,int){throw std::runtime_error("cold expert CPU unavailable");}
void ExpertHost::prepare_input(const float*){throw std::runtime_error("cold expert CPU unavailable");}
void ExpertHost::run_batch(int,int,int,const Job*,int,float,const std::function<void()>&,bool){throw std::runtime_error("cold expert CPU unavailable");}
void ExpertHost::copy_expert(uint8_t*,const uint8_t*,size_t,const uint8_t*,size_t,const uint8_t*,size_t){throw std::runtime_error("cold expert CPU unavailable");}
void ExpertHost::report()const{}
}
#endif
