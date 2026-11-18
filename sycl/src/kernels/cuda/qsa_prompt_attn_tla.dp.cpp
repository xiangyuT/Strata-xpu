// FP32 sparse prompt attention with TLA block K loads and cached query tiles.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/qsa_prompt_attn_tla.hpp"
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>
#include <cfloat>
#include <cstdlib>
#include <type_traits>
#include <cstdio>
#ifdef printf
#undef printf
#endif

namespace strata::kernels::tla_prompt_fp32 {
constexpr int HD=256,G=12,CH=64,Threads=64;
class Core;
class Merge;
class CorePV;
class MergeVec2;
class MergeSubgroup;
inline float h2f(uint16_t b){return float(sycl::bit_cast<sycl::half>(b));}
// For every signed byte x, x is reconstructed exactly as
// bit_cast<float>(0x4b000000 | (uint8_t(x)^0x80)) - (2^23+128).
// The subtraction is exact over [-128,127]; scaling remains original FP32.
inline auto exact_signed_bytes(uint64_t bits){
 using UInt8=uint32_t __attribute__((ext_vector_type(8)));
 using Float8=float __attribute__((ext_vector_type(8)));
 const auto words=sycl::bit_cast<sycl::uint2>(bits);
 const UInt8 data{words[0],words[0],words[0],words[0],words[1],words[1],words[1],words[1]};
 const UInt8 shifts{0,8,16,24,0,8,16,24};
 const UInt8 biased=((data>>shifts)&UInt8(255))^UInt8(0x4b000080U);
 return sycl::bit_cast<Float8>(biased)-Float8(8388736.f);
}
// The same exact signed-byte reconstruction used for K, now on four V codes.
// Keep this opt-in separate from the original convert path.
template<bool Optimize>inline sycl::float4 value_codes(sycl::vec<int8_t,4> codes){
 if constexpr(Optimize){
  using UInt4=uint32_t __attribute__((ext_vector_type(4)));
  const uint32_t bits=sycl::bit_cast<uint32_t>(codes);
  const UInt4 shifts{0,8,16,24};
  const UInt4 biased=((UInt4(bits)>>shifts)&UInt4(255))^UInt4(0x4b000080U);
  return sycl::bit_cast<sycl::float4>(biased)-sycl::float4(8388736.f);
 }else return codes.convert<float,sycl::rounding_mode::automatic>();
}
// One nontranspose TLA copy supplies the original paired eight-byte inputs.
// The pinned copy operation declares at least64 elements of operand storage;
// keep four uint64 registers per lane behind the two-element logical view.
inline void key16_tla(const QsaAttnPools&p,long long row,int lane,sycl::sub_group,float*out0,float*out16){
 using namespace cute;
 const auto*base=reinterpret_cast<const uint64_t*>(p.k_q+row*HD);
 auto K=make_tensor(make_gmem_ptr(base),make_shape(_4{},_8{}),make_stride(_8{},_1{}));
 auto cp=make_block_2d_copy(XE_LOAD_2D<64,4,8>{},K);
 auto ct=cp.get_slice(lane);auto coords=make_identity_tensor(K.shape());
 auto storage=make_tensor<uint64_t>(make_shape(_4{}));
 auto layout=make_fragment_like<uint64_t>(ct.partition_S(coords)).layout();
 auto codes=make_tensor(make_rmem_ptr(storage.data()),layout);
 copy(cp,ct.partition_S(coords),codes);
 const auto lo=exact_signed_bytes(codes(0));
 const auto hi=exact_signed_bytes(codes(1));
 const float sc0=h2f(p.k_scale[row*4+lane/8]),sc16=h2f(p.k_scale[row*4+lane/8+2]);
#pragma unroll
 for(int j=0;j<8;++j){out0[j]=lo[j]*sc0;out16[j]=hi[j]*sc16;}
}
// Retain the scalar dot rounding order.
// Explicit vector FMAs avoid loss of contraction in the vector expression.
inline sycl::float4 dot8_head4(const float q[4][8],const float*k){
 // The scalar dot rounds product1, then fuses product0 into it.
 sycl::float4 v=k[1]*sycl::float4(q[0][1],q[1][1],q[2][1],q[3][1]);
 v=sycl::fma(sycl::float4(k[0]),sycl::float4(q[0][0],q[1][0],q[2][0],q[3][0]),v);
#pragma unroll
 for(int j=2;j<8;++j)v=sycl::fma(sycl::float4(k[j]),sycl::float4(q[0][j],q[1][j],q[2][j],q[3][j]),v);
 return v;
}
// Each subgroup loads the chunk weights once, then broadcasts them to its
// independent output dimensions. Keep the original serial FMA order.
// This path is selected only for cap2051 (33 chunks).
inline void merge_subgroup(sycl::queue&queue,const float*scratch,float*out,int nq,int cap,QsaShapes sh){
 const int chunks=(cap+CH-1)/CH,heads=sh.n_head;
 const long long stride=qsa_decode_attn_scratch_floats(cap,sh);
 const float*pm=scratch+size_t(chunks)*heads*HD,*pl=pm+size_t(chunks)*heads;
 queue.parallel_for<MergeSubgroup>(sycl::nd_range<3>(sycl::range<3>(1,nq,size_t(heads)*128),sycl::range<3>(1,1,128)),
  [=](sycl::nd_item<3>it)[[sycl::reqd_sub_group_size(32)]]{
   const int qi=it.get_group(1),head=it.get_group(2),d=2*it.get_local_id(2),kv=head/G,h=head%G;
   const auto sg=it.get_sub_group();const int lane=sg.get_local_linear_id();
   const float*pa=scratch+size_t(qi)*stride,*ms=pm+size_t(qi)*stride,*ls=pl+size_t(qi)*stride;
   const int slot0=kv*chunks+lane,slot1=slot0+32;
   const float m0=lane<chunks?ms[slot0*G+h]:-FLT_MAX,m1=lane+32<chunks?ms[slot1*G+h]:-FLT_MAX;
   const float maximum=sycl::reduce_over_group(sg,sycl::fmax(m0,m1),sycl::maximum<float>{});
   const int valid0=m0!=-FLT_MAX,valid1=m1!=-FLT_MAX;
   const float w0=valid0?sycl::native::exp(m0-maximum):0.f,w1=valid1?sycl::native::exp(m1-maximum):0.f;
   const float l0=valid0?ls[slot0*G+h]:0.f,l1=valid1?ls[slot1*G+h]:0.f;
   float sum=0.f;sycl::float2 value(0.f);
   for(int c=0;c<chunks;++c){
    const int i=c&31;
    const float w=sycl::select_from_group(sg,c<32?w0:w1,i);
    const float l=sycl::select_from_group(sg,c<32?l0:l1,i);
    const int valid=sycl::select_from_group(sg,c<32?valid0:valid1,i);
    if(!valid)continue;
    sum=sycl::fma(l,w,sum);
    const auto v=*reinterpret_cast<const sycl::float2*>(pa+(size_t(kv*chunks+c)*G+h)*HD+d);
    value=sycl::fma(v,sycl::float2(w),value);
   }
   *reinterpret_cast<sycl::float2*>(out+(size_t(qi)*heads+head)*HD+d)=sum>0.f?value/sum:sycl::float2(0.f);
  });
}
template<bool Optimize>void launch(sycl::queue&queue,const float*q,QsaAttnPools p,const int32_t*ids,const int32_t*steps,
            int cap,QsaShapes sh,float*scratch,float*out,int nq,bool subgroup_merge=false){
 using namespace cute;
 const int chunks=(cap+CH-1)/CH,nkv=sh.n_head_kv,heads=sh.n_head,page=sh.page_size;
 const long long stride=qsa_decode_attn_scratch_floats(cap,sh);
 float*pm=scratch+size_t(chunks)*heads*HD,*pl=pm+size_t(chunks)*heads;
 using CoreType=std::conditional_t<Optimize,CorePV,Core>;
 queue.parallel_for<CoreType>(
  sycl::nd_range<3>(sycl::range<3>(nq,nkv,size_t(chunks)*Threads),sycl::range<3>(1,1,Threads)),
  [=](sycl::nd_item<3>it)[[sycl::reqd_sub_group_size(16)]]{
   const int qi=it.get_group(0),kv=it.get_group(1),chunk=it.get_group(2),t=it.get_local_id(2),lane=t&15,sgid=t>>4;
   const auto sg=it.get_sub_group();
   auto&sp=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][CH]>(it.get_group());
   auto&srow=*sycl::ext::oneapi::group_local_memory_for_overwrite<long long[CH]>(it.get_group());
   const int width=steps[size_t(qi)*kStepCount+kStepWidth],c0=chunk*CH,n_here=sycl::min(CH,width-c0),slot=kv*chunks+chunk;
   float*pa=scratch+size_t(qi)*stride,*ms=pm+size_t(qi)*stride,*ls=pl+size_t(qi)*stride;
   const float*query=q+(size_t(qi)*heads+kv*G)*HD;
   if(n_here<=0){if(t<G){ms[slot*G+t]=-FLT_MAX;ls[slot*G+t]=0.f;}return;}
   if(t<CH){long long r=-1;if(t<n_here){const int cell=ids[size_t(qi)*cap+c0+t];const int pg=p.page_table[cell/page];if(pg>=0)r=(static_cast<long long>(pg)*nkv+kv)*page+cell%page;}srow[t]=r;}
   const float down=1.f/16.f;
   it.barrier(sycl::access::fence_space::local_space);
    // Existing cute universal 128-bit vector copy, preserving each lane's
    // original eight FP32 dimensions. Keep TLA block key copy unchanged.
#pragma clang loop unroll(disable)
    for(int hg=0;hg<G;hg+=4){
     float q0[4][8],q16[4][8];
#pragma unroll
     for(int h=0;h<4;++h){
      auto src0=make_tensor(make_gmem_ptr(query+(hg+h)*HD+lane*8),make_shape(_8{}),make_stride(_1{}));
      auto src16=make_tensor(make_gmem_ptr(query+(hg+h)*HD+(lane+16)*8),make_shape(_8{}),make_stride(_1{}));
      auto dst0=make_tensor(make_rmem_ptr(&q0[h][0]),make_shape(_8{}),make_stride(_1{}));
      auto dst16=make_tensor(make_rmem_ptr(&q16[h][0]),make_shape(_8{}),make_stride(_1{}));
      copy(AutoVectorizingCopyWithAssumedAlignment<128>{},src0,dst0);
      copy(AutoVectorizingCopyWithAssumedAlignment<128>{},src16,dst16);
     }
     for(int c=sgid;c<CH;c+=Threads/16){
      if(c>=n_here||srow[c]<0){if(lane<4)sp[hg+lane][c]=-FLT_MAX;continue;}
      float k0[8],k16[8];key16_tla(p,srow[c],lane,sg,k0,k16);
      sycl::float4 v=dot8_head4(q0,k0)+dot8_head4(q16,k16);
      // Native reduction retains the lane-zero offset8/4/2/1 tree.
      v=sycl::reduce_over_group(sg,v,sycl::plus<sycl::float4>{});
      if(lane==0){
#pragma unroll
       for(int h=0;h<4;++h)sp[hg+h][c]=v[h]*down;
      }
     }
    }
   it.barrier(sycl::access::fence_space::local_space);
   // Fold original warp32 offset16 first, then reproduce offsets8/4/2/1.
   for(int h=sgid;h<G;h+=Threads/16){
    float m=sycl::fmax(sycl::fmax(sp[h][lane],sp[h][lane+32]),sycl::fmax(sp[h][lane+16],sp[h][lane+48]));
#pragma unroll
    for(int off=8;off>0;off>>=1)m=sycl::fmax(m,sycl::permute_group_by_xor(sg,m,off));
    const float ea=(lane<n_here&&srow[lane]>=0)?sycl::native::exp(sp[h][lane]-m):0.f;
    const float eb=(lane+32<n_here&&srow[lane+32]>=0)?sycl::native::exp(sp[h][lane+32]-m):0.f;
    const float ec=(lane+16<n_here&&srow[lane+16]>=0)?sycl::native::exp(sp[h][lane+16]-m):0.f;
    const float ed=(lane+48<n_here&&srow[lane+48]>=0)?sycl::native::exp(sp[h][lane+48]-m):0.f;
    sp[h][lane]=ea;sp[h][lane+32]=eb;sp[h][lane+16]=ec;sp[h][lane+48]=ed;
    float l=(ea+eb)+(ec+ed);
#pragma unroll
    for(int off=8;off>0;off>>=1)l+=sycl::permute_group_by_xor(sg,l,off);
    if(lane==0){ms[slot*G+h]=m;ls[slot*G+h]=l;}
   }
   it.barrier(sycl::access::fence_space::local_space);
   const int d=4*t;sycl::float4 acc[G];
#pragma unroll
   for(int h=0;h<G;++h)acc[h]=sycl::float4(0.f);
   for(int c=0;c<n_here;c+=8){long long rows[8];bool full=c+8<=n_here;
#pragma unroll
    for(int j=0;j<8;++j){rows[j]=c+j<n_here?srow[c+j]:-1;full=full&&rows[j]>=0;}
    if(full){sycl::float4 values[8];
#pragma unroll
     for(int j=0;j<8;++j){const auto codes=*reinterpret_cast<const sycl::vec<int8_t,4>*>(p.v_q+rows[j]*HD+d);
      values[j]=value_codes<Optimize>(codes)*h2f(p.v_scale[rows[j]*4+d/64]);}
#pragma unroll
     for(int h=0;h<G;++h){const auto p0=*reinterpret_cast<const sycl::float4*>(&sp[h][c]);const auto p1=*reinterpret_cast<const sycl::float4*>(&sp[h][c+4]);
      acc[h]=sycl::fma(sycl::float4(p0[0]),values[0],acc[h]);acc[h]=sycl::fma(sycl::float4(p0[1]),values[1],acc[h]);
      acc[h]=sycl::fma(sycl::float4(p0[2]),values[2],acc[h]);acc[h]=sycl::fma(sycl::float4(p0[3]),values[3],acc[h]);
      acc[h]=sycl::fma(sycl::float4(p1[0]),values[4],acc[h]);acc[h]=sycl::fma(sycl::float4(p1[1]),values[5],acc[h]);
      acc[h]=sycl::fma(sycl::float4(p1[2]),values[6],acc[h]);acc[h]=sycl::fma(sycl::float4(p1[3]),values[7],acc[h]);
     }
    }else{
#pragma unroll
     for(int j=0;j<8;++j){if(rows[j]<0)continue;
      const auto codes=*reinterpret_cast<const sycl::vec<int8_t,4>*>(p.v_q+rows[j]*HD+d);
      const auto value=value_codes<Optimize>(codes)*h2f(p.v_scale[rows[j]*4+d/64]);
#pragma unroll
      for(int h=0;h<G;++h)acc[h]=sycl::fma(sycl::float4(sp[h][c+j]),value,acc[h]);
     }
    }
   }
#pragma unroll
   for(int h=0;h<G;++h){
#pragma unroll
    for(int j=0;j<4;++j)pa[(size_t(slot)*G+h)*HD+d+j]=acc[h][j];}
  });
 if(subgroup_merge){merge_subgroup(queue,scratch,out,nq,cap,sh);return;}
 if constexpr(Optimize){
  // Two independent dimensions share max/exp/sum while keeping each output's
  // chunk order, FMA chain and division unchanged.
  queue.parallel_for<MergeVec2>(sycl::nd_range<3>(sycl::range<3>(1,nq,size_t(heads)*HD/2),sycl::range<3>(1,1,HD/2)),
   [=](sycl::nd_item<3>it)[[sycl::reqd_sub_group_size(32)]]{
    const int qi=it.get_group(1),head=it.get_group(2),d=2*it.get_local_id(2),kv=head/G,h=head%G;
    const float*pa=scratch+size_t(qi)*stride,*ms=pm+size_t(qi)*stride,*ls=pl+size_t(qi)*stride;
    float maximum=-FLT_MAX;
#pragma unroll
    for(int c=0;c<chunks;++c)maximum=sycl::fmax(maximum,ms[(kv*chunks+c)*G+h]);
    float sum=0.f;sycl::float2 value(0.f);
    for(int c=0;c<chunks;++c){const int slot=kv*chunks+c;const float m=ms[slot*G+h];if(m==-FLT_MAX)continue;
     const float w=sycl::native::exp(m-maximum);sum=sycl::fma(ls[slot*G+h],w,sum);
     const auto v=*reinterpret_cast<const sycl::float2*>(pa+(size_t(slot)*G+h)*HD+d);
     value=sycl::fma(v,sycl::float2(w),value);
    }
    *reinterpret_cast<sycl::float2*>(out+(size_t(qi)*heads+head)*HD+d)=sum>0.f?value/sum:sycl::float2(0.f);
   });
 }else{
 queue.parallel_for<Merge>(sycl::nd_range<3>(sycl::range<3>(1,nq,size_t(heads)*HD),sycl::range<3>(1,1,HD)),
  [=](sycl::nd_item<3>it)[[sycl::reqd_sub_group_size(32)]]{
   const int qi=it.get_group(1),head=it.get_group(2),d=it.get_local_id(2),kv=head/G,h=head%G;
   const float*pa=scratch+size_t(qi)*stride,*ms=pm+size_t(qi)*stride,*ls=pl+size_t(qi)*stride;
   float maximum=-FLT_MAX;
#pragma unroll
   for(int c=0;c<chunks;++c)maximum=sycl::fmax(maximum,ms[(kv*chunks+c)*G+h]);
   float sum=0.f,value=0.f;
   for(int c=0;c<chunks;++c){const int slot=kv*chunks+c;const float m=ms[slot*G+h];if(m==-FLT_MAX)continue;
    const float w=sycl::native::exp(m-maximum);sum=sycl::fma(ls[slot*G+h],w,sum);value=sycl::fma(pa[(size_t(slot)*G+h)*HD+d],w,value);}
   out[(size_t(qi)*heads+head)*HD+d]=sum>0.f?value/sum:0.f;
  });
 }
}
}
namespace strata::kernels::tla_prompt_resident {
using namespace tla_prompt_fp32;
template<int>class Core;
template<int>class Merge;

// Full chunks have 64 valid rows. Reuse four decoded keys while visiting the
// twelve query heads in pairs. The checked path still handles partial chunks.
inline void qk_reuse_four_keys(const QsaAttnPools& p, const long long* srow,
                               const float* query, float (&sp)[G][CH],
                               int lane, int subgroup_id, sycl::sub_group sg) {
 using namespace cute;
 constexpr float down = 1.f / 16.f;
     // Keep four decoded keys in registers. Each query fragment is reused for
     // these four cells before loading the next two heads.
     for(int first=subgroup_id; first<CH; first+=16){
      float lo[4][8],hi[4][8];
#pragma unroll
      for(int i=0;i<4;++i)key16_tla(p,srow[first+4*i],lane,sg,lo[i],hi[i]);
#pragma clang loop unroll(disable)
      for(int hg=0;hg<G;hg+=2){
       float qlo[2][8],qhi[2][8];
#pragma unroll
       for(int h=0;h<2;++h){
        auto copy_query=[&](int d,float* dst){
         auto src=make_tensor(make_gmem_ptr(query+(hg+h)*HD+d),make_shape(_8{}),make_stride(_1{}));
         auto reg=make_tensor(make_rmem_ptr(dst),make_shape(_8{}),make_stride(_1{}));
         copy(AutoVectorizingCopyWithAssumedAlignment<128>{},src,reg);
        };
        copy_query(lane*8,qlo[h]);copy_query((lane+16)*8,qhi[h]);
       }
#pragma unroll
       for(int i=0;i<4;++i){
        auto dot=[&](const float (&qv)[2][8],const float *k){
         sycl::float2 v=sycl::float2(qv[0][1],qv[1][1])*k[1];
         v=sycl::fma(sycl::float2(qv[0][0],qv[1][0]),sycl::float2(k[0]),v);
#pragma unroll
         for(int j=2;j<8;++j)v=sycl::fma(sycl::float2(qv[0][j],qv[1][j]),sycl::float2(k[j]),v);
         return v;
        };
        auto v=sycl::reduce_over_group(sg,dot(qlo,lo[i])+dot(qhi,hi[i]),sycl::plus<sycl::float2>{});
        if(lane==0){sp[hg][first+4*i]=v[0]*down;sp[hg+1][first+4*i]=v[1]*down;}
       }
      }
     }
}
template<int Mode>void launch(sycl::queue&queue,const float*q,QsaAttnPools p,const int32_t*ids,const int32_t*steps,
            int cap,QsaShapes sh,float*scratch,float*out,int nq){
 using namespace cute;
 constexpr bool scheduled = Mode == 5;
 const int chunks = scheduled ? 33 : (cap+CH-1)/CH;
 const int nkv = scheduled ? 2 : sh.n_head_kv, heads = scheduled ? 24 : sh.n_head;
 const int page = scheduled ? 4 : sh.page_size;
 const long long stride=qsa_decode_attn_scratch_floats(cap,sh);
 float*pm=scratch+size_t(chunks)*heads*HD,*pl=pm+size_t(chunks)*heads;
 using CoreType=Core<Mode>;
 queue.parallel_for<CoreType>(
  sycl::nd_range<3>((scheduled ? sycl::range<3>(1,1,1056*Threads) : sycl::range<3>(nq,nkv,size_t(chunks)*Threads)),sycl::range<3>(1,1,Threads)),
  [=](sycl::nd_item<3>it)[[sycl::reqd_sub_group_size(16)]]{
   int qi, kv, chunk;
   if constexpr (scheduled) {
    // Schedule 16 * 2 * 32 complete chunks before the 32 three-cell tails.
    const unsigned group = it.get_group(2);
    const unsigned pair = group < 1024 ? group / 32 : group - 1024;
    qi = pair / 2; kv = pair % 2; chunk = group < 1024 ? group % 32 : 32;
   } else {
    qi = it.get_group(0); kv = it.get_group(1); chunk = it.get_group(2);
   }
   const int t=it.get_local_id(2),lane=t&15,sgid=t>>4;
   const auto sg=it.get_sub_group();
   auto&sp=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[G][CH]>(it.get_group());
   auto&srow=*sycl::ext::oneapi::group_local_memory_for_overwrite<long long[CH]>(it.get_group());
   const int width=steps[size_t(qi)*kStepCount+kStepWidth],c0=chunk*CH,n_here=sycl::min(CH,width-c0),slot=kv*chunks+chunk;
   float*pa=scratch+size_t(qi)*stride,*ms=pm+size_t(qi)*stride,*ls=pl+size_t(qi)*stride;
   const float*query=q+(size_t(qi)*heads+kv*G)*HD;
   if(n_here<=0){if(t<G){ms[slot*G+t]=-FLT_MAX;ls[slot*G+t]=0.f;}return;}
   long long row_check=-1;
   if(t<CH){long long r=-1;if(t<n_here){const int cell=ids[size_t(qi)*(scheduled?2051:cap)+c0+t];const int pg=p.page_table[cell/page];if(pg>=0)r=(static_cast<long long>(pg)*nkv+kv)*page+cell%page;}srow[t]=r;row_check=r;}
   const float down=1.f/16.f;
   bool full_resident=false;
   if constexpr(Mode!=0){
    auto&flags=*sycl::ext::oneapi::group_local_memory_for_overwrite<int[4]>(it.get_group());
    const bool valid=sycl::all_of_group(sg,row_check>=0);
    if(lane==0)flags[sgid]=int(valid);
    it.barrier(sycl::access::fence_space::local_space);
    full_resident=n_here==CH&&flags[0]&&flags[1]&&flags[2]&&flags[3];
   }else it.barrier(sycl::access::fence_space::local_space);
   auto do_qk=[&](auto tag){constexpr bool All=decltype(tag)::value;
    // Existing cute universal 128-bit vector copy, preserving each lane's
    // original eight FP32 dimensions. Keep TLA block key copy unchanged.
#pragma clang loop unroll(disable)
    for(int hg=0;hg<G;hg+=4){
     float q0[4][8],q16[4][8];
#pragma unroll
     for(int h=0;h<4;++h){
      auto src0=make_tensor(make_gmem_ptr(query+(hg+h)*HD+lane*8),make_shape(_8{}),make_stride(_1{}));
      auto src16=make_tensor(make_gmem_ptr(query+(hg+h)*HD+(lane+16)*8),make_shape(_8{}),make_stride(_1{}));
      auto dst0=make_tensor(make_rmem_ptr(&q0[h][0]),make_shape(_8{}),make_stride(_1{}));
      auto dst16=make_tensor(make_rmem_ptr(&q16[h][0]),make_shape(_8{}),make_stride(_1{}));
      copy(AutoVectorizingCopyWithAssumedAlignment<128>{},src0,dst0);
      copy(AutoVectorizingCopyWithAssumedAlignment<128>{},src16,dst16);
     }
     for(int c=sgid;c<CH;c+=Threads/16){
      if constexpr(!All){if(c>=n_here||srow[c]<0){if(lane<4)sp[hg+lane][c]=-FLT_MAX;continue;}}
      float k0[8],k16[8];key16_tla(p,srow[c],lane,sg,k0,k16);
      sycl::float4 v=dot8_head4(q0,k0)+dot8_head4(q16,k16);
      // Native reduction retains the lane-zero offset8/4/2/1 tree.
      v=sycl::reduce_over_group(sg,v,sycl::plus<sycl::float4>{});
      if(lane==0){
#pragma unroll
       for(int h=0;h<4;++h)sp[hg+h][c]=v[h]*down;
      }
     }
    }
   };
   if constexpr(Mode==4){
    if(full_resident)qk_reuse_four_keys(p,srow,query,sp,lane,sgid,sg);
    else do_qk(std::false_type{});
   }else if constexpr(Mode!=0){if(full_resident)do_qk(std::true_type{});else do_qk(std::false_type{});}
   else do_qk(std::false_type{});
   it.barrier(sycl::access::fence_space::local_space);
   // Fold original warp32 offset16 first, then reproduce offsets8/4/2/1.
   for(int h=sgid;h<G;h+=Threads/16){
    float m=sycl::fmax(sycl::fmax(sp[h][lane],sp[h][lane+32]),sycl::fmax(sp[h][lane+16],sp[h][lane+48]));
#pragma unroll
    for(int off=8;off>0;off>>=1)m=sycl::fmax(m,sycl::permute_group_by_xor(sg,m,off));
    const float ea=(lane<n_here&&srow[lane]>=0)?sycl::native::exp(sp[h][lane]-m):0.f;
    const float eb=(lane+32<n_here&&srow[lane+32]>=0)?sycl::native::exp(sp[h][lane+32]-m):0.f;
    const float ec=(lane+16<n_here&&srow[lane+16]>=0)?sycl::native::exp(sp[h][lane+16]-m):0.f;
    const float ed=(lane+48<n_here&&srow[lane+48]>=0)?sycl::native::exp(sp[h][lane+48]-m):0.f;
    sp[h][lane]=ea;sp[h][lane+32]=eb;sp[h][lane+16]=ec;sp[h][lane+48]=ed;
    float l=(ea+eb)+(ec+ed);
#pragma unroll
    for(int off=8;off>0;off>>=1)l+=sycl::permute_group_by_xor(sg,l,off);
    if(lane==0){ms[slot*G+h]=m;ls[slot*G+h]=l;}
   }
   it.barrier(sycl::access::fence_space::local_space);
   auto do_pv=[&](auto tag){constexpr bool All=decltype(tag)::value;
   const int d=4*t;sycl::float4 acc[G];
#pragma unroll
   for(int h=0;h<G;++h)acc[h]=sycl::float4(0.f);
   for(int c=0;c<(All?CH:n_here);c+=8){long long rows[8];bool full=All||c+8<=n_here;
#pragma unroll
    for(int j=0;j<8;++j){if constexpr(All)rows[j]=srow[c+j];else{rows[j]=c+j<n_here?srow[c+j]:-1;full=full&&rows[j]>=0;}}
    if(full){sycl::float4 values[8];
#pragma unroll
     for(int j=0;j<8;++j){const auto codes=*reinterpret_cast<const sycl::vec<int8_t,4>*>(p.v_q+rows[j]*HD+d);
      values[j]=value_codes<false>(codes)*h2f(p.v_scale[rows[j]*4+d/64]);}
#pragma unroll
     for(int h=0;h<G;++h){const auto p0=*reinterpret_cast<const sycl::float4*>(&sp[h][c]);const auto p1=*reinterpret_cast<const sycl::float4*>(&sp[h][c+4]);
      acc[h]=sycl::fma(sycl::float4(p0[0]),values[0],acc[h]);acc[h]=sycl::fma(sycl::float4(p0[1]),values[1],acc[h]);
      acc[h]=sycl::fma(sycl::float4(p0[2]),values[2],acc[h]);acc[h]=sycl::fma(sycl::float4(p0[3]),values[3],acc[h]);
      acc[h]=sycl::fma(sycl::float4(p1[0]),values[4],acc[h]);acc[h]=sycl::fma(sycl::float4(p1[1]),values[5],acc[h]);
      acc[h]=sycl::fma(sycl::float4(p1[2]),values[6],acc[h]);acc[h]=sycl::fma(sycl::float4(p1[3]),values[7],acc[h]);
     }
    }else{
#pragma unroll
     for(int j=0;j<8;++j){if(rows[j]<0)continue;
      const auto codes=*reinterpret_cast<const sycl::vec<int8_t,4>*>(p.v_q+rows[j]*HD+d);
      const auto value=value_codes<false>(codes)*h2f(p.v_scale[rows[j]*4+d/64]);
#pragma unroll
      for(int h=0;h<G;++h)acc[h]=sycl::fma(sycl::float4(sp[h][c+j]),value,acc[h]);
     }
    }
   }
#pragma unroll
   for(int h=0;h<G;++h){
#pragma unroll
    for(int j=0;j<4;++j)pa[(size_t(slot)*G+h)*HD+d+j]=acc[h][j];}
   };
   if constexpr(Mode==2||Mode==4||scheduled){if(full_resident)do_pv(std::true_type{});else do_pv(std::false_type{});}
   else do_pv(std::false_type{});

  });
 if constexpr (scheduled) {
  // A workgroup owns one output head. Its first subgroup computes rescaling
  // weights once; every dimension then folds the chunks in their original order.
  constexpr int chunk_count = 33;
  queue.parallel_for<Merge<Mode>>(
   sycl::nd_range<3>({size_t(nq), size_t(heads), HD}, {1, 1, HD}),
   [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(32)]] {
    const int qi = it.get_group(0), head = it.get_group(1), dim = it.get_local_id(2);
    const int head_slot = (head / G) * chunk_count * G + head % G;
    const size_t query_offset = size_t(qi) * stride;
    const float* maxima = pm + query_offset + head_slot;
    const float* sums = pl + query_offset + head_slot;
    const float* partial = scratch + query_offset + head_slot * HD + dim;
    auto group = it.get_group();
    auto& weights = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[chunk_count]>(group);
    auto& lengths = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[chunk_count]>(group);
    auto& valid = *sycl::ext::oneapi::group_local_memory_for_overwrite<unsigned[2]>(group);
    if (dim < 32) {
     const auto sg = it.get_sub_group();
     const float m = maxima[dim * G], tail = maxima[32 * G];
     const float maximum = sycl::reduce_over_group(
      sg, sycl::fmax(-FLT_MAX, sycl::fmax(m, tail)), sycl::maximum<float>());
     weights[dim] = m != -FLT_MAX ? sycl::native::exp(m - maximum) : 0.f;
     lengths[dim] = sums[dim * G];
     const unsigned bits = sycl::reduce_over_group(
      sg, m != -FLT_MAX ? (1u << dim) : 0u, sycl::bit_or<unsigned>());
     if (dim == 0) {
      valid[0] = bits;
      valid[1] = unsigned(tail != -FLT_MAX);
      weights[32] = tail != -FLT_MAX ? sycl::native::exp(tail - maximum) : 0.f;
      lengths[32] = sums[32 * G];
     }
    }
    it.barrier(sycl::access::fence_space::local_space);
    float sum = 0.f, value = 0.f;
    auto accumulate = [&](auto all_valid) {
#pragma unroll 4
     for (int c = 0; c < chunk_count; ++c) {
      if constexpr (!decltype(all_valid)::value) {
       if (!(valid[c / 32] & (1u << (c % 32)))) continue;
      }
      sum = sycl::fma(lengths[c], weights[c], sum);
      value = sycl::fma(partial[c * G * HD], weights[c], value);
     }
    };
    if (valid[0] == 0xffffffffu && valid[1]) accumulate(std::true_type{});
    else accumulate(std::false_type{});
    out[(size_t(qi) * heads + head) * HD + dim] = sum > 0.f ? value / sum : 0.f;
   });
 } else {
 queue.parallel_for<Merge<Mode>>(sycl::nd_range<3>(sycl::range<3>(1,nq,size_t(heads)*HD),sycl::range<3>(1,1,HD)),
  [=](sycl::nd_item<3>it)[[sycl::reqd_sub_group_size(32)]]{
   const int qi=it.get_group(1),head=it.get_group(2),d=it.get_local_id(2),kv=head/G,h=head%G;
   const float*pa=scratch+size_t(qi)*stride,*ms=pm+size_t(qi)*stride,*ls=pl+size_t(qi)*stride;
   float maximum=-FLT_MAX;
#pragma unroll
   for(int c=0;c<chunks;++c)maximum=sycl::fmax(maximum,ms[(kv*chunks+c)*G+h]);
   float sum=0.f,value=0.f;
   for(int c=0;c<chunks;++c){const int slot=kv*chunks+c;const float m=ms[slot*G+h];if(m==-FLT_MAX)continue;
    const float w=sycl::native::exp(m-maximum);sum=sycl::fma(ls[slot*G+h],w,sum);value=sycl::fma(pa[(size_t(slot)*G+h)*HD+d],w,value);}
   out[(size_t(qi)*heads+head)*HD+d]=sum>0.f?value/sum:0.f;
  });
 }
}
}

namespace strata::kernels {
bool qsa_prompt_attn_tla_batch(const float*q,const QsaAttnPools&p,const int32_t*ids,const int32_t*steps,int64_t cap,
 const QsaShapes&sh,float*scratch,float*out,int64_t nq,void*stream){
 if(!q||!ids||!steps||!scratch||!out||!p.k_q||!p.v_q||!p.k_scale||!p.v_scale||!p.page_table||p.k_q4||p.v_q4||
    nq<=0||nq>32||cap<=0||cap>32768||sh.head_dim!=256||sh.n_head_kv!=2||sh.n_head!=24||sh.page_size<=0)return false;
 if((reinterpret_cast<uintptr_t>(p.v_q)&3)||(reinterpret_cast<uintptr_t>(q)&63)||(reinterpret_cast<uintptr_t>(p.k_q)&63))return false;
 auto*queue=strata::q_of(stream);
#ifdef STRATA_HAVE_ZE
 if(queue->get_backend()!=sycl::backend::ext_oneapi_level_zero)return false;
 const auto device=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue->get_device());
 static thread_local ze_device_handle_t cached_device=nullptr;
 static thread_local bool supported=false;
 if(device!=cached_device){
  ze_device_properties_t props{};props.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  supported=zeDeviceGetProperties(device,&props)==ZE_RESULT_SUCCESS&&props.vendorId==0x8086&&props.deviceId==0xe223;
  cached_device=device;
 }
 if(!supported)return false;
#else
 return false;
#endif
 static const bool pv_vec2=[] {const char*v=std::getenv("STRATA_SYCL_TLA_PV_VEC2");return v&&v[0]=='1';}();
 static const bool merge_sg=[] {const char*v=std::getenv("STRATA_SYCL_TLA_MERGE_SG");return v&&v[0]=='1';}();
 static const bool resident=[] {const char*v=std::getenv("STRATA_SYCL_TLA_RESIDENT");return v&&v[0]=='1';}();
 const bool subgroup_merge=merge_sg&&nq==16&&cap==2051;
 if(resident&&nq==16&&cap==2051&&!pv_vec2&&!subgroup_merge){
  static const bool key_reuse=[] {const char* v=std::getenv("STRATA_SYCL_TLA_KEY_REUSE");return v&&v[0]=='1';}();
  static const bool schedule=[] {const char* v=std::getenv("STRATA_SYCL_TLA_SCHEDULE");return v&&v[0]=='1';}();
  if(schedule&&!key_reuse&&sh.page_size==4){
   tla_prompt_resident::launch<5>(*queue,q,p,ids,steps,int(cap),sh,scratch,out,int(nq));
   static thread_local bool announced_schedule=false;
   if(!announced_schedule){std::fprintf(stderr,"strata: using SYCL TLA scheduled prompt attention\n");announced_schedule=true;}
  }else if(key_reuse){
   tla_prompt_resident::launch<4>(*queue,q,p,ids,steps,int(cap),sh,scratch,out,int(nq));
   static thread_local bool announced=false;
   if(!announced){std::fprintf(stderr,"strata: using SYCL TLA four-key reuse\n");announced=true;}
  }else tla_prompt_resident::launch<2>(*queue,q,p,ids,steps,int(cap),sh,scratch,out,int(nq));
  static thread_local bool announced_resident=false;
  if(!announced_resident){std::fprintf(stderr,"strata: using SYCL TLA resident-chunk prompt attention\n");announced_resident=true;}
 }else if(pv_vec2&&nq==16&&cap==2051){
  tla_prompt_fp32::launch<true>(*queue,q,p,ids,steps,int(cap),sh,scratch,out,int(nq),subgroup_merge);
  static thread_local bool announced_pv=false;
  if(!announced_pv){std::fprintf(stderr,"strata: using SYCL TLA exact PV bytes and two-dimension merge\n");announced_pv=true;}
 }else tla_prompt_fp32::launch<false>(*queue,q,p,ids,steps,int(cap),sh,scratch,out,int(nq),subgroup_merge);
 if(subgroup_merge){
  static thread_local bool announced_sg=false;
  if(!announced_sg){std::fprintf(stderr,"strata: using SYCL TLA subgroup-shared prompt merge\n");announced_sg=true;}
 }
 static thread_local bool announced=false;
 if(!announced){std::fprintf(stderr,"strata: using SYCL TLA FP32 prompt attention\n");announced=true;}
 return true;
}
}
