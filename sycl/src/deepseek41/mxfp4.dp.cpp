// SYCL counterpart of src/deepseek41/mxfp4.cu from mmnga/Strata:deepseek.
// Keep the existing family API and its 32-lane FMA/reduction order.
#include "strata/deepseek41/kernels.hpp"
#include "strata/deepseek41/mxfp4.hpp"
#include <sycl/sycl.hpp>
#include "strata/deepseek41/iq_weights.hpp"
#include <array>
#include <algorithm>
#include <stdexcept>

namespace strata::deepseek41::cuda {
namespace {
template<bool Alu> float weight(const uint8_t* row, int i) {
    const uint8_t* block = row + size_t(i / 32) * 17;
    const int q = (block[1 + i % 16] >> ((i % 32) / 16 * 4)) & 15;
    const uint32_t scale = block[0] < 2 ? 0x00200000u << block[0] : uint32_t(block[0] - 1) << 23;
    if constexpr (Alu) {
        const int m = q & 7;
        const int value = m < 4 ? m : (2 + (m & 1)) << ((m >> 1) - 1);
        return float((q & 8) ? -value : value) * sycl::bit_cast<float>(scale);
    } else {
        constexpr int8_t fp4[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
        return float(fp4[q]) * sycl::bit_cast<float>(scale);
    }
}

template<int Type> float expert_weight(const uint8_t* row,int i) {
    if constexpr(Type==39) return weight<false>(row,i);
    else return iq::weight<Type>(row,i);
}

float reduce(float v, sycl::sub_group sg) {
    for (int offset = 16; offset; offset >>= 1) {
        const float other = sycl::shift_group_left(sg, v, offset);
        if (sg.get_local_linear_id() + offset < 32) v += other;
    }
    return v;
}

sycl::queue& checked(void* stream, int input, int output, int tokens) {
    if (!stream || input < 32 || input % 32 || output < 1 || tokens < 1 || tokens > max_prefill_batch)
        throw std::invalid_argument("DeepSeek MXFP4 geometry");
    return *static_cast<sycl::queue*>(stream);
}

template<bool Alu> void mmv(sycl::queue& q, const uint8_t* w, const float* x, float* y,
                           int input, int output, int tokens) {
    q.parallel_for(sycl::nd_range<2>({size_t(tokens), ((size_t(output)+3)/4)*128}, {1,128}),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
            const int row = item.get_group(1)*4 + item.get_local_id(1)/32;
            if (row >= output) return;
            const int token = item.get_group(0), lane = item.get_local_id(1)%32;
            const auto* wr = w + size_t(row)*(input/32)*17;
            const auto* xr = x + size_t(token)*input;
            float v = 0;
            for (int i = lane; i < input; i += 32) v = sycl::fma(weight<Alu>(wr,i),xr[i],v);
            v = reduce(v,item.get_sub_group());
            if (!lane) y[size_t(token)*output+row] = v;
        });
}

template<bool Alu,bool Grouped=false> void gu(sycl::queue& q, const uint8_t* gate, const uint8_t* up,
    const float* x, const int32_t* ids, float* hidden, int input, int ff, int k, int tokens, float limit,
    std::array<int32_t,max_prefill_batch> destinations={},int top_k=1) {
    q.parallel_for(sycl::nd_range<2>({size_t(k*tokens), ((size_t(ff)+3)/4)*128}, {1,128}),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
            const int row = item.get_group(1)*4 + item.get_local_id(1)/32;
            if (row >= ff) return;
            const int entry = item.get_group(0), lane = item.get_local_id(1)%32;
            const size_t expert = Grouped ? 0 : size_t(ids[entry]);
            const size_t token = Grouped ? size_t(destinations[entry]/top_k) : size_t(entry/k);
            const size_t offset = (expert*ff+row)*(input/32)*17;
            const float* xr = x + token*input;
            float g = 0, u = 0;
            for (int i = lane; i < input; i += 32) {
                g = sycl::fma(weight<Alu>(gate+offset,i),xr[i],g);
                u = sycl::fma(weight<Alu>(up+offset,i),xr[i],u);
            }
            g = reduce(g,item.get_sub_group()); u = reduce(u,item.get_sub_group());
            if (!lane) {
                g = sycl::fmin(g,limit); u = sycl::fmax(-limit,sycl::fmin(u,limit));
                hidden[size_t(entry)*ff+row] = (g/(1+sycl::exp(-g)))*u;
            }
        });
}

template<bool Alu,bool Grouped=false> void down(sycl::queue& q, const uint8_t* w, const float* hidden,
    const int32_t* ids, float* out, int ff, int output, int k, int tokens,
    std::array<int32_t,max_prefill_batch> destinations={}) {
    q.parallel_for(sycl::nd_range<2>({size_t(k*tokens), ((size_t(output)+3)/4)*128}, {1,128}),
        [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
            const int row = item.get_group(1)*4 + item.get_local_id(1)/32;
            if (row >= output) return;
            const int entry = item.get_group(0), lane = item.get_local_id(1)%32;
            const size_t expert = Grouped ? 0 : size_t(ids[entry]);
            const size_t destination = Grouped ? size_t(destinations[entry]) : size_t(entry);
            const uint8_t* wr = w + (expert*output+row)*(ff/32)*17;
            const float* hr = hidden + size_t(entry)*ff;
            float v = 0;
            for (int i = lane; i < ff; i += 32) v = sycl::fma(weight<Alu>(wr,i),hr[i],v);
            v = reduce(v,item.get_sub_group());
            if (!lane) out[destination*output+row] = v;
        });
}

// An IQ2_XXS block has 32 packed 16-bit payload words. Read them once
// across the subgroup, then distribute the original index/sign metadata from
// registers. This keeps the lane+32*k dot order and every decoder rounding.
template<bool Packed=false>
inline uint32_t iq2xxs_payload(const uint8_t* block,int lane) {
 if constexpr(Packed)return lane<16?iq::u32(block+2+lane*4):0;
 else return iq::u16(block+2+lane*2);
}
template<bool Packed=false>
inline float iq2xxs_group_weight(sycl::sub_group sg, uint32_t payload, int at, float d) {
    const int ib=at/32, lane=at%32, l=lane/8, j=lane%8;
    uint32_t indices,meta;int index;
    if constexpr(Packed) {
        indices=sycl::select_from_group(sg,payload,ib*2);
        meta=sycl::select_from_group(sg,payload,ib*2+1);
        index=(indices>>(8*l))&255;
    }else {
        indices=sycl::select_from_group(sg,payload,ib*4+l/2);
        meta=sycl::select_from_group(sg,payload,ib*4+2) |
            (sycl::select_from_group(sg,payload,ib*4+3)<<16);
        index=(indices>>(8*(l%2)))&255;
    }
    const uint8_t signs=iq::iq2_signs(meta>>(7*l));
    const float db=d*(0.5f+float(meta>>28))*0.25f;
    const float magnitude=db*float(iq::codebook_byte(iq2xxs_grid,index,j));
    return (signs&(1<<j)) ? -magnitude : magnitude;
}

// Single-token layers need one accumulator and one destination per row.
// Keep the general grouped kernels intact for multi-token prefill.
template<int Type,bool Packed=false> void single_gu(sycl::queue& q,const uint8_t* gate,const uint8_t* up,const float* x,float* hidden,
    int input,int ff,int token,float limit){
    const size_t stride=iq::row_bytes(Type,input);const float* in=x+size_t(token)*input;
    q.parallel_for(sycl::nd_range<1>({size_t((ff+3)/4)*128},{128}),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const int row=int(item.get_group(0))*4+item.get_local_id(0)/32,lane=item.get_local_id(0)%32;
        if(row>=ff)return;
        const auto* g=gate+size_t(row)*stride;const auto* u=up+size_t(row)*stride;
        float ag=0,au=0;
        for(int base=0;base<input;base+=256){
            const auto* gb=g+size_t(base/256)*iq::block_bytes<Type>;
            const auto* ub=u+size_t(base/256)*iq::block_bytes<Type>;
            const float gd=iq::block_scale<Type>(gb),ud=iq::block_scale<Type>(ub);
            uint32_t gp=0,up=0;
            if constexpr(Type==16){gp=iq2xxs_payload<Packed>(gb,lane);up=iq2xxs_payload<Packed>(ub,lane);}
#pragma unroll
            for(int ib=0;ib<8;++ib){
                const int at=ib*32+lane;float gw,uw;
                if constexpr(Type==16){gw=iq2xxs_group_weight<Packed>(item.get_sub_group(),gp,at,gd);uw=iq2xxs_group_weight<Packed>(item.get_sub_group(),up,at,ud);}
                else {gw=iq::block_weight<Type>(gb,at,gd);uw=iq::block_weight<Type>(ub,at,ud);}
                const float v=in[base+at];ag=sycl::fma(gw,v,ag);au=sycl::fma(uw,v,au);
            }
        }
        const float gv=sycl::fmin(reduce(ag,item.get_sub_group()),limit);
        const float uv=sycl::fmax(-limit,sycl::fmin(reduce(au,item.get_sub_group()),limit));
        if(!lane)hidden[row]=(gv/(1+sycl::exp(-gv)))*uv;
    });
}
template<int Type,bool Packed=false> void single_down(sycl::queue& q,const uint8_t* weights,const float* hidden,float* parts,
    int ff,int output,int destination){
    const size_t stride=iq::row_bytes(Type,ff);
    q.parallel_for(sycl::nd_range<1>({size_t((output+3)/4)*128},{128}),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const int row=int(item.get_group(0))*4+item.get_local_id(0)/32,lane=item.get_local_id(0)%32;
        if(row>=output)return;
        const auto* w=weights+size_t(row)*stride;float acc=0;
        for(int base=0;base<ff;base+=256){
            const auto* b=w+size_t(base/256)*iq::block_bytes<Type>;const float d=iq::block_scale<Type>(b);
            uint32_t payload=0;if constexpr(Type==16)payload=iq2xxs_payload<Packed>(b,lane);
#pragma unroll
            for(int ib=0;ib<8;++ib){
                const int at=ib*32+lane;float value;
                if constexpr(Type==16)value=iq2xxs_group_weight<Packed>(item.get_sub_group(),payload,at,d);
                else value=iq::block_weight<Type>(b,at,d);
                acc=sycl::fma(value,hidden[base+at],acc);
            }
        }
        const float v=reduce(acc,item.get_sub_group());if(!lane)parts[size_t(destination)*output+row]=v;
    });
}

// A four-token matrix tile shares every decoded weight while preserving each
// token's lane+32*k FMA recurrence and the original 32-lane reduction tree.
template<int Type=39,int Tile=4,bool Packed=false,int ActiveRows=0> void grouped_gu(sycl::queue& q,const uint8_t* gate,const uint8_t* up,const float* x,float* hidden,
    int input,int ff,int top_k,int runtime_count,float limit,std::array<int32_t,ActiveRows?ActiveRows:max_prefill_batch> rows){
    static_assert(ActiveRows==0 || (ActiveRows>=2 && ActiveRows<=Tile));
    const int count=ActiveRows?ActiveRows:runtime_count;
    static_assert(Tile==4 || Tile==8 || Tile==16);
    const size_t stride=iq::row_bytes(Type,input);
    q.parallel_for(sycl::nd_range<2>({size_t((count+Tile-1)/Tile),size_t((ff+3)/4)*128},{1,128}),
      [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
        const int row=int(item.get_group(1))*4+item.get_local_id(1)/32,lane=item.get_local_id(1)%32;
        if(row>=ff)return;
        const int first=ActiveRows?0:int(item.get_group(0))*Tile;
        const auto* g=gate+size_t(row)*stride;const auto* u=up+size_t(row)*stride;
        const float* inputs[Tile]={};
#pragma unroll
        for(int t=0;t<Tile;++t)if(first+t<count)
            inputs[t]=x+size_t(rows[first+t]/top_k)*input;
        float ag[Tile]={},au[Tile]={};
        auto accumulate=[&](int i,float gw,float uw){
#pragma unroll
            for(int t=0;t<Tile;++t)if(first+t<count){
                const float v=inputs[t][i];
                ag[t]=sycl::fma(gw,v,ag[t]);au[t]=sycl::fma(uw,v,au[t]);
            }
        };
        if constexpr(Type==39){
            for(int i=lane;i<input;i+=32)accumulate(i,expert_weight<Type>(g,i),expert_weight<Type>(u,i));
        }else{
            for(int base=0;base<input;base+=256){
                const auto* gb=g+size_t(base/256)*iq::block_bytes<Type>;
                const auto* ub=u+size_t(base/256)*iq::block_bytes<Type>;
                const float gd=iq::block_scale<Type>(gb),ud=iq::block_scale<Type>(ub);
                uint32_t gp=0,up=0;
                if constexpr(Type==16){gp=iq2xxs_payload<Packed>(gb,lane);up=iq2xxs_payload<Packed>(ub,lane);}
#pragma unroll
                for(int ib=0;ib<8;++ib){
                    const int at=ib*32+lane;
                    if constexpr(Type==16)
                        accumulate(base+at,iq2xxs_group_weight<Packed>(item.get_sub_group(),gp,at,gd),
                                           iq2xxs_group_weight<Packed>(item.get_sub_group(),up,at,ud));
                    else accumulate(base+at,iq::block_weight<Type>(gb,at,gd),iq::block_weight<Type>(ub,at,ud));
                }
            }
        }
#pragma unroll
        for(int t=0;t<Tile;++t)if(first+t<count){
            float gv=reduce(ag[t],item.get_sub_group()),uv=reduce(au[t],item.get_sub_group());
            if(!lane){gv=sycl::fmin(gv,limit);uv=sycl::fmax(-limit,sycl::fmin(uv,limit));
                hidden[size_t(first+t)*ff+row]=(gv/(1+sycl::exp(-gv)))*uv;}
        }
    });
}
// Keep each original little-endian metadata halfword a separate read. The
// packed alias preserves the byte-alignment contract of the GGUF row.
struct __attribute__((packed,may_alias)) IQ1Metadata16 { uint16_t value; };
static_assert(sizeof(IQ1Metadata16)==2 && alignof(IQ1Metadata16)==1);
inline float iq1_weight_narrow(const uint8_t* b,int at,float d){
    const int ib=at/32,l=(at%32)/8,j=at%8;
    const uint8_t* p=b+34+ib*2;
    const uint16_t h=std::endian::native==std::endian::little
        ?reinterpret_cast<const volatile IQ1Metadata16*>(p)->value:iq::u16(p);
    const int index=b[2+ib*4+l] | (((h>>(3*l))&7)<<8);
    const int grid=int(int8_t(iq::codebook_byte(iq1s_grid,index,j)));
    const float dl=d*float(2*((h>>12)&7)+1);
    return dl*(float(grid)+((h&0x8000)?-0.125f:0.125f));
}

template<int Tile,int ActiveRows=0> void iq1s_grouped_gu_narrow(sycl::queue& q,const uint8_t* gate,const uint8_t* up,const float* x,float* hidden,
    int input,int ff,int top_k,int runtime_count,float limit,std::array<int32_t,ActiveRows?ActiveRows:max_prefill_batch> rows){
    static_assert(ActiveRows==0 || (ActiveRows>=1 && ActiveRows<=Tile));
    const int count=ActiveRows?ActiveRows:runtime_count;
    static_assert(Tile==1 || Tile==4 || Tile==8 || Tile==16);
    const size_t stride=iq::row_bytes(19,input);
    q.parallel_for(sycl::nd_range<2>({size_t((count+Tile-1)/Tile),size_t((ff+3)/4)*128},{1,128}),
      [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
        const int row=int(item.get_group(1))*4+item.get_local_id(1)/32,lane=item.get_local_id(1)%32;
        if(row>=ff)return;
        const int first=ActiveRows?0:int(item.get_group(0))*Tile;
        const auto* g=gate+size_t(row)*stride;const auto* u=up+size_t(row)*stride;
        const float* inputs[Tile]={};
#pragma unroll
        for(int t=0;t<Tile;++t)if(first+t<count)
            inputs[t]=x+size_t(rows[first+t]/top_k)*input;
        float ag[Tile]={},au[Tile]={};
        auto accumulate=[&](int i,float gw,float uw){
#pragma unroll
            for(int t=0;t<Tile;++t)if(first+t<count){
                const float v=inputs[t][i];
                ag[t]=sycl::fma(gw,v,ag[t]);au[t]=sycl::fma(uw,v,au[t]);
            }
        };
        for(int base=0;base<input;base+=256){
            const auto* gb=g+size_t(base/256)*iq::block_bytes<19>;
            const auto* ub=u+size_t(base/256)*iq::block_bytes<19>;
            const float gd=iq::block_scale<19>(gb),ud=iq::block_scale<19>(ub);
#pragma unroll
            for(int ib=0;ib<8;ib+=2){
                const int at0=ib*32+lane,at1=at0+32;
                const float gw0=iq1_weight_narrow(gb,at0,gd);
                const float uw0=iq1_weight_narrow(ub,at0,ud);
                const float gw1=iq1_weight_narrow(gb,at1,gd);
                const float uw1=iq1_weight_narrow(ub,at1,ud);
                accumulate(base+at0,gw0,uw0);
                accumulate(base+at1,gw1,uw1);
            }
        }
#pragma unroll
        for(int t=0;t<Tile;++t)if(first+t<count){
            float gv=reduce(ag[t],item.get_sub_group()),uv=reduce(au[t],item.get_sub_group());
            if(!lane){gv=sycl::fmin(gv,limit);uv=sycl::fmax(-limit,sycl::fmin(uv,limit));
                hidden[size_t(first+t)*ff+row]=(gv/(1+sycl::exp(-gv)))*uv;}
        }
    });
}
template<int Type=39,int Tile=4,bool Packed=false,int ActiveRows=0> void grouped_down(sycl::queue& q,const uint8_t* weights,const float* hidden,float* parts,
    int ff,int output,int runtime_count,std::array<int32_t,ActiveRows?ActiveRows:max_prefill_batch> rows){
    static_assert(ActiveRows==0 || (ActiveRows>=2 && ActiveRows<=Tile));
    const int count=ActiveRows?ActiveRows:runtime_count;
    static_assert(Tile==4 || Tile==8 || Tile==16);
    const size_t stride=iq::row_bytes(Type,ff);
    q.parallel_for(sycl::nd_range<2>({size_t((count+Tile-1)/Tile),size_t((output+3)/4)*128},{1,128}),
      [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
        const int row=int(item.get_group(1))*4+item.get_local_id(1)/32,lane=item.get_local_id(1)%32;
        if(row>=output)return;
        const int first=ActiveRows?0:int(item.get_group(0))*Tile;const auto* w=weights+size_t(row)*stride;
        float acc[Tile]={};
        auto accumulate=[&](int i,float value){
#pragma unroll
            for(int t=0;t<Tile;++t)if(first+t<count)acc[t]=sycl::fma(value,hidden[size_t(first+t)*ff+i],acc[t]);
        };
        if constexpr(Type==39){
            for(int i=lane;i<ff;i+=32)accumulate(i,expert_weight<Type>(w,i));
        }else{
            for(int base=0;base<ff;base+=256){
                const auto* b=w+size_t(base/256)*iq::block_bytes<Type>;
                const float d=iq::block_scale<Type>(b);
                uint32_t payload=0;
                if constexpr(Type==16)payload=iq2xxs_payload<Packed>(b,lane);
#pragma unroll
                for(int ib=0;ib<8;++ib){
                    const int at=ib*32+lane;
                    if constexpr(Type==16)accumulate(base+at,iq2xxs_group_weight<Packed>(item.get_sub_group(),payload,at,d));
                    else accumulate(base+at,iq::block_weight<Type>(b,at,d));
                }
            }
        }
#pragma unroll
        for(int t=0;t<Tile;++t)if(first+t<count){
            const float v=reduce(acc[t],item.get_sub_group());
            if(!lane)parts[size_t(rows[first+t])*output+row]=v;
        }
    });
}
}

void mxfp4_mmv(const void* w, const float* x, float* y, int input, int output, int tokens, void* stream, bool alu) {
    auto& q = checked(stream,input,output,tokens);
    if (alu) mmv<true>(q,static_cast<const uint8_t*>(w),x,y,input,output,tokens);
    else mmv<false>(q,static_cast<const uint8_t*>(w),x,y,input,output,tokens);
}
void mxfp4_gu(const void* g, const void* u, const float* x, const int32_t* ids, float* hidden,
              int input, int ff, int k, int tokens, float limit, void* stream, bool alu) {
    if (k < 1 || k > 64) throw std::invalid_argument("DeepSeek MXFP4 routing");
    auto& q = checked(stream,input,ff,tokens);
    if (alu) gu<true>(q,static_cast<const uint8_t*>(g),static_cast<const uint8_t*>(u),x,ids,hidden,input,ff,k,tokens,limit);
    else gu<false>(q,static_cast<const uint8_t*>(g),static_cast<const uint8_t*>(u),x,ids,hidden,input,ff,k,tokens,limit);
}
void mxfp4_down(const void* w, const float* hidden, const int32_t* ids, float* out,
                int ff, int output, int k, int tokens, void* stream, bool alu) {
    if (k < 1 || k > 64) throw std::invalid_argument("DeepSeek MXFP4 routing");
    auto& q = checked(stream,ff,output,tokens);
    if (alu) down<true>(q,static_cast<const uint8_t*>(w),hidden,ids,out,ff,output,k,tokens);
    else down<false>(q,static_cast<const uint8_t*>(w),hidden,ids,out,ff,output,k,tokens);
}
void mxfp4_expert_group(const void* gate,const void* up,const void* weights,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,
    const int32_t* destinations,int count,float limit,void* stream) {
    auto& q=checked(stream,input,ff,tokens);
    if(ff%32 || top_k<1 || top_k>64 || count<1 || count>tokens || !destinations)
        throw std::invalid_argument("DeepSeek grouped expert geometry");
    std::array<int32_t,max_prefill_batch> rows{};
    for(int i=0;i<count;++i){
        if(destinations[i]<0 || destinations[i]>=tokens*top_k)
            throw std::invalid_argument("DeepSeek grouped expert destination");
        for(int j=0;j<i;++j)if(destinations[j]/top_k==destinations[i]/top_k)
            throw std::invalid_argument("duplicate expert within one token route");
        rows[i]=destinations[i];
    }
    if(count>8){
        grouped_gu(q,static_cast<const uint8_t*>(gate),static_cast<const uint8_t*>(up),
            x,hidden,input,ff,top_k,count,limit,rows);
        grouped_down(q,static_cast<const uint8_t*>(weights),hidden,parts,ff,input,count,rows);
        return;
    }
    gu<false,true>(q,static_cast<const uint8_t*>(gate),static_cast<const uint8_t*>(up),
        x,nullptr,hidden,input,ff,1,count,limit,rows,top_k);
    down<false,true>(q,static_cast<const uint8_t*>(weights),hidden,nullptr,parts,
        ff,input,1,count,rows);
}

namespace {
// A bounded active row count specializes the existing math, rather than
// introducing a format tag or copying another decoder implementation.
template<class Launch> bool dispatch_small_rows(int count,Launch&& launch){
    switch(count){
        case 2:launch.template operator()<2>();return true;
        case 3:launch.template operator()<3>();return true;
        case 4:launch.template operator()<4>();return true;
        case 5:launch.template operator()<5>();return true;
        case 6:launch.template operator()<6>();return true;
        case 7:launch.template operator()<7>();return true;
        case 8:launch.template operator()<8>();return true;
        default:return false;
    }
}
template<bool Tiled>
void expert_group_impl(int gt,int ut,int dt,const void* gate,const void* up,const void* weights,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,
    const int32_t* destinations,int count,float limit,void* stream,int gate_tile,int down_tile,bool packed_metadata,bool iq1_gu_narrow,bool small_rows=false) {
    if constexpr(Tiled){
        if((gate_tile!=4 && gate_tile!=8 && gate_tile!=16) ||
           (down_tile!=4 && down_tile!=8 && down_tile!=16))
            throw std::invalid_argument("DeepSeek expert row tile must be4,8 or16");
    }
    if(gt==39 && ut==39 && dt==39) {
        mxfp4_expert_group(gate,up,weights,x,hidden,parts,input,ff,top_k,tokens,destinations,count,limit,stream);
        return;
    }
    auto& q=checked(stream,input,ff,tokens);
    if(gt!=ut || !iq::supported(gt) || !iq::supported(dt) ||
       top_k<1 || top_k>64 || count<1 || count>tokens || !destinations ||
       !gate || !up || !weights || !x || !hidden || !parts)
        throw std::invalid_argument("DeepSeek mixed expert geometry/format");
    iq::row_bytes(gt,input);iq::row_bytes(dt,ff);
    std::array<int32_t,max_prefill_batch> rows{};
    for(int i=0;i<count;++i){
        if(destinations[i]<0 || destinations[i]>=tokens*top_k)
            throw std::invalid_argument("DeepSeek grouped expert destination");
        for(int j=0;j<i;++j)if(destinations[j]/top_k==destinations[i]/top_k)
            throw std::invalid_argument("duplicate expert within one token route");
        rows[i]=destinations[i];
    }
    const auto g=static_cast<const uint8_t*>(gate),u=static_cast<const uint8_t*>(up),d=static_cast<const uint8_t*>(weights);
    auto launch_gu=[&]<int Type>(){
        if constexpr(Type==19){
            if(small_rows && iq1_gu_narrow && count==1){
                std::array<int32_t,1> active{rows[0]};
                iq1s_grouped_gu_narrow<1,1>(q,g,u,x,hidden,input,ff,top_k,1,limit,active);
                return;
            }
        }
        if constexpr(Type==16 || Type==19){
            if(small_rows && ((Type==16 && packed_metadata) || (Type==19 && iq1_gu_narrow)) &&
               count>=2 && count<=8 && (count<=4 || gate_tile==8)){
                dispatch_small_rows(count,[&]<int Rows>(){
                    constexpr int Tile=Rows<=4?4:8;
                    std::array<int32_t,Rows> active{};
                    std::copy_n(rows.begin(),Rows,active.begin());
                    if constexpr(Type==19)
                        iq1s_grouped_gu_narrow<Tile,Rows>(q,g,u,x,hidden,input,ff,top_k,count,limit,active);
                    else grouped_gu<Type,Tile,true,Rows>(q,g,u,x,hidden,input,ff,top_k,count,limit,active);
                });
                return;
            }
        }
        if constexpr(Type==19)if(iq1_gu_narrow){
            if constexpr(Tiled){
                if(count>8 && gate_tile==16){iq1s_grouped_gu_narrow<16>(q,g,u,x,hidden,input,ff,top_k,count,limit,rows);return;}
                if(count>8 && gate_tile==8){iq1s_grouped_gu_narrow<8>(q,g,u,x,hidden,input,ff,top_k,count,limit,rows);return;}
            }
            iq1s_grouped_gu_narrow<4>(q,g,u,x,hidden,input,ff,top_k,count,limit,rows);return;
        }
        if constexpr(Type==16)if(packed_metadata){
            if(tokens==1){single_gu<16,true>(q,g,u,x,hidden,input,ff,rows[0]/top_k,limit);return;}
            if constexpr(Tiled){
                if(count>8 && gate_tile==16){grouped_gu<16,16,true>(q,g,u,x,hidden,input,ff,top_k,count,limit,rows);return;}
                if(count>8 && gate_tile==8){grouped_gu<16,8,true>(q,g,u,x,hidden,input,ff,top_k,count,limit,rows);return;}
            }
            grouped_gu<16,4,true>(q,g,u,x,hidden,input,ff,top_k,count,limit,rows);return;
        }
        if constexpr(Type==16 || Type==17){if(tokens==1){single_gu<Type>(q,g,u,x,hidden,input,ff,rows[0]/top_k,limit);return;}}
        if constexpr(Tiled && (Type==16 || Type==19)){
            if(count>8 && gate_tile==16){grouped_gu<Type,16>(q,g,u,x,hidden,input,ff,top_k,count,limit,rows);return;}
            if(count>8 && gate_tile==8){grouped_gu<Type,8>(q,g,u,x,hidden,input,ff,top_k,count,limit,rows);return;}
        }
        grouped_gu<Type>(q,g,u,x,hidden,input,ff,top_k,count,limit,rows);
    };
    auto launch_down=[&]<int Type>(){
        // Generic IQ2_XS/IQ1_S/IQ2_S down retain their original dispatch:
        // constant active rows regressed those decoders in direct components.
        if constexpr(Type==16){
            if(small_rows && packed_metadata && count>=2 && count<=8 &&
               (count<=4 || down_tile==8)){
                dispatch_small_rows(count,[&]<int Rows>(){
                    constexpr int Tile=Rows<=4?4:8;
                    std::array<int32_t,Rows> active{};
                    std::copy_n(rows.begin(),Rows,active.begin());
                    grouped_down<Type,Tile,Rows<=4,Rows>(q,d,hidden,parts,ff,input,count,active);
                });
                return;
            }
        }
        if constexpr(Type==16)if(packed_metadata && (tokens==1 || count<=8 || down_tile==4)){
            if(tokens==1){single_down<16,true>(q,d,hidden,parts,ff,input,rows[0]);return;}
            grouped_down<16,4,true>(q,d,hidden,parts,ff,input,count,rows);return;
        }
        if constexpr(Type==16){if(tokens==1){single_down<Type>(q,d,hidden,parts,ff,input,rows[0]);return;}}
        if constexpr(Tiled && (Type==16 || Type==22)){
            if(count>8 && down_tile==16){grouped_down<Type,16>(q,d,hidden,parts,ff,input,count,rows);return;}
            if(count>8 && down_tile==8){grouped_down<Type,8>(q,d,hidden,parts,ff,input,count,rows);return;}
        }
        grouped_down<Type>(q,d,hidden,parts,ff,input,count,rows);
    };
    switch(gt){case 16:launch_gu.template operator()<16>();break;case 17:launch_gu.template operator()<17>();break;
        case 19:launch_gu.template operator()<19>();break;case 22:launch_gu.template operator()<22>();break;
        case 29:launch_gu.template operator()<29>();break;case 39:launch_gu.template operator()<39>();break;}
    switch(dt){case 16:launch_down.template operator()<16>();break;case 17:launch_down.template operator()<17>();break;
        case 19:launch_down.template operator()<19>();break;case 22:launch_down.template operator()<22>();break;
        case 29:launch_down.template operator()<29>();break;case 39:launch_down.template operator()<39>();break;}
}
} // namespace

void expert_group(int gt,int ut,int dt,const void* gate,const void* up,const void* weights,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,
    const int32_t* destinations,int count,float limit,void* stream) {
    expert_group_impl<false>(gt,ut,dt,gate,up,weights,x,hidden,parts,input,ff,top_k,tokens,destinations,count,limit,stream,4,4,false,false);
}
void expert_group_tiled(int gt,int ut,int dt,const void* gate,const void* up,const void* weights,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,
    const int32_t* destinations,int count,float limit,void* stream,int gate_tile,int down_tile) {
    expert_group_impl<true>(gt,ut,dt,gate,up,weights,x,hidden,parts,input,ff,top_k,tokens,destinations,count,limit,stream,gate_tile,down_tile,false,false);
}

void expert_group_tiled(int gt,int ut,int dt,const void* gate,const void* up,const void* weights,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,
    const int32_t* destinations,int count,float limit,void* stream,int gate_tile,int down_tile,bool packed_metadata) {
    expert_group_impl<true>(gt,ut,dt,gate,up,weights,x,hidden,parts,input,ff,top_k,tokens,destinations,count,limit,stream,gate_tile,down_tile,packed_metadata,false);
}

void expert_group_tiled(int gt,int ut,int dt,const void* gate,const void* up,const void* weights,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,
    const int32_t* destinations,int count,float limit,void* stream,int gate_tile,int down_tile,bool packed_metadata,bool iq1_gu_narrow) {
    expert_group_impl<true>(gt,ut,dt,gate,up,weights,x,hidden,parts,input,ff,top_k,tokens,destinations,count,limit,stream,gate_tile,down_tile,packed_metadata,iq1_gu_narrow);
}

void expert_group_tiled(int gt,int ut,int dt,const void* gate,const void* up,const void* weights,
    const float* x,float* hidden,float* parts,int input,int ff,int top_k,int tokens,
    const int32_t* destinations,int count,float limit,void* stream,int gate_tile,int down_tile,
    bool packed_metadata,bool iq1_gu_narrow,bool small_rows) {
    expert_group_impl<true>(gt,ut,dt,gate,up,weights,x,hidden,parts,input,ff,top_k,tokens,
        destinations,count,limit,stream,gate_tile,down_tile,packed_metadata,iq1_gu_narrow,small_rows);
}
} // namespace strata::deepseek41::cuda

namespace strata::deepseek41::cuda::expert_batch_detail {
inline constexpr int max_jobs=16;
using Job=ExpertBatchJob;
using Jobs=std::array<Job,max_jobs>;
inline constexpr int kernel_capacity=4;
using KernelJobs=std::array<Job,kernel_capacity>;
static_assert(sizeof(Jobs)==512);

template<int Type,bool Packed=false> void batch_single_gu(sycl::queue& q,KernelJobs jobs,int jobs_count,const float* x,float* hidden,int input,int ff,int top_k,float limit){
    const size_t stride=iq::row_bytes(Type,input);
    q.parallel_for(sycl::nd_range<2>({size_t(jobs_count),size_t((ff+3)/4)*128},{1,128}),
      [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
        const int expert=int(item.get_group(0));const auto job=jobs[expert];
        const int row=int(item.get_group(1))*4+item.get_local_id(1)/32,lane=item.get_local_id(1)%32;
        if(row>=ff)return;
        const float* in=x+size_t(job.entry/top_k)*input;
        const auto* g=job.gate+size_t(row)*stride;const auto* u=job.up+size_t(row)*stride;
        float ag=0,au=0;
        for(int base=0;base<input;base+=256){
            const auto* gb=g+size_t(base/256)*iq::block_bytes<Type>;
            const auto* ub=u+size_t(base/256)*iq::block_bytes<Type>;
            const float gd=iq::block_scale<Type>(gb),ud=iq::block_scale<Type>(ub);
            uint32_t gp=0,up=0;
            if constexpr(Type==16){gp=iq2xxs_payload<Packed>(gb,lane);up=iq2xxs_payload<Packed>(ub,lane);}
#pragma unroll
            for(int ib=0;ib<8;++ib){
                const int at=ib*32+lane;float gw,uw;
                if constexpr(Type==16){gw=iq2xxs_group_weight<Packed>(item.get_sub_group(),gp,at,gd);uw=iq2xxs_group_weight<Packed>(item.get_sub_group(),up,at,ud);}
                else {gw=iq::block_weight<Type>(gb,at,gd);uw=iq::block_weight<Type>(ub,at,ud);}
                const float v=in[base+at];ag=sycl::fma(gw,v,ag);au=sycl::fma(uw,v,au);
            }
        }
        const float gv=sycl::fmin(reduce(ag,item.get_sub_group()),limit);
        const float uv=sycl::fmax(-limit,sycl::fmin(reduce(au,item.get_sub_group()),limit));
        if(!lane)hidden[size_t(expert)*4*ff+row]=(gv/(1+sycl::exp(-gv)))*uv;
    });
}

template<int Type,bool Packed=false> void batch_single_down(sycl::queue& q,KernelJobs jobs,int jobs_count,const float* hidden,float* parts,int ff,int output){
    const size_t stride=iq::row_bytes(Type,ff);
    q.parallel_for(sycl::nd_range<2>({size_t(jobs_count),size_t((output+3)/4)*128},{1,128}),
      [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(32)]] {
        const int expert=int(item.get_group(0));const auto job=jobs[expert];
        const int row=int(item.get_group(1))*4+item.get_local_id(1)/32,lane=item.get_local_id(1)%32;
        if(row>=output)return;
        const auto* w=job.down+size_t(row)*stride;float acc=0;
        for(int base=0;base<ff;base+=256){
            const auto* b=w+size_t(base/256)*iq::block_bytes<Type>;const float d=iq::block_scale<Type>(b);
            uint32_t payload=0;if constexpr(Type==16)payload=iq2xxs_payload<Packed>(b,lane);
#pragma unroll
            for(int ib=0;ib<8;++ib){
                const int at=ib*32+lane;float value;
                if constexpr(Type==16)value=iq2xxs_group_weight<Packed>(item.get_sub_group(),payload,at,d);
                else value=iq::block_weight<Type>(b,at,d);
                acc=sycl::fma(value,hidden[size_t(expert)*4*ff+base+at],acc);
            }
        }
        const float v=reduce(acc,item.get_sub_group());if(!lane)parts[size_t(job.entry)*output+row]=v;
    });
}

template<int Tile,bool OneRow=false> void batch_iq1_gu(sycl::queue& q,KernelJobs jobs,int jobs_count,const float* x,float* hidden,int input,int ff,int top_k,float limit){
    static_assert(OneRow?Tile==1:(Tile==4 || Tile==8 || Tile==16));
    const size_t stride=iq::row_bytes(19,input);
    q.parallel_for(sycl::nd_range<3>({size_t(jobs_count),1,size_t((ff+3)/4)*128},{1,1,128}),
      [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
        const int expert=int(item.get_group(0));const auto job=jobs[expert];const int count=OneRow?1:job.count;
        const int row=int(item.get_group(2))*4+item.get_local_id(2)/32,lane=item.get_local_id(2)%32;
        if(row>=ff)return;
        const int first=OneRow?0:int(item.get_group(1))*Tile;
        const auto* g=job.gate+size_t(row)*stride;const auto* u=job.up+size_t(row)*stride;
        const float* inputs[Tile]={};
#pragma unroll
        for(int t=0;t<Tile;++t)if(first+t<count)
            inputs[t]=x+size_t((first+t==0?job.entry:0)/top_k)*input;
        float ag[Tile]={},au[Tile]={};
        auto accumulate=[&](int i,float gw,float uw){
#pragma unroll
            for(int t=0;t<Tile;++t)if(first+t<count){
                const float v=inputs[t][i];
                ag[t]=sycl::fma(gw,v,ag[t]);au[t]=sycl::fma(uw,v,au[t]);
            }
        };
        for(int base=0;base<input;base+=256){
            const auto* gb=g+size_t(base/256)*iq::block_bytes<19>;
            const auto* ub=u+size_t(base/256)*iq::block_bytes<19>;
            const float gd=iq::block_scale<19>(gb),ud=iq::block_scale<19>(ub);
#pragma unroll
            for(int ib=0;ib<8;ib+=2){
                const int at0=ib*32+lane,at1=at0+32;
                const float gw0=iq1_weight_narrow(gb,at0,gd);
                const float uw0=iq1_weight_narrow(ub,at0,ud);
                const float gw1=iq1_weight_narrow(gb,at1,gd);
                const float uw1=iq1_weight_narrow(ub,at1,ud);
                accumulate(base+at0,gw0,uw0);
                accumulate(base+at1,gw1,uw1);
            }
        }
#pragma unroll
        for(int t=0;t<Tile;++t)if(first+t<count){
            float gv=reduce(ag[t],item.get_sub_group()),uv=reduce(au[t],item.get_sub_group());
            if(!lane){gv=sycl::fmin(gv,limit);uv=sycl::fmax(-limit,sycl::fmin(uv,limit));
                hidden[(size_t(expert)*4+first+t)*ff+row]=(gv/(1+sycl::exp(-gv)))*uv;}
        }
    });
}

inline float iq2xs_weight_narrow(const uint8_t* b,int at,float d){
    const int ib=at/32,l=(at%32)/8,j=at%8;const auto* p=b+2+(ib*4+l)*2;
    const uint16_t q=std::endian::native==std::endian::little
        ?reinterpret_cast<const volatile IQ1Metadata16*>(p)->value:iq::u16(p);
    const int index=q&511,signs=iq::iq2_signs(q>>9),scale=(b[66+ib]>>(4*(l/2)))&15;
    const float db=d*(0.5f+float(scale))*0.25f;
    const float magnitude=db*float(iq::codebook_byte(iq2xs_grid,index,j));
    return (signs&(1<<j))?-magnitude:magnitude;
}
template<int Type,int Tile=4,bool Packed=false,bool Narrow=false> void batch_grouped_down(sycl::queue& q,KernelJobs jobs,int jobs_count,const float* hidden,float* parts,int ff,int output){
    static_assert(Tile==4 || Tile==8 || Tile==16);
    const size_t stride=iq::row_bytes(Type,ff);
    q.parallel_for(sycl::nd_range<3>({size_t(jobs_count),1,size_t((output+3)/4)*128},{1,1,128}),
      [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(32)]] {
        const int expert=int(item.get_group(0));const auto job=jobs[expert];const int count=job.count;
        const int row=int(item.get_group(2))*4+item.get_local_id(2)/32,lane=item.get_local_id(2)%32;
        if(row>=output)return;
        const int first=int(item.get_group(1))*Tile;const auto* w=job.down+size_t(row)*stride;
        float acc[Tile]={};
        auto accumulate=[&](int i,float value){
#pragma unroll
            for(int t=0;t<Tile;++t)if(first+t<count)acc[t]=sycl::fma(value,hidden[(size_t(expert)*4+first+t)*ff+i],acc[t]);
        };
        if constexpr(Type==39){
            for(int i=lane;i<ff;i+=32)accumulate(i,expert_weight<Type>(w,i));
        }else{
            for(int base=0;base<ff;base+=256){
                const auto* b=w+size_t(base/256)*iq::block_bytes<Type>;
                const float d=iq::block_scale<Type>(b);
                uint32_t payload=0;
                if constexpr(Type==16)payload=iq2xxs_payload<Packed>(b,lane);
                if constexpr(Narrow && Type==19){
#pragma unroll
                    for(int ib=0;ib<8;ib+=2){
                        const int at0=ib*32+lane,at1=at0+32;
                        const float w0=iq1_weight_narrow(b,at0,d),w1=iq1_weight_narrow(b,at1,d);
                        accumulate(base+at0,w0);accumulate(base+at1,w1);
                    }
                }else{
#pragma unroll
                for(int ib=0;ib<8;++ib){
                    const int at=ib*32+lane;
                    if constexpr(Type==16)accumulate(base+at,iq2xxs_group_weight<Packed>(item.get_sub_group(),payload,at,d));
                    else if constexpr(Narrow && Type==17)accumulate(base+at,iq2xs_weight_narrow(b,at,d));
                    else accumulate(base+at,iq::block_weight<Type>(b,at,d));
                }
                }
            }
        }
#pragma unroll
        for(int t=0;t<Tile;++t)if(first+t<count){
            const float v=reduce(acc[t],item.get_sub_group());
            if(!lane)parts[size_t((first+t==0?job.entry:0))*output+row]=v;
        }
    });
}

void launch_chunk(sycl::queue& q,int gt,int dt,KernelJobs jobs,int n,const float* x,float* hidden,float* parts,int H,int F,int top_k,int tokens,float limit,bool one_row=false,bool narrow_down=false){
 if(n<1 || n>kernel_capacity || H<256 || H%256 || F<256 || F%256 || top_k<1 || tokens<1 || tokens>128 || !x || !hidden || !parts)
  throw std::invalid_argument("private batched expert geometry");
 if((gt!=16&&gt!=17&&gt!=19)||(dt!=16&&dt!=17&&dt!=19&&dt!=22))throw std::invalid_argument("private batched expert formats");
 for(int j=0;j<n;++j){
  const auto& a=jobs[j];if(!a.gate||!a.up||!a.down||a.count!=1||a.entry<0||a.entry>=tokens*top_k)throw std::invalid_argument("private one-row descriptor");
  for(int k=0;k<j;++k)if(a.entry==jobs[k].entry)throw std::invalid_argument("duplicate output destination");
 }
 if(gt==16)batch_single_gu<16,true>(q,jobs,n,x,hidden,H,F,top_k,limit);
 else if(gt==17)batch_single_gu<17>(q,jobs,n,x,hidden,H,F,top_k,limit);
 else if(one_row)batch_iq1_gu<1,true>(q,jobs,n,x,hidden,H,F,top_k,limit);
 else batch_iq1_gu<4>(q,jobs,n,x,hidden,H,F,top_k,limit);
 if(dt==16)batch_single_down<16,true>(q,jobs,n,hidden,parts,F,H);
 else if(dt==17){
  if(narrow_down)batch_grouped_down<17,4,false,true>(q,jobs,n,hidden,parts,F,H);
  else batch_grouped_down<17>(q,jobs,n,hidden,parts,F,H);
 }
 else if(dt==19){
  if(narrow_down)batch_grouped_down<19,4,false,true>(q,jobs,n,hidden,parts,F,H);
  else batch_grouped_down<19>(q,jobs,n,hidden,parts,F,H);
 }
 else batch_grouped_down<22>(q,jobs,n,hidden,parts,F,H);
}

void launch(sycl::queue& q,int gt,int dt,Jobs jobs,int n,const float* x,float* hidden,float* parts,int H,int F,int top_k,int tokens,float limit,bool one_row=false,bool narrow_down=false){
 if(n<2 || n>max_jobs)throw std::invalid_argument("private logical batch count");
 for(int j=0;j<n;++j)for(int k=0;k<j;++k)
  if(jobs[j].entry==jobs[k].entry)throw std::invalid_argument("duplicate output destination");
 for(int first=0;first<n;first+=kernel_capacity){
  KernelJobs chunk{};const int count=std::min(kernel_capacity,n-first);
  for(int j=0;j<count;++j)chunk[j]=jobs[first+j];
  launch_chunk(q,gt,dt,chunk,count,x,hidden+size_t(first)*4*F,parts,H,F,top_k,tokens,limit,one_row,narrow_down);
 }
}
} // namespace strata::deepseek41::cuda::batch_trial


namespace strata::deepseek41::cuda {
static void expert_group_batch_impl(int gt,int dt,const ExpertBatchJob* input_jobs,int n,
    const float* x,float* hidden,float* parts,int H,int F,int top_k,int tokens,float limit,void* stream,bool one_row,bool narrow_down=false){
 if(!input_jobs || !stream || n<2 || n>max_expert_batch)throw std::invalid_argument("expert batch count/queue");
 expert_batch_detail::Jobs jobs{};
 for(int j=0;j<n;++j)jobs[j]=input_jobs[j];
 expert_batch_detail::launch(*static_cast<sycl::queue*>(stream),gt,dt,jobs,n,x,hidden,parts,H,F,top_k,tokens,limit,one_row,narrow_down);
}
void expert_group_batch(int gt,int dt,const ExpertBatchJob* jobs,int n,
    const float* x,float* hidden,float* parts,int H,int F,int top_k,int tokens,float limit,void* stream){
    expert_group_batch_impl(gt,dt,jobs,n,x,hidden,parts,H,F,top_k,tokens,limit,stream,false);
}
void expert_group_batch(int gt,int dt,const ExpertBatchJob* jobs,int n,
    const float* x,float* hidden,float* parts,int H,int F,int top_k,int tokens,float limit,void* stream,bool one_row){
    expert_group_batch_impl(gt,dt,jobs,n,x,hidden,parts,H,F,top_k,tokens,limit,stream,one_row);
}

void expert_group_batch(int gt,int dt,const ExpertBatchJob* jobs,int n,
    const float* x,float* hidden,float* parts,int H,int F,int top_k,int tokens,float limit,void* stream,bool one_row,bool narrow_down){
    expert_group_batch_impl(gt,dt,jobs,n,x,hidden,parts,H,F,top_k,tokens,limit,stream,one_row,narrow_down);
}

}
