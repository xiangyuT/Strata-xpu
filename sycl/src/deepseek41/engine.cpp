// Experimental single-GPU native DeepSeek driver. Large prefill batches share
// dense/expert weights; original quantized blobs use a bounded device cache.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include "strata/deepseek41/engine.hpp"
#include "strata/deepseek41/model.hpp"
#include "strata/deepseek41/resident.hpp"
#include "strata/deepseek41/expert_cache.hpp"
#include "strata/deepseek41/mxfp4.hpp"
#include "strata/deepseek41/gemm.hpp"
#include "strata/deepseek41/attention.hpp"
#include "strata/deepseek41/expert_host.hpp"
#include "strata/deepseek41/iq_weights.hpp"
#include "strata/deepseek41/kernels.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/sampler.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <poll.h>
#include <unistd.h>

namespace strata::deepseek41 {
namespace {
namespace C = strata::deepseek41::cuda;
namespace K = strata::kernels;
std::string block(int l) { return "blk."+std::to_string(l)+"."; }
struct Buffer {
    sycl::queue& q; void* p; size_t bytes;
    Buffer(sycl::queue& queue,size_t n):q(queue),p(sycl::malloc_device(n,q)),bytes(n) { if(!p)throw std::bad_alloc(); }
    ~Buffer(){sycl::free(p,q);}
    float* f(size_t off=0)const{return static_cast<float*>(p)+off;}
    int32_t* i()const{return static_cast<int32_t*>(p);}
    uint16_t* h()const{return static_cast<uint16_t*>(p);}
};
struct Host {
    sycl::queue& q; uint8_t* p; size_t bytes;
    Host(sycl::queue& queue,size_t n):q(queue),p(static_cast<uint8_t*>(sycl::malloc_host(n,q))),bytes(n){if(!p)throw std::bad_alloc();}
    ~Host(){sycl::free(p,q);}
};
struct Matrix { std::unique_ptr<Buffer> storage; int type,input,output; };
struct Engine {
    Model model; const Config& c; sycl::queue q;
    ResidentWeights backing;
    const int Batch;
    int context,position=0,last_rows=1,prefill_batch;
    std::map<std::string,Matrix> matrices;
    std::map<std::string,std::unique_ptr<Buffer>> buffers;
    std::vector<std::unique_ptr<Buffer>> raw,compressed,keys,projections,scores,selected;
    std::unique_ptr<Host> engram_stage;
    std::unique_ptr<ExpertCache> expert_cache;
    std::unique_ptr<ExpertHost> cpu_experts;
    bool cpu_cold=true,collect_features=false,speculative=false;
    // Internal rollback/ablation control, cached at engine construction:
    // 0 original; 1 normalized; 2 score-grid; 3 row views; 4 prefill row tiles;
    // 5 Q6 columns; 6 existing-pool host packing; 7 CPU IQ coefficients; 8 prefill attention.
    int sycl_tuning=0;
    uint64_t sycl_gpu_threads=0;
    std::unique_ptr<Buffer> attention_scratch,prefill_attention_scratch;
    std::vector<int32_t> history,routes;
    std::vector<std::vector<int32_t>> window_routes;
    Engine(const std::string& file,int ctx,int cache_mib=-1,int batch=128,bool dspark=false):model(file),c(model.config()),
        q(sycl::gpu_selector_v,[](sycl::exception_list list){for(auto& e:list)std::rethrow_exception(e);},sycl::property::queue::in_order{}),Batch(std::max(8,batch)),context(ctx),prefill_batch(batch),collect_features(dspark){
        if(c.hc!=4 || c.index_heads>32 || c.experts>512 || c.hidden%32 || c.expert_ff%32 || ctx<1 || ctx>c.context_train)
            throw std::runtime_error("unsupported single-GPU layout or context");
        if(batch<1 || batch>C::max_prefill_batch)throw std::invalid_argument("prefill batch must be in 1..128");
        bool tuned_arch=false;
        if(collect_features){
            namespace ex=sycl::ext::oneapi::experimental;
            try{tuned_arch=q.get_device().get_info<ex::info::device::architecture>()==ex::architecture::intel_gpu_bmg_g31;}catch(...){}
            sycl_tuning=tuned_arch?8:0;
            if(tuned_arch){
                const auto device=q.get_device();
                try{if(device.has(sycl::aspect::ext_intel_gpu_eu_count) && device.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu))
                    sycl_gpu_threads=uint64_t(device.get_info<sycl::ext::intel::info::device::gpu_eu_count>())*
                        device.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();
                }catch(...){sycl_gpu_threads=0;}
            }
            if(const char* value=std::getenv("STRATA_DS41_SYCL_DSPARK_TUNING")){
                char* end=nullptr;const long mode=std::strtol(value,&end,10);
                if(end==value || *end || mode<0 || mode>17)throw std::invalid_argument("STRATA_DS41_SYCL_DSPARK_TUNING must be 0..17");
                sycl_tuning=int(mode);
            }
        }
        K::native_mmvq_set_multi_exact(true);
        std::fprintf(stderr,"strata-deepseek41-sycl: device=%s; prefill batch=%d, TP1, DSpark=%d\n",q.get_device().get_info<sycl::info::device::name>().c_str(),batch,int(dspark));
        for(int l=0;l<c.layers;++l){
            const auto p=block(l);
            const auto gt=model.tensor(p+"ffn_gate_exps.weight").tensor->type;
            const auto ut=model.tensor(p+"ffn_up_exps.weight").tensor->type;
            const auto dt=model.tensor(p+"ffn_down_exps.weight").tensor->type;
            if(gt!=ut || !iq::supported(gt) || !iq::supported(dt))
                throw std::runtime_error("unsupported routed expert format combination");
            iq::row_bytes(gt,c.hidden);iq::row_bytes(dt,c.expert_ff);
        }
        size_t dense=0;
        for(size_t shard=0;shard<model.gguf().size();++shard)for(const auto& t:model.gguf().shard(shard).tensors()){
            if(t.name.find("_exps.weight")!=std::string::npos || (t.name.ends_with(".engram_embd.weight") || t.name=="per_layer_token_embd.weight"))continue;
            auto ref=model.tensor(t.name); load(t.name,ref); dense+=matrices.at(t.name).storage->bytes;
        }
        std::fprintf(stderr,"strata-deepseek41-sycl: dense uploads %.3f GiB complete\n",double(dense)/(1ull<<30));
        backing.load(model); allocate();
        if(collect_features){
            f("features",size_t(c.hidden)*3);window_routes.resize(c.layers);
            // Keep cache capacity equal in the original and tuned BMG arms.
            // The score-grid API supports at most eight query rows per call.
            if((tuned_arch || sycl_tuning>=2) && c.head_dim==512 && c.sliding_window==128 && c.index_top_k==512 && c.heads%2==0){
                const size_t cells=size_t(c.heads)*std::min(Batch,8)*(c.sliding_window+c.index_top_k);
                attention_scratch=std::make_unique<Buffer>(q,((cells*4+255)&~size_t(255))+cells*8);
            }
            if((tuned_arch || sycl_tuning>=8) && c.head_dim==512 && c.sliding_window==128 && c.index_top_k==512 && c.heads%4==0){
                const size_t rows=size_t(Batch)*(c.sliding_window+c.index_top_k),cells=rows*c.heads;
                // Reserve before cache sizing, identically for internal modes.
                prefill_attention_scratch=std::make_unique<Buffer>(q,((cells*4+255)&~size_t(255))+rows*8);
            }
        }
        std::fprintf(stderr,"strata-deepseek41-sycl: DSpark tuning=%d; original=0 normalized=1 score-grid=2 row-views=3 prefill-tiles=4 q6-columns=5 pool-packing=6 CPU-coeff=7 prefill-attention=8 paired-dots-QK-metadata=9 narrow-GU-raw-down=10 window-heat=11 prefill-transfer=12 verify-expert-batch-row-chunks=13 prefill-small-rows=14 verify-small-rows=15 verify-iq1-single=16 verify-down-metadata=17; GPU threads=%llu\n",sycl_tuning,(unsigned long long)sycl_gpu_threads);
        engram_stage=std::make_unique<Host>(q,c.engram.buckets()*size_t(c.engram.head_dim)*Batch*4);
        routes.resize(c.top_k*Batch);
        // Allocate the same small staging in both diagnostic arms before
        // automatic cache sizing, so the switch cannot change cache capacity.
        if(ExpertHost::available())cpu_experts=std::make_unique<ExpertHost>(&q,c.hidden,c.expert_ff,0,std::min(c.experts,c.top_k*Batch),Batch,sycl_tuning>=7,sycl_tuning>=9,sycl_tuning>=10);
        if(const char* value=std::getenv("STRATA_DS41_CPU_COLD"))cpu_cold=std::atoi(value)!=0;
        std::fprintf(stderr,"strata-deepseek41-sycl: CPU cold single-row fallback=%d\n",int(cpu_cold && bool(cpu_experts)));
        // Load the optional draft before sizing the target expert cache.
        if(!dspark)enable_cache(cache_mib);
    }
    void enable_cache(int mib){expert_cache=std::make_unique<ExpertCache>(model,backing,&q,mib,sycl_tuning>=6?cpu_experts.get():nullptr,sycl_tuning>=11,sycl_tuning>=12);}
    ~Engine(){try{q.wait_and_throw();}catch(...){}}
    Buffer& buffer(const std::string& name,size_t bytes){
        auto& b=buffers[name];if(!b || b->bytes<bytes){q.wait_and_throw();b=std::make_unique<Buffer>(q,bytes);}return *b;
    }
    // count is the width of one token row. Allocate the maximum batch before
    // sizing the expert cache, so later batch changes cannot steal its reserve.
    float* f(const std::string& name,size_t count){return buffer(name,count*Batch*4).f();}
    float* weight(const std::string& name){auto& m=matrices.at(name);if(m.type!=0)throw std::runtime_error("expected floating weight: "+name);return m.storage->f();}
    void load(const std::string& name,TensorRef ref){
        const auto& shape=ref.tensor->shape;size_t elements=1;for(auto d:shape)elements*=d;
        Matrix m; m.type=int(ref.tensor->type);m.input=int(shape[0]);m.output=int(elements/shape[0]);
        const uint8_t* src=ref.data();size_t bytes=ref.bytes();std::vector<float> floats;
        if(m.type==1 || m.type==30){floats.resize(elements);if(m.type==30)dequantize_bf16(src,floats.data(),int(elements));else dequantize_f16(src,floats.data(),int(elements));m.type=0;bytes=elements*4;src=reinterpret_cast<const uint8_t*>(floats.data());}
        if(m.type!=0 && !K::native_mmvq_supported(m.type))throw std::runtime_error("unsupported dense matrix: "+name);
        m.storage=std::make_unique<Buffer>(q,bytes);copy_registered_slice(m.storage->p,src,0,bytes,&q);matrices.emplace(name,std::move(m));
    }
    void quant(int type,const void* weights,const float* x,float* y,int input,int output,int tokens=1){
        if(tokens<1 || tokens>Batch || input<1 || input>INT_MAX/tokens)
            throw std::invalid_argument("DeepSeek dense quantization geometry");
        void* scratch=buffer("q8",K::native_q8_1_bytes(input)*Batch).p;
        // Flattening columns preserves every original32-element quant block.
        K::native_quantize_q8_1(x,scratch,input*tokens,1,&q);
        if(type==8 && C::q8_0_gemm(weights,scratch,y,input,output,tokens,&q))return;
        if(tokens>8 && sycl_tuning>=5 && type==14 && C::q6_k_gemm(weights,scratch,y,input,output,tokens,&q))return;
        const size_t stride=K::native_q8_1_bytes(input);
        for(int t=0;t<tokens;t+=8)K::native_mmvq(type,weights,
            static_cast<uint8_t*>(scratch)+size_t(t)*stride,y+size_t(t)*output,
            input,output,std::min(8,tokens-t),&q);
    }
    void multiply(const std::string& name,const float* x,float* y,int tokens=1){
        const auto& m=matrices.at(name);if(m.type==0)C::dense(m.storage->f(),x,y,m.input,m.output,tokens,&q);
        else quant(m.type,m.storage->p,x,y,m.input,m.output,tokens);
    }
    void norm(const std::string& name,const float* x,float* y,int width,int tokens=1){C::rms(x,weight(name),y,width,tokens,float(c.rms_eps),&q);}
    void allocate(){
        const size_t H=c.hidden,D=c.head_dim,A=c.heads,I=c.index_dim,F=c.expert_ff*c.shared_experts;
        for(const char* n:{"residual","flat_norm"})f(n,H*4);
        for(const char* n:{"pre","attn_pre","post"})f(n,4);f("comb",16);f("mix",24);
        for(const char* n:{"x","norm","partial","shared","ffn"})f(n,H);
        f("qr",c.q_rank);f("q",D*A);f("kv",D);f("attn",D*A);f("oa",size_t(c.o_rank)*c.o_groups);
        for(const char* n:{"gate","up","hidden"})f(n,F);
        f("expert_hidden",c.expert_ff);f("parts",H*c.top_k);f("router",c.experts);f("route_weights",c.top_k);buffer("route_ids",size_t(c.top_k)*Batch*4);
        f("index_q",size_t(c.index_heads)*I);f("index_w",c.index_heads);
        // Queries reuse scores in queue order; only the selected IDs persist
        // per token, bounded by index_top_k rather than the full context.
        buffer("index_scores",size_t(context)*4);buffer("index_sort",C::top_k_scratch_bytes(context));
        f("latent",D);f("index_key",I);f("engram_input",c.engram.buckets()*c.engram.head_dim);f("engram_kv",H*5);
        f("compress_values",D);f("compress_scores",D);f("logits",c.vocab);
        f("group_in",A/c.o_groups*D);f("group_out",c.o_rank);
        for(const char* n:{"token","position","compress_position_2"})buffer(n,Batch*4);
        buffer("q8",K::native_q8_1_bytes(std::max(int(H*4),int(A*D)))*Batch);
        raw.resize(c.layers);compressed.resize(c.layers);keys.resize(c.layers);projections.resize(c.layers);scores.resize(c.layers);selected.resize(c.layers);
        for(int l=0;l<c.layers;++l){
            raw[l]=std::make_unique<Buffer>(q,size_t(c.sliding_window+Batch)*D*2);const auto& s=model.streams()[l];int ratio=int(c.compress_ratios[l]);
            if(s.publishes_kv){compressed[l]=std::make_unique<Buffer>(q,size_t((context+ratio-1)/ratio)*D*2);projections[l]=std::make_unique<Buffer>(q,D*size_t(Batch*2)*4);scores[l]=std::make_unique<Buffer>(q,D*size_t(Batch*2)*4);}
            if(s.publishes_keys)keys[l]=std::make_unique<Buffer>(q,size_t((context+ratio-1)/ratio)*I*2);
            if(s.publishes_top_k)selected[l]=std::make_unique<Buffer>(q,size_t(c.index_top_k)*Batch*4);
        }
    }
    C::Rope rope(int layer)const{
        C::Rope r;r.dim=c.rope_dim;
        if(c.compress_ratios[layer]){r.base=float(c.compress_rope_base);r.scale=float(1/c.rope_scale);r.original_context=c.rope_context_original;r.beta_fast=float(c.beta_fast);r.beta_slow=float(c.beta_slow);r.yarn=true;}else r.base=float(c.rope_base);return r;
    }
    void hc(int layer,const char* sub,float* pre,int tokens){
        const auto p=block(layer)+"hc_"+sub;
        C::rms(f("residual",c.hidden*4),nullptr,f("flat_norm",c.hidden*4),c.hidden*4,tokens,float(c.rms_eps),&q);
        multiply(p+"_fn.weight",f("flat_norm",c.hidden*4),f("mix",24),tokens);
        C::hc_coeff(f("mix",24),weight(p+"_scale.weight"),weight(p+"_base.weight"),pre,f("post",4),f("comb",16),tokens,float(c.hc_eps),c.sinkhorn_iterations,&q);
    }
    void engram(int layer,const int32_t* inputs,int tokens){
        const auto& s=model.streams()[layer];if(s.engram_index<0)return;
        const auto p=block(layer);auto table=model.tensor(p+"engram_embd.weight");
        const size_t width=c.engram.buckets()*size_t(c.engram.head_dim),row=table.bytes()/table.tensor->shape[1];
        q.wait_and_throw();auto* dst=reinterpret_cast<float*>(engram_stage->p);
        // Later tokens in this batch must hash the earlier batch tokens too.
        const auto keep=std::min(history.size(),size_t(c.engram.max_ngram-1));
        std::vector<int32_t> local_history(history.end()-keep,history.end());
        for(int t=0;t<tokens;++t){
            const auto ids=engram_rows(c.engram,size_t(s.engram_index),inputs[t],local_history);
            local_history.push_back(inputs[t]);
            for(size_t j=0;j<ids.size();++j)dequantize_engram_row(table.tensor->type,
                backing.data(p+"engram_embd.weight")+size_t(ids[j])*row,c.engram.head_dim,
                dst+size_t(t)*width+j*c.engram.head_dim);
        }
        q.memcpy(f("engram_input",width),dst,size_t(tokens)*width*4);
        multiply(s.engram_projection,f("engram_input",width),f("engram_kv",c.hidden*5),tokens);
        C::engram_add(f("residual",c.hidden*4),f("engram_kv",c.hidden*5),weight(p+(s.engram_fused_gate?"engram_gate.weight":"engram_q.weight")),s.engram_fused_gate?nullptr:weight(p+"engram_k.weight"),nullptr,c.hidden,tokens,float(c.rms_eps),&q);
    }
    void attention(int l,int tokens){
        const auto p=block(l);const auto& s=model.streams()[l];const int H=c.hidden,D=c.head_dim,A=c.heads,ratio=int(c.compress_ratios[l]);
        hc(l,"attn",f("attn_pre",4),tokens);C::hc_pre(f("residual",H*4),f("pre",4),f("x",H),H,tokens,&q);norm(p+"attn_norm.weight",f("x",H),f("norm",H),H,tokens);
        multiply(p+"attn_q_a.weight",f("norm",H),f("qr",c.q_rank),tokens);norm(p+"attn_q_a_norm.weight",f("qr",c.q_rank),f("qr",c.q_rank),c.q_rank,tokens);
        multiply(p+"attn_q_b.weight",f("qr",c.q_rank),f("q",D*A),tokens);multiply(p+"attn_kv.weight",f("norm",H),f("kv",D),tokens);norm(p+"attn_kv_a_norm.weight",f("kv",D),f("kv",D),D,tokens);
        auto* pos=buffer("position",Batch*4).i();C::rope(f("q",D*A),D,A,tokens,pos,rope(l),false,&q);C::rope(f("kv",D),D,1,tokens,pos,rope(l),false,&q);
        // The extra Batch rows protect the earliest query's ring while all
        // the full new batch is stored. Attention still sees only causal rows.
        C::store_f16_positions(f("kv",D),raw[l]->h(),D,tokens,pos,c.sliding_window+Batch,0,&q);
        if(ratio && s.publishes_kv){
            multiply(p+"attn_compressor_kv.weight",f("norm",H),f("compress_values",D),tokens);
            if(ratio>1)multiply(p+"attn_compressor_gate.weight",f("norm",H),f("compress_scores",D),tokens);
            C::projection_ring(f("compress_values",D),ratio>1?f("compress_scores",D):nullptr,
                projections[l]->f(),ratio>1?scores[l]->f():nullptr,D,tokens,pos,Batch*2,&q);
            C::pool_positions(projections[l]->f(),scores[l]->f(),f("latent",D),D,tokens,pos,Batch*2,ratio,&q);
            norm(p+"attn_compressor_norm.weight",f("latent",D),f("latent",D),D,tokens);
            auto* cp=buffer(ratio==2?"compress_position_2":"position",Batch*4).i();
            const int capacity=(context+ratio-1)/ratio;
            if(s.publishes_keys){
                multiply(p+"indexer.attn_k.weight",f("latent",D),f("index_key",c.index_dim),tokens);
                norm(p+"indexer.k_norm.weight",f("index_key",c.index_dim),f("index_key",c.index_dim),c.index_dim,tokens);
                C::rope(f("index_key",c.index_dim),c.index_dim,1,tokens,cp,rope(l),false,&q);
                C::store_f16_positions(f("index_key",c.index_dim),keys[l]->h(),c.index_dim,tokens,pos,capacity,ratio,&q);
            }
            C::rope(f("latent",D),D,1,tokens,cp,rope(l),false,&q);
            C::store_f16_positions(f("latent",D),compressed[l]->h(),D,tokens,pos,capacity,ratio,&q);
        }
        if(ratio && s.publishes_top_k){
            multiply(p+"indexer.attn_q_b.weight",f("qr",c.q_rank),f("index_q",c.index_heads*c.index_dim),tokens);
            C::rope(f("index_q",c.index_heads*c.index_dim),c.index_dim,c.index_heads,tokens,pos,rope(l),false,&q);
            multiply(p+"indexer.proj.weight",f("norm",H),f("index_w",c.index_heads),tokens);
            auto* index_score=buffer("index_scores",size_t(context)*4).f();
            for(int t=0;t<tokens;++t){
                const int rows=(position+t+1)/ratio;if(!rows)continue;
                C::index_scores(f("index_q",c.index_heads*c.index_dim)+size_t(t)*c.index_heads*c.index_dim,
                    f("index_w",c.index_heads)+size_t(t)*c.index_heads,keys[s.key_source]->h(),index_score,c.index_dim,c.index_heads,rows,&q);
                auto& sort=buffer("index_sort",C::top_k_scratch_bytes(context));
                C::top_k_sorted(index_score,selected[l]->i()+size_t(t)*c.index_top_k,rows,std::min(rows,c.index_top_k),sort.p,sort.bytes,&q);
            }
        }
        if(sycl_tuning>=8 && !speculative && prefill_attention_scratch &&
           C::attention_positions_prefill(f("q",D*A),raw[l]->h(),ratio?compressed[s.kv_source]->h():nullptr,
                ratio?selected[s.top_k_source]->i():nullptr,weight(p+"attn_sinks.weight"),f("attn",D*A),prefill_attention_scratch->p,
                D,A,tokens,c.sliding_window,pos,ratio,ratio?c.index_top_k:0,c.index_top_k,c.sliding_window+Batch,&q,sycl_tuning>=9)){
            // Enqueued the validated large-batch path on the same in-order queue.
        }else if(sycl_tuning>=9 && speculative && tokens>1 && tokens<=8 && attention_scratch &&
           C::attention_positions_score_grid_paired(f("q",D*A),raw[l]->h(),ratio?compressed[s.kv_source]->h():nullptr,
                ratio?selected[s.top_k_source]->i():nullptr,weight(p+"attn_sinks.weight"),f("attn",D*A),attention_scratch->p,
                D,A,tokens,c.sliding_window,pos,ratio,ratio?c.index_top_k:0,c.index_top_k,c.sliding_window+Batch,&q,true)){
            // Same workspace, softmax and value path; only QK shares key reads.
        }else if(sycl_tuning>=2 && speculative && tokens>1 && tokens<=8 && attention_scratch){
            C::attention_positions_score_grid(f("q",D*A),raw[l]->h(),ratio?compressed[s.kv_source]->h():nullptr,
                ratio?selected[s.top_k_source]->i():nullptr,weight(p+"attn_sinks.weight"),f("attn",D*A),attention_scratch->p,
                D,A,tokens,c.sliding_window,pos,ratio,ratio?c.index_top_k:0,c.index_top_k,c.sliding_window+Batch,&q,true);
        }else{
            auto launch=sycl_tuning && tokens>1 ? C::attention_positions_normalized : C::attention_positions;
            launch(f("q",D*A),raw[l]->h(),ratio?compressed[s.kv_source]->h():nullptr,
                ratio?selected[s.top_k_source]->i():nullptr,weight(p+"attn_sinks.weight"),f("attn",D*A),
                D,A,tokens,c.sliding_window,pos,ratio,ratio?c.index_top_k:0,c.index_top_k,c.sliding_window+Batch,&q);
        }
        C::rope(f("attn",D*A),D,A,tokens,pos,rope(l),true,&q);
        const int group_dim=A/c.o_groups*D;const auto& oa=matrices.at(p+"attn_output_a.weight");const size_t bytes=K::native_mmvq_weight_bytes(oa.type,group_dim,c.o_rank);
        for(int g=0;g<c.o_groups;++g){
            const float* input=f("attn",D*A)+g*group_dim;float* output=f("oa",c.o_rank*c.o_groups)+g*c.o_rank;
            if(tokens>1)C::strided_copy(input,f("group_in",group_dim),group_dim,tokens,D*A,group_dim,&q);
            quant(oa.type,static_cast<uint8_t*>(oa.storage->p)+size_t(g)*bytes,
                tokens>1?f("group_in",group_dim):input,tokens>1?f("group_out",c.o_rank):output,group_dim,c.o_rank,tokens);
            if(tokens>1)C::strided_copy(f("group_out",c.o_rank),output,c.o_rank,tokens,c.o_rank,c.o_rank*c.o_groups,&q);
        }
        multiply(p+"attn_output_b.weight",f("oa",c.o_rank*c.o_groups),f("partial",H),tokens);
    }
    void ffn(int l,int tokens){
        const auto p=block(l);const int H=c.hidden,F=c.expert_ff*c.shared_experts,entries=c.top_k*tokens;
        C::hc_post(f("partial",H),f("residual",H*4),f("post",4),f("comb",16),f("residual",H*4),H,tokens,&q);
        hc(l,"ffn",f("pre",4),tokens);C::hc_pre(f("residual",H*4),f("attn_pre",4),f("x",H),H,tokens,&q);norm(p+"ffn_norm.weight",f("x",H),f("norm",H),H,tokens);
        multiply(p+"ffn_gate_inp.weight",f("norm",H),f("router",c.experts),tokens);
        C::route(f("router",c.experts),weight(p+"exp_probs_b.bias"),buffer("route_ids",size_t(c.top_k)*Batch*4).i(),f("route_weights",c.top_k),c.experts,c.top_k,tokens,float(c.expert_scale),&q);
        auto route_ready=q.memcpy(routes.data(),buffer("route_ids",size_t(c.top_k)*Batch*4).p,size_t(entries)*4);
        multiply(p+"ffn_gate_shexp.weight",f("norm",H),f("gate",F),tokens);multiply(p+"ffn_up_shexp.weight",f("norm",H),f("up",F),tokens);
        C::swiglu(f("gate",F),f("up",F),f("hidden",F),F*tokens,float(c.clamp_shared[l]),&q);multiply(p+"ffn_down_shexp.weight",f("hidden",F),f("shared",H),tokens);
        // Shared work is independent of the host route grouping. Wait only
        // for route IDs, allowing it to run while the CPU prepares experts.
        route_ready.wait_and_throw();
        expert_cache->observe(l,routes.data(),entries,!speculative);
        if(speculative)window_routes.at(l).assign(routes.begin(),routes.begin()+entries);
        // Group repeated routes without changing their destination or combine
        // order. Each expert appears at most once per token, hence <=Batch rows.
        std::map<int,std::vector<int>> groups;
        for(int j=0;j<entries;++j)groups[routes[j]].push_back(j);
        const int gt=model.tensor(p+"ffn_gate_exps.weight").tensor->type;
        const int ut=model.tensor(p+"ffn_up_exps.weight").tensor->type;
        const int dt=model.tensor(p+"ffn_down_exps.weight").tensor->type;
        const size_t gb=model.tensor(p+"ffn_gate_exps.weight").bytes()/c.experts;
        const size_t ub=model.tensor(p+"ffn_up_exps.weight").bytes()/c.experts;
        const size_t db=model.tensor(p+"ffn_down_exps.weight").bytes()/c.experts;
        const auto* host_g=backing.data(p+"ffn_gate_exps.weight");
        const auto* host_u=backing.data(p+"ffn_up_exps.weight");
        const auto* host_d=backing.data(p+"ffn_down_exps.weight");
        const bool batch_cpu=cpu_cold && bool(cpu_experts);
        // No acquire may replace a retained device slot. A CPU-eligible miss
        // can still be admitted when hot, so upload_transient=false is insufficient.
        bool deferred_resident=batch_cpu;
        if(deferred_resident)for(const auto& [id,destinations]:groups){
            if(expert_cache->requires_upload(l,id,destinations.size()!=1)){
                deferred_resident=false;break;
            }
        }
        std::vector<ExpertHost::Job> cold_jobs;
        struct ResidentJob{const uint8_t* device;const std::vector<int32_t>* destinations;};
        std::vector<ResidentJob> resident_jobs;
        auto launch_resident_one=[&](const ResidentJob& job){
            const auto* device=job.device;const auto& destinations=*job.destinations;
            if(sycl_tuning>=3 && destinations.size()==1 && (gt==16 || gt==17 || dt==16)){
                // A one-row expert is a one-token view even when its parent
                // verification batch has several tokens. Preserve global slots.
                const int entry=destinations[0];const int32_t zero=0;
                if(sycl_tuning>=9)C::expert_group_tiled(gt,ut,dt,device,device+gb,device+gb+ub,
                    f("norm",H)+size_t(entry/c.top_k)*H,f("expert_hidden",c.expert_ff),
                    f("parts",H*c.top_k)+size_t(entry)*H,H,c.expert_ff,1,1,&zero,1,float(c.clamp_expert[l]),&q,4,4,true,sycl_tuning>=10,sycl_tuning>=16 && speculative);
                else C::expert_group(gt,ut,dt,device,device+gb,device+gb+ub,
                    f("norm",H)+size_t(entry/c.top_k)*H,f("expert_hidden",c.expert_ff),
                    f("parts",H*c.top_k)+size_t(entry)*H,H,c.expert_ff,1,1,&zero,1,float(c.clamp_expert[l]),&q);
                return;
            }
            if(sycl_tuning>=4 && (!speculative || (sycl_tuning>=15 && destinations.size()>=2 && destinations.size()<=4)) && sycl_gpu_threads){
                const int count=int(destinations.size());
                int gate_tile=4,down_tile=4;
                if(gt==16 && count>8)gate_tile=count>=32?16:8;
                else if(gt==19 && count>=16)gate_tile=8;
                if(dt==16 && count>8)down_tile=16;
                else if(dt==22 && count>8)down_tile=8;
                if(sycl_tuning>=14 && count>=5 && count<=8){
                    if(gt==16 || gt==19)gate_tile=8;
                    if(dt==16)down_tile=8;
                }
                // Small grids retain the original path: more rows per thread
                // cannot help when the proposed grid cannot fill one wave.
                if(uint64_t(c.expert_ff)*uint64_t((count+gate_tile-1)/gate_tile)<sycl_gpu_threads)gate_tile=4;
                if(uint64_t(H)*uint64_t((count+down_tile-1)/down_tile)<sycl_gpu_threads)down_tile=4;
                if(gate_tile!=4 || down_tile!=4 || sycl_tuning>=9){
                    C::expert_group_tiled(gt,ut,dt,device,device+gb,device+gb+ub,f("norm",H),
                        f("expert_hidden",c.expert_ff),f("parts",H*c.top_k),H,c.expert_ff,c.top_k,tokens,
                        destinations.data(),count,float(c.clamp_expert[l]),&q,gate_tile,down_tile,sycl_tuning>=9,sycl_tuning>=10,sycl_tuning>=14);
                    return;
                }
            }
            if(sycl_tuning>=9){
                C::expert_group_tiled(gt,ut,dt,device,device+gb,device+gb+ub,f("norm",H),
                    f("expert_hidden",c.expert_ff),f("parts",H*c.top_k),H,c.expert_ff,
                    c.top_k,tokens,destinations.data(),int(destinations.size()),float(c.clamp_expert[l]),&q,4,4,true,sycl_tuning>=10,sycl_tuning>=16 && speculative && destinations.size()==1);
                return;
            }
            C::expert_group(gt,ut,dt,device,device+gb,device+gb+ub,f("norm",H),
                f("expert_hidden",c.expert_ff),f("parts",H*c.top_k),H,c.expert_ff,
                c.top_k,tokens,destinations.data(),int(destinations.size()),float(c.clamp_expert[l]),&q);
        };

        const bool batch_resident=sycl_tuning>=13 && speculative && gt==ut &&
            (gt==16 || gt==17 || gt==19) && (dt==16 || dt==17 || dt==19 || dt==22) &&
            H>=256 && H%256==0 && c.expert_ff>=256 && c.expert_ff%256==0 &&
            4*C::max_expert_batch<=Batch;
        std::array<ResidentJob,C::max_expert_batch> pending_residents{};
        int pending_count=0;
        auto flush_residents=[&]{
            if(pending_count==1)launch_resident_one(pending_residents[0]);
            else if(pending_count>1){
                std::array<C::ExpertBatchJob,C::max_expert_batch> jobs{};
                for(int j=0;j<pending_count;++j){const auto& a=pending_residents[j];
                    jobs[j]={a.device,a.device+gb,a.device+gb+ub,a.destinations->at(0),1};}
                C::expert_group_batch(gt,dt,jobs.data(),pending_count,f("norm",H),
                    f("expert_hidden",c.expert_ff),f("parts",H*c.top_k),H,c.expert_ff,
                    c.top_k,tokens,float(c.clamp_expert[l]),&q,sycl_tuning>=15,sycl_tuning>=17);
            }
            pending_count=0;
        };
        auto launch_resident=[&](const ResidentJob& job){
            if(!batch_resident){launch_resident_one(job);return;}
            if(job.destinations->size()!=1){flush_residents();launch_resident_one(job);return;}
            pending_residents[pending_count++]=job;
            if(pending_count==C::max_expert_batch)flush_residents();
        };
        if(batch_cpu)cpu_experts->prepare_inputs(f("norm",H),tokens);
        auto next=groups.cbegin();
        auto collect=[&](size_t n,std::vector<ExpertHost::Job>& cold){
            while(n-- && next!=groups.cend()){
                const auto& [id,destinations]=*next++;
                const int count=int(destinations.size());
                const bool cpu=cpu_cold && cpu_experts && count==1;
                // Pipeline the long prefill batches; verification keeps its
                // original direct upload path and avoids a staging-copy cost.

                // Submit every retained read before acquire can overwrite its
                // slot or shared scratch. CPU-eligible misses may still admit.
                if(pending_count && expert_cache->requires_upload(l,id,!cpu))flush_residents();
                const auto* device=expert_cache->acquire(l,id,!cpu,!speculative && tokens>1);
                if(!device){
                    const int entry=destinations[0];
                    if(batch_cpu){cold.push_back({host_g+size_t(id)*gb,host_u+size_t(id)*ub,host_d+size_t(id)*db,
                        f("parts",H*c.top_k)+size_t(entry)*H,entry/c.top_k});continue;}
                    cpu_experts->run(gt,ut,dt,host_g+size_t(id)*gb,host_u+size_t(id)*ub,host_d+size_t(id)*db,
                        f("norm",H)+size_t(entry/c.top_k)*H,f("parts",H*c.top_k)+size_t(entry)*H,float(c.clamp_expert[l]));
                }else if(deferred_resident){
                    // The read-only preflight proved that this entire batch
                    // cannot write a retained device slot.
                    resident_jobs.push_back({device,&destinations});
                }else launch_resident({device,&destinations});
            }
        };
        auto run_cold=[&](const std::vector<ExpertHost::Job>& jobs,const std::function<void()>& after={}){

            if(sycl_tuning<13 || !speculative){
                if(!jobs.empty())cpu_experts->run_batch(gt,ut,dt,jobs.data(),int(jobs.size()),float(c.clamp_expert[l]),after);
                return;
            }
            flush_residents();
            if(!jobs.empty()){
                std::function<void()> callback;
                if(after)callback=[&]{after();flush_residents();};
                cpu_experts->run_batch(gt,ut,dt,jobs.data(),int(jobs.size()),
                    float(c.clamp_expert[l]),callback,sycl_tuning>=13 && speculative);
            }
        };
        if(batch_cpu && tokens>1 && !deferred_resident){
            // Keep acquire/submission order across both halves: prefill admission
            // may recycle a slot, so resident pointers are never stashed here.
            collect(groups.size()/2,cold_jobs);
            if(cold_jobs.empty()){
                collect(groups.size(),cold_jobs);run_cold(cold_jobs);
            }else{
                std::vector<ExpertHost::Job> late_cold;
                run_cold(cold_jobs,[&]{collect(groups.size(),late_cold);});
                run_cold(late_cold);
            }
        }else{
            collect(groups.size(),cold_jobs);
            if(batch_cpu && !cold_jobs.empty()){
                const size_t first=resident_jobs.size()/2;
                for(size_t j=0;j<first;++j)launch_resident(resident_jobs[j]);
                run_cold(cold_jobs,[&]{for(size_t j=first;j<resident_jobs.size();++j)launch_resident(resident_jobs[j]);});
            }else for(const auto& job:resident_jobs)launch_resident(job);
        }
        flush_residents();
        C::combine(f("parts",H*c.top_k),f("route_weights",c.top_k),f("shared",H),f("ffn",H),H,c.top_k,tokens,&q);
        C::hc_post(f("ffn",H),f("residual",H*4),f("post",4),f("comb",16),f("residual",H*4),H,tokens,&q);
    }
    void process(const int32_t* inputs,int tokens,bool output,bool verify=false){
        if(tokens<1 || tokens>Batch || position+tokens>context)throw std::runtime_error("input/context out of range");
        if(verify && !collect_features)throw std::runtime_error("speculative verification requires DSpark state");
        speculative=verify;
        int32_t pos[C::max_prefill_batch],cp[C::max_prefill_batch];
        for(int t=0;t<tokens;++t){
            if(inputs[t]<0 || inputs[t]>=c.vocab)throw std::runtime_error("token out of range");
            pos[t]=position+t;cp[t]=position+t-1;
        }
        q.memcpy(buffer("token",Batch*4).p,inputs,size_t(tokens)*4);
        q.memcpy(buffer("position",Batch*4).p,pos,size_t(tokens)*4);
        q.memcpy(buffer("compress_position_2",Batch*4).p,cp,size_t(tokens)*4);
        const auto& emb=matrices.at("token_embd.weight");
        K::iq_embed_rows(emb.type,emb.storage->p,emb.storage->bytes/size_t(c.vocab),buffer("token",Batch*4).i(),tokens,c.hidden,f("x",c.hidden),&q);
        C::hc_init(f("x",c.hidden),f("residual",c.hidden*4),f("pre",4),c.hidden,tokens,&q);
        for(int l=0;l<c.layers;++l){
            if(collect_features && l>=c.layers-3)C::hc_mean(f("residual",c.hidden*4),
                f("features",c.hidden*3)+size_t(l-(c.layers-3))*c.hidden,c.hidden,tokens,c.hidden*3,&q);
            engram(l,inputs,tokens);attention(l,tokens);ffn(l,tokens);
            if(position==0 && l%8==7){q.wait_and_throw();std::fprintf(stderr,"strata-deepseek41-sycl: first block layer %d/%d\n",l+1,c.layers);}
        }
        if(output){
            C::hc_pre(f("residual",c.hidden*4),f("pre",4),f("x",c.hidden),c.hidden,tokens,&q);
            norm("output_norm.weight",f("x",c.hidden),f("norm",c.hidden),c.hidden,tokens);
            multiply("output.weight",f("norm",c.hidden),f("logits",c.vocab),tokens);
        }
        q.wait_and_throw();history.insert(history.end(),inputs,inputs+tokens);position+=tokens;last_rows=tokens;
    }
    void step(int32_t token,bool output=true){process(&token,1,output);}
    void retain(int count){
        if(!speculative || count<1 || count>last_rows)throw std::runtime_error("invalid retained verification window");
        for(int l=0;l<c.layers;++l)expert_cache->commit(l,window_routes.at(l).data(),count*c.top_k);
        // Rings reserve Batch extra rows; causal positions mask rejected tails.
        // Compression rows are addressed by absolute position and overwritten
        // before reuse. Engram hashes must see only the committed history.
        position-=last_rows-count;history.resize(size_t(position));last_rows=count;speculative=false;
    }
    void reset(){q.wait_and_throw();position=0;last_rows=1;history.clear();speculative=false;}
    void report(){expert_cache->report();if(cpu_experts)cpu_experts->report();}
    std::vector<float> logits(){
        std::vector<float> out(c.vocab);q.memcpy(out.data(),f("logits",c.vocab)+size_t(last_rows-1)*c.vocab,out.size()*4).wait_and_throw();
        for(float v:out)if(!std::isfinite(v))throw std::runtime_error("nonfinite logits");return out;
    }
    std::vector<int32_t> greedy_rows(){
        std::vector<float> out(size_t(last_rows)*c.vocab);
        q.memcpy(out.data(),f("logits",c.vocab),out.size()*4).wait_and_throw();
        std::vector<int32_t> ids(last_rows);
        for(int t=0;t<last_rows;++t){
            const auto first=out.begin()+size_t(t)*c.vocab,last=first+c.vocab;
            for(auto i=first;i!=last;++i)if(!std::isfinite(*i))throw std::runtime_error("nonfinite verification logits");
            ids[t]=int32_t(std::max_element(first,last)-first);
        }
        return ids;
    }
    int sample(K::SamplerParams params){
        const bool neutral=params.penalty_last_n<=0 || (params.penalty_repeat==1 && params.penalty_freq==0 && params.penalty_present==0);
        if(params.greedy && neutral){auto out=logits();return int(std::max_element(out.begin(),out.end())-out.begin());}
        const int n=std::min(int(history.size()),std::max(0,params.penalty_last_n));
        if(n)q.memcpy(buffer("penalty_history",size_t(context)*4).p,history.data()+history.size()-n,size_t(n)*4);
        params.counter=position;
        K::sample_tokens(f("logits",c.vocab)+size_t(last_rows-1)*c.vocab,1,c.vocab,
            n?buffer("penalty_history",size_t(context)*4).i():nullptr,n,params,buffer("pick",Batch*4).i(),&q);
        int32_t id=-1;q.memcpy(&id,buffer("pick",Batch*4).p,4).wait_and_throw();
        if(id<0 || id>=c.vocab)throw std::runtime_error("sample outside vocabulary");return id;
    }
};

// Same three-stage DSpark graph as the CUDA driver. Immutable target embedding
// and output weights are shared on one in-order queue. Draft KV is separate;
// only committed target features enter its history.
class Draft {
public:
    GgufFile file;
    Config c;
    Engine& target;
    sycl::queue& q;
    const int Batch;
    void* stream;
    int position=0,block_size=5,mask=0,context_limit=0;
    float min_confidence=0.75f;
    bool mxfp4_alu=false;
    std::map<std::string,Matrix> matrices;
    std::map<std::string,std::unique_ptr<Buffer>> buffers;
    std::vector<std::unique_ptr<Buffer>> raw;
    std::array<int32_t,C::max_prefill_batch> proposals{};
    std::array<float,C::max_prefill_batch> confidence{};
    Draft(const std::string& path,Engine& engine):file(path),c(engine.c),target(engine),q(engine.q),Batch(engine.Batch),stream(&q){
        auto integer=[&](const std::string& key){
            const auto* v=file.get(key);if(!v || !v->is_num() || v->u>INT32_MAX)throw std::runtime_error("DSpark missing/invalid "+key);return int(v->u);
        };
        auto same=[&](const char* key,int expected){if(integer(std::string("dflash.")+key)!=expected)throw std::runtime_error(std::string("DSpark incompatible ")+key);};
        const auto* arch=file.get("general.architecture");
        if(!arch || arch->s!="dflash" || file.find("output_hc_fn.weight"))throw std::runtime_error("expected 1017 dflash DSpark without an HC output head");
        same("embedding_length",c.hidden);same("block_count",3);same("attention.head_count",c.heads);
        same("attention.key_length",c.head_dim);same("attention.value_length",c.head_dim);same("attention.q_lora_rank",c.q_rank);
        same("attention.output_group_count",c.o_groups);same("attention.output_lora_rank",c.o_rank);same("rope.dimension_count",c.rope_dim);
        same("hyper_connection.count",4);same("expert_gating_func",4);same("expert_feed_forward_length",c.expert_ff);
        same("expert_count",128);same("expert_used_count",3);same("expert_shared_count",1);
        if(c.layers!=40 || file.integer_array("dflash.target_layers")!=std::vector<uint64_t>({37,38,39}) ||
           file.integer_array("dflash.attention.compress_ratios")!=std::vector<uint64_t>({0,0,0}))throw std::runtime_error("DSpark requires target layers 37/38/39 and three plain sliding-attention stages");
        c.layers=3;c.experts=128;c.top_k=3;c.shared_experts=1;
        c.sliding_window=integer("dflash.attention.sliding_window");
        c.clamp_expert=file.real_array("dflash.swiglu_clamp_exp");c.clamp_shared=file.real_array("dflash.swiglu_clamp_shexp");
        if(c.sliding_window<1 || c.clamp_expert.size()!=3 || c.clamp_shared.size()!=3)throw std::runtime_error("DSpark attention/clamp metadata");
        block_size=integer("dflash.block_size");mask=integer("tokenizer.ggml.mask_token_id");context_limit=target.context;
        if(block_size!=5 || block_size>=Batch || mask<0 || mask>=c.vocab)throw std::runtime_error("DSpark noise block geometry");
        if(const char* v=std::getenv("STRATA_DS41_DSPARK_P_MIN"))min_confidence=std::stof(v);
        if(!std::isfinite(min_confidence) || min_confidence<0 || min_confidence>1)throw std::runtime_error("DSpark confidence threshold must be 0..1");
        uint64_t bytes=0;
        for(const auto& tensor:file.tensors()){
            TensorRef ref{&file,&tensor};
            if(!ref.bytes() || tensor.offset>file.file_size()-file.data_start() || ref.bytes()>file.file_size()-file.data_start()-tensor.offset)throw std::runtime_error("truncated DSpark tensor");
            load(tensor.name,ref);bytes+=matrices.at(tensor.name).storage->bytes;
        }
        auto check=[&](const std::string& name,std::vector<uint64_t> shape,int type=-1,std::vector<uint64_t> alternative={}){
            const auto* t=file.find(name);
            if(!t || (t->shape!=shape && t->shape!=alternative) || (type>=0 && w(name).type!=type))throw std::runtime_error("DSpark tensor geometry/type: "+name);
        };
        const uint64_t H=c.hidden,F=c.expert_ff,V=c.vocab,D=c.head_dim,A=c.heads;
        check("fc.weight",{H*3,H});check("enc.output_norm.weight",{H},0);check("output_norm.weight",{H},0);
        check("markov_w1.weight",{256,V},0);check("markov_w2.weight",{256,V});
        if(min_confidence>0){check("conf_proj.weight",{H+256,1},0);if(file.find("conf_proj.bias"))check("conf_proj.bias",{1},0);}
        for(int l=0;l<3;++l){
            const auto p=block(l);
            for(const char* sub:{"attn","ffn"}){
                const auto h=p+"hc_"+sub;check(h+"_fn.weight",{H*4,24});check(h+"_scale.weight",{3},0);check(h+"_base.weight",{24},0);
            }
            check(p+"attn_norm.weight",{H},0);check(p+"ffn_norm.weight",{H},0);
            check(p+"attn_q_a.weight",{H,uint64_t(c.q_rank)});check(p+"attn_q_a_norm.weight",{uint64_t(c.q_rank)},0);
            check(p+"attn_q_b.weight",{uint64_t(c.q_rank),A*D});check(p+"attn_kv.weight",{H,D});check(p+"attn_kv_a_norm.weight",{D},0);
            check(p+"attn_sinks.weight",{A},0);
            // GGUF may describe the same contiguous grouped matrix with the
            // group axis explicit or flattened into its output dimension.
            check(p+"attn_output_a.weight",{A/uint64_t(c.o_groups)*D,uint64_t(c.o_rank),uint64_t(c.o_groups)},-1,
                {A/uint64_t(c.o_groups)*D,uint64_t(c.o_rank*c.o_groups)});
            check(p+"attn_output_b.weight",{uint64_t(c.o_rank*c.o_groups),H});
            check(p+"ffn_gate_inp.weight",{H,128});check(p+"exp_probs_b.bias",{128},0);
            for(const char* name:{"ffn_gate_exps.weight","ffn_up_exps.weight"})check(p+name,{H,F,128},39);
            check(p+"ffn_down_exps.weight",{F,H,128},39);
            for(const char* name:{"ffn_gate_shexp.weight","ffn_up_shexp.weight"})check(p+name,{H,F});
            check(p+"ffn_down_shexp.weight",{F,H});
        }
        allocate();q.wait_and_throw();
        std::fprintf(stderr,"strata-deepseek41-sycl: DSpark stages=3 block=%d target_layers=37,38,39 confidence_min=%.2f MXFP4 weights=%.3f GiB; shared target embedding/head\n",block_size,min_confidence,double(bytes)/(1ull<<30));
    }
    ~Draft(){try{q.wait_and_throw();}catch(...){}}
    Buffer& buffer(const std::string& name,size_t row_bytes){
        auto& b=buffers[name];const size_t bytes=row_bytes*Batch;
        if(!b || b->bytes<bytes){q.wait_and_throw();b=std::make_unique<Buffer>(q,bytes);}return *b;
    }
    float* f(const std::string& name,size_t width){return buffer(name,width*4).f();}
    Matrix& w(const std::string& name){
        if(name=="target_embedding")return target.matrices.at("token_embd.weight");
        if(name=="target_output")return target.matrices.at("output.weight");
        return matrices.at(name);
    }
    float* weight(const std::string& name){auto& m=w(name);if(m.type!=0)throw std::runtime_error("expected floating draft weight: "+name);return m.storage->f();}
    void load(const std::string& name,TensorRef ref){
        const auto& shape=ref.tensor->shape;size_t elements=1;
        for(auto d:shape){if(!d || d>SIZE_MAX/elements)throw std::runtime_error("DSpark tensor size overflow");elements*=d;}
        if(shape.empty() || shape[0]>INT_MAX || elements/shape[0]>INT_MAX)throw std::runtime_error("DSpark tensor dimensions");
        Matrix m;m.type=ref.tensor->type;m.input=int(shape[0]);m.output=int(elements/shape[0]);
        const uint8_t* src=ref.data();size_t bytes=ref.bytes();std::vector<float> floats;
        if(m.type==1 || m.type==30){
            if(elements>INT_MAX)throw std::runtime_error("DSpark floating tensor size");
            floats.resize(elements);if(m.type==30)dequantize_bf16(src,floats.data(),int(elements));else dequantize_f16(src,floats.data(),int(elements));
            m.type=0;bytes=elements*4;src=reinterpret_cast<const uint8_t*>(floats.data());
        }
        if(m.type!=0 && m.type!=39 && !K::native_mmvq_supported(m.type))throw std::runtime_error("unsupported DSpark matrix: "+name);
        m.storage=std::make_unique<Buffer>(q,bytes);copy_registered_slice(m.storage->p,src,0,bytes,&q);matrices.emplace(name,std::move(m));
    }
    void mat(const std::string& name,const float* x,float* y,int n=1){
        const auto& m=w(name);
        if(m.type==0)C::dense(m.storage->f(),x,y,m.input,m.output,n,&q);
        else if(m.type==39)C::mxfp4_mmv(m.storage->p,x,y,m.input,m.output,n,&q,mxfp4_alu);
        else target.quant(m.type,m.storage->p,x,y,m.input,m.output,n);
    }
    void norm(const std::string& name,const float* x,float* y,int width,float eps,int n){C::rms(x,weight(name),y,width,n,eps,&q);}
    void allocate() {
        auto& r = *this; const int H = c.hidden, D = c.head_dim, A = c.heads, F = c.expert_ff;
        r.f("residual",H*4); r.f("flat_norm",H*4); r.f("pre",4); r.f("attn_pre",4); r.f("post",4); r.f("comb",16); r.f("mix",24);
        r.f("x",H); r.f("norm",H); r.f("qr",c.q_rank); r.f("q",D*A); r.f("kv",D); r.f("attn",D*A);
        r.f("oa",c.o_rank*c.o_groups); r.f("group_in",A/c.o_groups*D); r.f("group_out",c.o_rank); r.f("out",H);
        r.f("gate",F); r.f("up",F); r.f("hidden",F); r.f("shared",H); r.f("ffn",H);
        r.f("router",c.experts); r.f("route_weights",c.top_k); r.buffer("route_ids",size_t(c.top_k)*4);
        r.f("expert_hidden",F*c.top_k); r.f("parts",H*c.top_k);
        r.f("features",H*3); r.f("encoded",H); r.f("logits",c.vocab); r.f("markov",256); r.f("bias",c.vocab); r.f("confidence",1);
        r.buffer("token",4); r.buffer("position",4); r.buffer("pick",4); r.buffer("q8",K::native_q8_1_bytes(H*4));
        r.raw.resize(3); for (auto& raw : r.raw) raw = std::make_unique<Buffer>(q,size_t(c.sliding_window+Batch)*D*2);
    }
    C::Rope rope_config() const { C::Rope rc; rc.dim = c.rope_dim; rc.base = float(c.rope_base); return rc; }
    void reset() { q.wait_and_throw(); position = 0; }
    void inject(Engine& target,int start,int count) {
        if (start != position || count < 1 || count > Batch) throw std::runtime_error("DSpark committed feature position mismatch");
        auto& r = *this; const int H = c.hidden, D = c.head_dim; int32_t positions[Batch]; for (int t = 0; t < count; ++t) positions[t] = start+t;
        q.memcpy(r.f("features",H*3),target.f("features",H*3),size_t(count)*H*3*4);
        q.memcpy(r.buffer("position",4).p,positions,size_t(count)*4);
        r.mat("fc.weight",r.f("features",H*3),r.f("encoded",H),count);
        r.norm("enc.output_norm.weight",r.f("encoded",H),r.f("encoded",H),H,float(c.rms_eps),count);
        for (int l = 0; l < 3; ++l) {
            const auto p = block(l); r.mat(p+"attn_kv.weight",r.f("encoded",H),r.f("kv",D),count);
            r.norm(p+"attn_kv_a_norm.weight",r.f("kv",D),r.f("kv",D),D,float(c.rms_eps),count);
            C::rope(r.f("kv",D),D,1,count,r.buffer("position",4).i(),rope_config(),false,r.stream);
            for (int t = 0; t < count; ++t) C::store_f16(r.f("kv",D)+size_t(t)*D,r.raw[l]->h(),D,(start+t)%(c.sliding_window+Batch),r.stream);
        }
        q.wait_and_throw(); position += count;
    }
    void hc(int l,const char* sub,float* pre,int n) {
        auto& r = *this; const auto p = block(l)+"hc_"+sub;
        C::rms(r.f("residual",c.hidden*4),nullptr,r.f("flat_norm",c.hidden*4),c.hidden*4,n,float(c.rms_eps),r.stream);
        r.mat(p+"_fn.weight",r.f("flat_norm",c.hidden*4),r.f("mix",24),n);
        C::hc_coeff(r.f("mix",24),r.weight(p+"_scale.weight"),r.weight(p+"_base.weight"),pre,r.f("post",4),r.f("comb",16),n,float(c.hc_eps),c.sinkhorn_iterations,r.stream);
    }
    void attention(int l,int n) {
        auto& r = *this; const auto p = block(l); const int H = c.hidden, D = c.head_dim, A = c.heads;
        r.mat(p+"attn_q_a.weight",r.f("norm",H),r.f("qr",c.q_rank),n);
        r.norm(p+"attn_q_a_norm.weight",r.f("qr",c.q_rank),r.f("qr",c.q_rank),c.q_rank,float(c.rms_eps),n);
        r.mat(p+"attn_q_b.weight",r.f("qr",c.q_rank),r.f("q",A*D),n);
        r.mat(p+"attn_kv.weight",r.f("norm",H),r.f("kv",D),n);
        r.norm(p+"attn_kv_a_norm.weight",r.f("kv",D),r.f("kv",D),D,float(c.rms_eps),n);
        C::rope(r.f("q",A*D),D,A,n,r.buffer("position",4).i(),rope_config(),false,r.stream);
        C::rope(r.f("kv",D),D,1,n,r.buffer("position",4).i(),rope_config(),false,r.stream);
        C::store_f16_positions(r.f("kv",D),r.raw[l]->h(),D,n,r.buffer("position",4).i(),c.sliding_window+Batch,0,r.stream);
        C::draft_attention_positions(r.f("q",A*D),r.raw[l]->h(),r.weight(p+"attn_sinks.weight"),r.f("attn",A*D),D,A,n,c.sliding_window,r.buffer("position",4).i(),c.sliding_window+Batch,r.stream);
        C::rope(r.f("attn",A*D),D,A,n,r.buffer("position",4).i(),rope_config(),true,r.stream);
        const int width = A/c.o_groups*D; const auto& matrix = r.w(p+"attn_output_a.weight");
        const size_t group_bytes = K::native_mmvq_weight_bytes(matrix.type,width,c.o_rank);
        for (int g = 0; g < c.o_groups; ++g) {
            C::strided_copy(r.f("attn",A*D)+size_t(g)*width,r.f("group_in",width),width,n,A*D,width,r.stream);
            target.quant(matrix.type,static_cast<const uint8_t*>(matrix.storage->p)+size_t(g)*group_bytes,
                r.f("group_in",width),r.f("group_out",c.o_rank),width,c.o_rank,n);
            C::strided_copy(r.f("group_out",c.o_rank),r.f("oa",c.o_groups*c.o_rank)+size_t(g)*c.o_rank,c.o_rank,n,c.o_rank,c.o_groups*c.o_rank,r.stream);
        }
        r.mat(p+"attn_output_b.weight",r.f("oa",c.o_groups*c.o_rank),r.f("out",H),n);
    }
    void ffn(int l,int n) {
        auto& r = *this; const auto p = block(l); const int H = c.hidden, F = c.expert_ff;
        r.mat(p+"ffn_gate_inp.weight",r.f("norm",H),r.f("router",c.experts),n);
        C::route(r.f("router",c.experts),r.weight(p+"exp_probs_b.bias"),r.buffer("route_ids",size_t(c.top_k)*4).i(),r.f("route_weights",c.top_k),c.experts,c.top_k,n,float(c.expert_scale),r.stream);
        r.mat(p+"ffn_gate_shexp.weight",r.f("norm",H),r.f("gate",F),n); r.mat(p+"ffn_up_shexp.weight",r.f("norm",H),r.f("up",F),n);
        C::swiglu(r.f("gate",F),r.f("up",F),r.f("hidden",F),n*F,float(c.clamp_shared[l]),r.stream);
        r.mat(p+"ffn_down_shexp.weight",r.f("hidden",F),r.f("shared",H),n);
        auto* ids = r.buffer("route_ids",size_t(c.top_k)*4).i();
        C::mxfp4_gu(r.w(p+"ffn_gate_exps.weight").storage->p,r.w(p+"ffn_up_exps.weight").storage->p,r.f("norm",H),ids,r.f("expert_hidden",F*c.top_k),H,F,c.top_k,n,float(c.clamp_expert[l]),r.stream,mxfp4_alu);
        C::mxfp4_down(r.w(p+"ffn_down_exps.weight").storage->p,r.f("expert_hidden",F*c.top_k),ids,r.f("parts",H*c.top_k),F,H,c.top_k,n,r.stream,mxfp4_alu);
        C::combine(r.f("parts",H*c.top_k),r.f("route_weights",c.top_k),r.f("shared",H),r.f("ffn",H),H,c.top_k,n,r.stream);
    }
    void noise_decode(int n) {
        auto& r=*this; const int H=c.hidden;
        const auto& embedding = r.w("target_embedding");
        K::iq_embed_rows(embedding.type,embedding.storage->p,embedding.storage->bytes/size_t(c.vocab),r.buffer("token",4).i(),n,H,r.f("x",H),r.stream);
        C::hc_init(r.f("x",H),r.f("residual",H*4),r.f("pre",4),H,n,r.stream);
        for (int l = 0; l < 3; ++l) {
            hc(l,"attn",r.f("attn_pre",4),n); C::hc_pre(r.f("residual",H*4),r.f("pre",4),r.f("x",H),H,n,r.stream);
            r.norm(block(l)+"attn_norm.weight",r.f("x",H),r.f("norm",H),H,float(c.rms_eps),n); attention(l,n);
            C::hc_post(r.f("out",H),r.f("residual",H*4),r.f("post",4),r.f("comb",16),r.f("residual",H*4),H,n,r.stream);
            hc(l,"ffn",r.f("pre",4),n); C::hc_pre(r.f("residual",H*4),r.f("attn_pre",4),r.f("x",H),H,n,r.stream);
            r.norm(block(l)+"ffn_norm.weight",r.f("x",H),r.f("norm",H),H,float(c.rms_eps),n); ffn(l,n);
            C::hc_post(r.f("ffn",H),r.f("residual",H*4),r.f("post",4),r.f("comb",16),r.f("residual",H*4),H,n,r.stream);
        }
        C::hc_pre(r.f("residual",H*4),r.f("pre",4),r.f("x",H),H,n,r.stream);
        r.norm("output_norm.weight",r.f("x",H),r.f("norm",H),H,float(c.rms_eps),n);
        r.mat("target_output",r.f("norm",H),r.f("logits",c.vocab),n);
    }
    std::vector<int32_t> propose(int anchor,int count) {
        if (count<=0) return {};
        auto& r=*this; const int H=c.hidden;
        // Preserve the trained five-position noncausal noise block; only the
        // emitted prefix is shortened by confidence or the request boundary.
        const int n=std::min(block_size,context_limit-position);
        int32_t ids[Batch], positions[Batch];
        for (int t=0;t<n;++t) { ids[t]=t ? mask : anchor; positions[t]=position+t; }
        q.memcpy(r.buffer("token",4).p,ids,size_t(n)*4);
        q.memcpy(r.buffer("position",4).p,positions,size_t(n)*4);
        auto body=[&] {
            noise_decode(n);
            const auto& w1=r.w("markov_w1.weight");
            K::SamplerParams params; params.greedy=true; params.temperature=0;
            for (int t=0;t<n;++t) {
                const auto* previous=t ? r.buffer("pick",4).i()+t-1 : r.buffer("token",4).i();
                C::gather_row(w1.storage->f(),previous,r.f("markov",256),256,c.vocab,r.stream);
                r.mat("markov_w2.weight",r.f("markov",256),r.f("bias",c.vocab));
                auto* logits=r.f("logits",c.vocab)+size_t(t)*c.vocab;
                C::add(logits,r.f("bias",c.vocab),logits,c.vocab,r.stream);
                if (min_confidence>0) {
                    const auto bias=r.matrices.find("conf_proj.bias");
                    C::draft_confidence(r.f("x",H)+size_t(t)*H,r.f("markov",256),r.weight("conf_proj.weight"),
                        bias==r.matrices.end() ? nullptr : bias->second.storage->f(),r.f("confidence",1)+t,H,256,r.stream);
                }
                K::sample_tokens(logits,1,c.vocab,nullptr,0,params,r.buffer("pick",4).i()+t,r.stream);
            }
        };
        body();
        q.memcpy(proposals.data(),r.buffer("pick",4).p,size_t(n)*4);
        if (min_confidence>0) q.memcpy(confidence.data(),r.f("confidence",1),size_t(n)*4);
        q.wait_and_throw();
        std::vector<int32_t> output;
        for (int t=0;t<std::min(count,n);++t) {
            const int id=proposals.data()[t];
            if (id<0 || id>=c.vocab) throw std::runtime_error("DSpark prediction outside vocabulary");
            if (min_confidence>0 && !std::isfinite(confidence.data()[t])) throw std::runtime_error("DSpark confidence is not finite");
            const bool uncertain=min_confidence>0 && confidence.data()[t]<min_confidence;
            if (uncertain && !output.empty()) break;
            output.push_back(id); // Existing accepted policy: always offer first.
            if (uncertain) break;
        }
        return output;
    }

};

std::vector<int32_t> tokens(std::string s){std::replace(s.begin(),s.end(),',',' ');std::istringstream in(s);std::vector<int32_t> out;int32_t id;while(in>>id)out.push_back(id);if(!in.eof() || out.empty())throw std::runtime_error("--tokens requires integer token IDs");return out;}
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Control {
    std::string pending;
    std::deque<std::string> lines;
    bool stopped = false, eof = false;
    void pump(int timeout) {
        if (eof) return;
        pollfd fd{STDIN_FILENO,POLLIN,0}; const int result = poll(&fd,1,timeout);
        if (result < 0) { if (errno == EINTR) return; throw std::runtime_error("control poll failed"); }
        if (!result) return;
        char bytes[65536]; const auto n = read(STDIN_FILENO,bytes,sizeof(bytes));
        if (n == 0) { eof = true; return; }
        if (n < 0) { if (errno == EINTR) return; throw std::runtime_error("control read failed"); }
        pending.append(bytes,size_t(n));
        for (;;) {
            const size_t at = pending.find('\n'); if (at == std::string::npos) break;
            std::string line = pending.substr(0,at); pending.erase(0,at+1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line == "STOP") stopped = true; else lines.push_back(std::move(line));
        }
    }
    bool next(std::string& line) {
        while (lines.empty() && !eof) pump(-1);
        if (lines.empty()) return false;
        line = std::move(lines.front()); lines.pop_front(); return true;
    }
};
int eos(const Model& model) {
    const auto* value = model.gguf().meta().get("tokenizer.ggml.eos_token_id");
    return value && value->is_num() ? int(value->u) : -1;
}
struct Request {
    int maximum = 64;
    int draft_depth = -1;
    K::SamplerParams sampling;
    std::vector<int32_t> prompt;
    Request() { sampling.temperature = 0; sampling.greedy = true; }
};
Request request(const std::string& line) {
    std::istringstream in(line); std::string command, item; Request r;
    if (!(in >> command >> r.maximum) || command != "GEN" || r.maximum < 0) throw std::runtime_error("expected GEN <max_new> [keys] <id,id,...>");
    while (in >> item) {
        const auto eq = item.find('=');
        if (eq == std::string::npos) { r.prompt = tokens(item); if (in >> item) throw std::runtime_error("extra GEN input"); break; }
        const std::string key = item.substr(0,eq), value = item.substr(eq+1);
        if (key == "temperature") { r.sampling.temperature = std::stof(value); r.sampling.greedy = r.sampling.temperature <= 0; }
        else if (key == "top_p") r.sampling.top_p = std::stof(value);
        else if (key == "top_k") r.sampling.top_k = std::stoi(value);
        else if (key == "min_p") r.sampling.min_p = std::stof(value);
        else if (key == "seed") r.sampling.seed = std::stoull(value);
        else if (key == "penalty_last_n") r.sampling.penalty_last_n = std::stoi(value);
        else if (key == "penalty_repeat") r.sampling.penalty_repeat = std::stof(value);
        else if (key == "penalty_freq") r.sampling.penalty_freq = std::stof(value);
        else if (key == "penalty_present") r.sampling.penalty_present = std::stof(value);
        else if (key == "draft_depth") {
            r.draft_depth = std::stoi(value);
            if (r.draft_depth < 0 || r.draft_depth > 5) throw std::runtime_error("draft_depth must be 0..5");
        }
        else if (key != "cache_id" && key != "pin" && key != "cache_mode") throw std::runtime_error("unsupported GEN key: "+key);
    }
    if (r.prompt.empty()) throw std::runtime_error("empty native prompt");
    r.sampling.penalty_last_n = std::max(0,r.sampling.penalty_last_n);
    return r;
}
void generate(Engine& e,const Request& request,Control* control,const std::vector<int32_t>& forced = {},const std::string& dump = {},Draft* draft = nullptr,int max_drafts = 5) {
    if (request.prompt.size()+size_t(request.maximum) > size_t(e.context)) throw std::runtime_error("prompt/output exceed native context");
    const auto& sampling = request.sampling;
    if (request.draft_depth >= 0) max_drafts = request.draft_depth;
    const bool neutral = sampling.penalty_last_n <= 0 || (sampling.penalty_repeat == 1 && sampling.penalty_freq == 0 && sampling.penalty_present == 0);
    Draft* active = draft && max_drafts && sampling.greedy && neutral && forced.empty() ? draft : nullptr;
    e.reset(); const auto prompt_start = Clock::now();
    if (active) active->reset();
    size_t read = 0; bool stopped = false;
    while (read < request.prompt.size()) {
        const int count = int(std::min(size_t(e.prefill_batch),request.prompt.size()-read));
        e.process(request.prompt.data()+read,count,read+size_t(count) == request.prompt.size());
        if (active) active->inject(e,int(read),count);
        read += size_t(count);
        if (control) { control->pump(0); stopped = control->stopped; }
        if (control && (read%8 == 0 || read == request.prompt.size())) {
            std::printf("PP %zu %zu %.0f %.1f\n",read,request.prompt.size(),elapsed(prompt_start),1000*read/std::max(1e-9,elapsed(prompt_start))); std::fflush(stdout);
        }
        if (stopped) break;
    }
    const double prompt_ms = elapsed(prompt_start);
    if (!dump.empty() && !stopped) {
        const auto logits=e.logits();std::ofstream out(dump,std::ios::binary);
        out.write(reinterpret_cast<const char*>(logits.data()),logits.size()*4);
        if(!out)throw std::runtime_error("logit output failed");
    }
    const auto decode_start = Clock::now();
    int produced = 0, offered = 0, accepted = 0, verified_rows = 0, windows=0;
    double draft_ms = 0, verify_ms = 0, inject_ms=0;
    const int maximum = std::min(request.maximum,e.context-int(request.prompt.size()));
    const char* finish = stopped ? "cancel" : "length";
    while (!stopped && produced < maximum) {
        const int token = size_t(produced) < forced.size() ? forced[size_t(produced)] : e.sample(request.sampling);
        // The anchor is already selected by the target. Publish it immediately,
        // as on the Qwen path, rather than delaying TTFT behind a draft window.
        std::printf("T %d\n",token);std::fflush(stdout);++produced;
        if(token==eos(e.model)){finish="stop";break;}
        if(produced>=maximum || e.position>=e.context)break;
        if (active && e.position+1 < e.context) {
            const auto d0 = Clock::now();
            const auto proposals = active->propose(token,std::min({max_drafts,maximum-produced,e.context-e.position-1}));
            draft_ms += elapsed(d0); offered += int(proposals.size());++windows;
            std::vector<int32_t> window = {token}; window.insert(window.end(),proposals.begin(),proposals.end());
            const int start = e.position; const auto v0 = Clock::now();
            e.process(window.data(),int(window.size()),true,true); const auto actual = e.greedy_rows();
            verify_ms += elapsed(v0); verified_rows += int(window.size());
            int keep = 1;
            while (size_t(keep) < window.size() && proposals[size_t(keep-1)] == actual[size_t(keep-1)]) ++keep;
            for (int j = 0; j < keep; ++j) if (window[size_t(j)] == eos(e.model)) { keep = j+1; finish = "stop"; break; }
            accepted += keep-1; e.retain(keep); const auto i0=Clock::now();active->inject(e,start,keep);inject_ms+=elapsed(i0);
            for (int j = 1; j < keep; ++j) { std::printf("T %d\n",window[size_t(j)]); ++produced; }
            std::fflush(stdout);
            if (std::strcmp(finish,"stop") == 0) break;
            if (control) { control->pump(0); stopped = control->stopped; if (stopped) finish = "cancel"; }
            continue;
        }
        e.step(token); ++verified_rows;
        if (control) { control->pump(0); stopped = control->stopped; if (stopped) finish = "cancel"; }
    }
    const double decode_ms = elapsed(decode_start);
    const int steps = std::max(0,produced-1);
    std::printf("DONE %d %zu %.1f %.1f %s %d %d 0 0 0 0 0 0.0 %zu 0\n",produced,request.prompt.size(),prompt_ms,decode_ms,finish,accepted,offered,read); std::fflush(stdout);
    std::fprintf(stderr,"strata-deepseek41: native prompt_tokens=%zu prompt_ms=%.3f decode_steps=%d decode_ms=%.3f tok/s=%.6f\n",read,prompt_ms,steps,decode_ms,decode_ms > 0 ? 1000*steps/decode_ms : 0);
    if (active) std::fprintf(stderr,"strata-deepseek41: native DSpark accepted=%d offered=%d verified_rows=%d draft_ms=%.3f verify_ms=%.3f inject_ms=%.3f windows=%d depth=%d\n",accepted,offered,verified_rows,draft_ms,verify_ms,inject_ms,windows,max_drafts);
    e.backing.verify();
    e.report();
}
} // namespace
int run(int argc,char** argv){
    try{
        std::setvbuf(stdout,nullptr,_IONBF,0);
        std::string file,input,dump,dspark,force;int ctx=2048,n=8,cache_mib=-1,batch=128,max_drafts=5;bool serve=false;
        for(int i=1;i<argc;++i){
            const std::string a=argv[i];auto value=[&](){if(++i>=argc)throw std::runtime_error("missing value: "+a);return std::string(argv[i]);};
            if(a=="--native" || a=="--model")file=value();
            else if(a=="--tokens")input=value();else if(a=="--ctx")ctx=std::stoi(value());else if(a=="--n")n=std::stoi(value());
            else if(a=="--prefill-batch")batch=std::stoi(value());else if(a=="--expert-cache-mib")cache_mib=std::stoi(value());
            else if(a=="--dump-full-logits")dump=value();else if(a=="--force")force=value();
            else if(a=="--dspark")dspark=value();else if(a=="--no-dspark")dspark.clear();
            else if(a=="--draft-depth")max_drafts=std::stoi(value());else if(a=="--serve")serve=true;
            else if(a=="--tp"){if(value()!="1")throw std::runtime_error("this experimental driver supports TP1 only");}
            else if(a=="generate"){}
            else if(a=="--version"){std::puts(STRATA_VERSION);return 0;}
            else if(a=="--help"){
                std::puts("strata-deepseek41 --model GGUF [--tokens IDS --n 8 | --serve] [--ctx 2048]\n  --prefill-batch 1..128 (default 128) --expert-cache-mib -1|0|MiB (auto|off|budget)\n  --dspark DRAFT.gguf --draft-depth 1..5 (opt-in; greedy without active penalties)\n  --no-dspark disables the draft; --serve uses the standard Strata engine protocol.");return 0;
            }else throw std::runtime_error("unknown argument: "+a);
        }
        if(file.empty() || (!serve && input.empty()) || n<0 || max_drafts<1 || max_drafts>5)throw std::runtime_error("model and token IDs/--serve required; n>=0 and draft-depth=1..5");
        Engine e(file,ctx,cache_mib,batch,!dspark.empty());std::unique_ptr<Draft> draft;
        if(!dspark.empty()){draft=std::make_unique<Draft>(dspark,e);e.enable_cache(cache_mib);}
        if(!serve){
            Request r;r.prompt=tokens(input);r.maximum=n;generate(e,r,nullptr,force.empty()?std::vector<int32_t>{}:tokens(force),dump,draft.get(),max_drafts);return 0;
        }
        std::printf("INFO engine=%s arch=deepseek41 tp=1 kv=F16 dspark=%d draft_depth=%d prefill_batch=%d\n",STRATA_VERSION,draft?1:0,max_drafts,e.prefill_batch);
        std::printf("READY %d stop\n",ctx);Control control;std::string line;
        while(control.next(line)){
            if(line=="QUIT" || line=="EXIT")break;
            if(line.empty() || line=="PING"){if(line=="PING")std::puts("PONG");continue;}
            try{control.stopped=false;generate(e,request(line),&control,{},"",draft.get(),max_drafts);}
            catch(const std::exception& error){std::printf("ERR %s\n",error.what());std::fflush(stdout);}
        }
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"strata-deepseek41-sycl: %s\n",e.what());return 1;}
}
} // namespace strata::deepseek41
