#include "strata/deepseek41/model.hpp"
#include "gguf_fixture.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <iostream>

namespace ds = strata::deepseek41;
void check(bool v,const char* s) { if (!v) throw std::runtime_error(s); }
template<class F> void rejects(F&& f) {
    bool rejected=false;
    try { f(); } catch (const std::exception&) { rejected=true; }
    check(rejected,"invalid input accepted");
}
void decode() {
    std::array<uint8_t,33> block{};
    std::array<float,34> out;
    for (int scale=0;scale<256;++scale) for (int code=0;code<256;++code) {
        block[0]=uint8_t(scale); std::fill(block.begin()+1,block.end(),uint8_t(code)); out.fill(12345.0f);
        ds::dequantize_engram_row(67,block.data(),32,out.data()+1);
        const int sign=code&128?-1:1,e=(code>>3)&15,m=code&7;
        // Independent double-precision rational oracle, rounded once to F32.
        const float expected=scale==255 || (e==15 && m==7) ? std::numeric_limits<float>::quiet_NaN() :
            float(std::ldexp(double(sign)*(e ? 8+m : m), (e ? e-10 : -9)+scale-127));
        for (int i=1;i<=32;++i)
            check(std::isnan(expected) ? std::isnan(out[i]) :
                std::bit_cast<uint32_t>(expected)==std::bit_cast<uint32_t>(out[i]),"FP8 reference mismatch");
        check(out[0]==12345.0f && out[33]==12345.0f,"output guard overwritten");
    }
    // New MXFP8 rows retain the producer's original 33-byte representation.
    block[0]=127;
    for(int c=0;c<256;++c){
        std::fill(block.begin()+1,block.end(),uint8_t(c));
        std::array<float,32> a,b;
        ds::dequantize_engram_row(43,block.data(),32,a.data());
        ds::dequantize_engram_row(67,block.data(),32,b.data());
        check(std::memcmp(a.data(),b.data(),sizeof(a))==0,"MXFP8 Engram layout mismatch");
    }
    std::array<uint8_t,330> rows{};
    for (int b=0;b<10;++b) { rows[b*33]=uint8_t(126+b); rows[b*33+1]=0x38; }
    std::array<float,320> values;
    ds::dequantize_engram_row(67,rows.data(),320,values.data());
    for(int b=0;b<10;++b) check(values[b*32]==std::ldexp(1.0f,b-1),"block stride");
    rejects([&]{ds::dequantize_engram_row(67,rows.data(),33,values.data());});
    rejects([&]{ds::dequantize_engram_row(67,nullptr,32,values.data());});
    rejects([&]{ds::dequantize_engram_row(68,rows.data(),32,values.data());});
    check(ds::engram_row_bytes(67,0)==0 && ds::engram_row_bytes(67,UINT64_MAX-31)==0,"row size overflow");
    strata::TensorInfo t{"blk.1.engram_embd.weight",{256,384006168},67,0};
    ds::TensorRef r{nullptr,&t}; check(r.bytes()==101377628352ull,"large table bytes");
    t.shape[1]=UINT64_MAX;check(r.bytes()==0,"table size overflow");
    t.shape={256,1,1};check(r.bytes()==0,"non-matrix accepted");
    t.shape={256,1};t.name="blk.1.attn_q.weight";check(r.bytes()==0,"global type-67 interpretation leaked");
    int elems=0,bytes=0;check(!strata::block_geometry(67,elems,bytes),"global type registry changed");
    // The Q4_K adapter must preserve the existing decoder byte-for-byte.
    std::array<uint8_t,288> q4{};
    for(size_t i=0;i<q4.size();++i)q4[i]=uint8_t(i*7);
    q4[0]=q4[144]=0;q4[1]=q4[145]=0x3c; // FP16 scale = 1
    q4[2]=q4[146]=0;q4[3]=q4[147]=0;
    std::array<float,512> old,now;
    strata::dequantize_q4_K(q4.data(),old.data());strata::dequantize_q4_K(q4.data()+144,old.data()+256);
    ds::dequantize_engram_row(12,q4.data(),512,now.data());
    check(std::memcmp(old.data(),now.data(),sizeof(old))==0,"legacy Q4_K changed");
    std::array<uint8_t,36> iq4{};
    for(size_t i=0;i<iq4.size();++i)iq4[i]=uint8_t(i*19);
    iq4[0]=iq4[18]=0;iq4[1]=iq4[19]=0x38;
    std::array<float,64> expected,actual;
    strata::dequantize_iq4_nl(iq4.data(),expected.data());strata::dequantize_iq4_nl(iq4.data()+18,expected.data()+32);
    ds::dequantize_engram_row(20,iq4.data(),64,actual.data());
    check(std::memcmp(expected.data(),actual.data(),sizeof(expected))==0,"IQ4_NL Engram row stride");
}
std::vector<fixture::Kv> config(bool aliases) {
    using namespace fixture;
    std::vector<Kv> v{str("general.architecture","deepseek41"),str("deepseek41.rope.scaling.type","yarn"),
        boolean("deepseek41.expert_weights_norm",true),strs("tokenizer.ggml.tokens",{"a","b"}),
        u32s("deepseek41.attention.compress_ratios",{0,0}),u32s("deepseek41.engram.token_map",{0,1}),
        u32s("deepseek41.engram.multipliers",{3,5,7,9}),u32s("deepseek41.engram.primes",{101,103,107,109,113,127}),
        u32s("deepseek41.engram.offsets",{0,101,204,311,420,533}),
        u32s(aliases?"deepseek41.engram.layers":"deepseek41.engram.layer_ids",{1}),
        u32(aliases?"deepseek41.engram.head_dim":"deepseek41.engram.key_length",256),
        u32(aliases?"deepseek41.engram.pad_token_id":"deepseek41.engram.pad_id",2)};
    for (auto [k,n]:std::vector<std::pair<const char*,uint64_t>>{
        {"block_count",2},{"embedding_length",5120},{"attention.head_count",64},{"attention.head_count_kv",1},
        {"attention.key_length",512},{"attention.value_length",512},{"rope.dimension_count",64},
        {"attention.q_lora_rank",1280},{"attention.output_lora_rank",1024},{"attention.output_group_count",8},
        {"attention.sliding_window",128},{"expert_count",384},{"expert_used_count",6},{"expert_feed_forward_length",2304},
        {"expert_shared_count",1},{"expert_gating_func",4},{"hyper_connection.count",4},
        {"hyper_connection.sinkhorn_iterations",20},{"attention.indexer.head_count",32},{"attention.indexer.key_length",128},
        {"attention.indexer.top_k",512},{"context_length",1048576},{"rope.scaling.original_context_length",65536},
        {"engram.head_count",2},{"engram.max_ngram_size",4}}) v.push_back(u32(std::string("deepseek41.")+k,n));
    for(auto [k,n]:std::vector<std::pair<const char*,float>>{
        {"expert_weights_scale",1.5f},{"hyper_connection.epsilon",1e-6f},{"attention.layer_norm_rms_epsilon",1e-20f},
        {"rope.freq_base",10000},{"attention.compress_rope_freq_base",160000},{"rope.scaling.factor",16},
        {"rope.scaling.yarn_beta_fast",32},{"rope.scaling.yarn_beta_slow",1},{"swiglu_clamp_exp",10}})
        v.push_back(f32(std::string("deepseek41.")+k,n));
    return v;
}
void aliases() {
    const auto path=std::filesystem::temp_directory_path()/"strata-deepseek41-engram-aliases.gguf";
    auto read=[&](const auto& kv) {fixture::write(path,kv,{});strata::GgufFile f(path.string());return ds::Config::read(f);};
    try {
        for(bool alternate:{false,true}) {
            const auto c=read(config(alternate));
            check(c.engram.head_dim==256 && c.engram.pad_id==2 && c.engram.layers==std::vector<uint64_t>{1},"Engram aliases");
        }
        auto both=config(true);both.push_back(fixture::u32("deepseek41.engram.key_length",256));
        both.push_back(fixture::u32("deepseek41.engram.pad_id",2));both.push_back(fixture::u32s("deepseek41.engram.layer_ids",{1}));
        check(read(both).engram.head_dim==256,"equal aliases rejected");
        for(const auto& conflict:std::vector<fixture::Kv>{fixture::u32("deepseek41.engram.key_length",512),
            fixture::u32("deepseek41.engram.pad_id",3),fixture::u32s("deepseek41.engram.layer_ids",{0})}) {
            auto bad=config(true);bad.push_back(conflict);rejects([&]{read(bad);});
        }
        auto bad=config(true);
        for(auto& kv:bad)if(kv.key=="deepseek41.engram.head_dim")kv=fixture::boolean(kv.key,true);
        rejects([&]{read(bad);});
        std::filesystem::remove(path);
    } catch(...) {std::filesystem::remove(path);throw;}
}
void ple() {
    const auto path=std::filesystem::temp_directory_path()/"strata-deepseek41-ple.gguf";
    auto kv=config(true);
    kv.erase(std::remove_if(kv.begin(),kv.end(),[](const auto& x){
        return x.key.starts_with("deepseek41.engram.") && x.key!="deepseek41.engram.token_map" && x.key!="deepseek41.engram.pad_token_id";
    }),kv.end());
    for(auto& x:kv){
        if(x.key=="deepseek41.engram.pad_token_id")x=fixture::u32(x.key,1);
        if(x.key=="deepseek41.engram.token_map")x=fixture::u32s(x.key,{17,23});
    }
    kv.push_back(fixture::u32s("deepseek41.ple.layers",{1}));
    kv.push_back(fixture::u32("deepseek41.ple.heads_per_ngram",2));
    kv.push_back(fixture::u32("deepseek41.ple.ngram_size",4));
    kv.push_back(fixture::u32("deepseek41.embedding_length_per_layer_input",256));
    kv.push_back(fixture::u64s("deepseek41.ple.layer_multipliers",{UINT64_MAX-2,5,7,9}));
    const std::vector<uint64_t> primes{101,103,107,109,113,127},offsets{1000,1101,1204,1311,1420,1533};
    kv.push_back(fixture::u64s("deepseek41.ple.head_vocab_sizes",primes));
    kv.push_back(fixture::u64s("deepseek41.ple.head_offsets",offsets));
    auto read=[&](const auto& v){fixture::write(path,v,{});strata::GgufFile f(path.string());return ds::Config::read(f);};
    try {
        const auto e=read(kv).engram;
        check(e.table_offsets==std::vector<uint64_t>{1000} && e.pad_id==23,"PLE base/pad mapping");
        for(const auto& history:std::vector<std::vector<int32_t>>{{},{0},{0,1},{1,0,1},{0,-1,1}}){
            const auto rows=ds::engram_rows(e,0,1,history);
            uint64_t hash=23*(UINT64_MAX-2);bool blocked=false;
            for(int n=1;n<4;++n){
                const int32_t prior=size_t(n)<=history.size()?history[history.size()-n]:-1;
                blocked=blocked || prior<0;
                hash^=uint64_t(blocked?23:(prior==0?17:23))*uint64_t(n==1?5:n==2?7:9);
                for(int h=0;h<2;++h){const int at=(n-1)*2+h;
                    check(uint64_t(rows[at])+1000==hash%primes[at]+offsets[at],"PLE global/local hash mismatch");}
            }
        }
        auto bad=kv;bad.push_back(fixture::u32("deepseek41.engram.head_count",2));rejects([&]{read(bad);});
        bad=kv;for(auto& x:bad)if(x.key=="deepseek41.engram.pad_token_id")x=fixture::u32(x.key,2);
        rejects([&]{read(bad);});std::filesystem::remove(path);
    }catch(...){std::filesystem::remove(path);throw;}
}
int main() {
    try {decode();aliases();ple();std::cout<<"Engram FP8 65536 combinations, guards, aliases/conflicts, payload overflow Q4/IQ4/MXFP8 compatibility and PLE hashing PASS\n";}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
