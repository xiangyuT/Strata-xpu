// Same native IQ4_XS geometry/arithmetic, with isolated lookup alternatives.
// Mode 2 is a counterfactual cost control, NOT a numerically valid candidate.
#define DPCT_PROFILING_ENABLED
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq4_lut.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_mmvq_tuning.hpp"
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using namespace strata::kernels;
constexpr int Guard = 128, MaxT = 8;
static int candidate_mode = 1;
struct IQ4Block { sycl::half d; uint16_t scales_h; uint8_t scales_l[4], qs[128]; };
struct ActBlock { sycl::half2 ds; int8_t qs[32]; };
static_assert(sizeof(IQ4Block) == 136 && offsetof(IQ4Block, qs) == 8);
static_assert(sizeof(ActBlock) == 36 && offsetof(ActBlock, qs) == 4);
struct Fixture { int N,K; std::string name; IQ4Block* w; float *x,*out; ActBlock* xq; };

void run(Fixture& f,sycl::queue& q,int T,int mode){
    native_iq4_xs_decode_probe(mode,f.w,f.xq,f.out+Guard,f.K,f.N,T,&q);
}
void reset(Fixture& f,sycl::queue& q,int T,int pattern){
    std::vector<float> x((size_t)T*f.K);
    for(size_t i=0;i<x.size();++i){const float v=float(int((i*7919+17)%2001)-1000)/1000.f;
        x[i]=pattern==1?0.f:pattern==2?v*1e-20f:pattern==3?v*1000.f:pattern==4?std::copysign(0.f,v):v;}
    q.memcpy(f.x,x.data(),x.size()*4);native_quantize_q8_1(f.x,f.xq,f.K,T,&q);
    q.fill(f.out,13.75f,(size_t)MaxT*f.N+2*Guard);q.wait_and_throw();
}
std::vector<float> read(Fixture& f,sycl::queue& q,int T){
    std::vector<float> y((size_t)MaxT*f.N+2*Guard);q.memcpy(y.data(),f.out,y.size()*4).wait_and_throw();
    for(size_t i=0;i<y.size();++i){if(!std::isfinite(y[i]))throw std::runtime_error("nonfinite output");
        if((i<Guard||i>=Guard+(size_t)T*f.N)&&y[i]!=13.75f)throw std::runtime_error("guard/inactive output changed");}
    return y;
}
void exhaustive(sycl::queue& q){
    constexpr int n=65536;auto* out=sycl::malloc_device<sycl::uint4>(n,q);
    q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> id){uint32_t v=id[0];
        const uint32_t x=(v&15)|((v&240)<<4)|((v&3840)<<8)|((v&61440)<<12)|((v*7919)&0xf0f0f0f0);
        out[v]={iq4_decode::scalar(x),iq4_decode::bit_select(x),iq4_decode::bit_select_add_xor(x),0};});
    std::vector<sycl::uint4> h(n);q.memcpy(h.data(),out,n*sizeof(sycl::uint4)).wait_and_throw();
    const int values[]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    for(uint32_t v=0;v<n;++v){uint32_t expected=0;for(int j=0;j<4;++j)expected|=(uint32_t(values[(v>>(4*j))&15])&255)<<(8*j);
        if(h[v].x()!=expected||h[v].y()!=expected||h[v].z()!=expected)throw std::runtime_error("exhaustive codebook mismatch");}
    sycl::free(out,q);std::printf("{\"kind\":\"LUT_correctness\",\"cases\":65536,\"all_four_byte_indices\":true,\"upper_bits_ignored\":true,\"byte_exact\":true}\n");std::fflush(stdout);
}
void parity(Fixture& f,sycl::queue& q){
    for(int T=1;T<=8;++T)for(int pattern=0;pattern<5;++pattern){
        reset(f,q,T,pattern);native_mmvq(23,f.w,f.xq,f.out+Guard,f.K,f.N,T,&q);const auto ref=read(f,q,T);
        q.fill(f.out,13.75f,(size_t)MaxT*f.N+2*Guard);
        dpct::experimental::command_graph_ptr graph=nullptr;dpct::experimental::begin_recording(&q);run(f,q,T,candidate_mode);
        dpct::experimental::end_recording(&q,&graph);auto exec=graph->finalize();q.ext_oneapi_graph(exec);q.wait_and_throw();
        const auto cand=read(f,q,T);delete graph;
        if(std::memcmp(ref.data(),cand.data(),ref.size()*4)){
            size_t first=0;while(first<ref.size()&&std::memcmp(&ref[first],&cand[first],4)==0)++first;
            uint32_t rb=0,cb=0;std::memcpy(&rb,&ref[first],4);std::memcpy(&cb,&cand[first],4);
            std::fprintf(stderr,"first mismatch index=%zu native=%g(%08x) candidate=%g(%08x)\n",first,ref[first],rb,cand[first],cb);
            throw std::runtime_error("candidate/native mismatch "+f.name+" T="+std::to_string(T)+" pattern="+std::to_string(pattern));
        }
    }
    std::printf("{\"kind\":\"matrix_correctness\",\"name\":\"%s\",\"N\":%d,\"K\":%d,\"cases\":40,\"native_baseline_and_candidate\":\"byte-exact\",\"candidate_graph\":true,\"guards\":true,\"finite\":true}\n",f.name.c_str(),f.N,f.K);std::fflush(stdout);
}
double measure(std::vector<Fixture*>& fs,sycl::queue& q,int T,int mode,const char* arm){
    for(auto* f:fs)reset(*f,q,T,0);
    dpct::experimental::command_graph_ptr graph=nullptr;dpct::experimental::begin_recording(&q);
    for(auto* f:fs)run(*f,q,T,mode);dpct::experimental::end_recording(&q,&graph);auto exec=graph->finalize();
    const int per=fs.size();auto start=Clock::now();int calls=0;
    do{for(int i=0;i<16;++i){q.ext_oneapi_graph(exec);calls+=per;}q.wait_and_throw();}
    while(calls<1000||std::chrono::duration<double>(Clock::now()-start).count()<2);
    const int batch=std::max(per*16,int(.15*calls/std::chrono::duration<double>(Clock::now()-start).count()/per)*per);
    start=Clock::now();calls=0;int chunk=0;double device_us=0;
    do{auto before=Clock::now();auto first=q.ext_oneapi_submit_barrier();for(int i=0;i<batch/per;++i)q.ext_oneapi_graph(exec);
        auto last=q.ext_oneapi_submit_barrier();last.wait_and_throw();
        const double span=(last.get_profiling_info<sycl::info::event_profiling::command_start>()-first.get_profiling_info<sycl::info::event_profiling::command_end>())/1000.0/batch;
        const double wall=std::chrono::duration<double,std::micro>(Clock::now()-before).count()/batch;
        std::printf("{\"kind\":\"graph_chunk\",\"arm\":\"%s\",\"N\":%d,\"K\":%d,\"T\":%d,\"mode\":%d,\"weight_sets\":%d,\"chunk\":%d,\"calls\":%d,\"event_span_us\":%.6f,\"wall_us\":%.6f}\n",arm,fs[0]->N,fs[0]->K,T,mode,per,chunk++,batch,span,wall);calls+=batch;device_us+=span*batch;
    }while(calls<2000||std::chrono::duration<double>(Clock::now()-start).count()<3);
    std::printf("{\"kind\":\"block_complete\",\"arm\":\"%s\",\"N\":%d,\"K\":%d,\"T\":%d,\"mode\":%d,\"calls\":%d}\n",arm,fs[0]->N,fs[0]->K,T,mode,calls);std::fflush(stdout);delete graph;return device_us/calls;
}
int main(int argc,char** argv)try{
    if(argc<3||argc>5)throw std::runtime_error("usage: iq4_decode_cost_bench <GGUF> <draft_vocab.bin> [--micro] [--esimd|--group|--add-xor]");
    bool micro=false;
    for(int i=3;i<argc;++i){if(std::string(argv[i])=="--micro")micro=true;else if(std::string(argv[i])=="--esimd")candidate_mode=3;else if(std::string(argv[i])=="--group")candidate_mode=4;else if(std::string(argv[i])=="--add-xor")candidate_mode=5;else throw std::runtime_error("unexpected argument");}
    auto& q=dpct::get_in_order_queue();strata::GgufFile g(argv[1]);exhaustive(q);
    if(candidate_mode==3||candidate_mode==4){
        auto* device=sycl::malloc_device<uint32_t>(65536,q);
        if(candidate_mode==3)native_iq4_esimd_lut_probe(device,&q);else native_iq4_group_lut_probe(device,&q);
        std::vector<uint32_t> result(65536);q.memcpy(result.data(),device,result.size()*4).wait_and_throw();
        const int values[]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        for(uint32_t v=0;v<65536;++v){uint32_t expected=0;for(int j=0;j<4;++j)expected|=(uint32_t(values[(v>>(4*j))&15])&255)<<(8*j);
            if(result[v]!=expected)throw std::runtime_error("register exhaustive lookup mismatch");}
        sycl::free(device,q);std::printf("{\"kind\":\"register_LUT_correctness\",\"mode\":%d,\"cases\":65536,\"byte_exact\":true}\n",candidate_mode);std::fflush(stdout);
    }
    std::vector<Fixture> fixtures;
    auto add=[&](int K,int N,const std::string& name,const uint8_t* bytes){
        Fixture f{};f.K=K;f.N=N;f.name=name;const size_t size=(size_t)N*(K/256)*136;
        f.w=(IQ4Block*)sycl::malloc_device(size,q);f.x=sycl::malloc_device<float>((size_t)MaxT*K,q);
        f.xq=sycl::malloc_device<ActBlock>((size_t)MaxT*K/32,q);f.out=sycl::malloc_device<float>((size_t)MaxT*N+2*Guard,q);
        if(!f.w||!f.x||!f.xq||!f.out)throw std::bad_alloc();q.memcpy(f.w,bytes,size).wait_and_throw();fixtures.push_back(std::move(f));};
    for(const auto [K,N]:{std::pair{2560,512},{2560,640},{6144,2560},{2560,6144},{2560,10240},{2560,12288}}){
        int count=0;for(const auto& t:g.tensors())if(t.type==23&&t.shape==std::vector<uint64_t>{(uint64_t)K,(uint64_t)N}){
            add(K,N,t.name,g.tensor_data(t));if(++count==4)break;}
        if(!count)throw std::runtime_error("real shape weights missing");
    }
    const auto* head=g.find("output.weight");if(!head||head->type!=23||head->shape!=std::vector<uint64_t>{2560,248320})throw std::runtime_error("head contract");
    add(2560,248320,head->name,g.tensor_data(*head));
    std::ifstream vocab(argv[2],std::ios::binary);std::vector<int32_t> ids(106299);vocab.read((char*)ids.data(),ids.size()*4);
    if(!vocab||vocab.peek()!=EOF)throw std::runtime_error("draft subset count");
    const size_t row=10*136;std::vector<uint8_t> subset(ids.size()*row);
    for(size_t i=0;i<ids.size();++i){if(ids[i]<0||ids[i]>=248320)throw std::runtime_error("draft subset id");
        std::memcpy(subset.data()+i*row,g.tensor_data(*head)+(size_t)ids[i]*row,row);}
    add(2560,106299,"actual-MTP-draft-subset",subset.data());
    for(auto& f:fixtures)parity(f,q);
    std::printf("{\"kind\":\"all_correctness\",\"fixtures\":%zu,\"cases\":%zu,\"result\":\"passed\",\"control_mode2_semantically_valid\":false}\n",fixtures.size(),fixtures.size()*40);std::fflush(stdout);
    if(micro)for(const auto [K,N]:{std::pair{2560,10240},{6144,2560},{2560,6144},{2560,12288},{2560,106299},{2560,248320}}){
        std::vector<Fixture*> fs;for(auto& f:fixtures)if(f.N==N&&f.K==K)fs.push_back(&f);
        for(int T:{1,2,3,4}){if(N==106299&&T!=1)continue;
            const double before=measure(fs,q,T,0,"baseline-before");
            const double candidate=measure(fs,q,T,candidate_mode,candidate_mode==3?"esimd-register":candidate_mode==4?"group-register":candidate_mode==5?"bit-select-add-xor":"bit-select");
            const double after=measure(fs,q,T,0,"baseline-after");
            const double change=100*(candidate/((before+after)/2)-1),drift=100*(after/before-1);
            std::printf("{\"kind\":\"micro_cell\",\"N\":%d,\"K\":%d,\"T\":%d,\"change_pct\":%.6f,\"baseline_drift_pct\":%.6f}\n",N,K,T,change,drift);std::fflush(stdout);
            if(change>1||std::abs(drift)>1)throw std::runtime_error("micro rejected: regression or baseline drift above1pct");
            if(candidate_mode==1)measure(fs,q,T,2,"no-codebook-cost-control");}
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"IQ4 decode cost: %s\n",e.what());return 1;}
