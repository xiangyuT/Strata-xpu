// The native SYCL family uses the existing GGUF mappings as CPU backing.
// Engram gathers are sparse; do not allocate/lock a second 188 GiB table copy.
#include "strata/deepseek41/resident.hpp"
#include "strata/sycl_queue.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>

namespace strata::deepseek41 {
struct ResidentWeights::Impl {
    struct Range { const uint8_t* address; uint64_t bytes; };
    std::map<std::string,Range> ranges;
    uint64_t total=0;
    void add(const Model& m,const std::string& name) {
        const auto t=m.tensor(name); const auto bytes=t.bytes();
        if(!bytes || total>UINT64_MAX-bytes || !ranges.emplace(name,Range{t.data(),bytes}).second)
            throw std::runtime_error("invalid or duplicate mapped tensor: "+name);
        total+=bytes;
    }
};
ResidentWeights::ResidentWeights():p_(std::make_unique<Impl>()){}
ResidentWeights::~ResidentWeights()=default;
const uint8_t* ResidentWeights::data(const std::string& name) const {return p_->ranges.at(name).address;}
uint64_t ResidentWeights::bytes() const {return p_->total;}
void ResidentWeights::load(const Model& m) {
    if(!p_->ranges.empty())throw std::runtime_error("model backing already initialized");
    for(int l=0;l<m.config().layers;++l)for(const char* n:{"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"})
        p_->add(m,"blk."+std::to_string(l)+"."+n);
    for(auto l:m.config().engram.layers)p_->add(m,"blk."+std::to_string(l)+".engram_embd.weight");
    verify();
}
void ResidentWeights::verify() const {
    for(const auto& [name,r]:p_->ranges)if(!r.address || !r.bytes)throw std::runtime_error("invalid model mapping: "+name);
    std::fprintf(stderr,"strata-deepseek41-sycl: read-only backing %.3f GiB; demand paging, no full-RAM residency claim\n",double(p_->total)/(1ull<<30));
}
void copy_registered_slice(void* destination,const uint8_t* base,size_t offset,size_t bytes,void* stream) {
    if(!stream || !base || !destination || offset>SIZE_MAX-bytes)throw std::invalid_argument("invalid model upload");
    if(!bytes)return;
    auto& q=*strata::q_of(stream);const size_t capacity=std::min(bytes,size_t(4)<<20);
    auto* stage=static_cast<uint8_t*>(sycl::malloc_host(capacity,q));
    if(!stage)throw std::bad_alloc();
    try {
        for(size_t done=0;done<bytes;) {
            const size_t n=std::min(capacity,bytes-done);
            std::memcpy(stage,base+offset+done,n);
            q.memcpy(static_cast<uint8_t*>(destination)+done,stage,n).wait_and_throw();done+=n;
        }
    }catch(...){sycl::free(stage,q);throw;}
    sycl::free(stage,q);
}
} // namespace strata::deepseek41
