#include "strata/deepseek41/expert_cache.hpp"
#include "strata/deepseek41/expert_host.hpp"
#include "strata/deepseek41/window_heat.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::deepseek41 {
namespace { constexpr size_t MiB=1ull<<20; }
struct ExpertCache::Impl {
    struct Slot { int expert=-1; };
    struct Layer {
        size_t gate=0,up=0,down=0,blob=0;
        const uint8_t *g=nullptr,*u=nullptr,*d=nullptr;
        uint8_t* storage=nullptr;
        std::vector<Slot> slots;
        std::vector<int> lookup;
        std::vector<uint64_t> heat,age;
        std::vector<uint32_t> frequency;
        uint64_t clock=0;
    };
    sycl::queue& q;
    std::unique_ptr<sycl::queue> transfer;
    std::vector<Layer> layers;
    struct Stage {
        uint8_t* host=nullptr;
        uint8_t* device=nullptr;
        sycl::event uploaded,consumed;
        bool pending=false,read_pending=false;
    };
    std::array<Stage,2> stages;
    size_t next_stage=0;
    uint8_t* scratch=nullptr;
    size_t capacity=0,allocated=0,slot_count=0;
    Stats counters;
    int experts,top_k;
    bool window_heat;
    ExpertHost* host_stager;

    Impl(const Model& model,const ResidentWeights& backing,void* queue,int mib,ExpertHost* stager,bool window,bool pipelined)
        :q(*static_cast<sycl::queue*>(queue)),experts(model.config().experts),top_k(model.config().top_k),window_heat(window),host_stager(stager) {
        if(mib < -1)throw std::invalid_argument("expert cache MiB must be -1 (auto), zero or positive");
        if(window_heat && top_k<1)throw std::invalid_argument("expert cache requires positive top_k");
        try {
            if(pipelined)transfer=std::make_unique<sycl::queue>(q.get_context(),q.get_device(),
                [](sycl::exception_list errors){for(auto& error:errors)std::rethrow_exception(error);},
                sycl::property::queue::in_order{});
            layers.resize(model.config().layers);
            for(size_t l=0;l<layers.size();++l){
                auto& c=layers[l];const auto p="blk."+std::to_string(l)+".";
                const auto g=model.tensor(p+"ffn_gate_exps.weight"),u=model.tensor(p+"ffn_up_exps.weight"),d=model.tensor(p+"ffn_down_exps.weight");
                c.gate=g.bytes()/experts;c.up=u.bytes()/experts;c.down=d.bytes()/experts;
                c.blob=c.gate+c.up+c.down;capacity=std::max(capacity,c.blob);
                c.g=backing.data(p+"ffn_gate_exps.weight");c.u=backing.data(p+"ffn_up_exps.weight");c.d=backing.data(p+"ffn_down_exps.weight");
                c.lookup.assign(experts,-1);c.heat.assign(experts,0);c.age.assign(experts,0);
                if(window_heat)c.frequency.assign(experts,0);
            }
            for(auto& stage:stages){
                stage.host=sycl::malloc_host<uint8_t>(capacity,q);
                if(!stage.host)throw std::bad_alloc();
                if(transfer){
                    stage.device=sycl::malloc_device<uint8_t>(capacity,q);
                    if(!stage.device)throw std::bad_alloc();
                }
            }
            scratch=sycl::malloc_device<uint8_t>(capacity,q);
            if(!scratch)throw std::bad_alloc();
            size_t budget=0;
            if(mib){
                const auto device=q.get_device();
                if(!device.has(sycl::aspect::ext_intel_free_memory)){
                    if(mib<0)throw std::runtime_error("automatic expert cache sizing requires free-memory reporting; use --expert-cache-mib with an explicit budget or zero");
                    budget=size_t(mib)*MiB;
                }else{
                    const size_t free=device.get_info<sycl::ext::intel::info::device::free_memory>();
                    budget=free>1536*MiB?free-1536*MiB:0;
                    if(mib>0)budget=std::min(budget,size_t(mib)*MiB);
                }
            }
            size_t left=budget;bool added=true;
            // Same round-robin per-layer capacity distribution as the community
            // cache. Reserve is applied after dense, KV and batch scratch exist.
            while(added){
                added=false;
                for(auto& l:layers)if(l.slots.size()<size_t(experts) && l.blob<=left){
                    l.slots.emplace_back();left-=l.blob;added=true;
                }
            }
            for(auto& l:layers)if(!l.slots.empty()){
                l.storage=sycl::malloc_device<uint8_t>(l.slots.size()*l.blob,q);
                if(!l.storage)throw std::bad_alloc();
                allocated+=l.slots.size()*l.blob;slot_count+=l.slots.size();
            }
            std::fprintf(stderr,"strata-deepseek41-sycl: expert cache slots=%zu MiB=%.1f reserve_MiB=1536; raw GGUF, ordered queue\n",slot_count,double(allocated)/MiB);
            if(transfer)std::fprintf(stderr,"strata-deepseek41-sycl: expert transfer stages=%zu extra_device_MiB=%.3f; slot writes remain ordered\n",stages.size(),double(stages.size()*capacity)/MiB);
        }catch(...){cleanup();throw;}
    }
    void cleanup(){
        try{q.wait_and_throw();}catch(...){}
        if(transfer)try{transfer->wait_and_throw();}catch(...){}
        for(auto& l:layers)if(l.storage)sycl::free(l.storage,q);
        if(scratch)sycl::free(scratch,q);
        for(auto& stage:stages){
            if(stage.device)sycl::free(stage.device,q);
            if(stage.host)sycl::free(stage.host,q);
        }
    }
    ~Impl(){cleanup();}
    void train(int layer,const int32_t* ids,int count){
        if(count<1 || !ids)throw std::invalid_argument("empty expert observation");
        auto& l=layers.at(size_t(layer));
        // Validate the whole batch before changing metadata.
        for(int j=0;j<count;++j)if(ids[j]<0 || ids[j]>=experts)throw std::invalid_argument("expert route out of range");
        if(window_heat){
            // At most eight accepted tokens per heat update, bounded by the
            // portable helper's 48-route contract. Validate all chunks above
            // before changing anything. This metadata has one submitting thread.
            const int chunk=int(std::min<size_t>(size_t(top_k)*8,48));
            for(int first=0;first<count;first+=chunk)
                observe_window_heat(l.heat,l.age,l.frequency,l.clock,ids+first,std::min(chunk,count-first));
            return;
        }
        ++l.clock;if(l.clock%128==0)for(auto& h:l.heat)h=h*7/10;
        for(int j=0;j<count;++j){l.heat[ids[j]]+=256;l.age[ids[j]]=l.clock;}
    }
    void observe(int layer,const int32_t* ids,int count,bool committed){
        if(committed)train(layer,ids,count);
        else {
            if(count<1 || !ids)throw std::invalid_argument("empty expert observation");
            layers.at(size_t(layer));
            for(int j=0;j<count;++j)if(ids[j]<0 || ids[j]>=experts)throw std::invalid_argument("expert route out of range");
        }
        counters.route_entries+=count;
    }
    int admission_slot(const Layer& l,int e) const {
        int victim=-1;uint64_t cold=UINT64_MAX,old=UINT64_MAX;
        // Retained occurrence counts guard window admission. Keep the existing
        // least-heat/oldest victim order and replacement margin in both modes.
        const bool learned=window_heat?l.frequency[e]>=2:l.heat[e]>=512;
        if(learned)for(size_t j=0;j<l.slots.size();++j){
            const int other=l.slots[j].expert;
            if(other<0){victim=int(j);break;}
            if(l.heat[other]<cold || (l.heat[other]==cold && l.age[other]<old)){
                victim=int(j);cold=l.heat[other];old=l.age[other];
            }
        }
        if(victim>=0 && l.slots[size_t(victim)].expert>=0 && l.heat[e]<cold+384)victim=-1;
        return victim;
    }
    bool requires_upload(int layer,int e,bool upload_transient) const {
        const auto& l=layers.at(size_t(layer));
        if(e<0 || e>=experts)throw std::invalid_argument("expert route out of range");
        return l.lookup[e]<0 && (upload_transient || admission_slot(l,e)>=0);
    }
    const uint8_t* acquire(int layer,int e,bool upload_transient,bool allow_pipeline){
        auto& l=layers.at(size_t(layer));
        if(e<0 || e>=experts)throw std::invalid_argument("expert route out of range");
        ++counters.groups;
        if(l.lookup[e]>=0){++counters.hits;return l.storage+size_t(l.lookup[e])*l.blob;}
        const int victim=admission_slot(l,e);
        if(victim<0 && !upload_transient){++counters.cpu_misses;return nullptr;}
        // Only the previous DMA reading this host buffer must finish before
        // CPU reuse. Device-slot/scratch reuse is ordered after all consumers
        // by the in-order queue; it needs no whole-queue host fence.
        auto& stage=stages[next_stage];
        next_stage=(next_stage+1)%stages.size();
        if(stage.pending)stage.uploaded.wait_and_throw();
        uint8_t* destination=scratch;
        if(victim>=0){
            auto& slot=l.slots[size_t(victim)];
            if(slot.expert>=0)l.lookup[slot.expert]=-1;
            slot.expert=e;l.lookup[e]=victim;destination=l.storage+size_t(victim)*l.blob;
        }
        if(host_stager)host_stager->copy_expert(stage.host,l.g+size_t(e)*l.gate,l.gate,
            l.u+size_t(e)*l.up,l.up,l.d+size_t(e)*l.down,l.down);
        else{
            std::memcpy(stage.host,l.g+size_t(e)*l.gate,l.gate);
            std::memcpy(stage.host+l.gate,l.u+size_t(e)*l.up,l.up);
            std::memcpy(stage.host+l.gate+l.up,l.d+size_t(e)*l.down,l.down);
        }
        if(transfer && allow_pipeline){
            // Only private stages are written from the transfer queue. Reusing
            // a stage waits on its previous main-queue read, not on all compute.
            auto* staged=stage.device;auto* source=stage.host;const auto bytes=l.blob;
            const auto previous=stage.consumed;const bool reuse=stage.read_pending;
            stage.uploaded=transfer->submit([=](sycl::handler& handler){
                if(reuse)handler.depends_on(previous);
                handler.memcpy(staged,source,bytes);
            });
            const auto uploaded=stage.uploaded;
            // This copy follows every earlier consumer before replacing its
            // original slot/scratch. Later consumers follow it in the same queue.
            stage.consumed=q.submit([=](sycl::handler& handler){
                handler.depends_on(uploaded);
                handler.memcpy(destination,staged,bytes);
            });
            stage.read_pending=true;
        }else stage.uploaded=q.memcpy(destination,stage.host,l.blob);
        stage.pending=true;
        ++counters.uploads;counters.bytes+=l.blob;
        return destination;
    }
};
ExpertCache::ExpertCache(const Model& m,const ResidentWeights& r,void* q,int mib):ExpertCache(m,r,q,mib,nullptr){}
ExpertCache::ExpertCache(const Model& m,const ResidentWeights& r,void* q,int mib,ExpertHost* stager):ExpertCache(m,r,q,mib,stager,false){}
ExpertCache::ExpertCache(const Model& m,const ResidentWeights& r,void* q,int mib,ExpertHost* stager,bool window):ExpertCache(m,r,q,mib,stager,window,false){}
ExpertCache::ExpertCache(const Model& m,const ResidentWeights& r,void* q,int mib,ExpertHost* stager,bool window,bool transfer){
    if(!q)throw std::invalid_argument("expert cache requires an in-order queue");
    auto& queue=*static_cast<sycl::queue*>(q);
    if(!queue.has_property<sycl::property::queue::in_order>())throw std::invalid_argument("expert cache requires an in-order queue");
    p_=std::make_unique<Impl>(m,r,q,mib,stager,window,transfer);
}
ExpertCache::~ExpertCache()=default;
void ExpertCache::observe(int l,const int32_t* ids,int count,bool committed){p_->observe(l,ids,count,committed);}
void ExpertCache::commit(int l,const int32_t* ids,int count){p_->train(l,ids,count);}
const uint8_t* ExpertCache::acquire(int l,int e,bool upload_transient){return p_->acquire(l,e,upload_transient,true);}
const uint8_t* ExpertCache::acquire(int l,int e,bool upload_transient,bool pipeline){return p_->acquire(l,e,upload_transient,pipeline);}
bool ExpertCache::requires_upload(int l,int e,bool upload)const{return p_->requires_upload(l,e,upload);}
ExpertCache::Stats ExpertCache::stats()const{return p_->counters;}
size_t ExpertCache::allocated_bytes()const{return p_->allocated;}
size_t ExpertCache::slots()const{return p_->slot_count;}
void ExpertCache::report()const{
    const auto s=stats();
    std::fprintf(stderr,"strata-deepseek41-sycl: expert routes=%llu groups=%llu batch_reuse=%llu hits=%llu uploads=%llu bytes=%llu cpu_misses=%llu\n",
        (unsigned long long)s.route_entries,(unsigned long long)s.groups,(unsigned long long)(s.route_entries-s.groups),
        (unsigned long long)s.hits,(unsigned long long)s.uploads,(unsigned long long)s.bytes,(unsigned long long)s.cpu_misses);
}
} // namespace strata::deepseek41
