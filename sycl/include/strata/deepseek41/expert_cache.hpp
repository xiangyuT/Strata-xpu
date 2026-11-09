#pragma once
#include "strata/deepseek41/model.hpp"
#include "strata/deepseek41/resident.hpp"
#include <cstdint>
#include <memory>

namespace strata::deepseek41 {
class ExpertHost;
// In-order queue: consumers must be queued before any later slot writes.
// Slot writes follow queued consumers. Two host staging buffers are reused
// only after their upload events complete, without draining queued compute.
// Original gate/up/down bytes are copied without format conversion.
class ExpertCache {
public:
    struct Stats {
        uint64_t route_entries=0, groups=0, hits=0, uploads=0, bytes=0, cpu_misses=0;
    };
    ExpertCache(const Model&,const ResidentWeights&,void* queue,int budget_mib=-1);
    // Borrow an existing host pool only for synchronous acquire-time packing.
    ExpertCache(const Model&,const ResidentWeights&,void* queue,int budget_mib,ExpertHost* stager);
    ExpertCache(const Model&,const ResidentWeights&,void* queue,int budget_mib,ExpertHost* stager,bool window_heat);
    // Optional transfer queue writes only private stages. Slot/scratch writes
    // and every consumer stay on the supplied in-order queue.
    ExpertCache(const Model&,const ResidentWeights&,void* queue,int budget_mib,ExpertHost* stager,bool window_heat,bool staged_transfer);
    ~ExpertCache();
    // Executed speculative routes count as work but do not train admission.
    void observe(int layer,const int32_t* ids,int count,bool committed=true);
    // Train only the accepted prefix, without counting its work twice.
    void commit(int layer,const int32_t* ids,int count);
    // A non-admitted miss may be computed on the CPU instead of uploaded.
    // Hits and cache admissions keep their existing device path.
    const uint8_t* acquire(int layer,int expert,bool upload_transient=true);
    // The caller may retain direct uploads where there is no profitable batch.
    const uint8_t* acquire(int layer,int expert,bool upload_transient,bool allow_pipeline);
    // Read-only prediction: no heat, slot, queue or counter changes.
    // A batch may defer device consumers only if every acquire needs no upload.
    bool requires_upload(int layer,int expert,bool upload_transient=true) const;
    Stats stats() const;
    size_t allocated_bytes() const;
    size_t slots() const;
    void report() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
} // namespace strata::deepseek41
