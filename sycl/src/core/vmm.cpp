// CUDA driver VMM is unavailable in this SYCL port. Keep the shared cache's empty VmmRange safe to destroy.
#include <cstddef>
#include "strata/core/vmm.hpp"
#include <cstdlib>

namespace strata::core {
bool vmm_available() { return false; }
uint64_t vmm_granularity() { return 0; }
VmmChunk vmm_chunk_new() { return 0; }
void vmm_chunk_free(VmmChunk h) { if (h != 0) std::abort(); }
bool VmmRange::reserve(uint64_t) { return false; }
void VmmRange::release() {
    if (base_ != 0 || !h_.empty()) std::abort();
}
int64_t VmmRange::mapped_count() const { return 0; }
VmmChunk VmmRange::unmap(int64_t) { return 0; }
bool VmmRange::map_one(int64_t, VmmChunk) { return false; }
bool VmmRange::set_access(int64_t, int64_t) { return false; }
bool VmmRange::commit_run(int64_t, int64_t) { return false; }
}  // namespace strata::core
