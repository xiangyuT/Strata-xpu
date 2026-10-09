// src/core/gguf_expert_source.cpp - see the header. Plain C++, no device code.
#include "strata/core/gguf_expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include <thread>
#include <atomic>
#include <algorithm>
#include <utility>
#include <stdexcept>

namespace strata::core {

namespace {
constexpr size_t kRing = 512;   // blobs alive at once: the prompt path holds a layer's worth of streamed experts
}

GgufExpertSource::~GgufExpertSource() { close(); }

void GgufExpertSource::close() {
    for (auto* region : mirror_regions_) sycl::free(region, dpct::get_in_order_queue());
    mirror_regions_.clear();
    mirror_bytes_ = 0; mirror_ptr_.clear(); layer_first_.clear();
    for (int fd : fds_) if (fd >= 0) ::close(fd);
    fds_.clear(); names_.clear(); layer_fd_.clear(); ring_.clear(); ring_key_.clear(); where_.clear();
    ring_next_ = 0;
}

bool GgufExpertSource::open(const std::string& shard1, int64_t n_layers, int64_t n_expert, std::string& err) {
    close();
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native || lay.gguf_off.size() < (size_t) (3 * n_layers)) {
        err = "--stream-experts needs a native (IQ) pack whose native_experts.txt carries the GGUF tensor offsets";
        return false;
    }
    for (int64_t l = 0; l < n_layers; ++l)
        for (int r = 0; r < 3; ++r)
            if (lay.gguf_off[(size_t) (3 * l + r)] == 0) {
                err = "--stream-experts: layer " + std::to_string(l) + " has no GGUF offset for its experts";
                return false;
            }
    shard_ = shard1;
    const size_t cut = shard1.find_last_of("/\\");
    dir_ = cut == std::string::npos ? std::string() : shard1.substr(0, cut + 1);
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    layer_fd_.assign((size_t) (3 * n_layers), -1);
    for (int64_t l = 0; l < n_layers; ++l)
        for (int r = 0; r < 3; ++r)
            if (fd_of(l, r, err) < 0) { close(); return false; }
    ring_.resize(kRing);
    for (auto& b : ring_) b.resize((size_t) lay.max_blob);
    ring_key_.assign(kRing, -1);
    return true;
}

// The file of role `role` (0 gate, 1 up, 2 down) of `layer`. ExpertLayout::gguf_file is per layer AND role
// (`3 * layer + role`) since upstream 0.1.31; indexing it by layer alone read a shard-2 layer from shard 1 (Swift 1.5,
// whose layers 13-47 are in shard 2: garbage IQ1_M scales, NaN logits - 2026-10-01).
int GgufExpertSource::fd_of(int64_t layer, int role, std::string& err) {
    const size_t i = (size_t) (3 * layer + role);
    if (layer_fd_[i] >= 0) return fds_[(size_t) layer_fd_[i]];
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::string name = shard_;
    if (lay.gguf_file.size() > i && !lay.gguf_file[i].empty()) name = dir_ + lay.gguf_file[i];
    for (size_t k = 0; k < names_.size(); ++k)
        if (names_[k] == name) { layer_fd_[i] = (int) k; return fds_[k]; }
    const int fd = ::open(name.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) { err = "--stream-experts: cannot open " + name; return -1; }
    names_.push_back(name);
    fds_.push_back(fd);
    layer_fd_[i] = (int) fds_.size() - 1;
    return fd;
}

const uint8_t* GgufExpertSource::blob(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_ || ring_.empty()) return nullptr;
    if (!mirror_ptr_.empty()) {
        const auto* p = mirror_ptr_[(size_t) (layer * n_expert_ + expert)];
        if (p != nullptr) return p;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t blob = lay.bytes[(size_t) layer];
    // the arena loader's gather, for one expert: [gate | up | down] from the three tensors
    const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    const int64_t key = ((int64_t) layer << 20) | expert;
    size_t slot;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = where_.find(key);
        if (it != where_.end()) { ++reads_; return ring_[it->second].data(); }   // still resident
        slot = ring_next_;
        ring_next_ = (ring_next_ + 1) % ring_.size();
        if (ring_key_[slot] >= 0) where_.erase(ring_key_[slot]);
        ring_key_[slot] = key;
        where_[key] = slot;
    }
    std::vector<uint8_t>& buf = ring_[slot];
    for (int r = 0; r < 3; ++r) {
        const int fd = fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]];
        const uint64_t src = lay.gguf_off[(size_t) (3 * layer + r)] + per[r] * (uint64_t) expert;
        uint64_t done = 0;
        while (done < per[r]) {
            const ssize_t n = ::pread(fd, buf.data() + at[r] + done, (size_t) (per[r] - done), (off_t) (src + done));
            if (n <= 0) { std::lock_guard<std::mutex> lk(mu_); where_.erase(key); ring_key_[slot] = -1; return nullptr; }
            done += (uint64_t) n;
        }
    }
    ++reads_;
    return buf.data();
}

int64_t GgufExpertSource::mirror(const std::vector<std::pair<int64_t, int64_t>>& pairs, uint64_t cap, int threads,
                                 std::string& err) {
    err.clear();
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<std::pair<int64_t, int64_t>> take;
    std::vector<uint64_t> offs;
    uint64_t total = 0;
    for (const auto& [l, e] : pairs) {
        if (l < 0 || l >= n_layers_ || e < 0 || e >= n_expert_) continue;
        const uint64_t b = (lay.bytes[(size_t) l] + 255) / 256 * 256;
        if (b > cap - total) break;
        take.push_back({l, e});
        offs.push_back(total);
        total += b;
    }
    if (take.empty()) return 0;
    auto& queue = dpct::get_in_order_queue();
    std::vector<uint8_t*> regions;
    std::vector<uint8_t*> destinations(take.size(), nullptr);
    auto release = [&] {
        for (auto* p : regions) sycl::free(p, queue);
        regions.clear();
    };
    try {
        const uint64_t allocation_limit = queue.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
        uint8_t* base = total <= allocation_limit ? (uint8_t*) sycl::malloc_host(total, queue) : nullptr;
        if (base != nullptr) {
            regions.push_back(base);
            for (size_t j = 0; j < take.size(); ++j) destinations[j] = base + offs[j];
        } else {
            // USM host allocations are also bounded by max_mem_alloc_size on the Level Zero runtime.
            // The original IQ2_XS model needs ~30 GiB of host mirror on a 12 GiB B580. Pack whole experts
            // into <=2 GiB regions: the kernels consume individual blob pointers, not a single arena base.
            const uint64_t region_limit = std::min<uint64_t>(2ull << 30, allocation_limit);
            for (size_t begin = 0; begin < take.size();) {
                uint64_t bytes = 0;
                size_t end = begin;
                while (end < take.size()) {
                    const uint64_t b = (lay.bytes[(size_t) take[end].first] + 255) / 256 * 256;
                    if (b > region_limit - bytes) break;
                    bytes += b;
                    ++end;
                }
                if (end == begin) {
                    release();
                    err = "mirror: an expert exceeds the USM host allocation limit";
                    return -1;
                }
                auto* p = (uint8_t*) sycl::malloc_host(bytes, queue);
                if (p == nullptr) {
                    release();
                    err = "mirror: no pinned host memory for a " + std::to_string(bytes >> 20) + " MiB region";
                    return -1;
                }
                regions.push_back(p);
                uint64_t offset = 0;
                for (size_t j = begin; j < end; ++j) {
                    destinations[j] = p + offset;
                    offset += (lay.bytes[(size_t) take[j].first] + 255) / 256 * 256;
                }
                begin = end;
            }
        }
    } catch (const sycl::exception& ex) {
        release();
        err = std::string("mirror: ") + ex.what();
        return -1;
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    std::vector<std::thread> ts;
    for (int i = 0; i < std::max(1, threads); ++i)
        ts.emplace_back([&] {
            for (size_t j; (j = next.fetch_add(1)) < take.size() && !bad.load();) {
                const auto [l, e] = take[j];
                if (!read_into(l, e, destinations[j], (size_t) lay.bytes[(size_t) l])) bad.store(true);
            }
        });
    for (auto& th : ts) th.join();
    if (bad.load()) {
        release();
        err = "mirror: reading an expert from the GGUF failed";
        return -1;
    }
    for (auto* p : mirror_regions_) sycl::free(p, queue);
    mirror_regions_ = std::move(regions);
    mirror_bytes_ = total;
    mirror_ptr_.assign((size_t) (n_layers_ * n_expert_), nullptr);
    layer_first_.assign((size_t) n_layers_, nullptr);
    for (size_t j = 0; j < take.size(); ++j) {
        const auto [l, e] = take[j];
        mirror_ptr_[(size_t) (l * n_expert_ + e)] = destinations[j];
        if (layer_first_[(size_t) l] == nullptr) layer_first_[(size_t) l] = destinations[j];
    }
    return (int64_t) take.size();
}

bool GgufExpertSource::pinned(int64_t layer, int64_t expert) const {
    if (mirror_ptr_.empty() || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return false;
    return mirror_ptr_[(size_t) (layer * n_expert_ + expert)] != nullptr;
}

const uint8_t* GgufExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (mirror_ptr_.empty() || layer < 0 || layer >= n_layers_) return nullptr;
    if (expert >= 0 && expert < n_expert_) {
        const auto* p = mirror_ptr_[(size_t) (layer * n_expert_ + expert)];
        if (p != nullptr) return p;
    }
    return layer_first_[(size_t) layer];
}

bool GgufExpertSource::read_into(int64_t layer, int64_t expert, uint8_t* dst, size_t bytes) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_ || dst == nullptr) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t blob = lay.bytes[(size_t) layer];
    if (bytes < blob) return false;
    const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    for (int r = 0; r < 3; ++r) {
        const int fd = fds_[(size_t) layer_fd_[(size_t) (3 * layer + r)]];
        const uint64_t src = lay.gguf_off[(size_t) (3 * layer + r)] + per[r] * (uint64_t) expert;
        uint64_t done = 0;
        while (done < per[r]) {
            const ssize_t n = ::pread(fd, dst + at[r] + done, (size_t) (per[r] - done), (off_t) (src + done));
            if (n <= 0) return false;
            done += (uint64_t) n;
        }
    }
    return true;
}

// Exchange existing host/device bytes without an additional expert blob.
namespace {
class MirrorCacheExchange;
void exchange_mirror_bytes(uint8_t* device, uint8_t* mirror, size_t bytes, sycl::queue& q) {
    if (!device || !mirror || !bytes || bytes % 16 ||
        (reinterpret_cast<uintptr_t>(device) & 15) || (reinterpret_cast<uintptr_t>(mirror) & 15) ||
        sycl::get_pointer_type(device, q.get_context()) != sycl::usm::alloc::device ||
        sycl::get_pointer_type(mirror, q.get_context()) != sycl::usm::alloc::host)
        throw std::invalid_argument("mirror cache exchange requires aligned device/host whole blobs");
    const size_t vectors = bytes / 16;
    auto* dp = reinterpret_cast<uint32_t*>(device);
    auto* hp = reinterpret_cast<uint32_t*>(mirror);
    q.parallel_for<MirrorCacheExchange>(
        sycl::nd_range<1>((vectors + 255) / 256 * 256, 256),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const size_t i = item.get_global_id(0);
            if (i >= vectors) return;
            sycl::uint4 old_device, old_host;
            old_device.load(0, dp + 4 * i);
            old_host.load(0, hp + 4 * i);
            old_host.store(0, dp + 4 * i);
            old_device.store(0, hp + 4 * i);
        });
    q.wait_and_throw();
}
}  // namespace

bool GgufExpertSource::exchange_cached_expert(int64_t layer, int32_t incoming, int32_t outgoing,
                                             void* device_slot, void* stream, std::string& err) {
    if (layer < 0 || layer >= n_layers_ || incoming < 0 || outgoing < 0 ||
        incoming >= n_expert_ || outgoing >= n_expert_ || incoming == outgoing ||
        mirror_ptr_.empty() || !stream) {
        err = "mirror cache exchange invalid identity"; return false;
    }
    const size_t in = (size_t)(layer * n_expert_ + incoming);
    const size_t out = (size_t)(layer * n_expert_ + outgoing);
    std::lock_guard<std::mutex> lock(mu_);
    if (!mirror_ptr_[in] || mirror_ptr_[out]) {
        err = "mirror cache exchange requires incoming mirror and unmirrored outgoing"; return false;
    }
    try {
        auto* memory = const_cast<uint8_t*>(mirror_ptr_[in]);
        exchange_mirror_bytes(static_cast<uint8_t*>(device_slot), memory,
            (size_t)strata::kernels::cpu::expert_layout().blob_bytes(layer),
            *static_cast<sycl::queue*>(stream));
        mirror_ptr_[out] = memory;
        mirror_ptr_[in] = nullptr;
        return true;
    } catch (const std::exception& e) { err = e.what(); return false; }
}
}  // namespace strata::core
