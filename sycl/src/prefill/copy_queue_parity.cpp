#define DPCT_PROFILING_ENABLED
#include "strata/prefill_copy_queue.hpp"
#include <array>
#include <chrono>
#include <iostream>

using Clock = std::chrono::steady_clock;

void run(bool profiling, size_t bytes) {
    auto& compute = dpct::get_in_order_queue();
    std::unique_ptr<sycl::queue> plain;
    sycl::queue* transfer;
    if (profiling) transfer = dpct::get_current_device().create_queue(true);
    else { plain = strata::make_prefill_copy_queue(compute); transfer = plain.get(); }
    if (!transfer->has_property<sycl::property::queue::in_order>() ||
        transfer->has_property<sycl::property::queue::enable_profiling>() != profiling ||
        transfer->get_context() != compute.get_context() ||
        transfer->get_device() != compute.get_device()) std::exit(2);
    constexpr size_t slots = 8, sources = 16, guard = 64;
    std::array<uint8_t*, sources> input;
    std::array<uint8_t*, slots> device;
    std::array<sycl::event, slots> used;
    std::array<bool, slots> live{};
    for (size_t s=0;s<sources;++s) {
        input[s] = sycl::malloc_host<uint8_t>(bytes, compute);
        for (size_t k=0;k<bytes;++k) input[s][k] = (uint8_t)((k*13+s*17)^((k>>8)+s));
    }
    for (auto& p : device) p = sycl::malloc_device<uint8_t>(bytes+2*guard, compute);
    auto* marker = sycl::malloc_host<uint64_t>(slots, compute);
    auto* error = sycl::malloc_device<uint32_t>(1, compute);
    for (size_t s=0;s<slots;++s) marker[s]=0;
    compute.fill(error, uint32_t{0}, 1).wait();
    size_t next = 0;
    auto batch = [&](size_t count) {
        for (size_t k=0;k<count;++k,++next) {
            const size_t slot=next%slots, source=next%sources;
            if (live[slot]) transfer->ext_oneapi_submit_barrier({used[slot]});
            transfer->memset(device[slot],0x5a,bytes+2*guard);
            transfer->memcpy(device[slot]+guard,input[source],bytes);
            transfer->fill(marker+slot,(uint64_t)(next+1),1);
            auto copied=transfer->ext_oneapi_submit_barrier();
            compute.ext_oneapi_submit_barrier({copied});
            auto* ptr=device[slot];
            compute.parallel_for(sycl::range<1>(bytes+2*guard),[=](sycl::id<1> item) {
                const size_t i=item[0];
                const size_t offset=i-guard;
                const uint8_t expected=i<guard || i>=guard+bytes ? 0x5a :
                    (uint8_t)((offset*13+source*17)^((offset>>8)+source));
                if (ptr[i]!=expected) {
                    sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,
                        sycl::memory_scope::device,sycl::access::address_space::global_space>(*error).fetch_or(1);
                }
            });
            used[slot]=compute.ext_oneapi_submit_barrier();
            live[slot]=true;
        }
        compute.wait_and_throw();
        transfer->wait_and_throw();
        for (size_t s=0;s<slots;++s) {
            if (next<=s) continue;
            const uint64_t expected=(next-1-(next-1-s)%slots)+1;
            if (marker[s]!=expected) std::exit(3);
        }
    };
    auto warm=Clock::now();
    do { batch(128); } while(next<1000 || std::chrono::duration<double>(Clock::now()-warm).count()<2);
    size_t timed=0;
    auto begin=Clock::now();
    do { batch(256); timed+=256; } while(timed<2000 || std::chrono::duration<double>(Clock::now()-begin).count()<3);
    const double elapsed=std::chrono::duration<double>(Clock::now()-begin).count();
    uint32_t errors;
    compute.memcpy(&errors,error,sizeof errors).wait();
    if (errors) std::exit(4);
    std::cout << "{\"profiling\":" << (profiling?"true":"false") << ",\"bytes\":" << bytes
              << ",\"timed_calls\":" << timed << ",\"wall_us\":" << elapsed*1e6/timed
              << ",\"byte_exact\":true,\"guards\":true,\"markers\":true}" << std::endl;
    for (auto p:input) sycl::free(p,compute);
    for (auto p:device) sycl::free(p,compute);
    sycl::free(marker,compute);sycl::free(error,compute);
    if (profiling) dpct::get_current_device().destroy_queue(transfer);
}

int main() {
    for (size_t bytes : {size_t{1510400},size_t{1305600},size_t{1177600}})
        for (bool profiling : {true,false}) run(profiling,bytes);
}
