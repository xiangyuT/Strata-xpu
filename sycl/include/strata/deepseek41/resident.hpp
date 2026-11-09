#pragma once
#include "strata/deepseek41/model.hpp"
#include <memory>

namespace strata::deepseek41 {
// SYCL bring-up: model-owned read-only mappings back expert and Engram data.
// This interface does not promise that all mapped pages are resident in RAM.
// Uploads use bounded host-USM staging so GPU kernels never read a file mapping.
void copy_registered_slice(void*,const uint8_t*,size_t,size_t,void*);
class ResidentWeights {
public:
    ResidentWeights();
    ~ResidentWeights();
    void load(const Model&);
    const uint8_t* data(const std::string&) const;
    uint64_t bytes() const;
    void verify() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
} // namespace strata::deepseek41
