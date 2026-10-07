#pragma once

namespace strata::kernels {
// Bench only: wait for outstanding queue work before changing the route.
// Production reads STRATA_SYCL_GR_LO_ONCE before graph capture (default off).
void fused_gr_set_lo_once(bool enabled);
}
