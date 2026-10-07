#pragma once

namespace strata::kernels {
// Diagnostic bench only. Wait for outstanding queue work before changing routes;
// production selects the optional route through STRATA_SYCL_GR_UP_STATIC_T.
void fused_gr_set_up_static(bool enabled);
void fused_gr_set_lo_once(bool enabled);
void fused_gr_set_up_sg(int lanes);
}
