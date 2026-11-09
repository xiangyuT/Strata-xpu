#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/deepseek41/kernels.hpp"
#include "strata/deepseek41/attention.hpp"

#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>
#include <time.h>

namespace strata::deepseek41::cuda {
namespace {
constexpr int threads = 256;
dpct::queue_ptr stream_cast(void *stream) {
    if (!stream) throw std::invalid_argument("DeepSeek CUDA requires an explicit stream");
    return strata::q_of(stream);
}
inline float sum(float v) {

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());


#pragma unroll
    for (int n = 16; n; n >>= 1) v += dpct::experimental::shift_sub_group_left(
        0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, n);
    if (!(item_ct1.get_local_id(2) & 31))
        partial[item_ct1.get_local_id(2) >> 5] = v;

    item_ct1.barrier();
    v = item_ct1.get_local_id(2) < 8 ? partial[item_ct1.get_local_id(2)] : 0;


    if (item_ct1.get_local_id(2) < 32)
#pragma unroll
    for (int n = 16; n; n >>= 1) v += dpct::experimental::shift_sub_group_left(
        0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, n);
    if (!item_ct1.get_local_id(2)) partial[0] = v;

    item_ct1.barrier();
    const float result = partial[0];

    item_ct1.barrier();
    return result;
}
inline float sigmoid(float x) { return 1.0f / (1.0f + sycl::native::exp(-x)); }
inline float maximum(float v) {

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &partial = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[8]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());


#pragma unroll
    for (int n = 16; n; n >>= 1) v = sycl::fmax(
        v, dpct::experimental::shift_sub_group_left(
               0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
               v, n));
    if (!(item_ct1.get_local_id(2) & 31))
        partial[item_ct1.get_local_id(2) >> 5] = v;

    item_ct1.barrier();
    v = item_ct1.get_local_id(2) < 8 ? partial[item_ct1.get_local_id(2)]
                                     : -INFINITY;


    if (item_ct1.get_local_id(2) < 32)
#pragma unroll
    for (int n = 16; n; n >>= 1) v = sycl::fmax(
        v, dpct::experimental::shift_sub_group_left(
               0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
               v, n));
    if (!item_ct1.get_local_id(2)) partial[0] = v;

    item_ct1.barrier();
    const float result = partial[0];

    item_ct1.barrier();
    return result;
}
__dpct_inline__ void rms_kernel(const float *x, const float *w, float *y,
                                int width, float eps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    x += size_t(item_ct1.get_group(2)) * width;
        y += size_t(item_ct1.get_group(2)) * width;
    float s = 0;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < width;
         i += item_ct1.get_local_range(2)) s += x[i] * x[i];
    s = sycl::rsqrt(sum(s) / width + eps);
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < width;
         i += item_ct1.get_local_range(2))
        y[i] = (x[i] * s) * (w ? w[i] : 1.0f);
}
__dpct_inline__ void init_kernel(const float *x, float *r, float *pre,
                                 int hidden) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(2);
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < hidden * 4;
         i += item_ct1.get_local_range(2))
        r[size_t(t) * hidden * 4 + i] = x[size_t(t) * hidden + i % hidden];
    if (item_ct1.get_local_id(2) < 4) pre[t * 4 + item_ct1.get_local_id(2)] =
        item_ct1.get_local_id(2) == 0 ? 1.0f : 0.0f;
}

__dpct_inline__ void coeff_kernel(const float *mix, const float *scale,
                                  const float *base, float *pre, float *post,
                                  float *comb, float eps, int iterations) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (item_ct1.get_local_id(2)) return;
    const int t = item_ct1.get_group(2);
    mix += t*24; pre += t*4; post += t*4; comb += t*16;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        pre[h] = sigmoid(sycl::fma((float)(mix[h]), (float)(scale[0]),
                                   (float)(base[h]))) +
                 eps;
        post[h] = 2 * sigmoid(sycl::fma((float)(mix[4 + h]), (float)(scale[1]),
                                        (float)(base[4 + h])));
    }
    float c[16];
    for (int src = 0; src < 4; ++src) {
        float maximum = -INFINITY, total = 0;
        for (int dst = 0; dst < 4; ++dst) {
            const int i = dst+4*src;
            c[i] = sycl::fma((float)(mix[8 + i]), (float)(scale[2]),
                             (float)(base[8 + i]));
                maximum = sycl::fmax(maximum, c[i]);
        }
        for (int dst = 0; dst < 4; ++dst) {
          const int i = dst + 4 * src; c[i] = sycl::native::exp(c[i] - maximum);
          total += c[i];
        }
        const float inv = 1.0f/total;
#pragma unroll
        for (int dst = 0; dst < 4; ++dst)
            c[dst + 4 * src] = c[dst + 4 * src] * inv + eps;
    }
    for (int it = 0; it < iterations; ++it) {
        if (it) for (int src = 0; src < 4; ++src) {
            float total = eps;
#pragma unroll
            for (int dst = 0; dst < 4; ++dst) total += c[dst + 4 * src];
            const float inv = 1.0f/total;
#pragma unroll
            for (int dst = 0; dst < 4; ++dst) c[dst + 4 * src] *= inv;
        }
        for (int dst = 0; dst < 4; ++dst) {
            float total = eps;
#pragma unroll
            for (int src = 0; src < 4; ++src) total += c[dst + 4 * src];
            const float inv = 1.0f/total;
#pragma unroll
            for (int src = 0; src < 4; ++src) c[dst + 4 * src] *= inv;
        }
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) comb[i] = c[i];
}
__dpct_inline__ void pre_kernel(const float *r, const float *pre, float *y,
                                int hidden) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1),
              i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= hidden) return;
    r += size_t(t)*hidden*4+i; pre += t*4;
    float v = 0;
#pragma unroll
    for (int h = 0; h < 4; ++h)
        v = sycl::fma((float)(r[h * hidden]), (float)(pre[h]), v);
    y[size_t(t)*hidden+i] = v;
}
__dpct_inline__ void post_kernel(const float *x, const float *r,
                                 const float *post, const float *c, float *y,
                                 int hidden) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1),
              i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= hidden) return;
    x += size_t(t)*hidden; r += size_t(t)*hidden*4+i; y += size_t(t)*hidden*4+i;
    post += t*4; c += t*16;
    // Reading all four source values before writing makes in-place residual safe.
    float src[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) src[h] = r[h * hidden];
    for (int dst = 0; dst < 4; ++dst) {

        float v = x[i] * post[dst];
#pragma unroll
        for (int h = 0; h < 4; ++h)
            v = sycl::fma(src[h], (float)(c[dst + 4 * h]), v);
        y[dst*hidden] = v;
    }
}
__dpct_inline__ void engram_kernel(float *r, const float *kv, const float *qw,
                                   const float *kw, float *gates, int hidden,
                                   float eps) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int h = item_ct1.get_group(2) % 4, t = item_ct1.get_group(2) / 4;
    r += (size_t(t)*4+h)*hidden; kv += size_t(t)*hidden*5;
    const float* key = kv+size_t(h)*hidden;
    qw += size_t(h)*hidden; if (kw) kw += size_t(h)*hidden;
    float xx = 0, kk = 0;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < hidden;
         i += item_ct1.get_local_range(2)) {
      xx += r[i] * r[i]; kk += key[i] * key[i];
    }
    const float xn = sycl::rsqrt(sum(xx) / hidden + eps),
                kn = sycl::rsqrt(sum(kk) / hidden + eps);
    float dot = 0;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < hidden;
         i += item_ct1.get_local_range(2)) dot += kw ?
        ((r[i] * xn) * qw[i]) * ((key[i] * kn) * kw[i]) :
        ((r[i] * xn) * qw[i]) * (key[i] * kn);
    const float score = sum(dot) * sycl::rsqrt(float(hidden));
    const float gate = sigmoid(sycl::copysign(
        sycl::sqrt(sycl::fmax(sycl::fabs(score), 1e-6f)), (float)score));
    if (!item_ct1.get_local_id(2) && gates) gates[t * 4 + h] = gate;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < hidden;
         i += item_ct1.get_local_range(2)) r[i] += kv[hidden * 4 + i] * gate;
}
__dpct_inline__ void route_kernel(const float *logits, const float *bias,
                                  int32_t *ids, float *weights, int experts,
                                  int k, float scale) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &probs = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[512]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &values =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &selected =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[64]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &indices =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &winner = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int lane = item_ct1.get_local_id(2), t = item_ct1.get_group(2);
    logits += size_t(t)*experts; ids += t*k; weights += t*k;
    for (int i = lane; i < experts; i += item_ct1.get_local_range(2)) {
        const float v = logits[i];
            probs[i] =
                sycl::sqrt(sycl::fmax(v, 0.0f) +
                           sycl::log1p(sycl::native::exp(-sycl::fabs(v))));
    }

    item_ct1.barrier();
    for (int rank = 0; rank < k; ++rank) {
        float best = -INFINITY; int id = INT_MAX;
        for (int i = lane; i < experts; i += item_ct1.get_local_range(2)) {
            bool used = false;
#pragma unroll
            for (int j = 0; j < rank; ++j) used |= ids[j] == i;
            const float v = probs[i]+bias[i];
            if (!used && (v > best || (v == best && i < id))) { best = v; id = i; }
        }


        values[lane] = best; indices[lane] = id; item_ct1.barrier();
        for (int n = 128; n; n >>= 1) {
            if (lane < n) {
                const float v = values[lane+n]; const int j = indices[lane+n];
                if (v > values[lane] || (v == values[lane] && j < indices[lane])) { values[lane] = v; indices[lane] = j; }
            }


            item_ct1.barrier();
        }
        if (!lane) { winner = indices[0]; ids[rank] = winner; selected[rank] = probs[winner]; }


        item_ct1.barrier();
    }
    if (!lane) {
        float total = 0;
#pragma unroll
        for (int i = 0; i < k; ++i) total += selected[i];
        const float inv = 1.0f / sycl::fmax(total, 6.103515625e-5f);
#pragma unroll
        for (int i = 0; i < k; ++i) weights[i] = (selected[i] * inv) * scale;
    }
}
__dpct_inline__ void swiglu_kernel(const float *g, const float *u, float *y,
                                   int n, float limit) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i < n) {
      const float v = sycl::fmin(g[i], limit);
      y[i] = (v * sigmoid(v)) * sycl::fmin(sycl::fmax(u[i], -limit), limit);
    }
}
__dpct_inline__ void combine_kernel(const float *parts, const float *w,
                                    const float *shared, float *y, int h,
                                    int k) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1),
              i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= h) return;
    parts += size_t(t)*k*h+i; w += t*k;
    float v = 0;
#pragma unroll
    for (int j = 0; j < k; ++j) v += parts[size_t(j) * h] * w[j];
    y[size_t(t)*h+i] = v+(shared ? shared[size_t(t)*h+i] : 0.0f);
}
template <bool System = false>
__dpct_inline__ void
combine_sources_kernel(const float *device_parts, const float *host_parts,
                       const uint64_t *gpu_mask, const float *w,
                       const float *shared, float *y, int h, int k) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1),
              i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if(i>=h) return;
    const uint64_t mask=*gpu_mask; w+=t*k;
    float v=0;
    for(int j=0;j<k;++j) {
        const int row=t*k+j;
        float part;
        if constexpr(System) {
            const size_t at=size_t(row)*h+i;
            part=(mask & (uint64_t(1)<<row)) ? device_parts[at] :
                reinterpret_cast<const volatile float*>(host_parts)[at];
        } else {
            const float* parts=(mask & (uint64_t(1)<<row)) ? device_parts : host_parts;
            part=parts[size_t(row)*h+i];
        }
        v+=part*w[j];
    }
    y[size_t(t)*h+i]=v+(shared ? shared[size_t(t)*h+i] : 0.0f);
}
__dpct_inline__ void dense_kernel(const float *w, const float *x, float *y,
                                  int n_in, int n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2), t = item_ct1.get_group(1);
    w += size_t(row)*n_in; x += size_t(t)*n_in;
    float v = 0;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n_in;
         i += item_ct1.get_local_range(2))
        v = sycl::fma((float)(w[i]), (float)(x[i]), v);
    v = sum(v); if (!item_ct1.get_local_id(2)) y[size_t(t) * n_out + row] = v;
}
__dpct_inline__ void add_kernel(const float *x, const float *y, float *out,
                                int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
        if (i < n) out[i] = x[i] + y[i];
}
__dpct_inline__ void strided_copy_kernel(const float *x, float *y, int width,
                                         int xs, int ys) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2),
              row = item_ct1.get_group(1);
    if (i < width) y[size_t(row)*ys+i] = x[size_t(row)*xs+i];
}
__dpct_inline__ void gather_row_kernel(const float *rows, const int32_t *id,
                                       float *out, int width, int count) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i<width) out[i]=(*id>=0 && *id<count) ? rows[size_t(*id)*width+i] : 0.0f;
}
__dpct_inline__ void draft_confidence_kernel(const float *hidden,
                                             const float *markov,
                                             const float *w, const float *bias,
                                             float *out, int h, int m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    float v = 0;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < h + m;
         i += item_ct1.get_local_range(2)) v =
        sycl::fma((float)(w[i]), (float)(i < h ? hidden[i] : markov[i - h]), v);
    v=sum(v);
    if (!item_ct1.get_local_id(2)) *out = sigmoid(v + (bias ? *bias : 0.0f));
}
__dpct_inline__ void mean_kernel(const float *r, float *y, int h, int stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2),
              t = item_ct1.get_group(1);
    if (i >= h) return;
    r += size_t(t)*h*4+i;
    float v = r[0];
#pragma unroll
    for (int s = 1; s < 4; ++s) v += r[size_t(s) * h];
    y[size_t(t)*stride+i] = v*0.25f;
}
__dpct_inline__ void rope_kernel(float *x, int d, int heads, const int32_t *pos,
                                 Rope c, float low, float high, bool inverse) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int pair = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2);
    if (pair >= heads*c.dim/2) return;
    const int h = pair / (c.dim / 2), i = pair % (c.dim / 2),
              t = item_ct1.get_group(1);
    const float theta_scale = dpct::pow(c.base, -2.0f / c.dim);
    const float extrap = float(pos[t]) * dpct::pow(theta_scale, float(i));
    float theta = c.scale*extrap;
    if (c.yarn) {
        const float ramp =
            1.0f -
            sycl::fmin(
                1.0f,
                sycl::fmax(0.0f, (i - low) / sycl::fmax(0.001f, high - low)));
        theta = theta*(1.0f-ramp)+extrap*ramp;
    }
    // DeepSeek's attention factor cancels YaRN's magnitude factor, so both
    // forward and inverse rotations have unit magnitude, also in long context.
    const float cs = sycl::cos(theta),
                sn = sycl::sin(theta) * (inverse ? -1.0f : 1.0f);
    x += (size_t(t)*heads+h)*d+d-c.dim+2*i;
    const float a = x[0], b = x[1]; x[0] = a*cs-b*sn; x[1] = a*sn+b*cs;
}
__dpct_inline__ void store_kernel(const float *x, sycl::half *cache, int n,
                                  int row) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
        if (i < n) cache[size_t(row) * n + i] =
            sycl::vec<float, 1>(x[i])
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
__dpct_inline__ void pool_kernel(const float *v, const float *s, float *y,
                                 int n, int ratio) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (i >= n) return;
    if (ratio == 1) { y[i] = v[i]; return; }
    const float m = sycl::fmax(s[i], s[n + i]), a = sycl::native::exp(s[i] - m),
                b = sycl::native::exp(s[n + i] - m);
    // Pool in projection precision; the cache converts to F16 only after norm+RoPE.
    y[i] = v[i]*(a/(a+b))+v[n+i]*(b/(a+b));
}
__dpct_inline__ void store_positions_kernel(const float *x, sycl::half *cache,
                                            int width, const int32_t *positions,
                                            int capacity, int ratio) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2),
              t = item_ct1.get_group(1), pos = positions[t];
    if (i >= width || (ratio && (pos+1)%ratio)) return;
    const int row = ratio ? pos/ratio : pos%capacity;
    cache[size_t(row) * width + i] =
        sycl::vec<float, 1>(x[size_t(t) * width + i])
            .convert<sycl::half, sycl::rounding_mode::rte>()[0];
}
__dpct_inline__ void projection_ring_kernel(const float *v, const float *s,
                                            float *rv, float *rs, int width,
                                            const int32_t *positions,
                                            int capacity) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2),
              t = item_ct1.get_group(1);
    if (i >= width) return;
    const size_t dst = size_t(positions[t]%capacity)*width+i, src = size_t(t)*width+i;
    rv[dst] = v[src]; if (s && rs) rs[dst] = s[src];
}
__dpct_inline__ void pool_positions_kernel(const float *v, const float *s,
                                           float *y, int width,
                                           const int32_t *positions,
                                           int capacity, int ratio) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2),
              t = item_ct1.get_group(1), pos = positions[t];
    if (i >= width) return;
    y += size_t(t)*width;
    if ((pos+1)%ratio) { y[i] = 0; return; }
    const size_t first = size_t((pos+1-ratio)%capacity)*width+i;
    if (ratio == 1) { y[i] = v[first]; return; }
    const size_t second = size_t(pos%capacity)*width+i;
    const float m = sycl::fmax(s[first], s[second]),
                a = sycl::native::exp(s[first] - m),
                b = sycl::native::exp(s[second] - m);
    y[i] = v[first]*(a/(a+b))+v[second]*(b/(a+b));
}
template <bool DevicePositions, bool DraftBlock = false, bool Normalize = false,
          bool Emit = false, bool QueryPrefetch = false>

__dpct_inline__ void
attention_kernel(const float *q, const sycl::half *raw, const sycl::half *comp,
                 const int32_t *ids, const float *sinks, float *out, int d,
                 int window, int position, int selected, int capacity, int last,
                 const int32_t *positions, int heads, int ratio,
                 int max_selected, int ids_stride, uint8_t *dpct_local,
                 float *score_out = nullptr,
                 unsigned long long *row_out = nullptr, int output_stride = 0) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto scores = (float *)dpct_local;
    if constexpr (DevicePositions) {
        const int t = item_ct1.get_group(1); position = positions[t];
        if constexpr (DraftBlock) last =
            positions[item_ct1.get_group_range(1) - 1];
        else last = position;
        selected = ratio ? sycl::min((position + 1) / ratio, max_selected) : 0;
        q += size_t(t)*heads*d; out += size_t(t)*heads*d;
        if (ids) ids += size_t(t)*ids_stride;
    }
    const int head = item_ct1.get_group(2),
              lane = item_ct1.get_local_id(2) % 32,
              warp = item_ct1.get_local_id(2) / 32;
    if constexpr (Emit) {
        const size_t slot =
            (DevicePositions ? size_t(item_ct1.get_group(1)) * heads : 0) +
            head;
        score_out+=slot*output_stride;row_out+=slot*output_stride;
    }
    const int first = sycl::max(0, position - window + 1),
              nr = last - first + 1, n = nr + selected;
    q += size_t(head)*d; out += size_t(head)*d;
    float query_fragment[16];
    if constexpr (QueryPrefetch) {
#pragma unroll
        for(int k=0;k<16;++k) query_fragment[k]=q[lane+32*k];
    }
    for (int row = warp; row < n; row += 8) {
        const sycl::half *key =
            row < nr ? raw + size_t((first + row) % capacity) * d
                     : comp + size_t(ids ? ids[row - nr] : row - nr) * d;
        float dot = 0;
        if constexpr (QueryPrefetch) {
            uint16_t fragment[16];
#pragma unroll
            for(int k=0;k<16;++k) fragment[k]=reinterpret_cast<const uint16_t*>(key)[lane+32*k];
#pragma unroll
            for (int k = 0; k < 16; ++k) dot +=
                query_fragment[k] *
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(fragment[k]))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
        } else
#pragma unroll
        for (int i = lane; i < d; i += 32) dot +=
            q[i] * sycl::vec<sycl::half, 1>(key[i])
                       .convert<float, sycl::rounding_mode::automatic>()[0];

#pragma unroll
        for (int k = 16; k; k >>= 1) dot +=
            dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                dot, k);
        if (!lane) {
            scores[row] = dot * sycl::rsqrt(float(d));
            if constexpr (Emit) row_out[row]=reinterpret_cast<unsigned long long>(key);
        }
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    float m = sinks[head];
#pragma unroll
    for (int row = item_ct1.get_local_id(2); row < n;
         row += item_ct1.get_local_range(2)) m = sycl::fmax(m, scores[row]);
    m = maximum(m);
    float denom =
        item_ct1.get_local_id(2) == 0 ? sycl::native::exp(sinks[head] - m) : 0;
#pragma unroll
    for (int row = item_ct1.get_local_id(2); row < n;
         row += item_ct1.get_local_range(2)) {
      scores[row] = sycl::native::exp(scores[row] - m); denom += scores[row];
    }
    denom = sum(denom);
    if constexpr (Emit) {
#pragma unroll
        for (int row = item_ct1.get_local_id(2); row < n;
             row += item_ct1.get_local_range(2))
            score_out[row] = scores[row] / denom;
        return;
    }
    if constexpr (Normalize) {
        // Every output channel uses the same rounded weight. Compute each
        // division once, with the original denominator and row order.
#pragma unroll
        for (int row = item_ct1.get_local_id(2); row < n;
             row += item_ct1.get_local_range(2))
            scores[row] = scores[row] / denom;
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    for (int i = item_ct1.get_local_id(2); i < d;
         i += item_ct1.get_local_range(2)) {
        float v = 0;
        for (int row = 0; row < n; ++row) {
            const sycl::half *value =
                row < nr ? raw + size_t((first + row) % capacity) * d
                         : comp + size_t(ids ? ids[row - nr] : row - nr) * d;
            const float weight=Normalize ? scores[row] : scores[row]/denom;
            v += sycl::vec<sycl::half, 1>(value[i])
                     .convert<float, sycl::rounding_mode::automatic>()[0] *
                 weight;
        }
        out[i] = v;
    }
}
template <bool DevicePositions>
__dpct_inline__ void index_kernel(const float *q, const float *w,
                                  const sycl::half *keys, float *scores, int d,
                                  int heads, const int32_t *positions,
                                  int ratio, int stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &dots = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int lane = item_ct1.get_local_id(2) % 32,
              warp = item_ct1.get_local_id(2) / 32, row = item_ct1.get_group(2);
    if constexpr (DevicePositions) {
        const int t = item_ct1.get_group(1);
        scores += size_t(t)*stride; q += size_t(t)*heads*d; w += size_t(t)*heads;
        if (row >= (positions[t] + 1) / ratio) {
          if (!item_ct1.get_local_id(2)) scores[row] = -INFINITY; return;
        }
    }
    keys += size_t(row)*d;
    for (int h = warp; h < heads; h += 8) {
        float dot = 0;
#pragma unroll
        for (int i = lane; i < d; i += 32) dot +=
            q[size_t(h) * d + i] *
            sycl::vec<sycl::half, 1>(keys[i])
                .convert<float, sycl::rounding_mode::automatic>()[0];

#pragma unroll
        for (int k = 16; k; k >>= 1) dot +=
            dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                dot, k);
        if (!lane) dots[h] =
            sycl::fmax(dot, 0.0f) * (w[h] * sycl::rsqrt(float(d * heads)));
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (!item_ct1.get_local_id(2)) { float v = 0;
#pragma unroll
    for (int h = 0; h < heads; ++h) v += dots[h]; scores[row] = v; }
}
__dpct_inline__ void topk_kernel(float *scores, int32_t *ids, int rows, int k) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &values = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &indices =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[256]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int tid = item_ct1.get_local_id(2);
    for (int rank = 0; rank < k; ++rank) {
        float best = -INFINITY; int id = INT_MAX;
        for (int i = tid; i < rows; i += item_ct1.get_local_range(2)) {
            const float v = scores[i]; if (v > best || (v == best && i < id)) { best = v; id = i; }
        }


        values[tid] = best; indices[tid] = id; item_ct1.barrier();
        for (int n = 128; n; n >>= 1) {
            if (tid < n) {
                const int j = indices[tid+n]; const float v = values[tid+n];
                if (v > values[tid] || (v == values[tid] && j < indices[tid])) { values[tid] = v; indices[tid] = j; }
            }


            item_ct1.barrier();
        }
        if (!tid) { ids[rank] = indices[0]; scores[indices[0]] = -INFINITY; }


        item_ct1.barrier();
    }
}
__dpct_inline__ void sequence_kernel(int32_t *ids, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
        if (i < n) ids[i] = i;
}
__dpct_inline__ uint32_t load_system(const uint32_t *pointer) {
#if DPCT_COMPATIBILITY_TEMP >= 700
    uint32_t value;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(pointer)
                 : "memory");
#else
    value = *pointer;
#endif
    return value;
#else
    return *reinterpret_cast<const volatile uint32_t*>(pointer);
#endif
}
__dpct_inline__ void host_wait_kernel(const uint32_t *ready,
                                      const uint32_t *expected,
                                      const uint32_t *abort) {
    if (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(
            2)) return;
    const uint32_t want=load_system(expected);
    while(load_system(ready)!=want && !load_system(abort)) {
#if DPCT_COMPATIBILITY_TEMP >= 700

        /* no device nanosleep in SYCL */;
#endif
    }

    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}
__dpct_inline__ void tp_sum_wait_kernel(const float *local,
                                        const volatile float *peer,
                                        const volatile uint32_t *own_ready,
                                        const volatile uint32_t *peer_ready,
                                        const volatile uint32_t *abort,
                                        float *out, int n, int rank) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &stopped = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) == 0) {
        const uint32_t want=*own_ready;
        while(*peer_ready<want && !*abort) {
#if DPCT_COMPATIBILITY_TEMP >= 700

            /* no device nanosleep in SYCL */;
#endif
        }
        stopped=*abort!=0;

        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if(stopped) return;
    const int i = int(item_ct1.get_group(2)) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if(i<n) {
        const float a=rank==0 ? local[i] : peer[i];
        const float b=rank==0 ? peer[i] : local[i];

        out[i] = 0.0f + a + b;
    }
}

// Adapt Qwen's fixed 64-cell split-K decode design to the original D512 MLA
// ring and selected compressed rows. Query dot order stays lane+32*k, down
// shuffles16..1; only softmax/value aggregation is split into chunks.
constexpr int DsChunk=64,DsHeads=4,DsDim=512,DsStride=544;
inline float ds_warp_sum(float v) {


#pragma unroll
    for (int k = 16; k; k >>= 1) v += dpct::experimental::shift_sub_group_left(
        0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, k);


    return dpct::experimental::select_from_sub_group(
        0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, 0);
}
inline float ds_warp_max(float v) {


#pragma unroll
    for (int k = 16; k; k >>= 1) v = sycl::fmax(
        v, dpct::experimental::shift_sub_group_left(
               0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
               v, k));


    return dpct::experimental::select_from_sub_group(
        0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v, 0);
}

__dpct_inline__ void
attention_chunk_kernel(const float *q, const sycl::half *raw,
                       const sycl::half *comp, const int32_t *ids,
                       float *scratch, int heads, int window,
                       const int32_t *positions, int ratio, int selected_limit,
                       int ids_stride, int capacity, int chunks) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &sq =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<float[DsHeads][DsDim]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sp = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[DsHeads][DsChunk]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &rows = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        const sycl::half *[DsChunk]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &maxima =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[DsHeads]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &totals =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[DsHeads]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int tid = item_ct1.get_local_id(2), lane = tid % 32, warp = tid / 32;
    const int token = item_ct1.get_group(0),
              head0 = int(item_ct1.get_group(1)) * DsHeads,
              chunk = item_ct1.get_group(2);
    const int pos = positions[token], first = sycl::max(0, pos - window + 1),
              nr = pos - first + 1;
    const int selected = ratio ? sycl::min((pos + 1) / ratio, selected_limit) : 0;
    const int begin = chunk * DsChunk,
              here = sycl::min(DsChunk, nr + selected - begin);
    q+=size_t(token)*heads*DsDim+head0*DsDim;
    if(ids) ids+=size_t(token)*ids_stride;
    if(here<=0) {
        if(tid<DsHeads) {
            float* p=scratch+((size_t(token)*heads+head0+tid)*chunks+chunk)*DsStride;
            p[DsDim]=-INFINITY;p[DsDim+1]=0;
        }
        return;
    }
#pragma unroll
    for (int i = tid; i < DsHeads * DsDim; i += 256)
        sq[i / DsDim][i % DsDim] = q[i];
    if(tid<DsChunk) {
        const int row=begin+tid;
        rows[tid]=tid>=here ? nullptr : row<nr ? raw+size_t((first+row)%capacity)*DsDim :
            comp+size_t(ids ? ids[row-nr] : row-nr)*DsDim;
    }

    item_ct1.barrier();
    for(int cell=warp;cell<DsChunk;cell+=8) {
        if(cell>=here) {
            if(lane<DsHeads) sp[lane][cell]=-INFINITY;
            continue;
        }
        float key[16];
#pragma unroll
        for (int k = 0; k < 16; ++k)
            key[k] = sycl::vec<sycl::half, 1>(rows[cell][lane + 32 * k])
                         .convert<float, sycl::rounding_mode::automatic>()[0];
#pragma unroll
        for(int h=0;h<DsHeads;++h) {
            float dot=0;
#pragma unroll
            for(int k=0;k<16;++k) dot+=sq[h][lane+32*k]*key[k];

#pragma unroll
            for (int k = 16; k; k >>= 1) dot +=
                dpct::experimental::shift_sub_group_left(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), dot, k);
            if (!lane) sp[h][cell] = dot * sycl::rsqrt(float(DsDim));
        }
    }

    item_ct1.barrier();
    if(warp<DsHeads) {
        const float m =
            ds_warp_max(sycl::fmax(sp[warp][lane], sp[warp][lane + 32]));
        const float a = lane < here ? sycl::native::exp(sp[warp][lane] - m) : 0;
        const float b =
            lane + 32 < here ? sycl::native::exp(sp[warp][lane + 32] - m) : 0;
        sp[warp][lane]=a;sp[warp][lane+32]=b;
        const float total=ds_warp_sum(a+b);
        if(!lane) {maxima[warp]=m;totals[warp]=total;}
    }

    item_ct1.barrier();
    float acc[DsHeads][2]={};
    for(int cell=0;cell<here;++cell) {
        const float a =
            sycl::vec<sycl::half, 1>(rows[cell][tid])
                .convert<float, sycl::rounding_mode::automatic>()[0];
        const float b =
            sycl::vec<sycl::half, 1>(rows[cell][tid + 256])
                .convert<float, sycl::rounding_mode::automatic>()[0];
#pragma unroll
        for(int h=0;h<DsHeads;++h) {
            const float weight=sp[h][cell];
            acc[h][0]+=a*weight;acc[h][1]+=b*weight;
        }
    }
#pragma unroll
    for(int h=0;h<DsHeads;++h) {
        float* p=scratch+((size_t(token)*heads+head0+h)*chunks+chunk)*DsStride;
        p[tid]=acc[h][0];p[tid+256]=acc[h][1];
        if(!tid) {p[DsDim]=maxima[h];p[DsDim+1]=totals[h];}
    }
}
__dpct_inline__ void attention_chunk_merge_kernel(
    const float *scratch, const float *sinks, float *out, int heads, int window,
    const int32_t *positions, int ratio, int selected_limit, int chunks) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &scales = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[10]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2), token = item_ct1.get_group(1),
              tid = item_ct1.get_local_id(2);
    const int pos = positions[token], first = sycl::max(0, pos - window + 1);
    const int n = pos - first + 1 +
                  (ratio ? sycl::min((pos + 1) / ratio, selected_limit) : 0);
    const int active=(n+DsChunk-1)/DsChunk;
    const float* partial=scratch+(size_t(token)*heads+head)*chunks*DsStride;
    float m = sinks[head];
        if (tid < active) m =
            sycl::fmax(m, partial[size_t(tid) * DsStride + DsDim]);
    m=maximum(m);
    const float scale =
        tid < active
            ? sycl::native::exp(partial[size_t(tid) * DsStride + DsDim] - m)
            : 0;
    float denom = tid == 0 ? sycl::native::exp(sinks[head] - m) : 0;
    if(tid<active) denom+=scale*partial[size_t(tid)*DsStride+DsDim+1];
    denom=sum(denom);
    if(tid<active) scales[tid]=scale/denom;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    for(int i=tid;i<DsDim;i+=256) {
        float v = 0;
#pragma unroll
        for (int c = 0; c < active; ++c) v +=
            partial[size_t(c) * DsStride + i] * scales[c];
        out[(size_t(token)*heads+head)*DsDim+i]=v;
    }
}

// Parallelize independent rows without changing any row's original dot order.
// The following pass keeps the original whole-head 256-thread reductions.

template<bool SharedRows=false,int HeadGroup=1>
__dpct_inline__ void attention_score_grid_kernel(
    const float *q, const sycl::half *raw, const sycl::half *comp,
    const int32_t *ids, float *scores, unsigned long long *pointers, int d,
    int heads, int window, const int32_t *positions, int ratio,
    int max_selected, int ids_stride, int capacity, int stride) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int token = item_ct1.get_group(1), head = HeadGroup*int(item_ct1.get_group(2)),
              lane = item_ct1.get_local_id(2) % 32,
              warp = item_ct1.get_local_id(2) / 32;
    const int pos = positions[token], first = sycl::max(0, pos - window + 1),
              nr = pos - first + 1;
    const int selected = ratio ? sycl::min((pos + 1) / ratio, max_selected) : 0,
              n = nr + selected;
    const int base = int(item_ct1.get_group(0)) * 8;
    if(base>=n) return; // Uniform across the CTA, before its query barrier.
    auto &query =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[HeadGroup][512]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const size_t slot=size_t(token)*heads+head;
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < d;
         i += item_ct1.get_local_range(2)) for(int h=0;h<HeadGroup;++h) query[h][i] = q[(slot+h)*d+i];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const int row=base+warp;
    if(row>=n) return; // Entire warp returns; all query loads already completed.
    if(ids) ids+=size_t(token)*ids_stride;
    const sycl::half *key =
        row < nr ? raw + size_t((first + row) % capacity) * d
                 : comp + size_t(ids ? ids[row - nr] : row - nr) * d;
    float query_fragment[HeadGroup][16];uint16_t fragment[16];
#pragma unroll
    for(int k=0;k<16;++k) for(int h=0;h<HeadGroup;++h) query_fragment[h][k]=query[h][lane+32*k];
#pragma unroll
    for(int k=0;k<16;++k) fragment[k]=reinterpret_cast<const uint16_t*>(key)[lane+32*k];
    float dot[HeadGroup]={};
#pragma unroll
    for(int k=0;k<16;++k) {
        const float value=sycl::vec<sycl::half,1>(
            sycl::bit_cast<sycl::half,unsigned short>(fragment[k]))
                .convert<float,sycl::rounding_mode::automatic>()[0];
#pragma unroll
        for(int h=0;h<HeadGroup;++h) dot[h]+=query_fragment[h][k]*value;
    }
#pragma unroll
    for(int k=16;k;k>>=1) {
#pragma unroll
        for(int h=0;h<HeadGroup;++h) dot[h]+=dpct::experimental::shift_sub_group_left(
            0xffffffffu,sycl::ext::oneapi::this_work_item::get_sub_group(),dot[h],k);
    }
    if(!lane) {
#pragma unroll
        for(int h=0;h<HeadGroup;++h) {
            scores[(slot+h)*stride+row]=dot[h]*sycl::rsqrt(float(d));
            if constexpr(SharedRows) {
                if(!head && !h)pointers[size_t(token)*stride+row]=reinterpret_cast<unsigned long long>(key);
            }else pointers[(slot+h)*stride+row]=reinterpret_cast<unsigned long long>(key);
        }
    }
}
__dpct_inline__ void
attention_score_softmax_kernel(float *weights, const float *sinks, int heads,
                               int window, const int32_t *positions, int ratio,
                               int max_selected, int stride,
                               uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto scores = (float *)dpct_local;
    const int head = item_ct1.get_group(2), token = item_ct1.get_group(1),
              pos = positions[token], first = sycl::max(0, pos - window + 1);
    const int n = pos - first + 1 +
                  (ratio ? sycl::min((pos + 1) / ratio, max_selected) : 0);
    weights+=(size_t(token)*heads+head)*stride;
#pragma unroll
    for (int row = item_ct1.get_local_id(2); row < n;
         row += item_ct1.get_local_range(2)) scores[row] = weights[row];

    item_ct1.barrier();
    float m = sinks[head];
#pragma unroll
    for (int row = item_ct1.get_local_id(2); row < n;
         row += item_ct1.get_local_range(2)) m = sycl::fmax(m, scores[row]);
    m=maximum(m);
    float denom =
        item_ct1.get_local_id(2) == 0 ? sycl::native::exp(sinks[head] - m) : 0;
#pragma unroll
    for (int row = item_ct1.get_local_id(2); row < n;
         row += item_ct1.get_local_range(2)) {
      scores[row] = sycl::native::exp(scores[row] - m); denom += scores[row];
    }
    denom=sum(denom);
#pragma unroll
    for (int row = item_ct1.get_local_id(2); row < n;
         row += item_ct1.get_local_range(2)) weights[row] = scores[row] / denom;
}
inline uint16_t global_value_bits(const uint16_t* pointer) {
    // CUDA's ld.global.ca cache policy has no SYCL spelling; the bits are an ordinary F16 load.
    return *pointer;
}

template <bool Prefetch = false, bool Global = false>

__dpct_inline__ void
attention_value_kernel(const float *weights, const unsigned long long *pointers,
                       float *out, int d, int heads, int window,
                       const int32_t *positions, int ratio, int selected_limit,
                       int stride, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto stage = (unsigned char *)dpct_local;
    auto* probability=reinterpret_cast<float*>(stage);
    auto* row_address=reinterpret_cast<unsigned long long*>(stage+((size_t(stride)*4+7)&~size_t(7)));
    const int head = item_ct1.get_group(2), token = item_ct1.get_group(1),
              tid = item_ct1.get_local_id(2),
              dim = int(item_ct1.get_group(0)) * 64 + tid;
    const int pos = positions[token], first = sycl::max(0, pos - window + 1);
    const int n = pos - first + 1 +
                  (ratio ? sycl::min((pos + 1) / ratio, selected_limit) : 0);
    const size_t slot=(size_t(token)*heads+head)*stride;
#pragma unroll
    for (int row = tid; row < n; row += 64) {
      probability[row] = weights[slot + row];
      row_address[row] = pointers[slot + row];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if(dim<d) {
        float v=0;int row=0;
        if constexpr (Prefetch) {
            for(;row+7<n;row+=8) {
                uint16_t raw[8];float weight[8];
#pragma unroll
                for(int j=0;j<8;++j) {
                    const auto* value=reinterpret_cast<const uint16_t*>(row_address[row+j]);
                    if constexpr (Global) raw[j]=global_value_bits(value+dim); else raw[j]=value[dim];
                    weight[j]=probability[row+j];
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) v +=
                    sycl::vec<sycl::half, 1>(
                        sycl::bit_cast<sycl::half, unsigned short>(raw[j]))
                        .convert<float, sycl::rounding_mode::automatic>()[0] *
                    weight[j];
            }
        }
        for(;row<n;++row) {
            if constexpr (Global) {
                const auto* value=reinterpret_cast<const uint16_t*>(row_address[row]);
                v += sycl::vec<sycl::half, 1>(
                         sycl::bit_cast<sycl::half, unsigned short>(
                             global_value_bits(value + dim)))
                         .convert<float, sycl::rounding_mode::automatic>()[0] *
                     probability[row];
            } else {
                const auto *value =
                    reinterpret_cast<const sycl::half *>(row_address[row]);
                v += sycl::vec<sycl::half, 1>(value[dim])
                         .convert<float, sycl::rounding_mode::automatic>()[0] *
                     probability[row];
            }
        }
        out[(size_t(token)*heads+head)*d+dim]=v;
    }
}

// Every head of a token has the same raw/sparse row list. Share only its
// pointer and exact half read; each head keeps its original probability and
// sequential row-ordered FMA recurrence. Two heads retain128 tiles at NT1.
__dpct_inline__ void attention_value_head_pair_kernel(
    const float *weights, const unsigned long long *pointers, float *out, int d,
    int heads, int window, const int32_t *positions, int ratio,
    int selected_limit, int stride, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto stage = (unsigned char *)dpct_local;
    auto* p0=reinterpret_cast<float*>(stage);
    auto* p1=p0+stride;
    auto* rows=reinterpret_cast<unsigned long long*>(p1+stride);
    const int head = 2 * int(item_ct1.get_group(2)),
              token = item_ct1.get_group(1), tid = item_ct1.get_local_id(2),
              dim = int(item_ct1.get_group(0)) * 64 + tid;
    const int pos = positions[token], first = sycl::max(0, pos - window + 1);
    const int n = pos - first + 1 +
                  (ratio ? sycl::min((pos + 1) / ratio, selected_limit) : 0);
    const size_t slot=(size_t(token)*heads+head)*stride;
#pragma unroll
    for (int row = tid; row < n; row += 64) {
        p0[row]=weights[slot+row];p1[row]=weights[slot+stride+row];rows[row]=pointers[slot+row];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if(dim<d) {
        float v0=0,v1=0;
        for(int row=0;row<n;++row) {
            const auto* value=reinterpret_cast<const uint16_t*>(rows[row]);
            const float x =
                sycl::vec<sycl::half, 1>(
                    sycl::bit_cast<sycl::half, unsigned short>(
                        global_value_bits(value + dim)))
                    .convert<float, sycl::rounding_mode::automatic>()[0];
            v0+=x*p0[row];v1+=x*p1[row];
        }
        out[(size_t(token)*heads+head)*d+dim]=v0;
        out[(size_t(token)*heads+head+1)*d+dim]=v1;
    }
}

// Preserve each head's rounded softmax weights and row-ordered FMA.
template<int Group>
__dpct_inline__ void attention_value_head_group_kernel(
    const float* weights, const unsigned long long* pointers, float* out,
    int d, int heads, int window, const int32_t* positions, int ratio,
    int selected_limit, int stride, uint8_t* local, bool shared_rows) {
    const auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto* probabilities = reinterpret_cast<float*>(local);
    auto* rows = reinterpret_cast<unsigned long long*>(probabilities + Group*stride);
    const int head = Group*int(item.get_group(2));
    const int token = item.get_group(1), tid = item.get_local_id(2);
    const int dim = 64*int(item.get_group(0)) + tid;
    const int pos = positions[token], first = sycl::max(0, pos-window+1);
    const int count = pos-first+1 +
        (ratio ? sycl::min((pos+1)/ratio, selected_limit) : 0);
    const size_t slot = (size_t(token)*heads+head)*stride;
    for (int row=tid; row<count; row+=64) {
        #pragma unroll
        for (int h=0; h<Group; ++h)
            probabilities[h*stride+row] = weights[slot+size_t(h)*stride+row];
        rows[row] = pointers[(shared_rows?size_t(token)*stride:slot)+row];
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (dim<d) {
        float values[Group] = {};
        for (int row=0; row<count; ++row) {
            const auto* value = reinterpret_cast<const uint16_t*>(rows[row]);
            const float x = sycl::vec<sycl::half,1>(
                sycl::bit_cast<sycl::half,unsigned short>(global_value_bits(value+dim)))
                .convert<float,sycl::rounding_mode::automatic>()[0];
            #pragma unroll
            for (int h=0; h<Group; ++h)
                values[h] += x*probabilities[h*stride+row];
        }
        #pragma unroll
        for (int h=0; h<Group; ++h)
            out[(size_t(token)*heads+head+h)*d+dim] = values[h];
    }
}

}  // namespace

void rms(const float *x, const float *w, float *y, int n, int rows, float eps,
         void *s) {


  stream_cast(s)->parallel_for(
    sycl::nd_range<3>(sycl::range(1, 1, rows) * sycl::range(1, 1, threads), sycl::range(1, 1, threads)),
[=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
        rms_kernel(x, w, y, n, eps);
    }); }
void hc_init(const float *x, float *r, float *pre, int n, int t, void *s) {


  stream_cast(s)->parallel_for(
    sycl::nd_range<3>(sycl::range(1, 1, t) * sycl::range(1, 1, threads), sycl::range(1, 1, threads)),
[=](sycl::nd_item<3> item_ct1) {
        init_kernel(x, r, pre, n);
    }); }
void hc_coeff(const float* m,const float* sc,const float* b,float* pre,float* post,float* c,int t,float eps,int it,void* s) {
    if (it < 1) throw std::invalid_argument("Sinkhorn needs at least one iteration");
    {


        stream_cast(s)
            ->parallel_for(
                sycl::nd_range<3>(sycl::range(1, 1, t) * sycl::range(1, 1, 32),
                                  sycl::range(1, 1, 32)),
                [=](sycl::nd_item<3> item_ct1) {
                    coeff_kernel(m, sc, b, pre, post, c, eps, it);
                });
    }
}
void hc_pre(const float *r, const float *pre, float *y, int n, int t, void *s) {


  stream_cast(s)->parallel_for(
    sycl::nd_range<3>(sycl::range(1, t, (n+255)/256) * sycl::range(1, 1, threads), sycl::range(1, 1, threads)),
[=](sycl::nd_item<3> item_ct1) {
        pre_kernel(r, pre, y, n);
    }); }
void hc_post(const float *x, const float *r, const float *post, const float *c,
             float *y, int n, int t, void *s) {


  stream_cast(s)->parallel_for(
    sycl::nd_range<3>(sycl::range(1, t, (n+255)/256) * sycl::range(1, 1, threads), sycl::range(1, 1, threads)),
[=](sycl::nd_item<3> item_ct1) {
        post_kernel(x, r, post, c, y, n);
    }); }
void engram_add(float *r, const float *kv, const float *qw, const float *kw,
                float *gates, int n, int t, float eps, void *s) {


  stream_cast(s)->parallel_for(
    sycl::nd_range<3>(sycl::range(1, 1, t*4) * sycl::range(1, 1, threads), sycl::range(1, 1, threads)),
[=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
        engram_kernel(r, kv, qw, kw, gates, n, eps);
    }); }
void route(const float* l,const float* b,int32_t* ids,float* w,int e,int k,int t,float sc,void* s) {
    if (e < 1 || e > 512 || k < 1 || k > 64 || k > e) throw std::invalid_argument("DeepSeek routing geometry");
    {


        stream_cast(s)
            ->parallel_for(
                sycl::nd_range<3>(sycl::range(1, 1, t) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    route_kernel(l, b, ids, w, e, k, sc);
                });
    }
}
void swiglu(const float *g, const float *u, float *y, int n, float limit,
            void *s) {


  stream_cast(s)->parallel_for(
    sycl::nd_range<3>(sycl::range(1, 1, (n+255)/256) * sycl::range(1, 1, threads), sycl::range(1, 1, threads)),
[=](sycl::nd_item<3> item_ct1) {
        swiglu_kernel(g, u, y, n, limit);
    }); }
void combine(const float *p, const float *w, const float *sh, float *y, int h,
             int k, int t, void *s) {


  stream_cast(s)->parallel_for(
    sycl::nd_range<3>(sycl::range(1, t, (h+255)/256) * sycl::range(1, 1, threads), sycl::range(1, 1, threads)),
[=](sycl::nd_item<3> item_ct1) {
        combine_kernel(p, w, sh, y, h, k);
    }); }
void combine_sources(const float* p,const float* cpu,const uint64_t* mask,const float* w,const float* sh,float* y,int h,int k,int t,void* s) {


    stream_cast(s)
        ->parallel_for(
            sycl::nd_range<3>(sycl::range(1, t, (h + 255) / 256) *
                                  sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            [=](sycl::nd_item<3> item_ct1) {
                combine_sources_kernel<false>(p, cpu, mask, w, sh, y, h, k);
            });
}
void combine_sources_system(const float* p,const float* cpu,const uint64_t* mask,const float* w,const float* sh,float* y,int h,int k,int t,void* s) {


    stream_cast(s)
        ->parallel_for(
            sycl::nd_range<3>(sycl::range(1, t, (h + 255) / 256) *
                                  sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            [=](sycl::nd_item<3> item_ct1) {
                combine_sources_kernel<true>(p, cpu, mask, w, sh, y, h, k);
            });
}
void wait_host(const uint32_t* ready,const uint32_t* expected,const uint32_t* abort,void* s) {
    throw std::runtime_error("DeepSeek SYCL graph/TP host waits are not implemented");
}
void dense(const float *w, const float *x, float *y, int ni, int no, int t,
           void *s) {


  stream_cast(s)->parallel_for(
    sycl::nd_range<3>(sycl::range(1, t, no) * sycl::range(1, 1, threads), sycl::range(1, 1, threads)),
[=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
        dense_kernel(w, x, y, ni, no);
    }); }
void add(const float *x, const float *y, float *out, int n, void *s) {


  stream_cast(s)->parallel_for(
    sycl::nd_range<3>(sycl::range(1, 1, (n+255)/256) * sycl::range(1, 1, threads), sycl::range(1, 1, threads)),
[=](sycl::nd_item<3> item_ct1) {
        add_kernel(x, y, out, n);
    }); }
void tp_sum_wait(const float* local,const float* peer,const uint32_t* own_ready,
    const uint32_t* peer_ready,const uint32_t* abort,float* out,int n,int rank,void* s) {
    throw std::runtime_error("DeepSeek SYCL graph/TP host waits are not implemented");
}
void strided_copy(const float *x, float *y, int w, int rows, int xs, int ys,
                  void *s) {


  stream_cast(s)->parallel_for(
    sycl::nd_range<3>(sycl::range(1, rows, (w+255)/256) * sycl::range(1, 1, threads), sycl::range(1, 1, threads)),
[=](sycl::nd_item<3> item_ct1) {
        strided_copy_kernel(x, y, w, xs, ys);
    }); }
void gather_row(const float* rows,const int32_t* id,float* out,int w,int count,void* s) {


    stream_cast(s)
        ->parallel_for(
            sycl::nd_range<3>(sycl::range(1, 1, (w + 255) / 256) *
                                  sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            [=](sycl::nd_item<3> item_ct1) {
                gather_row_kernel(rows, id, out, w, count);
            });
}
void draft_confidence(const float* h,const float* m,const float* w,const float* b,float* out,int nh,int nm,void* s) {


    stream_cast(s)
        ->parallel_for(
            sycl::nd_range<3>(sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                draft_confidence_kernel(h, m, w, b, out, nh, nm);
            });
}
void hc_mean(const float *r, float *y, int h, int t, int stride, void *s) {


  stream_cast(s)->parallel_for(
    sycl::nd_range<3>(sycl::range(1, t, (h+255)/256) * sycl::range(1, 1, threads), sycl::range(1, 1, threads)),
[=](sycl::nd_item<3> item_ct1) {
        mean_kernel(r, y, h, stride);
    }); }
void rope(float* x,int d,int heads,int t,const int32_t* pos,Rope c,bool inv,void* s) {
    if (c.dim < 2 || c.dim%2 || c.dim > d || c.base <= 1 || c.scale <= 0) throw std::invalid_argument("DeepSeek RoPE geometry");
    float low = 0, high = 0;
    if (c.yarn) {
        if (c.original_context <= 0 || c.beta_fast <= 0 || c.beta_slow <= 0) throw std::invalid_argument("DeepSeek YaRN configuration");
        auto corr = [&](float beta) {
                                      return c.dim *
                                             sycl::log(
                                                 c.original_context /
                                                 (beta * 2.0f *
                                                  3.14159265358979323846f)) /
                                             (2 * sycl::log(c.base)); };
        low = std::fmax(0.0f,std::floor(corr(c.beta_fast))); high = std::fmin(float(c.dim-1),std::ceil(corr(c.beta_slow)));
    }
    {


        stream_cast(s)
            ->parallel_for(
                sycl::nd_range<3>(
                    sycl::range(1, t, (heads * c.dim / 2 + 255) / 256) *
                        sycl::range(1, 1, threads),
                    sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    rope_kernel(x, d, heads, pos, c, low, high, inv);
                });
    }
}
void store_f16(const float *x, uint16_t *c, int n, int row, void *s) {

    dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                 {sycl::aspect::fp16});

  stream_cast(s)->parallel_for(
    sycl::nd_range<3>(sycl::range(1, 1, (n+255)/256) * sycl::range(1, 1, threads), sycl::range(1, 1, threads)),
[=](sycl::nd_item<3> item_ct1) {
        store_kernel(x, reinterpret_cast<sycl::half *>(c), n, row);
    }); }
void pool(const float* v,const float* scores,float* y,int n,int ratio,void* s) {
    if (ratio != 1 && ratio != 2) throw std::invalid_argument("DeepSeek pool ratio");
    {


        stream_cast(s)
            ->parallel_for(
                sycl::nd_range<3>(sycl::range(1, 1, (n + 255) / 256) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    pool_kernel(v, scores, y, n, ratio);
                });
    }
}
void store_f16_positions(const float* x,uint16_t* cache,int w,int t,const int32_t* pos,int capacity,int ratio,void* s) {
    if (capacity < 1 || ratio < 0 || ratio > 2) throw std::invalid_argument("DeepSeek position store geometry");
    {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)
            ->parallel_for(
                sycl::nd_range<3>(sycl::range(1, t, (w + 255) / 256) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    store_positions_kernel(
                        x, reinterpret_cast<sycl::half *>(cache), w, pos,
                        capacity, ratio);
                });
    }
}
void projection_ring(const float* v,const float* scores,float* rv,float* rs,int w,int t,const int32_t* pos,int capacity,void* s) {
    if (capacity < t) throw std::invalid_argument("DeepSeek projection ring capacity");
    {


        stream_cast(s)
            ->parallel_for(
                sycl::nd_range<3>(sycl::range(1, t, (w + 255) / 256) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    projection_ring_kernel(v, scores, rv, rs, w, pos, capacity);
                });
    }
}
void pool_positions(const float* v,const float* scores,float* y,int w,int t,const int32_t* pos,int capacity,int ratio,void* s) {
    if (capacity < t+ratio-1 || (ratio != 1 && ratio != 2)) throw std::invalid_argument("DeepSeek position pool geometry");
    {


        stream_cast(s)
            ->parallel_for(
                sycl::nd_range<3>(sycl::range(1, t, (w + 255) / 256) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    pool_positions_kernel(v, scores, y, w, pos, capacity,
                                          ratio);
                });
    }
}
void attention(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,const float* sinks,float* out,
               int d,int heads,int window,int pos,int selected,void* s,int capacity) {
    if (pos < 0 || window < 1 || selected < 0 || (selected && !comp)) throw std::invalid_argument("DeepSeek attention state");
    {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(size_t(window + selected) * 4), cgh);


            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(1, 1, heads) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attention_kernel<false>(
                            q, reinterpret_cast<const sycl::half *>(raw),
                            reinterpret_cast<const sycl::half *>(comp), ids,
                            sinks, out, d, window, pos, selected,
                            capacity ? capacity : window, pos, nullptr, heads,
                            0, 0, 0,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            nullptr, nullptr, 0);
                    });
        });
    }
}
void draft_attention(const float* q,const uint16_t* raw,const float* sinks,float* out,int d,int heads,int window,int pos,int end,int capacity,void* s) {
    if (end < pos || capacity < window+end-pos) throw std::invalid_argument("DSpark noncausal window capacity");
    {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(size_t(window + end - pos) * 4), cgh);


            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(1, 1, heads) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attention_kernel<false>(
                            q, reinterpret_cast<const sycl::half *>(raw),
                            nullptr, nullptr, sinks, out, d, window, pos, 0,
                            capacity, end, nullptr, heads, 0, 0, 0,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            nullptr, nullptr, 0);
                    });
        });
    }
}
void draft_attention_positions(const float* q,const uint16_t* raw,const float* sinks,float* out,int d,int heads,int tokens,int window,const int32_t* pos,int capacity,void* s) {

    dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                 {sycl::aspect::fp16});

    stream_cast(s)->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range(size_t(window + tokens - 1) * 4), cgh);


        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range(1, tokens, heads) *
                                  sycl::range(1, 1, threads),
                              sycl::range(1, 1, threads)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                attention_kernel<true, true>(
                    q, reinterpret_cast<const sycl::half *>(raw), nullptr,
                    nullptr, sinks, out, d, window, 0, 0, capacity, 0, pos,
                    heads, 0, 0, 0,
                    dpct_local_acc_ct1
                        .get_multi_ptr<sycl::access::decorated::no>()
                        .get(),
                    nullptr, nullptr, 0);
            });
    });
}
void index_scores(const float* q,const float* w,const uint16_t* keys,float* scores,int d,int heads,int rows,void* s) {
    if (heads > 32 || heads < 1) throw std::invalid_argument("DeepSeek indexer heads");
    if (rows) {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)
            ->parallel_for(
                sycl::nd_range<3>(sycl::range(1, 1, rows) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        index_kernel<false>(
                            q, w, reinterpret_cast<const sycl::half *>(keys),
                            scores, d, heads, nullptr, 1, 0);
                    });
    }
}
void index_scores_positions(const float* q,const float* w,const uint16_t* keys,float* scores,int d,int heads,
                            int bucket,int tokens,const int32_t* pos,int ratio,int stride,void* s) {
    if (heads < 1 || heads > 32 || bucket < 1 || bucket > stride || (ratio != 1 && ratio != 2)) throw std::invalid_argument("DeepSeek bucket index geometry");
    {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)
            ->parallel_for(
                sycl::nd_range<3>(sycl::range(1, tokens, bucket) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        index_kernel<true>(
                            q, w, reinterpret_cast<const sycl::half *>(keys),
                            scores, d, heads, pos, ratio, stride);
                    });
    }
}
void attention_positions(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,const float* sinks,
                         float* out,int d,int heads,int tokens,int window,const int32_t* pos,int ratio,int max_selected,
                         int ids_stride,int raw_capacity,void* s) {
    if (window < 1 || raw_capacity < window+tokens || ratio < 0 || ratio > 2 || max_selected < 0 || (ratio && !comp)) throw std::invalid_argument("DeepSeek position attention geometry");
    {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(size_t(window + max_selected) * 4), cgh);


            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(1, tokens, heads) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attention_kernel<true>(
                            q, reinterpret_cast<const sycl::half *>(raw),
                            reinterpret_cast<const sycl::half *>(comp), ids,
                            sinks, out, d, window, 0, 0, raw_capacity, 0, pos,
                            heads, ratio, max_selected, ids_stride,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            nullptr, nullptr, 0);
                    });
        });
    }
}
void attention_positions_normalized(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,const float* sinks,
                         float* out,int d,int heads,int tokens,int window,const int32_t* pos,int ratio,int max_selected,
                         int ids_stride,int raw_capacity,void* s) {
    if (window < 1 || raw_capacity < window+tokens || ratio < 0 || ratio > 2 || max_selected < 0 || (ratio && !comp)) throw std::invalid_argument("DeepSeek position attention geometry");
    {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(size_t(window + max_selected) * 4), cgh);


            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(1, tokens, heads) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attention_kernel<true, false, true>(
                            q, reinterpret_cast<const sycl::half *>(raw),
                            reinterpret_cast<const sycl::half *>(comp), ids,
                            sinks, out, d, window, 0, 0, raw_capacity, 0, pos,
                            heads, ratio, max_selected, ids_stride,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            nullptr, nullptr, 0);
                    });
        });
    }
}
void attention_positions_chunked(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,float* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s) {
    if(d!=DsDim || heads<1 || heads%DsHeads || tokens<1 || tokens>8 || window<1 || window>128 ||
       raw_capacity<window+tokens || ratio<0 || ratio>2 || max_selected<0 || max_selected>512 ||
       (ratio && !comp) || !scratch) throw std::invalid_argument("DeepSeek split attention geometry");
    const int chunks=(window+max_selected+DsChunk-1)/DsChunk;
    {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)
            ->parallel_for(
                sycl::nd_range<3>(sycl::range(tokens, heads / DsHeads, chunks) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attention_chunk_kernel(
                            q, reinterpret_cast<const sycl::half *>(raw),
                            reinterpret_cast<const sycl::half *>(comp), ids,
                            scratch, heads, window, pos, ratio, max_selected,
                            ids_stride, raw_capacity, chunks);
                    });
    }
    {


        stream_cast(s)
            ->parallel_for(
                sycl::nd_range<3>(sycl::range(1, tokens, heads) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attention_chunk_merge_kernel(scratch, sinks, out, heads,
                                                     window, pos, ratio,
                                                     max_selected, chunks);
                    });
    }
}
void attention_positions_values(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s) {
    if(d!=512 || heads<1 || tokens<1 || tokens>8 || window!=128 || raw_capacity<window+tokens ||
       ratio<0 || ratio>2 || (max_selected!=0 && max_selected!=512) || (ratio && !comp) || !scratch)
        throw std::invalid_argument("DeepSeek parallel value geometry");
    const int stride=window+max_selected;
    const size_t scores=size_t(heads)*tokens*stride;
    auto* weights=static_cast<float*>(scratch);
    auto* rows=reinterpret_cast<unsigned long long*>(static_cast<unsigned char*>(scratch)+((scores*4+255)&~size_t(255)));
    {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(size_t(stride) * 4), cgh);


            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(1, tokens, heads) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attention_kernel<true, false, false, true>(
                            q, reinterpret_cast<const sycl::half *>(raw),
                            reinterpret_cast<const sycl::half *>(comp), ids,
                            sinks, out, d, window, 0, 0, raw_capacity, 0, pos,
                            heads, ratio, max_selected, ids_stride,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            weights, rows, stride);
                    });
        });
    }
    const size_t shared=((size_t(stride)*4+7)&~size_t(7))+size_t(stride)*8;
    {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(shared), cgh);


            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(8, tokens, heads) *
                                      sycl::range(1, 1, 64),
                                  sycl::range(1, 1, 64)),
                [=](sycl::nd_item<3> item_ct1) {
                    attention_value_kernel<false>(
                        weights, rows, out, d, heads, window, pos, ratio,
                        max_selected, stride,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
}
void attention_positions_values_prefetch(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s) {
    if(d!=512 || heads<1 || tokens<1 || tokens>8 || window!=128 || raw_capacity<window+tokens ||
       ratio<0 || ratio>2 || (max_selected!=0 && max_selected!=512) || (ratio && !comp) || !scratch)
        throw std::invalid_argument("DeepSeek parallel value geometry");
    const int stride=window+max_selected;
    const size_t scores=size_t(heads)*tokens*stride;
    auto* weights=static_cast<float*>(scratch);
    auto* rows=reinterpret_cast<unsigned long long*>(static_cast<unsigned char*>(scratch)+((scores*4+255)&~size_t(255)));
    {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(size_t(stride) * 4), cgh);


            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(1, tokens, heads) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attention_kernel<true, false, false, true>(
                            q, reinterpret_cast<const sycl::half *>(raw),
                            reinterpret_cast<const sycl::half *>(comp), ids,
                            sinks, out, d, window, 0, 0, raw_capacity, 0, pos,
                            heads, ratio, max_selected, ids_stride,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            weights, rows, stride);
                    });
        });
    }
    const size_t shared=((size_t(stride)*4+7)&~size_t(7))+size_t(stride)*8;
    {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(shared), cgh);


            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(8, tokens, heads) *
                                      sycl::range(1, 1, 64),
                                  sycl::range(1, 1, 64)),
                [=](sycl::nd_item<3> item_ct1) {
                    attention_value_kernel<true>(
                        weights, rows, out, d, heads, window, pos, ratio,
                        max_selected, stride,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
}
void attention_positions_query_prefetch(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s,bool value_prefetch) {
    if(d!=512 || heads<1 || tokens<1 || tokens>8 || window!=128 || raw_capacity<window+tokens ||
       ratio<0 || ratio>2 || (max_selected!=0 && max_selected!=512) || (ratio && !comp) || !scratch)
        throw std::invalid_argument("DeepSeek parallel value geometry");
    const int stride=window+max_selected;
    const size_t scores=size_t(heads)*tokens*stride;
    auto* weights=static_cast<float*>(scratch);
    auto* rows=reinterpret_cast<unsigned long long*>(static_cast<unsigned char*>(scratch)+((scores*4+255)&~size_t(255)));
    {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(size_t(stride) * 4), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(1, tokens, heads) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attention_kernel<true, false, false, true, true>(
                            q, reinterpret_cast<const sycl::half *>(raw),
                            reinterpret_cast<const sycl::half *>(comp), ids,
                            sinks, out, d, window, 0, 0, raw_capacity, 0, pos,
                            heads, ratio, max_selected, ids_stride,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get(),
                            weights, rows, stride);
                    });
        });
    }
    const size_t shared=((size_t(stride)*4+7)&~size_t(7))+size_t(stride)*8;
    if (value_prefetch) {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(shared), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(8, tokens, heads) *
                                      sycl::range(1, 1, 64),
                                  sycl::range(1, 1, 64)),
                [=](sycl::nd_item<3> item_ct1) {
                    attention_value_kernel<true, true>(
                        weights, rows, out, d, heads, window, pos, ratio,
                        max_selected, stride,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } else {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(shared), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(8, tokens, heads) *
                                      sycl::range(1, 1, 64),
                                  sycl::range(1, 1, 64)),
                [=](sycl::nd_item<3> item_ct1) {
                    attention_value_kernel<false, true>(
                        weights, rows, out, d, heads, window, pos, ratio,
                        max_selected, stride,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
}
template<bool PairedQK>
void attention_positions_score_grid_impl(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s,bool pair_heads) {
    if(d!=512 || heads<1 || tokens<1 || tokens>8 || window!=128 || raw_capacity<window+tokens ||
       ratio<0 || ratio>2 || (max_selected!=0 && max_selected!=512) || (ratio && !comp) || !scratch || (pair_heads && heads%2))
        throw std::invalid_argument("DeepSeek score-grid geometry");
    const int stride=window+max_selected;
    const size_t count=size_t(heads)*tokens*stride;
    auto* weights=static_cast<float*>(scratch);
    auto* rows=reinterpret_cast<unsigned long long*>(static_cast<unsigned char*>(scratch)+((count*4+255)&~size_t(255)));
    {

        dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                     {sycl::aspect::fp16});

        stream_cast(s)
            ->parallel_for(
                sycl::nd_range<3>(sycl::range((stride + 7) / 8, tokens, heads/(PairedQK?2:1)) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attention_score_grid_kernel<false,PairedQK?2:1>(
                            q, reinterpret_cast<const sycl::half *>(raw),
                            reinterpret_cast<const sycl::half *>(comp), ids,
                            weights, rows, d, heads, window, pos, ratio,
                            max_selected, ids_stride, raw_capacity, stride);
                    });
    }
    {


        stream_cast(s)->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range(size_t(stride) * 4), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range(1, tokens, heads) *
                                      sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        attention_score_softmax_kernel(
                            weights, sinks, heads, window, pos, ratio,
                            max_selected, stride,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
        });
    }
    if(pair_heads) {

        const size_t shared =
            size_t(stride) * (2 * sizeof(float) + sizeof(unsigned long long));
        {


            stream_cast(s)->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range(8, tokens, heads / 2) *
                                          sycl::range(1, 1, 64),
                                      sycl::range(1, 1, 64)),
                    [=](sycl::nd_item<3> item_ct1) {
                        attention_value_head_pair_kernel(
                            weights, rows, out, d, heads, window, pos, ratio,
                            max_selected, stride,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        }
    } else {
        const size_t shared=((size_t(stride)*4+7)&~size_t(7))+size_t(stride)*8;
        {

            dpct::has_capability_or_fail(stream_cast(s)->get_device(),
                                         {sycl::aspect::fp16});

            stream_cast(s)->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range(8, tokens, heads) *
                                          sycl::range(1, 1, 64),
                                      sycl::range(1, 1, 64)),
                    [=](sycl::nd_item<3> item_ct1) {
                        attention_value_kernel<false, true>(
                            weights, rows, out, d, heads, window, pos, ratio,
                            max_selected, stride,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
        }
    }
}
void attention_positions_score_grid(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s,bool pair_heads) {
    attention_positions_score_grid_impl<false>(q,raw,comp,ids,sinks,out,scratch,d,heads,tokens,window,pos,ratio,max_selected,ids_stride,raw_capacity,s,pair_heads);
}

bool attention_positions_score_grid_paired(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s,bool pair_heads) {
    if(heads<2 || heads%2)return false;
    attention_positions_score_grid_impl<true>(q,raw,comp,ids,sinks,out,scratch,d,heads,tokens,window,pos,ratio,max_selected,ids_stride,raw_capacity,s,pair_heads);
    return true;
}

void top_k(float* scores,int32_t* ids,int rows,int k,void* s) {
    if (k < 0 || k > rows) throw std::invalid_argument("DeepSeek indexer top-k");
    if (k) {


        stream_cast(s)
            ->parallel_for(
                sycl::nd_range<3>(sycl::range(1, 1, threads),
                                  sycl::range(1, 1, threads)),
                [=](sycl::nd_item<3> item_ct1) {
                    topk_kernel(scores, ids, rows, k);
                });
    }
}
size_t top_k_scratch_bytes(int rows) {
    if (rows < 1) throw std::invalid_argument("DeepSeek top-k rows");
    // Two merge planes. A key includes the original row so equal scores have
    // a deterministic order without another payload plane or host sorting.
    return size_t(rows)*2*sizeof(uint64_t);
}
void top_k_sorted(const float* scores,int32_t* ids,int rows,int k,void* scratch,size_t bytes,void* stream) {
    if (k < 0 || k > rows) throw std::invalid_argument("DeepSeek top-k geometry");
    if (!k) return;
    if (!scratch || bytes < top_k_scratch_bytes(rows)) throw std::invalid_argument("DeepSeek top-k scratch");
    auto& queue=*stream_cast(stream);
    if(k==1){
        queue.memcpy(scratch,scores,size_t(rows)*sizeof(float));
        top_k(static_cast<float*>(scratch),ids,rows,k,stream);
        return;
    }
    auto* source=static_cast<uint64_t*>(scratch);
    auto* destination=source+rows;
    constexpr size_t tile=256;
    const size_t padded=(size_t(rows)+tile-1)/tile*tile;
    queue.submit([&](sycl::handler& handler){
        sycl::local_accessor<uint64_t,1> local(tile,handler);
        handler.parallel_for(sycl::nd_range<1>(padded,tile),[=](sycl::nd_item<1> item){
            const size_t at=item.get_global_id(0),lane=item.get_local_id(0);
            uint64_t key=UINT64_MAX;
            if(at<size_t(rows)){
                const float score=scores[at];
                // Numeric ordering, with +0 and -0 tied, then original row.
                // NaNs sort last; ordinary indexer scores are finite.
                const uint32_t bits=sycl::bit_cast<uint32_t>(score==0.0f?0.0f:score);
                const uint32_t ascending=(bits&0x80000000u)?~bits:(bits^0x80000000u);
                const uint32_t descending=(bits&0x7fffffffu)>0x7f800000u?UINT32_MAX:~ascending;
                key=(uint64_t(descending)<<32)|uint32_t(at);
            }
            local[lane]=key;
            item.barrier(sycl::access::fence_space::local_space);
            // Sort one tile cooperatively, rather than taking k successive
            // maxima with a full global scan and block reduction each time.
            for(size_t width=2;width<=tile;width*=2)for(size_t stride=width/2;stride;stride/=2){
                const uint64_t own=local[lane],other=local[lane^stride];
                const bool minimum=((lane&width)==0)==((lane&stride)==0);
                item.barrier(sycl::access::fence_space::local_space);
                local[lane]=minimum?sycl::min(own,other):sycl::max(own,other);
                item.barrier(sycl::access::fence_space::local_space);
            }
            if(at<size_t(rows))source[at]=local[lane];
        });
    });
    // Each key finds its unique rank in the adjacent sorted run. All passes
    // are queued asynchronously in order, with no host round-trip or wait.
    for(size_t width=tile;width<size_t(rows);width*=2){
        queue.parallel_for(sycl::range<1>(size_t(rows)),[=](sycl::id<1> item){
            const size_t at=item[0],base=at/(2*width)*(2*width);
            const bool right=at-base>=width;
            const size_t own=base+(right?width:0),other=base+(right?0:width);
            const size_t count=other<size_t(rows)?sycl::min(width,size_t(rows)-other):0;
            const uint64_t key=source[at];
            size_t low=0,high=count;
            while(low<high){
                const size_t middle=low+(high-low)/2;
                if(source[other+middle]<key)low=middle+1;else high=middle;
            }
            destination[base+(at-own)+low]=key;
        });
        std::swap(source,destination);
    }
    queue.parallel_for(sycl::range<1>(size_t(k)),[=](sycl::id<1> item){
        ids[item[0]]=int32_t(uint32_t(source[item[0]]));
    });
}

template<bool PairedQK>
bool attention_positions_prefill_impl(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s) {
    // Small/speculative batches and unsupported layouts keep their original
    // caller dispatch. Four heads reuse one value while retaining each FMA.
    if(d!=512 || heads<4 || heads%4 || tokens<=8 || tokens>128 || window!=128 ||
       raw_capacity<window+tokens || ratio<1 || ratio>2 || max_selected!=512 ||
       !q || !raw || !comp || !sinks || !out || !scratch || !pos ||
       (ids && ids_stride<max_selected)) return false;
    const int stride=window+max_selected;
    const size_t count=size_t(heads)*tokens*stride;
    auto* weights=static_cast<float*>(scratch);
    // All heads of one token use exactly the same row-address list.
    auto* rows=reinterpret_cast<unsigned long long*>(static_cast<unsigned char*>(scratch)+((count*4+255)&~size_t(255)));
    auto* queue=stream_cast(s);
    dpct::has_capability_or_fail(queue->get_device(),{sycl::aspect::fp16});
    queue->parallel_for(
        sycl::nd_range<3>(sycl::range<3>((stride+7)/8,tokens,heads/(PairedQK?2:1))*sycl::range<3>(1,1,threads),
                          sycl::range<3>(1,1,threads)),
        [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
            attention_score_grid_kernel<true,PairedQK?2:1>(
                q,reinterpret_cast<const sycl::half*>(raw),reinterpret_cast<const sycl::half*>(comp),
                ids,weights,rows,d,heads,window,pos,ratio,max_selected,ids_stride,raw_capacity,stride);
        });
    queue->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<uint8_t,1> local(sycl::range<1>(size_t(stride)*4),cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1,tokens,heads)*sycl::range<3>(1,1,threads),
                              sycl::range<3>(1,1,threads)),
            [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                attention_score_softmax_kernel(weights,sinks,heads,window,pos,ratio,max_selected,stride,
                    local.get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
    queue->submit([&](sycl::handler& cgh) {
        const size_t shared=size_t(stride)*(4*sizeof(float)+sizeof(unsigned long long));
        sycl::local_accessor<uint8_t,1> local(sycl::range<1>(shared),cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(8,tokens,heads/4)*sycl::range<3>(1,1,64),
                              sycl::range<3>(1,1,64)),
            [=](sycl::nd_item<3>) {
                attention_value_head_group_kernel<4>(weights,rows,out,d,heads,window,pos,ratio,max_selected,
                    stride,local.get_multi_ptr<sycl::access::decorated::no>().get(),true);
            });
    });
    return true;
}

bool attention_positions_prefill(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s) {return attention_positions_prefill_impl<false>(q,raw,comp,ids,sinks,out,scratch,d,heads,tokens,window,pos,ratio,max_selected,ids_stride,raw_capacity,s);}

bool attention_positions_prefill(const float* q,const uint16_t* raw,const uint16_t* comp,const int32_t* ids,
    const float* sinks,float* out,void* scratch,int d,int heads,int tokens,int window,const int32_t* pos,
    int ratio,int max_selected,int ids_stride,int raw_capacity,void* s,bool paired_qk) {
    return paired_qk?attention_positions_prefill_impl<true>(q,raw,comp,ids,sinks,out,scratch,d,heads,tokens,window,pos,ratio,max_selected,ids_stride,raw_capacity,s):attention_positions_prefill_impl<false>(q,raw,comp,ids,sinks,out,scratch,d,heads,tokens,window,pos,ratio,max_selected,ids_stride,raw_capacity,s);
}

}  // namespace strata::deepseek41::cuda
