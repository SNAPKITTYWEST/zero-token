/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-013
   File:              zt_kernels.cuh
   Parent-Work:       zero-token
   Copyright:         2026 BEL ESPRIT D ACCORD
   License-ID:        GPL-2.0 OR AGPL-3.0
   Covenant-Version:  1.0
   Compliance:        FAIL-CLOSED

   This file is governed by the GNU General Public License version 2.0 or the GNU Affero
   General Public License version 3.0, at your option, together with the applicable Sovereign
   Leviathan Covenant.

   Whatsoever branch this root shall bear,
   Must breathe the exact and sovereign air.
   Touch but a leaf, invoke a single thread,
   And honor still the terms beneath it spread.

   Ignorantia juris non excusat.

   Clone-Gate: sha256:55d190b20a25692c75993517dc7eb46ce0b8a73017a643320f3e4e869a7bc4b6

   See: LICENSE
   ======================================================================== */
#ifndef ZT_KERNELS_CUH
#define ZT_KERNELS_CUH

#include <cuda_runtime.h>
#include <mma.h>
#include "zt_core.h"

#define ZT_WARP       32u
#define ZT_HEAD_WARPS 4u          /* warps per block in the head kernels (128 threads) */
#define ZT_TILE       16u         /* WMMA m = n = k */

/* ------------------------------------------------------- warp reductions */

__device__ __forceinline__ float zt_warp_max(float v)
{
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}
__device__ __forceinline__ float zt_warp_sum(float v)
{
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}
__device__ __forceinline__ uint32_t zt_warp_min_u(uint32_t v)
{
    for (int off = 16; off > 0; off >>= 1) {
        uint32_t o = __shfl_xor_sync(0xffffffffu, v, off);
        v = (o < v) ? o : v;
    }
    return v;
}

/* --------------------------------------------------------------- probe */
/* Compiled for every target in ZT_CUDA_ARCH; if the device cannot run WMMA the
 * launch fails and the host falls back to the SIMT kernel.  A kernel that did
 * not actually touch tensor cores could be optimised away and would report a
 * false positive, so it performs a real fragment multiply and stores it. */
__global__ void zt_k_probe_wmma(float *out)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    using namespace nvcuda::wmma;
    __shared__ float sC[ZT_TILE * ZT_TILE];
    fragment<matrix_a, 16, 16, 16, __half, row_major> a;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b;
    fragment<accumulator, 16, 16, 16, float> c;
    fill_fragment(a, __float2half(1.0f));
    fill_fragment(b, __float2half(1.0f));
    fill_fragment(c, 0.0f);
    mma_sync(c, a, b, c);
    store_matrix_sync(sC, c, 16, mem_row_major);
    __syncthreads();
    if (threadIdx.x == 0) out[0] = sC[0];   /* == 16.0f on a working device */
#else
    if (threadIdx.x == 0) out[0] = 0.0f;
#endif
}

/* --------------------------------------------------------------- gather */

__global__ void zt_k_gather_logits(const uint8_t *logits, const int32_t *col_token,
                                   float *out, uint32_t ld_out, zt_gather_params prm)
{
    uint32_t n = prm.rows * prm.K;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    size_t esz = (size_t)zt_dtype_size(prm.dtype);
    for (; i < n; i += stride) {
        uint32_t r = i / prm.K, k = i - r * prm.K;
        const uint8_t *row = logits + (size_t)r * prm.ld * esz;
        out[(size_t)r * ld_out + k] = zt_load_f(row, prm.dtype, (uint32_t)col_token[k]);
    }
}

/* V x H head -> Kpad x H fp16 slice.  Rows K..Kpad are zeroed so the padded
 * WMMA tiles contribute exactly zero to the product. */
__global__ void zt_k_gather_head(const uint8_t *head, const int32_t *col_token,
                                 uint16_t *out, uint32_t Kpad, zt_head_params prm)
{
    uint32_t n = Kpad * prm.H;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;
    size_t esz = (size_t)zt_dtype_size(prm.dtype);
    for (; i < n; i += stride) {
        uint32_t k = i / prm.H, h = i - k * prm.H;
        if (k >= prm.K) { out[i] = 0u; continue; }
        {
            const uint8_t *row = head + (size_t)col_token[k] * prm.ld * esz;
            out[i] = zt_f32_to_f16_(zt_load_f(row, prm.dtype, h));
        }
    }
}

/* ------------------------------------------------------------ head: WMMA */
/* out[rows][Kpad] = hidden[rows][H] * W[Kpad][H]^T, fp16 inputs, fp32 accum.
 * grid = (ceil(Kpad/16/WARPS), ceil(rows/16)); block = 128 (4 warps).
 * Each warp owns one 16x16 output tile and stages its own A and B tiles, so
 * there is no block-wide barrier and out-of-range warps can return early. */
__global__ void zt_k_head_wmma(const uint8_t *hidden, const uint16_t *W, float *out,
                               uint32_t ld_out, zt_head_params prm)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    using namespace nvcuda::wmma;
    __shared__ __align__(32) __half sA[ZT_HEAD_WARPS][ZT_TILE * ZT_TILE];
    __shared__ __align__(32) __half sB[ZT_HEAD_WARPS][ZT_TILE * ZT_TILE];
    __shared__ __align__(32) float  sC[ZT_HEAD_WARPS][ZT_TILE * ZT_TILE];

    const uint32_t warp = threadIdx.x / ZT_WARP;
    const uint32_t lane = threadIdx.x % ZT_WARP;
    const uint32_t k0   = (blockIdx.x * ZT_HEAD_WARPS + warp) * ZT_TILE;
    const uint32_t r0   = blockIdx.y * ZT_TILE;
    const size_t   esz  = (size_t)zt_dtype_size(prm.dtype);
    uint32_t h0, i;

    if (k0 >= prm.Kpad || r0 >= prm.rows) return;   /* warp-uniform: safe */

    fragment<matrix_a, 16, 16, 16, __half, row_major> fa;
    fragment<matrix_b, 16, 16, 16, __half, col_major> fb;
    fragment<accumulator, 16, 16, 16, float> fc;
    fill_fragment(fc, 0.0f);

    for (h0 = 0; h0 < prm.H; h0 += ZT_TILE) {
        /* 256 elements per tile, 8 per lane. */
        for (i = lane; i < ZT_TILE * ZT_TILE; i += ZT_WARP) {
            uint32_t rr = i / ZT_TILE, hh = i - rr * ZT_TILE;
            uint32_t r = r0 + rr, h = h0 + hh;
            float a = 0.0f, b = 0.0f;
            if (r < prm.rows && h < prm.H) {
                const uint8_t *row = hidden + (size_t)r * prm.ld * esz;
                a = zt_load_f(row, prm.dtype, h);          /* dtype conversion happens here */
            }
            if (h < prm.H) b = zt_f16_to_f32_(W[(size_t)(k0 + rr) * prm.H + h]);
            sA[warp][rr * ZT_TILE + hh] = __float2half(a);
            /* sB holds W[k][h] row-major; loaded below as col_major so that
             * fragment element (h, k) == W[k][h], i.e. the transpose we want. */
            sB[warp][rr * ZT_TILE + hh] = __float2half(b);
        }
        __syncwarp();
        load_matrix_sync(fa, sA[warp], ZT_TILE);
        load_matrix_sync(fb, sB[warp], ZT_TILE);
        mma_sync(fc, fa, fb, fc);
        __syncwarp();
    }
    store_matrix_sync(sC[warp], fc, ZT_TILE, mem_row_major);
    __syncwarp();
    for (i = lane; i < ZT_TILE * ZT_TILE; i += ZT_WARP) {
        uint32_t rr = i / ZT_TILE, kk = i - rr * ZT_TILE;
        uint32_t r = r0 + rr, k = k0 + kk;
        if (r < prm.rows && k < prm.Kpad) out[(size_t)r * ld_out + k] = sC[warp][i];
    }
#else
    (void)hidden; (void)W; (void)out; (void)ld_out; (void)prm;
#endif
}

/* ------------------------------------------------------ head: SIMT path */
/* One warp per (row, k).  Used when the device has no tensor cores or when H
 * is not a multiple of 16.  Same accumulation order as the reference. */
__global__ void zt_k_head_simt(const uint8_t *hidden, const uint16_t *W, float *out,
                               uint32_t ld_out, zt_head_params prm)
{
    uint32_t warps_per_block = blockDim.x / ZT_WARP;
    uint32_t gw = blockIdx.x * warps_per_block + threadIdx.x / ZT_WARP;
    uint32_t lane = threadIdx.x % ZT_WARP;
    uint32_t r = gw / prm.K, k = gw - r * prm.K;
    size_t esz = (size_t)zt_dtype_size(prm.dtype);
    const uint8_t *row;
    const uint16_t *w;
    float acc = 0.0f;
    uint32_t h;

    if (r >= prm.rows) return;
    row = hidden + (size_t)r * prm.ld * esz;
    w   = W + (size_t)k * prm.H;
    for (h = lane; h < prm.H; h += ZT_WARP) acc += zt_load_f(row, prm.dtype, h) * zt_f16_to_f32_(w[h]);
    acc = zt_warp_sum(acc);
    if (lane == 0) out[(size_t)r * ld_out + k] = acc;
}

/* ----------------------------------------------------------- decide */
/* One warp per row.  Lane L handles choices L, L+32, ... of that row's field.
 * Tie-break is the lowest choice index, computed without atomics or sorting:
 *   m1  = max over lanes of the lane's best probability
 *   win = min over lanes of (best_idx where best == m1), i.e. lowest index
 *   p2  = max over lanes of (lane's second if it owns the winner, else best) */
__global__ void zt_k_decide(const float *x, const zt_field_dev *fields,
                            const uint32_t *choice_col_start, const float *choice_value,
                            const zt_knot *knots, zt_decision *dec, float *probs,
                            zt_decide_params prm)
{
    uint32_t warps_per_block = blockDim.x / ZT_WARP;
    uint32_t r    = blockIdx.x * warps_per_block + threadIdx.x / ZT_WARP;
    uint32_t lane = threadIdx.x % ZT_WARP;
    zt_field_dev f;
    const float *xs;
    uint32_t i, c;
    float m = -ZT_INF, z = 0.0f;
    float best = -1.0f, second = -1.0f, ent = 0.0f, ex = 0.0f, p0 = 0.0f;
    uint32_t best_i = 0u, win;
    float m1, p2;

    if (r >= prm.rows) return;
    f  = fields[r % prm.F];
    xs = x + (size_t)r * prm.ld + f.col_start;

    for (i = lane; i < f.n_cols; i += ZT_WARP) m = ZT_FMAXF(m, xs[i] * f.inv_temp);
    m = zt_warp_max(m);
    for (i = lane; i < f.n_cols; i += ZT_WARP) z += ZT_EXPF(xs[i] * f.inv_temp - m);
    z = zt_warp_sum(z);

    for (c = lane; c < f.n_choices; c += ZT_WARP) {
        uint32_t g  = f.choice_start + c;
        uint32_t lo = choice_col_start[g]     - f.col_start;
        uint32_t hi = choice_col_start[g + 1] - f.col_start;
        float part = 0.0f, pc;
        for (i = lo; i < hi; i++) part += ZT_EXPF(xs[i] * f.inv_temp - m);
        pc = part / z;
        if (pc > best) { second = best; best = pc; best_i = c; }
        else if (pc > second) second = pc;
        if (pc > 0.0f) ent -= pc * ZT_LOGF(pc);
        ex += pc * choice_value[g];
        if (c == 0u) p0 = pc;
        if (prm.has_probs) probs[(size_t)r * prm.c_max + c] = pc;
    }
    m1  = zt_warp_max(best);
    win = zt_warp_min_u((best == m1) ? best_i : 0xffffffffu);
    p2  = zt_warp_max((best_i == win && best == m1) ? second : best);
    ent = zt_warp_sum(ent);
    ex  = zt_warp_sum(ex);
    p0  = __shfl_sync(0xffffffffu, p0, 0);           /* lane 0 always owns choice 0 */

    if (lane == 0) zt_finalize(&dec[r], &f, knots, (int32_t)win, m1, p2, p0, ent, ex);
}

#endif /* ZT_KERNELS_CUH */
