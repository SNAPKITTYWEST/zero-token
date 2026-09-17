/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-014
   File:              zt_kernels.metal
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

   Clone-Gate: sha256:d8b58eef4859c058b8812610f0836a7aad6b742397382eed96b060a02e9a41f3

   See: LICENSE
   ======================================================================== */
#include "zt_core.h"

using namespace metal;

#define ZT_SIMD    32u
#define ZT_HEAD_SG 4u      /* simdgroups per threadgroup (128 threads) */
#define ZT_MTILE   8u      /* simdgroup_matrix edge */

/* --------------------------------------------------------------- gather */

kernel void zt_k_gather_logits(device const uint8_t   *logits    [[buffer(0)]],
                               device const int32_t   *col_token [[buffer(1)]],
                               device float           *out       [[buffer(2)]],
                               constant uint32_t      &ld_out    [[buffer(3)]],
                               constant zt_gather_params &prm    [[buffer(4)]],
                               uint gid [[thread_position_in_grid]])
{
    uint32_t n = prm.rows * prm.K;
    if (gid >= n) return;
    uint32_t r = gid / prm.K, k = gid - r * prm.K;
    size_t esz = (size_t)zt_dtype_size(prm.dtype);
    device const uint8_t *row = logits + (size_t)r * prm.ld * esz;
    out[(size_t)r * ld_out + k] = zt_load_f(row, prm.dtype, (uint32_t)col_token[k]);
}

kernel void zt_k_gather_head(device const uint8_t  *head      [[buffer(0)]],
                             device const int32_t  *col_token [[buffer(1)]],
                             device uint16_t       *out       [[buffer(2)]],
                             constant uint32_t     &Kpad      [[buffer(3)]],
                             constant zt_head_params &prm     [[buffer(4)]],
                             uint gid [[thread_position_in_grid]])
{
    uint32_t n = Kpad * prm.H;
    if (gid >= n) return;
    uint32_t k = gid / prm.H, h = gid - k * prm.H;
    if (k >= prm.K) { out[gid] = 0u; return; }
    size_t esz = (size_t)zt_dtype_size(prm.dtype);
    device const uint8_t *row = head + (size_t)col_token[k] * prm.ld * esz;
    out[gid] = as_type<uint16_t>((half)zt_load_f(row, prm.dtype, h));
}

/* ------------------------------------------------- head: simdgroup matrix */
/* out[rows][Kpad] = hidden[rows][H] * W[Kpad][H]^T.
 * grid: x = Kpad/8 tiles split over ZT_HEAD_SG simdgroups, y = rows/8 tiles.
 * Each simdgroup owns one 8x8 output tile and stages its own A/B tiles, so
 * only simdgroup_barrier is needed and out-of-range groups may return. */
kernel void zt_k_head_simd(device const uint8_t    *hidden [[buffer(0)]],
                           device const uint16_t   *W      [[buffer(1)]],
                           device float            *out    [[buffer(2)]],
                           constant uint32_t       &ld_out [[buffer(3)]],
                           constant zt_head_params &prm    [[buffer(4)]],
                           uint3 tgid  [[threadgroup_position_in_grid]],
                           uint  lane  [[thread_index_in_simdgroup]],
                           uint  sg    [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sA[ZT_HEAD_SG][ZT_MTILE * ZT_MTILE];
    threadgroup float sB[ZT_HEAD_SG][ZT_MTILE * ZT_MTILE];
    threadgroup float sC[ZT_HEAD_SG][ZT_MTILE * ZT_MTILE];

    uint32_t k0  = (tgid.x * ZT_HEAD_SG + sg) * ZT_MTILE;
    uint32_t r0  = tgid.y * ZT_MTILE;
    size_t   esz = (size_t)zt_dtype_size(prm.dtype);

    if (k0 >= prm.Kpad || r0 >= prm.rows) return;

    simdgroup_float8x8 fa, fb;
    simdgroup_float8x8 fc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint32_t h0 = 0; h0 < prm.H; h0 += ZT_MTILE) {
        for (uint32_t i = lane; i < ZT_MTILE * ZT_MTILE; i += ZT_SIMD) {
            uint32_t rr = i / ZT_MTILE, hh = i - rr * ZT_MTILE;
            uint32_t r = r0 + rr, h = h0 + hh;
            float a = 0.0f, b = 0.0f;
            if (r < prm.rows && h < prm.H) {
                device const uint8_t *row = hidden + (size_t)r * prm.ld * esz;
                a = zt_load_f(row, prm.dtype, h);
            }
            if (h < prm.H) b = (float)as_type<half>(W[(size_t)(k0 + rr) * prm.H + h]);
            sA[sg][rr * ZT_MTILE + hh] = a;
            sB[sg][rr * ZT_MTILE + hh] = b;   /* row-major W[k][h]; loaded transposed below */
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(fa, &sA[sg][0], ZT_MTILE);
        simdgroup_load(fb, &sB[sg][0], ZT_MTILE, ulong2(0, 0), true);   /* transpose */
        simdgroup_multiply_accumulate(fc, fa, fb, fc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(fc, &sC[sg][0], ZT_MTILE);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint32_t i = lane; i < ZT_MTILE * ZT_MTILE; i += ZT_SIMD) {
        uint32_t rr = i / ZT_MTILE, kk = i - rr * ZT_MTILE;
        uint32_t r = r0 + rr, k = k0 + kk;
        if (r < prm.rows && k < prm.Kpad) out[(size_t)r * ld_out + k] = sC[sg][i];
    }
}

/* ------------------------------------------------------ head: SIMT path */
kernel void zt_k_head_simt(device const uint8_t    *hidden [[buffer(0)]],
                           device const uint16_t   *W      [[buffer(1)]],
                           device float            *out    [[buffer(2)]],
                           constant uint32_t       &ld_out [[buffer(3)]],
                           constant zt_head_params &prm    [[buffer(4)]],
                           uint gid  [[threadgroup_position_in_grid]],
                           uint tid  [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg   [[simdgroup_index_in_threadgroup]])
{
    uint32_t gw = gid * ZT_HEAD_SG + sg;
    uint32_t r = gw / prm.K, k = gw - r * prm.K;
    (void)tid;
    if (r >= prm.rows) return;
    size_t esz = (size_t)zt_dtype_size(prm.dtype);
    device const uint8_t  *row = hidden + (size_t)r * prm.ld * esz;
    device const uint16_t *w   = W + (size_t)k * prm.H;
    float acc = 0.0f;
    for (uint32_t h = lane; h < prm.H; h += ZT_SIMD)
        acc += zt_load_f(row, prm.dtype, h) * (float)as_type<half>(w[h]);
    acc = simd_sum(acc);
    if (lane == 0) out[(size_t)r * ld_out + k] = acc;
}

/* ----------------------------------------------------------- decide */
kernel void zt_k_decide(device const float        *x                [[buffer(0)]],
                        device const zt_field_dev *fields           [[buffer(1)]],
                        device const uint32_t     *choice_col_start [[buffer(2)]],
                        device const float        *choice_value     [[buffer(3)]],
                        device const zt_knot      *knots            [[buffer(4)]],
                        device zt_decision        *dec              [[buffer(5)]],
                        device float              *probs            [[buffer(6)]],
                        constant zt_decide_params &prm              [[buffer(7)]],
                        uint gid  [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg   [[simdgroup_index_in_threadgroup]])
{
    uint32_t r = gid * ZT_HEAD_SG + sg;
    if (r >= prm.rows) return;

    zt_field_dev f = fields[r % prm.F];
    device const float *xs = x + (size_t)r * prm.ld + f.col_start;
    float m = -ZT_INF, z = 0.0f;
    float best = -1.0f, second = -1.0f, ent = 0.0f, ex = 0.0f, p0 = 0.0f;
    uint32_t best_i = 0u;

    for (uint32_t i = lane; i < f.n_cols; i += ZT_SIMD) m = ZT_FMAXF(m, xs[i] * f.inv_temp);
    m = simd_max(m);
    for (uint32_t i = lane; i < f.n_cols; i += ZT_SIMD) z += ZT_EXPF(xs[i] * f.inv_temp - m);
    z = simd_sum(z);

    for (uint32_t c = lane; c < f.n_choices; c += ZT_SIMD) {
        uint32_t g  = f.choice_start + c;
        uint32_t lo = choice_col_start[g]     - f.col_start;
        uint32_t hi = choice_col_start[g + 1] - f.col_start;
        float part = 0.0f;
        for (uint32_t i = lo; i < hi; i++) part += ZT_EXPF(xs[i] * f.inv_temp - m);
        float pc = part / z;
        if (pc > best) { second = best; best = pc; best_i = c; }
        else if (pc > second) second = pc;
        if (pc > 0.0f) ent -= pc * ZT_LOGF(pc);
        ex += pc * choice_value[g];
        if (c == 0u) p0 = pc;
        if (prm.has_probs != 0u) probs[(size_t)r * prm.c_max + c] = pc;
    }
    float    m1  = simd_max(best);
    uint32_t win = simd_min((best == m1) ? best_i : 0xffffffffu);
    float    p2  = simd_max((best_i == win && best == m1) ? second : best);
    ent = simd_sum(ent);
    ex  = simd_sum(ex);
    p0  = simd_broadcast(p0, 0);

    if (lane == 0) {
        zt_decision d;
        zt_finalize(&d, &f, knots, (int32_t)win, m1, p2, p0, ent, ex);
        dec[r] = d;
    }
}
