/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-015
   File:              zt_kernels.metal.h
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

   Clone-Gate: sha256:cfe4f1c9782ff4d74b1d11ac989fc5f1e278c3d0019d6e0ac47a631604e22093

   See: LICENSE
   ======================================================================== */
#ifndef ZT_KERNELS_METAL_H
#define ZT_KERNELS_METAL_H

static const char *zt_kernels_msl_src = R"MSL(
#include <metal_stdlib>
using namespace metal;

/* ---- parameter structs (must match host mirror in zt_metal.mm) ---- */
struct zt_head_params   { uint dtype; uint ld; uint rows; uint H; uint K; uint Kpad; };
struct zt_gather_params { uint dtype; uint ld; uint K; uint rows; };
struct zt_decide_params { uint K; uint ld; uint F; uint rows; uint c_max; uint has_probs; };

/* ---- program-side structs (must match zt_internal.h) ---- */
struct zt_field_dev { uint kind; uint c_start; uint c_count; uint k_start; uint k_count; float lo; float hi; };
struct zt_knot      { float x; float y; };
struct zt_decision  { uint class_index; float confidence; uint bitmask; uchar valid; uchar _pad[3]; };

/* ---- SIMD-group probe ----
 * Writes [width, sum(lane_ids)]. The host verifies the triangular
 * identity: only satisfied if simd_sum reduced correctly. */
kernel void zt_k_probe_simd(
    device uint* out [[buffer(0)]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sg_size [[threads_per_simdgroup]])
{
    float total = simd_sum(float(lane));
    if (lane == 0) { out[0] = sg_size; out[1] = uint(total); }
}

/* ---- gather head rows ----
 * dst[k][h] = src[tok[k]][h] for k < K, else 0.
 * One thread per output element. */
kernel void zt_k_gather_head_f16(
    device const half* src [[buffer(0)]],
    device const int*  tok [[buffer(1)]],
    device half*       dst [[buffer(2)]],
    constant zt_head_params& p [[buffer(30)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.Kpad * p.H) return;
    uint k = gid / p.H, h = gid % p.H;
    half v = half(0);
    if (k < p.K) { int row = tok[k]; if (row >= 0 && row < (int)p.rows) v = src[(uint)row * p.ld + h]; }
    dst[gid] = v;
}

/* ---- gather logits into sub-logit workspace (ZT_INPUT_LOGITS path) ---- */
kernel void zt_k_gather_logits_f16(
    device const half* src [[buffer(0)]],
    device const int*  tok [[buffer(1)]],
    device float*      x   [[buffer(2)]],
    constant zt_gather_params& p [[buffer(30)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.rows * p.Kpad) return;
    uint row = gid / p.Kpad, k = gid % p.Kpad;
    float v = 0.0f;
    if (k < p.K) { int col = tok[k]; v = float(src[row * p.ld + (uint)col]); }
    x[gid] = v;
}

/* ---- head: x = input @ W^T ----
 * One SIMD-group per (row, col) output. Reads threads_per_simdgroup
 * so it is correct on any Apple GPU — never hard-codes 32. */
kernel void zt_k_head_simd(
    device const half* in [[buffer(0)]],
    device const half* W  [[buffer(1)]],
    device float*      x  [[buffer(2)]],
    constant zt_head_params& p [[buffer(30)]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint sg_size [[threads_per_simdgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint gsg = tg_id * (tg_size / sg_size) + sg_id;
    if (gsg >= p.rows * p.K) return;
    uint row = gsg / p.K, col = gsg % p.K;
    float partial = 0.0f;
    for (uint h = lane; h < p.H; h += sg_size)
        partial += float(in[row * p.ld + h]) * float(W[col * p.H + h]);
    float s = simd_sum(partial);
    if (lane == 0) x[row * p.Kpad + col] = s;
}

/* ---- decide: one thread per row ---- */
kernel void zt_k_decide(
    device const float*        x            [[buffer(0)]],
    device const zt_field_dev* fields       [[buffer(1)]],
    device const uint*         choice_start [[buffer(2)]],
    device const float*        choice_value [[buffer(3)]],
    device const zt_knot*      knots        [[buffer(4)]],
    device zt_decision*        dec          [[buffer(5)]],
    device float*              probs        [[buffer(6)]],
    constant zt_decide_params& p            [[buffer(30)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.rows) return;
    uint f = gid % p.F;
    zt_field_dev fd = fields[f];
    device const float* row = x + gid * p.ld;

    if (fd.kind == 0) { // BOOL
        dec[gid].class_index = (row[0] >= row[1]) ? 0u : 1u;
        dec[gid].confidence = 1.0f; dec[gid].bitmask = 0; dec[gid].valid = 1;
    } else if (fd.kind == 1) { // CHOICE
        uint best = 0; float bv = row[0];
        for (uint i = 1; i < fd.c_count; ++i) if (row[i] > bv) { bv = row[i]; best = i; }
        float sum = 0.0f;
        for (uint i = 0; i < fd.c_count; ++i) sum += exp(row[i] - bv);
        dec[gid].class_index = best; dec[gid].confidence = 1.0f / sum;
        dec[gid].bitmask = 0; dec[gid].valid = 1;
    } else if (fd.kind == 2) { // SCORE
        dec[gid].class_index = 0;
        dec[gid].confidence = 1.0f / (1.0f + exp(-row[0]));
        dec[gid].bitmask = 0; dec[gid].valid = 1;
    } else { // MULTI
        uint mask = 0;
        for (uint i = 0; i < fd.c_count; ++i) {
            if (1.0f / (1.0f + exp(-row[i])) >= 0.5f) mask |= (1u << i);
        }
        dec[gid].class_index = 0; dec[gid].confidence = 0.0f;
        dec[gid].bitmask = mask; dec[gid].valid = 1;
    }

    if (p.has_probs) {
        device float* pr = probs + gid * p.c_max;
        for (uint i = 0; i < p.c_max; ++i) pr[i] = 0.0f;
        if (fd.kind == 0) {
            pr[0] = dec[gid].confidence; pr[1] = 1.0f - dec[gid].confidence;
        } else if (fd.kind == 1) {
            float bv = row[0];
            for (uint i = 1; i < fd.c_count; ++i) if (row[i] > bv) bv = row[i];
            float sum = 0.0f;
            for (uint i = 0; i < fd.c_count; ++i) sum += exp(row[i] - bv);
            for (uint i = 0; i < fd.c_count; ++i) pr[i] = exp(row[i] - bv) / sum;
        } else if (fd.kind == 2) { pr[0] = dec[gid].confidence; }
    }
}
)MSL";

#endif /* ZT_KERNELS_METAL_H */
