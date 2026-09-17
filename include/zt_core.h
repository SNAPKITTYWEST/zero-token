/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-003
   File:              zt_core.h
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

   Clone-Gate: sha256:2d4b1c34849ac76d3e8c31aa2c60de98d56cbd8e5417e6ad652410525b9cf4cb

   See: LICENSE

   -------------------------------------------------------------------------
   zt_core.h — the shared decision core.

   Included, unchanged, by:
     src/zt_cpu.c              (C99 reference backend; host tables)
     kernels/zt_kernels.cuh    (CUDA device code)
     metal/zt_kernels.metal    (Metal Shading Language)

   Contents: device-visible POD tables whose layout is identical on all three
   compilers (fixed-width scalars only, no pointers, no enums), dtype loaders,
   the calibration map, and the decision finalizer.  Everything here is pure
   scalar math with no allocation and no target-specific intrinsics.  The
   per-target parallel structure (warp/simdgroup reductions, tensor-core
   tiles) lives in the kernel files, which call into this header.
   ======================================================================== */
#ifndef ZT_CORE_H
#define ZT_CORE_H

#include "zt_device.h"

/* Field kinds */
#define ZT_KIND_BOOL   0u
#define ZT_KIND_CHOICE 1u
#define ZT_KIND_SCORE  2u

/* Element types */
#define ZT_DTYPE_F32   0u
#define ZT_DTYPE_F16   1u
#define ZT_DTYPE_BF16  2u

/* Sentinel choice values */
#define ZT_ABSTAIN (-1)   /* withheld: confidence or margin below the field's floor */
#define ZT_SKIPPED (-2)   /* not evaluated: the field's dependency guard failed     */

/* Decision flags */
#define ZT_FLAG_LOW_CONF   1u
#define ZT_FLAG_LOW_MARGIN 2u

/* One compiled field.  48 bytes, 4-byte aligned on every target. */
typedef struct {
    uint32_t kind;          /* ZT_KIND_*                                          */
    uint32_t col_start;     /* first column of this field in the sub-logit matrix */
    uint32_t n_cols;        /* alias tokens summed over all choices               */
    uint32_t n_choices;
    uint32_t choice_start;  /* first entry of this field in per-choice tables     */
    uint32_t calib_off;     /* offset into the knot table                         */
    uint32_t calib_n;       /* 0 = identity calibration                           */
    uint32_t reserved;
    float    inv_temp;      /* 1 / temperature                                    */
    float    threshold;     /* BOOL: choose true iff P(true) >= threshold         */
    float    min_conf;      /* abstain iff calibrated confidence < min_conf       */
    float    min_margin;    /* abstain iff log P(top) - log P(second) < min_margin */
} zt_field_dev;

/* Piecewise-linear calibration knot (x = raw probability, y = calibrated). */
typedef struct { float x; float y; } zt_knot;

/* One decision.  32 bytes.  Identical on host and device. */
typedef struct {
    int32_t  choice;        /* index into the field's choices, ZT_ABSTAIN or ZT_SKIPPED */
    uint32_t flags;         /* ZT_FLAG_*                                                 */
    float    confidence;    /* calibrated probability of the selected choice             */
    float    p_top;         /* raw probability of the selected choice                    */
    float    p_second;      /* raw probability of the runner-up                          */
    float    margin;        /* log p_top - log p_second (nats)                           */
    float    score;         /* SCORE: expectation of choice values; BOOL: P(true); else 0 */
    float    entropy;       /* entropy of the choice distribution (nats)                 */
} zt_decision;

/* Launch parameter blocks (Metal constant buffers; also used by the host bridge). */
typedef struct { uint32_t dtype; uint32_t ld; uint32_t K; uint32_t rows; } zt_gather_params;
typedef struct { uint32_t dtype; uint32_t ld; uint32_t rows; uint32_t H; uint32_t K; uint32_t Kpad; } zt_head_params;
typedef struct { uint32_t K; uint32_t ld; uint32_t F; uint32_t rows; uint32_t c_max; uint32_t has_probs; } zt_decide_params;

ZT_FN uint32_t zt_dtype_size(uint32_t dtype) { return (dtype == ZT_DTYPE_F32) ? 4u : 2u; }

/* Read element i of a row stored as f32 / f16 / bf16 and return it as float. */
ZT_FN float zt_load_f(ZT_DEV const uint8_t *base, uint32_t dtype, uint32_t i)
{
    if (dtype == ZT_DTYPE_F32) return ((ZT_DEV const float *)base)[i];
    if (dtype == ZT_DTYPE_F16) return ZT_F16_TO_F32(((ZT_DEV const uint16_t *)base)[i]);
    return ZT_BITS_TO_F32(((uint32_t)((ZT_DEV const uint16_t *)base)[i]) << 16);
}

/* Monotone piecewise-linear map fitted offline (see zt_calib_fit_bins).  n == 0 is identity. */
ZT_FN float zt_calib_apply(ZT_DEV const zt_knot *k, uint32_t n, float p)
{
    uint32_t i;
    float x0, x1, t;
    if (n == 0u) return p;
    if (p <= k[0].x) return k[0].y;
    if (p >= k[n - 1u].x) return k[n - 1u].y;
    i = 1u;
    while (i < n - 1u && p > k[i].x) i++;
    x0 = k[i - 1u].x; x1 = k[i].x;
    t = (x1 > x0) ? (p - x0) / (x1 - x0) : 0.0f;
    return k[i - 1u].y + t * (k[i].y - k[i - 1u].y);
}

/* Turn the reduced statistics of one row into a decision.
 *   top        argmax choice (lowest index on ties)
 *   p_top      its probability;  p_second: runner-up probability
 *   p_true     probability of choice 0 (used by BOOL fields)
 *   entropy    entropy of the choice distribution
 *   expect     sum_c p_c * value_c (used by SCORE fields)
 * Called by exactly one thread per row on every target. */
ZT_FN void zt_finalize(ZT_THREAD zt_decision *d, ZT_THREAD const zt_field_dev *f,
                       ZT_DEV const zt_knot *knots,
                       int32_t top, float p_top, float p_second, float p_true,
                       float entropy, float expect)
{
    int32_t  choice;
    float    p_sel, p_alt, conf, margin;
    uint32_t flags = 0u;

    if (f->kind == ZT_KIND_BOOL) {
        choice = (p_true >= f->threshold) ? 0 : 1;
        p_sel  = (choice == 0) ? p_true : 1.0f - p_true;
        p_alt  = 1.0f - p_sel;
    } else {
        choice = top;
        p_sel  = p_top;
        p_alt  = p_second;
    }
    conf   = zt_calib_apply(knots + f->calib_off, f->calib_n, p_sel);
    margin = ZT_LOGF(ZT_FMAXF(p_sel, 1e-30f)) - ZT_LOGF(ZT_FMAXF(p_alt, 1e-30f));
    if (conf   < f->min_conf)   flags |= ZT_FLAG_LOW_CONF;
    if (margin < f->min_margin) flags |= ZT_FLAG_LOW_MARGIN;

    d->choice     = (flags != 0u) ? ZT_ABSTAIN : choice;
    d->flags      = flags;
    d->confidence = conf;
    d->p_top      = p_sel;
    d->p_second   = p_alt;
    d->margin     = margin;
    d->score      = (f->kind == ZT_KIND_BOOL) ? p_true : (f->kind == ZT_KIND_SCORE ? expect : 0.0f);
    d->entropy    = entropy;
}

#endif /* ZT_CORE_H */
