/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-011
   File:              zt_metal.mm
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

   Clone-Gate: sha256:b7a32a2db36ca76bbde8fd4ee2d42f103be1a27fd9d67d359695883694366412

   See: LICENSE
   ======================================================================== */
#include "zt_kernels.metal.h"

extern "C" {
#include "zt_internal.h"
}

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <cstdlib>
#include <cstdio>
#include <cstring>

/* ------------------------------------------------------------------ params
 *
 * Kernel parameter blobs. These must match the struct definitions at the
 * top of zt_kernels.metal.h exactly, and they mirror the CUDA __global__
 * parameter lists in zt_kernels.cuh. Field order, field widths and total
 * size are all part of the ABI.
 */
typedef struct {
    uint32_t dtype;
    uint32_t ld;
    uint32_t rows;
    uint32_t H;
    uint32_t K;
    uint32_t Kpad;
} zt_head_params;

typedef struct {
    uint32_t dtype;
    uint32_t ld;
    uint32_t K;
    uint32_t rows;
} zt_gather_params;

typedef struct {
    uint32_t K;
    uint32_t ld;
    uint32_t F;
    uint32_t rows;
    uint32_t c_max;
    uint32_t has_probs;
} zt_decide_params;

/* ------------------------------------------------------------------ context */

typedef struct {
    MTL::Device *device;
    MTL::CommandQueue *queue;
    MTL::Library *library;
    MTL::ComputePipelineState *p_gather_head;
    MTL::ComputePipelineState *p_gather_logits;
    MTL::ComputePipelineState *p_head_simd;
    MTL::ComputePipelineState *p_decide;
    MTL::ComputePipelineState *p_probe;
    uint32_t sg_width;
    int has_simdgroup;
} metal_ctx;

typedef struct {
    metal_ctx *ctx;
    const zt_program *prog;
    uint32_t max_batch, max_rows;
    uint32_t H, Kpad;
    MTL::Buffer *b_fields;
    MTL::Buffer *b_col_token;
    MTL::Buffer *b_choice_start;
    MTL::Buffer *b_choice_value;
    MTL::Buffer *b_knots;
    MTL::Buffer *b_W;
    MTL::Buffer *b_x;
    MTL::Buffer *b_in;
    size_t in_cap;
    MTL::Buffer *b_dec;
    MTL::Buffer *b_probs;
    size_t probs_cap;
} metal_sess;

/* ------------------------------------------------------------------ helpers */

static zt_status mtl_fail(NS::Error *e, const char *what,
                          char *err, size_t errcap)
{
    if (e && e->localizedDescription())
        ZT_SETERR(err, errcap, "%s: %s", what,
                  e->localizedDescription()->utf8String());
    else
        ZT_SETERR(err, errcap, "%s: unknown Metal error", what);
    return ZT_ERR_BACKEND;
}

static MTL::ComputePipelineState *
mtl_pipeline(metal_ctx *c, const char *name, char *err, size_t errcap)
{
    NS::String *fn_name = NS::String::string(name, NS::UTF8StringEncoding);
    MTL::Function *fn = c->library->newFunction(fn_name);
    if (!fn) {
        ZT_SETERR(err, errcap, "Metal function not found: %s", name);
        return nullptr;
    }
    NS::Error *e = nullptr;
    MTL::ComputePipelineState *ps =
        c->device->newComputePipelineState(fn, &e);
    fn->release();
    if (!ps) { mtl_fail(e, name, err, errcap); return nullptr; }
    return ps;
}

/* ------------------------------------------------------------------ context */

static zt_status metal_create(void **out, char *err, size_t errcap)
{
    metal_ctx *c = (metal_ctx *)calloc(1, sizeof *c);
    if (!c) { ZT_SETERR(err, errcap, "out of memory"); return ZT_ERR_NO_MEMORY; }

    c->device = MTL::CreateSystemDefaultDevice();
    if (!c->device) { free(c); ZT_SETERR(err, errcap, "no Metal device found"); return ZT_ERR_UNAVAILABLE; }
    c->queue = c->device->newCommandQueue();
    if (!c->queue) {
        c->device->release(); free(c);
        ZT_SETERR(err, errcap, "cannot create Metal command queue");
        return ZT_ERR_UNAVAILABLE;
    }

    NS::String *src = NS::String::string(zt_kernels_msl_src, NS::UTF8StringEncoding);
    NS::Error *e = nullptr;
    c->library = c->device->newLibrary(src, nullptr, &e);
    if (!c->library) {
        c->queue->release(); c->device->release(); free(c);
        return mtl_fail(e, "newLibrary", err, errcap);
    }

    if (!(c->p_probe        = mtl_pipeline(c, "zt_k_probe_simd",        err, errcap)) ||
        !(c->p_gather_head  = mtl_pipeline(c, "zt_k_gather_head_f16",   err, errcap)) ||
        !(c->p_gather_logits= mtl_pipeline(c, "zt_k_gather_logits_f16", err, errcap)) ||
        !(c->p_head_simd    = mtl_pipeline(c, "zt_k_head_simd",         err, errcap)) ||
        !(c->p_decide       = mtl_pipeline(c, "zt_k_decide",            err, errcap))) {
        extern void metal_destroy(void *);
        metal_destroy(c);
        return ZT_ERR_BACKEND;
    }

    c->has_simdgroup = 0;
    {
        MTL::Buffer *buf = c->device->newBuffer(2 * sizeof(uint32_t),
                                                 MTL::ResourceStorageModeShared);
        if (buf) {
            MTL::CommandBuffer *cb = c->queue->commandBuffer();
            MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
            enc->setComputePipelineState(c->p_probe);
            enc->setBuffer(buf, 0, 0);
            enc->dispatchThreadgroups(MTL::Size::Make(1, 1, 1),
                                       MTL::Size::Make(32, 1, 1));
            enc->endEncoding();
            cb->commit();
            cb->waitUntilCompleted();
            uint32_t *r = (uint32_t *)buf->contents();
            uint32_t w = r[0], sum = r[1];
            c->sg_width = w;
            if (w > 0 && sum == w * (w - 1) / 2) c->has_simdgroup = 1;
            buf->release();
        }
    }

    *out = c;
    return ZT_OK;
}

static void metal_destroy(void *ctx)
{
    metal_ctx *c = (metal_ctx *)ctx;
    if (!c) return;
    if (c->p_decide)        c->p_decide->release();
    if (c->p_head_simd)     c->p_head_simd->release();
    if (c->p_gather_logits) c->p_gather_logits->release();
    if (c->p_gather_head)   c->p_gather_head->release();
    if (c->p_probe)         c->p_probe->release();
    if (c->library)         c->library->release();
    if (c->queue)           c->queue->release();
    if (c->device)          c->device->release();
    free(c);
}

/* ------------------------------------------------------------------ session */

static void metal_session_destroy(void *sess)
{
    metal_sess *s = (metal_sess *)sess;
    if (!s) return;
#define REL(b) do { if (s->b) { s->b->release(); s->b = nullptr; } } while (0)
    REL(b_fields); REL(b_col_token); REL(b_choice_start);
    REL(b_choice_value); REL(b_knots);
    REL(b_W); REL(b_x); REL(b_in); REL(b_dec); REL(b_probs);
#undef REL
    free(s);
}

static MTL::Buffer *mtl_private_init(metal_ctx *c, const void *src, size_t bytes)
{
    MTL::Buffer *shared = c->device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (!shared) return nullptr;
    memcpy(shared->contents(), src, bytes);
    MTL::Buffer *priv = c->device->newBuffer(bytes, MTL::ResourceStorageModePrivate);
    if (!priv) { shared->release(); return nullptr; }
    MTL::CommandBuffer *cb = c->queue->commandBuffer();
    MTL::BlitCommandEncoder *blit = cb->blitCommandEncoder();
    blit->copyFromBuffer(shared, 0, priv, 0, bytes);
    blit->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    shared->release();
    return priv;
}

static zt_status metal_session_create(void *ctx, const zt_program *p,
                                       uint32_t max_batch, void **out,
                                       char *err, size_t errcap)
{
    metal_ctx *c = (metal_ctx *)ctx;
    metal_sess *s = (metal_sess *)calloc(1, sizeof *s);
    if (!s) { ZT_SETERR(err, errcap, "out of memory"); return ZT_ERR_NO_MEMORY; }
    s->ctx = c; s->prog = p;
    s->max_batch = max_batch;
    s->max_rows = max_batch * p->n_fields;
    s->Kpad = zt_round_up(p->n_cols, ZT_TILE);
    uint32_t nk = p->n_knots ? p->n_knots : 1u;

#define ALLOC_TABLE(field, src, count, type) \
    do { \
        s->field = mtl_private_init(c, (src), (size_t)(count) * sizeof(type)); \
        if (!s->field) { \
            metal_session_destroy(s); \
            ZT_SETERR(err, errcap, "Metal table alloc failed: %s", #field); \
            return ZT_ERR_BACKEND; \
        } \
    } while (0)
    ALLOC_TABLE(b_fields,       p->fields,          p->n_fields,       zt_field_dev);
    ALLOC_TABLE(b_col_token,    p->col_token,        p->n_cols,         int32_t);
    ALLOC_TABLE(b_choice_start, p->choice_col_start, p->n_choices + 1,  uint32_t);
    ALLOC_TABLE(b_choice_value, p->choice_value,     p->n_choices,      float);
    ALLOC_TABLE(b_knots,        p->knots,            nk,                zt_knot);
#undef ALLOC_TABLE

    size_t x_bytes   = (size_t)s->max_rows * s->Kpad * sizeof(float);
    size_t dec_bytes = (size_t)s->max_rows * sizeof(zt_decision);
    s->b_x   = c->device->newBuffer(x_bytes,   MTL::ResourceStorageModePrivate);
    s->b_dec = c->device->newBuffer(dec_bytes,  MTL::ResourceStorageModeShared);
    if (!s->b_x || !s->b_dec) {
        metal_session_destroy(s);
        ZT_SETERR(err, errcap, "device workspace allocation failed");
        return ZT_ERR_BACKEND;
    }
    *out = s;
    return ZT_OK;
}

/* ------------------------------------------------------------------ head */

static zt_status metal_bind_head(void *sess, const zt_tensor *head,
                                  char *err, size_t errcap)
{
    metal_sess *s = (metal_sess *)sess;
    const zt_program *p = s->prog;
    uint32_t K = p->n_cols, H = head->cols;
    size_t bytes = (size_t)s->Kpad * H * sizeof(uint16_t);

    if (s->b_W) { s->b_W->release(); s->b_W = nullptr; }
    s->b_W = s->ctx->device->newBuffer(bytes, MTL::ResourceStorageModePrivate);
    if (!s->b_W) { ZT_SETERR(err, errcap, "W alloc failed"); return ZT_ERR_BACKEND; }

    size_t src_bytes = (size_t)head->rows * head->ld * sizeof(uint16_t);
    MTL::Buffer *b_src = s->ctx->device->newBuffer(src_bytes,
                                                    MTL::ResourceStorageModeShared);
    if (!b_src) { ZT_SETERR(err, errcap, "head staging alloc failed"); return ZT_ERR_BACKEND; }

    if (head->mem == ZT_MEM_HOST) {
        uint16_t *tmp = (uint16_t *)malloc(bytes);
        if (!tmp) { b_src->release(); ZT_SETERR(err, errcap, "out of memory"); return ZT_ERR_NO_MEMORY; }
        zt_host_gather_head_f16(head, p->col_token, K, s->Kpad, H, tmp);
        memcpy(b_src->contents(), tmp, bytes);
        free(tmp);
    } else {
        memcpy(b_src->contents(), (const uint8_t *)head->data + head->offset, src_bytes);
    }

    zt_head_params hp = {0};
    hp.dtype = head->dtype; hp.ld = head->ld; hp.rows = head->rows;
    hp.H = H; hp.K = K; hp.Kpad = s->Kpad;

    MTL::CommandBuffer *cb = s->ctx->queue->commandBuffer();
    MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(s->ctx->p_gather_head);
    enc->setBuffer(b_src,          0, 0);
    enc->setBuffer(s->b_col_token, 0, 1);
    enc->setBuffer(s->b_W,         0, 2);
    enc->setBytes(&hp, sizeof hp, 30);
    uint32_t n = s->Kpad * H, blocks = (n + 255u) / 256u;
    enc->dispatchThreadgroups(MTL::Size::Make(blocks, 1, 1), MTL::Size::Make(256, 1, 1));
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    b_src->release();
    s->H = H;
    return ZT_OK;
}

/* ------------------------------------------------------------------ run */

static zt_status metal_stage_input(metal_sess *s, const zt_tensor *in,
                                    uint32_t rows, MTL::Buffer **out_buf,
                                    char *err, size_t errcap)
{
    size_t esz = zt_dtype_size(in->dtype);
    size_t bytes = (size_t)rows * in->ld * esz;
    if (bytes > s->in_cap) {
        if (s->b_in) { s->b_in->release(); s->b_in = nullptr; }
        s->b_in = s->ctx->device->newBuffer(bytes, MTL::ResourceStorageModeShared);
        if (!s->b_in) { ZT_SETERR(err, errcap, "input staging alloc failed"); return ZT_ERR_BACKEND; }
        s->in_cap = bytes;
    }
    memcpy(s->b_in->contents(), (const uint8_t *)in->data + in->offset, bytes);
    *out_buf = s->b_in;
    return ZT_OK;
}

static zt_status metal_run(void *sess, zt_input_kind kind,
                           const zt_tensor *in, uint32_t batch,
                           zt_result *out, char *err, size_t errcap)
{
    metal_sess *s = (metal_sess *)sess;
    const zt_program *p = s->prog;
    uint32_t rows = batch * p->n_fields;
    MTL::Buffer *b_in = nullptr;
    zt_status st = metal_stage_input(s, in, rows, &b_in, err, errcap);
    if (st != ZT_OK) return st;

    MTL::CommandBuffer *cb = s->ctx->queue->commandBuffer();

    /* Stage 1: gather sub-logits */
    {
        MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
        if (kind == ZT_INPUT_LOGITS) {
            zt_gather_params gp = {0};
            gp.dtype = in->dtype; gp.ld = in->ld; gp.K = p->n_cols; gp.rows = rows;
            enc->setComputePipelineState(s->ctx->p_gather_logits);
            enc->setBuffer(b_in,          0, 0);
            enc->setBuffer(s->b_col_token, 0, 1);
            enc->setBuffer(s->b_x,         0, 2);
            enc->setBytes(&gp, sizeof gp, 30);
            uint32_t n = rows * s->Kpad, blocks = (n + 255u) / 256u;
            enc->dispatchThreadgroups(MTL::Size::Make(blocks, 1, 1), MTL::Size::Make(256, 1, 1));
        } else {
            if (!s->b_W) { enc->endEncoding(); ZT_SETERR(err, errcap, "no lm_head bound"); return ZT_ERR_NOT_BOUND; }
            zt_head_params hp = {0};
            hp.dtype = in->dtype; hp.ld = in->ld; hp.rows = rows;
            hp.H = s->H; hp.K = p->n_cols; hp.Kpad = s->Kpad;
            enc->setComputePipelineState(s->ctx->p_head_simd);
            enc->setBuffer(b_in,   0, 0);
            enc->setBuffer(s->b_W, 0, 1);
            enc->setBuffer(s->b_x, 0, 2);
            enc->setBytes(&hp, sizeof hp, 30);
            uint32_t total = rows * p->n_cols;
            uint32_t per_tg = 256 / s->ctx->sg_width;
            uint32_t tgs = (total + per_tg - 1) / per_tg;
            enc->dispatchThreadgroups(MTL::Size::Make(tgs, 1, 1), MTL::Size::Make(256, 1, 1));
        }
        enc->endEncoding();
    }

    /* Stage 2: decide */
    if (out->probs) {
        size_t need = (size_t)rows * p->c_max * sizeof(float);
        if (need > s->probs_cap) {
            if (s->b_probs) { s->b_probs->release(); s->b_probs = nullptr; }
            s->b_probs = s->ctx->device->newBuffer(need, MTL::ResourceStorageModeShared);
            if (!s->b_probs) { ZT_SETERR(err, errcap, "probs alloc failed"); return ZT_ERR_BACKEND; }
            s->probs_cap = need;
        }
    }
    {
        MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
        zt_decide_params dp = {0};
        dp.K = p->n_cols; dp.ld = s->Kpad; dp.F = p->n_fields;
        dp.rows = rows; dp.c_max = p->c_max; dp.has_probs = out->probs ? 1u : 0u;
        enc->setComputePipelineState(s->ctx->p_decide);
        enc->setBuffer(s->b_x,           0, 0);
        enc->setBuffer(s->b_fields,       0, 1);
        enc->setBuffer(s->b_choice_start, 0, 2);
        enc->setBuffer(s->b_choice_value, 0, 3);
        enc->setBuffer(s->b_knots,        0, 4);
        enc->setBuffer(s->b_dec,          0, 5);
        enc->setBuffer(s->b_probs ? s->b_probs : s->b_dec, 0, 6);
        enc->setBytes(&dp, sizeof dp, 30);
        uint32_t blocks = (rows + 255u) / 256u;
        enc->dispatchThreadgroups(MTL::Size::Make(blocks, 1, 1), MTL::Size::Make(256, 1, 1));
        enc->endEncoding();
    }

    cb->commit();
    cb->waitUntilCompleted();

    memcpy(out->dec, s->b_dec->contents(), (size_t)rows * sizeof(zt_decision));
    if (out->probs)
        memcpy(out->probs, s->b_probs->contents(), (size_t)rows * p->c_max * sizeof(float));

    return ZT_OK;
}

extern "C" const zt_backend_ops zt_metal_ops = {
    "metal", ZT_BACKEND_METAL,
    metal_create, metal_destroy,
    metal_session_create, metal_session_destroy,
    metal_bind_head, metal_run
};
