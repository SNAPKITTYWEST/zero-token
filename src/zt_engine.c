/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-005
   File:              zt_engine.c
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

   Clone-Gate: sha256:c66c40ee7311861b5eab55f211b625fa3bf1bd9c6c929a8bf00bba6d8b561a6d

   See: LICENSE
   ======================================================================== */
#include "zt_internal.h"

struct zt_engine {
    const zt_backend_ops *ops;
    void                 *ctx;
};

struct zt_session {
    zt_engine        *eng;
    const zt_program *prog;
    void             *bsess;
    uint32_t          max_batch;
    uint32_t          head_H;     /* 0 = no head bound */
    char              err[256];
};

const char *zt_backend_name(zt_backend b)
{
    switch (b) {
        case ZT_BACKEND_CPU:   return "cpu";
        case ZT_BACKEND_CUDA:  return "cuda";
        case ZT_BACKEND_METAL: return "metal";
        case ZT_BACKEND_AUTO:  return "auto";
        default:               return "unknown";
    }
}

const char *zt_status_str(zt_status s)
{
    switch (s) {
        case ZT_OK:                   return "ok";
        case ZT_ERR_INVALID_ARG:      return "invalid argument";
        case ZT_ERR_NO_MEMORY:        return "out of memory";
        case ZT_ERR_BAD_SCHEMA:       return "invalid schema";
        case ZT_ERR_AMBIGUOUS:        return "ambiguous token assignment";
        case ZT_ERR_SHAPE:            return "tensor shape mismatch";
        case ZT_ERR_BACKEND:          return "backend failure";
        case ZT_ERR_UNAVAILABLE:      return "backend unavailable";
        case ZT_ERR_BUFFER_TOO_SMALL: return "buffer too small";
        case ZT_ERR_NOT_BOUND:        return "no lm_head bound";
        default:                      return "unknown status";
    }
}

const char *zt_version(void) { return "0.1.0"; }

static const zt_backend_ops *ops_for(zt_backend b)
{
    switch (b) {
        case ZT_BACKEND_CPU:   return &zt_cpu_ops;
#if defined(ZT_HAVE_CUDA)
        case ZT_BACKEND_CUDA:  return &zt_cuda_ops;
#endif
#if defined(ZT_HAVE_METAL)
        case ZT_BACKEND_METAL: return &zt_metal_ops;
#endif
        default: return NULL;
    }
}

int zt_backend_available(zt_backend b)
{
    const zt_backend_ops *o;
    void *ctx = NULL;
    char err[64];
    if (b == ZT_BACKEND_AUTO) return 1;
    o = ops_for(b);
    if (!o) return 0;
    if (o->create(&ctx, err, sizeof err) != ZT_OK) return 0;
    o->destroy(ctx);
    return 1;
}

zt_engine *zt_engine_create(zt_backend backend, char *err, size_t errcap)
{
    static const zt_backend order[3] = { ZT_BACKEND_CUDA, ZT_BACKEND_METAL, ZT_BACKEND_CPU };
    const zt_backend_ops *o = NULL;
    zt_engine *e;
    void *ctx = NULL;
    char local[256];
    uint32_t i;

    local[0] = '\0';
    if (backend == ZT_BACKEND_AUTO) {
        for (i = 0; i < 3; i++) {
            o = ops_for(order[i]);
            if (o && o->create(&ctx, local, sizeof local) == ZT_OK) break;
            o = NULL;
        }
        if (!o) { ZT_SETERR(err, errcap, "no backend available"); return NULL; }
    } else {
        o = ops_for(backend);
        if (!o) {
            ZT_SETERR(err, errcap, "backend '%s' is not compiled into this build", zt_backend_name(backend));
            return NULL;
        }
        if (o->create(&ctx, local, sizeof local) != ZT_OK) {
            ZT_SETERR(err, errcap, "backend '%s' unavailable: %s", o->name, local[0] ? local : "no device");
            return NULL;
        }
    }
    e = (zt_engine *)calloc(1, sizeof *e);
    if (!e) { o->destroy(ctx); ZT_SETERR(err, errcap, "out of memory"); return NULL; }
    e->ops = o;
    e->ctx = ctx;
    return e;
}

void zt_engine_destroy(zt_engine *e)
{
    if (!e) return;
    e->ops->destroy(e->ctx);
    free(e);
}

zt_backend zt_engine_backend(const zt_engine *e) { return e ? e->ops->kind : ZT_BACKEND_AUTO; }

zt_session *zt_session_create(zt_engine *e, const zt_program *p, uint32_t max_batch,
                              char *err, size_t errcap)
{
    zt_session *s;
    void *bs = NULL;
    char local[256];

    if (!e || !p || max_batch == 0) { ZT_SETERR(err, errcap, "invalid argument"); return NULL; }
    local[0] = '\0';
    if (e->ops->session_create(e->ctx, p, max_batch, &bs, local, sizeof local) != ZT_OK) {
        ZT_SETERR(err, errcap, "%s: %s", e->ops->name, local[0] ? local : "session creation failed");
        return NULL;
    }
    s = (zt_session *)calloc(1, sizeof *s);
    if (!s) { e->ops->session_destroy(bs); ZT_SETERR(err, errcap, "out of memory"); return NULL; }
    s->eng = e; s->prog = p; s->bsess = bs; s->max_batch = max_batch;
    return s;
}

void zt_session_destroy(zt_session *s)
{
    if (!s) return;
    s->eng->ops->session_destroy(s->bsess);
    free(s);
}

const char *zt_session_error(const zt_session *s) { return s ? s->err : ""; }

zt_status zt_session_bind_head(zt_session *s, const zt_tensor *h)
{
    zt_status st;
    if (!s || !h || !h->data) return ZT_ERR_INVALID_ARG;
    if (h->cols == 0 || h->ld < h->cols) return ZT_ERR_SHAPE;
    if (h->dtype > ZT_DTYPE_BF16) return ZT_ERR_INVALID_ARG;
    if ((int64_t)h->rows <= (int64_t)s->prog->max_token) {
        snprintf(s->err, sizeof s->err,
                 "lm_head has %u rows but the program references token %d",
                 h->rows, s->prog->max_token);
        return ZT_ERR_SHAPE;
    }
    s->err[0] = '\0';
    st = s->eng->ops->bind_head(s->bsess, h, s->err, sizeof s->err);
    if (st == ZT_OK) s->head_H = h->cols;
    return st;
}

zt_status zt_session_run(zt_session *s, zt_input_kind kind, const zt_tensor *in, zt_result *out)
{
    const zt_program *p;
    uint32_t F, batch, b;
    zt_status st;

    if (!s || !in || !out || !in->data || !out->dec) return ZT_ERR_INVALID_ARG;
    p = s->prog;
    F = p->n_fields;
    s->err[0] = '\0';

    if (in->dtype > ZT_DTYPE_BF16) return ZT_ERR_INVALID_ARG;
    if (in->ld < in->cols) return ZT_ERR_SHAPE;
    if (in->rows == 0 || in->rows % F != 0) {
        snprintf(s->err, sizeof s->err, "input rows (%u) must be a multiple of n_fields (%u)", in->rows, F);
        return ZT_ERR_SHAPE;
    }
    batch = in->rows / F;
    if (batch > s->max_batch) {
        snprintf(s->err, sizeof s->err, "batch %u exceeds session max_batch %u", batch, s->max_batch);
        return ZT_ERR_SHAPE;
    }
    if (batch > out->cap_batch || out->n_fields != F) {
        snprintf(s->err, sizeof s->err, "result buffer too small (cap_batch %u, n_fields %u)", out->cap_batch, out->n_fields);
        return ZT_ERR_SHAPE;
    }
    if (kind == ZT_INPUT_LOGITS) {
        if ((int64_t)in->cols <= (int64_t)p->max_token) {
            snprintf(s->err, sizeof s->err, "logits have %u columns but the program references token %d",
                     in->cols, p->max_token);
            return ZT_ERR_SHAPE;
        }
    } else if (kind == ZT_INPUT_HIDDEN) {
        if (s->head_H == 0) { snprintf(s->err, sizeof s->err, "hidden input requires zt_session_bind_head"); return ZT_ERR_NOT_BOUND; }
        if (in->cols != s->head_H) {
            snprintf(s->err, sizeof s->err, "hidden size %u does not match bound lm_head H=%u", in->cols, s->head_H);
            return ZT_ERR_SHAPE;
        }
    } else {
        return ZT_ERR_INVALID_ARG;
    }

    st = s->eng->ops->run(s->bsess, kind, in, batch, out, s->err, sizeof s->err);
    if (st != ZT_OK) return st;
    out->batch = batch;
    for (b = 0; b < batch; b++) (void)zt_fsm_walk(p, out, b);
    return ZT_OK;
}

/* ----------------------------------------------------------------- results */

zt_status zt_result_alloc(zt_result *r, const zt_program *p, uint32_t max_batch, int with_probs)
{
    if (!r || !p || max_batch == 0) return ZT_ERR_INVALID_ARG;
    memset(r, 0, sizeof *r);
    r->cap_batch = max_batch;
    r->n_fields  = p->n_fields;
    r->c_max     = p->c_max;
    r->dec = (zt_decision *)calloc((size_t)max_batch * p->n_fields, sizeof *r->dec);
    if (!r->dec) return ZT_ERR_NO_MEMORY;
    if (with_probs) {
        r->probs = (float *)calloc((size_t)max_batch * p->n_fields * p->c_max, sizeof *r->probs);
        if (!r->probs) { free(r->dec); r->dec = NULL; return ZT_ERR_NO_MEMORY; }
    }
    return ZT_OK;
}

void zt_result_free(zt_result *r)
{
    if (!r) return;
    free(r->dec); free(r->probs);
    memset(r, 0, sizeof *r);
}

const zt_decision *zt_result_decision(const zt_result *r, uint32_t b, uint32_t field)
{
    if (!r || !r->dec || b >= r->cap_batch || field >= r->n_fields) return NULL;
    return &r->dec[(size_t)b * r->n_fields + field];
}

const float *zt_result_probs(const zt_result *r, uint32_t b, uint32_t field)
{
    if (!r || !r->probs || b >= r->cap_batch || field >= r->n_fields) return NULL;
    return r->probs + ((size_t)b * r->n_fields + field) * r->c_max;
}

/* ------------------------------------------------------------------- utils */
uint16_t zt_f32_to_f16(float f) { return zt_f32_to_f16_(f); }
float    zt_f16_to_f32(uint16_t h) { return zt_f16_to_f32_(h); }
