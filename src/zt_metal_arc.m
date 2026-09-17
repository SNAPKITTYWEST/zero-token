/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-012
   File:              zt_metal_arc.m
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

   Clone-Gate: sha256:ce1a255368fee1bd487bd80880ee1b3ca403428ed267d55f6368a57d5c515c08

   See: LICENSE
   ======================================================================== */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>

#include "zt_internal.h"

#if defined(ZT_METAL_EMBED_SRC)
# include "zt_metal_src.h"
#endif

/* ------------------------------------------------------------------ classes */

@interface ZTMetal : NSObject
@property (strong) id<MTLDevice> dev;
@property (strong) id<MTLCommandQueue> queue;
@property (strong) id<MTLComputePipelineState> gatherLogits;
@property (strong) id<MTLComputePipelineState> gatherHead;
@property (strong) id<MTLComputePipelineState> headSimd;
@property (strong) id<MTLComputePipelineState> headSimt;
@property (strong) id<MTLComputePipelineState> decide;
@property (assign) BOOL hasSimdMatrix;
@end
@implementation ZTMetal
@end

@interface ZTMetalSession : NSObject
@property (strong) id<MTLBuffer> fields;
@property (strong) id<MTLBuffer> colToken;
@property (strong) id<MTLBuffer> choiceStart;
@property (strong) id<MTLBuffer> choiceValue;
@property (strong) id<MTLBuffer> knots;
@property (strong) id<MTLBuffer> W;
@property (strong) id<MTLBuffer> x;
@property (strong) id<MTLBuffer> staging;
@property (strong) id<MTLBuffer> dec;
@property (strong) id<MTLBuffer> probs;
@end
@implementation ZTMetalSession
@end

typedef struct { void *obj; ZTMetal *unsafe; } metal_ctx_arc;
typedef struct {
    metal_ctx_arc *ctx;
    void *obj; ZTMetalSession *unsafe;
    const zt_program *prog;
    uint32_t max_rows, Kpad, H;
    size_t staging_cap, probs_cap;
} metal_sess_arc;

/* ------------------------------------------------------------------ helpers */

static NSString *zt_metal_source_arc(void)
{
#if defined(ZT_METAL_EMBED_SRC)
    return [[NSString alloc] initWithBytes:zt_metal_src
                                    length:(NSUInteger)zt_metal_src_len
                                  encoding:NSUTF8StringEncoding];
#else
    return nil;
#endif
}

static NSString *zt_sibling_metallib_arc(void)
{
    Dl_info info;
    if (dladdr((const void *)&zt_sibling_metallib_arc, &info) && info.dli_fname) {
        NSString *dir = [[NSString stringWithUTF8String:info.dli_fname]
                          stringByDeletingLastPathComponent];
        return [dir stringByAppendingPathComponent:@"zt.metallib"];
    }
    return nil;
}

static id<MTLLibrary> zt_load_library_arc(id<MTLDevice> dev, NSError **err)
{
    const char *envp = getenv("ZT_METALLIB");
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *path = envp ? [NSString stringWithUTF8String:envp]
                           : zt_sibling_metallib_arc();
    if (path && [fm fileExistsAtPath:path]) {
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:path] error:err];
        if (lib) return lib;
    }
    NSString *src = zt_metal_source_arc();
    if (!src) return nil;
    MTLCompileOptions *opt = [MTLCompileOptions new];
    opt.languageVersion = MTLLanguageVersion2_3;
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    opt.mathMode = MTLMathModeSafe;
#else
    opt.fastMathEnabled = NO;
#endif
    return [dev newLibraryWithSource:src options:opt error:err];
}

static id<MTLComputePipelineState>
zt_pipeline_arc(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name, NSError **err)
{
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        if (err) *err = [NSError errorWithDomain:@"zt" code:1 userInfo:
            @{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"kernel '%@' not found", name]}];
        return nil;
    }
    return [dev newComputePipelineStateWithFunction:fn error:err];
}

static id<MTLBuffer> zt_buf_arc(id<MTLDevice> dev, const void *src, size_t bytes)
{
    if (bytes == 0) bytes = 4;
    if (src) return [dev newBufferWithBytes:src length:bytes
                                    options:MTLResourceStorageModeShared];
    return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

/* ------------------------------------------------------------------ context */

static zt_status metal_arc_create(void **out, char *err, size_t errcap)
{
    @autoreleasepool {
        NSError *e = nil;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { ZT_SETERR(err, errcap, "no Metal device"); return ZT_ERR_UNAVAILABLE; }
        id<MTLLibrary> lib = zt_load_library_arc(dev, &e);
        if (!lib) {
            ZT_SETERR(err, errcap, "metal library: %s",
                      e ? [[e localizedDescription] UTF8String] : "no source or metallib available");
            return ZT_ERR_BACKEND;
        }
        ZTMetal *m = [ZTMetal new];
        m.dev = dev;
        m.queue = [dev newCommandQueue];
        m.gatherLogits = zt_pipeline_arc(dev, lib, @"zt_k_gather_logits", &e);
        m.gatherHead   = zt_pipeline_arc(dev, lib, @"zt_k_gather_head",   &e);
        m.headSimd     = zt_pipeline_arc(dev, lib, @"zt_k_head_simd",     &e);
        m.headSimt     = zt_pipeline_arc(dev, lib, @"zt_k_head_simt",     &e);
        m.decide       = zt_pipeline_arc(dev, lib, @"zt_k_decide",        &e);
        if (!m.queue || !m.gatherLogits || !m.gatherHead || !m.headSimt || !m.decide) {
            ZT_SETERR(err, errcap, "pipeline creation failed: %s",
                      e ? [[e localizedDescription] UTF8String] : "unknown");
            return ZT_ERR_BACKEND;
        }
        m.hasSimdMatrix = (m.headSimd != nil) && (m.decide.threadExecutionWidth == 32);
        if (m.decide.threadExecutionWidth != 32) {
            ZT_SETERR(err, errcap, "unexpected SIMD width %lu",
                      (unsigned long)m.decide.threadExecutionWidth);
            return ZT_ERR_UNAVAILABLE;
        }
        metal_ctx_arc *c = (metal_ctx_arc *)calloc(1, sizeof *c);
        if (!c) { ZT_SETERR(err, errcap, "out of memory"); return ZT_ERR_NO_MEMORY; }
        c->obj = (__bridge_retained void *)m;
        c->unsafe = m;
        *out = c;
        return ZT_OK;
    }
}

static void metal_arc_destroy(void *ctx)
{
    metal_ctx_arc *c = (metal_ctx_arc *)ctx;
    if (!c) return;
    if (c->obj) { ZTMetal *m = (__bridge_transfer ZTMetal *)c->obj; m = nil; }
    free(c);
}

/* ------------------------------------------------------------------ session */

static void metal_arc_session_destroy(void *sess)
{
    metal_sess_arc *s = (metal_sess_arc *)sess;
    if (!s) return;
    if (s->obj) { ZTMetalSession *o = (__bridge_transfer ZTMetalSession *)s->obj; o = nil; }
    free(s);
}

static zt_status metal_arc_session_create(void *ctx, const zt_program *p,
                                           uint32_t max_batch, void **out,
                                           char *err, size_t errcap)
{
    @autoreleasepool {
        metal_ctx_arc *c = (metal_ctx_arc *)ctx;
        ZTMetal *m = c->unsafe;
        ZTMetalSession *o = [ZTMetalSession new];
        metal_sess_arc *s = (metal_sess_arc *)calloc(1, sizeof *s);
        if (!s) { ZT_SETERR(err, errcap, "out of memory"); return ZT_ERR_NO_MEMORY; }
        uint32_t nk = p->n_knots ? p->n_knots : 1u;
        s->ctx = c; s->prog = p;
        s->max_rows = max_batch * p->n_fields;
        s->Kpad = zt_round_up(p->n_cols, 8u);

        o.fields      = zt_buf_arc(m.dev, p->fields,          (size_t)p->n_fields       * sizeof(zt_field_dev));
        o.colToken    = zt_buf_arc(m.dev, p->col_token,        (size_t)p->n_cols         * sizeof(int32_t));
        o.choiceStart = zt_buf_arc(m.dev, p->choice_col_start, (size_t)(p->n_choices + 1)* sizeof(uint32_t));
        o.choiceValue = zt_buf_arc(m.dev, p->choice_value,     (size_t)p->n_choices      * sizeof(float));
        o.knots       = zt_buf_arc(m.dev, p->knots,            (size_t)nk                * sizeof(zt_knot));
        o.x   = zt_buf_arc(m.dev, NULL, (size_t)s->max_rows * s->Kpad * sizeof(float));
        o.dec = zt_buf_arc(m.dev, NULL, (size_t)s->max_rows * sizeof(zt_decision));
        if (!o.fields || !o.colToken || !o.choiceStart || !o.choiceValue ||
            !o.knots || !o.x || !o.dec) {
            free(s);
            ZT_SETERR(err, errcap, "buffer allocation failed");
            return ZT_ERR_BACKEND;
        }
        s->obj = (__bridge_retained void *)o;
        s->unsafe = o;
        *out = s;
        return ZT_OK;
    }
}

static zt_status metal_arc_bind_head(void *sess, const zt_tensor *head,
                                      char *err, size_t errcap)
{
    @autoreleasepool {
        metal_sess_arc *s = (metal_sess_arc *)sess;
        ZTMetal *m = s->ctx->unsafe;
        ZTMetalSession *o = s->unsafe;
        const zt_program *p = s->prog;
        uint32_t K = p->n_cols, H = head->cols;
        size_t bytes = (size_t)s->Kpad * H * sizeof(uint16_t);

        if (head->mem == ZT_MEM_HOST) {
            uint16_t *tmp = (uint16_t *)malloc(bytes);
            if (!tmp) { ZT_SETERR(err, errcap, "out of memory"); return ZT_ERR_NO_MEMORY; }
            zt_host_gather_head_f16(head, p->col_token, K, s->Kpad, H, tmp);
            o.W = zt_buf_arc(m.dev, tmp, bytes);
            free(tmp);
            if (!o.W) { ZT_SETERR(err, errcap, "head buffer allocation failed"); return ZT_ERR_BACKEND; }
        } else {
            id<MTLBuffer> src = (__bridge id<MTLBuffer>)(void *)head->data;
            o.W = zt_buf_arc(m.dev, NULL, bytes);
            if (!o.W) { ZT_SETERR(err, errcap, "head buffer allocation failed"); return ZT_ERR_BACKEND; }
            zt_head_params prm;
            prm.dtype = head->dtype; prm.ld = head->ld; prm.rows = head->rows;
            prm.H = H; prm.K = K; prm.Kpad = s->Kpad;
            uint32_t n = s->Kpad * H, Kpad = s->Kpad;
            id<MTLCommandBuffer> cb = [m.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:m.gatherHead];
            [enc setBuffer:src offset:head->offset atIndex:0];
            [enc setBuffer:o.colToken offset:0 atIndex:1];
            [enc setBuffer:o.W offset:0 atIndex:2];
            [enc setBytes:&Kpad length:sizeof Kpad atIndex:3];
            [enc setBytes:&prm length:sizeof prm atIndex:4];
            [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            if (cb.error) {
                ZT_SETERR(err, errcap, "head gather failed: %s",
                          [[cb.error localizedDescription] UTF8String]);
                return ZT_ERR_BACKEND;
            }
        }
        s->H = H;
        return ZT_OK;
    }
}

static zt_status metal_arc_run(void *sess, zt_input_kind kind, const zt_tensor *in,
                                uint32_t batch, zt_result *out,
                                char *err, size_t errcap)
{
    @autoreleasepool {
        metal_sess_arc *s = (metal_sess_arc *)sess;
        ZTMetal *m = s->ctx->unsafe;
        ZTMetalSession *o = s->unsafe;
        const zt_program *p = s->prog;
        uint32_t rows = batch * p->n_fields;
        uint32_t ld_out = s->Kpad;
        id<MTLBuffer> input;
        size_t in_off = 0;

        if (in->mem == ZT_MEM_DEVICE) {
            input = (__bridge id<MTLBuffer>)(void *)in->data;
            in_off = in->offset;
        } else {
            size_t bytes = (size_t)rows * in->ld * zt_dtype_size(in->dtype);
            if (bytes > s->staging_cap) {
                o.staging = zt_buf_arc(m.dev, NULL, bytes);
                if (!o.staging) { ZT_SETERR(err, errcap, "staging alloc failed"); return ZT_ERR_BACKEND; }
                s->staging_cap = bytes;
            }
            memcpy([o.staging contents], (const uint8_t *)in->data + in->offset, bytes);
            input = o.staging;
        }

        if (out->probs) {
            size_t need = (size_t)rows * p->c_max * sizeof(float);
            if (need > s->probs_cap) {
                o.probs = zt_buf_arc(m.dev, NULL, need);
                if (!o.probs) { ZT_SETERR(err, errcap, "probs alloc failed"); return ZT_ERR_BACKEND; }
                s->probs_cap = need;
            }
        } else if (!o.probs) {
            o.probs = zt_buf_arc(m.dev, NULL, 4);
        }

        id<MTLCommandBuffer> cb = [m.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        if (kind == ZT_INPUT_LOGITS) {
            zt_gather_params gp;
            gp.dtype = in->dtype; gp.ld = in->ld; gp.K = p->n_cols; gp.rows = rows;
            [enc setComputePipelineState:m.gatherLogits];
            [enc setBuffer:input offset:in_off atIndex:0];
            [enc setBuffer:o.colToken offset:0 atIndex:1];
            [enc setBuffer:o.x offset:0 atIndex:2];
            [enc setBytes:&ld_out length:sizeof ld_out atIndex:3];
            [enc setBytes:&gp length:sizeof gp atIndex:4];
            [enc dispatchThreads:MTLSizeMake(rows * p->n_cols, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            if (!o.W) { [enc endEncoding]; ZT_SETERR(err, errcap, "no lm_head bound"); return ZT_ERR_NOT_BOUND; }
            zt_head_params hp;
            hp.dtype = in->dtype; hp.ld = in->ld; hp.rows = rows;
            hp.H = s->H; hp.K = p->n_cols; hp.Kpad = s->Kpad;
            [enc setBuffer:input offset:in_off atIndex:0];
            [enc setBuffer:o.W offset:0 atIndex:1];
            [enc setBuffer:o.x offset:0 atIndex:2];
            [enc setBytes:&ld_out length:sizeof ld_out atIndex:3];
            [enc setBytes:&hp length:sizeof hp atIndex:4];
            if (m.hasSimdMatrix && (s->H % 8u) == 0u) {
                MTLSize grid = MTLSizeMake((s->Kpad/8u+3u)/4u, (rows+7u)/8u, 1);
                [enc setComputePipelineState:m.headSimd];
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            } else {
                uint32_t groups = (rows * p->n_cols + 3u) / 4u;
                [enc setComputePipelineState:m.headSimt];
                [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
            }
        }

        zt_decide_params dp;
        dp.K = p->n_cols; dp.ld = s->Kpad; dp.F = p->n_fields;
        dp.rows = rows; dp.c_max = p->c_max; dp.has_probs = out->probs ? 1u : 0u;
        [enc setComputePipelineState:m.decide];
        [enc setBuffer:o.x           offset:0 atIndex:0];
        [enc setBuffer:o.fields      offset:0 atIndex:1];
        [enc setBuffer:o.choiceStart offset:0 atIndex:2];
        [enc setBuffer:o.choiceValue offset:0 atIndex:3];
        [enc setBuffer:o.knots       offset:0 atIndex:4];
        [enc setBuffer:o.dec         offset:0 atIndex:5];
        [enc setBuffer:o.probs       offset:0 atIndex:6];
        [enc setBytes:&dp length:sizeof dp atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((rows+3u)/4u,1,1)
              threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.error) {
            ZT_SETERR(err, errcap, "dispatch failed: %s",
                      [[cb.error localizedDescription] UTF8String]);
            return ZT_ERR_BACKEND;
        }

        memcpy(out->dec, [o.dec contents], (size_t)rows * sizeof(zt_decision));
        if (out->probs)
            memcpy(out->probs, [o.probs contents], (size_t)rows * p->c_max * sizeof(float));
        return ZT_OK;
    }
}

const zt_backend_ops zt_metal_arc_ops = {
    "metal", ZT_BACKEND_METAL,
    metal_arc_create, metal_arc_destroy,
    metal_arc_session_create, metal_arc_session_destroy,
    metal_arc_bind_head, metal_arc_run
};
