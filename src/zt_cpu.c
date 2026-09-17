/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-007
   File:              zt_cpu.c
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

   Clone-Gate: sha256:123acf95fc32b23b1cfaf0609f524e8435841ffce9574f3af8cea97cb9a68ca2

   See: LICENSE
   ======================================================================== */
#include <math.h>
#include "zt_internal.h"

/* --------------------------------------------------------------- helpers */

/* Element (r, c) of a host tensor as float. */
static float host_at(const zt_tensor *t, uint32_t r, uint32_t c)
{
    const uint8_t *row = (const uint8_t *)t->data + t->offset +
                         (size_t)r * t->ld * zt_dtype_size(t->dtype);
    return zt_load_f(row, t->dtype, c);
}

void zt_host_gather_head_f16(const zt_tensor *head, const int32_t *col_token,
                             uint32_t K, uint32_t Kpad, uint32_t H, uint16_t *out)
{
    uint32_t k, h;
    for (k = 0; k < K; k++) {
        uint32_t v = (uint32_t)col_token[k];
        if (head->dtype == ZT_DTYPE_F16) {
            const uint8_t *row = (const uint8_t *)head->data + head->offset +
                                 (size_t)v * head->ld * 2u;
            memcpy(out + (size_t)k * H, row, (size_t)H * 2u);   /* already fp16: bit-exact copy */
        } else {
            for (h = 0; h < H; h++) out[(size_t)k * H + h] = zt_f32_to_f16_(host_at(head, v, h));
        }
    }
    if (Kpad > K) memset(out + (size_t)K * H, 0, (size_t)(Kpad - K) * H * 2u);
}

/* Sequential reference of the warp/simdgroup decision kernel. */
void zt_decide_row_ref(const zt_program *p, const float *x, uint32_t field,
                       zt_decision *d, float *probs)
{
    const zt_field_dev *f = &p->fields[field];
    const float *xs = x + f->col_start;
    uint32_t i, c;
    float m = -INFINITY, z = 0.0f;
    float best = -1.0f, second = -1.0f, ent = 0.0f, ex = 0.0f, p0 = 0.0f;
    int32_t best_i = 0;

    for (i = 0; i < f->n_cols; i++) m = fmaxf(m, xs[i] * f->inv_temp);
    for (i = 0; i < f->n_cols; i++) z += expf(xs[i] * f->inv_temp - m);
    for (c = 0; c < f->n_choices; c++) {
        uint32_t g  = f->choice_start + c;
        uint32_t lo = p->choice_col_start[g] - f->col_start;
        uint32_t hi = p->choice_col_start[g + 1] - f->col_start;
        float part = 0.0f, pc;
        for (i = lo; i < hi; i++) part += expf(xs[i] * f->inv_temp - m);
        pc = part / z;
        if (pc > best) { second = best; best = pc; best_i = (int32_t)c; }
        else if (pc > second) second = pc;
        if (pc > 0.0f) ent -= pc * logf(pc);
        ex += pc * p->choice_value[g];
        if (c == 0) p0 = pc;
        if (probs) probs[c] = pc;
    }
    zt_finalize(d, f, p->knots, best_i, best, second, p0, ent, ex);
}

/* ------------------------------------------------------------ backend ops */

typedef struct {
    const zt_program *prog;
    uint32_t          max_batch;
    uint32_t          H;        /* 0 until a head is bound */
    float            *wsub;     /* [K][H] fp16-rounded weights as float */
    float            *rowbuf;   /* [max(H, K)] scratch */
    float            *sub;      /* [K] one row of sub-logits */
} cpu_sess;

static zt_status cpu_create(void **ctx, char *err, size_t errcap)
{
    (void)err; (void)errcap;
    *ctx = NULL;           /* stateless */
    return ZT_OK;
}

static void cpu_destroy(void *ctx) { (void)ctx; }

static zt_status cpu_session_create(void *ctx, const zt_program *p, uint32_t max_batch,
                                    void **sess, char *err, size_t errcap)
{
    cpu_sess *s;
    (void)ctx;
    s = (cpu_sess *)calloc(1, sizeof *s);
    if (!s) { ZT_SETERR(err, errcap, "out of memory"); return ZT_ERR_NO_MEMORY; }
    s->prog = p;
    s->max_batch = max_batch;
    s->sub = (float *)malloc((size_t)p->n_cols * sizeof(float));
    if (!s->sub) { free(s); ZT_SETERR(err, errcap, "out of memory"); return ZT_ERR_NO_MEMORY; }
    *sess = s;
    return ZT_OK;
}

static void cpu_session_destroy(void *sess)
{
    cpu_sess *s = (cpu_sess *)sess;
    if (!s) return;
    free(s->wsub); free(s->rowbuf); free(s->sub); free(s);
}

static zt_status cpu_bind_head(void *sess, const zt_tensor *head, char *err, size_t errcap)
{
    cpu_sess *s = (cpu_sess *)sess;
    const zt_program *p = s->prog;
    uint32_t K = p->n_cols, H = head->cols, i;
    uint16_t *tmp;
    float *w, *rb;

    if (head->mem != ZT_MEM_HOST) { ZT_SETERR(err, errcap, "cpu backend needs a host-memory lm_head"); return ZT_ERR_INVALID_ARG; }
    tmp = (uint16_t *)malloc((size_t)K * H * 2u);
    w   = (float *)malloc((size_t)K * H * sizeof(float));
    rb  = (float *)malloc((size_t)(H > K ? H : K) * sizeof(float));
    if (!tmp || !w || !rb) { free(tmp); free(w); free(rb); ZT_SETERR(err, errcap, "out of memory"); return ZT_ERR_NO_MEMORY; }
    zt_host_gather_head_f16(head, p->col_token, K, K, H, tmp);
    for (i = 0; i < K * H; i++) w[i] = zt_f16_to_f32_(tmp[i]);
    free(tmp);
    free(s->wsub); free(s->rowbuf);
    s->wsub = w; s->rowbuf = rb; s->H = H;
    return ZT_OK;
}

static zt_status cpu_run(void *sess, zt_input_kind kind, const zt_tensor *in, uint32_t batch,
                         zt_result *out, char *err, size_t errcap)
{
    cpu_sess *s = (cpu_sess *)sess;
    const zt_program *p = s->prog;
    uint32_t F = p->n_fields, K = p->n_cols, rows = batch * F, r, k, h;

    if (in->mem != ZT_MEM_HOST) { ZT_SETERR(err, errcap, "cpu backend needs host-memory input"); return ZT_ERR_INVALID_ARG; }
    for (r = 0; r < rows; r++) {
        uint32_t f = r % F;
        float *probs = out->probs ? out->probs + (size_t)r * p->c_max : NULL;
        if (kind == ZT_INPUT_LOGITS) {
            for (k = 0; k < K; k++) s->sub[k] = host_at(in, r, (uint32_t)p->col_token[k]);
        } else {
            const uint32_t H = s->H;
            for (h = 0; h < H; h++) s->rowbuf[h] = host_at(in, r, h);
            for (k = 0; k < K; k++) {
                const float *w = s->wsub + (size_t)k * H;
                float acc = 0.0f;
                for (h = 0; h < H; h++) acc += s->rowbuf[h] * w[h];
                s->sub[k] = acc;
            }
        }
        zt_decide_row_ref(p, s->sub, f, &out->dec[r], probs);
    }
    return ZT_OK;
}

const zt_backend_ops zt_cpu_ops = {
    "cpu", ZT_BACKEND_CPU,
    cpu_create, cpu_destroy, cpu_session_create, cpu_session_destroy, cpu_bind_head, cpu_run
};
