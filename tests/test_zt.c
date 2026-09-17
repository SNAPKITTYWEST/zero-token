/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-018
   File:              test_zt.c
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

   Clone-Gate: sha256:9e93b5ed40fc0fcd182c7ac1645ab58ff76e82f2e466beeb8e7b13d520ebe3f4

   See: LICENSE
   ======================================================================== */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zt.h"

static int failures = 0, checks = 0;

#define CHECK(cond) do {                                                     \
        checks++;                                                             \
        if (!(cond)) { failures++;                                            \
            printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); }        \
    } while (0)

#define CHECK_NEAR(a, b, tol) do {                                           \
        double a_ = (double)(a), b_ = (double)(b);                            \
        checks++;                                                             \
        if (!(fabs(a_ - b_) <= (tol))) { failures++;                          \
            printf("  FAIL %s:%d: %s = %.9g, expected %.9g (tol %g)\n",       \
                   __FILE__, __LINE__, #a, a_, b_, (double)(tol)); }          \
    } while (0)

static void section(const char *s) { printf("[%s]\n", s); }

/* ---------------------------------------------------------------- program */

static zt_program *build_program(void)
{
    zt_schema *s = zt_schema_new();
    zt_program *p = NULL;
    char err[256];
    int f_ref, f_cat, f_urg;
    const int32_t t_true[2] = { 10, 11 }, t_false[1] = { 12 };
    const int32_t c0[1] = { 20 }, c1[1] = { 21 }, c2[1] = { 22 }, c3[1] = { 23 };
    const int32_t u0[1] = { 30 }, u1[1] = { 31 }, u2[1] = { 32 };

    f_ref = zt_schema_add_field(s, "is_refund", ZT_KIND_BOOL);
    f_cat = zt_schema_add_field(s, "category",  ZT_KIND_CHOICE);
    f_urg = zt_schema_add_field(s, "urgency",   ZT_KIND_SCORE);
    if (f_ref < 0 || f_cat < 0 || f_urg < 0) { zt_schema_free(s); return NULL; }

    zt_schema_add_choice(s, f_ref, "yes", t_true, 2, 0.0f);
    zt_schema_add_choice(s, f_ref, "no",  t_false, 1, 0.0f);
    zt_schema_add_choice(s, f_cat, "billing",  c0, 1, 0.0f);
    zt_schema_add_choice(s, f_cat, "shipping", c1, 1, 0.0f);
    zt_schema_add_choice(s, f_cat, "product",  c2, 1, 0.0f);
    zt_schema_add_choice(s, f_cat, "other",    c3, 1, 0.0f);
    zt_schema_add_choice(s, f_urg, "low",    u0, 1, 1.0f);
    zt_schema_add_choice(s, f_urg, "medium", u1, 1, 2.0f);
    zt_schema_add_choice(s, f_urg, "high",   u2, 1, 3.0f);
    zt_schema_set_dependency(s, f_cat, f_ref, 1);              /* only when not a refund */
    zt_schema_set_field_params(s, f_urg, 2.0f, 0.5f, 0.0f, -INFINITY);

    if (zt_compile(s, &p, err, sizeof err) != ZT_OK) { printf("  compile failed: %s\n", err); p = NULL; }
    zt_schema_free(s);
    return p;
}

/* ------------------------------------------------------------ logits path */

static void test_logits(zt_engine *e, const zt_program *p)
{
    const uint32_t V = 64, F = 3, B = 2;
    float *logits = (float *)calloc((size_t)B * F * V, sizeof(float));
    zt_session *ss;
    zt_result r;
    zt_tensor in;
    char err[256];
    const zt_decision *d;
    const float *pr;
    double e10, e11, e12, z, p_true;

    section("logits path");
    /* batch 0 */
    logits[0 * V + 10] = 2.0f; logits[0 * V + 11] = 1.0f; logits[0 * V + 12] = 0.5f;  /* is_refund */
    logits[1 * V + 20] = 0.1f; logits[1 * V + 21] = 3.0f; logits[1 * V + 22] = 0.2f; logits[1 * V + 23] = 0.0f;
    logits[2 * V + 30] = 0.0f; logits[2 * V + 31] = 0.0f; logits[2 * V + 32] = 4.0f;  /* urgency, T=2 */
    /* batch 1: not a refund -> category is evaluated */
    logits[3 * V + 10] = 0.0f; logits[3 * V + 11] = 0.0f; logits[3 * V + 12] = 5.0f;
    logits[4 * V + 20] = 1.0f; logits[4 * V + 21] = 1.0f; logits[4 * V + 22] = 1.0f; logits[4 * V + 23] = 1.0f;
    logits[5 * V + 30] = 1.0f; logits[5 * V + 31] = 1.0f; logits[5 * V + 32] = 1.0f;

    ss = zt_session_create(e, p, B, err, sizeof err);
    CHECK(ss != NULL);
    if (!ss) { free(logits); return; }
    CHECK(zt_result_alloc(&r, p, B, 1) == ZT_OK);

    memset(&in, 0, sizeof in);
    in.data = logits; in.rows = B * F; in.cols = V; in.ld = V;
    in.dtype = ZT_DTYPE_F32; in.mem = ZT_MEM_HOST;
    CHECK(zt_session_run(ss, ZT_INPUT_LOGITS, &in, &r) == ZT_OK);
    CHECK(r.batch == B);

    /* batch 0, is_refund: P(true) = (e^2 + e^1) / (e^2 + e^1 + e^0.5) */
    e10 = exp(2.0); e11 = exp(1.0); e12 = exp(0.5);
    z = e10 + e11 + e12;
    p_true = (e10 + e11) / z;
    d = zt_result_decision(&r, 0, 0);
    CHECK(d->choice == 0);
    CHECK_NEAR(d->score, p_true, 1e-5);
    CHECK_NEAR(d->p_top, p_true, 1e-5);
    CHECK_NEAR(d->p_second, 1.0 - p_true, 1e-5);
    CHECK_NEAR(d->margin, log(p_true) - log(1.0 - p_true), 1e-4);

    /* batch 0, category: skipped (guard wanted choice 1, got 0) */
    d = zt_result_decision(&r, 0, 1);
    CHECK(d->choice == ZT_SKIPPED);

    /* batch 0, urgency with T = 2: softmax(0, 0, 2) -> expectation */
    {
        double a = exp(0.0), b = exp(0.0), c = exp(2.0), zz = a + b + c;
        double ex = 1.0 * a / zz + 2.0 * b / zz + 3.0 * c / zz;
        d = zt_result_decision(&r, 0, 2);
        CHECK(d->choice == 2);
        CHECK_NEAR(d->score, ex, 1e-5);
        CHECK_NEAR(d->p_top, c / zz, 1e-5);
    }

    /* batch 1: category evaluated, all logits equal -> uniform, lowest index wins */
    d = zt_result_decision(&r, 1, 0);
    CHECK(d->choice == 1);                      /* "no" */
    d = zt_result_decision(&r, 1, 1);
    CHECK(d->choice == 0);                      /* tie -> lowest index */
    CHECK_NEAR(d->p_top, 0.25, 1e-5);
    CHECK_NEAR(d->margin, 0.0, 1e-5);
    CHECK_NEAR(d->entropy, log(4.0), 1e-5);
    pr = zt_result_probs(&r, 1, 1);
    CHECK(pr != NULL);
    if (pr) { CHECK_NEAR(pr[0], 0.25, 1e-5); CHECK_NEAR(pr[3], 0.25, 1e-5); }

    CHECK(zt_fsm_walk(p, &r, 1) == ZT_ROUTE_COMPLETE);

    zt_result_free(&r);
    zt_session_destroy(ss);
    free(logits);
}

/* ------------------------------------------------------------ hidden path */

static void test_hidden(zt_engine *e, const zt_program *p, uint32_t dtype, const char *name)
{
    const uint32_t V = 64, H = 64, F = 3, B = 2;
    const uint32_t rows = B * F;
    float *head = (float *)malloc((size_t)V * H * sizeof(float));
    float *hid  = (float *)malloc((size_t)rows * H * sizeof(float));
    void  *hid_t;
    uint16_t *hid16 = NULL;
    zt_session *ss;
    zt_result r;
    zt_tensor th, ti;
    char err[256];
    uint32_t i, k, h, K = zt_program_n_cols(p);
    const int32_t *toks = zt_program_col_tokens(p, NULL);
    unsigned seed = 12345u;

    printf("[hidden path: %s]\n", name);
    for (i = 0; i < V * H; i++) { seed = seed * 1103515245u + 12345u; head[i] = ((float)((seed >> 16) & 0x7fffu) / 16384.0f - 1.0f) * 0.2f; }
    for (i = 0; i < rows * H; i++) { seed = seed * 1103515245u + 12345u; hid[i] = ((float)((seed >> 16) & 0x7fffu) / 16384.0f - 1.0f); }

    if (dtype == ZT_DTYPE_F32) {
        hid_t = hid;
    } else {
        hid16 = (uint16_t *)malloc((size_t)rows * H * sizeof(uint16_t));
        for (i = 0; i < rows * H; i++) {
            if (dtype == ZT_DTYPE_F16) { hid16[i] = zt_f32_to_f16(hid[i]); hid[i] = zt_f16_to_f32(hid16[i]); }
            else { /* bf16: truncate toward zero at 16 bits (round-to-nearest not required here) */
                uint32_t bits; memcpy(&bits, &hid[i], 4); hid16[i] = (uint16_t)(bits >> 16);
                bits = (uint32_t)hid16[i] << 16; memcpy(&hid[i], &bits, 4);
            }
        }
        hid_t = hid16;
    }

    ss = zt_session_create(e, p, B, err, sizeof err);
    CHECK(ss != NULL);
    if (!ss) { free(head); free(hid); free(hid16); return; }
    CHECK(zt_result_alloc(&r, p, B, 1) == ZT_OK);

    memset(&th, 0, sizeof th);
    th.data = head; th.rows = V; th.cols = H; th.ld = H; th.dtype = ZT_DTYPE_F32; th.mem = ZT_MEM_HOST;
    CHECK(zt_session_bind_head(ss, &th) == ZT_OK);

    memset(&ti, 0, sizeof ti);
    ti.data = hid_t; ti.rows = rows; ti.cols = H; ti.ld = H; ti.dtype = dtype; ti.mem = ZT_MEM_HOST;
    CHECK(zt_session_run(ss, ZT_INPUT_HIDDEN, &ti, &r) == ZT_OK);

    /* Independent naive check of the projection for row 0, field 0 (BOOL). */
    {
        float sub[8];
        double m, zz, pt;
        for (k = 0; k < K && k < 8; k++) {
            double acc = 0.0;
            for (h = 0; h < H; h++)
                acc += (double)hid[h] * (double)zt_f16_to_f32(zt_f32_to_f16(head[(size_t)toks[k] * H + h]));
            sub[k] = (float)acc;
        }
        m = sub[0] > sub[1] ? sub[0] : sub[1];
        if (sub[2] > m) m = sub[2];
        zz = exp(sub[0] - m) + exp(sub[1] - m) + exp(sub[2] - m);
        pt = (exp(sub[0] - m) + exp(sub[1] - m)) / zz;
        CHECK_NEAR(zt_result_decision(&r, 0, 0)->score, pt, 2e-3);
    }

    zt_result_free(&r);
    zt_session_destroy(ss);
    free(head); free(hid); free(hid16);
}

/* ------------------------------------------------- aliases, abstain, errors */

static void test_aliases_and_abstain(zt_engine *e)
{
    zt_schema *s = zt_schema_new();
    zt_program *p = NULL;
    zt_session *ss;
    zt_result r;
    zt_tensor in;
    char err[256];
    const uint32_t V = 16;
    float logits[16];
    const int32_t yes[3] = { 1, 2, 3 }, no[1] = { 4 };
    int f;

    section("alias merging and abstain");
    f = zt_schema_add_field(s, "flag", ZT_KIND_BOOL);
    zt_schema_add_choice(s, f, "yes", yes, 3, 0.0f);
    zt_schema_add_choice(s, f, "no",  no,  1, 0.0f);
    zt_schema_set_field_params(s, f, 1.0f, 0.5f, 0.9f, -INFINITY);   /* abstain below 0.9 */
    CHECK(zt_compile(s, &p, err, sizeof err) == ZT_OK);
    zt_schema_free(s);
    if (!p) return;
    CHECK(zt_program_n_cols(p) == 4);

    memset(logits, 0, sizeof logits);
    logits[1] = 1.0f; logits[2] = 1.0f; logits[3] = 1.0f; logits[4] = 1.0f;   /* 3 aliases vs 1 */

    ss = zt_session_create(e, p, 1, err, sizeof err);
    CHECK(ss != NULL);
    CHECK(zt_result_alloc(&r, p, 1, 0) == ZT_OK);
    memset(&in, 0, sizeof in);
    in.data = logits; in.rows = 1; in.cols = V; in.ld = V; in.dtype = ZT_DTYPE_F32; in.mem = ZT_MEM_HOST;
    CHECK(zt_session_run(ss, ZT_INPUT_LOGITS, &in, &r) == ZT_OK);
    /* alias mass: P(true) = 3/4 -> selected, but 0.75 < min_conf 0.9 -> abstain */
    CHECK_NEAR(zt_result_decision(&r, 0, 0)->score, 0.75, 1e-5);
    CHECK(zt_result_decision(&r, 0, 0)->choice == ZT_ABSTAIN);
    CHECK(zt_result_decision(&r, 0, 0)->flags & ZT_FLAG_LOW_CONF);
    CHECK(zt_fsm_walk(p, &r, 0) == ZT_ROUTE_ABSTAIN);

    zt_result_free(&r);
    zt_session_destroy(ss);
    zt_program_free(p);
}

static void test_schema_errors(void)
{
    zt_schema *s;
    zt_program *p = NULL;
    char err[256];
    const int32_t a[2] = { 5, 6 }, b[2] = { 6, 7 }, c[1] = { 9 };

    section("schema validation");
    s = zt_schema_new();
    CHECK(zt_schema_add_field(s, "9bad", ZT_KIND_BOOL) < 0);
    CHECK(zt_schema_add_field(s, "ok_name", ZT_KIND_BOOL) == 0);
    CHECK(zt_schema_add_field(s, "ok_name", ZT_KIND_BOOL) < 0);            /* duplicate id */
    zt_schema_add_choice(s, 0, "yes", a, 2, 0.0f);
    zt_schema_add_choice(s, 0, "no",  b, 2, 0.0f);                          /* token 6 shared */
    CHECK(zt_compile(s, &p, err, sizeof err) == ZT_ERR_AMBIGUOUS);
    zt_schema_free(s);

    s = zt_schema_new();
    CHECK(zt_schema_add_field(s, "solo", ZT_KIND_BOOL) == 0);
    zt_schema_add_choice(s, 0, "only", c, 1, 0.0f);
    CHECK(zt_compile(s, &p, err, sizeof err) == ZT_ERR_BAD_SCHEMA);         /* BOOL needs 2 */
    CHECK(zt_schema_set_field_params(s, 0, 0.0f, 0.5f, 0.0f, 0.0f) == ZT_ERR_INVALID_ARG);
    zt_schema_free(s);

    s = zt_schema_new();
    zt_schema_add_field(s, "first",  ZT_KIND_BOOL);
    zt_schema_add_field(s, "second", ZT_KIND_BOOL);
    CHECK(zt_schema_set_dependency(s, 0, 1, 0) == ZT_ERR_BAD_SCHEMA);       /* backward dep */
    zt_schema_free(s);
}

/* ------------------------------------------------------------- emitters */

static void test_emit(zt_engine *e, const zt_program *p)
{
    const uint32_t V = 64, F = 3;
    float *logits = (float *)calloc((size_t)F * V, sizeof(float));
    zt_session *ss;
    zt_result r;
    zt_tensor in;
    char err[256], buf[4096];
    size_t len = 0;

    section("emitters");
    logits[0 * V + 12] = 5.0f;                       /* is_refund = no  */
    logits[1 * V + 21] = 5.0f;                       /* category = shipping */
    logits[2 * V + 32] = 6.0f;                       /* urgency high    */

    ss = zt_session_create(e, p, 1, err, sizeof err);
    CHECK(zt_result_alloc(&r, p, 1, 0) == ZT_OK);
    memset(&in, 0, sizeof in);
    in.data = logits; in.rows = F; in.cols = V; in.ld = V; in.dtype = ZT_DTYPE_F32; in.mem = ZT_MEM_HOST;
    CHECK(zt_session_run(ss, ZT_INPUT_LOGITS, &in, &r) == ZT_OK);

    CHECK(zt_emit_json(p, &r, 0, 0, buf, sizeof buf, &len) == ZT_OK);
    CHECK(strstr(buf, "\"is_refund\":false") != NULL);
    CHECK(strstr(buf, "\"category\":\"shipping\"") != NULL);
    CHECK(strstr(buf, "\"urgency\":") != NULL);
    CHECK(len == strlen(buf));
    printf("  json: %s\n", buf);

    /* short buffer reports the exact required length and still NUL-terminates */
    {
        char small[8];
        size_t need = 0;
        CHECK(zt_emit_json(p, &r, 0, 0, small, sizeof small, &need) == ZT_ERR_BUFFER_TOO_SMALL);
        CHECK(need == len);
        CHECK(small[sizeof small - 1] == '\0');
    }

    CHECK(zt_emit_json(p, &r, 0, ZT_EMIT_META, buf, sizeof buf, &len) == ZT_OK);
    CHECK(strstr(buf, "\"status\":\"ok\"") != NULL);
    CHECK(strstr(buf, "\"confidence\":") != NULL);

    CHECK(zt_emit_xml(p, &r, 0, 0, buf, sizeof buf, &len) == ZT_OK);
    CHECK(strstr(buf, "<decision>") != NULL);
    CHECK(strstr(buf, "<category kind=\"choice\" status=\"ok\">shipping</category>") != NULL);
    printf("  xml:  %s\n", buf);

    CHECK(zt_emit_json_schema(p, 0, buf, sizeof buf, &len) == ZT_OK);
    CHECK(strstr(buf, "\"additionalProperties\":false") != NULL);
    CHECK(strstr(buf, "\"enum\":[\"billing\",\"shipping\",\"product\",\"other\",null]") != NULL);
    CHECK(strstr(buf, "\"minimum\":1,\"maximum\":3") != NULL);

    /* a skipped field emits null, never a fabricated label */
    logits[0 * V + 12] = 0.0f; logits[0 * V + 10] = 5.0f;   /* is_refund = yes -> category skipped */
    CHECK(zt_session_run(ss, ZT_INPUT_LOGITS, &in, &r) == ZT_OK);
    CHECK(zt_emit_json(p, &r, 0, 0, buf, sizeof buf, &len) == ZT_OK);
    CHECK(strstr(buf, "\"category\":null") != NULL);

    zt_result_free(&r);
    zt_session_destroy(ss);
    free(logits);
}

/* ---------------------------------------------------------- calibration */

static void test_calibration(void)
{
    const uint32_t n = 2000, K = 4;
    float *logits = (float *)malloc((size_t)n * K * sizeof(float));
    int32_t *labels = (int32_t *)malloc(n * sizeof(int32_t));
    float T = 0.0f;
    unsigned seed = 999u;
    uint32_t i, k;
    zt_knot knots[16];
    uint32_t nk = 0;
    float p[64];
    uint8_t ok[64];

    section("calibration");
    /* Generate labels from softmax(z / 2) but present the *unscaled* z, so the
     * fitted temperature should recover roughly 2. */
    for (i = 0; i < n; i++) {
        double probs[4], zz = 0.0, u;
        double m = -1e30;
        for (k = 0; k < K; k++) {
            seed = seed * 1103515245u + 12345u;
            logits[i * K + k] = ((float)((seed >> 16) & 0x7fffu) / 16384.0f - 1.0f) * 4.0f;
            if (logits[i * K + k] / 2.0 > m) m = logits[i * K + k] / 2.0;
        }
        for (k = 0; k < K; k++) { probs[k] = exp(logits[i * K + k] / 2.0 - m); zz += probs[k]; }
        seed = seed * 1103515245u + 12345u;
        u = (double)((seed >> 16) & 0x7fffu) / 32768.0;
        labels[i] = (int32_t)(K - 1);
        for (k = 0; k < K; k++) { u -= probs[k] / zz; if (u <= 0.0) { labels[i] = (int32_t)k; break; } }
    }
    CHECK(zt_calib_fit_temperature(logits, labels, n, K, &T) == ZT_OK);
    printf("  fitted temperature: %.3f (target ~2)\n", (double)T);
    CHECK(T > 1.3f && T < 3.2f);

    for (i = 0; i < 40; i++) { p[i] = 0.5f + 0.0125f * (float)i; ok[i] = (uint8_t)(i % 5 == 0 ? 0 : 1); }
    CHECK(zt_calib_fit_bins(p, ok, 40, 8, knots, &nk) == ZT_OK);
    CHECK(nk >= 2);
    for (i = 1; i < nk; i++) { CHECK(knots[i].x > knots[i - 1].x); CHECK(knots[i].y >= knots[i - 1].y); }

    free(logits); free(labels);
}

static void test_f16(void)
{
    section("fp16 round trips");
    CHECK(zt_f32_to_f16(1.0f) == 0x3c00u);
    CHECK(zt_f32_to_f16(0.5f) == 0x3800u);
    CHECK(zt_f32_to_f16(-2.0f) == 0xc000u);
    CHECK(zt_f32_to_f16(65504.0f) == 0x7bffu);
    CHECK(zt_f32_to_f16(ldexpf(1.0f, -24)) == 0x0001u);   /* smallest subnormal */
    CHECK_NEAR(zt_f16_to_f32(0x3c00u), 1.0, 0.0);
    CHECK_NEAR(zt_f16_to_f32(0x7bffu), 65504.0, 0.0);
    CHECK_NEAR(zt_f16_to_f32(zt_f32_to_f16(0.1f)), 0.1, 1e-3);
}

/* ----------------------------------------------------- GPU vs CPU check */

static void test_gpu_matches_cpu(const zt_program *p)
{
    zt_engine *cpu, *gpu = NULL;
    zt_backend b = ZT_BACKEND_CUDA;
    char err[256];
    const uint32_t V = 128, F = 3, B = 8, rows = B * F;
    float *logits;
    zt_session *sc, *sg;
    zt_result rc, rg;
    zt_tensor in;
    uint32_t i, f, bi;
    unsigned seed = 7u;

    if (!zt_backend_available(ZT_BACKEND_CUDA)) {
        b = ZT_BACKEND_METAL;
        if (!zt_backend_available(ZT_BACKEND_METAL)) {
            section("gpu cross-check");
            printf("  skipped: no GPU backend available in this build\n");
            return;
        }
    }
    printf("[gpu cross-check: %s]\n", zt_backend_name(b));
    cpu = zt_engine_create(ZT_BACKEND_CPU, err, sizeof err);
    gpu = zt_engine_create(b, err, sizeof err);
    CHECK(cpu && gpu);
    if (!cpu || !gpu) { zt_engine_destroy(cpu); zt_engine_destroy(gpu); return; }

    logits = (float *)calloc((size_t)rows * V, sizeof(float));
    for (i = 0; i < rows * V; i++) { seed = seed * 1103515245u + 12345u; logits[i] = ((float)((seed >> 16) & 0x7fffu) / 16384.0f - 1.0f) * 3.0f; }

    sc = zt_session_create(cpu, p, B, err, sizeof err);
    sg = zt_session_create(gpu, p, B, err, sizeof err);
    CHECK(zt_result_alloc(&rc, p, B, 1) == ZT_OK);
    CHECK(zt_result_alloc(&rg, p, B, 1) == ZT_OK);
    memset(&in, 0, sizeof in);
    in.data = logits; in.rows = rows; in.cols = V; in.ld = V; in.dtype = ZT_DTYPE_F32; in.mem = ZT_MEM_HOST;
    CHECK(zt_session_run(sc, ZT_INPUT_LOGITS, &in, &rc) == ZT_OK);
    CHECK(zt_session_run(sg, ZT_INPUT_LOGITS, &in, &rg) == ZT_OK);

    for (bi = 0; bi < B; bi++) {
        for (f = 0; f < zt_program_n_fields(p); f++) {
            const zt_decision *a = zt_result_decision(&rc, bi, f);
            const zt_decision *g = zt_result_decision(&rg, bi, f);
            CHECK_NEAR(g->p_top, a->p_top, 5e-3);
            CHECK_NEAR(g->confidence, a->confidence, 5e-3);
            /* choices may legitimately differ only when the CPU margin is tiny */
            if (a->margin > 0.1f) CHECK(g->choice == a->choice);
        }
    }
    zt_result_free(&rc); zt_result_free(&rg);
    zt_session_destroy(sc); zt_session_destroy(sg);
    zt_engine_destroy(cpu); zt_engine_destroy(gpu);
    free(logits);
}

/* ------------------------------------------------------------------ main */

int main(void)
{
    zt_engine *e;
    zt_program *p;
    char err[256];

    printf("zero-token %s — tests\n", zt_version());
    p = build_program();
    if (!p) { printf("could not build the reference program\n"); return 1; }

    e = zt_engine_create(ZT_BACKEND_AUTO, err, sizeof err);
    if (!e) { printf("engine: %s\n", err); zt_program_free(p); return 1; }
    printf("backend: %s\n", zt_backend_name(zt_engine_backend(e)));

    test_logits(e, p);
    test_hidden(e, p, ZT_DTYPE_F32,  "f32");
    test_hidden(e, p, ZT_DTYPE_F16,  "f16");
    test_hidden(e, p, ZT_DTYPE_BF16, "bf16");
    test_aliases_and_abstain(e);
    test_schema_errors();
    test_emit(e, p);
    test_calibration();
    test_f16();
    zt_engine_destroy(e);
    test_gpu_matches_cpu(p);

    zt_program_free(p);
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
