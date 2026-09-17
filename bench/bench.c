/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-019
   File:              bench.c
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

   Clone-Gate: sha256:1980cdb8e67676dbb8f7947a531d28c004e3563245d4f631873db7eb40fcfb84

   See: LICENSE
   ======================================================================== */
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "zt.h"

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static zt_program *make_program(uint32_t n_choices)
{
    zt_schema *s = zt_schema_new();
    zt_program *p = NULL;
    char err[256];
    int32_t tok[4];
    uint32_t c;
    int f = zt_schema_add_field(s, "route", ZT_KIND_CHOICE);
    for (c = 0; c < n_choices; c++) {
        char label[32];
        snprintf(label, sizeof label, "c%u", c);
        tok[0] = (int32_t)(100 + c * 2);
        tok[1] = (int32_t)(101 + c * 2);
        zt_schema_add_choice(s, f, label, tok, 2, (float)c);
    }
    if (zt_compile(s, &p, err, sizeof err) != ZT_OK) { fprintf(stderr, "compile: %s\n", err); p = NULL; }
    zt_schema_free(s);
    return p;
}

static void run_case(zt_backend backend, const zt_program *p, uint32_t rows, uint32_t H,
                     uint32_t V, uint32_t iters, int hidden)
{
    char err[256];
    zt_engine *e = zt_engine_create(backend, err, sizeof err);
    zt_session *s;
    zt_result r;
    zt_tensor in, head;
    float *data, *headw = NULL;
    double t0, t1;
    uint32_t i;
    unsigned seed = 4242u;
    uint32_t cols = hidden ? H : V;

    if (!e) { printf("  %-6s : unavailable (%s)\n", zt_backend_name(backend), err); return; }
    s = zt_session_create(e, p, rows, err, sizeof err);
    if (!s) { printf("  %-6s : session failed (%s)\n", zt_backend_name(backend), err); zt_engine_destroy(e); return; }

    data = (float *)malloc((size_t)rows * cols * sizeof(float));
    for (i = 0; i < rows * cols; i++) { seed = seed * 1103515245u + 12345u; data[i] = (float)((seed >> 16) & 0x7fffu) / 16384.0f - 1.0f; }
    memset(&in, 0, sizeof in);
    in.data = data; in.rows = rows; in.cols = cols; in.ld = cols;
    in.dtype = ZT_DTYPE_F32; in.mem = ZT_MEM_HOST;

    if (hidden) {
        headw = (float *)malloc((size_t)V * H * sizeof(float));
        for (i = 0; i < V * H; i++) { seed = seed * 1103515245u + 12345u; headw[i] = ((float)((seed >> 16) & 0x7fffu) / 16384.0f - 1.0f) * 0.1f; }
        memset(&head, 0, sizeof head);
        head.data = headw; head.rows = V; head.cols = H; head.ld = H;
        head.dtype = ZT_DTYPE_F32; head.mem = ZT_MEM_HOST;
        if (zt_session_bind_head(s, &head) != ZT_OK) {
            printf("  %-6s : bind_head failed (%s)\n", zt_backend_name(backend), zt_session_error(s));
            free(data); free(headw); zt_session_destroy(s); zt_engine_destroy(e);
            return;
        }
    }
    zt_result_alloc(&r, p, rows, 0);

    if (zt_session_run(s, hidden ? ZT_INPUT_HIDDEN : ZT_INPUT_LOGITS, &in, &r) != ZT_OK) {
        printf("  %-6s : run failed (%s)\n", zt_backend_name(backend), zt_session_error(s));
        goto done;
    }
    t0 = now_us();
    for (i = 0; i < iters; i++) (void)zt_session_run(s, hidden ? ZT_INPUT_HIDDEN : ZT_INPUT_LOGITS, &in, &r);
    t1 = now_us();
    printf("  %-6s : %8.1f us/run  (%.2f us/row)\n", zt_backend_name(zt_engine_backend(e)),
           (t1 - t0) / iters, (t1 - t0) / iters / rows);

done:
    zt_result_free(&r);
    free(data); free(headw);
    zt_session_destroy(s);
    zt_engine_destroy(e);
}

int main(int argc, char **argv)
{
    uint32_t rows  = (argc > 1) ? (uint32_t)atoi(argv[1]) : 256;
    uint32_t H     = (argc > 2) ? (uint32_t)atoi(argv[2]) : 4096;
    uint32_t K     = (argc > 3) ? (uint32_t)atoi(argv[3]) : 4;
    uint32_t iters = (argc > 4) ? (uint32_t)atoi(argv[4]) : 50;
    const uint32_t V = 32000;
    zt_program *p;

    if (rows == 0 || H == 0 || K < 2 || K > 4 || iters == 0) {
        fprintf(stderr, "usage: bench [rows] [H] [choices 2..4] [iters]\n");
        return 1;
    }
    p = make_program(K);
    if (!p) return 1;

    printf("zero-token %s bench: rows=%u H=%u choices=%u (K=%u columns) V=%u iters=%u\n",
           zt_version(), rows, H, K, zt_program_n_cols(p), V, iters);

    printf("logits path (rows x %u):\n", V);
    run_case(ZT_BACKEND_AUTO, p, rows, H, V, iters, 0);
    run_case(ZT_BACKEND_CPU,  p, rows, H, V, iters, 0);

    printf("hidden path (rows x %u against a %u-row head slice):\n", H, zt_program_n_cols(p));
    run_case(ZT_BACKEND_AUTO, p, rows, H, V, iters, 1);
    run_case(ZT_BACKEND_CPU,  p, rows, H, V, iters, 1);

    zt_program_free(p);
    return 0;
}
