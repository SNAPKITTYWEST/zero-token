/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-004
   File:              zt_internal.h
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

   Clone-Gate: sha256:6e732081916a56b63833151739407b6c6f5bb4a2e8c3137aa2c9505243cf4a1f

   See: LICENSE

   -------------------------------------------------------------------------
   zt_internal.h — private declarations shared by the host sources.
   Included from the C sources, from nvcc inside extern "C" (src/zt_cuda.cu)
   and from Objective-C (metal/zt_metal.m).  Keep it C89-compatible in syntax:
   no designated initialisers, no VLAs, no _Generic.
   ======================================================================== */
#ifndef ZT_INTERNAL_H
#define ZT_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zt.h"

/* Compiled, immutable program.  All tables are dense and device-ready. */
struct zt_program {
    uint32_t      n_fields;
    uint32_t      n_cols;            /* K: total gathered columns                       */
    uint32_t      n_choices;         /* over all fields                                 */
    uint32_t      c_max;             /* max choices in one field                        */
    uint32_t      n_knots;           /* may be 0; knots always has >= 1 entry           */
    int32_t       max_token;
    zt_field_dev *fields;            /* [n_fields]                                      */
    int32_t      *col_token;         /* [n_cols]      vocabulary id per column          */
    uint32_t     *choice_col_start;  /* [n_choices+1] columns of global choice g are
                                        [choice_col_start[g], choice_col_start[g+1])    */
    float        *choice_value;      /* [n_choices]                                     */
    zt_knot      *knots;             /* [max(n_knots,1)]                                */
    int32_t      *dep_field;         /* [n_fields] -1 = none                            */
    int32_t      *dep_choice;        /* [n_fields]                                      */
    char        **field_id;          /* [n_fields]                                      */
    char        **choice_label;      /* [n_choices]                                     */
};

/* Backend vtable.  `run` receives an already-validated request:
 *   rows = batch * n_fields, kind/cols consistent, out large enough.
 * It must fill out->dec[0 .. rows) and, if out->probs, the probs block. */
typedef struct zt_backend_ops {
    const char *name;
    zt_backend  kind;
    zt_status (*create)(void **ctx, char *err, size_t errcap);
    void      (*destroy)(void *ctx);
    zt_status (*session_create)(void *ctx, const zt_program *p, uint32_t max_batch,
                                void **sess, char *err, size_t errcap);
    void      (*session_destroy)(void *sess);
    zt_status (*bind_head)(void *sess, const zt_tensor *head, char *err, size_t errcap);
    zt_status (*run)(void *sess, zt_input_kind kind, const zt_tensor *in, uint32_t batch,
                     zt_result *out, char *err, size_t errcap);
} zt_backend_ops;

extern const zt_backend_ops zt_cpu_ops;
#if defined(ZT_HAVE_CUDA)
extern const zt_backend_ops zt_cuda_ops;
#endif
#if defined(ZT_HAVE_METAL)
extern const zt_backend_ops zt_metal_ops;
#endif

/* Gather the K head rows named by col_token from a host-memory lm_head into a
 * dense fp16 [Kpad][H] block (rows K..Kpad zero-filled).  Used by every
 * backend when the head lives in host memory. */
void zt_host_gather_head_f16(const zt_tensor *head, const int32_t *col_token,
                             uint32_t K, uint32_t Kpad, uint32_t H, uint16_t *out);

/* Scalar reference decision for one row.  x points at the row's K sub-logits. */
void zt_decide_row_ref(const zt_program *p, const float *x, uint32_t field,
                       zt_decision *d, float *probs /* c_max or NULL */);

static inline uint32_t zt_round_up(uint32_t v, uint32_t m) { return (v + m - 1u) / m * m; }

#define ZT_SETERR(err, cap, ...) \
    do { if ((err) && (cap) > 0) snprintf((err), (cap), __VA_ARGS__); } while (0)

#endif /* ZT_INTERNAL_H */
