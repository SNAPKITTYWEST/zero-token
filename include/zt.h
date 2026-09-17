/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-001
   File:              zt.h
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

   Clone-Gate: sha256:a0591fb614abc0c0930b581da40497b7243995f2b6280f6518198db5dc8b212b

   See: LICENSE

   -------------------------------------------------------------------------
   zt.h — public C99 API of the zero-token decision & routing engine.

   The engine never generates text.  Given, for every (batch item, field), the
   host model's *next-token* distribution — either as full-vocabulary logits or
   as the final hidden state plus a bound lm_head — it produces, in one pass,
   a structured decision per field:

     BOOL    two choices, thresholded on the calibrated P(true)
     CHOICE  categorical argmax over N labelled choices
     SCORE   ordinal expectation over N choices that carry numeric values

   plus calibrated confidence, raw probabilities, log-margin and entropy, an
   abstain floor, and forward-only dependency guards that form a deterministic
   finite-state machine over the fields.  Output is emitted *by construction*
   from the compiled program (JSON / XML / JSON-Schema), so a value can never
   have the wrong type or an unknown label.

   Row layout (the one convention you must follow):
     input rows = batch * n_fields;  row r = (b = r / F, f = r % F)
     i.e. each batch item contributes one next-token distribution per field,
     in field order.  Results are indexed the same way: dec[b * F + f].

   Thread safety: a zt_engine may be shared; a zt_session must not be used by
   two threads at once.  A zt_program is immutable after zt_compile and must
   outlive every session and result built from it.
   ======================================================================== */
#ifndef ZT_H
#define ZT_H

#include <stddef.h>
#include <stdint.h>
#include "zt_core.h"   /* zt_decision, zt_knot, ZT_KIND_*, ZT_DTYPE_*, ZT_ABSTAIN, ZT_SKIPPED */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define ZT_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define ZT_API __attribute__((visibility("default")))
#else
#  define ZT_API
#endif

#define ZT_VERSION_MAJOR 0
#define ZT_VERSION_MINOR 1
#define ZT_VERSION_PATCH 0

#define ZT_MAX_FIELDS          256u
#define ZT_MAX_COLS_PER_FIELD  1024u
#define ZT_MAX_ID_LEN          64u

/* ------------------------------------------------------------------ status */
typedef enum {
    ZT_OK                   = 0,
    ZT_ERR_INVALID_ARG      = 1,
    ZT_ERR_NO_MEMORY        = 2,
    ZT_ERR_BAD_SCHEMA       = 3,   /* schema failed validation (see error text) */
    ZT_ERR_AMBIGUOUS        = 4,   /* one token appears in two choices of one field */
    ZT_ERR_SHAPE            = 5,   /* tensor shape does not match the program */
    ZT_ERR_BACKEND          = 6,   /* device / driver failure (see zt_session_error) */
    ZT_ERR_UNAVAILABLE      = 7,   /* backend not compiled in or no device found */
    ZT_ERR_BUFFER_TOO_SMALL = 8,
    ZT_ERR_NOT_BOUND        = 9    /* hidden-state input without a bound lm_head */
} zt_status;

/* ----------------------------------------------------------------- backends */
typedef enum {
    ZT_BACKEND_AUTO  = 0,   /* CUDA, then Metal, then CPU */
    ZT_BACKEND_CPU   = 1,   /* C99 reference; always available */
    ZT_BACKEND_CUDA  = 2,   /* NVIDIA, WMMA tensor cores when present (SIMT fallback) */
    ZT_BACKEND_METAL = 3    /* Apple Silicon, simdgroup matrices (SIMT fallback) */
} zt_backend;

/* ------------------------------------------------------------------ tensors */
typedef enum {
    ZT_MEM_HOST   = 0,   /* data is a host pointer (staged to the device once per call) */
    ZT_MEM_DEVICE = 1    /* CUDA: device pointer.  Metal: id<MTLBuffer> passed as void*.
                            Zero-copy: the engine reads it in place. */
} zt_mem;

typedef enum {
    ZT_INPUT_LOGITS = 0,   /* rows x V   full-vocabulary next-token logits          */
    ZT_INPUT_HIDDEN = 1    /* rows x H   final hidden state; needs zt_session_bind_head */
} zt_input_kind;

/* A strided row-major 2-D view.  Never owns memory. */
typedef struct {
    const void *data;     /* see zt_mem                                          */
    size_t      offset;   /* byte offset into data (Metal buffer offset; else added to the pointer) */
    uint32_t    rows;
    uint32_t    cols;
    uint32_t    ld;       /* leading dimension in elements, >= cols               */
    uint32_t    dtype;    /* ZT_DTYPE_F32 / ZT_DTYPE_F16 / ZT_DTYPE_BF16          */
    uint32_t    mem;      /* ZT_MEM_HOST / ZT_MEM_DEVICE                          */
} zt_tensor;

/* ------------------------------------------------------------------ results */
typedef struct {
    uint32_t     batch;       /* items written by the last run                  */
    uint32_t     cap_batch;
    uint32_t     n_fields;
    uint32_t     c_max;       /* max choices over all fields (probs stride)     */
    zt_decision *dec;         /* [cap_batch * n_fields]                         */
    float       *probs;       /* [cap_batch * n_fields * c_max] or NULL         */
} zt_result;

typedef enum {
    ZT_ROUTE_COMPLETE = 0,   /* every evaluated field produced a choice        */
    ZT_ROUTE_ABSTAIN  = 1    /* at least one evaluated field abstained         */
} zt_route;

/* Emitter flags */
#define ZT_EMIT_META   1u   /* per-field objects with confidence, margin, entropy, status */
#define ZT_EMIT_PRETTY 2u   /* newlines + indentation                                    */

/* ------------------------------------------------------------ opaque types */
typedef struct zt_schema  zt_schema;    /* mutable builder                         */
typedef struct zt_program zt_program;   /* immutable compiled tables               */
typedef struct zt_engine  zt_engine;    /* one backend + device                    */
typedef struct zt_session zt_session;   /* program bound to an engine + workspace  */

/* ------------------------------------------------------------------ schema */
ZT_API zt_schema *zt_schema_new(void);
ZT_API void       zt_schema_free(zt_schema *s);

/* Returns the new field's index (>= 0) or a negative zt_status.
 * id: [A-Za-z_][A-Za-z0-9_]{0,63}, unique.  kind: ZT_KIND_BOOL / CHOICE / SCORE.
 * Defaults: temperature 1, threshold 0.5, min_confidence 0, min_margin -inf. */
ZT_API int zt_schema_add_field(zt_schema *s, const char *id, uint32_t kind);

ZT_API zt_status zt_schema_set_field_params(zt_schema *s, int field,
                                            float temperature,      /* > 0            */
                                            float threshold,        /* BOOL: [0,1]    */
                                            float min_confidence,   /* abstain floor  */
                                            float min_margin);      /* nats; -inf ok  */

/* Append a choice.  first_tokens are the vocabulary ids whose next-token mass
 * is summed for this choice (aliases: " yes", "Yes", "yes", ...).  value is
 * used by SCORE fields (ordinal value); ignored otherwise.
 * For BOOL fields, choice 0 is the TRUE branch and choice 1 the FALSE branch. */
ZT_API zt_status zt_schema_add_choice(zt_schema *s, int field, const char *label,
                                      const int32_t *first_tokens, uint32_t n_tokens,
                                      float value);

/* Evaluate `field` only if `on_field` (which must come earlier) selected
 * `required_choice`; otherwise the decision is ZT_SKIPPED.  Forward-only, so
 * the dependency graph is a DAG and the FSM walk is a single ordered pass. */
ZT_API zt_status zt_schema_set_dependency(zt_schema *s, int field, int on_field, int required_choice);

/* Monotone piecewise-linear calibration map applied to the selected choice's
 * raw probability (xs strictly increasing in [0,1], ys non-decreasing). */
ZT_API zt_status zt_schema_set_calibration(zt_schema *s, int field,
                                           const float *xs, const float *ys, uint32_t n);

/* Validate + flatten into device-ready tables.  err receives a message on failure. */
ZT_API zt_status zt_compile(const zt_schema *s, zt_program **out, char *err, size_t errcap);
ZT_API void      zt_program_free(zt_program *p);

/* Program introspection (all O(1)). */
ZT_API uint32_t       zt_program_n_fields(const zt_program *p);
ZT_API uint32_t       zt_program_n_cols(const zt_program *p);            /* K: gathered columns */
ZT_API int32_t        zt_program_max_token(const zt_program *p);         /* V must exceed this */
ZT_API uint32_t       zt_program_c_max(const zt_program *p);
ZT_API uint32_t       zt_program_field_kind(const zt_program *p, uint32_t field);
ZT_API const char    *zt_program_field_id(const zt_program *p, uint32_t field);
ZT_API uint32_t       zt_program_n_choices(const zt_program *p, uint32_t field);
ZT_API const char    *zt_program_choice_label(const zt_program *p, uint32_t field, uint32_t choice);
ZT_API float          zt_program_choice_value(const zt_program *p, uint32_t field, uint32_t choice);
ZT_API const int32_t *zt_program_col_tokens(const zt_program *p, uint32_t *n_out);

/* ------------------------------------------------------------------ engine */
ZT_API zt_engine  *zt_engine_create(zt_backend backend, char *err, size_t errcap);
ZT_API void        zt_engine_destroy(zt_engine *e);
ZT_API zt_backend  zt_engine_backend(const zt_engine *e);
ZT_API const char *zt_backend_name(zt_backend b);
ZT_API int         zt_backend_available(zt_backend b);   /* 1 = compiled in and a device answers */

/* ----------------------------------------------------------------- session */
ZT_API zt_session *zt_session_create(zt_engine *e, const zt_program *p, uint32_t max_batch,
                                     char *err, size_t errcap);
ZT_API void        zt_session_destroy(zt_session *s);

/* lm_head: rows = V (>= max_token + 1), cols = H, row-major [V][H], any dtype,
 * host or device memory.  Only the program's K rows are gathered (as fp16). */
ZT_API zt_status   zt_session_bind_head(zt_session *s, const zt_tensor *lm_head);

/* One pass.  input->rows must be batch * n_fields (batch <= max_batch and
 * <= out->cap_batch).  LOGITS: cols = V.  HIDDEN: cols = H of the bound head.
 * Dependency guards are applied on return; out->batch is set. */
ZT_API zt_status   zt_session_run(zt_session *s, zt_input_kind kind, const zt_tensor *input,
                                  zt_result *out);
ZT_API const char *zt_session_error(const zt_session *s);

/* ----------------------------------------------------------------- results */
ZT_API zt_status          zt_result_alloc(zt_result *r, const zt_program *p, uint32_t max_batch, int with_probs);
ZT_API void               zt_result_free(zt_result *r);
ZT_API const zt_decision *zt_result_decision(const zt_result *r, uint32_t b, uint32_t field);
ZT_API const float       *zt_result_probs(const zt_result *r, uint32_t b, uint32_t field); /* n_choices floats or NULL */

/* Apply the dependency guards of batch item b (idempotent; zt_session_run
 * already does this) and classify the route. */
ZT_API zt_route zt_fsm_walk(const zt_program *p, zt_result *r, uint32_t b);

/* ---------------------------------------------------------------- emitters */
/* All emitters write a NUL-terminated string into buf and store the length
 * (excluding NUL) in *len_out; return ZT_ERR_BUFFER_TOO_SMALL with *len_out =
 * required length if cap is insufficient.  Values are produced from the
 * program's own label tables — nothing is parsed from model output. */
ZT_API zt_status zt_emit_json(const zt_program *p, const zt_result *r, uint32_t b, uint32_t flags,
                              char *buf, size_t cap, size_t *len_out);
ZT_API zt_status zt_emit_xml (const zt_program *p, const zt_result *r, uint32_t b, uint32_t flags,
                              char *buf, size_t cap, size_t *len_out);
/* JSON Schema (draft 2020-12) that every zt_emit_json output of this program
 * satisfies, for the same flags (ZT_EMIT_META changes the shape). */
ZT_API zt_status zt_emit_json_schema(const zt_program *p, uint32_t flags,
                                     char *buf, size_t cap, size_t *len_out);

/* ------------------------------------------------------------- calibration */
/* Temperature scaling: minimise NLL of labels under softmax(logits / T).
 * logits: [n][K] host floats; labels: [n] in [0,K).  Writes T > 0. */
ZT_API zt_status zt_calib_fit_temperature(const float *logits, const int32_t *labels,
                                          uint32_t n, uint32_t K, float *temperature_out);
/* Reliability-bin fit: p[i] = predicted prob of the chosen choice, correct[i] =
 * 1 if that choice was right.  Produces <= n_bins monotone knots (PAV-pooled)
 * suitable for zt_schema_set_calibration.  knots must hold n_bins entries. */
ZT_API zt_status zt_calib_fit_bins(const float *p, const uint8_t *correct, uint32_t n,
                                   uint32_t n_bins, zt_knot *knots, uint32_t *n_knots_out);

/* ------------------------------------------------------------------- utils */
ZT_API uint16_t    zt_f32_to_f16(float f);
ZT_API float       zt_f16_to_f32(uint16_t h);
ZT_API const char *zt_status_str(zt_status s);
ZT_API const char *zt_version(void);

#ifdef __cplusplus
}
#endif
#endif /* ZT_H */
