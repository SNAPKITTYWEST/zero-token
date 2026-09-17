/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-006
   File:              zt_schema.c
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

   Clone-Gate: sha256:72757b7434a006f91d53d3707c262054230d59e7ee147df7ab96363925a77dc5

   See: LICENSE
   ======================================================================== */
#include <math.h>
#include "zt_internal.h"

typedef struct {
    char    *label;
    int32_t *tokens;
    uint32_t n_tokens;
    float    value;
} sch_choice;

typedef struct {
    char       *id;
    uint32_t    kind;
    float       temperature, threshold, min_conf, min_margin;
    sch_choice *choices;
    uint32_t    n_choices, cap_choices;
    int32_t     dep_field, dep_choice;
    float      *cal_x, *cal_y;
    uint32_t    n_cal;
} sch_field;

struct zt_schema {
    sch_field *fields;
    uint32_t   n_fields, cap_fields;
};

static char *zt_strdup(const char *s)
{
    size_t n = strlen(s) + 1;
    char *d = (char *)malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

zt_schema *zt_schema_new(void)
{
    zt_schema *s = (zt_schema *)calloc(1, sizeof *s);
    return s;
}

static void free_field(sch_field *f)
{
    uint32_t c;
    for (c = 0; c < f->n_choices; c++) {
        free(f->choices[c].label);
        free(f->choices[c].tokens);
    }
    free(f->choices);
    free(f->id);
    free(f->cal_x);
    free(f->cal_y);
}

void zt_schema_free(zt_schema *s)
{
    uint32_t i;
    if (!s) return;
    for (i = 0; i < s->n_fields; i++) free_field(&s->fields[i]);
    free(s->fields);
    free(s);
}

static int valid_ident(const char *id)
{
    size_t i, n;
    if (!id || !*id) return 0;
    n = strlen(id);
    if (n > ZT_MAX_ID_LEN) return 0;
    for (i = 0; i < n; i++) {
        char c = id[i];
        int alpha = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
        int digit = (c >= '0' && c <= '9');
        if (i == 0 ? !alpha : !(alpha || digit)) return 0;
    }
    return 1;
}

int zt_schema_add_field(zt_schema *s, const char *id, uint32_t kind)
{
    sch_field *f;
    uint32_t i;
    if (!s || !valid_ident(id)) return -(int)ZT_ERR_INVALID_ARG;
    if (kind > ZT_KIND_SCORE) return -(int)ZT_ERR_INVALID_ARG;
    if (s->n_fields >= ZT_MAX_FIELDS) return -(int)ZT_ERR_BAD_SCHEMA;
    for (i = 0; i < s->n_fields; i++)
        if (strcmp(s->fields[i].id, id) == 0) return -(int)ZT_ERR_BAD_SCHEMA;
    if (s->n_fields == s->cap_fields) {
        uint32_t nc = s->cap_fields ? s->cap_fields * 2 : 8;
        sch_field *nf = (sch_field *)realloc(s->fields, nc * sizeof *nf);
        if (!nf) return -(int)ZT_ERR_NO_MEMORY;
        s->fields = nf;
        s->cap_fields = nc;
    }
    f = &s->fields[s->n_fields];
    memset(f, 0, sizeof *f);
    f->id = zt_strdup(id);
    if (!f->id) return -(int)ZT_ERR_NO_MEMORY;
    f->kind        = kind;
    f->temperature = 1.0f;
    f->threshold   = 0.5f;
    f->min_conf    = 0.0f;
    f->min_margin  = -INFINITY;
    f->dep_field   = -1;
    f->dep_choice  = -1;
    return (int)s->n_fields++;
}

static sch_field *get_field(zt_schema *s, int field)
{
    if (!s || field < 0 || (uint32_t)field >= s->n_fields) return NULL;
    return &s->fields[field];
}

zt_status zt_schema_set_field_params(zt_schema *s, int field, float temperature, float threshold,
                                     float min_confidence, float min_margin)
{
    sch_field *f = get_field(s, field);
    if (!f) return ZT_ERR_INVALID_ARG;
    if (!(temperature > 0.0f) || temperature != temperature) return ZT_ERR_INVALID_ARG;
    if (!(threshold >= 0.0f && threshold <= 1.0f)) return ZT_ERR_INVALID_ARG;
    if (!(min_confidence >= 0.0f && min_confidence <= 1.0f)) return ZT_ERR_INVALID_ARG;
    if (min_margin != min_margin) return ZT_ERR_INVALID_ARG;   /* NaN */
    f->temperature = temperature;
    f->threshold   = threshold;
    f->min_conf    = min_confidence;
    f->min_margin  = min_margin;
    return ZT_OK;
}

zt_status zt_schema_add_choice(zt_schema *s, int field, const char *label,
                               const int32_t *first_tokens, uint32_t n_tokens, float value)
{
    sch_field  *f = get_field(s, field);
    sch_choice *c;
    uint32_t i, j, n = 0;
    int32_t *toks;
    if (!f || !label || !*label || !first_tokens || n_tokens == 0) return ZT_ERR_INVALID_ARG;
    if (strlen(label) > 256) return ZT_ERR_INVALID_ARG;
    if (value != value) return ZT_ERR_INVALID_ARG;
    for (i = 0; i < n_tokens; i++) if (first_tokens[i] < 0) return ZT_ERR_INVALID_ARG;
    for (i = 0; i < f->n_choices; i++)
        if (strcmp(f->choices[i].label, label) == 0) return ZT_ERR_BAD_SCHEMA;
    if (f->n_choices >= ZT_MAX_COLS_PER_FIELD) return ZT_ERR_BAD_SCHEMA;
    if (f->n_choices == f->cap_choices) {
        uint32_t nc = f->cap_choices ? f->cap_choices * 2 : 4;
        sch_choice *nch = (sch_choice *)realloc(f->choices, nc * sizeof *nch);
        if (!nch) return ZT_ERR_NO_MEMORY;
        f->choices = nch;
        f->cap_choices = nc;
    }
    toks = (int32_t *)malloc(n_tokens * sizeof *toks);
    if (!toks) return ZT_ERR_NO_MEMORY;
    /* dedupe within the choice, preserving order */
    for (i = 0; i < n_tokens; i++) {
        int dup = 0;
        for (j = 0; j < n; j++) if (toks[j] == first_tokens[i]) { dup = 1; break; }
        if (!dup) toks[n++] = first_tokens[i];
    }
    c = &f->choices[f->n_choices];
    c->label = zt_strdup(label);
    if (!c->label) { free(toks); return ZT_ERR_NO_MEMORY; }
    c->tokens   = toks;
    c->n_tokens = n;
    c->value    = value;
    f->n_choices++;
    return ZT_OK;
}

zt_status zt_schema_set_dependency(zt_schema *s, int field, int on_field, int required_choice)
{
    sch_field *f = get_field(s, field);
    sch_field *g = get_field(s, on_field);
    if (!f || !g) return ZT_ERR_INVALID_ARG;
    if (on_field >= field) return ZT_ERR_BAD_SCHEMA;          /* forward-only */
    if (required_choice < 0) return ZT_ERR_INVALID_ARG;         /* range checked at compile */
    f->dep_field  = on_field;
    f->dep_choice = required_choice;
    return ZT_OK;
}

zt_status zt_schema_set_calibration(zt_schema *s, int field, const float *xs, const float *ys, uint32_t n)
{
    sch_field *f = get_field(s, field);
    float *nx, *ny;
    uint32_t i;
    if (!f) return ZT_ERR_INVALID_ARG;
    if (n == 0) { free(f->cal_x); free(f->cal_y); f->cal_x = f->cal_y = NULL; f->n_cal = 0; return ZT_OK; }
    if (!xs || !ys || n < 2 || n > 4096) return ZT_ERR_INVALID_ARG;
    for (i = 0; i < n; i++) {
        if (!(xs[i] >= 0.0f && xs[i] <= 1.0f) || !(ys[i] >= 0.0f && ys[i] <= 1.0f)) return ZT_ERR_INVALID_ARG;
        if (i > 0 && !(xs[i] > xs[i - 1])) return ZT_ERR_BAD_SCHEMA;    /* strictly increasing */
        if (i > 0 && ys[i] < ys[i - 1]) return ZT_ERR_BAD_SCHEMA;       /* monotone */
    }
    nx = (float *)malloc(n * sizeof *nx);
    ny = (float *)malloc(n * sizeof *ny);
    if (!nx || !ny) { free(nx); free(ny); return ZT_ERR_NO_MEMORY; }
    memcpy(nx, xs, n * sizeof *nx);
    memcpy(ny, ys, n * sizeof *ny);
    free(f->cal_x); free(f->cal_y);
    f->cal_x = nx; f->cal_y = ny; f->n_cal = n;
    return ZT_OK;
}

/* ------------------------------------------------------------------ compile */

void zt_program_free(zt_program *p)
{
    uint32_t i;
    if (!p) return;
    if (p->field_id)     for (i = 0; i < p->n_fields;  i++) free(p->field_id[i]);
    if (p->choice_label) for (i = 0; i < p->n_choices; i++) free(p->choice_label[i]);
    free(p->fields); free(p->col_token); free(p->choice_col_start); free(p->choice_value);
    free(p->knots); free(p->dep_field); free(p->dep_choice); free(p->field_id); free(p->choice_label);
    free(p);
}

zt_status zt_compile(const zt_schema *s, zt_program **out, char *err, size_t errcap)
{
    zt_program *p;
    uint32_t i, c, t, n_cols = 0, n_choices = 0, n_knots = 0, c_max = 0, col = 0, g = 0, kn = 0;
    int32_t max_token = -1;

    if (!s || !out) { ZT_SETERR(err, errcap, "null argument"); return ZT_ERR_INVALID_ARG; }
    *out = NULL;
    if (s->n_fields == 0) { ZT_SETERR(err, errcap, "schema has no fields"); return ZT_ERR_BAD_SCHEMA; }

    /* ---- validate ---- */
    for (i = 0; i < s->n_fields; i++) {
        const sch_field *f = &s->fields[i];
        uint32_t fcols = 0;
        if (f->kind == ZT_KIND_BOOL && f->n_choices != 2) {
            ZT_SETERR(err, errcap, "field '%s': BOOL needs exactly 2 choices (true, false), has %u", f->id, f->n_choices);
            return ZT_ERR_BAD_SCHEMA;
        }
        if (f->n_choices < 2) {
            ZT_SETERR(err, errcap, "field '%s': needs at least 2 choices, has %u", f->id, f->n_choices);
            return ZT_ERR_BAD_SCHEMA;
        }
        for (c = 0; c < f->n_choices; c++) {
            const sch_choice *ch = &f->choices[c];
            uint32_t d, u;
            fcols += ch->n_tokens;
            for (t = 0; t < ch->n_tokens; t++) {
                if (ch->tokens[t] > max_token) max_token = ch->tokens[t];
                for (d = 0; d < c; d++)
                    for (u = 0; u < f->choices[d].n_tokens; u++)
                        if (f->choices[d].tokens[u] == ch->tokens[t]) {
                            ZT_SETERR(err, errcap, "field '%s': token %d belongs to both '%s' and '%s'",
                                      f->id, ch->tokens[t], f->choices[d].label, ch->label);
                            return ZT_ERR_AMBIGUOUS;
                        }
            }
        }
        if (fcols > ZT_MAX_COLS_PER_FIELD) {
            ZT_SETERR(err, errcap, "field '%s': %u alias tokens exceed ZT_MAX_COLS_PER_FIELD", f->id, fcols);
            return ZT_ERR_BAD_SCHEMA;
        }
        if (f->dep_field >= 0) {
            const sch_field *on = &s->fields[f->dep_field];
            if ((uint32_t)f->dep_field >= i) { ZT_SETERR(err, errcap, "field '%s': dependency must be an earlier field", f->id); return ZT_ERR_BAD_SCHEMA; }
            if ((uint32_t)f->dep_choice >= on->n_choices) {
                ZT_SETERR(err, errcap, "field '%s': required choice %d out of range for '%s'", f->id, f->dep_choice, on->id);
                return ZT_ERR_BAD_SCHEMA;
            }
        }
        n_cols    += fcols;
        n_choices += f->n_choices;
        n_knots   += f->n_cal;
        if (f->n_choices > c_max) c_max = f->n_choices;
    }

    /* ---- allocate ---- */
    p = (zt_program *)calloc(1, sizeof *p);
    if (!p) { ZT_SETERR(err, errcap, "out of memory"); return ZT_ERR_NO_MEMORY; }
    p->n_fields  = s->n_fields;
    p->n_cols    = n_cols;
    p->n_choices = n_choices;
    p->c_max     = c_max;
    p->n_knots   = n_knots;
    p->max_token = max_token;
    p->fields           = (zt_field_dev *)calloc(p->n_fields, sizeof *p->fields);
    p->col_token        = (int32_t *)calloc(n_cols, sizeof *p->col_token);
    p->choice_col_start = (uint32_t *)calloc(n_choices + 1, sizeof *p->choice_col_start);
    p->choice_value     = (float *)calloc(n_choices, sizeof *p->choice_value);
    p->knots            = (zt_knot *)calloc(n_knots ? n_knots : 1, sizeof *p->knots);
    p->dep_field        = (int32_t *)calloc(p->n_fields, sizeof *p->dep_field);
    p->dep_choice       = (int32_t *)calloc(p->n_fields, sizeof *p->dep_choice);
    p->field_id         = (char **)calloc(p->n_fields, sizeof *p->field_id);
    p->choice_label     = (char **)calloc(n_choices, sizeof *p->choice_label);
    if (!p->fields || !p->col_token || !p->choice_col_start || !p->choice_value || !p->knots ||
        !p->dep_field || !p->dep_choice || !p->field_id || !p->choice_label) {
        zt_program_free(p); ZT_SETERR(err, errcap, "out of memory"); return ZT_ERR_NO_MEMORY;
    }

    /* ---- flatten ---- */
    for (i = 0; i < s->n_fields; i++) {
        const sch_field *f = &s->fields[i];
        zt_field_dev *d = &p->fields[i];
        d->kind         = f->kind;
        d->col_start    = col;
        d->n_choices    = f->n_choices;
        d->choice_start = g;
        d->calib_off    = kn;
        d->calib_n      = f->n_cal;
        d->inv_temp     = 1.0f / f->temperature;
        d->threshold    = f->threshold;
        d->min_conf     = f->min_conf;
        d->min_margin   = f->min_margin;
        p->dep_field[i]  = f->dep_field;
        p->dep_choice[i] = f->dep_choice;
        p->field_id[i]   = zt_strdup(f->id);
        if (!p->field_id[i]) { zt_program_free(p); ZT_SETERR(err, errcap, "out of memory"); return ZT_ERR_NO_MEMORY; }
        for (c = 0; c < f->n_choices; c++, g++) {
            const sch_choice *ch = &f->choices[c];
            p->choice_col_start[g] = col;
            p->choice_value[g]     = ch->value;
            p->choice_label[g]     = zt_strdup(ch->label);
            if (!p->choice_label[g]) { p->n_choices = g + 1; zt_program_free(p); ZT_SETERR(err, errcap, "out of memory"); return ZT_ERR_NO_MEMORY; }
            for (t = 0; t < ch->n_tokens; t++) p->col_token[col++] = ch->tokens[t];
        }
        d->n_cols = col - d->col_start;
        for (t = 0; t < f->n_cal; t++, kn++) { p->knots[kn].x = f->cal_x[t]; p->knots[kn].y = f->cal_y[t]; }
    }
    p->choice_col_start[g] = col;
    *out = p;
    return ZT_OK;
}

/* ------------------------------------------------------------ introspection */
uint32_t zt_program_n_fields(const zt_program *p)  { return p ? p->n_fields : 0; }
uint32_t zt_program_n_cols(const zt_program *p)    { return p ? p->n_cols : 0; }
int32_t  zt_program_max_token(const zt_program *p) { return p ? p->max_token : -1; }
uint32_t zt_program_c_max(const zt_program *p)     { return p ? p->c_max : 0; }
uint32_t zt_program_field_kind(const zt_program *p, uint32_t field)
{ return (p && field < p->n_fields) ? p->fields[field].kind : 0xffffffffu; }
const char *zt_program_field_id(const zt_program *p, uint32_t field)
{ return (p && field < p->n_fields) ? p->field_id[field] : NULL; }
uint32_t zt_program_n_choices(const zt_program *p, uint32_t field)
{ return (p && field < p->n_fields) ? p->fields[field].n_choices : 0; }
const char *zt_program_choice_label(const zt_program *p, uint32_t field, uint32_t choice)
{
    if (!p || field >= p->n_fields || choice >= p->fields[field].n_choices) return NULL;
    return p->choice_label[p->fields[field].choice_start + choice];
}
float zt_program_choice_value(const zt_program *p, uint32_t field, uint32_t choice)
{
    if (!p || field >= p->n_fields || choice >= p->fields[field].n_choices) return 0.0f;
    return p->choice_value[p->fields[field].choice_start + choice];
}
const int32_t *zt_program_col_tokens(const zt_program *p, uint32_t *n_out)
{
    if (n_out) *n_out = p ? p->n_cols : 0;
    return p ? p->col_token : NULL;
}
