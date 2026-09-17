/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-009
   File:              zt_emit.c
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

   Clone-Gate: sha256:27ac6bc30244d6ee629cf57183e662fea1535bfd0e8531b6d42a9f61db47e424

   See: LICENSE
   ======================================================================== */
#include <math.h>
#include <stdarg.h>
#include "zt_internal.h"

typedef struct { char *buf; size_t cap; size_t len; } wr;

static void w_raw(wr *w, const char *s, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) {
        if (w->len + 1 < w->cap) w->buf[w->len] = s[i];
        w->len++;
    }
}
static void w_str(wr *w, const char *s) { w_raw(w, s, strlen(s)); }
static void w_ch(wr *w, char c) { w_raw(w, &c, 1); }

static void w_fmt(wr *w, const char *fmt, ...)
{
    char tmp[64];
    int n;
    va_list ap;
    va_start(ap, fmt);
    n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n > 0) w_raw(w, tmp, (size_t)n < sizeof tmp ? (size_t)n : sizeof tmp - 1);
}

/* Finite floats print as %.6g; non-finite ones are not valid JSON, so they
 * degrade to null / 0 rather than producing an unparseable document. */
static void w_num(wr *w, float v, int json)
{
    if (v != v || v > 3.0e38f || v < -3.0e38f) { w_str(w, json ? "null" : "0"); return; }
    w_fmt(w, "%.6g", (double)v);
}

static void w_json_str(wr *w, const char *s)
{
    w_ch(w, '"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
            case '"':  w_str(w, "\\\""); break;
            case '\\': w_str(w, "\\\\"); break;
            case '\n': w_str(w, "\\n");  break;
            case '\r': w_str(w, "\\r");  break;
            case '\t': w_str(w, "\\t");  break;
            case '\b': w_str(w, "\\b");  break;
            case '\f': w_str(w, "\\f");  break;
            default:
                if (c < 0x20u) w_fmt(w, "\\u%04x", (unsigned)c);
                else           w_ch(w, (char)c);
        }
    }
    w_ch(w, '"');
}

static void w_xml_text(wr *w, const char *s)
{
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
            case '&':  w_str(w, "&amp;");  break;
            case '<':  w_str(w, "&lt;");   break;
            case '>':  w_str(w, "&gt;");   break;
            case '"':  w_str(w, "&quot;"); break;
            case '\'': w_str(w, "&apos;"); break;
            default:
                if (c < 0x20u && c != '\t' && c != '\n' && c != '\r') w_ch(w, ' ');
                else w_ch(w, (char)c);
        }
    }
}

static zt_status w_done(wr *w, char *buf, size_t cap, size_t *len_out)
{
    if (len_out) *len_out = w->len;
    if (cap == 0) return ZT_ERR_BUFFER_TOO_SMALL;
    buf[w->len < cap ? w->len : cap - 1] = '\0';
    return (w->len + 1 <= cap) ? ZT_OK : ZT_ERR_BUFFER_TOO_SMALL;
}

static const char *status_word(int32_t choice)
{
    if (choice == ZT_ABSTAIN) return "abstain";
    if (choice == ZT_SKIPPED) return "skipped";
    return "ok";
}

static void nl(wr *w, uint32_t flags, int depth)
{
    int i;
    if (!(flags & ZT_EMIT_PRETTY)) return;
    w_ch(w, '\n');
    for (i = 0; i < depth; i++) w_str(w, "  ");
}

/* ------------------------------------------------------------------- JSON */

zt_status zt_emit_json(const zt_program *p, const zt_result *r, uint32_t b, uint32_t flags,
                       char *buf, size_t cap, size_t *len_out)
{
    wr w;
    uint32_t f;
    const zt_decision *dec;

    if (!p || !r || !r->dec || b >= r->cap_batch || (!buf && cap)) return ZT_ERR_INVALID_ARG;
    w.buf = buf; w.cap = cap; w.len = 0;
    dec = r->dec + (size_t)b * p->n_fields;

    w_ch(&w, '{');
    for (f = 0; f < p->n_fields; f++) {
        const zt_decision *d = &dec[f];
        const zt_field_dev *fd = &p->fields[f];
        int ok = d->choice >= 0 && (uint32_t)d->choice < fd->n_choices;
        if (f) w_ch(&w, ',');
        nl(&w, flags, 1);
        w_json_str(&w, p->field_id[f]);
        w_ch(&w, ':');
        if (flags & ZT_EMIT_PRETTY) w_ch(&w, ' ');

        if (flags & ZT_EMIT_META) {
            w_ch(&w, '{');
            nl(&w, flags, 2);
            w_str(&w, "\"value\":");
            if (flags & ZT_EMIT_PRETTY) w_ch(&w, ' ');
        }

        if (!ok) {
            w_str(&w, "null");
        } else if (fd->kind == ZT_KIND_BOOL) {
            w_str(&w, d->choice == 0 ? "true" : "false");
        } else if (fd->kind == ZT_KIND_SCORE) {
            w_num(&w, d->score, 1);
        } else {
            w_json_str(&w, p->choice_label[fd->choice_start + (uint32_t)d->choice]);
        }

        if (flags & ZT_EMIT_META) {
            w_ch(&w, ',');
            nl(&w, flags, 2);
            w_str(&w, "\"label\":");
            if (flags & ZT_EMIT_PRETTY) w_ch(&w, ' ');
            if (ok) w_json_str(&w, p->choice_label[fd->choice_start + (uint32_t)d->choice]);
            else    w_str(&w, "null");
            w_ch(&w, ',');
            nl(&w, flags, 2);
            w_str(&w, "\"status\":");
            if (flags & ZT_EMIT_PRETTY) w_ch(&w, ' ');
            w_json_str(&w, status_word(d->choice));
            w_ch(&w, ',');
            nl(&w, flags, 2);
            w_str(&w, "\"confidence\":");
            if (flags & ZT_EMIT_PRETTY) w_ch(&w, ' ');
            w_num(&w, d->confidence, 1);
            w_ch(&w, ',');
            nl(&w, flags, 2);
            w_str(&w, "\"p_top\":");
            if (flags & ZT_EMIT_PRETTY) w_ch(&w, ' ');
            w_num(&w, d->p_top, 1);
            w_ch(&w, ',');
            nl(&w, flags, 2);
            w_str(&w, "\"p_second\":");
            if (flags & ZT_EMIT_PRETTY) w_ch(&w, ' ');
            w_num(&w, d->p_second, 1);
            w_ch(&w, ',');
            nl(&w, flags, 2);
            w_str(&w, "\"margin\":");
            if (flags & ZT_EMIT_PRETTY) w_ch(&w, ' ');
            w_num(&w, d->margin, 1);
            w_ch(&w, ',');
            nl(&w, flags, 2);
            w_str(&w, "\"entropy\":");
            if (flags & ZT_EMIT_PRETTY) w_ch(&w, ' ');
            w_num(&w, d->entropy, 1);
            nl(&w, flags, 1);
            w_ch(&w, '}');
        }
    }
    nl(&w, flags, 0);
    w_ch(&w, '}');
    return w_done(&w, buf, cap, len_out);
}

/* -------------------------------------------------------------------- XML */

zt_status zt_emit_xml(const zt_program *p, const zt_result *r, uint32_t b, uint32_t flags,
                      char *buf, size_t cap, size_t *len_out)
{
    wr w;
    uint32_t f;
    const zt_decision *dec;

    if (!p || !r || !r->dec || b >= r->cap_batch || (!buf && cap)) return ZT_ERR_INVALID_ARG;
    w.buf = buf; w.cap = cap; w.len = 0;
    dec = r->dec + (size_t)b * p->n_fields;

    w_str(&w, "<decision>");
    for (f = 0; f < p->n_fields; f++) {
        const zt_decision *d = &dec[f];
        const zt_field_dev *fd = &p->fields[f];
        int ok = d->choice >= 0 && (uint32_t)d->choice < fd->n_choices;
        nl(&w, flags, 1);
        w_ch(&w, '<');
        w_str(&w, p->field_id[f]);
        w_str(&w, " kind=\"");
        w_str(&w, fd->kind == ZT_KIND_BOOL ? "bool" : (fd->kind == ZT_KIND_SCORE ? "score" : "choice"));
        w_str(&w, "\" status=\"");
        w_str(&w, status_word(d->choice));
        if (flags & ZT_EMIT_META) {
            w_str(&w, "\" confidence=\"");
            w_num(&w, d->confidence, 0);
            w_str(&w, "\" margin=\"");
            w_num(&w, d->margin, 0);
            w_str(&w, "\" entropy=\"");
            w_num(&w, d->entropy, 0);
        }
        w_ch(&w, '"');
        if (!ok) {
            w_str(&w, "/>");
            continue;
        }
        w_ch(&w, '>');
        if (fd->kind == ZT_KIND_BOOL)       w_str(&w, d->choice == 0 ? "true" : "false");
        else if (fd->kind == ZT_KIND_SCORE) w_num(&w, d->score, 0);
        else                                w_xml_text(&w, p->choice_label[fd->choice_start + (uint32_t)d->choice]);
        w_str(&w, "</");
        w_str(&w, p->field_id[f]);
        w_ch(&w, '>');
    }
    nl(&w, flags, 0);
    w_str(&w, "</decision>");
    return w_done(&w, buf, cap, len_out);
}

/* ------------------------------------------------------------- JSON Schema */

static void schema_value(wr *w, const zt_program *p, uint32_t f, uint32_t flags)
{
    const zt_field_dev *fd = &p->fields[f];
    uint32_t c;
    if (fd->kind == ZT_KIND_BOOL) {
        w_str(w, "{\"type\":[\"boolean\",\"null\"]}");
        return;
    }
    if (fd->kind == ZT_KIND_SCORE) {
        float lo = p->choice_value[fd->choice_start], hi = lo;
        for (c = 1; c < fd->n_choices; c++) {
            float v = p->choice_value[fd->choice_start + c];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        w_str(w, "{\"type\":[\"number\",\"null\"],\"minimum\":");
        w_num(w, lo, 1);
        w_str(w, ",\"maximum\":");
        w_num(w, hi, 1);
        w_ch(w, '}');
        return;
    }
    (void)flags;
    w_str(w, "{\"type\":[\"string\",\"null\"],\"enum\":[");
    for (c = 0; c < fd->n_choices; c++) {
        if (c) w_ch(w, ',');
        w_json_str(w, p->choice_label[fd->choice_start + c]);
    }
    w_str(w, ",null]}");
}

zt_status zt_emit_json_schema(const zt_program *p, uint32_t flags, char *buf, size_t cap, size_t *len_out)
{
    wr w;
    uint32_t f, c;

    if (!p || (!buf && cap)) return ZT_ERR_INVALID_ARG;
    w.buf = buf; w.cap = cap; w.len = 0;

    w_str(&w, "{\"$schema\":\"https://json-schema.org/draft/2020-12/schema\",");
    nl(&w, flags, 1);
    w_str(&w, "\"type\":\"object\",\"additionalProperties\":false,");
    nl(&w, flags, 1);
    w_str(&w, "\"required\":[");
    for (f = 0; f < p->n_fields; f++) { if (f) w_ch(&w, ','); w_json_str(&w, p->field_id[f]); }
    w_str(&w, "],");
    nl(&w, flags, 1);
    w_str(&w, "\"properties\":{");
    for (f = 0; f < p->n_fields; f++) {
        const zt_field_dev *fd = &p->fields[f];
        if (f) w_ch(&w, ',');
        nl(&w, flags, 2);
        w_json_str(&w, p->field_id[f]);
        w_ch(&w, ':');
        if (!(flags & ZT_EMIT_META)) {
            schema_value(&w, p, f, flags);
            continue;
        }
        w_str(&w, "{\"type\":\"object\",\"additionalProperties\":false,");
        w_str(&w, "\"required\":[\"value\",\"label\",\"status\",\"confidence\",\"p_top\",\"p_second\",\"margin\",\"entropy\"],");
        w_str(&w, "\"properties\":{\"value\":");
        schema_value(&w, p, f, flags);
        w_str(&w, ",\"label\":{\"type\":[\"string\",\"null\"],\"enum\":[");
        for (c = 0; c < fd->n_choices; c++) {
            if (c) w_ch(&w, ',');
            w_json_str(&w, p->choice_label[fd->choice_start + c]);
        }
        w_str(&w, ",null]},\"status\":{\"type\":\"string\",\"enum\":[\"ok\",\"abstain\",\"skipped\"]},");
        w_str(&w, "\"confidence\":{\"type\":\"number\",\"minimum\":0,\"maximum\":1},");
        w_str(&w, "\"p_top\":{\"type\":\"number\",\"minimum\":0,\"maximum\":1},");
        w_str(&w, "\"p_second\":{\"type\":\"number\",\"minimum\":0,\"maximum\":1},");
        w_str(&w, "\"margin\":{\"type\":\"number\"},");
        w_str(&w, "\"entropy\":{\"type\":\"number\",\"minimum\":0}}}");
    }
    nl(&w, flags, 1);
    w_ch(&w, '}');
    nl(&w, flags, 0);
    w_ch(&w, '}');
    return w_done(&w, buf, cap, len_out);
}
