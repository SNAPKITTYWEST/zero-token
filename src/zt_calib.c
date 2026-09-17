/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-010
   File:              zt_calib.c
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

   Clone-Gate: sha256:4788154baade21ba247cdf20b153ff42cc8ef511a0f313b732a50852d88d994a

   See: LICENSE
   ======================================================================== */
#include <math.h>
#include "zt_internal.h"

/* mean NLL of labels under softmax(beta * logits) */
static double nll_at(const float *logits, const int32_t *labels, uint32_t n, uint32_t K, double beta)
{
    uint32_t i, k;
    double total = 0.0;
    for (i = 0; i < n; i++) {
        const float *x = logits + (size_t)i * K;
        double m = -1e300, z = 0.0;
        for (k = 0; k < K; k++) { double v = beta * (double)x[k]; if (v > m) m = v; }
        for (k = 0; k < K; k++) z += exp(beta * (double)x[k] - m);
        total += -(beta * (double)x[labels[i]] - m - log(z));
    }
    return total / (double)n;
}

zt_status zt_calib_fit_temperature(const float *logits, const int32_t *labels,
                                   uint32_t n, uint32_t K, float *temperature_out)
{
    const double phi = 0.6180339887498949;   /* 1/golden ratio */
    double a = log(0.02), b = log(50.0);     /* beta in [0.02, 50] -> T in [0.02, 50] */
    double c, d, fc, fd;
    uint32_t i, it;

    if (!logits || !labels || !temperature_out || n == 0 || K < 2) return ZT_ERR_INVALID_ARG;
    for (i = 0; i < n; i++) if (labels[i] < 0 || (uint32_t)labels[i] >= K) return ZT_ERR_INVALID_ARG;

    c = b - phi * (b - a); d = a + phi * (b - a);
    fc = nll_at(logits, labels, n, K, exp(c));
    fd = nll_at(logits, labels, n, K, exp(d));
    for (it = 0; it < 80 && (b - a) > 1e-6; it++) {
        if (fc < fd) { b = d; d = c; fd = fc; c = b - phi * (b - a); fc = nll_at(logits, labels, n, K, exp(c)); }
        else         { a = c; c = d; fc = fd; d = a + phi * (b - a); fd = nll_at(logits, labels, n, K, exp(d)); }
    }
    *temperature_out = (float)(1.0 / exp(0.5 * (a + b)));
    return ZT_OK;
}

zt_status zt_calib_fit_bins(const float *p, const uint8_t *correct, uint32_t n,
                            uint32_t n_bins, zt_knot *knots, uint32_t *n_knots_out)
{
    double *sum_p, *sum_y, *wgt;
    uint32_t i, j, m = 0;

    if (!p || !correct || !knots || !n_knots_out || n == 0 || n_bins < 2 || n_bins > 1024)
        return ZT_ERR_INVALID_ARG;
    for (i = 0; i < n; i++) if (!(p[i] >= 0.0f && p[i] <= 1.0f)) return ZT_ERR_INVALID_ARG;

    sum_p = (double *)calloc(n_bins, sizeof *sum_p);
    sum_y = (double *)calloc(n_bins, sizeof *sum_y);
    wgt   = (double *)calloc(n_bins, sizeof *wgt);
    if (!sum_p || !sum_y || !wgt) { free(sum_p); free(sum_y); free(wgt); return ZT_ERR_NO_MEMORY; }

    for (i = 0; i < n; i++) {
        uint32_t b = (uint32_t)(p[i] * (float)n_bins);
        if (b >= n_bins) b = n_bins - 1u;
        sum_p[b] += (double)p[i];
        sum_y[b] += correct[i] ? 1.0 : 0.0;
        wgt[b]   += 1.0;
    }
    /* compact non-empty bins into (x = mean p, y = empirical accuracy, w = count) */
    for (i = 0; i < n_bins; i++) {
        if (wgt[i] <= 0.0) continue;
        sum_p[m] = sum_p[i] / wgt[i];
        sum_y[m] = sum_y[i] / wgt[i];
        wgt[m]   = wgt[i];
        m++;
    }
    if (m == 0) { free(sum_p); free(sum_y); free(wgt); return ZT_ERR_INVALID_ARG; }

    /* pool-adjacent-violators: make y non-decreasing, weighted */
    for (i = 1; i < m; ) {
        if (sum_y[i] >= sum_y[i - 1]) { i++; continue; }
        {
            double w = wgt[i - 1] + wgt[i];
            sum_y[i - 1] = (sum_y[i - 1] * wgt[i - 1] + sum_y[i] * wgt[i]) / w;
            sum_p[i - 1] = (sum_p[i - 1] * wgt[i - 1] + sum_p[i] * wgt[i]) / w;
            wgt[i - 1]   = w;
            for (j = i; j + 1 < m; j++) { sum_y[j] = sum_y[j + 1]; sum_p[j] = sum_p[j + 1]; wgt[j] = wgt[j + 1]; }
            m--;
            if (i > 1) i--;
        }
    }
    /* x must be strictly increasing for zt_schema_set_calibration */
    for (i = 1; i < m; i++) if (sum_p[i] <= sum_p[i - 1]) sum_p[i] = sum_p[i - 1] + 1e-6;
    if (m == 1) {   /* a single pooled bin degenerates to a constant map */
        knots[0].x = 0.0f; knots[0].y = (float)sum_y[0];
        knots[1].x = 1.0f; knots[1].y = (float)sum_y[0];
        *n_knots_out = 2u;
    } else {
        for (i = 0; i < m; i++) {
            knots[i].x = (float)(sum_p[i] > 1.0 ? 1.0 : sum_p[i]);
            knots[i].y = (float)sum_y[i];
        }
        *n_knots_out = m;
    }
    free(sum_p); free(sum_y); free(wgt);
    return ZT_OK;
}
