/* ========================================================================
   SOVEREIGN LEVIATHAN COVENANT
   ========================================================================

   Node-ID:           ZERO-TOKEN-008
   File:              zt_fsm.c
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

   Clone-Gate: sha256:62a0b739741fd84fb98e43e9e79a155b051e3edff568ca1d46077ae1f78bbb61

   See: LICENSE
   ======================================================================== */
#include "zt_internal.h"

zt_route zt_fsm_walk(const zt_program *p, zt_result *r, uint32_t b)
{
    uint32_t F, f;
    zt_decision *dec;
    zt_route route = ZT_ROUTE_COMPLETE;

    if (!p || !r || !r->dec || b >= r->cap_batch) return ZT_ROUTE_ABSTAIN;
    F = p->n_fields;
    dec = r->dec + (size_t)b * F;

    for (f = 0; f < F; f++) {
        int32_t on = p->dep_field[f];
        if (on >= 0 && dec[on].choice != p->dep_choice[f]) {
            /* guard failed (including because the guard itself abstained or was skipped) */
            dec[f].choice = ZT_SKIPPED;
            continue;
        }
        if (dec[f].choice == ZT_ABSTAIN) route = ZT_ROUTE_ABSTAIN;
    }
    return route;
}
