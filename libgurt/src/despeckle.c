/* SPDX-License-Identifier: GPL-2.0-only
 * libgurt - Copyright (C) the GURT authors.
 * Algorithms ported from DREADER (GPL-2.0). This program is free software;
 * you can redistribute it and/or modify it under the terms of the GNU General
 * Public License version 2 as published by the Free Software Foundation.
 * WITHOUT ANY WARRANTY; see the file LICENSE.
 */
#include "gurt.h"
#include "util.h"

/* Summed-area table of `good` over a ray axis extended by r rows each side
 * (circular if wrap, zeros otherwise). sat has (nrays+2r+1) x (ngates+1) entries. */
static void build_sat(const uint8_t *good, int32_t nr, int32_t ng, int32_t r,
                      int wrap, int32_t *sat)
{
    int32_t rows = nr + 2 * r, W = ng + 1;
    for (int32_t j = 0; j < W; j++) sat[j] = 0;
    for (int32_t i = 0; i < rows; i++) {
        int32_t src = i - r;
        int inside = 1;
        if (src < 0 || src >= nr) {
            if (wrap) src = ((src % nr) + nr) % nr; else inside = 0;
        }
        int32_t *prev = sat + (size_t)i * W, *cur = sat + (size_t)(i + 1) * W;
        int32_t run = 0;
        cur[0] = 0;
        for (int32_t j = 0; j < ng; j++) {
            run += (inside && good[IDX(src, j, ng)]) ? 1 : 0;
            cur[j + 1] = prev[j + 1] + run;
        }
    }
}

GURT_API int gurt_despeckle(float *data, int32_t nrays, int32_t ngates,
                            int32_t radius, int32_t min_good, int32_t max_pass,
                            int32_t wrap, const uint8_t *region,
                            int64_t *n_removed, int32_t *n_passes)
{
    if (!data || nrays < 1 || ngates < 1 || radius < 0 || min_good < 0 || max_pass < 0)
        return GURT_EARG;
    const int32_t r = radius;
    const int do_wrap = (wrap && nrays >= 2 * r + 1) ? 1 : 0;
    const size_t n = (size_t)nrays * (size_t)ngates;
    const int32_t W = ngates + 1;

    uint8_t *good = (uint8_t *)malloc(n);
    int32_t *sat = (int32_t *)malloc((size_t)(nrays + 2 * r + 1) * (size_t)W * sizeof(int32_t));
    int32_t *need = (int32_t *)malloc(n * sizeof(int32_t));
    if (!good || !sat || !need) { free(good); free(sat); free(need); return GURT_ENOMEM; }

    for (size_t i = 0; i < n; i++) good[i] = !IS_BAD(data[i]);

    /* minimum good neighbours per gate: ceil(frac * clipped-window-size-minus-self) */
    const double maxgood = (double)(2 * r + 1) * (2 * r + 1) - 1.0;
    const double frac = maxgood > 0 ? (double)min_good / maxgood : 0.0;
    for (int32_t i = 0; i < nrays; i++) {
        int32_t h = 2 * r + 1;
        if (!do_wrap) {
            int32_t a = i - r < 0 ? 0 : i - r, b = i + r >= nrays ? nrays - 1 : i + r;
            h = b - a + 1;
        }
        for (int32_t j = 0; j < ngates; j++) {
            int32_t a = j - r < 0 ? 0 : j - r, b = j + r >= ngates ? ngates - 1 : j + r;
            double reduced = (double)h * (double)(b - a + 1) - 1.0;
            need[IDX(i, j, ngates)] = (int32_t)ceil(frac * reduced - 1e-9);
        }
    }

    int64_t total = 0;
    int32_t passes = 0;
    for (;;) {
        build_sat(good, nrays, ngates, r, do_wrap, sat);
        int64_t removed = 0;
        /* decide from the pass-start state; apply afterwards */
        uint8_t *kill = (uint8_t *)calloc(n, 1);
        if (!kill) { free(good); free(sat); free(need); return GURT_ENOMEM; }
        for (int32_t i = 0; i < nrays; i++) {
            for (int32_t j = 0; j < ngates; j++) {
                size_t k = IDX(i, j, ngates);
                if (!good[k] || (region && !region[k])) continue;
                int32_t j0 = j - r < 0 ? 0 : j - r;
                int32_t j1 = j + r >= ngates ? ngates : j + r + 1;       /* exclusive */
                int32_t r0 = i, r1 = i + 2 * r + 1;                      /* SAT rows, offset by r */
                int32_t cnt = sat[(size_t)r1 * W + j1] - sat[(size_t)r0 * W + j1]
                            - sat[(size_t)r1 * W + j0] + sat[(size_t)r0 * W + j0] - 1;
                if (cnt < need[k]) { kill[k] = 1; removed++; }
            }
        }
        for (size_t k = 0; k < n; k++)
            if (kill[k]) { good[k] = 0; data[k] = NAN; }
        free(kill);
        passes++;
        total += removed;
        if (removed == 0) break;
        if (max_pass > 0 && passes >= max_pass) break;
    }
    if (n_removed) *n_removed = total;
    if (n_passes) *n_passes = passes;
    free(good); free(sat); free(need);
    return GURT_OK;
}
