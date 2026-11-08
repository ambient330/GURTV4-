/* SPDX-License-Identifier: GPL-2.0-only
 * libgurt - Copyright (C) the GURT authors.
 * Algorithms ported from DREADER (GPL-2.0). This program is free software;
 * you can redistribute it and/or modify it under the terms of the GNU General
 * Public License version 2 as published by the Free Software Foundation.
 * WITHOUT ANY WARRANTY; see the file LICENSE.
 */
#include "gurt.h"
#include "util.h"

/* DREADER dealias_parm (unfold mode). All arithmetic is float, as in the C original. */

static int median_f(float *v, int n, float *out)
{
    if (n <= 0) return 0;
    for (int i = 1; i < n; i++) {                       /* insertion sort: n is tiny */
        float x = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > x) { v[j + 1] = v[j]; j--; }
        v[j + 1] = x;
    }
    *out = (n % 2 == 0) ? (v[n / 2 - 1] + v[n / 2]) / 2.0f : v[(n - 1) / 2];
    return 1;
}

GURT_API int gurt_unfold(float *data, int32_t nrays, int32_t ngates,
                         const double *nyq, int32_t radius, double thresh,
                         const float *ref, int32_t ref_nrays, int32_t ref_ngates,
                         const int32_t *ray_map, const int32_t *gate_map,
                         const uint8_t *region, int64_t *n_unfolded)
{
    if (!data || !nyq || nrays < 1 || ngates < 1 || radius < 0) return GURT_EARG;
    if (ref && (ref_nrays < 1 || ref_ngates < 1 || !ray_map || !gate_map)) return GURT_EARG;
    float *good = (float *)malloc((size_t)(2 * radius + 4) * sizeof(float));
    if (!good) return GURT_ENOMEM;
    const float fthresh = (float)thresh;
    int64_t changed = 0;

    for (int32_t ray = 0; ray < nrays; ray++) {
        const float nq = (float)nyq[ray];
        if (!(nq > 0.0f) || !isfinite(nq)) continue;
        for (int32_t gate = 0; gate < ngates; gate++) {
            size_t k = IDX(ray, gate, ngates);
            if (IS_BAD(data[k]) || (region && !region[k])) continue;

            int abort = 0, numgood = 0;
            float locvalue = 0.0f, median = 0.0f;

            if (ref) {                                       /* reference sweep */
                int32_t rr = ray_map[ray], rg = gate_map[gate];
                if (rr < 0 || rg < 0) abort = 1;
                else {
                    const float *row = ref + (size_t)rr * (size_t)ref_ngates;
                    int32_t lg = rg, count = 0;
                    while (lg >= 0 && count <= radius) {
                        if (!IS_BAD(row[lg])) good[numgood++] = row[lg];
                        count++; lg--;
                    }
                    lg = rg + 1; count = 0;
                    while (lg < ref_ngates && count <= radius) {
                        if (!IS_BAD(row[lg])) good[numgood++] = row[lg];
                        count++; lg++;
                    }
                    if (numgood == 0) {                      /* nearest good gate beyond, then behind */
                        while (lg < ref_ngates && numgood == 0) {
                            if (!IS_BAD(row[lg])) good[numgood++] = row[lg];
                            lg++;
                        }
                    }
                    if (numgood == 0) {
                        lg = rg - radius;
                        while (lg >= 0 && numgood == 0) {
                            if (!IS_BAD(row[lg])) good[numgood++] = row[lg];
                            lg--;
                        }
                    }
                    if (median_f(good, numgood, &median)) locvalue = median; else abort = 1;
                }
            }

            if (!ref || abort) {                             /* same ray, previous gates */
                int32_t lg = gate - 1, count = 0;
                numgood = 0;
                while (lg >= 0 && count <= radius) {
                    float v = data[IDX(ray, lg, ngates)];
                    if (!IS_BAD(v)) good[numgood++] = v;
                    count++; lg--;
                }
                if (median_f(good, numgood, &median)) { locvalue = median; abort = 0; }
                else abort = 1;
            }

            const float orig = data[k];
            float newvalue = orig;
            float mindiff = fabsf(newvalue - locvalue);
            if (mindiff > fthresh * nq && !abort) {
                int unf = 0, prev = 0, steps = 0;
                while (!abort && steps++ < 100000) {
                    if (unf == prev || prev == 0) {
                        prev = unf;
                        if (newvalue < locvalue) { newvalue = newvalue + 2.0f * nq; unf = 1; }
                        else                     { newvalue = newvalue - 2.0f * nq; unf = -1; }
                        float newdiff = fabsf(newvalue - locvalue);
                        if (newdiff < mindiff) { data[k] = newvalue; mindiff = newdiff; }
                    } else abort = 1;
                }
            }
            if (data[k] != orig) changed++;
        }
    }
    free(good);
    if (n_unfolded) *n_unfolded = changed;
    return GURT_OK;
}
