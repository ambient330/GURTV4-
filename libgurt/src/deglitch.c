/* SPDX-License-Identifier: GPL-2.0-only
 * libgurt - Copyright (C) the GURT authors.
 * Algorithms ported from DREADER (GPL-2.0). This program is free software;
 * you can redistribute it and/or modify it under the terms of the GNU General
 * Public License version 2 as published by the Free Software Foundation.
 * WITHOUT ANY WARRANTY; see the file LICENSE.
 */
#include "gurt.h"
#include "util.h"
#include <string.h>

GURT_API int gurt_deglitch(float *data, int32_t nrays, int32_t ngates,
                           const double *nyq, int32_t radius, double thresh,
                           int32_t max_pass, int32_t delete_bad, int32_t numprt,
                           const float *prt1, const float *prt2,
                           const double *nyq1, const double *nyq2,
                           const uint8_t *region,
                           int64_t *n_glitch, int32_t *n_passes)
{
    if (!data || !nyq || nrays < 1 || ngates < 1 || radius < 0 || thresh < 0.0 ||
        max_pass < 0 || numprt < 1 || numprt > 3)
        return GURT_EARG;
    if (max_pass == 0 && !delete_bad)       /* cannot reach a stable solution without deleting */
        return GURT_EARG;
    if (numprt == 2 && (!prt1 || !prt2 || !nyq1 || !nyq2)) return GURT_EARG;
    if (numprt == 3 && (!nyq1 || !nyq2)) return GURT_EARG;

    const size_t n = (size_t)nrays * (size_t)ngates;
    const int32_t side = 2 * radius + 1;
    float *temp = (float *)malloc(n * sizeof(float));
    double *goodvals = (double *)malloc((size_t)side * (size_t)side * sizeof(double));
    if (!temp || !goodvals) { free(temp); free(goodvals); return GURT_ENOMEM; }
    memcpy(temp, data, n * sizeof(float));

    int64_t numglitch = -1, prevnumglitch = 0;
    int32_t pass = 0;

    while (numglitch != prevnumglitch) {
        prevnumglitch = numglitch;
        numglitch = 0;
        for (int32_t ray = 0; ray < nrays; ray++) {
            int32_t minray = ray - radius < 0 ? 0 : ray - radius;
            int32_t maxray = ray + radius >= nrays ? nrays - 1 : ray + radius;
            for (int32_t gate = 0; gate < ngates; gate++) {
                size_t k = IDX(ray, gate, ngates);
                double val = data[k];
                if (IS_BAD(data[k]) || (region && !region[k])) continue;

                int32_t mingate = gate - radius < 0 ? 0 : gate - radius;
                int32_t maxgate = gate + radius >= ngates ? ngates - 1 : gate + radius;
                int numgood = 0;
                for (int32_t lr = minray; lr <= maxray; lr++)
                    for (int32_t lg = mingate; lg <= maxgate; lg++) {
                        if (lr == ray && lg == gate) continue;
                        float v = data[IDX(lr, lg, ngates)];
                        if (!IS_BAD(v)) goodvals[numgood++] = (double)v;
                    }
                if (numgood == 0) continue;

                double median = median_inplace(goodvals, numgood);
                double nq = nyq[ray];
                double origvalue = val;
                double mindiff = fabs(origvalue - median);
                double newvalue = origvalue;

                if (mindiff > thresh) {
                    numglitch++;
                    int unfdirect = 0, prevunfdirect = 0, prt = 0, abort = 0;
                    while (!abort) {
                        if (unfdirect == prevunfdirect || prevunfdirect == 0) {
                            prevunfdirect = unfdirect;
                            if (newvalue < median) { newvalue += 2.0 * nq; unfdirect = 1; }
                            else                   { newvalue -= 2.0 * nq; unfdirect = -1; }
                            double newdiff = fabs(newvalue - median);
                            if (newdiff < mindiff) { temp[k] = (float)newvalue; mindiff = newdiff; }
                        } else if (prt == 0 && numprt == 2) {
                            newvalue = (double)prt1[k]; prt = 1; unfdirect = 0; prevunfdirect = 0; nq = nyq1[ray];
                        } else if (prt == 1 && numprt == 2) {
                            newvalue = (double)prt2[k]; prt = 2; unfdirect = 0; prevunfdirect = 0; nq = nyq2[ray];
                        } else if (prt == 0 && numprt == 3) {
                            newvalue = origvalue; prt = 1; unfdirect = 0; prevunfdirect = 0; nq = nyq1[ray];
                        } else if (prt == 1 && numprt == 3) {
                            newvalue = origvalue; prt = 2; unfdirect = 0; prevunfdirect = 0; nq = nyq2[ray];
                        } else {
                            abort = 1;
                        }
                        /* a missing PRT value (NaN) can never get closer; the loop ends via the
                         * unfold-direction logic like the original, but guard runaway values */
                        if (!abort && !isfinite(newvalue)) abort = 1;
                    }
                    if (mindiff > thresh)
                        temp[k] = delete_bad ? NAN : (float)origvalue;
                }
            }
        }
        pass++;
        memcpy(data, temp, n * sizeof(float));
        if (max_pass > 0 && pass >= max_pass) prevnumglitch = numglitch;
    }
    if (n_glitch) *n_glitch = numglitch;
    if (n_passes) *n_passes = pass;
    free(temp); free(goodvals);
    return GURT_OK;
}
