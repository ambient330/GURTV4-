/* SPDX-License-Identifier: GPL-2.0-only
 * libgurt - Copyright (C) the GURT authors.
 * Algorithms ported from DREADER (GPL-2.0). This program is free software;
 * you can redistribute it and/or modify it under the terms of the GNU General
 * Public License version 2 as published by the Free Software Foundation.
 * WITHOUT ANY WARRANTY; see the file LICENSE.
 */
#include "gurt.h"
#include "util.h"

GURT_API const char *gurt_version(void) { return GURT_VERSION; }

GURT_API int gurt_gapfill(float *data, int32_t nrays, int32_t ngates,
                          int32_t n_neighbors, int32_t use_median,
                          int32_t n_passes, const uint8_t *protect,
                          const uint8_t *region, int64_t *n_filled)
{
    if (!data || nrays < 1 || ngates < 1 || n_neighbors < 1 || n_passes < 1) return GURT_EARG;
    double *nb = (double *)malloc((size_t)n_neighbors * sizeof(double));
    if (!nb) return GURT_ENOMEM;
    int64_t filled = 0;
    for (int32_t pass = 0; pass < n_passes; pass++) {
        for (int32_t ray = 0; ray < nrays; ray++) {
            for (int32_t gate = 0; gate < ngates; gate++) {
                size_t k = IDX(ray, gate, ngates);
                if ((protect && protect[k]) || (region && !region[k])) continue;
                if (!IS_BAD(data[k])) continue;
                int count = 0;
                int32_t left = gate - 1, right = gate + 1;
                while ((left >= 0 || right < ngates) && count < n_neighbors) {
                    if (left >= 0) {
                        float lv = data[IDX(ray, left, ngates)];
                        if (!IS_BAD(lv)) nb[count++] = (double)lv;
                    }
                    if (count >= n_neighbors) break;
                    if (right < ngates) {
                        float rv = data[IDX(ray, right, ngates)];
                        if (!IS_BAD(rv)) nb[count++] = (double)rv;
                    }
                    left--; right++;
                }
                if (count == 0) continue;
                double v;
                if (use_median) v = median_inplace(nb, count);
                else { double s = 0.0; for (int i = 0; i < count; i++) s += nb[i]; v = s / count; }
                data[k] = (float)v;
                filled++;
            }
        }
    }
    if (n_filled) *n_filled = filled;
    free(nb);
    return GURT_OK;
}
