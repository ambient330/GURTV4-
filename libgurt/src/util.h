/* SPDX-License-Identifier: GPL-2.0-only
 * libgurt - Copyright (C) the GURT authors.
 * Algorithms ported from DREADER (GPL-2.0). This program is free software;
 * you can redistribute it and/or modify it under the terms of the GNU General
 * Public License version 2 as published by the Free Software Foundation.
 * WITHOUT ANY WARRANTY; see the file LICENSE.
 */
#ifndef GURT_UTIL_H
#define GURT_UTIL_H
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#define IS_BAD(v) isnan(v)
#define IDX(r, g, ng) ((size_t)(r) * (size_t)(ng) + (size_t)(g))

static inline int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* median of n values (average of the two middle values when n is even); sorts v */
static inline double median_inplace(double *v, int n)
{
    if (n < 24) {                                   /* insertion sort: windows are tiny */
        for (int i = 1; i < n; i++) {
            double x = v[i];
            int j = i - 1;
            while (j >= 0 && v[j] > x) { v[j + 1] = v[j]; j--; }
            v[j + 1] = x;
        }
    } else {
        qsort(v, (size_t)n, sizeof(double), cmp_double);
    }
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
#endif
