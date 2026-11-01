/* SPDX-License-Identifier: GPL-2.0-only
 * libgurt - Copyright (C) the GURT authors.
 * Algorithms ported from DREADER (GPL-2.0). This program is free software;
 * you can redistribute it and/or modify it under the terms of the GNU General
 * Public License version 2 as published by the Free Software Foundation.
 * WITHOUT ANY WARRANTY; see the file LICENSE.
 */
/*
 * libgurt -- GURT's radar editing / QC algorithms in portable C99.
 *
 * Data model
 * ----------
 * A field is a row-major float array [nrays][ngates]; a missing / masked gate
 * is NaN. Rays are in the sweep's own order (not azimuth sorted); functions
 * that need azimuth adjacency (despeckle's `wrap`) document it.
 *
 * Regions (the hook for brushes)
 * ------------------------------
 * Every editing routine takes an optional `region` mask, uint8 [nrays][ngates].
 * NULL = the whole sweep. Otherwise a gate is only *modified* where the mask
 * is non-zero, but its neighbours are always read from the whole field. That is
 * exactly a brush or polygon edit: the footprint (circle, polygon, ...) is
 * turned into a mask by the caller, and the algorithm runs only on it, with
 * context from the gates around it. Brush footprints and the remaining brush
 * modes (unfold, gradient, GateFilter) are planned on top of this same shape.
 *
 * All routines return GURT_OK (0) or a negative GURT_E* code.
 */
#ifndef GURT_H
#define GURT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define GURT_API __declspec(dllexport)
#else
#  define GURT_API __attribute__((visibility("default")))
#endif

#define GURT_VERSION "0.1.0"

enum {
    GURT_OK        =  0,
    GURT_EARG      = -1,   /* bad argument */
    GURT_ENOMEM    = -2
};

GURT_API const char *gurt_version(void);

/* ---------------------------------------------------------------- despeckle
 * DREADER despeck_parm. A good gate is deleted (set NaN) when fewer than
 * `min_good` of the good gates in its (2r+1)x(2r+1) window (itself excluded)
 * remain; the minimum scales down at clipped sweep edges. Repeats until a pass
 * removes nothing, or `max_pass` passes (0 = until stable). Each pass reads
 * the data as it stood at the start of the pass.
 * wrap != 0 treats the ray axis as circular (full-circle sweeps).
 * n_removed / n_passes may be NULL.
 */
GURT_API int gurt_despeckle(float *data, int32_t nrays, int32_t ngates,
                            int32_t radius, int32_t min_good, int32_t max_pass,
                            int32_t wrap, const uint8_t *region,
                            int64_t *n_removed, int32_t *n_passes);

/* ---------------------------------------------------------------- deglitch
 * DREADER deglitch_parm as used by GURT (edit/deglitch.py).
 *
 * A gate whose |value - median of its good neighbours| exceeds `thresh` is
 * folded by +/- 2*nyquist steps toward the median; the closest result wins. If
 * none lands within `thresh` the gate is deleted (delete_bad) or left alone.
 * Passes repeat until the glitch count stops changing, or max_pass (0 =
 * until stable).
 *
 * nyq       per-ray Nyquist of `data`, length nrays (required)
 * numprt    1: fold against nyq only
 *           2: two-PRT: after nyq fails, retry prt1 (with nyq1) then prt2
 *              (with nyq2), each a separate velocity field
 *           3: same-field retry: re-fold the original value against nyq1 then
 *              nyq2 (staggered PRT with one combined velocity field)
 * prt1, prt2, nyq1, nyq2 used for numprt 2 / 3 as described (else NULL).
 * n_glitch / n_passes may be NULL (n_glitch = glitches found in the last pass).
 */
GURT_API int gurt_deglitch(float *data, int32_t nrays, int32_t ngates,
                           const double *nyq, int32_t radius, double thresh,
                           int32_t max_pass, int32_t delete_bad, int32_t numprt,
                           const float *prt1, const float *prt2,
                           const double *nyq1, const double *nyq2,
                           const uint8_t *region,
                           int64_t *n_glitch, int32_t *n_passes);

/* ---------------------------------------------------------------- gap fill
 * Fill missing gates along each ray from the nearest `n_neighbors` good gates
 * (alternating left / right outward), by median or mean, over `n_passes`
 * sweeps (a filled gate counts as a neighbour afterwards).
 * `protect` (uint8, may be NULL): non-zero gates are never filled (gates that
 * were missing in the original data, before deglitching).
 * Returns the number of gates filled in *n_filled (may be NULL).
 */
GURT_API int gurt_gapfill(float *data, int32_t nrays, int32_t ngates,
                          int32_t n_neighbors, int32_t use_median,
                          int32_t n_passes, const uint8_t *protect,
                          const uint8_t *region, int64_t *n_filled);

#ifdef __cplusplus
}
#endif
#endif /* GURT_H */
