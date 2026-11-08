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

/* ------------------------------------------------------------------ unfold
 * DREADER dealias_parm (unfold mode), Bargen-Brown style. For each good gate
 * (ray by ray, range increasing, edited in place) a local reference velocity is
 * the median of the good gates among the previous radius+1 gates on the same
 * ray (already-unfolded values). If the gate differs from it by more than
 * thresh*nyquist it is stepped by +/-2*nyquist toward the reference for as
 * long as it stays on the same side, and the closest value seen is kept.
 * Gates with no reference value are left alone.
 *
 * Optional reference sweep (ref != NULL, [ref_nrays][ref_ngates], NaN = missing):
 * the reference is instead the median of the good ref gates within +/-radius
 * gates of the matching ref gate (ray_map[ray] / gate_map[gate] give the ref
 * ray / gate index, or -1 for none), then the nearest good gate along that ref
 * ray. When the reference gives nothing the same-ray rule above is used.
 *
 * nyq   per-ray Nyquist (length nrays). n_unfolded may be NULL.
 */
GURT_API int gurt_unfold(float *data, int32_t nrays, int32_t ngates,
                         const double *nyq, int32_t radius, double thresh,
                         const float *ref, int32_t ref_nrays, int32_t ref_ngates,
                         const int32_t *ray_map, const int32_t *gate_map,
                         const uint8_t *region, int64_t *n_unfolded);


/* ---------------------------------------------------------------- tormodel
 * DREADER tormodel: an analytic tornado wind field driven by radar observations.
 * Observations (time, vortex position as radar azimuth / range, radius of maximum
 * wind and tangential / radial wind-profile attributes) are interpolated to a
 * regular time axis (cubic spline for the position, linear for everything else,
 * translation from the position track), and the horizontal wind is
 *     V = translation + Vt(r) * tangential + Vr(r) * radial,
 *     V*(r) = Vmax* * (r / R)^exp   (core exponent for r <= R, decay exponent beyond).
 * Arrays are float32; angles are degrees (meteorological: 0 = north, clockwise);
 * x east, y north, metres from the radar.
 *
 * gurt_tor_interp  obs: [numobs][GURT_TOR_NOBS] (order GURT_OBS_*), tobs: seconds
 *                  (strictly increasing, int32 -- pass them relative to any origin),
 *                  step t is at t0 + t*dt (same origin). state: [numt][GURT_TOR_NSTATE]
 *                  (order GURT_ST_*). Needs numobs >= 2 and numt >= 3.
 * gurt_tor_grid    wind on an x/y grid at one time step. Outputs [numy][numx], any may be NULL.
 * gurt_tor_maxgrid maximum speed over the first `nsteps` steps (0 = the first step only).
 * gurt_tor_point   time series at a point: out [numt][4] = tornado distance, tornado
 *                  direction, wind speed, wind direction.
 * gurt_tor_latlon  latitude / longitude of a ground point x, y metres from the radar.
 * gurt_tor_polar / gurt_tor_cart  DREADER's coordinate conversions (x, y <-> range, azimuth).
 */
#define GURT_TOR_NOBS   9
#define GURT_TOR_NSTATE 13
enum { GURT_OBS_AZIM, GURT_OBS_RANG, GURT_OBS_RADIUS, GURT_OBS_MAXTAN, GURT_OBS_CORETAN,
       GURT_OBS_DECAYTAN, GURT_OBS_MAXRAD, GURT_OBS_CORERAD, GURT_OBS_DECAYRAD };
enum { GURT_ST_X, GURT_ST_Y, GURT_ST_AZIM, GURT_ST_RANG, GURT_ST_RADIUS, GURT_ST_TRANSVEL,
       GURT_ST_TRANSANG, GURT_ST_MAXTAN, GURT_ST_CORETAN, GURT_ST_DECAYTAN, GURT_ST_MAXRAD,
       GURT_ST_CORERAD, GURT_ST_DECAYRAD };

GURT_API int gurt_tor_interp(int32_t numobs, const int32_t *tobs, const float *obs,
                             int32_t numt, int32_t t0, int32_t dt, float *state);
GURT_API int gurt_tor_grid(const float *state, int32_t numt, int32_t tstep,
                           int32_t x0, int32_t dx, int32_t numx, int32_t y0, int32_t dy, int32_t numy,
                           float *speed, float *direc, float *tordist, float *posang);
GURT_API int gurt_tor_maxgrid(const float *state, int32_t numt, int32_t nsteps,
                              int32_t x0, int32_t dx, int32_t numx, int32_t y0, int32_t dy, int32_t numy,
                              float *maxspeed);
GURT_API int gurt_tor_point(const float *state, int32_t numt, float xpos, float ypos, float *out);
GURT_API int gurt_tor_latlon(float radar_lat, float radar_lon, float xpos, float ypos,
                             float *lat, float *lon);
GURT_API int gurt_tor_polar(float x, float y, float *mag, float *angle);
GURT_API int gurt_tor_cart(float mag, float angle, float *x, float *y);

#ifdef __cplusplus
}
#endif
#endif /* GURT_H */
