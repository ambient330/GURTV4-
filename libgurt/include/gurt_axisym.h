/* SPDX-License-Identifier: GPL-2.0-only
 * libgurt - Copyright (C) the GURT authors.
 * Algorithms ported from DREADER (GPL-2.0). This program is free software;
 * you can redistribute it and/or modify it under the terms of the GNU General
 * Public License version 2 as published by the Free Software Foundation.
 * WITHOUT ANY WARRANTY; see the file LICENSE.
 */
/*
 * Axisymmetric analysis and patch dynamics (DREADER extractor "axisymmetric" and "patch"
 * modes), full GURT build only. Velocity is a double array [nrays][ngates], NaN = missing;
 * az_deg per ray (compass, any range), rng_km per gate; the vortex centre xc, yc is in km,
 * x east, y north of the radar.
 */
#ifndef GURT_AXISYM_H
#define GURT_AXISYM_H

#include "gurt.h"

#ifdef __cplusplus
extern "C" {
#endif

/* DREADER calc_annwidth: the smaller of the angular spacing at center_range_km and the gate
 * spacing (0 if neither is positive). */
GURT_API double gurt_annulus_width(const double *az_deg, int32_t nrays, const double *rng_km,
                                   int32_t ngates, double center_range_km);

/* Axisymmetric mode. Rings of width delta_km stepped by delta_km/2; in each ring a least-squares
 * fit of VD = Vr cos(phi - az) + Vt sin(phi - az) (phi = bearing centre -> gate) gives the mean
 * radial and tangential wind. VD is velocity / cos(elev), minus the vortex translation (u, v)
 * m/s along each beam when has_trans. Rings with fewer than min_samples gates, or an
 * ill-conditioned fit, are skipped; stops when the sample count collapses (edge of the data).
 * rings: [max_rings][GURT_AXI_NCOL] =
 *   radius_km, n, vr, vt, vtot, radial momentum, angular momentum, circulation, pressure deficit,
 *   flow angle (deg). n_rings receives the number written. */
#define GURT_AXI_NCOL 10
GURT_API int gurt_axisym(const double *vel, int32_t nrays, int32_t ngates,
                         const double *az_deg, const double *rng_km, double xc, double yc,
                         double elev_deg, double delta_km, double max_radius_km,
                         int32_t has_trans, double u, double v, double air_density,
                         int32_t min_samples, double *rings, int32_t max_rings, int32_t *n_rings);

/* Patch dynamics: couplet extrema in a circular patch (or at user-placed points when
 * has_extrema), aspect-ratio correction, circulation, radius of maximum vorticity and the
 * vorticity / divergence / stretching / angular momentum at fixed fractions of the mean radius.
 * out: GURT_PATCH_NOUT doubles (order P_*), levels: [GURT_PATCH_NLEV][6] = fraction, radius_km,
 * vorticity, divergence, stretching, angular momentum (NaN where not computable).
 * *status = 0 on a result, 1 when there is no valid inbound / outbound pair (Python: None). */
#define GURT_PATCH_NLEV 8
enum {
    P_PATCH_RADIUS, P_MIN_V, P_MIN_AZ, P_MIN_RNG, P_MIN_X, P_MIN_Y, P_MAX_V, P_MAX_AZ, P_MAX_RNG,
    P_MAX_X, P_MAX_Y, P_MIN_RADIUS, P_MAX_RADIUS, P_MEAN_RADIUS, P_DIAM, P_AXIS, P_ASPECT,
    P_MIN_ADJ, P_MAX_ADJ, P_V_MEAN, P_V_ASYM, P_DELTA_V, P_CIRC, P_DELTA_P, P_VORT_RADIUS,
    P_MIN_GR, P_MAX_GR, P_MIN_GR_ADJ, P_MAX_GR_ADJ, P_LARGEST_DOPPLER, P_LARGEST_ADJ,
    P_LARGEST_GROUND, P_LARGEST_GROUND_ADJ, GURT_PATCH_NOUT
};
GURT_API int gurt_patch(const double *vel, int32_t nrays, int32_t ngates,
                        const double *az_deg, const double *rng_km, double xc, double yc,
                        double base_diam_km, double beamwidth_deg, double numdiam,
                        double storm_speed_ms, double storm_dir_deg, double air_density,
                        int32_t has_extrema, double x_min, double y_min, double x_max, double y_max,
                        double *out, double *levels, int32_t *status);

#ifdef __cplusplus
}
#endif
#endif
