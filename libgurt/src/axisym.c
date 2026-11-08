/* SPDX-License-Identifier: GPL-2.0-only
 * libgurt - Copyright (C) the GURT authors.
 * Algorithms ported from DREADER (GPL-2.0), extractor "axisymmetric" and "patch" modes
 * (C. Alexander), as implemented in GURT's gui/dreader_modes.py. This program is free
 * software; you can redistribute it and/or modify it under the terms of the GNU General
 * Public License version 2 as published by the Free Software Foundation. WITHOUT ANY
 * WARRANTY; see the file LICENSE.
 *
 * Doubles throughout. Everything mirrors the Python reference statement by statement
 * (same ring stepping, tie-breaking by first index in ray-major order, same stop rules).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gurt.h"
#include "gurt_axisym.h"

#define KM 1.0e3
#define PI_ 3.14159265358979323846
#define D2R (PI_ / 180.0)

static double pymod(double x, double m)                    /* Python's % for floats */
{
    double r = fmod(x, m);
    if (r != 0.0 && ((r < 0.0) != (m < 0.0))) r += m;
    return r;
}

static double bearing_deg(double dx, double dy)            /* compass bearing of (dx east, dy north) */
{
    return pymod(atan2(dx, dy) / D2R, 360.0);
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_d(double *v, int n)                   /* sorts v */
{
    qsort(v, (size_t)n, sizeof(double), cmp_d);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* DREADER calc_annwidth: the smaller of the angular spacing (at the centre range) and the gate spacing */
GURT_API double gurt_annulus_width(const double *az_deg, int32_t nrays, const double *rng_km,
                                   int32_t ngates, double center_range_km)
{
    double d_az = 0.0, d_r = 0.0;
    if (nrays > 1) {
        double *d = (double *)malloc((size_t)(nrays - 1) * sizeof(double));
        if (!d) return 0.0;
        double prev = pymod(az_deg[0], 360.0) * D2R, cum = 0.0, last = prev;   /* np.unwrap */
        for (int32_t i = 1; i < nrays; i++) {
            double cur = pymod(az_deg[i], 360.0) * D2R;
            double dd = cur - prev;
            double ddmod = pymod(dd + PI_, 2.0 * PI_) - PI_;
            if (ddmod == -PI_ && dd > 0) ddmod = PI_;
            double corr = (fabs(dd) < PI_) ? 0.0 : ddmod - dd;
            cum += corr;
            double un = cur + cum;
            d[i - 1] = fabs(un - last);
            last = un; prev = cur;
        }
        d_az = median_d(d, nrays - 1);
        free(d);
    }
    if (ngates > 1) {
        double *d = (double *)malloc((size_t)(ngates - 1) * sizeof(double));
        if (!d) return 0.0;
        for (int32_t i = 1; i < ngates; i++) d[i - 1] = fabs(rng_km[i] - rng_km[i - 1]);
        d_r = median_d(d, ngates - 1);
        free(d);
    }
    double c1 = center_range_km * d_az, best = 0.0;
    int have = 0;
    if (c1 > 0) { best = c1; have = 1; }
    if (d_r > 0 && (!have || d_r < best)) { best = d_r; have = 1; }
    return have ? best : 0.0;
}

/* ============================================================================ axisymmetric */

/* 2-parameter least squares  b ~ x0*a1 + x1*a2  by Gram-Schmidt QR (re-orthogonalised);
 * returns cond(A) (inf when rank deficient) */
static double lsq2(const double *a1, const double *a2, const double *b, int n, double *x0, double *x1)
{
    double r11 = 0.0;
    for (int i = 0; i < n; i++) r11 += a1[i] * a1[i];
    r11 = sqrt(r11);
    if (!(r11 > 0.0)) return INFINITY;
    double r12 = 0.0;
    double *q2 = (double *)malloc((size_t)n * sizeof(double));
    if (!q2) return INFINITY;
    for (int i = 0; i < n; i++) r12 += (a1[i] / r11) * a2[i];
    for (int i = 0; i < n; i++) q2[i] = a2[i] - r12 * (a1[i] / r11);
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (a1[i] / r11) * q2[i];
    for (int i = 0; i < n; i++) q2[i] -= s * (a1[i] / r11);
    r12 += s;
    double r22 = 0.0;
    for (int i = 0; i < n; i++) r22 += q2[i] * q2[i];
    r22 = sqrt(r22);
    if (!(r22 > 0.0)) { free(q2); return INFINITY; }
    double y1 = 0.0, y2 = 0.0;
    for (int i = 0; i < n; i++) { y1 += (a1[i] / r11) * b[i]; y2 += (q2[i] / r22) * b[i]; }
    *x1 = y2 / r22;
    *x0 = (y1 - r12 * (*x1)) / r11;
    free(q2);
    /* singular values of R = [[r11, r12], [0, r22]] */
    double tr = r11 * r11 + r12 * r12 + r22 * r22, det = r11 * r22;
    double disc = sqrt(fmax(tr * tr - 4.0 * det * det, 0.0));
    double lmax = 0.5 * (tr + disc), lmin = (det * det) / lmax;
    return (lmin > 0.0) ? sqrt(lmax / lmin) : INFINITY;
}

GURT_API int gurt_axisym(const double *vel, int32_t nrays, int32_t ngates,
                         const double *az_deg, const double *rng_km, double xc, double yc,
                         double elev_deg, double delta_km, double max_radius_km,
                         int32_t has_trans, double u, double v, double air_density,
                         int32_t min_samples, double *rings, int32_t max_rings, int32_t *n_rings)
{
    if (!vel || !az_deg || !rng_km || !rings || !n_rings || nrays < 1 || ngates < 1 ||
        !(delta_km > 0.0) || max_rings < 1) return GURT_EARG;
    size_t N = (size_t)nrays * (size_t)ngates;
    double *dist = (double *)malloc(N * sizeof(double));
    double *rc = (double *)malloc(N * sizeof(double));
    double *tc = (double *)malloc(N * sizeof(double));
    double *vd = (double *)malloc(N * sizeof(double));
    unsigned char *good = (unsigned char *)malloc(N);
    int32_t *idx = (int32_t *)malloc(N * sizeof(int32_t));
    double *a1 = (double *)malloc(N * sizeof(double)), *a2 = (double *)malloc(N * sizeof(double)),
           *bb = (double *)malloc(N * sizeof(double));
    if (!dist || !rc || !tc || !vd || !good || !idx || !a1 || !a2 || !bb) {
        free(dist); free(rc); free(tc); free(vd); free(good); free(idx); free(a1); free(a2); free(bb);
        return GURT_ENOMEM;
    }
    const double ce = cos(elev_deg * D2R);
    const double cosel = ce > 1e-3 ? ce : 1e-3;
    for (int32_t ia = 0; ia < nrays; ia++) {
        double az = pymod(az_deg[ia], 360.0) * D2R, sa = sin(az), ca = cos(az);
        for (int32_t ir = 0; ir < ngates; ir++) {
            size_t k = (size_t)ia * (size_t)ngates + (size_t)ir;
            double X = rng_km[ir] * sa, Y = rng_km[ir] * ca;
            double dx = X - xc, dy = Y - yc;
            double val = vel[k] / cosel;
            if (has_trans) val = val - (u * sa + v * ca);
            double phi = atan2(dx, dy);
            dist[k] = hypot(dx, dy);
            tc[k] = sin(phi - az);
            rc[k] = cos(phi - az);
            vd[k] = val;
            good[k] = isfinite(val) ? 1 : 0;
        }
    }

    int32_t nr = 0, prev_n = 0;
    double ring_r = 0.0;
    int rc_out = GURT_OK;
    while (ring_r < max_radius_km) {
        ring_r += delta_km / 2.0;
        const double lo = ring_r - delta_km / 2.0, hi = ring_r + delta_km / 2.0;
        int32_t n = 0;
        for (size_t k = 0; k < N; k++)
            if (good[k] && dist[k] > lo && dist[k] < hi) { a1[n] = rc[k]; a2[n] = tc[k]; bb[n] = vd[k]; n++; }
        if (n >= min_samples) {
            double vr, vt;
            double cond = lsq2(a1, a2, bb, n, &vr, &vt);
            if (cond < 1e6) {
                if (nr >= max_rings) { rc_out = GURT_EARG; break; }
                double r_m = ring_r * KM;
                double *o = rings + (size_t)nr * GURT_AXI_NCOL;
                o[0] = ring_r; o[1] = (double)n; o[2] = vr; o[3] = vt; o[4] = hypot(vr, vt);
                o[5] = r_m * vr; o[6] = r_m * vt; o[7] = 2.0 * PI_ * r_m * vt;
                o[8] = air_density * vt * vt; o[9] = atan2(vr, vt) / D2R;
                nr++;
            }
        }
        if (prev_n && 2 * n < prev_n && ring_r > 2 * delta_km) break;     /* edge of the data */
        prev_n = n;
    }
    *n_rings = nr;
    (void)idx;
    free(dist); free(rc); free(tc); free(vd); free(good); free(idx); free(a1); free(a2); free(bb);
    return rc_out;
}

/* ============================================================================ patch dynamics */

static double aspect_ratio_correction(double beamwidth_deg, double range_km, double diam_km)
{
    if (!(beamwidth_deg > 0) || diam_km <= 0 || range_km <= 0) return 1.0;
    double bw_km = range_km * (beamwidth_deg * D2R);
    if (bw_km <= 0.25 * diam_km) return 1.0 / (1.0 - 0.24 * (2.0 * bw_km / diam_km));
    return 1.14;
}

typedef struct { const double *vel; const double *X, *Y; int32_t ng; size_t N; } Sweep;

/* min / max valid velocity in the annulus |d - radius| <= width/2 (first index wins ties) */
static int ring_extrema(const Sweep *s, double xc, double yc, double radius, double width,
                        int64_t *imin, int64_t *imax)
{
    double vmin = 0, vmax = 0;
    int64_t a = -1, b = -1;
    for (size_t k = 0; k < s->N; k++) {
        double v = s->vel[k];
        if (!isfinite(v)) continue;
        double d = hypot(s->X[k] - xc, s->Y[k] - yc);
        if (d >= radius - width / 2.0 && d <= radius + width / 2.0) {
            if (a < 0 || v < vmin) { vmin = v; a = (int64_t)k; }
            if (b < 0 || v > vmax) { vmax = v; b = (int64_t)k; }
        }
    }
    if (a < 0) return 0;
    *imin = a; *imax = b;
    return 1;
}

static const double FRACS[GURT_PATCH_NLEV] = {0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00};

GURT_API int gurt_patch(const double *vel, int32_t nrays, int32_t ngates,
                        const double *az_deg, const double *rng_km, double xc, double yc,
                        double base_diam_km, double beamwidth_deg, double numdiam,
                        double storm_speed_ms, double storm_dir_deg, double air_density,
                        int32_t has_extrema, double x_min, double y_min, double x_max, double y_max,
                        double *out, double *levels, int32_t *status)
{
    if (!vel || !az_deg || !rng_km || !out || !levels || !status || nrays < 1 || ngates < 1) return GURT_EARG;
    *status = 1;                                          /* 1 = no result (Python: None) */
    size_t N = (size_t)nrays * (size_t)ngates;
    double *X = (double *)malloc(N * sizeof(double)), *Y = (double *)malloc(N * sizeof(double));
    double *az = (double *)malloc((size_t)nrays * sizeof(double));
    if (!X || !Y || !az) { free(X); free(Y); free(az); return GURT_ENOMEM; }
    for (int32_t ia = 0; ia < nrays; ia++) {
        az[ia] = pymod(az_deg[ia], 360.0);
        double sa = sin(az[ia] * D2R), ca = cos(az[ia] * D2R);
        for (int32_t ir = 0; ir < ngates; ir++) {
            X[(size_t)ia * ngates + ir] = rng_km[ir] * sa;
            Y[(size_t)ia * ngates + ir] = rng_km[ir] * ca;
        }
    }
    Sweep sw = {vel, X, Y, ngates, N};
    int rcode = GURT_OK;

    const double rc = hypot(xc, yc), bc = bearing_deg(xc, yc);
    const double patch_radius = (numdiam != 0.0 && base_diam_km != 0.0) ? numdiam * base_diam_km : base_diam_km;
    if (patch_radius <= 0) goto done;

    int64_t i_min = -1, i_max = -1;
    if (has_extrema) {
        size_t nvalid = 0;
        for (size_t k = 0; k < N; k++) if (isfinite(vel[k])) nvalid++;
        if (nvalid < 2) goto done;
        double bd0 = INFINITY, bd1 = INFINITY;
        for (size_t k = 0; k < N; k++) {
            if (!isfinite(vel[k])) continue;
            double d0 = (X[k] - x_min) * (X[k] - x_min) + (Y[k] - y_min) * (Y[k] - y_min);
            double d1 = (X[k] - x_max) * (X[k] - x_max) + (Y[k] - y_max) * (Y[k] - y_max);
            if (d0 < bd0) { bd0 = d0; i_min = (int64_t)k; }
            if (d1 < bd1) { bd1 = d1; i_max = (int64_t)k; }
        }
        if (i_min == i_max) goto done;
    } else {
        size_t cnt = 0;
        double vmin = 0, vmax = 0;
        for (size_t k = 0; k < N; k++) {
            if (!isfinite(vel[k]) || !(hypot(X[k] - xc, Y[k] - yc) <= patch_radius)) continue;
            cnt++;
            if (i_min < 0 || vel[k] < vmin) { vmin = vel[k]; i_min = (int64_t)k; }
            if (i_max < 0 || vel[k] > vmax) { vmax = vel[k]; i_max = (int64_t)k; }
        }
        if (cnt < 2) goto done;
        if (vel[i_min] >= 0 || vel[i_max] <= 0) goto done;
    }

    int32_t amin = (int32_t)(i_min / ngates), rmin = (int32_t)(i_min % ngates);
    int32_t amax = (int32_t)(i_max / ngates), rmax = (int32_t)(i_max % ngates);
    double mn_v = vel[i_min], mn_az = az[amin], mn_rng = rng_km[rmin], mn_x = X[i_min], mn_y = Y[i_min];
    double mx_v = vel[i_max], mx_az = az[amax], mx_rng = rng_km[rmax], mx_x = X[i_max], mx_y = Y[i_max];
    double min_radius = hypot(mn_x - xc, mn_y - yc), max_radius = hypot(mx_x - xc, mx_y - yc);
    double diam_km = hypot(mx_x - mn_x, mx_y - mn_y);
    if (diam_km <= 0) goto done;
    double axis_ang = bearing_deg(mx_x - mn_x, mx_y - mn_y);
    double mean_radius = 0.5 * (min_radius + max_radius);

    double asp = aspect_ratio_correction(beamwidth_deg, rc, diam_km);
    double min_adj = mn_v * asp, max_adj = mx_v * asp;
    double v_mean = 0.5 * (fabs(min_adj) + fabs(max_adj));
    double v_asym = fabs(max_adj) - fabs(min_adj);
    double delta_v = max_adj - min_adj;
    double d_m = diam_km * KM;
    double circ = (PI_ / 2.0) * delta_v * d_m;
    double delta_p = air_density * v_mean * v_mean;
    double sin_term = sin((axis_ang - bc) * D2R), cos_term = cos((axis_ang - bc) * D2R);

    /* radius of maximum vorticity from an annulus scan */
    double width = gurt_annulus_width(az, nrays, rng_km, ngates, rc);
    double vvort_max_radius = 0.0;
    if (width > 0) {
        double best = -1.0;
        double start = fmax(width, 0.25 * mean_radius), stop = 4.0 * mean_radius + 1e-9;
        double delta = (start + width) - start;                       /* np.arange's step */
        double cnt = ceil((stop - start) / width);
        for (int64_t i = 0; i < (int64_t)cnt; i++) {
            double rad = start + (double)i * delta;
            int64_t a, b;
            if (!ring_extrema(&sw, xc, yc, rad, width, &a, &b)) continue;
            double dia = hypot(X[b] - X[a], Y[b] - Y[a]);
            if (dia < 1.5 * width) continue;
            double ang = bearing_deg(X[b] - X[a], Y[b] - Y[a]);
            double vv = 2.0 * (vel[b] - vel[a]) / (dia * KM) * sin((ang - bc) * D2R);
            if (fabs(vv) > best) { best = fabs(vv); vvort_max_radius = dia / 2.0; }
        }
    }

    for (int l = 0; l < GURT_PATCH_NLEV; l++) {
        double frac = FRACS[l], rad = frac * mean_radius;
        double *row = levels + (size_t)l * 6;
        row[0] = frac; row[1] = rad;
        row[2] = row[3] = row[4] = row[5] = NAN;
        if (fabs(frac - 1.0) < 1e-9) {
            row[2] = 2.0 * delta_v / d_m * sin_term;
            row[3] = 2.0 * delta_v / d_m * cos_term;
            row[4] = -row[2] * row[3];
            row[5] = mean_radius * KM * v_mean;
        } else {
            int64_t a, b;
            if (width > 0 && ring_extrema(&sw, xc, yc, rad, width, &a, &b)) {
                double dia = hypot(X[b] - X[a], Y[b] - Y[a]);
                if (dia > 0) {
                    double ang = bearing_deg(X[b] - X[a], Y[b] - Y[a]), dv = vel[b] - vel[a];
                    row[2] = 2.0 * dv / (dia * KM) * sin((ang - bc) * D2R);
                    row[3] = 2.0 * dv / (dia * KM) * cos((ang - bc) * D2R);
                    row[4] = -row[2] * row[3];
                    row[5] = rad * KM * 0.5 * (fabs(vel[a]) + fabs(vel[b]));
                }
            }
        }
    }

    double beta = storm_dir_deg * D2R;
    double min_gr = mn_v + storm_speed_ms * sin(beta - mn_az * D2R);
    double max_gr = mx_v + storm_speed_ms * sin(beta - mx_az * D2R);
    double min_gr_adj = min_adj + storm_speed_ms * sin(beta - mn_az * D2R);
    double max_gr_adj = max_adj + storm_speed_ms * sin(beta - mx_az * D2R);

    double r[GURT_PATCH_NOUT] = {0};
    r[P_PATCH_RADIUS] = patch_radius;
    r[P_MIN_V] = mn_v; r[P_MIN_AZ] = mn_az; r[P_MIN_RNG] = mn_rng; r[P_MIN_X] = mn_x; r[P_MIN_Y] = mn_y;
    r[P_MAX_V] = mx_v; r[P_MAX_AZ] = mx_az; r[P_MAX_RNG] = mx_rng; r[P_MAX_X] = mx_x; r[P_MAX_Y] = mx_y;
    r[P_MIN_RADIUS] = min_radius; r[P_MAX_RADIUS] = max_radius; r[P_MEAN_RADIUS] = mean_radius;
    r[P_DIAM] = diam_km; r[P_AXIS] = axis_ang; r[P_ASPECT] = asp;
    r[P_MIN_ADJ] = min_adj; r[P_MAX_ADJ] = max_adj; r[P_V_MEAN] = v_mean; r[P_V_ASYM] = v_asym;
    r[P_DELTA_V] = delta_v; r[P_CIRC] = circ; r[P_DELTA_P] = delta_p; r[P_VORT_RADIUS] = vvort_max_radius;
    r[P_MIN_GR] = min_gr; r[P_MAX_GR] = max_gr; r[P_MIN_GR_ADJ] = min_gr_adj; r[P_MAX_GR_ADJ] = max_gr_adj;
    r[P_LARGEST_DOPPLER] = fmax(fabs(mn_v), fabs(mx_v));
    r[P_LARGEST_ADJ] = fmax(fabs(min_adj), fabs(max_adj));
    r[P_LARGEST_GROUND] = fmax(fabs(min_gr), fabs(max_gr));
    r[P_LARGEST_GROUND_ADJ] = fmax(fabs(min_gr_adj), fabs(max_gr_adj));
    memcpy(out, r, sizeof(r));
    *status = 0;
done:
    free(X); free(Y); free(az);
    return rcode;
}
