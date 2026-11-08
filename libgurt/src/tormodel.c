/* SPDX-License-Identifier: GPL-2.0-only
 * libgurt - Copyright (C) the GURT authors.
 * Algorithms ported from DREADER (GPL-2.0), tormodel / procmodel.c / interpolate.c /
 * coordinates.c (C. Alexander). This program is free software; you can redistribute it
 * and/or modify it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation. WITHOUT ANY WARRANTY; see the file LICENSE.
 *
 * Arithmetic follows the original statement by statement (float storage, double libm
 * calls) so results match DREADER's tormodel digit for digit.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "gurt.h"

#define TPI      3.14159265             /* DREADER's PI */
#define TDTR     (TPI / 180.0)
#define TRTD     (180.0 / TPI)
#define APPEARTH 8483000.0              /* 4/3 earth radius */
#define EARTHRAD 6371000.0

/* ---- coordinates.c ------------------------------------------------------------------- */

static void convert_angle(float *angle)
{
    if (*angle < 0) *angle = *angle + 360.0;
    *angle = 90.0 - *angle;                          /* meteorological coordinates */
    if (*angle < 0) *angle = *angle + 360.0;
}

static void convert_polar(float x, float y, float *mag, float *angle)
{
    *mag = sqrt(fabs(x * x + y * y));
    *angle = TRTD * atan2(y, x);
    convert_angle(angle);
}

static int convert_cart(float mag, float angle, float *x, float *y)
{
    if (mag < 0.0 || angle < 0.0) return -1;
    *x = mag * sinf(TDTR * angle);
    *y = mag * cosf(TDTR * angle);
    return 0;
}

/* ---- interpolate.c ------------------------------------------------------------------- */

static int calc_cubicspline(int32_t numpts, const int32_t *x, const float *y, float *coeffs)
{
    int32_t n1 = numpts - 1;
    float *buf = (float *)malloc((size_t)(4 * n1) * sizeof(float));
    if (!buf) return GURT_ENOMEM;
    float *dx = buf, *dydx = buf + n1, *sdx = buf + 2 * n1, *sdydx = buf + 3 * n1;

    for (int32_t i = 0; i <= numpts - 2; i++) {
        dx[i] = x[i + 1] - x[i];
        dydx[i] = (y[i + 1] - y[i]) / dx[i];
    }
    for (int32_t i = 1; i <= numpts - 2; i++) {
        if (i == 1) {
            sdx[1] = 2 * (dx[0] + dx[1]);
            sdydx[1] = 6 * (dydx[1] - dydx[0]);
        } else {
            sdx[i] = 2 * (dx[i] + dx[i - 1]) - (dx[i - 1] * dx[i - 1]) / sdx[i - 1];
            sdydx[i] = 6 * (dydx[i] - dydx[i - 1]) - (dx[i - 1] * sdydx[i - 1]) / sdx[i - 1];
        }
    }
    for (int32_t i = numpts - 1; i >= 0; i--) {
        if (i == numpts - 1 || i == 0) coeffs[i] = 0.0;
        else coeffs[i] = (sdydx[i] - dx[i] * coeffs[i + 1]) / sdx[i];
    }
    free(buf);
    return 0;
}

static void eval_cubicspline(int32_t numpts, const int32_t *x, const float *y, const float *coeff,
                             int32_t nint, const int32_t *xi, float *yi)
{
    for (int32_t j = 0; j < nint; j++) {
        int32_t i = numpts - 2;
        while (i > 0 && (xi[j] - x[i]) < 0) i--;
        float dx = x[i + 1] - x[i];
        float spline = (coeff[i] / 2) + ((xi[j] - x[i]) * (coeff[i + 1] - coeff[i])) / (6 * dx);
        spline = -(dx / 6) * (coeff[i + 1] + 2 * coeff[i]) + (y[i + 1] - y[i]) / dx + (xi[j] - x[i]) * spline;
        yi[j] = y[i] + (xi[j] - x[i]) * spline;
    }
}

static void eval_linearspline(int32_t numpts, const int32_t *x, const float *y,
                              int32_t nint, const int32_t *xi, float *yi)
{
    for (int32_t j = 0; j < nint; j++) {
        int32_t i = numpts - 2;
        while (i > 0 && (xi[j] - x[i]) < 0) i--;
        yi[j] = y[i] + (xi[j] - x[i]) * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]));
    }
}

/* ---- procmodel.c: interp_tornado ----------------------------------------------------- */

GURT_API int gurt_tor_interp(int32_t numobs, const int32_t *tobs, const float *obs,
                             int32_t numt, int32_t t0, int32_t dt, float *state)
{
    if (!tobs || !obs || !state || numobs < 2 || numt < 3 || dt <= 0) return GURT_EARG;
    for (int32_t i = 1; i < numobs; i++) if (tobs[i] <= tobs[i - 1]) return GURT_EARG;

    int rc = GURT_OK;
    float *ov = (float *)malloc((size_t)numobs * 2 * sizeof(float));
    float *iv = (float *)malloc((size_t)numt * sizeof(float));
    int32_t *it = (int32_t *)malloc((size_t)numt * sizeof(int32_t));
    float *px = (float *)malloc((size_t)numt * 2 * sizeof(float));
    float *coeffs = ov ? ov + numobs : NULL;            /* second half of ov */
    if (!ov || !iv || !it || !px) { rc = GURT_ENOMEM; goto done; }
    for (int32_t t = 0; t < numt; t++) it[t] = t0 + t * dt;

    /* positions: observation azimuth / range -> cartesian, cubic spline in time */
    float *ox = (float *)malloc((size_t)numobs * 2 * sizeof(float));
    if (!ox) { rc = GURT_ENOMEM; goto done; }
    for (int32_t o = 0; o < numobs; o++) {
        const float *r = obs + (size_t)o * GURT_TOR_NOBS;
        if (convert_cart(r[GURT_OBS_RANG], r[GURT_OBS_AZIM], &ox[o], &ox[numobs + o]) != 0) { free(ox); rc = GURT_EARG; goto done; }
    }
    for (int k = 0; k < 2; k++) {                       /* 0 = x, 1 = y */
        for (int32_t o = 0; o < numobs; o++) ov[o] = ox[k * numobs + o];
        if ((rc = calc_cubicspline(numobs, tobs, ov, coeffs)) != 0) { free(ox); goto done; }
        eval_cubicspline(numobs, tobs, ov, coeffs, numt, it, iv);
        for (int32_t t = 0; t < numt; t++) px[k * numt + t] = iv[t];
    }
    free(ox);

    for (int32_t t = 0; t < numt; t++) {
        float *s = state + (size_t)t * GURT_TOR_NSTATE;
        s[GURT_ST_X] = px[t];
        s[GURT_ST_Y] = px[numt + t];
        convert_polar(s[GURT_ST_X], s[GURT_ST_Y], &s[GURT_ST_RANG], &s[GURT_ST_AZIM]);
    }

    /* translation: 3-point (end points: 2nd-order one-sided) derivative of the position */
    for (int32_t t = 0; t < numt; t++) {
        float *s = state + (size_t)t * GURT_TOR_NSTATE;
        const float *p1 = state + (size_t)(t + 1) * GURT_TOR_NSTATE;
        const float *p2 = state + (size_t)(t + 2) * GURT_TOR_NSTATE;
        const float *m1 = state + (size_t)(t - 1) * GURT_TOR_NSTATE;
        const float *m2 = state + (size_t)(t - 2) * GURT_TOR_NSTATE;
        float dx, dy;
        int32_t d2 = 2 * dt;
        if (t == 0) {
            dx = 4 * (p1[GURT_ST_X]) - 3 * (s[GURT_ST_X]) - p2[GURT_ST_X];
            dy = 4 * (p1[GURT_ST_Y]) - 3 * (s[GURT_ST_Y]) - p2[GURT_ST_Y];
        } else if (t == numt - 1) {
            dx = 3 * (s[GURT_ST_X]) - 4 * (m1[GURT_ST_X]) + m2[GURT_ST_X];
            dy = 3 * (s[GURT_ST_Y]) - 4 * (m1[GURT_ST_Y]) + m2[GURT_ST_Y];
        } else {
            dx = p1[GURT_ST_X] - m1[GURT_ST_X];
            dy = p1[GURT_ST_Y] - m1[GURT_ST_Y];
        }
        convert_polar(dx / d2, dy / d2, &s[GURT_ST_TRANSVEL], &s[GURT_ST_TRANSANG]);
    }

    /* vortex size and wind-profile attributes: linear in time */
    static const int map[8][2] = {
        {GURT_OBS_RADIUS, GURT_ST_RADIUS}, {GURT_OBS_MAXTAN, GURT_ST_MAXTAN},
        {GURT_OBS_MAXRAD, GURT_ST_MAXRAD}, {GURT_OBS_CORETAN, GURT_ST_CORETAN},
        {GURT_OBS_DECAYTAN, GURT_ST_DECAYTAN}, {GURT_OBS_CORERAD, GURT_ST_CORERAD},
        {GURT_OBS_DECAYRAD, GURT_ST_DECAYRAD}, {-1, -1}};
    for (int m = 0; map[m][0] >= 0; m++) {
        for (int32_t o = 0; o < numobs; o++) {
            float v = obs[(size_t)o * GURT_TOR_NOBS + map[m][0]];
            ov[o] = (map[m][0] == GURT_OBS_RADIUS) ? fabsf(v) : v;
        }
        eval_linearspline(numobs, tobs, ov, numt, it, iv);
        for (int32_t t = 0; t < numt; t++) state[(size_t)t * GURT_TOR_NSTATE + map[m][1]] = iv[t];
    }
done:
    free(ov); free(iv); free(it); free(px);
    return rc;
}

/* ---- procmodel.c: calc_gridpoint ------------------------------------------------------ */

typedef struct { float tordist, posang, speed, direc; } Wind;

static void gridpoint_wind(const float *s, float xpos, float ypos, Wind *w)
{
    float dx = xpos - s[GURT_ST_X];
    float dy = ypos - s[GURT_ST_Y];
    float Vm, Vr, Vt, Vx, Vy;
    const float radius = s[GURT_ST_RADIUS];
    convert_polar(dx, dy, &w->tordist, &w->posang);

    if (w->tordist <= radius) {                          /* inside the radius of maximum wind: core */
        Vm = s[GURT_ST_TRANSVEL];
        Vr = (s[GURT_ST_MAXRAD]) * pow(w->tordist / radius, s[GURT_ST_CORERAD]);
        Vt = (s[GURT_ST_MAXTAN]) * pow(w->tordist / radius, s[GURT_ST_CORETAN]);
    } else {                                             /* outside: decay */
        Vm = s[GURT_ST_TRANSVEL];
        Vr = (s[GURT_ST_MAXRAD]) * pow(w->tordist / radius, s[GURT_ST_DECAYRAD]);
        Vt = (s[GURT_ST_MAXTAN]) * pow(w->tordist / radius, s[GURT_ST_DECAYTAN]);
    }
    Vx = Vm * sin(TDTR * (s[GURT_ST_TRANSANG])) - Vt * cos(TDTR * (w->posang)) + Vr * sin(TDTR * (w->posang));
    Vy = Vm * cos(TDTR * (s[GURT_ST_TRANSANG])) + Vt * sin(TDTR * (w->posang)) + Vr * cos(TDTR * (w->posang));

    convert_polar(-dx, -dy, &w->tordist, &w->posang);   /* tornado bearing from the grid point */
    convert_polar(-Vx, -Vy, &w->speed, &w->direc);      /* wind speed and meteorological direction */
}

static int check_grid(int32_t dx, int32_t numx, int32_t dy, int32_t numy)
{
    return (dx > 0 && dy > 0 && numx > 0 && numy > 0) ? 0 : -1;
}

GURT_API int gurt_tor_grid(const float *state, int32_t numt, int32_t tstep,
                           int32_t x0, int32_t dx, int32_t numx, int32_t y0, int32_t dy, int32_t numy,
                           float *speed, float *direc, float *tordist, float *posang)
{
    if (!state || tstep < 0 || tstep >= numt || check_grid(dx, numx, dy, numy)) return GURT_EARG;
    const float *s = state + (size_t)tstep * GURT_TOR_NSTATE;
    for (int32_t y = 0; y < numy; y++) {
        float ypos = y0 + y * dy;
        for (int32_t x = 0; x < numx; x++) {
            Wind w;
            gridpoint_wind(s, (float)(x0 + x * dx), ypos, &w);
            size_t k = (size_t)y * (size_t)numx + (size_t)x;
            if (speed)   speed[k] = w.speed;
            if (direc)   direc[k] = w.direc;
            if (tordist) tordist[k] = w.tordist;
            if (posang)  posang[k] = w.posang;
        }
    }
    return GURT_OK;
}

/* Maximum-wind grid. Same result as evaluating gridpoint_wind() for every cell and step (DREADER's maxgrid),
 * but cheaper in three exact ways: (1) the translation part of the wind is computed once per step, not per
 * cell; (2) only the wind speed is needed, so the bearing / distance bookkeeping of the full routine is
 * skipped; (3) a step is dropped without any trigonometry when |Vm| + |Vt| + |Vr| (an upper bound of the
 * speed) cannot beat the best value found so far, and each cell starts from the step that won in the cell
 * before it, so that bound is usually tight from the first step. Output is bit-identical to the plain loop. */
typedef struct { double ax, ay, vm; } StepC;

static float cell_speed_pruned(const float *s, const StepC *c, float xpos, float ypos, float best, int *skip)
{
    float dx = xpos - s[GURT_ST_X], dy = ypos - s[GURT_ST_Y];
    const float radius = s[GURT_ST_RADIUS];
    float dist = (float)sqrt(fabs(dx * dx + dy * dy));
    float Vr, Vt;
    int core = dist <= radius;
    if (dist == 0.0f) { *skip = 0; return -1.0f; }       /* degenerate: use the full routine */
    if (s[GURT_ST_MAXRAD] == 0.0f) Vr = 0.0f;            /* 0 * finite = 0, no pow needed */
    else Vr = (s[GURT_ST_MAXRAD]) * pow(dist / radius, core ? s[GURT_ST_CORERAD] : s[GURT_ST_DECAYRAD]);
    Vt = (s[GURT_ST_MAXTAN]) * pow(dist / radius, core ? s[GURT_ST_CORETAN] : s[GURT_ST_DECAYTAN]);
    {
        double ub = fabs(c->vm) + fabs((double)Vt) + fabs((double)Vr);
        if (ub * 1.00001 + 1e-4 <= (double)best) { *skip = 1; return 0.0f; }
    }
    *skip = 0;
    {
        float posang, Vx, Vy, mag, ang;
        convert_polar(dx, dy, &mag, &posang);
        Vx = c->ax - Vt * cos(TDTR * posang) + Vr * sin(TDTR * posang);
        Vy = c->ay + Vt * sin(TDTR * posang) + Vr * cos(TDTR * posang);
        convert_polar(-Vx, -Vy, &mag, &ang);
        return mag;
    }
}

GURT_API int gurt_tor_maxgrid(const float *state, int32_t numt, int32_t nsteps,
                              int32_t x0, int32_t dx, int32_t numx, int32_t y0, int32_t dy, int32_t numy,
                              float *maxspeed)
{
    if (!state || !maxspeed || nsteps < 0 || nsteps > numt || check_grid(dx, numx, dy, numy)) return GURT_EARG;
    StepC *sc = (StepC *)malloc((size_t)(nsteps > 0 ? nsteps : 1) * sizeof(StepC));
    if (!sc) return GURT_ENOMEM;
    for (int32_t t = 0; t < nsteps; t++) {
        const float *s = state + (size_t)t * GURT_TOR_NSTATE;
        float Vm = s[GURT_ST_TRANSVEL];
        sc[t].vm = Vm;
        sc[t].ax = Vm * sin(TDTR * (s[GURT_ST_TRANSANG]));
        sc[t].ay = Vm * cos(TDTR * (s[GURT_ST_TRANSANG]));
    }
    for (int32_t y = 0; y < numy; y++) {
        float ypos = y0 + y * dy;
        int32_t seed = 0;
        for (int32_t x = 0; x < numx; x++) {
            float xpos = (float)(x0 + x * dx), best = 0.0f;
            int32_t bt = 0;
            for (int32_t i = -1; i < nsteps; i++) {      /* i = -1: the neighbour's winning step first */
                int32_t t = (i < 0) ? (seed < nsteps ? seed : 0) : i;
                const float *s = state + (size_t)t * GURT_TOR_NSTATE;
                int skip = 0;
                float sp = cell_speed_pruned(s, &sc[t], xpos, ypos, best, &skip);
                if (sp < 0.0f) {                         /* dist == 0: exact original path */
                    Wind w;
                    gridpoint_wind(s, xpos, ypos, &w);
                    sp = w.speed;
                } else if (skip) {
                    continue;
                }
                /* the original scans t = 0.. with a strict '>': on a tie the earliest step wins, which the
                 * output value does not depend on (equal speed) */
                if (sp > best) { best = sp; bt = t; }
            }
            seed = bt;
            {
                Wind w;
                gridpoint_wind(state + (size_t)bt * GURT_TOR_NSTATE, xpos, ypos, &w);
                maxspeed[(size_t)y * (size_t)numx + (size_t)x] = w.speed;
            }
        }
    }
    free(sc);
    return GURT_OK;
}

GURT_API int gurt_tor_point(const float *state, int32_t numt, float xpos, float ypos, float *out)
{
    if (!state || !out || numt < 1) return GURT_EARG;
    for (int32_t t = 0; t < numt; t++) {
        Wind w;
        gridpoint_wind(state + (size_t)t * GURT_TOR_NSTATE, xpos, ypos, &w);
        float *o = out + (size_t)t * 4;
        o[0] = w.tordist; o[1] = w.posang; o[2] = w.speed; o[3] = w.direc;
    }
    return GURT_OK;
}

/* ---- coordinates.c: calc_latlon for a point on the ground (elevation 0) --------------- */

GURT_API int gurt_tor_latlon(float radar_lat, float radar_lon, float xpos, float ypos,
                             float *lat, float *lon)
{
    if (!lat || !lon) return GURT_EARG;
    float rang, azim, hdist, vdist, glat, glon;
    float elev = 0.0f;
    convert_polar(xpos, ypos, &rang, &azim);
    hdist = rang * cos(TDTR * elev);
    {   /* calc_cartesian: z = rang*sin(elev); vdist from the 4/3 earth radius */
        float z = rang * sin(TDTR * elev);
        vdist = sqrt(rang * rang + APPEARTH * APPEARTH + 2 * APPEARTH * z) - APPEARTH;
    }
    float greatcirc = asin(hdist / (vdist + EARTHRAD));
    float la = asin(sin(TDTR * radar_lat) * cos(greatcirc) + cos(TDTR * radar_lat) * sin(greatcirc) * cos(TDTR * azim));
    float lo = TDTR * radar_lon + atan2(sin(greatcirc) * cos(TDTR * radar_lat) * sin(TDTR * azim),
                                        cos(greatcirc) - sin(TDTR * radar_lat) * sin(la));
    while (lo > TPI || lo < -TPI) {
        if (lo > TPI) lo = lo - 2 * TPI; else lo = lo + 2 * TPI;
    }
    glat = TRTD * la;
    glon = TRTD * lo;
    *lat = glat; *lon = glon;
    return GURT_OK;
}

/* position -> (range, azimuth) with DREADER's convert_polar; exported so the Python layer
 * resolves "-azim/-rang" and "-xpos/-ypos" exactly like the original */
GURT_API int gurt_tor_polar(float x, float y, float *mag, float *angle)
{
    if (!mag || !angle) return GURT_EARG;
    convert_polar(x, y, mag, angle);
    return GURT_OK;
}

GURT_API int gurt_tor_cart(float mag, float angle, float *x, float *y)
{
    if (!x || !y) return GURT_EARG;
    return convert_cart(mag, angle, x, y) == 0 ? GURT_OK : GURT_EARG;
}
