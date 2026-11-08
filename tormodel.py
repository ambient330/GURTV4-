# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""Tornado wind model -- a port of DREADER's `tormodel` (C. Alexander, SourceForge "dreader").

Reads a parameter file of radar tornado observations plus an output grid specification and
writes a comma-delimited text file of the interpolated analytic wind field:

  point    time series of the wind at the grid point nearest an (x, y) or (azimuth, range)
  grid     (x, y) grid of the wind speed at one time
  maxgrid  (x, y) grid of the maximum wind speed up to a time

Parameter file (whitespace separated, '#' starts a comment, blank lines ignored)::

    RADAR  lat  lon  heading  elevation_m
    x0  dx  nx                 grid x:  start, step, number of steps (metres from the radar, east)
    y0  dy  ny                 grid y:  start, step, number of steps (metres from the radar, north)
    t0  dt  nt                 time:    start (epoch seconds, -1 = first observation), step (s),
                               number of steps (-1 = through the last observation)
    yyyy mm dd HH MM SS  azim rang radius maxtan coretan decaytan maxrad corerad decayrad
    ...                        one line per observation (UTC), at least two

``azim`` / ``rang`` give the vortex centre (degrees clockwise from north, metres), ``radius``
the radius of maximum wind (m), and for the tangential (``tan``) and radial (``rad``) wind
profiles the maximum speed (m/s) and the exponents inside (``core``) and outside (``decay``)
the radius of maximum wind: V(r) = Vmax * (r / radius) ** exponent. The wind is the vortex
translation (from its interpolated track) plus the tangential and radial parts.

Differences from DREADER's `tormodel` (deliberate):
  * grid mode takes the time step from the time in seconds divided by the step (DREADER used
    the seconds as an array index, which is only right when the step is 1 s);
  * maxgrid without a time covers the whole period (DREADER: just the first step);
  * the point header shows the position actually used, also when given as azimuth / range;
  * the output file is overwritten, not appended to; tabs count as spaces in the parameter file.
Everything else, including the digits printed, matches when libgurt is used (a NumPy fallback
agrees to rounding).
"""

import calendar
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np

OBS_COLS = ("azim", "rang", "radius", "maxtan", "coretan", "decaytan", "maxrad", "corerad", "decayrad")
ST = {n: i for i, n in enumerate(("x", "y", "azim", "rang", "radius", "transvel", "transang", "maxtan",
                                  "coretan", "decaytan", "maxrad", "corerad", "decayrad"))}


class TorError(ValueError):
    pass


EXAMPLE_PARAMS = """\
# GURT tormodel parameter file (DREADER format)
# radar   lat      lon       heading  elevation(m)
KOUN      35.2358  -97.4619  0.0      379
# x0   dx   nx
-4000  200  40
# y0   dy   ny
-1000  200  40
# t0 (epoch s, -1 = first obs)   dt (s)   nt (-1 = to the last obs)
-1  1  -1
# yyyy mm dd HH MM SS  azim  rang    radius  maxtan coretan decaytan maxrad corerad decayrad
2013 5 20 20 0 0       30.0  5000.0  150.0   60.0   1.0     -0.6      -10.0  1.0     -0.6
2013 5 20 20 1 0       32.0  5400.0  200.0   75.0   1.0     -0.6      -12.0  1.0     -0.6
2013 5 20 20 2 10      36.0  5900.0  260.0   90.0   1.2     -0.5      -15.0  1.0     -0.5
2013 5 20 20 3 0       41.0  6300.0  230.0   80.0   1.0     -0.6      -9.0   1.0     -0.6
"""


# --------------------------------------------------------------------------------- backends
def _native():
    try:
        from edit.native import native
    except Exception:                                  # running without the GURT package on the path
        return None
    nat = native()
    return nat if nat is not None and getattr(nat, "has_tormodel", lambda: False)() else None


def _convert_angle(a):
    a = np.where(a < 0, a + 360.0, a)
    a = 90.0 - a
    return np.where(a < 0, a + 360.0, a)


def _py_polar(x, y):
    return np.hypot(x, y), _convert_angle(np.degrees(np.arctan2(y, x)))


def _py_cart(rang, azim):
    return rang * np.sin(np.radians(azim)), rang * np.cos(np.radians(azim))


def _py_spline_coeffs(x, y):
    n = len(x)
    dx = np.diff(x).astype(float)
    dydx = np.diff(y) / dx
    sdx, sdy = np.zeros(n), np.zeros(n)
    for i in range(1, n - 1):
        if i == 1:
            sdx[1], sdy[1] = 2 * (dx[0] + dx[1]), 6 * (dydx[1] - dydx[0])
        else:
            sdx[i] = 2 * (dx[i] + dx[i - 1]) - dx[i - 1] ** 2 / sdx[i - 1]
            sdy[i] = 6 * (dydx[i] - dydx[i - 1]) - dx[i - 1] * sdy[i - 1] / sdx[i - 1]
    c = np.zeros(n)
    for i in range(n - 2, 0, -1):
        c[i] = (sdy[i] - dx[i] * c[i + 1]) / sdx[i]
    return c


def _py_segment(x, xi):
    return np.clip(np.searchsorted(x, xi, side="right") - 1, 0, len(x) - 2)


def _py_interp(tobs, obs, numt, t0, dt):
    tobs = np.asarray(tobs, float)
    obs = np.asarray(obs, float)
    ti = t0 + dt * np.arange(numt)
    seg = _py_segment(tobs, ti)
    ox, oy = _py_cart(obs[:, 1], obs[:, 0])
    pos = []
    for y in (ox, oy):
        c = _py_spline_coeffs(tobs, y)
        d = tobs[seg + 1] - tobs[seg]
        s = c[seg] / 2 + (ti - tobs[seg]) * (c[seg + 1] - c[seg]) / (6 * d)
        s = -(d / 6) * (c[seg + 1] + 2 * c[seg]) + (y[seg + 1] - y[seg]) / d + (ti - tobs[seg]) * s
        pos.append(y[seg] + (ti - tobs[seg]) * s)
    st = np.zeros((numt, len(ST)))
    st[:, ST["x"]], st[:, ST["y"]] = pos
    st[:, ST["rang"]], st[:, ST["azim"]] = _py_polar(pos[0], pos[1])
    dx, dy = np.empty(numt), np.empty(numt)
    dx[1:-1], dy[1:-1] = pos[0][2:] - pos[0][:-2], pos[1][2:] - pos[1][:-2]
    dx[0], dy[0] = 4 * pos[0][1] - 3 * pos[0][0] - pos[0][2], 4 * pos[1][1] - 3 * pos[1][0] - pos[1][2]
    dx[-1], dy[-1] = 3 * pos[0][-1] - 4 * pos[0][-2] + pos[0][-3], 3 * pos[1][-1] - 4 * pos[1][-2] + pos[1][-3]
    st[:, ST["transvel"]], st[:, ST["transang"]] = _py_polar(dx / (2 * dt), dy / (2 * dt))
    for name in ("radius", "maxtan", "maxrad", "coretan", "decaytan", "corerad", "decayrad"):
        y = obs[:, OBS_COLS.index(name)]
        y = np.abs(y) if name == "radius" else y
        st[:, ST[name]] = y[seg] + (ti - tobs[seg]) * ((y[seg + 1] - y[seg]) / (tobs[seg + 1] - tobs[seg]))
    return st.astype(np.float32)


def _py_wind(s, xpos, ypos):
    """Wind at points (arrays) for one state row: returns tordist, posang, speed, direc."""
    dx, dy = xpos - float(s[ST["x"]]), ypos - float(s[ST["y"]])
    dist, ang = _py_polar(dx, dy)
    R = float(s[ST["radius"]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = dist / R
        core = dist <= R
        vr = float(s[ST["maxrad"]]) * np.where(core, ratio ** float(s[ST["corerad"]]), ratio ** float(s[ST["decayrad"]]))
        vt = float(s[ST["maxtan"]]) * np.where(core, ratio ** float(s[ST["coretan"]]), ratio ** float(s[ST["decaytan"]]))
    vm, ta = float(s[ST["transvel"]]), np.radians(float(s[ST["transang"]]))
    pa = np.radians(ang)
    vx = vm * np.sin(ta) - vt * np.cos(pa) + vr * np.sin(pa)
    vy = vm * np.cos(ta) + vt * np.sin(pa) + vr * np.cos(pa)
    tdist, tang = _py_polar(-dx, -dy)
    spd, dire = _py_polar(-vx, -vy)
    return tdist, tang, spd, dire


def _py_latlon(rlat, rlon, x, y):
    rang, azim = _py_polar(x, y)
    vdist = math.sqrt(rang * rang + 8483000.0 ** 2) - 8483000.0
    gc = math.asin(rang / (vdist + 6371000.0))
    la = math.asin(math.sin(math.radians(rlat)) * math.cos(gc) +
                   math.cos(math.radians(rlat)) * math.sin(gc) * math.cos(math.radians(azim)))
    lo = math.radians(rlon) + math.atan2(math.sin(gc) * math.cos(math.radians(rlat)) * math.sin(math.radians(azim)),
                                         math.cos(gc) - math.sin(math.radians(rlat)) * math.sin(la))
    lo = (lo + math.pi) % (2 * math.pi) - math.pi
    return math.degrees(la), math.degrees(lo)


# --------------------------------------------------------------------------------- parameters
def _tokens(line):
    line = line.split("#", 1)[0]
    return line.split()


def parse_params(text):
    """Parse a DREADER tormodel parameter file (text). Returns a dict."""
    rows = [t for t in (_tokens(l) for l in text.splitlines()) if t]
    if len(rows) < 6:
        raise TorError("parameter file needs a radar line, x, y and t grid lines and at least two observations")
    try:
        f32 = lambda v: float(np.float32(float(v)))        # DREADER stores these as float    # noqa: E731
        radar = {"name": rows[0][0], "lat": f32(rows[0][1]), "lon": f32(rows[0][2]),
                 "head": f32(rows[0][3]), "elev": int(float(rows[0][4]))}
    except (IndexError, ValueError):
        raise TorError("malformed radar line (need: name lat lon heading elevation)")
    grid = {}
    for key, row, what in (("x", rows[1], "x"), ("y", rows[2], "y"), ("t", rows[3], "t")):
        try:
            grid[key + "0"], grid["d" + key], grid["n" + key] = (int(float(v)) for v in row[:3])
        except (IndexError, ValueError):
            raise TorError(f"malformed {what} dimension line (need: start step count)")
    obs, times = [], []
    for k, row in enumerate(rows[4:], 1):
        try:
            y, mo, d, h, mi, s = (int(float(v)) for v in row[:6])
            vals = [float(v) for v in row[6:15]]
            if len(vals) != 9:
                raise ValueError
            times.append(calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0)))
        except (IndexError, ValueError, OverflowError):
            raise TorError(f"malformed radar observation {k} (need: yyyy mm dd HH MM SS + 9 values)")
        obs.append(vals)
    if len(obs) < 2:
        raise TorError("at least two radar observations are needed")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise TorError("observation times must be strictly increasing")
    return {"radar": radar, "grid": grid, "times": times, "obs": np.array(obs, np.float32)}


# --------------------------------------------------------------------------------- track smoothing
def _loess(t, y, h, iters=2):
    """Robust local-linear (LOESS, tricube weights, bisquare robustness) smoothing of y(t) evaluated at
    the sample times. h = half-width of the window in seconds (widened to at least 6 samples)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(t)
    rob = np.ones(n)
    out = y.copy()
    for it in range(iters + 1):
        for i in range(n):
            d = np.abs(t - t[i])
            hh = max(h, float(np.sort(d)[min(5, n - 1)]) * 1.000001, 1e-9)
            w = np.clip(1.0 - (d / hh) ** 3, 0.0, None) ** 3 * rob
            if w.sum() <= 1e-12:
                w = np.clip(1.0 - (d / hh) ** 3, 0.0, None) ** 3
            sw = w.sum()
            tm = t - t[i]
            sx, sxx = (w * tm).sum(), (w * tm * tm).sum()
            sy, sxy = (w * y).sum(), (w * tm * y).sum()
            den = sw * sxx - sx * sx
            out[i] = (sy * sxx - sx * sxy) / den if abs(den) > 1e-12 * max(1.0, sw * sxx) else sy / sw
        res = y - out
        sc = float(np.median(np.abs(res)))
        if sc <= 0:
            break
        u = res / (6.0 * sc)
        rob = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
    return out


def _despike(t, x, y, h, min_dev=150.0, k=4.0):
    """Boolean mask of the observations to keep: those whose position lies within max(min_dev, k robust sigma)
    metres of a robust smooth of the track (bad detections form short runs of positions hundreds of metres to
    kilometres off the path). Repeated until nothing more is rejected."""
    t = np.asarray(t, float)
    keep = np.ones(len(t), bool)
    for _ in range(4):
        idx = np.nonzero(keep)[0]
        if len(idx) < 6:
            break
        fx, fy = _loess(t[idx], np.asarray(x)[idx], h, 3), _loess(t[idx], np.asarray(y)[idx], h, 3)
        dev = np.hypot(np.asarray(x)[idx] - fx, np.asarray(y)[idx] - fy)
        thr = max(min_dev, k * 1.4826 * float(np.median(dev)))
        bad = dev > thr
        if not bad.any():
            break
        keep[idx[bad]] = False
    return keep


def _select(t, x, y, h, max_speed=60.0):
    """Mask of the observations to keep: bad detections removed (_despike), then the track cut at any jump faster
    than max_speed m/s (a mislinked detection, not a tornado), keeping the longest continuous segment. The jump
    speed compares the median positions of the three observations either side of each gap, so ordinary detection
    jitter does not count. max_speed <= 0 disables the cut."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = _despike(t, x, y, h)
    if not (max_speed and max_speed > 0):
        return keep
    for _ in range(8):
        idx = np.nonzero(keep)[0]
        n = len(idx)
        if n < 4:
            break
        v = np.zeros(n - 1)
        for i in range(n - 1):
            L, R = idx[max(0, i - 2):i + 1], idx[i + 1:i + 4]
            dt = float(np.median(t[R]) - np.median(t[L]))
            v[i] = np.hypot(np.median(x[R]) - np.median(x[L]), np.median(y[R]) - np.median(y[L])) / max(dt, 1e-9)
        hot = v > max_speed
        if not hot.any():
            break
        cuts, i = [], 0
        while i < n - 1:                                  # one cut per run of hot gaps: the fastest
            if hot[i]:
                j = i
                while j + 1 < n - 1 and hot[j + 1]:
                    j += 1
                cuts.append(i + int(np.argmax(v[i:j + 1])))
                i = j + 1
            else:
                i += 1
        edges = [0] + [c + 1 for c in cuts] + [n]
        segs = [idx[a:b] for a, b in zip(edges[:-1], edges[1:]) if b > a]
        best = max(segs, key=lambda sg: (t[sg[-1]] - t[sg[0]], len(sg)))
        keep = np.zeros(len(t), bool)
        keep[best] = True
    return keep


def _track_speed(t, x, y):
    t, x, y = (np.asarray(a, float) for a in (t, x, y))
    return np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 1e-9)


def smooth_params(params, smooth_s=60.0, max_speed=60.0):
    """Smooth the observations of a parameter set: the centre track (x, y), the radius and the tangential /
    radial maxima are fitted with a robust local-linear regression over +-smooth_s seconds, so detection
    jitter cannot make the spline track jump (the model adds the track's translation speed to the wind, so a
    track that jumps hundreds of metres in seconds gives hundreds of m/s). 0 = no change. Returns new params
    plus a dict with the raw / smoothed maximum track speed."""
    if not smooth_s or smooth_s <= 0:
        return params, {"raw_max_speed": None, "max_speed": None}
    o = np.array(params["obs"], np.float64)
    times = np.array(params["times"], float)
    t = times - times[0]
    az = np.radians(o[:, 0])
    x, y = o[:, 1] * np.sin(az), o[:, 1] * np.cos(az)
    raw = float(_track_speed(t, x, y).max())
    keep = _select(t, x, y, smooth_s, max_speed)
    n_drop = int((~keep).sum())
    if n_drop:
        o, times, t, x, y = o[keep], times[keep], t[keep], x[keep], y[keep]
    xs, ys = _loess(t, x, smooth_s), _loess(t, y, smooth_s)
    o[:, 2] = np.maximum(_loess(t, o[:, 2], smooth_s), 1.0)
    o[:, 3] = np.maximum(_loess(t, o[:, 3], smooth_s), 0.0)
    o[:, 6] = _loess(t, o[:, 6], smooth_s)
    o[:, 0] = np.degrees(np.arctan2(xs, ys)) % 360.0
    o[:, 1] = np.hypot(xs, ys)
    out = dict(params)
    out["obs"] = o.astype(np.float32)
    out["times"] = [int(v) for v in times]
    return out, {"raw_max_speed": raw, "max_speed": float(_track_speed(t, xs, ys).max()), "dropped": n_drop}


def params_text(params):
    """Parameter set -> text of a DREADER tormodel parameter file."""
    r, g = params["radar"], params["grid"]
    lines = ["# GURT tormodel parameter file (DREADER format)",
             f"{r['name']} {r['lat']:.6f} {r['lon']:.6f} {r['head']:.1f} {int(r['elev'])}",
             f"{g['x0']} {g['dx']} {g['nx']}", f"{g['y0']} {g['dy']} {g['ny']}", f"{g['t0']} {g['dt']} {g['nt']}"]
    for tt, row in zip(params["times"], params["obs"]):
        tm = datetime.fromtimestamp(int(tt), timezone.utc)
        lines.append("%04d %02d %02d %02d %02d %02d  %.3f %.1f %.1f %.2f %.2f %.2f %.2f %.2f %.2f" % (
            tm.year, tm.month, tm.day, tm.hour, tm.minute, tm.second, *[float(v) for v in row]))
    return "\n".join(lines) + "\n"


def apply_overrides(params, res=None, dt=None):
    """Change the output grid without editing the file: res = new x and y spacing in metres (the grid
    keeps its extent, rounded to whole cells), dt = new time step in seconds. Returns new params."""
    g = dict(params["grid"])
    if res is not None:
        res = int(round(res))
        if res <= 0:
            raise TorError("grid resolution must be a positive number of metres")
        for a, d, n in (("x0", "dx", "nx"), ("y0", "dy", "ny")):
            span = g[d] * g[n]
            g[n] = max(1, int(round(span / res)))
            g[d] = res
    if dt is not None:
        if int(dt) <= 0:
            raise TorError("time step must be a positive number of seconds")
        g["dt"] = int(dt)
    out = dict(params)
    out["grid"] = g
    return out


def load_params(path, res=None, dt=None, smooth=None):
    with open(path, "r", errors="replace") as fh:
        p = parse_params(fh.read())
    if smooth:
        p = smooth_params(p, smooth)[0]
    return apply_overrides(p, res, dt) if (res is not None or dt is not None) else p


# --------------------------------------------------------------------------------- the model
class TorModel:
    """Interpolated tornado track plus the analytic wind. Build with TorModel(params)."""

    def __init__(self, params, use_native=True):
        g = dict(params["grid"])
        times, obs = params["times"], params["obs"]
        if min(g["dx"], g["dy"], g["dt"]) <= 0:
            raise TorError(f"grid x, y, t steps {g['dx']} {g['dy']} {g['dt']} must be positive integers")
        if g["nx"] < 0 or g["ny"] < 0:
            raise TorError(f"number of grid space steps {g['nx']} {g['ny']} must be non-negative")
        first, last = times[0], times[-1]
        if g["t0"] < 0:
            start = first
        elif first <= g["t0"] <= last:
            start = g["t0"]
        else:
            raise TorError(f"grid start time {g['t0']} must be between {first} and {last} seconds")
        if g["nt"] < 0:
            end = last
        elif start + g["dt"] * g["nt"] <= last:
            end = start + g["dt"] * g["nt"]
        else:
            raise TorError(f"grid end time {start + g['dt'] * g['nt']} must be between {first} and {last} seconds")
        self.numt = (end - start + 1) // g["dt"]
        if self.numt < 3:
            raise TorError("the time axis needs at least 3 steps (decrease dt or lengthen the observations)")
        self.numx, self.numy = g["nx"] + 1, g["ny"] + 1
        self.grid, self.radar = g, params["radar"]
        self.stime = start
        self.times = start + g["dt"] * np.arange(self.numt, dtype=np.int64)
        for k in range(len(obs)):
            if obs[k, 0] < 0 or obs[k, 1] < 0:
                raise TorError(f"polar coordinate magnitude {obs[k, 1]:.1f} and angle {obs[k, 0]:.1f} must be non-negative")
        rel = np.array(times, np.int64) - first
        self._nat = _native() if use_native else None
        if self._nat is not None:
            self.state = self._nat.tor_interp(rel, obs, self.numt, start - first, g["dt"])
        else:
            self.state = _py_interp(rel, obs, self.numt, start - first, g["dt"])

    # grid coordinates (float32 like DREADER)
    def _xs(self):
        return (self.grid["x0"] + self.grid["dx"] * np.arange(self.numx)).astype(np.float32)

    def _ys(self):
        return (self.grid["y0"] + self.grid["dy"] * np.arange(self.numy)).astype(np.float32)

    def step_for(self, t_seconds):
        """(time step index, number of steps up to t, effective seconds) like DREADER's -time."""
        dt, numt = self.grid["dt"], self.numt
        if t_seconds < 0 or t_seconds > dt * numt:
            return numt - 1, numt, dt * numt
        n = int(round(t_seconds / dt))
        return min(n, numt - 1), n, t_seconds

    def latlon(self, x, y):
        if self._nat is not None:
            return self._nat.tor_latlon(self.radar["lat"], self.radar["lon"], x, y)
        return _py_latlon(self.radar["lat"], self.radar["lon"], x, y)

    def polar(self, x, y):
        if self._nat is not None:
            return self._nat.tor_polar(x, y)
        r, a = _py_polar(np.float32(x), np.float32(y))
        return float(r), float(a)

    def cart(self, rang, azim):
        if rang < 0 or azim < 0:
            raise TorError(f"polar coordinate magnitude {rang:.1f} and angle {azim:.1f} must be non-negative")
        if self._nat is not None:
            return self._nat.tor_cart(rang, azim)
        x, y = _py_cart(np.float32(rang), np.float32(azim))
        return float(x), float(y)

    def locate(self, xpos=None, ypos=None, azim=None, rang=None):
        """DREADER locate_gridpoint: the grid node nearest a point. Returns (ix, iy, x_node, y_node)."""
        g = self.grid
        if azim is None and rang is None and xpos is not None:
            x, y = float(xpos), float(ypos if ypos is not None else 0.0)
        elif azim is None and rang is None:
            x, y = 0.0, 0.0
        else:
            x, y = self.cart(abs(rang or 0.0), abs(azim or 0.0))
        if x < g["x0"] or x > g["x0"] + g["dx"] * g["nx"]:
            raise TorError(f"x position {x:.1f} must be between {g['x0']} and {g['x0'] + g['dx'] * g['nx']}")
        if y < g["y0"] or y > g["y0"] + g["dy"] * g["ny"]:
            raise TorError(f"y position {y:.1f} must be between {g['y0']} and {g['y0'] + g['dy'] * g['ny']}")
        ix = int(np.rint((x - g["x0"]) / g["dx"]))
        iy = int(np.rint((y - g["y0"]) / g["dy"]))
        return ix, iy, x, y

    # ---- products
    def point(self, xnode, ynode):
        """[numt, 4]: tornado distance, tornado direction, wind speed, wind direction."""
        if self._nat is not None:
            return self._nat.tor_point(self.state, xnode, ynode)
        out = np.empty((self.numt, 4), np.float32)
        for t in range(self.numt):
            out[t] = [float(v) for v in _py_wind(self.state[t], np.float64(xnode), np.float64(ynode))]
        return out

    def grid_at(self, tstep, want=("speed",)):
        g = self.grid
        if self._nat is not None:
            return self._nat.tor_grid(self.state, tstep, g["x0"], g["dx"], self.numx, g["y0"], g["dy"], self.numy, want)
        X, Y = np.meshgrid(self._xs(), self._ys())
        td, pa, sp, di = _py_wind(self.state[tstep], X.astype(np.float64), Y.astype(np.float64))
        allv = {"tordist": td, "posang": pa, "speed": sp, "direc": di}
        return {n: allv[n].astype(np.float32) for n in want}

    def maxgrid(self, nsteps, first=0):
        """Maximum wind speed over time steps first .. first + nsteps - 1 (default: from the start)."""
        g = self.grid
        if first < 0 or first + nsteps > self.numt:
            raise TorError(f"time steps {first}..{first + nsteps - 1} are outside 0..{self.numt - 1}")
        if self._nat is not None:
            st = self.state[first:first + nsteps] if first else self.state
            return self._nat.tor_maxgrid(st, nsteps, g["x0"], g["dx"], self.numx, g["y0"], g["dy"], self.numy)
        X, Y = np.meshgrid(self._xs(), self._ys())
        best = _py_wind(self.state[first], X.astype(float), Y.astype(float))[2]
        for t in range(first + 1, first + nsteps):
            best = np.maximum(best, _py_wind(self.state[t], X.astype(float), Y.astype(float))[2])
        return best.astype(np.float32)

    def window_steps(self, t_from, t_to):
        """Time steps whose time lies in [t_from, t_to] (epoch seconds): (first, nsteps), or None if
        the interval does not overlap the model period."""
        dt = self.grid["dt"]
        i0 = max(0, -((self.stime - t_from) // dt))                      # ceil((t_from - stime) / dt)
        i1 = min(self.numt - 1, int((t_to - self.stime) // dt))
        if i1 < i0 or t_to < self.stime or t_from > self.times[-1]:
            return None
        return int(i0), int(i1 - i0 + 1)


# --------------------------------------------------------------------------------- text output
def grid_header(m):
    r, g = m.radar, m.grid
    return (f"Radar: {r['name'][:8]:>8}, Lat: {r['lat']:16.6f}, Lon: {r['lon']:16.6f}, Heading: {r['head']:8.1f}, "
            f"Grid Elev: {r['elev']:8d} (m),\n"
            f"X-Begin: {g['x0']:8d} m, X-End: {g['x0'] + g['dx'] * g['nx']:8d} m, X-Step: {g['dx']:8d} m,\n"
            f"Y-Begin: {g['y0']:8d} m, Y-End: {g['y0'] + g['dy'] * g['ny']:8d} m, Y-Step: {g['dy']:8d} m,\n"
            f"T-Begin: {0:8d} s, T-End: {g['dt'] * m.numt - 1:8d} s, T-Step: {g['dt']:8d} s,\n")


def point_header(m, xpos, ypos):
    r = m.radar
    return (f"Radar: {r['name'][:8]:>8}, Lat: {r['lat']:16.5f}, Lon: {r['lon']:16.5f}, Heading: {r['head']:6.1f}, "
            f"Grid Elev: {r['elev']:8d} (m), Grid Point Nearest to, X-Coordinate: {xpos:9.1f} (m), "
            f"Y-Coordinate: {ypos:9.1f} (m),\n"
            "Mo/Dy/Year Hr:Mn:Sc,JulianTime---(s),TornadoAzim--(m),TornadoRange-(m),RadiusMaxTVel(m),"
            "TransSpeed-(m/s),TransDirect--(*),MaxTanVeloc(m/s),TanVelCoreExpon-,TanVelDecayExpon,"
            "MaxRadVeloc(m/s),RadVelCoreExpon-,RadVelDecayExpon,TornadoDist--(m),TorandoDirect(*),"
            "WindSpeed--(m/s),WindDirection(*),\n")


def render_point(m, xpos=None, ypos=None, azim=None, rang=None, header=False):
    ix, iy, x, y = m.locate(xpos, ypos, azim, rang)
    xn, yn = m.grid["x0"] + ix * m.grid["dx"], m.grid["y0"] + iy * m.grid["dy"]
    w = m.point(xn, yn)
    out = [point_header(m, x, y)] if header else []
    s = m.state
    f = lambda v: float(v)                                    # noqa: E731
    for t in range(m.numt):
        tm = datetime.fromtimestamp(int(m.times[t]), timezone.utc)
        row = s[t]
        out.append("%02d/%02d/%4d %02d:%02d:%02d,%16d," % (tm.month, tm.day, tm.year, tm.hour, tm.minute, tm.second,
                                                          int(m.times[t])))
        out.append("%16.1f,%16.1f,%16.1f," % (f(row[ST["azim"]]), f(row[ST["rang"]]), f(row[ST["radius"]])))
        out.append("%16.1f,%16.1f," % (f(row[ST["transvel"]]), f(row[ST["transang"]])))
        out.append("%16.1f,%16.2f,%16.2f," % (f(row[ST["maxtan"]]), f(row[ST["coretan"]]), f(row[ST["decaytan"]])))
        out.append("%16.1f,%16.2f,%16.2f," % (f(row[ST["maxrad"]]), f(row[ST["corerad"]]), f(row[ST["decayrad"]])))
        out.append("%16.1f,%16.1f,%16.1f,%16.1f,\n" % tuple(f(v) for v in w[t]))
    return "".join(out)


def render_grid(m, time=0, maxgrid=False, header=False, window=None):
    """`time` in seconds from the start. grid mode: the wind at that time; maxgrid: the maximum
    through that time (negative or beyond the end = the whole period)."""
    g = m.grid
    tstep, nsteps, tsec = m.step_for(time)
    out = []
    if header:
        out.append(grid_header(m))
    if maxgrid and window is not None:
        speed = m.maxgrid(window[1], window[0])
        out.append(f"Maximum {g['dt']}-sec wind speed (m/s) from {window[0] * g['dt']} through "
                   f"{(window[0] + window[1] - 1) * g['dt']} sec.\n\n")
    elif maxgrid:
        speed = m.maxgrid(nsteps)
        out.append(f"Maximum {g['dt']}-sec wind speed (m/s) through {tsec} sec.\n\n")
    else:
        speed = m.grid_at(tstep)["speed"]
        out.append(f"The {g['dt']}-sec wind speed (m/s) at {tsec} sec.\n\n")
    xs, ys = m._xs(), m._ys()
    if header:
        lon = [m.latlon(float(x), float(g["y0"]))[1] for x in xs]
        out.append("Coordinates  (*)," + "".join("%16.6f," % v for v in lon) + "\n")
        out.append("Coordinates  (m)," + "".join("%16d," % int(x) for x in xs) + "\n")
    for iy in range(m.numy - 1, -1, -1):
        if header:
            out.append("%16.6f,%16d," % (m.latlon(float(g["x0"]), float(ys[iy]))[0], int(ys[iy])))
        out.append("".join("%16.1f," % float(v) for v in speed[iy]) + "\n")
    return "".join(out)


# --------------------------------------------------------------------------------- swaths
def swath_grid(m, time=-1, maxgrid=True, window=None):
    """The wind-speed grid as a swath dict like the box-tracking wind swath:
    {'M': (ny, nx) m/s, 'x0', 'y0' (metres, node of the first cell), 'res_x', 'res_y' (m), 'ut' (epoch s)}.
    maxgrid=True: maximum through `time` seconds (negative / beyond the end = whole period);
    False: the wind at `time`. Row 0 is the southernmost row; cell centres sit on the grid nodes."""
    tstep, nsteps, tsec = m.step_for(time)
    if maxgrid and window is not None:                  # (first step, number of steps)
        M = np.asarray(m.maxgrid(window[1], window[0]), np.float64)
        ut = int(m.times[window[0]])
    elif maxgrid:
        M = np.asarray(m.maxgrid(nsteps), np.float64)
        ut = int(m.stime)
    else:
        M = np.asarray(m.grid_at(tstep)["speed"], np.float64)
        ut = int(m.times[tstep])
    g = m.grid
    return {"M": M, "x0": float(g["x0"]), "y0": float(g["y0"]), "res_x": float(g["dx"]),
            "res_y": float(g["dy"]), "ut": ut}


def swath_path(path):
    """Swath files are always named swath.<name>: the basename gets a leading 'swath.' unless it has one
    (25mtornado.txt -> swath.25mtornado.txt)."""
    d, b = os.path.split(str(path))
    if not b.startswith("swath."):
        b = "swath." + b
    return os.path.join(d, b)


def swath_ascii(m, path, time=-1, maxgrid=True, sensor=None, product=None, unit="m/s", fill=0.0,
                south_first=False, window=None):
    """Write the grid as a swath ASCII text file, the same layout GURT's box-tracking wind swath
    export uses: header lines `KEY value` (UT, XG / YG = radar lon / lat, ZG = radar elevation,
    X0 / Y0 = grid origin in metres east / north of XG, YG, nx, ny, dx, dy, sn, pr, un), then ny
    rows of nx space-delimited values (north row first; south_first=True flips it. X0 / Y0 always locate the south-west corner of the grid).
    CRLF line endings. The file is named swath.<name> (swath_path); returns the path written."""
    path = swath_path(path)
    sw = swath_grid(m, time, maxgrid, window)
    M = np.where(np.isfinite(sw["M"]), sw["M"], fill)
    if not south_first:
        M = M[::-1]
    ny, nx = M.shape
    r = m.radar

    def num(v):
        return f"{int(round(v))}" if abs(v - round(v)) < 1e-6 else f"{v:.3f}"

    y0 = sw["y0"]                       # south edge, whichever row comes first
    head = [("UT", str(sw["ut"])), ("XG", f"{r['lon']:.6f}"), ("YG", f"{r['lat']:.6f}"),
            ("ZG", f"{float(r['elev']):.2f}"), ("X0", num(sw["x0"])), ("Y0", num(y0)),
            ("nx", str(nx)), ("ny", str(ny)), ("dx", num(sw["res_x"])), ("dy", num(sw["res_y"])),
            ("sn", sensor or r["name"] or "Radar"), ("pr", product or ("vmax" if maxgrid else "v")), ("un", unit)]
    with open(path, "w", newline="") as fh:
        for k, v in head:
            fh.write(f"{k} {v}\r\n")
        for row in M:
            fh.write(" ".join(f"{x:.4f}" for x in row) + "\r\n")
    return path


# --------------------------------------------------------------------------------- folders
def volume_time_span(vol):
    """(first, last) UTC epoch seconds covered by a loaded radar volume, from per-ray times when the
    file has them, else from the sweep / volume scan time. None when no time is known."""
    lo, hi = [], []
    for sw in getattr(vol, "sweeps", []):
        rt = getattr(sw, "ray_time", None)
        if rt is not None and len(rt):
            a = np.asarray(rt, float)
            a = a[np.isfinite(a) & (a >= 0)]
            st0 = getattr(sw, "scan_time", None) or getattr(vol, "scan_time", None)
            if a.size and a.max() > 1.0e8:                      # epoch seconds
                lo.append(float(a.min()))
                hi.append(float(a.max()))
                continue
            if a.size and st0 is not None:                      # offsets from the sweep start
                try:
                    if st0.tzinfo is None:
                        st0 = st0.replace(tzinfo=timezone.utc)
                    base = st0.timestamp()
                    lo.append(base + float(a.min()))
                    hi.append(base + float(a.max()))
                    continue
                except Exception:
                    pass
        st = getattr(sw, "scan_time", None) or getattr(vol, "scan_time", None)
        if st is not None:
            try:
                if st.tzinfo is None:
                    st = st.replace(tzinfo=timezone.utc)
                lo.append(st.timestamp())
                hi.append(st.timestamp())
            except Exception:
                pass
    if not lo:
        st = getattr(vol, "scan_time", None)
        if st is not None:
            try:
                if st.tzinfo is None:
                    st = st.replace(tzinfo=timezone.utc)
                lo = hi = [st.timestamp()]
            except Exception:
                return None
    return (min(lo), max(hi)) if lo else None


def folder_window(m, spans, time_offset=0.0):
    """Model time window covered by a folder: spans = [(path, (first, last) | None), ...].
    Returns (first_step, nsteps, used, skipped) where `used` lists the paths whose scans overlap the
    model period and `skipped` the (path, reason) pairs that did not. Raises TorError if none overlap."""
    used, skipped = [], []
    t0 = t1 = None
    for path, sp in spans:
        if sp is None:
            skipped.append((path, "no scan time"))
            continue
        a, b = sp[0] + time_offset, sp[1] + time_offset
        if m.window_steps(a, b) is None:
            skipped.append((path, "outside the model period"))
            continue
        used.append(path)
        t0 = a if t0 is None else min(t0, a)
        t1 = b if t1 is None else max(t1, b)
    if not used:
        raise TorError("none of the files overlap the model period "
                       f"({datetime.fromtimestamp(int(m.times[0]), timezone.utc):%Y-%m-%d %H:%M:%S} to "
                       f"{datetime.fromtimestamp(int(m.times[-1]), timezone.utc):%Y-%m-%d %H:%M:%S} UTC)")
    first, n = m.window_steps(t0, t1)
    return first, n, used, skipped


def scan_folder(folder, glob="*", loader=None, progress=None):
    """Time span of every radar file in a folder: [(path, (first, last) | None), ...] sorted by name.
    Files that cannot be read get None. loader(path) -> volume (default gurt_core.load_radar_file)."""
    import fnmatch
    if loader is None:
        from gurt_core import load_radar_file as loader
    paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if fnmatch.fnmatch(f, glob) and not f.startswith(".") and os.path.isfile(os.path.join(folder, f)))
    out = []
    for i, p in enumerate(paths):
        try:
            sp = volume_time_span(loader(p))
        except Exception:
            sp = None
        out.append((p, sp))
        if progress:
            progress(i + 1, len(paths), p)
    return out


def run_folder(params_path, folder, out_path, glob="*", swath=True, time_offset=0.0, loader=None,
               progress=None, header=False, **kw):
    """Maximum tornado wind over the time covered by a folder of radar scans, written as ONE file:
    a swath ASCII (.txt, swath=True) or a grid CSV. Returns a summary dict (window, files used / skipped)."""
    m = TorModel(load_params(params_path, kw.pop("res", None), kw.pop("dt", None), kw.pop("smooth", None)))
    spans = scan_folder(folder, glob, loader, progress)
    if not spans:
        raise TorError(f"no files match {glob!r} in {folder}")
    first, n, used, skipped = folder_window(m, spans, time_offset)
    if swath:
        swath_ascii(m, out_path, -1, True, kw.get("sensor"), kw.get("product"), kw.get("unit", "m/s"),
                    kw.get("fill", 0.0), kw.get("south_first", False), window=(first, n))
        out_path = swath_path(out_path)
    else:
        with open(out_path, "w", newline="") as fh:
            fh.write(render_grid(m, -1, True, header, window=(first, n)))
    t_a, t_b = int(m.times[first]), int(m.times[first + n - 1])
    return {"output": out_path, "first_step": first, "nsteps": n, "start": t_a, "end": t_b, "used": used, "skipped": skipped}


# --------------------------------------------------------------------------------- from tracking
def params_from_track(dets, radar, res=100, margin_m=None, core_tan=1.0, decay_tan=-0.6, radial_frac=0.0,
                      core_rad=1.0, decay_rad=-0.6, min_spacing_s=0.0, track_ids=None, track=None, smooth_s=60.0, max_speed=60.0):
    """Build a tormodel parameter file (text) from GURT vortex detections (box tracking / detection).

    dets: detection dicts (keys time, lat, lon, diameter_m, delta_v or min_vel / max_vel);
    radar: {"name", "lat", "lon", "elev"} = the fixed radar point the model is referenced to;
    track_ids / track: keep only the detections of that track id. One observation per detection, in time order:
      position  = the detection's gate lat / lon as azimuth (deg) and range (m) from `radar`;
      radius    = half of the couplet diameter (m);
      maxtan    = half of the couplet delta-V (m/s) (ground-relative translation is part of the track);
      maxrad    = radial_frac * maxtan (negative = inflow; 0 = purely tangential);
      exponents = core_tan / decay_tan and core_rad / decay_rad (solid-body core, decay outside).
    Detections without a usable time / position / size are skipped, as are those closer than
    `min_spacing_s` to the previous kept one; equal times keep the largest delta-V. The output grid
    covers the track plus `margin_m` (default 3 x the largest radius, at least 1 km) at `res` metres.
    smooth_s: the track, radius and wind are smoothed over +-smooth_s seconds (robust local-linear fit) because
    raw detections jitter by hundreds of metres, which the model would add to the wind as translation speed;
    0 = use the raw detections. Detections far off the track are dropped, and the track is cut at any jump faster
    than max_speed m/s (a mislinked detection), keeping the longest continuous part. Returns (text, info dict) (info has raw_max_speed / max_speed in m/s)."""
    rows = []
    for i, d in enumerate(dets):
        if track is not None and track_ids is not None and track_ids[i] != track:
            continue
        try:
            t = float(d["time"])
            lat, lon = float(d["lat"]), float(d["lon"])
            dv = float(d.get("delta_v", float("nan")))
            if not np.isfinite(dv):
                dv = float(d["max_vel"]) - float(d["min_vel"])
            diam = float(d["diameter_m"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(t) and t > 1.0e8 and np.isfinite(lat) and np.isfinite(lon)
                and np.isfinite(dv) and np.isfinite(diam)) or diam <= 0 or dv <= 0:
            continue
        x, y = _py_cart_ll(radar["lat"], radar["lon"], lat, lon)
        rows.append((int(round(t)), x, y, 0.5 * diam, 0.5 * dv))
    if not rows:
        raise TorError("no usable detections (need a UTC time, position, couplet diameter and delta-V; "
                       "times before 1973 are treated as missing - detections made with an older GURT build carry seconds since the sweep "
                       "start instead of UTC; re-run the track)")
    rows.sort(key=lambda r: (r[0], -r[4]))
    kept = []
    for r in rows:
        if kept and r[0] == kept[-1][0]:
            continue                                   # same second: keep the strongest (sorted first)
        if kept and r[0] - kept[-1][0] < min_spacing_s:
            continue
        kept.append(r)
    if len(kept) < 2:
        raise TorError(f"{len(kept)} usable detection(s) - the model needs at least two observations")
    tt = np.array([r[0] for r in kept], float)
    sp_raw = float(_track_speed(tt, [r[1] for r in kept], [r[2] for r in kept]).max())
    sp_new = sp_raw
    n_drop = 0
    if smooth_s and smooth_s > 0:
        keep = _select(tt - tt[0], [r[1] for r in kept], [r[2] for r in kept], smooth_s, max_speed)
        n_drop = int((~keep).sum())
        if n_drop:
            kept = [r for r, kp in zip(kept, keep) if kp]
            tt = np.array([r[0] for r in kept], float)
        t0_ = tt - tt[0]
        xs_, ys_ = _loess(t0_, [r[1] for r in kept], smooth_s), _loess(t0_, [r[2] for r in kept], smooth_s)
        rad_ = np.maximum(_loess(t0_, [r[3] for r in kept], smooth_s), 1.0)
        vt_ = np.maximum(_loess(t0_, [r[4] for r in kept], smooth_s), 0.0)
        kept = [(r[0], float(xs_[i]), float(ys_[i]), float(rad_[i]), float(vt_[i])) for i, r in enumerate(kept)]
        sp_new = float(_track_speed(tt, xs_, ys_).max())
    res = int(round(res))
    if res <= 0:
        raise TorError("grid resolution must be a positive number of metres")
    rmax = max(r[3] for r in kept)
    mg = max(1000.0, 3.0 * rmax) if margin_m is None else float(margin_m)
    xs, ys = [r[1] for r in kept], [r[2] for r in kept]
    x0, x1 = int(np.floor((min(xs) - mg) / res)) * res, int(np.ceil((max(xs) + mg) / res)) * res
    y0, y1 = int(np.floor((min(ys) - mg) / res)) * res, int(np.ceil((max(ys) + mg) / res)) * res
    nx, ny = (x1 - x0) // res, (y1 - y0) // res
    name = (str(radar.get("name") or "RADAR").replace(" ", "_"))[:8] or "RADAR"
    out = ["# GURT tormodel parameter file (DREADER format) written from GURT vortex tracking",
           "# radar   lat   lon   heading  elevation(m)",
           f"{name} {float(radar['lat']):.6f} {float(radar['lon']):.6f} 0.0 {int(round(float(radar.get('elev', 0.0))))}",
           "# x0 dx nx", f"{x0} {res} {nx}", "# y0 dy ny", f"{y0} {res} {ny}",
           "# t0 (-1 = first obs) dt nt (-1 = last obs)", "-1 1 -1",
           "# yyyy mm dd HH MM SS  azim  rang  radius  maxtan  coretan  decaytan  maxrad  corerad  decayrad"]
    for t, x, y, rad, vt in kept:
        az, rg = _py_polar_xy(x, y)
        tm = datetime.fromtimestamp(t, timezone.utc)
        out.append("%04d %02d %02d %02d %02d %02d  %.3f %.1f %.1f %.2f %.2f %.2f %.2f %.2f %.2f" % (
            tm.year, tm.month, tm.day, tm.hour, tm.minute, tm.second, az, rg, rad, vt, core_tan, decay_tan,
            radial_frac * vt, core_rad, decay_rad))
    info = {"n": len(kept), "start": kept[0][0], "end": kept[-1][0], "nx": nx, "ny": ny, "res": res,
            "x_range": (x0, x1), "y_range": (y0, y1), "skipped": len(dets) - len(kept),
            "raw_max_speed": sp_raw, "max_speed": sp_new, "dropped": n_drop}
    return "\n".join(out) + "\n", info


def _py_cart_ll(rlat, rlon, lat, lon):
    """East / north (m) of a point from the radar by the same flat-earth factors GURT's tracker uses."""
    return ((lon - rlon) * 111.32 * math.cos(math.radians(rlat)) * 1000.0, (lat - rlat) * 111.19 * 1000.0)


def _py_polar_xy(x, y):
    """(azimuth deg clockwise from north in [0, 360), range m)."""
    return (math.degrees(math.atan2(x, y)) % 360.0, math.hypot(x, y))


def run(params_path, mode, out_path=None, **kw):
    """Convenience: parse, run one mode and write / return the text.
    mode: 'point' (xpos, ypos | azim, rang), 'grid' / 'maxgrid' (time); header=True for the header lines."""
    m = TorModel(load_params(params_path, kw.get("res"), kw.get("dt"), kw.get("smooth")))
    if kw.get("swath"):
        if mode not in ("grid", "maxgrid"):
            raise TorError("swath ASCII output needs grid or maxgrid mode")
        t = kw.get("time")
        if out_path is None:
            raise TorError("swath ASCII output needs an output file")
        swath_ascii(m, out_path, -1 if t is None else t, mode == "maxgrid", kw.get("sensor"), kw.get("product"),
                    kw.get("unit", "m/s"), kw.get("fill", 0.0), kw.get("south_first", False))
        return ""
    if mode == "point":
        text = render_point(m, kw.get("xpos"), kw.get("ypos"), kw.get("azim"), kw.get("rang"), kw.get("header", False))
    elif mode in ("grid", "maxgrid"):
        t = kw.get("time")
        text = render_grid(m, (-1 if mode == "maxgrid" else 0) if t is None else t, mode == "maxgrid", kw.get("header", False))
    else:
        raise TorError(f"unknown mode {mode!r}")
    if out_path:
        with open(out_path, "w", newline="") as fh:
            fh.write(text)
    return text


# --------------------------------------------------------------------------------- command line
def add_arguments(ap):
    ap.add_argument("params", help="tormodel parameter file (see `tormodel --example`)")
    mode = ap.add_mutually_exclusive_group()          # one is required unless --write-params
    mode.add_argument("--point", action="store_true", help="time series at a point (use --xpos/--ypos or --azim/--rang)")
    mode.add_argument("--grid", action="store_true", help="wind-speed grid at --time")
    mode.add_argument("--maxgrid", action="store_true", help="maximum wind-speed grid through --time (default: whole period)")
    ap.add_argument("--xpos", type=float, help="point x (m east of the radar)")
    ap.add_argument("--ypos", type=float, help="point y (m north of the radar)")
    ap.add_argument("--azim", type=float, help="point azimuth (deg)")
    ap.add_argument("--rang", type=float, help="point range (m)")
    ap.add_argument("--time", type=int, help="seconds from the start; negative = whole period")
    ap.add_argument("--header", action="store_true", help="print the header lines (point: -pheader; grid: -gheader)")
    ap.add_argument("--swath", action="store_true",
                    help="write a swath ASCII (.txt) file (as the box-tracking wind swath export) from --grid / --maxgrid")
    ap.add_argument("--sensor", help="swath header 'sn' (default: radar name)")
    ap.add_argument("--product", help="swath header 'pr' (default: vmax / v)")
    ap.add_argument("--fill", type=float, default=0.0, help="swath value for cells without data (default 0)")
    ap.add_argument("--south-first", action="store_true", help="swath rows south to north (default north to south)")
    ap.add_argument("--folder", help="run over a folder of radar files: the maximum wind (--maxgrid) over the time the "
                                     "scans cover, written as one file (-o; --swath for swath ASCII)")
    ap.add_argument("--res", type=float, help="override the grid spacing (m) in x and y, keeping the extent: high resolution "
                                              "without editing the file (e.g. --res 50)")
    ap.add_argument("--smooth", type=float, help="smooth the observed track / radius / wind over +-SECONDS (robust local-linear "
                                                 "fit); fixes tracks whose jitter gives absurd translation speeds")
    ap.add_argument("--write-params", metavar="FILE", help="write the (smoothed / re-gridded) parameter file and stop")
    ap.add_argument("--dt", type=int, help="override the time step (s); larger is faster for maxgrid")
    ap.add_argument("--glob", default="*", help="file pattern inside --folder (default *)")
    ap.add_argument("--time-offset", type=float, default=0.0, help="seconds added to the scan times (clock mismatch)")
    ap.add_argument("-o", "--output", help="output file (default: standard output; required with --swath, which names it swath.<name>)")


def run_args(a):
    if a.write_params:
        try:
            p = load_params(a.params, a.res, a.dt, a.smooth)
            with open(a.write_params, "w", newline="\n") as fh:
                fh.write(params_text(p))
        except (TorError, OSError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"wrote {a.write_params}", file=sys.stderr)
        return 0
    if not (a.point or a.grid or a.maxgrid):
        print("error: choose one of --point, --grid, --maxgrid (or --write-params)", file=sys.stderr)
        return 2
    if a.folder:
        if not a.output:
            print("error: --folder needs -o OUTPUT", file=sys.stderr)
            return 1
        if a.point or a.grid:
            print("error: --folder computes the maximum wind: use --maxgrid", file=sys.stderr)
            return 1
        try:
            r = run_folder(a.params, a.folder, a.output, a.glob, a.swath, a.time_offset, header=a.header,
                           sensor=a.sensor, product=a.product, fill=a.fill, south_first=a.south_first, res=a.res, dt=a.dt, smooth=a.smooth,
                           progress=lambda i, n, p: print(f"\r[{i}/{n}] {os.path.basename(p)}", end="", file=sys.stderr))
        except (TorError, OSError) as e:
            print(f"\nerror: {e}", file=sys.stderr)
            return 1
        print(f"\nwrote {r['output']}: maximum over {r['nsteps']} step(s) "
              f"({datetime.fromtimestamp(r['start'], timezone.utc):%H:%M:%S}-"
              f"{datetime.fromtimestamp(r['end'], timezone.utc):%H:%M:%S} UTC), {len(r['used'])} file(s) used, "
              f"{len(r['skipped'])} skipped", file=sys.stderr)
        for p, why in r["skipped"]:
            print(f"  skipped {os.path.basename(p)}: {why}", file=sys.stderr)
        return 0
    if a.point:
        mode = "point"
    else:
        mode = "maxgrid" if a.maxgrid else "grid"
    try:
        text = run(a.params, mode, a.output, xpos=a.xpos, ypos=a.ypos, azim=a.azim, rang=a.rang,
                   time=a.time, header=a.header, swath=a.swath, sensor=a.sensor,
                   product=a.product, fill=a.fill, south_first=a.south_first, res=a.res, dt=a.dt, smooth=a.smooth)
    except (TorError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not a.output:
        sys.stdout.write(text)
    else:
        print(f"wrote {swath_path(a.output) if a.swath else a.output}", file=sys.stderr)
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="tormodel", description=__doc__.split("\n\n")[0])
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["--example"]:
        sys.stdout.write(EXAMPLE_PARAMS)
        return 0
    add_arguments(ap)
    return run_args(ap.parse_args(argv))


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    sys.exit(main())
