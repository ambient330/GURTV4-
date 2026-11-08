# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""Variational fit of the tormodel radial-wind profile to the observed Doppler velocities.

tormodel's wind at distance r from the vortex centre (bearing phi from the centre) is
    Vt = maxtan * (r/R)^coretan | decaytan,   Vr = maxrad * (r/R)^corerad | decayrad   (core | outside R),
    V  = (-Vt cos phi + Vr sin phi,  Vt sin phi + Vr cos phi)      east / north, Vr > 0 outward.
GURT's tracker supplies the position, R and maxtan of every observation, and the tangential profile stays as
tracked; this module finds the three radial parameters (maxrad, corerad, decayrad) of every observation that
minimise one cost function over the whole track:

    J = sum_k sum_gates ((Vobs - Vmodel_k) / sigma)^2                    data: Doppler velocity near vortex k
      + strength * sum_k sum_j ((p_kj - p_(k-1)j) / step_j)^2            smoothness: a scan to scan change
      + sum_k sum_j ((p_kj - prior_j) / width_j)^2                       prior: stay near the starting profile

with p = (maxrad / maxtan, corerad, decayrad). The model Doppler velocity is the beam component of V (times
cos elevation) plus a uniform background wind (u, v) fitted per observation by linear least squares, so the
environmental flow and the vortex translation are not mistaken for radial inflow. The minimisation is a
Levenberg-Marquardt iteration on all observations at once (the normal equations are block tri-diagonal and are
solved densely: a few hundred observations take well under a second). NumPy only."""
import math

import numpy as np

# parameter bounds: maxrad / maxtan, core exponent, decay exponent
LO = np.array([-1.5, 0.2, -3.0])
HI = np.array([1.5, 3.0, -0.05])


# ------------------------------------------------------------------------------------------- gates near a vortex
def gather_samples(sweep, field, xc, yc, radius, shift=(0.0, 0.0), rho_min=1200.0, rho_fac=3.0, nmax=700):
    """Velocity gates within max(rho_min, rho_fac * radius) metres of the vortex centre.

    (xc, yc): the centre in the model's frame (m east / north of the model radar); shift: where this sweep's radar
    is in that frame (non-zero for a moving radar). Returns a dict of arrays (x, y in the model frame, sinaz,
    cosaz, cosel of the beam, vel) or None when fewer than 12 gates are usable."""
    az = np.radians(np.asarray(sweep.azimuth, float))
    rng = np.asarray(sweep.ranges, float)
    el = np.radians(np.asarray(getattr(sweep, "elevation", np.zeros(len(az))), float))
    v = np.ma.filled(np.ma.asarray(sweep.fields[field], float), np.nan)
    rho = max(float(rho_min), float(rho_fac) * float(radius))
    cx, cy = xc - shift[0], yc - shift[1]                  # centre in the sweep radar's frame
    rc = math.hypot(cx, cy)
    if rc < 1.0:
        return None
    azc = math.atan2(cx, cy)
    dwin = math.asin(min(1.0, rho / rc)) + math.radians(1.0)
    daz = np.angle(np.exp(1j * (az - azc)))
    rays = np.where(np.abs(daz) <= dwin)[0]
    gates = np.where((rng * np.cos(np.nanmean(el)) >= rc - rho) & (rng * np.cos(np.nanmean(el)) <= rc + rho))[0]
    if rays.size == 0 or gates.size == 0:
        return None
    A, G = np.meshgrid(rays, gates, indexing="ij")
    ce = np.cos(el[A])
    gr = rng[G] * ce                                        # ground range
    sa, ca = np.sin(az[A]), np.cos(az[A])
    vv = v[A, G]
    px, py = gr * sa, gr * ca                               # in the sweep radar's frame
    ok = np.isfinite(vv) & (np.hypot(px - cx, py - cy) <= rho)
    if ok.sum() < 12:
        return None
    idx = np.where(ok.ravel())[0]
    if idx.size > nmax:
        idx = idx[np.linspace(0, idx.size - 1, nmax).astype(int)]
    f = lambda a: a.ravel()[idx]
    return {"x": f(px) + shift[0], "y": f(py) + shift[1], "sinaz": f(sa), "cosaz": f(ca), "cosel": f(ce), "vel": f(vv)}


# ------------------------------------------------------------------------------------------------ the forward model
def model_doppler(S, xc, yc, radius, maxtan, coretan, decaytan, maxrad, corerad, decayrad):
    """Doppler velocity (m/s, away from the radar positive) of the tormodel vortex at the sample gates."""
    dx, dy = S["x"] - xc, S["y"] - yc
    r = np.maximum(np.hypot(dx, dy), 1e-3)
    q = r / radius
    core = r <= radius
    vt = maxtan * np.where(core, q ** coretan, q ** decaytan)
    vr = maxrad * np.where(core, q ** corerad, q ** decayrad)
    sp, cp = dx / r, dy / r                                  # sin / cos of the bearing from the centre
    vx = -vt * cp + vr * sp
    vy = vt * sp + vr * cp
    return (vx * S["sinaz"] + vy * S["cosaz"]) * S["cosel"]


def _residual(S, obs, k, theta, sigma):
    """Standardised residual of observation k at theta = (maxrad/maxtan, corerad, decayrad) with the uniform
    background wind projected out (linear least squares)."""
    m = model_doppler(S, obs["x"][k], obs["y"][k], obs["radius"][k], obs["maxtan"][k], obs["coretan"][k],
                      obs["decaytan"][k], theta[0] * obs["maxtan"][k], theta[1], theta[2])
    r = S["vel"] - m
    B = np.column_stack((S["sinaz"] * S["cosel"], S["cosaz"] * S["cosel"]))
    coef, *_ = np.linalg.lstsq(B, r, rcond=None)
    return (r - B @ coef) / sigma


# ----------------------------------------------------------------------------------------------------- the fit
def fit_radial(obs, samples, prior, sigma=2.0, strength=1.0, steps=(0.15, 0.25, 0.25), widths=(0.5, 0.4, 0.4),
               iters=30, reject=True):
    """Fit (maxrad, corerad, decayrad) of every observation.

    obs: dict of arrays x, y, radius, maxtan, coretan, decaytan (metres, m/s) for K observations in time order;
    samples: list of K gate dicts from gather_samples (None = no data: that observation follows the
    smoothness / prior terms only); prior: (maxrad / maxtan, corerad, decayrad) the fit starts from and is
    pulled towards; sigma: velocity error (m/s); strength: weight of the scan to scan smoothness (0 = each scan
    on its own, larger = stiffer). Returns (theta[K, 3] = maxrad / maxtan, corerad, decayrad, info dict)."""
    K = len(obs["x"])
    p0 = np.tile(np.asarray(prior, float), (K, 1))
    theta = np.clip(p0.copy(), LO, HI)
    S = list(samples)
    if reject:                                               # drop gates far from the rest (folded velocity, clutter)
        for k, s in enumerate(S):
            if s is None:
                continue
            r0 = _residual(s, obs, k, theta[k], 1.0)
            med = np.median(r0)
            mad = 1.4826 * np.median(np.abs(r0 - med)) + 1e-6
            keep = np.abs(r0 - med) <= 4.0 * mad + 6.0
            if keep.sum() < 12:
                S[k] = None
                continue
            S[k] = {key: val[keep] for key, val in s.items()}
    have = np.array([s is not None for s in S])
    sw = math.sqrt(max(float(strength), 0.0))
    st = np.asarray(steps, float)
    wd = np.asarray(widths, float)

    def regs(th):
        """Smoothness and prior residuals, stacked (3 per prior, 3 per scan to scan difference)."""
        out = [((th - p0) / wd).ravel()]
        if K > 1 and sw > 0:
            out.append((sw * (th[1:] - th[:-1]) / st).ravel())
        return np.concatenate(out)

    def data(th):
        return [_residual(S[k], obs, k, th[k], sigma) if have[k] else np.zeros(0) for k in range(K)]

    def cost(th, dr=None):
        dr = data(th) if dr is None else dr
        return float(sum((d @ d) for d in dr) + (lambda g: g @ g)(regs(th)))

    # regulariser as a sparse-ish matrix D (dense is fine): rows act on the flat theta (K*3)
    n = 3 * K
    rows = [np.eye(n) / np.tile(wd, K)[:, None]]
    if K > 1 and sw > 0:
        Dd = np.zeros((3 * (K - 1), n))
        for k in range(1, K):
            for j in range(3):
                Dd[3 * (k - 1) + j, 3 * k + j] = sw / st[j]
                Dd[3 * (k - 1) + j, 3 * (k - 1) + j] = -sw / st[j]
        rows.append(Dd)
    D = np.vstack(rows)
    DtD = D.T @ D
    dr = data(theta)
    c = cost(theta, dr)
    c_first = c
    mu = 1e-2
    eps = 1e-3
    rms0 = _rms(dr)
    for it in range(int(iters)):
        H = DtD.copy()
        g = D.T @ regs(theta)
        for k in range(K):
            if not have[k]:
                continue
            Jk = np.empty((dr[k].size, 3))
            for j in range(3):
                t2 = theta[k].copy()
                t2[j] += eps
                Jk[:, j] = (_residual(S[k], obs, k, t2, sigma) - dr[k]) / eps
            sl = slice(3 * k, 3 * k + 3)
            H[sl, sl] += Jk.T @ Jk
            g[sl] += Jk.T @ dr[k]
        improved = False
        for _ in range(8):
            try:
                step = np.linalg.solve(H + mu * np.diag(np.diag(H) + 1e-9), -g)
            except np.linalg.LinAlgError:
                mu *= 10
                continue
            th2 = np.clip(theta + step.reshape(K, 3), LO, HI)
            dr2 = data(th2)
            c2 = cost(th2, dr2)
            if c2 < c:
                done = (c - c2) <= 1e-5 * max(c, 1e-12)
                theta, dr, c = th2, dr2, c2
                mu = max(mu / 3.0, 1e-7)
                improved = True
                break
            mu *= 4.0
        if not improved or done:
            break
    info = {"cost0": c_first, "cost": c, "iterations": it + 1, "rms_before": rms0, "rms_after": _rms(dr),
            "n_fit": int(have.sum()), "n_obs": K,
            "rms_each": [float(math.sqrt(d @ d / d.size) * sigma) if d.size else float("nan") for d in dr]}
    return theta, info


def _rms(drs):
    n = sum(d.size for d in drs)
    return float(math.sqrt(sum(d @ d for d in drs) / n)) if n else float("nan")      # in units of sigma


# ------------------------------------------------------------------------- applying it to a parameter file's text
def fit_params_text(text, dets, get_sweep, radar_lat, radar_lon, sigma=2.0, strength=1.0, rho_min=1200.0):
    """Fit the radial columns of a tormodel parameter file written by tormodel.params_from_track.

    dets: the detections it was built from (each with time, file, sweep, field); get_sweep(det) -> (sweep,
    radar_lat, radar_lon) loads that detection's sweep (return None to skip it). Observations are matched to
    detections by time. Returns (new text, info)."""
    import tormodel as TM
    lines = text.split("\n")
    first = next(i for i, ln in enumerate(lines) if ln.strip() and not ln.startswith("#") and len(ln.split()) >= 15)
    idx = [i for i in range(first, len(lines)) if len(lines[i].split()) >= 15]
    from datetime import datetime, timezone
    rows = [lines[i].split() for i in idx]
    t = np.array([datetime(*map(int, r[:6]), tzinfo=timezone.utc).timestamp() for r in rows])
    az, rg = np.radians([float(r[6]) for r in rows]), np.array([float(r[7]) for r in rows])
    num = np.array([[float(x) for x in r[8:]] for r in rows])        # radius maxtan coretan decaytan maxrad corerad decayrad
    obs = {"x": rg * np.sin(az), "y": rg * np.cos(az), "radius": num[:, 0], "maxtan": num[:, 1],
           "coretan": num[:, 2], "decaytan": num[:, 3]}
    prior = (float(np.median(num[:, 4] / np.maximum(num[:, 1], 1e-6))), float(np.median(num[:, 5])), float(np.median(num[:, 6])))
    by_t = {}
    for d in dets:
        try:
            by_t[int(round(float(d["time"])))] = d
        except (KeyError, TypeError, ValueError):
            pass
    samples, errs = [], []
    for k in range(len(rows)):
        d = by_t.get(int(round(t[k])))
        s = None
        if d is not None:
            try:
                got = get_sweep(d)
                if got is not None:
                    sw, la, lo = got
                    shift = TM._py_cart_ll(radar_lat, radar_lon, la, lo)
                    s = gather_samples(sw, d.get("field") or "VEL", obs["x"][k], obs["y"][k], obs["radius"][k], shift, rho_min)
            except Exception as e:                                   # a file that will not load: no data for this scan
                errs.append(str(e))
        samples.append(s)
    theta, info = fit_radial(obs, samples, prior, sigma=sigma, strength=strength)
    for j, i in enumerate(idx):
        r = lines[i].split()
        r[12] = "%.2f" % (theta[j, 0] * obs["maxtan"][j])
        r[13] = "%.2f" % theta[j, 1]
        r[14] = "%.2f" % theta[j, 2]
        lines[i] = " ".join(r[:15])
    info["errors"] = errs[:3]
    info["maxrad_range"] = (float(np.min(theta[:, 0] * obs["maxtan"])), float(np.max(theta[:, 0] * obs["maxtan"])))
    info["theta"] = theta
    return "\n".join(lines), info
