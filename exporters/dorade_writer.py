# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""GURT's DORADE sweepfile export.

The file layout comes from dorade_lrose.py (LROSE-style writer). This module adapts a GURT
RadarVolume to it: unique <= 8 character parameter names, unit / description metadata, a
per-field scale that always fits int16, sectors that cross north put in scan order, a
sector / surveillance scan mode, and strictly increasing ray times.

    write_dorade(vol, out_dir, sweep_indices=None, fields=None, scales=None, volume_number=1,
                 compat="standard", project="GURT", compress=False, history="")

compat="dreader" writes the 196-byte SSWB that DREADER's own writer produces; "standard" the
200-byte one current LROSE writes. Both are read by GURT, Radx / Solo3 and DREADER.
"""
import os
import re
import types
from datetime import datetime, timezone

import numpy as np

from . import dorade_lrose as _L

_UNITS = [
    (r"^(VE|VEL|VR|VELOC|V\d?)", "M/S", "Radial velocity"),
    (r"^(SW|WIDTH|W$)", "M/S", "Spectrum width"),
    (r"^(DBZ|DZ|ZH|REF|Z$)", "DBZ", "Reflectivity"),
    (r"^(DBM|DM|POWER)", "DBM", "Power"),
    (r"^ZDR", "DB", "Differential reflectivity"),
    (r"^(RHO|RH)", "RATIO", "Correlation coefficient"),
    (r"^(PHI|KDP)", "DEG", "Differential phase"),
    (r"^(NCP|SQI|SNR)", "RATIO", "Signal quality"),
]


def _units_for(name):
    up = name.upper()
    for pat, unit, desc in _UNITS:
        if re.match(pat, up):
            return unit, desc
    return "", name


def dorade_names(field_names):
    """Map GURT field names to unique <= 8 character DORADE parameter names."""
    out, used = {}, set()
    for f in field_names:
        base = re.sub(r"[^A-Za-z0-9_\-]", "_", str(f))[:8] or "FIELD"
        n, k = base, 1
        while n.upper() in used:
            suf = str(k)
            n = base[:8 - len(suf)] + suf
            k += 1
        used.add(n.upper())
        out[f] = n
    return out




def _scale_for(name, arr, override=None):
    """Power-of-ten scale (raw = value * scale) that fits int16; LROSE's usual one when it fits."""
    if override:
        return float(override)
    pref = 1.0 / _L.field_scale(name)
    a = np.ma.filled(np.ma.asarray(arr, dtype="float64"), np.nan)
    a = a[np.isfinite(a)]
    m = float(np.max(np.abs(a))) if a.size else 0.0
    if m * pref <= 32000.0:
        return pref
    for k in (3, 2, 1, 0):
        if m * 10.0 ** k <= 32000.0:
            return 10.0 ** k
    return 32000.0 / m


def _scan_order(sw, az):
    """Rotation that puts a north-crossing sector (stored sorted by azimuth) in scan order, or None."""
    n = len(az)
    if getattr(sw, "scan_type", "ppi") == "rhi" or n < 4 or not np.all(np.diff(az) >= 0):
        return None
    gaps = np.append(np.diff(az), az[0] + 360.0 - az[-1])
    k = int(np.argmax(gaps))
    if gaps[k] > 5.0 and gaps[k] > 3.0 * float(np.median(gaps)) and k != n - 1:
        return np.roll(np.arange(n), -(k + 1))
    return None


def _is_sector(az):
    if len(az) < 4:
        return False
    s = np.sort(np.mod(az, 360.0))
    gaps = np.append(np.diff(s), s[0] + 360.0 - s[-1])
    return bool(gaps.max() > 5.0 and gaps.max() > 3.0 * float(np.median(gaps)))


def _proxy_sweep(vol, sw, si, names, nmap, scales):
    az = np.asarray(sw.azimuth, dtype="float64")
    el = np.asarray(sw.elevation, dtype="float64")
    rt = getattr(sw, "ray_time", None)
    have_times = rt is not None and len(rt) == len(az) and float(np.ptp(np.asarray(rt, float))) > 0
    order = None if have_times else _scan_order(sw, az)
    if order is not None:
        az, el = az[order], el[order]
    p = types.SimpleNamespace()
    p.azimuth, p.elevation = az, el
    p.ranges = np.asarray(sw.ranges, dtype="float64")
    p.scan_type = getattr(sw, "scan_type", "ppi")
    if p.scan_type != "rhi":
        p.sweep_mode = "sector" if _is_sector(az) else "azimuth_surveillance"
    p.fixed_angle = getattr(sw, "fixed_angle", None)
    p.scan_time = getattr(sw, "scan_time", None) or getattr(vol, "scan_time", None)
    p.instrument_parameters = getattr(sw, "instrument_parameters", None)
    p.sweep_number = si + 1
    p.fields, p.field_meta = {}, {}
    for f in names:
        a = np.ma.asarray(sw.fields[f])
        p.fields[nmap[f]] = a[order] if order is not None else a
        unit, desc = _units_for(f)
        p.field_meta[nmap[f]] = {"units": unit, "long_name": desc if desc != f else f}
    if have_times:
        t = np.asarray(rt, dtype="float64")
        p.ray_times = t - t[0]
    else:
        p.ray_times = None
        p.duration_s = max(len(az) - 1, 1) * 0.001          # strictly increasing times, 1 ms per ray
    p.scales = {nmap[f]: _scale_for(f, p.fields[nmap[f]], scales.get(f)) ** -1 for f in names}
    return p


def write_dorade(vol, out_dir, sweep_indices=None, fields=None, scales=None, volume_number=1,
                 compat="standard", project="GURT", compress=False, history=""):
    """Write one DORADE sweepfile per selected sweep. Returns the list of written paths."""
    idx = list(range(len(vol.sweeps))) if sweep_indices is None else list(sweep_indices)
    scales = scales or {}
    # all sweeps share one volume header, so give every written sweep the same field list
    proxies, sel = [], None
    for si in idx:
        sw = vol.sweeps[si]
        want = [f for f in (fields or vol.field_names) if f in sw.fields]
        if not want:
            raise ValueError(f"sweep {si} has none of the requested fields")
        nmap = dorade_names(want)
        proxies.append(_proxy_sweep(vol, sw, si, want, nmap, scales))
        sel = sel or want
    v = types.SimpleNamespace(
        sweeps=proxies, latitude=vol.latitude, longitude=vol.longitude, altitude=vol.altitude,
        radar_name=vol.radar_name, scan_time=vol.scan_time, volume_number=int(volume_number),
        title=project, references="", source="GURT",
        instrument_parameters=getattr(vol, "instrument_parameters", None),
        comment=f"Written by GURT ({datetime.now(timezone.utc):%Y-%m-%d %H:%MZ}). Source fields: {', '.join(sel or [])}")
    # per-field scale overrides are applied through LROSE writer's scale_overrides (per sweep file)
    paths = []
    for k, p in enumerate(proxies):
        v.sweeps = [p]
        out = _L.write_dorade(v, out_dir, [0], compress=compress, swib_pad=(compat != "dreader"),
                              include_asib=False, scale_overrides=p.scales, comment=v.comment,
                              history=history)
        paths.extend(out)
    return paths


# kept for older callers
export_dorade = write_dorade
