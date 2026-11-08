# Adapted from NCAR LROSE code (Radx DoradeRadxFile); this is NOT the version available from UCAR.
# Copyright (c) 1990 - 2023, University Corporation for Atmospheric Research (UCAR), Boulder,
# Colorado, USA. All rights reserved. Used under the LROSE BSD license - see
# THIRD_PARTY_NOTICES.md for the full text and disclaimer.
# Modifications for GURT: SPDX-License-Identifier: GPL-2.0-only (BSD is GPL-compatible).
"""LROSE-style DORADE sweepfile writer. GURT calls it through exporters/dorade_writer.py."""
from __future__ import annotations

import math
import os
import struct
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np

# constants
MISSING_F = -9999.0
MISSING_POWER = -999.0
MISSING_I16 = -32768
HRD_END = 1
SIGN16 = 0x8000
NDX_QUE_SIZE = 480
MAX_KEYS = 8

SCAN_CAL, SCAN_PPI, SCAN_COP, SCAN_RHI, SCAN_VER = 0, 1, 2, 3, 4
SCAN_TAR, SCAN_MAN, SCAN_IDL, SCAN_SUR, SCAN_AIR, SCAN_HOR = 5, 6, 7, 8, 9, 10
_SCAN_SHORT = {0: "CAL", 1: "PPI", 2: "COP", 3: "RHI", 4: "VER", 5: "TAR",
               6: "MAN", 7: "IDL", 8: "SUR", 9: "AIR", 10: "HOR"}
_MODE_TO_SCAN = {
    "sector": SCAN_PPI, "coplane": SCAN_COP, "rhi": SCAN_RHI,
    "vertical_pointing": SCAN_VER, "idle": SCAN_IDL,
    "azimuth_surveillance": SCAN_SUR, "elevation_surveillance": SCAN_AIR,
    "sunscan": SCAN_CAL, "pointing": SCAN_MAN, "calibration": SCAN_CAL,
    "manual_ppi": SCAN_MAN, "manual_rhi": SCAN_MAN, "ppi": SCAN_SUR,
}

# on-disk struct formats (big-endian, packed)
_F_COMM = ">4si500s"
_F_SSWB = ">4si7i8sddiii7i24i" # 196
_F_SSWB_PAD = ">4si7i8sidd" + "iii7i24i" # 200
_F_VOLD = ">4sihhi20s6h8s8s4h"
_F_RADD = (">4si8s8f2h3f4h2f5f2h5f5fi8si3f11f11f4fi20s")
_F_CFAC = ">4si16f"
_F_PARM = ">4si8s40s8shhf4h8sfffii8siifi32si32sifff"
_F_CSFD = ">4siif8f8h"
_F_SWIB = ">4si8siifffi"
_F_ASIB = ">4si18f"
_F_RYIB = ">4siii4h4fi"
_F_RDAT = ">4si8s"
_F_NULL = ">4si"
_F_RKTB = ">4sifiiii"
_F_RKEY = ">fii"

for _name, _fmt, _size in (
        ("COMM", _F_COMM, 508), ("SSWB", _F_SSWB, 196), ("SSWB_PAD", _F_SSWB_PAD, 200),
        ("VOLD", _F_VOLD, 72), ("RADD", _F_RADD, 300), ("CFAC", _F_CFAC, 72),
        ("PARM", _F_PARM, 216), ("CSFD", _F_CSFD, 64), ("SWIB", _F_SWIB, 40),
        ("ASIB", _F_ASIB, 80), ("RYIB", _F_RYIB, 44), ("RDAT", _F_RDAT, 16),
        ("NULL", _F_NULL, 8), ("RKTB", _F_RKTB, 28), ("RKEY", _F_RKEY, 12)):
    assert struct.calcsize(_fmt) == _size, (_name, struct.calcsize(_fmt), _size)


# helpers
def _b(s, n: int) -> bytes:
    return (s or "").encode("ascii", "replace")[:n].ljust(n, b"\0")


def _fmod360(x: float) -> float:
    return math.fmod(x + 720.0, 360.0)


def _utc(dt: Optional[datetime]) -> datetime:
    if dt is None:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def field_scale(name: str) -> float:
    n = name.lower()
    if "rho" in n or "ncp" in n or "sqi" in n:
        return 0.0001
    if "kdp" in n:
        return 0.001
    if "phi" in n:
        return 0.02
    return 0.01


def scan_mode_for(sw) -> int:
    if getattr(sw, "scan_type", "ppi") == "rhi":
        return SCAN_RHI
    mode = getattr(sw, "sweep_mode", None)
    if mode:
        return _MODE_TO_SCAN.get(str(mode).lower(), SCAN_SUR)
    return SCAN_SUR


def compute_file_name(vol_num: int, n_sweeps: int, fixed_angle: float,
                      instrument: str, scan_type: str, t: datetime,
                      millisecs: int) -> str:
    instrument = (instrument or "")[:8]
    instrument = "".join("_" if (c.isspace() or c == "/") else c for c in instrument)
    scan_type = "".join("_" if (c.isspace() or c == "/") else c for c in scan_type)
    if n_sweeps != 1:
        fixed_angle = 0.0
    return "swp.%d%02d%02d%02d%02d%02d.%s.%d.%.1f_%s_v%03d" % (
        t.year - 1900, t.month, t.day, t.hour, t.minute, t.second,
        instrument, millisecs, fixed_angle, scan_type, vol_num)


def compress_hrd16(uncomp: Sequence[int], bad_val: int = 32768) -> List[int]:
    n = len(uncomp)
    if n < 2:
        return list(uncomp)
    comp: List[int] = []
    ss = 0
    end = n - 1
    kount = 0
    data_run = False
    rl = -1                                   
    while ss < end:
        kount = 2
        rl = len(comp)
        comp.append(0)
        if uncomp[ss + 1] != bad_val or uncomp[ss] != bad_val:
            data_run = True
            comp.append(uncomp[ss])
            comp.append(uncomp[ss + 1])
            ss += 2
        else:
            data_run = False
            ss += 2
        while ss < end:
            if data_run:
                if uncomp[ss - 1] == bad_val and uncomp[ss] == bad_val and kount > 2:
                    kount -= 1
                    comp[rl] = SIGN16 | kount
                    ss -= 1
                    comp.pop()                
                    break
                kount += 1
                comp.append(uncomp[ss])
                ss += 1
            else:
                if uncomp[ss] != bad_val:
                    comp[rl] = kount
                    break
                ss += 1
                kount += 1
    if ss > end:
        comp[rl] = (SIGN16 | kount) if data_run else kount
    elif data_run:
        if ss == end:
            comp.append(uncomp[ss])
            kount += 1
        comp[rl] = SIGN16 | kount
    elif uncomp[ss] == bad_val:
        kount += 1
        comp[rl] = kount
    else:
        if kount == 2:
            comp[rl] = SIGN16 | 3
            comp.extend((bad_val, bad_val, uncomp[ss]))
        else:
            kount -= 1
            comp[rl] = kount
            comp.extend((SIGN16 | 2, bad_val, uncomp[ss]))
    comp.append(HRD_END)
    return comp


def decompress_hrd16(comp: Sequence[int], max_uncomp: int, bad_val: int = 32768) -> List[int]:
    out: List[int] = []
    i = 0
    n_bads = 0
    while i < len(comp):
        w = comp[i]
        if w == HRD_END or w == 0:
            return out
        nn = w & 0x7FFF
        if w & SIGN16:
            i += 1
            if len(out) + nn > max_uncomp:
                return out
            n_bads = 0
            out.extend(comp[i:i + nn])
            i += nn
        else:
            if n_bads:
                return out
            if len(out) + nn > max_uncomp:
                return out
            n_bads = nn
            i += 1
            out.extend([bad_val] * nn)
    return out


# field encoding
def encode_field(arr2d, ngates: int, scale: float) -> np.ndarray:
    a = np.ma.filled(np.ma.asarray(arr2d, dtype="float64"), np.nan)
    if a.shape[1] < ngates:
        pad = np.full((a.shape[0], ngates - a.shape[1]), np.nan)
        a = np.concatenate([a, pad], axis=1)
    elif a.shape[1] > ngates:
        a = a[:, :ngates]
    with np.errstate(invalid="ignore"):
        raw = np.rint(a / scale)
    bad = ~np.isfinite(raw) | (np.abs(raw) > 32767)
    raw = np.where(bad, MISSING_I16, raw)
    return raw.astype(np.int16)


# volume metadata helpers
def _param(sw, vol, key: str) -> Optional[float]:
    for src in (getattr(sw, "instrument_parameters", None),
                getattr(vol, "instrument_parameters", None)):
        if not src:
            continue
        entry = src.get(key)
        if entry is None:
            continue
        try:
            v = float(np.asarray(entry["data"]).ravel()[0])
            if np.isfinite(v):
                return v
        except Exception:
            continue
    return None


def _field_meta(sw, vol, name: str) -> dict:
    for src in (getattr(sw, "field_meta", None), getattr(vol, "field_meta", None)):
        if src and name in src:
            return src[name] or {}
    return {}


def _ray_times(sw, nrays: int, base: datetime) -> List[datetime]:
    """Absolute time of every ray. Uses sw.ray_times (s since sw.scan_time) when
    present, otherwise spreads rays evenly over `sweep_duration` (default 1 s)
    so that times are at least monotonic."""
    rt = getattr(sw, "ray_times", None)
    if rt is not None and len(rt) == nrays:
        return [base + timedelta(seconds=float(x)) for x in rt]
    dur = float(getattr(sw, "duration_s", 0.0) or 0.0)
    return [base + timedelta(seconds=dur * i / max(nrays - 1, 1)) for i in range(nrays)]


# ── block builders ─────────────────────────────────────────────────────────
def _b_comment(text: str) -> bytes:
    t = (text or "").strip() or "Written by GURT dorade_writer.py"
    return struct.pack(_F_COMM, b"COMM", 508, _b(t, 499) + b"\0")


def _b_sswb(start: datetime, stop: datetime, nparams: int, radar: str,
            file_size: int, rot_off: int, rot_size: int,
            seds_off: int, seds_size: int, pad: bool) -> bytes:
    keys = []
    if rot_size > 0:
        keys.append((rot_off, rot_size, 2))        # KEYED_BY_ROT_ANG
    if seds_size > 0:
        keys.append((seds_off, seds_size, 3))      # SOLO_EDIT_SUMMARY
    keys += [(0, 0, 0)] * (MAX_KEYS - len(keys))
    flat = [x for k in keys for x in k]
    s0, s1 = int(start.timestamp()), int(stop.timestamp())
    d0 = s0 + (start.microsecond / 1e6)
    d1 = s1 + (stop.microsecond / 1e6)
    head = [int(_time.time()), s0, s1, file_size, 0, s0, nparams]
    nbytes = 200 if pad else 196
    # version_num is 0 in lrose (DoradeData::init memsets the struct and never sets it)
    tail = [0, len([k for k in keys if k[2]]), 0] + [0] * 7 + flat   # version, nkeys, status, holders
    if pad:
        return struct.pack(_F_SSWB_PAD, b"SSWB", nbytes, *head, _b(radar, 8), 0, d0, d1, *tail)
    return struct.pack(_F_SSWB, b"SSWB", nbytes, *head, _b(radar, 8), d0, d1, *tail)


def _b_vold(vol_num: int, max_bytes: int, proj: str, flight: str, facility: str,
            start: datetime) -> bytes:
    now = datetime.now(timezone.utc)
    return struct.pack(_F_VOLD, b"VOLD", 72, 2, vol_num, max(max_bytes, 32768),
                       _b(proj, 20), start.year, start.month, start.day,
                       start.hour, start.minute, start.second,
                       _b(flight, 8), _b(facility, 8),
                       now.year, now.month, now.day, 1)


def _b_radd(vol, sw, radar: str, site: str, scan_mode: int, nfields: int,
            first_ray_params: dict, compress: bool) -> bytes:
    f = first_ray_params
    freq = [MISSING_F] * 5
    if f.get("frequency_hz"):
        freq[0] = f["frequency_hz"] / 1e9
    prt = [0.0] * 5
    if f.get("prt_s") is not None:
        prt[0] = f["prt_s"] * 1000.0               # msec
    lat, lon = float(vol.latitude), float(vol.longitude)
    alt_km = float(vol.altitude) / 1000.0
    vals = [
        MISSING_F,                                 # radar_const
        MISSING_POWER,                             # peak_power (lrose: -999 when unknown)
        MISSING_F,                                 # noise_power
        MISSING_F,                                 # receiver_gain
        MISSING_F,                                 # antenna_gain
        MISSING_F,                                 # system_gain
        float(getattr(vol, "beam_width_h", MISSING_F) or MISSING_F),
        float(getattr(vol, "beam_width_v", MISSING_F) or MISSING_F),
    ]
    return struct.pack(
        _F_RADD, b"RADD", 300, _b(radar, 8), *vals,
        0, scan_mode,                              # radar_type = GROUND
        float(f.get("scan_rate", MISSING_F)), MISSING_F, MISSING_F,
        nfields, nfields, 1 if compress else 0, 0,
        MISSING_F, MISSING_F,                      # data_red_parm0/1
        lon, lat, alt_km,
        float(f.get("nyquist", MISSING_F)),
        float(f.get("unamb_range_km", MISSING_F)),
        1 if f.get("frequency_hz") else 0, 1,
        *freq, *prt,
        1,                                         # extension_num: primary axis Z
        _b("", 8), 0,                              # config_name, config_num
        MISSING_F, MISSING_F, MISSING_F,           # lidar aperture/fov/eff
        *([MISSING_F] * 11), *([MISSING_F] * 11),  # aux_freq, aux_prt
        float(f.get("pulse_width_us") or MISSING_F),
        MISSING_F, MISSING_F, MISSING_F,           # cop baselines, pc bandwidth
        0,                                         # pc_waveform_type
        _b(site, 20))


CFAC_FIELDS = ("azimuth_corr", "elevation_corr", "range_delay_corr", "longitude_corr",
               "latitude_corr", "pressure_alt_corr", "radar_alt_corr", "ew_gndspd_corr",
               "ns_gndspd_corr", "vert_vel_corr", "heading_corr", "roll_corr", "pitch_corr",
               "drift_corr", "rot_angle_corr", "tilt_corr")


def _b_cfac(cfac: Dict[str, float]) -> bytes:
    """CFAC block; missing keys are 0.0 (DoradeData::init zero-fills the struct)."""
    return struct.pack(_F_CFAC, b"CFAC", 72, *[float(cfac.get(k, 0.0)) for k in CFAC_FIELDS])


def _b_parm(name: str, long_name: str, units: str, ngates: int, r0_m: float,
            dr_m: float, scale: float, nyquist: float, f: dict) -> bytes:
    pw_m = MISSING_I16
    rbw = MISSING_F
    if f.get("pulse_width_us"):
        pw_m = int(f["pulse_width_us"] / 1e6 * 299792458.0 * 0.5 + 0.5)
        rbw = 1.0 / f["pulse_width_us"]
    return struct.pack(
        _F_PARM, b"PARM", 216, _b(name, 8), _b(long_name or name, 40), _b(units, 8),
        1, 1, rbw, pw_m, 0, int(f.get("nsamples", 1)), 2,   # binary_format INT16
        _b("NONE", 8), MISSING_F, 1.0 / scale, 0.0, MISSING_I16,
        0, _b("", 8), 0,                           # extension, config_name, config_num
        16,                                        # offset_to_data = sizeof(RDAT)
        0.0, 0, _b("", 32), 0, _b("", 32),
        ngates, r0_m, dr_m, nyquist)


def _b_csfd(r0_m: float, dr_m: float, ngates: int) -> bytes:
    spacing = [dr_m] + [0.0] * 7
    cells = [ngates] + [0] * 7
    return struct.pack(_F_CSFD, b"CSFD", 64, 1, r0_m, *spacing, *cells)


def _b_swib(radar: str, sweep_num: int, nrays: int, start_ang: float,
            stop_ang: float, fixed: float) -> bytes:
    return struct.pack(_F_SWIB, b"SWIB", 40, _b(radar, 8), sweep_num, nrays,
                       start_ang, stop_ang, fixed, 0)


def _b_ryib(sweep_num: int, t: datetime, az: float, el: float,
            scan_rate: float) -> bytes:
    # Round to the nearest ms FIRST and take every field from the rounded time, so a
    # time like 45.9996 s becomes 46.000 s rather than 45 s + 0 ms (which ran 1 s backwards).
    t = t.replace(microsecond=0) + timedelta(milliseconds=int(t.microsecond / 1000.0 + 0.5))
    yday = t.timetuple().tm_yday
    return struct.pack(_F_RYIB, b"RYIB", 44, sweep_num, yday,
                       t.hour, t.minute, t.second, t.microsecond // 1000,
                       az, el, MISSING_POWER, scan_rate, 0)


def _b_asib(vol, az: float, el: float) -> bytes:
    v = [float(vol.longitude), float(vol.latitude), float(vol.altitude) / 1000.0,
         0.0,                                      # altitude_agl
         0.0, 0.0, 0.0,                            # ew/ns/vert velocity
         0.0, 0.0, 0.0, 0.0,                       # heading roll pitch drift
         az, el,                                   # rotation_angle, tilt
         0.0, 0.0, 0.0, 0.0, 0.0]                  # winds, heading/pitch change
    return struct.pack(_F_ASIB, b"ASIB", 80, *v)


def _b_rdat(name: str, words: bytes) -> bytes:
    return struct.pack(_F_RDAT, b"RDAT", 16 + len(words), _b(name, 8)) + words


def _b_rotation_table(entries: List[tuple], lrose_compat: bool = True) -> bytes:
    """entries: (rotation_angle, offset, size) per ray, in file order.
    Layout (offsets from start of this block, as lrose writes them):
        RKTB struct | pad to 8 | si32 index[480] | rot_table_entry x nrays [| zero pad]
    lrose_compat: lrose sizes the block with sizeof(rot_angle_table_t)=28 per ray instead of
    sizeof(rot_table_entry_t)=12 (DoradeRadxFile::_writeRotAngTable), leaving 16 bytes/ray of
    trailing zeros that nothing reads. True reproduces that, False writes the tight block."""
    n = len(entries)
    index_off = ((28 - 1) // 8 + 1) * 8
    key_off = index_off + NDX_QUE_SIZE * 4
    block_size = key_off + n * (28 if lrose_compat else 12)
    angle2ndx = NDX_QUE_SIZE / 360.0
    head = struct.pack(_F_RKTB, b"RKTB", block_size, angle2ndx, NDX_QUE_SIZE,
                       key_off, index_off, n)
    index = [-1] * NDX_QUE_SIZE
    for j, (ang, _, _) in enumerate(entries):
        ii = int(ang * angle2ndx)
        if ii < 0 or ii > NDX_QUE_SIZE - 1:
            ii = 0
        index[ii] = j
    body = head + b"\0" * (index_off - len(head))
    body += struct.pack(">%di" % NDX_QUE_SIZE, *index)
    for ang, off, size in entries:
        body += struct.pack(_F_RKEY, ang, off, size)
    return body.ljust(block_size, b"\0")


def _b_seds(history: str) -> bytes:
    text = (history or "").encode("ascii", "replace")
    size = (((len(text) + 12) // 4) + 1) * 4
    blk = bytearray(size)
    blk[0:4] = b"SEDS"
    blk[4:8] = struct.pack(">i", size)
    blk[8:8 + len(text)] = text
    return bytes(blk)

def write_dorade(vol, out_dir: str,
                 sweep_indices: Optional[Sequence[int]] = None,
                 *, compress: bool = True,
                 swib_pad: bool = True,
                 include_asib: bool = False,
                 cfac: Optional[Dict[str, float]] = None,
                 lrose_compat: bool = True,
                 scale_overrides: Optional[Dict[str, float]] = None,
                 comment: str = "",
                 history: str = "",
                 site_name: Optional[str] = None) -> List[str]:
    """Write each selected sweep of a GURT ``RadarVolume`` as a DORADE file.

    compress        HRD run-length compression of the bad-data runs (RADD
                    data_compress = 1). Off by default for maximum reader
                    compatibility; Radx/Solo read both. Output is byte-identical to
                    lrose's compressHrd16.
    swib_pad        True (default, what current lrose writes): 200-byte SSWB with the
                    4-byte pad that puts the doubles on an 8-byte boundary.
                    False: the old 196-byte layout (lrose DORADE_WRITE_32BIT_SWIB=TRUE).
    include_asib    write a platform (ASIB) block per ray. Default False: lrose only
                    writes ASIB when the ray has a georeference, which a fixed ground
                    radar normally does not.
    cfac            None (default): no CFAC block, as lrose does when the volume has no
                    cfactors. A dict {azimuth_corr: .., elevation_corr: .., ...} (keys in
                    CFAC_FIELDS, missing ones 0) writes the block, and azimuth_corr /
                    elevation_corr are added into the rotation-angle table like lrose does.
    lrose_compat    pad the rotation-angle table to lrose's (oversized) block length.
    scale_overrides {field name: scale}; raw = round(value / scale).
    history         text for the SEDS block (omitted when empty).
    Returns the list of absolute paths written (one per sweep)."""
    os.makedirs(out_dir, exist_ok=True)
    if sweep_indices is None:
        sweep_indices = list(range(len(vol.sweeps)))
    picked = [(i, vol.sweeps[i]) for i in sweep_indices if 0 <= i < len(vol.sweeps)]
    if not picked:
        raise ValueError("No sweeps selected for export.")
    paths = []
    for i, sw in picked:
        paths.append(_write_sweep(vol, sw, getattr(sw, "sweep_number", None) or (i + 1), out_dir, compress, swib_pad, include_asib, cfac,
                                  lrose_compat, scale_overrides or {}, comment, history, site_name))
    return paths


def _write_sweep(vol, sw, sweep_num, out_dir, compress, swib_pad, include_asib, cfac,
                 lrose_compat, scale_overrides, comment, history, site_name) -> str:
    nrays = len(sw.azimuth)
    if nrays == 0:
        raise ValueError("Sweep has no rays.")
    az = np.asarray(sw.azimuth, dtype="float64")
    el = np.asarray(sw.elevation, dtype="float64")
    ranges = np.asarray(sw.ranges, dtype="float64")
    ngates = len(ranges)
    r0 = float(ranges[0]) if ngates else 0.0
    dr = float(np.median(np.diff(ranges))) if ngates > 1 else 1.0
    names = list(sw.fields.keys())
    scan_mode = scan_mode_for(sw)
    is_rhi = scan_mode == SCAN_RHI
    radar = (getattr(vol, "radar_name", "") or "UNKN").strip()
    site = site_name if site_name is not None else radar

    base = _utc(getattr(sw, "scan_time", None) or getattr(vol, "scan_time", None))
    times = _ray_times(sw, nrays, base)
    start_t, stop_t = times[0], times[-1]

    # per-sweep instrument parameters
    nyq = _param(sw, vol, "nyquist_velocity")
    prt = _param(sw, vol, "prt")
    freq = _param(sw, vol, "frequency")
    pw = _param(sw, vol, "pulse_width")            # microseconds if provided
    params = {
        "nyquist": nyq if nyq is not None else MISSING_F,
        "prt_s": prt,
        "frequency_hz": freq,
        "unamb_range_km": (299792458.0 * prt / 2000.0) if prt else MISSING_F,
        "pulse_width_us": pw,
        "scan_rate": float(getattr(sw, "scan_rate", MISSING_F) or MISSING_F),
        "nsamples": int(_param(sw, vol, "n_samples") or 1),
    }

    # encode fields to int16 (nrays, ngates) once
    scales = {n: float(scale_overrides.get(n, field_scale(n))) for n in names}
    enc = {n: encode_field(sw.fields[n], ngates, scales[n]) for n in names}

    vol_num = int(getattr(vol, "volume_number", 0) or 0)
    # lrose: NPoints (nrays * ngates, NOT times nfields) * 4 + sizeof(RDAT), floor 32768
    max_bytes = nrays * ngates * 4 + 16
    cf = cfac or {}

    buf = bytearray()

    def pos() -> int:
        return len(buf)

    buf += _b_comment(comment or getattr(vol, "comment", ""))
    sswb_pos = pos()
    buf += _b_sswb(start_t, stop_t, len(names), radar, 0, 0, 0, 0, 0, swib_pad)
    buf += _b_vold(vol_num, max_bytes, getattr(vol, "title", "") or "GURT",
                   getattr(vol, "references", "") or "", getattr(vol, "source", "") or "GURT",
                   start_t)
    buf += _b_radd(vol, sw, radar, site, scan_mode, len(names), params, compress)
    for n in names:
        meta = _field_meta(sw, vol, n)
        buf += _b_parm(n, meta.get("long_name", n), meta.get("units", ""), ngates,
                       r0, dr, scales[n],
                       params["nyquist"], params)
    buf += _b_csfd(r0, dr, ngates)
    if cfac is not None:
        buf += _b_cfac(cf)

    fixed = getattr(sw, "fixed_angle", None)
    if fixed is None or not np.isfinite(fixed):
        fixed = np.nanmedian(az) if is_rhi else np.nanmean(el)
    fixed = float(fixed)
    # lrose takes start/stop from elevation for RHI, elevation-surveillance and manual RHI
    el_sweep = is_rhi or str(getattr(sw, "sweep_mode", "")).lower() in ("elevation_surveillance", "manual_rhi")
    buf += _b_swib(radar, sweep_num, nrays,
                   float(el[0] if el_sweep else az[0]), float(el[-1] if el_sweep else az[-1]), fixed)

    entries = []
    for i in range(nrays):
        off = pos()
        buf += _b_ryib(sweep_num, times[i], float(az[i]), float(el[i]), params["scan_rate"])
        if include_asib:
            buf += _b_asib(vol, float(az[i]), float(el[i]))
        for n in names:
            raw = enc[n][i]
            if compress:
                words = compress_hrd16(raw.view(np.uint16).tolist(), 32768)
                data = struct.pack(">%dH" % len(words), *words)
            else:
                data = raw.astype(">i2").tobytes()
            buf += _b_rdat(n, data)
        # rotation-angle table key, as DoradeRadxFile::_writeRayGeoref:
        #   RHI           -> FMOD360(450 - (el + elevation_corr))   (with or without ASIB)
        #   no ASIB       -> az + azimuth_corr, NOT wrapped
        #   with ASIB     -> FMOD360(az + azimuth_corr)             (_ddRotation default case)
        if is_rhi:
            rot = _fmod360(450.0 - (float(el[i]) + cf.get("elevation_corr", 0.0)))
        elif include_asib:
            rot = _fmod360(float(az[i]) + cf.get("azimuth_corr", 0.0))
        else:
            rot = float(az[i]) + cf.get("azimuth_corr", 0.0)
        entries.append((rot, off, pos() - off))

    buf += struct.pack(_F_NULL, b"NULL", 8)
    rot_off = pos()
    rkb = _b_rotation_table(entries, lrose_compat)
    buf += rkb
    seds_off = seds_size = 0
    if history:
        seds_off = pos()
        sb = _b_seds(history)
        buf += sb
        seds_size = len(sb)

    sswb = _b_sswb(start_t, stop_t, len(names), radar, len(buf),
                   rot_off, len(rkb), seds_off, seds_size, swib_pad)
    buf[sswb_pos:sswb_pos + len(sswb)] = sswb

    fixed_name = fixed
    scan_short = _SCAN_SHORT.get(scan_mode, "XXX")
    # seconds in the name are the floor of the start time (as in lrose), so the ms must be
    # clamped to 999 rather than carried or wrapped
    name = compute_file_name(vol_num, 1, fixed_name, radar, scan_short, start_t,
                             min(int(start_t.microsecond / 1000.0 + 0.5), 999))
    dest = os.path.join(out_dir, name)
    tmp = dest + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(buf)
    os.replace(tmp, dest)
    return os.path.abspath(dest)


def read_dorade_blocks(path: str):
    """Walk a DORADE sweepfile and return [(id, offset, nbytes)] for the main
    sequence (up to NULL), plus the trailing blocks addressed by the key table."""
    data = open(path, "rb").read()
    out, off = [], 0
    while off + 8 <= len(data):
        bid = data[off:off + 4].decode("ascii", "replace")
        nb = struct.unpack_from(">i", data, off + 4)[0]
        out.append((bid, off, nb))
        if bid == "NULL":
            break
        if nb < 8:
            raise ValueError(f"bad block length {nb} for {bid} at {off}")
        off += nb
    return out, data


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        blocks, _ = read_dorade_blocks(sys.argv[1])
        for b in blocks[:20]:
            print(b)
        print(f"... {len(blocks)} blocks to NULL")
    else:
        print(__doc__)