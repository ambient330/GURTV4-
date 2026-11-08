#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
"""GURT core data model and file loaders (no GUI dependencies).

Shared by the GUI (main.py) and the headless command-line tools (gurt_cli.py).
"""

import os
from datetime import datetime, timezone

import numpy as np

from readers.detect_format import detect_filetype
from readers.nclock import serialized as _serialized
from readers import (
    DoradeFile,
    Grib2File,
    read_cfradial,
    read_odim_h5, is_odim_h5,
    read_ukmo, is_ukmo,
    read_nexrad_level2, is_nexrad_level2,
)

# ══════════════════════════════════════════════════════════════════════════════
# Radar data model
# ══════════════════════════════════════════════════════════════════════════════

class SweepData:
    """Single PPI sweep ready for rendering."""
    def __init__(self):
        self.azimuth     = None   # (nrays,)  degrees
        self.elevation   = None   # (nrays,)  degrees
        self.ranges      = None   # (ngates,) metres
        self.fixed_angle = 0.0
        self.fields      = {}     # name -> (nrays, ngates) float32 masked array
        # name -> (nrays, ngates) bool array, True where the source data
        # flagged the gate as range-folded (distinct from ordinary missing/
        # below-threshold data). Optional -- loaders that can't distinguish
        # RF from plain missing data just leave this empty.
        self.range_folded = {}
        self.radar_name  = "Unknown"
        self.scan_time   = None
        self.scan_type   = "ppi"  # "ppi" or "rhi"
        # Per-sweep instrument parameters (nyquist_velocity, prt, prt_ratio,
        # frequency), same dict shape as RadarVolume.instrument_parameters.
        # Populated by loaders that can resolve metadata per-sweep (CfRadial);
        # left empty otherwise, in which case detect_stagger_nyquist() falls
        # back to the volume-wide instrument_parameters.
        self.instrument_parameters = {}


def sane_altitude_m(v, default=0.0):
    """A radar altitude in metres, or `default` when the file holds a garbage value (some DORADE
    files carry e.g. -431602 km). Plausible range: -500 m .. 12 km."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if (v == v and -500.0 <= v <= 12000.0) else default


class RadarVolume:
    """All sweeps from one file."""
    def __init__(self):
        self.sweeps      = []
        self.field_names = []
        self.radar_name  = "Unknown"
        self.scan_time   = None
        self.latitude    = 0.0    # degrees_north
        self.longitude   = 0.0    # degrees_east
        self.altitude    = 0.0    # metres MSL


# ══════════════════════════════════════════════════════════════════════════════
# DORADE loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_dorade(path) -> RadarVolume:
    df  = DoradeFile(path)
    vol = RadarVolume()
    vol.radar_name  = getattr(df, "instrument_name", "Unknown")
    vol.scan_time   = getattr(df, "sweep_start_time", None)
    vol.latitude    = float(getattr(df, "lat", 0.0) or 0.0)
    vol.longitude   = float(getattr(df, "lon", 0.0) or 0.0)
    vol.altitude    = sane_altitude_m(getattr(df, "alt", 0.0) or 0.0)
    vol.field_names = list(df.params.keys())
    vol.instrument_parameters = {}
    freq_ghz = float(getattr(df, "frequency", 0.0) or 0.0)
    if np.isfinite(freq_ghz) and freq_ghz > 0.0:
        vol.instrument_parameters["frequency"] = {
            "data": np.array([freq_ghz * 1.0e9], dtype="float64")
        }
    nyq = float(getattr(df, "nyquist_velocity", 0.0) or 0.0)
    if np.isfinite(nyq) and nyq > 0.0:
        vol.instrument_parameters["nyquist_velocity"] = {
            "data": np.array([nyq], dtype="float32")
        }
    p1_ms = float(getattr(df, "_radd_prt1", 0.0) or 0.0)
    p2_ms = float(getattr(df, "_radd_prt2", 0.0) or 0.0)
    if p1_ms > 0.0:
        prt_short_s = p1_ms / 1000.0
        vol.instrument_parameters["prt"] = {
            "data": np.array([prt_short_s], dtype="float64")
        }
        if p2_ms > 0.0:
            prt_long_s = p2_ms / 1000.0
            ps, pl = sorted((prt_short_s, prt_long_s))
            vol.instrument_parameters["prt"] = {
                "data": np.array([ps], dtype="float64")
            }
            if int(getattr(df, "_radd_num_ipps", 1) or 1) >= 2 and pl > 0.0:
                vol.instrument_parameters["prt_ratio"] = {
                    "data": np.array([ps / pl], dtype="float64")
                }

    sw = SweepData()
    sw.radar_name  = vol.radar_name
    sw.scan_time   = vol.scan_time
    sw.fixed_angle = getattr(df, "fixed_angle", 0.0)
    # DORADE stores scan mode as a string ("RHI", "PPI", "SUR", etc.)
    # For RHI, fixed_angle is already the azimuth (the angle held constant
    # while elevation scans), so no remapping is needed.
    dorade_mode = str(getattr(df, "sweep_mode", "PPI") or "PPI").upper()
    sw.scan_type = "rhi" if dorade_mode == "RHI" else "ppi"

    first_pname = next(iter(df.params))
    first_p     = df.params[first_pname]
    if "meters_to_first_cell" in first_p and first_p["meters_to_first_cell"] != 0:
        r0      = float(first_p["meters_to_first_cell"])
        spacing = float(first_p["meters_between_cells"])
        ngates  = max(len(r["rdat"][first_pname])
                      for r in df.rays if first_pname in r["rdat"])
    elif hasattr(df, "dist_cells") and df.dist_cells is not None:
        r0      = float(df.dist_cells[0])
        spacing = float(df.dist_cells[1] - df.dist_cells[0])
        ngates  = len(df.dist_cells)
    else:
        raise RuntimeError("Cannot determine range geometry from DORADE file.")

    sw.ranges    = r0 + np.arange(ngates, dtype="float32") * spacing
    sw.azimuth   = np.array([r["azimuth"]   for r in df.rays], dtype="float32")
    sw.elevation = np.array([r["elevation"] for r in df.rays], dtype="float32")

    # per-ray times, seconds since the first ray (kept so exports carry real ray times)
    try:
        t = np.array([r["jd"] * 86400.0 + r["tod"] for r in df.rays], dtype="float64")
        t -= t[0]
        if len(t) > 1 and np.all(np.isfinite(t)) and np.all(np.diff(t) >= 0) and t[-1] > 0:
            sw.ray_time = t
    except (KeyError, TypeError):
        pass

    for pname, parm in df.params.items():
        scale = parm["parameter_scale"]
        bias  = parm["parameter_bias"]
        bad   = parm["bad_data"]
        rows  = []
        for ray in df.rays:
            raw = ray["rdat"].get(pname)
            if raw is None:
                rows.append(np.full(ngates, np.nan, dtype="float32"))
                continue
            phys = np.where(raw == bad, np.nan,
                            (raw - bias) / scale).astype("float32")
            if len(phys) < ngates:
                phys = np.concatenate([phys,
                    np.full(ngates - len(phys), np.nan, "float32")])
            rows.append(phys[:ngates])
        sw.fields[pname] = np.ma.masked_invalid(np.vstack(rows))

    vol.sweeps = [sw]
    sw.instrument_parameters = dict(vol.instrument_parameters)
    vol._dorade_file = df   # carry raw DORADE object for stagger detection
    return vol


# ══════════════════════════════════════════════════════════════════════════════
# GRIB2 (JMA) loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_grib2(path) -> RadarVolume:
    """
    Load a JMA GRIB2 radar file and return a RadarVolume.
    Ported from test3.py. Pass a single refl .bin; vel merging is not
    supported via load_radar_file (open the file pair via Grib2File directly
    if needed).
    """
    from datetime import timedelta
    gf  = Grib2File(path)
    vol = RadarVolume()

    vol.radar_name = (gf.icao or "Unknown").strip()
    vol.scan_time  = gf.time_reference
    vol.latitude   = gf.latitude
    vol.longitude  = gf.longitude
    vol.altitude   = gf.altitude
    vol.instrument_parameters = {}
    if gf.frequency and gf.frequency > 0.0:
        vol.instrument_parameters["frequency"] = {
            "data": np.array([gf.frequency], dtype="float64")
        }

    # flat ray arrays
    ray_times    = np.array(gf.time,      dtype="float64")
    azimuth      = np.array(gf.azimuth,   dtype="float32")
    elevation    = np.array(gf.elevation, dtype="float32")
    ranges       = np.array(gf.radar_range_all, dtype="float32")
    start_idxs   = gf.sweep_start_ray_index
    end_idxs     = gf.sweep_end_ray_index
    fixed_angles = gf.fixed_angle

    # DBZ: stored as (physical*100 - 32), fill=MISSING_VALUE
    # decode: (int16 + 32) * 0.01, fill -> NaN
    FILL = -32768 + 32
    refl_raw = gf.refl_data.astype("float32")
    refl_phys = np.where(refl_raw == FILL, np.nan, (refl_raw + 32) * 0.01)

    # VEL: stored as physical*100, fill=MISSING_VALUE
    # decode: int16 * 0.01, fill -> NaN
    if gf.has_velocity and gf.vel_data is not None:
        vel_raw  = gf.vel_data.astype("float32")
        vel_phys = np.where(vel_raw == FILL, np.nan, vel_raw * 0.01)
        vol.field_names = ["DBZ", "VEL"]
    else:
        vel_phys = None
        vol.field_names = ["DBZ"]

    for si in range(len(start_idxs)):
        s, e = int(start_idxs[si]), int(end_idxs[si])
        sw = SweepData()
        sw.radar_name  = vol.radar_name
        sw.fixed_angle = float(fixed_angles[si])
        sw.azimuth     = azimuth[s:e + 1]
        sw.elevation   = elevation[s:e + 1]
        sw.ranges      = ranges
        try:
            sw.scan_time = gf.time_reference + timedelta(seconds=float(ray_times[s]))
        except Exception:
            sw.scan_time = gf.time_reference
        sw.ray_time = ray_times[s:e + 1] - ray_times[s]
        sw.fields["DBZ"] = np.ma.masked_invalid(refl_phys[s:e + 1, :])
        if vel_phys is not None:
            sw.fields["VEL"] = np.ma.masked_invalid(vel_phys[s:e + 1, :])
        vol.sweeps.append(sw)

    return vol


# ══════════════════════════════════════════════════════════════════════════════
# File format detection + dispatch
# ══════════════════════════════════════════════════════════════════════════════
#
# detect_filetype() sniffs magic bytes (transparently unwrapping one level
# of gzip/bz2) and returns a short tag: "DORADE", "GRIB", "NETCDF3",
# "NETCDF4", "ODIM_H5", "NEXRAD_L2", "NEXRAD_L3", "UF", "IRIS_PF", "SIGMET",
# "MDV", "GZ", "BZ2", or "UNKNOWN". See readers/detect_format.py.
#
# Several tags need a second, structural check before dispatch:
#   - "NETCDF4" / "ODIM_H5": detect_filetype() tags a file "NETCDF4" when
#     the HDF5 signature sits at byte 0 and "ODIM_H5" when it's found
#     further in (behind a vendor prefix) — but that offset alone doesn't
#     tell flat CfRadial-HDF5 apart from grouped ODIM_H5, since most
#     ODIM_H5 producers *also* put the signature at byte 0 with no prefix
#     at all. is_odim_h5() does the real check (walks the group tree) and
#     is run for both tags rather than trusted to the byte offset.
#   - "UNKNOWN": UKMO polar files have no documented magic number at all,
#     so they always land here by design (see detect_format.py's module
#     docstring) — is_ukmo() confirms them structurally instead. If that
#     also fails the file is reported as unrecognised.
#
# "NEXRAD_L2" is unambiguous by signature alone, but is_nexrad_level2() is
# still run as a cheap defensive check so a truncated/corrupt file that
# happens to pass the magic-byte test fails with a clear message here
# rather than a confusing traceback inside the reader.

@_serialized
def load_radar_file(path) -> RadarVolume:
    ftype = detect_filetype(path)

    if ftype == "DORADE":
        return _load_dorade(path)

    if ftype == "GRIB":
        return _load_grib2(path)

    if ftype == "NETCDF3":
        return read_cfradial(path, RadarVolume=RadarVolume, SweepData=SweepData)

    if ftype in ("NETCDF4", "ODIM_H5"):
        if is_odim_h5(path):
            return read_odim_h5(path, RadarVolume=RadarVolume, SweepData=SweepData)
        if ftype == "ODIM_H5":
            # Signature was behind a vendor prefix (nonzero offset) but the
            # structural check says it isn't ODIM_H5 either — most likely
            # grouped CfRadial2-HDF5, which isn't supported yet.
            raise RuntimeError(
                f"HDF5 file has a non-root signature but isn't ODIM_H5 layout — "
                f"likely grouped CfRadial2-HDF5, which isn't supported yet: "
                f"{os.path.basename(path)}"
            )
        return read_cfradial(path, RadarVolume=RadarVolume, SweepData=SweepData)

    if ftype == "NEXRAD_L2":
        if not is_nexrad_level2(path):
            raise RuntimeError(
                f"Detected NEXRAD_L2 by signature but failed verification: "
                f"{os.path.basename(path)}"
            )
        return read_nexrad_level2(path, RadarVolume, SweepData)

    if ftype == "UNKNOWN":
        # UKMO polar files have no documented magic number, so they always
        # land here — confirm structurally before giving up.
        if is_ukmo(path):
            return read_ukmo(path, RadarVolume=RadarVolume, SweepData=SweepData)
        raise RuntimeError(f"Unrecognised format: {os.path.basename(path)}")

    raise RuntimeError(
        f"Unsupported format '{ftype}' detected for {os.path.basename(path)}\n"
        "GURT's readers/ package currently handles: DORADE, GRIB2 (JMA), "
        "CfRadial (.nc), ODIM_H5, UKMO polar files, NEXRAD Level II.\n"
        "(MDV, NEXRAD Level III, UF, SIGMET/IRIS are detected but have no "
        "reader yet.)"
    )

