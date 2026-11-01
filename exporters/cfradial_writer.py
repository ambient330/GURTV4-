#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""
exporters/cfradial_writer.py — Pure-netCDF4 CF-Radial 1.4 writer for GURT.

No pyart dependency.  Lives in the ``exporters`` package alongside the
NEXRAD MSG31 writer.  Re-exported from ``exporters/__init__.py`` so
``main.py`` can simply do:

    from exporters import write_cfradial

    dest = write_cfradial(vol, out_dir)                    # all sweeps
    dest = write_cfradial(vol, out_dir, sweep_indices=[0]) # one sweep

Naming and structure follow lrose-core's Radx CfRadial writer
(``RadxNcfStr.cc`` / ``NcfRadxFile_write.cc``, Mike Dixon / NCAR — BSD
licence) as closely as GURT's data model allows, so files written here
read cleanly in RadxConvert, Radx-based displays, and other strict
CF-Radial-1.4 tools.  Anything lrose-core tracks that GURT's
``RadarVolume`` / ``SweepData`` model has no data for (calibration
tables, georeference corrections, lidar parameters, per-ray transmit
power / estimated noise) is simply omitted rather than fabricated.

Global attributes written
--------------------------
    Conventions ("CF-1.7"), sub_conventions ("CF-Radial instrument_parameters
    radar_parameters"), version ("CF-Radial-1.4"), title, institution,
    references, source, history, comment, original_format, driver, created
    instrument_name, site_name, scan_name, scan_id
    platform_is_mobile, n_gates_vary, ray_times_increase
    start_datetime / start_time, end_datetime / end_time
    time_coverage_start, time_coverage_end

Variables beyond the CF-Radial minimum (following RadxNcfStr naming)
----------------------------------------------------------------------
    volume_number, platform_type, primary_axis, status_xml, instrument_type
    grid_mapping (+ projection-origin attrs), latitude, longitude, altitude
    sweep_number, sweep_mode, fixed_angle,
    sweep_start_ray_index, sweep_end_ray_index,
    rays_are_indexed, ray_angle_res         — derived from actual ray spacing
    ray_start_range, ray_gate_spacing
    nyquist_velocity, prt, prt_ratio, unambiguous_range
        — broadcast per-ray from SweepData.instrument_parameters (falling
          back to RadarVolume.instrument_parameters), only written for
          parameters that actually have data somewhere in the volume
    frequency                               — if present in instrument_parameters

RHI sweeps
----------
Sweeps with ``sw.scan_type == "rhi"`` are written with
``sweep_mode = "rhi"`` and ``fixed_angle`` set to the sweep's azimuth
(median of ray azimuths) rather than the elevation, matching the CF-Radial
convention for RHI files.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import numpy as np

try:
    import netCDF4
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False

# ── RadxNcfStr-style name/long_name/units constants (subset GURT needs) ────
_CF_CONVENTION      = "CF-1.7"
_BASE_CONVENTION    = "CF-Radial"
_CF_VERSION         = "CF-Radial-1.4"
_MODE_PPI           = "azimuth_surveillance"
_MODE_RHI           = "rhi"

_LONG = {
    "range":                "range_to_center_of_measurement_volume",
    "volume_number":        "data_volume_index_number",
    "platform_type":        "platform_type",
    "primary_axis":         "primary_axis_of_rotation",
    "instrument_type":      "type_of_instrument",
    "time_coverage_start":  "data_volume_start_time_utc",
    "time_coverage_end":    "data_volume_end_time_utc",
    "latitude":              "latitude",
    "longitude":             "longitude",
    "altitude":              "altitude",
    "sweep_number":          "sweep_index_number_0_based",
    "sweep_mode":            "scan_mode_for_sweep",
    "fixed_angle":           "fixed_angle_for_sweep",
    "sweep_start_ray_index": "index_of_first_ray_in_sweep",
    "sweep_end_ray_index":   "index_of_last_ray_in_sweep",
    "rays_are_indexed":      "flag_for_indexed_rays",
    "ray_angle_res":         "angular_resolution_between_rays",
    "azimuth":               "azimuth_angle_from_true_north",
    "elevation":             "elevation_angle_from_horizontal_plane",
    "nyquist_velocity":      "unambiguous_doppler_velocity",
    "prt":                   "pulse_repetition_time",
    "prt_ratio":             "pulse_repetition_frequency_ratio",
    "unambiguous_range":     "unambiguous_range",
    "frequency":             "transmission_frequency",
}
_AZIMUTH_STANDARD   = "ray_azimuth_angle"
_ELEVATION_STANDARD = "ray_elevation_angle"


# ──────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────

def _w3c_str(dt: Optional[datetime]) -> str:
    """W3C datetime string, e.g. 2024-01-01T00:00:00Z (matches RadxTime::getW3cStr)."""
    if dt is None:
        return "unknown"
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _dashed_str(dt: Optional[datetime]) -> str:
    """Dashed datetime string with millisecond precision (matches
    RadxTime::asStringDashed(3)), e.g. 2024-01-01 00:00:00.000"""
    if dt is None:
        return "unknown"
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def _gate_spacing(ranges: np.ndarray) -> float:
    if len(ranges) > 1:
        return float(np.median(np.diff(ranges.astype("float64"))))
    return float(ranges[0]) if len(ranges) else 1.0


def _pad_gates(arr2d: np.ndarray, n: int) -> np.ndarray:
    """Pad or truncate the gate (column) axis of arr2d to width n."""
    n_have = arr2d.shape[1]
    if n_have == n:
        return arr2d
    if n_have > n:
        return arr2d[:, :n]
    pad = np.full((arr2d.shape[0], n - n_have), np.nan, dtype="float32")
    filled = np.ma.filled(arr2d, np.nan).astype("float32")
    return np.concatenate([filled, pad], axis=1)


def _str_to_chars(s: str, length: int) -> np.ndarray:
    """
    Encode a string into a fixed-length numpy 'S1' char array.

    Avoids netCDF4.stringtochar(), whose behavior with bytes-dtype input
    changed across netCDF4-python versions (it raises
    AttributeError: 'numpy.bytes_' object has no attribute 'encode' on
    1.7.x with an 'SN' input array) -- this implementation behaves the
    same regardless of installed netCDF4 version.
    """
    b = (s or "").encode("ascii", "replace").ljust(length)[:length]
    return np.frombuffer(b, dtype="S1")


def _write_strvar(nc, varname: str, dim_name: str, str_dim: str,
                  strings: list[str]) -> None:
    """Write a list of fixed-width strings as a netCDF4 char variable."""
    max_len = int(nc.dimensions[str_dim].size)
    var = nc.createVariable(varname, "S1", (dim_name, str_dim))
    for i, s in enumerate(strings):
        var[i, :] = _str_to_chars(s, max_len)


def _write_strscalar(nc, varname: str, str_dim: str, value: str) -> None:
    """
    Write a single string as a 1-D netCDF4 char variable over `str_dim`.

    Matches the CF/Radial + Radx convention where fields such as
    time_coverage_start, time_coverage_end, platform_type, instrument_type,
    and primary_axis are stored as char-array *variables* (not just global
    attributes). Some strict CfRadial readers expect these as variables and
    will throw a text/number type-mismatch if they only find a global
    string attribute (or nothing) where a char variable is expected.
    """
    max_len = int(nc.dimensions[str_dim].size)
    var = nc.createVariable(varname, "S1", (str_dim,))
    var[:] = _str_to_chars(value, max_len)


def _ray_indexing(angles: np.ndarray, wrap360: bool):
    """
    Inspect the actual per-ray angle spacing to determine whether the sweep
    was scanned at a fixed angular step (rays_are_indexed) and, if so, what
    that step is (ray_angle_res). This mirrors what Radx reports, but is
    derived here from the real ray angles rather than assumed.
    """
    if len(angles) < 3:
        return False, float("nan")
    diffs = np.diff(angles.astype("float64"))
    if wrap360:
        diffs = ((diffs + 180.0) % 360.0) - 180.0
    diffs = np.abs(diffs)
    med = float(np.median(diffs))
    if med <= 0.0:
        return False, float("nan")
    # "indexed" if the spacing is consistent to within ~10% of the median step
    is_indexed = bool(np.all(np.abs(diffs - med) < max(0.1 * med, 0.02)))
    return is_indexed, med


def _sweep_param(sw, vol, key: str) -> Optional[float]:
    """Look up a scalar instrument_parameters value for this sweep, falling
    back to the volume-wide value (mirrors the reader's per-sweep-then-
    volume-wide resolution used by detect_stagger_nyquist())."""
    for src in (getattr(sw, "instrument_parameters", None), getattr(vol, "instrument_parameters", None)):
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


# ──────────────────────────────────────────────────────────────────────────
# Public writer
# ──────────────────────────────────────────────────────────────────────────

def write_cfradial(
    vol,
    out_dir: str,
    sweep_indices: Optional[Sequence[int]] = None,
) -> str:
    """
    Write a GURT ``RadarVolume`` as a CF-Radial 1.4 NetCDF4 file, following
    lrose-core's Radx CfRadial writer conventions.

    Parameters
    ----------
    vol : RadarVolume
        The source data.
    out_dir : str
        Destination directory (created if it does not exist).
    sweep_indices : sequence of int, optional
        Which sweep indices from *vol* to include.  Defaults to all.

    Returns
    -------
    str
        Absolute path of the written file.
    """
    if not NETCDF4_AVAILABLE:
        raise RuntimeError(
            "netCDF4 is required to write CFRadial files (pip install netCDF4)."
        )

    os.makedirs(out_dir, exist_ok=True)

    if sweep_indices is None:
        sweep_indices = list(range(len(vol.sweeps)))
    sweeps = [vol.sweeps[i] for i in sweep_indices if 0 <= i < len(vol.sweeps)]
    if not sweeps:
        raise ValueError("No sweeps selected for export.")

    # ── Common range axis: pad shorter sweeps to the longest ──────────────
    ngates_each = [len(sw.ranges) for sw in sweeps]
    ngates      = max(ngates_each)
    ref_sw      = sweeps[ngates_each.index(ngates)]
    ranges      = np.asarray(ref_sw.ranges, dtype="float32")
    spacing     = _gate_spacing(ranges)
    r0          = float(ranges[0]) if len(ranges) else 0.0
    n_gates_vary = len(set(ngates_each)) > 1

    # ── Concatenate per-sweep geometry ────────────────────────────────────
    az_parts, el_parts   = [], []
    sweep_start, sweep_end = [], []
    fixed_angles_out     = []
    sweep_mode_strs      = []
    ray_times_parts: list[np.ndarray] = []
    rays_indexed_out: list[str] = []
    ray_angle_res_out: list[float] = []

    # per-ray instrument-parameter broadcasts (only built for keys that have
    # real data somewhere in the volume -- see below)
    ip_keys = ("nyquist_velocity", "prt", "prt_ratio")
    ip_parts: dict[str, list[np.ndarray]] = {k: [] for k in ip_keys}
    ip_has_any: dict[str, bool] = {k: False for k in ip_keys}

    cursor = 0
    base_dt = sweeps[0].scan_time or vol.scan_time or datetime(2000, 1, 1)

    for sw in sweeps:
        nrays = len(sw.azimuth)
        az = np.asarray(sw.azimuth,   dtype="float32")
        el = np.asarray(sw.elevation, dtype="float32")
        az_parts.append(az)
        el_parts.append(el)

        is_rhi = getattr(sw, "scan_type", "ppi") == "rhi"
        sweep_mode_strs.append(_MODE_RHI if is_rhi else _MODE_PPI)

        # For RHI, fixed_angle is the azimuth (it's fixed while elevation scans)
        if is_rhi:
            fixed_angles_out.append(float(np.nanmedian(az)))
        else:
            fixed_angles_out.append(float(sw.fixed_angle))

        # Derive rays_are_indexed / ray_angle_res from the actual ray angles
        indexed, res = _ray_indexing(el if is_rhi else az, wrap360=not is_rhi)
        rays_indexed_out.append("true" if indexed else "false")
        ray_angle_res_out.append(res)

        sweep_start.append(cursor)
        sweep_end.append(cursor + nrays - 1)
        cursor += nrays

        sw_t0  = sw.scan_time or base_dt
        offset = (sw_t0 - base_dt).total_seconds()
        ray_times_parts.append(
            offset + np.linspace(0.0, max(nrays - 1, 0), nrays, dtype="float64")
        )

        for key in ip_keys:
            val = _sweep_param(sw, vol, key)
            if val is not None:
                ip_has_any[key] = True
                ip_parts[key].append(np.full(nrays, val, dtype="float32"))
            else:
                ip_parts[key].append(np.full(nrays, np.nan, dtype="float32"))

    azimuth_data   = np.concatenate(az_parts)
    elevation_data = np.concatenate(el_parts)
    time_data      = np.concatenate(ray_times_parts)
    nrays_total    = len(azimuth_data)

    try:
        end_dt = base_dt + timedelta(seconds=float(time_data[-1]))
    except Exception:
        end_dt = base_dt

    # unambiguous_range: derive per-ray from nyquist-implied max range isn't
    # meaningful here, so use gate_spacing * n_gates like the previous
    # writer -- broadcast the same scalar to every ray for consistency with
    # the other instrument_parameters variables.
    unambiguous_range_m = spacing * ngates

    # ── Field union across all selected sweeps ────────────────────────────
    field_names = list(dict.fromkeys(
        name for sw in sweeps for name in sw.fields.keys()
    ))

    # ── Output filename ───────────────────────────────────────────────────
    ts        = base_dt.strftime("%Y%m%d_%H%M%S")
    site      = (vol.radar_name or "UNKN").strip().replace(" ", "_") or "UNKN"
    scan_tag  = "RHI" if sweep_mode_strs and sweep_mode_strs[0] == _MODE_RHI else "SUR"
    dest      = os.path.join(out_dir, f"cfrad.{ts}_{site}_{scan_tag}.nc")

    # ── Write NetCDF4 ──────────────────────────────────────────────────────
    nc = netCDF4.Dataset(dest, "w", format="NETCDF4")

    # Dimensions (RadxNcfStr::_addDimensions)
    nc.createDimension("time",             nrays_total)
    nc.createDimension("range",            ngates)
    nc.createDimension("sweep",            len(sweeps))
    nc.createDimension("string_length_8",  8)
    nc.createDimension("string_length_32", 32)
    nc.createDimension("status_xml_length", 1)

    # ── Global attributes (RadxNcfStr::_addGlobalAttributes) ───────────────
    sub_conventions = f"{_BASE_CONVENTION} instrument_parameters radar_parameters"

    nc.Conventions          = _CF_CONVENTION
    nc.sub_conventions      = sub_conventions
    nc.version              = _CF_VERSION
    nc.title                = "GURT CFRadial export"
    nc.institution           = ""
    nc.references            = ""
    nc.source                = "GURT"
    nc.history               = ""
    nc.comment               = ""
    nc.original_format       = "CFRADIAL"
    nc.driver                = "GURT exporters/cfradial_writer.py"
    nc.created               = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    nc.instrument_name      = vol.radar_name or "Unknown"
    nc.site_name            = vol.radar_name or "Unknown"
    nc.scan_name            = scan_tag
    nc.scan_id              = int(getattr(vol, "vcp_number", 0) or 0)
    nc.platform_is_mobile   = "false"
    nc.n_gates_vary         = "true" if n_gates_vary else "false"
    nc.ray_times_increase   = "true"

    nc.start_datetime       = _w3c_str(base_dt)
    nc.time_coverage_start  = _w3c_str(base_dt)
    nc.start_time           = _dashed_str(base_dt)

    nc.end_datetime         = _w3c_str(end_dt)
    nc.time_coverage_end    = _w3c_str(end_dt)
    nc.end_time             = _dashed_str(end_dt)

    # ── "Global variables" (RadxNcfStr / CF-Radial §4.3) ────────────────────
    # These MUST be real variables, not just global attributes -- strict
    # readers (incl. RadX/Radx-derived and some GIS NetCDF importers) look
    # them up as typed variables and raise a text/number conversion error
    # if only a string global attribute is present where a variable (or an
    # int variable, in volume_number's case) is expected.
    v = nc.createVariable("volume_number", "i4", ())
    v.long_name = _LONG["volume_number"]
    v.units     = "unitless"
    v.assignValue(int(getattr(vol, "volume_number", 0) or 0))

    _write_strscalar(nc, "platform_type", "string_length_32", "fixed")
    nc.variables["platform_type"].long_name = _LONG["platform_type"]
    nc.variables["platform_type"].options   = (
        "fixed vehicle ship aircraft_fore aircraft_aft aircraft_tail "
        "aircraft_belly aircraft_roof aircraft_nose satellite_orbit "
        "satellite_geostat"
    )

    _write_strscalar(nc, "primary_axis", "string_length_32", "axis_z")
    nc.variables["primary_axis"].long_name = _LONG["primary_axis"]
    nc.variables["primary_axis"].options   = "axis_z axis_y axis_x axis_z_prime axis_y_prime axis_x_prime"

    _write_strscalar(nc, "status_xml", "status_xml_length", "")
    nc.variables["status_xml"].long_name = "status_of_instrument"

    _write_strscalar(nc, "instrument_type", "string_length_32", "radar")
    nc.variables["instrument_type"].long_name  = _LONG["instrument_type"]
    nc.variables["instrument_type"].options    = "radar lidar"
    nc.variables["instrument_type"].meta_group = "instrument_parameters"

    _write_strscalar(nc, "time_coverage_start", "string_length_32",
                      base_dt.isoformat() + "Z")
    nc.variables["time_coverage_start"].long_name = _LONG["time_coverage_start"]
    nc.variables["time_coverage_start"].units     = "unitless"

    _write_strscalar(nc, "time_coverage_end", "string_length_32",
                      end_dt.isoformat() + "Z")
    nc.variables["time_coverage_end"].long_name = _LONG["time_coverage_end"]
    nc.variables["time_coverage_end"].units     = "unitless"

    # ── time ──────────────────────────────────────────────────────────────
    v = nc.createVariable("time", "f8", ("time",))
    v.units         = f"seconds since {base_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    v.standard_name = "time"
    v.long_name     = "time_in_seconds_since_volume_start"
    v.calendar      = "gregorian"
    v.comment       = "times are relative to the volume start_time"
    v[:]            = time_data

    # ── range ─────────────────────────────────────────────────────────────
    v = nc.createVariable("range", "f4", ("range",))
    v.units                          = "meters"
    v.standard_name                  = "projection_range_coordinate"
    v.long_name                      = _LONG["range"]
    v.axis                           = "radial_range_coordinate"
    v.spacing_is_constant            = "true"
    v.meters_to_center_of_first_gate = r0
    v.meters_between_gates           = spacing
    v[:]                             = ranges

    # ── azimuth ───────────────────────────────────────────────────────────
    v = nc.createVariable("azimuth", "f4", ("time",))
    v.units         = "degrees"
    v.standard_name = _AZIMUTH_STANDARD
    v.long_name     = _LONG["azimuth"]
    v[:]            = azimuth_data

    # ── elevation ─────────────────────────────────────────────────────────
    v = nc.createVariable("elevation", "f4", ("time",))
    v.units         = "degrees"
    v.standard_name = _ELEVATION_STANDARD
    v.long_name     = _LONG["elevation"]
    v.positive      = "up"
    v[:]            = elevation_data

    # ── grid_mapping (projection) variable ──────────────────────────────────
    proj = nc.createVariable("grid_mapping", "i4", ())
    proj.grid_mapping_name = "radar_lidar_radial_scan"
    proj.longitude_of_projection_origin = float(vol.longitude)
    proj.latitude_of_projection_origin  = float(vol.latitude)
    proj.altitude_of_projection_origin  = float(vol.altitude)
    proj.false_northing = 0.0
    proj.false_easting  = 0.0

    # ── latitude / longitude / altitude ───────────────────────────────────
    v = nc.createVariable("latitude",  "f8", ())
    v.long_name = _LONG["latitude"];  v.units = "degrees_north"; v.assignValue(vol.latitude)
    v = nc.createVariable("longitude", "f8", ())
    v.long_name = _LONG["longitude"]; v.units = "degrees_east";  v.assignValue(vol.longitude)
    v = nc.createVariable("altitude",  "f8", ())
    v.long_name = _LONG["altitude"];  v.units = "meters"; v.positive = "up"
    v.assignValue(vol.altitude)

    # ── sweep index arrays (RadxNcfStr::_addSweepVariables) ─────────────────
    v = nc.createVariable("sweep_number", "i4", ("sweep",))
    v.long_name = _LONG["sweep_number"]
    v[:] = np.arange(len(sweeps), dtype="int32")

    _write_strvar(nc, "sweep_mode", "sweep", "string_length_32", sweep_mode_strs)
    nc.variables["sweep_mode"].long_name = _LONG["sweep_mode"]
    nc.variables["sweep_mode"].options   = "sector coplane rhi vertical_pointing idle azimuth_surveillance elevation_surveillance sunscan pointing manual_ppi manual_rhi"

    v = nc.createVariable("fixed_angle", "f4", ("sweep",))
    v.units     = "degrees"
    v.long_name = _LONG["fixed_angle"]
    v[:] = np.array(fixed_angles_out, dtype="float32")

    v = nc.createVariable("sweep_start_ray_index", "i4", ("sweep",))
    v.long_name = _LONG["sweep_start_ray_index"]
    v[:] = np.array(sweep_start, dtype="int32")

    v = nc.createVariable("sweep_end_ray_index", "i4", ("sweep",))
    v.long_name = _LONG["sweep_end_ray_index"]
    v[:] = np.array(sweep_end, dtype="int32")

    _write_strvar(nc, "rays_are_indexed", "sweep", "string_length_8", rays_indexed_out)
    nc.variables["rays_are_indexed"].long_name = _LONG["rays_are_indexed"]

    v = nc.createVariable("ray_angle_res", "f4", ("sweep",))
    v.units     = "degrees"
    v.long_name = _LONG["ray_angle_res"]
    v[:] = np.array(ray_angle_res_out, dtype="float32")

    # ── Extra range-derived ray variables ────────────────────────────────────
    v = nc.createVariable("ray_start_range", "f4", ("time",))
    v.units     = "meters"
    v.long_name = "start_range_for_ray"
    v[:] = np.full(nrays_total, r0, dtype="float32")

    v = nc.createVariable("ray_gate_spacing", "f4", ("time",))
    v.units     = "meters"
    v.long_name = "gate_spacing_for_ray"
    v[:] = np.full(nrays_total, spacing, dtype="float32")

    # ── unambiguous_range (per-ray, RadxNcfStr::UNAMBIGUOUS_RANGE) ─────────
    v = nc.createVariable("unambiguous_range", "f4", ("time",))
    v.units     = "meters"
    v.long_name = _LONG["unambiguous_range"]
    v.meta_group = "instrument_parameters"
    v[:] = np.full(nrays_total, unambiguous_range_m, dtype="float32")

    # ── nyquist_velocity / prt / prt_ratio (per-ray, only if real data) ────
    for key, units in (
        ("nyquist_velocity", "meters per second"),
        ("prt",              "seconds"),
        ("prt_ratio",        "seconds"),
    ):
        if not ip_has_any[key]:
            continue
        data = np.concatenate(ip_parts[key])
        v = nc.createVariable(key, "f4", ("time",), fill_value=np.nan)
        v.units      = units
        v.long_name  = _LONG[key]
        v.meta_group = "instrument_parameters"
        v[:] = data

    # ── frequency (volume-wide, RadxNcfStr::_addFrequencyVariable) ─────────
    freq_val = _sweep_param(sweeps[0], vol, "frequency")
    if freq_val is not None:
        nc.createDimension("frequency", 1)
        v = nc.createVariable("frequency", "f4", ("frequency",))
        v.units      = "s-1"
        v.long_name  = _LONG["frequency"]
        v.meta_group = "instrument_parameters"
        v[:] = np.array([freq_val], dtype="float32")

    # ── Field data ─────────────────────────────────────────────────────────
    coordinates_str = "elevation azimuth range"
    for fname in field_names:
        rows = []
        for sw in sweeps:
            nrays = len(sw.azimuth)
            arr   = sw.fields.get(fname)
            if arr is None:
                rows.append(np.full((nrays, ngates), np.nan, dtype="float32"))
            else:
                a = np.ma.filled(arr, np.nan).astype("float32")
                rows.append(_pad_gates(a, ngates) if a.shape[1] != ngates else a)
        data = np.ma.masked_invalid(np.concatenate(rows, axis=0))

        vf = nc.createVariable(
            fname, "f4", ("time", "range"),
            fill_value=-32767, zlib=True, complevel=4,
        )
        vf.long_name     = fname
        vf.standard_name = fname
        vf.grid_mapping  = "grid_mapping"
        vf.coordinates   = coordinates_str
        vf.units = " "
        vf[:] = data

    nc.close()
    return os.path.abspath(dest)