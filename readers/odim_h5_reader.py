#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""
readers/odim_h5_reader.py — Pure-netCDF4 ODIM_H5 (EUMETNET OPERA) reader for GURT.

Lives in the ``readers/`` package alongside ``cfradial_reader.py``,
``dorade_reader.py`` and ``grib2_reader.py``.  Re-exported from
``readers/__init__.py`` so ``main.py`` can do:

    from readers import read_odim_h5, is_odim_h5

``read_odim_h5(path)`` returns a ``RadarVolume`` populated with
``SweepData`` objects — the same types every other GURT reader produces.

ODIM_H5 files are HDF5, but their layout is completely different from
CfRadial: instead of flat (nrays_total, ngates) variables at the root,
each sweep lives in its own group (``/dataset1``, ``/dataset2``, ...)
and each moment/quantity lives in a sub-group of that
(``/dataset1/data1``, ``/dataset1/data2``, ...). Metadata is stored as
HDF5 attributes on ``what``/``where``/``how`` sub-groups rather than as
CF variable attributes. netCDF4 can still open these files (they are valid HDF5), it just has to
walk the group tree instead of reading top-level variables. Some vendor
exports (Environment Canada/ECCC's ODIM_H5 in particular) prepend an
arbitrary-length ASCII header before the actual HDF5 stream begins;
libhdf5's own superblock search only checks a fixed set of offsets
(0, then powers of two from 512 up), so a header of unusual length is
often invisible to it, and a plain ``netCDF4.Dataset(path)`` call fails
outright. ``_open_hdf5()`` below finds the real signature by scanning
the file's bytes directly and, when it isn't at offset 0, opens the
file in netCDF4's in-memory mode starting from that offset instead.

Layout this reader understands (ODIM_H5 v2.1 / v2.2, PVOL/SCAN objects)
------------------------------------------------------------------------
/what          @source, @date, @time, @object
/where         @lat, @lon, @height
/datasetN/where   @elangle, @nbins, @nrays, @rstart, @rscale, @a1gate
/datasetN/what    @startdate, @starttime, @product
/datasetN/how     @startazA, @stopazA, @elangles, @NI       (optional)
/datasetN/dataM/what   @quantity, @gain, @offset, @nodata, @undetect
/datasetN/dataM/data   raw counts, shape (nrays, nbins)

Azimuth resolution (in order of preference), per sweep
--------------------------------------------------------
1. ``how/startazA`` + ``how/stopazA`` (mean of start/stop per ray)
2. Uniform spacing derived from ``nrays`` and ``a1gate`` (ray 0 sits at
   ``a1gate`` in the physical scan, i.e. azimuth 0 is ``nrays - a1gate``
   slots after the first stored ray).

Only SCAN/PVOL (PPI) volumes are handled; ODIM RHI/XSEC objects are not
supported by this reader.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

try:
    import netCDF4
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _attr(group, name, default=None):
    """Read an HDF5 attribute off a netCDF4 Group/Dataset, or return default."""
    try:
        v = getattr(group, name)
    except AttributeError:
        return default
    if isinstance(v, (bytes, np.bytes_)):
        v = v.decode(errors="ignore")
    if isinstance(v, np.ndarray) and v.size == 1:
        v = v.item()
    return v


def _dataset_groups(nc) -> list[str]:
    """Sorted list of 'datasetN' group names actually present."""
    nums = []
    for name in nc.groups:
        if name.startswith("dataset") and name[7:].isdigit():
            nums.append(int(name[7:]))
    return [f"dataset{n}" for n in sorted(nums)]


def _data_groups(dset_grp) -> list[str]:
    nums = []
    for name in dset_grp.groups:
        if name.startswith("data") and name[4:].isdigit():
            nums.append(int(name[4:]))
    return [f"data{n}" for n in sorted(nums)]


def _parse_odim_datetime(date_str, time_str) -> Optional[datetime]:
    """ODIM dates are 'YYYYMMDD', times are 'HHMMSS'."""
    if not date_str or not time_str:
        return None
    try:
        return datetime.strptime(f"{date_str}{time_str}"[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _radar_name_from_source(source: str) -> str:
    """
    ODIM 'source' is a comma-separated key:value list, e.g.
    'WMO:02588,RAD:SE41,PLC:Angelholm,NOD:sease'. Prefer a human-readable
    place name (PLC), then node id (NOD), then WMO number.
    """
    if not source:
        return "Unknown"
    parts = {}
    for kv in str(source).split(","):
        if ":" in kv:
            k, _, v = kv.partition(":")
            parts[k.strip().upper()] = v.strip()
    for key in ("PLC", "NOD", "RAD", "WMO"):
        if parts.get(key):
            return parts[key]
    return str(source).strip() or "Unknown"


def _sweep_azimuths(how_grp, nrays: int, a1gate: int) -> np.ndarray:
    start = _attr(how_grp, "startazA") if how_grp is not None else None
    stop  = _attr(how_grp, "stopazA")  if how_grp is not None else None
    if start is not None and stop is not None:
        start = np.asarray(start, dtype="float64")
        stop  = np.asarray(stop, dtype="float64")
        if len(start) == nrays and len(stop) == nrays:
            diff = stop - start
            diff = np.where(diff < -180.0, diff + 360.0, diff)
            diff = np.where(diff > 180.0, diff - 360.0, diff)
            az = (start + diff / 2.0) % 360.0
            return az.astype("float32")
    # Fallback: uniform ray spacing, ray index 0 stored at physical
    # azimuth position a1gate.
    step = 360.0 / nrays
    idx = np.arange(nrays, dtype="float64")
    az = (idx * step - a1gate * step) % 360.0
    return az.astype("float32")


def _sweep_elevations(how_grp, nrays: int, elangle: float) -> np.ndarray:
    elangles = _attr(how_grp, "elangles") if how_grp is not None else None
    if elangles is not None:
        arr = np.asarray(elangles, dtype="float32")
        if len(arr) == nrays:
            return arr
    return np.full(nrays, float(elangle), dtype="float32")


# ──────────────────────────────────────────────────────────────────────────
# HDF5 open — scan for the real signature rather than assume offset 0
# ──────────────────────────────────────────────────────────────────────────

_HDF5_SIGNATURE  = b"\x89HDF\r\n\x1a\n"
_HDF5_SCAN_BYTES = 65536  # generous headroom for vendor prefixes


def _find_hdf5_offset(path: str) -> Optional[int]:
    """
    Byte offset of the real HDF5 superblock signature in *path*, scanning
    the first ``_HDF5_SCAN_BYTES`` bytes directly rather than assuming
    it's at offset 0. Returns ``None`` if no signature is found in that
    window.
    """
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(_HDF5_SCAN_BYTES)
    except OSError:
        return None
    idx = chunk.find(_HDF5_SIGNATURE)
    return idx if idx >= 0 else None


def _open_hdf5(path: str):
    """
    Open *path* as a netCDF4/HDF5 ``Dataset``, transparently skipping any
    prefix bytes before the real HDF5 signature. Some ECCC ODIM_H5 files
    carry a short ASCII header before the HDF5 stream at an offset libhdf5
    doesn't check on its own, which makes a plain
    ``netCDF4.Dataset(path)`` raise. When the signature isn't at byte 0,
    this reads the file from that offset onward into memory and opens it
    via netCDF4's in-memory mode instead; the common byte-0 case is opened
    straight from disk as usual, with no extra I/O.
    """
    offset = _find_hdf5_offset(path)
    if offset is None:
        raise OSError(
            f"No HDF5 signature found in the first {_HDF5_SCAN_BYTES} "
            f"bytes: {path}"
        )
    if offset == 0:
        return netCDF4.Dataset(path, "r")
    with open(path, "rb") as fh:
        fh.seek(offset)
        data = fh.read()
    # The filename here is just a label for the in-memory dataset, not
    # touched on disk.
    return netCDF4.Dataset(os.path.basename(path), memory=data, mode="r")


# ──────────────────────────────────────────────────────────────────────────
# Format detection
# ──────────────────────────────────────────────────────────────────────────

def is_odim_h5(path: str) -> bool:
    """
    True if ``path`` looks like an ODIM_H5 (EUMETNET OPERA) file rather
    than plain CfRadial/CF-NetCDF. Both are HDF5, so we distinguish them
    by layout: ODIM_H5 has top-level 'what'/'where' *groups* holding
    attributes (not root variables) plus 'datasetN' sweep groups; a
    CfRadial file has flat root variables named 'time'/'azimuth'/etc.
    and no such groups.
    """
    if not NETCDF4_AVAILABLE:
        return False
    try:
        with _open_hdf5(path) as nc:
            conv = str(getattr(nc, "Conventions", "") or "")
            if "ODIM_H5" in conv.upper():
                return True
            has_dataset_groups = any(
                g.startswith("dataset") and g[7:].isdigit() for g in nc.groups
            )
            has_what_where_groups = "what" in nc.groups and "where" in nc.groups
            looks_cfradial = "time" in nc.variables and "azimuth" in nc.variables
            return has_dataset_groups and has_what_where_groups and not looks_cfradial
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────

def read_odim_h5(path: str, RadarVolume=None, SweepData=None):
    """
    Read an ODIM_H5 (EUMETNET OPERA) polar volume/scan and return a
    ``RadarVolume``.

    ``RadarVolume`` and ``SweepData`` are imported from ``main`` by
    default; pass them explicitly to use this module standalone.
    """
    if not NETCDF4_AVAILABLE:
        raise RuntimeError(
            "netCDF4 is required to read ODIM_H5 files (pip install netCDF4)."
        )

    if RadarVolume is None or SweepData is None:
        from main import RadarVolume, SweepData  # type: ignore[assignment]

    nc  = _open_hdf5(path)
    vol = RadarVolume()

    root_what  = nc.groups.get("what")
    root_where = nc.groups.get("where")
    root_how   = nc.groups.get("how")

    vol.radar_name = _radar_name_from_source(_attr(root_what, "source", ""))
    vol.latitude   = float(_attr(root_where, "lat", 0.0) or 0.0)
    vol.longitude  = float(_attr(root_where, "lon", 0.0) or 0.0)
    vol.altitude   = float(_attr(root_where, "height", 0.0) or 0.0)

    vol.scan_time = (
        _parse_odim_datetime(_attr(root_what, "date"), _attr(root_what, "time"))
        or datetime(2000, 1, 1)
    )

    dataset_names = _dataset_groups(nc)
    if not dataset_names:
        nc.close()
        raise ValueError(f"No ODIM 'datasetN' sweep groups found: {path}")

    all_field_names: set[str] = set()
    vol.instrument_parameters = {}

    # NI ('Nyquist Interval', i.e. Nyquist velocity in m/s) usually lives
    # in the root /how group as a volume-wide value, and/or per-sweep in
    # /datasetN/how — the per-sweep value (read below) takes precedence
    # when both are present, since PRF can differ cut-to-cut.
    root_ni = _attr(root_how, "NI", None)
    if root_ni is not None:
        vol.instrument_parameters["nyquist_velocity"] = {
            "data": np.array([float(root_ni)], dtype="float32")
        }

    for dname in dataset_names:
        dset = nc.groups[dname]
        d_where = dset.groups.get("where")
        d_what  = dset.groups.get("what")
        d_how   = dset.groups.get("how")

        if d_where is None:
            continue  # malformed sweep group, skip it

        elangle = float(_attr(d_where, "elangle", 0.0) or 0.0)
        nbins   = int(_attr(d_where, "nbins", 0) or 0)
        nrays   = int(_attr(d_where, "nrays", 0) or 0)
        rstart  = float(_attr(d_where, "rstart", 0.0) or 0.0)   # km
        rscale  = float(_attr(d_where, "rscale", 0.0) or 0.0)   # m
        a1gate  = int(_attr(d_where, "a1gate", 0) or 0)

        if nbins == 0 or nrays == 0:
            continue

        sw = SweepData()
        sw.radar_name  = vol.radar_name
        sw.fixed_angle = elangle
        sw.scan_type   = "ppi"
        sw.ranges      = (rstart * 1000.0
                           + np.arange(nbins, dtype="float64") * rscale
                           ).astype("float32")
        sw.azimuth     = _sweep_azimuths(d_how, nrays, a1gate)
        sw.elevation   = _sweep_elevations(d_how, nrays, elangle)

        sweep_dt = (
            _parse_odim_datetime(_attr(d_what, "startdate"), _attr(d_what, "starttime"))
            or vol.scan_time
        )
        sw.scan_time = sweep_dt

        sweep_ni = _attr(d_how, "NI", None)
        if sweep_ni is not None:
            sw.instrument_parameters["nyquist_velocity"] = {
                "data": np.array([float(sweep_ni)], dtype="float32")
            }

        for data_name in _data_groups(dset):
            dgrp   = dset.groups[data_name]
            d_data_what = dgrp.groups.get("what")
            if d_data_what is None or "data" not in dgrp.variables:
                continue

            quantity = str(_attr(d_data_what, "quantity", data_name)).strip()
            gain     = float(_attr(d_data_what, "gain", 1.0) or 1.0)
            offset   = float(_attr(d_data_what, "offset", 0.0) or 0.0)
            nodata   = _attr(d_data_what, "nodata", None)
            undetect = _attr(d_data_what, "undetect", None)

            raw = np.asarray(dgrp.variables["data"][:])
            if raw.shape != (nrays, nbins):
                # Some encoders swap axes or pad; try a best-effort reshape,
                # otherwise skip this quantity rather than corrupting data.
                if raw.T.shape == (nrays, nbins):
                    raw = raw.T
                else:
                    continue

            phys = raw.astype("float64") * gain + offset
            # 'nodata'   = no coverage at this gate (out of range, blocked) — mask.
            # 'undetect' = radar looked and found nothing above the noise
            # floor. Technically a real measurement (often ~-32 dBZ), but
            # rendering it as data paints the whole no-echo background a
            # solid color, so treat it as missing too.
            invalid = np.zeros(raw.shape, dtype=bool)
            if nodata is not None:
                invalid |= (raw == nodata)
            if undetect is not None:
                invalid |= (raw == undetect)
            # Write NaN into the array itself, not just the mask flag.
            # GURT's PPI gridding step (_grid_ppi_for_image in gurt_gui.py)
            # copies ray data into a plain ndarray, which silently drops a
            # masked array's mask and keeps only its underlying numbers —
            # so a *mask-only* flag doesn't survive to the renderer. Every
            # other GURT reader sidesteps this by using masked_invalid on
            # data that's already NaN at those spots; do the same here.
            phys[invalid] = np.nan

            sw.fields[quantity] = np.ma.masked_invalid(phys.astype("float32"))
            all_field_names.add(quantity)

        vol.sweeps.append(sw)

    vol.field_names = sorted(all_field_names)
    nc.close()
    return vol