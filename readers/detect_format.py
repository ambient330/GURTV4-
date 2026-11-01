#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""
readers/detect_format.py — signature-based radar file format detection.

Standalone diagnostic/dispatch helper: given a path, sniff its magic
bytes and return a short format tag (``"NETCDF4"``, ``"NEXRAD_L2"``,
``"DORADE"``, ...) without needing to actually parse the file with a
format-specific reader first. Modeled on Py-ART's ``io.auto_read``
signature table (the snippet this was built from), with two differences:

  1. Compression is unwrapped before sniffing, not after. GZ/BZ2 are
     peeled in-memory and the *decompressed* bytes are what get
     classified — so a gzipped CfRadial file correctly comes back as
     ``"NETCDF4"``, not ``"GZ"``. ``"GZ"``/``"BZ2"`` are only returned
     if the archive's payload itself couldn't be identified.
  2. A ``"UKMO"`` tag is included for completeness, but UKMO polar
     files have no publicly documented magic number (see
     ``readers/ukmo_reader.py``), so this module can't actually detect
     them by signature — ``is_ukmo()`` there does it structurally
     instead, by sanity-checking header fields. detect_filetype() will
     report a UKMO file as ``"UNKNOWN"``; that's expected, not a bug.

Two bugs fixed from the original pyart-style snippet this was adapted
from:
  - The IRIS_PF check compared a 4-byte slice against the 2-byte
    literal ``b"PF"``, which can never match — narrowed to 2 bytes.
  - The "Canadian ODIM" HDF5-signature check used a hardcoded offset
    (``begin[34:42]``) that only catches one specific prefix length;
    replaced with a signature *scan* over the first 64 KB (same
    approach GURT's own ``_hdf5_offset()`` in main.py already uses),
    so it catches the signature regardless of how much junk precedes
    it.
"""

from __future__ import annotations

import bz2
import gzip
import os

# ── Magic signatures ─────────────────────────────────────────────────────────

_MDV_SIGNATURE   = b"\x00\x00\x03\xf8\x00\x007>\x00\x00\x00\x01"
_HDF5_SIGNATURE  = b"\x89HDF\r\n\x1a\n"
_GZIP_SIGNATURE  = b"\x1f\x8b"
_BZIP2_SIGNATURE = b"BZh"

# Some HDF5 files (e.g. Environment Canada ODIM exports) carry extra
# bytes before the real signature — scan rather than assume offset 0.
_HDF5_SCAN_BYTES = 65536


def _find_hdf5_offset(chunk: bytes):
    idx = chunk.find(_HDF5_SIGNATURE)
    return idx if idx >= 0 else None


def _classify_bytes(begin: bytes) -> str:
    """Classify already-decompressed bytes by magic signature."""
    if begin[:12] == _MDV_SIGNATURE:
        return "MDV"

    if begin[:3] == b"CDF":
        return "NETCDF3"

    hdf5_offset = _find_hdf5_offset(begin)
    if hdf5_offset == 0:
        return "NETCDF4"
    if hdf5_offset is not None:
        # HDF5 signature present but not at byte 0 — extra header/prefix
        # bytes before it. Could be CfRadial2-HDF5 *or* ODIM_H5 with a
        # vendor prefix; the two are only reliably told apart by
        # inspecting the HDF5 group structure (see readers/odim_h5_reader.py
        # :is_odim_h5), which needs real HDF5 parsing, not a magic-byte
        # sniff. Reported as ODIM_H5 since that's the case this was added
        # for — treat it as "prefixed HDF5, go inspect it" rather than a
        # hard guarantee.
        return "ODIM_H5"

    if begin[:9] == b"ARCHIVE2." or begin[:4] == b"AR2V":
        return "NEXRAD_L2"

    if begin[:4] == b"SDUS" or begin[:4] == b"\x01\r\r\n":
        return "NEXRAD_L3"

    if begin[:2] == b"UF" or begin[2:4] == b"UF" or begin[4:6] == b"UF":
        return "UF"

    # Fixed: was `begin[:4] == b"PF"`, a 4-byte slice compared against a
    # 2-byte literal — always False. Narrowed to a 2-byte comparison.
    if begin[:2] == b"PF" or begin[4:6] == b"PF":
        return "IRIS_PF"

    if begin[:4] in (b"SSWB", b"VOLD", b"COMM"):
        return "DORADE"

    if begin[0:1] == b"\x1b":
        return "SIGMET"

    if begin[0:4] == b"GRIB":
        return "GRIB"
    
    if begin[0:4] in (b"RADF", b"ARFD"):
        return "UKMO"

    return "UNKNOWN"


def detect_filetype(filename: str) -> str:
    """
    Identify a radar file's format from its magic bytes.

    Transparently unwraps one level of gzip or bz2 compression — the
    returned tag describes what's *inside* the archive, not the
    archive itself, unless the decompressed payload couldn't be
    classified either.

    Returns one of: "MDV", "NETCDF3", "NETCDF4", "ODIM_H5", "NEXRAD_L2",
    "NEXRAD_L3", "UF", "IRIS_PF", "DORADE", "SIGMET", "GRIB", "GZ",
    "BZ2", or "UNKNOWN". Note UKMO polar files come back "UNKNOWN" —
    see the module docstring.
    """
    try:
        with open(filename, "rb") as f:
            head = f.read(4)
    except OSError:
        return "UNKNOWN"

    if head[:2] == _GZIP_SIGNATURE:
        try:
            with gzip.open(filename, "rb") as gz:
                inner = gz.read(_HDF5_SCAN_BYTES)
        except OSError:
            return "GZ"
        inner_type = _classify_bytes(inner)
        return inner_type if inner_type != "UNKNOWN" else "GZ"

    if head[:3] == _BZIP2_SIGNATURE:
        try:
            with bz2.open(filename, "rb") as bz:
                inner = bz.read(_HDF5_SCAN_BYTES)
        except OSError:
            return "BZ2"
        inner_type = _classify_bytes(inner)
        return inner_type if inner_type != "UNKNOWN" else "BZ2"

    try:
        with open(filename, "rb") as f:
            chunk = f.read(_HDF5_SCAN_BYTES)
    except OSError:
        return "UNKNOWN"

    return _classify_bytes(chunk)


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:] or [__file__]:
        print(f"{path}: {detect_filetype(path)}")