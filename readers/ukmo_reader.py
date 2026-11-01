#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""
readers/ukmo_reader.py — Pure-Python UKMO polar-file reader for GURT.

Lives in the ``readers/`` package alongside ``cfradial_reader.py``,
``odim_h5_reader.py``, ``dorade_reader.py`` and ``grib2_reader.py``.
Re-exported from ``readers/__init__.py`` so ``main.py`` can do:

    from readers import read_ukmo, is_ukmo

``read_ukmo(path)`` returns a ``RadarVolume`` populated with a single
``SweepData`` — the same types every other GURT reader produces. UKMO
polar files are single-scan-per-file, so there's exactly one sweep per
volume. They're normally shipped gzip-compressed, but ``_open_ukmo()``
sniffs the gzip magic number and transparently falls back to a plain
``open()`` for already-decompressed files, so both ``.dat.gz`` and
``.dat`` work.

The binary parsing (``UKMOPolarFile`` and the header/bit-layout tables)
is unchanged from the reference reader; only ``is_ukmo`` (format
detection) and ``read_ukmo`` (GURT data-model adapter) are new.
"""

from __future__ import annotations

import gzip
import struct
from datetime import datetime, timedelta
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Transparent gzip / raw file opening
# ──────────────────────────────────────────────────────────────────────────

_GZIP_MAGIC = b"\x1f\x8b"


def _open_ukmo(path: str):
    """
    Open a UKMO file for binary reading, transparently decompressing it
    if it's gzipped and passing it through unchanged otherwise.

    UKMO polar files are normally shipped as ``.dat.gz``, but the same
    binary layout also turns up already-decompressed (``.dat``, or a
    ``.gz`` that's been unzipped by hand/by another pipeline stage), so
    rather than assume compression from the extension we sniff the first
    two bytes for gzip's magic number (``1f 8b``) and pick the right
    opener. Cheap (two bytes) and avoids gzip.BadGzipFile blowing up on
    plain files.
    """
    with open(path, 'rb') as fh:
        head = fh.read(2)
    if head == _GZIP_MAGIC:
        return gzip.open(path, 'rb')
    return open(path, 'rb')


# ──────────────────────────────────────────────────────────────────────────
# Header layout tables
# ──────────────────────────────────────────────────────────────────────────

VOLUME_HEADER = {
    'identification': [
        (0,  4, 1, 'magic number'),
        (4,  4, 1, 'bit-byte ordering'),
        (8,  2, 2, 'version number'),
        (10, 2, 2, 'mode of operation'),
        (12, 12, 2, 'creation time and date'),
        (24, 12, 2, 'volume start time and date'),
        (36, 12, 2, 'volume stop time and date'),
    ],
    'site information': [
        (48, 2, 2, 'WMO country code'),
        (50, 2, 2, 'WMO site number'),
        (52, 6, 2, 'radar site longitude'),
        (58, 6, 2, 'radar site latitude'),
        (64, 2, 2, 'local site number'),
        (66, 2, 2, 'local grid easting'),
        (68, 2, 2, 'local grid northing'),
        (70, 2, 2, 'sensor height'),
    ],
    'hardware configuration': [
        (72, 2, 2, 'hardware type'),
        (74, 2, 2, 'software type'),
        (76, 2, 2, 'number of channels'),
        (80, 2, 2, 'channel 1 polarisation mode'),
        (82, 2, 2, 'channel 2 polarisation mode'),
        (84, 2, 2, 'channel 1 radar constant'),
        (86, 2, 2, 'channel 2 radar constant'),
        (88, 2, 2, 'channel 1 wavelength'),
        (90, 2, 2, 'channel 2 wavelength'),
        (92, 2, 2, 'channel 1 beam width'),
        (94, 2, 2, 'channel 2 beam width'),
    ],
    'processor configuration': [
        (108, 2, 2, 'type of scan'),
        (110, 2, 2, 'number of scans in volume'),
        (112, 2, 2, 'number of rays per scan'),
        (114, 2, 2, 'number of bins per ray'),
        (116, 2, 2, 'processed range bin length'),
        (118, 2, 2, 'pulse length'),
        (120, 2, 2, 'antenna rotation rate'),
        (122, 2, 2, 'number of samples'),
        (124, 2, 2, 'primary PRF'),
        (126, 2, 2, 'secondary PRF'),
        (128, 2, 2, 'unambiguous range'),
        (130, 2, 2, 'unambiguous velocity'),
    ],
    'data processing': [
        (144, 2, 2, 'processor flags'),
        (146, 2, 2, 'clutter filter type'),
        (148, 2, 2, 'clutter filter response'),
        (150, 2, 2, 'clutter power threshold'),
        (152, 2, 2, 'noise threshold'),
        (154, 2, 2, 'SQI threshold'),
    ],
    'data storage': [
        (168, 2, 2, 'derived data type'),
        (170, 2, 2, 'source data type'),
        (172, 2, 2, 'number of bytes per element'),
        (174, 2, 2, 'number of values per element'),
        (176, 2, 2, 'compression'),
    ],
    'diagnostics section': [
        (192, 2, 2, 'diagnostic flags'),
    ],
}

SCAN_HEADER = [
    (0,  2, 2, 'scan index'),
    (2,  2, 2, 'number of rays in scan'),
    (4,  2, 2, 'number of bins per ray'),
    (6,  2, 2, 'range to first bin'),
    (8,  2, 2, 'scan start seconds'),
    (10, 2, 2, 'scan stop seconds'),
    (12, 2, 2, 'scan start azimuth'),
    (14, 2, 2, 'scan stop azimuth'),
    (16, 2, 2, 'scan requested elevation'),
    (18, 2, 2, 'scan average elevation'),
    (20, 2, 2, 'beam number'),
]

# Scaling factors for scan header angular fields (spec: tenths of a degree)
SCAN_HEADER_SCALE = {
    'scan start azimuth':       0.1,
    'scan stop azimuth':        0.1,
    'scan requested elevation': 0.1,
    'scan average elevation':   0.1,
}

RAY_HEADER = [
    (0, 2, 2, 'ray index'),
    (2, 2, 2, 'azimuth centre angle'),
    (4, 2, 2, 'elevation angle'),
    (6, 2, 2, 'number of bytes of data'),
    (8, 2, 2, 'number of rays per degree'),
]
RAY_HEADER_BYTES = 10

# Scaling factors for ray header angular fields (spec: hundredths of a degree)
RAY_HEADER_SCALE = {
    'azimuth centre angle': 0.01,
    'elevation angle':      0.1,
}

# ---------------------------------------------------------------------------
# Per the spec (Table 6b), data type 2213 is undocumented but observed in
# files with the 'augzdr' designation. Based on the old reader + spec context:
#
#   bits  0-11: REF  (1114: offset=-32 dBZ,  gain=0.1 dBZ)
#   bits 12-15: CI   (1191: lookup table)
#   bits 16-27: VEL  (1126: offset=-π,       gain=π/2^11)
#   bits 28-31: SQI  (1123: offset=0,        gain=1/15)
#   bits 32-39: SW   (1122: offset=0,        gain=π/256)
#   bits 40-47: ZDR  (1511: offset=-8 dB,    gain=0.0625 dB)  ← 8-bit per spec
#   bits 48-55: RHO  (1514: offset=-0.2,     gain=0.005)      ← 8-bit per spec
#   bits 56-63: PHI  (1513: offset=-π,       gain=π/256)      ← 8-bit per spec
#   Total: 64 bits = 8 bytes/bin  ✓  matches 'number of bytes per element'
# ---------------------------------------------------------------------------
p_bits = {
    2111: {'REF': [0,  12], 'CI':  [12, 16]},
    2122: {'REF': [0,  12], 'CI':  [12, 16], 'VEL': [16, 28], 'SQI': [28, 32]},
    2179: {'REF': [0,  10], 'CPA': [10, 16]},
    2213: {
        'REF': [0,  12],
        'CI':  [12, 16],
        'VEL': [16, 28],
        'SQI': [28, 32],
        'SW':  [32, 40],
        'ZDR': [40, 48],
        'RHO': [48, 64],
        'PHI': [64, 76],
    },
}

# Physical offset per product — directly from Table 6a of the spec
p_offset = {
    'REF': -32.0,
    'CI':   0.0,
    'VEL':  np.pi,    # will be negated by negative gain
    'SQI':  0.0,
    'SW':   0.0,
    'ZDR': -8.0,      # dB
    'RHO': -20/100,
    'PHI': -180,
    'CPA':  0.0,
}

# Physical gain per count — directly from Table 6a of the spec
p_gain = {
    'REF':  0.1,
    'CI':   1.0,
    'VEL': -np.pi / 2**11,   # negative so offset+gain*0 = +π → -Vn after Vn scaling
    'SQI':  1.0 / 15.0,
    'SW':   np.pi / 256.0,
    'ZDR':  0.0625,
    'RHO':  0.002/100,
    'PHI':  180 / 2**11,
    'CPA':  1.0 / 64.0,
}

p_lookup_table = {
    'CI': np.array([0, 1.5, 2, 2.5, 2.75, 3, 3.25, 3.5,
                    3.75, 4, 4.25, 4.5, 5, 5.5, 6, 7]),
}

def _bad_value(n_bits):
    return (1 << n_bits) - 1


fmts          = {1: 'b', 2: 'h', 4: 'l'}   # signed
fmts_unsigned = {1: 'B', 2: 'H', 4: 'L'}   # unsigned

def unpack_struct(data, endianness, length, unsigned=False):
    fmt = endianness + (fmts_unsigned if unsigned else fmts)[length]
    if len(data) == length:
        return struct.unpack(fmt, data)[0]
    return [struct.unpack(fmt, data[i:i + length])[0]
            for i in range(0, len(data), length)]


def bits_to_n(bits, endianness, dtype):
    """
    Convert packed bit arrays to unsigned integers.
    Respects endianness: big-endian = MSB first, little-endian = LSB first.
    bits shape: (..., n_bits)
    """
    n = bits.shape[-1]
    if endianness == '>':
        weights = np.array([2**j for j in reversed(range(n))], dtype=dtype)
    else:
        weights = np.array([2**j for j in range(n)], dtype=dtype)
    return np.sum(bits * weights, axis=-1)


class UKMOPolarFile:
    def __init__(self):
        self.filepath = None
        self.data_bits = None

    def __call__(self, filepath, products='all'):
        need_data = products == 'all' or bool(products)

        if filepath != self.filepath or (need_data and self.data_bits is None):
            self.read_file(filepath, obtain_products=need_data)
            self.filepath = filepath
            self.data = {}

        dtype_id = self.volume_header['derived data type']
        available = p_bits.get(dtype_id, {})

        if products == 'all':
            wanted = list(available.keys())
        else:
            wanted = [p for p in products if p in available]

        for p in wanted:
            if p not in self.data:
                self.data[p] = self.read_product(p)

        if need_data:
            return self.data, self.volume_header, self.scan_header, self.ray_headers
        return self.volume_header, self.scan_header

    def read_file(self, filepath, obtain_products=True):
        with _open_ukmo(filepath) as f:
            raw_vh = f.read(256)
            self.volume_header = {}
            for section, items in VOLUME_HEADER.items():
                for start, full_length, length, name in items:
                    chunk = raw_vh[start:start + full_length]
                    if name == 'magic number':
                        self.endianness = '<' if chunk[0] == 0x41 else '>'
                        self.volume_header[name] = chunk
                    else:
                        self.volume_header[name] = unpack_struct(
                            chunk, self.endianness, length)

            raw_sh = f.read(64)
            self.scan_header = {}
            for start, full_length, length, name in SCAN_HEADER:
                chunk = raw_sh[start:start + full_length]
                self.scan_header[name] = unpack_struct(
                    chunk, self.endianness, length)

            for field, scale in SCAN_HEADER_SCALE.items():
                if field in self.scan_header:
                    self.scan_header[field] = self.scan_header[field] * scale

            self.data_bits = None
            if not obtain_products:
                return

            scan_content = f.read()
            nrays         = self.scan_header['number of rays in scan']
            ngates        = self.scan_header['number of bins per ray']
            bytes_per_bin = self.volume_header['number of bytes per element']
            ray_bytes     = RAY_HEADER_BYTES + ngates * bytes_per_bin

            # Azimuth and elevation are unsigned 16-bit per spec (range 0-36000
            # and 0-9000 hundredths of a degree).  Reading them as signed int16
            # causes wrap-around for azimuths > 327.67deg (raw > 32767).
            _unsigned_ray_fields = {'azimuth centre angle', 'elevation angle'}

            self.ray_headers = {name: [] for _, _, _, name in RAY_HEADER}
            for i in range(nrays):
                raw_rh = scan_content[i * ray_bytes: i * ray_bytes + RAY_HEADER_BYTES]
                for start, full_length, length, name in RAY_HEADER:
                    chunk = raw_rh[start:start + full_length]
                    self.ray_headers[name].append(
                        unpack_struct(chunk, self.endianness, length,
                                      unsigned=(name in _unsigned_ray_fields)))

            for field, scale in RAY_HEADER_SCALE.items():
                if field in self.ray_headers:
                    self.ray_headers[field] = [v * scale for v in self.ray_headers[field]]

            self.ray_headers['azimuth centre angle'] = [
                az % 360.0 for az in self.ray_headers['azimuth centre angle']
            ]

            data_uint8 = (
                np.frombuffer(scan_content, dtype=np.uint8)
                .reshape(nrays, ray_bytes)
                [:, RAY_HEADER_BYTES:]
                .reshape(nrays, ngates, bytes_per_bin)
            )
            bitorder = 'big' if self.endianness == '>' else 'little'
            self.data_bits = np.unpackbits(data_uint8, axis=-1, bitorder=bitorder)

    def read_product(self, p):
        dtype_id   = self.volume_header['derived data type']
        bit_range  = p_bits[dtype_id][p]
        offset     = p_offset[p]
        gain       = p_gain[p]
        n_bits     = bit_range[1] - bit_range[0]

        raw_bits = self.data_bits[..., bit_range[0]:bit_range[1]]
        raw_int  = bits_to_n(raw_bits, self.endianness, 'uint32')

        bad_val  = _bad_value(n_bits)
        bad_mask = (raw_int == bad_val)

        data = (offset + gain * raw_int.astype(np.float32))

        if p in p_lookup_table:
            idx  = np.clip(raw_int, 0, len(p_lookup_table[p]) - 1)
            data = p_lookup_table[p][idx].astype(np.float32)

        if p in ('VEL', 'SW'):
            vn   = self.volume_header['unambiguous velocity'] / 100.0  # cm/s → m/s
            data = data * (vn / np.pi)

        return np.ma.array(data, mask=bad_mask)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _decode_ukmo_datetime(raw_values) -> datetime:
    """
    Decode a UKMO 12-byte time field (six int16s: year, month, day, hour,
    min, sec). raw_values is a list of 6 integers.
    """
    if isinstance(raw_values, (int, float)) or raw_values is None:
        return datetime(1970, 1, 1)
    try:
        year, month, day, hour, minute, second = raw_values[:6]
        return datetime(year, month, day, hour, minute, second)
    except (ValueError, TypeError):
        return datetime(1970, 1, 1)


def _decode_angle_3x2(parts) -> float:
    """Degrees/minutes/seconds triple (as stored in the volume header) → decimal degrees."""
    if isinstance(parts, (int, float)):
        return float(parts)
    try:
        d, m, s = parts
        return d + m / 60.0 + s / 3600.0
    except (ValueError, TypeError):
        return 0.0


def _radar_name_from_header(vol_hdr) -> str:
    """
    UKMO volume headers carry only numeric site identifiers (no plain-text
    name), so build something identifiable out of the local site number:
    "UK03", "UK10", "UK112", etc. (zero-padded to at least 2 digits — a
    3+ digit local site number is used as-is rather than truncated).
    Falls back to the WMO country/site codes, then a bare "UKMO", for the
    rare file that has a local site number of 0/None but does carry WMO
    identifiers.
    """
    local = vol_hdr.get('local site number')
    if local is not None:
        return f"UK{int(local):02d}"
    country = vol_hdr.get('WMO country code')
    site    = vol_hdr.get('WMO site number')
    if country is not None and site is not None and (country or site):
        return f"WMO {int(country):03d}{int(site):03d}"
    return "UKMO"


# ──────────────────────────────────────────────────────────────────────────
# Format detection
# ──────────────────────────────────────────────────────────────────────────

def is_ukmo(path: str) -> bool:
    """
    Heuristic UKMO polar-file detection.

    These files are usually gzip-compressed but sometimes show up already
    decompressed, and have no publicly documented magic number either way,
    so rather than trust a signature we open the volume/scan headers (via
    ``_open_ukmo``, which handles both cases) at their known byte offsets
    and sanity-check that: the
    'derived data type' is one of the bit layouts we know how to decode,
    and the ray/gate counts are physically reasonable. Cheap enough to
    run against every candidate file, and much more reliable than
    guessing from the extension alone.
    """
    try:
        with _open_ukmo(path) as f:
            raw_vh = f.read(256)
            if len(raw_vh) < 256:
                return False
            endianness = '<' if raw_vh[0] == 0x41 else '>'
            dtype_id = unpack_struct(raw_vh[168:170], endianness, 2)
            if dtype_id not in p_bits:
                return False

            raw_sh = f.read(64)
            if len(raw_sh) < 64:
                return False
            nrays  = unpack_struct(raw_sh[2:4], endianness, 2, unsigned=True)
            ngates = unpack_struct(raw_sh[4:6], endianness, 2, unsigned=True)
            return 0 < nrays <= 20000 and 0 < ngates <= 20000
    except OSError:
        return False
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────

def read_ukmo(path: str, RadarVolume=None, SweepData=None):
    """
    Read a UKMO polar file and return a ``RadarVolume``.

    ``RadarVolume`` and ``SweepData`` are imported from ``main`` by
    default; pass them explicitly to use this module standalone. UKMO
    files hold a single scan each, so the returned volume always has
    exactly one sweep.
    """
    if RadarVolume is None or SweepData is None:
        from main import RadarVolume, SweepData  # type: ignore[assignment]

    reader = UKMOPolarFile()
    data, vol_hdr, scan_hdr, ray_hdrs = reader(path, products='all')

    vol = RadarVolume()
    vol.radar_name = _radar_name_from_header(vol_hdr)
    vol.latitude    = _decode_angle_3x2(vol_hdr.get('radar site latitude', 0.0))
    vol.longitude   = _decode_angle_3x2(vol_hdr.get('radar site longitude', 0.0))
    vol.altitude    = float(vol_hdr.get('sensor height', 0.0) or 0.0)
    vol.scan_time   = _decode_ukmo_datetime(vol_hdr.get('volume start time and date'))

    nrays  = scan_hdr['number of rays in scan']
    ngates = scan_hdr['number of bins per ray']
    bin_len        = float(vol_hdr['processed range bin length'])   # metres
    range_to_first = float(scan_hdr['range to first bin'])          # metres

    sw = SweepData()
    sw.radar_name  = vol.radar_name
    sw.fixed_angle = float(scan_hdr.get('scan average elevation',
                                        scan_hdr.get('scan requested elevation', 0.0)))
    sw.scan_type   = "ppi"   # UKMO files carry a 'type of scan' code, but its
                              # meaning isn't documented here; every field
                              # this reader decodes is designed for PPI.
    sw.ranges      = (range_to_first
                       + np.arange(ngates, dtype="float64") * bin_len
                       ).astype("float32")
    sw.azimuth     = np.asarray(ray_hdrs['azimuth centre angle'], dtype="float32")
    sw.elevation   = np.asarray(
        ray_hdrs.get('elevation angle', [sw.fixed_angle] * nrays), dtype="float32"
    )

    scan_start_sec = scan_hdr.get('scan start seconds', 0)
    try:
        sw.scan_time = vol.scan_time + timedelta(seconds=float(scan_start_sec))
    except (TypeError, OverflowError, OSError):
        sw.scan_time = vol.scan_time

    # Nyquist velocity / PRT, when present. Same 'assumed units' caveats as
    # the original reference reader: PRF is assumed Hz, unambiguous
    # velocity is stored in cm/s per the spec.
    vol.instrument_parameters = {}
    prf = vol_hdr.get('primary PRF', 0)
    vn  = float(vol_hdr.get('unambiguous velocity', 0) or 0) / 100.0  # cm/s -> m/s
    if vn:
        ni = {"data": np.array([vn], dtype="float32")}
        vol.instrument_parameters["nyquist_velocity"] = ni
        sw.instrument_parameters["nyquist_velocity"] = ni
    if prf:
        vol.instrument_parameters["prt"] = {
            "data": np.array([1.0 / prf], dtype="float64")
        }

    field_names = []
    for name, field_data in data.items():
        arr = np.ma.filled(field_data, np.nan).astype("float32")
        # Bake NaN into the underlying array (not just the mask flag) so it
        # survives GURT's PPI gridding step, matching every other reader
        # in this codebase — see the ODIM_H5 reader for the full story on
        # why a mask-only flag silently gets dropped there.
        sw.fields[name] = np.ma.masked_invalid(arr)
        field_names.append(name)

    vol.field_names = field_names
    vol.sweeps.append(sw)
    return vol