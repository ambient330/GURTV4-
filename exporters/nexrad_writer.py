# Modified from NCAR LROSE code; this is NOT the version available from UCAR.
# Copyright (c) 1990 - 2023, University Corporation for Atmospheric Research
# (UCAR), Boulder, Colorado, USA. All rights reserved. Used under the LROSE BSD
# license - see THIRD_PARTY_NOTICES.md for the full text and disclaimer.
# Modifications for GURT: SPDX-License-Identifier: GPL-2.0-only (BSD is GPL-compatible).
import bz2
import gzip
import io
import struct
import math
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class Ray:
    azimuth: float
    elevation: float
    time_unix: float
    start_range_m: int
    gate_width_m: int
    fields: Dict[str, np.ndarray]
    nyquist_mps: float = 29.1
    unamb_range_km: float = 230.0
    noise_h_dbm: float = -74.0
    noise_v_dbm: float = -74.0
    field_geometry: Dict[str, tuple] = None
    # Optional per-field boolean masks marking gates that were flagged
    # range-folded (CINRAD/NEXRAD digital code 1) in the source data.
    # These gates get forced to the NEXRAD RF sentinel byte (1) after
    # normal scale/offset encoding, instead of being lost as plain
    # "missing" (0) once decoded to physical units.
    range_folded: Dict[str, np.ndarray] = None


@dataclass
class Sweep:
    fixed_elevation: float
    rays: List[Ray]
    scan_rate_deg_s: float = 18.0


VOL_TITLE_FMT  = ">9s3shhI4s"
VOL_TITLE_SIZE = struct.calcsize(VOL_TITLE_FMT)

CTM_INFO_SIZE  = 12

MSG_HDR_FMT    = ">HBBHHIHH"
MSG_HDR_SIZE   = struct.calcsize(MSG_HDR_FMT)

PACKET_SIZE    = 2432
MAX_PAYLOAD    = PACKET_SIZE - CTM_INFO_SIZE - MSG_HDR_SIZE

VCP_HDR_FMT    = ">HHHHHBBHHHHH"
VCP_HDR_SIZE   = struct.calcsize(VCP_HDR_FMT)

PPI_HDR_FMT    = ">H BBBBHhhhh HHH HHHHHHHHHHHh"
PPI_HDR_SIZE   = struct.calcsize(PPI_HDR_FMT)

MSG31_HDR_FMT  = ">4sIHHfBBHBBBBfBBH9I"
MSG31_HDR_SIZE = struct.calcsize(MSG31_HDR_FMT)

MSG31_VOL_FMT  = ">4sHBBffhHfffffHH"
MSG31_VOL_SIZE = struct.calcsize(MSG31_VOL_FMT)

MSG31_ELV_FMT  = ">4sHHf"
MSG31_ELV_SIZE = struct.calcsize(MSG31_ELV_FMT)

MSG31_RAD_FMT  = ">4sHHffHH"
MSG31_RAD_SIZE = struct.calcsize(MSG31_RAD_FMT)

MSG31_FLD_FMT  = ">4sfHhHHhBBff"
MSG31_FLD_SIZE = struct.calcsize(MSG31_FLD_FMT)

ADAPTATION_DATA_SIZE    = 8808
RDA_ADAPTATION_DATA     = 18
VOLUME_COVERAGE_PATTERN = 5
ANGLE_MULT              = 360.0 / 65536.0

FIELD_PARAMS = {
    "REF": (2.0,    66.0,   1),
    "VEL": (2.0,   129.0,   1),
    "SW":  (2.0,   129.0,   1),
    "ZDR": (16.0,  128.0,   1),
    "PHI": (2.8361,  2.0,   2),
    "RHO": (300.0,  -60.0,  1),
    "CFP": (2.0,    66.0,   1),
}


def _encode_u8(data: np.ndarray, scale: float, offset: float) -> np.ndarray:
    out = np.zeros(len(data), dtype=np.uint8)
    finite = np.isfinite(data)
    iv = np.rint(data[finite] * scale + offset).astype(np.int32)
    out[finite] = np.clip(iv, 1, 255).astype(np.uint8)
    return out


def _encode_u16(data: np.ndarray, scale: float, offset: float) -> np.ndarray:
    out = np.zeros(len(data), dtype=np.uint16)
    finite = np.isfinite(data)
    iv = np.rint(data[finite] * scale + offset).astype(np.int32)
    out[finite] = np.clip(iv, 1, 65535).astype(np.uint16)
    return out


def _encode_field(data: np.ndarray, scale: float, offset: float,
                  byte_width: int, max_vel: float = 30.0,
                  field_name: str = "") -> bytes:
    if field_name == "VEL" and max_vel >= 64.0:
        scale = 2.0
    data = np.asarray(data, dtype=np.float32)
    if byte_width == 2:
        arr = _encode_u16(data, scale, offset)
        return arr.astype(">u2").tobytes()
    else:
        return _encode_u8(data, scale, offset).tobytes()


def _time_to_nexrad(time_unix: float):
    total_secs  = int(time_unix)
    frac_secs   = time_unix - total_secs
    julian      = total_secs // 86400
    secs_in_day = total_secs - julian * 86400
    ms          = int((secs_in_day + frac_secs) * 1e3 + 0.5)
    return julian + 1, ms



class NexradMsg31Writer:

    def __init__(self,
                 site_id: str        = "KXXX",
                 latitude: float     = 0.0,
                 longitude: float    = 0.0,
                 height_m: int       = 0,
                 feedhorn_height_m: int = 15,
                 vcp_number: int     = 11,
                 dbz0: float         = 0.0,
                 volume_number: int  = 1,
                 frequency_hz: float = 0.0,
                 compress: bool      = False,
                 native_bz2: bool    = True,
                 bz2_chunk_bytes: int = 200_000):
        self.site_id            = site_id[:4].ljust(4)
        self.latitude           = latitude
        self.longitude          = longitude
        self.height_m           = height_m
        self.feedhorn_height_m  = feedhorn_height_m
        self.vcp_number         = vcp_number
        self.dbz0               = dbz0
        self.volume_number      = volume_number
        self._seq_num           = 1
        self.frequency_hz       = frequency_hz
        self.compress           = compress
        self.native_bz2         = native_bz2
        self.bz2_chunk_bytes    = bz2_chunk_bytes

    def write(self, path: str, sweeps: List[Sweep]) -> None:
        all_times = [r.time_unix for sw in sweeps for r in sw.rays]
        end_time  = max(all_times) if all_times else 0.0

        max_vel = 0.0
        for sw in sweeps:
            for r in sw.rays:
                vel = r.fields.get("VEL")
                if vel is not None and len(vel):
                    mv = float(np.nanmax(np.abs(vel)))
                    if mv > max_vel:
                        max_vel = mv

        if self.native_bz2:
            # Native Archive II framing: 24-byte volume header written
            # uncompressed, followed by a series of independently
            # bz2-compressed "LDM compressed records". Each record is
            # prefixed with a 4-byte big-endian signed control word giving
            # the byte length of the bz2 stream that follows it; the
            # control word of the final record is negated as an EOF flag.
            # The decompressed payloads of every record are simply
            # concatenated by readers before message parsing, so record
            # boundaries need not align with message boundaries.
            body = io.BytesIO()
            self._write_adaptation_data(body, end_time)
            self._write_vcp(body, sweeps, end_time)
            for sweep_idx, sweep in enumerate(sweeps):
                for ray_idx, ray in enumerate(sweep.rays):
                    self._write_msg31(body, sweep_idx, ray_idx,
                                      sweep, ray, sweeps, max_vel)

            raw = body.getvalue()
            with open(path, "wb") as fh:
                self._write_vol_title(fh, end_time)
                pos = 0
                n = len(raw)
                while pos < n:
                    chunk_end = min(pos + self.bz2_chunk_bytes, n)
                    is_last = chunk_end >= n
                    compressed = bz2.compress(raw[pos:chunk_end])
                    ctrl = len(compressed)
                    if is_last:
                        ctrl = -ctrl
                    fh.write(struct.pack(">i", ctrl))
                    fh.write(compressed)
                    pos = chunk_end
        elif self.compress:
            # Write all records through a gzip file handle.
            with gzip.open(path, "wb", compresslevel=6) as fh:
                self._write_vol_title(fh, end_time)
                self._write_adaptation_data(fh, end_time)
                self._write_vcp(fh, sweeps, end_time)
                for sweep_idx, sweep in enumerate(sweeps):
                    for ray_idx, ray in enumerate(sweep.rays):
                        self._write_msg31(fh, sweep_idx, ray_idx,
                                          sweep, ray, sweeps, max_vel)
        else:
            with open(path, "wb") as fh:
                self._write_vol_title(fh, end_time)
                self._write_adaptation_data(fh, end_time)
                self._write_vcp(fh, sweeps, end_time)
                for sweep_idx, sweep in enumerate(sweeps):
                    for ray_idx, ray in enumerate(sweep.rays):
                        self._write_msg31(fh, sweep_idx, ray_idx,
                                          sweep, ray, sweeps, max_vel)

    def _write_vol_title(self, fh, end_time: float) -> None:
        jul, ms = _time_to_nexrad(end_time)
        data = struct.pack(VOL_TITLE_FMT,
                           f"AR2V0006.{self.volume_number:03d}".encode()[:9],
                           b"001",
                           jul, 0,
                           ms,
                           self.site_id.encode()[:4])
        fh.write(data)

    def _write_adaptation_data(self, fh, end_time: float) -> None:
        payload = bytearray(ADAPTATION_DATA_SIZE)
        sid = self.site_id.encode()[:4]
        payload[8368:8368 + len(sid)] = sid

        if self.frequency_hz > 0:
            freq_mhz = int(round(self.frequency_hz / 1e6))
            struct.pack_into(">I", payload, 1092, freq_mhz)  # uint32 MHz

        self._write_message(fh, bytes(payload), RDA_ADAPTATION_DATA, end_time)

    def _write_vcp(self, fh, sweeps: List[Sweep], end_time: float) -> None:
        n_sweeps    = len(sweeps)
        n_halfwords = (VCP_HDR_SIZE + n_sweeps * PPI_HDR_SIZE) // 2
        max_vel     = max((r.nyquist_mps for sw in sweeps for r in sw.rays), default=29.1)
        dop_vel_res = 2 if max_vel < 64.0 else 4
        RATE_MULT   = (22.5 / 2048.0) / 8.0

        hdr = struct.pack(VCP_HDR_FMT,
                          n_halfwords, 2, self.vcp_number, n_sweeps, 1,
                          dop_vel_res, 4,
                          0, 0, 0, 0, 0)

        ppis = bytearray()
        for sweep in sweeps:
            elev_code = int(sweep.fixed_elevation / ANGLE_MULT + 0.5) & 0xFFFF
            rate_code = int(sweep.scan_rate_deg_s / RATE_MULT + 0.5)
            ppis += struct.pack(PPI_HDR_FMT,
                                elev_code,
                                0, 3, 0, 0, 0,
                                rate_code,
                                -160, -160, -160,
                                0, 0, 0,
                                0, 4, 128, 0,
                                0, 0, 0, 0,
                                0, 0, 0, 0)

        self._write_message(fh, hdr + bytes(ppis), VOLUME_COVERAGE_PATTERN, end_time)

    def _write_message(self, fh, payload: bytes, msg_type: int,
                       time_unix: float) -> None:
        jul, ms      = _time_to_nexrad(time_unix)
        n_bytes_left = len(payload)
        n_messages   = max(1, math.ceil(n_bytes_left / MAX_PAYLOAD))
        ptr          = 0

        for seg in range(n_messages):
            chunk_size = min(n_bytes_left, MAX_PAYLOAD)
            chunk      = payload[ptr:ptr + chunk_size]

            fh.write(b"\x00" * CTM_INFO_SIZE)
            fh.write(struct.pack(MSG_HDR_FMT,
                                 chunk_size // 2, 0, msg_type,
                                 self._seq_num & 0x7FFF,
                                 jul, ms,
                                 n_messages, seg + 1))
            self._seq_num += 1
            fh.write(chunk)
            if chunk_size < MAX_PAYLOAD:
                fh.write(b"\x00" * (MAX_PAYLOAD - chunk_size))

            n_bytes_left -= chunk_size
            ptr          += chunk_size

    def _write_msg31(self, fh,
                     sweep_idx: int, ray_idx: int,
                     sweep: Sweep, ray: Ray,
                     all_sweeps: List[Sweep],
                     max_vel: float) -> None:
        jul, ms  = _time_to_nexrad(ray.time_unix)
        n_sweeps = len(all_sweeps)
        n_rays   = len(sweep.rays)

        if sweep_idx == 0 and ray_idx == 0:
            radial_status = 3
        elif sweep_idx == n_sweeps - 1 and ray_idx == n_rays - 1:
            radial_status = 4
        elif ray_idx == 0:
            radial_status = 0
        elif ray_idx == n_rays - 1:
            radial_status = 2
        else:
            radial_status = 1

        az_spacing = 1 if n_rays >= 600 else 2

        encoded    = {}
        field_hdrs = {}
        for fname, (fscale, foffset, fwidth) in FIELD_PARAMS.items():
            arr = ray.fields.get(fname)
            if arr is None or len(arr) == 0:
                continue
            
            # Use per-field geometry if available, else fall back to ray-level
            geom = (ray.field_geometry or {}).get(fname)
            start_m = geom[0] if geom else ray.start_range_m
            gate_m  = geom[1] if geom else ray.gate_width_m

            encoded[fname]    = _encode_field(arr, fscale, foffset, fwidth, max_vel, fname)

            # Force range-folded gates to the NEXRAD RF sentinel (byte 1),
            # overriding whatever the normal scale/offset encoding produced.
            rf_mask = (ray.range_folded or {}).get(fname)
            if rf_mask is not None and np.any(rf_mask):
                enc = bytearray(encoded[fname])
                if fwidth == 2:
                    rf_bytes = (1).to_bytes(2, "big")
                    for gi in np.nonzero(rf_mask)[0]:
                        enc[gi * 2:gi * 2 + 2] = rf_bytes
                else:
                    for gi in np.nonzero(rf_mask)[0]:
                        enc[gi] = 1
                encoded[fname] = bytes(enc)

            field_hdrs[fname] = struct.pack(MSG31_FLD_FMT,
                                            ("D" + fname.ljust(3)).encode()[:4],
                                            0.0,
                                            len(arr),
                                            start_m,
                                            gate_m,
                                            0, 0, 0,
                                            fwidth * 8,
                                            fscale, foffset)

        n_blocks   = 3 + len(encoded)
        offset     = MSG31_HDR_SIZE
        vol_offset = offset; offset += MSG31_VOL_SIZE
        elv_offset = offset; offset += MSG31_ELV_SIZE
        rad_offset = offset; offset += MSG31_RAD_SIZE

        field_offsets = {}
        for fname in FIELD_PARAMS:
            if fname in encoded:
                field_offsets[fname] = offset
                offset += MSG31_FLD_SIZE + len(encoded[fname])

        boffsets = [0] * 9
        bi = 0
        boffsets[bi] = vol_offset; bi += 1
        boffsets[bi] = elv_offset; bi += 1
        boffsets[bi] = rad_offset; bi += 1
        for fname in FIELD_PARAMS:
            if fname in field_offsets:
                boffsets[bi] = field_offsets[fname]; bi += 1

        msg31_hdr = struct.pack(MSG31_HDR_FMT,
                                self.site_id.encode()[:4],
                                ms, jul,
                                ray_idx + 1,
                                float(ray.azimuth),
                                0, 0,
                                offset,
                                az_spacing, radial_status,
                                sweep_idx + 1, 1,
                                float(ray.elevation),
                                0, 0,
                                n_blocks,
                                *boffsets)

        vol_blk = struct.pack(MSG31_VOL_FMT,
                              b"RVOL", MSG31_VOL_SIZE, 1, 0,
                              float(self.latitude), float(self.longitude),
                              self.height_m, self.feedhorn_height_m,
                              self.dbz0, 0.0, 0.0, 0.0, 0.0,
                              self.vcp_number, 0)

        elv_blk = struct.pack(MSG31_ELV_FMT,
                              b"RELV", MSG31_ELV_SIZE, 0, self.dbz0)

        rad_blk = struct.pack(MSG31_RAD_FMT,
                              b"RRAD", MSG31_RAD_SIZE,
                              int(ray.unamb_range_km * 10 + 0.5),
                              float(ray.noise_h_dbm), float(ray.noise_v_dbm),
                              int(ray.nyquist_mps * 100 + 0.5), 0)
        

        body = msg31_hdr + vol_blk + elv_blk + rad_blk
        for fname in FIELD_PARAMS:
            if fname in encoded:
                body += field_hdrs[fname] + encoded[fname]

        if len(body) < MAX_PAYLOAD:
            body += b"\x00" * (MAX_PAYLOAD - len(body))
        elif len(body) % 2:
            body += b"\x00"

        fh.write(b"\x00" * CTM_INFO_SIZE)
        fh.write(struct.pack(MSG_HDR_FMT,
                             (len(body) + MSG_HDR_SIZE) // 2,
                             0, 31,
                             self._seq_num & 0x7FFF,
                             jul, ms,
                             1, 1))
        self._seq_num += 1
        fh.write(body)


if __name__ == "__main__":
    pass