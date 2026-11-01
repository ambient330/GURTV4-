# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
import struct
import re
import warnings
import numpy as np
import datetime

class DoradeFile:
    def __init__(self, filename, endian="<"):
        if hasattr(filename, "read"):
            self._fh = filename
        else:
            self._fh = open(filename, "rb")

        self.buf = self._fh.read()
        #self.endian = endian
        self.endian = self._detect_endian()

        self.blocks = []
        self.params = {}
        self.rdat = []
        self.rays = []
        

        self._parse()

    def _detect_endian(self):
        buf = self.buf
        block_id = buf[0:4]
        nbytes_le = struct.unpack("<i", buf[4:8])[0]
        nbytes_be = struct.unpack(">i", buf[4:8])[0]
        file_len = len(buf)
        le_ok = 0 < nbytes_le < file_len
        be_ok = 0 < nbytes_be < file_len
        if le_ok and not be_ok:
            return "<"
        elif be_ok and not le_ok:
            return ">"
        elif be_ok and le_ok:
            if block_id in BLOCK_MAP:
                return ">"
            else:
                return "<"
        else:
            raise RuntimeError("Cannot determine endian (both invalid)")


    def _parse(self):
        pos = 0
        buf = self.buf
        while pos < len(buf):
            block_id = buf[pos:pos+4]
            if block_id == b"PARM":
                block = read_parm_block(buf, pos, self.endian)
                name = block["parameter_name"].decode(errors="ignore").strip("\x00").strip()
                scale = block["parameter_scale"]
                offset = block["parameter_bias"]
                bad_data = block["bad_data"]
                
                #print(bad_data)
                self.params[name] = block

            if block_id == b"RYIB":
                block = _unpack_from_buf(buf, pos, RYIB_BLOCK, self.endian)
                current_ray = {
                    "azimuth": block["azimuth"],
                    "elevation": block["elevation"],
                    "status": block["ray_status"],
                    "rdat": {}
                }
                self.rays.append(current_ray)

            elif block_id == b"RADD":
                block = _unpack_from_buf(buf, pos, RADD_BLOCK, self.endian)
                self.instrument_name = block["radar_name"].decode(errors="ignore").strip("\x00").strip()
                self.compression = block["data_compress"]
                #print(self.compression)
                self.lat = block["radar_latitude"]
                self.lon = block["radar_longitude"]
                self.alt = block["radar_altitude"] * 1000 # this is in km
                self.frequency = block["freq1"] # freq1 is always the usual frequency block with the true value, when working with DORADE data
                self.sweep_mode = _sweep_mode(block["scan_mode"])
                self.nyquist_velocity = block["eff_unamb_vel"]
                self.unambiguous_range = block["eff_unamb_range"]
                self._radd_num_freq = block["num_freq_trans"]
                self._radd_num_ipps = block["num_ipps_trans"]
                self._radd_prt1 = block["prt1"]
                self._radd_prt2 = block["prt2"]
                self._radd_prt3 = block["prt3"]
                self._radd_prt4 = block["prt4"]
                self._radd_prt5 = block["prt5"]
                #print(alt)
                #site_name = block[""]
                #print(lat, lon, alt)

            elif block_id == b"RDAT":
                block = _unpack_from_buf(buf, pos, RDAT_BLOCK, self.endian)
                min_offset = struct.calcsize(self.endian + "4si8s")  # 16
                product_name = block["pdata_name"].decode(errors="ignore").strip("\x00").strip()
                data_length = block["nbytes"]

                # Prefer the PARM block's offset_to_data when available
                # (newer 216-byte PARM blocks carry it); older 104-byte
                # PARM blocks don't have the field, and some files leave
                # it at 0, so fall back to the fixed RDAT header size in
                # that case. This mirrors lrose-core's DoradeRadxFile::
                # _handleField16, which does the same min-clamp.
                offset_to_data = min_offset
                parm = self.params.get(product_name)
                if parm is not None and "offset_to_data" in parm:
                    parm_offset = int(parm["offset_to_data"])
                    if parm_offset >= min_offset:
                        offset_to_data = parm_offset

                raw = buf[pos + offset_to_data : pos + data_length]
                data = np.frombuffer(raw, dtype=self.endian + "i2")

                if self.compression == 1:  # COMPRESSION_HRD
                    if hasattr(self, "dist_cells") and self.dist_cells is not None:
                        ngates = len(self.dist_cells)
                    elif product_name in self.params and "number_cells" in self.params[product_name]:
                        ngates = self.params[product_name]["number_cells"]
                    else:
                        raise RuntimeError(f"Cannot determine ngates for compressed param '{product_name}'")
                    data = self._rle_decode_row(data, ngates)

                data = data.astype("float32")
                if current_ray is not None:
                    current_ray["rdat"][product_name] = data

                
            elif block_id == b"SSWB":
                block = _unpack_from_buf(buf, pos, SSWB_BLOCK, self.endian)
                volume_time = datetime.datetime.fromtimestamp(block["volume_time_stamp"], tz=datetime.timezone.utc).replace(tzinfo=None)
                #radar_name = block["radar_name"]
                self.sweep_start_time = datetime.datetime.fromtimestamp(block["start_time"], tz=datetime.timezone.utc).replace(tzinfo=None)
                self.sweep_stop_time = datetime.datetime.fromtimestamp(block["stop_time"], tz=datetime.timezone.utc).replace(tzinfo=None)

            elif block_id == b"VOLD":
                block = _unpack_from_buf(buf, pos, VOLD_BLOCK, self.endian)
                self.volume_number = block["volume_num"]

                def get_volume_datetime():
                    year = block["year"]
                    month = block["month"]
                    day = block["day"]
                    hour = block["data_set_hour"]
                    minute = block["data_set_minute"]
                    second = block["data_set_second"]

                    return datetime.datetime(year, month, day, hour, minute, second)
                
            elif block_id == b"SWIB":
                block = _unpack_from_buf(buf, pos, SWIB_BLOCK, self.endian)
                self.sweep_number = block["sweep_num"]
                start_angle = block["start_angle"]
                end_angle = block["stop_angle"]
                self.fixed_angle = block["fixed_angle"]
                self.num_rays = block["num_rays"]

            elif block_id == b"RYIB":
                block = _unpack_from_buf(buf, pos, RYIB_BLOCK, self.endian)
                azimuth = block["azimuth"]
                elevation = block["elevation"]
                ray_status = _ray_status(block["ray_status"])
                #print(ray_status)

                self.rays.append({"azimuth": block["azimuth"], "elevation": block["elevation"], "status": block["ray_status"]})

            elif block_id == b"CELV":
                block = _unpack_from_buf(buf, pos, CELV_BLOCK, self.endian)
                ncells = block["number_cells"]
                raw_dist = block["dist_cells"][:ncells * 4]
                self.dist_cells = np.frombuffer(raw_dist, dtype=self.endian + "f4").copy()

            elif block_id == b"CSFD":
                block = _unpack_from_buf(buf, pos, CSFD_BLOCK, self.endian)
                num_segments = block["num_segments"]
                dist_to_first = block["dist_to_first"]
                spacing = np.frombuffer(
                    block["spacing"], dtype=self.endian + "f4"
                )[:num_segments]
                num_cells = np.frombuffer(
                    block["num_cells"], dtype=self.endian + "i2"
                )[:num_segments]

                segments = []
                r = float(dist_to_first)
                for seg_spacing, seg_ncells in zip(spacing, num_cells):
                    seg_spacing = float(seg_spacing)
                    seg_ncells = int(seg_ncells)
                    if seg_ncells <= 0:
                        continue
                    seg_ranges = r + np.arange(seg_ncells, dtype="float32") * seg_spacing
                    segments.append(seg_ranges)
                    r += seg_ncells * seg_spacing

                if segments and getattr(self, "dist_cells", None) is None:
                    self.dist_cells = np.concatenate(segments)

            elif block_id in BLOCK_MAP:
                block_def = BLOCK_MAP[block_id]
                block = _unpack_from_buf(buf, pos, block_def, self.endian)
                self.blocks.append((block_id, block))

            nbytes = block["nbytes"]
            pos += nbytes

    def get_param(self, name):
        return self.params.get(name)
    
    def _rle_decode_row(self, data_comp, ngates):
        """Decompress one ray's MIT/HRD run-length-encoded 16-bit field
        data. Ported from lrose-core's DoradeData::decompressHrd16
        (DoradeData.cc):

          - a count word's low 15 bits (MASK15 = 0x7fff) give a run
            length
          - if the high bit (SIGN16 = 0x8000) is set, the run is
            literal data: that many words immediately follow and are
            copied verbatim
          - if the high bit is clear, the run is a run of bad/missing
            gates: no data words follow, just fill with the bad value
          - a count word of 0 or 1 ends the row
          - two bad-data runs in a row (with no literal run between
            them) is used as an alternate termination flag

        `data_comp` carries this file's byte order (self.endian) in its
        dtype. We need that converted to native order before treating
        it as plain integers; `.astype()` does that conversion
        correctly, unlike `.view()`, which only reinterprets the
        existing bytes and silently produces garbage when the source
        byte order doesn't match the host's.
        """
        comp = data_comp.astype(np.uint16)
        n_words = len(comp)
        out = np.full(ngates, -32768, dtype=np.int16)

        i = j = 0
        prev_was_bad_run = False
        while i < n_words and j < ngates:
            val = int(comp[i])
            if val == 0 or val == 1:
                break  # end-of-row sentinel

            n = val & 0x7FFF
            is_literal = bool(val & 0x8000)

            if is_literal:
                prev_was_bad_run = False
                i += 1
                available = n_words - i
                n = min(n, available, ngates - j)
                if n <= 0:
                    break
                out[j:j+n] = comp[i:i+n].view(np.int16)
                i += n
            else:
                if prev_was_bad_run:
                    # Two consecutive bad-data runs with no data run in
                    # between -- per lrose this signals an alternate
                    # termination condition, not real data.
                    break
                prev_was_bad_run = True
                n = min(n, ngates - j)
                out[j:j+n] = -32768
                i += 1

            j += n

        if j == 0:
            warnings.warn(
                f"RLE decode produced no gates (ngates={ngates}, "
                f"words={n_words}); filling this ray with bad/missing data.",
                RuntimeWarning,
            )

        return out
    
    def get_sweep(self, param_name):
        parm = self.params.get(param_name)
        if parm is None:
            raise KeyError(f"Product '{param_name}' not found")

        scale = parm["parameter_scale"]
        bias  = parm["parameter_bias"]
        bad   = parm["bad_data"]

        azimuth   = []
        elevation = []
        data      = []

        for ray in self.rays:
            if param_name not in ray["rdat"]:
                continue
            raw = ray["rdat"][param_name]
            masked = np.where(raw == bad, np.nan, (raw - bias) / scale)
            azimuth.append(ray["azimuth"])
            elevation.append(ray["elevation"])
            data.append(masked)

        # DORADE products will have different shapes at times, so as a fallback we find the product with the largest shape, and pad them with bad data (it gets filtered out)
        target_len = max(len(d) for d in data)
        padded = []
        for d in data:
            if len(d) < target_len:
                pad = np.full(target_len - len(d), bad, dtype="float32")
                padded.append(np.concatenate([d, pad]))
            else:
                padded.append(d)

        data_arr = np.vstack(padded)
        ngates = data_arr.shape[1]

        if "meters_to_first_cell" in parm and parm["meters_to_first_cell"] != 0:
            r0      = parm["meters_to_first_cell"]
            spacing = parm["meters_between_cells"]
        elif hasattr(self, "dist_cells") and self.dist_cells is not None:
            r0      = self.dist_cells[0]
            spacing = self.dist_cells[1] - self.dist_cells[0]
        else:
            raise RuntimeError("No range information available")

        ranges = r0 + np.arange(ngates) * spacing

        return {
            "azimuth":    np.array(azimuth),
            "elevation":  np.array(elevation),
            "data":       data_arr,
            "range":      ranges,
            "radar_name": self.instrument_name,
            "start_time": self.sweep_start_time,
            "stop_time":  self.sweep_stop_time,
        }

def read_parm_block(buf, pos, endian="<"):
    header = struct.unpack(endian + "4s i", buf[pos:pos+8])
    block_id, nbytes = header

    if block_id != b"PARM":
        raise ValueError("Not a PARM block")

    if nbytes == 104:
        block_def = PARM_BLOCK_104
    elif nbytes == 216:
        block_def = PARM_BLOCK_216
    else:
        raise ValueError(f"Unknown PARM size: {nbytes}")

    return _unpack_from_buf(buf, pos, block_def, endian)

def _sweep_mode(value):
    return {
        0: "Cal",
        1: "PPI",
        2: "Coplane",
        3: "RHI",
        4: "Vertical",
        5: "Target",
        6: "Manual",
        7: "Idle",
        8: "PPI",
    }[value]

def _ray_status(value):
    return {
        0: "normal",
        1: "transition",
        2: "bad",
    }[value]

def _structure_size(structure, endian=">"):
    return struct.calcsize(endian + "".join([i[1] for i in structure]))


def _unpack_from_buf(buf, pos, structure, endian=">"):
    size = _structure_size(structure, endian)
    return _unpack_structure(buf[pos : pos + size], structure, endian)


def _unpack_structure(string, structure, endian=">"):
    fmt = endian + "".join([i[1] for i in structure])
    lst = struct.unpack(fmt, string)
    return dict(zip([i[0] for i in structure], lst))

BYTE = "B"
INT1 = "B"
INT2 = "H"
INT4 = "I"
REAL4 = "f"
REAL8 = "d"
SINT1 = "b"
SINT2 = "h"
SINT4 = "i"

# below claude did the block organizing, saved me a lot of time

# ── SSWB  (196 bytes) ────────────────────────────────────────────────────────
SSWB_BLOCK = (
    ("id",                   "4s"),
    ("nbytes",               SINT4),
    ("last_used",            SINT4),   # Unix time; 0 = never age off
    ("start_time",           SINT4),   # Unix time
    ("stop_time",            SINT4),   # Unix time
    ("sizeof_file",          SINT4),
    ("compression_flag",     SINT4),
    ("volume_time_stamp",    SINT4),
    ("num_params",           SINT4),
    ("radar_name",           "8s"),
    ("d_start_time",         REAL8),   # high-precision volume start
    ("d_stop_time",          REAL8),   # high-precision volume stop
    ("version_num",          SINT4),
    ("num_key_tables",       SINT4),
    ("status",               SINT4),
    ("place_holder",         "28s"),   # 7 × si32 unused
    # key_table[0]
    ("key_table_0_offset",   SINT4),
    ("key_table_0_size",     SINT4),
    ("key_table_0_type",     SINT4),
    # key_table[1]
    ("key_table_1_offset",   SINT4),
    ("key_table_1_size",     SINT4),
    ("key_table_1_type",     SINT4),
    # key_table[2]
    ("key_table_2_offset",   SINT4),
    ("key_table_2_size",     SINT4),
    ("key_table_2_type",     SINT4),
    # key_table[3]
    ("key_table_3_offset",   SINT4),
    ("key_table_3_size",     SINT4),
    ("key_table_3_type",     SINT4),
    # key_table[4]
    ("key_table_4_offset",   SINT4),
    ("key_table_4_size",     SINT4),
    ("key_table_4_type",     SINT4),
    # key_table[5]
    ("key_table_5_offset",   SINT4),
    ("key_table_5_size",     SINT4),
    ("key_table_5_type",     SINT4),
    # key_table[6]
    ("key_table_6_offset",   SINT4),
    ("key_table_6_size",     SINT4),
    ("key_table_6_type",     SINT4),
    # key_table[7]
    ("key_table_7_offset",   SINT4),
    ("key_table_7_size",     SINT4),
    ("key_table_7_type",     SINT4),
)

# ── COMM  (508 bytes) ────────────────────────────────────────────────────────
COMM_BLOCK = (
    ("id",      "4s"),
    ("nbytes",  SINT4),
    ("comment", "500s"),
)

# ── VOLD  (72 bytes) ─────────────────────────────────────────────────────────
VOLD_BLOCK = (
    ("id",                "4s"),
    ("nbytes",            SINT4),
    ("format_version",    SINT2),
    ("volume_num",        SINT2),
    ("maximum_bytes",     SINT4),
    ("proj_name",         "20s"),
    ("year",              SINT2),
    ("month",             SINT2),
    ("day",               SINT2),
    ("data_set_hour",     SINT2),
    ("data_set_minute",   SINT2),
    ("data_set_second",   SINT2),
    ("flight_num",        "8s"),
    ("gen_facility",      "8s"),
    ("gen_year",          SINT2),
    ("gen_month",         SINT2),
    ("gen_day",           SINT2),
    ("number_sensor_des", SINT2),
)

# ── RADD  (300 bytes) ────────────────────────────────────────────────────────
RADD_BLOCK = (
    ("id",                    "4s"),
    ("nbytes",                SINT4),
    ("radar_name",            "8s"),
    ("radar_const",           REAL4),
    ("peak_power",            REAL4),
    ("noise_power",           REAL4),
    ("receiver_gain",         REAL4),
    ("antenna_gain",          REAL4),
    ("system_gain",           REAL4),
    ("horz_beam_width",       REAL4),
    ("vert_beam_width",       REAL4),
    ("radar_type",            SINT2),
    ("scan_mode",             SINT2),
    ("req_rotat_vel",         REAL4),
    ("scan_mode_pram0",       REAL4),
    ("scan_mode_pram1",       REAL4),
    ("num_parameter_des",     SINT2),
    ("total_num_des",         SINT2),
    ("data_compress",         SINT2),
    ("data_reduction",        SINT2),
    ("data_red_parm0",        REAL4),
    ("data_red_parm1",        REAL4),
    ("radar_longitude",       REAL4),
    ("radar_latitude",        REAL4),
    ("radar_altitude",        REAL4),
    ("eff_unamb_vel",         REAL4),
    ("eff_unamb_range",       REAL4),
    ("num_freq_trans",        SINT2),
    ("num_ipps_trans",        SINT2),
    ("freq1",                 REAL4),
    ("freq2",                 REAL4),
    ("freq3",                 REAL4),
    ("freq4",                 REAL4),
    ("freq5",                 REAL4),
    ("prt1",                  REAL4),
    ("prt2",                  REAL4),
    ("prt3",                  REAL4),
    ("prt4",                  REAL4),
    ("prt5",                  REAL4),
)

# ── CFAC  (72 bytes) ─────────────────────────────────────────────────────────
CFAC_BLOCK = (
    ("id",                "4s"),
    ("nbytes",            SINT4),
    ("azimuth_corr",      REAL4),
    ("elevation_corr",    REAL4),
    ("range_delay_corr",  REAL4),
    ("longitude_corr",    REAL4),
    ("latitude_corr",     REAL4),
    ("pressure_alt_corr", REAL4),
    ("radar_alt_corr",    REAL4),
    ("ew_gndspd_corr",    REAL4),
    ("ns_gndspd_corr",    REAL4),
    ("vert_vel_corr",     REAL4),
    ("heading_corr",      REAL4),
    ("roll_corr",         REAL4),
    ("pitch_corr",        REAL4),
    ("drift_corr",        REAL4),
    ("rot_angle_corr",    REAL4),
    ("tilt_corr",         REAL4),
)

PARM_BLOCK_104 = (
    ("id",                    "4s"),
    ("nbytes",                SINT4),
    ("parameter_name",        "8s"),
    ("param_description",     "40s"),
    ("param_units",           "8s"),
    ("interpulse_time",       SINT2),
    ("xmitted_freq",          SINT2),
    ("recvr_bandwidth",       REAL4),
    ("pulse_width",           SINT2),
    ("polarization",          SINT2),
    ("num_samples",           SINT2),
    ("binary_format",         SINT2),
    ("threshold_field",       "8s"),
    ("threshold_value",       REAL4),
    ("parameter_scale",       REAL4),
    ("parameter_bias",        REAL4),
    ("bad_data",              SINT4),
)

PARM_BLOCK_216 = (
    ("id",                    "4s"),
    ("nbytes",                SINT4),
    ("parameter_name",        "8s"),
    ("param_description",     "40s"),
    ("param_units",           "8s"),
    ("interpulse_time",       SINT2),
    ("xmitted_freq",          SINT2),
    ("recvr_bandwidth",       REAL4),
    ("pulse_width",           SINT2),
    ("polarization",          SINT2),
    ("num_samples",           SINT2),
    ("binary_format",         SINT2),
    ("threshold_field",       "8s"),
    ("threshold_value",       REAL4),
    ("parameter_scale",       REAL4),
    ("parameter_bias",        REAL4),
    ("bad_data",              SINT4),
    ("extension_num",         SINT4),
    ("config_name",           "8s"),
    ("config_num",            SINT4),
    ("offset_to_data",        SINT4),
    ("mks_conversion",        REAL4),
    ("num_qnames",            SINT4),
    ("qdata_names",           "32s"),
    ("num_criteria",          SINT4),
    ("criteria_names",        "32s"),
    ("number_cells",          SINT4),
    ("meters_to_first_cell",  REAL4),
    ("meters_between_cells",  REAL4),
    ("eff_unamb_vel",         REAL4),
)

# ── CELV  (6012 bytes) ───────────────────────────────────────────────────────
CELV_BLOCK = (
    ("id",           "4s"),
    ("nbytes",       SINT4),
    ("number_cells", SINT4),
    ("dist_cells",   "6000s"),  # fl32[1500] — parse separately
)

# ── CSFD  (64 bytes) ─────────────────────────────────────────────────────────
CSFD_BLOCK = (
    ("id",            "4s"),
    ("nbytes",        SINT4),
    ("num_segments",  SINT4),
    ("dist_to_first", REAL4),
    ("spacing",       "32s"),   # fl32[8]
    ("num_cells",     "16s"),   # si16[8]
)

# ── SWIB  (40 bytes) ─────────────────────────────────────────────────────────
SWIB_BLOCK = (
    ("id",          "4s"),
    ("nbytes",      SINT4),
    ("radar_name",  "8s"),
    ("sweep_num",   SINT4),
    ("num_rays",    SINT4),
    ("start_angle", REAL4),
    ("stop_angle",  REAL4),
    ("fixed_angle", REAL4),
    ("filter_flag", SINT4),
)

# ── ASIB  (80 bytes) ─────────────────────────────────────────────────────────
ASIB_BLOCK = (
    ("id",              "4s"),
    ("nbytes",          SINT4),
    ("longitude",       REAL4),
    ("latitude",        REAL4),
    ("altitude_msl",    REAL4),
    ("altitude_agl",    REAL4),
    ("ew_velocity",     REAL4),
    ("ns_velocity",     REAL4),
    ("vert_velocity",   REAL4),
    ("heading",         REAL4),
    ("roll",            REAL4),
    ("pitch",           REAL4),
    ("drift_angle",     REAL4),
    ("rotation_angle",  REAL4),
    ("tilt",            REAL4),
    ("ew_horiz_wind",   REAL4),
    ("ns_horiz_wind",   REAL4),
    ("vert_wind",       REAL4),
    ("heading_change",  REAL4),
    ("pitch_change",    REAL4),
)

# ── RYIB  (44 bytes) ─────────────────────────────────────────────────────────
RYIB_BLOCK = (
    ("id",             "4s"),
    ("nbytes",         SINT4),
    ("sweep_num",      SINT4),
    ("julian_day",     SINT4),
    ("hour",           SINT2),
    ("minute",         SINT2),
    ("second",         SINT2),
    ("millisecond",    SINT2),
    ("azimuth",        REAL4),
    ("elevation",      REAL4),
    ("peak_power",     REAL4),
    ("true_scan_rate", REAL4),
    ("ray_status",     SINT4),
)

# ── RDAT  (16 bytes header; field data follows) ──────────────────────────────
RDAT_BLOCK = (
    ("id",         "4s"),
    ("nbytes",     SINT4),
    ("pdata_name", "8s"),
)

# ── QDAT  (56 bytes header; field data follows) ──────────────────────────────
QDAT_BLOCK = (
    ("id",              "4s"),
    ("nbytes",          SINT4),
    ("pdata_name",      "8s"),
    ("extension_num",   SINT4),
    ("config_num",      SINT4),
    ("first_cell",      "8s"),   # si16[4]
    ("num_cells",       "8s"),   # si16[4]
    ("criteria_value",  "16s"),  # fl32[4]
)

# ── XSTF  (24 bytes) ─────────────────────────────────────────────────────────
XSTF_BLOCK = (
    ("id",                  "4s"),
    ("nbytes",              SINT4),
    ("one",                 SINT4),   # always 1 (endian flag)
    ("source_format",       SINT4),
    ("offset_to_first_item", SINT4),
    ("transition_flag",     SINT4),
)

# ── NULL  (8 bytes) ──────────────────────────────────────────────────────────
NULL_BLOCK = (
    ("id",     "4s"),
    ("nbytes", SINT4),
)

# ── RKTB  (28 bytes header) ──────────────────────────────────────────────────
RKTB_BLOCK = (
    ("id",                  "4s"),
    ("nbytes",              SINT4),
    ("angle2ndx",           REAL4),   # 360.0 / ndx_que_size
    ("ndx_que_size",        SINT4),
    ("first_key_offset",    SINT4),
    ("angle_table_offset",  SINT4),
    ("num_rays",            SINT4),
)

# ── rot_table_entry  (12 bytes, repeated num_rays times after RKTB) ──────────
ROT_TABLE_ENTRY = (
    ("rotation_angle", REAL4),
    ("offset",         SINT4),
    ("size",           SINT4),
)

# ── FRAD  (52 bytes) ─────────────────────────────────────────────────────────
FRAD_BLOCK = (
    ("id",                "4s"),
    ("nbytes",            SINT4),
    ("data_sys_status",   SINT4),
    ("radar_name",        "8s"),
    ("test_pulse_level",  REAL4),
    ("test_pulse_dist",   REAL4),
    ("test_pulse_width",  REAL4),
    ("test_pulse_freq",   REAL4),
    ("test_pulse_atten",  SINT2),
    ("test_pulse_fnum",   SINT2),
    ("noise_power",       REAL4),
    ("ray_count",         SINT4),
    ("first_rec_gate",    SINT2),
    ("last_rec_gate",     SINT2),
)

# ── FRIB  (264 bytes) ────────────────────────────────────────────────────────
FRIB_BLOCK = (
    ("id",                      "4s"),
    ("nbytes",                  SINT4),
    ("data_sys_id",             SINT4),
    ("loss_out",                REAL4),
    ("loss_in",                 REAL4),
    ("loss_rjoint",             REAL4),
    ("ant_v_dim",               REAL4),
    ("ant_h_dim",               REAL4),
    ("ant_noise_temp",          REAL4),
    ("r_noise_figure",          REAL4),
    ("xmit_power",              "20s"),  # fl32[5]
    ("x_band_gain",             REAL4),
    ("receiver_gain",           "20s"),  # fl32[5]
    ("if_gain",                 "20s"),  # fl32[5]
    ("conversion_gain",         REAL4),
    ("scale_factor",            "20s"),  # fl32[5]
    ("processor_const",         REAL4),
    ("dly_tube_antenna",        SINT4),
    ("dly_rndtrip_chip_atod",   SINT4),
    ("dly_timmod_testpulse",    SINT4),
    ("dly_modulator_on",        SINT4),
    ("dly_modulator_off",       SINT4),
    ("peak_power_offset",       REAL4),
    ("test_pulse_offset",       REAL4),
    ("E_plane_angle",           REAL4),
    ("H_plane_angle",           REAL4),
    ("encoder_antenna_up",      REAL4),
    ("pitch_antenna_up",        REAL4),
    ("indepf_times_flg",        SINT2),
    ("indep_freq_gate",         SINT2),
    ("time_series_gate",        SINT2),
    ("num_base_params",         SINT2),
    ("file_name",               "80s"),
)

# ── LIDR  (148 bytes) ────────────────────────────────────────────────────────
LIDR_BLOCK = (
    ("id",                "4s"),
    ("nbytes",            SINT4),
    ("lidar_name",        "8s"),
    ("lidar_const",       REAL4),
    ("pulse_energy",      REAL4),
    ("peak_power",        REAL4),
    ("pulse_width",       REAL4),
    ("aperture_size",     REAL4),
    ("field_of_view",     REAL4),
    ("aperture_eff",      REAL4),
    ("beam_divergence",   REAL4),
    ("lidar_type",        SINT2),
    ("scan_mode",         SINT2),
    ("req_rotat_vel",     REAL4),
    ("scan_mode_pram0",   REAL4),
    ("scan_mode_pram1",   REAL4),
    ("num_parameter_des", SINT2),
    ("total_num_des",     SINT2),
    ("data_compress",     SINT2),
    ("data_reduction",    SINT2),
    ("data_red_parm0",    REAL4),
    ("data_red_parm1",    REAL4),
    ("lidar_longitude",   REAL4),
    ("lidar_latitude",    REAL4),
    ("lidar_altitude",    REAL4),
    ("eff_unamb_vel",     REAL4),
    ("eff_unamb_range",   REAL4),
    ("num_wvlen_trans",   SINT4),
    ("prf",               REAL4),
    ("wavelength",        "40s"),  # fl32[10]
)

# ── FLIB  (748 bytes) ────────────────────────────────────────────────────────
FLIB_BLOCK = (
    ("id",                  "4s"),
    ("nbytes",              SINT4),
    ("data_sys_id",         SINT4),
    ("transmit_beam_div",   "40s"),  # fl32[10]
    ("xmit_power",          "40s"),  # fl32[10]
    ("receiver_fov",        "40s"),  # fl32[10]
    ("receiver_type",       "40s"),  # si32[10]
    ("r_noise_floor",       "40s"),  # fl32[10]
    ("receiver_spec_bw",    "40s"),  # fl32[10]
    ("receiver_elec_bw",    "40s"),  # fl32[10]
    ("calibration",         "40s"),  # fl32[10]
    ("range_delay",         SINT4),
    ("peak_power_multi",    "40s"),  # fl32[10]
    ("encoder_mirror_up",   REAL4),
    ("pitch_mirror_up",     REAL4),
    ("max_digitizer_count", SINT4),
    ("max_digitizer_volt",  REAL4),
    ("digitizer_rate",      REAL4),
    ("total_num_samples",   SINT4),
    ("samples_per_cell",    SINT4),
    ("cells_per_ray",       SINT4),
    ("pmt_temp",            REAL4),
    ("pmt_gain",            REAL4),
    ("apd_temp",            REAL4),
    ("apd_gain",            REAL4),
    ("transect",            SINT4),
    ("derived_names",       "120s"),  # char[10][12]
    ("derived_units",       "80s"),   # char[10][8]
    ("temp_names",          "120s"),  # char[10][12]
)

# ── SITU  (4108 bytes) ───────────────────────────────────────────────────────
SITU_BLOCK = (
    ("id",            "4s"),
    ("nbytes",        SINT4),
    ("number_params", SINT4),
    ("params",        "4096s"),  # insitu_parameter_t[256]: name[8]+units[8] each
)

# ── ISIT  (16 bytes) ─────────────────────────────────────────────────────────
ISIT_BLOCK = (
    ("id",         "4s"),
    ("nbytes",     SINT4),
    ("julian_day", SINT2),
    ("hours",      SINT2),
    ("minutes",    SINT2),
    ("seconds",    SINT2),
)

# ── INDF  (8 bytes) ──────────────────────────────────────────────────────────
INDF_BLOCK = (
    ("id",     "4s"),
    ("nbytes", SINT4),
)

# ── MINI  (4112 bytes) ───────────────────────────────────────────────────────
MINI_BLOCK = (
    ("id",           "4s"),
    ("nbytes",       SINT4),
    ("command",      SINT2),
    ("status",       SINT2),
    ("temperature",  REAL4),
    ("x_axis_gyro",  "512s"),   # fl32[128]
    ("y_axis_gyro",  "512s"),   # fl32[128]
    ("z_axis_gyro",  "512s"),   # fl32[128]
    ("xr_axis_gyro", "512s"),   # fl32[128]
    ("x_axis_vel",   "512s"),   # fl32[128]
    ("y_axis_vel",   "512s"),   # fl32[128]
    ("z_axis_vel",   "512s"),   # fl32[128]
    ("x_axis_pos",   "512s"),   # fl32[128]
)

# ── NDDS  (16 bytes) ─────────────────────────────────────────────────────────
NDDS_BLOCK = (
    ("id",             "4s"),
    ("nbytes",         SINT4),
    ("ins_flag",       SINT2),
    ("gps_flag",       SINT2),
    ("minirims_flag",  SINT2),
    ("kalman_flag",    SINT2),
)

# ── TIME  (8 bytes) ──────────────────────────────────────────────────────────
TIME_BLOCK = (
    ("id",     "4s"),
    ("nbytes", SINT4),
)

# ── WAVE  (364 bytes) ────────────────────────────────────────────────────────
WAVE_BLOCK = (
    ("id",              "4s"),
    ("nbytes",          SINT4),
    ("ps_file_name",    "16s"),
    ("num_chips",       "12s"),   # si16[6]
    ("blank_chip",      "256s"),
    ("repeat_seq",      REAL4),
    ("repeat_seq_dwel", SINT2),
    ("total_pcp",       SINT2),
    ("chip_offset",     "12s"),   # si16[6]
    ("chip_width",      "12s"),   # si16[6]
    ("ur_pcp",          REAL4),
    ("uv_pcp",          REAL4),
    ("num_gates",       "12s"),   # si16[6]
    ("gate_dist1",      "4s"),    # si16[2]
    ("gate_dist2",      "4s"),    # si16[2]
    ("gate_dist3",      "4s"),    # si16[2]
    ("gate_dist4",      "4s"),    # si16[2]
    ("gate_dist5",      "4s"),    # si16[2]
)


# ── Helper: build a struct format string from a block definition ──────────────
def _block_fmt(block, big_endian=True):
    endian = ">" if big_endian else "<"
    return endian + "".join(fmt for _, fmt in block)


def _block_size(block):
    return struct.calcsize(_block_fmt(block))


def unpack_block(block_def, data, big_endian=True):
    """Unpack raw bytes into an OrderedDict using a block definition tuple."""
    fmt = _block_fmt(block_def, big_endian)
    size = struct.calcsize(fmt)
    values = struct.unpack(fmt, data[:size])
    return dict(zip((name for name, _ in block_def), values))


# ── Block-ID → definition map ────────────────────────────────────────────────
BLOCK_MAP = {
    b"COMM": COMM_BLOCK,
    b"SSWB": SSWB_BLOCK,
    b"VOLD": VOLD_BLOCK,
    b"RADD": RADD_BLOCK,
    b"CFAC": CFAC_BLOCK,
    b"CELV": CELV_BLOCK,
    b"CSFD": CSFD_BLOCK,
    b"SWIB": SWIB_BLOCK,
    b"ASIB": ASIB_BLOCK,
    b"RYIB": RYIB_BLOCK,
    b"RDAT": RDAT_BLOCK,
    b"QDAT": QDAT_BLOCK,
    b"XSTF": XSTF_BLOCK,
    b"NULL": NULL_BLOCK,
    b"RKTB": RKTB_BLOCK,
    b"FRAD": FRAD_BLOCK,
    b"FRIB": FRIB_BLOCK,
    b"LIDR": LIDR_BLOCK,
    b"FLIB": FLIB_BLOCK,
    b"SITU": SITU_BLOCK,
    b"ISIT": ISIT_BLOCK,
    b"INDF": INDF_BLOCK,
    b"MINI": MINI_BLOCK,
    b"NDDS": NDDS_BLOCK,
    b"TIME": TIME_BLOCK,
    b"WAVE": WAVE_BLOCK,
}