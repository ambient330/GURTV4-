# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""
readers/grib2_reader.py
-----------------------
Reader for JMA-style GRIB2 radar files, ported from test3.py.

Supports reflectivity-only or reflectivity + velocity pair.
No volume splitting — all sweeps are exposed as-is.

Usage
-----
    gf = Grib2File("refl.bin")
    gf = Grib2File("refl.bin", vel_path="vel.bin")

After construction the following attributes are ready (same names as
test3.py after convert() finishes, minus the write step):

    gf.time_reference       datetime
    gf.icao                 str
    gf.site_id              int
    gf.latitude             float
    gf.longitude            float
    gf.altitude             float
    gf.frequency            float   Hz
    gf.parameter_number     int     1=DBZ, 2=VEL
    gf.has_velocity         bool

    # flat ray arrays
    gf.time                 list[float]   seconds since time_reference
    gf.azimuth              list[float]   degrees
    gf.elevation            list[float]   degrees

    # per-sweep metadata
    gf.fixed_angle          list[float]
    gf.sweep_mode           list[str]
    gf.nb_list              list[int]
    gf.nr_list              list[int]
    gf.sweep_start_ray_index list[int]
    gf.sweep_end_ray_index   list[int]
    gf.sweep_number         list[int]
    gf.max_nb               int

    # range axis
    gf.radar_range_all      np.ndarray float32  shape (max_nb,)

    # field data — int16, scale_factor=0.01
    # DBZ: stored as (raw - 32), fill = MISSING_VALUE - 32  => read back: int16 * 0.01, fill->NaN
    # VEL: stored raw,           fill = MISSING_VALUE       => read back: int16 * 0.01, fill->NaN
    gf.refl_data            np.ndarray int16  shape (total_rays, max_nb)
    gf.vel_data             np.ndarray int16 or None
"""

import numpy as np
from datetime import datetime

MISSING_VALUE = -32768 + 32
_FillValueI16 = np.int16(MISSING_VALUE)
LONG_RANGE_BINS = 800


def read_int(ary, i0, i1, offset):
    return int.from_bytes(ary[offset + i0 - 1: offset + i1], "big")


def read_sint(ary, i0, i1, offset):
    l = 8 * (i1 - i0 + 1)
    result = read_int(ary, i0, i1, offset)
    if result & 1 << (l - 1) > 0:
        result = -(result & ~(1 << (l - 1)))
    return result


class Grib2File:

    def __init__(self, refl_path, vel_path=None):
        self.refl_path = refl_path
        self.vel_path  = vel_path
        self._read_refl()
        if self.vel_path:
            self._read_vel()
            self._filter_long_range_and_merge()
        else:
            self._expose_refl_as_is()

    def _read_refl(self):
        self._read_header(self.refl_path, is_vel=False)
        self._read_values(self.refl_path, is_vel=False)

    def _read_vel(self):
        saved = self.parameter_number
        self.parameter_number = 2
        self._read_header(self.vel_path, is_vel=True)
        self._read_values(self.vel_path, is_vel=True)
        self.parameter_number = saved

    def _read_header(self, filepath, is_vel):
        with open(filepath, "rb") as f:
            data = bytearray(f.read())

        offset = 0
        offset += 16  # Section 0

        year   = read_int(data, 13, 14, offset)
        month  = read_int(data, 15, 15, offset)
        day    = read_int(data, 16, 16, offset)
        hour   = read_int(data, 17, 17, offset)
        minute = read_int(data, 18, 18, offset)
        second = read_int(data, 19, 19, offset)
        time_string    = (f"{year:0>4}-{month:0>2}-{day:0>2}"
                          f"T{hour:0>2}:{minute:0>2}:{second:0>2}")
        time_reference = datetime.strptime(time_string, "%Y-%m-%dT%H:%M:%S")

        time_list      = []
        elevation_list = []
        azimuth_list   = []
        sweep_number   = []
        sweep_mode     = []
        fixed_angle    = []
        sweep_start    = []
        sweep_end      = []
        nb_list        = []
        nr_list        = []
        max_nb         = 0
        radar_range    = None
        parameter_number = None
        latitude = longitude = altitude = None
        icao = site_id = frequency = None
        bins = radials = 0

        while True:
            if data[offset:offset + 4] == b"7777":
                break

            sec_num = data[offset + 4]

            if sec_num == 3:
                bins    = read_int(data, 15, 18, offset)
                radials = read_int(data, 19, 22, offset)
                gate_spacing        = read_int(data, 31, 34, offset) * 1e-3
                range_to_first_gate = read_int(data, 35, 38, offset)
                rng = (range_to_first_gate
                       + np.arange(bins) * gate_spacing
                       + gate_spacing / 2)
                az  = read_int(data, 40, 41, offset) * 1e-2
                if bins > max_nb:
                    max_nb      = bins
                    radar_range = rng

            if sec_num == 4:
                parameter_number = read_int(data, 11, 11, offset)
                latitude   = read_int(data, 15, 18, offset) * 1e-6
                longitude  = read_int(data, 19, 22, offset) * 1e-6
                altitude   = read_int(data, 23, 24, offset) * 1e-1
                icao       = read_int(data, 25, 28, offset).to_bytes(4, "big").decode("ascii")
                site_id    = read_int(data, 29, 30, offset)
                frequency  = read_int(data, 33, 36, offset) * 1e3
                time_start = read_sint(data, 51, 52, offset)
                time_end   = read_sint(data, 53, 54, offset)

                time_list.extend(np.linspace(time_start, time_end, radials))
                nb_list.append(bins)
                nr_list.append(radials)

                sweep_start.append(len(azimuth_list))
                azimuth_list.extend(
                    [(az + 360 * i / radials) % 360 for i in range(radials)])
                sweep_end.append(len(azimuth_list) - 1)

                for i in range(radials):
                    elev = read_sint(data, 61 + 4 * i, 62 + 4 * i, offset)
                    elevation_list.append(elev * 1e-2)

                sweep_number.append(len(sweep_number))
                fa = read_sint(data, 42, 43, offset) * 1e-2
                fixed_angle.append(fa)
                sweep_mode.append(
                    "azimuth_surveillance" if fa < 90 else "rhi")

            offset += read_int(data, 1, 4, offset)

        if not is_vel:
            self.time_reference      = time_reference
            self.time_all            = time_list
            self.elevation_all       = elevation_list
            self.azimuth_all         = azimuth_list
            self.sweep_number_all    = sweep_number
            self.sweep_mode_all      = sweep_mode
            self.fixed_angle_all     = fixed_angle
            self.sweep_start_all     = sweep_start
            self.sweep_end_all       = sweep_end
            self.nb_list_all         = nb_list
            self.nr_list_all         = nr_list
            self.max_nb_all          = max_nb
            self.radar_range_all     = radar_range
            self.parameter_number    = parameter_number
            self.latitude            = latitude
            self.longitude           = longitude
            self.altitude            = altitude
            self.icao                = icao
            self.site_id             = site_id
            self.frequency           = frequency
        else:
            self.vel_time_all        = time_list
            self.vel_elevation_all   = elevation_list
            self.vel_azimuth_all     = azimuth_list
            self.vel_nb_list         = nb_list
            self.vel_nr_list         = nr_list
            self.vel_max_nb          = max_nb
            self.vel_sweep_start     = sweep_start
            self.vel_sweep_end       = sweep_end
            self.vel_fixed_angle     = fixed_angle

    def _read_values(self, filepath, is_vel):
        with open(filepath, "rb") as f:
            data = bytearray(f.read())

        nb_list = self.vel_nb_list if is_vel else self.nb_list_all
        nr_list = self.vel_nr_list if is_vel else self.nr_list_all
        max_nb  = self.vel_max_nb  if is_vel else self.max_nb_all

        total_rays = sum(nr_list)
        out_data   = np.full((total_rays, max_nb), _FillValueI16, dtype="int16")

        offset = 0
        offset += 16
        offset += read_int(data, 1, 4, offset)

        sweep_index = 0
        r_offset    = 0

        while True:
            if data[offset:offset + 4] == b"7777":
                break

            sec_num = data[offset + 4]
            sec_len = read_int(data, 1, 4, offset)

            if sec_num == 3:
                offset += sec_len
                continue

            if sec_num == 4:
                nb = nb_list[sweep_index]
                nr = nr_list[sweep_index]
                offset += sec_len

                s5_len    = read_int(data, 1, 4, offset)
                vmax      = read_int(data, 13, 14, offset)
                mlevels   = read_int(data, 15, 16, offset)
                scale_exp = data[offset + 16]

                value_table = []
                for i in range(mlevels):
                    raw = read_sint(data, 18 + 2*i, 19 + 2*i, offset)
                    if self.parameter_number == 2:
                        unsigned = read_int(data, 18 + 2*i, 19 + 2*i, offset)
                        raw = -(unsigned & 0x7FFF) if unsigned & 0x8000 else unsigned
                        raw = -raw
                    value_table.append(raw / (10 ** scale_exp))
                offset += s5_len

                if data[offset + 4] == 6:
                    offset += read_int(data, 1, 4, offset)

                s7_len   = read_int(data, 1, 4, offset)
                raw_data = data[offset + 5 + 6: offset + s7_len]
                offset  += s7_len

                total    = nb * nr
                out_vals = []
                i        = 0
                current_level = None

                while i < len(raw_data) and len(out_vals) < total:
                    byte_val = raw_data[i]
                    i += 1

                    if byte_val <= vmax:
                        current_level = byte_val
                        val = (value_table[current_level]
                               if current_level < len(value_table) else None)
                        out_vals.append(val)
                    else:
                        run_len  = 0
                        pwr      = 0
                        run_byte = byte_val
                        while run_byte > vmax:
                            run_code = run_byte - (vmax + 1)
                            run_len += (255 - vmax) ** pwr * run_code
                            pwr += 1
                            if i < len(raw_data):
                                run_byte = raw_data[i]
                                i += 1
                            else:
                                break
                        if run_byte <= vmax:
                            i -= 1
                        if current_level is not None and run_len > 0:
                            val = (value_table[current_level]
                                   if current_level < len(value_table) else None)
                            out_vals.extend([val] * min(run_len, total - len(out_vals)))

                if len(out_vals) < total:
                    out_vals = [None] * (total - len(out_vals)) + out_vals

                grid = np.full((nr, nb), _FillValueI16, dtype="int16")
                flat = out_vals[:total]
                for flat_i, v in enumerate(flat):
                    if v is not None:
                        grid[flat_i // nb, flat_i % nb] = np.int16(round(v * 100))

                out_data[r_offset:r_offset + nr, 0:nb] = grid
                r_offset    += nr
                sweep_index += 1
                continue

            offset += sec_len

        if is_vel:
            self.vel_data_all = out_data
        else:
            self.refl_data_all = out_data

    def _expose_refl_as_is(self):
        self.time      = self.time_all
        self.elevation = self.elevation_all
        self.azimuth   = self.azimuth_all
        self.fixed_angle           = self.fixed_angle_all
        self.sweep_mode            = self.sweep_mode_all
        self.nb_list               = self.nb_list_all
        self.nr_list               = self.nr_list_all
        self.sweep_start_ray_index = self.sweep_start_all
        self.sweep_end_ray_index   = self.sweep_end_all
        self.sweep_number          = self.sweep_number_all
        self.max_nb                = self.max_nb_all
        self.refl_data             = self.refl_data_all
        self.vel_data              = None
        self.has_velocity          = False

    def _filter_long_range_and_merge(self):
        doppler_sweep_idx = [i for i, nb in enumerate(self.nb_list_all)
                             if nb != LONG_RANGE_BINS]

        new_time      = []
        new_elevation = []
        new_azimuth   = []
        new_fixed     = []
        new_mode      = []
        new_nb        = []
        new_nr        = []
        new_start     = []
        new_end       = []
        new_data_rows = []

        for si in doppler_sweep_idx:
            s  = self.sweep_start_all[si]
            e  = self.sweep_end_all[si] + 1
            nr = self.nr_list_all[si]
            nb = self.nb_list_all[si]

            new_start.append(len(new_azimuth))
            new_time.extend(self.time_all[s:e])
            new_elevation.extend(self.elevation_all[s:e])
            new_azimuth.extend(self.azimuth_all[s:e])
            new_end.append(len(new_azimuth) - 1)
            new_fixed.append(self.fixed_angle_all[si])
            new_mode.append(self.sweep_mode_all[si])
            new_nb.append(nb)
            new_nr.append(nr)

            r_start = sum(self.nr_list_all[:si])
            new_data_rows.append(self.refl_data_all[r_start:r_start + nr, :])

        new_max_nb         = self.max_nb_all
        total_doppler_rays = sum(new_nr)

        refl_filtered = np.full((total_doppler_rays, new_max_nb),
                                _FillValueI16, dtype="int16")
        row = 0
        for block, nb in zip(new_data_rows, new_nb):
            nr = block.shape[0]
            refl_filtered[row:row + nr, 0:nb] = block[:, 0:nb]
            row += nr

        vel_max_nb      = self.vel_max_nb
        vel_data_interp = np.full((total_doppler_rays, vel_max_nb),
                                  _FillValueI16, dtype="int16")

        refl_row = 0
        for si_local, (refl_si_global, nb_r, nr_r) in enumerate(
                zip(doppler_sweep_idx, new_nb, new_nr)):

            best_vi = si_local
            if best_vi >= len(self.vel_fixed_angle):
                refl_row += nr_r
                continue

            vel_r_start = sum(self.vel_nr_list[:best_vi])
            vel_nr      = self.vel_nr_list[best_vi]
            vel_nb      = self.vel_nb_list[best_vi]
            vel_az      = np.array(self.vel_azimuth_all[
                self.vel_sweep_start[best_vi]:self.vel_sweep_end[best_vi] + 1])
            vel_rows    = self.vel_data_all[
                vel_r_start:vel_r_start + vel_nr, 0:vel_nb]

            refl_az = np.array(new_azimuth[new_start[si_local]:new_end[si_local] + 1])

            for ri, raz in enumerate(refl_az):
                diffs   = np.abs(vel_az - raz)
                diffs   = np.minimum(diffs, 360.0 - diffs)
                nearest = int(np.argmin(diffs))
                vel_data_interp[refl_row + ri, 0:vel_nb] = vel_rows[nearest, :]

            refl_row += nr_r

        vel_padded = np.full((total_doppler_rays, new_max_nb),
                             _FillValueI16, dtype="int16")
        vel_padded[:, :vel_max_nb] = vel_data_interp

        self.time      = new_time
        self.elevation = new_elevation
        self.azimuth   = new_azimuth
        self.fixed_angle           = new_fixed
        self.sweep_mode            = new_mode
        self.nb_list               = new_nb
        self.nr_list               = new_nr
        self.sweep_start_ray_index = new_start
        self.sweep_end_ray_index   = new_end
        self.sweep_number          = list(range(len(new_nb)))
        self.max_nb                = new_max_nb
        self.refl_data             = refl_filtered
        self.vel_data              = vel_padded
        self.has_velocity          = False