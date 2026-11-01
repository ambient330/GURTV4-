"""
Functions for reading NEXRAD level 2 files.
"""

# This file is part of the Py-ART, the Python ARM Radar Toolkit
# https://github.com/ARM-DOE/pyart

# Care has been taken to keep this file free from extraneous dependancies
# so that it can be used by other projects with no/minimal modification.

# Please feel free to use this file in other project provided the license
# below is followed. Keeping the above comment lines would also be helpful
# to direct other back to the Py-ART project and the source of this file.


LICENSE = """
Copyright (c) 2013, UChicago Argonne, LLC
All rights reserved.

Copyright 2013 UChicago Argonne, LLC. This software was produced under U.S.
Government contract DE-AC02-06CH11357 for Argonne National Laboratory (ANL),
which is operated by UChicago Argonne, LLC for the U.S. Department of Energy.
The U.S. Government has rights to use, reproduce, and distribute this
software. NEITHER THE GOVERNMENT NOR UCHICAGO ARGONNE, LLC MAKES ANY
WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS
SOFTWARE. If software is modified to produce derivative works, such modified
software should be clearly marked, so as not to confuse it with the version
available from ANL.

Additionally, redistribution and use in source and binary forms, with or
without modification, are permitted provided that the following conditions
are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of UChicago Argonne, LLC, Argonne National
      Laboratory, ANL, the U.S. Government, nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY UCHICAGO ARGONNE, LLC AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL UCHICAGO ARGONNE, LLC OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import bz2
import gzip
import io
import struct
import warnings
import zlib
from datetime import datetime, timedelta

import numpy as np


def _open_nexrad_stream(filename):
    """
    Open a NEXRAD Archive2 file, transparently unwrapping outer-file
    compression if present.

    Files distributed with an outer gzip, bzip2, or zlib wrapper (as
    opposed to the internal per-block bz2 compression used natively by
    Archive2) are detected via their magic bytes and fully decompressed
    into memory before parsing begins. Files with no outer wrapper are
    passed through unchanged.

    Parameters
    ----------
    filename : str or file-like
        Path to a NEXRAD Level II file, or an already-open file-like
        object (in which case it is returned unmodified; the caller is
        responsible for having already dealt with any outer compression
        in that case).

    Returns
    -------
    file-like
        A seekable, read()-able binary stream positioned at the start of
        the (now outer-decompressed) Archive2 data.
    """
    if hasattr(filename, "read"):
        return filename

    with open(filename, "rb") as f:
        head = f.read(4)
        f.seek(0)
        raw = f.read()

    if head[:2] == b"\x1f\x8b":
        # gzip
        data = gzip.decompress(raw)
    elif head[:3] == b"BZh":
        # bzip2 - whole file is a single bz2 stream
        data = bz2.decompress(raw)
    elif len(head) >= 2 and head[0] == 0x78 and head[1] in (
        0x01, 0x5E, 0x9C, 0xDA, 0xF9,
    ):
        # zlib
        data = zlib.decompress(raw)
    else:
        # no recognized outer wrapper - use as-is
        data = raw

    return io.BytesIO(data)


class NEXRADLevel2File:
    """
    Class for accessing data in a NEXRAD (WSR-88D) Level II file.

    NEXRAD Level II files [1]_, also know as NEXRAD Archive Level II or
    WSR-88D Archive level 2, are available from the NOAA National Climate Data
    Center [2]_ as well as on the UCAR THREDDS Data Server [3]_. Files with
    uncompressed messages and compressed messages are supported. This class
    supports reading both "message 31" and "message 1" type files.

    Parameters
    ----------
    filename : str
        Filename of Archive II file to read.

    Attributes
    ----------
    radial_records : list
        Radial (1 or 31) messages in the file.
    nscans : int
        Number of scans in the file.
    scan_msgs : list of arrays
        Each element specifies the indices of the message in the
        radial_records attribute which belong to a given scan.
    volume_header : dict
        Volume header.
    vcp : dict
        VCP information dictionary.
    _records : list
        A list of all records (message) in the file.
    _fh : file-like
        File like object from which data is read.
    _msg_type : '31' or '1':
        Type of radial messages in file.

    References
    ----------
    .. [1] http://www.roc.noaa.gov/WSR88D/Level_II/Level2Info.aspx
    .. [2] http://www.ncdc.noaa.gov/
    .. [3] http://thredds.ucar.edu/thredds/catalog.html

    """

    def __init__(self, filename):
        """initalize the object."""
        # read in the volume header and compression_record
        # (transparently unwraps an outer gzip/bzip2/zlib wrapper, if any)
        fh = _open_nexrad_stream(filename)
        size = _structure_size(VOLUME_HEADER)
        self.volume_header = _unpack_structure(fh.read(size), VOLUME_HEADER)
        compression_record = fh.read(COMPRESSION_RECORD_SIZE)

        # read the records in the file, decompressing as needed
        compression_slice = slice(CONTROL_WORD_SIZE, CONTROL_WORD_SIZE + 2)
        compression_or_ctm_info = compression_record[compression_slice]
        # The 4-byte control word immediately after the volume header is,
        # for internally (per-block) bz2-compressed files, the big-endian
        # length of the first bz2 block, and is immediately followed by
        # that block's data, which always starts with the bz2 stream
        # magic b"BZh". Checking for that magic (rather than demanding an
        # exact match against b"BZ") mirrors how LRose's NexradRadxFile
        # detects compression, and is more tolerant of files whose control
        # word doesn't hit one of the two known "uncompressed" patterns
        # below. See NexradRadxFile::isNexrad().
        if compression_record[CONTROL_WORD_SIZE:CONTROL_WORD_SIZE + 3] == b"BZh":
            buf = _decompress_records(fh)
        # The 12-byte compression record previously held the Channel Terminal
        # Manager (CTM) information. Bytes 4 through 6 contain the size of the
        # record (2432) as a big endian unsigned short, which is encoded as
        # b'\t\x80' == struct.pack('>H', 2432).
        # Newer files zero out this section.
        elif compression_or_ctm_info in (b"\x00\x00", b"\t\x80"):
            buf = fh.read()
        else:
            # Unrecognized control word. Rather than failing outright,
            # fall back to LRose's simpler heuristic: try treating the
            # data as internally bz2-compressed first (a nonzero control
            # word most often means compression, per
            # NexradRadxFile::isNexrad's `itest != 0` check), and if that
            # doesn't look like a valid bz2 stream, treat the file as
            # uncompressed.
            try:
                buf = _decompress_records(fh)
            except OSError as err:
                warnings.warn(
                    "Unrecognized compression control word "
                    f"{compression_or_ctm_info!r}; bz2 decompression "
                    f"failed ({err}). Falling back to treating the file "
                    "as uncompressed.",
                    stacklevel=2,
                )
                buf = fh.read()
        self._fh = fh

        # read the records from the buffer
        self._records = []
        buf_length = len(buf)
        pos = 0
        while pos < buf_length:
            pos, dic = _get_record_from_buf(buf, pos)
            self._records.append(dic)

        # pull out radial records (1 or 31) which contain the moment data.
        self.radial_records = [r for r in self._records if r["header"]["type"] == 31]
        self._msg_type = "31"
        if len(self.radial_records) == 0:
            self.radial_records = [r for r in self._records if r["header"]["type"] == 1]
            self._msg_type = "1"
        if len(self.radial_records) == 0:
            raise ValueError("No MSG31 records found, cannot read file")
        elev_nums = np.array(
            [m["msg_header"]["elevation_number"] for m in self.radial_records]
        )
        self.scan_msgs = [
            np.where(elev_nums == i + 1)[0] for i in range(elev_nums.max())
        ]
        self.nscans = len(self.scan_msgs)

        # pull out the vcp record
        msg_5 = [r for r in self._records if r["header"]["type"] == 5]

        if len(msg_5):
            self.vcp = msg_5[0]
        else:
            # There is no VCP Data.. This is uber dodgy
            warnings.warn(
                "No MSG5 detected. Setting to meaningless data. "
                "Rethink your life choices and be ready for errors."
                "Specifically fixed angle data will be missing"
            )

            self.vcp = None
        return

    def close(self):
        """Close the file."""
        self._fh.close()

    def location(self):
        """
        Find the location of the radar.

        Returns all zeros if location is not available.

        Returns
        -------
        latitude : float
            Latitude of the radar in degrees.
        longitude : float
            Longitude of the radar in degrees.
        height : int
            Height of radar and feedhorn in meters above mean sea level.

        """
        if self._msg_type == "31":
            dic = self.radial_records[0]["VOL"]
            height = dic["height"] + dic["feedhorn_height"]
            return dic["lat"], dic["lon"], height
        else:
            return 0.0, 0.0, 0.0

    def scan_info(self, scans=None):
        """
        Return a list of dictionaries with scan information.

        Parameters
        ----------
        scans : list ot None
            Scans (0 based) for which ray (radial) azimuth angles will be
            retrieved.  None (the default) will return the angles for all
            scans in the volume.

        Returns
        -------
        scan_info : list, optional
            A list of the scan performed with a dictionary with keys
            'moments', 'ngates', 'nrays', 'first_gate' and 'gate_spacing'
            for each scan.  The 'moments', 'ngates', 'first_gate', and
            'gate_spacing' keys are lists of the NEXRAD moments and gate
            information for that moment collected during the specific scan.
            The 'nrays' key provides the number of radials collected in the
            given scan.

        """
        info = []

        if scans is None:
            scans = range(self.nscans)
        for scan in scans:
            nrays = self.get_nrays(scan)
            if nrays < 2:
                self.nscans -= 1
                continue
            msg31_number = self.scan_msgs[scan][0]
            msg = self.radial_records[msg31_number]
            # Any block that isn't a known non-moment key and has an
            # "ngates" entry is a data-moment field, matching the
            # no-whitelist detection in _get_msg31_data_block(). Known
            # moments are listed first (for a stable, familiar ordering);
            # anything else found (a new/renamed moment block) is
            # appended afterwards rather than dropped.
            _non_moment_keys = {"header", "msg_header", "VOL", "ELV", "RAD"}
            known_order = ["REF", "VEL", "SW", "ZDR", "PHI", "RHO", "CFP"]
            present = {
                k for k, v in msg.items()
                if k not in _non_moment_keys and isinstance(v, dict) and "ngates" in v
            }
            moments = [f for f in known_order if f in present]
            moments += sorted(present - set(known_order))
            # Gate geometry (ngates/first_gate/gate_spacing) is reported
            # per-scan using the geometry shared by the most rays in the
            # scan, not just whichever ray happens to come first. A
            # handful of NEXRAD files have individual rays within a scan
            # that disagree with the rest (a truncated/corrupt message, a
            # SAILS/MRLE insertion, an odd legacy split-cut variant) and
            # trusting ray zero for the whole scan would silently treat
            # that minority ray's raw gate indices as if they were on the
            # majority grid. See _scan_moment_geometry() and get_data(),
            # which resamples any disagreeing ray onto this grid.
            ngates = []
            gate_spacing = []
            first_gate = []
            for f in moments:
                geom = self._scan_moment_geometry(scan, f)
                ngates.append(geom["ngates"])
                gate_spacing.append(geom["gate_spacing"])
                first_gate.append(geom["first_gate"])

            # The classic low-res-to-high-res gate spacing ratio ("the
            # scan's spacing is either 1/2 or 1/4 of the finest moment in
            # the scan") that upstream Py-ART uses to pick between its
            # _fast_interpolate_scan_2/_4 routines - kept here for
            # compatibility/inspection. Unlike Py-ART, which raises
            # ValueError for any other ratio, a moment whose spacing isn't
            # cleanly 2x or 4x the finest moment's spacing just gets a
            # multiplier of 1 (no special-cased ratio; get_data() still
            # resamples it correctly via plain nearest-neighbor regardless
            # of what this value is).
            finest_spacing = min(gate_spacing) if gate_spacing else None
            multiplier = [
                _gate_spacing_multiplier(sp, finest_spacing) for sp in gate_spacing
            ]

            info.append(
                {
                    "nrays": nrays,
                    "ngates": ngates,
                    "gate_spacing": gate_spacing,
                    "first_gate": first_gate,
                    "moments": moments,
                    "spacing_multiplier": multiplier,
                }
            )
        return info

    def _scan_moment_geometry(self, scan, moment):
        """
        Determine the canonical (first_gate, gate_spacing, ngates) gate
        geometry for a moment within a scan.

        Individual rays occasionally report different gate geometry than
        the rest of the scan for the same moment - a truncated or corrupt
        message, a SAILS/MRLE insertion, an odd legacy split-cut variant,
        etc. Rather than trusting whichever ray happens to be first (which
        may itself be the outlier, and which is what would silently
        misalign that ray's data if unchecked), the geometry shared by the
        largest number of rays in the scan is taken as canonical. Rays
        that disagree are left for get_data() to resample onto this grid
        with nearest-neighbor lookup, rather than being copied as if they
        matched.

        Ties are broken in favor of the finer-resolution (larger ngates)
        candidate so real data isn't unnecessarily discarded.

        Parameters
        ----------
        scan : int
            Scan of interest (0 based).
        moment : str
            Moment name, e.g. 'REF', 'VEL', 'SW'.

        Returns
        -------
        geometry : dict or None
            Dictionary with 'first_gate', 'gate_spacing' and 'ngates' keys,
            or None if the moment is not present in any ray of the scan.

        """
        counts = {}
        for msg_num in self.scan_msgs[scan]:
            msg = self.radial_records[msg_num]
            if moment not in msg:
                continue
            dic = msg[moment]
            key = (dic["first_gate"], dic["gate_spacing"], dic["ngates"])
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return None
        best_key = max(counts.items(), key=lambda kv: (kv[1], kv[0][2]))[0]
        first_gate, gate_spacing, ngates = best_key
        return {
            "first_gate": first_gate,
            "gate_spacing": gate_spacing,
            "ngates": ngates,
        }

    def get_vcp_pattern(self):
        """
        Return the numerical volume coverage pattern (VCP) or None if unknown.
        """
        if self.vcp is None:
            return None
        else:
            return self.vcp["msg5_header"]["pattern_number"]

    def get_vcp_supplemental_info(self, scan=None):
        """
        Return SAILS/MRLE/MPDA supplemental scan information.

        Parameters
        ----------
        scan : int or None
            Scan (0 based) to retrieve per-cut supplemental info for. If
            None, only the VCP-wide info is returned (no 'cut' key).

        Returns
        -------
        info : dict
            'sails_sequences' / 'mrle_sequences' : int, total number of
                SAILS/MRLE sequences configured for this VCP (0 if none).
            'mpda_vcp' : bool, whether this is an MPDA (Multi-Purpose
                Dwell Aloft) VCP.
            'cut' : dict, present only if `scan` is given. Per-cut flags:
                'sails_sequence', 'mrle_sequence' (int, this cut's
                sequence number, 0 if not applicable), 'mpda_cut' and
                'base_tilt_cut' (bool).

        """
        if self.vcp is None:
            info = {"sails_sequences": 0, "mrle_sequences": 0, "mpda_vcp": False}
        else:
            info = dict(self.vcp["msg5_header"]["supplemental_data"])

        if scan is not None and self.vcp is not None:
            msg31_number = self.scan_msgs[scan][0]
            elevation_number = self.radial_records[msg31_number]["msg_header"][
                "elevation_number"
            ]
            cut = self.vcp["cut_parameters"][elevation_number - 1]
            info["cut"] = dict(cut["supplemental_data_decoded"])

        return info

    def get_nrays(self, scan):
        """
        Return the number of rays in a given scan.

        Parameters
        ----------
        scan : int
            Scan of interest (0 based).

        Returns
        -------
        nrays : int
            Number of rays (radials) in the scan.

        """
        return len(self.scan_msgs[scan])

    def get_range(self, scan_num, moment):
        """
        Return an array of gate ranges for a given scan and moment.

        Parameters
        ----------
        scan_num : int
            Scan number (0 based).
        moment : 'REF', 'VEL', 'SW', 'ZDR', 'PHI', 'RHO', or 'CFP'
            Moment of interest.

        Returns
        -------
        range : ndarray
            Range in meters from the antenna to the center of gate (bin).

        """
        dic = self.radial_records[self.scan_msgs[scan_num][0]][moment]
        ngates = dic["ngates"]
        first_gate = dic["first_gate"]
        gate_spacing = dic["gate_spacing"]
        return np.arange(ngates) * gate_spacing + first_gate

    # helper functions for looping over scans
    def _msg_nums(self, scans):
        """Find the all message number for a list of scans."""
        return np.concatenate([self.scan_msgs[i] for i in scans])

    def _radial_array(self, scans, key):
        """
        Return an array of radial header elements for all rays in scans.
        """
        msg_nums = self._msg_nums(scans)
        temp = [self.radial_records[i]["msg_header"][key] for i in msg_nums]
        return np.array(temp)

    def _radial_sub_array(self, scans, key):
        """
        Return an array of RAD or msg_header elements for all rays in scans.
        """
        msg_nums = self._msg_nums(scans)
        if self._msg_type == "31":
            tmp = [self.radial_records[i]["RAD"][key] for i in msg_nums]
        else:
            tmp = [self.radial_records[i]["msg_header"][key] for i in msg_nums]
        return np.array(tmp)

    def get_times(self, scans=None):
        """
        Retrieve the times at which the rays were collected.

        Parameters
        ----------
        scans : list or None
            Scans (0-based) to retrieve ray (radial) collection times from.
            None (the default) will return the times for all scans in the
            volume.

        Returns
        -------
        time_start : Datetime
            Initial time.
        time : ndarray
            Offset in seconds from the initial time at which the rays
            in the requested scans were collected.

        """
        if scans is None:
            scans = range(self.nscans)
        days = self._radial_array(scans, "collect_date")
        secs = self._radial_array(scans, "collect_ms") / 1000.0
        offset = timedelta(days=int(days[0]) - 1, seconds=int(secs[0]))
        time_start = datetime(1970, 1, 1) + offset
        time = secs - int(secs[0]) + (days - days[0]) * 86400
        return time_start, time

    def get_azimuth_angles(self, scans=None):
        """
        Retrieve the azimuth angles of all rays in the requested scans.

        Parameters
        ----------
        scans : list ot None
            Scans (0 based) for which ray (radial) azimuth angles will be
            retrieved. None (the default) will return the angles for all
            scans in the volume.

        Returns
        -------
        angles : ndarray
            Azimuth angles in degress for all rays in the requested scans.

        """
        if scans is None:
            scans = range(self.nscans)
        if self._msg_type == "1":
            scale = 180 / (4096 * 8.0)
        else:
            scale = 1.0
        return self._radial_array(scans, "azimuth_angle") * scale

    def get_elevation_angles(self, scans=None):
        """
        Retrieve the elevation angles of all rays in the requested scans.

        Parameters
        ----------
        scans : list or None
            Scans (0 based) for which ray (radial) azimuth angles will be
            retrieved. None (the default) will return the angles for
            all scans in the volume.

        Returns
        -------
        angles : ndarray
            Elevation angles in degress for all rays in the requested scans.

        """
        if scans is None:
            scans = range(self.nscans)
        if self._msg_type == "1":
            scale = 180 / (4096 * 8.0)
        else:
            scale = 1.0
        return self._radial_array(scans, "elevation_angle") * scale

    def get_target_angles(self, scans=None):
        """
        Retrieve the target elevation angle of the requested scans.

        Parameters
        ----------
        scans : list or None
            Scans (0 based) for which the target elevation angles will be
            retrieved. None (the default) will return the angles for all
            scans in the volume.

        Returns
        -------
        angles : ndarray
            Target elevation angles in degress for the requested scans.

        """
        if scans is None:
            scans = range(self.nscans)
        if self._msg_type == "31":
            if self.vcp is not None:
                cut_parameters = self.vcp["cut_parameters"]
            else:
                cut_parameters = [{"elevation_angle_deg": 0.0}] * self.nscans
            return np.array(
                [cut_parameters[i]["elevation_angle_deg"] for i in scans],
                dtype="float32",
            )
        else:
            scale = 180 / (4096 * 8.0)
            msgs = [self.radial_records[self.scan_msgs[i][0]] for i in scans]
            return np.round(
                np.array(
                    [m["msg_header"]["elevation_angle"] * scale for m in msgs],
                    dtype="float32",
                ),
                1,
            )

    def get_nyquist_vel(self, scans=None):
        """
        Retrieve the Nyquist velocities of the requested scans.

        Parameters
        ----------
        scans : list or None
            Scans (0 based) for which the Nyquist velocities will be
            retrieved. None (the default) will return the velocities for all
            scans in the volume.

        Returns
        -------
        velocities : ndarray
            Nyquist velocities (in m/s) for the requested scans.

        """
        if scans is None:
            scans = range(self.nscans)
        return self._radial_sub_array(scans, "nyquist_vel") * 0.01

    def get_unambigous_range(self, scans=None):
        """
        Retrieve the unambiguous range of the requested scans.

        Parameters
        ----------
        scans : list or None
            Scans (0 based) for which the unambiguous range will be retrieved.
            None (the default) will return the range for all scans in the
            volume.

        Returns
        -------
        unambiguous_range : ndarray
            Unambiguous range (in meters) for the requested scans.

        """
        if scans is None:
            scans = range(self.nscans)
        # unambiguous range is stored in tenths of km, x100 for meters
        return self._radial_sub_array(scans, "unambig_range") * 100.0

    def get_data(self, moment, max_ngates, scans=None, raw_data=False,
                 first_gate=None, gate_spacing=None):
        """
        Retrieve moment data for a given set of scans.

        Masked points indicate that the data was not collected, below
        threshold or is range folded.

        Parameters
        ----------
        moment : 'REF', 'VEL', 'SW', 'ZDR', 'PHI', 'RHO', or 'CFP'
            Moment for which to to retrieve data.
        max_ngates : int
            Maximum number of gates (bins) in any ray.
            requested.
        raw_data : bool
            True to return the raw data, False to perform masking as well as
            applying the appropiate scale and offset to the data.  When
            raw_data is True values of 1 in the data likely indicate that
            the gate was not present in the sweep, in some cases in will
            indicate range folded data.
        scans : list or None.
            Scans to retrieve data from (0 based). None (the default) will
            get the data for all scans in the volume.
        first_gate, gate_spacing : float or None
            The canonical gate geometry (as returned by scan_info(), e.g.
            via _scan_moment_geometry()) that every ray in the output
            should end up on. If given, any ray whose own first_gate/
            gate_spacing/ngates for this moment doesn't match is resampled
            onto this grid using nearest-neighbor lookup rather than
            having its raw (differently-spaced) gate indices copied in
            directly; gates with no corresponding source gate are masked.
            The ratio between a ray's native spacing and the canonical
            spacing is not restricted to any particular value (e.g. 2x or
            4x) - any ratio is handled the same way. If omitted, rays are
            copied as-is (the previous, non-resampling behavior).

        Returns
        -------
        data : ndarray

        """
        if scans is None:
            scans = range(self.nscans)

        # determine the number of rays
        msg_nums = self._msg_nums(scans)
        nrays = len(msg_nums)
        # extract the data
        set_datatype = False
        data = np.ones((nrays, max_ngates), ">B")

        resample = first_gate is not None and gate_spacing is not None
        target_ranges = None
        if resample and gate_spacing != 0:
            target_ranges = first_gate + np.arange(max_ngates, dtype="float64") * gate_spacing

        for i, msg_num in enumerate(msg_nums):
            msg = self.radial_records[msg_num]
            if moment not in msg.keys():
                continue
            if not set_datatype:
                data = data.astype(">" + _bits_to_code(msg, moment))
                set_datatype = True

            mdic = msg[moment]
            raw = mdic["data"]
            ngates_ray = mdic["ngates"]
            ray_first = mdic["first_gate"]
            ray_spacing = mdic["gate_spacing"]

            on_target_grid = (
                not resample
                or (ngates_ray == max_ngates
                    and ray_first == first_gate
                    and ray_spacing == gate_spacing)
            )

            if on_target_grid:
                ngates = min(ngates_ray, max_ngates, len(raw))
                data[i, :ngates] = raw[:ngates]
                continue

            # This ray's own gate geometry disagrees with the canonical
            # grid for this scan/moment (a truncated or corrupt message, a
            # SAILS/MRLE insertion, a legacy split-cut oddity, etc). Rather
            # than copying its raw indices as though they lined up with
            # the target grid - which would silently place its gates at
            # the wrong ranges - resample it with nearest-neighbor lookup.
            # No assumption is made about the ratio between the ray's
            # spacing and the target spacing (unlike some readers, which
            # only handle an exact 1/2 or 1/4 ratio and error out
            # otherwise); any ratio, including non-integer ones, works.
            if ngates_ray <= 0 or not ray_spacing or target_ranges is None:
                continue
            src_last = ray_first + (ngates_ray - 1) * ray_spacing
            idx = np.round((target_ranges - ray_first) / ray_spacing).astype(np.int64)
            idx_clipped = np.clip(idx, 0, ngates_ray - 1)
            usable = min(ngates_ray, len(raw))
            in_range = (
                (target_ranges >= ray_first - ray_spacing / 2.0)
                & (target_ranges <= src_last + ray_spacing / 2.0)
                & (idx_clipped < usable)
            )
            row = np.ones(max_ngates, dtype=data.dtype)
            row[in_range] = raw[idx_clipped[in_range]]
            data[i, :] = row

        # return raw data if requested
        if raw_data:
            return data

        # mask, scan and offset, assume that the offset and scale
        # are the same in all scans/gates
        for scan in scans:  # find a scan which contains the moment
            msg_num = self.scan_msgs[scan][0]
            msg = self.radial_records[msg_num]
            if moment in msg.keys():
                offset = np.float32(msg[moment]["offset"])
                scale = np.float32(msg[moment]["scale"])
                mask = data <= 1
                scaled_data = ((data - offset) / scale).astype("float64")
                # Write NaN into the array itself, not just the mask
                # flag. GURT's PPI gridding step reads sw.fields[...]
                # via a plain np.asarray(..., dtype="float32") call
                # (see gurt_gui.py) before ever checking
                # np.ma.isMaskedArray() — which silently drops a masked
                # array's mask and keeps only its underlying (raw
                # sentinel-derived) numbers. A mask-only flag on "below
                # threshold"/"range folded" gates therefore doesn't
                # survive to the renderer and paints bogus colored data
                # instead of a transparent gap. Baking the NaN in here,
                # mirroring readers/odim_h5_reader.py, fixes that.
                scaled_data[mask] = np.nan
                return np.ma.masked_invalid(scaled_data)

        # moment is not present in any scan: there's no real data to
        # return, so every gate should render as missing rather than as
        # spurious values built from the raw sentinel array.
        return np.ma.masked_invalid(np.full(data.shape, np.nan, dtype="float64"))

    def get_range_folded_mask(self, moment, max_ngates, scans=None,
                               first_gate=None, gate_spacing=None):
        """
        Boolean mask, same shape/gate-alignment as get_data()'s output,
        True where the raw digital code is 1 -- range folded, per the
        Archive II convention (distinct from 0, "below threshold").

        get_data() collapses both codes into one NaN mask (``data <= 1``)
        since for rendering they're both "no usable value". This method
        reuses get_data()'s raw_data=True path (same resampling/alignment
        as the real call, just before scale/offset is applied) so a
        caller that also wants to flag RF specifically -- e.g. when
        exporting -- can tell the two apart without duplicating the gate
        resampling logic here.
        """
        raw = self.get_data(moment, max_ngates, scans=scans, raw_data=True,
                             first_gate=first_gate, gate_spacing=gate_spacing)
        return raw == 1


def is_nexrad_level2(path):
    """
    Detect whether *path* is a NEXRAD Level II archive (Msg1 or Msg31),
    compressed or not. Handles the modern 'AR2V00xx.xxx' tape name and the
    legacy 'ARCHIVE2.' name, and transparently unwraps a whole-file gzip
    wrapper before checking (per-record BZ2 compression, the far more
    common case for real archive files, is detected from the same first
    few bytes either way since it only affects data *after* the volume
    header).
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(32)
    except OSError:
        return False

    if head[:2] == b"\x1f\x8b":
        try:
            with gzip.open(path, "rb") as gz:
                head = gz.read(32)
        except OSError:
            return False

    return head[:9] == b"ARCHIVE2." or head[:6] == b"AR2V00"


def read_nexrad_level2(path, RadarVolume, SweepData):
    """
    Read a NEXRAD Level II file into a GURT RadarVolume.

    Follows the same calling convention as GURT's other native readers
    (e.g. readers/ukmo_reader.py, readers/odim_h5_reader.py): the caller's
    RadarVolume/SweepData classes are passed in so this module has no
    hard dependency on GURT itself.

    Handles BZ2 (LDM) and uncompressed archives, both MSG31 and legacy
    MSG1 files, real per-scan collection time, real site location, real
    VCP target angles (via the corrected BAM decode), Nyquist velocity,
    unambiguous range, and per-moment gate geometry (REF/VEL/SW etc. are
    resampled onto a common per-sweep range grid rather than assumed to
    share one, since split-cut super-res volumes commonly give them
    different first_gate/gate_spacing).
    """
    reader = NEXRADLevel2File(path)
    try:
        vol = RadarVolume()

        lat, lon, height = reader.location()
        vol.latitude = float(lat)
        vol.longitude = float(lon)
        vol.altitude = float(height)

        icao = reader.volume_header.get("icao", b"")
        if isinstance(icao, bytes):
            icao = icao.decode("ascii", errors="ignore").strip()
        vol.radar_name = icao or "NEXRAD"

        vol.scan_time, _ = reader.get_times()

        all_moments = set()
        for i in range(reader.nscans):
            info = reader.scan_info([i])
            if info:
                all_moments.update(info[0]["moments"])
        vol.field_names = sorted(all_moments)

        for i in range(reader.nscans):
            if reader.get_nrays(i) < 2:
                continue
            info_list = reader.scan_info([i])
            if not info_list:
                continue
            info = info_list[0]
            moments = info["moments"]
            if not moments:
                continue

            # Canonical range grid for this sweep: the moment with the most
            # gates (normally REF in super-res data) sets the geometry that
            # every other moment gets resampled onto.
            ref_idx = int(np.argmax(info["ngates"]))
            ref_ngates = info["ngates"][ref_idx]
            ref_spacing = float(info["gate_spacing"][ref_idx])
            ref_first = float(info["first_gate"][ref_idx])
            ranges = (ref_first + np.arange(ref_ngates, dtype="float32") * ref_spacing)

            sw = SweepData()
            sw.radar_name = vol.radar_name
            azimuth = reader.get_azimuth_angles([i]).astype("float32")
            elevation = reader.get_elevation_angles([i]).astype("float32")

            # GURT's PPI mesh builder (_build_ppi_geometry) calls
            # np.unwrap() directly on the azimuth array and assumes it's
            # already in clean angular order with at most one 0/360 wrap.
            # NEXRAD rays normally arrive that way, but split cuts,
            # SAILS/MRLE insertions, or a single out-of-sequence radial
            # in the record stream are enough to break that assumption —
            # unwrap() then produces garbage cell edges and the mesh
            # connects triangles between rays that aren't actually
            # angular neighbors, which renders as the periodic radial
            # striping/tearing artifact. Sorting defensively here
            # guarantees the contract the renderer actually needs.
            az_order = np.argsort(azimuth, kind="stable")
            sw.azimuth = azimuth[az_order]
            sw.elevation = elevation[az_order]

            target_angle = float(reader.get_target_angles([i])[0])
            # Some files (msg31 with no VCP metadata, certain replay/research
            # products) report a sentinel instead of a real target angle.
            # Fall back to the median of the actual per-ray elevations for this
            # sweep as an approximation. Median (not mean) so a handful of
            # stray/out-of-sequence rays near a split cut don't skew it.
            if (not np.isfinite(target_angle)
                    or abs(target_angle) >= 9999
                    or target_angle == 0.0):
                target_angle = float(np.median(elevation))
            sw.fixed_angle = target_angle

            #sw.fixed_angle = float(reader.get_target_angles([i])[0])
            sw.ranges = ranges

            scan_time, _ = reader.get_times([i])
            sw.scan_time = scan_time

            nyquist = reader.get_nyquist_vel([i])[az_order]
            unamb_range = reader.get_unambigous_range([i])[az_order]
            sw.instrument_parameters = {
                "nyquist_velocity": {"data": nyquist, "units": "meters_per_second"},
                "unambiguous_range": {"data": unamb_range, "units": "meters"},
            }

            try:
                sw.vcp_supplemental_info = reader.get_vcp_supplemental_info(i)
            except Exception:
                pass

            for m_idx, moment in enumerate(moments):
                ngates_m = info["ngates"][m_idx]
                spacing_m = float(info["gate_spacing"][m_idx])
                first_m = float(info["first_gate"][m_idx])

                # first_gate/gate_spacing here is this moment's own
                # canonical (majority) grid for the scan, per
                # _scan_moment_geometry() - this lets get_data()
                # nearest-neighbor resample any individual ray whose
                # gate geometry doesn't match the rest of the scan
                # (corrupt message, SAILS/MRLE insertion, etc), instead
                # of silently copying its raw, differently-spaced gate
                # indices as if they lined up.
                data = reader.get_data(
                    moment, ngates_m, scans=[i],
                    first_gate=first_m, gate_spacing=spacing_m,
                )
                data = data[az_order]

                rf_mask = reader.get_range_folded_mask(
                    moment, ngates_m, scans=[i],
                    first_gate=first_m, gate_spacing=spacing_m,
                )
                rf_mask = rf_mask[az_order]

                if ngates_m == ref_ngates and spacing_m == ref_spacing and first_m == ref_first:
                    field_data = data
                    field_rf = rf_mask
                else:
                    # Resample onto the canonical grid: nearest source gate
                    # for every canonical range, masking any canonical gate
                    # that falls outside this moment's actual coverage.
                    src_last = first_m + (ngates_m - 1) * spacing_m
                    idx = np.clip(
                        np.round((ranges - first_m) / spacing_m).astype(int),
                        0, max(ngates_m - 1, 0),
                    )
                    in_range = (ranges >= first_m - spacing_m / 2) & (
                        ranges <= src_last + spacing_m / 2
                    )
                    field_data = data[:, idx]
                    field_rf = rf_mask[:, idx]
                    # As in get_data() above, bake this into real NaNs
                    # rather than leaving it as a MaskedArray-only flag —
                    # gurt_gui.py's np.asarray(sweep.fields[...]) calls
                    # would otherwise silently drop the mask and let
                    # whichever nearest-neighbor gate happened to land
                    # there leak through as if it were real data outside
                    # this moment's actual coverage.
                    out_of_range = np.broadcast_to(
                        ~in_range[np.newaxis, :], field_data.shape
                    )
                    field_data = np.ma.filled(field_data.astype("float64"), np.nan)
                    field_data[out_of_range] = np.nan
                    field_data = np.ma.masked_invalid(field_data)
                    field_rf = field_rf & ~out_of_range

                sw.fields[moment] = field_data.astype("float32")
                if field_rf.any():
                    sw.range_folded[moment] = field_rf

            vol.sweeps.append(sw)

        if not vol.sweeps:
            raise ValueError("No usable sweeps found in NEXRAD Level II file.")

        return vol
    finally:
        reader.close()


def _gate_spacing_multiplier(spacing, finest_spacing, rel_tol=0.02):
    """
    Classify a moment's gate spacing relative to the finest (smallest)
    gate spacing present in the same scan, the same ratio upstream
    Py-ART uses to choose between its ``_fast_interpolate_scan_2`` and
    ``_fast_interpolate_scan_4`` routines for mixed-resolution rays::

        if spacing == gate_spacing * 4:
            interpolate['multiplier'] = '4'
        elif spacing == gate_spacing * 2:
            interpolate['multiplier'] = '2'
        else:
            raise ValueError('Gate spacing is neither 1/4 or 1/2')

    Some real NEXRAD files have moments/rays whose spacing isn't a clean
    2x or 4x multiple of the finest spacing in the scan (odd legacy
    values, or spacing that only misses the exact ratio because of an
    upstream truncation/rounding quirk). Rather than raising like Py-ART
    does, those simply get a multiplier of 1. This value is informational
    only - get_data()'s nearest-neighbor resampling doesn't use it and
    handles any ratio, clean or not, the same way.

    Parameters
    ----------
    spacing : float
        Gate spacing of the moment being classified.
    finest_spacing : float or None
        Smallest gate spacing among the moments in the scan.
    rel_tol : float
        Relative tolerance for matching the 2x/4x ratio, to absorb minor
        floating point/encoding noise.

    Returns
    -------
    multiplier : int
        4, 2, or 1 (default when the ratio doesn't cleanly match either).

    """
    if not finest_spacing or not spacing:
        return 1
    ratio = spacing / finest_spacing
    if abs(ratio - 4) <= 4 * rel_tol:
        return 4
    if abs(ratio - 2) <= 2 * rel_tol:
        return 2
    return 1


def _bits_to_code(msg, moment):
    """
    Convert number of bits to the proper code for unpacking.
    Based on the code found in MetPy:
    https://github.com/Unidata/MetPy/blob/40d5c12ab341a449c9398508bd41
    d010165f9eeb/src/metpy/io/_tools.py#L313-L321
    """
    if msg["header"]["type"] == 1:
        word_size = msg[moment]["data"].dtype
        if word_size == "uint16":
            return "H"
        elif word_size == "uint8":
            return "B"
        else:
            warnings.warn(('Unsupported bit size: %s. Returning "B"', word_size))
            return "B"

    elif msg["header"]["type"] == 31:
        word_size = msg[moment]["word_size"]
        if word_size == 16:
            return "H"
        elif word_size == 8:
            return "B"
        else:
            warnings.warn(('Unsupported bit size: %s. Returning "B"', word_size))
            return "B"
    else:
        raise TypeError("Unsupported msg type %s", msg["header"]["type"])


def _decompress_records(file_handler):
    """
    Decompressed the records from an BZ2 compressed Archive 2 file.
    """
    file_handler.seek(0)
    cbuf = file_handler.read()  # read all data from the file
    decompressor = bz2.BZ2Decompressor()
    skip = _structure_size(VOLUME_HEADER) + CONTROL_WORD_SIZE
    buf = bytearray(decompressor.decompress(cbuf[skip:]))
    while len(decompressor.unused_data):
        cbuf = decompressor.unused_data
        decompressor = bz2.BZ2Decompressor()
        buf += decompressor.decompress(cbuf[CONTROL_WORD_SIZE:])

    return buf[COMPRESSION_RECORD_SIZE:]


def _get_record_from_buf(buf, pos):
    """Retrieve and unpack a NEXRAD record from a buffer."""
    dic = {"header": _unpack_from_buf(buf, pos, MSG_HEADER)}
    msg_type = dic["header"]["type"]

    if msg_type == 31:
        new_pos = _get_msg31_from_buf(buf, pos, dic)
    elif msg_type == 5:
        # Sometimes we encounter incomplete buffers
        try:
            new_pos = _get_msg5_from_buf(buf, pos, dic)
        except struct.error:
            warnings.warn(
                "Encountered incomplete MSG5. File may be corrupt.", RuntimeWarning
            )
            new_pos = pos + RECORD_SIZE
    elif msg_type == 29:
        new_pos = _get_msg29_from_buf(pos, dic)
        warnings.warn("Message 29 encountered, not parsing.", RuntimeWarning)
    elif msg_type == 1:
        new_pos = _get_msg1_from_buf(buf, pos, dic)
    else:  # not message 31 or 1, no decoding performed
        new_pos = pos + RECORD_SIZE

    return new_pos, dic


def _get_msg29_from_buf(pos, dic):
    msg_size = dic["header"]["size"]
    if msg_size == 65535:
        msg_size = dic["header"]["segments"] << 16 | dic["header"]["seg_num"]
    msg_header_size = _structure_size(MSG_HEADER)
    new_pos = pos + msg_header_size + msg_size
    return new_pos


def _get_msg31_from_buf(buf, pos, dic):
    """Retrieve and unpack a MSG31 record from a buffer."""
    msg_size = dic["header"]["size"] * 2 - 4
    msg_header_size = _structure_size(MSG_HEADER)
    new_pos = pos + msg_header_size + msg_size
    mbuf = buf[pos + msg_header_size : new_pos]
    msg_31_header = _unpack_from_buf(mbuf, 0, MSG_31)
    block_pointers = [
        v for k, v in msg_31_header.items() if k.startswith("block_pointer") and v > 0
    ]
    for block_pointer in block_pointers:
        block_name, block_dic = _get_msg31_data_block(mbuf, block_pointer)
        dic[block_name] = block_dic

    dic["msg_header"] = msg_31_header
    return new_pos


def _get_msg31_data_block(buf, ptr):
    """
    Unpack a msg_31 data block into a dictionary.

    Block type is identified the same way LROSE's
    NexradRadxFile::_handleDataBlockType31() does it: by the full 4-byte
    identifier at the start of the block ("RVOL", "RELV", "RRAD" for the
    three structural blocks). Anything else is treated as a data-moment
    field block and parsed generically — we do *not* whitelist moment
    names (REF/VEL/SW/ZDR/PHI/RHO/CFP/...). LROSE reads the field's own
    3-character name purely for metadata (units/long_name), never to
    decide whether to bother parsing it, so a moment block with an
    unexpected or future name still gets its data read instead of being
    silently dropped.
    """
    block_id = buf[ptr : ptr + 4]
    block_name = block_id[1:4].decode("ascii", errors="ignore").strip()

    if block_id == b"RVOL":
        min_len = ptr + _structure_size(VOLUME_DATA_BLOCK)
        if len(buf) < min_len:
            warnings.warn(
                "RVOL block truncated (buffer too short: %d, need %d). "
                "Skipping block." % (len(buf), min_len), RuntimeWarning
            )
            return block_name, {}
        dic = _unpack_from_buf(buf, ptr, VOLUME_DATA_BLOCK)
    elif block_id == b"RELV":
        min_len = ptr + _structure_size(ELEVATION_DATA_BLOCK)
        if len(buf) < min_len:
            warnings.warn(
                "RELV block truncated (buffer too short: %d, need %d). "
                "Skipping block." % (len(buf), min_len), RuntimeWarning
            )
            return block_name, {}
        dic = _unpack_from_buf(buf, ptr, ELEVATION_DATA_BLOCK)
    elif block_id == b"RRAD":
        min_len = ptr + _structure_size(RADIAL_DATA_BLOCK)
        if len(buf) < min_len:
            warnings.warn(
                "RRAD block truncated (buffer too short: %d, need %d). "
                "Skipping block." % (len(buf), min_len), RuntimeWarning
            )
            return block_name, {}
        dic = _unpack_from_buf(buf, ptr, RADIAL_DATA_BLOCK)
    else:
        # Generic data-moment field block (REF, VEL, SW, ZDR, PHI, RHO,
        # CFP, or any other/future moment) — parsed the same way
        # regardless of name, matching LROSE's fallthrough behavior.
        # LROSE guards this with "if (len < minLen) { ...warn...; return; }"
        # (NexradRadxFile::_handleFieldType31) rather than trusting the
        # block pointer — a corrupt/out-of-range pointer here (truncated
        # message, unusual file variant) would otherwise raise
        # struct.error instead of just skipping the one bad block.
        min_len = ptr + _structure_size(GENERIC_DATA_BLOCK)
        if len(buf) < min_len:
            warnings.warn(
                "Data field block truncated (buffer too short: %d, need "
                "%d). Skipping block." % (len(buf), min_len), RuntimeWarning
            )
            return block_name, {}
        dic = _unpack_from_buf(buf, ptr, GENERIC_DATA_BLOCK)
        ngates = dic["ngates"]
        word_size = dic["word_size"]
        ptr2 = ptr + _structure_size(GENERIC_DATA_BLOCK)
        if word_size == 16:
            nbytes = ngates * 2
            if ptr2 + nbytes > len(buf):
                warnings.warn(
                    'Field "%s" data truncated (need %d bytes, have %d). '
                    "Skipping field." % (block_name, nbytes, len(buf) - ptr2),
                    RuntimeWarning
                )
                return block_name, {}
            data = np.frombuffer(buf[ptr2 : ptr2 + nbytes], ">u2")
        elif word_size == 8:
            if ptr2 + ngates > len(buf):
                warnings.warn(
                    'Field "%s" data truncated (need %d bytes, have %d). '
                    "Skipping field." % (block_name, ngates, len(buf) - ptr2),
                    RuntimeWarning
                )
                return block_name, {}
            data = np.frombuffer(buf[ptr2 : ptr2 + ngates], ">u1")
        else:
            warnings.warn(
                'Unsupported bit size: %s. Skipping field "%s".'
                % (word_size, block_name)
            )
            return block_name, {}
        dic["data"] = data
    return block_name, dic


def _get_msg1_from_buf(buf, pos, dic):
    """Retrieve and unpack a MSG1 record from a buffer."""
    msg_header_size = _structure_size(MSG_HEADER)
    msg1_header = _unpack_from_buf(buf, pos + msg_header_size, MSG_1)
    dic["msg_header"] = msg1_header

    sur_nbins = int(msg1_header["sur_nbins"])
    doppler_nbins = int(msg1_header["doppler_nbins"])

    sur_step = int(msg1_header["sur_range_step"])
    doppler_step = int(msg1_header["doppler_range_step"])

    sur_first = int(msg1_header["sur_range_first"])
    doppler_first = int(msg1_header["doppler_range_first"])
    if doppler_first > 2**15:
        doppler_first = doppler_first - 2**16

    if msg1_header["sur_pointer"]:
        offset = pos + msg_header_size + msg1_header["sur_pointer"]
        data = np.frombuffer(buf[offset : offset + sur_nbins], ">u1")
        dic["REF"] = {
            "ngates": sur_nbins,
            "gate_spacing": sur_step,
            "first_gate": sur_first,
            "data": data,
            "scale": 2.0,
            "offset": 66.0,
        }
    if msg1_header["vel_pointer"]:
        offset = pos + msg_header_size + msg1_header["vel_pointer"]
        data = np.frombuffer(buf[offset : offset + doppler_nbins], ">u1")
        dic["VEL"] = {
            "ngates": doppler_nbins,
            "gate_spacing": doppler_step,
            "first_gate": doppler_first,
            "data": data,
            "scale": 2.0,
            "offset": 129.0,
        }
        if msg1_header["doppler_resolution"] == 4:
            # 1 m/s resolution velocity, offset remains 129.
            dic["VEL"]["scale"] = 1.0
    if msg1_header["width_pointer"]:
        offset = pos + msg_header_size + msg1_header["width_pointer"]
        data = np.frombuffer(buf[offset : offset + doppler_nbins], ">u1")
        dic["SW"] = {
            "ngates": doppler_nbins,
            "gate_spacing": doppler_step,
            "first_gate": doppler_first,
            "data": data,
            "scale": 2.0,
            "offset": 129.0,
        }
    return pos + RECORD_SIZE


def _get_msg5_from_buf(buf, pos, dic):
    """Retrieve and unpack a MSG1 record from a buffer."""
    msg_header_size = _structure_size(MSG_HEADER)
    msg5_header_size = _structure_size(MSG_5)
    msg5_elev_size = _structure_size(MSG_5_ELEV)

    dic["msg5_header"] = _unpack_from_buf(buf, pos + msg_header_size, MSG_5)
    dic["msg5_header"]["supplemental_data"] = _decode_vcp_supplemental_data(
        dic["msg5_header"]["vcp_supplemental_data"]
    )
    dic["cut_parameters"] = []
    for i in range(dic["msg5_header"]["num_cuts"]):
        pos2 = pos + msg_header_size + msg5_header_size + msg5_elev_size * i
        cut = _unpack_from_buf(buf, pos2, MSG_5_ELEV)
        cut["elevation_angle_deg"] = _vcp_elevation_angle_to_deg(
            cut["elevation_angle"]
        )
        cut["supplemental_data_decoded"] = _decode_vcp_elevation_supplemental_data(
            cut["supplemental_data"]
        )
        dic["cut_parameters"].append(cut)
    return pos + RECORD_SIZE


def _vcp_elevation_angle_to_deg(raw_value):
    """
    Convert a raw MSG5 (VCP table) elevation angle code to degrees.

    The VCP elevation angle field is a 16-bit binary angular measurement
    (BAM), but the RDA only ever populates it with 12 bits of real
    precision (the ICD-documented resolution for the VCP elevation table
    is 0.043945313 deg/LSB, not the full 16-bit 360/65536 resolution used
    for per-radial elevation angles). Decoding it bit-by-bit like the RDA
    itself does, rather than a flat ``raw * 360/65536`` scale, avoids
    small systematic errors and correctly handles angles that wrap to a
    small negative value (e.g. slightly-below-horizon cuts) instead of
    reporting them as ~360 degrees.

    Parameters
    ----------
    raw_value : int
        Raw unsigned 16-bit value read from the MSG5 elevation_angle field.

    Returns
    -------
    angle_deg : float
        Elevation angle in degrees, in the range (-180, 180].
    """
    sbf = raw_value if raw_value < 0x8000 else raw_value - 0x10000  # treat as signed

    angle = 0.0
    if sbf < 0:
        angle += 180.0

    for bitmask, degrees in (
        (0x4000, 90.0),
        (0x2000, 45.0),
        (0x1000, 22.5),
        (0x0800, 11.25),
        (0x0400, 5.625),
        (0x0200, 2.8125),
        (0x0100, 1.40625),
        (0x0080, 0.70313001),
        (0x0040, 0.35156),
        (0x0020, 0.17578),
        (0x0010, 0.087889999),
        (0x0008, 0.043945),
    ):
        if raw_value & bitmask:
            angle += degrees

    if angle > 90.0:
        angle -= 360.0

    return angle


def _decode_vcp_supplemental_data(raw_value):
    """
    Decode the VCP-level supplemental data bitfield (MSG5 header).

    Bit layout (per RDA/RPG ICD, VCP supplemental data field):
      bit 0        : SAILS active
      bits 1-3     : number of SAILS sequences (0-7), valid only if bit 0 set
      bit 4        : MRLE active
      bits 5-7     : number of MRLE sequences (0-7), valid only if bit 4 set
      bit 11       : MPDA VCP

    Returns
    -------
    dic : dict
        'sails_sequences', 'mrle_sequences' (ints, 0 if not active) and
        'mpda_vcp' (bool).
    """
    sails_sequences = (raw_value >> 1) & 0x7 if raw_value & 0x1 else 0
    mrle_sequences = (raw_value >> 5) & 0x7 if raw_value & 0x10 else 0
    mpda_vcp = bool(raw_value & 0x800)
    return {
        "sails_sequences": sails_sequences,
        "mrle_sequences": mrle_sequences,
        "mpda_vcp": mpda_vcp,
    }


def _decode_vcp_elevation_supplemental_data(raw_value):
    """
    Decode the per-elevation-cut supplemental data bitfield (MSG5 elevation
    data, last 2 bytes of each cut record).

    Bit layout:
      bit 0        : SAILS cut
      bits 1-3     : SAILS sequence number for this cut, valid if bit 0 set
      bit 4        : MRLE cut
      bits 5-7     : MRLE sequence number for this cut, valid if bit 4 set
      bit 9        : MPDA cut
      bit 10       : base tilt cut (the "normal" VCP cut this extra
                     low-level scan is inserted alongside)

    Returns
    -------
    dic : dict
        'sails_sequence', 'mrle_sequence' (ints, 0 if not active),
        'mpda_cut' and 'base_tilt_cut' (bools).
    """
    sails_sequence = (raw_value >> 1) & 0x7 if raw_value & 0x1 else 0
    mrle_sequence = (raw_value >> 5) & 0x7 if raw_value & 0x10 else 0
    mpda_cut = bool(raw_value & 0x200)
    base_tilt_cut = bool(raw_value & 0x400)
    return {
        "sails_sequence": sails_sequence,
        "mrle_sequence": mrle_sequence,
        "mpda_cut": mpda_cut,
        "base_tilt_cut": base_tilt_cut,
    }


def _structure_size(structure):
    """Find the size of a structure in bytes."""
    return struct.calcsize(">" + "".join([i[1] for i in structure]))


def _unpack_from_buf(buf, pos, structure):
    """Unpack a structure from a buffer."""
    size = _structure_size(structure)
    return _unpack_structure(buf[pos : pos + size], structure)


def _unpack_structure(string, structure):
    """Unpack a structure from a string."""
    fmt = ">" + "".join([i[1] for i in structure])  # NEXRAD is big-endian
    lst = struct.unpack(fmt, string)
    return dict(zip([i[0] for i in structure], lst))


# NEXRAD Level II file structures and sizes
# The deails on these structures are documented in:
# "Interface Control Document for the Achive II/User" RPG Build 12.0
# Document Number 2620010E
# and
# "Interface Control Document for the RDA/RPG" Open Build 13.0
# Document Number 2620002M
# Tables and page number refer to those in the second document unless
# otherwise noted.
RECORD_SIZE = 2432
COMPRESSION_RECORD_SIZE = 12
CONTROL_WORD_SIZE = 4

# format of structure elements
# section 3.2.1, page 3-2
CODE1 = "B"
CODE2 = "H"
INT1 = "B"
INT2 = "H"
INT4 = "I"
REAL4 = "f"
REAL8 = "d"
SINT1 = "b"
SINT2 = "h"
SINT4 = "i"

# Figure 1 in Interface Control Document for the Archive II/User
# page 7-2
VOLUME_HEADER = (
    ("tape", "9s"),
    ("extension", "3s"),
    ("date", "I"),
    ("time", "I"),
    ("icao", "4s"),
)

# Table II Message Header Data
# page 3-7
MSG_HEADER = (
    ("size", INT2),  # size of data, no including header
    ("channels", INT1),
    ("type", INT1),
    ("seq_id", INT2),
    ("date", INT2),
    ("ms", INT4),
    ("segments", INT2),
    ("seg_num", INT2),
)

# Table XVII Digital Radar Generic Format Blocks (Message Type 31)
# pages 3-87 to 3-89
MSG_31 = (
    ("id", "4s"),  # 0-3
    ("collect_ms", INT4),  # 4-7
    ("collect_date", INT2),  # 8-9
    ("azimuth_number", INT2),  # 10-11
    ("azimuth_angle", REAL4),  # 12-15
    ("compress_flag", CODE1),  # 16
    ("spare_0", INT1),  # 17
    ("radial_length", INT2),  # 18-19
    ("azimuth_resolution", CODE1),  # 20
    ("radial_spacing", CODE1),  # 21
    ("elevation_number", INT1),  # 22
    ("cut_sector", INT1),  # 23
    ("elevation_angle", REAL4),  # 24-27
    ("radial_blanking", CODE1),  # 28
    ("azimuth_mode", SINT1),  # 29
    ("block_count", INT2),  # 30-31
    ("block_pointer_1", INT4),  # 32-35  Volume Data Constant XVII-E
    ("block_pointer_2", INT4),  # 36-39  Elevation Data Constant XVII-F
    ("block_pointer_3", INT4),  # 40-43  Radial Data Constant XVII-H
    ("block_pointer_4", INT4),  # 44-47  Moment "REF" XVII-{B/I}
    ("block_pointer_5", INT4),  # 48-51  Moment "VEL"
    ("block_pointer_6", INT4),  # 52-55  Moment "SW"
    ("block_pointer_7", INT4),  # 56-59  Moment "ZDR"
    ("block_pointer_8", INT4),  # 60-63  Moment "PHI"
    ("block_pointer_9", INT4),  # 64-67  Moment "RHO"
    ("block_pointer_10", INT4),  # Moment "CFP"
)


# Table III Digital Radar Data (Message Type 1)
# pages 3-7 to
MSG_1 = (
    ("collect_ms", INT4),  # 0-3
    ("collect_date", INT2),  # 4-5
    ("unambig_range", SINT2),  # 6-7
    ("azimuth_angle", CODE2),  # 8-9
    ("azimuth_number", INT2),  # 10-11
    ("radial_status", CODE2),  # 12-13
    ("elevation_angle", INT2),  # 14-15
    ("elevation_number", INT2),  # 16-17
    ("sur_range_first", CODE2),  # 18-19
    ("doppler_range_first", CODE2),  # 20-21
    ("sur_range_step", CODE2),  # 22-23
    ("doppler_range_step", CODE2),  # 24-25
    ("sur_nbins", INT2),  # 26-27
    ("doppler_nbins", INT2),  # 28-29
    ("cut_sector_num", INT2),  # 30-31
    ("calib_const", REAL4),  # 32-35
    ("sur_pointer", INT2),  # 36-37
    ("vel_pointer", INT2),  # 38-39
    ("width_pointer", INT2),  # 40-41
    ("doppler_resolution", CODE2),  # 42-43
    ("vcp", INT2),  # 44-45
    ("spare_1", "8s"),  # 46-53
    ("spare_2", "2s"),  # 54-55
    ("spare_3", "2s"),  # 56-57
    ("spare_4", "2s"),  # 58-59
    ("nyquist_vel", SINT2),  # 60-61
    ("atmos_attenuation", SINT2),  # 62-63
    ("threshold", SINT2),  # 64-65
    ("spot_blank_status", INT2),  # 66-67
    ("spare_5", "32s"),  # 68-99
    # 100+  reflectivity, velocity and/or spectral width data, CODE1
)

# Table XI Volume Coverage Pattern Data (Message Type 5 & 7)
# pages 3-51 to 3-54
MSG_5 = (
    ("msg_size", INT2),
    ("pattern_type", CODE2),
    ("pattern_number", INT2),
    ("num_cuts", INT2),
    ("clutter_map_group", INT2),
    ("doppler_vel_res", CODE1),  # 2: 0.5 degrees, 4: 1.0 degrees
    ("pulse_width", CODE1),  # 2: short, 4: long
    ("vcp_sequencing", INT2),  # SAILS/MRLE cut-count sequencing info
    ("vcp_supplemental_data", INT2),  # SAILS/MRLE/MPDA bitfield, see
    # _decode_vcp_supplemental_data()
    ("spare", "6s"),
)

MSG_5_ELEV = (
    ("elevation_angle", CODE2),  # scaled by 360/65536 for value in degrees.
    ("channel_config", CODE1),
    ("waveform_type", CODE1),
    ("super_resolution", CODE1),
    ("prf_number", INT1),
    ("prf_pulse_count", INT2),
    ("azimuth_rate", CODE2),
    ("ref_thresh", SINT2),
    ("vel_thresh", SINT2),
    ("sw_thresh", SINT2),
    ("zdr_thres", SINT2),
    ("phi_thres", SINT2),
    ("rho_thres", SINT2),
    ("edge_angle_1", CODE2),
    ("dop_prf_num_1", INT2),
    ("dop_prf_pulse_count_1", INT2),
    ("spare_1", "2s"),
    ("edge_angle_2", CODE2),
    ("dop_prf_num_2", INT2),
    ("dop_prf_pulse_count_2", INT2),
    ("spare_2", "2s"),
    ("edge_angle_3", CODE2),
    ("dop_prf_num_3", INT2),
    ("dop_prf_pulse_count_3", INT2),
    ("supplemental_data", INT2),  # SAILS/MRLE/MPDA/base-tilt bitfield, see
    # _decode_vcp_elevation_supplemental_data()
)

# Table XVII-B Data Block (Descriptor of Generic Data Moment Type)
# pages 3-90 and 3-91
GENERIC_DATA_BLOCK = (
    ("block_type", "1s"),
    ("data_name", "3s"),  # VEL, REF, SW, RHO, PHI, ZDR
    ("reserved", INT4),
    ("ngates", INT2),
    ("first_gate", SINT2),
    ("gate_spacing", SINT2),
    ("thresh", SINT2),
    ("snr_thres", SINT2),
    ("flags", CODE1),
    ("word_size", INT1),
    ("scale", REAL4),
    ("offset", REAL4),
    # then data
)

# Table XVII-E Data Block (Volume Data Constant Type)
# page 3-92
VOLUME_DATA_BLOCK = (
    ("block_type", "1s"),
    ("data_name", "3s"),
    ("lrtup", INT2),
    ("version_major", INT1),
    ("version_minor", INT1),
    ("lat", REAL4),
    ("lon", REAL4),
    ("height", SINT2),
    ("feedhorn_height", INT2),
    ("refl_calib", REAL4),
    ("power_h", REAL4),
    ("power_v", REAL4),
    ("diff_refl_calib", REAL4),
    ("init_phase", REAL4),
    ("vcp", INT2),
    ("spare", "2s"),
)

# Table XVII-F Data Block (Elevation Data Constant Type)
# page 3-93
ELEVATION_DATA_BLOCK = (
    ("block_type", "1s"),
    ("data_name", "3s"),
    ("lrtup", INT2),
    ("atmos", SINT2),
    ("refl_calib", REAL4),
)

# Table XVII-H Data Block (Radial Data Constant Type)
# pages 3-93
RADIAL_DATA_BLOCK = (
    ("block_type", "1s"),
    ("data_name", "3s"),
    ("lrtup", INT2),
    ("unambig_range", SINT2),
    ("noise_h", REAL4),
    ("noise_v", REAL4),
    ("nyquist_vel", SINT2),
    ("spare", "2s"),
)