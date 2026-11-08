import bz2
import os
import re

import numpy as np

class FileFormatError(Exception):
    pass

def _make_iq_lut():
    codes = np.arange(2 ** 16, dtype=np.int64)
    exp = (codes >> 12) & 0xF
    mant = codes & 0x7FF
    sign = (codes >> 11) & 1

    out = np.empty(2 ** 16, dtype=np.float32)

    # exp == 0 branch (denormalized)
    m0 = exp == 0
    mant_a = mant.copy()
    sign_mask_a = m0 & (sign == 1)
    mant_a[sign_mask_a] |= 0xFFFFF800
    neg_a = m0 & ((mant_a & 0x80000000) != 0)
    mant_a[neg_a] = -(0x100000000 - mant_a[neg_a])
    out[m0] = mant_a[m0].astype(np.float32) * np.float32(2.0 ** -25)

    # exp != 0 branch (normalized)
    m1 = ~m0
    mant_b = mant.copy()
    sign_mask_b = m1 & (sign == 1)
    mant_b[sign_mask_b] |= 0xFFFFF000
    neg_b = sign_mask_b & ((mant_b & 0x80000000) != 0)
    mant_b[neg_b] = -(0x100000000 - mant_b[neg_b])
    nosign_mask_b = m1 & (sign == 0)
    mant_b[nosign_mask_b] |= 0x00000800

    out[m1] = (mant_b[m1].astype(np.float32)
               * np.power(2.0, exp[m1]).astype(np.float32)
               / np.float32(3.3554432e7))

    return out


_IQ_LUT = _make_iq_lut()


def unpack_iq_block(raw_bytes: bytes) -> np.ndarray:
    """Unpack a single pulse's raw I/Q bytes. Kept for API compatibility /
    one-off use; the bulk file reader below uses _batch_unpack_iq instead,
    which does one contiguous copy + one LUT gather for an entire file
    rather than one small gather per pulse."""
    u16 = np.frombuffer(raw_bytes, dtype='<u2')
    return _IQ_LUT[u16].reshape(-1, 2)


def _batch_unpack_iq(data: bytes, spans):
    """Unpack many (start_byte, nbytes) spans out of `data` in one shot.

    Instead of calling unpack_iq_block() once per pulse (many small
    allocations + many small LUT gathers), this copies every span's raw
    bytes into a single contiguous uint8 buffer via np.ascontiguousarray,
    then does exactly one LUT gather over the whole thing. Returns a list
    of (n_gates, 2) float32 arrays, one per span, in input order.
    """
    lengths = [n for _, n in spans]
    total = sum(lengths)
    if total == 0:
        return [np.empty((0, 2), dtype=np.float32) for _ in spans]

    # Single contiguous allocation covering every pulse's IQ payload.
    buf = np.empty(total, dtype=np.uint8)
    off = 0
    for start, n in spans:
        if n:
            buf[off:off + n] = np.frombuffer(data, dtype=np.uint8, count=n, offset=start)
            off += n
    buf = np.ascontiguousarray(buf)

    # One LUT gather for the entire file's worth of I/Q samples.
    flat = _IQ_LUT[buf.view('<u2')]

    out = []
    off = 0
    for n in lengths:
        k = n // 4           # number of (I, Q) pairs in this span
        span_len = 2 * k     # number of u16 samples consumed
        out.append(flat[off:off + span_len].reshape(k, 2))
        off += span_len
    return out

def _read_until(data: bytes, pos: int, prefix: bytes):
    while pos < len(data):
        nl = data.find(b'\n', pos)
        end = nl + 1 if nl != -1 else len(data)
        line = data[pos:end]
        if line.startswith(prefix):
            return line, end
        pos = end
    return None, pos


def _next_line(buf: bytes, pos: int):
    nl = buf.find(b'\n', pos)
    if nl == -1:
        return None, len(buf)
    end = nl + 1
    line = buf[pos:end].decode('ascii', errors='ignore')
    return line, end


def _parse_scalar(line: str):
    name, val = line.strip().split('=', 1)
    if re.fullmatch(r'[-+]?\d+', val):
        return name, int(val)
    try:
        return name, float(val)
    except ValueError:
        return name, val


def _parse_array(line: str):
    name, rest = line.strip().split('=', 1)
    vals = []
    for p in rest.split():
        try:
            vals.append(float(p))
        except ValueError:
            vals.append(p)
    return name, vals


def _parse_block(buf: bytes, pos: int, end_markers):
    line, pos = _read_until(buf, pos, b'rvp')
    if line is None:
        return None, pos

    fields = {}
    while pos < len(buf):
        ln, pos = _next_line(buf, pos)
        if ln is None:
            break
        if any(marker in ln for marker in end_markers):
            break
        if '=' not in ln:
            continue
        if ' ' in ln.strip().split('=', 1)[1]:
            name, vals = _parse_array(ln)
            fields[name] = vals
        else:
            name, val = _parse_scalar(ln)
            fields[name] = val
    return fields, pos


class Pulse:
    __slots__ = ('hdr', 'iq_h', 'iq_v', 'dual_pol')

    def __init__(self, hdr, iq_h, iq_v, dual_pol: bool = False):
        self.hdr = hdr
        self.iq_h = iq_h
        self.iq_v = iq_v
        self.dual_pol = dual_pol


class vaisalafile:
    def __init__(self, filename, verbose=True):
        self.filename = filename
        self.header = {}
        self.pulses = []

        if verbose:
            print(f"Filename: {filename}")

        data = self._load_bytes(filename, verbose=verbose)
        self._read_pulse_info(data, verbose=verbose)
        self._read_pulses(data, verbose=verbose)

    # ---- internal helpers -------------------------------------------------
    @staticmethod
    def _load_bytes(filename, verbose=True):
        if filename.lower().endswith('.bz2'):
            with bz2.open(filename, 'rb') as f:
                raw = f.read()
            if verbose:
                print(f"[info] bz2 decompressed {os.path.basename(filename)} "
                      f"({len(raw) / 1_048_576:.1f} MB uncompressed)")
        else:
            with open(filename, 'rb') as f:
                raw = f.read()
        return raw

    def _read_pulse_info(self, data, verbose=True):
        info, pos = _parse_block(
            data, 0, end_markers=('rvp8PulseInfo end', 'rvptsPulseInfo end'))
        if info is None:
            raise FileFormatError('rvp8PulseInfo / rvptsPulseInfo not found')
        self.header = info
        self._data_pos = pos
        if verbose:
            print(f"Parsed {len(self.header)} pulse-info fields.")

    def _read_pulses(self, data, verbose=True):
        pos = self._data_pos
        headers = []
        h_spans = []   # (start_byte, nbytes) per pulse
        v_spans = []
        dual_pol_flags = []

        # Pass 1: walk the text headers and record where each pulse's raw
        # I/Q bytes live, without touching/copying the I/Q payloads yet.
        while pos < len(data):
            hdr, pos = _parse_block(
                data, pos, end_markers=('rvp8PulseHdr end', 'rvptsPulseHdr end'))
            if hdr is None:
                break

            numvec = hdr.get('iNumVecs', 0)
            numvec = int(numvec[0]) if isinstance(numvec, list) else int(numvec)

            num_pol = hdr.get('iVIQPerBin', 1)
            num_pol = int(num_pol[0]) if isinstance(num_pol, list) else int(num_pol)
            dual_pol = num_pol >= 2

            nbytes = numvec * 4

            if pos + nbytes > len(data):
                raise FileFormatError('Unexpected end of file inside H IQ payload')
            h_spans.append((pos, nbytes))
            pos += nbytes

            if dual_pol:
                if pos + nbytes > len(data):
                    raise FileFormatError('Unexpected end of file inside V IQ payload')
                v_spans.append((pos, nbytes))
                pos += nbytes
            else:
                v_spans.append((pos, 0))

            headers.append(hdr)
            dual_pol_flags.append(dual_pol)

        # Pass 2: one contiguous copy + one LUT gather for ALL pulses'
        # H data, and likewise for all V data, instead of one small
        # allocation + gather per pulse.
        iq_h_all = _batch_unpack_iq(data, h_spans)
        iq_v_all = _batch_unpack_iq(data, v_spans)

        self.pulses = [
            Pulse(headers[i], iq_h_all[i], iq_v_all[i], dual_pol=dual_pol_flags[i])
            for i in range(len(headers))
        ]
        if verbose:
            print(f"Number of pulses: {len(self.pulses)}")

    def azArray(self):
        return np.array(
            [p.hdr.get('iAz', 0) * (360.0 / 65536.0) for p in self.pulses])

    def elArray(self):
        return np.array(
            [p.hdr.get('iEl', 0) * (360.0 / 65536.0) for p in self.pulses])

    def isDualPol(self):
        return any(p.dual_pol for p in self.pulses)

    @staticmethod
    def _rvp_angle_to_deg(value: float) -> float:
        """Convert an RVP8 32-bit binary angle (as found in the INU/GPS
        pulse-header fields) to degrees. Same convention used for iAz/iEl,
        just full 32-bit resolution instead of 16-bit."""
        return value * (180.0 / (2 ** 31))

    def geolocation(self):
        """Return (lat_deg, lon_deg, alt_m) read from the first pulse's
        INU/GPS header fields: `inu.iLatitude`, `inu.iLongitude`,
        `inu.iHeight`.

        Not every Vaisala IQ dump carries INU data (rvp8PulseInfo itself
        never does -- it's a per-pulse header field only present on units
        with GPS/INU logging enabled), so this returns (None, None, None)
        if those fields aren't present. Callers should fall back to a
        known site lat/lon/alt in that case, e.g. via
        IQMoments(..., lat=..., lon=..., alt=...).
        """
        if not self.pulses:
            return None, None, None

        hdr = self.pulses[0].hdr
        lat_raw = hdr.get('inu.iLatitude')
        lon_raw = hdr.get('inu.iLongitude')
        if lat_raw is None or lon_raw is None:
            return None, None, None

        lat_deg = self._rvp_angle_to_deg(float(lat_raw))
        lon_deg = self._rvp_angle_to_deg(float(lon_raw))
        # CONUS radars fall in the western hemisphere; wrap into [-180, 180]
        # to match how the reference script normalizes this value.
        if lon_deg > 180:
            lon_deg -= 360.0

        alt_raw = hdr.get('inu.iHeight')
        alt_m = float(alt_raw) if alt_raw is not None else None

        return lat_deg, lon_deg, alt_m

    def pulseToDict(self, idx: int):
        return self.pulses[idx].hdr