import os, re, sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from clutter_filter import GroundClutterFilter

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


if _HAVE_NUMBA:
    @njit(cache=True, parallel=True)
    def _autocorrelate_ray_numba(Xh_re, Xh_im, Xv_re, Xv_im, N):
        n_gates, n_pulses = Xh_re.shape
        Rh = np.empty((n_gates, N + 1), dtype=np.complex128)
        Rv = np.empty((n_gates, N + 1), dtype=np.complex128)
        C = np.empty((n_gates, 2 * N + 1), dtype=np.complex128)
        nan = np.nan

        for g in prange(n_gates):
            # --- Rh, Rv: lags 0..N ---
            for lag in range(N + 1):
                cnt = n_pulses - lag
                if cnt <= 0:
                    Rh[g, lag] = complex(nan, nan)
                    Rv[g, lag] = complex(nan, nan)
                    continue
                hsre = hsim = 0.0
                vsre = vsim = 0.0
                for p in range(cnt):
                    q = p + lag
                    # X[q] * conj(X[p]) = (a+bi)(c-di) = (ac+bd) + (bc-ad)i
                    a, b = Xh_re[g, q], Xh_im[g, q]
                    c, d = Xh_re[g, p], Xh_im[g, p]
                    hsre += a * c + b * d
                    hsim += b * c - a * d
                    a, b = Xv_re[g, q], Xv_im[g, q]
                    c, d = Xv_re[g, p], Xv_im[g, p]
                    vsre += a * c + b * d
                    vsim += b * c - a * d
                Rh[g, lag] = complex(hsre / cnt, hsim / cnt)
                Rv[g, lag] = complex(vsre / cnt, vsim / cnt)

            # --- C: cross-correlation, lags -N..N ---
            for k in range(2 * N + 1):
                lag = k - N
                if lag < 0:
                    cnt = n_pulses + lag
                    if cnt <= 0:
                        C[g, k] = complex(nan, nan)
                        continue
                    csre = csim = 0.0
                    for p in range(cnt):
                        q = p - lag
                        # Xh[p] * conj(Xv[q])
                        a, b = Xh_re[g, p], Xh_im[g, p]
                        c, d = Xv_re[g, q], Xv_im[g, q]
                        csre += a * c + b * d
                        csim += b * c - a * d
                    C[g, k] = complex(csre / cnt, csim / cnt)
                else:
                    cnt = n_pulses - lag
                    if cnt <= 0:
                        C[g, k] = complex(nan, nan)
                        continue
                    csre = csim = 0.0
                    for p in range(cnt):
                        q = p + lag
                        # Xh[q] * conj(Xv[p])
                        a, b = Xh_re[g, q], Xh_im[g, q]
                        c, d = Xv_re[g, p], Xv_im[g, p]
                        csre += a * c + b * d
                        csim += b * c - a * d
                    C[g, k] = complex(csre / cnt, csim / cnt)

        return Rh, Rv, C


class PulsePairMoments:
    def __init__(self, wavelength_m: float, prt_s: float, n_lags: int = 1):
        self.wavelength_m = wavelength_m
        self.prt_s = prt_s
        self.n_lags = n_lags
        self.va = 0.25 * wavelength_m / prt_s 

    def autocorrelate_ray(self, Xh: np.ndarray, Xv: np.ndarray):
        N = self.n_lags

        if _HAVE_NUMBA:
            # .real/.imag on a complex128 array are strided (non-contiguous)
            # views, so copy each into a contiguous float64 buffer once;
            # the numba kernel then fuses the whole lag loop +
            # conjugate-multiply + mean into one pass per gate with no
            # temporary product arrays (unlike the numpy path below, which
            # allocates a full temp array per lag).
            return _autocorrelate_ray_numba(
                np.ascontiguousarray(Xh.real), np.ascontiguousarray(Xh.imag),
                np.ascontiguousarray(Xv.real), np.ascontiguousarray(Xv.imag),
                N,
            )

        n_gates = Xh.shape[0]
        Rh = np.empty((n_gates, N + 1), dtype=np.complex128)
        Rv = np.empty((n_gates, N + 1), dtype=np.complex128)
        C = np.empty((n_gates, 2 * N + 1), dtype=np.complex128)

        for i in range(N + 1):
            end = None if i == 0 else -i
            Rh[:, i] = np.mean(Xh[:, i:] * np.conjugate(Xh[:, :end]), axis=1)
            Rv[:, i] = np.mean(Xv[:, i:] * np.conjugate(Xv[:, :end]), axis=1)

        for i in range(-N, N + 1):
            if i < 0:
                C[:, i + N] = np.mean(Xh[:, :i] * np.conjugate(Xv[:, -i:]), axis=1)
            else:
                end = None if i == 0 else -i
                C[:, i + N] = np.mean(Xh[:, i:] * np.conjugate(Xv[:, :end]), axis=1)

        return Rh, Rv, C

    def autocorrelate_volume(self, ray_iq_pairs):
        Rh_list, Rv_list, C_list = [], [], []
        for Xh, Xv in ray_iq_pairs:
            Rh, Rv, C = self.autocorrelate_ray(Xh, Xv)
            Rh_list.append(Rh)
            Rv_list.append(Rv)
            C_list.append(C)
        return (np.stack(Rh_list, axis=1),
                np.stack(Rv_list, axis=1),
                np.stack(C_list, axis=1))

    def moments_from_covariances(self, Rh, Rv, C, range_km, zcal=0.0, dcal=0.0,
                                  pcal=0.0, noise_h_dB=0.0, noise_v_dB=0.0,
                                  noiseCorrection=True,
                                  elev_deg=None, wavelength_cm=None,
                                  attenCorrection=True, dbzMethod='pulsepair',
                                  phidpSign=1):
        """
        noiseCorrection : if True (default, matches prior behavior), the
            lag-0 H/V power (Sh/Sv) used for DBZ, WIDTH, ZDR, RHOHV, and
            SNR all has the noise floor subtracted first, and gates whose
            noise-subtracted power goes negative are masked to NaN. If
            False, every one of those products is computed from the raw
            lag-0 power instead. This one flag controls all of them
            consistently, not just WIDTH.
        dbzMethod : 'pulsepair' (default) computes DBZ the original way
            this class always has: 10*log10(Sh * range_km**2) + zcal. This
            is what IQMoments uses for rkcfile input, and matches the
            raxpolIQ.py reference exactly (no attenuation term).
            'snr' instead reproduces iqtest_clutter.py's
            compute_dbz_from_power for Vaisala files: it divides the
            noise-subtracted power by the noise floor first (an actual
            SNR ratio, not just noise-subtracted signal) before taking
            the log, and adds the elevation/range attenuation-table term
            when attenCorrection is on:
                DBZ = 10*log10(Sh / noise_h_lin) + zcal
                      + 20*log10(range_km) + atten_correction
            IQMoments picks this automatically for vaisalafile input --
            you shouldn't normally need to set it directly.
        elev_deg, wavelength_cm : per-ray elevation (degrees) and radar
            wavelength (cm), used only by dbzMethod='snr' when
            attenCorrection is on. Ignored by dbzMethod='pulsepair'.
        attenCorrection : only applies when dbzMethod='snr'; adds the
            same gaseous-attenuation correction table
            iqtest_clutter.py's compute_dbz_from_power looks up
            (build_atten_table). Requires elev_deg and wavelength_cm.
        phidpSign : +1 (default) or -1, multiplies the raw angle(C_lag0)
            before scaling to degrees. IQMoments passes -1 automatically
            for vaisalafile input -- see IQMoments' docstring.
        """
        N = self.n_lags
        n_gates, n_rays = Rh.shape[0], Rh.shape[1]
        range_km = np.asarray(range_km).reshape(-1, 1)

        noise_h_lin = np.power(10, 0.1 * np.broadcast_to(noise_h_dB, (n_rays,)))
        noise_v_lin = np.power(10, 0.1 * np.broadcast_to(noise_v_dB, (n_rays,)))

        Sh_raw = np.real(Rh[:, :, 0])
        Sv_raw = np.real(Rv[:, :, 0])

        if noiseCorrection:
            Sh = Sh_raw - noise_h_lin
            Sv = Sv_raw - noise_v_lin
            Sh = np.where(Sh < 0, np.nan, Sh)
            Sv = np.where(Sv < 0, np.nan, Sv)
        else:
            Sh = Sh_raw
            Sv = Sv_raw

        if dbzMethod == 'snr':
            with np.errstate(invalid='ignore', divide='ignore'):
                DBZ = (10 * np.log10(Sh / noise_h_lin) + zcal
                       + 10 * np.log10(range_km ** 2))
            if attenCorrection and elev_deg is not None and wavelength_cm is not None:
                DBZ = DBZ + _attenuation_correction_dB(range_km, elev_deg, wavelength_cm)
        elif dbzMethod == 'pulsepair':
            DBZ = 10 * np.log10(Sh * range_km ** 2) + zcal
        else:
            raise ValueError(f"dbzMethod must be 'pulsepair' or 'snr', got {dbzMethod!r}.")

        VEL = self.va / np.pi * np.angle(Rh[:, :, 1])

        with np.errstate(invalid='ignore', divide='ignore'):
            WIDTH = (np.sqrt(2) * self.va / np.pi
                     * np.sqrt(np.abs(np.log(Sh / np.abs(Rh[:, :, 1])))))
        WIDTH = WIDTH.astype(np.float64)

        ZDR = 10 * np.log10(Sh / Sv) + dcal

        PHIDP = phidpSign * np.angle(C[:, :, 0 + N]) / np.pi * 180 + pcal
        PHIDP = np.where(PHIDP < -180, PHIDP + 360, PHIDP)
        PHIDP = np.where(PHIDP > 180, PHIDP - 360, PHIDP)

        RHOHV = None
        if N == 1:
            RHOHV = np.abs(C[:, :, 0 + N]) / np.sqrt(Sh * Sv)

        SNRHC = 10 * np.log10(Sh / noise_h_lin)
        SNRVC = 10 * np.log10(Sv / noise_v_lin)

        return {
            'DBZ': DBZ, 'VEL': VEL, 'WIDTH': WIDTH, 'ZDR': ZDR,
            'PHIDP': PHIDP, 'RHOHV': RHOHV, 'SNRHC': SNRHC, 'SNRVC': SNRVC,
        }

    def process_volume(self, ray_iq_pairs, range_km, gstep: int = 1, **cal_kwargs):
        Rh, Rv, C = self.autocorrelate_volume(ray_iq_pairs)
        if gstep > 1:
            Rh = self.average_by_gate_step(Rh, gstep)
            Rv = self.average_by_gate_step(Rv, gstep)
            C = self.average_by_gate_step(C, gstep)
        return self.moments_from_covariances(Rh, Rv, C, range_km, **cal_kwargs)

    @staticmethod
    def average_by_gate_step(arr: np.ndarray, gstep: int):
        if gstep <= 1:
            return arr
        m, n, o = arr.shape
        full_groups = m // gstep
        if full_groups > 0:
            main = arr[:full_groups * gstep].reshape((full_groups, gstep, n, o)).mean(axis=1)
        else:
            main = np.array([]).reshape((0, n, o))
        if m % gstep != 0:
            remainder = arr[full_groups * gstep:].mean(axis=0, keepdims=True)
            return np.concatenate((main, remainder), axis=0)
        return main
    
def _in_degree_range(val, low, high):
    low = low if low > 0 else low + 360
    high = high if high < 360 else high - 360
    return (val > low and val < high
            if low < high
            else val < (high + 360)
                if val > low
                else (val + 360) < (high + 360))


if _HAVE_NUMBA:
    @njit(cache=True)
    def _bin_boundaries_numba(az, az_unique, half_swath):
        """For each unique (discretized) azimuth center, find the first and
        last pulse index whose *raw* azimuth falls within
        [center - half_swath, center + half_swath] (wrapped at 0/360).

        Replaces np.tile(az, (n_unique, n_pulses)) [a full n_unique x
        n_pulses temp array] + np.vectorize(...) [a disguised Python loop,
        not real vectorization] with a single allocation-free pass.
        """
        n_unique = az_unique.shape[0]
        n_pulses = az.shape[0]
        boundaries = np.empty((n_unique, 2), dtype=np.int32)

        for u in range(n_unique):
            low = az_unique[u] - half_swath
            high = az_unique[u] + half_swath
            low = low if low > 0.0 else low + 360.0
            high = high if high < 360.0 else high - 360.0

            first = -1
            last = -1
            for p in range(n_pulses):
                v = az[p]
                if low < high:
                    hit = low < v < high
                else:
                    hit = (v < (high + 360.0)) if v > low else ((v + 360.0) < (high + 360.0))
                if hit:
                    if first == -1:
                        first = p
                    last = p

            boundaries[u, 0] = first
            boundaries[u, 1] = last

        return boundaries


class AzimuthBinner:
    def __init__(self, azimuth_beamwidth_deg: float = 1.0, beam_overlap_deg: float = 0.0):
        self.az_spacing = azimuth_beamwidth_deg
        self.az_swath = azimuth_beamwidth_deg + 2 * beam_overlap_deg
 
    def bin_azimuths(self, az: np.ndarray):
        az_discrete = np.rint(az / self.az_spacing) * self.az_spacing
        az_discrete[az_discrete == 360] = 0
 
        az_unique = [az_discrete[0]]
        curr_az = az_discrete[0]
        for azimuth in az_discrete:
            if azimuth != curr_az:
                az_unique.append(azimuth)
                curr_az = azimuth
        az_unique = np.array(az_unique, dtype=np.float32)

        if _HAVE_NUMBA:
            pulse_boundaries = _bin_boundaries_numba(
                np.ascontiguousarray(az, dtype=np.float64),
                np.ascontiguousarray(az_unique, dtype=np.float64),
                0.5 * self.az_swath,
            ).astype(np.int32)
            if (pulse_boundaries == -1).any():
                bad = int(np.where(pulse_boundaries[:, 0] == -1)[0][0])
                raise IndexError(
                    f"No pulses found within the azimuth swath around "
                    f"{az_unique[bad]!r} degrees.")
        else:
            rep_az = np.tile(az, (len(az_unique), 1))
            az_col = np.array([az_unique]).T

            is_in_range = np.vectorize(_in_degree_range)
            az_bool = is_in_range(rep_az, az_col - 0.5 * self.az_swath, az_col + 0.5 * self.az_swath)

            pulse_boundaries = []
            for swath in az_bool:
                hits = np.where(swath)[0]
                pulse_boundaries.append([hits[0], hits[-1]])
            pulse_boundaries = np.array(pulse_boundaries, dtype=np.int32)
 
        middle_pulses = np.rint(np.mean(pulse_boundaries, axis=1)).astype(np.int32)
 
        return az_unique, pulse_boundaries, middle_pulses
 
    @staticmethod
    def extract_rays(Xh_full: np.ndarray, Xv_full: np.ndarray, pulse_boundaries: np.ndarray):
        rays = []
        for start, end in pulse_boundaries:
            rays.append((Xh_full[:, start:end + 1], Xv_full[:, start:end + 1]))
        return rays
 
    @staticmethod
    def is_likely_rhi(az: np.ndarray, el: np.ndarray, el_var_threshold_deg: int = 5) -> bool:
        n_el = len(np.unique(np.rint(el)))
        n_az = len(np.unique(np.rint(az)))
        return n_el > n_az and n_el > el_var_threshold_deg
 

# =============================================================================
# Generic, format-agnostic moment computation
# =============================================================================
#
# IQMoments takes an already-loaded I/Q file object -- from radarkitIQ.rkcfile,
# vaisalaIQ.vaisalafile, or any other loader you register -- and produces
# ray-binned pulse-pair moments, using the same call signature regardless of
# which file format was loaded:
#
#     from radarkitIQ import rkcfile
#     from calcIQ import IQMoments
#     m = IQMoments(rkcfile('myfile.rkc'), azDeg=0.1, flipVel=True)
#
#     from vaisalaIQ import vaisalafile
#     from calcIQ import IQMoments
#     m = IQMoments(vaisalafile('myfile.iq.bz2'), azDeg=1.0)
#
# Support for a format is provided by a small "adapter" function that pulls a
# canonical set of arrays/scalars out of the file object. Adapters for
# rkcfile and vaisalafile are registered below. To support another format
# without modifying this file, write a similarly-shaped function and call
# register_iq_adapter('YourClassName', yourFunc) before constructing
# IQMoments.

_IQ_ADAPTERS = {}


def _angular_velocity_deg_s(angle_deg: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    """Per-pulse angular velocity (deg/s), estimated by finite difference
    against the neighboring pulse(s). Used (when an explicit velocity
    threshold is given) to flag antenna-transition pulses for file formats
    that don't already report velocity directly.

    Wraps angle differences to the shortest path (so 359 -> 1 deg reads as
    a 2 deg step, not 358), and is tolerant of bad/duplicate timestamps --
    a non-finite or non-positive dt contributes 0 rather than raising or
    producing inf, so one bad time sample doesn't corrupt the whole array.
    Each interior pulse takes the larger of its left/right neighbor step,
    which is the more conservative (safer) choice for transition flagging.
    """
    n = len(angle_deg)
    vel = np.zeros(n, dtype=np.float64)
    if n < 2:
        return vel
    angle_deg = angle_deg.astype(np.float64)
    dt = np.diff(times_s.astype(np.float64))
    dang = (np.diff(angle_deg) + 180.0) % 360.0 - 180.0
    with np.errstate(invalid='ignore', divide='ignore'):
        step_vel = np.abs(dang) / dt
    step_vel[~np.isfinite(step_vel)] = 0.0
    vel[:-1] = step_vel
    vel[1:] = np.maximum(vel[1:], step_vel)
    return vel


def _elevation_mode_deg(el: np.ndarray, decimals: int = 2) -> float:
    """The single most common elevation angle in the pulse stream, rounded
    to `decimals` places first so encoder jitter on an otherwise-steady
    tilt doesn't fragment one true mode into many near-identical floats
    that individually lose to noise. This is the "no threshold given"
    fallback for transition detection: a normal PPI/RHI sweep spends the
    overwhelming majority of its pulses sitting at one fixed elevation
    (the sweep's target angle), so whatever value pulses pile up on the
    most *is* that angle -- and anything else is, by construction, not a
    normal on-tilt pulse.
    """
    el_rounded = np.round(el.astype(np.float64), decimals)
    values, counts = np.unique(el_rounded, return_counts=True)
    return float(values[np.argmax(counts)])


def register_iq_adapter(class_name: str, extractor):
    """Register an extractor function so IQMoments can support a new I/Q
    file class without any change to IQMoments itself.

    `extractor(fileobj, verbose=True)` must return a dict with these keys:
        Xh_full, Xv_full   : complex128 ndarray, shape (n_gates, n_pulses)
        dual_pol           : bool
        az, el             : ndarray, shape (n_pulses,), degrees
        times              : ndarray, shape (n_pulses,), unix seconds (float64)
        wavelength_m       : float or None
        prt_s              : float or None
        pulse_width_s      : float or None
        gate_size_m        : float or None
        lat, lon, alt      : float or None
        instrument_name    : str or None
        zcal, dcal, pcal   : float or None (dB, dB, degrees)
        noise_h_dB         : float or None
        noise_v_dB         : float or None
    Any None values can be filled in by the caller via IQMoments(...)
    keyword overrides.
    """
    _IQ_ADAPTERS[class_name] = extractor


def _header_get(header: dict, *candidates, required=True, default=None):
    """Look up the first matching key out of several candidate spellings
    in a parsed text header dict, unwrapping single-element lists."""
    for c in candidates:
        if c in header:
            val = header[c]
            if isinstance(val, list) and len(val) == 1:
                val = val[0]
            return val
    if required:
        available = sorted(header.keys())
        raise KeyError(
            f"None of {candidates} found in header. Available keys "
            f"(first 25): {available[:25]}. Pass the value directly via "
            "the matching IQMoments(...) keyword override instead.")
    return default


def _decode_range_mask(header):
    """Reproduce iqtest_clutter.py's decode_range_mask: iRangeMask is a
    bitmask (one bit per fRangeMaskRes-sized range bin) marking which bins
    actually have data. fRangeMaskRes is the bin size of that mask, not
    necessarily the spacing between *populated* bins -- if the mask is
    decimated (every Nth bin set), the true gate spacing is N times
    fRangeMaskRes. Returns (start_range_m, spacing_m) or (None, None) if
    the mask/res aren't present or usable.
    """
    mask = header.get('iRangeMask')
    res = _header_get(header, 'fRangeMaskRes', required=False)
    if mask is None or res is None:
        return None, None
    if not isinstance(mask, list):
        mask = [mask]
    res = float(res)

    start_bin = None
    end_bin = None
    count = 0
    for ii, m in enumerate(mask):
        m = int(m)
        if m > 0:
            for bit in range(16):
                if (m >> bit) & 1:
                    b = ii * 16 + bit
                    if start_bin is None:
                        start_bin = b
                    end_bin = b
                    count += 1

    if start_bin is None or count == 0:
        return None, None

    start_range_m = start_bin * res
    if count > 1:
        spacing_m = (end_bin * res - start_range_m) / (count - 1)
    else:
        spacing_m = res
    return start_range_m, spacing_m


_ATTEN_TABLE_CACHE = {}


def _build_atten_table(wavelength_cm):
    """Direct port of iqtest_clutter.py's build_atten_table: an empirical
    gaseous-attenuation table indexed by [elevation_bin(0.1 deg steps),
    range_bin(1 km steps)], scaled by a wavelength-dependent factor."""
    if wavelength_cm in _ATTEN_TABLE_CACHE:
        return _ATTEN_TABLE_CACHE[wavelength_cm]
    N_ELEV, N_RANGE = 200, 500
    table = np.zeros((N_ELEV, N_RANGE), dtype=np.float32)
    wc = (1.5 - 0.3 * (wavelength_cm - 3.3) / 1.7) if wavelength_cm < 5 \
        else (1.2 - 0.2 * (wavelength_cm - 5) / 5)
    elev = np.arange(N_ELEV) * 0.1
    rm = np.arange(N_RANGE)
    for ie in range(N_ELEV):
        table[ie, :] = ((0.4 + 3.45 * np.exp(-elev[ie] / 1.8))
                        * (1 - np.exp(-rm / (27.8 + 154 * np.exp(-elev[ie] / 2.2))))) * wc
    _ATTEN_TABLE_CACHE[wavelength_cm] = table
    return table


def _attenuation_correction_dB(range_km, elev_deg, wavelength_cm):
    """Vectorized version of the ie/ir table lookup in iqtest_clutter.py's
    compute_dbz_from_power. range_km: (n_gates, 1) array (km). elev_deg:
    (n_rays,) array (degrees). Returns an (n_gates, n_rays) dB array to
    add to DBZ, matching `ac` in the reference implementation exactly
    (same 0.1-deg elevation bins, same 1-km range bins, same clipping)."""
    table = _build_atten_table(wavelength_cm)
    ie = np.clip(np.round(np.asarray(elev_deg, dtype=np.float64) / 0.1).astype(int), 0, 199)
    ir = np.clip(np.round(range_km.reshape(-1)).astype(int), 0, 499)
    return table[ie[np.newaxis, :], ir[:, np.newaxis]].astype(np.float64)


def _first_nonzero(arr, default=0.0):
    a = np.atleast_1d(np.asarray(arr, dtype=np.float64))
    nz = a[a > 0]
    return float(nz[0]) if nz.size else float(default)


def _noise_to_dB(noise_linear):
    noise_linear = float(noise_linear)
    return float(10.0 * np.log10(noise_linear)) if noise_linear > 0 else 0.0


def _average_range_km(range_km_full, gstep):
    if gstep <= 1:
        return range_km_full
    avg = PulsePairMoments.average_by_gate_step(range_km_full.reshape(-1, 1, 1), gstep)
    return avg.reshape(-1)


# =============================================================================
# Optional clutter filtering (opt-in via IQMoments(clutterFilter=...))
# =============================================================================
#
# Vectorized ports of the per-gate filters from iqtest_clutter.py, adapted to
# operate on a whole ray's IQ matrix at once (shape (n_gates, n_pulses),
# complex) instead of looping gate-by-gate. Math is unchanged from the
# reference: 'fft' is the Gaussian spectral notch, 'regression' is the
# linear-detrend (GMAP-style) filter, and 'cpa' doesn't touch the IQ at all --
# it just flags gates whose lag-1 phase alignment looks like clutter, for the
# caller to mask out of the final moments.

def _clutter_filter_fft(X: np.ndarray, filter_width: float = 3.0) -> np.ndarray:
    """Gaussian notch filter around zero Doppler, applied along the pulse
    axis (axis=1) for every gate (row) at once."""
    n_pulses = X.shape[1]
    if n_pulses < 8:
        return X
    spectrum = np.fft.fft(X, axis=1)
    freqs = np.fft.fftfreq(n_pulses)
    sigma = filter_width / 6.0
    notch = 1.0 - np.exp(-(freqs ** 2) / (2 * sigma ** 2))
    return np.fft.ifft(spectrum * notch[np.newaxis, :], axis=1)


def _clutter_filter_regression(X: np.ndarray) -> np.ndarray:
    """Linear detrend (GMAP-style) along the pulse axis for every gate at
    once, fitting I and Q separately just like the reference implementation
    but via closed-form least squares instead of a per-gate np.polyfit
    loop."""
    n_pulses = X.shape[1]
    if n_pulses < 4:
        return X
    x = np.arange(n_pulses, dtype=np.float64)
    x_mean = x.mean()
    x_centered = x - x_mean
    denom = np.sum(x_centered ** 2)

    def _detrend(component):
        c_mean = component.mean(axis=1, keepdims=True)
        slope = (component - c_mean) @ x_centered / denom
        trend = c_mean + slope[:, np.newaxis] * x_centered[np.newaxis, :]
        return component - trend

    return _detrend(X.real) + 1j * _detrend(X.imag)


def _clutter_phase_alignment(X: np.ndarray) -> np.ndarray:
    """Per-gate CPA metric (0-1, 1 = perfectly phase-aligned = likely
    clutter), computed along the pulse axis for every gate at once."""
    n_pulses = X.shape[1]
    if n_pulses < 3:
        return np.zeros(X.shape[0], dtype=np.float64)
    lag1 = np.mean(X[:, 1:] * np.conj(X[:, :-1]), axis=1)
    power = np.mean(np.abs(X) ** 2, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cpa = np.where(power > 1e-10, np.abs(lag1) / power, 0.0)
    return np.clip(cpa, 0.0, 1.0)


# ---- adapter: radarkitIQ.rkcfile -------------------------------------------
def _extract_from_rkcfile(rk, verbose=True):
    desc = rk.header['desc']
    config = rk.header['config']
    build_no = rk.header['buildNo']

    if build_no < 6 or build_no > 8:
        raise NotImplementedError(
            f".rkc build {build_no} is not supported by the built-in rkcfile "
            "adapter (only builds 6-8 have PRT/pulse-width/gate-spacing "
            "extraction wired up). Register a custom adapter for it.")

    iq_raw = rk.pulses['iq']  # (numPulses, 2[I/Q], gateCount, 2[pol])
    # The original built a full 4D float64 copy of iq_raw (astype), then a
    # full complex copy on top of that, before ever slicing out H/V -- for
    # a big volume that's two ~2x-oversized transient allocations. Slice
    # straight to the (numPulses, gateCount) I and Q planes we actually
    # need per channel and build the complex arrays directly from those.
    Xh_full = (iq_raw[:, 0, :, 0].astype(np.float64)
               + 1j * iq_raw[:, 1, :, 0].astype(np.float64)).T  # H = pol 0, (n_gates, n_pulses)
    Xv_full = (iq_raw[:, 0, :, 1].astype(np.float64)
               + 1j * iq_raw[:, 1, :, 1].astype(np.float64)).T  # V = pol 1

    # raxpolIQ.py, the reference implementation, does NOT actually apply noise-floor
    # subtraction -- its noise line is commented out and replaced with a hardcoded
    # zero (`noisedB = np.zeros((nRay, 2))`). Pulling the real noise floor from
    # config['noise'] here (as this adapter used to) makes Sh/Sv diverge from the
    # reference, which throws off anything derived from them -- most visibly RHOHV,
    # since it's a bounded ratio that's especially sensitive to the Sh*Sv denominator.
    noise_h_dB = 0.0
    noise_v_dB = 0.0

    # For compressed/downsampled I/Q, each downsampled gate spans pulseToRayRatio
    # raw gates worth of range -- the true gate spacing is pulseToRayRatio x
    # pulseGateSize, not pulseGateSize alone (which is only correct for raw data).
    if rk.header['dataType'] == 'compressed':
        gate_size_m = float(desc['pulseToRayRatio']) * float(config['pulseGateSize'])
    else:
        gate_size_m = float(config['pulseGateSize'])

    # RadarKit reports this directly per pulse -- no need to estimate it.
    # It's only used when IQMoments' excludeTransition is given an explicit
    # transitionElVelDegS threshold; the no-threshold (mode-based) path
    # below doesn't need it.
    el_vel = np.abs(rk.pulses['elevationVelocityDegreesPerSecond'].astype(np.float64))

    return dict(
        Xh_full=Xh_full, Xv_full=Xv_full, dual_pol=True,
        az=rk.azArray(), el=rk.elArray(), el_vel=el_vel,
        # rk.pulses['timeDouble'] is NOT a reliable absolute epoch on this RadarKit
        # build -- for build 7 .rkc files it comes back as (time_tv_sec % 86400) + 86400,
        # i.e. correct time-of-day but with the actual date discarded (interpreting it as
        # a Unix timestamp lands on 1970-01-02). time_tv_sec/time_tv_usec are the real,
        # correct epoch fields -- use those instead.
        times=(rk.pulses['time_tv_sec'].astype(np.float64)
               + rk.pulses['time_tv_usec'].astype(np.float64) / 1e6),
        wavelength_m=float(desc['wavelength']),
        prt_s=_first_nonzero(config['prt']),
        pulse_width_s=_first_nonzero(config['pw']),
        gate_size_m=gate_size_m,
        lat=float(desc['latitude']), lon=float(desc['longitude']),
        alt=float(desc['radarHeight']),
        instrument_name=desc.get('name', 'Unknown'),
        # raxpolIQ.py's reference implementation applies a fixed +8 dB offset on top
        # of the raw systemZCal header value before using it as the DBZ calibration
        # constant -- this adapter was dropping that offset, biasing DBZ 8 dB low.
        zcal=float(np.atleast_1d(config['systemZCal'])[0]) + 8.0,
        dcal=float(config['systemDCal']), pcal=float(config['systemPCal']),
        noise_h_dB=noise_h_dB, noise_v_dB=noise_v_dB,
    )


register_iq_adapter('rkcfile', _extract_from_rkcfile)


# ---- adapter: vaisalaIQ.vaisalafile ----------------------------------------
def _extract_from_vaisalafile(vf, verbose=True):
    header = vf.header
    pulses = vf.pulses
    if not pulses:
        raise ValueError("vaisalafile has no pulses.")

    n_gates = pulses[0].iq_h.shape[0]
    n_pulses = len(pulses)
    dual_pol = vf.isDualPol()

    Xh_full = np.full((n_gates, n_pulses), np.nan, dtype=np.complex128)
    Xv_full = np.zeros((n_gates, n_pulses), dtype=np.complex128)
    times = np.full(n_pulses, np.nan, dtype=np.float64)

    for i, p in enumerate(pulses):
        g = min(n_gates, p.iq_h.shape[0])
        Xh_full[:g, i] = p.iq_h[:g, 0] + 1j * p.iq_h[:g, 1]
        if dual_pol and p.iq_v.shape[0] >= n_gates:
            Xv_full[:, i] = p.iq_v[:n_gates, 0] + 1j * p.iq_v[:n_gates, 1]
        t_utc = p.hdr.get('iTimeUTC')
        if t_utc is not None:
            t_ms = p.hdr.get('iMSecUTC', 0)
            times[i] = float(t_utc) + float(t_ms) / 1000.0

    # The raw I/Q coming out of the RVP8 LUT unpack is in arbitrary
    # normalized ADC-code units, not the physical dBm scale that
    # fNoiseDBm/fDBzCalib are calibrated against. fSaturationDBM is the
    # header field that bridges the two: it's the dBm level a raw
    # unpacked amplitude of 1.0 corresponds to. iqtest_clutter.py applies
    # this by scaling *power* (`power * 10**(fSaturationDBM/10)`) right
    # before comparing to the noise floor; scaling the I/Q amplitude here
    # by sqrt(10**(fSaturationDBM/10)) is equivalent (power scales as
    # amplitude squared) and keeps the noise-floor subtraction, ZDR, and
    # SNR all self-consistent downstream, instead of just tacking the
    # offset onto zcal at the end.
    sat_dbm = _header_get(header, 'fSaturationDBM', required=False)
    if sat_dbm is not None:
        iq_scale = np.sqrt(10.0 ** (float(sat_dbm) / 10.0))
        Xh_full *= iq_scale
        Xv_full *= iq_scale
    elif verbose:
        print("[warn] fSaturationDBM not found in header -- DBZ/SNR will "
              "be off by whatever that offset should have been.")

    wavelength_cm = _header_get(header, 'fWavelengthCM', required=False)
    wavelength_m = float(wavelength_cm) / 100.0 if wavelength_cm is not None else None

    pw_usec = _header_get(header, 'fPWidthUSec', required=False)
    pulse_width_s = float(pw_usec) * 1e-6 if pw_usec is not None else None

    prt_s = None
    sy_clk_mhz = _header_get(header, 'fSyClkMHz', required=False)
    if sy_clk_mhz:
        prt_ticks = [p.hdr.get('iPrevPRT') for p in pulses[1:] if p.hdr.get('iPrevPRT')]
        if prt_ticks:
            prt_s = float(np.median(prt_ticks)) / (float(sy_clk_mhz) * 1e6)

    # NOTE: fRangeMaskRes alone is the *bit* size of the iRangeMask bitmask,
    # not necessarily the spacing between the range bins that actually
    # have data -- if iRangeMask is decimated, using fRangeMaskRes directly
    # (as this adapter used to) undercounts the true gate spacing.
    # iqtest_clutter.py derives the real spacing from where the mask's
    # bits actually start/end (decode_range_mask); do the same here.
    start_range_m, gate_size_m = _decode_range_mask(header)
    if gate_size_m is None:
        # Fall back to the raw resolution field if there's no usable
        # iRangeMask to decode (e.g. header variant without it).
        fallback_res = _header_get(header, 'fRangeMaskRes', required=False)
        gate_size_m = float(fallback_res) if fallback_res is not None else None
        start_range_m = 0.0

    zcal = _header_get(header, 'fDBzCalib', required=False, default=None)
    noise_dbm = _header_get(header, 'fNoiseDBm', required=False, default=None)
    if isinstance(noise_dbm, list) and len(noise_dbm) >= 2:
        noise_h_dB, noise_v_dB = float(noise_dbm[0]), float(noise_dbm[1])
    elif noise_dbm is not None:
        noise_h_dB = noise_v_dB = float(noise_dbm)
    else:
        noise_h_dB = noise_v_dB = None

    site_name = _header_get(header, 'sSiteName', required=False)

    # lat/lon/alt aren't in rvp8PulseInfo itself, but files logged with
    # GPS/INU enabled carry them per-pulse under inu.iLatitude/iLongitude/
    # iHeight. Falls back to (None, None, None) if not present, in which
    # case the caller should pass lat=/lon=/alt= explicitly to IQMoments.
    lat, lon, alt = vf.geolocation()

    # rvp8PulseHdr carries no antenna-velocity field the way RadarKit does,
    # so estimate elevation angular velocity from consecutive iEl/time
    # samples instead. Only needed for the explicit-threshold path of
    # excludeTransition; the no-threshold (mode-based) path doesn't need it.
    el_full = vf.elArray()
    el_vel = _angular_velocity_deg_s(el_full, times)

    task_name = _header_get(header, 'taskID.sTaskName', required=False, default='')
    scan_id = 0
    if isinstance(task_name, str) and 'vcp' in task_name.lower():
        match = re.search(r'vcp(\d+)', task_name.lower())
        if match:
            scan_id = int(match.group(1))

    # Extract Sweep Number
    sweep_number = _header_get(header, 'taskID.iSweep', required=False, default=0)

    # dcal (ZDR) / pcal (system PhiDP) calibration constants are genuinely
    # not part of the rvp8PulseInfo/rvptsPulseInfo header format -- it only
    # ever carries fDBzCalib. There's nothing to extract here; pass them
    # explicitly via IQMoments(dcal=..., pcal=...) if you have them from
    # elsewhere (e.g. a separate cal sheet).
    return dict(
        Xh_full=Xh_full, Xv_full=Xv_full, dual_pol=dual_pol,
        az=vf.azArray(), el=el_full, el_vel=el_vel, times=times,
        wavelength_m=wavelength_m, prt_s=prt_s, pulse_width_s=pulse_width_s,
        gate_size_m=gate_size_m, start_range_m=start_range_m,
        lat=lat, lon=lon, alt=alt,
        instrument_name=site_name or 'Unknown',
        zcal=float(zcal) if zcal is not None else None, dcal=None, pcal=None,
        noise_h_dB=noise_h_dB, noise_v_dB=noise_v_dB, scan_id=scan_id, sweep_number=sweep_number,
    )


register_iq_adapter('vaisalafile', _extract_from_vaisalafile)


class IQMoments:
    """Ray-binned pulse-pair moments computed from any registered I/Q file
    object (see register_iq_adapter). Works identically for rkcfile and
    vaisalafile inputs -- same constructor call either way."""

    def __init__(self, fileobj, azDeg: float = 1.0, azOverlapDeg: float = 0.0,
                 gstep: int = 1, nLags: int = 1,
                 flipVel: bool = False, singlePol: bool = False,
                 wavelengthM: float = None, prtS: float = None,
                 pulseWidthS: float = None, gateSizeM: float = None,
                 lat: float = None, lon: float = None, alt: float = None,
                 instrumentName: str = None,
                 zcal: float = None, dcal: float = None, pcal: float = None,
                 noiseHdB: float = None, noiseVdB: float = None,
                 clutterFilter: str = None, clutterFilterWidth: float = 3.0,
                 clutterThreshold: float = 1.0,
                 noiseCorrection: bool = True,
                 attenCorrection: bool = True,
                 dbzMethod: str = None,
                 phidpSign: int = None,
                 fixTransitionRays: bool = True,
                 verbose: bool = True):
        """
        clutterFilter : None, 'fft', 'regression', or 'cpa'. Off by default
            (None), matching prior behavior exactly.
            - 'fft':        Gaussian spectral notch around zero Doppler,
                             applied to the IQ before autocorrelation.
            - 'regression': linear detrend (GMAP-style) applied to the IQ
                             before autocorrelation.
            - 'cpa':        doesn't touch the IQ; flags gates whose lag-1
                             phase alignment exceeds clutterThreshold and
                             masks them to NaN in the final moments.
        clutterFilterWidth : notch width in FFT bins, only used when
            clutterFilter='fft' (default 3.0, matching iqtest_clutter.py).
        clutterThreshold : CPA threshold above which a gate is masked as
            clutter, only used when clutterFilter='cpa' (default 1.0 --
            raise this, e.g. to 0.6-0.8, to actually start masking).
        noiseCorrection : if True (default), DBZ, WIDTH, ZDR, RHOHV, and
            SNR are all computed from the noise-subtracted H/V power
            (gates that go negative after subtraction are masked to NaN).
            Set to False to use the raw, un-noise-corrected power for all
            of them instead -- matching iqtest_clutter.py, which never
            subtracts noise before forming its power ratios. This is one
            switch for every product, not just WIDTH.
        attenCorrection : only relevant when dbzMethod='snr' (i.e. Vaisala
            input by default); adds the same per-gate/per-ray gaseous-
            attenuation correction iqtest_clutter.py looks up from its
            elevation/range table (build_atten_table), on top of the SNR
            + zcal + range term. Has no effect on dbzMethod='pulsepair'
            (rkcfile input), which never applied an attenuation term.
        dbzMethod : 'pulsepair' or 'snr'. If left as None (default),
            IQMoments picks it for you based on the input file type:
            'pulsepair' for rkcfile (the original, unchanged formula this
            class has always used -- 10*log10(Sh*range_km**2) + zcal,
            matching the raxpolIQ.py reference), and 'snr' for vaisalafile
            (iqtest_clutter.py's compute_dbz_from_power: an actual SNR
            ratio -- noise-subtracted power divided by the noise floor --
            plus zcal, range correction, and the attenuation table).
            RadarKit and Vaisala files are calibrated differently, so they
            need different formulas; you normally shouldn't need to
            override this.
        phidpSign : PHIDP = phidpSign * angle(C_lag0)/pi*180 + pcal, where
            C_lag0 = <Xh * conj(Xv)> (Doviak & Zrnic's Rhv(0) convention).
            Default is +1. IQMoments passes -1 for vaisalafile input,
            since Vaisala's rvp8 dual-pol pulses need the cross term
            conjugated the other way (equivalent to using Rvh(0) =
            <Xv * conj(Xh)> instead) to come out in the expected sense --
            without the flip PHIDP comes out mirrored (-180..0 instead of
            0..180). rkcfile input keeps phidpSign=+1 (unchanged).
        excludeTransition : if True, pulses caught while the antenna is
            actively changing tilt between sweeps are dropped before rays
            are formed, instead of being binned in as if they were part of
            a normal, elevation-constant sweep. Off by default -- matching
            prior behavior exactly. Without this, a ray whose pulses span
            a transition mixes pulses from genuinely different antenna
            positions into one average, which is what shows up as
            smearing/striping near sweep boundaries.

            Two different detection modes, depending on whether you pass
            transitionElVelDegS:

            - transitionElVelDegS given: velocity mode. A pulse is flagged
              as transition if its elevation angular velocity exceeds that
              threshold (deg/s). rkcfile reports this directly per pulse;
              vaisalafile has it estimated from consecutive iEl/time
              samples.
            - transitionElVelDegS left as None (default): mode mode. There
              is no velocity threshold to tune at all -- instead, the
              single most common elevation angle across the whole pulse
              stream is taken as the sweep's nominal fixed tilt (its
              "mode"), and *every* pulse whose elevation isn't within
              transitionElToleranceDeg of that value is dropped as a
              transition pulse. e.g. if the sweep's mode elevation is
              0.33 deg, any pulse sitting at anything other than ~0.33 deg
              is removed. This is the simpler, more aggressive option --
              it doesn't care how fast the antenna was moving, only
              whether a pulse was actually on the sweep's target angle --
              and needs no rate tuning, at the cost of being unusable for
              genuine RHIs or multi-tilt-per-file volumes (where more than
              one elevation is legitimately "normal").
        transitionElVelDegS : elevation angular-velocity threshold in
            deg/s above which a pulse is dropped as transition, when using
            velocity mode. Leave as None (default) to use mode mode
            instead -- see excludeTransition above.
        transitionElToleranceDeg : in mode mode only (transitionElVelDegS
            left as None), how far a pulse's elevation may sit from the
            sweep's mode elevation and still count as "on tilt" (default
            0.05 deg -- just enough to absorb encoder jitter on an
            otherwise steady sweep).
        transitionMarginPulses : also drops this many pulses immediately
            before and after each detected transition pulse (default 4),
            in either detection mode. A per-pulse test alone tends to
            leave a thin residual streak: right at the edges of a
            transition, a pulse can briefly look "clean" (within velocity
            or tolerance) while the antenna is still not settled, biasing
            its ray's average just enough to stand out from its
            neighbors. Set to 0 to disable and use the raw per-pulse test
            only.
        """
        cls_name = type(fileobj).__name__
        if cls_name not in _IQ_ADAPTERS:
            raise TypeError(
                f"No I/Q adapter registered for file object type {cls_name!r}. "
                f"Supported types: {sorted(_IQ_ADAPTERS)}. Add support with "
                "calcIQ.register_iq_adapter('YourClassName', yourExtractorFunc).")

        info = _IQ_ADAPTERS[cls_name](fileobj, verbose=verbose)

        if dbzMethod is None:
            dbzMethod = 'snr' if cls_name == 'vaisalafile' else 'pulsepair'
        elif dbzMethod not in ('pulsepair', 'snr'):
            raise ValueError(f"dbzMethod must be 'pulsepair', 'snr', or None, got {dbzMethod!r}.")
        self.dbzMethod = dbzMethod

        if phidpSign is None:
            phidpSign = -1 if cls_name == 'vaisalafile' else 1
        elif phidpSign not in (1, -1):
            raise ValueError(f"phidpSign must be 1, -1, or None, got {phidpSign!r}.")
        self.phidpSign = phidpSign

        overrides = dict(
            wavelength_m=wavelengthM, prt_s=prtS, pulse_width_s=pulseWidthS,
            gate_size_m=gateSizeM, lat=lat, lon=lon, alt=alt,
            instrument_name=instrumentName, zcal=zcal, dcal=dcal, pcal=pcal,
            noise_h_dB=noiseHdB, noise_v_dB=noiseVdB,
        )
        for k, v in overrides.items():
            if v is not None:
                info[k] = v

        required = ['wavelength_m', 'prt_s', 'pulse_width_s', 'gate_size_m']
        missing = [k for k in required if info.get(k) is None]
        if missing:
            raise ValueError(
                f"Missing required metadata {missing} for file type {cls_name!r}; "
                "the adapter couldn't find them in the header. Pass them "
                "explicitly, e.g. IQMoments(fileobj, wavelengthM=..., prtS=..., "
                "pulseWidthS=..., gateSizeM=...).")

        for k in ('lat', 'lon', 'alt', 'zcal', 'dcal', 'pcal', 'noise_h_dB', 'noise_v_dB'):
            if info.get(k) is None:
                info[k] = 0.0
        if not info.get('instrument_name'):
            info['instrument_name'] = 'Unknown'

        self.source_type = cls_name
        self.azDeg = azDeg
        self.gstep = gstep
        self.nLags = nLags
        self.flipVel = flipVel
        self.singlePol = singlePol

        self.instrument_name = info['instrument_name']
        self.lat, self.lon, self.alt = info['lat'], info['lon'], info['alt']
        self.scan_id = info.get('scan_id', 0)
        self.sweep_number = info.get('sweep_number', 0)

        az, el = info['az'], info['el']
        self.is_rhi = bool(AzimuthBinner.is_likely_rhi(az, el))
        if self.is_rhi and verbose:
            print("[warn] Scan looks like an RHI (elevation varies more "
                  "than azimuth). IQMoments bins by azimuth, which is "
                  "usually wrong for an RHI -- results may not make sense.")

        Xh_full, Xv_full = info['Xh_full'], info['Xv_full']
        n_gates = Xh_full.shape[0]
        times = info['times']



        binner = AzimuthBinner(azimuth_beamwidth_deg=azDeg, beam_overlap_deg=azOverlapDeg)
        az_unique, pulse_boundaries, middle_pulses = binner.bin_azimuths(az)

        if fixTransitionRays:
            # --- ENFORCE UNIFORM RADIALS (Legacy Behavior) ---
            pulse_counts = pulse_boundaries[:, 1] - pulse_boundaries[:, 0] + 1
            pulse_per_rad = int(np.median(pulse_counts))

            # Transpose to (n_pulses, n_gates) to reshape properly
            iq_h = Xh_full.T
            iq_v = Xv_full.T

            num_rad = iq_h.shape[0] // pulse_per_rad

            # Truncate leftovers
            iq_h = iq_h[:num_rad * pulse_per_rad]
            iq_v = iq_v[:num_rad * pulse_per_rad]

            # Reshape into 3D and swap axes back to (n_gates, n_pulses)
            iq_h = iq_h.reshape((num_rad, pulse_per_rad, iq_h.shape[-1])).swapaxes(1, 2)
            iq_v = iq_v.reshape((num_rad, pulse_per_rad, iq_v.shape[-1])).swapaxes(1, 2)

            # Pack back into the expected list format
            ray_iq_pairs = [(iq_h[i], iq_v[i]) for i in range(num_rad)]

            # Re-align metadata
            middle_pulses = (np.arange(num_rad) * pulse_per_rad + (pulse_per_rad // 2)).astype(np.int32)
            az_unique = az[middle_pulses]
            n_rays = num_rad
            
            if verbose:
                print(f"[{cls_name}] Fixed transition rays applied: {n_rays} uniform radials formed "
                      f"({pulse_per_rad} pulses per radial).")
        else:
            # --- STANDARD VARIABLE-LENGTH RAYS ---
            ray_iq_pairs = binner.extract_rays(Xh_full, Xv_full, pulse_boundaries)
            n_rays = len(ray_iq_pairs)
            
            if verbose:
                print(f"[{cls_name}] Formed {n_rays} rays from {len(az)} pulses "
                      f"(azDeg={azDeg}).")

        cpa_full = None

        self.wavelength_m = info['wavelength_m']
        self.prt_s = info['prt_s']
        self.pulse_width_s = info['pulse_width_s']
        gate_size_m = info['gate_size_m']

        if clutterFilter is True:
            # 1. Estimate scan rate (degrees/sec) for the empirical formula
            dang = (np.diff(az) + 180.0) % 360.0 - 180.0
            dt = np.diff(times)
            valid = dt > 0
            scan_rate = np.nanmedian(np.abs(dang[valid] / dt[valid])) if valid.any() else 15.0
            
            filtered_pairs = []
            for Xh, Xv in ray_iq_pairs:
                n_pulses = Xh.shape[1]
                
                # Safety check: polynomials need a minimum number of points
                if n_pulses < 4:
                    filtered_pairs.append((Xh, Xv))
                    continue

                # Instantiate GCF dynamically for this ray's specific pulse count
                gcf = GroundClutterFilter(
                    wavelength=self.wavelength_m,
                    scan_rate=scan_rate,
                    prt=self.prt_s,
                    num_samples=n_pulses
                )
                
                # 2. Expand to 3D arrays: (n_gates, 1 ray, n_pulses)
                Xh_3d = np.expand_dims(Xh, axis=1)
                Xv_3d = np.expand_dims(Xv, axis=1)
                
                # 3. Filter H and V channels (Notice apply_interpolation=False)
                f_ih, f_qh, _, _, _, _ = gcf.filter_iq_data(
                    Xh_3d.real, Xh_3d.imag, apply_interpolation=False
                )
                f_iv, f_qv, _, _, _, _ = gcf.filter_iq_data(
                    Xv_3d.real, Xv_3d.imag, apply_interpolation=False
                )
                
                # 4. Repackage back into complex 2D arrays and append
                f_Xh = f_ih[:, 0, :] + 1j * f_qh[:, 0, :]
                f_Xv = f_iv[:, 0, :] + 1j * f_qv[:, 0, :]
                
                filtered_pairs.append((f_Xh, f_Xv))
            
            # 5. Replace the original pairs with the filtered ones
            ray_iq_pairs = filtered_pairs

        start_range_m = info.get('start_range_m') or 0.0
        range_km_full = (start_range_m
                          + (np.arange(n_gates, dtype=np.float64) + 0.5) * gate_size_m) / 1000.0
        range_km = _average_range_km(range_km_full, gstep)
        self.range_m = (range_km * 1000.0).astype(np.float32)

        clutter_mask = None
        if cpa_full is not None:
            # cpa_full is (n_gates_full, n_rays); average to the same gate
            # resolution as the moments (gstep may bin multiple raw gates
            # into one output gate), then transpose to (n_rays, n_gates)
            # to match self.DBZ/self.VEL layout.
            cpa_avg = PulsePairMoments.average_by_gate_step(
                cpa_full[:, :, np.newaxis], gstep)[:, :, 0]
            clutter_mask = (cpa_avg > clutterThreshold).T

        ppm = PulsePairMoments(wavelength_m=self.wavelength_m, prt_s=self.prt_s, n_lags=nLags)
        elev_deg_mid = el[middle_pulses].astype(np.float64)
        wavelength_cm = self.wavelength_m * 100.0
        moments = ppm.process_volume(
            ray_iq_pairs, range_km, gstep=gstep,
            zcal=info['zcal'], dcal=info['dcal'], pcal=info['pcal'],
            noise_h_dB=info['noise_h_dB'], noise_v_dB=info['noise_v_dB'],
            noiseCorrection=noiseCorrection,
            elev_deg=elev_deg_mid, wavelength_cm=wavelength_cm,
            attenCorrection=attenCorrection, dbzMethod=dbzMethod,
            phidpSign=phidpSign,
        )
        self.noiseCorrection = noiseCorrection
        self.attenCorrection = attenCorrection



        self.DBZ = moments['DBZ'].T.astype(np.float64)
        self.VEL = moments['VEL'].T.astype(np.float64)
        if flipVel:
            self.VEL = -self.VEL
        self.WIDTH = moments['WIDTH'].T.astype(np.float64)
        self.SNRH = moments['SNRHC'].T.astype(np.float64)

        if singlePol or not info['dual_pol']:
            self.ZDR = None
            self.PHIDP = None
            self.RHOHV = None
            self.SNRV = None
        else:
            self.ZDR = moments['ZDR'].T.astype(np.float64)
            self.PHIDP = moments['PHIDP'].T.astype(np.float64)
            self.RHOHV = moments['RHOHV'].T.astype(np.float64) if moments['RHOHV'] is not None else None
            self.SNRV = moments['SNRVC'].T.astype(np.float64)
            if moments['RHOHV'] is None and verbose:
                print("[warn] RHOHV not computed (requires nLags == 1).")

        if clutter_mask is not None:
            for field in (self.DBZ, self.VEL, self.WIDTH, self.SNRH,
                          self.ZDR, self.PHIDP, self.RHOHV, self.SNRV):
                if field is not None:
                    field[clutter_mask] = np.nan

        self.time = times[middle_pulses].astype(np.float64)
        self.azimuth = az_unique.astype(np.float32)
        self.elevation = el[middle_pulses].astype(np.float32)
        self.n_rays = n_rays
        self.n_gates = len(self.range_m)
        self.clutter_filter = clutterFilter

    def __repr__(self):
        return (f"<IQMoments source={self.source_type} instrument={self.instrument_name!r} "
                f"n_rays={self.n_rays} n_gates={self.n_gates} is_rhi={self.is_rhi}>")