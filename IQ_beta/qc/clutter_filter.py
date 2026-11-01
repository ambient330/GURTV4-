import numpy as np
from numpy.polynomial import polynomial as P
from numba import njit, prange
import math

# ================================================================
# Numba-compiled helpers
# ================================================================

@njit(cache=True)
def _get_polynomial_order_numba(cnr_db_flat, scan_rate, nyquist_velocity, num_samples):
    """
    Vectorised polynomial-order selection compiled with Numba.
    Mirrors the original empirical formula from Appendix A of Hubbert et al. (2025).
    """
    n = cnr_db_flat.shape[0]
    orders = np.empty(n, dtype=np.int64)

    wc  = 0.03 + 0.017 * scan_rate          # b_factor == 1 (default)
    wcn = wc / nyquist_velocity

    On = -2.0428 * (wcn * wcn) + 0.6490 * wcn

    for i in range(n):
        cnr_lin = 10.0 ** (cnr_db_flat[i] / 10.0)
        if cnr_lin < 1e-5:
            cnr_lin = 1e-5
        o = On * (cnr_lin ** (2.0 / 3.0)) * num_samples
        o = math.ceil(o)
        if o < 1:
            o = 1
        if o > num_samples - 1:
            o = num_samples - 1
        orders[i] = int(o)
    return orders


@njit(parallel=True, cache=True)
def _gaussian_interpolation_numba(spectrum_batch, poly_orders, E_points, velocity_threshold_interp):
    """
    Gaussian interpolation across the zero-velocity gap.
    Compiled with Numba; each gate is processed in parallel via prange.

    Parameters
    ----------
    spectrum_batch : float32 array (num_series, num_samples)
        Doppler power spectra in linear units.
    poly_orders : int64 array (num_series,)
        Polynomial order used when filtering each series.
    E_points : int
        Points on either side of the gap used for Gaussian fitting.
    velocity_threshold_interp : float
        Skip interpolation when |Vest/Nyq| exceeds this threshold.

    Returns
    -------
    out : float32 array (num_series, num_samples)
        Spectra with the zero-velocity gap filled.
    """
    num_series  = spectrum_batch.shape[0]
    num_samples = spectrum_batch.shape[1]
    out         = np.empty_like(spectrum_batch)

    # Copy input into output first (prange requires we write independently)
    for idx in prange(num_series):
        for j in range(num_samples):
            out[idx, j] = spectrum_batch[idx, j]

    zero_vel_idx = num_samples // 2

    for idx in prange(num_series):
        poly_order = poly_orders[idx]

        # Interpolation half-width L from Table B1
        if poly_order <= 3:
            L = 1
        elif poly_order <= 5:
            L = 2
        elif poly_order <= 8:
            L = 3
        elif poly_order <= 10:
            L = 4
        elif poly_order <= 13:
            L = 5
        elif poly_order <= 16:
            L = 6
        elif poly_order <= 19:
            L = 7
        else:
            L = 8

        gap_half = L
        gap_start_idx = zero_vel_idx - gap_half
        gap_end_idx   = zero_vel_idx + gap_half + 1
        if gap_start_idx < 0:
            gap_start_idx = 0
        if gap_end_idx > num_samples:
            gap_end_idx = num_samples

        # Velocity estimate from centre-of-mass of full spectrum
        total_power = 0.0
        for j in range(num_samples):
            total_power += spectrum_batch[idx, j]

        if total_power < 1e-30:
            continue

        weighted_vel = 0.0
        for j in range(num_samples):
            weighted_vel += (j - zero_vel_idx) * spectrum_batch[idx, j]
        vel_est_norm = weighted_vel / (total_power * (num_samples / 2.0))

        if vel_est_norm < 0.0:
            vel_est_norm = -vel_est_norm
        if vel_est_norm > velocity_threshold_interp:
            continue

        # Collect interp points on either side of the gap
        interp_start_left  = gap_start_idx - E_points
        interp_end_left    = gap_start_idx
        interp_start_right = gap_end_idx
        interp_end_right   = gap_end_idx + E_points

        if interp_start_left < 0:
            interp_start_left = 0
        if interp_end_left < 0:
            interp_end_left = 0
        if interp_start_right > num_samples:
            interp_start_right = num_samples
        if interp_end_right > num_samples:
            interp_end_right = num_samples

        n_interp = (interp_end_left - interp_start_left) + (interp_end_right - interp_start_right)
        if n_interp < 2:
            continue

        # Gaussian fit parameters from interp points
        PT = 0.0
        for j in range(interp_start_left, interp_end_left):
            PT += spectrum_batch[idx, j]
        for j in range(interp_start_right, interp_end_right):
            PT += spectrum_batch[idx, j]

        if PT == 0.0:
            continue

        a = 0.0
        for j in range(interp_start_left, interp_end_left):
            a += (j - zero_vel_idx) * spectrum_batch[idx, j]
        for j in range(interp_start_right, interp_end_right):
            a += (j - zero_vel_idx) * spectrum_batch[idx, j]
        a /= PT

        s2 = 0.0
        for j in range(interp_start_left, interp_end_left):
            d = (j - zero_vel_idx) - a
            s2 += spectrum_batch[idx, j] * d * d
        for j in range(interp_start_right, interp_end_right):
            d = (j - zero_vel_idx) - a
            s2 += spectrum_batch[idx, j] * d * d
        s2 /= PT
        s = s2 ** 0.5
        s_eps = 1e-9
        if s < s_eps:
            s = s_eps

        # Two-pass Gaussian fill
        for _ in range(2):
            scale = PT / (s * (2.0 * math.pi) ** 0.5)
            for j in range(gap_start_idx, gap_end_idx):
                rel = (j - zero_vel_idx) - a
                out[idx, j] = scale * math.exp(-(rel * rel) / (2.0 * s * s))

            # Refit with filled gap included
            PT2 = 0.0
            for j in range(interp_start_left, interp_end_left):
                PT2 += out[idx, j]
            for j in range(interp_start_right, interp_end_right):
                PT2 += out[idx, j]
            for j in range(gap_start_idx, gap_end_idx):
                PT2 += out[idx, j]

            if PT2 == 0.0:
                break

            a2 = 0.0
            for j in range(interp_start_left, interp_end_left):
                a2 += (j - zero_vel_idx) * out[idx, j]
            for j in range(interp_start_right, interp_end_right):
                a2 += (j - zero_vel_idx) * out[idx, j]
            for j in range(gap_start_idx, gap_end_idx):
                a2 += (j - zero_vel_idx) * out[idx, j]
            a2 /= PT2

            s2_2 = 0.0
            for j in range(interp_start_left, interp_end_left):
                d = (j - zero_vel_idx) - a2
                s2_2 += out[idx, j] * d * d
            for j in range(interp_start_right, interp_end_right):
                d = (j - zero_vel_idx) - a2
                s2_2 += out[idx, j] * d * d
            for j in range(gap_start_idx, gap_end_idx):
                d = (j - zero_vel_idx) - a2
                s2_2 += out[idx, j] * d * d
            s2_2 /= PT2

            s_new = s2_2 ** 0.5
            if s_new < s_eps:
                s_new = s_eps

            PT = PT2
            a  = a2
            s  = s_new

    return out


# ================================================================
# GroundClutterFilter class
# ================================================================

class GroundClutterFilter:
    """
    Ground clutter filter based on global polynomial regression,
    as described in Hubbert et al. (2025).

    I/Q time-series data flows through:
      1. CNR estimation (if not supplied)
      2. Polynomial-order selection  (_get_polynomial_order  — Numba JIT)
      3. Regression filtering        (_perform_regression_filter — NumPy)
      4. Gaussian gap interpolation  (_gaussian_interpolation_batch — Numba parallel)
    """

    def __init__(self, wavelength, scan_rate, prt, num_samples):
        self.wavelength       = wavelength
        self.scan_rate        = scan_rate
        self.prt              = prt
        self.num_samples      = num_samples
        self.nyquist_velocity = wavelength / (4.0 * prt)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_clutter_power_from_polyfit(self, i_series, q_series, order=2):
        t          = np.arange(self.num_samples)
        coeffs_i   = P.polyfit(t, i_series.T, order)
        clutter_i  = P.polyval(t, coeffs_i)
        coeffs_q   = P.polyfit(t, q_series.T, order)
        clutter_q  = P.polyval(t, coeffs_q)

        total_clutter_power = (
            np.mean(clutter_i ** 2, axis=-1)
            + np.mean(clutter_q ** 2, axis=-1)
        )
        return total_clutter_power

    def _get_polynomial_order(self, cnr_db, b_factor=1.0):
        # Delegate to the Numba kernel (b_factor != 1 not yet exposed;
        # extend the kernel signature if needed).
        cnr_arr = np.asarray(cnr_db, dtype=np.float64)
        return _get_polynomial_order_numba(
            cnr_arr,
            float(self.scan_rate),
            float(self.nyquist_velocity),
            float(self.num_samples),
        )

    def _perform_regression_filter(self, time_series_batch, orders):
        filtered_batch    = np.zeros_like(time_series_batch)
        clutter_trend_map = np.zeros_like(time_series_batch)
        t                 = np.arange(self.num_samples)

        for order in np.unique(orders):
            idx = np.where(orders == order)[0]
            if idx.size == 0:
                continue
            sel       = time_series_batch[idx]          # (k, num_samples)
            coeffs    = P.polyfit(t, sel.T, order)
            trend     = P.polyval(t, coeffs)            # (num_samples, k)

            filtered_batch[idx]    = sel - trend
            clutter_trend_map[idx] = trend

        return filtered_batch, clutter_trend_map

    def _gaussian_interpolation_batch(self, spectrum_batch, poly_orders,
                                      E_points=3, velocity_threshold_interp=0.2):
        """Thin wrapper — delegates to the Numba parallel kernel."""
        sb = np.ascontiguousarray(spectrum_batch, dtype=np.float32)
        po = np.ascontiguousarray(poly_orders,    dtype=np.int64)
        return _gaussian_interpolation_numba(sb, po, int(E_points),
                                             float(velocity_threshold_interp))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_iq_data(self, i_data, q_data, cnr_db_map=None,
                       apply_interpolation=True,
                       interpolation_E_points=3,
                       velocity_threshold_interp=0.2):
        """
        Filter 3-D I/Q data (num_ranges, num_azimuths, num_samples).

        Returns
        -------
        filtered_i, filtered_q, poly_order_map,
        clutter_i_trend_map, clutter_q_trend_map, interpolated_spectrum_map
        """
        if i_data.shape != q_data.shape:
            raise ValueError("i_data and q_data must have the same shape.")
        if i_data.ndim != 3 or i_data.shape[2] != self.num_samples:
            raise ValueError(
                f"Input must be (range, azimuth, {self.num_samples}) — "
                f"got {i_data.shape}."
            )

        original_shape             = i_data.shape
        num_ranges, num_azimuths, num_samples = original_shape
        num_cells                  = num_ranges * num_azimuths

        i_flat = np.ascontiguousarray(i_data.reshape(num_cells, num_samples))
        q_flat = np.ascontiguousarray(q_data.reshape(num_cells, num_samples))

        # CNR estimation
        if cnr_db_map is None:
            clutter_power  = self._estimate_clutter_power_from_polyfit(i_flat, q_flat)
            raw_power      = np.mean(i_flat ** 2 + q_flat ** 2, axis=-1)
            noise_power    = np.maximum(1e-10, raw_power * 0.1)
            cnr_db_flat    = np.where(
                noise_power > 1e-10,
                10.0 * np.log10(clutter_power / noise_power),
                -99.0,
            )
        else:
            cnr_db_flat = cnr_db_map.flatten()

        # Polynomial order selection (Numba)
        poly_order_flat = self._get_polynomial_order(cnr_db_flat)

        # Regression filtering (NumPy — uses numpy.polynomial)
        filtered_i_flat, clutter_i_flat = self._perform_regression_filter(i_flat, poly_order_flat)
        filtered_q_flat, clutter_q_flat = self._perform_regression_filter(q_flat, poly_order_flat)

        # Doppler spectrum
        cx            = filtered_i_flat + 1j * filtered_q_flat
        spectrum_flat = np.abs(
            np.fft.fftshift(np.fft.fft(cx, axis=-1), axes=-1)
        ) ** 2

        # Gaussian gap interpolation (Numba parallel)
        if apply_interpolation:
            interp_flat = self._gaussian_interpolation_batch(
                spectrum_flat, poly_order_flat,
                E_points=interpolation_E_points,
                velocity_threshold_interp=velocity_threshold_interp,
            )
        else:
            interp_flat = spectrum_flat.astype(np.float32)

        # Reshape back
        return (
            filtered_i_flat.reshape(original_shape),
            filtered_q_flat.reshape(original_shape),
            poly_order_flat.reshape(original_shape[:-1]),
            clutter_i_flat.reshape(original_shape),
            clutter_q_flat.reshape(original_shape),
            interp_flat.reshape(original_shape),
        )