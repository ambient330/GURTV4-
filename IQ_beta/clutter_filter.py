import numpy as np
from numpy.polynomial import polynomial as P
from scipy import signal
from tqdm.auto import tqdm

class GroundClutterFilter:
    """
    Implements a ground clutter filter based on global polynomial regression,
    as described in the provided article "A New Paradigm for Automated Ground
    Clutter Removal: Global Regression Filtering" by Hubbert et al. (2025).

    This filter operates on I (in-phase) and Q (quadrature) time series data.
    It identifies clutter by fitting a polynomial to the time series and
    subtracting this fitted trend. It also includes automated polynomial
    order selection and an optional Gaussian interpolation for the
    zero-velocity gap.

    Python version created by Vitor Goede.
    """

    def __init__(self, wavelength, scan_rate, prt, num_samples):
        """
        Defines important parameters for initializing the clutter filtering.

        Args:
            wavelength (float): Radar wavelength in meters.
            scan_rate (float): antenna rotation speed in degrees/second.
            prt (float): pulse repetition time in 1/s.
            num_samples (float): number of samples per ray.
        """
        self.wavelength = wavelength
        self.scan_rate = scan_rate
        self.prt = prt
        self.num_samples = num_samples
        self.nyquist_velocity = self.wavelength / (4 * self.prt)

        print(f"Filter initialized with:")
        print(f"  Wavelength: {self.wavelength:.4f} m")
        print(f"  Scan Rate: {self.scan_rate} deg/s")
        print(f"  PRT: {self.prt} s")
        print(f"  Number of Samples per gate: {self.num_samples}")
        print(f"  Nyquist Velocity: {self.nyquist_velocity:.2f} m/s")

    def _estimate_clutter_power_from_polyfit(self, i_series, q_series, order=2):
        """
        Estimates the power backscattered by the ground clutter

        Args:
            i_series (np.ndarray): in-phase time series of complex numbers.
            q_series (np.ndarray): quadratyre time series of complex numbers.
            order (int): polynomial fit order.

        Returns:
            np.ndarray: Total clutter power array.
        """
        print(f"  _estimate_clutter_power_from_polyfit: i_series.shape = {i_series.shape}")
        print(f"  _estimate_clutter_power_from_polyfit: q_series.shape = {q_series.shape}")
        t = np.arange(self.num_samples)
        print(f"  _estimate_clutter_power_from_polyfit: t.shape = {t.shape}")
        i_series_T = i_series.T
        q_series_T = q_series.T
        print(f"  _estimate_clutter_power_from_polyfit: i_series_T.shape = {i_series_T.shape}")
        coeffs_i = P.polyfit(t, i_series_T, order)
        print(f"  _estimate_clutter_power_from_polyfit: coeffs_i.shape = {coeffs_i.shape}")

        clutter_i = P.polyval(t, coeffs_i)
        print(f"  _estimate_clutter_power_from_polyfit: clutter_i.shape = {clutter_i.shape}")

        coeffs_q = P.polyfit(t, q_series_T, order)
        clutter_q = P.polyval(t, coeffs_q) # Same for Q component
        print(f"  _estimate_clutter_power_from_polyfit: clutter_q.shape = {clutter_q.shape}")

        clutter_power_i = np.mean(clutter_i**2, axis=-1)
        clutter_power_q = np.mean(clutter_q**2, axis=-1)
        print(f"  _estimate_clutter_power_from_polyfit: clutter_power_i.shape = {clutter_power_i.shape}")

        total_clutter_power = clutter_power_i + clutter_power_q
        print(f"  _estimate_clutter_power_from_polyfit: total_clutter_power.shape = {total_clutter_power.shape}")
        return total_clutter_power

    def _get_polynomial_order(self, cnr_db, b_factor=1.0):
        """
        Determines the required polynomial order for the regression filter
        based on CNR and radar parameters, using the empirical formulas
        from Appendix A of the article.

        Args:
            cnr_db (np.ndarray): Clutter-to-Noise Ratio in dB (1D array or scalar).
            b_factor (float): Multiplicative parameter for clutter spectrum width.

        Returns:
            np.ndarray: The calculated polynomial order (integer array).
        """
        wc = b_factor * (0.03 + 0.017 * self.scan_rate)
        wcn = wc / self.nyquist_velocity

        On = -2.0428 * (wcn**2) + 0.6490 * wcn

        cnr_linear = 10**(cnr_db / 10.0)
        cnr_linear = np.maximum(cnr_linear, 1e-5)

        order = np.ceil(On * (cnr_linear**(2/3.0)) * self.num_samples)

        order = np.maximum(1, np.minimum(order, self.num_samples - 1)).astype(int)
        return order

    def _perform_regression_filter(self, time_series_batch, orders):
        """
        Performs the regression filter

        Args:
            time_series_batch (np.ndarray): Batch I/Q time series.
            orders (np.ndarray): Polynomial order.

        Returns:
           tuple: (filtered_batch, clutter_trend_map)
                    filtered_batch (np.ndarray): Clutter-suppressed I/Q.
                    orders (np.ndarray): Polynomial orders.
        """
        filtered_batch = np.zeros_like(time_series_batch)
        clutter_trend_map = np.zeros_like(time_series_batch)
        unique_orders = np.unique(orders)

        for order in unique_orders:
            indices_for_this_order = np.where(orders == order)[0]
            if len(indices_for_this_order) == 0:
                continue

            selected_series = time_series_batch[indices_for_this_order, :]
            
            t = np.arange(self.num_samples)
            series_T = selected_series.T

            coeffs = P.polyfit(t, series_T, order)

            clutter_trend = P.polyval(t, coeffs)

            print(f"  _perform_regression_filter: selected_series.shape = {selected_series.shape}")
            print(f"  _perform_regression_filter: clutter_trend.shape = {clutter_trend.shape}")

            filtered_series = selected_series - clutter_trend

            filtered_batch[indices_for_this_order, :] = filtered_series
            clutter_trend_map[indices_for_this_order, :] = clutter_trend

        return filtered_batch, clutter_trend_map

    def _gaussian_interpolation_batch(self, spectrum_batch, poly_orders, E_points=3, velocity_threshold_interp=0.2):
        """
        Applies Gaussian interpolation across the zero-velocity gap in the
        Doppler spectrum for a batch of spectra.

        Args:
            spectrum_batch (np.ndarray): Batch of Doppler power spectra (linear units) (num_series, num_samples).
            poly_orders (np.ndarray): Polynomial order used for filtering for each series (1D array).
            E_points (int): Number of points on either side of the gap used for Gaussian fitting.
            velocity_threshold_interp (float): Threshold for estimated velocity relative to Nyquist velocity.

        Returns:
            np.ndarray: Spectra with the zero-velocity gap interpolated (num_series, num_samples).
        """
        num_series, num_samples = spectrum_batch.shape
        interpolated_spectrum_batch = np.copy(spectrum_batch)

        zero_vel_idx = num_samples // 2
        nyquist_velocity_per_bin = (2 * self.nyquist_velocity / num_samples)

        mean_vel_indices = np.argmax(spectrum_batch, axis=-1)
        estimated_velocities = (mean_vel_indices - zero_vel_idx) * nyquist_velocity_per_bin

        apply_interp_mask = (np.abs(estimated_velocities / self.nyquist_velocity) <= velocity_threshold_interp)

        L_values = np.array([self._get_interpolation_width_L(order) for order in poly_orders])

        indices_to_interp = np.where(apply_interp_mask)[0]

        if len(indices_to_interp) == 0:
            return interpolated_spectrum_batch

        for idx in tqdm(indices_to_interp, desc="Batch Gaussian Interpolation"):
            spectrum = spectrum_batch[idx, :]
            L = L_values[idx]

            gap_start_idx = zero_vel_idx - L
            gap_end_idx = zero_vel_idx + L + 1

            interp_start_left = zero_vel_idx - L - E_points
            interp_end_left = zero_vel_idx - L

            interp_start_right = zero_vel_idx + L + 1
            interp_end_right = zero_vel_idx + L + 1 + E_points

            interp_indices_left = np.arange(max(0, interp_start_left), max(0, interp_end_left))
            interp_indices_right = np.arange(min(num_samples, interp_start_right), min(num_samples, interp_end_right))
            interp_indices = np.concatenate((interp_indices_left, interp_indices_right))

            if len(interp_indices) < 2:
                continue

            interp_values = spectrum[interp_indices]
            relative_vel_indices = interp_indices - zero_vel_idx

            PT = np.sum(interp_values)
            if PT == 0:
                continue

            a = np.sum(relative_vel_indices * interp_values) / PT
            s_squared = np.sum(interp_values * (relative_vel_indices - a)**2) / PT
            s = np.sqrt(s_squared)
            s_epsilon = 1e-9
            s = s + s_epsilon if s < s_epsilon else s

            gap_indices = np.arange(gap_start_idx, gap_end_idx)
            relative_gap_indices = gap_indices - zero_vel_idx

            for _ in range(2):
                estimated_gaussian = (PT / (s * np.sqrt(2 * np.pi))) * \
                                     np.exp(-(relative_gap_indices - a)**2 / (2 * s**2))
                interpolated_spectrum_batch[idx, gap_indices] = estimated_gaussian

                combined_indices = np.concatenate((interp_indices, gap_indices))
                combined_values = interpolated_spectrum_batch[idx, combined_indices]
                combined_relative_vel_indices = combined_indices - zero_vel_idx

                PT = np.sum(combined_values)
                if PT == 0:
                    break
                a = np.sum(combined_relative_vel_indices * combined_values) / PT
                s_squared = np.sum(combined_values * (combined_relative_vel_indices - a)**2) / PT
                s = np.sqrt(s_squared)
                s = s + s_epsilon if s < s_epsilon else s
                if s == 0:
                    break
        return interpolated_spectrum_batch

    # def _get_interpolation_width_L(self, poly_order):
    #     """
    #     Determines the interpolation half-width L based on polynomial order.
    #     This is a simplified lookup based on Table B1.
    #     In a real application, this would be derived from the filter's
    #     frequency response.

    #     Args:
    #         poly_order (int): The polynomial order used for filtering.

    #     Returns:
    #         int: The half-width L for interpolation.
    #     """
    #     if poly_order <= 3: return 1
    #     elif poly_order <= 5: return 2
    #     elif poly_order <= 8: return 3
    #     elif poly_order <= 10: return 4
    #     elif poly_order <= 13: return 5
    #     elif poly_order <= 16: return 6
    #     elif poly_order <= 19: return 7
    #     else: return 8

    def _get_interpolation_width_L(self, poly_order):
        """
        Determines the interpolation half-width L based on polynomial order.
        This is a simplified lookup based on Table B1.
        In a real application, this would be derived from the filter's
        frequency response.

        Args:
            poly_order (int): The polynomial order used for filtering.

        Returns:
            int: The half-width L for interpolation.
        """
        if poly_order <= 3: return 1
        elif poly_order <= 5: return 2
        elif poly_order <= 8: return 3
        elif poly_order <= 10: return 4
        elif poly_order <= 13: return 5
        elif poly_order <= 16: return 6
        elif poly_order <= 19: return 7
        elif poly_order <= 22: return 8
        elif poly_order <= 25: return 9
        else: return 10

    def filter_iq_data(self, i_data, q_data, cnr_db_map=None, apply_interpolation=True,
                             interpolation_E_points=3, velocity_threshold_interp=0.2):
        """
        Main method to filter I/Q time series data for ground clutter.
        This method now handles 3D input data (range, azimuth, sample)
        by flattening the first two dimensions for batch processing.

        Args:
            i_data (np.ndarray): Input in-phase time series of shape (num_ranges, num_azimuths, num_samples).
            q_data (np.ndarray): Input quadrature time series of shape (num_ranges, num_azimuths, num_samples).
            cnr_db_map (np.ndarray, optional): 2D array of CNR in dB of shape (num_ranges, num_azimuths).
                                                If None, CNR will be estimated for each (range, azimuth) cell.
            apply_interpolation (bool): Whether to apply Gaussian interpolation
                                        across the zero-velocity gap. Defaults to True.
            interpolation_E_points (int): Number of points on either side of the
                                          gap used for Gaussian interpolation (E).
                                          Defaults to 3.
            velocity_threshold_interp (float): Threshold for estimated velocity
                                              relative to Nyquist velocity. If
                                              abs(Vest/Nyq) > Vth, interpolation
                                              is not applied. Defaults to 0.2.

        Returns:
            tuple: (filtered_i, filtered_q, poly_order_map, clutter_i_trend_map,
                    clutter_q_trend_map, interpolated_spectrum_map)
                    filtered_i (np.ndarray): Clutter-suppressed I series (num_ranges, num_azimuths, num_samples).
                    filtered_q (np.ndarray): Clutter-suppressed Q series (num_ranges, num_azimuths, num_samples).
                    poly_order_map (np.ndarray): Polynomial order used for filtering for each cell (num_ranges, num_azimuths).
                    clutter_i_trend_map (np.ndarray): Estimated clutter trend for I series (num_ranges, num_azimuths, num_samples).
                    clutter_q_trend_map (np.ndarray): Estimated clutter trend for Q series (num_ranges, num_azimuths, num_samples).
                    interpolated_spectrum_map (np.ndarray): Interpolated Doppler spectrum for each cell
                                                             (num_ranges, num_azimuths, num_samples).
        """
        if i_data.shape != q_data.shape:
            raise ValueError("i_data and q_data must have the same shape.")
        if i_data.ndim != 3 or i_data.shape[2] != self.num_samples:
            raise ValueError(f"Input data must be of shape (range, azimuth, sample) "
                             f"where sample dimension is {self.num_samples}.")

        original_shape = i_data.shape
        num_ranges, num_azimuths, num_samples = original_shape
        num_cells = num_ranges * num_azimuths

        i_data_flat = i_data.reshape(num_cells, num_samples)
        q_data_flat = q_data.reshape(num_cells, num_samples)

        print(f"Processing {num_cells} cells ({num_ranges} range gates, {num_azimuths} azimuthal positions)...")

        filtered_i_flat = np.zeros_like(i_data_flat, dtype=float)
        filtered_q_flat = np.zeros_like(q_data_flat, dtype=float)
        poly_order_flat = np.zeros(num_cells, dtype=int)
        clutter_i_trend_flat = np.zeros_like(i_data_flat, dtype=float)
        clutter_q_trend_flat = np.zeros_like(q_data_flat, dtype=float)
        interpolated_spectrum_flat = np.zeros_like(i_data_flat, dtype=float)

        if cnr_db_map is None:
            print("Estimating CNR for all cells...")
            clutter_power_linear_flat = self._estimate_clutter_power_from_polyfit(i_data_flat, q_data_flat)
            total_raw_power_flat = np.mean(i_data_flat**2 + q_data_flat**2, axis=-1)
            noise_power_linear_flat = np.maximum(1e-10, total_raw_power_flat * 0.1)
            
            cnr_db_flat = np.where(noise_power_linear_flat > 1e-10,
                                   10 * np.log10(clutter_power_linear_flat / noise_power_linear_flat),
                                   -99.0)
        else:
            cnr_db_flat = cnr_db_map.flatten()

        print("Determining polynomial orders...")
        poly_order_flat = self._get_polynomial_order(cnr_db_flat)
        
        print("Performing regression filtering on I series...")
        filtered_i_flat, clutter_i_trend_flat = self._perform_regression_filter(i_data_flat, poly_order_flat)
        print("Performing regression filtering on Q series...")
        filtered_q_flat, clutter_q_trend_flat = self._perform_regression_filter(q_data_flat, poly_order_flat)

        if apply_interpolation:
            print("Applying Gaussian interpolation (if conditions met)...")
            complex_signal_flat = filtered_i_flat + 1j * filtered_q_flat

            doppler_spectrum_flat = np.abs(np.fft.fftshift(np.fft.fft(complex_signal_flat, axis=-1), axes=-1))**2

            interpolated_spectrum_flat = self._gaussian_interpolation_batch(
                doppler_spectrum_flat,
                poly_order_flat,
                E_points=interpolation_E_points,
                velocity_threshold_interp=velocity_threshold_interp
            )
        else: 
            
            complex_signal_flat = filtered_i_flat + 1j * filtered_q_flat
            interpolated_spectrum_flat = np.abs(np.fft.fftshift(np.fft.fft(complex_signal_flat, axis=-1), axes=-1))**2

        filtered_i = filtered_i_flat.reshape(original_shape)
        filtered_q = filtered_q_flat.reshape(original_shape)
        poly_order_map = poly_order_flat.reshape(original_shape[:-1])
        clutter_i_trend_map = clutter_i_trend_flat.reshape(original_shape)
        clutter_q_trend_map = clutter_q_trend_flat.reshape(original_shape)
        interpolated_spectrum_map = interpolated_spectrum_flat.reshape(original_shape) 
        return filtered_i, filtered_q, poly_order_map, clutter_i_trend_map, clutter_q_trend_map, interpolated_spectrum_map
