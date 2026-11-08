"""
IQ_beta/iq_engine.py

Core engine behind "Open I/Q File (Beta)": loads a Vaisala rvp8/rvpts I/Q
dump (vaisalaIQ.py), runs calcIQ.IQMoments on it at a user-chosen azimuth
resolution (optionally with GRF ground-clutter filtering), derives the
extra H/V-named products the dialog offers, and packages the result as a
duck-typed RadarVolume/SweepData pair — same attribute shape as the
RadarVolume/SweepData classes in main.py — so gurt_gui.py's existing PPI
pipeline can render it exactly like any other loaded file.

Vaisala-only for now. RadarKit (.rkc) I/Q input isn't wired up here yet;
add a second adapter call (calcIQ already supports rkcfile via its own
adapter) when that's ready, alongside a format-detection step in
iq_dialog.py's file browser.
"""

import os
import sys
from datetime import datetime

import numpy as np

# calcIQ.py / vaisalaIQ.py live at the repo root, one level up from this
# package — same sys.path fallback calcIQ.py itself uses to reach `qc`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from vaisalaIQ import vaisalafile, FileFormatError  # noqa: E402
from calcIQ import IQMoments  # noqa: E402

from .products import product_by_key  # noqa: E402


class IQLoadError(Exception):
    """Raised for any failure between opening the file and producing a
    renderable volume. str(e) is shown to the user as-is, so messages
    here are written to be read directly in a dialog box."""
    pass


class SweepData:
    """Duck-typed stand-in for main.py's SweepData. Kept local to IQ_beta
    (instead of importing main.py) to avoid a circular import — main.py
    imports gurt_gui, and gurt_gui only imports IQ_beta lazily, inside the
    File ▸ Open I/Q File (Beta)… handler. gurt_gui.py's rendering code
    never does isinstance() checks against RadarVolume/SweepData, only
    attribute access, so this works as a drop-in."""
    def __init__(self):
        self.azimuth = None
        self.elevation = None
        self.ranges = None
        self.fixed_angle = 0.0
        self.fields = {}
        self.radar_name = "Unknown"
        self.scan_time = None
        self.scan_type = "ppi"
        self.instrument_parameters = {}


class RadarVolume:
    """Duck-typed stand-in for main.py's RadarVolume — see SweepData
    docstring for why this isn't just imported from main.py."""
    def __init__(self):
        self.sweeps = []
        self.field_names = []
        self.radar_name = "Unknown"
        self.scan_time = None
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0
        self.instrument_parameters = {}


def is_koun_file(path, instrument_name=None):
    """Best-effort heuristic for the KOUN disclaimer: True if 'koun'
    shows up in the filename or the site name IQMoments resolved from the
    header. Used only to make the dialog's warning a bit more pointed —
    the general KOUN caveat is shown regardless of this check."""
    name = os.path.basename(path).lower()
    if 'koun' in name:
        return True
    if instrument_name and 'koun' in str(instrument_name).lower():
        return True
    return False


def _noise_floor_dbm(header):
    """Decode rvp8PulseInfo's fNoiseDBm the same way
    calcIQ._extract_from_vaisalafile does internally. IQMoments doesn't
    hand the value it used back out on the returned object, and DBMHC/
    DBMVC need it (DBM = SNR_dB + noise_floor_dBm), so it's re-derived
    here directly from the file's own header instead of reaching into
    calcIQ's private helpers.
    """
    noise_dbm = header.get('fNoiseDBm')
    if isinstance(noise_dbm, list) and len(noise_dbm) >= 2:
        return float(noise_dbm[0]), float(noise_dbm[1])
    if noise_dbm is not None:
        v = float(noise_dbm[0]) if isinstance(noise_dbm, list) else float(noise_dbm)
        return v, v
    return 0.0, 0.0


def load_iq_volume(path, selected_keys, az_deg=1.0, clutter_filter=False,
                    progress_cb=None, verbose=False):
    """
    Load a Vaisala I/Q file and compute the requested moment products.

    Parameters
    ----------
    path : str
        Path to a .iq / .iq.bz2 (rvp8PulseInfo/rvptsPulseInfo) file.
    selected_keys : dict
        {product_key: output_field_name}, one entry per product the user
        checked in the dialog. product_key must be one of the keys in
        IQ_beta.products.PRODUCT_SPECS; output_field_name is whatever the
        user typed (or left as the default) and becomes both the key in
        sweep.fields and the name shown in GURT's field picker.
    az_deg : float
        Azimuth binning resolution in degrees — IQMoments' azDeg. This is
        the "render at 0.5 degrees" resolution control from the dialog.
    clutter_filter : bool
        If True, runs calcIQ's Global Regression Filter (GRF, Hubbert et
        al. 2025 — qc.clutter_filter.GroundClutterFilter) on the I/Q
        before moments are formed. Passed through as
        IQMoments(clutterFilter=True), which is the flag that actually
        gates calcIQ's GRF code path (its docstring also mentions
        'fft'/'regression'/'cpa' string modes, but only the `is True`
        GRF path is implemented right now).
    progress_cb : callable(str) or None
        Optional status-string callback, called from whatever thread
        load_iq_volume runs on — the caller is responsible for marshaling
        it back to the UI thread if needed (iq_dialog.py does this).
    verbose : bool
        Forwarded to vaisalafile/IQMoments for their own progress prints.

    Returns
    -------
    (RadarVolume, list[str])
        The renderable volume, and a list of human-readable warnings
        (e.g. products skipped because the file wasn't dual-pol).

    Raises
    ------
    IQLoadError
        On any failure reading the file or computing moments, or if none
        of the selected products could be produced at all.
    """
    def _progress(msg):
        if progress_cb:
            progress_cb(msg)

    if not selected_keys:
        raise IQLoadError("No products were selected to render.")

    _progress("Reading pulses…")
    try:
        vf = vaisalafile(path, verbose=verbose)
    except FileFormatError as e:
        raise IQLoadError(f"Not a recognized Vaisala I/Q file:\n{e}") from e
    except Exception as e:
        raise IQLoadError(f"Failed to read file:\n{e}") from e

    if not vf.pulses:
        raise IQLoadError("File contains no pulses.")

    dual_pol = vf.isDualPol()

    warnings = []
    active = {}
    for key, out_name in selected_keys.items():
        spec = product_by_key(key)
        if spec is None:
            continue
        if spec['dual_pol_required'] and not dual_pol:
            warnings.append(
                f"Skipped {spec['label']} ({out_name}) — file is single-polarization.")
            continue
        active[key] = out_name

    if not active:
        raise IQLoadError(
            "None of the selected products could be produced — this file "
            "is single-polarization and every checked product requires "
            "dual-pol data.")

    _progress("Computing moments (this can take a while)…")
    try:
        moments = IQMoments(
            vf,
            azDeg=float(az_deg),
            clutterFilter=bool(clutter_filter),
            fixTransitionRays=True,   # always used for reshaping, per GURT IQ Beta convention
            verbose=verbose,
        )
    except Exception as e:
        raise IQLoadError(f"Moment computation failed:\n{e}") from e

    noise_h_dB, noise_v_dB = _noise_floor_dbm(vf.header)

    fields = {}

    def _want(key):
        return key in active

    if _want('dbz_h'):
        fields[active['dbz_h']] = moments.DBZ

    if _want('dbz_v'):
        if moments.ZDR is not None:
            # Zv(dBZ) = Zh(dBZ) - Zdr(dB) — standard relation, both sides
            # already carry the same zcal, so it cancels cleanly.
            fields[active['dbz_v']] = moments.DBZ - moments.ZDR
        else:
            warnings.append(f"Skipped {active['dbz_v']} — ZDR unavailable.")

    if _want('vel'):
        fields[active['vel']] = moments.VEL

    if _want('vel_h'):
        fields[active['vel_h']] = moments.VEL

    if _want('vel_v'):
        # See the module docstring in products.py: this is the same
        # H-channel pulse-pair velocity estimate as VEL/VELH, exposed
        # under the VELV name for downstream consumers that expect it —
        # not an independently measured V-channel velocity.
        fields[active['vel_v']] = moments.VEL

    if _want('width'):
        fields[active['width']] = moments.WIDTH

    if _want('snr_h'):
        fields[active['snr_h']] = moments.SNRH

    if _want('snr_v'):
        if moments.SNRV is not None:
            fields[active['snr_v']] = moments.SNRV
        else:
            warnings.append(f"Skipped {active['snr_v']} — SNRV unavailable.")

    if _want('dbm_h'):
        fields[active['dbm_h']] = moments.SNRH + noise_h_dB

    if _want('dbm_v'):
        if moments.SNRV is not None:
            fields[active['dbm_v']] = moments.SNRV + noise_v_dB
        else:
            warnings.append(f"Skipped {active['dbm_v']} — SNRV unavailable.")

    if _want('zdr'):
        if moments.ZDR is not None:
            fields[active['zdr']] = moments.ZDR
        else:
            warnings.append(f"Skipped {active['zdr']} — ZDR unavailable.")

    if _want('phidp'):
        if moments.PHIDP is not None:
            fields[active['phidp']] = moments.PHIDP
        else:
            warnings.append(f"Skipped {active['phidp']} — PHIDP unavailable.")

    if _want('rhohv'):
        if moments.RHOHV is not None:
            fields[active['rhohv']] = moments.RHOHV
        else:
            warnings.append(
                f"Skipped {active['rhohv']} — RHOHV unavailable (requires nLags == 1).")

    if not fields:
        raise IQLoadError(
            "No products could be computed for this file.\n\n" + "\n".join(warnings))

    sw = SweepData()
    sw.azimuth = moments.azimuth
    sw.elevation = moments.elevation
    sw.ranges = moments.range_m
    sw.fixed_angle = (float(np.nanmedian(moments.elevation))
                       if moments.elevation is not None and len(moments.elevation)
                       else 0.0)
    sw.radar_name = moments.instrument_name
    sw.scan_type = "rhi" if moments.is_rhi else "ppi"
    sw.fields = {name: np.ma.masked_invalid(np.asarray(arr, dtype=np.float32))
                 for name, arr in fields.items()}

    if moments.time is not None and len(moments.time):
        try:
            sw.scan_time = datetime.utcfromtimestamp(float(moments.time[0]))
        except (ValueError, OSError, OverflowError):
            sw.scan_time = None

    wavelength_m = moments.wavelength_m
    prt_s = moments.prt_s
    if wavelength_m and prt_s:
        nyq = 0.25 * wavelength_m / prt_s
        sw.instrument_parameters['nyquist_velocity'] = {
            'data': np.array([nyq], dtype='float32')}
        sw.instrument_parameters['prt'] = {
            'data': np.array([prt_s], dtype='float64')}
        freq_hz = 299792458.0 / wavelength_m
        sw.instrument_parameters['frequency'] = {
            'data': np.array([freq_hz], dtype='float64')}

    vol = RadarVolume()
    vol.sweeps = [sw]
    vol.field_names = list(fields.keys())
    vol.radar_name = moments.instrument_name
    vol.scan_time = sw.scan_time
    vol.latitude = moments.lat or 0.0
    vol.longitude = moments.lon or 0.0
    vol.altitude = moments.alt or 0.0
    vol.instrument_parameters = dict(sw.instrument_parameters)

    _progress("Done.")
    return vol, warnings
