"""
IQ_beta — "Open I/Q File (Beta)" feature for GURT Vispy.

Loads Vaisala rvp8/rvpts I/Q dumps (vaisalaIQ.py), computes pulse-pair
moments via calcIQ.IQMoments at a user-chosen azimuth resolution
(optionally with GRF ground-clutter filtering), and renders the result
through GURT's existing PPI pipeline exactly like any other loaded file.

RadarKit (.rkc) I/Q input is not supported yet — Vaisala only, for now.

Wired into gurt_gui.py's File menu as:

    File ▸ Open I/Q File (Beta)…
        -> GURTVispy.open_iq_beta_file()
        -> from IQ_beta import IQBetaDialog; IQBetaDialog(self)

Layout:
    products.py   — the moment-product registry (names, groups, dual-pol
                     requirements) shown in the dialog.
    iq_engine.py  — file loading + moment computation + duck-typed
                     RadarVolume/SweepData construction. No Tkinter here;
                     safe to use headlessly (e.g. from a script).
    iq_dialog.py  — the Tk dialog UI and the install_iq_volume() glue
                     that hands a finished volume back to a running app.
"""

from .iq_dialog import IQBetaDialog, install_iq_volume
from .iq_engine import load_iq_volume, IQLoadError, RadarVolume, SweepData, is_koun_file
from .products import PRODUCT_SPECS, PRODUCT_GROUPS, product_by_key

__all__ = [
    "IQBetaDialog", "install_iq_volume",
    "load_iq_volume", "IQLoadError", "RadarVolume", "SweepData", "is_koun_file",
    "PRODUCT_SPECS", "PRODUCT_GROUPS", "product_by_key",
]
