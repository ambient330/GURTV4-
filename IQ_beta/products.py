"""
IQ_beta/products.py

Registry of moment products the "Open I/Q File (Beta)" dialog can render,
grouped for display and tagged with whether they require dual-pol data.

Default output field names follow the convention requested for GURT's I/Q
Beta feature:
    DBZHC / DBZVC     — reflectivity, horizontal / vertical
    VEL / VELH / VELV — velocity (see note below)
    SNRHC / SNRVC     — signal-to-noise ratio, horizontal / vertical
    DBMHC / DBMVC     — received power, horizontal / vertical
    WIDTH             — spectrum width
    ZDR / PHIDP / RHOHV — polarimetric products

Note on VEL/VELH/VELV: RVP8-style pulse-pair processing (calcIQ's
PulsePairMoments) produces exactly one Doppler velocity estimate, taken
from the H-channel autocorrelation — there is no independently computed
V-channel velocity to report. VELH and VELV are offered as alternate
*names* for that same estimate (for pipelines/scripts downstream that
expect one of those product names), not as distinct measurements. This
is spelled out again where they're assigned in iq_engine.load_iq_volume.
"""

PRODUCT_SPECS = [
    dict(key='dbz_h', group='Reflectivity', label='Reflectivity (Horizontal)',
         default_name='DBZHC', dual_pol_required=False, default_checked=True),
    dict(key='dbz_v', group='Reflectivity', label='Reflectivity (Vertical)',
         default_name='DBZVC', dual_pol_required=True, default_checked=False),

    dict(key='vel', group='Velocity', label='Velocity',
         default_name='VEL', dual_pol_required=False, default_checked=True),
    dict(key='vel_h', group='Velocity', label='Velocity (Horizontal)',
         default_name='VELH', dual_pol_required=False, default_checked=False),
    dict(key='vel_v', group='Velocity', label='Velocity (Vertical)',
         default_name='VELV', dual_pol_required=True, default_checked=False),

    dict(key='width', group='Spectrum Width', label='Spectrum Width',
         default_name='WIDTH', dual_pol_required=False, default_checked=False),

    dict(key='snr_h', group='Signal Quality', label='SNR (Horizontal)',
         default_name='SNRHC', dual_pol_required=False, default_checked=False),
    dict(key='snr_v', group='Signal Quality', label='SNR (Vertical)',
         default_name='SNRVC', dual_pol_required=True, default_checked=False),
    dict(key='dbm_h', group='Signal Quality', label='DBM (Horizontal)',
         default_name='DBMHC', dual_pol_required=False, default_checked=False),
    dict(key='dbm_v', group='Signal Quality', label='DBM (Vertical)',
         default_name='DBMVC', dual_pol_required=True, default_checked=False),

    dict(key='zdr', group='Polarimetric', label='Differential Reflectivity (ZDR)',
         default_name='ZDR', dual_pol_required=True, default_checked=False),
    dict(key='phidp', group='Polarimetric', label='Differential Phase (PHIDP)',
         default_name='PHIDP', dual_pol_required=True, default_checked=False),
    dict(key='rhohv', group='Polarimetric', label='Correlation Coefficient (RHOHV)',
         default_name='RHOHV', dual_pol_required=True, default_checked=False),
]

_BY_KEY = {spec['key']: spec for spec in PRODUCT_SPECS}


def product_by_key(key):
    return _BY_KEY.get(key)


def _grouped():
    groups = []
    by_group = {}
    for spec in PRODUCT_SPECS:
        g = spec['group']
        if g not in by_group:
            by_group[g] = []
            groups.append((g, by_group[g]))
        by_group[g].append(spec)
    return groups


# [(group_label, [spec, ...]), ...] in declaration order — what the dialog
# actually iterates over to build its checkbox/name-entry rows.
PRODUCT_GROUPS = _grouped()
