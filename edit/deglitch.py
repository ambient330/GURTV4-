# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""
Python translation of `deglitch_parm` (qualitycontrol.c) for use with Py-ART
radar objects.

Original algorithm (per ray/gate, iterated until stable):
  1. Skip gates that are already "bad" (masked).
  2. Build a local window of +/- `radius` gates and rays (clamped at sweep
     edges) and take the median of the *good* neighbors (self excluded).
  3. If |value - median| > threshold, the gate is a "glitch". Try to bring
     it closer to the local median by adding/subtracting 2x Nyquist
     (velocity folding correction). If a second/third field is supplied
     (the C code's prt1/prt2 case, for staggered-PRT data with separate
     short-PRT and long-PRT velocity moments), the correction switches to
     those fields' own values and Nyquist intervals in turn before giving up.
  4. If no correction gets within threshold, either delete the gate (mask
     it) or leave the original value, depending on `delete_bad`.
  5. Repeat passes until the number of glitches found stops changing, or
     until `max_pass` is reached (0 = unlimited passes until stable).

Two entry points are provided:
  - `deglitch_field` / `add_deglitched_field`: single combined field, one
    Nyquist value per ray (the C code's numprt == 1 case). This is what you
    want when a file only has one combined VEL field referenced to the
    extended Nyquist (post-processed staggered data, or plain fixed-PRF).
  - `deglitch_field_staggered` / `add_deglitched_field_staggered`: the full
    two-PRT case (C code's numprt == 2), for files that carry separate
    short-PRT and long-PRT velocity fields alongside prt/prt_ratio.

`compute_stagger_nyquist` re-implements the Nyquist math from
detect_stagger.py so both entry points can auto-derive nyquist_short,
nyquist_long, and nyquist_extended straight from the radar object.
"""

import numpy as np
from fractions import Fraction
from numba import njit

TOLERANCE = 1e-5


def compute_stagger_nyquist(radar):
    """
    Re-implementation of detect_stagger.py's logic, operating on a Py-ART
    Radar object instead of a raw netCDF4.Dataset.

    Returns a dict with keys:
        wavelength_m, prt (array), is_staggered (bool),
        m_stag, n_stag (ints, stagger ratio numerator/denominator; 1,1 if
            not staggered),
        nyquist_short, nyquist_long, nyquist_extended (floats, m/s)
    Raises ValueError if the required fields aren't present.
    """
    ip = radar.instrument_parameters or {}

    # --- wavelength ---
    wavelength_cm = 3.2  # fallback, matches detect_stagger.py
    if 'frequency' in ip:
        freq_hz = float(np.asarray(ip['frequency']['data']).flat[0])
        if np.isfinite(freq_hz) and freq_hz > 0:
            wavelength_cm = (3e8 / freq_hz) * 100.0
    wl_m = wavelength_cm / 100.0

    # --- PRT ---
    if 'prt' not in ip:
        raise ValueError("No 'prt' found in radar.instrument_parameters.")
    prt = np.ma.filled(ip['prt']['data'], np.nan).astype(np.float64)

    result = {'wavelength_m': wl_m, 'prt': prt}

    # --- stagger ratio ---
    if 'prt_ratio' not in ip:
        nyq = float(np.nanmedian(wl_m / (4.0 * prt)))
        result.update(is_staggered=False, m_stag=1, n_stag=1,
                      nyquist_short=nyq, nyquist_long=nyq,
                      nyquist_extended=nyq)
        return result

    ratio_vals = np.ma.filled(ip['prt_ratio']['data'], np.nan)
    r = float(np.nanmedian(ratio_vals))

    if abs(r - 1.0) < 0.01:
        nyq = float(np.nanmedian(wl_m / (4.0 * prt)))
        result.update(is_staggered=False, m_stag=1, n_stag=1,
                      nyquist_short=nyq, nyquist_long=nyq,
                      nyquist_extended=nyq)
        return result

    frac = Fraction(r).limit_denominator(10)
    m_stag, n_stag = frac.numerator, frac.denominator

    prt_short = prt
    prt_long = prt_short * (n_stag / m_stag)

    nyquist_short = float(np.nanmedian(wl_m / (4.0 * prt_short)))
    nyquist_long = float(np.nanmedian(wl_m / (4.0 * prt_long)))
    nyquist_extended = m_stag * nyquist_short

    result.update(is_staggered=True, m_stag=m_stag, n_stag=n_stag,
                  nyquist_short=nyquist_short, nyquist_long=nyquist_long,
                  nyquist_extended=nyquist_extended)
    return result


@njit(cache=True)
def _gap_fill_core(data, bad_val, orig_bad, n_neighbors, use_median, n_passes):
    """
    data       : 2D float64 array (nrays, ngates), gates to be filled are
                 set to bad_val (this includes gates deleted by deglitching)
    orig_bad   : 2D boolean array, same shape. True where the gate was
                 already missing/bad in the *original* data (before any
                 deglitching). These gates are never touched or filled.
    n_neighbors: int, number of nearest valid gates to use (searched
                 outward along the same ray, alternating left/right)
    use_median : bool, True = median of the neighbors found, False = mean
    n_passes   : int, number of full sweeps over every ray. Filling is
                 immediate (a filled gate becomes usable as a neighbor for
                 later gates in the same and later passes), so gates whose
                 nearest neighbors were themselves gaps on pass 1 can get
                 filled once those neighbors are resolved on a later pass.

    Returns the data array with fillable gaps filled in place (bad_val
    remains wherever orig_bad is True, or where not enough valid neighbors
    were ever found within the ray).
    """
    nrays, ngates = data.shape
    neighbors = np.empty(n_neighbors, dtype=np.float64)

    for _pass in range(n_passes):
        for ray in range(nrays):
            for gate in range(ngates):
                if orig_bad[ray, gate]:
                    continue  # never fill originally-missing/NaN gates

                val = data[ray, gate]
                if abs(val - bad_val) > TOLERANCE:
                    continue  # already valid, nothing to do

                # Search outward along this ray (range direction) for the
                # nearest valid gates, alternating left/right by distance.
                count = 0
                left = gate - 1
                right = gate + 1
                while (left >= 0 or right < ngates) and count < n_neighbors:
                    if left >= 0:
                        lv = data[ray, left]
                        if abs(lv - bad_val) > TOLERANCE:
                            neighbors[count] = lv
                            count += 1
                    if count >= n_neighbors:
                        break
                    if right < ngates:
                        rv = data[ray, right]
                        if abs(rv - bad_val) > TOLERANCE:
                            neighbors[count] = rv
                            count += 1
                    left -= 1
                    right += 1

                if count == 0:
                    continue  # no valid neighbor found yet; try next pass

                if use_median:
                    data[ray, gate] = np.median(neighbors[:count])
                else:
                    s = 0.0
                    for i in range(count):
                        s += neighbors[i]
                    data[ray, gate] = s / count

    return data


def gap_fill(masked_array, orig_mask=None, n_neighbors=3, method='median',
             n_passes=5):
    """
    Fill masked gates along each ray using the nearest valid neighboring
    gates in range, run iteratively over multiple passes.

    Parameters
    ----------
    masked_array : numpy.ma.MaskedArray, shape (nrays, ngates)
        The field to fill (typically the deglitched result).
    orig_mask : numpy.ma.MaskedArray, numpy bool array, or None
        Mask of gates that were already missing/bad in the *original*
        (pre-deglitch) data. These are always left alone -- only gaps
        that are masked in `masked_array` but NOT in `orig_mask` get
        filled. If None, every masked gate in `masked_array` is treated
        as fillable (no original-missing gates are protected).
    n_neighbors : int
        Number of nearest valid gates (searched outward along the ray,
        left/right) to use for each fill, typically 2 or 3.
    method : 'median' or 'mean'
        How to combine the neighboring good values.
    n_passes : int
        Number of full sweeps over every ray. Each pass can resolve gaps
        that only became fillable because a neighbor was filled in a
        previous pass.

    Returns
    -------
    numpy.ma.MaskedArray, same shape. Originally-missing gates and any
    gaps that never found a valid neighbor within their ray stay masked.
    """
    if method not in ('median', 'mean'):
        raise ValueError("method must be 'median' or 'mean'")

    bad_val = -9999.0
    filled = np.ma.filled(masked_array, bad_val).astype(np.float64)

    if orig_mask is None:
        orig_bad = np.zeros(filled.shape, dtype=np.bool_)
    else:
        orig_bad = np.ma.getdata(np.ma.filled(orig_mask, True)) if isinstance(
            orig_mask, np.ma.MaskedArray) else np.asarray(orig_mask)
        orig_bad = orig_bad.astype(np.bool_)

    from edit.native import native
    nat = native()
    if nat is not None:
        nan_in = np.where(np.abs(filled - bad_val) <= TOLERANCE, np.nan, filled)
        res, _ = nat.gapfill(np.ma.masked_invalid(nan_in), int(n_neighbors), method,
                             int(n_passes), protect=orig_bad)
        had = np.abs(filled - bad_val) > TOLERANCE          # gap fill only ever fills missing gates
        merged = np.where(had, filled, np.ma.getdata(res).astype(np.float64))
        return np.ma.masked_where(np.ma.getmaskarray(res) & ~had, merged)
    out = _gap_fill_core(filled, bad_val, orig_bad, int(n_neighbors),
                          method == 'median', int(n_passes))

    return np.ma.masked_where(np.abs(out - bad_val) <= TOLERANCE, out)



@njit(cache=True)
def _deglitch_core(data, bad_val, nyquist, radius, thresh, max_pass, delete_bad,
                    numprt, prt1data, prt2data, nyq1, nyq2):
    """
    data     : 2D float64 array (nrays, ngates), bad gates already set to bad_val.
               This is the field actually being edited/output.
    nyquist  : 1D float64 array of length nrays (Nyquist for `data`, C `nyq`)
    numprt   : 1 or 2. If 2, prt1data/prt2data/nyq1/nyq2 are used exactly like
               the C code's prt1parm/prt2parm branch: when folding `data`
               against its own Nyquist fails, retry using prt1data's value
               folded against nyq1, then prt2data's value folded against nyq2.
    prt1data, prt2data : 2D float64 arrays, same shape as data (dummy zeros
               array if numprt == 1; unused in that case)
    nyq1, nyq2 : 1D float64 arrays of length nrays (unused if numprt == 1)

    Returns corrected data array (same shape).
    """
    nrays, ngates = data.shape
    tempdata = data.copy()
    goodvals = np.empty(((2 * radius + 1) * (2 * radius + 1),), dtype=np.float64)

    numglitch = -1
    prevnumglitch = 0
    pas = 0

    while numglitch != prevnumglitch:
        prevnumglitch = numglitch
        numglitch = 0

        for ray in range(nrays):
            minray = ray - radius
            maxray = ray + radius
            if minray < 0:
                minray = 0
            if maxray >= nrays:
                maxray = nrays - 1

            for gate in range(ngates):
                val = data[ray, gate]
                if abs(val - bad_val) <= TOLERANCE:
                    continue

                mingate = gate - radius
                maxgate = gate + radius
                if mingate < 0:
                    mingate = 0
                if maxgate >= ngates:
                    maxgate = ngates - 1

                numgood = 0
                for locray in range(minray, maxray + 1):
                    for locgate in range(mingate, maxgate + 1):
                        if locray == ray and locgate == gate:
                            continue
                        v = data[locray, locgate]
                        if abs(v - bad_val) > TOLERANCE:
                            goodvals[numgood] = v
                            numgood += 1

                if numgood == 0:
                    continue

                median = np.median(goodvals[:numgood])

                nyq = nyquist[ray]
                origvalue = val
                mindiff = abs(origvalue - median)
                newvalue = origvalue

                if mindiff > thresh:
                    numglitch += 1
                    unfdirect = 0
                    prevunfdirect = 0
                    prt = 0
                    abort = False

                    while not abort:
                        if (unfdirect == prevunfdirect) or (prevunfdirect == 0):
                            prevunfdirect = unfdirect
                            if newvalue < median:
                                newvalue = newvalue + 2 * nyq
                                unfdirect = 1
                            else:
                                newvalue = newvalue - 2 * nyq
                                unfdirect = -1

                            newdiff = abs(newvalue - median)
                            if newdiff < mindiff:
                                tempdata[ray, gate] = newvalue
                                mindiff = newdiff
                        elif prt == 0 and numprt == 2:
                            # real two-PRT case: switch to the short-PRT field's
                            # own value and Nyquist
                            newvalue = prt1data[ray, gate]
                            prt = 1
                            unfdirect = 0
                            prevunfdirect = 0
                            nyq = nyq1[ray]
                        elif prt == 1 and numprt == 2:
                            # switch to the long-PRT field's own value and Nyquist
                            newvalue = prt2data[ray, gate]
                            prt = 2
                            unfdirect = 0
                            prevunfdirect = 0
                            nyq = nyq2[ray]
                        elif prt == 0 and numprt == 3:
                            # same-field retry: no separate PRT field available,
                            # so re-try folding the *original* value against the
                            # short Nyquist instead
                            newvalue = origvalue
                            prt = 1
                            unfdirect = 0
                            prevunfdirect = 0
                            nyq = nyq1[ray]
                        elif prt == 1 and numprt == 3:
                            # then against the long Nyquist
                            newvalue = origvalue
                            prt = 2
                            unfdirect = 0
                            prevunfdirect = 0
                            nyq = nyq2[ray]
                        else:
                            abort = True

                    if mindiff > thresh:
                        if delete_bad:
                            tempdata[ray, gate] = bad_val
                        else:
                            tempdata[ray, gate] = origvalue

        pas += 1
        data[:, :] = tempdata

        if max_pass > 0 and pas >= max_pass:
            prevnumglitch = numglitch

    return data


def _run_core_native(nat, filled, bad_val, nyquist, radius, thresh, max_pass, delete_bad,
                     numprt, prt1data, prt2data, nyq1, nyq2):
    """libgurt version of _deglitch_core, with the same in/out convention
    (float64, bad gates = bad_val). Gates the C code leaves unchanged keep
    their exact float64 input values (the C library works in float32), so
    callers that diff the result against the input see only real changes."""
    miss = np.abs(filled - bad_val) <= TOLERANCE
    x = np.where(miss, np.nan, filled)
    p1 = p2 = None
    if numprt == 2:
        p1 = np.where(np.abs(prt1data - bad_val) <= TOLERANCE, np.nan, prt1data)
        p2 = np.where(np.abs(prt2data - bad_val) <= TOLERANCE, np.nan, prt2data)
    out, _, _ = nat.deglitch(np.ma.masked_invalid(x), nyquist, int(radius), float(thresh),
                             int(max_pass), bool(delete_bad), int(numprt),
                             prt1=p1, prt2=p2, nyq1=nyq1, nyq2=nyq2)
    omask = np.ma.getmaskarray(out)
    new = np.ma.getdata(out).astype(np.float64)
    unchanged = (~omask) & (~miss) & (new == filled.astype(np.float32).astype(np.float64))
    res = np.where(unchanged, filled, new)
    res[omask] = bad_val
    return res


def _run_core(filled, bad_val, nyquist, radius, thresh, max_pass, delete_bad,
              numprt=1, prt1data=None, prt2data=None, nyq1=None, nyq2=None):
    nrays, ngates = filled.shape
    from edit.native import native
    nat = native()
    if nat is not None:
        if numprt == 2 and (prt1data is None or prt2data is None):
            nat = None
    if nat is not None:
        return _run_core_native(nat, np.asarray(filled, dtype=np.float64), bad_val, nyquist,
                                radius, thresh, max_pass, delete_bad, numprt,
                                prt1data, prt2data, nyq1, nyq2)
    if prt1data is None:
        prt1data = np.zeros((nrays, ngates), dtype=np.float64)
    if prt2data is None:
        prt2data = np.zeros((nrays, ngates), dtype=np.float64)
    if nyq1 is None:
        nyq1 = np.zeros(nrays, dtype=np.float64)
    if nyq2 is None:
        nyq2 = np.zeros(nrays, dtype=np.float64)

    return _deglitch_core(
        filled, bad_val, nyquist, int(radius), float(thresh),
        int(max_pass), bool(delete_bad), int(numprt),
        prt1data, prt2data, nyq1, nyq2
    )


def deglitch_field(radar, field_name, radius=2, thresh=15.0, max_pass=0,
                    delete_bad=True, nyquist=None, try_stagger_nyquists=None,
                    nyquist_short=None, nyquist_long=None,
                    gap_fill_enabled=False, gap_fill_n_neighbors=3,
                    gap_fill_method='median', gap_fill_n_passes=5):
    """
    Single-field deglitch (C code's numprt == 1 case, extended with an
    optional same-field multi-Nyquist retry). Use this when the radar only
    has one combined velocity field (no separate short-PRT/long-PRT
    velocity moments).

    Parameters
    ----------
    radar : pyart.core.Radar
    field_name : str
        e.g. 'VEL'
    radius : int
        Search radius in gates/rays (C `glitchradius`).
    thresh : float
        Glitch threshold, same units as the field (C `glitchthresh`).
    max_pass : int
        0 = iterate to a stable solution, >0 caps number of passes
        (C `glitchmaxpass`).
    delete_bad : bool
        If a glitch can't be corrected, mask it out rather than keep the
        original value (C `glitchdelete`).
    nyquist : float or None
        Nyquist velocity to use for the primary folding attempt. If None,
        tries radar.instrument_parameters['nyquist_velocity'] first, then
        falls back to the extended Nyquist from compute_stagger_nyquist.
    try_stagger_nyquists : bool or None
        If True, after folding against `nyquist` fails to resolve a glitch,
        retry folding the *same* original value against the short-PRT
        Nyquist, then the long-PRT Nyquist (each with its own multi-fold
        passes), before giving up. This matters for staggered-PRT data
        where residual glitches can sit at multiples of the short or long
        Nyquist rather than the extended one, and there's no separate VE1/
        VE2 field to switch to. If None (default), this is turned on
        automatically when compute_stagger_nyquist detects staggered PRT
        on the radar, and left off otherwise.
    nyquist_short, nyquist_long : float or None
        Override the short/long Nyquist used for the retries above. If
        None, derived from compute_stagger_nyquist.
    gap_fill_enabled : bool
        If True, after deglitching, fill gaps that were newly created by
        this call (deleted glitches) using the nearest 2-3 valid neighbors
        along each ray, over multiple passes. Gates that were *already*
        missing/masked in the original field (before deglitching) are
        never filled. See `gap_fill`.
    gap_fill_n_neighbors : int
        Number of nearest valid gates (along the ray) to use per fill,
        typically 2 or 3.
    gap_fill_method : 'median' or 'mean'
        How to combine neighboring good values when gap filling.
    gap_fill_n_passes : int
        Number of full sweeps over every ray during gap filling (default
        5), so gaps that only become fillable once a neighbor is itself
        filled get resolved on a later pass.

    Returns
    -------
    corrected : numpy.ma.MaskedArray, same shape as the input field.
    """
    field = radar.fields[field_name]
    data = np.ma.array(field['data'], copy=True)
    bad_val = -9999.0

    orig_bad_mask = np.ma.getmaskarray(data).copy()
    filled = data.filled(bad_val).astype(np.float64)

    stagger_info = None
    if (nyquist is None or try_stagger_nyquists is None
            or nyquist_short is None or nyquist_long is None):
        try:
            stagger_info = compute_stagger_nyquist(radar)
        except ValueError:
            stagger_info = None

    if nyquist is None:
        if (radar.instrument_parameters is not None
                and 'nyquist_velocity' in radar.instrument_parameters):
            nyq_arr = radar.instrument_parameters['nyquist_velocity']['data']
            nyquist = np.asarray(nyq_arr, dtype=np.float64)
        elif stagger_info is not None:
            nyquist = np.full(radar.nrays, stagger_info['nyquist_extended'], dtype=np.float64)
        else:
            raise ValueError(
                "No Nyquist velocity found on radar; pass `nyquist=` explicitly."
            )
    else:
        nyquist = np.full(radar.nrays, float(nyquist), dtype=np.float64)

    if try_stagger_nyquists is None:
        try_stagger_nyquists = bool(stagger_info is not None
                                     and stagger_info.get('is_staggered', False))

    if try_stagger_nyquists:
        if nyquist_short is None:
            if stagger_info is None:
                raise ValueError(
                    "try_stagger_nyquists=True but no stagger info could be "
                    "computed; pass nyquist_short/nyquist_long explicitly."
                )
            nyquist_short = stagger_info['nyquist_short']
        if nyquist_long is None:
            if stagger_info is None:
                raise ValueError(
                    "try_stagger_nyquists=True but no stagger info could be "
                    "computed; pass nyquist_short/nyquist_long explicitly."
                )
            nyquist_long = stagger_info['nyquist_long']

        nyq1 = np.full(radar.nrays, float(nyquist_short), dtype=np.float64)
        nyq2 = np.full(radar.nrays, float(nyquist_long), dtype=np.float64)

        corrected = _run_core(filled, bad_val, nyquist, radius, thresh, max_pass,
                               delete_bad, numprt=3, nyq1=nyq1, nyq2=nyq2)
    else:
        corrected = _run_core(filled, bad_val, nyquist, radius, thresh, max_pass,
                               delete_bad, numprt=1)

    result = np.ma.masked_where(np.abs(corrected - bad_val) <= TOLERANCE, corrected)

    if gap_fill_enabled:
        result = gap_fill(result, orig_mask=orig_bad_mask,
                           n_neighbors=gap_fill_n_neighbors,
                           method=gap_fill_method,
                           n_passes=gap_fill_n_passes)

    return result


def deglitch_field_staggered(radar, field_name, prt1_field, prt2_field,
                              radius=2, thresh=15.0, max_pass=0,
                              delete_bad=True, nyquist=None,
                              nyquist1=None, nyquist2=None,
                              gap_fill_enabled=False, gap_fill_n_neighbors=3,
                              gap_fill_method='median', gap_fill_n_passes=5):
    """
    Two-PRT deglitch (C code's numprt == 2 case), for staggered-PRT files
    that carry separate short-PRT and long-PRT velocity fields (e.g. VE1
    and VE2) in addition to the combined/edited field.

    Parameters
    ----------
    radar : pyart.core.Radar
    field_name : str
        The combined field being edited (C `editname`), e.g. 'VEL'.
    prt1_field, prt2_field : str
        Field names for the short-PRT and long-PRT velocity moments
        (C `prt1name`/`prt2name`), e.g. 'VE1', 'VE2'.
    radius, thresh, max_pass, delete_bad : same as deglitch_field.
    nyquist, nyquist1, nyquist2 : float or None
        Nyquist for field_name, prt1_field, prt2_field respectively. If any
        is None, it's derived from compute_stagger_nyquist: `nyquist` uses
        the extended Nyquist, `nyquist1`/`nyquist2` use the short/long
        Nyquist.
    gap_fill_enabled : bool
        If True, after deglitching, fill gaps newly created by this call
        (deleted glitches) using the nearest 2-3 valid neighbors along
        each ray, over multiple passes. Gates already missing/masked in
        the original field are never filled. See `gap_fill`.
    gap_fill_n_neighbors : int
        Number of nearest valid gates (along the ray) to use per fill.
    gap_fill_method : 'median' or 'mean'
        How to combine neighboring good values when gap filling.
    gap_fill_n_passes : int
        Number of full sweeps over every ray during gap filling.

    Returns
    -------
    corrected : numpy.ma.MaskedArray, same shape as field_name's data.
    """
    bad_val = -9999.0

    data = np.ma.array(radar.fields[field_name]['data'], copy=True)
    orig_bad_mask = np.ma.getmaskarray(data).copy()
    filled = data.filled(bad_val).astype(np.float64)

    prt1data = np.ma.filled(radar.fields[prt1_field]['data'], bad_val).astype(np.float64)
    prt2data = np.ma.filled(radar.fields[prt2_field]['data'], bad_val).astype(np.float64)

    if nyquist is None or nyquist1 is None or nyquist2 is None:
        info = compute_stagger_nyquist(radar)

    nyq = np.full(radar.nrays, float(nyquist) if nyquist is not None
                  else info['nyquist_extended'], dtype=np.float64)
    nyq1 = np.full(radar.nrays, float(nyquist1) if nyquist1 is not None
                   else info['nyquist_short'], dtype=np.float64)
    nyq2 = np.full(radar.nrays, float(nyquist2) if nyquist2 is not None
                   else info['nyquist_long'], dtype=np.float64)

    corrected = _run_core(filled, bad_val, nyq, radius, thresh, max_pass,
                           delete_bad, numprt=2, prt1data=prt1data,
                           prt2data=prt2data, nyq1=nyq1, nyq2=nyq2)

    result = np.ma.masked_where(np.abs(corrected - bad_val) <= TOLERANCE, corrected)

    if gap_fill_enabled:
        result = gap_fill(result, orig_mask=orig_bad_mask,
                           n_neighbors=gap_fill_n_neighbors,
                           method=gap_fill_method,
                           n_passes=gap_fill_n_passes)

    return result




def add_deglitched_field(radar, field_name, new_field_name=None, **kwargs):
    """
    Run deglitch_field and add the result back into the Radar object as a
    new field (so it can be plotted with pyart's RadarDisplay like any
    other field).
    """
    if new_field_name is None:
        new_field_name = field_name + '_DEGLITCHED'

    corrected = deglitch_field(radar, field_name, **kwargs)

    new_field = radar.fields[field_name].copy()
    new_field['data'] = corrected
    new_field['long_name'] = field_name + ' (deglitched)'
    radar.add_field(new_field_name, new_field, replace_existing=True)
    return new_field_name


def add_deglitched_field_staggered(radar, field_name, prt1_field, prt2_field,
                                    new_field_name=None, **kwargs):
    """
    Run deglitch_field_staggered and add the result back into the Radar
    object as a new field.
    """
    if new_field_name is None:
        new_field_name = field_name + '_DEGLITCHED'

    corrected = deglitch_field_staggered(radar, field_name, prt1_field,
                                          prt2_field, **kwargs)

    new_field = radar.fields[field_name].copy()
    new_field['data'] = corrected
    new_field['long_name'] = field_name + ' (deglitched, staggered PRT)'
    radar.add_field(new_field_name, new_field, replace_existing=True)
    return new_field_name