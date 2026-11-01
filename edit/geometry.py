# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
import numpy as np

def rotate_sweep(sweep, deg: float) -> bool:
    if sweep.azimuth is None or deg == 0.0:
        return False
    sweep.azimuth = (np.asarray(sweep.azimuth, dtype="float64") + deg) % 360.0
    return True


def offset_elevations(sweep, deg: float) -> bool:
    if deg == 0.0:
        return False
    if sweep.elevation is not None:
        sweep.elevation = np.asarray(sweep.elevation, dtype="float64") + deg
    sweep.fixed_angle = float(sweep.fixed_angle) + deg
    return True


def estimate_fixed_angle(sweep) -> float:
    if sweep.elevation is None or len(np.atleast_1d(sweep.elevation)) == 0:
        raise ValueError("This sweep has no elevation data to estimate from.")
    elev = np.asarray(sweep.elevation, dtype="float64")
    valid = elev[np.isfinite(elev) & (elev > -90.0) & (elev < 90.0)]
    if valid.size == 0:
        raise ValueError("No valid elevation values found in this sweep.")
    new_fixed = float(np.median(valid))
    sweep.fixed_angle = new_fixed
    return new_fixed