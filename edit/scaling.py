# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
import numpy as np

def offset_scale_field(sweep, fname: str, scale: float, offset: float) -> bool:
    if fname not in sweep.fields:
        return False
    data = sweep.fields[fname]
    new_data = np.ma.filled(data, np.nan).astype("float64") * scale + offset
    sweep.fields[fname] = np.ma.masked_invalid(new_data).astype("float32")
    return True


def offset_scale_volume(sweeps, fname: str, scale: float, offset: float) -> int:
    n = 0
    for sw in sweeps:
        if offset_scale_field(sw, fname, scale, offset):
            n += 1
    return n