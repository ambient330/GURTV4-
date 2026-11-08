# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""One process-wide lock for every netCDF4 / HDF5 call.

The HDF5 library bundled with netCDF4 (conda, pip wheels) is not thread-safe: two threads opening files at once
produce "HDF5-DIAG: invalid location identifier" errors and can crash the program. GURT prefetches the next files
on background threads while the foreground loads or writes one, so every function that touches a netCDF / HDF5
file runs under this lock (re-entrant, so nested calls are fine)."""
import functools
import threading

NC_LOCK = threading.RLock()


def serialized(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with NC_LOCK:
            return fn(*a, **k)
    return wrapper
