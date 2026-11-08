# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""Optional libgurt (C) acceleration.

`native()` returns the gurt_native module when the shared library is built and
loadable, else None, so every caller keeps its pure-Python path as a fallback.
Set GURT_NO_NATIVE=1 to force the Python code (handy for comparing results).
Build the library with `make` in GURTv4/libgurt/.
"""
import os
import sys

_state = {"mod": None, "tried": False}


def native():
    if os.environ.get("GURT_NO_NATIVE"):
        return None
    if _state["tried"]:
        return _state["mod"]
    _state["tried"] = True
    here = os.path.dirname(os.path.abspath(__file__))
    pydir = os.path.join(here, "..", "libgurt", "python")
    if os.path.isdir(pydir) and pydir not in sys.path:
        sys.path.append(pydir)
    try:
        import gurt_native
        if gurt_native.available():
            _state["mod"] = gurt_native
            print(f"[GURT] libgurt {gurt_native.version()} loaded: C despeckle / deglitch / gap fill / unfold / tormodel / axisymmetric", file=sys.stderr)
        else:
            print("[GURT] libgurt library not found for this platform (see libgurt/README.md); using the Python versions", file=sys.stderr)
    except Exception as e:                       # missing module, bad binary, ...
        print(f"[GURT] libgurt unavailable ({e}); using the Python versions", file=sys.stderr)
    return _state["mod"]


def status():
    """Short text for the UI: 'C' or 'Python'."""
    return "C" if native() is not None else "Python"
