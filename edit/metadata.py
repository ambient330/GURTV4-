# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
def set_radar_name(volume, name: str):
    """
    Rename the radar/site/instrument for a volume and every sweep in it.

    Mirrors ``geolocation.set_location``: validates, then stamps the new
    value onto the volume and cascades it to each sweep so downstream
    code (title bar, CfRadial/NEXRAD export, per-sweep metadata) all
    agree on the name.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Radar name cannot be empty.")
    volume.radar_name = name
    for sw in volume.sweeps:
        sw.radar_name = name