# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
def is_geolocated(volume, tol: float = 1e-6) -> bool:
    return not (abs(volume.latitude) < tol and abs(volume.longitude) < tol
               and abs(volume.altitude) < tol)


def set_location(volume, latitude: float, longitude: float, altitude: float):
    if not (-90.0 <= latitude <= 90.0):
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not (-180.0 <= longitude <= 180.0):
        raise ValueError("Longitude must be between -180 and 180 degrees.")
    volume.latitude = float(latitude)
    volume.longitude = float(longitude)
    volume.altitude = float(altitude)