# GURT

GURT is a vispy + Tk radar viewer/editor (with libgurt, an optional C backend for the
despeckle / deglitch / gap-fill routines).

This release ("lite") does **not** include vortex detection/tracking or axisymmetric
analysis. Their menu entries remain as empty placeholders.

## License

GURT is free software, licensed under the **GNU General Public License, version 2 only**
(`GPL-2.0-only`). See [LICENSE](LICENSE). Each GURT source file carries an
`SPDX-License-Identifier: GPL-2.0-only` header.

## Credits and third-party code

* **DREADER** (GPL-2.0, per its SourceForge page) - the despeckle, deglitch, threshold and cross-section algorithms
  (`edit/despeckle.py`, `edit/deglitch.py`, `libgurt/`, `gui/velmag.py`, `gui/dreader_xsect.py`)
  are ports of DREADER's C code (`qualitycontrol.c` and related files).
* `readers/nexrad_level2.py` - derived from PyART, BSD-3-Clause (UChicago Argonne, LLC). Its
  original notice is kept in the file.
* `exporters/nexrad_writer.py` - 
  NEXRAD Level II writer is modified NCAR LROSE code. Full license text in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
* Runtime dependencies (vispy, matplotlib, numpy, Py-ART, netCDF4, Pillow, ...) keep their own
  licenses. 
