#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""
edit/gatefilter.py — pyart-style GateFilter for GURT Vispy.

Lets you pick a "filter field" (e.g. DBZ), a condition (e.g. "Equal to"),
and a value (e.g. 0), then masks out gates that match in one or more
"target fields" — exactly like building a pyart.filters.GateFilter and
calling gatefilter.exclude_equal('reflectivity', 0) then applying the
resulting mask to whichever fields you choose.

Both the math (GateFilter class) and the Qt dialog live here so
gurt_gui.py only needs a menu entry and a one-line callback.
"""

import numpy as np
try:
    from PyQt5 import QtWidgets, QtCore, QtGui
except ImportError:                      # headless / command-line use: only the math is needed
    import types as _types
    QtWidgets = _types.SimpleNamespace(QDialog=object)
    QtCore = QtGui = None

DLG_BG    = "#e6e6e6"
DLG_FG    = "black"
DLG_ENTRY = "#ffffff"
DLG_ABG   = "#cfcfcf"


DLG_BG    = "#e6e6e6"
DLG_FG    = "black"
DLG_ENTRY = "#ffffff"
DLG_ABG   = "#cfcfcf"


# ══════════════════════════════════════════════════════════════════════════════
# GateFilter — pyart-style condition mask builder
# ══════════════════════════════════════════════════════════════════════════════

class GateFilter:
    """Builds a boolean "excluded" mask (nrays x ngates) for one sweep,
    based on conditions evaluated against a chosen field — mirrors the
    common pyart.filters.GateFilter calls (exclude_equal, exclude_below,
    exclude_above, exclude_outside, ...).
    """

    def __init__(self, sweep):
        self.sweep = sweep
        shape = None
        for arr in sweep.fields.values():
            shape = arr.shape
            break
        if shape is None:
            raise ValueError("Sweep has no fields to build a GateFilter from.")
        self._excluded = np.zeros(shape, dtype=bool)

    def _values(self, field: str) -> np.ndarray:
        if field not in self.sweep.fields:
            raise ValueError(f"Field '{field}' not found in this sweep.")
        return np.ma.filled(self.sweep.fields[field], np.nan).astype("float64")

    @property
    def gate_excluded(self) -> np.ndarray:
        return self._excluded

    def exclude_equal(self, field: str, value: float, atol: float = 1e-6):
        v = self._values(field)
        self._excluded |= np.isclose(v, value, atol=atol, equal_nan=False)

    def exclude_not_equal(self, field: str, value: float, atol: float = 1e-6):
        v = self._values(field)
        self._excluded |= ~np.isclose(v, value, atol=atol, equal_nan=False) & np.isfinite(v)

    def exclude_below(self, field: str, value: float, inclusive: bool = False):
        v = self._values(field)
        self._excluded |= (v <= value) if inclusive else (v < value)

    def exclude_above(self, field: str, value: float, inclusive: bool = False):
        v = self._values(field)
        self._excluded |= (v >= value) if inclusive else (v > value)

    def exclude_inside(self, field: str, v1: float, v2: float):
        lo, hi = (v1, v2) if v1 <= v2 else (v2, v1)
        v = self._values(field)
        self._excluded |= (v >= lo) & (v <= hi)

    def exclude_outside(self, field: str, v1: float, v2: float):
        lo, hi = (v1, v2) if v1 <= v2 else (v2, v1)
        v = self._values(field)
        self._excluded |= (v < lo) | (v > hi)

    def exclude_invalid(self, field: str):
        """Exclude NaN / masked gates."""
        v = self._values(field)
        self._excluded |= ~np.isfinite(v)

    def apply(self, target_field: str) -> int:
        """Mask out (NaN-fill) every excluded gate in sweep.fields[target_field].

        Returns the number of gates newly masked."""
        if target_field not in self.sweep.fields:
            return 0
        data = self.sweep.fields[target_field]
        vals = np.ma.filled(data, np.nan).astype("float32", copy=True)
        n = int(np.count_nonzero(self._excluded & np.isfinite(vals)))
        vals[self._excluded] = np.nan
        self.sweep.fields[target_field] = np.ma.masked_invalid(vals)
        return n


# Condition labels shown in the dialog -> whether they need 1 or 2 values
CONDITIONS = {
    "Equal to (==)":            1,
    "Not equal to (!=)":        1,
    "Less than (<)":            1,
    "Less than or equal (<=)":  1,
    "Greater than (>)":         1,
    "Greater than or equal (>=)": 1,
    "Between (inclusive)":      2,
    "Outside range":            2,
    "Is NaN / missing":         0,
}


def build_filter(sweep, filter_field: str, condition: str,
                  value: float = 0.0, value2: float = 0.0) -> GateFilter:
    """Construct a GateFilter for *sweep* applying *condition* against
    *filter_field*. Raises ValueError on bad input."""
    gf = GateFilter(sweep)
    if condition == "Equal to (==)":
        gf.exclude_equal(filter_field, value)
    elif condition == "Not equal to (!=)":
        gf.exclude_not_equal(filter_field, value)
    elif condition == "Less than (<)":
        gf.exclude_below(filter_field, value, inclusive=False)
    elif condition == "Less than or equal (<=)":
        gf.exclude_below(filter_field, value, inclusive=True)
    elif condition == "Greater than (>)":
        gf.exclude_above(filter_field, value, inclusive=False)
    elif condition == "Greater than or equal (>=)":
        gf.exclude_above(filter_field, value, inclusive=True)
    elif condition == "Between (inclusive)":
        gf.exclude_inside(filter_field, value, value2)
    elif condition == "Outside range":
        gf.exclude_outside(filter_field, value, value2)
    elif condition == "Is NaN / missing":
        gf.exclude_invalid(filter_field)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return gf


def apply_gate_filter(sweep, filter_field: str, condition: str,
                      target_fields, value: float = 0.0, value2: float = 0.0) -> int:
    """Build the filter for *sweep* and apply it to every field in
    *target_fields*. Returns total number of gates masked (summed across
    target fields)."""
    gf = build_filter(sweep, filter_field, condition, value, value2)
    total = 0
    for tf in target_fields:
        total += gf.apply(tf)
    return total


# ══════════════════════════════════════════════════════════════════════════════
# Compound conditions (AND/OR across possibly-different fields) — used by
# the GateFilter Brush (edit/brushes.py), which can combine several
# conditions like "DBZ < 20 or VEL == NaN" instead of just one.
# ══════════════════════════════════════════════════════════════════════════════

def _condition_matches(sweep, field: str, condition: str,
                       value: float = 0.0, value2: float = 0.0) -> np.ndarray:
    """Boolean mask (same shape as the field) of gates satisfying
    *condition* against *field*'s values. This is the positive form of
    each test in CONDITIONS — unlike GateFilter.exclude_*, which always
    OR's everything it's given into one running mask, this returns a
    single condition's match mask so the caller can combine several of
    them with AND just as easily as OR."""
    if field not in sweep.fields:
        raise ValueError(f"Field '{field}' not found in this sweep.")
    v = np.ma.filled(sweep.fields[field], np.nan).astype("float64")

    if condition == "Equal to (==)":
        return np.isclose(v, value, atol=1e-6, equal_nan=False)
    elif condition == "Not equal to (!=)":
        return ~np.isclose(v, value, atol=1e-6, equal_nan=False) & np.isfinite(v)
    elif condition == "Less than (<)":
        return v < value
    elif condition == "Less than or equal (<=)":
        return v <= value
    elif condition == "Greater than (>)":
        return v > value
    elif condition == "Greater than or equal (>=)":
        return v >= value
    elif condition == "Between (inclusive)":
        lo, hi = (value, value2) if value <= value2 else (value2, value)
        return (v >= lo) & (v <= hi)
    elif condition == "Outside range":
        lo, hi = (value, value2) if value <= value2 else (value2, value)
        return (v < lo) | (v > hi)
    elif condition == "Is NaN / missing":
        return ~np.isfinite(v)
    else:
        raise ValueError(f"Unknown condition: {condition}")


def evaluate_conditions(sweep, conditions, combine: str = "or") -> np.ndarray:
    """Combine several per-field conditions into a single boolean
    "matches" mask.

    *conditions* is a list of dicts, each with keys "field", "condition",
    "value", "value2" ("value2" only matters for Between/Outside). The
    same *combine* ("and" or "or") applies across every condition in the
    list — e.g. [DBZ < 20, VEL is NaN] with combine="or" reproduces
    "DBZ < 20 or VEL == NaN"; the same two conditions with combine="and"
    reproduce a "DBZ < 20 and VEL > 90"-style compound expression (swap
    in a Greater-than condition on VEL for that one).

    Raises ValueError (propagated from _condition_matches) if a
    condition names a field this sweep doesn't have, or an unknown
    condition label. Raises ValueError if *conditions* is empty."""
    if not conditions:
        raise ValueError("Add at least one condition.")

    masks = [
        _condition_matches(sweep, c["field"], c["condition"],
                          c.get("value", 0.0), c.get("value2", 0.0))
        for c in conditions
    ]
    result = masks[0].copy()
    for m in masks[1:]:
        if combine == "and":
            result &= m
        else:
            result |= m
    return result


# ══════════════════════════════════════════════════════════════════════════════
# GateFilterDialog — Qt UI (Edit ▸ GateFilter…)
# ══════════════════════════════════════════════════════════════════════════════

class GateFilterDialog(QtWidgets.QDialog):
    """Pick a product to filter on, a condition, and which product(s) to
    mask the matching gates out of — then Apply.

    Calls back into the app via `app._do_gatefilter(filter_field, condition,
    target_fields, value, value2, scope)` so undo/refresh stay in gurt_gui.
    """

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.setWindowTitle("GateFilter")
        self.setStyleSheet(f"background-color: {DLG_BG}; color: {DLG_FG};")
        self.setModal(True)

        if app.volume is None:
            QtWidgets.QMessageBox.warning(self, "GateFilter", "No radar data loaded.")
            QtCore.QTimer.singleShot(0, self.reject)
            return

        self.fields = list(app.volume.field_names)
        if not self.fields:
            QtWidgets.QMessageBox.warning(self, "GateFilter", "No fields available.")
            QtCore.QTimer.singleShot(0, self.reject)
            return

        self._build()

    def _build(self):
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setVerticalSpacing(6)

        def label(text):
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(f"color: {DLG_FG};")
            return lbl

        grid.addWidget(label("Filter on product:"), 0, 0)
        active_panel = getattr(self.app, "_active_panel", None)
        default_field = (active_panel.field_combo.get()
                         if active_panel and
                         active_panel.field_combo.get() in self.fields
                         else self.fields[0])
        self.filter_field_combo = QtWidgets.QComboBox()
        self.filter_field_combo.addItems(self.fields)
        self.filter_field_combo.setCurrentText(default_field)
        grid.addWidget(self.filter_field_combo, 0, 1)

        grid.addWidget(label("Condition:"), 1, 0)
        self.condition_combo = QtWidgets.QComboBox()
        self.condition_combo.addItems(list(CONDITIONS.keys()))
        self.condition_combo.setCurrentText("Equal to (==)")
        self.condition_combo.currentTextChanged.connect(self._refresh_value_state)
        grid.addWidget(self.condition_combo, 1, 1)

        grid.addWidget(label("Value:"), 2, 0)
        self.value_edit = QtWidgets.QLineEdit("0.0")
        self.value_edit.setFixedWidth(90)
        grid.addWidget(self.value_edit, 2, 1)

        grid.addWidget(label("Value 2 (for between/outside):"), 3, 0)
        self.value2_edit = QtWidgets.QLineEdit("0.0")
        self.value2_edit.setFixedWidth(90)
        grid.addWidget(self.value2_edit, 3, 1)

        grid.addWidget(label("Mask out matching gates in:"), 4, 0, 1, 2)
        self.target_list = QtWidgets.QListWidget()
        self.target_list.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection)
        self.target_list.addItems(self.fields)
        self.target_list.setFixedHeight(min(6, len(self.fields)) * 20 + 6)
        grid.addWidget(self.target_list, 5, 0, 1, 2)
        # No default selection: which product(s) get masked is exactly
        # what's checked here, never anything implied by the "Filter on
        # product" choice above. (An earlier version auto-checked the
        # filter-on field here and re-synced it on every change to the
        # filter field dropdown — which silently masked the filter-on
        # field itself whenever a *different* target was also checked,
        # and wiped out manual target picks whenever the filter field
        # was changed afterward.)

        note = label("e.g. filter on DBZ, condition 'Equal to', value 0\n"
                     "→ masks out gates where DBZ == 0, but only in the "
                     "product(s) checked above\n"
                     "(checking nothing above is a no-op — the filter-on "
                     "product is never masked unless you check it too).")
        note.setStyleSheet("color: #6a6a9a; font-size: 8pt;")
        note.setWordWrap(True)
        grid.addWidget(note, 6, 0, 1, 2)

        scope_group = QtWidgets.QGroupBox("Scope")
        scope_layout = QtWidgets.QVBoxLayout(scope_group)
        self.scope_sweep_radio = QtWidgets.QRadioButton("Current sweep only")
        self.scope_volume_radio = QtWidgets.QRadioButton("All sweeps in this volume")
        self.scope_sweep_radio.setChecked(True)
        scope_layout.addWidget(self.scope_sweep_radio)
        scope_layout.addWidget(self.scope_volume_radio)
        grid.addWidget(scope_group, 7, 0, 1, 2)

        btns = QtWidgets.QHBoxLayout()
        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.clicked.connect(self._on_apply)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(apply_btn)
        btns.addWidget(close_btn)
        btns.addStretch(1)
        grid.addLayout(btns, 8, 0, 1, 2)

        self._refresh_value_state()

    def _refresh_value_state(self, *_):
        n = CONDITIONS.get(self.condition_combo.currentText(), 1)
        self.value_edit.setEnabled(n >= 1)
        self.value2_edit.setEnabled(n >= 2)

    def _on_apply(self):
        filter_field = self.filter_field_combo.currentText().strip()
        condition = self.condition_combo.currentText()
        if not filter_field:
            QtWidgets.QMessageBox.warning(self, "GateFilter", "Choose a product to filter on.")
            return
        n_vals = CONDITIONS.get(condition, 1)
        try:
            value = float(self.value_edit.text()) if n_vals >= 1 else 0.0
            value2 = float(self.value2_edit.text()) if n_vals >= 2 else 0.0
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "GateFilter", "Value(s) must be numeric.")
            return

        target_fields = [item.text() for item in self.target_list.selectedItems()]
        if not target_fields:
            QtWidgets.QMessageBox.warning(
                self, "GateFilter",
                "Choose at least one product to mask out gates in.")
            return

        scope = "volume" if self.scope_volume_radio.isChecked() else "sweep"
        if scope == "volume":
            btn = QtWidgets.QMessageBox.question(
                self, "GateFilter",
                "Apply to all sweeps in this volume?\nThis cannot be undone with Ctrl+Z.")
            if btn != QtWidgets.QMessageBox.Yes:
                return

        self.accept()
        self.app._do_gatefilter(filter_field, condition, target_fields,
                                value, value2, scope)