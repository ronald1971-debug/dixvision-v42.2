"""Dashboard OS package (Phase 6).

Per Build Compiler Spec §6: the dashboard is a *Control Plane*, not a
restructurable UI layer. This package contains the Python control-plane
backend for the 5 immutable widgets:

* :mod:`dashboard.control_plane.mode_control_bar`        DASH-02
* :mod:`dashboard.control_plane.engine_status_grid`      DASH-EG-01
* :mod:`dashboard.control_plane.decision_trace`          DASH-04
* :mod:`dashboard.control_plane.strategy_lifecycle_panel` DASH-SLP-01
* :mod:`dashboard.control_plane.memecoin_control_panel`  DASH-MCP-01

Authority constraints (Build Compiler Spec §6 + INV-12 + INV-37):

- Dashboard *reads* engine state for display.
- Dashboard *requests* mode changes through GOV-CP-07
  ``OperatorInterfaceBridge``; never writes ledger / mode directly.
- Dashboard does not route, mutate, or otherwise transform engine
  state. It is a one-way projection plus a request seam.

The TypeScript client surface lives elsewhere; this package is the
deterministic Python control-plane logic that backs the widgets.
"""
