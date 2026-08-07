"""
PINAGMM Run Callbacks
=====================
Handles Predict / Simulate / Clear and sidebar UI logic.

Data flow:
  Predict  → pred-store  (JSON-serialised DataFrame via pandas)
  Simulate → sim-store   (JSON dict with serialised arrays + pre-computed spectra)

sim-store structure:
  {
    "major": [ {gm_dict}, ... ],
    "inter": [ {gm_dict}, ... ],
    "vert":  [ {gm_dict}, ... ],
    "spec_T": [...],        # shared period array for spectra
    "dt": float
  }

Each gm_dict:
  {
    "t":   [...], "dt": float,
    "ac":  [[...], ...],  # shape (n_sim, npts)
    "vel": [[...], ...],
    "disp":[[...], ...],
    "freq":[...],
    "fas": [[...], ...],
    "sa":  [[...], ...],  # PRE-COMPUTED response spectra — shape (n_sim, n_periods)
  }
"""

from __future__ import annotations

import json
import traceback

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import ALL, Input, Output, State, callback_context, html, no_update
from dash.exceptions import PreventUpdate

from pinagmm import PINAGMM
from pinagmm.core.variables import yvars

# Spectral periods pre-computed (80 log-spaced from 0.01 s to 4 s)
SPEC_PERIODS = np.logspace(-2, np.log10(4.0), 80)

# ── Module-level lazy singleton ───────────────────────────────────────────────
_gmm_instance: PINAGMM | None = None


def _get_gmm() -> PINAGMM:
    global _gmm_instance
    if _gmm_instance is None:
        _gmm_instance = PINAGMM()
    return _gmm_instance


_G_CMSS = 980.665


# ── Serialisation helpers ─────────────────────────────────────────────────────
def _serialise_gm(gm) -> dict:
    """Convert a GroundMotion object to a JSON-safe dict, pre-computing spectra."""
    ac = np.atleast_2d(gm.ac) / _G_CMSS  # cm/s² → g
    vel = np.atleast_2d(gm.vel)  # cm/s
    disp = np.atleast_2d(gm.disp)  # cm
    fas = np.atleast_2d(gm.fas) / _G_CMSS  # (cm/s²)·s → g·s

    # Pre-compute response spectra so callbacks don't need to re-instantiate GroundMotion
    try:
        _, _, sa = gm.response_spectra(SPEC_PERIODS)
        sa = np.atleast_2d(sa) / _G_CMSS  # cm/s² → g
    except Exception:
        # Fallback: zeros if spectra computation fails
        sa = np.zeros((ac.shape[0], len(SPEC_PERIODS)))

    return {
        "t": gm.t.tolist(),
        "dt": float(gm.dt),
        "ac": ac.tolist(),
        "vel": vel.tolist(),
        "disp": disp.tolist(),
        "freq": gm.freq.tolist(),
        "fas": fas.tolist(),
        "sa": sa.tolist(),  # (n_sim, n_periods) in g
    }


def _serialise_ts_list(gm_list: list) -> list[dict]:
    return [_serialise_gm(gm) for gm in gm_list]


# ─── Helper: build conditional row DOM elements ───────────────────────────────
def _build_cond_rows(cond_data: list) -> list:
    """Return html/dbc elements for the cond-rows-container."""

    def _nice(v):
        c_map = {"M": "Major", "I": "Interm.", "V": "Vertical"}
        if "PGV" in v:
            comp = v.split("_")[0]
            return f"PGV [{c_map.get(comp, comp)}]"
        if "Sa" in v:
            parts = v.split("_")
            comp, per = parts[0], parts[2]
            return f"Sa(T={per}s) [{c_map.get(comp, comp)}]"
        return v

    im_opts = [
        {"label": _nice(v), "value": v}
        for v in yvars
        if any(s in v for s in ("PGV", "Sa"))
    ]

    rows = []
    for i, row in enumerate(cond_data):
        rows.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                f"Condition {i + 1}",
                                style={
                                    "fontSize": "0.65rem",
                                    "color": "var(--sb-text-muted)",
                                    "fontWeight": "700",
                                    "letterSpacing": "0.05em",
                                    "textTransform": "uppercase",
                                },
                            ),
                            dbc.Button(
                                html.I(className="fas fa-times"),
                                id={"type": "btn-remove-cond", "index": i},
                                size="sm",
                                style={
                                    "background": "transparent",
                                    "border": "none",
                                    "color": "#fca5a5",
                                    "padding": "0 2px",
                                    "cursor": "pointer",
                                    "fontSize": "0.75rem",
                                },
                                n_clicks=0,
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "marginBottom": "5px",
                        },
                    ),
                    dbc.Select(
                        id={"type": "cond-im-select", "index": i},
                        options=im_opts,
                        value=row.get("im", "M_Sa_1"),
                        size="sm",
                        style={"marginBottom": "5px"},
                    ),
                    dbc.Input(
                        id={"type": "cond-val-input", "index": i},
                        type="number",
                        value=row.get("val", 0.9),
                        min=1e-4,
                        step=0.01,
                        placeholder="Target value (e.g. g)",
                        size="sm",
                        debounce=True,
                    ),
                ],
                className="cond-row-card",
            )
        )
    return rows


# ── Private helpers ───────────────────────────────────────────────────────────
def _parse_conditions(cond_data: list) -> dict | None:
    if not cond_data:
        return None
    conds = {}
    for row in cond_data:
        im = row.get("im")
        val = row.get("val")
        if im and val is not None:
            try:
                conds[str(im)] = float(val)
            except (TypeError, ValueError):
                pass
    return conds if conds else None


def _n_sample_from_mode(mode: str, nsamples) -> int:
    """
    Return n_sample count:
    – median mode → 0 (deterministic median prediction, 1 row)
    – sample mode → N (N random samples drawn from GMM distribution)
    """
    if mode == "sample":
        return max(1, int(nsamples or 10))
    return 0


# ── Busy / idle indicator helpers ─────────────────────────────────────────
# Note: 'Computing...' is set by a clientside callback in app.py (fires instantly
# in-browser on button click). Server callbacks only need to signal completion.
_DONE_TEXT = "Computed"
_DONE_CLASS = "status-pill done"
_ERR_CLASS = "status-pill error"


# ── Callback registration ─────────────────────────────────────────────────────
def register_run_callbacks(app) -> None:

    # ── 0. Toggle sample-count input visibility based on pred-mode ────────────
    @app.callback(
        Output("sb-nsamples-div", "style"),
        Input("sb-pred-mode", "value"),
        prevent_initial_call=False,
    )
    def toggle_nsamples(mode):
        if mode == "sample":
            return {"display": "block"}
        return {"display": "none"}

    # ── 1. Add or remove a conditional row ───────────────────────────────────
    @app.callback(
        Output("cond-store", "data"),
        Output("cond-rows-container", "children"),
        Input("btn-add-cond", "n_clicks"),
        Input({"type": "btn-remove-cond", "index": ALL}, "n_clicks"),
        State("cond-store", "data"),
        prevent_initial_call=True,
    )
    def manage_cond_rows(add_clicks, remove_clicks_list, cond_data):
        cond_data = list(cond_data or [])
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_id == "btn-add-cond":
            cond_data.append({"im": "M_Sa_1", "val": 0.9})
        else:
            try:
                t_dict = json.loads(triggered_id)
                if t_dict.get("type") == "btn-remove-cond":
                    idx = int(t_dict["index"])
                    if 0 <= idx < len(cond_data):
                        cond_data = [c for i, c in enumerate(cond_data) if i != idx]
            except Exception:
                pass
        return cond_data, _build_cond_rows(cond_data)

    # ── 2. Sync cond-store when selects / inputs change ──────────────────────
    @app.callback(
        Output("cond-store", "data", allow_duplicate=True),
        Input({"type": "cond-im-select", "index": ALL}, "value"),
        Input({"type": "cond-val-input", "index": ALL}, "value"),
        State("cond-store", "data"),
        prevent_initial_call=True,
    )
    def sync_cond_values(im_values, val_values, cond_data):
        cond_data = list(cond_data or [])
        for i, (im, val) in enumerate(zip(im_values or [], val_values or [])):
            if i < len(cond_data):
                if im is not None:
                    cond_data[i]["im"] = im
                if val is not None:
                    try:
                        cond_data[i]["val"] = float(val)
                    except (TypeError, ValueError):
                        pass
        return cond_data

    # ── 3. Predict ────────────────────────────────────────────────────────────
    @app.callback(
        Output("pred-store", "data"),
        Output("status-pill", "children"),
        Output("status-pill", "className"),
        Input("btn-predict", "n_clicks"),
        State("sb-mw", "value"),
        State("sb-ztor", "value"),
        State("sb-rrup", "value"),
        State("sb-vs30", "value"),
        State("sb-fm", "value"),
        State("sb-pred-mode", "value"),
        State("sb-nsamples", "value"),
        State("cond-store", "data"),
        prevent_initial_call=True,
    )
    def on_predict(n_clicks, mw, ztor, rrup, vs30, fm, pred_mode, nsamples, cond_data):
        if not n_clicks:
            raise PreventUpdate
        try:
            gmm = _get_gmm()
            conditions = _parse_conditions(cond_data)
            n_sample = _n_sample_from_mode(pred_mode, nsamples)
            pred = gmm.predict(
                Mw=float(mw or 6.5),
                Ztor=float(ztor or 3.0),
                Rrup=float(rrup or 15.0),
                Vs30=float(vs30 or 800.0),
                Fm=str(fm or "0"),
                n_sample=n_sample,
                conditions=conditions,
            )
            return pred.to_json(orient="split"), _DONE_TEXT, _DONE_CLASS
        except Exception as exc:
            traceback.print_exc()
            return no_update, f"Error: {str(exc)[:48]}", _ERR_CLASS

    # ── 4. Simulate ───────────────────────────────────────────────────────────
    @app.callback(
        Output("sim-store", "data"),
        Output("status-pill", "children", allow_duplicate=True),
        Output("status-pill", "className", allow_duplicate=True),
        Input("btn-simulate", "n_clicks"),
        State("sb-mw", "value"),
        State("sb-ztor", "value"),
        State("sb-rrup", "value"),
        State("sb-vs30", "value"),
        State("sb-fm", "value"),
        State("sb-pred-mode", "value"),
        State("sb-nsamples", "value"),
        State("sb-nsim", "value"),
        State("sb-dt", "value"),
        State("cond-store", "data"),
        prevent_initial_call=True,
    )
    def on_simulate(
        n_clicks, mw, ztor, rrup, vs30, fm, pred_mode, nsamples, nsim, dt, cond_data
    ):
        if not n_clicks:
            raise PreventUpdate
        try:
            gmm = _get_gmm()
            conditions = _parse_conditions(cond_data)
            n_samples = _n_sample_from_mode(pred_mode, nsamples)

            ts_m_list, ts_i_list, ts_v_list = gmm.simulate(
                Mw=float(mw or 6.5),
                Ztor=float(ztor or 3.0),
                Rrup=float(rrup or 15.0),
                Vs30=float(vs30 or 800.0),
                Fm=str(fm or "0"),
                n_samples=n_samples,
                n_simulations=int(nsim or 1),
                dt=float(dt or 0.005),
                conditions=conditions,
            )
            sim_data = {
                "major": _serialise_ts_list(ts_m_list),
                "inter": _serialise_ts_list(ts_i_list),
                "vert": _serialise_ts_list(ts_v_list),
                "dt": float(dt or 0.005),
                "spec_T": SPEC_PERIODS.tolist(),
            }
            return json.dumps(sim_data), _DONE_TEXT, _DONE_CLASS
        except Exception as exc:
            traceback.print_exc()
            return no_update, f"Error: {str(exc)[:48]}", _ERR_CLASS

    # ── 5. Clear ─────────────────────────────────────────────────────────────
    @app.callback(
        Output("pred-store", "data", allow_duplicate=True),
        Output("sim-store", "data", allow_duplicate=True),
        Output("status-pill", "children", allow_duplicate=True),
        Output("status-pill", "className", allow_duplicate=True),
        Input("btn-clear", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_clear(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return None, None, "Ready", "status-pill"
