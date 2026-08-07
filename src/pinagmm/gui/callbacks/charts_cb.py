"""
PINAGMM Chart Callbacks
=======================

Section-switcher callbacks (show/hide one card at a time):
  - Prediction tab:   pred-view       → spectra | ims
  - Spectra tab:      simsa-comp-view → all | major | inter | vert
  - FAS tab:          fas-comp-view   → all | major | inter | vert

Time Series:
  - ts-channel radio → update graph-timeseries (combined 3-row subplot)

All CSV download callbacks with UTF-8 BOM byte stream.
Full IM table includes ALL spectral accelerations (PGA + 15 periods + PGV) with 3 rows per sample.
DataTable container forced to 100% max-width with explicit horizontal scrollbar.

Strict plotting rules (zero fake averaging or confusing geomeans):
  - Median mode (1 row): bold solid lines with markers.
  - Sampling mode (N rows): all N rows plotted individually;
    first realization is solid + opaque (legend entry with "(solid = first, cloud = rest)" hint),
    remaining realizations are semi-transparent cloud lines of the same color — no mixing.

Units:
  - Time series in GUI/exports: ac (g), vel (cm/s), disp (cm)
  - Fourier Amplitude Spectra in GUI/exports: FAS (g·s)
  - Response Spectra in GUI/exports: Sa (g)
  - GMM outputs PGV in normalised g-seconds. Multiply by 980.665 → cm/s.
"""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, dcc, html, dash_table
from dash.exceptions import PreventUpdate

from ..components.chart_helpers import (
    COMP_PALETTE,
    COMP_PALETTE_FAINT,
    GRAPH_CONFIG,
    PINAGMM_TEMPLATE,
    make_fig,
)
from pinagmm.core.variables import periods as SA_PERIODS

_G_CMSS = 980.665
_COMP_KEYS = [
    ("Major", "major", "M"),
    ("Intermediate", "inter", "I"),
    ("Vertical", "vert", "V"),
]
_SHOW = {"display": "block", "width": "100%"}
_HIDE = {"display": "none"}

# Cloud trace style — visible but clearly secondary
_CLOUD_OPACITY = 0.55
_CLOUD_WIDTH = 1.2
_CLOUD_ALPHA = 0.55   # rgba alpha for COMP_PALETTE_FAINT format string

# Solid (first realization) style
_SOLID_OPACITY = 1.0
_SOLID_WIDTH_LINE = 2.0
_SOLID_WIDTH_SUBPLOT = 1.8


def _send_csv_with_bom(df: pd.DataFrame, filename: str):
    """Return a dcc.send_bytes object formatted with UTF-8 BOM for clean Excel export."""
    csv_str = df.to_csv(index=False)
    csv_bytes = "\ufeff".encode("utf-8") + csv_str.encode("utf-8")
    return dcc.send_bytes(csv_bytes, filename)


def _send_text_with_bom(text: str, filename: str):
    """Return a dcc.send_bytes object for a pre-built CSV string."""
    csv_bytes = "\ufeff".encode("utf-8") + text.encode("utf-8")
    return dcc.send_bytes(csv_bytes, filename)


def make_empty_fig(message: str = "No data computed yet.") -> go.Figure:
    """Return a clean empty figure without giant log axes or grids."""
    fig = go.Figure()
    fig.update_layout(
        template=PINAGMM_TEMPLATE,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=12, color="#94a3b8", family="Inter, sans-serif"),
            )
        ],
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
#  Section-switcher helpers
# ═══════════════════════════════════════════════════════════════════


def _section_styles(active: str, keys) -> list[dict]:
    return [_SHOW if k == active else _HIDE for k in keys]


# ═══════════════════════════════════════════════════════════════════
#  Legend annotation helper
# ═══════════════════════════════════════════════════════════════════


def _add_cloud_legend_note(fig: go.Figure, multi: bool) -> go.Figure:
    """
    When multi=True, add a compact paper-space annotation explaining the
    solid vs. cloud convention — legend names remain short.
    """
    if not multi:
        return fig
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.0,
        xanchor="left",
        yanchor="bottom",
        text="solid = 1st realization · cloud = remaining",
        showarrow=False,
        font=dict(size=9, color="#94a3b8", family="Inter, sans-serif"),
        bgcolor="rgba(255,255,255,0.0)",
        borderpad=2,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
#  Figure & Table builders
# ═══════════════════════════════════════════════════════════════════


def fig_pred_spectra(pred_df: pd.DataFrame, n_rows: int) -> go.Figure:
    """
    GMM-predicted response spectra plot.
      • Median mode (n_rows == 1): bold solid line with markers.
      • Sample mode (n_rows > 1): realization 1 solid, realizations 2+ as cloud
        (no averaging; every row plotted individually at full resolution).
    """
    fig = make_fig("Period (s)", "Spectral Acceleration (g)", log_x=True, log_y=True)
    multi = n_rows > 1

    for comp, _, prefix in _COMP_KEYS:
        color = COMP_PALETTE[comp]
        faint = COMP_PALETTE_FAINT[comp].format(a=_CLOUD_ALPHA)
        pga_col = f"{prefix}_Sa_0"

        # Build x axis
        x = []
        if pga_col in pred_df.columns:
            x.append(0.01)
        for T in SA_PERIODS:
            if f"{prefix}_Sa_{T:g}" in pred_df.columns:
                x.append(float(T))
        if not x:
            continue

        def _row_y(row, _prefix=prefix, _pga_col=pga_col):
            y = []
            if _pga_col in pred_df.columns:
                y.append(float(row[_pga_col]))
            for T in SA_PERIODS:
                c = f"{_prefix}_Sa_{T:g}"
                if c in pred_df.columns:
                    y.append(float(row[c]))
            return y

        if n_rows == 1:
            # Single Median row: bold solid line with markers
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=_row_y(pred_df.iloc[0]),
                    mode="lines+markers",
                    name=comp,
                    legendgroup=comp,
                    line=dict(color=color, width=2.5),
                    marker=dict(size=5, opacity=0.85),
                )
            )
        else:
            # N Samples: realization 1 solid, 2+ cloud — no mixing
            for ri in range(n_rows):
                first = ri == 0
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=_row_y(pred_df.iloc[ri]),
                        mode="lines",
                        name=comp if first else None,
                        showlegend=first,
                        legendgroup=comp,
                        line=dict(
                            color=color if first else faint,
                            width=2.2 if first else _CLOUD_WIDTH,
                        ),
                        opacity=_SOLID_OPACITY if first else _CLOUD_OPACITY,
                    )
                )

    _add_cloud_legend_note(fig, multi)
    return fig


def _build_im_table(pred_df: pd.DataFrame):
    """
    Build a comprehensive DataTable with ALL predicted intensity measures.
    Rows: 3 rows per sample (Major, Intermediate, Vertical).
    Columns: Sample | Component | PGA (g) | Sa(0.03s) (g) | ... | PGV (cm/s)
    """
    rows = []
    n = len(pred_df)
    for ri in range(n):
        r = pred_df.iloc[ri]
        label = "Median" if n == 1 else f"Sample {ri + 1}"
        for comp, _, prefix in _COMP_KEYS:
            entry: dict[str, Any] = {"Sample": label, "Component": comp}

            # PGA (Sa_0)
            pga_col = f"{prefix}_Sa_0"
            if pga_col in pred_df.columns:
                entry["PGA (g)"] = round(float(r[pga_col]), 4)

            # Spectral Accelerations for all periods
            for T in SA_PERIODS:
                col_src = f"{prefix}_Sa_{T:g}"
                if col_src in pred_df.columns:
                    col_name = f"Sa({T:g}s) (g)"
                    entry[col_name] = round(float(r[col_src]), 4)

            # PGV (cm/s)
            pgv_col = f"{prefix}_PGV"
            if pgv_col in pred_df.columns:
                entry["PGV (cm/s)"] = round(float(r[pgv_col]) * _G_CMSS, 4)

            rows.append(entry)

    cols = [{"name": c, "id": c} for c in (rows[0] if rows else {})]
    return rows, cols


def fig_timeseries(sim_data: dict, plot_type: str) -> go.Figure:
    """3-row subplot: Major / Intermediate / Vertical for a given signal type.
    Realization 1 (first GMM sample, first stochastic sim) → solid, legend entry.
    All others → same-color cloud (thicker than a hairline, clearly visible).
    No averaging or mixing of any kind.
    """
    unit_map = {"ac": "g", "vel": "cm/s", "disp": "cm"}
    lname = {"ac": "Acceleration", "vel": "Velocity", "disp": "Displacement"}
    unit = unit_map.get(plot_type, "g")
    ylabel = f"{lname.get(plot_type, plot_type)} ({unit})"

    # Count total traces to detect multi mode for annotation
    total_traces = sum(
        sum(
            np.atleast_2d(np.array(gm[plot_type])).shape[0]
            for gm in sim_data.get(key, [])
        )
        for _, key, _ in _COMP_KEYS
    )
    multi = total_traces > 3  # more than one realization per component

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["<b>Major</b>", "<b>Intermediate</b>", "<b>Vertical</b>"],
        vertical_spacing=0.07,
    )
    for row_i, (comp, key, _) in enumerate(_COMP_KEYS, start=1):
        color = COMP_PALETTE[comp]
        faint = COMP_PALETTE_FAINT[comp].format(a=_CLOUD_ALPHA)
        global_trace = 0
        for gi, gm_dict in enumerate(sim_data.get(key, [])):
            data = np.atleast_2d(np.array(gm_dict[plot_type]))
            t = np.array(gm_dict["t"])
            for si in range(data.shape[0]):
                first = gi == 0 and si == 0
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=data[si],
                        mode="lines",
                        name=comp if first else None,
                        showlegend=first,
                        legendgroup=comp,
                        line=dict(
                            color=color if first else faint,
                            width=_SOLID_WIDTH_SUBPLOT if first else _CLOUD_WIDTH,
                        ),
                        opacity=_SOLID_OPACITY if first else _CLOUD_OPACITY,
                    ),
                    row=row_i,
                    col=1,
                )
                global_trace += 1

    fig.update_layout(template=PINAGMM_TEMPLATE, margin=dict(l=60, r=18, t=40, b=50))
    for r in range(1, 4):
        fig.update_yaxes(title_text=ylabel, row=r, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    # Re-apply subplot title font after update (update_layout resets annotations)
    for ann in fig.layout.annotations:
        if ann.text and ann.text not in ("", None):
            ann.font = dict(size=11, color="#475569")
    if multi:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.02,
            xanchor="left",
            yanchor="bottom",
            text="solid = 1st realization · cloud = remaining",
            showarrow=False,
            font=dict(size=9, color="#94a3b8", family="Inter, sans-serif"),
            bgcolor="rgba(255,255,255,0.0)",
            borderpad=2,
        )
    return fig


def _read_sa(sim_data: dict, key: str):
    T, sa_rows = np.array(sim_data.get("spec_T", [])), []
    for gm_dict in sim_data.get(key, []):
        sa = np.atleast_2d(np.array(gm_dict.get("sa", [])))
        for si in range(sa.shape[0]):
            sa_rows.append(sa[si])
    return T, np.vstack(sa_rows) if sa_rows else np.zeros((0, len(T)))


def fig_simsa_combined(sim_data: dict, log_y: bool) -> go.Figure:
    """Simulated spectra: realization 1 solid line, realizations 2+ cloud traces (no geomean).
    Cloud traces use the same component color at reduced opacity — thicker than a hairline."""
    fig = make_fig("Period (s)", "Sa (g)", log_x=True, log_y=log_y)
    all_rows = sum(
        _read_sa(sim_data, key)[1].shape[0] for _, key, _ in _COMP_KEYS
    )
    multi = all_rows > 3

    for comp, key, _ in _COMP_KEYS:
        color = COMP_PALETTE[comp]
        faint = COMP_PALETTE_FAINT[comp].format(a=_CLOUD_ALPHA)
        T, sa = _read_sa(sim_data, key)
        if sa.shape[0] == 0:
            continue

        # Realization 1 (solid line, legend entry — always just comp name)
        fig.add_trace(
            go.Scatter(
                x=T,
                y=sa[0],
                mode="lines",
                name=comp,
                legendgroup=comp,
                line=dict(color=color, width=2.5),
                opacity=_SOLID_OPACITY,
            )
        )

        # Realizations 2+ (cloud traces, no legend entry, visible but secondary)
        for row in sa[1:]:
            fig.add_trace(
                go.Scatter(
                    x=T,
                    y=row,
                    mode="lines",
                    showlegend=False,
                    legendgroup=comp,
                    line=dict(color=faint, width=_CLOUD_WIDTH),
                    opacity=_CLOUD_OPACITY,
                )
            )

    _add_cloud_legend_note(fig, multi)
    return fig


def fig_simsa_single(sim_data: dict, comp: str, key: str, log_y: bool) -> go.Figure:
    color = COMP_PALETTE[comp]
    faint = COMP_PALETTE_FAINT[comp].format(a=_CLOUD_ALPHA)
    fig = make_fig("Period (s)", "Sa (g)", log_x=True, log_y=log_y)
    T, sa = _read_sa(sim_data, key)
    multi = sa.shape[0] > 1
    if sa.shape[0] > 0:
        fig.add_trace(
            go.Scatter(
                x=T,
                y=sa[0],
                mode="lines",
                name=comp,
                line=dict(color=color, width=2.5),
                opacity=_SOLID_OPACITY,
            )
        )
        for row in sa[1:]:
            fig.add_trace(
                go.Scatter(
                    x=T,
                    y=row,
                    mode="lines",
                    showlegend=False,
                    line=dict(color=faint, width=_CLOUD_WIDTH),
                    opacity=_CLOUD_OPACITY,
                )
            )
    _add_cloud_legend_note(fig, multi)
    fig.update_layout(margin=dict(l=50, r=10, t=36, b=44))
    return fig


def fig_fas_combined(sim_data: dict, log_y: bool) -> go.Figure:
    """Combined FAS: realization 1 solid, remaining cloud — no mixing."""
    fig = make_fig("Frequency (Hz)", "FAS (g·s)", log_x=True, log_y=log_y)

    # Count total traces per component to determine multi mode
    def _count(key):
        return sum(
            np.atleast_2d(np.array(gm.get("fas", []))).shape[0]
            for gm in sim_data.get(key, [])
        )

    multi = any(_count(key) > 1 for _, key, _ in _COMP_KEYS)

    for comp, key, _ in _COMP_KEYS:
        color = COMP_PALETTE[comp]
        faint = COMP_PALETTE_FAINT[comp].format(a=_CLOUD_ALPHA)
        global_idx = 0
        for gi, gm_dict in enumerate(sim_data.get(key, [])):
            fas = np.atleast_2d(np.array(gm_dict.get("fas", [])))
            freq = np.array(gm_dict.get("freq", []))
            if len(freq) == 0:
                continue
            freq, fas = freq[1:], fas[:, 1:]
            for si in range(fas.shape[0]):
                first = global_idx == 0
                fig.add_trace(
                    go.Scatter(
                        x=freq,
                        y=fas[si],
                        mode="lines",
                        name=comp if first else None,
                        showlegend=first,
                        legendgroup=comp,
                        line=dict(
                            color=color if first else faint,
                            width=2.0 if first else _CLOUD_WIDTH,
                        ),
                        opacity=_SOLID_OPACITY if first else _CLOUD_OPACITY,
                    )
                )
                global_idx += 1

    _add_cloud_legend_note(fig, multi)
    return fig


def fig_fas_single(sim_data: dict, comp: str, key: str, log_y: bool) -> go.Figure:
    color = COMP_PALETTE[comp]
    faint = COMP_PALETTE_FAINT[comp].format(a=_CLOUD_ALPHA)
    fig = make_fig("Frequency (Hz)", "FAS (g·s)", log_x=True, log_y=log_y)
    global_idx = 0
    for gi, gm_dict in enumerate(sim_data.get(key, [])):
        fas = np.atleast_2d(np.array(gm_dict.get("fas", [])))
        freq = np.array(gm_dict.get("freq", []))
        if len(freq) == 0:
            continue
        freq, fas = freq[1:], fas[:, 1:]
        for si in range(fas.shape[0]):
            first = global_idx == 0
            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=fas[si],
                    mode="lines",
                    name=comp if first else None,
                    showlegend=first,
                    legendgroup=comp,
                    line=dict(
                        color=color if first else faint,
                        width=2.0 if first else _CLOUD_WIDTH,
                    ),
                    opacity=_SOLID_OPACITY if first else _CLOUD_OPACITY,
                )
            )
            global_idx += 1
    multi = global_idx > 1
    _add_cloud_legend_note(fig, multi)
    fig.update_layout(margin=dict(l=50, r=10, t=36, b=44))
    return fig


# ═══════════════════════════════════════════════════════════════════
#  CSV export helpers
# ═══════════════════════════════════════════════════════════════════


def _build_ts_csv(sim_data: dict) -> str:
    """
    Build a flat columnar time-series CSV for all simulated components, GM samples, and realizations.
    Each trace (Component, GMM sample gi, Realization si) gets its own quadruplet of columns:
      Time_{comp}_gm{gi}_r{si}_s, Acc_{comp}_gm{gi}_r{si}_g, Vel_{comp}_gm{gi}_r{si}_cms, Disp_{comp}_gm{gi}_r{si}_cm

    This guarantees that even if time series have different lengths or sampling,
    each trace's time vector aligns perfectly with its acceleration, velocity, and displacement values.
    """
    if not sim_data:
        return "Time_s\n"

    series_list = []
    max_len = 0

    for comp_label, comp_key, _ in _COMP_KEYS:
        gm_list = sim_data.get(comp_key, [])
        for gi, gm_dict in enumerate(gm_list):
            t = np.array(gm_dict.get("t", []), dtype=float)
            ac = np.atleast_2d(np.array(gm_dict.get("ac", []), dtype=float))
            vel = np.atleast_2d(np.array(gm_dict.get("vel", []), dtype=float))
            disp = np.atleast_2d(np.array(gm_dict.get("disp", []), dtype=float))

            n_sim = ac.shape[0]
            for si in range(n_sim):
                ac_i = ac[si] if si < ac.shape[0] else np.array([], dtype=float)
                vel_i = vel[si] if si < vel.shape[0] else np.array([], dtype=float)
                disp_i = disp[si] if si < disp.shape[0] else np.array([], dtype=float)

                cur_max = max(len(t), len(ac_i), len(vel_i), len(disp_i))
                max_len = max(max_len, cur_max)
                series_list.append((comp_label, gi + 1, si + 1, t, ac_i, vel_i, disp_i))

    if max_len == 0 or not series_list:
        return "Time_s\n"

    def _pad(arr: np.ndarray) -> np.ndarray:
        out = np.full(max_len, np.nan, dtype=float)
        n = min(len(arr), max_len)
        if n > 0:
            out[:n] = arr[:n]
        return out

    out: dict[str, np.ndarray] = {}

    for comp_label, gm_idx, r_idx, t_arr, ac_arr, vel_arr, disp_arr in series_list:
        tag = f"{comp_label}_gm{gm_idx}_r{r_idx}"
        out[f"Time_{tag}_s"] = _pad(t_arr)
        out[f"Acc_{tag}_g"] = _pad(ac_arr)
        out[f"Vel_{tag}_cms"] = _pad(vel_arr)
        out[f"Disp_{tag}_cm"] = _pad(disp_arr)

    return pd.DataFrame(out).to_csv(index=False)



def _build_sa_csv(sim_data: dict, keys_labels) -> str:
    """
    Build a flat response-spectra CSV.

    Columns: Period_s | Sa_{comp}_gm{gi+1}_r{si+1}_g  ...
    All components share the same period axis (spec_T).
    """
    T = np.array(sim_data.get("spec_T", []))
    out: dict[str, Any] = {"Period_s": T}
    for comp_label, comp_key in keys_labels:
        for gi, gm_dict in enumerate(sim_data.get(comp_key, [])):
            sa = np.atleast_2d(np.array(gm_dict.get("sa", [])))
            for si in range(sa.shape[0]):
                col = f"Sa_{comp_label}_gm{gi + 1}_r{si + 1}_g"
                row_sa = sa[si]
                if len(row_sa) == len(T):
                    out[col] = row_sa
                else:
                    padded = np.full(len(T), np.nan)
                    n = min(len(row_sa), len(T))
                    padded[:n] = row_sa[:n]
                    out[col] = padded
    return pd.DataFrame(out).to_csv(index=False)


def _build_fas_csv(sim_data: dict, keys_labels) -> str:
    """
    Build a flat Fourier-spectra CSV — same style as Response Spectra.

    Columns: Frequency_Hz | FAS_{comp}_gm{gi+1}_r{si+1}_gs  ...

    All GM samples from the same simulation share the same frequency axis
    (same dt/npts), so a single flat DataFrame is used.
    If axes differ in length (edge case), shorter columns are NaN-padded.
    """
    # Collect all frequency axes and FAS arrays
    freq_ref: np.ndarray | None = None
    max_freq_len = 0
    for _, comp_key in keys_labels:
        for gm_dict in sim_data.get(comp_key, []):
            f = np.array(gm_dict.get("freq", []))
            if len(f) > 1:
                f = f[1:]  # drop DC bin
                if freq_ref is None or len(f) > max_freq_len:
                    freq_ref = f
                    max_freq_len = len(f)

    if freq_ref is None:
        return "Frequency_Hz\n"

    def _pad(arr: np.ndarray, length: int) -> np.ndarray:
        if len(arr) == length:
            return arr
        out = np.full(length, np.nan)
        out[: len(arr)] = arr
        return out

    out: dict[str, Any] = {"Frequency_Hz": freq_ref}
    for comp_label, comp_key in keys_labels:
        for gi, gm_dict in enumerate(sim_data.get(comp_key, [])):
            fas = np.atleast_2d(np.array(gm_dict.get("fas", [])))
            freq_gi = np.array(gm_dict.get("freq", []))
            if len(freq_gi) <= 1:
                continue
            fas = fas[:, 1:]  # drop DC bin column
            for si in range(fas.shape[0]):
                col = f"FAS_{comp_label}_gm{gi + 1}_r{si + 1}_gs"
                out[col] = _pad(fas[si], max_freq_len)
    return pd.DataFrame(out).to_csv(index=False)


# ═══════════════════════════════════════════════════════════════════
#  Callback registration
# ═══════════════════════════════════════════════════════════════════


def register_charts_callbacks(app) -> None:

    # ── Prediction: section switcher ─────────────────────────────
    @app.callback(
        Output("pred-section-spectra", "style"),
        Output("pred-section-ims", "style"),
        Input("pred-view", "value"),
        Input("pred-store", "data"),
        prevent_initial_call=False,
    )
    def switch_pred_view(view, pred_json):
        if not pred_json:
            return _HIDE, _HIDE
        return (
            _SHOW if view == "spectra" else _HIDE,
            _SHOW if view == "ims" else _HIDE,
        )

    # ── Prediction: charts + IM table ───────────────────────────
    @app.callback(
        Output("graph-pred-spectra", "figure"),
        Output("table-pred-ims-container", "children"),
        Output("pred-hint", "style"),
        Input("pred-store", "data"),
        prevent_initial_call=False,
    )
    def update_pred_charts(pred_json):
        empty = make_empty_fig(
            "Click Predict in the sidebar to compute GMM predictions."
        )
        if not pred_json:
            return empty, html.Div(), _SHOW

        pred_df = pd.read_json(io.StringIO(pred_json), orient="split")
        n_rows = len(pred_df)
        spec_fig = fig_pred_spectra(pred_df, n_rows)
        rows, col_defs = _build_im_table(pred_df)

        table = dash_table.DataTable(
            data=rows,
            columns=col_defs,
            style_table={
                # Tell Dash the table is wider than the card — parent div scrolls
                "minWidth": f"{len(col_defs) * 115}px",
                "width": f"{len(col_defs) * 115}px",
            },
            style_cell={
                "fontFamily": "Inter, sans-serif",
                "fontSize": "0.78rem",
                "padding": "8px 12px",
                "textAlign": "center",
                "border": "1px solid #e2e8f0",
                "minWidth": "110px",
                "width": "115px",
                "maxWidth": "160px",
                "whiteSpace": "nowrap",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            },
            style_header={
                "backgroundColor": "#f1f5f9",
                "fontWeight": "700",
                "fontSize": "0.74rem",
                "color": "#475569",
                "border": "1px solid #cbd5e1",
                "whiteSpace": "nowrap",
                "position": "sticky",
                "top": 0,
                "zIndex": 2,
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#fafbfd"},
                {
                    "if": {
                        "filter_query": "{Component} = Major",
                        "column_id": "Component",
                    },
                    "color": "#4f46e5",
                    "fontWeight": "700",
                },
                {
                    "if": {
                        "filter_query": "{Component} = Intermediate",
                        "column_id": "Component",
                    },
                    "color": "#e16b16",
                    "fontWeight": "700",
                },
                {
                    "if": {
                        "filter_query": "{Component} = Vertical",
                        "column_id": "Component",
                    },
                    "color": "#059669",
                    "fontWeight": "700",
                },
            ],
            page_size=30,
        )
        return spec_fig, table, _HIDE

    # ── Time Series: combined 3-row chart ────────────────────────
    @app.callback(
        Output("graph-timeseries", "figure"),
        Output("ts-hint", "style"),
        Input("sim-store", "data"),
        Input("ts-channel", "value"),
        prevent_initial_call=False,
    )
    def update_timeseries(sim_json, channel):
        empty = make_empty_fig(
            "Click Simulate in the sidebar to generate ground motion time series."
        )
        if not sim_json:
            return empty, _SHOW
        sim_data = json.loads(sim_json)
        return fig_timeseries(sim_data, channel or "ac"), _HIDE

    # ── Response Spectra: section switcher ───────────────────────
    @app.callback(
        Output("simsa-section-all", "style"),
        Output("simsa-section-major", "style"),
        Output("simsa-section-inter", "style"),
        Output("simsa-section-vert", "style"),
        Input("simsa-comp-view", "value"),
        Input("sim-store", "data"),
        prevent_initial_call=False,
    )
    def switch_simsa_view(view, sim_json):
        if not sim_json:
            return _HIDE, _HIDE, _HIDE, _HIDE
        return _section_styles(view, ["all", "major", "inter", "vert"])

    # ── Response Spectra: combined + per-component ────────────────
    @app.callback(
        Output("graph-simspectra", "figure"),
        Output("graph-simsa-major", "figure"),
        Output("graph-simsa-inter", "figure"),
        Output("graph-simsa-vert", "figure"),
        Output("simsa-hint", "style"),
        Input("sim-store", "data"),
        Input("simsa-yscale", "value"),
        prevent_initial_call=False,
    )
    def update_simspectra(sim_json, yscale):
        log_y = yscale != "linear"
        empty = make_empty_fig(
            "Click Simulate in the sidebar to compute response spectra."
        )
        if not sim_json:
            return empty, empty, empty, empty, _SHOW
        sim_data = json.loads(sim_json)
        combined = fig_simsa_combined(sim_data, log_y)
        major_fig = fig_simsa_single(sim_data, "Major", "major", log_y)
        inter_fig = fig_simsa_single(sim_data, "Intermediate", "inter", log_y)
        vert_fig = fig_simsa_single(sim_data, "Vertical", "vert", log_y)
        return combined, major_fig, inter_fig, vert_fig, _HIDE

    # ── FAS: section switcher ────────────────────────────────────
    @app.callback(
        Output("fas-section-all", "style"),
        Output("fas-section-major", "style"),
        Output("fas-section-inter", "style"),
        Output("fas-section-vert", "style"),
        Input("fas-comp-view", "value"),
        Input("sim-store", "data"),
        prevent_initial_call=False,
    )
    def switch_fas_view(view, sim_json):
        if not sim_json:
            return _HIDE, _HIDE, _HIDE, _HIDE
        return _section_styles(view, ["all", "major", "inter", "vert"])

    # ── FAS: combined + per-component ────────────────────────────
    @app.callback(
        Output("graph-fas", "figure"),
        Output("graph-fas-major", "figure"),
        Output("graph-fas-inter", "figure"),
        Output("graph-fas-vert", "figure"),
        Output("fas-hint", "style"),
        Input("sim-store", "data"),
        Input("fas-yscale", "value"),
        prevent_initial_call=False,
    )
    def update_fas(sim_json, yscale):
        log_y = yscale != "linear"
        empty = make_empty_fig(
            "Click Simulate in the sidebar to compute Fourier spectra."
        )
        if not sim_json:
            return empty, empty, empty, empty, _SHOW
        sim_data = json.loads(sim_json)
        combined = fig_fas_combined(sim_data, log_y)
        major_fig = fig_fas_single(sim_data, "Major", "major", log_y)
        inter_fig = fig_fas_single(sim_data, "Intermediate", "inter", log_y)
        vert_fig = fig_fas_single(sim_data, "Vertical", "vert", log_y)
        return combined, major_fig, inter_fig, vert_fig, _HIDE

    # ── DOWNLOAD CALLBACKS ───────────────────────────────────────

    # ── Prediction: Response Spectra CSV ─────────────────────────
    @app.callback(
        Output("download-pred", "data"),
        Input("btn-dl-pred", "n_clicks"),
        State("pred-store", "data"),
        prevent_initial_call=True,
    )
    def dl_pred(n, pred_json):
        if not n or not pred_json:
            raise PreventUpdate
        pred_df = pd.read_json(io.StringIO(pred_json), orient="split")
        n_rows = len(pred_df)
        # Build a tidy IM table identical to what the DataTable shows
        rows, _ = _build_im_table(pred_df)
        out_df = pd.DataFrame(rows)
        return _send_csv_with_bom(out_df, "pinagmm_predicted_spectra.csv")

    # ── Prediction: IM Table CSV ──────────────────────────────────
    @app.callback(
        Output("download-ims", "data"),
        Input("btn-dl-ims", "n_clicks"),
        State("pred-store", "data"),
        prevent_initial_call=True,
    )
    def dl_ims(n, pred_json):
        if not n or not pred_json:
            raise PreventUpdate
        pred_df = pd.read_json(io.StringIO(pred_json), orient="split")
        rows, _ = _build_im_table(pred_df)
        out_df = pd.DataFrame(rows)
        return _send_csv_with_bom(out_df, "pinagmm_intensity_measures.csv")

    # ── Response Spectra: All Components CSV ──────────────────────
    @app.callback(
        Output("download-simsa", "data"),
        Input("btn-dl-simsa", "n_clicks"),
        State("sim-store", "data"),
        prevent_initial_call=True,
    )
    def dl_simsa(n, sim_json):
        if not n or not sim_json:
            raise PreventUpdate
        sim_data = json.loads(sim_json)
        csv_str = _build_sa_csv(
            sim_data,
            [("Major", "major"), ("Intermediate", "inter"), ("Vertical", "vert")],
        )
        return _send_text_with_bom(csv_str, "pinagmm_response_spectra_all.csv")

    for _comp_label, _comp_key in [
        ("Major", "major"),
        ("Intermediate", "inter"),
        ("Vertical", "vert"),
    ]:
        _reg_simsa_comp_dl(app, _comp_label, _comp_key)

    # ── Fourier Spectra: All Components CSV ───────────────────────
    @app.callback(
        Output("download-fas", "data"),
        Input("btn-dl-fas", "n_clicks"),
        State("sim-store", "data"),
        prevent_initial_call=True,
    )
    def dl_fas(n, sim_json):
        if not n or not sim_json:
            raise PreventUpdate
        sim_data = json.loads(sim_json)
        csv_str = _build_fas_csv(
            sim_data,
            [("Major", "major"), ("Intermediate", "inter"), ("Vertical", "vert")],
        )
        return _send_text_with_bom(csv_str, "pinagmm_fourier_spectra_all.csv")

    for _comp_label, _comp_key in [
        ("Major", "major"),
        ("Intermediate", "inter"),
        ("Vertical", "vert"),
    ]:
        _reg_fas_comp_dl(app, _comp_label, _comp_key)

    # ── Time Series: Download CSV ─────────────────────────────────
    @app.callback(
        Output("download-ts", "data"),
        Input("btn-dl-ts", "n_clicks"),
        State("sim-store", "data"),
        prevent_initial_call=True,
    )
    def dl_ts(n, sim_json):
        if not n or not sim_json:
            raise PreventUpdate
        try:
            import traceback as _tb
            sim_data = json.loads(sim_json)
            csv_str = _build_ts_csv(sim_data)
            return _send_text_with_bom(csv_str, "pinagmm_timeseries.csv")
        except Exception as exc:
            _tb.print_exc()
            raise PreventUpdate from exc


def _reg_simsa_comp_dl(app, comp_label: str, comp_key: str):
    @app.callback(
        Output(f"download-simsa-{comp_key}", "data"),
        Input(f"btn-dl-simsa-{comp_key}", "n_clicks"),
        State("sim-store", "data"),
        prevent_initial_call=True,
    )
    def _dl(n, sim_json, _cl=comp_label, _ck=comp_key):
        if not n or not sim_json:
            raise PreventUpdate
        sim_data = json.loads(sim_json)
        csv_str = _build_sa_csv(sim_data, [(_cl, _ck)])
        return _send_text_with_bom(
            csv_str, f"pinagmm_response_spectra_{_ck}.csv"
        )


def _reg_fas_comp_dl(app, comp_label: str, comp_key: str):
    @app.callback(
        Output(f"download-fas-{comp_key}", "data"),
        Input(f"btn-dl-fas-{comp_key}", "n_clicks"),
        State("sim-store", "data"),
        prevent_initial_call=True,
    )
    def _dl(n, sim_json, _cl=comp_label, _ck=comp_key):
        if not n or not sim_json:
            raise PreventUpdate
        sim_data = json.loads(sim_json)
        csv_str = _build_fas_csv(sim_data, [(_cl, _ck)])
        return _send_text_with_bom(
            csv_str, f"pinagmm_fourier_spectra_{_ck}.csv"
        )
