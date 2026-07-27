"""
PINAGMM Interactive GUI
=======================
A modern web-based desktop application powered by NiceGUI + Plotly.

Launch with:
    python -m pinagmm.gui
    # or
    python pinagmm/gui/app.py

The GUI opens automatically in your default browser at http://localhost:8080.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nicegui import ui, run

from pinagmm import PINAGMM, save_timeseries, save_spectra
from pinagmm.core.variables import periods as SA_PERIODS, yvars

# ─────────────────────────────────────────────────────────────────────────────
#  Constants & theming
# ─────────────────────────────────────────────────────────────────────────────
# Component colors are pulled straight from the Nord palette so the plotly
# figures and the surrounding UI chrome always feel like one coherent theme.
COMP_COLORS = {
    "Major": "#88C0D0",  # nord8  - frost cyan
    "Intermediate": "#EBCB8B",  # nord13 - aurora yellow
    "Vertical": "#A3BE8C",  # nord14 - aurora green
}

OUT_DIR = Path.home() / "Desktop" / "PINAGMM Results"

# Plotly layout shared across all charts
_PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="Inter, system-ui, sans-serif",
        color="#D8DEE9",  # nord4
        size=11,
    ),
    legend=dict(
        bgcolor="rgba(46,52,64,0.8)",  # nord0
        bordercolor="#4C566A",  # nord3
        borderwidth=1,
        font=dict(size=10),
    ),
    margin=dict(l=58, r=18, t=46, b=46),
    hovermode="x unified",
    xaxis=dict(
        gridcolor="#3B4252",  # nord1
        zerolinecolor="#434C5E",  # nord2
        showline=True,
        linecolor="#4C566A",  # nord3
        showspikes=False,
    ),
    yaxis=dict(
        gridcolor="#3B4252",  # nord1
        zerolinecolor="#434C5E",  # nord2
        showline=True,
        linecolor="#4C566A",  # nord3
        showspikes=False,
    ),
)

# IM columns available for conditional targeting
_COND_IMS = [v for v in yvars if any(s in v for s in ("PGV", "Sa"))]

# Fault mechanism dropdown options
_FM_LABELS = {
    "0 — Strike Slip": "0",
    "1 — Normal": "1",
    "2 — Reverse": "2",
    "3 — Reverse Oblique": "3",
    "4 — Normal Oblique": "4",
}

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg:           #2E3440;   /* nord0 */
    --surface:      #3B4252;   /* nord1 */
    --card:         #3B4252;   /* nord1 */
    --border:       #4C566A;   /* nord3 */
    --border-hover: #5E81AC;   /* nord10 */
    --text:         #ECEFF4;   /* nord6 */
    --text-muted:   #D8DEE9;   /* nord4 */

    /* Single accent family used everywhere instead of Quasar's default
       purple, so switches / tab indicators / spinners / flat buttons all
       read as part of the same Nord-based theme. */
    --primary:   #88C0D0;      /* nord8  - frost cyan   */
    --secondary: #81A1C1;      /* nord9  - frost blue    */
    --accent:    #A3BE8C;      /* nord14 - aurora green  */
    --major:     #88C0D0;
    --inter:     #EBCB8B;
    --vert:      #A3BE8C;

    /* Re-point Quasar's own CSS variables so color="primary" etc. pick up
       the Nord palette automatically, without needing purple- overrides
       scattered through the markup. */
    --q-primary:   var(--primary);
    --q-secondary: var(--secondary);
    --q-accent:    var(--accent);
}

body {
    font-family: "Inter", system-ui, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.q-header {
    background-color: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    color: var(--text) !important;
}

.pcard {
    background-color: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25) !important;
    transition: border-color 0.15s ease;
}
.pcard:hover {
    border-color: var(--border-hover) !important;
}

.slabel {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
}

.q-field--outlined .q-field__control {
    border-radius: 6px !important;
    border-color: var(--border) !important;
}
.q-field--outlined:hover .q-field__control { border-color: var(--primary) !important; }
.q-field--focused .q-field__control { border-color: var(--primary) !important; }

.primary-btn {
    background-color: var(--secondary) !important;
    color: #fff !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    transition: filter 0.15s ease;
}
.primary-btn:hover { filter: brightness(1.08); }

.accent-btn {
    background-color: var(--accent) !important;
    color: #1B1F27 !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    transition: filter 0.15s ease;
}
.accent-btn:hover { filter: brightness(1.08); }

.status-pill {
    font-size: 0.75rem;
    padding: 4px 12px;
    border-radius: 16px;
    background-color: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-muted);
}

.q-tabs {
    background: var(--surface);
    border-radius: 8px;
    padding: 2px;
    border: 1px solid var(--border);
}
.q-tab {
    border-radius: 6px !important;
    min-height: 32px !important;
}
.q-tab--active {
    background: var(--bg) !important;
    color: var(--primary) !important;
    border: 1px solid var(--border) !important;
}

/* Quasar uppercases tab labels by default — switch to normal title case
   so "Time Series" / "Response Spectra" / "Fourier" render as written. */
.q-tab__label {
    text-transform: none !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.01em !important;
    font-weight: 500 !important;
}
.q-tab--active .q-tab__label {
    font-weight: 600 !important;
}

.q-btn {
    text-transform: none !important;
    letter-spacing: 0.01em;
}

.result-hint {
    color: var(--text-muted);
    font-size: 0.88rem;
    padding: 3rem 2rem;
    text-align: center;
}

/* thin, theme-matched scrollbar for the sidebar */
.sidebar-scroll::-webkit-scrollbar { width: 8px; }
.sidebar-scroll::-webkit-scrollbar-track { background: transparent; }
.sidebar-scroll::-webkit-scrollbar-thumb {
    background-color: var(--border);
    border-radius: 8px;
}
.sidebar-scroll::-webkit-scrollbar-thumb:hover { background-color: var(--border-hover); }
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Application state
# ─────────────────────────────────────────────────────────────────────────────
class _State:
    gmm: PINAGMM | None = None
    prediction: pd.DataFrame | None = None
    ts_m = None  # GroundMotion or list[GroundMotion]
    ts_i = None
    ts_v = None
    sim_dt: float = 0.005


_s = _State()


def _get_gmm() -> PINAGMM:
    if _s.gmm is None:
        _s.gmm = PINAGMM()
    return _s.gmm


# ─────────────────────────────────────────────────────────────────────────────
#  Figure builders
# ─────────────────────────────────────────────────────────────────────────────
def _layout(**extra) -> dict:
    return {**_PL, **extra}


def _title(text: str) -> dict:
    return dict(text=text, font=dict(size=13, color="#88C0D0"), pad=dict(b=6))


def fig_spectrum_predicted(pred: pd.DataFrame) -> go.Figure:
    """Predicted median response spectrum for all three components."""
    row0 = pred.iloc[0]
    fig = go.Figure()

    for comp, prefix in [("Major", "M"), ("Intermediate", "I"), ("Vertical", "V")]:
        x, y = [], []
        pga = f"{prefix}_Sa_0"
        if pga in pred.columns:
            x.append(0.01)
            y.append(float(row0[pga]))
        for T in SA_PERIODS:
            col = f"{prefix}_Sa_{T:g}"
            if col in pred.columns:
                x.append(float(T))
                y.append(float(row0[col]))
        if x:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=comp,
                    line=dict(color=COMP_COLORS[comp], width=2.5),
                    marker=dict(size=5, opacity=0.85),
                )
            )

    fig.update_layout(
        **_layout(
            title=_title("Predicted Median Response Spectrum"),
            xaxis=dict(**_PL["xaxis"], title="Period (s)", type="log"),
            yaxis=dict(**_PL["yaxis"], title="Spectral Acceleration (g)", type="log"),
        )
    )
    return fig


def fig_timeseries(ts_m, ts_i, ts_v, max_traces: int = 30) -> go.Figure:
    """3-component acceleration time series."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["<b>Major</b>", "<b>Intermediate</b>", "<b>Vertical</b>"],
        vertical_spacing=0.08,
    )
    for row_idx, (comp, ts) in enumerate(
        [("Major", ts_m), ("Intermediate", ts_i), ("Vertical", ts_v)], start=1
    ):
        color = COMP_COLORS[comp]
        gm_list = ts
        count = 0
        for gi, gm in enumerate(gm_list):
            ac = np.atleast_2d(gm.ac)
            t = gm.t
            for si in range(ac.shape[0]):
                if count >= max_traces:
                    break
                first = gi == 0 and si == 0
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=ac[si],
                        mode="lines",
                        name=comp if first else None,
                        showlegend=first,
                        legendgroup=comp,
                        line=dict(color=color, width=2.0 if first else 0.55),
                        opacity=1.0 if first else 0.15,
                    ),
                    row=row_idx,
                    col=1,
                )
                count += 1

    base = _layout(
        height=580,
        title=_title("Simulated Ground Motion Time Series  (Acceleration)"),
    )
    base.update(
        {
            "xaxis": dict(**_PL["xaxis"], showticklabels=False),
            "xaxis2": dict(**_PL["xaxis"], showticklabels=False),
            "xaxis3": dict(**_PL["xaxis"], title="Time (s)"),
            "yaxis": dict(**_PL["yaxis"], title="Major (g)"),
            "yaxis2": dict(**_PL["yaxis"], title="Interm. (g)"),
            "yaxis3": dict(**_PL["yaxis"], title="Vertical (g)"),
        }
    )
    fig.update_layout(**base)
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color="#81A1C1")
    return fig


def fig_simulated_spectra(ts_m, ts_i, ts_v, periods: np.ndarray) -> go.Figure:
    """Response spectra computed from simulated time series."""
    fig = go.Figure()
    for comp, ts in [("Major", ts_m), ("Intermediate", ts_i), ("Vertical", ts_v)]:
        if ts is None:
            continue
        color = COMP_COLORS[comp]
        gm_list = ts
        sa_rows: list[np.ndarray] = []
        for gm in gm_list:
            _, _, sa = gm.response_spectra(periods)
            for row in np.atleast_2d(sa):
                sa_rows.append(row)

        sa_mat = np.vstack(sa_rows)  # (total_sims, n_periods)

        # Individual traces (faint cloud)
        for i, row in enumerate(sa_mat):
            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=row,
                    mode="lines",
                    name=comp if i == 0 else None,
                    showlegend=(i == 0),
                    legendgroup=comp,
                    line=dict(color=color, width=0.6),
                    opacity=0.12,
                )
            )

        # Geometric-mean median
        sa_med = np.exp(np.mean(np.log(np.clip(sa_mat, 1e-12, None)), axis=0))
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=sa_med,
                mode="lines",
                name=f"{comp} median",
                legendgroup=comp + "_med",
                line=dict(color=color, width=2.8),
            )
        )

    fig.update_layout(
        **_layout(
            title=_title("Simulated Response Spectra (5% damping)"),
            xaxis=dict(**_PL["xaxis"], title="Period (s)", type="log"),
            yaxis=dict(**_PL["yaxis"], title="Spectral Acceleration (g)", type="log"),
        )
    )
    return fig


def fig_fas(ts_m, ts_i, ts_v) -> go.Figure:
    """Fourier Amplitude Spectra."""
    fig = go.Figure()
    for comp, ts in [("Major", ts_m), ("Intermediate", ts_i), ("Vertical", ts_v)]:
        if ts is None:
            continue
        color = COMP_COLORS[comp]
        gm_list = ts
        for gi, gm in enumerate(gm_list):
            fas_data = np.atleast_2d(gm.fas)
            freq = gm.freq
            for si, fas_row in enumerate(fas_data):
                first = gi == 0 and si == 0
                fig.add_trace(
                    go.Scatter(
                        x=freq,
                        y=fas_row,
                        mode="lines",
                        name=comp if first else None,
                        showlegend=first,
                        legendgroup=comp,
                        line=dict(color=color, width=1.5 if first else 0.5),
                        opacity=1.0 if first else 0.15,
                    )
                )

    fig.update_layout(
        **_layout(
            title=_title("Fourier Amplitude Spectrum"),
            xaxis=dict(**_PL["xaxis"], title="Frequency (Hz)", type="log"),
            yaxis=dict(**_PL["yaxis"], title="Fourier Amplitude (g·s)", type="log"),
        )
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Page definition
# ─────────────────────────────────────────────────────────────────────────────
def build_page() -> None:
    from nicegui import app

    app.add_static_files("/assets", str(Path(__file__).parent / "assets"))
    ui.add_head_html(f"<style>{_CSS}</style>")
    ui.dark_mode().enable()

    # ── Header ────────────────────────────────────────────────────────────────
    with ui.header().classes("items-center q-py-xs q-px-lg"):
        ui.image("/assets/logo.png").style("width: 32px; height: 32px;")
        ui.label("PINAGMM").classes("text-weight-bold q-ml-sm").style(
            "font-size:1.22rem;letter-spacing:.06em;"
        )
        ui.label("Physics-Informed Neural Additive Ground Motion Model").style(
            "font-size:.78rem;opacity:.7;margin-left:.75rem;"
        )
        ui.space()
        status_lbl = ui.label("Ready").classes("status-pill")

    # ── Main two-column layout ─────────────────────────────────────────────────
    with (
        ui.row()
        .classes("w-full no-wrap items-start")
        .style("gap:0;min-height:calc(100vh - 52px)")
    ):
        # ── LEFT sidebar ──────────────────────────────────────────────────────
        with (
            ui.column()
            .classes("q-pa-sm q-gutter-sm sidebar-scroll")
            .style(
                "width:296px;min-width:296px;overflow-y:auto;"
                "height:calc(100vh - 52px);position:sticky;top:52px;"
                "background:rgba(46,52,64,.55);border-right:1px solid var(--border);"
            )
        ):
            with ui.card().classes("pcard q-pa-md w-full"):
                ui.label("Earthquake and Site Scenario").classes("slabel q-mb-sm")
                i_mw = (
                    ui.number(
                        "Moment Magnitude Mw",
                        value=6.5,
                        min=4.0,
                        max=8.5,
                        step=0.1,
                        format="%.1f",
                    )
                    .props("dense outlined")
                    .classes("w-full")
                )
                i_ztor = (
                    ui.number(
                        "Rupture Depth Ztor (km)",
                        value=3.0,
                        min=0.0,
                        max=30.0,
                        step=0.5,
                        format="%.1f",
                    )
                    .props("dense outlined")
                    .classes("w-full q-mt-sm")
                )
                i_fm = (
                    ui.select(
                        options=list(_FM_LABELS),
                        label="Fault Mechanism",
                        value="0 — Strike Slip",
                    )
                    .props("dense outlined")
                    .classes("w-full q-mt-sm")
                )
                i_rrup = (
                    ui.number(
                        "Rupture Distance Rrup (km)",
                        value=15.0,
                        min=1.0,
                        max=400.0,
                        step=1.0,
                        format="%.1f",
                    )
                    .props("dense outlined")
                    .classes("w-full q-mt-sm")
                )
                i_vs30 = (
                    ui.number(
                        "VS30 (m/s)",
                        value=800,
                        min=100,
                        max=2000,
                        step=10,
                        format="%.0f",
                    )
                    .props("dense outlined")
                    .classes("w-full q-mt-sm")
                )

                ui.separator().classes("q-my-md")

                ui.label("Model Settings").classes("slabel q-mb-sm")
                i_nsmpl = (
                    ui.number(
                        "GMM Samples (0 = median only)",
                        value=0,
                        min=0,
                        max=1000,
                        step=1,
                    )
                    .props("dense outlined")
                    .classes("w-full")
                )
                i_dt = (
                    ui.number(
                        "Time Step dt (s)",
                        value=0.005,
                        min=0.001,
                        max=0.05,
                        step=0.001,
                        format="%.3f",
                    )
                    .props("dense outlined")
                    .classes("w-full q-mt-sm")
                )
                i_nsim = (
                    ui.number(
                        "Stochastic Realizations", value=1, min=1, max=100, step=1
                    )
                    .props("dense outlined")
                    .classes("w-full q-mt-sm")
                )

                ui.separator().classes("q-my-md")

                with ui.row().classes("items-center w-full q-mb-xs"):
                    ui.label("Conditional Target").classes("slabel")
                    ui.space()
                    cond_toggle = ui.switch("").props("dense color=primary")

                _cond_inputs = []

                with ui.column().classes("w-full") as cond_box:
                    cond_list_container = ui.column().classes("w-full q-gutter-y-xs")

                    def add_cond_row(im_val="M_Sa_1", num_val=0.9):
                        with cond_list_container:
                            with ui.column().classes("w-full q-pa-sm q-mb-xs").style("border: 1px dashed var(--border); border-radius: 6px; background: rgba(0,0,0,0.1);") as cond_item:
                                with ui.row().classes("w-full items-center justify-between q-mb-xs"):
                                    ui.label("Condition").style("font-size: 0.75rem; color: var(--text-muted); font-weight: 600;")
                                    close_btn = ui.button(icon="close").props("flat dense color=negative").style("padding: 0; min-height: 0; min-width: 0;")
                                
                                cim = (
                                    ui.select(
                                        options=_COND_IMS,
                                        label="Target IM",
                                        value=im_val,
                                    )
                                    .props("dense outlined")
                                    .classes("w-full")
                                )
                                cval = (
                                    ui.number(
                                        "Target Value (e.g. g)",
                                        value=num_val,
                                        min=1e-4,
                                        step=0.01,
                                        format="%.4f",
                                    )
                                    .props("dense outlined")
                                    .classes("w-full q-mt-xs")
                                )

                                def remove_self(e, r=cond_item, c=cim, v=cval):
                                    r.delete()
                                    if (c, v) in _cond_inputs:
                                        _cond_inputs.remove((c, v))
                                        
                                close_btn.on_click(remove_self)
                                
                            _cond_inputs.append((cim, cval))

                    add_cond_row()

                    ui.button(
                        "Add Condition",
                        icon="add",
                        on_click=lambda: add_cond_row(im_val="M_Sa_1", num_val=0.9),
                    ).props("flat dense color=primary").classes("w-full q-mt-xs")

                cond_box.set_visibility(False)
                cond_toggle.on_value_change(lambda e: cond_box.set_visibility(e.value))

                ui.separator().classes("q-my-md")

                ui.label("Output Settings").classes("slabel q-mb-sm")
                with ui.row().classes("w-full"):
                    i_outdir = (
                        ui.input("Save Directory", value=str(OUT_DIR))
                        .props("outlined dense")
                        .classes("flex-grow")
                    )

                ui.separator().classes("q-my-md")

                with ui.column().classes("w-full q-gutter-xs"):
                    btn_pred = ui.button(
                        "Predict IMs", icon="analytics", on_click=lambda: on_predict()
                    ).classes("w-full primary-btn q-py-xs")
                    btn_sim = ui.button(
                        "Simulate", icon="waves", on_click=lambda: on_simulate()
                    ).classes("w-full accent-btn q-py-xs")
                    ui.button(
                        "Clear", icon="clear_all", on_click=lambda: on_clear()
                    ).props("flat").classes("w-full text-grey-5")

        # ── RIGHT results panel ───────────────────────────────────────────────
        with ui.column().classes("flex-grow q-pa-sm").style("min-width:0"):
            with ui.card().classes("pcard w-full q-pa-none").style("overflow:hidden"):
                # Tab bar
                with (
                    ui.tabs()
                    .props("dense active-color=primary indicator-color=primary")
                    .classes("w-full q-px-sm") as tabs
                ):
                    t_ims = ui.tab("Intensity Measures", icon="analytics")
                    t_ts = ui.tab("Time Series", icon="waves")
                    t_spectra = ui.tab("Response Spectra", icon="show_chart")
                    t_fas = ui.tab("Fourier", icon="graphic_eq")

                ui.separator().props("dark").classes("w-full")

                # Spinner row (shown while computing)
                with ui.row().classes("items-center q-px-md q-py-sm") as spin_row:
                    ui.spinner("dots", size="xs", color="primary")
                    ui.label("Computing …").style(
                        "font-size:.83rem;color:var(--text-muted);margin-left:.4rem"
                    )
                spin_row.set_visibility(False)

                with ui.tab_panels(tabs, value=t_ims).classes("w-full"):
                    # ── Intensity Measures panel ─────────────────────────────
                    with ui.tab_panel(t_ims):
                        im_panel = ui.column().classes("w-full q-pa-sm q-gutter-sm")
                        with im_panel:
                            ui.label("Run a prediction to see results here.").classes(
                                "result-hint"
                            )

                    # ── Time Series panel ────────────────────────────────────
                    with ui.tab_panel(t_ts):
                        ts_panel = ui.column().classes("w-full q-pa-sm q-gutter-sm")
                        with ts_panel:
                            ui.label(
                                "Run a simulation to see time series here."
                            ).classes("result-hint")

                    # ── Simulated Spectra panel ──────────────────────────────
                    with ui.tab_panel(t_spectra):
                        sa_panel = ui.column().classes("w-full q-pa-sm q-gutter-sm")
                        with sa_panel:
                            ui.label(
                                "Run a simulation to see response spectra here."
                            ).classes("result-hint")

                    # ── FAS panel ────────────────────────────────────────────
                    with ui.tab_panel(t_fas):
                        fas_panel = ui.column().classes("w-full q-pa-sm q-gutter-sm")
                        with fas_panel:
                            ui.label(
                                "Run a simulation to see Fourier spectra here."
                            ).classes("result-hint")

    # ── Helpers: read inputs ──────────────────────────────────────────────────
    def _scenario() -> dict:
        return dict(
            Mw=float(i_mw.value or 6.5),
            Ztor=float(i_ztor.value or 3.0),
            Rrup=float(i_rrup.value or 15.0),
            Vs30=float(i_vs30.value or 800.0),
            Fm=_FM_LABELS.get(str(i_fm.value), "0"),
        )

    def _conditions() -> dict | None:
        if cond_toggle.value and _cond_inputs:
            conds = {}
            for cim, cval in _cond_inputs:
                if cim.value:
                    conds[str(cim.value)] = float(cval.value or 0.9)
            return conds if conds else None
        return None

    def _out_dir() -> Path:
        d = Path(str(i_outdir.value or OUT_DIR))
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _busy(state: bool, btn=None) -> None:
        spin_row.set_visibility(state)
        if btn:
            if state:
                btn.disable()
            else:
                btn.enable()

    # ── Render helpers ────────────────────────────────────────────────────────
    def _render_ims(pred: pd.DataFrame) -> None:
        im_panel.clear()
        with im_panel:
            # Response spectrum chart
            ui.plotly(fig_spectrum_predicted(pred)).classes("w-full").style(
                "min-height:380px"
            )

            # Key IM summary table
            key = [
                c
                for c in [
                    "M_Sa_0",
                    "M_Sa_0.2",
                    "M_Sa_1",
                    "M_PGV",
                    "I_Sa_0",
                    "I_Sa_0.2",
                    "I_Sa_1",
                    "I_PGV",
                    "V_Sa_0",
                    "V_Sa_0.2",
                    "V_Sa_1",
                    "V_PGV",
                ]
                if c in pred.columns
            ]

            ui.label("Key Intensity Measures").classes("slabel q-mt-sm")
            ui.table(
                columns=[
                    {"name": c, "label": c, "field": c, "align": "left"} for c in key
                ],
                rows=pred[key].round(5).to_dict("records"),
            ).props("dark dense flat bordered").classes("w-full")

            with ui.row().classes("q-mt-xs"):

                async def _save_pred():
                    fp = _out_dir() / "prediction.csv"
                    pred.to_csv(fp, index=False)
                    ui.notify(f"Saved: {fp}", type="positive")

                ui.button(
                    "Save prediction csv", icon="download", on_click=_save_pred
                ).props("flat color=primary").style("font-size:.82rem")

    def _render_ts() -> None:
        ts_panel.clear()
        with ts_panel:
            ui.plotly(fig_timeseries(_s.ts_m, _s.ts_i, _s.ts_v)).classes(
                "w-full"
            ).style("min-height:540px")

            with ui.row().classes("q-mt-xs q-gutter-xs"):
                for comp, attr, label in [
                    ("Major", "ts_m", "Major"),
                    ("Intermediate", "ts_i", "Intermediate"),
                    ("Vertical", "ts_v", "Vertical"),
                ]:

                    async def _save_ts(a=attr, lbl=label):
                        gm_list = getattr(_s, a)
                        fp = _out_dir() / f"ts_{lbl.lower()}.csv"
                        save_timeseries(gm_list, fp, dt=_s.sim_dt)
                        ui.notify(f"Saved: {fp}", type="positive")

                    ui.button(
                        f"Save {label.lower()}", icon="save", on_click=_save_ts
                    ).props("flat color=grey-4").style("font-size:.8rem")

    def _render_spectra() -> None:
        sa_panel.clear()
        with sa_panel:
            T = np.logspace(-2, np.log10(4), 80)
            ui.plotly(fig_simulated_spectra(_s.ts_m, _s.ts_i, _s.ts_v, T)).classes(
                "w-full"
            ).style("min-height:420px")

            with ui.row().classes("q-mt-xs q-gutter-xs"):
                for attr, label in [
                    ("ts_m", "Major"),
                    ("ts_i", "Intermediate"),
                    ("ts_v", "Vertical"),
                ]:

                    async def _save_sa(a=attr, lbl=label):
                        gm_list = getattr(_s, a)
                        fp = _out_dir() / f"spectra_{lbl.lower()}.csv"
                        save_spectra(gm_list, fp, T)
                        ui.notify(f"Saved: {fp}", type="positive")

                    ui.button(
                        f"Save {label.lower()} spectra", icon="save", on_click=_save_sa
                    ).props("flat color=grey-4").style("font-size:.8rem")

    def _render_fas() -> None:
        fas_panel.clear()
        with fas_panel:
            ui.plotly(fig_fas(_s.ts_m, _s.ts_i, _s.ts_v)).classes("w-full").style(
                "min-height:420px"
            )

    # ── Event handlers ─────────────────────────────────────────────────────────
    async def on_predict():
        status_lbl.text = "Predicting…"
        _busy(True, btn_pred)
        try:
            gmm = await run.io_bound(_get_gmm)
            pred = await run.io_bound(
                lambda: gmm.predict(
                    **_scenario(),
                    n_sample=int(i_nsmpl.value or 0),
                    conditions=_conditions(),
                )
            )
            _s.prediction = pred
            _render_ims(pred)
            tabs.set_value(t_ims)
            status_lbl.text = "Prediction done"
            ui.notify("Prediction complete ✓", type="positive", timeout=2500)
        except Exception as exc:
            status_lbl.text = "Error"
            ui.notify(f"Error: {exc}", type="negative", timeout=6000)
        finally:
            _busy(False, btn_pred)

    async def on_simulate():
        status_lbl.text = "Simulating…"
        _busy(True, btn_sim)
        try:
            gmm = await run.io_bound(_get_gmm)
            dt = float(i_dt.value or 0.005)
            _s.sim_dt = dt

            result = await run.io_bound(
                lambda: gmm.simulate(
                    **_scenario(),
                    conditions=_conditions(),
                    n_samples=int(i_nsmpl.value or 0),
                    n_simulations=int(i_nsim.value or 1),
                    dt=dt,
                )
            )
            _s.ts_m, _s.ts_i, _s.ts_v = result

            _render_ts()
            _render_spectra()
            _render_fas()
            tabs.set_value(t_ts)
            status_lbl.text = "Simulation done"
            ui.notify("Simulation complete ✓", type="positive", timeout=2500)
        except Exception as exc:
            status_lbl.text = "Error"
            ui.notify(f"Error: {exc}", type="negative", timeout=6000)
        finally:
            _busy(False, btn_sim)

    def on_clear():
        _s.prediction = None
        _s.ts_m = _s.ts_i = _s.ts_v = None
        for panel in (im_panel, ts_panel, sa_panel, fas_panel):
            panel.clear()
            with panel:
                ui.label("No data.").classes("result-hint")
        status_lbl.text = "Ready"
        ui.notify("Cleared.", timeout=1500)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_app():
    import os
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "127.0.0.1")
    show = os.environ.get("SHOW", "true").lower() == "true"

    ui.run(
        root=build_page,
        title="PINAGMM",
        dark=True,
        favicon="🌊",
        host=host,
        port=port,
        reload=False,
        show=show,
    )


if __name__ == "__main__":
    run_app()
