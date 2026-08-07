"""
PINAGMM top-level layout.
navbar → error-banner → app-body (dark sidebar + dcc.Loading main content).
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from .components.navbar import create_navbar
from .components.sidebar import create_sidebar
from .pages.tab_fourier import tab_fourier
from .pages.tab_prediction import tab_prediction
from .pages.tab_simspectra import tab_simspectra
from .pages.tab_timeseries import tab_timeseries


def serve_layout() -> html.Div:
    return html.Div(
        [
            # ── Persistent data stores ────────────────────────────────
            dcc.Store(id="pred-store", data=None),
            dcc.Store(id="sim-store", data=None),
            # ── Top navigation bar (contains status-pill) ─────────────
            create_navbar(),
            # ── App body: dark sidebar + tabbed main content ───────────
            html.Div(
                [
                    # Dark left panel
                    create_sidebar(),
                    # Main content area — wrapped in Loading so a spinner
                    # appears over the graphs while callbacks are computing
                    dcc.Loading(
                        id="main-loading",
                        type="circle",
                        color="#4f46e5",
                        overlay_style={
                            "visibility": "visible",
                            "opacity": 0.10,
                            "backgroundColor": "white",
                        },
                        children=html.Div(
                            [
                                dbc.Tabs(
                                    [
                                        tab_prediction(),
                                        tab_timeseries(),
                                        tab_simspectra(),
                                        tab_fourier(),
                                    ],
                                    id="main-tabs",
                                    active_tab="tab-pred",
                                ),
                            ],
                            className="main-content",
                        ),
                        style={"flex": 1, "minWidth": 0, "overflowY": "auto"},
                    ),
                ],
                className="app-body",
            ),
        ],
        style={
            "fontFamily": "'Inter', 'Segoe UI', -apple-system, sans-serif",
            "minHeight": "100vh",
        },
    )
