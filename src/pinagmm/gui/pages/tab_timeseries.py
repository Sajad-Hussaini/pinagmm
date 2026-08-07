import dash_bootstrap_components as dbc
from dash import dcc, html

from ..components.chart_helpers import GRAPH_CONFIG


def tab_timeseries() -> dbc.Tab:
    return dbc.Tab(
        label="Simulated Time Series",
        tab_id="tab-ts",
        children=[
            # ── Signal type switcher ──────────────────────────────
            dbc.RadioItems(
                id="ts-channel",
                options=[
                    {"label": "Acceleration", "value": "ac"},
                    {"label": "Velocity", "value": "vel"},
                    {"label": "Displacement", "value": "disp"},
                ],
                value="ac",
                inline=True,
                class_name="ts-switcher",
            ),
            # ── 3-row combined chart (Major / Intermediate / Vertical) ─────
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "Simulated Ground Motion — Three Components",
                                className="graph-card-title",
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-download me-1"),
                                            "Download CSV",
                                        ],
                                        id="btn-dl-ts",
                                        class_name="btn-dl",
                                    ),
                                    dcc.Download(id="download-ts"),
                                ],
                            ),
                        ],
                        className="graph-card-header",
                    ),
                    dcc.Graph(
                        id="graph-timeseries",
                        config=GRAPH_CONFIG,
                        style={
                            "height": "480px",
                            "overflowY": "visible",
                        },
                    ),
                ],
                className="pinagmm-card",
            ),
            # ── Empty-state hint ─────────────────────────────────
            html.Div(
                [
                    html.I(
                        className="fas fa-wave-square",
                        style={
                            "fontSize": "2rem",
                            "color": "#cbd5e1",
                            "marginBottom": "12px",
                        },
                    ),
                    html.Br(),
                    "Click ",
                    html.Strong("Simulate"),
                    " to generate ground-motion time series.",
                ],
                id="ts-hint",
                className="hint-card",
                style={"marginTop": "12px"},
            ),
        ],
    )
