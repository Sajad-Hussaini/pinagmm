import dash_bootstrap_components as dbc
from dash import dcc, html

from ..components.chart_helpers import GRAPH_CONFIG


def tab_simspectra() -> dbc.Tab:
    return dbc.Tab(
        label="Simulated Response Spectra",
        tab_id="tab-simsa",
        children=[
            # ── Dual switcher row ─────────────────────────────────
            html.Div(
                [
                    dbc.RadioItems(
                        id="simsa-comp-view",
                        options=[
                            {"label": "All Components", "value": "all"},
                            {"label": "Major", "value": "major"},
                            {"label": "Intermediate", "value": "inter"},
                            {"label": "Vertical", "value": "vert"},
                        ],
                        value="all",
                        inline=True,
                        class_name="ts-switcher",
                    ),
                    dbc.RadioItems(
                        id="simsa-yscale",
                        options=[
                            {"label": "Log Y", "value": "log"},
                            {"label": "Linear Y", "value": "linear"},
                        ],
                        value="log",
                        inline=True,
                        class_name="ts-switcher",
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "18px",
                    "flexWrap": "wrap",
                    "alignItems": "center",
                },
            ),
            # ── Combined chart ────────────────────────────────────
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "Simulated Response Spectra  "
                                "(cloud = individual simulations - solid = first simulation)",
                                className="graph-card-title",
                            ),
                            html.Div([
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-download me-1"),
                                        "Download CSV",
                                    ],
                                    id="btn-dl-simsa",
                                    class_name="btn-dl",
                                ),
                                dcc.Download(id="download-simsa"),
                            ]),
                        ],
                        className="graph-card-header",
                    ),
                    dcc.Graph(
                        id="graph-simspectra",
                        config=GRAPH_CONFIG,
                        style={"height": "480px"},
                    ),
                ],
                className="pinagmm-card",
                id="simsa-section-all",
            ),
            # ── Per-component charts ──────────────────────────────
            *[
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    f"{label} — Simulated Response Spectra",
                                    className="graph-card-title",
                                ),
                                html.Div([
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-download me-1"),
                                            "Download CSV",
                                        ],
                                        id=f"btn-dl-simsa-{key}",
                                        class_name="btn-dl",
                                    ),
                                    dcc.Download(id=f"download-simsa-{key}"),
                                ]),
                            ],
                            className="graph-card-header",
                        ),
                        dcc.Graph(
                            id=f"graph-simsa-{key}",
                            config=GRAPH_CONFIG,
                            style={"height": "480px"},
                        ),
                    ],
                    className="pinagmm-card",
                    id=f"simsa-section-{key}",
                    style={"display": "none"},
                )
                for label, key in [
                    ("Major", "major"),
                    ("Intermediate", "inter"),
                    ("Vertical", "vert"),
                ]
            ],
            # ── Empty-state hint ─────────────────────────────────
            html.Div(
                [
                    html.I(
                        className="fas fa-chart-line",
                        style={
                            "fontSize": "2rem",
                            "color": "#cbd5e1",
                            "marginBottom": "12px",
                        },
                    ),
                    html.Br(),
                    "Click ",
                    html.Strong("Simulate"),
                    " to compute simulated response spectra.",
                ],
                id="simsa-hint",
                className="hint-card",
                style={"marginTop": "12px"},
            ),
        ],
    )
