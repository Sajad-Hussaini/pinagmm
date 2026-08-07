import dash_bootstrap_components as dbc
from dash import dcc, html

from ..components.chart_helpers import GRAPH_CONFIG


def tab_fourier() -> dbc.Tab:
    return dbc.Tab(
        label="Simulated Fourier Spectra",
        tab_id="tab-fas",
        children=[
            # ── Dual switcher row ─────────────────────────────────
            html.Div(
                [
                    dbc.RadioItems(
                        id="fas-comp-view",
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
                        id="fas-yscale",
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
            # ── Combined FAS chart ────────────────────────────────
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "Fourier Amplitude Spectra — Three Components",
                                className="graph-card-title",
                            ),
                            html.Div([
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-download me-1"),
                                        "Download CSV",
                                    ],
                                    id="btn-dl-fas",
                                    class_name="btn-dl",
                                ),
                                dcc.Download(id="download-fas"),
                            ]),
                        ],
                        className="graph-card-header",
                    ),
                    dcc.Graph(
                        id="graph-fas", config=GRAPH_CONFIG, style={"height": "480px"}
                    ),
                ],
                className="pinagmm-card",
                id="fas-section-all",
            ),
            # ── Per-component charts ──────────────────────────────
            *[
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    f"{label} — Fourier Amplitude Spectra",
                                    className="graph-card-title",
                                ),
                                html.Div([
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-download me-1"),
                                            "Download CSV",
                                        ],
                                        id=f"btn-dl-fas-{key}",
                                        class_name="btn-dl",
                                    ),
                                    dcc.Download(id=f"download-fas-{key}"),
                                ]),
                            ],
                            className="graph-card-header",
                        ),
                        dcc.Graph(
                            id=f"graph-fas-{key}",
                            config=GRAPH_CONFIG,
                            style={"height": "480px"},
                        ),
                    ],
                    className="pinagmm-card",
                    id=f"fas-section-{key}",
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
                        className="fas fa-signal",
                        style={
                            "fontSize": "2rem",
                            "color": "#cbd5e1",
                            "marginBottom": "12px",
                        },
                    ),
                    html.Br(),
                    "Click ",
                    html.Strong("Simulate"),
                    " to compute Fourier amplitude spectra.",
                ],
                id="fas-hint",
                className="hint-card",
                style={"marginTop": "12px"},
            ),
        ],
    )
