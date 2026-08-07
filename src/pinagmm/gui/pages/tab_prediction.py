import dash_bootstrap_components as dbc
from dash import dcc, html

from ..components.chart_helpers import GRAPH_CONFIG


def tab_prediction() -> dbc.Tab:
    return dbc.Tab(
        label="GMM Prediction",
        tab_id="tab-pred",
        children=[
            # ── Section switcher ─────────────────────────────────
            dbc.RadioItems(
                id="pred-view",
                options=[
                    {"label": "Response Spectra", "value": "spectra"},
                    {"label": "Intensity Measures", "value": "ims"},
                ],
                value="spectra",
                inline=True,
                class_name="ts-switcher",
            ),
            # ── SECTION: Response Spectra ─────────────────────────
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "GMM-Predicted Response Spectra",
                                className="graph-card-title",
                            ),
                            html.Div([
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-download me-1"),
                                        "Download CSV",
                                    ],
                                    id="btn-dl-pred",
                                    class_name="btn-dl",
                                ),
                                dcc.Download(id="download-pred"),
                            ]),
                        ],
                        className="graph-card-header",
                    ),
                    dcc.Graph(
                        id="graph-pred-spectra",
                        config=GRAPH_CONFIG,
                        style={"height": "480px"},
                    ),
                ],
                className="pinagmm-card",
                id="pred-section-spectra",
            ),
            # ── SECTION: IM Table ─────────────────────────────────
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "GMM-Predicted Intensity Measures",
                                className="graph-card-title",
                            ),
                            html.Div([
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-download me-1"),
                                        "Download CSV",
                                    ],
                                    id="btn-dl-ims",
                                    class_name="btn-dl",
                                ),
                                dcc.Download(id="download-ims"),
                            ]),
                        ],
                        className="graph-card-header",
                    ),
                    html.Div(id="table-pred-ims-container"),
                ],
                className="pinagmm-card",
                id="pred-section-ims",
                style={"display": "none"},
            ),
            # ── Empty-state hint ─────────────────────────────────
            html.Div(
                [
                    html.I(
                        className="fas fa-bullseye",
                        style={
                            "fontSize": "2rem",
                            "color": "#cbd5e1",
                            "marginBottom": "12px",
                        },
                    ),
                    html.Br(),
                    "Click ",
                    html.Strong("Predict"),
                    " to run the GMM and view predicted spectra & IMs.",
                ],
                id="pred-hint",
                className="hint-card",
                style={"marginTop": "12px"},
            ),
        ],
    )
