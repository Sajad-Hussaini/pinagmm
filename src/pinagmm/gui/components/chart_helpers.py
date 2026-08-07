"""
PINAGMM Chart Helpers
=====================
Plotly template, colour palette, and graph_card() helper — fully analogous
to sgsim's chart_helpers.py, adapted for three-component GMM output.

Component colour palette:
  Major       → indigo   #4f46e5
  Intermediate → cyan    #0891b2
  Vertical    → emerald  #059669
"""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html


# ─── Colour Palette ───────────────────────────────────────────────────────────
# Colours chosen for maximum visual discrimination:
#   Major        → indigo  (cool, dominant)
#   Intermediate → vivid orange (warm, clearly different from both others)
#   Vertical     → emerald green (neutral mid-tone)
COMP_PALETTE = {
    "Major": "#4f46e5",  # indigo
    "Intermediate": "#e16b16",  # vivid orange
    "Vertical": "#059669",  # emerald
}

# Faint versions for cloud/ensemble traces
COMP_PALETTE_FAINT = {
    "Major": "rgba(79, 70, 229, {a})",
    "Intermediate": "rgba(225, 107, 22, {a})",
    "Vertical": "rgba(5, 150, 105, {a})",
}

# Generic series palette (for multi-sample traces ordered by sample index)
SERIES_PALETTE = [
    "#4f46e5",  # indigo
    "#e16b16",  # orange
    "#059669",  # emerald
    "#d97706",  # amber
    "#dc2626",  # red
    "#9333ea",  # purple
    "#be185d",  # pink
    "#0f766e",  # teal
]

FAINT_LINE = {"color": "rgba(148, 163, 184, 0.35)", "width": 1.0}


# ─── Custom Plotly Template ────────────────────────────────────────────────────
PINAGMM_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font={
            "family": "Inter, 'Segoe UI', -apple-system, sans-serif",
            "color": "#0f172a",
            "size": 12,
        },
        colorway=list(COMP_PALETTE.values()),
        xaxis={
            "gridcolor": "#e8ecf0",
            "gridwidth": 1,
            "linecolor": "#cbd5e1",
            "linewidth": 1,
            "zerolinecolor": "#e2e8f0",
            "zerolinewidth": 1,
            "tickfont": {"size": 11, "color": "#64748b"},
            "title_font": {
                "size": 12,
                "color": "#475569",
                "family": "Inter, sans-serif",
            },
            "exponentformat": "power",
            "minor": {
                "showgrid": True,
                "gridcolor": "#f1f5f9",
                "gridwidth": 0.5,
            },
        },
        yaxis={
            "gridcolor": "#e8ecf0",
            "gridwidth": 1,
            "linecolor": "#cbd5e1",
            "linewidth": 1,
            "zerolinecolor": "#e2e8f0",
            "zerolinewidth": 1,
            "tickfont": {"size": 11, "color": "#64748b"},
            "title_font": {
                "size": 12,
                "color": "#475569",
                "family": "Inter, sans-serif",
            },
            "exponentformat": "power",
            "minor": {
                "showgrid": True,
                "gridcolor": "#f1f5f9",
                "gridwidth": 0.5,
            },
        },
        legend={
            "bgcolor": "rgba(255,255,255,0.90)",
            "bordercolor": "#e2e8f0",
            "borderwidth": 1,
            "font": {"size": 11, "color": "#475569"},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 58, "r": 16, "t": 30, "b": 48},
        hoverlabel={
            "bgcolor": "#1e293b",
            "font_color": "#f1f5f9",
            "bordercolor": "#1e293b",
            "font_size": 12,
        },
    )
)


# ─── Common graph config ──────────────────────────────────────────────────────
GRAPH_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "svg", "scale": 2},
}


def make_fig(
    x_label: str,
    y_label: str,
    log_x: bool = False,
    log_y: bool = False,
    uirevision=True,
) -> go.Figure:
    """Return an empty figure with the PINAGMM design template applied."""
    fig = go.Figure()
    fig.update_layout(
        template=PINAGMM_TEMPLATE,
        xaxis_title=x_label,
        yaxis_title=y_label,
        uirevision=uirevision,
    )
    if log_x:
        fig.update_xaxes(type="log", dtick=1)
    if log_y:
        fig.update_yaxes(type="log", dtick=1)
    return fig


def graph_card(
    graph_id: str,
    title: str,
    dl_btn_id: str,
    dl_comp_id: str,
    height: int = 280,
) -> html.Div:
    """A white card containing a titled chart with a CSV download button."""
    return html.Div(
        [
            html.Div(
                [
                    html.Span(title, className="graph-card-title"),
                    dbc.Button(
                        [html.I(className="fas fa-download me-1"), "CSV"],
                        id=dl_btn_id,
                        class_name="btn-dl",
                    ),
                ],
                className="graph-card-header",
            ),
            dcc.Graph(
                id=graph_id,
                config=GRAPH_CONFIG,
                style={"height": f"{height}px"},
            ),
            dcc.Download(id=dl_comp_id),
        ],
        className="pinagmm-card",
    )
