import dash_bootstrap_components as dbc
from dash import dcc, html


# ─── Fault mechanism options ──────────────────────────────────────────────────
_FM_OPTIONS = [
    {"label": "Strike-Slip", "value": "0"},
    {"label": "Normal", "value": "1"},
    {"label": "Reverse", "value": "2"},
    {"label": "Reverse-Oblique", "value": "3"},
    {"label": "Normal-Oblique", "value": "4"},
]


# ─── Label helpers ────────────────────────────────────────────────────────────
def _label(children) -> html.Span:
    """Accepts a string or list of html elements (for subscripts)."""
    return html.Span(children, className="sb-label")


def _tip(text: str) -> html.Small:
    return html.Small(text, className="sb-tip")


def _num_input(
    id_: str,
    placeholder=None,
    value=None,
    step=None,
    min_=None,
    max_=None,
    debounce=True,
) -> dbc.Input:
    return dbc.Input(
        id=id_,
        type="number",
        placeholder=placeholder,
        value=value,
        step=step,
        min=min_,
        max=max_,
        debounce=debounce,
        size="sm",
    )


# ─── Accordion items ──────────────────────────────────────────────────────────


def _scenario_item() -> dbc.AccordionItem:
    """Earthquake & site scenario — with subscripted parameter labels."""
    return dbc.AccordionItem(
        item_id="scenario",
        title="Event Scenario Parameters",
        children=[
            # Mw
            _label(["Moment Magnitude  M", html.Sub("w")]),
            _num_input("sb-mw", value=6.5, min_=4.0, max_=8.5, step=0.1),
            # Ztor
            _label(["Depth to Top of Rupture  Z", html.Sub("tor"), "  (km)"]),
            _num_input("sb-ztor", value=3.0, min_=0.0, max_=30.0, step=0.5),
            # Rrup
            _label(["Source to Site Distance  R", html.Sub("rup"), "  (km)"]),
            _num_input("sb-rrup", value=15.0, min_=1.0, max_=400.0, step=1.0),
            # VS30
            _label(["Site Shear Wave Velocity  V", html.Sub("S30"), "  (m/s)"]),
            _num_input("sb-vs30", value=800, min_=100, max_=2000, step=10),
            # Fault mechanism — no tip, the labels are self-explanatory
            _label("Fault Mechanism"),
            dbc.Select(
                id="sb-fm",
                options=_FM_OPTIONS,
                value="0",
                size="sm",
            ),
        ],
    )


def _simulation_item() -> dbc.AccordionItem:
    """Simulation settings with clear median vs. sampling UI."""
    return dbc.AccordionItem(
        item_id="simulation",
        title="Simulation Settings",
        children=[
            # ── Prediction mode ──────────────────────────────────────
            _label("GMM Prediction Mode"),
            dbc.RadioItems(
                id="sb-pred-mode",
                options=[
                    {
                        "label": "Median from GMM Distribution",
                        "value": "median",
                    },
                    {"label": "Sample from GMM Distribution", "value": "sample"},
                ],
                value="median",
                class_name="sb-radio",
            ),
            # Sample count — only visible when mode == "sample"
            html.Div(
                [
                    _label("Number of Samples"),
                    _num_input("sb-nsamples", value=10, min_=1, max_=500, step=1),
                    _tip(
                        "Each sample draws a unique set of model parameters from "
                        "the learned multivariate distribution."
                    ),
                ],
                id="sb-nsamples-div",
                style={"display": "none"},
            ),
            html.Hr(style={"borderColor": "var(--sb-border)", "margin": "10px 0 8px"}),
            # ── Stochastic simulation ────────────────────────────────
            _label("Stochastic Simulations per Prediction"),
            _num_input("sb-nsim", value=1, min_=1, max_=100, step=1),
            _tip(
                "Each prediction is passed through the stochastic model "
                "(sgsim) to generate independent ground motion time series."
            ),
            _label(["Time Step  Δt  (s)"]),
            _num_input("sb-dt", value=0.005, min_=0.001, max_=0.05, step=0.001),
            _tip("Default 0.005 s  →  Nyquist 100 Hz sampling rate."),
        ],
    )


def _conditional_item() -> dbc.AccordionItem:
    """Conditional hazard targeting accordion item."""
    return dbc.AccordionItem(
        item_id="conditional",
        title="Conditional Hazard Target (optional)",
        children=[
            html.Div(
                [
                    html.I(className="fas fa-info-circle me-1"),
                    "Pin one or more Intensity Measures to a target value. "
                    "Sampling is then conditioned on those targets via the "
                    "Schur complement of the GMM covariance.",
                ],
                className="pred-info-card",
                style={"marginBottom": "8px"},
            ),
            html.Div(id="cond-rows-container"),
            dbc.Button(
                [html.I(className="fas fa-plus me-1"), "Add Condition"],
                id="btn-add-cond",
                class_name="btn-add-cond",
                n_clicks=0,
            ),
        ],
    )


# ─── Main sidebar assembler ───────────────────────────────────────────────────
def create_sidebar() -> html.Div:
    return html.Div(
        [
            # ══ All input sections as accordion ══════════════════════════════
            dbc.Accordion(
                [
                    _scenario_item(),
                    _simulation_item(),
                    _conditional_item(),
                ],
                always_open=True,
                active_item=["scenario"],  # scenario open by default
                flush=True,
                class_name="ops-accordion",
            ),
            html.Hr(className="sb-divider"),
            # ══ § ACTION BUTTONS ═════════════════════════════════════════════
            # Predict
            dbc.Button(
                [html.I(className="fas fa-chart-area me-2"), "Predict"],
                id="btn-predict",
                class_name="btn-predict",
                n_clicks=0,
            ),
            dbc.Tooltip(
                [
                    html.Strong("Predict  "),
                    "runs the GMM to output: ",
                    html.Br(),
                    "• Median intensity measures (e.g., Sa) per component",
                    html.Br(),
                    "• Stochastic simulation parameters (ω, ζ, D, …)",
                    html.Br(),
                    html.Br(),
                    html.Em("No time series are generated at this stage."),
                ],
                target="btn-predict",
                placement="right",
                style={"maxWidth": "350px", "textAlign": "left"},
            ),
            # Simulate
            dbc.Button(
                [html.I(className="fas fa-wave-square me-2"), "Simulate"],
                id="btn-simulate",
                class_name="btn-simulate",
                n_clicks=0,
            ),
            dbc.Tooltip(
                [
                    html.Strong("Simulate  "),
                    "runs the end-to-end generative framework: ",
                    html.Br(),
                    "• Calls the GMM",
                    html.Br(),
                    "• Runs the stochastic model (sgsim) ",
                    html.Br(),
                    "• Computes simulated time series, Fourier, and response spectra",
                    html.Br(),
                    html.Br(),
                    html.Em(
                        "IMs from the simulated time series differ "
                        "from the GMM-predicted IMs due to the natural stochastic variability."
                    ),
                ],
                target="btn-simulate",
                placement="right",
                style={"maxWidth": "340px", "textAlign": "left"},
            ),
            # Clear
            dbc.Button(
                [html.I(className="fas fa-rotate-left me-2"), "Clear"],
                id="btn-clear",
                class_name="btn-clear",
                n_clicks=0,
            ),
            # Hidden stores
            dcc.Store(id="cond-store", data=[]),
        ],
        className="pinagmm-sidebar",
    )
