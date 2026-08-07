from dash import html
import pinagmm


def create_navbar() -> html.Div:
    """Top navigation bar — logo, title, status pill aligned with version badge."""
    return html.Div(
        [
            html.Img(src="/assets/logo.png", className="navbar-logo"),
            html.Span("PINAGMM", className="navbar-title"),
            html.Span(
                "Physics-Informed Neural Additive Ground Motion Model",
                className="navbar-subtitle",
            ),
            html.Span(className="navbar-sep"),
            # ── Status + version group (right side, same flex row) ────────
            html.Div(
                [
                    # Spinning icon — shown by clientside callback on button click
                    html.I(
                        className="fas fa-circle-notch fa-spin",
                        id="nav-spinner",
                        style={
                            "display": "none",
                            "color": "var(--accent)",
                            "fontSize": "0.78rem",
                        },
                    ),
                    # Status text pill
                    html.Span(
                        "Ready",
                        id="status-pill",
                        className="status-pill",
                    ),
                    # Vertical divider
                    html.Span(
                        style={
                            "width": "1px",
                            "height": "16px",
                            "background": "rgba(148,163,184,0.3)",
                            "margin": "0 8px",
                        }
                    ),
                    # Version badge
                    html.Span(
                        f"v{pinagmm.__version__}",
                        className="navbar-version",
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "7px",
                },
            ),
        ],
        className="pinagmm-navbar",
    )
