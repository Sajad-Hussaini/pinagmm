import dash
import dash_bootstrap_components as dbc

from pinagmm.gui.callbacks.run_cb import register_run_callbacks
from pinagmm.gui.callbacks.charts_cb import register_charts_callbacks
from pinagmm.gui.layout import serve_layout

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="PINAGMM — Neural Additive Ground Motion Model",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {
            "name": "description",
            "content": (
                "PINAGMM — Physics-Informed Neural Additive Ground Motion Model "
                "for Hazard-Compatible Three-Component Stochastic Simulation."
            ),
        },
    ],
)

# Use the PINAGMM logo as favicon (override default Dash favicon)
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/png" href="/assets/logo.png">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

app.layout = serve_layout()

# ── Register all callbacks ─────────────────────────────────────────────────
register_run_callbacks(app)
register_charts_callbacks(app)

# ── Clientside callbacks: immediately show "Computing..." on button click ──
for _btn_id in ("btn-predict", "btn-simulate"):
    app.clientside_callback(
        """
        function(n_clicks) {
            if (n_clicks && n_clicks > 0) {
                return ['Computing…', 'status-pill busy'];
            }
            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        """,
        dash.Output("status-pill", "children", allow_duplicate=True),
        dash.Output("status-pill", "className", allow_duplicate=True),
        dash.Input(_btn_id, "n_clicks"),
        prevent_initial_call=True,
    )

server = app.server  # WSGI entry-point


def run_app(
    host: str | None = None, port: int | None = None, debug: bool = False
) -> None:
    """Entry point for running the Dash application locally or in production."""
    import os
    import threading
    import webbrowser

    host = host or os.environ.get("HOST", "127.0.0.1")
    port = int(port or os.environ.get("PORT", "8050"))
    show_browser = os.environ.get("SHOW", "true").lower() != "false"

    if show_browser and host in ("127.0.0.1", "localhost"):
        threading.Timer(
            0.25, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")
        ).start()

    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    run_app()
