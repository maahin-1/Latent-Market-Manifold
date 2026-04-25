import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Config
from regime import REGIME_COLORS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static Plotly figure builder (used in both Dash and standalone export)
# ---------------------------------------------------------------------------

def build_manifold_figure(
    latent: np.ndarray,
    dates: pd.DatetimeIndex,
    color_by: np.ndarray,
    regime_labels: Optional[np.ndarray] = None,
    loadings: Optional[pd.DataFrame] = None,
    cfg: Config = None,
    title: str = '3D Latent Market Manifold',
) -> go.Figure:
    """
    Build a Plotly figure with:
      - 3D scatter + trajectory coloured by volatility
      - Regime cluster colouring option
      - PCA loadings bar chart (if loadings provided and method == pca)
      - Animated frame-by-frame playback
    """
    if cfg is None:
        cfg = Config()

    n = len(latent)
    x, y, z = latent[:, 0], latent[:, 1], latent[:, 2]
    date_strs = [str(d)[:10] for d in dates]

    show_loadings = cfg.show_loadings and loadings is not None
    cols = 2 if show_loadings else 1
    specs = [[{'type': 'scene'}, {'type': 'xy'}]] if show_loadings else [[{'type': 'scene'}]]
    fig = make_subplots(
        rows=1, cols=cols,
        specs=specs,
        subplot_titles=[title, 'PCA Component Loadings'] if show_loadings else [title],
        column_widths=[0.7, 0.3] if show_loadings else [1.0],
    )

    # --- Regime colour mapping ---
    if regime_labels is not None:
        point_colors = [REGIME_COLORS[int(l) % len(REGIME_COLORS)] for l in regime_labels]
        color_arr = point_colors
        colorscale_arg = None
        showscale = False
    else:
        color_arr = color_by.tolist()
        colorscale_arg = cfg.colorscale
        showscale = True

    # Hover text
    hover = [
        f"Date: {date_strs[i]}<br>"
        f"Volatility: {color_by[i]:.4f}<br>"
        f"Regime: {cfg.regime_labels[int(regime_labels[i])] if regime_labels is not None else 'N/A'}<br>"
        f"PC1: {x[i]:.3f} PC2: {y[i]:.3f} PC3: {z[i]:.3f}"
        for i in range(n)
    ]

    # Full trajectory line
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color='rgba(200,200,200,0.35)', width=2),
        showlegend=False,
        hoverinfo='skip',
    ), row=1, col=1)

    # Scatter points
    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=color_arr,
            colorscale=colorscale_arg,
            showscale=showscale,
            colorbar=dict(title='Volatility', x=0.55) if showscale else None,
            opacity=0.85,
        ),
        text=hover,
        hoverinfo='text',
        name='Market State',
    )
    fig.add_trace(scatter, row=1, col=1)

    # --- PCA loadings bar chart ---
    if show_loadings:
        for i, pc in enumerate(loadings.columns[:3]):
            fig.add_trace(go.Bar(
                name=pc,
                x=loadings.index.tolist(),
                y=loadings[pc].tolist(),
                marker_color=REGIME_COLORS[i % len(REGIME_COLORS)],
            ), row=1, col=2)

    # --- Animation frames ---
    if cfg.animate:
        frames = []
        step = max(1, n // 150)   # cap at ~150 frames for performance
        for end in range(step, n + 1, step):
            sl = slice(0, end)
            c = color_arr[:end] if isinstance(color_arr, list) else color_arr[:end].tolist()
            frames.append(go.Frame(
                data=[
                    go.Scatter3d(x=x[sl], y=y[sl], z=z[sl], mode='lines',
                                 line=dict(color='rgba(200,200,200,0.35)', width=2)),
                    go.Scatter3d(x=x[sl], y=y[sl], z=z[sl], mode='markers',
                                 marker=dict(size=4, color=c, colorscale=colorscale_arg,
                                             showscale=showscale, opacity=0.85),
                                 text=hover[:end], hoverinfo='text'),
                ],
                name=str(end),
            ))
        fig.frames = frames

        fig.update_layout(
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                y=0.02, x=0.1,
                xanchor='right',
                buttons=[
                    dict(label='▶ Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=60, redraw=True),
                                          fromcurrent=True, mode='immediate')]),
                    dict(label='⏸ Pause',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             mode='immediate')]),
                ],
            )],
            sliders=[dict(
                steps=[dict(method='animate',
                            args=[[f.name], dict(mode='immediate', frame=dict(duration=0, redraw=True))],
                            label=f.name) for f in frames],
                transition=dict(duration=0),
                x=0.15, y=0, len=0.85,
            )],
        )

    # Layout
    fig.update_layout(
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#0d0d0d',
        font=dict(color='white'),
        scene=dict(
            bgcolor='#0d0d0d',
            xaxis=dict(title='Latent X', color='white', gridcolor='#333'),
            yaxis=dict(title='Latent Y', color='white', gridcolor='#333'),
            zaxis=dict(title='Latent Z', color='white', gridcolor='#333'),
        ),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')),
        margin=dict(l=0, r=0, t=40, b=60),
        height=700,
    )
    return fig


# ---------------------------------------------------------------------------
# Regime timeline figure
# ---------------------------------------------------------------------------

def build_regime_timeline(
    dates: pd.DatetimeIndex,
    regime_labels: np.ndarray,
    cfg: Config = None,
) -> go.Figure:
    if cfg is None:
        cfg = Config()

    named = [cfg.regime_labels[int(l)] if l < len(cfg.regime_labels) else f'R{l}' for l in regime_labels]
    colors = [REGIME_COLORS[int(l) % len(REGIME_COLORS)] for l in regime_labels]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(dates),
        y=named,
        mode='markers',
        marker=dict(color=colors, size=6),
        hovertemplate='%{x|%Y-%m-%d}<br>Regime: %{y}<extra></extra>',
    ))
    fig.update_layout(
        title='Market Regime Timeline',
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#111',
        font=dict(color='white'),
        xaxis=dict(color='white', gridcolor='#333'),
        yaxis=dict(color='white', gridcolor='#333'),
        height=250,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Dash interactive dashboard
# ---------------------------------------------------------------------------

def launch_dashboard(
    latent: np.ndarray,
    dates: pd.DatetimeIndex,
    color_by: np.ndarray,
    regime_labels: Optional[np.ndarray] = None,
    loadings: Optional[pd.DataFrame] = None,
    cfg: Config = None,
):
    """Launch a Dash web app on localhost."""
    try:
        import dash
        from dash import dcc, html
        import dash_bootstrap_components as dbc
    except ImportError:
        raise ImportError(
            "Dash not installed. Run: pip install dash dash-bootstrap-components"
        )

    if cfg is None:
        cfg = Config()

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    manifold_fig = build_manifold_figure(latent, dates, color_by, regime_labels, loadings, cfg)
    regime_fig = build_regime_timeline(dates, regime_labels, cfg) if regime_labels is not None else go.Figure()

    method_options = [
        {'label': 'PCA', 'value': 'pca'},
        {'label': 't-SNE', 'value': 'tsne'},
        {'label': 'UMAP', 'value': 'umap'},
    ]
    color_options = [
        {'label': 'Volatility', 'value': 'plasma'},
        {'label': 'Viridis', 'value': 'viridis'},
        {'label': 'Inferno', 'value': 'inferno'},
    ]

    initial_tickers = ', '.join(cfg.tickers) if cfg.use_real_data else ''

    app.layout = dbc.Container([
        dbc.Row(dbc.Col(html.H2('3D Latent Market Manifold',
                                style={'color': 'white', 'textAlign': 'center', 'padding': '12px'}))),

        # Ticker input row
        dbc.Row([
            dbc.Col([
                dbc.Label('Tickers (comma-separated, e.g. AAPL, MSFT, TSLA)', style={'color': '#aaa'}),
                dbc.InputGroup([
                    dbc.Input(
                        id='ticker-input',
                        placeholder='AAPL, MSFT, GOOGL, TSLA…',
                        value=initial_tickers,
                        debounce=False,
                        style={'backgroundColor': '#1a1a1a', 'color': 'white', 'border': '1px solid #444'},
                    ),
                    dbc.Button('Fetch & Update', id='ticker-btn', color='primary', n_clicks=0),
                ]),
            ], width=8),
            dbc.Col([
                dbc.Label('Date Range', style={'color': '#aaa'}),
                dbc.InputGroup([
                    dbc.Input(id='start-date-input', placeholder='Start date', value=cfg.start_date,
                              style={'backgroundColor': '#1a1a1a', 'color': 'white', 'border': '1px solid #444'}),
                    dbc.Input(id='end-date-input', placeholder='End date (blank = today)', value='',
                              style={'backgroundColor': '#1a1a1a', 'color': 'white', 'border': '1px solid #444'}),
                ]),
            ], width=4),
        ], style={'padding': '8px 0'}),

        # Status/error banner
        dbc.Row(dbc.Col(html.Div(id='ticker-status', style={'color': '#f39c12', 'fontSize': '13px', 'minHeight': '20px'}))),

        # Controls row
        dbc.Row([
            dbc.Col([
                dbc.Label('Reduction Method', style={'color': '#aaa'}),
                dcc.Dropdown(id='method-dropdown', options=method_options,
                             value=cfg.method, clearable=False,
                             style={'backgroundColor': '#222', 'color': 'white'}),
            ], width=3),
            dbc.Col([
                dbc.Label('Colour Scale', style={'color': '#aaa'}),
                dcc.Dropdown(id='color-dropdown', options=color_options,
                             value=cfg.colorscale, clearable=False,
                             style={'backgroundColor': '#222', 'color': 'white'}),
            ], width=3),
            dbc.Col([
                dbc.Label('Regimes', style={'color': '#aaa'}),
                dcc.Slider(id='regime-slider', min=2, max=6, step=1, value=cfg.n_regimes,
                           marks={i: str(i) for i in range(2, 7)}),
            ], width=3),
            dbc.Col([
                dbc.Label('Show Regime Colours', style={'color': '#aaa'}),
                dbc.Switch(id='regime-toggle', value=True, label=''),
            ], width=3),
        ], style={'padding': '8px 0'}),

        dbc.Row(dbc.Col(dcc.Graph(id='manifold-graph', figure=manifold_fig,
                                  style={'height': '700px'}))),
        dbc.Row(dbc.Col(dcc.Graph(id='regime-timeline', figure=regime_fig,
                                  style={'height': '250px'}))),
    ], fluid=True, style={'backgroundColor': '#0d0d0d'})

    from dash import Input, Output, State, callback_context
    import visualizer as _self

    # Callback 1: fetch new tickers and rebuild the full pipeline
    @app.callback(
        Output('ticker-status', 'children'),
        Input('ticker-btn', 'n_clicks'),
        State('ticker-input', 'value'),
        State('start-date-input', 'value'),
        State('end-date-input', 'value'),
        State('method-dropdown', 'value'),
        State('regime-slider', 'value'),
        State('regime-toggle', 'value'),
        prevent_initial_call=True,
    )
    def fetch_tickers(n_clicks, ticker_str, start_date, end_date, method, n_regimes, show_regimes):
        from data_fetcher import fetch_real_data
        from features import compute_features
        from config import Config

        if not ticker_str or not ticker_str.strip():
            return 'Enter at least one ticker symbol.'

        tickers = [t.strip().upper() for t in ticker_str.replace(',', ' ').split() if t.strip()]
        if not tickers:
            return 'No valid tickers found.'

        try:
            df = fetch_real_data(tickers, start_date or '2020-01-01', end_date or None)
        except Exception as e:
            return f'Fetch error: {e}'

        fetch_cfg = Config(
            use_real_data=True,
            tickers=tickers,
            start_date=start_date or '2020-01-01',
            end_date=end_date or None,
            method=method,
            n_regimes=n_regimes,
        )
        features = compute_features(df, fetch_cfg)
        features_valid = features.dropna(subset=['Return', 'Volatility', 'Momentum'])
        if features_valid.empty:
            return 'No valid rows after feature computation. Try different tickers or date range.'

        # Store for the controls callback to pick up
        _self._stored_features = features_valid
        _self._stored_dates = pd.DatetimeIndex(features_valid['Date'].values)
        _self._stored_color_by = features_valid['Volatility'].fillna(0).astype('float32').values
        return f"Loaded {len(features_valid)} rows for: {', '.join(tickers)}"

    # Callback 2: rebuild manifold when controls change (or after ticker fetch triggers a re-render)
    @app.callback(
        Output('manifold-graph', 'figure'),
        Output('regime-timeline', 'figure'),
        Input('method-dropdown', 'value'),
        Input('color-dropdown', 'value'),
        Input('regime-slider', 'value'),
        Input('regime-toggle', 'value'),
        Input('ticker-status', 'children'),   # re-fires after a successful fetch
    )
    def update_graph(method, colorscale, n_regimes, show_regimes, _status):
        from config import Config
        from model import reduce_dimensionality
        from regime import detect_regimes

        features_src = _self._stored_features
        cur_dates = getattr(_self, '_stored_dates', dates)
        cur_color = getattr(_self, '_stored_color_by', color_by)

        if features_src is None:
            return build_manifold_figure(latent, dates, color_by, None, None, cfg), go.Figure()

        new_cfg = Config(
            method=method,
            colorscale=colorscale,
            n_regimes=n_regimes,
            animate=False,
            show_loadings=(method == 'pca'),
        )
        new_latent, new_loadings = reduce_dimensionality(features_src, new_cfg)
        new_regime_labels, _ = detect_regimes(new_latent, new_cfg) if show_regimes else (None, None)
        mfig = build_manifold_figure(new_latent, cur_dates, cur_color, new_regime_labels, new_loadings, new_cfg)
        rfig = build_regime_timeline(cur_dates, new_regime_labels, new_cfg) if new_regime_labels is not None else go.Figure()
        return mfig, rfig

    logger.info("Launching Dash dashboard at http://localhost:%d", cfg.port)
    app.run(debug=False, port=cfg.port)


# Module-level slots shared between launch_dashboard and its callbacks
_stored_features = None
_stored_dates = None
_stored_color_by = None
