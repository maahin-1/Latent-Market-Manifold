"""
Latent Market Manifold — entry point.

Usage examples
--------------
# Synthetic data, PCA, open Dash dashboard
python main.py

# Real tickers via yfinance, UMAP, 4 regimes
python main.py --real --tickers AAPL MSFT GOOGL TSLA --method umap --regimes 4

# Load from CSV, t-SNE, no animation, port 8080
python main.py --csv market_data.csv --method tsne --no-animate --port 8080

# Export static HTML instead of launching Dash
python main.py --export manifold.html
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s — %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='3D Latent Market Manifold')

    # Data source
    src = p.add_mutually_exclusive_group()
    src.add_argument('--real', action='store_true', help='Fetch real data via yfinance')
    src.add_argument('--csv', metavar='PATH', help='Load data from CSV file')

    p.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                   help='Tickers when --real is set (default: AAPL MSFT GOOGL AMZN TSLA)')
    p.add_argument('--start', default='2020-01-01', help='Start date (default: 2020-01-01)')
    p.add_argument('--end', default=None, help='End date (default: today)')

    # Synthetic data
    p.add_argument('--assets', type=int, default=5, help='Synthetic asset count (default: 5)')
    p.add_argument('--days', type=int, default=300, help='Synthetic day count (default: 300)')

    # Reduction
    p.add_argument('--method', choices=['pca', 'tsne', 'umap'], default='pca',
                   help='Dimensionality reduction method (default: pca)')

    # Regimes
    p.add_argument('--regimes', type=int, default=3, help='Number of GMM regimes (default: 3)')
    p.add_argument('--no-regimes', action='store_true', help='Disable regime detection')

    # Viz
    p.add_argument('--colorscale', default='plasma', help='Plotly colorscale name (default: plasma)')
    p.add_argument('--no-animate', action='store_true', help='Disable trajectory animation')
    p.add_argument('--no-loadings', action='store_true', help='Hide PCA loadings panel')
    p.add_argument('--port', type=int, default=8050, help='Dash server port (default: 8050)')
    p.add_argument('--export', metavar='FILE', help='Export static HTML instead of launching Dash')

    return p.parse_args()


def main():
    args = parse_args()

    from config import Config
    from data_fetcher import fetch_real_data, generate_synthetic_data
    from features import compute_features, FEATURE_COLS
    from model import reduce_dimensionality
    from regime import detect_regimes
    from visualizer import build_manifold_figure, launch_dashboard
    import visualizer as _vis

    cfg = Config(
        use_real_data=args.real,
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        csv_path=args.csv,
        n_assets=args.assets,
        n_days=args.days,
        method=args.method,
        n_regimes=args.regimes,
        colorscale=args.colorscale,
        animate=not args.no_animate,
        show_loadings=not args.no_loadings,
        port=args.port,
    )

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    if args.real:
        logger.info("Fetching real market data (%s)…", ', '.join(cfg.tickers))
        df = fetch_real_data(cfg.tickers, cfg.start_date, cfg.end_date)
    elif args.csv:
        from data_loader import load_market_data
        logger.info("Loading CSV: %s", args.csv)
        df = load_market_data(args.csv)
    else:
        logger.info("Generating synthetic data (%d assets, %d days)…", cfg.n_assets, cfg.n_days)
        df = generate_synthetic_data(cfg.n_assets, cfg.n_days, cfg.synthetic_seed)

    # ------------------------------------------------------------------ #
    # 2. Feature engineering
    # ------------------------------------------------------------------ #
    logger.info("Computing features…")
    features = compute_features(df, cfg)
    features_valid = features.dropna(subset=['Return', 'Volatility', 'Momentum'])
    if features_valid.empty:
        logger.error("No valid rows after feature computation. Check your data.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 3. Dimensionality reduction
    # ------------------------------------------------------------------ #
    logger.info("Running %s…", cfg.method.upper())
    latent, loadings = reduce_dimensionality(features_valid, cfg)

    # ------------------------------------------------------------------ #
    # 4. Regime detection
    # ------------------------------------------------------------------ #
    regime_labels = None
    if not args.no_regimes:
        logger.info("Detecting %d market regimes via GMM…", cfg.n_regimes)
        regime_labels, gmm = detect_regimes(latent, cfg)

    # ------------------------------------------------------------------ #
    # 5. Dates + colour array aligned to valid rows
    # ------------------------------------------------------------------ #
    dates = pd.DatetimeIndex(features_valid['Date'].values)
    color_by = features_valid['Volatility'].fillna(0).astype('float32').values

    # ------------------------------------------------------------------ #
    # 6. Visualise
    # ------------------------------------------------------------------ #
    if args.export:
        fig = build_manifold_figure(latent, dates, color_by, regime_labels, loadings, cfg)
        fig.write_html(args.export)
        logger.info("Exported to %s", args.export)
    else:
        # Wire up the stored features for the Dash callback
        _vis._stored_features = features_valid
        logger.info("Opening dashboard → http://localhost:%d", cfg.port)
        launch_dashboard(latent, dates, color_by, regime_labels, loadings, cfg)


if __name__ == '__main__':
    main()
