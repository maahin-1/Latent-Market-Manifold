import logging

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

FEATURE_COLS = ['Return', 'Volatility', 'Momentum', 'MeanCorrelation']


def compute_features(df: pd.DataFrame, cfg: Config = None) -> pd.DataFrame:
    """
    Compute returns, rolling volatility, momentum, and mean cross-asset correlation.

    All rolling windows are configured via Config. Output uses float32 to halve memory.
    A single pivot-based approach replaces the old fragile per-asset groupby/stack pattern.
    """
    if cfg is None:
        cfg = Config()

    vol_win = cfg.volatility_window
    mom_win = cfg.momentum_window
    cor_win = cfg.correlation_window

    df = df.sort_values(['Asset', 'Date']).copy()

    # Per-asset return, volatility, momentum — vectorized via pivot then stack
    pivot_price = df.pivot(index='Date', columns='Asset', values='Price').astype('float32')
    pivot_ret = pivot_price.pct_change().astype('float32')
    pivot_vol = pivot_ret.rolling(vol_win).std().astype('float32')
    pivot_mom = pivot_price.pct_change(periods=mom_win).astype('float32')

    # Mean pairwise correlation per date (scalar per date, same for all assets on that date)
    rolling_corr = pivot_ret.rolling(cor_win).corr()  # MultiIndex: (Date, Asset) x Asset
    # Average upper-triangle correlation across all pairs per date
    dates = pivot_price.index
    mean_corr = pd.Series(index=dates, dtype='float32', name='MeanCorrelation')
    n = pivot_ret.shape[1]
    if n > 1:
        for date in dates:
            try:
                mat = rolling_corr.loc[date].values
                # upper triangle excluding diagonal
                idx = np.triu_indices(n, k=1)
                mean_corr[date] = float(np.nanmean(mat[idx]))
            except KeyError:
                mean_corr[date] = np.nan
    else:
        mean_corr[:] = np.nan

    # Stack wide frames back to long format
    ret_long = pivot_ret.stack().rename('Return')
    vol_long = pivot_vol.stack().rename('Volatility')
    mom_long = pivot_mom.stack().rename('Momentum')

    features = (
        ret_long.to_frame()
        .join(vol_long, how='left')
        .join(mom_long, how='left')
        .reset_index()
        .rename(columns={'level_1': 'Asset'} if 'level_1' in ret_long.index.names else {})
    )
    # Normalise column name from MultiIndex stack
    if 'Asset' not in features.columns and pivot_ret.columns.name:
        features = features.rename(columns={pivot_ret.columns.name: 'Asset'})

    features = features.merge(mean_corr.rename('MeanCorrelation'), left_on='Date', right_index=True, how='left')
    features[FEATURE_COLS] = features[FEATURE_COLS].astype('float32')

    logger.info(
        "Features computed: %d rows, %d assets, windows vol=%d mom=%d cor=%d",
        len(features), features['Asset'].nunique(), vol_win, mom_win, cor_win,
    )
    return features
