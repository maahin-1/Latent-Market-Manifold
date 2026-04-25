import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_real_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Download OHLCV data via yfinance and return long-format DataFrame."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    logger.info("Fetching real market data for %s from %s to %s", tickers, start_date, end_date or "today")
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError("yfinance returned no data. Check tickers and date range.")

    # Extract Close prices; handle single vs multi-ticker response
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw[['Close']]
        close.columns = tickers

    close = close.dropna(how='all')
    long = close.reset_index().melt(id_vars='Date', var_name='Asset', value_name='Price')
    long = long.dropna(subset=['Price']).sort_values(['Asset', 'Date']).reset_index(drop=True)
    long['Price'] = long['Price'].astype('float32')
    logger.info("Fetched %d rows for %d assets", len(long), long['Asset'].nunique())
    return long


def generate_synthetic_data(
    n_assets: int = 5,
    n_days: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic multi-asset price data with regime shifts."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2020-01-01', periods=n_days)
    records = []

    for i in range(n_assets):
        price = 100.0
        prices = []
        # Three regime blocks: bear, recovery, bull
        regimes = [
            (n_days // 3, -0.0003, 0.012),
            (n_days // 3, 0.0005, 0.008),
            (n_days - 2 * (n_days // 3), 0.001, 0.015),
        ]
        for length, drift, vol in regimes:
            shocks = rng.normal(drift + rng.normal(0, 0.0002), vol, length)
            for s in shocks:
                price = max(price * (1 + s), 0.01)
                prices.append(price)

        asset_name = f'Asset{i + 1}'
        for d, p in zip(dates, prices):
            records.append({'Date': d, 'Asset': asset_name, 'Price': float(p)})

    df = pd.DataFrame(records)
    df['Price'] = df['Price'].astype('float32')
    return df
