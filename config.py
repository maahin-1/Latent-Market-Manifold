from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # Data source
    use_real_data: bool = False
    tickers: List[str] = field(default_factory=lambda: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
    start_date: str = '2020-01-01'
    end_date: Optional[str] = None  # None = today
    csv_path: Optional[str] = None

    # Synthetic data fallback
    n_assets: int = 5
    n_days: int = 300
    synthetic_seed: int = 42

    # Feature engineering
    volatility_window: int = 20
    momentum_window: int = 10
    correlation_window: int = 20

    # Dimensionality reduction
    method: str = 'pca'          # 'pca', 'tsne', 'umap'
    n_components: int = 3
    tsne_perplexity: int = 30
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1

    # Regime detection
    n_regimes: int = 3           # number of GMM clusters
    regime_labels: List[str] = field(default_factory=lambda: ['Bear', 'Neutral', 'Bull'])

    # Visualization
    colorscale: str = 'plasma'
    animate: bool = True         # show animated trajectory
    show_loadings: bool = True   # show PCA loadings panel (only for PCA)
    port: int = 8050             # Dash server port

    # Caching
    cache_dir: str = '.cache'
