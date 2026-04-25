import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import Config

logger = logging.getLogger(__name__)

# Default colour per regime index (used by the Dash visualiser)
REGIME_COLORS = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']


def detect_regimes(
    latent: np.ndarray,
    cfg: Config = None,
) -> Tuple[np.ndarray, GaussianMixture]:
    """
    Fit a Gaussian Mixture Model on the 3D latent coordinates and return
    a hard cluster label (0 … n_regimes-1) for every time-step.

    Returns:
        labels : (N,) int array of regime indices
        gmm    : fitted GaussianMixture model (useful for BIC / log-likelihood)
    """
    if cfg is None:
        cfg = Config()

    n = cfg.n_regimes
    X = StandardScaler().fit_transform(latent.astype('float32'))

    gmm = GaussianMixture(
        n_components=n,
        covariance_type='full',
        max_iter=300,
        random_state=42,
    )
    gmm.fit(X)
    labels = gmm.predict(X)

    # Re-order labels so the regime with the lowest mean PC1 is index 0
    # (heuristic: negative PC1 ≈ bear / stress, positive ≈ bull / risk-on)
    pc1_means = np.array([X[labels == k, 0].mean() for k in range(n)])
    order = np.argsort(pc1_means)          # ascending PC1 → label 0 = most bearish
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels], dtype=np.int32)

    counts = np.bincount(labels, minlength=n)
    logger.info(
        "Regime detection complete: %d regimes, counts %s, BIC=%.1f",
        n, counts.tolist(), gmm.bic(X),
    )
    return labels, gmm


def regime_label_series(
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    cfg: Config = None,
) -> pd.Series:
    """Map integer regime labels to named strings using cfg.regime_labels."""
    if cfg is None:
        cfg = Config()

    named = [cfg.regime_labels[l] if l < len(cfg.regime_labels) else f'Regime{l}' for l in labels]
    return pd.Series(named, index=dates, name='Regime')
