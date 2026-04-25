import hashlib
import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from config import Config
from features import FEATURE_COLS

logger = logging.getLogger(__name__)


def _cache_key(X: np.ndarray, method: str, cfg: Config) -> str:
    h = hashlib.md5(X.tobytes()).hexdigest()[:12]
    return f"{method}_{h}_n{cfg.n_components}"


def _load_cache(key: str, cache_dir: str):
    path = os.path.join(cache_dir, f"{key}.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def _save_cache(key: str, cache_dir: str, obj):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def reduce_dimensionality(
    features: pd.DataFrame,
    cfg: Config = None,
) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
    """
    Reduce feature space to 3D latent coordinates.

    Returns:
        latent   : (N, 3) float32 array of latent coordinates
        loadings : DataFrame of PCA component loadings (None for t-SNE/UMAP)
    """
    if cfg is None:
        cfg = Config()

    method = cfg.method
    n_components = cfg.n_components

    # Select and clean numeric feature columns
    cols = [c for c in FEATURE_COLS if c in features.columns]
    X = features[cols].dropna().astype('float32').values

    n_samples, n_features = X.shape
    logger.info("Reducing %d samples × %d features → %dD via %s", n_samples, n_features, n_components, method)

    if n_samples < 3 or n_features < 2:
        logger.warning("Insufficient data for embedding (samples=%d, features=%d).", n_samples, n_features)
        n_components = max(1, min(n_samples, n_features, 2))

    # Standardise
    X_scaled = StandardScaler().fit_transform(X).astype('float32')

    # Cache lookup
    cache_key = _cache_key(X_scaled, method, cfg)
    cached = _load_cache(cache_key, cfg.cache_dir)
    if cached is not None:
        logger.info("Loaded embedding from cache (%s).", cache_key)
        return cached

    loadings: Optional[pd.DataFrame] = None

    if method == 'pca':
        pca = PCA(n_components=n_components)
        latent = pca.fit_transform(X_scaled)
        explained = pca.explained_variance_ratio_
        logger.info("PCA explained variance: %s (total %.1f%%)", explained.round(3), explained.sum() * 100)
        loadings = pd.DataFrame(
            pca.components_.T,
            index=cols,
            columns=[f'PC{i+1}' for i in range(n_components)],
        )

    elif method == 'tsne':
        perp = min(cfg.tsne_perplexity, n_samples - 1)
        model = TSNE(n_components=min(n_components, 3), perplexity=perp, random_state=42, n_iter=1000)
        latent = model.fit_transform(X_scaled)

    elif method == 'umap':
        try:
            import umap as umap_lib
        except ImportError:
            raise ImportError("umap-learn not installed. Run: pip install umap-learn")
        model = umap_lib.UMAP(
            n_components=n_components,
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist,
            random_state=42,
        )
        latent = model.fit_transform(X_scaled)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'pca', 'tsne', or 'umap'.")

    # Pad to 3 columns if needed
    if latent.ndim == 1 or latent.shape[1] < 3:
        cols_needed = 3 - (latent.shape[1] if latent.ndim > 1 else 1)
        latent = np.hstack([latent.reshape(-1, latent.shape[-1] if latent.ndim > 1 else 1),
                            np.zeros((latent.shape[0], cols_needed), dtype='float32')])

    latent = latent.astype('float32')
    result = (latent, loadings)
    _save_cache(cache_key, cfg.cache_dir, result)
    return result
