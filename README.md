
# 3D Latent Market Manifold

##  What is this project?
This project visualizes the hidden geometry of financial markets by embedding time series data (like prices, returns, volatility, etc.) into a 3D latent space using dimensionality reduction (PCA, t-SNE, UMAP). Instead of plotting prices directly, it reveals the "shape" of the market as a 3D manifold, helping you see market regimes, transitions, and structure.

##  How to Fork & Run
1. **Fork this repo** on GitHub (top right corner).
2. **Clone your fork** to your computer:
   ```bash
   git clone https://github.com/YOUR_USERNAME/3d-latent-market-manifold.git
   cd 3d-latent-market-manifold
   ```
3. **Install dependencies** (Python 3.8+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the main script:**
   ```bash
   python main.py
   ```
   This will generate synthetic market data and show a 3D plot in a Python window.

##  Project Structure
- `data_loader.py`: Load historical price data
- `features.py`: Feature engineering (returns, volatility, momentum, correlations)
- `model.py`: Dimensionality reduction (PCA, t-SNE, UMAP)
- `visualizer.py`: 3D visualization (matplotlib)
- `main.py`: Pipeline entry point

##  Output
- Animated or static 3D latent space trajectory of the market
- Color by volatility, returns, or regime
- Interactive matplotlib window (not browser)

##  Advanced Features (Planned)
- Market regime clustering and labeling (bull, bear, volatile)
- Animated camera and glowing trajectory
- Gradient color mapping

---
**Made with ❤️ by tubakhxn**
