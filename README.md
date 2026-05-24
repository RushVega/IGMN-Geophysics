# IGMN: Incremental Gaussian Mixture Neural Model for Geophysical Inverse Problems

A full Python implementation reproducing the results of the paper:

> Mazzutti, Roisenberg, Rodrigues (2026). *Incremental Gaussian Mixture Neural Model for Geophysical inverse problems.* Computers & Geosciences.

This project shows how the Incremental Gaussian Mixture Network (IGMN) can be applied to complex, non-linear inverse problems in reservoir geophysics: one-dimensional petrophysical inversion and three-dimensional time-lapse uncertainty propagation using ordinary kriging.

## Repository Structure

```text
data/                      # Original and synthetic reservoir datasets
├── data4.mat              # Well-log data for 1D inversion (Experiment 1)
└── data_CAGEO.mat         # UNISIM-I reservoir grid data (Base 2013 & Monitor 2024) for time-lapse inversion (Experiment 2)
notebooks/                 # Interactive Jupyter notebooks
├── experiment1.ipynb      # 1D real-data application
└── experiment_2.ipynb     # 3D synthetic time-lapse application
igmn.py                    # Core IGMN algorithm (math, online learning, Schur complement)
pyproject.toml             # Project metadata and dependencies
uv.lock                    # Locked dependency versions for reproducibility
README.md                  # Project documentation
```

## Installation & Setup

This project uses [uv](https://github.com/astral-sh/uv) as an extremely fast Python package and project manager. 

1. **Install `uv`** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository and sync dependencies**:
   ```bash
   git clone https://github.com/RushVega/IGMN-Geophysics.git
   cd IGMN-Geophysics
   uv sync
   ```
   *This command will automatically create an isolated `.venv` environment and install all required libraries (`numpy`, `scipy`, `matplotlib`, `scikit-learn`, `optuna`, `pykrige`, `ipykernel`) with the exact versions used during development.*

##  Experiments

### Experiment 1: 1D Real-Data Application
Located in `notebooks/experiment_1.ipynb`.
- Generates a synthetic training dataset via Monte Carlo sampling using a linearized rock-physics model.
- Trains the IGMN sequentially to autonomously discover latent Gaussian components associated with litho-fluid facies without prior classification.
- Performs petrophysical inversion to estimate porosity ($\phi$), clay volume ($V_c$), and water saturation ($S_w$) from elastic well-log data.
- Visualizes the learned joint-distribution using covariance ellipsoids and propagates uncertainty analytically to generate probability heatmaps.

### Experiment 2: 3D Synthetic Time-Lapse Application
Located in `notebooks/experiment_2.ipynb`.
- Employs a non-linear rock-physics model (Soft Sand/Mindlin-Hertz and Gassmann fluid substitution) to build a robust 3D time-lapse training set.
- Optimizes IGMN hyperparameters using Bayesian optimization (`Optuna`), penalizing network over-fragmentation to maintain geological consistency.
- Executes a full 2D spatial reservoir inversion and applies Ordinary Kriging (`PyKrige`) to condition the predictions to hard well data.
- Derives point-wise standard deviation maps to quantify spatial uncertainty and mitigate reservoir exploration risks.

## License
This project is for educational and research purposes, designed to validate and expand upon the methodologies proposed in the original Computers & Geosciences publication.
```
