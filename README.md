# IGMN: Incremental Gaussian Mixture Neural Model for Geophysical Inverse Problems

A full Python implementation reproducing the results of the paper:

> Mazzutti, Roisenberg, Rodrigues (2026). *Incremental Gaussian Mixture Neural Model for Geophysical inverse problems.* Computers & Geosciences.

This project shows how the Incremental Gaussian Mixture Network (IGMN) can be applied to complex, non-linear inverse problems in reservoir geophysics: one-dimensional petrophysical inversion and three-dimensional time-lapse uncertainty propagation using ordinary kriging.

## Repository Structure

```text
data/                      # Original and synthetic reservoir datasets
├── data4.mat              # Well-log data for 1D inversion (Experiment 1)
└── data_CAGEO.mat         # 2D reservoir grids for time-lapse (Experiment 2)
notebooks/                 # Interactive Jupyter notebooks
├── experiment1.ipynb      # 1D real-data application
└── experiment_2.ipynb     # 3D synthetic time-lapse application
igmn.py                    # Core IGMN algorithm (math, online learning, Schur complement)
pyproject.toml             # Project metadata and dependencies
uv.lock                    # Locked dependency versions for reproducibility
README.md                  # Project documentation
```

## Installation and Setup

The project uses [uv](https://github.com/astral-sh/uv), a fast Python package and project manager.

1. Install `uv` if you don't have it yet:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Clone the repository and sync dependencies:

   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   uv sync
   ```

   This creates an isolated `.venv` environment and installs all required libraries (numpy, scipy, matplotlib, scikit-learn, optuna, pykrige, ipykernel) at the exact versions used during development.

## Experiments

### Experiment 1: 1D Real-Data Application

Located in `notebooks/experiment1.ipynb`.

- Builds a rock-physics model (linear regression) to generate a Monte Carlo synthetic training set.
- Trains the IGMN sequentially to discover the latent Gaussian components (facies).
- Runs petrophysical inversion (porosity, velocity, water saturation) on real well-log data.
- Reconstructs joint-distribution ellipsoids and uncertainty heatmaps.

### Experiment 2: 3D Synthetic Time-Lapse Application

Located in `notebooks/experiment_2.ipynb`.

- Uses a complex, non-linear rock-physics model (Mindlin-Hertz contact theory plus Gassmann fluid substitution).
- Tunes IGMN hyperparameters with Optuna (Bayesian optimization), penalizing excessive cluster formation.
- Applies ordinary kriging (via PyKrige) to condition the network's predictions on hard well data.
- Produces point-wise standard-deviation maps (uncertainty quantification) to help manage reservoir drilling risk.

## License

This project is intended for educational and research purposes, built to validate and extend the methods proposed in the original *Computers & Geosciences* publication.
