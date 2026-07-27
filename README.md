<div align="center">
  <img src="https://raw.githubusercontent.com/Sajad-Hussaini/pinagmm/main/pinagmm/gui/assets/logo.png" alt="PINAGMM Logo" width="300"/>
  <br>
  <h1>PINAGMM: Physics-Informed Neural Additive Ground Motion Model</h1>

  <p>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
    <a href="https://pinagmm.onrender.com"><img src="https://img.shields.io/badge/Live-Web_App-success?style=flat&logo=render" alt="Live Web App"></a>
    <a href="https://doi.org/10.5281/zenodo.20746843"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20746843.svg" alt="DOI"></a>
  </p>
</div>

**PINAGMM** is a unified multivariate generative framework that bridges the prediction of discrete intensity measures (IMs) with the synthesis of hazard-compatible, three-component stochastic ground motion time-series. 

Developed for performance-based earthquake engineering, this framework uses a Physics-Informed Neural Additive Model (NAM) coupled with Multivariate Mixed-Effects Regression (MMER). It enables the direct conditional simulation of physically coherent synthetic ground motions for prescribed hazard scenarios, completely bypassing the need for artificial time-domain spectral matching.

## Table of Contents
- [Installation](#installation)
- [Overview: Predict vs. Simulate](#overview-predict-vs-simulate)
- [Graphical User Interface (GUI)](#graphical-user-interface-gui)
- [Python API (For Power Users)](#python-api-for-power-users)
  - [1. Median Predictions](#1-median-predictions)
  - [2. Stochastic Simulation (Unconditional & Conditional)](#2-stochastic-simulation)
- [Contact & Support](#contact--support)
- [References](#references)

## Installation

We recommend installing the CPU-only version of PyTorch first to save disk space, followed by this package.

```bash
# 1. Clone the repository
git clone https://github.com/Sajad-Hussaini/PINAGMM.git
cd PINAGMM

# 2. Install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install PINAGMM and dependencies (sgsim, mmer, etc.)
pip install .
```


## Overview: Predict vs. Simulate

PINAGMM separates the statistical machine-learning predictions from the end-to-end stochastic ground motion simulation:

- **`predict()` (The Machine Learning layer):** Purely runs the neural network to predict the 74 output variables (Intensity Measures and stochastic simulation parameters) for a given earthquake scenario. No waveforms are generated here.
- **`simulate()` (The End-to-End Generative layer):** A comprehensive wrapper that takes the ML predictions, mathematically applies covariance sampling and conditional target matching (if requested), and automatically feeds the parameters into the stochastic engine (`sgsim`) to generate actual 3-component ground motions.


## Graphical User Interface (GUI)

For users who prefer a visual, no-code environment, we host a live version of the PINAGMM interactive dashboard. You can generate and download synthetic ground motions directly in your browser without installing any code!

🌐 **[Launch the Live Web App](https://pinagmm.onrender.com)**

> ⚠️ **Live Demo Limitations:** The web app above is hosted on a free cloud server with very limited memory. It is intended strictly as a **lightweight demo** to explore the interface and test small predictions (a few simulations). Attempting to run large batches of stochastic simulations online will not work and face reloading server. For practical, full-scale ground motion generation, please install and run the GUI locally!

The GUI allows you to:
- Seamlessly input scenario parameters (Mw, Ztor, Rrup, Vs30, Fm).
- Toggle conditional hazard targets (e.g., forcing Sa at 1.0s to equal 0.9g).
- Instantly visualize predicted median response spectra.
- Plot and explore 3-component time-series realizations and their Fourier Amplitude Spectra using interactive Plotly charts.
- Export all generated data and spectra directly to neatly formatted CSV files with a single click.

### Run the GUI Locally

If you prefer to run the GUI securely on your own machine, first follow the [Installation](#installation) instructions. Then, ensure your virtual environment is active and run the following command in your terminal:
```bash
pinagmm
```
This will automatically launch the dashboard (usually at `http://localhost:8080`) in your local web browser.

## Python API (For Power Users)

### 1. Median Predictions (`predict`)

The `PINAGMM` class generates the median physical scaling for both Intensity Measures (PGA, PGV, Sa) and the underlying parameters governing the stochastic simulation (Energy, Duration, Frequencies).

```python
from pinagmm import PINAGMM

# Initialize the model
model = PINAGMM()

# Predict median parameters for a specific earthquake scenario
predictions = model.predict(Mw=6.5, Ztor=1.0, Rrup=25.0, Vs30=560.0, Fm="0")

print(predictions[["PGA", "Sa(T=1)", "PGV"]])
```

> 💡 **Tip:** We highly recommend checking out the `example/run_me.py` file in this repository. It is a fully functional script that demonstrates how to generate predictions, simulate time-series, save them to CSV files, and use `matplotlib` to plot and save the results!

### 2. Stochastic Simulation

The true power of this framework lies in the generative engine, which leverages the learned inter-parameter covariance matrix to generate realistic 3-component ground motion seamlessly.

The model provides two completely independent layers of statistical generation natively through the `simulate()` method:
1. `n_samples`: The number of macroscopic parameter sets to sample from the underlying neural GMM's covariance matrix.
2. `n_simulations`: The number of discrete time-series realizations to generate for *each* parameter set using the stochastic simulation engine.

#### 1. Unconditional Simulation (Median vs Sampled)
Generate physically consistent ground motions based purely on the macroscopic earthquake scenario. 

```python
from pinagmm import PINAGMM

# Initialize the model
gmm = PINAGMM()

# Example A: 5 Stochastic Realizations using the exact Median GMM prediction
ts_m_med, ts_i_med, ts_v_med = gmm.simulate(
    Mw=6.5,
    Ztor=1.0,
    Rrup=25.0,
    Vs30=560.0,
    Fm="0",
    n_samples=0,  # 0 = Use the median GMM prediction
    n_simulations=5,  # 5 time-series realizations
)

# Example B: 10 Parameter Samples from the GMM, 1 Stochastic Realization each
ts_m_list, ts_i_list, ts_v_list = gmm.simulate(
    Mw=6.5,
    Ztor=1.0,
    Rrup=25.0,
    Vs30=560.0,
    Fm="0",
    n_samples=10,  # Sample 10 unique parameter conditions from the Neural NAM
    n_simulations=1,  # 1 time-series realization per parameter set
)
```

#### 2. Conditional Simulation (Hazard-Targeting)

Mathematically condition the generative model to match a specific hazard target (e.g., forcing Spectral Acceleration at 1.0s to a specific physical value like 0.9g). The framework automatically adjusts all other correlated IMs and physical simulation parameters across all three principal axes to physically justify the target.  

```python
# Condition the target IM (Major SA at 1s) to 0.9 g
target_conditions = {"M_Sa_1": 0.9}

ts_m_cond, ts_i_cond, ts_v_cond = gmm.simulate(
    Mw=6.5,
    Ztor=1.0,
    Rrup=25.0,
    Vs30=560.0,
    Fm="0",
    conditions=target_conditions,
    n_samples=10,  # 10 conditioned samples
    n_simulations=1,
)
```

## Contact & Support
For any questions, assistance, or suggestions, please feel free to contact:

**S. M. Sajad Hussaini**  
📧 [hussaini.smsajad@gmail.com](mailto:hussaini.smsajad@gmail.com)

> Please include "PINAGMM" in the subject line for a quicker response.

> If you find this package useful, contributions to help maintain and improve it, are always appreciated. [![PayPal](https://img.shields.io/badge/PayPal-Donate-blue.svg)](https://www.paypal.com/paypalme/sajadhussaini)

## References

Please cite the following references for any formal study:  

**[1] Primary Reference**  
*A Physics-Informed Neural Additive Ground Motion Model for Hazard-Compatible Three-Component Stochastic Simulation*  
*DOI: to be added later*  
(Journal of Earthquake Engineering & Structural Dynamics)

**[2] PINAGMM Package**  
*Physics-Informed Neural Additive Ground Motion Model*  
*DOI: [https://doi.org/10.5281/zenodo.18068839](https://doi.org/10.5281/zenodo.20746843)*  

