<div align="center">
  <img src="https://raw.githubusercontent.com/Sajad-Hussaini/pinagmm/main/src/pinagmm/gui/assets/logo.png" alt="PINAGMM Logo" width="300"/>
  <br>
  <h1>PINAGMM: Physics-Informed Neural Additive Ground Motion Model</h1>

  <p>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python"></a>
    <a href="https://opensource.org/licenses/GPL-3.0"><img src="https://img.shields.io/badge/license-GPLv3-blue.svg" alt="License"></a>
    <a href="https://pinagmm.onrender.com"><img src="https://img.shields.io/badge/Live-Web_App-success?style=flat&logo=render" alt="Live Web App"></a>
    <a href="https://doi.org/10.5281/zenodo.20746843"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20746843-blue.svg" alt="DOI"></a>
  </p>
</div>

**PINAGMM** is a unified generative framework designed for performance-based earthquake engineering. It bridges statistical machine-learning prediction of discrete intensity measures (IMs) with the synthesis of hazard-compatible, three-component stochastic ground motion time-series.

Powered by a Physics-Informed Neural Additive Model (NAM) coupled with [Multivariate Mixed-Effects Regression (MMER)](https://github.com/Sajad-Hussaini/mmer) and a [Stochastic Simulation Engine (SGSIM)](https://github.com/Sajad-Hussaini/sgsim), PINAGMM enables direct conditional simulation of physically coherent ground motions for prescribed earthquake hazard scenarios, bypassing the need for artificial time-domain spectral matching.


## Table of Contents
- [Installation](#installation)
- [Desktop Application (GUI)](#desktop-application-gui)
- [Web Demo App](#web-demo-app)
- [Python API & Custom Workflows](#python-api--custom-workflows)
- [Contact & Support](#contact--support)
- [License](#license)
- [References](#references)

## Installation

Install **PINAGMM** directly from GitHub with a single command in your terminal:

```bash
pip install git+https://github.com/Sajad-Hussaini/pinagmm.git
```

<details>
<summary><b>Updating to the Latest Version</b></summary>
To update to the latest version at any time, run:

```bash
pip install --upgrade git+https://github.com/Sajad-Hussaini/pinagmm.git
```
</details>

## Desktop Application (GUI)

The simplest way to use **PINAGMM** on your local machine is through its interactive Graphical User Interface (GUI). 

Once installed, launch the application by running a single command in your terminal:

```bash
pinagmm
```

This automatically opens the PINAGMM interface in your web browser.

<details>
<summary><b>Key Capabilities & Deliverables</b></summary>

- **Earthquake Scenario Inputs:**
- **Hazard-Targeting (Conditional Simulation):**
- **GMM Prediction and Response Spectra Visualization:**
- **Stochastic Time-Series Simulation:**
- **One-Click Downloading Data:**

</details>

## Web Demo App

For users who want to explore the interface without installing the package, a hosted cloud preview is available:

🌐 **[Launch the Live Web Demo](https://pinagmm.onrender.com)**

> ⚠️ **Demo Note:** The web demo runs on a lightweight free cloud tier with limited resources. It is designed for quick interface exploration. For full-scale ground motion generation and batch stochastic simulations, please run the desktop application locally via `pinagmm`.

## Python API & Custom Workflows

If you need to automate batch calculations, integrate PINAGMM into existing engineering scripts, or build custom simulation workflows, PINAGMM provides a simple Python API.

A fully functional reference script is provided in the repository example folder:
📄 **[`example/run_me.py`](example/run_me.py)**

<details>
<summary><b>Example Demonstration</b></summary>

1. **Median Predictions:**
2. **Unconditional Stochastic Simulation:**
3. **Conditional Hazard Simulation:**
4. **Data Export & Plotting:**

</details>

You can run the script directly from your terminal and see results on your desktop:
```bash
python example/run_me.py
```

## Contact & Support
For any questions, assistance, or suggestions, please feel free to contact:

**S. M. Sajad Hussaini**  
📧 [hussaini.smsajad@gmail.com](mailto:hussaini.smsajad@gmail.com)

> Please include "PINAGMM" in the subject line for a quicker response.

## License

**PINAGMM** is distributed under the [**GNU General Public License v3 (GPLv3)**](https://opensource.org/licenses/GPL-3.0). See the [LICENSE](LICENSE) file for the full text.

> You are free to use, modify, and distribute this software for academic and research purposes. Any commercial use or distribution of modified versions requires the entire project to be open-sourced under the same GPLv3 license. For proprietary commercial exemptions, please refer to the Contact section.

## References

If you use PINAGMM in your research, please cite the following references:

**[1] Primary Reference (Methodology)**  
*A Physics-Informed Neural Additive Ground Motion Model for Hazard-Compatible Three-Component Stochastic Simulation*  
*DOI: to be added later*  
(Journal of Earthquake Engineering & Structural Dynamics)

**[2] PINAGMM Software Package**  
*Physics-Informed Neural Additive Ground Motion Model*  
*DOI: [https://doi.org/10.5281/zenodo.20746843](https://doi.org/10.5281/zenodo.20746843)*
