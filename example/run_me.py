"""
PINAGMM — Quick-start example
==============================
Run this script from the repository root:

    python example/run_me.py

It demonstrates the three main use-cases:

  1. Median prediction of intensity measures -> saved to CSV
  2. Stochastic simulation using the median GMM parameters -> saved to CSV
  3. Conditional simulation targeting a specific hazard level -> saved to CSV

No arguments needed.  All output files are written to Desktop/PINAGMM Results/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pinagmm import PINAGMM, save_timeseries, save_spectra

OUT_DIR = Path.home() / "Desktop" / "PINAGMM Results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- 1. Initialise the model (loads bundled .joblib files once) -------------
print("Loading PINAGMM ...")
gmm = PINAGMM()
print("  Done.\n")


# ============================================================================
# USE-CASE 1: Median prediction
# ============================================================================
print("-" * 60)
print("1. Median prediction")
print("-" * 60)

prediction = gmm.predict(Mw=6.5, Ztor=3.0, Rrup=15.0, Vs30=800.0, Fm="0")

# The result is a DataFrame: input columns + all yvars columns.
# Column naming: {Component}_{IM}, where M=Major, I=Intermediate, V=Vertical.
print(prediction[["M_Sa_0", "M_Sa_0.2", "M_Sa_1", "M_PGV"]].to_string(index=False))

prediction.to_csv(OUT_DIR / "prediction_median.csv", index=False)
print(f"\n  [OK] Saved -> {OUT_DIR / 'prediction_median.csv'}\n")

# Plot: predicted response spectrum for all three components
periods = [
    0.03,
    0.05,
    0.08,
    0.1,
    0.16,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
]
comp_specs = {
    "Major": [prediction[f"M_Sa_{T:g}"].iloc[0] for T in periods],
    "Intermediate": [prediction[f"I_Sa_{T:g}"].iloc[0] for T in periods],
    "Vertical": [prediction[f"V_Sa_{T:g}"].iloc[0] for T in periods],
}

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#4da6ff", "#ffb84d", "#6ddb82"]
for (comp, sa), color in zip(comp_specs.items(), colors):
    ax.loglog(periods, sa, marker="o", ms=3, lw=1.8, label=comp, color=color)
ax.set_xlabel("Period (s)")
ax.set_ylabel("Sa (g)")
ax.set_title("Predicted Median Response Spectrum  (Mw 6.5 | Rrup 15 km | Vs30 800 m/s)")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "spectrum_median.png", dpi=200)
print(f"  [OK] Saved -> {OUT_DIR / 'spectrum_median.png'}")


# ============================================================================
# USE-CASE 2: Stochastic simulation (median GMM parameters)
# ============================================================================
print("\n" + "-" * 60)
print("2. Stochastic simulation (median parameters, 5 realisations)")
print("-" * 60)

# n_samples=0  -> use the deterministic median GMM prediction
# n_simulations=5  -> generate 5 independent time series from the stochastic engine
ts_m, ts_i, ts_v = gmm.simulate(
    Mw=6.5,
    Ztor=3.0,
    Rrup=15.0,
    Vs30=800.0,
    Fm="0",
    dt=0.005,
    n_samples=0,  # median GMM parameters
    n_simulations=5,  # 5 stochastic realisations
)

# ts_m is now a list of GroundMotion. First element has .ac shape (5, npts)
print(f"  Major component: ac shape = {ts_m[0].ac.shape}")

# -- Save time series --------------------------------------------------------
# Columns: t (s), ac_0, ac_1, ..., ac_4  (all in g)
save_timeseries(ts_m, OUT_DIR / "ts_major.csv")
save_timeseries(ts_i, OUT_DIR / "ts_intermediate.csv")
save_timeseries(ts_v, OUT_DIR / "ts_vertical.csv")
print(f"  [OK] Time series saved to {OUT_DIR}/ts_{{major,intermediate,vertical}}.csv")

# -- Save response spectra ----------------------------------------------------
T = np.logspace(-2, np.log10(4), 80)
save_spectra(ts_m, OUT_DIR / "spectra_major.csv", T)
save_spectra(ts_i, OUT_DIR / "spectra_intermediate.csv", T)
save_spectra(ts_v, OUT_DIR / "spectra_vertical.csv", T)
print(f"  [OK] Response spectra saved to {OUT_DIR}/spectra_{{...}}.csv")

# -- Plot ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for ax, (comp, ts, color) in zip(
    axes,
    [
        ("Major", ts_m, "#4da6ff"),
        ("Intermediate", ts_i, "#ffb84d"),
        ("Vertical", ts_v, "#6ddb82"),
    ],
):
    for row in ts[0].ac:
        ax.plot(ts[0].t, row, lw=0.5, alpha=0.6, color=color)
    ax.set_title(f"{comp} Component")
    ax.set_xlabel("Time (s)")
axes[0].set_ylabel("Acceleration (g)")
fig.tight_layout()
fig.savefig(OUT_DIR / "timeseries_simulation.png", dpi=200)
print(f"  [OK] Saved -> {OUT_DIR / 'timeseries_simulation.png'}")


# ============================================================================
# USE-CASE 3: Conditional simulation (hazard targeting)
# ============================================================================
print("\n" + "-" * 60)
print("3. Conditional simulation (targeting Sa(1s)=0.9 g in Major component)")
print("-" * 60)

# The GMM adjusts all correlated IMs and simulation parameters so they are
# physically consistent with the prescribed target value.
ts_m_list, ts_i_list, ts_v_list = gmm.simulate(
    Mw=6.5,
    Ztor=3.0,
    Rrup=15.0,
    Vs30=800.0,
    Fm="0",
    conditions={"M_Sa_1": 0.9},  # force Major-component Sa(1s) = 0.9 g
    n_samples=5,  # 5 independent GMM samples
    n_simulations=1,  # 1 stochastic realisation each
    dt=0.005,
)

# ts_m_list is a list of 5 GroundMotion objects
print(f"  Number of conditional realisations: {len(ts_m_list)}")

# Save all 5 major-component time series (zero-padded to the longest)
save_timeseries(ts_m_list, OUT_DIR / "ts_conditional_major.csv")
print(f"  [OK] Saved -> {OUT_DIR / 'ts_conditional_major.csv'}")

# Compute and plot response spectra of conditioned realisations
fig, ax = plt.subplots(figsize=(7, 4))
for i, sim in enumerate(ts_m_list):
    _, _, sa = sim.response_spectra(T)
    sa = np.atleast_2d(sa)[0]  # single realisation
    ax.loglog(
        T,
        sa,
        lw=0.8,
        alpha=0.6,
        color="#4da6ff",
        label="Conditional realisations" if i == 0 else None,
    )

ax.axvline(1.0, color="red", ls="--", lw=0.8, label="Target period (1 s)")
ax.plot([1.0], [0.9], "r*", ms=12, label="Target Sa = 0.9 g")
ax.set_xlabel("Period (s)")
ax.set_ylabel("Sa (g)")
ax.set_title("Conditional Response Spectra  (Target: M_Sa_1 = 0.9 g)")
ax.legend(fontsize=8)
ax.grid(True, which="both", ls="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "spectra_conditional.png", dpi=200)
print(f"  [OK] Saved -> {OUT_DIR / 'spectra_conditional.png'}")

print("\n" + "=" * 60)
print("All done!  Results are in the Desktop/PINAGMM Results/ directory.")
print("=" * 60)
