"""
PINAGMM — Advanced Plots Example
================================
Run this script from the repository root:

    python example/example_advanced_plots.py

It demonstrates advanced plotting and analysis using the PINAGMM model:
  1. Attenuation Curve plotting (PGA vs Distance)
  2. Multi-metric Time Series plotting (Acceleration, Velocity, Displacement)
  3. Conditional Hazard Targeting (Conditional Mean Spectra)

All output files are saved to Desktop/PINAGMM Results/ to keep your workspace clean.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pinagmm import PINAGMM


def main():
    print("Loading PINAGMM...")
    gmm = PINAGMM()

    OUT_DIR = Path.home() / "Desktop" / "PINAGMM Results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nAll outputs will be saved to: {OUT_DIR}")

    # =========================================================================
    # 1. Attenuation Curve Plotting (Vectorized Prediction)
    # =========================================================================
    print("\n--- 1. Generating Attenuation Curve ---")
    distances = np.linspace(1.0, 100.0, 50)

    attenuation_df = gmm.predict(
        Mw=7.0, Ztor=3.0, Rrup=distances, Vs30=800.0, Fm="0", n_sample=10
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        attenuation_df["ClstD (km)"],
        attenuation_df["M_Sa_0"],
        marker=".",
        linestyle="none",
        color="b",
        label="Mw=7.0, Normal",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Rupture Distance ($R_{rup}$) [km]")
    ax.set_ylabel("Peak Ground Acceleration (PGA) [g]")
    ax.set_title("PINAGMM Attenuation Curve")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    out_path = OUT_DIR / "attenuation_curve.png"
    fig.savefig(out_path, dpi=200)
    print(f"  [OK] Saved -> {out_path}")
    plt.close(fig)

    attenuation_df.to_csv(OUT_DIR / "attenuation_results.csv", index=False)
    print(f"  [OK] Saved -> {OUT_DIR / 'attenuation_results.csv'}")

    # =========================================================================
    # 2. Multi-metric Time Series (Acceleration, Velocity, Displacement)
    # =========================================================================
    print("\n--- 2. Visualizing 3-Component Time Series ---")

    # Simulate ground motions using the median parameter prediction (n_samples=0)
    # Generate 5 stochastic realizations (n_simulations=5) to see aleatory variability
    ts_m_med, ts_i_med, ts_v_med = gmm.simulate(
        Mw=6.5,
        Ztor=1.0,
        Rrup=25.0,
        Vs30=560.0,
        Fm="0",
        dt=0.005,
        n_samples=0,
        n_simulations=5,
    )

    # Scale factor to convert acceleration from 'g' to 'cm/s^2'
    scale = 980.665

    fig, axes = plt.subplots(
        3, 3, sharex="col", sharey="row", figsize=(12, 6), constrained_layout=True
    )

    comps = [
        ("Major", ts_m_med[0], "tab:blue"),
        ("Intermediate", ts_i_med[0], "tab:orange"),
        ("Vertical", ts_v_med[0], "tab:green"),
    ]

    metrics = [
        ("ac", "Acceleration\n(g)", 1.0),
        ("vel", "Velocity\n(cm/s)", scale),
        ("disp", "Displacement\n(cm)", scale),
    ]

    for col, (comp_name, sim, color) in enumerate(comps):
        axes[0, col].set_title(f"{comp_name} Component")

        for row, (metric, ylabel, scale_factor) in enumerate(metrics):
            ax = axes[row, col]

            # Transpose (n_simulations, npts) to (npts, n_simulations) for plotting
            data = getattr(sim, metric)
            if getattr(data, "ndim", 1) > 1:
                data = data.T

            # Plot ensemble cloud
            ax.plot(sim.t, data * scale_factor, color=color, alpha=0.3, linewidth=0.5)
            # Highlight one realization
            ax.plot(sim.t, data[:, 0] * scale_factor, color="k", linewidth=0.8)

            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("Time (s)")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.minorticks_on()
            ax.grid(axis="both", which="minor", linewidth=0.1, alpha=0.5)
            ax.grid(axis="both", which="major", linewidth=0.2, alpha=0.3)

    out_path = OUT_DIR / "simulated_3_component_traces.png"
    fig.savefig(out_path, dpi=200)
    print(f"  [OK] Saved -> {out_path}")
    plt.close(fig)

    # =========================================================================
    # 3. Conditional Hazard Targeting
    # =========================================================================
    print("\n--- 3. Visualizing Conditional Hazard Targeting ---")

    # High-hazard scenario: forcing Major Component Spectral Acceleration at 1.0s to 0.9g
    target_conditions = {"M_Sa_1": 0.9}

    ts_m_cond, ts_i_cond, ts_v_cond = gmm.simulate(
        Mw=6.5,
        Ztor=1.0,
        Rrup=25.0,
        Vs30=560.0,
        Fm="0",
        dt=0.005,
        conditions=target_conditions,
        n_samples=5,  # 5 conditioned parameter sets from the GMM
        n_simulations=1,  # 1 stochastic realization per set
    )

    periods_val = np.logspace(-2, 1, 50)
    sa_cond_m = np.zeros((5, len(periods_val)))

    for i, sim_realization in enumerate(ts_m_cond):
        _, _, sa = sim_realization.response_spectra(periods_val)
        sa_cond_m[i] = sa.flatten()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(periods_val, sa_cond_m.T, color="tab:blue", lw=0.5, alpha=0.5)

    sim_p50 = np.percentile(sa_cond_m, 50, axis=0)
    ax.loglog(
        periods_val, sim_p50, color="k", linewidth=1.5, label="Median Conditional Sa"
    )

    ax.plot(
        [1.0],
        [0.9],
        marker="*",
        color="red",
        markersize=15,
        label="User Target (0.9g at 1.0s)",
    )

    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Spectral Acceleration (g)")
    ax.set_title("Conditional Mean Spectra (Targeting 0.9g at 1.0s)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    out_path = OUT_DIR / "conditional_spectra_target.png"
    fig.savefig(out_path, dpi=200)
    print(f"  [OK] Saved -> {out_path}")
    plt.close(fig)

    print("\n" + "=" * 60)
    print("All done! Results are in the Desktop/PINAGMM Results/ directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
