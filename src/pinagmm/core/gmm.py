"""
PINAGMM — Physics-Informed Neural Additive Ground Motion Model.

This module exposes the ``PINAGMM`` class, the primary public interface for:

1. **Prediction** — ``predict()``:
   Returns a DataFrame of median (or conditionally-sampled) intensity measures
   and stochastic simulation parameters for the three principal components
   (Major, Intermediate, Vertical).

2. **Simulation** — ``simulate()``:
   Predicts parameters, passes them to the ``sgsim`` stochastic engine,
   and returns ``GroundMotion`` instances containing acceleration, velocity,
   and displacement time series.

References
----------
[1] Hussaini et al. (2025). "A Physics-Informed Neural Additive Ground Motion
    Model for Hazard-Compatible Three-Component Stochastic Simulation."
    Journal of Earthquake Engineering & Structural Dynamics.
    DOI: To be Added Later.

[2] Hussaini, S.S., Karimzadeh, S., Rezaeian, S. and Lourenço, P.B. (2025),
    Broadband stochastic simulation of earthquake ground motions with multiple strong phases with an application
    to the 2023 Kahramanmaraş, Turkey (Türkiye), earthquake. Earthquake Spectra, 41: 2399-2435.
    https://doi.org/10.1177/87552930251331981
"""

from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sgsim import StochasticModel
from .variables import yvars


class PINAGMM:
    """
    Physics-Informed Neural Additive Ground Motion Model.

    Predicts intensity measures (PGV, PGA, Sa up to 4 s) and stochastic
    simulation parameters across three principal statistical axes (Major,
    Intermediate, Vertical) using a Neural Additive Model coupled with
    Multivariate Mixed-Effects Regression.

    Examples
    --------
    >>> gmm = PINAGMM()

    Median prediction:
    >>> df = gmm.predict(Mw=6.5, Ztor=3.0, Rrup=15.0, Vs30=800.0, Fm="0")

    Simulate ground motions (median parameters for 3 realizations):
    >>> ts_m, ts_i, ts_v = gmm.simulate(Mw=6.5, Ztor=3.0, Rrup=15.0,
    ...                                  Vs30=800.0, Fm="0", n_simulations=3)

    Conditional simulation (target Sa(1s)=0.9 g) for 10 samples from GMM distribution and 1 realization each:
    >>> ts_m_list, ts_i_list, ts_v_list = gmm.simulate(
    ...     Mw=6.5, Ztor=3.0, Rrup=15.0, Vs30=800.0, Fm="0",
    ...     conditions={"M_Sa_1": 0.9}, n_samples=10, n_simulations=1)
    """

    def __init__(self):
        model_dir = Path(__file__).parent.parent / "model"
        self.preprocessor_x = joblib.load(model_dir / "xprocessor.joblib")
        self.scaler_y = joblib.load(model_dir / "yscaler.joblib")
        self.model = joblib.load(model_dir / "trained_model.joblib")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_scenario_df(self, Mw, Ztor, Rrup, Vs30, Fm="0") -> pd.DataFrame:
        """Broadcast scalar/array inputs into a consistent input DataFrame."""
        Mw, Ztor, Rrup, Vs30, Fm = np.broadcast_arrays(Mw, Ztor, Rrup, Vs30, Fm)
        return pd.DataFrame({
            "Earthquake Magnitude": Mw.flatten(),
            "Depth to Top Of Fault Rupture Model": Ztor.flatten(),
            "ClstD (km)": Rrup.flatten(),
            "Vs30 (m/s) selected for analysis": Vs30.flatten(),
            "Mechanism Based on Rake Angle": Fm.flatten(),
        })

    def _to_model_space(self, physical_values: np.ndarray, indices=None) -> np.ndarray:
        """
        Transform physical (un-logged) values to the model's standardised log-space.

        Parameters
        ----------
        physical_values : array-like
            Values in physical units (e.g., g for Sa, cm/s for PGV).
        indices : array-like of int, optional
            If given, apply only the mean/scale for these specific output columns
            (used for conditioning). If None, transform all columns.
        """
        log_vals = np.log(physical_values)
        if indices is None:
            return self.scaler_y.transform(np.atleast_2d(log_vals)).flatten()
        return (log_vals - self.scaler_y.mean_[indices]) / self.scaler_y.scale_[indices]

    def _to_physical_space(self, scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform from standardised log-space back to physical units."""
        return np.exp(self.scaler_y.inverse_transform(np.atleast_2d(scaled)))

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        Mw,
        Ztor,
        Rrup,
        Vs30,
        Fm,
        n_sample: int = 0,
        conditions: dict | None = None,
        random_state=None,
    ) -> pd.DataFrame:
        """
        Predict intensity measures and stochastic simulation parameters.

        Parameters
        ----------
        Mw, Ztor, Rrup, Vs30, Fm : float or array-like
            Earthquake scenario inputs (Magnitude, Ztor km, Rrup km, Vs30 m/s, rake mechanism).
        n_sample : int, optional
            Number of random GMM samples to draw.
            - ``0`` (default): Returns 1 row per scenario (the deterministic median prediction).
            - ``> 0``: Returns exactly ``n_sample`` rows per scenario sampled from the GMM
              multivariate normal distribution.
        conditions : dict, optional
           Conditional hazard target (e.g., ``{"M_Sa_1": 0.9}``). Constrains predictions to
           match target values while computing conditional mean and covariance.
        random_state : int, optional
            Seed for random number generator reproducibility.

        Returns
        -------
        pd.DataFrame
            Input scenario columns + predicted intensity measures and parameters.
            - When ``n_sample == 0``: Exactly 1 row per input scenario.
            - When ``n_sample > 0``: Exactly ``n_sample`` rows per input scenario.
        """
        if n_sample < 0:
            raise ValueError(f"n_sample must be >= 0, got {n_sample}.")

        rng = np.random.default_rng(random_state)
        df_input = self._build_scenario_df(Mw, Ztor, Rrup, Vs30, Fm)
        X = self.preprocessor_x.transform(df_input)

        # ── Unconditional median in model space (N, D) ──────────────────────
        mu = self.model.predict(X).value
        if mu.ndim == 1:
            mu = mu[np.newaxis, :]  # ensure (N, D)
        N, D = mu.shape
        marg_cov = self.model.marginal_cov().value  # (D, D)

        # ── Conditional adjustment (if conditions provided) ─────────────────
        if conditions:
            fixed_idx = np.array([yvars.index(k) for k in conditions])
            free_idx = np.setdiff1d(np.arange(D), fixed_idx)

            fixed_scaled = self._to_model_space(
                np.array(list(conditions.values()), dtype=float),
                indices=fixed_idx,
            )

            # Partition covariance
            Σ_ff = marg_cov[np.ix_(fixed_idx, fixed_idx)]
            Σ_rf = marg_cov[np.ix_(free_idx, fixed_idx)]
            Σ_fr = marg_cov[np.ix_(fixed_idx, free_idx)]
            Σ_rr = marg_cov[np.ix_(free_idx, free_idx)]

            # Conditional mean: μ_r|f = μ_r + Σ_rf Σ_ff⁻¹ (f - μ_f)
            delta = fixed_scaled - mu[:, fixed_idx]  # (N, |fixed|)
            cond_mean_r = mu[:, free_idx] + (Σ_rf @ np.linalg.solve(Σ_ff, delta.T)).T

            # Conditional covariance (Schur complement), symmetrised for stability
            cond_cov_rr = Σ_rr - Σ_rf @ np.linalg.solve(Σ_ff, Σ_fr)
            cond_cov_rr = 0.5 * (cond_cov_rr + cond_cov_rr.T)

            if n_sample > 0:
                noise = rng.multivariate_normal(
                    np.zeros(len(free_idx)), cond_cov_rr, size=(N, n_sample)
                )
                free_smp = noise + cond_mean_r[:, np.newaxis, :]  # (N, S, |free|)

                combined = np.empty((N, n_sample, D), dtype=float)
                combined[:, :, fixed_idx] = fixed_scaled
                combined[:, :, free_idx] = free_smp
            else:
                full_cond_mu = np.empty((N, D), dtype=float)
                full_cond_mu[:, fixed_idx] = fixed_scaled
                full_cond_mu[:, free_idx] = cond_mean_r
                combined = full_cond_mu[:, np.newaxis, :]  # (N, 1, D)

        else:
            if n_sample > 0:
                noise = rng.multivariate_normal(
                    np.zeros(D), marg_cov, size=(N, n_sample)
                )
                combined = noise + mu[:, np.newaxis, :]  # (N, S, D)
            else:
                combined = mu[:, np.newaxis, :]  # (N, 1, D) — median only

        repeats = 1 if n_sample == 0 else n_sample

        # combined shape: (N, repeats, D) → flatten to (N*repeats, D)
        combined = combined.reshape(-1, D)

        # ── Back to physical space ──────────────────────────────────────────
        physical = self._to_physical_space(combined).squeeze()
        if physical.ndim == 1:
            physical = physical[np.newaxis, :]

        df_pred = pd.DataFrame(physical, columns=yvars)

        df_input_rep = df_input.loc[df_input.index.repeat(repeats)].reset_index(
            drop=True
        )
        return pd.concat([df_input_rep, df_pred], axis=1)

    def extract_components(self, physical_row: np.ndarray, dt: float = 0.005) -> tuple:
        """
        Unpack one row of physical predictions into per-component parameter dicts
        and IM arrays suitable for ``StochasticModel.from_dict()``.

        Parameters
        ----------
        physical_row : ndarray of shape (len(yvars),)
            Physical-space values for one scenario (e.g., one row of ``predict()``
            output restricted to the ``yvars`` columns).
        dt : float
            Simulation time step (s). Used to compute the required ``npts``.

        Returns
        -------
        m_params, i_params, v_params : dict
            Parameter dicts for Major, Intermediate, and Vertical components.
        m_ims, i_ims, v_ims : ndarray
            Remaining IM values (PGV, Sa) for each component.
        """
        # Stochastic model keyword parameters predicted by the GMM
        _STOCH_PARAMS = {
            "q_centroid",
            "q_spread",
            "q_energy",
            "q_duration",
            "wu_value",
            "wl_value",
        }

        # Base parameter template required by StochasticModel.from_dict()
        params = {
            comp: {
                "q_type": "BetaCentroidSpread",
                "time_shift": 0.0,
                "wu_type": "Constant",
                "wl_type": "Constant",
                "zu_type": "Constant",
                "zu_value": 0.707,
                "zl_type": "Constant",
                "zl_value": 1.0,
            }
            for comp in ("m", "i", "v")
        }

        ims = {"m": [], "i": [], "v": []}

        for value, var_name in zip(physical_row, yvars):
            # Prefix is always exactly one uppercase letter before '_'
            prefix = var_name.split("_")[0].lower()  # 'm', 'i', or 'v'
            param_key = "_".join(var_name.split("_")[1:])  # e.g. 'q_duration', 'Sa_1'

            if prefix not in params:
                raise ValueError(
                    f"Unrecognised component prefix '{prefix}' in '{var_name}'. "
                    f"Expected one of M, I, V."
                )

            if param_key in _STOCH_PARAMS:
                params[prefix][param_key] = value
            else:
                ims[prefix].append(value)

        # Convert centroid/spread from absolute seconds → normalised ratios [0, 1]
        # and compute npts.  The ML model predicts absolute durations in seconds;
        # sgsim expects centroid and spread as fractions of total duration.
        for comp in ("m", "i", "v"):
            dur = params[comp]["q_duration"]

            # Normalise; clip to valid Beta-distribution domain to guard against
            # regression extrapolation.  Centroid ∈ (0,1), spread ∈ (0, 0.5).
            params[comp]["q_centroid"] = float(
                np.clip(params[comp]["q_centroid"] / dur, 0.05, 0.90)
            )
            params[comp]["q_spread"] = float(
                np.clip(params[comp]["q_spread"] / dur, 0.01, 0.45)
            )

            # Add 20 % padding to avoid truncating the signal coda
            params[comp]["npts"] = int(1.2 * np.ceil(dur / dt))
            params[comp]["dt"] = float(dt)

        return (
            params["m"],
            params["i"],
            params["v"],
            np.asarray(ims["m"]),
            np.asarray(ims["i"]),
            np.asarray(ims["v"]),
        )

    def simulate(
        self,
        Mw,
        Ztor,
        Rrup,
        Vs30,
        Fm,
        conditions: dict | None = None,
        random_state=None,
        dt: float = 0.005,
        n_samples: int = 0,
        n_simulations: int = 1,
    ):
        """
        Predict parameters from GMM and generate stochastic ground motion time-series.

        Combines parameter prediction with the stochastic simulation engine (sgsim).

        Parameters
        ----------
        Mw, Ztor, Rrup, Vs30, Fm :
            Earthquake scenario inputs.
        conditions : dict, optional
            Conditional hazard target (e.g. ``{"M_Sa_1": 0.9}``).
        random_state : int, optional
            Seed for random number generator reproducibility.
        dt : float
            Time step in seconds (default 0.005 s = 200 Hz).
        n_samples : int
            Number of parameter sets drawn from GMM distribution (default 0 → median prediction).
        n_simulations : int
            Number of independent stochastic time-series realisations per parameter set.

        Returns
        -------
        ts_m_list, ts_i_list, ts_v_list : tuple of lists
            Three lists of ``GroundMotion`` instances (Major, Intermediate, Vertical components).
            - When ``n_samples == 0``: Lists contain 1 ``GroundMotion`` object (for median parameters).
            - When ``n_samples > 0``: Lists contain ``n_samples`` ``GroundMotion`` objects.
        """
        if n_samples < 0:
            raise ValueError(f"n_samples must be >= 0, got {n_samples}.")
        if n_simulations < 1:
            raise ValueError(f"n_simulations must be >= 1, got {n_simulations}.")

        df_pred = self.predict(
            Mw=Mw,
            Ztor=Ztor,
            Rrup=Rrup,
            Vs30=Vs30,
            Fm=Fm,
            n_sample=n_samples,
            conditions=conditions,
            random_state=random_state,
        )

        def _build_trio(row_values: np.ndarray):
            """Create three GroundMotion objects from one parameter row."""
            m_p, i_p, v_p, _, _, _ = self.extract_components(row_values, dt=dt)
            return (
                StochasticModel.from_dict(m_p).simulate(n=n_simulations, unit="g"),
                StochasticModel.from_dict(i_p).simulate(n=n_simulations, unit="g"),
                StochasticModel.from_dict(v_p).simulate(n=n_simulations, unit="g"),
            )

        n_rows = len(df_pred)
        ts_m_list, ts_i_list, ts_v_list = [], [], []
        for r in range(n_rows):
            ts_m, ts_i, ts_v = _build_trio(df_pred[yvars].iloc[r].values)
            ts_m_list.append(ts_m)
            ts_i_list.append(ts_i)
            ts_v_list.append(ts_v)

        return ts_m_list, ts_i_list, ts_v_list
