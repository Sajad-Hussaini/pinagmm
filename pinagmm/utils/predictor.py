import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from .setup_variables import yvars
from sgsim import StochasticModel


class PINAGMM:
    """
    Physics-Informed Neural Additive Ground Motion Model

    This GMM predicts key IMs (PGV, PGA, SA upto 4s) and simulation parameters of the broadband stochastic simulation method [1] across three principal statistical axes

    [1] Broadband Stochastic Simulation of Earthquake Ground Motions with Multiple Strong Phases with an Application to the 2023 Kahramanmaraş, Turkey (Türkiye), Earthquake
        DOI: https://doi.org/10.1177/87552930251331981
        Journal of Earthquake Spectra
    """

    def __init__(self):
        model_dir = Path(__file__).resolve().parent.parent / "model"
        self.preprocessor_x = joblib.load(model_dir / "preprocessor_x.joblib")
        self.scaler_y = joblib.load(model_dir / "scaler_y.joblib")
        self.model = joblib.load(model_dir / "newensemble_pinam.joblib")

    def scenario(self, Mw, Ztor, Rrup, Vs30, Fm="0"):
        Mw, Ztor, Rrup, Vs30, Fm = np.broadcast_arrays(Mw, Ztor, Rrup, Vs30, Fm)
        return pd.DataFrame({
            "Earthquake Magnitude": Mw.flatten(),
            "Depth to Top Of Fault Rupture Model": Ztor.flatten(),
            "ClstD (km)": Rrup.flatten(),
            "Vs30 (m/s) selected for analysis": Vs30.flatten(),
            "Mechanism Based on Rake Angle": Fm.flatten(),
        })

    def physical_to_model(self, value, indices=None):
        value_log = np.log(value)
        if indices is None:
            value_log = np.atleast_2d(value_log)
            return self.scaler_y.transform(value_log).flatten()
        else:
            return (value_log - self.scaler_y.mean_[indices]) / self.scaler_y.scale_[
                indices
            ]

    def model_to_physical(self, y_scaled):
        y_scaled = np.atleast_2d(y_scaled)
        return np.exp(self.scaler_y.inverse_transform(y_scaled)).squeeze()

    def predict(
        self,
        Mw,
        Ztor,
        Rrup,
        Vs30,
        Fm,
        n_sample=0,
        conditions: dict = None,
        random_state=None,
    ):
        """
        Returns the predicted median (or conditional mean) and samples of the GMM.

        Parameters:
        -----------
        Mw : float
            Earthquake magnitude
        Ztor : float
            Depth to top of fault rupture model (km)
        Rrup : float
            Closest distance from the site to the rupture (km)
        Vs30 : float
            Shear wave velocity at 30 m depth (m/s)
        Fm : str
            Mechanism based on rake angle ("0" for normal fault, "1" for reverse fault, "2" for strike-slip fault)
        n_sample : int
            Number of samples to generate
        conditions : dict, optional
            A dictionary mapping output names (e.g., "PGA", "PGV") to their conditioned physical values.
            If provided, performs conditional sampling.
        random_state : int
            Random state for reproducibility

        Returns:
        --------
        pd.DataFrame
            DataFrame where the first row is the predicted median (or conditional mean),
            followed by `n_sample` rows of samples.
        """
        rng = np.random.default_rng(random_state)
        df_input = self.scenario(Mw, Ztor, Rrup, Vs30, Fm)
        X_processed = self.preprocessor_x.transform(df_input)

        # Original unconditional median and covariance
        # Ensure mu is strictly 2D: (N, D)
        mu = self.model.predict(X_processed).value.squeeze()
        if mu.ndim == 1:
            mu = np.atleast_2d(mu)

        N, D = mu.shape
        marg_cov = self.model.marginal_cov().value

        if conditions:
            # Map string keys to column indices using yvars
            fixed_indices = [yvars.index(k) for k in conditions.keys()]
            fixed_values_physical = np.array(list(conditions.values()))

            # Scale the physical condition values to model space
            fixed_values_scaled = self.physical_to_model(
                fixed_values_physical, indices=fixed_indices
            )

            free_idx = np.setdiff1d(np.arange(D), fixed_indices)

            mu_f = mu[:, fixed_indices]
            mu_r = mu[:, free_idx]

            Sigma_ff = marg_cov[np.ix_(fixed_indices, fixed_indices)]
            Sigma_fr = marg_cov[np.ix_(fixed_indices, free_idx)]
            Sigma_rf = marg_cov[np.ix_(free_idx, fixed_indices)]
            Sigma_rr = marg_cov[np.ix_(free_idx, free_idx)]

            # Compute conditional mean and covariance
            delta = fixed_values_scaled - mu_f

            # cond_mean_r: (N, len(free_idx))
            cond_mean_r = mu_r + (Sigma_rf @ np.linalg.solve(Sigma_ff, delta.T)).T

            cond_cov_rr = Sigma_rr - Sigma_rf @ np.linalg.solve(Sigma_ff, Sigma_fr)
            cond_cov_rr = 0.5 * (cond_cov_rr + cond_cov_rr.T)  # ensure symmetry

            full_cond_mean = np.empty((N, D), dtype=float)
            full_cond_mean[:, fixed_indices] = fixed_values_scaled
            full_cond_mean[:, free_idx] = cond_mean_r

            if n_sample > 0:
                base_samples = rng.multivariate_normal(
                    np.zeros(len(free_idx)), cond_cov_rr, size=(N, n_sample)
                )
                free_samples = base_samples + cond_mean_r[:, None, :]

                samples_scaled = np.empty((N, n_sample, D), dtype=float)
                samples_scaled[:, :, fixed_indices] = fixed_values_scaled
                samples_scaled[:, :, free_idx] = free_samples

                combined = np.concatenate(
                    (full_cond_mean[:, None, :], samples_scaled), axis=1
                )
                combined = combined.reshape(-1, D)
            else:
                combined = full_cond_mean
        else:
            if n_sample > 0:
                base_samples = rng.multivariate_normal(
                    np.zeros(D), marg_cov, size=(N, n_sample)
                )
                samples = base_samples + mu[:, None, :]
                combined = np.concatenate((mu[:, None, :], samples), axis=1)
                combined = combined.reshape(-1, D)
            else:
                combined = mu

        # Convert back to physical values
        physical_pred = np.exp(self.scaler_y.inverse_transform(combined))
        df_pred = pd.DataFrame(physical_pred, columns=yvars)

        # Attach the input scenario features so the user knows which row belongs to which scenario
        repeats = 1 + n_sample if n_sample > 0 else 1
        df_input_repeated = df_input.loc[df_input.index.repeat(repeats)].reset_index(
            drop=True
        )

        return pd.concat([df_input_repeated, df_pred], axis=1)

    def extract_components(self, phys_array, dt: float = 0.005):
        """
        Organize a 1D physical array (like mean_phys or a single row of samples_phys)
        into the format required by StochasticModel.load_from for major (m),
        intermediate (i), and vertical (v) components, along with their raw IM slices.
        """
        params_out = {"m": {}, "i": {}, "v": {}}
        ims_out = {"m": [], "i": [], "v": []}

        # Base structure template for StochasticModel.from_dict
        for comp in ["m", "i", "v"]:
            params_out[comp] = {
                "q_type": "BetaCentroidSpread",
                "time_shift": 0.0,
                "wu_type": "Constant",
                "wl_type": "Constant",
                "zu_type": "Constant",
                "zu_value": 0.707,
                "zl_type": "Constant",
                "zl_value": 1.0,
            }

        known_params = {
            "q_centroid",
            "q_spread",
            "q_energy",
            "q_duration",
            "wu_value",
            "wl_value",
        }

        # Map yvars names to the corresponding dict structure and IM arrays
        for val, var_name in zip(phys_array, yvars):
            comp = var_name[0].lower()  # 'm', 'i', or 'v'
            name_rest = var_name[2:]  # Extract exactly after 'M_' (e.g., 'q_centroid')

            if name_rest in known_params:
                params_out[comp][name_rest] = val
            else:
                # Anything else (PGV, Sa_*) goes to Intensity Measures
                ims_out[comp].append(val)

        # Convert centroid and spread to ratio values, and add dynamic npts calculation
        for comp in ["m", "i", "v"]:
            duration = params_out[comp]["q_duration"]

            c_ratio = params_out[comp]["q_centroid"] / duration
            s_ratio = params_out[comp]["q_spread"] / duration

            # Post-regression physical regularization:
            # The ML model generated unbounded continuous parameters. We must strictly clip the
            # resulting ratios so the Beta-distribution does not become mathematically
            # undefined or U-shaped (convex instead of concave).
            # Centroid ratio ideally bounded around [0.05, 0.95]
            # Spread ratio bounded around [0.01, 0.45]
            params_out[comp]["q_centroid"] = np.clip(c_ratio, 0.05, 0.9)
            params_out[comp]["q_spread"] = np.clip(s_ratio, 0.01, 0.45)

            params_out[comp]["npts"] = int(1.2 * np.ceil(duration / dt))
            params_out[comp]["dt"] = float(dt)

        # Convert IM lists to numpy arrays
        for comp in ["m", "i", "v"]:
            ims_out[comp] = np.array(ims_out[comp])

        return (
            params_out["m"],
            params_out["i"],
            params_out["v"],
            ims_out["m"],
            ims_out["i"],
            ims_out["v"],
        )

    def simulate(
        self,
        Mw,
        Ztor,
        Rrup,
        Vs30,
        Fm,
        conditions=None,
        random_state=None,
        dt=0.005,
        n_samples=0,
        n_simulations=1,
    ):
        """
        Convenience method to predict parameters and generate a time series realization.
        If conditions are provided, it performs conditional sampling automatically.

        Parameters:
        -----------
        Mw : float
            Earthquake magnitude
        Ztor : float
            Depth to top of fault rupture model (km)
        Rrup : float
            Closest distance from the site to the rupture (km)
        Vs30 : float
            Shear wave velocity at 30 m depth (m/s)
        Fm : str
            Mechanism based on rake angle ("0" for normal fault, "1" for reverse fault, "2" for strike-slip fault)
        conditions : dict, optional
            A dictionary mapping output names (e.g., "PGA", "PGV") to their conditioned physical values.
        random_state : int, optional
            Random state for reproducibility.
        dt : float
            Time step for the generated time series.
        n_samples : int
            Number of parameter sets to sample from the GMM.
            If 0, it uses the median GMM prediction (no parameter uncertainty).
        n_simulations : int
            Number of time series realizations to generate per parameter set (via the stochastic engine).

        Returns:
        --------
        ts_m, ts_i, ts_v : GroundMotion or list
            If `n_samples=0` or `n_samples=1`, returns a single GroundMotion object per component, where `.ac` is an array of shape (n_simulations, npts).
            If `n_samples > 1`, returns a list of GroundMotion objects per component, where the list has `n_samples` elements.
        """

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

        if n_samples == 0:
            sample_phys = df_pred[yvars].iloc[0].values
            m_params, i_params, v_params, _, _, _ = self.extract_components(
                sample_phys, dt=dt
            )

            model_m = StochasticModel.from_dict(m_params)
            model_i = StochasticModel.from_dict(i_params)
            model_v = StochasticModel.from_dict(v_params)

            ts_m = model_m.simulate(n=n_simulations)
            ts_i = model_i.simulate(n=n_simulations)
            ts_v = model_v.simulate(n=n_simulations)

            return ts_m, ts_i, ts_v
        else:
            ts_m_list = []
            ts_i_list = []
            ts_v_list = []

            # Row 0 is the median/mean. Row 1 to n_samples are the samples.
            for row in range(1, n_samples + 1):
                sample_phys = df_pred[yvars].iloc[row].values
                m_params, i_params, v_params, _, _, _ = self.extract_components(
                    sample_phys, dt=dt
                )

                model_m = StochasticModel.from_dict(m_params)
                model_i = StochasticModel.from_dict(i_params)
                model_v = StochasticModel.from_dict(v_params)

                ts_m_list.append(model_m.simulate(n=n_simulations))
                ts_i_list.append(model_i.simulate(n=n_simulations))
                ts_v_list.append(model_v.simulate(n=n_simulations))

            if n_samples == 1:
                return ts_m_list[0], ts_i_list[0], ts_v_list[0]

            return ts_m_list, ts_i_list, ts_v_list
