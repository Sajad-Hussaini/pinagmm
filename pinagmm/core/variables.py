"""
Centralized variable definitions for PINAGMM.

This module defines:
- Input feature names used by the preprocessor (xvars, xraw, xlog, xcat)
- Output target names in log-space (yvars) across three principal components:
    M = Major, I = Intermediate, V = Vertical
- Index slices into yvars for each physical group
- Group architectures for the ensemble neural network
- Scientific labels and spectral periods for plotting
"""

# ---------------------------------------------------------------------------
# Input feature columns
# ---------------------------------------------------------------------------
xraw = ["Earthquake Magnitude", "Depth to Top Of Fault Rupture Model"]
xlog = ["ClstD (km)", "Vs30 (m/s) selected for analysis"]
xcat = ["Mechanism Based on Rake Angle"]
xnumeric = xraw + xlog
xvars = xnumeric + xcat

# Grouping variables (not used for prediction, only for training data)
gvars = ["Earthquake Name", "Station Name"]

# ---------------------------------------------------------------------------
# Output target names  (order must match the trained model)
# ---------------------------------------------------------------------------
yvars = [
    # ── Temporal envelope parameters ──────────────────────────────────────
    "M_q_duration",
    "I_q_duration",
    "V_q_duration",
    "M_q_centroid",
    "I_q_centroid",
    "V_q_centroid",
    "M_q_spread",
    "I_q_spread",
    "V_q_spread",
    # ── Scale parameter (energy) ───────────────────────────────────────────
    "M_q_energy",
    "I_q_energy",
    "V_q_energy",
    # ── Frequency bandwidth ────────────────────────────────────────────────
    "M_wu_value",
    "I_wu_value",
    "V_wu_value",
    "M_wl_value",
    "I_wl_value",
    "V_wl_value",
    # ── Intensity measures: PGV ────────────────────────────────────────────
    "M_PGV",
    "I_PGV",
    "V_PGV",
    # ── Intensity measures: PGA (Sa at T=0) ───────────────────────────────
    "M_Sa_0",
    "I_Sa_0",
    "V_Sa_0",
    # ── Intensity measures: spectral accelerations ────────────────────────
    "M_Sa_0.03",
    "I_Sa_0.03",
    "V_Sa_0.03",
    "M_Sa_0.05",
    "I_Sa_0.05",
    "V_Sa_0.05",
    "M_Sa_0.08",
    "I_Sa_0.08",
    "V_Sa_0.08",
    "M_Sa_0.1",
    "I_Sa_0.1",
    "V_Sa_0.1",
    "M_Sa_0.16",
    "I_Sa_0.16",
    "V_Sa_0.16",
    "M_Sa_0.2",
    "I_Sa_0.2",
    "V_Sa_0.2",
    "M_Sa_0.3",
    "I_Sa_0.3",
    "V_Sa_0.3",
    "M_Sa_0.4",
    "I_Sa_0.4",
    "V_Sa_0.4",
    "M_Sa_0.5",
    "I_Sa_0.5",
    "V_Sa_0.5",
    "M_Sa_0.75",
    "I_Sa_0.75",
    "V_Sa_0.75",
    "M_Sa_1",
    "I_Sa_1",
    "V_Sa_1",
    "M_Sa_1.5",
    "I_Sa_1.5",
    "V_Sa_1.5",
    "M_Sa_2",
    "I_Sa_2",
    "V_Sa_2",
    "M_Sa_3",
    "I_Sa_3",
    "V_Sa_3",
    "M_Sa_4",
    "I_Sa_4",
    "V_Sa_4",
]

# ---------------------------------------------------------------------------
# Index slices into yvars
# ---------------------------------------------------------------------------
y_energy = [yvars.index(v) for v in yvars if "energy" in v]
y_duration = [
    yvars.index(v)
    for v in yvars
    if any(s in v for s in ["duration", "centroid", "spread"])
]
y_freq = [yvars.index(v) for v in yvars if any(s in v for s in ["wl_", "wu_"])]
y_ims = [yvars.index(v) for v in yvars if any(s in v for s in ["PGV", "Sa"])]

y_ims_major = [yvars.index(v) for v in yvars if any(s in v for s in ["M_PGV", "M_Sa"])]
y_ims_inter = [yvars.index(v) for v in yvars if any(s in v for s in ["I_PGV", "I_Sa"])]
y_ims_vert = [yvars.index(v) for v in yvars if any(s in v for s in ["V_PGV", "V_Sa"])]


def _ordered_model_indices(prefix: str) -> list[int]:
    """Return yvars indices for one component in the order: E, fU, fL, D, C, S."""
    patterns = [
        f"{prefix}_q_energy",
        f"{prefix}_wu_value",
        f"{prefix}_wl_value",
        f"{prefix}_q_duration",
        f"{prefix}_q_centroid",
        f"{prefix}_q_spread",
    ]
    seen: set[int] = set()
    out: list[int] = []
    for p in patterns:
        for i, v in enumerate(yvars):
            if p in v and i not in seen:
                seen.add(i)
                out.append(i)
    return out


y_model_major = _ordered_model_indices("M")
y_model_inter = _ordered_model_indices("I")
y_model_vert = _ordered_model_indices("V")

# ---------------------------------------------------------------------------
# Column index shortcuts (used in group_definitions below)
# ---------------------------------------------------------------------------
col_ids = {
    "Mw": 0,
    "Ztor": 1,
    "Rrup": 2,
    "Vs30": 3,
    "im_indices": [yvars.index(v) for v in yvars if any(s in v for s in ["PGV", "Sa"])],
    "energy_indices": [yvars.index(v) for v in yvars if "energy" in v],
    "duration_indices": [yvars.index(v) for v in yvars if "duration" in v],
    "centroid_indices": [yvars.index(v) for v in yvars if "centroid" in v],
    "spread_indices": [yvars.index(v) for v in yvars if "spread" in v],
    "wl_indices": [yvars.index(v) for v in yvars if "wl_" in v],
    "wu_indices": [yvars.index(v) for v in yvars if "wu_" in v],
}

# ---------------------------------------------------------------------------
# Ensemble group definitions (architecture config for training)
# ---------------------------------------------------------------------------
group_definitions = [
    {
        "name": "IM_Energy",
        "out_idx": col_ids["im_indices"] + col_ids["energy_indices"],
        "mono_in": [col_ids["Mw"], col_ids["Rrup"]],
        "signs": [1.0, -1.0],
        "interactions": [
            {
                "features": [col_ids["Mw"], col_ids["Rrup"]],
                "monotonic": True,
                "signs": [1.0, -1.0],
            },
        ],
    },
    {
        "name": "Duration",
        "out_idx": col_ids["duration_indices"]
        + col_ids["centroid_indices"]
        + col_ids["spread_indices"],
        "mono_in": [col_ids["Mw"], col_ids["Rrup"]],
        "signs": [1.0, 1.0],
        "interactions": [
            {
                "features": [col_ids["Mw"], col_ids["Rrup"]],
                "monotonic": True,
                "signs": [1.0, 1.0],
            },
        ],
    },
    {
        "name": "Freq_Lower",
        "out_idx": col_ids["wl_indices"],
        "mono_in": [col_ids["Mw"], col_ids["Rrup"]],
        "signs": [-1.0, -1.0],
        "interactions": [
            {
                "features": [col_ids["Mw"], col_ids["Rrup"]],
                "monotonic": True,
                "signs": [-1.0, -1.0],
            },
        ],
    },
    {
        "name": "Freq_Upper",
        "out_idx": col_ids["wu_indices"],
        "mono_in": [col_ids["Mw"], col_ids["Rrup"]],
        "signs": [-1.0, -1.0],
        "interactions": [
            {
                "features": [col_ids["Mw"], col_ids["Rrup"]],
                "monotonic": True,
                "signs": [-1.0, -1.0],
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# Spectral periods (excludes PGA / T=0)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Scientific axis labels (for plotting)
# ---------------------------------------------------------------------------
ylabels = [
    r"$D_{M}$",
    r"$D_{I}$",
    r"$D_{V}$",
    r"$C_{M}$",
    r"$C_{I}$",
    r"$C_{V}$",
    r"$S_{M}$",
    r"$S_{I}$",
    r"$S_{V}$",
    r"$E_{M}$",
    r"$E_{I}$",
    r"$E_{V}$",
    r"$f_{U,M}$",
    r"$f_{U,I}$",
    r"$f_{U,V}$",
    r"$f_{L,M}$",
    r"$f_{L,I}$",
    r"$f_{L,V}$",
    r"$PGV_{M}$",
    r"$PGV_{I}$",
    r"$PGV_{V}$",
    r"$PGA_{M}$",
    r"$PGA_{I}$",
    r"$PGA_{V}$",
    r"$SA_{0.03,M}$",
    r"$SA_{0.03,I}$",
    r"$SA_{0.03,V}$",
    r"$SA_{0.05,M}$",
    r"$SA_{0.05,I}$",
    r"$SA_{0.05,V}$",
    r"$SA_{0.08,M}$",
    r"$SA_{0.08,I}$",
    r"$SA_{0.08,V}$",
    r"$SA_{0.1,M}$",
    r"$SA_{0.1,I}$",
    r"$SA_{0.1,V}$",
    r"$SA_{0.16,M}$",
    r"$SA_{0.16,I}$",
    r"$SA_{0.16,V}$",
    r"$SA_{0.2,M}$",
    r"$SA_{0.2,I}$",
    r"$SA_{0.2,V}$",
    r"$SA_{0.3,M}$",
    r"$SA_{0.3,I}$",
    r"$SA_{0.3,V}$",
    r"$SA_{0.4,M}$",
    r"$SA_{0.4,I}$",
    r"$SA_{0.4,V}$",
    r"$SA_{0.5,M}$",
    r"$SA_{0.5,I}$",
    r"$SA_{0.5,V}$",
    r"$SA_{0.75,M}$",
    r"$SA_{0.75,I}$",
    r"$SA_{0.75,V}$",
    r"$SA_{1.0,M}$",
    r"$SA_{1.0,I}$",
    r"$SA_{1.0,V}$",
    r"$SA_{1.5,M}$",
    r"$SA_{1.5,I}$",
    r"$SA_{1.5,V}$",
    r"$SA_{2.0,M}$",
    r"$SA_{2.0,I}$",
    r"$SA_{2.0,V}$",
    r"$SA_{3.0,M}$",
    r"$SA_{3.0,I}$",
    r"$SA_{3.0,V}$",
    r"$SA_{4.0,M}$",
    r"$SA_{4.0,I}$",
    r"$SA_{4.0,V}$",
]

# ---------------------------------------------------------------------------
# Dtype mapping (for training data loaders)
# ---------------------------------------------------------------------------
dtype_mapping = {
    "Earthquake Name": "str",
    "Station Name": "str",
    "Mechanism Based on Rake Angle": "str",
    **{var: "float" for var in xnumeric},
}
