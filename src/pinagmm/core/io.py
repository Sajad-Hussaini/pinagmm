"""
Ground motion I/O utilities for PINAGMM.

Provides standalone utility functions to save acceleration time-series and
response spectra to CSV files. Operating on ``sgsim.GroundMotion`` objects
(or lists thereof), these functions handle zero-padding for records of varying
length and create destination parent directories automatically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_timeseries(
    ground_motions,
    filepath: str | Path,
    dt: float | None = None,
) -> pd.DataFrame:
    """
    Save acceleration time series to a CSV file.

    Parameters
    ----------
    ground_motions : GroundMotion or list of GroundMotion
        One or more ``sgsim.GroundMotion`` instances. If a list of records with
        differing lengths is provided, shorter records are zero-padded at the tail.
    filepath : str or Path
        Destination CSV file path. Parent directories are created automatically.
    dt : float, optional
        Time step in seconds. Defaults to the ``.dt`` attribute of the first record.

    Returns
    -------
    pd.DataFrame
        DataFrame written to disk with columns ``['t', 'ac_0', 'ac_1', ...]``.

    Examples
    --------
    >>> save_timeseries(ts_m, "major_component.csv")
    >>> save_timeseries([ts_m1, ts_m2], "major_ensemble.csv")
    """
    gm_list = (
        ground_motions
        if isinstance(ground_motions, (list, tuple))
        else [ground_motions]
    )
    if not gm_list:
        raise ValueError("ground_motions list is empty.")

    dt = dt if dt is not None else gm_list[0].dt

    # Collect every simulation acceleration array
    rows: list[np.ndarray] = []
    for gm in gm_list:
        for sim_row in np.atleast_2d(gm.ac):
            rows.append(np.asarray(sim_row, dtype=np.float64))

    # Zero-pad to max length
    max_npts = max(len(r) for r in rows)
    matrix = np.zeros((len(rows), max_npts), dtype=np.float64)
    for i, row in enumerate(rows):
        matrix[i, : len(row)] = row

    # Build and export DataFrame
    t = np.arange(max_npts) * dt
    df = pd.DataFrame({"t": t, **{f"ac_{i}": matrix[i] for i in range(len(rows))}})
    _write_csv(df, filepath)
    return df


def save_spectra(
    ground_motions,
    filepath: str | Path,
    periods: np.ndarray,
) -> pd.DataFrame:
    """
    Compute and save response spectra (Sa) to a CSV file.

    Parameters
    ----------
    ground_motions : GroundMotion or list of GroundMotion
        One or more ``sgsim.GroundMotion`` instances.
    filepath : str or Path
        Destination CSV file path. Parent directories are created automatically.
    periods : array-like
        Oscillator periods in seconds at which spectral acceleration (Sa) is computed.

    Returns
    -------
    pd.DataFrame
        DataFrame written to disk with columns ``['period_s', 'Sa_0', 'Sa_1', ...]``.

    Examples
    --------
    >>> import numpy as np
    >>> T = np.logspace(-2, 1, 50)
    >>> save_spectra(ts_m, "major_spectra.csv", T)
    """
    gm_list = (
        ground_motions
        if isinstance(ground_motions, (list, tuple))
        else [ground_motions]
    )
    if not gm_list:
        raise ValueError("ground_motions list is empty.")

    periods = np.asarray(periods, dtype=float)

    # Collect response spectrum for every simulation
    sa_rows: list[np.ndarray] = []
    for gm in gm_list:
        _, _, sa = gm.response_spectra(periods)
        for row in np.atleast_2d(sa):
            sa_rows.append(np.asarray(row, dtype=np.float64))

    sa_matrix = np.vstack(sa_rows)

    # Build and export DataFrame
    df = pd.DataFrame({
        "period_s": periods,
        **{f"Sa_{i}": sa_matrix[i] for i in range(len(sa_rows))},
    })
    _write_csv(df, filepath)
    return df


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _write_csv(df: pd.DataFrame, filepath: str | Path) -> None:
    """Ensure destination directory exists and write CSV with formatted floats."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False, float_format="%.8g")
