"""
Ground motion I/O utilities for PINAGMM.

Provides standalone functions to save acceleration time series and response
spectra to CSV files.  The functions operate on ``sgsim.GroundMotion``
objects (or lists thereof) and are independent of the ``PINAGMM`` model.

Saving strategy
---------------
When multiple ``GroundMotion`` objects with different record lengths (``npts``)
are combined, shorter records are **zero-padded at the tail**.  This is
physically sound: real ground motion energy decays to negligible amplitude
well before the end of the record duration, so appending zeros introduces no
meaningful distortion.

CSV format
----------
``save_timeseries``:  columns ``[t, ac_0, ac_1, …]`` — one column per
simulation; each value in ``g`` (the native sgsim unit).

``save_spectra``:     columns ``[period_s, Sa_0, Sa_1, …]`` — one column per
simulation; each value in ``g``.
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
        One or more ``sgsim.GroundMotion`` instances.  Each instance's
        ``.ac`` may be 1-D ``(npts,)`` (single simulation) or 2-D
        ``(n_simulations, npts)`` (multiple simulations).  When a list is
        provided and records differ in length, shorter records are
        zero-padded to ``max(npts)``.
    filepath : str or Path
        Destination CSV file path.  Parent directories are created
        automatically.
    dt : float, optional
        Time step in seconds.  Defaults to the ``.dt`` attribute of the
        first record.

    Returns
    -------
    pd.DataFrame
        The DataFrame that was written to disk, with columns
        ``['t', 'ac_0', 'ac_1', …]``.

    Examples
    --------
    Single GroundMotion with 5 simulations:

    >>> save_timeseries(ts_m, "major_timeseries.csv")

    List of GroundMotion objects (possibly different lengths):

    >>> save_timeseries([ts_m1, ts_m2, ts_m3], "major_timeseries.csv")
    """
    gm_list = _to_list(ground_motions)
    if not gm_list:
        raise ValueError("ground_motions is empty.")

    dt = dt if dt is not None else gm_list[0].dt

    # ── Collect every simulation as a 1-D array ─────────────────────────────
    rows: list[np.ndarray] = []
    for gm in gm_list:
        for sim_row in np.atleast_2d(gm.ac):  # always iterate over dim-0
            rows.append(np.asarray(sim_row, dtype=np.float64))

    # ── Zero-pad to max length ───────────────────────────────────────────────
    max_npts = max(len(r) for r in rows)
    matrix = np.zeros((len(rows), max_npts), dtype=np.float64)
    for i, row in enumerate(rows):
        matrix[i, : len(row)] = row

    # ── Build and write DataFrame ────────────────────────────────────────────
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

    Each simulation's 5%-damped spectral acceleration is computed via
    ``sgsim.GroundMotion.response_spectra()``.

    Parameters
    ----------
    ground_motions : GroundMotion or list of GroundMotion
        One or more ``sgsim.GroundMotion`` instances.
    filepath : str or Path
        Destination CSV file path.  Parent directories are created
        automatically.
    periods : array-like
        Oscillator periods in seconds at which Sa is evaluated.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['period_s', 'Sa_0', 'Sa_1', …]``.

    Examples
    --------
    >>> import numpy as np
    >>> T = np.logspace(-2, np.log10(4), 80)
    >>> save_spectra(ts_m, "major_spectra.csv", T)
    """
    gm_list = _to_list(ground_motions)
    if not gm_list:
        raise ValueError("ground_motions is empty.")

    periods = np.asarray(periods, dtype=float)

    # ── Collect every simulation's Sa vector ────────────────────────────────
    sa_rows: list[np.ndarray] = []
    for gm in gm_list:
        _, _, sa = gm.response_spectra(periods)
        # sa shape: (n_periods,) for 1 sim, or (n_sims, n_periods) for multiple
        for row in np.atleast_2d(sa):
            sa_rows.append(np.asarray(row, dtype=np.float64))

    sa_matrix = np.vstack(sa_rows)  # (total_sims, n_periods)

    # ── Build and write DataFrame ────────────────────────────────────────────
    df = pd.DataFrame({
        "period_s": periods,
        **{f"Sa_{i}": sa_matrix[i] for i in range(len(sa_rows))},
    })
    _write_csv(df, filepath)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_list(ground_motions) -> list:
    """Normalise a single GroundMotion or a list to a plain list."""
    if isinstance(ground_motions, list):
        return ground_motions
    return [ground_motions]


def _write_csv(df: pd.DataFrame, filepath: str | Path) -> None:
    """Ensure parent dir exists and write CSV with compact float formatting."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False, float_format="%.8g")
