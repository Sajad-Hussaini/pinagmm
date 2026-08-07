"""
PINAGMM — Physics-Informed Neural Additive Ground Motion Model.

Public API
----------
PINAGMM          : Ground motion model (predict + simulate).
save_timeseries  : Save acceleration time series to CSV.
save_spectra     : Save response spectra to CSV.

Quick start
-----------
>>> from pinagmm import PINAGMM, save_timeseries, save_spectra
>>> gmm = PINAGMM()
>>> df  = gmm.predict(Mw=6.5, Ztor=3.0, Rrup=15.0, Vs30=800.0, Fm="0")
>>> ts_m, ts_i, ts_v = gmm.simulate(Mw=6.5, Ztor=3.0, Rrup=15.0,
...                                   Vs30=800.0, Fm="0", n_simulations=3)
>>> save_timeseries(ts_m, "major_component.csv")

For the interactive GUI run in your terminal:
    pinagmm
"""

from .core.gmm import PINAGMM
from .core.io import save_timeseries, save_spectra

__version__ = "0.9.0"
__all__ = ["PINAGMM", "save_timeseries", "save_spectra"]
