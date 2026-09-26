Element-resolution Voigt cross-section grids computed by `saimon.trace_gases`
from HITRAN2020 line lists (Gordon et al., 2022) for the bands, gases and
(T, P) used by the paper's trace-gas budget steps (`floorw`, `oe`, `snr`,
`limb` in `make_all.py --list`).  File names encode gas, wavenumber window,
temperature, pressure and sampling.  Grids for the two auxiliary checks that
scan the whole spectrum (`cont`, `cfc`) are not shipped (their outputs are
archived); the full chain rebuilds any missing grid on first use once
`hitran-api` is installed (see `fetch/build_hitran_cache.py`).
