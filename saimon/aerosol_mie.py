"""
aerosol_mie.py — PSD+Mie extinction integrator and lookup-table builder.

Implements Wrana et al. (2021) Eq. (2):
    k_ext(λ) = ∫ Q_ext(x, m) · π r² · dN/dr dr   [m^-1]
where x = 2π r / λ.

The key property exploited here: when psd.total_number_density_m3 = 1.0,
k_ext is the per-unit-N0 extinction, so RATIOS k_ext(λ_a)/k_ext(λ_b) are
completely N0-independent (Wrana §3).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np
from .interfaces import LookupTableConfig, ExtinctionRatioLookupTable

logger = logging.getLogger(__name__)

# Default grids used when the corresponding LookupTableConfig field is None.
_DEFAULT_SIGMA_GRID = np.array([1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
_DEFAULT_RMED_GRID_M = np.logspace(np.log10(1e-9), np.log10(1000e-9), 60)    # 1–1000 nm
_DEFAULT_R_INTEGRATION_GRID_M = np.logspace(np.log10(1e-9), np.log10(10e-6), 1500)  # 1 nm–10 µm

from .config import CACHE
_CACHE_DIR_DEFAULT = str(CACHE / "mie")


def _make_cache_key(config, mie_model, refractive_index, psd_factory,
                    sigma_grid, rmed_grid, r_int_grid) -> str:
    h = hashlib.sha256()
    h.update(np.array([config.lambda_ref_m, config.lambda_y_m, config.lambda_x_m],
                      dtype=float).tobytes())
    h.update(np.asarray(sigma_grid, dtype=float).tobytes())
    h.update(np.asarray(rmed_grid, dtype=float).tobytes())
    h.update(np.asarray(r_int_grid, dtype=float).tobytes())
    h.update(type(mie_model).__name__.encode())
    h.update(repr(refractive_index).encode())
    sample_psd = psd_factory(100e-9, 1.5)
    h.update(type(sample_psd).__name__.encode())
    return h.hexdigest()


def mie_extinction_coefficient(
    psd,
    mie_model,
    m: complex,
    wavelength_m: float,
    r_grid_m: np.ndarray,
) -> float:
    """k_ext(λ) = ∫ Q_ext(r, m, λ) · π r² · dN/dr dr   [m^-1].

    x = 2*pi*r/λ; integrates with np.trapz over r_grid_m.
    With psd.total_number_density_m3 = 1.0 this is the per-unit-N0 extinction.
    """
    r = np.asarray(r_grid_m, dtype=float)
    x = 2.0 * np.pi * r / float(wavelength_m)
    q_ext, _, _ = mie_model.efficiencies(x, m)
    integrand = q_ext * np.pi * r ** 2 * psd.dn_dr(r)
    return float(np.trapezoid(integrand, r))


def mie_all_coefficients(
    psd,
    mie_model,
    m: complex,
    wavelength_m: float,
    r_grid_m: np.ndarray,
) -> tuple[float, float, float]:
    """Compute (k_ext, k_sca, k_abs) in one pass over the PSD.

    Returns (extinction, scattering, absorption) in m^-1.
    With psd.total_number_density_m3 = 1.0 these are per-unit-N0 cross sections.
    """
    r = np.asarray(r_grid_m, dtype=float)
    x = 2.0 * np.pi * r / float(wavelength_m)
    q_ext, q_sca, q_abs = mie_model.efficiencies(x, m)
    base = np.pi * r ** 2 * psd.dn_dr(r)
    return (
        float(np.trapezoid(q_ext * base, r)),
        float(np.trapezoid(q_sca * base, r)),
        float(np.trapezoid(q_abs * base, r)),
    )


def build_extinction_ratio_lookup_table(
    config: LookupTableConfig = None,
    mie_model=None,
    refractive_index=None,
    psd_factory=None,
    cache_dir: str = _CACHE_DIR_DEFAULT,
    use_cache: bool = True,
    force_recompute: bool = False,
) -> ExtinctionRatioLookupTable:
    """Build the 3-wavelength extinction-ratio lookup table (Wrana Fig 2a data product).

    For each (sigma, rmed) on config's grids, constructs a PSD via psd_factory,
    computes k_ext at lambda_y / lambda_ref / lambda_x using the provided Mie
    backend and refractive index, then stores
        ratio_y = k(lambda_y)  / k(lambda_ref)
        ratio_x = k(lambda_x)  / k(lambda_ref)

    Parameters
    ----------
    config           : LookupTableConfig.  None fields fall back to module defaults.
    mie_model        : MieModelProtocol.   Default: BohrenHuffmanMie().
    refractive_index : RefractiveIndexProtocol.  Default: ConstantRI(1.4, 0.0).
    psd_factory      : callable(rmed_m, sigma) -> PSDProtocol.
                       Default: lambda rmed, sig: LognormalPSD(rmed, sig).
    cache_dir        : Directory for .npz cache files.
    use_cache        : Read from and write to cache when True.
    force_recompute  : Ignore any existing cache and overwrite it.

    Returns
    -------
    ExtinctionRatioLookupTable
        kext shape (n_sigma, n_rmed, 3), columns = (lambda_y, lambda_ref, lambda_x).
    """
    from .mie import BohrenHuffmanMie
    from .refractive_index import ConstantRI
    from .psd import LognormalPSD

    if config is None:
        config = LookupTableConfig()
    if mie_model is None:
        mie_model = BohrenHuffmanMie()
    if refractive_index is None:
        refractive_index = ConstantRI(1.4, 0.0)
    if psd_factory is None:
        psd_factory = lambda rmed, sig: LognormalPSD(rmed, sig)

    sigma_grid = (config.sigma_grid if config.sigma_grid is not None
                  else _DEFAULT_SIGMA_GRID)
    rmed_grid  = (config.rmed_grid_m if config.rmed_grid_m is not None
                  else _DEFAULT_RMED_GRID_M)
    r_int_grid = (config.r_integration_grid_m if config.r_integration_grid_m is not None
                  else _DEFAULT_R_INTEGRATION_GRID_M)

    # --- caching: setup ---
    cache_path = None
    key = None
    if use_cache:
        try:
            key = _make_cache_key(config, mie_model, refractive_index, psd_factory,
                                  sigma_grid, rmed_grid, r_int_grid)
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"lut_{key[:16]}.npz")
        except Exception as exc:
            logger.warning("cache setup error (%s) — caching disabled for this run", exc)

    # --- caching: load ---
    if cache_path is not None and not force_recompute and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=False)
            stored_key = str(data['cache_key'])
            if stored_key == key:
                logger.info("cache hit — loading from %s", cache_path)
                filled_config = LookupTableConfig(
                    lambda_ref_m=config.lambda_ref_m,
                    lambda_y_m=config.lambda_y_m,
                    lambda_x_m=config.lambda_x_m,
                    rmed_grid_m=data['rmed_grid_m'],
                    sigma_grid=data['sigma_grid'],
                    r_integration_grid_m=data['r_integration_grid_m'],
                )
                return ExtinctionRatioLookupTable(
                    config=filled_config,
                    ratio_x=data['ratio_x'],
                    ratio_y=data['ratio_y'],
                    kext=data['kext'],
                )
            else:
                logger.warning("cache key mismatch in %s — recomputing", cache_path)
        except Exception as exc:
            logger.warning("cache read error (%s) — recomputing", exc)

    # --- compute ---
    if cache_path is not None:
        logger.info("cache miss — computing and caching to %s", cache_path)

    wavelengths = np.array([config.lambda_y_m, config.lambda_ref_m, config.lambda_x_m])
    m_vals = refractive_index(wavelengths)

    n_sigma = len(sigma_grid)
    n_rmed  = len(rmed_grid)

    kext    = np.empty((n_sigma, n_rmed, 3))
    ratio_x = np.empty((n_sigma, n_rmed))
    ratio_y = np.empty((n_sigma, n_rmed))

    for i, sigma in enumerate(sigma_grid):
        for j, rmed in enumerate(rmed_grid):
            psd = psd_factory(float(rmed), float(sigma))
            for k_idx in range(3):
                kext[i, j, k_idx] = mie_extinction_coefficient(
                    psd, mie_model, complex(m_vals[k_idx]),
                    wavelengths[k_idx], r_int_grid,
                )
            ratio_y[i, j] = kext[i, j, 0] / kext[i, j, 1]
            ratio_x[i, j] = kext[i, j, 2] / kext[i, j, 1]

    # Store the effective grids so callers (e.g. plot functions) can always read them.
    filled_config = LookupTableConfig(
        lambda_ref_m=config.lambda_ref_m,
        lambda_y_m=config.lambda_y_m,
        lambda_x_m=config.lambda_x_m,
        rmed_grid_m=rmed_grid,
        sigma_grid=sigma_grid,
        r_integration_grid_m=r_int_grid,
    )

    # --- caching: save ---
    if cache_path is not None:
        try:
            np.savez_compressed(
                cache_path,
                ratio_x=ratio_x,
                ratio_y=ratio_y,
                kext=kext,
                rmed_grid_m=rmed_grid,
                sigma_grid=sigma_grid,
                r_integration_grid_m=r_int_grid,
                cache_key=np.array(key),
            )
        except Exception as exc:
            logger.warning("cache write error (%s) — continuing without cache", exc)

    return ExtinctionRatioLookupTable(
        config=filled_config, ratio_x=ratio_x, ratio_y=ratio_y, kext=kext,
    )
