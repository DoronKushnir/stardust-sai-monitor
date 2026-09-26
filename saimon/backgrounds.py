"""
backgrounds.py -- Production stratospheric aerosol background presets.

Maps the three GloSSAC target states (quiet, elevated, post-eruption) to
physically consistent PSD+Mie models using empirical vertical profiles.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from scipy.interpolate import PchipInterpolator

from .refractive_index import ConstantRI, sulfuric_acid_at_temperature
from .sulfate_background import (
    TROPOPAUSE_M, SulfateAerosolColumn, LognormalModeProfile, sulfate_solution_density_kg_m3
)
from .mie import BohrenHuffmanMie
from .psd import LognormalPSD
from .aerosol_mie import mie_extinction_coefficient

logger = logging.getLogger(__name__)

# Empirical validation matrix at 525 nm extracted from GloSSAC
_EMPIRICAL_ALT_KM = np.array([16.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0])

_EMPIRICAL_EXT_525 = {
    "quiet": np.array([
        3.77e-07, 3.32e-07, 2.69e-07, 1.89e-07, 7.55e-08, 1e-9, 1e-10, 1e-12
    ]),
    "elevated": np.array([
        5.60e-07, 1.29e-06, 1.08e-06, 4.50e-07, 9.35e-08, 1e-9, 1e-10, 1e-12
    ]),
    "post_eruption": np.array([
        7.00e-06, 8.75e-06, 1.23e-05, 1.57e-05, 2.06e-06, 1e-8, 1e-10, 1e-12
    ]),
}


def make_empirical_column(
    altitude_m: np.ndarray,
    epoch_name: str,
    rmed_nm: float,
    sigma: float,
    weight_percent_h2so4: float = 75.0,
    temperature_k: float = 215.0,
    refractive_index=None,
    notes: str = "",
) -> SulfateAerosolColumn:
    """Build a SulfateAerosolColumn using an empirically defined extinction profile."""
    if refractive_index is None:
        refractive_index = ConstantRI(1.43, 0.0)
    
    # 1. Spline interpolation of empirical extinction profile
    alt_km = altitude_m / 1000.0
    ext_pts = _EMPIRICAL_EXT_525[epoch_name]
    
    # PchipInterpolator is monotonic and preserves shape without overshoots
    interpolator = PchipInterpolator(_EMPIRICAL_ALT_KM, ext_pts)
    alpha_empirical = interpolator(alt_km)
    
    # Enforce non-negative extinction and force to zero below 16 km
    alpha_empirical = np.maximum(alpha_empirical, 0.0)
    alpha_empirical[alt_km < 16.0] = 0.0

    return make_profile_column(
        altitude_m,
        alpha_empirical,
        rmed_nm,
        sigma,
        weight_percent_h2so4=weight_percent_h2so4,
        temperature_k=temperature_k,
        refractive_index=refractive_index,
        name=epoch_name,
        notes=notes,
    )


def make_profile_column(
    altitude_m: np.ndarray,
    alpha525_m1: np.ndarray,
    rmed_nm: float,
    sigma: float,
    weight_percent_h2so4: float = 75.0,
    temperature_k: float = 215.0,
    refractive_index=None,
    name: str = "profile",
    notes: str = "",
) -> SulfateAerosolColumn:
    """Build a SulfateAerosolColumn from an EXPLICIT 525-nm extinction profile.

    Same number-density mapping as make_empirical_column, but the 525-nm
    extinction profile alpha525_m1 [1/m, on altitude_m] is supplied directly
    instead of being interpolated from the named GloSSAC presets.  Used for
    calibrated backgrounds (e.g., a GloSSAC latitude-bin epoch median) and
    for the components of multi-mode backgrounds, where each mode carries a
    prescribed fraction of the total 525-nm extinction.
    """
    if refractive_index is None:
        refractive_index = ConstantRI(1.43, 0.0)
    alpha525_m1 = np.maximum(np.asarray(alpha525_m1, dtype=float), 0.0)

    # Physical number-density mapping: unit-normalized Mie extinction
    # coefficient at 525 nm
    psd = LognormalPSD(rmed_nm * 1e-9, sigma, n0_m3=1.0)
    mie_model = BohrenHuffmanMie()
    r_grid_m = np.logspace(np.log10(1e-9), np.log10(10e-6), 700)
    m_val = complex(refractive_index(np.array([525e-9]))[0])

    sigma_unit_N0 = mie_extinction_coefficient(psd, mie_model, m_val, 525e-9, r_grid_m)

    # N0(z) = alpha(z) / sigma_unit_N0
    n0_m3_profile = alpha525_m1 / sigma_unit_N0

    mode = LognormalModeProfile(
        name="mode1",
        rmed_m=float(rmed_nm) * 1e-9,
        sigma=float(sigma),
        n0_m3_profile=n0_m3_profile,
    )
    return SulfateAerosolColumn(
        altitude_m=np.asarray(altitude_m, dtype=float),
        modes=[mode],
        density_kg_m3=sulfate_solution_density_kg_m3(weight_percent_h2so4, temperature_k),
        weight_percent_h2so4=weight_percent_h2so4,
        refractive_index=refractive_index,
        name=name,
        notes=notes,
    )


def get_background_presets(altitude_m: np.ndarray) -> Dict[str, SulfateAerosolColumn]:
    """Return the three GloSSAC-anchored background presets.

    Microphysical parameters (r_med, sigma) are fixed to literature values
    representing different volcanic epochs. N0(z) is mapped directly from the
    interpolated GloSSAC 525 nm validation matrix.
    """
    try:
        ri = sulfuric_acid_at_temperature(215.0)
    except Exception as exc:
        logger.warning("Sulfuric acid RI table unavailable (%s); using ConstantRI(1.43, 0)", exc)
        ri = ConstantRI(1.43, 0.0)

    # Literature PSD parameters
    presets_config = {
        "quiet": {
            "rmed_nm": 80.0,
            "sigma": 1.6,
            "notes": "Empirical 'quiet' background; r_med=80nm, sigma=1.6.",
        },
        "elevated": {
            "rmed_nm": 100.0,
            "sigma": 1.6,
            "notes": "Empirical 'elevated' (moderate volcanic); r_med=100nm, sigma=1.6.",
        },
        "post_eruption": {
            "rmed_nm": 250.0,
            "sigma": 1.7,
            "notes": "Empirical 'post-eruption' (Pinatubo-like); r_med=250nm, sigma=1.7.",
        },
    }

    out = {}
    for name, p in presets_config.items():
        col = make_empirical_column(
            altitude_m,
            epoch_name=name,
            rmed_nm=p["rmed_nm"],
            sigma=p["sigma"],
            refractive_index=ri,
            notes=p["notes"],
        )
        out[name] = col
    return out


def verify_background_mass_audit(column: SulfateAerosolColumn):
    """Link-by-link audit of the mass and burden computation stack."""
    # Use 525 nm as the reference wavelength for diagnostics
    diag = column.diagnostics([525e-9])
    mode = column.modes[0]

    v_part = mode.volume_per_particle_m3
    rho_solution = column.density_kg_m3
    wt_h2so4 = column.weight_percent_h2so4

    # Report values
    print(f"\n--- Mass Audit Verification: {column.name} ---")
    print(f"PSD Parameters: r_med = {mode.rmed_m*1e9:.1f} nm, sigma = {mode.sigma:.2f}")
    print(f"Link 1 (Moment): Volume per particle <V> = {v_part:.4e} m^3")
    print(f"Link 2 (Density): Solution density rho_sol = {rho_solution:.1f} kg/m^3")
    print(f"Link 3 (Peak Mass): Peak solution conc. = {diag['mass_peak_ug_m3']:.3f} ug/m^3")
    print(f"Link 4 (H2SO4 fraction): wt% H2SO4 = {wt_h2so4:.1f}%")
    print(f"Link 5 (Integration): Stratospheric lower bound = {TROPOPAUSE_M/1000.0:.1f} km")
    print(f"RESULT: Global H2SO4 mass burden = {diag['global_mass_h2so4_kg']/1e9:.4f} Tg")
    print(f"RESULT: SO2-equivalent burden = {diag['global_mass_so2_equivalent_kg']/1e9:.4f} Tg SO2")
