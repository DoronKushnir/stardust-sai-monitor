"""
sai.py -- Stratospheric Aerosol Injection modeling.

Provides models and profiles for artificial injection layers.
"""

from __future__ import annotations

import numpy as np

from .sulfate_background import SulfateAerosolColumn, LognormalModeProfile
from .materials import (SilicaRefractiveIndex, CalciteRefractiveIndex,
                        DolomiteRefractiveIndex, AluminaRefractiveIndex)
from .psd import LognormalPSD
from .aerosol_mie import mie_extinction_coefficient
from .mie import BohrenHuffmanMie
from .atmosphere import us_standard_atmosphere

# Density of solid Silica in kg/m^3
SILICA_DENSITY_KG_M3 = 2200.0

# Density of solid calcite (CaCO3) in kg/m^3 (standard literature value).
CALCITE_DENSITY_KG_M3 = 2710.0

# Density of solid dolomite (CaMg(CO3)2) in kg/m^3 (Handbook of Mineralogy,
# D(meas.) = 2.86 g/cm^3).
DOLOMITE_DENSITY_KG_M3 = 2860.0

# Density of solid alpha-alumina (corundum) in kg/m^3 (standard sapphire value).
ALUMINA_DENSITY_KG_M3 = 3980.0


def _create_material_sai_layer(
    altitude_m: np.ndarray,
    target_mass_tg: float,
    density_kg_m3: float,
    refractive_index,
    rmed_nm: float,
    sigma: float,
    tropopause_km: float,
    top_pressure_hpa: float,
    name: str,
) -> SulfateAerosolColumn:
    """Generic single-material SAI column with a constant mixing ratio relative
    to air from the tropopause up to ``top_pressure_hpa``, scaled to
    ``target_mass_tg``.  Shared by the silica and calcite constructors -- the
    only material-specific inputs are ``density_kg_m3`` and ``refractive_index``.
    """
    atm = us_standard_atmosphere(altitude_m)

    # Identify top altitude based on pressure (30 hPa = 3000 Pa).
    pressure_pa = atm.pressure_pa
    top_idx = np.argmin(np.abs(pressure_pa - (top_pressure_hpa * 100.0)))
    top_altitude_km = altitude_m[top_idx] / 1000.0

    # Constant mixing ratio means N0(z) is proportional to n_air(z).
    alt_km = altitude_m / 1000.0
    mask = (alt_km >= tropopause_km) & (alt_km <= top_altitude_km)

    n0_shape = np.zeros_like(altitude_m, dtype=float)
    n0_shape[mask] = atm.number_density_m3[mask]

    # Temporary column to compute its natural mass, then scale to target.
    mode_tmp = LognormalModeProfile(
        name="sai_mode", rmed_m=rmed_nm * 1e-9, sigma=sigma, n0_m3_profile=n0_shape)
    col_tmp = SulfateAerosolColumn(
        altitude_m=altitude_m, modes=[mode_tmp],
        density_kg_m3=density_kg_m3,
        weight_percent_h2so4=100.0,  # Dummy, since density is fixed
        refractive_index=refractive_index, name=name)

    unscaled_mass_kg = col_tmp.global_mass_kg(alt_min_m=tropopause_km * 1000.0)
    scale_factor = (target_mass_tg * 1e9) / unscaled_mass_kg
    n0_scaled = n0_shape * scale_factor

    mode_final = LognormalModeProfile(
        name="sai_mode", rmed_m=rmed_nm * 1e-9, sigma=sigma, n0_m3_profile=n0_scaled)
    return SulfateAerosolColumn(
        altitude_m=altitude_m, modes=[mode_final],
        density_kg_m3=density_kg_m3, weight_percent_h2so4=100.0,
        refractive_index=refractive_index, name=name)


def create_silica_sai_layer(
    altitude_m: np.ndarray,
    target_mass_tg: float,
    rmed_nm: float = 250.0,
    sigma: float = 1.05,
    tropopause_km: float = 16.0,
    top_pressure_hpa: float = 30.0,
    refractive_index=None,
) -> SulfateAerosolColumn:
    """
    Creates a stratospheric aerosol injection layer of Silica.

    The layer follows a constant mixing ratio relative to atmospheric air
    from the tropopause up to a specified pressure level.

    Parameters
    ----------
    altitude_m : np.ndarray
        Altitude grid in meters.
    target_mass_tg : float
        Total mass of the injected material in Tg.
    rmed_nm : float
        Median radius of the lognormal PSD in nm (default 250 nm).
    sigma : float
        Geometric standard deviation of the lognormal PSD (default 1.05).
    tropopause_km : float
        Lower boundary of the layer in km (default 16.0).
    top_pressure_hpa : float
        Upper boundary of the layer defined by atmospheric pressure in hPa (default 30.0).
    refractive_index : RefractiveIndexProtocol or None
        Silica refractive-index callable. If None (default), uses
        ``SilicaRefractiveIndex()`` from silica_stardust.yml (validity 0.1–20 µm).
        Pass ``SilicaRefractiveIndex('CSVFiles/SiO2_Franta2016.yml')`` for the
        Franta 2016 dataset (0.025–125 µm), needed for analyses past 20 µm.

    Returns
    -------
    SulfateAerosolColumn
        The configured SAI column object.
    """
    if refractive_index is None:
        refractive_index = SilicaRefractiveIndex()
    return _create_material_sai_layer(
        altitude_m=altitude_m, target_mass_tg=target_mass_tg,
        density_kg_m3=SILICA_DENSITY_KG_M3, refractive_index=refractive_index,
        rmed_nm=rmed_nm, sigma=sigma, tropopause_km=tropopause_km,
        top_pressure_hpa=top_pressure_hpa,
        name=f"Silica_SAI_{target_mass_tg}Tg")


def create_calcite_sai_layer(
    altitude_m: np.ndarray,
    target_mass_tg: float,
    rmed_nm: float = 268.0,
    sigma: float = 1.31,
    tropopause_km: float = 16.0,
    top_pressure_hpa: float = 30.0,
    refractive_index=None,
) -> SulfateAerosolColumn:
    """
    Creates a stratospheric aerosol injection layer of calcite (CaCO3).

    Identical constant-mixing-ratio profile and mass scaling as the silica
    layer, but with the calcite solid density (2710 kg/m^3) and, by default,
    the ray-arithmetic-mean calcite refractive index (Lederer 2026 convention)
    built from the ordinary/extraordinary CSV tables in CSVFiles/.

    Parameters mirror ``create_silica_sai_layer``.  The PSD defaults
    (rmed_nm=268, sigma=1.31) reuse the silica Segev-2026 "D300o" fit so the two
    materials are compared at identical size; this is a first-pass assumption,
    not a calcite-specific size distribution.

    Returns
    -------
    SulfateAerosolColumn
        The configured calcite SAI column object.
    """
    if refractive_index is None:
        refractive_index = CalciteRefractiveIndex()
    return _create_material_sai_layer(
        altitude_m=altitude_m, target_mass_tg=target_mass_tg,
        density_kg_m3=CALCITE_DENSITY_KG_M3, refractive_index=refractive_index,
        rmed_nm=rmed_nm, sigma=sigma, tropopause_km=tropopause_km,
        top_pressure_hpa=top_pressure_hpa,
        name=f"Calcite_SAI_{target_mass_tg}Tg")


def create_dolomite_sai_layer(
    altitude_m: np.ndarray,
    target_mass_tg: float,
    rmed_nm: float = 268.0,
    sigma: float = 1.31,
    tropopause_km: float = 16.0,
    top_pressure_hpa: float = 30.0,
    refractive_index=None,
) -> SulfateAerosolColumn:
    """
    Creates a stratospheric aerosol injection layer of dolomite (CaMg(CO3)2).

    Identical constant-mixing-ratio profile and mass scaling as the silica and
    calcite layers, but with the dolomite solid density (2860 kg/m^3) and, by
    default, the ray-arithmetic-mean dolomite refractive index (Querry 1987
    rays) built from the CSV tables in CSVFiles/.

    Parameters mirror ``create_calcite_sai_layer``, including the reused
    silica Segev-2026 "D300o" PSD defaults (rmed_nm=268, sigma=1.31) for an
    apples-to-apples material comparison.
    """
    if refractive_index is None:
        refractive_index = DolomiteRefractiveIndex()
    return _create_material_sai_layer(
        altitude_m=altitude_m, target_mass_tg=target_mass_tg,
        density_kg_m3=DOLOMITE_DENSITY_KG_M3, refractive_index=refractive_index,
        rmed_nm=rmed_nm, sigma=sigma, tropopause_km=tropopause_km,
        top_pressure_hpa=top_pressure_hpa,
        name=f"Dolomite_SAI_{target_mass_tg}Tg")


def create_alumina_sai_layer(
    altitude_m: np.ndarray,
    target_mass_tg: float,
    rmed_nm: float = 268.0,
    sigma: float = 1.31,
    tropopause_km: float = 16.0,
    top_pressure_hpa: float = 30.0,
    refractive_index=None,
) -> SulfateAerosolColumn:
    """
    Creates a stratospheric aerosol injection layer of alumina (alpha-Al2O3).

    Identical constant-mixing-ratio profile and mass scaling as the silica and
    calcite layers, but with the corundum solid density (3980 kg/m^3) and, by
    default, the ray-arithmetic-mean alumina refractive index (Querry 1985
    rays) built from the CSV tables in CSVFiles/.

    Parameters mirror ``create_calcite_sai_layer``, including the reused
    silica Segev-2026 "D300o" PSD defaults (rmed_nm=268, sigma=1.31) for an
    apples-to-apples material comparison.
    """
    if refractive_index is None:
        refractive_index = AluminaRefractiveIndex()
    return _create_material_sai_layer(
        altitude_m=altitude_m, target_mass_tg=target_mass_tg,
        density_kg_m3=ALUMINA_DENSITY_KG_M3, refractive_index=refractive_index,
        rmed_nm=rmed_nm, sigma=sigma, tropopause_km=tropopause_km,
        top_pressure_hpa=top_pressure_hpa,
        name=f"Alumina_SAI_{target_mass_tg}Tg")
