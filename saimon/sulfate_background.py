"""
sulfate_background.py -- physically based stratospheric sulfuric-acid aerosol background models.

This module provides a flexible PSD + Mie background model for SAGE-style studies.
It is intentionally separate from the existing power-law aerosol profile so that the
old ATBD-derived profile can remain a sanity-check target.

Core idea
---------
For each lognormal mode and wavelength,

    alpha_ext(z, lambda) = N0(z) * integral Q_ext(r, lambda, m)
                               * pi r^2 * f(r; r_med, sigma) dr

where the integral is evaluated once per mode for N0 = 1 m^-3.  The same PSD
moments give mass concentration.  Diagnostics report:

    * column and global solution mass
    * H2SO4 and sulfur equivalent mass
    * nadir optical depth at each wavelength
    * slant optical depth for a spherical occultation geometry

The default literature cases are deliberately conservative and transparent:
    * Hamill et al. background: r_med ~ 70 nm, N ~ 10 cm^-3, sigma adopted 1.6
    * fixed limb-retrieval standard: r_med = 80 nm, sigma = 1.6
    * Wrana/SAGE-like 20 km median: r_med = 130.6 nm, sigma = 1.54, N = 3.17 cm^-3

The ATBD-calibrated constructors use the ATBD-derived 1020 nm extinction profile
to solve for N0(z).  That keeps the 1020 nm OD fixed and exposes how the PSD/Mie
spectral dependence compares with the older Angstrom-scaled profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .aerosol_mie import mie_extinction_coefficient
from .density import MyhreDensity
from .geometry import R_EARTH_M, build_occultation_geometry, global_shell_volume_integration
from .mie import BohrenHuffmanMie
from .psd import LognormalPSD, effective_radius, lognormal_volume_per_particle, mode_radius
from .refractive_index import ConstantRI

MOLAR_MASS_H2SO4_KG_MOL = 98.079e-3
MOLAR_MASS_SO2_KG_MOL = 64.066e-3
MOLAR_MASS_S_KG_MOL = 32.065e-3

# SAGE III stratospheric lower-bound integration limit (16 km).
# Used as the default for all stratospheric burden and OD integrals to avoid
# tropospheric contamination.
TROPOPAUSE_M: float = 16_000.0


@dataclass(frozen=True)
class LognormalModeProfile:
    """One altitude-dependent monomodal lognormal aerosol population.

    Parameters
    ----------
    name : label used in diagnostics
    rmed_m : number-median radius [m]
    sigma : geometric mode width (>1)
    n0_m3_profile : total particle number density profile [m^-3]
    """

    name: str
    rmed_m: float
    sigma: float
    n0_m3_profile: np.ndarray

    @property
    def reff_m(self) -> float:
        return float(effective_radius(self.rmed_m, self.sigma))

    @property
    def rmode_m(self) -> float:
        return float(mode_radius(self.rmed_m, self.sigma))

    @property
    def volume_per_particle_m3(self) -> float:
        """Lognormal mean particle volume [m^3 particle^-1].

        Delegates to psd.lognormal_volume_per_particle so there is a single
        canonical analytic-moment implementation in the codebase.
        """
        return lognormal_volume_per_particle(self.rmed_m, self.sigma)


@dataclass
class SulfateAerosolColumn:
    """Vertical column of 75 wt% aqueous H2SO4 modes on a SAGE altitude grid."""

    altitude_m: np.ndarray
    modes: list[LognormalModeProfile]
    density_kg_m3: float
    weight_percent_h2so4: float = 75.0
    refractive_index: object = field(default_factory=lambda: ConstantRI(1.43, 0.0))
    name: str = "sulfate_psd_mie"
    notes: str = ""

    def __post_init__(self):
        alt = np.asarray(self.altitude_m, dtype=float)
        if alt.ndim != 1 or len(alt) < 2:
            raise ValueError("altitude_m must be a 1-D array with at least two levels")
        if not np.all(np.diff(alt) > 0):
            raise ValueError("altitude_m must be strictly increasing")
        self.altitude_m = alt
        for mode in self.modes:
            arr = np.asarray(mode.n0_m3_profile, dtype=float)
            if arr.shape != alt.shape:
                raise ValueError(
                    f"mode {mode.name!r} n0_m3_profile has shape {arr.shape}, "
                    f"expected {alt.shape}"
                )

    @property
    def altitude_km(self) -> np.ndarray:
        return self.altitude_m / 1000.0

    @property
    def h2so4_solution_fraction(self) -> float:
        return self.weight_percent_h2so4 / 100.0

    @property
    def sulfur_mass_fraction_of_solution(self) -> float:
        return self.h2so4_solution_fraction * (MOLAR_MASS_S_KG_MOL / MOLAR_MASS_H2SO4_KG_MOL)

    @property
    def h2so4_mass_fraction_of_solution(self) -> float:
        return self.h2so4_solution_fraction

    @property
    def so2_equivalent_mass_fraction_of_solution(self) -> float:
        """Mass fraction of SO2 required to produce the solution [kg SO2 / kg solution].

        Assumes 1:1 molar conversion S -> H2SO4.
        m_SO2 = m_H2SO4 * (M_SO2 / M_H2SO4)
        """
        return self.h2so4_mass_fraction_of_solution * (MOLAR_MASS_SO2_KG_MOL / MOLAR_MASS_H2SO4_KG_MOL)

    def number_density_m3(self) -> np.ndarray:
        out = np.zeros_like(self.altitude_m, dtype=float)
        for mode in self.modes:
            out += np.asarray(mode.n0_m3_profile, dtype=float)
        return out

    def mass_concentration_kg_m3(self) -> np.ndarray:
        """Aqueous-solution aerosol mass concentration [kg m^-3]."""
        out = np.zeros_like(self.altitude_m, dtype=float)
        for mode in self.modes:
            out += np.asarray(mode.n0_m3_profile, dtype=float) * mode.volume_per_particle_m3 * self.density_kg_m3
        return out

    def column_mass_kg_m2(self, alt_min_m: Optional[float] = None, alt_max_m: Optional[float] = None) -> float:
        z, mask = _altitude_mask(self.altitude_m, alt_min_m, alt_max_m)
        return float(np.trapezoid(self.mass_concentration_kg_m3()[mask], z))

    def global_mass_kg(
        self,
        alt_min_m: Optional[float] = None,
        alt_max_m: Optional[float] = None,
        area_m2: Optional[float] = None,
    ) -> float:
        """Global mass [kg] if this 1-D column is interpreted as globally uniform.

        Integrates the mass concentration over global spherical shells.  If
        *area_m2* is provided, it uses the legacy flat-Earth approximation
        (column_mass * area_m2).
        """
        if area_m2 is not None:
            return self.column_mass_kg_m2(alt_min_m, alt_max_m) * area_m2

        # Physical shell integration: Integral[ 4*pi*(R+z)^2 * rho(z) dz ]
        z, mask = _altitude_mask(self.altitude_m, alt_min_m, alt_max_m)
        rho_z = self.mass_concentration_kg_m3()[mask]
        return global_shell_volume_integration(z, rho_z)

    def _mode_cross_sections_m2(
        self,
        wavelengths_m: np.ndarray,
        mie_model=None,
        r_grid_m: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Per-particle extinction cross section for each mode and wavelength."""
        wavelengths_m = np.asarray(wavelengths_m, dtype=float)
        if mie_model is None:
            mie_model = BohrenHuffmanMie()
        if r_grid_m is None:
            r_grid_m = np.logspace(np.log10(1e-9), np.log10(10e-6), 700)
        m_vals = self.refractive_index(wavelengths_m)
        out: dict[str, np.ndarray] = {}
        for mode in self.modes:
            psd = LognormalPSD(mode.rmed_m, mode.sigma, n0_m3=1.0)
            beta = np.empty_like(wavelengths_m, dtype=float)
            for j, lam in enumerate(wavelengths_m):
                beta[j] = mie_extinction_coefficient(
                    psd, mie_model, complex(m_vals[j]), float(lam), r_grid_m
                )
            out[mode.name] = beta
        return out

    def _mode_cross_sections_all_m2(
        self,
        wavelengths_m: np.ndarray,
        mie_model=None,
        r_grid_m: Optional[np.ndarray] = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per-particle (extinction, scattering, absorption) cross sections per mode.

        Returns dict mapping mode name to (beta_ext, beta_sca, beta_abs) arrays.
        """
        from .aerosol_mie import mie_all_coefficients
        wavelengths_m = np.asarray(wavelengths_m, dtype=float)
        if mie_model is None:
            mie_model = BohrenHuffmanMie()
        if r_grid_m is None:
            r_grid_m = np.logspace(np.log10(1e-9), np.log10(10e-6), 700)
        m_vals = self.refractive_index(wavelengths_m)
        out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for mode in self.modes:
            psd = LognormalPSD(mode.rmed_m, mode.sigma, n0_m3=1.0)
            b_ext = np.empty_like(wavelengths_m, dtype=float)
            b_sca = np.empty_like(wavelengths_m, dtype=float)
            b_abs = np.empty_like(wavelengths_m, dtype=float)
            for j, lam in enumerate(wavelengths_m):
                b_ext[j], b_sca[j], b_abs[j] = mie_all_coefficients(
                    psd, mie_model, complex(m_vals[j]), float(lam), r_grid_m
                )
            out[mode.name] = (b_ext, b_sca, b_abs)
        return out

    def extinction_profile_m1(
        self,
        wavelengths_m: Iterable[float],
        mie_model=None,
        r_grid_m: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Aerosol extinction profile [m^-1], shape (n_alt, n_wavelength)."""
        wavelengths_m = np.asarray(list(wavelengths_m), dtype=float)
        ext = np.zeros((len(self.altitude_m), len(wavelengths_m)), dtype=float)
        betas = self._mode_cross_sections_m2(wavelengths_m, mie_model=mie_model, r_grid_m=r_grid_m)
        for mode in self.modes:
            ext += np.asarray(mode.n0_m3_profile, dtype=float)[:, None] * betas[mode.name][None, :]
        return ext

    def scattering_profile_m1(
        self,
        wavelengths_m: Iterable[float],
        mie_model=None,
        r_grid_m: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Aerosol scattering profile [m^-1], shape (n_alt, n_wavelength)."""
        wavelengths_m = np.asarray(list(wavelengths_m), dtype=float)
        sca = np.zeros((len(self.altitude_m), len(wavelengths_m)), dtype=float)
        all_cs = self._mode_cross_sections_all_m2(wavelengths_m, mie_model=mie_model, r_grid_m=r_grid_m)
        for mode in self.modes:
            _, b_sca, _ = all_cs[mode.name]
            sca += np.asarray(mode.n0_m3_profile, dtype=float)[:, None] * b_sca[None, :]
        return sca

    def absorption_profile_m1(
        self,
        wavelengths_m: Iterable[float],
        mie_model=None,
        r_grid_m: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Aerosol absorption profile [m^-1], shape (n_alt, n_wavelength)."""
        wavelengths_m = np.asarray(list(wavelengths_m), dtype=float)
        abs_ = np.zeros((len(self.altitude_m), len(wavelengths_m)), dtype=float)
        all_cs = self._mode_cross_sections_all_m2(wavelengths_m, mie_model=mie_model, r_grid_m=r_grid_m)
        for mode in self.modes:
            _, _, b_abs = all_cs[mode.name]
            abs_ += np.asarray(mode.n0_m3_profile, dtype=float)[:, None] * b_abs[None, :]
        return abs_

    def nadir_od(
        self,
        wavelengths_m: Iterable[float],
        alt_min_m: Optional[float] = None,
        alt_max_m: Optional[float] = None,
        mie_model=None,
        r_grid_m: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vertical/nadir optical depth, integral alpha_ext dz."""
        ext = self.extinction_profile_m1(wavelengths_m, mie_model=mie_model, r_grid_m=r_grid_m)
        z, mask = _altitude_mask(self.altitude_m, alt_min_m, alt_max_m)
        return np.trapezoid(ext[mask, :], z, axis=0)

    def scattering_nadir_od(
        self,
        wavelengths_m: Iterable[float],
        alt_min_m: Optional[float] = None,
        alt_max_m: Optional[float] = None,
        mie_model=None,
        r_grid_m: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Nadir scattering optical depth, integral alpha_sca dz."""
        sca = self.scattering_profile_m1(wavelengths_m, mie_model=mie_model, r_grid_m=r_grid_m)
        z, mask = _altitude_mask(self.altitude_m, alt_min_m, alt_max_m)
        return np.trapezoid(sca[mask, :], z, axis=0)

    def absorption_nadir_od(
        self,
        wavelengths_m: Iterable[float],
        alt_min_m: Optional[float] = None,
        alt_max_m: Optional[float] = None,
        mie_model=None,
        r_grid_m: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Nadir absorption optical depth, integral alpha_abs dz."""
        abs_ = self.absorption_profile_m1(wavelengths_m, mie_model=mie_model, r_grid_m=r_grid_m)
        z, mask = _altitude_mask(self.altitude_m, alt_min_m, alt_max_m)
        return np.trapezoid(abs_[mask, :], z, axis=0)

    def all_nadir_od(
        self,
        wavelengths_m: Iterable[float],
        alt_min_m: Optional[float] = None,
        alt_max_m: Optional[float] = None,
        mie_model=None,
        r_grid_m: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (ext_od, sca_od, abs_od) in a single Mie pass.

        More efficient than calling nadir_od, scattering_nadir_od, and
        absorption_nadir_od separately when all three are needed.
        """
        wavelengths_m = np.asarray(list(wavelengths_m), dtype=float)
        n_alt = len(self.altitude_m)
        n_wl = len(wavelengths_m)
        ext = np.zeros((n_alt, n_wl), dtype=float)
        sca = np.zeros((n_alt, n_wl), dtype=float)
        abs_ = np.zeros((n_alt, n_wl), dtype=float)
        all_cs = self._mode_cross_sections_all_m2(wavelengths_m, mie_model=mie_model, r_grid_m=r_grid_m)
        for mode in self.modes:
            b_ext, b_sca, b_abs = all_cs[mode.name]
            n0 = np.asarray(mode.n0_m3_profile, dtype=float)[:, None]
            ext += n0 * b_ext[None, :]
            sca += n0 * b_sca[None, :]
            abs_ += n0 * b_abs[None, :]
        z, mask = _altitude_mask(self.altitude_m, alt_min_m, alt_max_m)
        return (
            np.trapezoid(ext[mask, :], z, axis=0),
            np.trapezoid(sca[mask, :], z, axis=0),
            np.trapezoid(abs_[mask, :], z, axis=0),
        )

    def slant_od(
        self,
        wavelengths_m: Iterable[float],
        tangent_altitudes_m: Optional[np.ndarray] = None,
        geometry=None,
        mie_model=None,
        r_grid_m: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spherical limb slant optical depths.

        Returns
        -------
        tangent_altitudes_m, tau_slant
            tau_slant has shape (n_tangent, n_wavelength).
        """
        if geometry is None:
            if tangent_altitudes_m is None:
                tangent_altitudes_m = self.altitude_m[::-1]
            geometry = build_occultation_geometry(np.asarray(tangent_altitudes_m, dtype=float), self.altitude_m)
        ext = self.extinction_profile_m1(wavelengths_m, mie_model=mie_model, r_grid_m=r_grid_m)
        return geometry.tangent_altitudes_m, geometry.slant_path_lengths_m @ ext

    def diagnostics(
        self,
        wavelengths_m: Iterable[float],
        tangent_altitudes_m: Optional[np.ndarray] = None,
        reference_tangent_km: float = 20.0,
        alt_min_m: Optional[float] = TROPOPAUSE_M,
        alt_max_m: Optional[float] = None,
        mie_model=None,
        r_grid_m: Optional[np.ndarray] = None,
    ) -> dict:
        """Return mass, nadir OD and reference slant OD diagnostics.

        Mass and nadir OD are integrated from *alt_min_m* to *alt_max_m* (both
        inclusive).  The default lower bound is ``TROPOPAUSE_M`` (16 km)
        so that all reported quantities refer to the stratosphere only.
        Pass ``alt_min_m=None`` explicitly to integrate over
        the full altitude grid.
        """
        wavelengths_m = np.asarray(list(wavelengths_m), dtype=float)
        if tangent_altitudes_m is None:
            tangent_altitudes_m = self.altitude_m[::-1]
        tangent_altitudes_m = np.asarray(tangent_altitudes_m, dtype=float)
        geometry = build_occultation_geometry(tangent_altitudes_m, self.altitude_m)
        tau_tangent_m, tau_slant = self.slant_od(
            wavelengths_m, geometry=geometry, mie_model=mie_model, r_grid_m=r_grid_m
        )
        idx_ref = int(np.argmin(np.abs(tau_tangent_m / 1000.0 - reference_tangent_km)))
        solution_mass_kg = self.global_mass_kg(alt_min_m=alt_min_m, alt_max_m=alt_max_m)
        return {
            "name": self.name,
            "n_modes": len(self.modes),
            "wavelengths_m": wavelengths_m,
            "column_mass_solution_kg_m2": self.column_mass_kg_m2(alt_min_m=alt_min_m, alt_max_m=alt_max_m),
            "global_mass_solution_kg": solution_mass_kg,
            "global_mass_h2so4_kg": solution_mass_kg * self.h2so4_mass_fraction_of_solution,
            "global_mass_so2_equivalent_kg": solution_mass_kg * self.so2_equivalent_mass_fraction_of_solution,
            "global_mass_sulfur_kg": solution_mass_kg * self.sulfur_mass_fraction_of_solution,
            "nadir_od": self.nadir_od(
                wavelengths_m, alt_min_m=alt_min_m, alt_max_m=alt_max_m, mie_model=mie_model, r_grid_m=r_grid_m
            ),
            "reference_tangent_km": float(tau_tangent_m[idx_ref] / 1000.0),
            "slant_od_at_reference_tangent": tau_slant[idx_ref, :],
            "tangent_altitudes_m": tau_tangent_m,
            "slant_od": tau_slant,
            "number_peak_cm3": float(np.max(self.number_density_m3()) / 1e6),
            "mass_peak_ug_m3": float(np.max(self.mass_concentration_kg_m3()) * 1e9),
        }

    def to_single_mode_aerosol_model(self):
        """Return an existing interfaces.AerosolModel for use with instrument.simulate_transmission.

        This only works for one mode because the current AerosolModel.psd_mie contract
        stores one r_med/sigma/N0 profile.
        """
        if len(self.modes) != 1:
            raise ValueError("The current AerosolModel.psd_mie interface only supports one lognormal mode")
        from .interfaces import AerosolModel

        mode = self.modes[0]
        return AerosolModel(
            mode="psd_mie",
            rmed_m_profile=np.full_like(self.altitude_m, mode.rmed_m, dtype=float),
            sigma_profile=np.full_like(self.altitude_m, mode.sigma, dtype=float),
            n0_m3_profile=np.asarray(mode.n0_m3_profile, dtype=float).copy(),
        )


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def sulfate_solution_density_kg_m3(weight_percent: float = 75.0, temperature_k: float = 215.0) -> float:
    """Density of aqueous sulfuric-acid solution using the project Myhre model."""
    return float(MyhreDensity()(weight_percent, temperature_k))


def gaussian_number_profile_m3(
    altitude_m: np.ndarray,
    n_peak_cm3: float,
    z_peak_km: float = 20.0,
    sigma_z_km: float = 5.0,
    alt_min_km: float = 16.0,
    alt_max_km: float = 35.0,
    floor_cm3: float = 0.0,
    bottom_scale_km: float = 1.0,
) -> np.ndarray:
    """Convenience Junge-layer number-density profile with sub-peak taper.

    Uses a Gaussian fall-off above the peak and a sharp exponential taper
    below the peak to avoid a flat floor or step-function cut at the tropopause.
    """
    z_km = np.asarray(altitude_m, dtype=float) / 1000.0

    # Gaussian part for the upper layer
    n_gauss = n_peak_cm3 * np.exp(-0.5 * ((z_km - z_peak_km) / sigma_z_km) ** 2)

    # Exponential taper part for z < z_peak (matching value at z_peak)
    n_taper = n_peak_cm3 * np.exp((z_km - z_peak_km) / bottom_scale_km)

    n_cm3 = np.where(z_km >= z_peak_km, n_gauss, n_taper)
    n_cm3 += floor_cm3

    # Force zero outside the stratospheric integration window
    n_cm3 = np.where((z_km >= alt_min_km) & (z_km <= alt_max_km), n_cm3, 0.0)
    return n_cm3 * 1e6


def make_single_mode_column(
    altitude_m: np.ndarray,
    rmed_nm: float,
    sigma: float,
    n_peak_cm3: float,
    name: str,
    z_peak_km: float = 20.0,
    sigma_z_km: float = 5.0,
    alt_min_km: float = 10.0,
    alt_max_km: float = 35.0,
    weight_percent_h2so4: float = 75.0,
    temperature_k: float = 215.0,
    refractive_index=None,
    notes: str = "",
) -> SulfateAerosolColumn:
    if refractive_index is None:
        refractive_index = ConstantRI(1.43, 0.0)
    n0 = gaussian_number_profile_m3(
        altitude_m, n_peak_cm3, z_peak_km=z_peak_km, sigma_z_km=sigma_z_km,
        alt_min_km=alt_min_km, alt_max_km=alt_max_km,
    )
    mode = LognormalModeProfile(
        name="mode1",
        rmed_m=float(rmed_nm) * 1e-9,
        sigma=float(sigma),
        n0_m3_profile=n0,
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


def make_atbd_calibrated_single_mode_column(
    altitude_m: np.ndarray,
    ext_ref_m1: np.ndarray,
    ref_wavelength_m: float,
    rmed_nm: float,
    sigma: float,
    name: str,
    weight_percent_h2so4: float = 75.0,
    temperature_k: float = 215.0,
    refractive_index=None,
    mie_model=None,
    r_grid_m: Optional[np.ndarray] = None,
    n0_ceiling_cm3: Optional[float] = None,
    tropopause_m: float = TROPOPAUSE_M,
    notes: str = "",
) -> SulfateAerosolColumn:
    """Solve N0(z) so this PSD+Mie model exactly matches ext_ref at ref_wavelength.

    Parameters
    ----------
    tropopause_m : float
        Altitudes strictly below this value (default ``TROPOPAUSE_M`` = 11 km)
        have their N0 forced to zero.  The ATBD reference extinction profile
        extends to the surface (an artefact of the occultation inversion), so
        without this mask the resulting N0 profile includes a spurious
        tropospheric aerosol layer that inflates the stratospheric mass burden
        by roughly a factor of two.  Pass ``tropopause_m=0.0`` to disable.
    """
    altitude_m = np.asarray(altitude_m, dtype=float)
    ext_ref_m1 = np.asarray(ext_ref_m1, dtype=float)
    if ext_ref_m1.shape != altitude_m.shape:
        raise ValueError("ext_ref_m1 must have the same shape as altitude_m")
    if refractive_index is None:
        refractive_index = ConstantRI(1.43, 0.0)
    if mie_model is None:
        mie_model = BohrenHuffmanMie()
    if r_grid_m is None:
        r_grid_m = np.logspace(np.log10(1e-9), np.log10(10e-6), 900)

    psd = LognormalPSD(float(rmed_nm) * 1e-9, float(sigma), n0_m3=1.0)
    m_val = complex(refractive_index(np.array([float(ref_wavelength_m)]))[0])
    beta_ref_m2 = mie_extinction_coefficient(psd, mie_model, m_val, float(ref_wavelength_m), r_grid_m)
    if beta_ref_m2 <= 0:
        raise ValueError("Mie extinction cross section is non-positive; cannot calibrate N0")
    n0_m3 = np.maximum(ext_ref_m1, 0.0) / beta_ref_m2
    # Zero out tropospheric layers — the reference profile has artefact extinction below
    # the tropopause from the limb/occultation inversion; those layers are not stratospheric.
    if tropopause_m > 0.0:
        n0_m3 = np.where(altitude_m < tropopause_m, 0.0, n0_m3)
    if n0_ceiling_cm3 is not None:
        n0_m3 = np.minimum(n0_m3, float(n0_ceiling_cm3) * 1e6)
    mode = LognormalModeProfile(
        name="mode1",
        rmed_m=float(rmed_nm) * 1e-9,
        sigma=float(sigma),
        n0_m3_profile=n0_m3,
    )
    return SulfateAerosolColumn(
        altitude_m=altitude_m,
        modes=[mode],
        density_kg_m3=sulfate_solution_density_kg_m3(weight_percent_h2so4, temperature_k),
        weight_percent_h2so4=weight_percent_h2so4,
        refractive_index=refractive_index,
        name=name,
        notes=notes,
    )


def literature_background_cases(altitude_m: np.ndarray, refractive_index=None) -> list[SulfateAerosolColumn]:
    """A small set of defensible background sulfate cases for first-pass tests."""
    return [
        make_single_mode_column(
            altitude_m, rmed_nm=70.0, sigma=1.6, n_peak_cm3=10.0,
            name="hamill_70nm_N10_sigma1p6",
            refractive_index=refractive_index,
            alt_min_km=16.0,
            notes="Hamill-style background peak radius and number; sigma=1.6 adopted as a common fixed width.",
        ),
        make_single_mode_column(
            altitude_m, rmed_nm=80.0, sigma=1.6, n_peak_cm3=10.0,
            name="limb_standard_80nm_N10_sigma1p6",
            refractive_index=refractive_index,
            alt_min_km=16.0,
            notes="Fixed 80 nm / sigma 1.6 hydrated sulfate PSD used by several limb-scatter retrievals; N profile chosen for comparison.",
        ),
        make_single_mode_column(
            altitude_m, rmed_nm=130.6, sigma=1.54, n_peak_cm3=3.17,
            name="wrana_20km_130p6nm_N3p17_sigma1p54",
            refractive_index=refractive_index,
            alt_min_km=16.0,
            notes="SAGE/ISS 20 km median values from Wrana et al. used as a Gaussian profile.",
        ),
    ]


# ---------------------------------------------------------------------------
# ATBD helpers
# ---------------------------------------------------------------------------


def build_atbd_reference_profiles(altitude_m: np.ndarray) -> dict[str, np.ndarray]:
    """ATBD-derived reference extinction profiles from the current analysis module.

    Returns 1020 nm extinction and the old Angstrom-scaled 521/756/1544 nm
    profiles used as a sanity-check target.
    """
    from .analysis import _ATBD_ALPHA_EXP, _ATBD_Z_ALPHA_KM, _build_atbd_aerosol_profile_1020

    altitude_m = np.asarray(altitude_m, dtype=float)
    ext_1020 = _build_atbd_aerosol_profile_1020(altitude_m)
    alpha_z = np.interp(altitude_m / 1000.0, _ATBD_Z_ALPHA_KM, _ATBD_ALPHA_EXP)
    wavelengths_nm = np.array([521.0, 756.0, 1020.0, 1544.0])
    ext = {}
    for wl_nm in wavelengths_nm:
        ext[f"ext_{int(round(wl_nm))}_m1"] = ext_1020 * (1020.0 / wl_nm) ** alpha_z
    ext["ext_1020_m1"] = ext_1020
    ext["angstrom_alpha"] = alpha_z
    ext["wavelengths_nm"] = wavelengths_nm
    return ext


def atbd_nadir_od(
    atbd_profiles: dict[str, np.ndarray],
    altitude_m: np.ndarray,
    wavelengths_nm: Iterable[float],
    alt_min_m: Optional[float] = None,
    alt_max_m: Optional[float] = None,
) -> np.ndarray:
    """Nadir (vertical) optical depth from the ATBD extinction profiles.

    Parameters
    ----------
    alt_min_m, alt_max_m : float, optional
        Integration bounds.  Pass ``alt_min_m=TROPOPAUSE_M`` to restrict to the
        stratosphere so that ATBD and model nadir ODs are on the same range.
    """
    altitude_m = np.asarray(altitude_m, dtype=float)
    _, mask = _altitude_mask(altitude_m, alt_min_m, alt_max_m)
    out = []
    for wl_nm in wavelengths_nm:
        key = f"ext_{int(round(wl_nm))}_m1"
        out.append(float(np.trapezoid(atbd_profiles[key][mask], altitude_m[mask])))
    return np.array(out)


def atbd_slant_od(
    atbd_profiles: dict[str, np.ndarray],
    altitude_m: np.ndarray,
    wavelengths_nm: Iterable[float],
    tangent_altitudes_m: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if tangent_altitudes_m is None:
        tangent_altitudes_m = altitude_m[::-1]
    geom = build_occultation_geometry(np.asarray(tangent_altitudes_m, dtype=float), altitude_m)
    ext = np.column_stack([atbd_profiles[f"ext_{int(round(wl_nm))}_m1"] for wl_nm in wavelengths_nm])
    return geom.tangent_altitudes_m, geom.slant_path_lengths_m @ ext


def stratospheric_burden_kg(
    column: "SulfateAerosolColumn",
    tropopause_m: float = TROPOPAUSE_M,
    area_m2: Optional[float] = None,
) -> float:
    """Global stratospheric H2SO4 mass burden [kg].

    Integrates the solution mass concentration from *tropopause_m* upward,
    then multiplies by the H2SO4 weight fraction and the Earth surface area.

    This is the canonical function for reporting stratospheric burdens; using
    it (rather than ``column.global_mass_kg()``) guarantees that:
    * the integration lower bound is explicitly the tropopause, not the surface;
    * the H2SO4 weight fraction (not the full solution mass) is returned.

    Parameters
    ----------
    column : SulfateAerosolColumn
    tropopause_m : float
        Lower altitude boundary [m].  Default ``TROPOPAUSE_M`` = 11 km.
    area_m2 : float, optional
        Global surface area [m^2].  Defaults to 4 pi R_Earth^2.

    Returns
    -------
    float  [kg]
    """
    solution_kg = column.global_mass_kg(alt_min_m=tropopause_m, area_m2=area_m2)
    return solution_kg * column.h2so4_mass_fraction_of_solution


def _altitude_mask(altitude_m: np.ndarray, alt_min_m: Optional[float], alt_max_m: Optional[float]) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones_like(altitude_m, dtype=bool)
    if alt_min_m is not None:
        mask &= altitude_m >= float(alt_min_m)
    if alt_max_m is not None:
        mask &= altitude_m <= float(alt_max_m)
    if not np.any(mask):
        raise ValueError("altitude mask selected no points")
    return altitude_m[mask], mask


def flatten_diagnostics_for_table(diag: dict) -> dict[str, float | str]:
    """Compact one-row representation for pandas/csv outputs."""
    row: dict[str, float | str] = {
        "model": diag["name"],
        "n_modes": diag["n_modes"],
        "column_mass_solution_kg_m2": diag["column_mass_solution_kg_m2"],
        "global_mass_solution_Tg": diag["global_mass_solution_kg"] / 1e9,
        "global_mass_h2so4_Tg": diag["global_mass_h2so4_kg"] / 1e9,
        "global_mass_so2_equivalent_Tg": diag["global_mass_so2_equivalent_kg"] / 1e9,
        "global_mass_sulfur_TgS": diag["global_mass_sulfur_kg"] / 1e9,
        "number_peak_cm3": diag["number_peak_cm3"],
        "mass_peak_ug_m3": diag["mass_peak_ug_m3"],
        "reference_tangent_km": diag["reference_tangent_km"],
    }
    for wl_m, od, sod in zip(
        diag["wavelengths_m"], diag["nadir_od"], diag["slant_od_at_reference_tangent"]
    ):
        wl = int(round(wl_m * 1e9))
        row[f"nadir_od_{wl}nm"] = float(od)
        row[f"slant_od_{wl}nm_at_ref"] = float(sod)
    return row
