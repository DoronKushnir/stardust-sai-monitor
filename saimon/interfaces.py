"""
interfaces.py — Shared dataclasses and protocols for the SAGE III retrieval system.

This file is the CONTRACT that all modules must respect.
No module may change these interfaces without updating all dependents.
All physical units are SI unless explicitly noted in the field docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol
import numpy as np


# ---------------------------------------------------------------------------
# Atmospheric state
# ---------------------------------------------------------------------------

@dataclass
class AtmosphereProfile:
    """
    Vertical profile of the atmospheric state on a fixed altitude grid.

    altitude_m   : 1-D array, metres above geoid, strictly increasing
    pressure_pa  : 1-D array, Pa
    temperature_k: 1-D array, K
    number_density_m3: 1-D array, total air number density (m^-3)
    species      : dict mapping species name -> number density array (m^-3)
                   Keys: 'aerosol', 'ozone', 'no2', 'h2o'  (any subset OK)
    """
    altitude_m: np.ndarray
    pressure_pa: np.ndarray
    temperature_k: np.ndarray
    number_density_m3: np.ndarray
    species: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.altitude_m)
        assert self.pressure_pa.shape == (n,)
        assert self.temperature_k.shape == (n,)
        assert self.number_density_m3.shape == (n,)
        for k, v in self.species.items():
            assert v.shape == (n,), f"species '{k}' shape mismatch"


# ---------------------------------------------------------------------------
# Aerosol optical properties
# ---------------------------------------------------------------------------

@dataclass
class AerosolModel:
    """
    Aerosol extinction model.  Two modes are supported:

    Mode 1 — power-law (initial implementation)
        extinction(λ) = extinction(λ_ref) * (λ / λ_ref)^(-angstrom)

    Mode 2 — PSD + Mie (future)
        Provide r_grid_m and n_r (particle size distribution) plus
        complex refractive index table m_r (real) and m_i (imag) vs wavelength.
    """
    mode: str = "power_law"   # "power_law" | "psd_mie"

    # --- power-law parameters ---
    angstrom_exponent: float = 1.6          # Ångström exponent
    ref_wavelength_m: float = 1020e-9       # reference wavelength [m]
    # extinction at ref wavelength is carried in AtmosphereProfile.species['aerosol']

    # --- PSD + Mie parameters (future) ---
    r_grid_m: Optional[np.ndarray] = None   # particle radii [m]
    n_r: Optional[np.ndarray] = None        # dn/d(ln r), normalised
    m_real: Optional[np.ndarray] = None     # real part of refractive index vs wavelength
    m_imag: Optional[np.ndarray] = None     # imaginary part vs wavelength

    # --- Per-altitude lognormal PSD parameters for psd_mie mode ---
    rmed_m_profile: Optional[np.ndarray] = None   # median radius per altitude [m]
    sigma_profile: Optional[np.ndarray] = None     # mode width per altitude (dimensionless)
    n0_m3_profile: Optional[np.ndarray] = None     # number density per altitude [m^-3]


# ---------------------------------------------------------------------------
# Instrument definition
# ---------------------------------------------------------------------------

@dataclass
class Channel:
    """Single instrument channel."""
    name: str                       # e.g. "CH1_1020nm"
    wavelength_m: float             # centre wavelength [m]
    bandwidth_m: float              # FWHM bandwidth [m]
    snr_at_toa: float               # signal-to-noise ratio at top-of-atmosphere
    is_aerosol_channel: bool = True
    is_ozone_channel: bool = False
    is_no2_channel: bool = False
    is_h2o_channel: bool = False


@dataclass
class InstrumentConfig:
    """
    Full instrument description.

    For multi-satellite studies, create one InstrumentConfig per satellite
    and pass a list to the retrieval orchestrator.
    """
    name: str                           # e.g. "SAGE_III_ISS"
    channels: list[Channel]
    altitude_resolution_m: float = 500.0  # vertical sampling [m]
    # Orbit parameters (for free-floating vs ISS studies)
    orbit_type: str = "ISS"             # "ISS" | "free_flying" | "constellation"
    orbit_altitude_km: float = 400.0
    inclination_deg: float = 51.6


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class OccultationGeometry:
    """
    Geometry for a single solar occultation event.

    tangent_altitudes_m: 1-D array of tangent point altitudes [m],
                         one per measurement (top-down order)
    slant_path_lengths_m: 2-D array [n_tangent x n_atm_layers],
                          path length through each atmospheric layer
                          for each ray (output of geometry engine)
    air_mass_factors:     2-D array [n_tangent x n_atm_layers]
    refraction_correction: bool flag — whether refraction was applied
    """
    tangent_altitudes_m: np.ndarray
    slant_path_lengths_m: np.ndarray      # shape (n_tangent, n_layers)
    air_mass_factors: np.ndarray          # shape (n_tangent, n_layers)
    refraction_correction: bool = False


# ---------------------------------------------------------------------------
# Transmission / Radiance measurement
# ---------------------------------------------------------------------------

@dataclass
class TransmissionSpectrum:
    """
    Observed (or simulated) transmission at each tangent altitude and channel.

    transmission[i, j] = T at tangent altitude i, channel j
    noise[i, j]        = 1-sigma measurement noise on transmission
    """
    tangent_altitudes_m: np.ndarray       # shape (n_tangent,)
    wavelengths_m: np.ndarray             # shape (n_channels,)
    transmission: np.ndarray              # shape (n_tangent, n_channels)
    noise: np.ndarray                     # shape (n_tangent, n_channels)


# ---------------------------------------------------------------------------
# Retrieval result
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    Output of the inversion engine for a single occultation.

    altitude_m          : retrieval altitude grid (m)
    aerosol_ext_m1      : retrieved aerosol extinction profile (m^-1) per channel
    aerosol_ext_error_m1: 1-sigma total error on aerosol extinction (m^-1) per channel
    aod                 : aerosol optical depth (integrated extinction, dimensionless)
    aod_error           : 1-sigma total error on AOD
    error_budget        : dict mapping error-source name -> error profile array
                          Keys: 'measurement_noise', 'rayleigh', 'ozone_xsec',
                                'no2_xsec', 'smoothing', 'total'
    species             : dict mapping gas name -> retrieved profile (optional)
    jacobian            : weighting function matrix (optional, for diagnostics)
    """
    altitude_m: np.ndarray
    aerosol_ext_m1: np.ndarray                     # shape (n_alt, n_channels)
    aerosol_ext_error_m1: np.ndarray               # shape (n_alt, n_channels)
    aod: np.ndarray                                # shape (n_channels,)
    aod_error: np.ndarray                          # shape (n_channels,)
    error_budget: dict[str, np.ndarray] = field(default_factory=dict)
    species: dict[str, np.ndarray] = field(default_factory=dict)
    jacobian: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Protocols (abstract interfaces for dependency injection / testing)
# ---------------------------------------------------------------------------

class ExtinctionModelProtocol(Protocol):
    """Anything that maps wavelength array -> extinction scaling array."""
    def extinction_ratio(self, wavelengths_m: np.ndarray,
                         ref_wavelength_m: float) -> np.ndarray: ...


class AtmosphereProviderProtocol(Protocol):
    """Anything that returns an AtmosphereProfile given a tangent altitude range."""
    def get_profile(self, min_alt_m: float, max_alt_m: float,
                    n_layers: int) -> AtmosphereProfile: ...


# ---------------------------------------------------------------------------
# PSD forward model protocols and containers
# ---------------------------------------------------------------------------

class MieModelProtocol(Protocol):
    """A Mie backend: maps size parameter + refractive index -> efficiencies.

    size_parameter : 1-D array of x = 2*pi*r / lambda  (dimensionless)
    m              : complex refractive index for ONE wavelength (n + i k),
                     with the convention k >= 0 for an absorbing particle.
    Returns (q_ext, q_sca, q_abs), each a 1-D array the same shape as
    size_parameter. q_abs = q_ext - q_sca.
    """
    def efficiencies(self, size_parameter: np.ndarray,
                     m: complex) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


class PSDProtocol(Protocol):
    """A particle size distribution evaluated on a radius grid.

    dn_dr(r_grid_m) returns dN/dr in units of [m^-3 per m] = [m^-4], i.e. the
    number of particles per unit volume per unit radius interval, scaled by the
    distribution's total number density N0 (default N0 = 1.0 m^-3, which is the
    correct choice for EXTINCTION RATIOS where N0 cancels exactly).
    total_number_density_m3 returns N0.
    """
    def dn_dr(self, r_grid_m: np.ndarray) -> np.ndarray: ...
    @property
    def total_number_density_m3(self) -> float: ...


class RefractiveIndexProtocol(Protocol):
    """Maps wavelength(s) -> complex refractive index m = n + i k.

    Constructed with a fixed composition and temperature; calling it with a
    wavelength array returns a complex array of the same shape.
    """
    def __call__(self, wavelength_m: np.ndarray) -> np.ndarray: ...


@dataclass
class LookupTableConfig:
    """Specification of a 3-wavelength extinction-ratio lookup table.

    Wrana et al. (2021) Fig 2a convention:
        ratio_y = k_ext(lambda_y) / k_ext(lambda_ref)   (449 / 756 nm)
        ratio_x = k_ext(lambda_x) / k_ext(lambda_ref)   (1544 / 756 nm)
    """
    lambda_ref_m: float = 755.979e-9      # denominator wavelength (756 nm)
    lambda_y_m: float = 448.511e-9        # y-axis numerator    (449 nm)
    lambda_x_m: float = 1543.92e-9        # x-axis numerator    (1544 nm)
    rmed_grid_m: np.ndarray = None        # 1-D, median radii [m] (e.g. 1..1000 nm)
    sigma_grid: np.ndarray = None         # 1-D, mode widths (e.g. 1.05..2.0)
    r_integration_grid_m: np.ndarray = None  # 1-D radius grid for the Eq.(2) integral


@dataclass
class ExtinctionRatioLookupTable:
    """Result of build_extinction_ratio_lookup_table().

    ratio_x, ratio_y : shape (n_sigma, n_rmed)
    kext             : shape (n_sigma, n_rmed, 3), columns = (y, ref, x)
                       in [m^-1] for N0 = 1 m^-3 (ratios are N0-independent)
    """
    config: LookupTableConfig
    ratio_x: np.ndarray
    ratio_y: np.ndarray
    kext: np.ndarray
