"""
atmosphere.py — Atmospheric state providers.

Provides:
  - US Standard Atmosphere 1976 (analytic approximation)
  - User-supplied profile loader (from dict or xarray)
  - Rayleigh scattering cross-section (King factor corrected)
  - Molecular extinction profile builder

Extensibility note
------------------
New gas species (O3, NO2, H2O) are added by:
  1. Adding a cross-section function (see xsec_ozone_serdyuchenko as an example stub)
  2. Calling add_species() on an AtmosphereProfile
  3. The retrieval engine will automatically pick them up if a corresponding
     Channel has the relevant is_*_channel flag set.

AGENT TASK: Implement all functions marked TODO.
"""

import logging
import numpy as np
from .interfaces import AtmosphereProfile, AerosolModel

_log = logging.getLogger(__name__)

# Physical constants
K_BOLTZMANN = 1.380649e-23   # J/K
AVOGADRO    = 6.02214076e23  # mol^-1
R_GAS       = 8.314462       # J/(mol·K)
M_AIR       = 28.9644e-3     # kg/mol  (dry air)
G_0         = 9.80665        # m/s^2


# ---------------------------------------------------------------------------
# US Standard Atmosphere 1976
# ---------------------------------------------------------------------------

def us_standard_atmosphere(altitude_m: np.ndarray) -> AtmosphereProfile:
    """
    Analytic US Standard Atmosphere 1976.

    Valid 0–86 km.  Uses the standard lapse-rate layer definitions.

    Returns an AtmosphereProfile with pressure, temperature, and number density.
    The 'species' dict is empty — call add_rayleigh() and aerosol helpers separately.
    """
    # Layer base altitudes (m), base temperatures (K), lapse rates (K/km)
    layers = [
        (0.0,     288.15, -6.5),
        (11000.0, 216.65,  0.0),
        (20000.0, 216.65,  1.0),
        (32000.0, 228.65,  2.8),
        (47000.0, 270.65,  0.0),
        (51000.0, 270.65, -2.8),
        (71000.0, 214.65, -2.0),
    ]
    # Base pressures computed sequentially from P0 = 101325 Pa
    base_pressures = [101325.0]
    for i in range(1, len(layers)):
        z0, T0, gamma = layers[i - 1]
        z1, T1, _     = layers[i]
        dz = z1 - z0          # metres
        P0 = base_pressures[-1]
        if gamma == 0.0:
            P1 = P0 * np.exp(-G_0 * M_AIR * dz / (R_GAS * T0))
        else:
            gamma_si = gamma / 1000.0  # K/m
            P1 = P0 * (T1 / T0) ** (-G_0 * M_AIR / (R_GAS * gamma_si))
        base_pressures.append(P1)

    altitude_m = np.asarray(altitude_m, dtype=float)
    T = np.empty_like(altitude_m)
    P = np.empty_like(altitude_m)

    # COESA 1976: layers are defined in geopotential altitude H.
    # Convert geometric altitude z to geopotential H before lookups.
    R_EARTH = 6356766.0  # effective radius used in COESA 1976 [m]

    for idx, z in enumerate(altitude_m):
        # Geopotential altitude [m]
        H = R_EARTH * z / (R_EARTH + z)
        # Find which layer H belongs to
        layer_idx = 0
        for i in range(len(layers) - 1, -1, -1):
            if H >= layers[i][0]:
                layer_idx = i
                break
        H0, T0, gamma = layers[layer_idx]
        P0 = base_pressures[layer_idx]
        dH = H - H0
        if gamma == 0.0:
            T[idx] = T0
            P[idx] = P0 * np.exp(-G_0 * M_AIR * dH / (R_GAS * T0))
        else:
            gamma_si = gamma / 1000.0  # K/m
            T[idx] = T0 + gamma_si * dH
            P[idx] = P0 * (T[idx] / T0) ** (-G_0 * M_AIR / (R_GAS * gamma_si))

    nd = P / (K_BOLTZMANN * T)
    profile = AtmosphereProfile(
        altitude_m=altitude_m,
        pressure_pa=P,
        temperature_k=T,
        number_density_m3=nd,
    )
    add_rayleigh_to_profile(profile)
    return profile


def add_rayleigh_to_profile(profile: AtmosphereProfile) -> AtmosphereProfile:
    """
    Compute Rayleigh (molecular) number density and add it as
    profile.species['rayleigh'].  Actually just copies number_density_m3
    since Rayleigh scattering is proportional to total air density.
    The wavelength dependence is handled in the extinction model.
    """
    profile.species['rayleigh'] = profile.number_density_m3.copy()
    return profile


def rayleigh_cross_section_m2(wavelength_m: np.ndarray) -> np.ndarray:
    """
    Rayleigh scattering cross-section per molecule [m^2].

    Uses the Bodhaine et al. (1999) formula with King factor for dry air.

        σ_R(λ) = (24π³ / (λ⁴ N²)) * ((n²-1)/(n²+2))² * F_K

    where n is the air refractive index and F_K is the King factor (~1.049).

    Parameters
    ----------
    wavelength_m : wavelengths in metres

    Returns
    -------
    sigma : cross-sections in m^2
    """
    wavelength_m = np.asarray(wavelength_m, dtype=float)
    lam_um = wavelength_m * 1e6          # convert to micrometres
    sigma_um = 1.0 / lam_um             # wavenumber in um^-1

    # Edlén 1966 refractive index formula (valid ~200–1000 nm; extrapolate beyond)
    sigma2 = sigma_um ** 2
    # Guard against division by zero near resonance denominators
    denom1 = 132.274 - sigma2
    denom2 = 39.32957 - sigma2
    # Clamp denominators to a small positive value to avoid blow-up
    denom1 = np.where(np.abs(denom1) < 1e-6, 1e-6, denom1)
    denom2 = np.where(np.abs(denom2) < 1e-6, 1e-6, denom2)
    n_minus1 = (8060.51 + 2480990.0 / denom1 + 17455.7 / denom2) * 1e-8
    n_s = 1.0 + n_minus1

    # Bodhaine et al. (1999) Loschmidt number at 15°C, 1013.25 hPa
    N_s = 2.546899e25  # m^-3

    # Wavelength-dependent King factor from Bates (1984), Eq. 15 of Bodhaine (1999)
    lam_um2 = lam_um ** 2
    F_N2  = 1.034 + 3.17e-4 / lam_um2
    F_O2  = 1.096 + 1.385e-3 / lam_um2 + 1.448e-4 / lam_um2**2
    F_Ar  = 1.000
    F_CO2 = 1.150
    F_K = (78.084 * F_N2 + 20.946 * F_O2 + 0.934 * F_Ar + 0.036 * F_CO2) / 100.0

    n2 = n_s ** 2
    king_bracket = ((n2 - 1.0) / (n2 + 2.0)) ** 2

    lam_m = wavelength_m
    sigma_R = (24.0 * np.pi**3 / (lam_m**4 * N_s**2)) * king_bracket * F_K
    return sigma_R


# ---------------------------------------------------------------------------
# Aerosol extinction
# ---------------------------------------------------------------------------

def aerosol_extinction_profile(
    profile: AtmosphereProfile,
    model: AerosolModel,
    wavelengths_m: np.ndarray,
) -> np.ndarray:
    """
    Compute aerosol extinction [m^-1] at each altitude and wavelength.

    Mode 'power_law':
        ext(z, λ) = ext_ref(z) * (λ / λ_ref)^(-α)
    where ext_ref(z) is profile.species['aerosol'] and α = model.angstrom_exponent.

    Mode 'psd_mie': NOT YET IMPLEMENTED — raises NotImplementedError.

    Returns
    -------
    ext : (n_alt, n_wavelengths) array [m^-1]
    """
    if model.mode == "power_law":
        ext_ref = profile.species.get('aerosol')
        if ext_ref is None:
            raise ValueError("profile.species['aerosol'] required for power_law mode")
        # Warn if ref_wavelength_m is not close to any wavelength in the array
        if not np.any(np.abs(wavelengths_m - model.ref_wavelength_m) / model.ref_wavelength_m < 0.01):
            _log.warning(
                "ref_wavelength_m=%.3e m is not within 1%% of any wavelength in "
                "wavelengths_m (range %.3e–%.3e m); power-law extrapolation may be "
                "inaccurate.", model.ref_wavelength_m, wavelengths_m.min(), wavelengths_m.max()
            )
        ratio = (wavelengths_m / model.ref_wavelength_m) ** (-model.angstrom_exponent)
        return ext_ref[:, np.newaxis] * ratio[np.newaxis, :]

    elif model.mode == "psd_mie":
        from .aerosol_mie import mie_extinction_coefficient
        from .psd import LognormalPSD
        from .refractive_index import ConstantRI
        from .mie import BohrenHuffmanMie

        if model.rmed_m_profile is None or model.sigma_profile is None:
            raise ValueError(
                "AerosolModel.rmed_m_profile and sigma_profile are required for "
                "psd_mie mode. Set per-altitude lognormal parameters."
            )
        ri   = ConstantRI(1.4, 0.0)
        mie  = BohrenHuffmanMie()
        r_int = np.logspace(np.log10(1e-9), np.log10(10e-6), 500)
        m_vals = ri(wavelengths_m)
        n_alt  = len(profile.altitude_m)
        n_lam  = len(wavelengths_m)
        ext = np.empty((n_alt, n_lam))
        n0_arr = (model.n0_m3_profile if model.n0_m3_profile is not None
                  else np.ones(n_alt))
        for i in range(n_alt):
            psd = LognormalPSD(model.rmed_m_profile[i], model.sigma_profile[i], n0_arr[i])
            for j in range(n_lam):
                ext[i, j] = mie_extinction_coefficient(psd, mie, complex(m_vals[j]),
                                                       wavelengths_m[j], r_int)
        return ext

    else:
        raise ValueError(f"Unknown aerosol model mode: {model.mode}")


# ---------------------------------------------------------------------------
# Gas cross-section stubs (future)
# ---------------------------------------------------------------------------

def xsec_ozone_serdyuchenko(wavelength_m: np.ndarray,
                             temperature_k: float = 230.0) -> np.ndarray:
    """
    Ozone UV/Vis absorption cross-section [m^2/molecule], Serdyuchenko et al.
    (2014) via the MPI-Mainz UV/VIS Atlas.  See sage3.trace_gases for the IR
    (9.6 µm) band and the full-spectrum combiner gas_total_cross_section('o3').
    """
    from .trace_gases import uvvis_cross_section
    return uvvis_cross_section("o3", wavelength_m)


def xsec_no2_vandaele(wavelength_m: np.ndarray,
                      temperature_k: float = 220.0) -> np.ndarray:
    """
    NO2 UV/Vis absorption cross-section [m^2/molecule], Vandaele et al. (1998)
    via the MPI-Mainz UV/VIS Atlas.
    """
    from .trace_gases import uvvis_cross_section
    return uvvis_cross_section("no2", wavelength_m)


def xsec_h2o_hitran(wavelength_m: np.ndarray,
                    temperature_k: float = 250.0,
                    pressure_pa: float = 1000.0) -> np.ndarray:
    """
    H2O absorption cross-section [m^2/molecule] from HITRAN line-by-line (HAPI).
    For the window continuum add sage3.trace_gases.mt_ckd_h2o_continuum.
    """
    from .trace_gases import hitran_cross_section
    return hitran_cross_section("h2o", wavelength_m, temperature_k, pressure_pa)


# ---------------------------------------------------------------------------
# Profile loader for user-supplied data
# ---------------------------------------------------------------------------

def profile_from_dict(data: dict) -> AtmosphereProfile:
    """
    Build an AtmosphereProfile from a plain dict with keys:
        'altitude_m', 'pressure_pa', 'temperature_k'
    and optionally any species arrays under their species name.
    """
    alt = np.asarray(data['altitude_m'])
    prs = np.asarray(data['pressure_pa'])
    tmp = np.asarray(data['temperature_k'])
    nd  = prs / (K_BOLTZMANN * tmp)   # ideal gas
    species = {k: np.asarray(v) for k, v in data.items()
               if k not in ('altitude_m', 'pressure_pa', 'temperature_k')}
    return AtmosphereProfile(
        altitude_m=alt,
        pressure_pa=prs,
        temperature_k=tmp,
        number_density_m3=nd,
        species=species,
    )
