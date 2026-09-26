"""
instrument.py — Instrument model: channels, noise, and forward simulation.

Responsibilities
----------------
1. Define standard channel sets (SAGE III heritage, hypothetical future missions)
2. Simulate transmission spectra from an atmospheric state + geometry
3. Generate measurement noise realizations and noise covariance matrices
4. Support multi-satellite constellation configurations

Extensibility hooks
-------------------
- add_channel() lets future agents append channels without touching this file
- NoiseModel is a Protocol so alternative noise models can be swapped in
- ConstellationConfig aggregates multiple InstrumentConfig objects

AGENT TASK: Implement all functions marked TODO.
"""

import numpy as np
from typing import Optional
from .interfaces import (
    Channel, InstrumentConfig, OccultationGeometry,
    AtmosphereProfile, TransmissionSpectrum, AerosolModel
)
from .atmosphere import (
    aerosol_extinction_profile,
    xsec_ozone_serdyuchenko, xsec_no2_vandaele
)


def _rayleigh_cross_section_m2(wavelength_m: float) -> float:
    """
    Bodhaine et al. (1999) Rayleigh cross-section [m^2/molecule].

    Uses Ciddor (1996) refractive index for standard dry air.
    Valid for ~0.2–1.1 µm.
    """
    lam_um = wavelength_m * 1e6
    sigma_um = 1.0 / lam_um
    ns = 1.0 + 1e-8 * (8342.54 + 2406147.0 / (130.0 - sigma_um ** 2)
                        + 15998.0 / (38.9 - sigma_um ** 2))
    Ns = 2.546899e25   # Loschmidt number [m^-3]
    F_K = 1.049        # King factor for dry air
    return (24.0 * np.pi ** 3 * (ns ** 2 - 1) ** 2 * F_K
            / (Ns ** 2 * wavelength_m ** 4 * (ns ** 2 + 2) ** 2))


# ---------------------------------------------------------------------------
# Standard channel sets
# ---------------------------------------------------------------------------

def sage3_iss_channels() -> list[Channel]:
    """
    SAGE III/ISS heritage channel set.
    Wavelengths and SNR from the SAGE III ATBD Table 2.x.

    Aerosol channels: 384, 448, 521, 602, 676, 756, 869, 1021 nm
    Ozone channel:    600 nm region (also used for aerosol)
    NO2 channel:      448 nm (also aerosol)
    """
    return [
        Channel("CH_384", 384e-9,  2e-9,  2000, is_aerosol_channel=True),
        Channel("CH_448", 448e-9,  2e-9,  2000, is_aerosol_channel=True,  is_no2_channel=True),
        Channel("CH_521", 521e-9,  2e-9,  2000, is_aerosol_channel=True),
        Channel("CH_602", 602e-9,  2e-9,  2000, is_aerosol_channel=True,  is_ozone_channel=True),
        Channel("CH_676", 676e-9,  2e-9,  2000, is_aerosol_channel=True),
        Channel("CH_756", 756e-9,  2e-9,  2000, is_aerosol_channel=True),
        Channel("CH_869", 869e-9,  2e-9,  2000, is_aerosol_channel=True),
        Channel("CH_1021",1021e-9, 2e-9,  2000, is_aerosol_channel=True),
        Channel("CH_1550",1550e-9, 2e-9,  2000, is_aerosol_channel=True),
    ]


def minimal_aerosol_channels() -> list[Channel]:
    """Minimal two-channel set for sensitivity studies."""
    return [
        Channel("CH_525", 525e-9, 10e-9, 400, is_aerosol_channel=True),
        Channel("CH_1020",1020e-9,10e-9, 300, is_aerosol_channel=True),
    ]


def future_mission_channels(
    wavelengths_nm: list[float],
    snr: float = 400,
    bandwidth_nm: float = 5.0,
) -> list[Channel]:
    """
    Generate an arbitrary channel set for trade studies.

    Parameters
    ----------
    wavelengths_nm : list of centre wavelengths in nm
    snr            : SNR assumed identical for all channels
    bandwidth_nm   : FWHM bandwidth for all channels
    """
    return [
        Channel(
            name=f"CH_{int(w)}",
            wavelength_m=w * 1e-9,
            bandwidth_m=bandwidth_nm * 1e-9,
            snr_at_toa=snr,
            is_aerosol_channel=True,
        )
        for w in wavelengths_nm
    ]


# ---------------------------------------------------------------------------
# Noise covariance
# ---------------------------------------------------------------------------

def measurement_noise_covariance(
    channels: list[Channel],
    n_tangent: int,
) -> np.ndarray:
    """
    Diagonal noise covariance matrix S_y.

    Shape: (n_tangent * n_channels, n_tangent * n_channels)
    Diagonal entries: (1 / SNR)^2  (fractional transmission noise variance)

    Assumes noise is uncorrelated between tangent altitudes and channels.
    """
    n_ch = len(channels)
    variances = np.array([(1.0 / ch.snr_at_toa) ** 2 for ch in channels])
    diag = np.tile(variances, n_tangent)   # ordering: i*n_ch + j
    return np.diag(diag)


def noise_sigma_matrix(channels: list[Channel], n_tangent: int) -> np.ndarray:
    """
    Theoretical noise standard deviations.

    Returns shape (n_tangent, n_channels) where entry [i, j] = 1/SNR_j.
    """
    sigmas = np.array([1.0 / ch.snr_at_toa for ch in channels])
    return np.tile(sigmas, (n_tangent, 1))


def noise_vector(channels: list[Channel], n_tangent: int,
                 seed: Optional[int] = None) -> np.ndarray:
    """
    Draw a random noise realisation.

    Returns shape (n_tangent, n_channels), units fractional transmission.
    """
    rng = np.random.default_rng(seed)
    sigmas = np.array([1.0 / ch.snr_at_toa for ch in channels])
    return rng.normal(0, sigmas[np.newaxis, :],
                      size=(n_tangent, len(channels)))


# ---------------------------------------------------------------------------
# Forward model: atmosphere + geometry -> transmission
# ---------------------------------------------------------------------------

def simulate_transmission(
    atmosphere: AtmosphereProfile,
    geometry: OccultationGeometry,
    instrument: InstrumentConfig,
    aerosol_model: AerosolModel,
    add_noise: bool = False,
    noise_seed: Optional[int] = None,
) -> TransmissionSpectrum:
    """
    Simulate limb transmission spectra via Beer-Lambert law.

    For each tangent altitude i and channel j:
        tau(i, j) = sum_k  [ext_total(k, j) * path_length(i, k)]
        T(i, j)   = exp(-tau(i, j))

    where ext_total includes aerosol + Rayleigh + (future) gas absorbers.

    Parameters
    ----------
    atmosphere   : AtmosphereProfile
    geometry     : OccultationGeometry
    instrument   : InstrumentConfig
    aerosol_model: AerosolModel
    add_noise    : whether to add photon noise
    noise_seed   : RNG seed for reproducibility

    Returns
    -------
    TransmissionSpectrum
    """
    channels = instrument.channels
    wavelengths = np.array([ch.wavelength_m for ch in channels])
    n_layers = atmosphere.number_density_m3.shape[0]
    n_ch = len(channels)

    ext_aerosol = aerosol_extinction_profile(atmosphere, aerosol_model, wavelengths)
    # shape: (n_layers, n_ch)

    ext_total = ext_aerosol.copy()
    for j, lam in enumerate(wavelengths):
        ext_total[:, j] += _rayleigh_cross_section_m2(lam) * atmosphere.number_density_m3
        ext_total[:, j] += (xsec_ozone_serdyuchenko(np.asarray(lam))
                            * atmosphere.species.get('ozone', 0))
        ext_total[:, j] += (xsec_no2_vandaele(np.asarray(lam))
                            * atmosphere.species.get('no2', 0))

    # tau[i, j] = slant_paths[i, :] @ ext_total[:, j]
    tau = geometry.slant_path_lengths_m @ ext_total   # (n_tangent, n_ch)
    T = np.exp(-tau)

    if add_noise:
        rng = np.random.default_rng(noise_seed)
        sigmas = np.array([1.0 / ch.snr_at_toa for ch in channels])
        epsilon = rng.normal(0.0, sigmas[np.newaxis, :],
                             size=(len(geometry.tangent_altitudes_m), n_ch))
        T = T * np.exp(epsilon)

    return TransmissionSpectrum(
        tangent_altitudes_m=geometry.tangent_altitudes_m,
        wavelengths_m=wavelengths,
        transmission=T,
        noise=noise_sigma_matrix(channels, len(geometry.tangent_altitudes_m)),
    )


# ---------------------------------------------------------------------------
# Constellation support
# ---------------------------------------------------------------------------

class ConstellationConfig:
    """
    Multiple satellites working together.

    For error studies: the combined measurement vector is the concatenation
    of each satellite's TransmissionSpectrum. The retrieval engine treats
    them as a joint inversion (common atmosphere, independent instruments).
    """
    def __init__(self, instruments: list[InstrumentConfig]):
        self.instruments = instruments

    def n_satellites(self) -> int:
        return len(self.instruments)

    def all_channels(self) -> list[Channel]:
        """Flat list of all channels across all satellites."""
        return [ch for inst in self.instruments for ch in inst.channels]
