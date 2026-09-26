"""
retrieval.py — Onion-peeling inversion and error propagation.

This is the scientific core of the system.  It implements:

1. Forward model Jacobian (weighting function matrix K)
       K[i,j] = d(ln T_i) / d(ext_j)   [layer j contribution to ray i]
       In the linear approximation: K = -path_length_matrix

2. Onion-peeling retrieval (top-down, layer by layer)
       At each layer k (top first):
           ext_k = [ln(T_k) - sum_{j>k} K_{kj} * ext_j] / K_{kk}

3. Linear error propagation following SAGE III ATBD Section 3.2:
       For a linear operator x = G y  (where G = K^{-1} for onion-peeling):
           S_x = G S_y G^T          (measurement noise)
       Plus systematic terms from parameter uncertainties:
           S_param = (dx/dp)^2 * sigma_p^2  for each parameter p

4. Error budget decomposition matching Figure 3.2.3:
       - measurement_noise
       - rayleigh (uncertainty in Rayleigh cross-section / T-P profile)
       - ozone_xsec (cross-section uncertainty, future)
       - no2_xsec  (cross-section uncertainty, future)
       - total (RSS of all terms)

Notation follows SAGE III ATBD (Rohen et al.) and Rodgers (2000) "Inverse
Methods for Atmospheric Sounding".

AGENT TASK: Implement all functions marked TODO.
"""

import numpy as np
from typing import Optional
from .interfaces import (
    OccultationGeometry, AtmosphereProfile, TransmissionSpectrum,
    InstrumentConfig, RetrievalResult, AerosolModel
)
from .atmosphere import rayleigh_cross_section_m2, aerosol_extinction_profile


# ---------------------------------------------------------------------------
# Jacobian / weighting function matrix
# ---------------------------------------------------------------------------

def compute_jacobian(geometry: OccultationGeometry) -> np.ndarray:
    """
    Compute the linear Jacobian K.

    In log-transmission space:
        K[i, j] = -path_length[i, j]   [m]

    This is exact for Beer-Lambert (no multiple scattering).

    Returns
    -------
    K : (n_tangent, n_layers) array  [m]
    """
    return -geometry.slant_path_lengths_m.copy()


# ---------------------------------------------------------------------------
# Onion-peeling retrieval
# ---------------------------------------------------------------------------

def onion_peel_single_channel(
    log_transmission: np.ndarray,
    jacobian: np.ndarray,
    total_extinction_known: np.ndarray,
) -> np.ndarray:
    """
    Classic onion-peeling for a single wavelength channel.

    Assumes tangent altitudes are ordered top-to-bottom (index 0 = highest).
    Layers are ordered top-to-bottom consistently.

    Parameters
    ----------
    log_transmission       : (n_tangent,)  ln(T) observed
    jacobian               : (n_tangent, n_layers)  K matrix  [m]
    total_extinction_known : (n_layers,)  extinction already accounted for
                             (Rayleigh + known gases); subtract before solving
                             for aerosol.

    Returns
    -------
    aerosol_ext : (n_layers,) retrieved aerosol extinction [m^-1]

    Algorithm
    ---------
    residual[i] = log_T[i] - K[i,:] @ total_extinction_known
    For k = 0, 1, ..., n_layers-1  (top down):
        ext_aer[k] = (residual[k] - sum_{j<k} K[k,j]*ext_aer[j]) / K[k,k]
    """
    n_layers = len(total_extinction_known)
    residual = log_transmission - jacobian @ total_extinction_known

    aerosol_ext = np.zeros(n_layers)
    for k in range(n_layers):
        K_kk = jacobian[k, k]
        if K_kk == 0.0:
            aerosol_ext[k] = 0.0
            continue
        already_accounted = np.dot(jacobian[k, :k], aerosol_ext[:k])
        aerosol_ext[k] = (residual[k] - already_accounted) / K_kk

    return aerosol_ext


def retrieve_aerosol(
    spectrum: TransmissionSpectrum,
    geometry: OccultationGeometry,
    atmosphere: AtmosphereProfile,
    instrument: InstrumentConfig,
    aerosol_model: AerosolModel,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve aerosol extinction at each altitude and channel.

    Returns
    -------
    ext : (n_layers, n_channels)   retrieved aerosol extinction [m^-1]
    log_T_residual : (n_tangent, n_channels)  post-fit residual (diagnostic)
    """
    K = compute_jacobian(geometry)
    n_layers = len(atmosphere.altitude_m)
    n_channels = len(instrument.channels)

    log_T = np.log(np.clip(spectrum.transmission, 1e-10, 1.0))  # (n_tangent, n_channels)

    # Reorder K to lower-triangular for onion peel: both tangent and layer dims top-down.
    # geometry.slant_path_lengths_m rows follow geometry.tangent_altitudes_m ordering;
    # columns follow atmosphere.altitude_m ordering (ascending from geometry engine).
    tang_td = np.argsort(geometry.tangent_altitudes_m)[::-1]   # top-down ray indices
    layer_td = np.argsort(atmosphere.altitude_m)[::-1]          # top-down layer indices
    K_td = K[np.ix_(tang_td, layer_td)]
    log_T_td = log_T[tang_td, :]

    ext = np.zeros((n_layers, n_channels))
    log_T_residual = np.zeros_like(log_T)

    for j, ch in enumerate(instrument.channels):
        sigma_R = rayleigh_cross_section_m2(np.array([ch.wavelength_m]))[0]
        tec = sigma_R * atmosphere.number_density_m3       # (n_layers,) in atm order
        tec_td = tec[layer_td]                              # reorder to top-down

        ext_td = onion_peel_single_channel(log_T_td[:, j], K_td, tec_td)

        # Map retrieved extinction back to atmosphere's original layer order
        ext_orig = np.empty(n_layers)
        ext_orig[layer_td] = ext_td
        ext[:, j] = ext_orig

        log_T_residual[:, j] = log_T[:, j] - K @ (ext_orig + tec)

    return ext, log_T_residual


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

def propagate_measurement_noise(
    jacobian: np.ndarray,
    noise_fractional: np.ndarray,
) -> np.ndarray:
    """
    Propagate transmission measurement noise into extinction uncertainty.

    Using linear error propagation through onion-peeling:
        The retrieval gain matrix G satisfies ext = G * ln(T)
        For onion-peeling, G is the lower-triangular inverse of K.

        S_ext = G * diag(sigma_lnT^2) * G^T
        sigma_ext[k] = sqrt(S_ext[k,k])

    Parameters
    ----------
    jacobian         : (n_tangent, n_layers)   K matrix [m]
    noise_fractional : (n_tangent,)  1-sigma noise on log-transmission,
                       sigma_lnT = sigma_T / T  (altitude-dependent; grows
                       large where T→0 at low tangent altitudes)

    Returns
    -------
    sigma_ext : (n_layers,)  1-sigma extinction error [m^-1]
    """
    # For onion-peeling, errors propagate top-down.
    # G = inv(K) is lower-triangular; row k of G gives the contribution
    # of each measurement to layer k's extinction.
    # Independent noise: sigma_ext[k]^2 = sum_i (G[k,i] * sigma_lnT[i])^2
    G = np.linalg.inv(jacobian)
    # G[k,i] * sigma_lnT[i] for each i, then sum in quadrature over i<=k
    variance = np.sum((G * noise_fractional[np.newaxis, :]) ** 2, axis=1)
    return np.sqrt(variance)

def propagate_rayleigh_uncertainty(
    atmosphere: AtmosphereProfile,
    geometry: OccultationGeometry,
    wavelength_m: float,
    relative_uncertainty: float = 0.003,
) -> np.ndarray:
    from .atmosphere import rayleigh_cross_section_m2
    sigma_R = rayleigh_cross_section_m2(np.array([wavelength_m]))[0]
    ext_ray = sigma_R * atmosphere.number_density_m3
    # The aerosol error from Rayleigh uncertainty at each layer is simply
    # the fractional uncertainty times the Rayleigh extinction at that layer.
    # Cross-layer propagation through onion-peeling is a second-order correction.
    return relative_uncertainty * ext_ray

def propagate_ozone_xsec_uncertainty(
    atmosphere: AtmosphereProfile,
    geometry: OccultationGeometry,
    wavelength_m: float,
    xsec_relative_uncertainty: float = 0.02,  # 2% from ATBD
) -> np.ndarray:
    """
    Error in aerosol extinction due to ozone cross-section uncertainty.
    Stub — returns zeros until ozone retrieval is implemented.

    Returns
    -------
    sigma_ext : (n_layers,)
    """
    n_layers = len(atmosphere.altitude_m)
    return np.zeros(n_layers)


def build_error_budget(
    spectrum: TransmissionSpectrum,
    geometry: OccultationGeometry,
    atmosphere: AtmosphereProfile,
    instrument: InstrumentConfig,
    aerosol_model: AerosolModel,
) -> dict[str, np.ndarray]:
    """
    Compute the full error budget for each channel.

    Returns
    -------
    budget : dict with keys
        'measurement_noise'  : (n_layers, n_channels)
        'rayleigh'           : (n_layers, n_channels)
        'ozone_xsec'         : (n_layers, n_channels)
        'no2_xsec'           : (n_layers, n_channels)  [zeros until implemented]
        'total'              : (n_layers, n_channels)  RSS of all terms
    """
    K = compute_jacobian(geometry)
    n_layers = len(atmosphere.altitude_m)
    n_channels = len(instrument.channels)

    # Reorder to top-down for gain matrix (consistent with onion peel)
    tang_td  = np.argsort(geometry.tangent_altitudes_m)[::-1]
    layer_td = np.argsort(atmosphere.altitude_m)[::-1]
    K_td = K[np.ix_(tang_td, layer_td)]

    noise_arr    = np.zeros((n_layers, n_channels))
    rayleigh_arr = np.zeros((n_layers, n_channels))
    ozone_arr    = np.zeros((n_layers, n_channels))
    no2_arr      = np.zeros((n_layers, n_channels))

    for j, ch in enumerate(instrument.channels):
        # sigma_lnT = sigma_T / T — amplifies where T→0 at low tangent altitudes
        T_td = np.clip(spectrum.transmission[tang_td, j], 1e-10, 1.0)
        noise_td = spectrum.noise[tang_td, j] / T_td
        sigma_td = propagate_measurement_noise(K_td, noise_td)
        noise_orig = np.empty(n_layers)
        noise_orig[layer_td] = sigma_td
        noise_arr[:, j] = noise_orig

        rayleigh_arr[:, j] = propagate_rayleigh_uncertainty(atmosphere, geometry, ch.wavelength_m)
        ozone_arr[:, j]    = propagate_ozone_xsec_uncertainty(atmosphere, geometry, ch.wavelength_m)

    total = np.sqrt(noise_arr**2 + rayleigh_arr**2 + ozone_arr**2 + no2_arr**2)
    return {
        'measurement_noise': noise_arr,
        'rayleigh':          rayleigh_arr,
        'ozone_xsec':        ozone_arr,
        'no2_xsec':          no2_arr,
        'total':             total,
    }


# ---------------------------------------------------------------------------
# Standalone onion-peeling: aerosol OD → extinction + error
# ---------------------------------------------------------------------------

def retrieve_extinction_onion(
    delta_aer: np.ndarray,
    sigma_delta: np.ndarray,
    P: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classic top-down onion-peeling retrieval of vertical extinction profile.

    Parameters
    ----------
    delta_aer   : (n_tangent,)  aerosol slant-path OD, index 0 = top of atmosphere
    sigma_delta : (n_tangent,)  1-sigma error on delta_aer
    P           : (n_tangent, n_layers)  path length matrix [m], lower-triangular
                  P[i,k] = path length of ray i through layer k;
                  P[i,k] = 0 for k > i (ray i only passes layers k <= i).

    Returns
    -------
    sigma_ext : (n_layers,)  extinction coefficient [m^-1]
    sigma_err : (n_layers,)  1-sigma error on sigma_ext

    Algorithm
    ---------
    Top-down forward substitution:
        sigma_ext[i] = (delta_aer[i] - sum_{j<i} P[i,j]*sigma_ext[j]) / P[i,i]

    Error propagation via gain matrix G = inv(P):
        sigma_err[k] = sqrt(sum_i (G[k,i]*sigma_delta[i])^2)
    """
    n = len(delta_aer)
    sigma_ext = np.zeros(n)
    for i in range(n):
        P_ii = P[i, i]
        if P_ii == 0.0:
            continue
        sigma_ext[i] = (delta_aer[i] - np.dot(P[i, :i], sigma_ext[:i])) / P_ii

    G = np.linalg.inv(P)
    sigma_err = np.sqrt(np.sum((G * sigma_delta[np.newaxis, :]) ** 2, axis=1))
    return sigma_ext, sigma_err


def diagonal_error_estimate(
    sigma_delta: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """
    Fast diagonal approximation to the onion-peeling error.

    Ignores off-diagonal terms of G = inv(P):
        sigma_err_diag[k] ≈ sigma_delta[k] / P[k,k]

    Returns
    -------
    sigma_err_diag : (n_layers,)
    """
    diag_P = np.diag(P)
    return np.where(diag_P != 0.0, sigma_delta / diag_P, 0.0)


# ---------------------------------------------------------------------------
# AOD integration
# ---------------------------------------------------------------------------

def integrate_aod(
    extinction_m1: np.ndarray,
    altitude_m: np.ndarray,
    alt_min_m: float = 0.0,
    alt_max_m: float = 40_000.0,
) -> np.ndarray:
    """
    Integrate extinction profile to get aerosol optical depth.

    AOD = integral_{alt_min}^{alt_max}  ext(z) dz

    Uses np.trapezoid.

    Parameters
    ----------
    extinction_m1 : (n_layers, n_channels)
    altitude_m    : (n_layers,)

    Returns
    -------
    aod : (n_channels,)
    """
    mask = (altitude_m >= alt_min_m) & (altitude_m <= alt_max_m)
    return np.trapezoid(extinction_m1[mask, :], altitude_m[mask], axis=0)


def integrate_aod_error(
    sigma_ext_m1: np.ndarray,
    altitude_m: np.ndarray,
    alt_min_m: float = 0.0,
    alt_max_m: float = 40_000.0,
) -> np.ndarray:
    """
    Propagate layer-wise extinction error into AOD error.

    Assumes errors are uncorrelated between layers (conservative for
    onion-peeling which introduces top-down correlation):
        sigma_AOD = sqrt( sum_k (sigma_ext_k * dz_k)^2 )

    Returns
    -------
    sigma_aod : (n_channels,)
    """
    mask = (altitude_m >= alt_min_m) & (altitude_m <= alt_max_m)
    dz = np.gradient(altitude_m[mask])
    return np.sqrt(np.sum((sigma_ext_m1[mask, :] * dz[:, np.newaxis])**2, axis=0))


# ---------------------------------------------------------------------------
# Top-level retrieval orchestrator
# ---------------------------------------------------------------------------

def run_retrieval(
    spectrum: TransmissionSpectrum,
    geometry: OccultationGeometry,
    atmosphere: AtmosphereProfile,
    instrument: InstrumentConfig,
    aerosol_model: AerosolModel,
) -> RetrievalResult:
    """
    Full retrieval pipeline: inversion + error budget + AOD.

    This is the main entry point called by the analysis layer.
    """
    ext, _ = retrieve_aerosol(
        spectrum, geometry, atmosphere, instrument, aerosol_model)

    budget = build_error_budget(
        spectrum, geometry, atmosphere, instrument, aerosol_model)

    aod = integrate_aod(ext, atmosphere.altitude_m)
    aod_error = integrate_aod_error(budget['total'], atmosphere.altitude_m)

    jacobian = compute_jacobian(geometry)

    return RetrievalResult(
        altitude_m=atmosphere.altitude_m,
        aerosol_ext_m1=ext,
        aerosol_ext_error_m1=budget['total'],
        aod=aod,
        aod_error=aod_error,
        error_budget=budget,
        jacobian=jacobian,
    )
