"""
analysis.py — Figure generation, error budget visualization, sensitivity studies.

Reproduces SAGE III ATBD Figure 3.2.3 and extends it to trade studies.

Key functions
-------------
plot_error_budget_fig323()   → recreates Fig 3.2.3 (error vs altitude per channel)
sensitivity_n_channels()     → how does AOD error change with number of channels?
sensitivity_orbit_type()     → ISS vs free-flying SNR impact
sensitivity_constellation()  → N satellites working together
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional

from .interfaces import (
    RetrievalResult, InstrumentConfig, AerosolModel, AtmosphereProfile, Channel,
    TransmissionSpectrum, OccultationGeometry,
)
from .instrument import (
    sage3_iss_channels, future_mission_channels, ConstellationConfig,
    simulate_transmission,
)
from .atmosphere import rayleigh_cross_section_m2, us_standard_atmosphere
from .retrieval import run_retrieval
from .geometry import build_occultation_geometry


# Observed Level-2 fractional extinction uncertainties at 20 km from
# Wrana et al. (2021), Table 1 (channel-averaged values).
WRANA2021_OBS = {
    384: 0.0526, 448: 0.0399, 521: 0.0566, 602: 0.1589,
    676: 0.0907, 756: 0.0319, 869: 0.0397, 1021: 0.0453, 1544: 0.0878,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_geometry(atmosphere: AtmosphereProfile):
    """Build a square occultation geometry from an atmosphere's altitude grid."""
    layer_alts = atmosphere.altitude_m          # bottom-up (ascending)
    tang_alts = layer_alts[::-1]                # top-down for rays
    return build_occultation_geometry(tang_alts, layer_alts)


# ---------------------------------------------------------------------------
# Figure 3.2.3 reproduction
# ---------------------------------------------------------------------------

def plot_error_budget_fig323(
    result: RetrievalResult,
    spectrum: TransmissionSpectrum,
    geometry: OccultationGeometry,
    channel_index: int = 2,              # default: 521 nm (index 2)
    ax: Optional[plt.Axes] = None,
    title: str = "Aerosol Slant Path OD Error",
) -> plt.Figure:
    """
    Reproduce ATBD Figure 3.2.3: error in slant-path aerosol OD (%) vs tangent altitude.

    X-axis: linear, error [%] on slant-path OD
    Y-axis: tangent altitude [km]

    Error components are computed as follows:
      - Measurement noise: sigma_lnT / OD * 100,  sigma_lnT = sigma_T / T
      - Rayleigh: project layer extinction errors onto slant-path OD via path lengths
      - Total: RSS of all components

    Parameters
    ----------
    result       : RetrievalResult from run_retrieval()
    spectrum     : TransmissionSpectrum used in the retrieval
    geometry     : OccultationGeometry used in the retrieval
    channel_index: which channel to plot (default 2 = 521 nm)
    ax           : existing axes or None (creates new figure)
    title        : plot title

    Returns
    -------
    fig : matplotlib Figure
    """
    j = channel_index

    T = np.clip(spectrum.transmission, 1e-10, 1.0)   # (n_tangent, n_channels)
    noise = spectrum.noise                             # (n_tangent, n_channels)
    tang_km = spectrum.tangent_altitudes_m / 1000.0   # (n_tangent,)

    # Slant-path OD; guard against near-zero to avoid exploding % errors
    slant_OD = -np.log(T)                             # (n_tangent, n_channels)
    OD_safe = np.where(slant_OD > 1e-4, slant_OD, np.nan)

    # --- Measurement noise ---
    # sigma_lnT = sigma_T / T  (grows large where T→0)
    sigma_lnT = noise / T
    pct_noise = sigma_lnT / OD_safe * 100             # (n_tangent, n_channels)

    # --- Systematic components: project layer extinction errors → slant OD error ---
    # path[i, k] × ext_err[k] sums to OD error for ray i
    path = geometry.slant_path_lengths_m              # (n_tangent, n_layers)
    budget = result.error_budget

    def _pct(key: str) -> np.ndarray:
        if key not in budget or not np.any(budget[key] > 0):
            return np.zeros_like(pct_noise)
        od_err = path @ budget[key]                   # (n_tangent, n_channels)
        return od_err / OD_safe * 100

    pct_rayleigh = _pct('rayleigh')
    pct_ozone    = _pct('ozone_xsec')
    pct_no2      = _pct('no2_xsec')

    # --- Total (RSS) ---
    pct_total = np.sqrt(pct_noise**2 + pct_rayleigh**2 + pct_ozone**2 + pct_no2**2)

    # --- Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
    else:
        fig = ax.get_figure()

    components = [
        (pct_total[:, j],    'black',  '-',             2.0, 'Total'),
        (pct_noise[:, j],    'blue',   '--',            1.5, 'Measurement Noise'),
        (pct_rayleigh[:, j], 'red',    ':',             1.5, 'Rayleigh/T-P'),
    ]
    if np.any(np.isfinite(pct_ozone[:, j]) & (pct_ozone[:, j] > 0)):
        components.append((pct_ozone[:, j],  'green',  '-.',            1.5, 'Ozone X-sec'))
    if np.any(np.isfinite(pct_no2[:, j]) & (pct_no2[:, j] > 0)):
        components.append((pct_no2[:, j],    'purple', (0, (3,1,1,1)), 1.5, 'NO₂ X-sec'))

    for err, color, ls, lw, label in components:
        mask = np.isfinite(err)
        if np.any(mask):
            ax.plot(err[mask], tang_km[mask], color=color, linestyle=ls,
                    linewidth=lw, label=label)

    ax.set_xlabel('Error in Slant Path OD [%]')
    ax.set_ylabel('Tangent Altitude [km]')
    ax.set_ylim(tang_km.min(), tang_km.max())
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    return fig


def plot_all_channels_error(
    result: RetrievalResult,
    spectrum: TransmissionSpectrum,
    geometry: OccultationGeometry,
    fig: Optional[plt.Figure] = None,
) -> plt.Figure:
    """
    Panel plot: one subplot per channel, each showing error budget vs tangent altitude.
    """
    n_channels = result.aerosol_ext_m1.shape[1]
    ncols = min(3, n_channels)
    nrows = int(np.ceil(n_channels / ncols))

    if fig is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 6 * nrows),
                                 squeeze=False)
    else:
        axes = np.array(fig.axes).reshape(nrows, ncols)

    for idx in range(n_channels):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        plot_error_budget_fig323(result, spectrum, geometry,
                                 channel_index=idx, ax=ax, title=f'Channel {idx}')

    # Hide unused axes
    for idx in range(n_channels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sensitivity studies (the science questions for future missions)
# ---------------------------------------------------------------------------

def sensitivity_n_channels(
    n_channel_list: list,
    atmosphere: AtmosphereProfile,
    aerosol_model: AerosolModel,
    base_snr: float = 400,
    wavelength_range_nm: tuple = (380, 1050),
) -> dict:
    """
    How does AOD error depend on the number of channels?

    For each N in n_channel_list, creates a uniformly-spaced channel set
    across wavelength_range_nm, runs the full retrieval + error budget,
    and records sigma_AOD per channel.

    Returns
    -------
    results : dict with keys
        'n_channels'          : list[int]
        'aod_error_per_channel': list of (n_channels,) arrays
        'mean_aod_error'       : list of scalars (mean over channels)
    """
    geometry = _make_geometry(atmosphere)

    aod_error_per_channel = []
    mean_aod_error = []

    for N in n_channel_list:
        wavelengths_nm = np.linspace(wavelength_range_nm[0], wavelength_range_nm[1], N)
        channels = future_mission_channels(list(wavelengths_nm), snr=base_snr)
        instrument = InstrumentConfig(f"SENS_{N}ch", channels)

        spectrum = simulate_transmission(
            atmosphere, geometry, instrument, aerosol_model, add_noise=False)
        result = run_retrieval(spectrum, geometry, atmosphere, instrument, aerosol_model)

        aod_error_per_channel.append(result.aod_error.copy())
        mean_aod_error.append(float(np.mean(result.aod_error)))

    return {
        'n_channels': list(n_channel_list),
        'aod_error_per_channel': aod_error_per_channel,
        'mean_aod_error': mean_aod_error,
    }


def plot_sensitivity_n_channels(results: dict) -> plt.Figure:
    """
    Plot mean AOD error vs number of channels.

    Parameters
    ----------
    results : output of sensitivity_n_channels()
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(results['n_channels'], results['mean_aod_error'], 'b-o', linewidth=2)
    ax.set_xlabel('Number of Channels')
    ax.set_ylabel('Mean AOD Error')
    ax.set_title('AOD Error vs Number of Channels')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def sensitivity_orbit_type(
    atmosphere: AtmosphereProfile,
    aerosol_model: AerosolModel,
    orbit_types: list = None,
    snr_scale: dict = None,
) -> dict:
    """
    Compare AOD error for ISS vs free-flying orbit.

    snr_scale : dict mapping orbit_type -> SNR multiplier relative to ISS
                default: {"ISS": 1.0, "free_flying": 1.3}

    Returns
    -------
    results : dict with orbit_type keys -> RetrievalResult
    """
    if orbit_types is None:
        orbit_types = ["ISS", "free_flying"]
    if snr_scale is None:
        snr_scale = {"ISS": 1.0, "free_flying": 1.3}

    geometry = _make_geometry(atmosphere)
    base_channels = sage3_iss_channels()
    results = {}

    for orbit_type in orbit_types:
        scale = snr_scale.get(orbit_type, 1.0)
        channels = [
            Channel(
                ch.name, ch.wavelength_m, ch.bandwidth_m,
                ch.snr_at_toa * scale,
                ch.is_aerosol_channel, ch.is_ozone_channel,
                ch.is_no2_channel, ch.is_h2o_channel,
            )
            for ch in base_channels
        ]
        instrument = InstrumentConfig(
            f"SAGE_{orbit_type}", channels, orbit_type=orbit_type)

        spectrum = simulate_transmission(
            atmosphere, geometry, instrument, aerosol_model, add_noise=False)
        result = run_retrieval(spectrum, geometry, atmosphere, instrument, aerosol_model)
        results[orbit_type] = result

    return results


def plot_sensitivity_orbit(results: dict) -> plt.Figure:
    """
    Overlay error budget profiles for each orbit type on the same axes.

    Parameters
    ----------
    results : output of sensitivity_orbit_type()
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    colors = ['blue', 'red', 'green', 'orange']

    for i, (orbit_type, result) in enumerate(results.items()):
        alt_km = result.altitude_m / 1000.0
        total_err = result.error_budget['total'][:, -1]
        ax.plot(total_err, alt_km, color=colors[i % len(colors)],
                linewidth=2, label=orbit_type)

    ax.set_xscale('log')
    ax.set_xlabel('Total AOD Error [m⁻¹]')
    ax.set_ylabel('Altitude [km]')
    ax.set_title('Error Budget by Orbit Type')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    return fig


def sensitivity_constellation(
    n_satellite_list: list,
    atmosphere: AtmosphereProfile,
    aerosol_model: AerosolModel,
    base_snr: float = 400,
) -> dict:
    """
    How does AOD error reduce with N identical satellites observing simultaneously?

    Simple model: N independent measurements reduce noise by sqrt(N).

    Returns
    -------
    results : dict
        'n_satellites' : list[int]
        'aod_error'    : list of (n_channels,) arrays
    """
    geometry = _make_geometry(atmosphere)
    aod_errors = []

    for N in n_satellite_list:
        scaled_snr = base_snr * np.sqrt(N)
        channels = future_mission_channels([525.0, 1020.0], snr=scaled_snr)
        instrument = InstrumentConfig(f"CONST_{N}sat", channels)

        spectrum = simulate_transmission(
            atmosphere, geometry, instrument, aerosol_model, add_noise=False)
        result = run_retrieval(spectrum, geometry, atmosphere, instrument, aerosol_model)
        aod_errors.append(result.aod_error.copy())

    return {
        'n_satellites': list(n_satellite_list),
        'aod_error': aod_errors,
    }


def plot_sensitivity_constellation(results: dict) -> plt.Figure:
    """
    Plot AOD error vs N satellites with theoretical sqrt(N) scaling reference.

    Parameters
    ----------
    results : output of sensitivity_constellation()
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    n_sats = np.array(results['n_satellites'])
    mean_errors = np.array([np.mean(e) for e in results['aod_error']])

    ax.plot(n_sats, mean_errors, 'b-o', linewidth=2, label='Simulated')

    # Theoretical sqrt(N) reference scaled to N=1
    ref = mean_errors[0] / np.sqrt(n_sats)
    ax.plot(n_sats, ref, 'k--', linewidth=1.5, label='1/√N scaling')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Satellites')
    ax.set_ylabel('Mean AOD Error')
    ax.set_title('AOD Error vs Constellation Size')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: run everything and produce a full report figure
# ---------------------------------------------------------------------------

def full_report(
    result: RetrievalResult,
    spectrum: TransmissionSpectrum,
    geometry: OccultationGeometry,
    sensitivity_results: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-panel summary figure:
      Panel A  : Retrieved aerosol extinction profile with error shading
      Panel B  : Error budget (Fig 3.2.3 style) for reference channel
      Panel C  : AOD error vs number of channels (if sensitivity_results given)
      Panel D  : AOD error vs N satellites (if sensitivity_results given)
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ---- Panel A: Retrieved aerosol extinction ± 1σ ----
    ax_a = fig.add_subplot(gs[0, 0])
    alt_km = result.altitude_m / 1000.0
    ext = result.aerosol_ext_m1[:, -1]
    err = result.aerosol_ext_error_m1[:, -1]
    ax_a.fill_betweenx(alt_km, np.maximum(ext - err, 0), ext + err,
                       alpha=0.3, color='blue', label='±1σ')
    ax_a.plot(ext, alt_km, 'b-', linewidth=2, label='Retrieved')
    ax_a.set_xlabel('Aerosol Extinction [m⁻¹]')
    ax_a.set_ylabel('Altitude [km]')
    ax_a.set_title('(A) Retrieved Aerosol Extinction')
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.3)

    # ---- Panel B: Error budget (Fig 3.2.3 style) ----
    ax_b = fig.add_subplot(gs[0, 1])
    plot_error_budget_fig323(result, spectrum, geometry, channel_index=-1, ax=ax_b,
                             title='(B) Error Budget')

    # ---- Panel C: AOD error vs number of channels ----
    ax_c = fig.add_subplot(gs[1, 0])
    sens_n = sensitivity_results.get('n_channels') if sensitivity_results else None
    if sens_n is not None:
        ax_c.plot(sens_n['n_channels'], sens_n['mean_aod_error'], 'b-o', linewidth=2)
        ax_c.set_xlabel('Number of Channels')
        ax_c.set_ylabel('Mean AOD Error')
        ax_c.grid(True, alpha=0.3)
    else:
        ax_c.text(0.5, 0.5, 'not available',
                  ha='center', va='center', transform=ax_c.transAxes, fontsize=12)
    ax_c.set_title('(C) AOD Error vs Channels')

    # ---- Panel D: AOD error vs N satellites ----
    ax_d = fig.add_subplot(gs[1, 1])
    sens_c = sensitivity_results.get('constellation') if sensitivity_results else None
    if sens_c is not None:
        n_sats = np.array(sens_c['n_satellites'])
        mean_errs = np.array([np.mean(e) for e in sens_c['aod_error']])
        ax_d.plot(n_sats, mean_errs, 'r-o', linewidth=2, label='Simulated')
        ref = mean_errs[0] / np.sqrt(n_sats)
        ax_d.plot(n_sats, ref, 'k--', linewidth=1.5, label='1/√N')
        ax_d.set_xscale('log')
        ax_d.set_xlabel('Number of Satellites')
        ax_d.set_ylabel('Mean AOD Error')
        ax_d.legend(fontsize=9)
        ax_d.grid(True, which='both', alpha=0.3)
    else:
        ax_d.text(0.5, 0.5, 'not available',
                  ha='center', va='center', transform=ax_d.transAxes, fontsize=12)
    ax_d.set_title('(D) AOD Error vs Constellation')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ---------------------------------------------------------------------------
# ATBD Figure 3.2.3 exact 4-panel reproduction
# ---------------------------------------------------------------------------

def plot_fig323_4panel(
    result: RetrievalResult,
    spectrum: TransmissionSpectrum,
    geometry: OccultationGeometry,
    atmosphere: AtmosphereProfile,
    instrument: InstrumentConfig,
    aerosol_model: AerosolModel,
    ch_idx_521: int = 2,    # index of 521 nm channel in instrument.channels
    ch_idx_1020: int = 7,   # index of 1020 nm channel in instrument.channels
    atbd_521_data: np.ndarray = None,   # (N,2): columns (error%, altitude_km)
    atbd_1020_data: np.ndarray = None,  # (N,2): columns (error%, altitude_km)
    atbd_od_521: np.ndarray = None,     # (N,2): columns (OD, altitude_km)
    atbd_od_1020: np.ndarray = None,    # (N,2): columns (OD, altitude_km)
) -> plt.Figure:
    """
    Reproduce SAGE III ATBD Figure 3.2.3 as a 4-panel figure.

    (a) Aerosol slant-path OD at 521 nm vs tangent altitude  (log x-axis, 1e-4–10)
    (b) Three error component curves at 521 nm               (linear 0–20%)
    (c) Aerosol slant-path OD at 1020 nm vs tangent altitude (log x-axis, 1e-4–10)
    (d) Three error component curves at 1020 nm              (linear 0–20%)

    Aerosol OD = tau_total - tau_rayleigh
    Components:
#      sigma_noise_pct  = (noise/T) / tau_aer * 100   = 1/(SNR*T) / tau_aer * 100
      sigma_noise_pct = noise[:, j] / tau_aer_safe * 100
      sigma_ray_pct    = sqrt(sum_k[(0.003*sigma_R*n_air[k]*path[i,k])^2]) / tau_aer * 100
      sigma_total_pct  = sqrt(sigma_noise_pct^2 + sigma_ray_pct^2)
    tau_aer clipped to minimum 1e-6 before dividing.
    """
    T = np.clip(spectrum.transmission, 1e-10, 1.0)   # (n_tangent, n_channels)
    noise = spectrum.noise                             # (n_tangent, n_channels)
    tang_km = spectrum.tangent_altitudes_m / 1000.0   # (n_tangent,)
    n_air = atmosphere.number_density_m3               # (n_layers,)
    path = geometry.slant_path_lengths_m               # (n_tangent, n_layers)
    channels = instrument.channels

    # n_air[k] * path[i,k] for every (ray, layer) pair — reused below
    n_air_path = n_air[np.newaxis, :] * path          # (n_tangent, n_layers)
    col_density = n_air_path.sum(axis=1)               # (n_tangent,) total column density

    def _channel_quantities(j: int, tau_aer_override: Optional[np.ndarray] = None):
        lam = channels[j].wavelength_m
        sigma_R = rayleigh_cross_section_m2(np.array([lam]))[0]
        tau_ray = sigma_R * col_density                     # (n_tangent,)

        if tau_aer_override is not None:
            tau_aer = tau_aer_override
        else:
            tau_total = -np.log(T[:, j])                   # (n_tangent,)
            tau_aer = np.maximum(tau_total - tau_ray, 0.0)

        tau_aer_safe = np.maximum(tau_aer, 1e-6)

        sigma_noise_pct = (noise[:, j]) / tau_aer_safe * 100
        sigma_tau_ray   = 0.004 * tau_ray
        sigma_ray_pct   = sigma_tau_ray / tau_aer_safe * 100
        sigma_total_pct = np.sqrt(sigma_noise_pct ** 2 + sigma_ray_pct ** 2)

        return tau_aer, tau_ray, sigma_noise_pct, sigma_ray_pct, sigma_total_pct

    # Altitude-dependent Ångström scaling: compute tau_aer at 521 nm from ext_1020
    ext_1020 = atmosphere.species.get('aerosol', np.zeros_like(n_air))
    if 'angstrom_alpha' in atmosphere.species:
        alpha_z = atmosphere.species['angstrom_alpha']
        ext_521 = ext_1020 * (1020.0 / 521.0) ** alpha_z
    else:
        ext_521 = ext_1020 * (1020.0 / 521.0) ** aerosol_model.angstrom_exponent
    tau_aer_521_scaled = path @ ext_521                     # (n_tangent,)

    tau_521,  tau_ray_521,  pct_noise_521,  pct_ray_521,  pct_total_521  = \
        _channel_quantities(ch_idx_521,  tau_aer_override=tau_aer_521_scaled)
    tau_1020, tau_ray_1020, pct_noise_1020, pct_ray_1020, pct_total_1020 = \
        _channel_quantities(ch_idx_1020)

    # Verification: print error components at tangent altitude closest to 25 km (521 nm)
    idx_25 = int(np.argmin(np.abs(tang_km - 25.0)))
    print(f"\n=== Verification at {tang_km[idx_25]:.1f} km tangent alt, "
          f"{channels[ch_idx_521].wavelength_m*1e9:.0f} nm ===")
    print(f"  tau_aer         = {tau_521[idx_25]:.4f}")
    print(f"  sigma_noise_pct = {pct_noise_521[idx_25]:.3f}%  (expected ~0.32%)")
    print(f"  sigma_ray_pct   = {pct_ray_521[idx_25]:.3f}%  (expected ~0.23%)")
    print(f"  total           = {pct_total_521[idx_25]:.3f}%  (expected ~0.39%)")

    fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(12, 14))
    fig.suptitle('SAGE III ATBD Figure 3.2.3', fontsize=14)

    def _plot_od(ax, tau, title, atbd_od):
        tau_log = np.maximum(tau, 1e-5)
        ax.plot(tau_log, tang_km, color='black', linewidth=2, label='Our profile')
        if atbd_od is not None:
            ax.scatter(atbd_od[:, 0], atbd_od[:, 1], color='red', s=15,
                       zorder=5, label='ATBD')
        ax.set_xscale('log')
        ax.set_xlim(1e-4, 10)
        ax.set_ylim(0, 60)
        ax.set_xlabel('Aerosol Slant Path OD')
        ax.set_ylabel('Tangent Altitude [km]')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)

    def _plot_pct(ax, pct_noise, pct_ray, pct_total, title, atbd_data):
        ax.plot(np.clip(pct_noise, 0, 20), tang_km,
                color='black', linestyle='--', linewidth=1.5, label='Measurement noise')
        ax.plot(np.clip(pct_ray, 0, 20), tang_km,
                color='black', linestyle=':', linewidth=1.5, label='Rayleigh')
        ax.plot(np.clip(pct_total, 0, 20), tang_km,
                color='black', linestyle='-', linewidth=2.0, label='Total')
        if atbd_data is not None:
            ax.scatter(atbd_data[:, 0], atbd_data[:, 1], color='red', s=15,
                       zorder=5, label='ATBD')
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 60)
        ax.set_xlabel('1-sigma Error [%]')
        ax.set_ylabel('Tangent Altitude [km]')
        ax.set_title(title)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

    _plot_od(ax_a, tau_521,  '(a) 521 nm — Aerosol Slant OD',  atbd_od_521)
    _plot_pct(ax_b, pct_noise_521,  pct_ray_521,  pct_total_521,  '(b) 521 nm — OD Error %',  atbd_521_data)
    _plot_od(ax_c, tau_1020, '(c) 1020 nm — Aerosol Slant OD', atbd_od_1020)
    _plot_pct(ax_d, pct_noise_1020, pct_ray_1020, pct_total_1020, '(d) 1020 nm — OD Error %', atbd_1020_data)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Vertical retrieval comparison figure
# ---------------------------------------------------------------------------

def plot_vertical_retrieval_comparison(
    results_dict: dict,
    true_profile: Optional[np.ndarray] = None,
    alt_km: Optional[np.ndarray] = None,
    sigma_delta: Optional[np.ndarray] = None,
    P_matrix: Optional[np.ndarray] = None,
    snr: float = None,
    eps_r: float = None,
) -> plt.Figure:
    """
    4-panel comparison figure for onion-peeling, Chahine, and Twomey retrievals.

    Parameters
    ----------
    results_dict : dict with keys 'onion', 'chahine', 'twomey'
        Each value is a dict with:
            'sigma_ext' : (n_layers,)  retrieved extinction [m^-1]
            'sigma_err' : (n_layers,)  1-sigma error [m^-1]
            'G'         : (n_layers, n_meas)  optional gain matrix for Panel D
    true_profile : (n_layers,) or None
    alt_km       : (n_layers,) altitude in km; if None, uses integer indices
    sigma_delta  : (n_layers,) measurement error — used for analytical reference curve
    P_matrix     : (n_layers, n_layers) path length matrix — used for analytical reference
    """
    methods = ['onion', 'chahine']
    colors = {'onion': 'blue', 'chahine': 'green'}
    linestyles = {'onion': '-', 'chahine': '--'}
    labels = {'onion': 'Onion-peeling', 'chahine': 'Chahine'}

    n_layers = len(results_dict[methods[0]]['sigma_ext'])
    if alt_km is None:
        alt_km = np.arange(n_layers, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # Panel A: Retrieved extinction profiles (log x)
    if true_profile is not None:
        ax_a.plot(np.maximum(true_profile, 1e-15), alt_km,
                  'k-', linewidth=2.5, label='True')
    for m in methods:
        r = results_dict[m]
        ext = r['sigma_ext']
        err = r['sigma_err']
        c, ls = colors[m], linestyles[m]
        ax_a.plot(np.maximum(ext, 1e-15), alt_km,
                  color=c, linestyle=ls, linewidth=1.5, label=labels[m])
        ax_a.fill_betweenx(alt_km,
                            np.maximum(ext - err, 1e-15),
                            np.maximum(ext + err, 1e-15),
                            alpha=0.15, color=c)
    ax_a.set_xscale('log')
    ax_a.set_xlabel('Extinction [m⁻¹]')
    ax_a.set_ylabel('Altitude [km]')
    ax_a.set_title('(A) Retrieved Profiles')
    ax_a.legend(fontsize=9)
    ax_a.grid(True, which='both', alpha=0.3)
    if snr is not None and eps_r is not None:
        ax_a.text(0.02, 0.97, f'SNR = {snr}\nε_R = {eps_r:.1%}',
                  transform=ax_a.transAxes, verticalalignment='top', fontsize=9,
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Analytical reference curve: 1.24 * sigma_delta[k] / P[k,k]
    sigma_err_analytical = None
    if sigma_delta is not None and P_matrix is not None:
        diag_P = np.diag(P_matrix)
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma_err_analytical = np.where(
                diag_P > 0, 1.24 * sigma_delta / diag_P, np.nan)

    # Panel B: 1-sigma errors (log x) + reference lines at 10% / 50% of true peak
    for m in methods:
        r = results_dict[m]
        ax_b.plot(r['sigma_err'], alt_km,
                  color=colors[m], linestyle=linestyles[m],
                  linewidth=1.5, label=labels[m])
    if sigma_err_analytical is not None:
        ax_b.plot(sigma_err_analytical, alt_km,
                  color='black', linestyle='--', linewidth=1.5,
                  label='Analytical (1.24 σ_δ/P_kk)')
    if true_profile is not None:
        peak = np.max(true_profile)
        ax_b.axvline(0.10 * peak, color='gray', linestyle='--',
                     linewidth=0.9, alpha=0.7, label='10% of peak')
        ax_b.axvline(0.50 * peak, color='gray', linestyle=':',
                     linewidth=0.9, alpha=0.7, label='50% of peak')
    ax_b.set_xlim(right=2e-8)
    ax_b.set_xlabel('Error [m⁻¹]')
    ax_b.set_ylabel('Altitude [km]')
    ax_b.set_title('(B) 1-sigma Errors')
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.3)

    # Panel C: Fractional error [%] — the key comparison
    for m in methods:
        r = results_dict[m]
        ext = r['sigma_ext']
        err = r['sigma_err']
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(np.abs(ext) > 1e-20, err / np.abs(ext) * 100.0, np.nan)
        ax_c.plot(frac, alt_km,
                  color=colors[m], linestyle=linestyles[m],
                  linewidth=1.5, label=labels[m])
    if sigma_err_analytical is not None and true_profile is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_analytical = np.where(
                np.abs(true_profile) > 1e-20,
                sigma_err_analytical / np.abs(true_profile) * 100.0,
                np.nan)
        ax_c.plot(frac_analytical, alt_km,
                  color='black', linestyle='--', linewidth=1.5,
                  label='Analytical (1.24 σ_δ/P_kk)')
    ax_c.axvline(5.0,  color='gray', linestyle='--', linewidth=0.9, alpha=0.7, label='5%')
    ax_c.axvline(20.0, color='gray', linestyle=':', linewidth=0.9, alpha=0.7, label='20%')
    ax_c.set_xlim(0, 40)
    ax_c.set_xlabel('Fractional Error [%]')
    ax_c.set_ylabel('Altitude [km]')
    ax_c.set_title('(C) Fractional Error')
    ax_c.legend(fontsize=9)
    ax_c.grid(True, alpha=0.3)

    # Panel D: Onion-peeling smoothing kernels at 15, 25, 35 km
    selected_km = [15, 25, 35]
    alt_colors = {15: 'steelblue', 25: 'darkorange', 35: 'firebrick'}
    G_onion = results_dict.get('onion', {}).get('G', None)

    if G_onion is not None:
        for z_tgt in selected_km:
            k = int(np.argmin(np.abs(alt_km - z_tgt)))
            actual_z = alt_km[k]
            kernel = G_onion[k, :]
            norm = np.max(np.abs(kernel))
            if norm > 0:
                kernel = kernel / norm
            x_axis = alt_km if G_onion.shape[1] == len(alt_km) else np.arange(G_onion.shape[1])
            ax_d.plot(x_axis, kernel,
                      color=alt_colors[z_tgt], linestyle='-', linewidth=1.5,
                      label=f'{actual_z:.0f} km')
        ax_d.axhline(0, color='black', linewidth=0.5)
        ax_d.set_xlabel('Altitude [km]')
        ax_d.set_ylabel('Kernel weight (normalized)')
        ax_d.set_title('(D) Onion-peeling Kernels')
        ax_d.legend(fontsize=8)
        ax_d.grid(True, alpha=0.3)
    else:
        ax_d.text(0.5, 0.5, 'Gain matrix\nnot provided',
                  ha='center', va='center', transform=ax_d.transAxes, fontsize=11)
        ax_d.set_title('(D) Onion-peeling Kernels')

    fig.tight_layout()
    return fig


def plot_lcurve(
    gammas: np.ndarray,
    residuals: np.ndarray,
    roughness: np.ndarray,
    gamma_opt: float,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot the L-curve for Twomey regularization selection.

    Residual norm (x) vs solution roughness (y) in log-log space.
    Marks the selected gamma_opt with a red dot.

    Parameters
    ----------
    gammas      : (n_points,)  regularization parameters tested
    residuals   : (n_points,)  residual norms
    roughness   : (n_points,)  solution roughness norms
    gamma_opt   : float        selected regularization parameter
    ax          : existing Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    ax.loglog(residuals, roughness, 'b-', linewidth=1.5)

    # Mark selected gamma
    corner_idx = int(np.argmin(np.abs(gammas - gamma_opt)))
    ax.loglog(residuals[corner_idx], roughness[corner_idx],
              'ro', markersize=10, label=f'γ_opt = {gamma_opt:.2e}')

    # Annotate a few gamma values along the curve
    for idx in np.linspace(0, len(gammas) - 1, 5, dtype=int):
        ax.annotate(f'{gammas[idx]:.0e}',
                    xy=(residuals[idx], roughness[idx]),
                    fontsize=7, color='gray',
                    xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel('Residual norm ‖Pσ − δ‖_W')
    ax.set_ylabel('Solution roughness ‖Hσ‖^{1/2}')
    ax.set_title('L-curve: Twomey regularization selection')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Extinction profile figure
# ---------------------------------------------------------------------------

def plot_extinction_profiles(
    atmosphere: AtmosphereProfile,
    aerosol_model: AerosolModel,
    instrument: InstrumentConfig,
) -> plt.Figure:
    """
    Single-panel extinction profile figure.

    Shows Rayleigh, aerosol, and total extinction at 521 nm and 1020 nm
    vs altitude on a log x-axis.  Marks the crossover altitude where
    Rayleigh_521 == Aerosol_521.
    """
    alt_km = atmosphere.altitude_m / 1000.0
    n_air  = atmosphere.number_density_m3
    ext_aer_1020 = atmosphere.species.get('aerosol', np.zeros_like(n_air))

    if 'angstrom_alpha' in atmosphere.species:
        alpha_z = atmosphere.species['angstrom_alpha']
    else:
        alpha_z = np.full_like(n_air, float(aerosol_model.angstrom_exponent))
    ext_aer_521 = ext_aer_1020 * (1020.0 / 521.0) ** alpha_z

    sigma_R_521  = rayleigh_cross_section_m2(np.array([521e-9]))[0]
    sigma_R_1020 = rayleigh_cross_section_m2(np.array([1020e-9]))[0]
    ext_ray_521  = sigma_R_521  * n_air
    ext_ray_1020 = sigma_R_1020 * n_air
    ext_tot_521  = ext_ray_521  + ext_aer_521
    ext_tot_1020 = ext_ray_1020 + ext_aer_1020

    fig, ax = plt.subplots(figsize=(8, 10))

    ax.plot(ext_ray_521,                       alt_km, 'b-',  linewidth=2.5, label='Rayleigh 521 nm')
    ax.plot(ext_ray_1020,                      alt_km, 'r-',  linewidth=2.5, label='Rayleigh 1020 nm')
    ax.plot(np.maximum(ext_aer_521,  1e-20),   alt_km, 'b--', linewidth=2.5, label='Aerosol 521 nm')
    ax.plot(np.maximum(ext_aer_1020, 1e-20),   alt_km, 'r--', linewidth=2.5, label='Aerosol 1020 nm')
    ax.plot(ext_tot_521,                        alt_km, 'b:',  linewidth=1.5, label='Total 521 nm')
    ax.plot(ext_tot_1020,                       alt_km, 'r:',  linewidth=1.5, label='Total 1020 nm')

    # Crossover: highest altitude where Rayleigh_521 transitions from < to > Aerosol_521
    diff = ext_ray_521 - ext_aer_521
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[-1]
        t = diff[idx] / (diff[idx] - diff[idx + 1])
        z_cross = alt_km[idx] + t * (alt_km[idx + 1] - alt_km[idx])
        ext_cross = ext_ray_521[idx] * (1 - t) + ext_ray_521[idx + 1] * t
        ax.axhline(z_cross, color='green', linewidth=1.0, linestyle='--', alpha=0.7)
        ax.annotate(f'Rayleigh = Aerosol\n@ {z_cross:.1f} km',
                    xy=(ext_cross, z_cross),
                    xytext=(ext_cross * 8, z_cross + 4),
                    fontsize=9, color='green',
                    arrowprops=dict(arrowstyle='->', color='green', lw=0.8))

    ax.set_xscale('log')
    ax.set_xlim(left=1e-9)
    ax.set_ylim(0, 60)
    ax.set_xlabel('Extinction [m⁻¹]')
    ax.set_ylabel('Altitude [km]')
    ax.set_title('Extinction Profiles')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Wrana et al. (2021) Table 1 comparison
# ---------------------------------------------------------------------------

# Altitude-dependent Angstrom exponent from the ATBD NNLS fit (demo_fig323.py).
# Shared by the profile builder and wrana_table_comparison.
_ATBD_Z_ALPHA_KM = np.array([0, 5, 10, 15, 20, 25, 30, 33, 34, 35, 36, 37, 38, 39, 40, 50, 60], dtype=float)
_ATBD_ALPHA_EXP  = np.array([1.73,1.73,1.73,1.73,1.73,1.73,1.64,1.87,1.97,2.35,2.37,2.38,2.38,2.40,2.40,2.40,2.40])


def _build_atbd_aerosol_profile_1020(alt_m: np.ndarray) -> np.ndarray:
    """
    NNLS fit of the ATBD Fig 3.2.3 aerosol extinction profile at 1020 nm.
    Same digitized data and fitting procedure as demo_fig323.py.
    Returns ext_ref [m^-1], shape (len(alt_m),).
    Raises ImportError if scipy is unavailable, ValueError on fit failure.
    """
    from scipy.optimize import nnls
    from scipy.ndimage import gaussian_filter1d

    z_alpha   = _ATBD_Z_ALPHA_KM
    alpha_pts = _ATBD_ALPHA_EXP

    data_1020 = np.array([
        [1.2881122e-04,4.0436681e+01],[1.5114652e-04,4.0524017e+01],
        [1.7500705e-04,4.0349345e+01],[2.3462288e-04,4.0349345e+01],
        [2.9037750e-04,4.0262009e+01],[3.3621754e-04,4.0000000e+01],
        [4.1611461e-04,3.9912664e+01],[4.8826704e-04,3.9825328e+01],
        [5.9629746e-04,3.9475983e+01],[7.1859052e-04,3.9126638e+01],
        [8.6596432e-04,3.8602620e+01],[1.0861245e-03,3.7816594e+01],
        [1.4368367e-03,3.6943231e+01],[1.8021338e-03,3.5982533e+01],
        [2.1717286e-03,3.5371179e+01],[2.5482967e-03,3.4497817e+01],
        [4.1722505e-03,3.3013100e+01],[5.3031882e-03,3.2227074e+01],
        [7.3996804e-03,3.1179039e+01],[1.0188303e-02,3.0218341e+01],
        [1.3478130e-02,2.9257642e+01],[1.6904770e-02,2.8733624e+01],
        [1.9835982e-02,2.8122271e+01],[2.3275453e-02,2.7423581e+01],
        [2.6949801e-02,2.6288210e+01],[3.2046975e-02,2.5065502e+01],
        [3.5651954e-02,2.4366812e+01],[3.8619403e-02,2.3755459e+01],
        [4.7164052e-02,2.3231441e+01],[5.9154902e-02,2.2620087e+01],
        [6.8493309e-02,2.2096070e+01],[8.0369745e-02,2.1310044e+01],
        [9.4305503e-02,2.0524017e+01],[1.0491397e-01,1.9737991e+01],
        [1.1517086e-01,1.8951965e+01],[1.3514097e-01,1.8253275e+01],
        [1.5647481e-01,1.7554585e+01],[1.6070098e-01,1.6855895e+01],
        [1.5857382e-01,1.6157205e+01],[1.5440359e-01,1.5109170e+01],
        [1.5857382e-01,1.4235808e+01],[1.7407674e-01,1.3187773e+01],
        [1.7177253e-01,1.1266376e+01],[1.6949882e-01,1.0218341e+01],
        [1.8117649e-01,8.6462882e+00],[2.0700089e-01,7.6855895e+00],
        [2.2723827e-01,6.5502183e+00],[2.3967880e-01,5.5895197e+00],
        [2.5280041e-01,4.2794760e+00],[2.5962819e-01,3.2314410e+00],
        [2.6311093e-01,2.1834061e+00],[2.6664038e-01,1.5720524e+00],
        [2.8501066e-01,7.8602620e-01],
    ])
    data_521 = np.array([
        [6.5136416e-04,4.0444444e+01],[8.4198931e-04,4.0296296e+01],
        [1.1907162e-03,4.0296296e+01],[1.6412005e-03,4.0074074e+01],
        [2.0944511e-03,3.9851852e+01],[2.7423766e-03,3.9703704e+01],
        [3.7799016e-03,3.9037037e+01],[5.2099540e-03,3.8222222e+01],
        [6.5639965e-03,3.7481481e+01],[8.3767764e-03,3.6592593e+01],
        [1.1846187e-02,3.5259259e+01],[1.5914160e-02,3.4148148e+01],
        [2.3090518e-02,3.3037037e+01],[3.2653921e-02,3.2074074e+01],
        [4.3867256e-02,3.1037037e+01],[5.7437737e-02,2.9777778e+01],
        [7.6177781e-02,2.8962963e+01],[9.3543689e-02,2.8222222e+01],
        [1.0499817e-01,2.7037037e+01],[1.2406392e-01,2.5925926e+01],
        [1.3572643e-01,2.4740741e+01],[1.5431417e-01,2.3777778e+01],
        [2.1822649e-01,2.2444444e+01],[2.7494253e-01,2.0814815e+01],
        [3.2486707e-01,1.9629630e+01],[3.8385699e-01,1.8444444e+01],
        [4.5355841e-01,1.7555556e+01],[4.5941731e-01,1.5629630e+01],
        [5.2233451e-01,1.1111111e+01],[5.4985131e-01,8.2222222e+00],
        [5.9386821e-01,5.1111111e+00],[6.4140877e-01,1.4074074e+00],
    ])

    alt_max_km = alt_m.max() / 1000.0
    mask_1020 = data_1020[:, 1] <= alt_max_km
    mask_521  = data_521[:,  1] <= alt_max_km
    data_1020 = data_1020[mask_1020]
    data_521  = data_521[mask_521]

    geom_tmp = build_occultation_geometry(alt_m, alt_m)
    n_1020, n_521 = len(data_1020), len(data_521)
    n_all = n_1020 + n_521
    z_all = np.concatenate([data_1020[:, 1], data_521[:, 1]])

    alpha_z_all = np.interp(alt_m / 1000.0, z_alpha, alpha_pts)

    P_combined = np.zeros((n_all, len(alt_m)))
    for k in range(n_1020):
        i = int(np.argmin(np.abs(alt_m - data_1020[k, 1] * 1000.0)))
        P_combined[k, :] = geom_tmp.slant_path_lengths_m[i, :]
    for k in range(n_521):
        i = int(np.argmin(np.abs(alt_m - data_521[k, 1] * 1000.0)))
        P_combined[n_1020 + k, :] = (
            geom_tmp.slant_path_lengths_m[i, :] * (1020.0 / 521.0) ** alpha_z_all
        )

    weights = np.ones(n_all)
    weights[z_all >= 30] = 5.0
    weights[z_all >= 35] = 10.0
    weights[n_1020:] *= 2.0

    ext_sol, _ = nnls(P_combined * weights[:, np.newaxis],
                      np.concatenate([data_1020[:, 0], data_521[:, 0]]) * weights)
    ext_ref = gaussian_filter1d(ext_sol, sigma=3)
    ext_ref = np.maximum(ext_ref, 1e-12)
    ext_ref[alt_m > 45000] = 1e-12
    return ext_ref


def wrana_table_comparison(
    snr: float = 2000,
    eps_R: float = 0.004,
    alpha_1020: float = 5e-8,
    angstrom_exp: float = 2.0,
    z_ref_km: float = 20.0,
    dz_km: float = 0.5,
    aerosol_profile_1020=None,
) -> dict:
    """Compute and print Table 1: predicted vs Wrana et al. (2021) fractional
    extinction errors at 20 km for all nine SAGE III/ISS aerosol channels.

    Returns a dict with keys: wavelengths, alpha_k, tau_ray,
    predicted_pct, observed_pct.
    """
    # Geometric correction factor from the onion-peeling error propagation.
    # Derived as sqrt(1 + (sqrt(3) - 1)^2) = sqrt(5 - 2*sqrt(3)) ≈ 1.2417
    geom_factor = math.sqrt(5.0 - 2.0 * math.sqrt(3.0))

    # Diagonal path length [m] for a tangent ray at z_ref through a dz-thick shell.
    # Formula: P_kk = 2 * sqrt(2 * R_Earth * dz)
    _R_EARTH_M = 6_371_000.0  # m  (task specification)
    P_kk_m = 2.0 * math.sqrt(2.0 * _R_EARTH_M * (dz_km * 1e3))

    # Full limb slant-path geometry: build atmosphere + path-length matrix,
    # then extract the row for tangent altitude z_ref_km.
    alt_m = np.arange(0.0, 40_001.0, dz_km * 1e3)
    atm_col = us_standard_atmosphere(alt_m)
    tang_alts_m = alt_m[::-1]
    geom = build_occultation_geometry(tang_alts_m, alt_m)
    i_ref = int(np.argmin(np.abs(tang_alts_m - z_ref_km * 1e3)))
    path_row = geom.slant_path_lengths_m[i_ref, :]   # (N_layers,) metres
    n_air = atm_col.number_density_m3                 # (N_layers,) m^-3

    # Index into alt_m (ascending) for z_ref_km, and altitude-dependent exponent there.
    i_z_ref = int(np.argmin(np.abs(alt_m - z_ref_km * 1e3)))
    alpha_exp_at_z_ref = float(np.interp(z_ref_km, _ATBD_Z_ALPHA_KM, _ATBD_ALPHA_EXP))

    # Aerosol profile at 1020 nm [m^-1], shape (N_layers,).
    if aerosol_profile_1020 is None:
        try:
            aerosol_profile_1020 = _build_atbd_aerosol_profile_1020(alt_m)
        except Exception as _e:
            print(f"WARNING: could not build ATBD aerosol profile ({_e}); tau_aer = N/A")
            aerosol_profile_1020 = None

    lam_ref_m = 1020e-9  # reference wavelength for Angstrom relation

    channels = sage3_iss_channels()

    wavelengths: list[int] = []
    alpha_k:     list[float] = []
    tau_ray:     list[float] = []
    tau_aer_list: list = []
    sigma_tau_list:   list[float] = []
    sigma_alpha_list: list[float] = []
    predicted_pct: list[float] = []
    observed_pct:  list[float] = []

    for ch in channels:
        lam_m = ch.wavelength_m
        lam_nm_int = round(lam_m * 1e9)

        # Map to nearest WRANA2021_OBS key (all nine channels match within 1 nm).
        wrana_key = min(WRANA2021_OBS, key=lambda k: abs(k - lam_nm_int))

        # Rayleigh cross-section [m²] at this wavelength.
        sigma_R = rayleigh_cross_section_m2(np.array([lam_m]))[0]

        # Rayleigh limb slant-path OD: dot-product of path row with n_air profile.
        tau_R = sigma_R * float(np.dot(path_row, n_air))

        # Aerosol extinction at z_ref at this wavelength.
        # When the ATBD profile is available use its value at z_ref with the
        # altitude-dependent Angstrom exponent; otherwise fall back to the
        # scalar alpha_1020 parameter (overridden when profile is present).
        if aerosol_profile_1020 is not None:
            alpha = (aerosol_profile_1020[i_z_ref]
                     * (lam_m / lam_ref_m) ** (-alpha_exp_at_z_ref))
        else:
            alpha = alpha_1020 * (lam_m / lam_ref_m) ** (-angstrom_exp)

        # Aerosol slant-path OD: scale 1020nm profile to this wavelength, then dot with path.
        if aerosol_profile_1020 is not None:
            alpha_aer_profile = aerosol_profile_1020 * (lam_m / lam_ref_m) ** (-angstrom_exp)
            tau_aer = float(np.dot(path_row, alpha_aer_profile))
        else:
            tau_aer = None

        sigma_tau = math.sqrt((1.0 / snr) ** 2 + (eps_R * tau_R) ** 2)
        sigma_alp = geom_factor * sigma_tau / P_kk_m

        # Predicted fractional extinction error (Eq. frac_error_pred).
        sigma_frac = sigma_alp / alpha

        wavelengths.append(wrana_key)
        alpha_k.append(alpha)
        tau_ray.append(tau_R)
        tau_aer_list.append(tau_aer)
        sigma_tau_list.append(sigma_tau)
        sigma_alpha_list.append(sigma_alp)
        predicted_pct.append(sigma_frac * 100.0)
        observed_pct.append(WRANA2021_OBS[wrana_key] * 100.0)

    # ------------------------------------------------------------------
    # Print formatted table
    # ------------------------------------------------------------------
    hdr = (
        f"{'λ (nm)':>7}  {'α_k (m⁻¹)':>11}  {'τ_Ray':>6}"
        f"  {'Predicted (%)':>13}  {'Observed (%)':>12}  Comment"
    )
    sep = (
        f"{'------':>7}  {'----------':>11}  {'------':>6}"
        f"  {'-------------':>13}  {'------------':>12}  {'----------------------'}"
    )
    print(hdr)
    print(sep)
    for wl, al, tau, pred, obs in zip(
        wavelengths, alpha_k, tau_ray, predicted_pct, observed_pct
    ):
        comment = "Rayleigh-dominated" if eps_R * tau > 1.0 / snr else ""
        print(
            f"{wl:>7}  {al:>11.2e}  {tau:>6.2f}"
            f"  {pred:>13.1f}  {obs:>12.1f}  {comment}"
        )

    # ------------------------------------------------------------------
    # Extended diagnostic table (not for publication)
    # tau_aer requires an aerosol extinction profile in atm_col.species['aerosol'],
    # which is not available here; pass it in to enable that column.
    # ------------------------------------------------------------------
    P_kk_km = P_kk_m / 1000.0
    print()
    print("--- Extended diagnostic table (not for publication) ---")
    ehdr = (
        f"{'λ(nm)':>6}  {'P_kk(km)':>8}  {'alpha_k(m⁻¹)':>12}  {'τ_Ray':>6}"
        f"  {'τ_aer':>6}  {'σ_tau':>8}  {'σ_alpha(m⁻¹)':>12}  {'σ_α/α (%)':>9}"
    )
    esep = (
        f"{'------':>6}  {'--------':>8}  {'------------':>12}  {'------':>6}"
        f"  {'------':>6}  {'--------':>8}  {'------------':>12}  {'---------':>9}"
    )
    print(ehdr)
    print(esep)
    for wl, al, tau_R, ta, st, sa, pred in zip(
        wavelengths, alpha_k, tau_ray, tau_aer_list, sigma_tau_list, sigma_alpha_list, predicted_pct
    ):
        tau_aer_str = f"{ta:>6.3f}" if ta is not None else f"{'N/A':>6}"
        print(
            f"{wl:>6}  {P_kk_km:>8.3f}  {al:>12.3e}  {tau_R:>6.3f}"
            f"  {tau_aer_str}  {st:>8.3e}  {sa:>12.3e}  {pred:>9.3f}"
        )

    return {
        "wavelengths":   wavelengths,
        "alpha_k":       alpha_k,
        "tau_ray":       tau_ray,
        "predicted_pct": predicted_pct,
        "observed_pct":  observed_pct,
    }


if __name__ == "__main__":
    wrana_table_comparison()
