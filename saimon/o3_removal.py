"""
o3_removal.py -- O3-removal error analysis for the silica SAI 8.80 um channel.

At a tangent altitude h_tan the slant optical depth at 8.80 um contains an
aerosol (silica) signal sitting on top of a non-trivial O3 contribution from
the 9.6 um band wing.  To detect SAI we must subtract that O3 using O3
measurements at other wavelengths -- the Chappuis band (~520-680 nm) constrains
the slant column cleanly, and a 9.6 um channel sits in the *same* vibrational
band as 8.80 um, so the cross-section ratio there is much better known than
the absolute IR cross section.  This module quantifies how well that
subtraction can be done and returns a single number sigma(tau_O3 @ 8.80 um)
that drops into the SAI detection-limit error budget.

HITRAN-uncertainty decomposition (the structural reason a 9.6 um channel helps)
-----------------------------------------------------------------------------
HITRAN cross sections at any wavelength carry three structurally different
error sources, which behave differently under wavelength ratios:

  1. Band-coherent line-intensity scaling (~5% typ.)
       Absolute calibration of the integrated band strength against high-
       resolution laboratory transmission spectra.  100% correlated across all
       wavelengths in the same vibrational/electronic band -- so a multiplicative
       error on sigma_O3(9.6 um) and sigma_O3(8.80 um) cancels in the ratio.
       This is the dominant systematic and the lever the 9.6 um channel pulls.

  2. Line-by-line scatter (~1-3% typ., band-averaged)
       Relative line-intensity uncertainty within a band, surviving instrumental
       band-averaging.  Uncorrelated across channels -- contributes to both
       measurement noise and the predictand independently.

  3. Temperature dependence (~3% per 10 K, channel-dependent)
       Lower-state energies are well known but T-extrapolation away from the
       laboratory reference T (~296 K) carries this residual.  Partially
       correlated across nearby wavelengths.

The default split below assigns band-coherent error to a "uv" band (Chappuis-
Wulf electronic transition) and an "ir" band (lumped IR vibrational bands).
A 9.6 um measurement pins down b_ir directly; without it, b_ir is constrained
only by the (loose) HITRAN prior.

Method (Rodgers 2000, Ch. 4)
----------------------------
The state vector is augmented:

    x = (x_O3[0], ..., x_O3[Nz-1],  b_uv, b_ir)

  - x_O3[j] : multiplier on the reference O3 number density at node z_j
              (x_ref = 1 everywhere; AFGL ozone profile).
  - b_band  : multiplier on sigma_O3(lambda) for wavelengths in `band`
              (prior b_ref = 1, prior std = band.sigma_rel).

Linearised around x_ref:

    tau_slant(lambda_i)  =  K_O3[i, :] @ x_O3  +  tau_ref(lambda_i) * b_band(lambda_i)
    K_O3[i, j]           =  sum_z chord(z) * sigma_O3(lambda_i; T(z), P(z))
                                            * n_O3_ref(z) * W[z, j]

With the augmented Jacobian K_full of shape ((n_ch + 1), (n_O3 + n_bands)),
the OE posterior is
    S_hat = (K_meas^T S_eps^-1 K_meas + S_a^-1)^-1
and the variance on the predictand is w_target^T S_hat w_target.

S_eps adds an uncorrelated line-by-line scatter sigma_line * tau_ref(lambda_i)
in quadrature with the photometric noise.  The 8.80 um T-dependence appears as
an additional, predictand-side variance.

h_tan_m is a free parameter; the module is altitude-agnostic, although the
default analysis runs at 20 km.

References
----------
- Rodgers (2000), Inverse Methods for Atmospheric Sounding, Ch. 4
- Damadeo et al. (2013), SAGE v7.0 algorithm, AMT, doi:10.5194/amt-6-3539-2013
- SAGE III ATBD (Rohen et al. 2002): 0.05% transmission precision per channel
- HITRAN 2020: Gordon et al. (2022), J. Quant. Spectrosc. Radiat. Transfer 277
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

from .atmosphere import us_standard_atmosphere
from .geometry import tangent_to_slant_paths
from . import trace_gases as tg


# SAGE-III/ISS Chappuis-band channels (operational; see Notes/sage_note_v2.tex).
SAGE3_CHAPPUIS_NM: tuple[float, ...] = (521.0, 602.0, 676.0)

# AFGL-consistent default altitude node grid for the O3 state vector (km).
DEFAULT_NODES_KM: tuple[float, ...] = (14.0, 18.0, 22.0, 28.0, 40.0)


# ---------------------------------------------------------------------------
# Spectroscopic-band model (HITRAN absolute line-intensity scaling)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class O3Band:
    """Band-coherent line-intensity systematic on O3 cross sections.

    Model: sigma_O3(lambda; T, P) = sigma_O3_base(lambda; T, P) * b, with b ~ 1
    and prior std = sigma_rel.  b is 100% correlated across all wavelengths in
    `wavelength_range_m` and uncorrelated between bands -- the formal way to say
    "the HITRAN absolute line-intensity systematic cancels in band-internal
    ratios but not in inter-band ratios."
    """
    name: str
    sigma_rel: float
    wavelength_range_m: tuple

    def contains(self, wl_m: float) -> bool:
        lo, hi = self.wavelength_range_m
        return lo <= wl_m <= hi


# Two-band default: lumped UV (Hartley/Huggins/Chappuis) + lumped IR.  Both
# carry a ~5% HITRAN absolute line-intensity systematic by default; these are
# defensible starting points, not definitive uncertainties (see the discussion
# in the module docstring above).
DEFAULT_O3_BANDS: tuple[O3Band, ...] = (
    O3Band(name="uv", sigma_rel=0.05, wavelength_range_m=(2e-7, 1e-6)),
    O3Band(name="ir", sigma_rel=0.05, wavelength_range_m=(1e-6, 3e-5)),
)


@dataclass
class O3RemovalSetup:
    """Inputs for the O3 removal error analysis at one tangent altitude."""

    h_tan_m: float = 20_000.0
    target_wavelength_m: float = 8.80e-6   # round 43: reference element moved from 8.74 to 8.80 um

    # Measurement channels in three semantic groups (concatenated, in order):
    chappuis_channels_m: tuple = tuple(c * 1e-9 for c in SAGE3_CHAPPUIS_NM)
    aux_channels_m: tuple = ()                      # extra UV/Vis/SWIR co-retrieved channels
    ir_o3_channels_m: tuple = ()                    # IR channels for direct O3 constraint
                                                    # (e.g. 9.6 um band centre/shoulder)

    nodes_km: tuple = DEFAULT_NODES_KM                # O3-state altitude grid

    sigma_lnT: float = 5e-4                           # ATBD: 0.05% transmission precision
                                                       # at TOA / unsaturated channels.
    photon_limited_noise: bool = True                  # if True, scale sigma_lnT by
                                                       # exp(tau_ref/2) per channel to
                                                       # model photon-shot-noise growth
                                                       # in saturated channels.
    prior_sigma_rel: float = 0.3                      # +/- 30% per-node O3 prior
    prior_correlation_length_km: float = 6.0          # vertical corr. length of prior

    # Spectroscopic model (see module docstring for decomposition).
    bands: tuple = DEFAULT_O3_BANDS
    sigma_line_rel: float = 0.02                      # per-channel uncorrelated spec scatter
    delta_T_K: float = 10.0                           # T uncertainty for xsec at predictand

    # Optional N2O co-retrieval.  When True the state is augmented with N2O
    # number-density node multipliers + an N2O line-intensity band scaling, the
    # IR channels see N2O absorption, and the predictand becomes the total gas OD
    # to remove at lambda_target, tau_O3(target) + tau_N2O(target).  N2O is
    # globally well-mixed, so its prior is tighter than O3's; the stratospheric
    # profile gradient above the tangent carries the residual uncertainty that a
    # channel in the N2O nu1 band (7.78 um) constrains directly.
    coretrieve_n2o: bool = False
    n2o_nodes_km: tuple = DEFAULT_NODES_KM
    n2o_prior_sigma_rel: float = 0.05                 # +/- 5% per-node N2O prior
    n2o_prior_correlation_length_km: float = 6.0
    n2o_band_sigma_rel: float = 0.05                  # N2O HITRAN band systematic
    n2o_sigma_line_rel: float = 0.002                 # N2O per-channel line scatter

    fwhm_um_target: float = 0.25                      # MIR resolution element
    nu_pad_cm_ir: float = 500.0                       # Voigt-fetch pad for IR channels;
                                                       # >= ~500 cm^-1 needed so that
                                                       # cross sections at 8.80 um are
                                                       # converged against wings of the
                                                       # strong O3 nu1/nu3 lines around
                                                       # 1000-1100 cm^-1 and consistent
                                                       # across measurement-channel sets.

    # Altitude integration grid (layer-centre, metres).  Default 14-80 km @ 0.5 km.
    altitude_grid_m: np.ndarray = field(
        default_factory=lambda: np.arange(14_000.0, 80_001.0, 500.0))

    def all_channels_m(self) -> np.ndarray:
        return np.array(
            tuple(self.chappuis_channels_m)
            + tuple(self.aux_channels_m)
            + tuple(self.ir_o3_channels_m),
            float,
        )

    # Legacy alias used by older callers and tests.
    def channels_m(self) -> np.ndarray:
        return self.all_channels_m()


# ---------------------------------------------------------------------------
# Generalised multi-gas removal (config-agnostic): the engine run_o3_removal_
# analysis delegates to.  O3 is just one RetrievalGas; config C is target
# 20.4 um with gases = (H2O, HNO3); any channel/gas set plugs in here.
# ---------------------------------------------------------------------------

@dataclass
class RetrievalGas:
    """One co-retrieved gas in the augmented OE state.

    Carries its own altitude-node profile multipliers (hat basis), a Gaussian-
    correlated per-node prior, a set of line-intensity band systematics (the
    O3Band model -- usually one IR band per gas, two for O3 to split UV/IR),
    and an uncorrelated per-channel line-scatter level.
    """
    name: str
    nodes_km: tuple = DEFAULT_NODES_KM
    prior_sigma_rel: float = 0.30
    prior_correlation_length_km: float = 6.0
    bands: tuple = ()                 # tuple[O3Band] line-intensity systematics
    sigma_line_rel: float = 0.002


def n2o_default_gas(setup: "O3RemovalSetup") -> RetrievalGas:
    """The N2O RetrievalGas matching the legacy coretrieve_n2o parameters."""
    return RetrievalGas(
        name="n2o", nodes_km=setup.n2o_nodes_km,
        prior_sigma_rel=setup.n2o_prior_sigma_rel,
        prior_correlation_length_km=setup.n2o_prior_correlation_length_km,
        bands=(O3Band("n2o", setup.n2o_band_sigma_rel, (1e-6, 3e-5)),),
        sigma_line_rel=setup.n2o_sigma_line_rel)


@dataclass
class GasRemovalSetup:
    """Config-agnostic inputs for gas-removal error at one tangent altitude.

    The predictand is the TOTAL gas OD to subtract at target_wavelength_m,
    sum_g tau_g(target), with its full joint posterior error.  `gases` lists the
    species to co-retrieve; `channels_m` are all measurement channels (any mix
    of UV/Vis/IR).  Other fields mirror O3RemovalSetup.
    """
    target_wavelength_m: float
    channels_m: tuple
    gases: tuple                                      # tuple[RetrievalGas]
    h_tan_m: float = 20_000.0
    sigma_lnT: float = 5e-4
    photon_limited_noise: bool = True
    delta_T_K: float = 2.0
    fwhm_um_target: float = 0.25
    nu_pad_cm_ir: float = 500.0
    altitude_grid_m: np.ndarray = field(
        default_factory=lambda: np.arange(14_000.0, 80_001.0, 500.0))

    # Optional silica (aerosol) signal in the joint state.  Provide the reference
    # silica slant-OD spectrum at [channels..., target] (shape n_ch+1) for some
    # reference loading; the retrieval then co-fits a single silica amplitude
    # multiplier x_sil alongside the gases, and the engine reports the posterior
    # error on the silica OD at the target -- the degeneracy-aware detection
    # limit (silica = broad reststrahlen shape vs sharp gas lines).  None -> the
    # original gas-removal-only predictand.  The prior on x_sil is left loose
    # (we are detecting silica, not assuming it).
    silica_tau_ref: object = None                     # np.ndarray (n_ch+1,) or None
    silica_prior_sigma: float = 1.0e3                 # ~unconstrained x_sil prior


# ---------------------------------------------------------------------------
# Hat-function basis for the O3 state vector
# ---------------------------------------------------------------------------

def _hat_basis(nodes_m: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
    """Hat-function (linear-interpolation) basis W[k, j] = w_j(z_k).

    Each column j is a triangular function peaking at nodes_m[j] and reaching
    zero at the neighbouring nodes; the outer two columns extend as constant
    half-hats so the basis is a partition of unity on the full altitude range.
    Together W lets us write n_O3(z) ~ n_O3_ref(z) * sum_j W[z,j] * x_j.
    """
    nodes_m = np.asarray(nodes_m, float)
    alt_m = np.asarray(alt_m, float)
    n_z, n_j = len(alt_m), len(nodes_m)
    W = np.zeros((n_z, n_j))
    for j in range(n_j):
        z_c = nodes_m[j]
        if j > 0:
            z_l = nodes_m[j - 1]
            m = (alt_m > z_l) & (alt_m <= z_c)
            W[m, j] = (alt_m[m] - z_l) / (z_c - z_l)
        else:
            W[alt_m <= z_c, j] = 1.0
        if j < n_j - 1:
            z_r = nodes_m[j + 1]
            m = (alt_m > z_c) & (alt_m < z_r)
            W[m, j] = (z_r - alt_m[m]) / (z_r - z_c)
        else:
            W[alt_m >= z_c, j] = 1.0
    return W


# ---------------------------------------------------------------------------
# O3 cross sections at the state-vector altitude nodes (then interpolated)
# ---------------------------------------------------------------------------

def _xsec_o3_on_nodes(channels_m: np.ndarray, node_atm, fwhm_um: float,
                      nu_pad_cm: float = 300.0) -> np.ndarray:
    """sigma_O3(lambda, z_node) [m^2 molecule^-1] at each (channel, node).

    UV/Vis channels (< 1 um) use the MPI-Mainz Chappuis table (no T dependence
    in that table).  IR channels (>= 1 um) use HITRAN line-by-line cross sections
    evaluated at the node's (T, P), with a wide Voigt pad (`nu_pad_cm`) so that
    band-wing contributions from strong central lines are correctly included.
    Returns shape (n_nodes, n_channels).
    """
    channels_m = np.asarray(channels_m, float)
    n_node, n_ch = len(node_atm.altitude_m), len(channels_m)
    sig = np.zeros((n_node, n_ch))
    is_uv = channels_m < 1e-6
    if is_uv.any():
        s_uv = tg.uvvis_cross_section("o3", channels_m[is_uv])
        sig[:, is_uv] = s_uv[None, :]
    if (~is_uv).any():
        ir_idx = np.where(~is_uv)[0]
        for k in range(n_node):
            T = float(node_atm.temperature_k[k])
            P = float(node_atm.pressure_pa[k])
            sig[k, ir_idx] = tg.gas_xsec_rm("o3", channels_m[ir_idx], T, P,
                                            fwhm_um=fwhm_um,
                                            nu_pad_cm=nu_pad_cm)
    return sig


def _interp_xsec_to_grid(channels_m, alt_m, nodes_m, sig_nodes) -> np.ndarray:
    """Interpolate node-resolution xsec onto the fine altitude grid (n_z, n_ch)."""
    n_ch = sig_nodes.shape[1]
    sig = np.empty((len(alt_m), n_ch))
    for c in range(n_ch):
        sig[:, c] = np.interp(alt_m, nodes_m, sig_nodes[:, c],
                              left=sig_nodes[0, c], right=sig_nodes[-1, c])
    return sig


# ---------------------------------------------------------------------------
# Forward kernel for the O3-state part of the augmented state
# ---------------------------------------------------------------------------

def build_kernel(setup: O3RemovalSetup) -> tuple[np.ndarray, dict]:
    """O3-state Jacobian K_O3 (no band columns yet).

    Returns
    -------
    K_O3   : ((n_ch + 1), n_nodes) -- last row is the predictand kernel at
             lambda_target; first n_ch rows are the measurement Jacobian.
    info   : dict with chord, n_O3_ref, sig_z, alt_m, nodes_m, atm, channels_m,
             target_wavelength_m.

    Notes
    -----
    K_O3[i, j] = sum_z chord[z] * sigma_O3(lambda_i; T(z), P(z)) * n_O3_ref(z)
                                                                  * W[z, j]
    and tau_slant_ref(lambda_i) = K_O3[i, :].sum() (sum over node multipliers
    at x = 1).
    """
    alt_m = np.asarray(setup.altitude_grid_m, float)
    nodes_m = np.asarray(setup.nodes_km, float) * 1e3

    atm = us_standard_atmosphere(alt_m)
    node_atm = us_standard_atmosphere(nodes_m)

    chord = tangent_to_slant_paths(np.array([setup.h_tan_m]), alt_m)[0]    # (n_z,)
    n_o3_ref = tg.number_density_profiles(alt_m, atm.number_density_m3,
                                          gases=["o3"])["o3"]              # (n_z,)

    W = _hat_basis(nodes_m, alt_m)                                          # (n_z, n_nodes)

    channels = setup.all_channels_m()                                       # (n_ch,)
    target = np.array([setup.target_wavelength_m])
    all_lambdas = np.concatenate([channels, target])                        # (n_ch + 1,)

    sig_nodes = _xsec_o3_on_nodes(all_lambdas, node_atm, setup.fwhm_um_target,
                                  nu_pad_cm=setup.nu_pad_cm_ir)
    sig_z = _interp_xsec_to_grid(all_lambdas, alt_m, nodes_m, sig_nodes)    # (n_z, n_ch+1)

    weight = chord * n_o3_ref                                               # (n_z,)
    K_O3 = np.einsum("z, zi, zj -> ij", weight, sig_z, W)                   # (n_ch+1, n_nodes)

    info = {
        "channels_m": channels,
        "target_wavelength_m": setup.target_wavelength_m,
        "chord": chord,
        "n_O3_ref": n_o3_ref,
        "sig_z": sig_z,
        "alt_m": alt_m,
        "nodes_m": nodes_m,
        "atm": atm,
    }
    return K_O3, info


def build_band_jacobian(setup: O3RemovalSetup, K_O3: np.ndarray,
                        info: dict) -> np.ndarray:
    """Jacobian rows for the band-scaling parameters b_band.

    Linearised forward model contribution from band b at channel lambda_i:
        d tau_slant(lambda_i) / d b_band = tau_O3_ref(lambda_i)  if band contains lambda_i
                                         = 0                     otherwise
    where tau_O3_ref = K_O3.sum(axis=1) (the slant OD at x_O3 = 1).

    Returns K_band of shape ((n_ch + 1), n_bands).
    """
    all_lambdas = np.concatenate([info["channels_m"],
                                  [setup.target_wavelength_m]])
    tau_ref = K_O3.sum(axis=1)                                              # (n_ch+1,)
    K_band = np.zeros((K_O3.shape[0], len(setup.bands)))
    for b_idx, band in enumerate(setup.bands):
        for i, wl in enumerate(all_lambdas):
            if band.contains(float(wl)):
                K_band[i, b_idx] = tau_ref[i]
    return K_band


def build_T_jacobian(setup: O3RemovalSetup, K_O3: np.ndarray,
                     info: dict, delta_K_for_fd: float = 5.0) -> np.ndarray:
    """Jacobian column for a uniform-shift T-offset state parameter T_eff.

    For each channel lambda_i:
        d tau_slant(lambda_i) / d T_eff
            = sum_z chord(z) * (d sigma_O3(lambda_i; T(z))/dT) * n_O3_ref(z)
           ~= tau_O3_ref(lambda_i) * a(lambda_i)
    where a(lambda_i) = (1/sigma_ref) * d sigma_O3 / dT, computed by finite
    difference of the band-averaged HITRAN cross section at the slant-weighted
    effective (T_eff, P_eff).  delta_K_for_fd is the finite-difference step,
    independent of the user's T-uncertainty prior.

    UV/Chappuis channels: our MPI table is at a single T, so a = 0.  The actual
    Chappuis T-dependence (~0.05%/K) is small enough to ignore here.

    Returns K_T of shape ((n_ch + 1), 1).
    """
    w = info["chord"] * info["n_O3_ref"]
    T_eff = float(np.sum(w * info["atm"].temperature_k) / np.sum(w))
    P_eff = float(np.sum(w * info["atm"].pressure_pa) / np.sum(w))

    all_lambdas = np.concatenate([info["channels_m"],
                                  [setup.target_wavelength_m]])
    tau_ref = K_O3.sum(axis=1)                                              # (n_ch+1,)
    K_T = np.zeros((K_O3.shape[0], 1))
    for i, wl in enumerate(all_lambdas):
        wl = float(wl)
        if wl < 1e-6:
            continue  # UV: no T-dep in our cross-section path
        wl_arr = np.array([wl])
        s_hi = tg.gas_xsec_rm("o3", wl_arr, T_eff + delta_K_for_fd, P_eff,
                              fwhm_um=setup.fwhm_um_target,
                              nu_pad_cm=setup.nu_pad_cm_ir)[0]
        s_lo = tg.gas_xsec_rm("o3", wl_arr, T_eff - delta_K_for_fd, P_eff,
                              fwhm_um=setup.fwhm_um_target,
                              nu_pad_cm=setup.nu_pad_cm_ir)[0]
        s0 = tg.gas_xsec_rm("o3", wl_arr, T_eff, P_eff,
                            fwhm_um=setup.fwhm_um_target,
                            nu_pad_cm=setup.nu_pad_cm_ir)[0]
        if s0 > 0:
            a_lambda = (s_hi - s_lo) / (2.0 * delta_K_for_fd * s0)
            K_T[i, 0] = tau_ref[i] * a_lambda
    return K_T


# ---------------------------------------------------------------------------
# Co-retrieved-gas kernels (N2O), parallel to the O3 blocks above
# ---------------------------------------------------------------------------

def _xsec_gas_on_nodes(gas: str, channels_m: np.ndarray, node_atm,
                       fwhm_um: float, nu_pad_cm: float) -> np.ndarray:
    """sigma_gas(lambda, z_node) [m^2/molecule] via gas_xsec_rm at each node's
    (T, P).  UV/Vis channels return 0 for a gas with no UV path (e.g. N2O), so
    this is correct for an IR co-retrieved gas; returns shape (n_nodes, n_ch)."""
    channels_m = np.asarray(channels_m, float)
    n_node, n_ch = len(node_atm.altitude_m), len(channels_m)
    sig = np.zeros((n_node, n_ch))
    for k in range(n_node):
        T = float(node_atm.temperature_k[k])
        P = float(node_atm.pressure_pa[k])
        sig[k, :] = tg.gas_xsec_rm(gas, channels_m, T, P, fwhm_um=fwhm_um,
                                   nu_pad_cm=nu_pad_cm)
    return sig


def build_gas_kernel(gas: str, setup: O3RemovalSetup, nodes_m: np.ndarray,
                     all_lambdas: np.ndarray, info: dict) -> tuple[np.ndarray, np.ndarray]:
    """Augmented-state Jacobian for a co-retrieved gas, parallel to build_kernel's
    O3 block: K_gas[i, j] = sum_z chord(z) * sigma_gas(lambda_i; T,P) * n_gas_ref(z)
                                                                       * W[z, j].

    Returns K_gas of shape ((n_ch + 1), n_nodes) (last row = predictand kernel)
    and the reference number-density profile n_gas_ref on info["alt_m"].
    """
    alt_m = info["alt_m"]
    node_atm = us_standard_atmosphere(np.asarray(nodes_m, float))
    n_gas_ref = tg.number_density_profiles(alt_m, info["atm"].number_density_m3,
                                           gases=[gas])[gas]
    W = _hat_basis(nodes_m, alt_m)
    sig_nodes = _xsec_gas_on_nodes(gas, all_lambdas, node_atm,
                                   setup.fwhm_um_target, setup.nu_pad_cm_ir)
    sig_z = _interp_xsec_to_grid(all_lambdas, alt_m, nodes_m, sig_nodes)
    weight = info["chord"] * n_gas_ref
    K_gas = np.einsum("z, zi, zj -> ij", weight, sig_z, W)
    return K_gas, n_gas_ref


def gas_T_column(gas: str, all_lambdas: np.ndarray, tau_ref: np.ndarray,
                 n_gas_ref: np.ndarray, info: dict, setup: O3RemovalSetup,
                 delta_K_for_fd: float = 5.0) -> np.ndarray:
    """T_eff Jacobian column contributed by a co-retrieved gas (same shared
    atmospheric T offset as O3): d tau_gas(lambda_i)/dT_eff = tau_gas_ref(i) *
    (1/sigma) dsigma/dT, evaluated at the gas-column-weighted slant (T, P).
    Returns shape ((n_ch + 1), 1).  UV channels contribute 0.

    The three finite-difference cross sections are computed in ONE vectorised
    gas_xsec_rm call each (over all IR wavelengths at once), not per wavelength:
    a single Voigt fetch spans the channel set, so cost is O(3) Voigt evals, not
    O(3 * n_channels) -- essential for many-channel configs with heavy line lists
    (e.g. HNO3/H2O at config C)."""
    all_lambdas = np.asarray(all_lambdas, float)
    w = info["chord"] * n_gas_ref
    T_eff = float(np.sum(w * info["atm"].temperature_k) / np.sum(w))
    P_eff = float(np.sum(w * info["atm"].pressure_pa) / np.sum(w))
    K_T = np.zeros((len(all_lambdas), 1))
    ir = all_lambdas >= 1e-6          # UV/Vis channels have no T-dep xsec path
    if not ir.any():
        return K_T
    wl_ir = all_lambdas[ir]
    kw = dict(fwhm_um=setup.fwhm_um_target, nu_pad_cm=setup.nu_pad_cm_ir)
    s_hi = tg.gas_xsec_rm(gas, wl_ir, T_eff + delta_K_for_fd, P_eff, **kw)
    s_lo = tg.gas_xsec_rm(gas, wl_ir, T_eff - delta_K_for_fd, P_eff, **kw)
    s0 = tg.gas_xsec_rm(gas, wl_ir, T_eff, P_eff, **kw)
    a = np.zeros_like(s0)
    m = s0 > 0
    a[m] = (s_hi[m] - s_lo[m]) / (2.0 * delta_K_for_fd * s0[m])
    K_T[np.where(ir)[0], 0] = tau_ref[ir] * a
    return K_T


# ---------------------------------------------------------------------------
# Optimal estimation (Rodgers 2000)
# ---------------------------------------------------------------------------

def gaussian_prior_covariance(nodes_m: np.ndarray, sigma_rel: float,
                              correlation_length_m: float) -> np.ndarray:
    """Exponential-Gaussian prior on the O3 multipliers x_j.

    S_a[i, j] = sigma_rel^2 * exp(-0.5 * ((z_i - z_j) / L)^2)
    """
    z = np.asarray(nodes_m, float)
    dz = z[:, None] - z[None, :]
    return (sigma_rel ** 2) * np.exp(-0.5 * (dz / correlation_length_m) ** 2)


def posterior_covariance(K_meas: np.ndarray, S_eps: np.ndarray,
                         S_a: np.ndarray) -> np.ndarray:
    """S_hat = (K^T S_eps^-1 K + S_a^-1)^-1  (Rodgers 2000, eq. 4.5)."""
    S_a_inv = np.linalg.inv(S_a)
    S_eps_inv = np.linalg.inv(S_eps)
    return np.linalg.inv(K_meas.T @ S_eps_inv @ K_meas + S_a_inv)


def _block_diag(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Two-block diagonal stack (np.block helper)."""
    Z1 = np.zeros((A.shape[0], B.shape[1]))
    Z2 = np.zeros((B.shape[0], A.shape[1]))
    return np.block([[A, Z1], [Z2, B]])


# ---------------------------------------------------------------------------
# Spectroscopic-uncertainty diagnostics
# ---------------------------------------------------------------------------

def _xsec_T_sensitivity_at_target(setup: O3RemovalSetup, info: dict) -> float:
    """Relative half-spread |sigma(T+dT) - sigma(T-dT)| / (2 sigma(T)) at lambda_target.

    Uses the contribution-weighted (T, P) along the slant path as the reference,
    since that is what tau_O3(lambda_target) is most sensitive to.
    """
    w = info["chord"] * info["n_O3_ref"]
    T_eff = float(np.sum(w * info["atm"].temperature_k) / np.sum(w))
    P_eff = float(np.sum(w * info["atm"].pressure_pa) / np.sum(w))

    wl = np.array([setup.target_wavelength_m])
    s0 = tg.gas_xsec_rm("o3", wl, T_eff, P_eff,
                        fwhm_um=setup.fwhm_um_target,
                        nu_pad_cm=setup.nu_pad_cm_ir)[0]
    s_hi = tg.gas_xsec_rm("o3", wl, T_eff + setup.delta_T_K, P_eff,
                          fwhm_um=setup.fwhm_um_target,
                          nu_pad_cm=setup.nu_pad_cm_ir)[0]
    s_lo = tg.gas_xsec_rm("o3", wl, T_eff - setup.delta_T_K, P_eff,
                          fwhm_um=setup.fwhm_um_target,
                          nu_pad_cm=setup.nu_pad_cm_ir)[0]
    if s0 <= 0.0:
        return 0.0
    return 0.5 * abs(s_hi - s_lo) / s0


# ---------------------------------------------------------------------------
# Legacy helper (kept for direct use; no longer used by run_o3_removal_analysis)
# ---------------------------------------------------------------------------

def spectroscopic_uncertainty_rel(setup: O3RemovalSetup, info: dict) -> dict:
    """Legacy lump-sum spectroscopic uncertainty.  Equivalent to the old
    quadrature of (band absolute + T-extrapolation) at lambda_target -- useful
    for sanity-checking the augmented-state model.
    """
    sigma_T_rel = _xsec_T_sensitivity_at_target(setup, info)
    # Pull the band that contains the predictand, default to the largest sigma_rel.
    target = setup.target_wavelength_m
    band_sigma = max((b.sigma_rel for b in setup.bands if b.contains(target)),
                     default=max(b.sigma_rel for b in setup.bands))
    total_rel = float(np.sqrt(band_sigma ** 2 + setup.sigma_line_rel ** 2
                              + sigma_T_rel ** 2))
    return {
        "band_rel":  float(band_sigma),
        "line_rel":  float(setup.sigma_line_rel),
        "T_rel":     float(sigma_T_rel),
        "total_rel": total_rel,
    }


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def _build_bands_for_gas(gas: RetrievalGas, all_lambdas: np.ndarray,
                         tau_gas_ref: np.ndarray) -> np.ndarray:
    """Band-scaling Jacobian columns for one gas: d tau_gas(lambda_i)/d b =
    tau_gas_ref(i) if the band contains lambda_i, else 0.  Shape (n_lam, n_bands)."""
    K = np.zeros((len(all_lambdas), len(gas.bands)))
    for b_idx, band in enumerate(gas.bands):
        for i, wl in enumerate(all_lambdas):
            if band.contains(float(wl)):
                K[i, b_idx] = tau_gas_ref[i]
    return K


def run_gas_removal_analysis(setup: GasRemovalSetup) -> dict:
    """Config-agnostic gas-removal OE.  State = [for each gas: node multipliers]
    + [for each gas: band scalings] + [shared T offset]; predictand = total gas
    OD to subtract at the target, sum_g tau_g(target).  Returns sigma_total (OD,
    the number for the SAI detection-limit pipeline), a sub-contribution
    breakdown, per-gas diagnostics, and the usual OE matrices.
    """
    alt_m = np.asarray(setup.altitude_grid_m, float)
    atm = us_standard_atmosphere(alt_m)
    chord = tangent_to_slant_paths(np.array([setup.h_tan_m]), alt_m)[0]
    channels = np.asarray(setup.channels_m, float)
    n_ch = len(channels)
    all_lambdas = np.concatenate([channels, [setup.target_wavelength_m]])
    info = {"channels_m": channels, "target_wavelength_m": setup.target_wavelength_m,
            "chord": chord, "alt_m": alt_m, "atm": atm}

    gas_blocks, band_blocks, node_priors, band_priors = [], [], [], []
    K_T = np.zeros((n_ch + 1, 1))
    tau_meas_total = np.zeros(n_ch)
    tau_target_total = 0.0
    line_meas_sq = np.zeros(n_ch)
    line_pred_sq = 0.0
    n_nodes_list, n_bands_list = [], []
    tau_target_by_gas, band_prior_sigma = {}, {}

    for g in setup.gases:
        nodes_m = np.asarray(g.nodes_km, float) * 1e3
        K_g, n_g_ref = build_gas_kernel(g.name, setup, nodes_m, all_lambdas, info)
        tau_g_ref = K_g.sum(axis=1)                            # (n_ch+1,)
        K_bg = _build_bands_for_gas(g, all_lambdas, tau_g_ref)
        K_T = K_T + gas_T_column(g.name, all_lambdas, tau_g_ref, n_g_ref,
                                 info, setup)                   # shared T offset
        gas_blocks.append(K_g)
        band_blocks.append(K_bg)
        node_priors.append(gaussian_prior_covariance(
            nodes_m, g.prior_sigma_rel, g.prior_correlation_length_km * 1e3))
        if g.bands:
            band_priors.append(np.diag([b.sigma_rel ** 2 for b in g.bands]))
        tau_meas_total += tau_g_ref[:n_ch]
        tau_target_total += float(tau_g_ref[n_ch])
        tau_target_by_gas[g.name] = float(tau_g_ref[n_ch])
        n_nodes_list.append(K_g.shape[1])
        n_bands_list.append(K_bg.shape[1])
        for b in g.bands:
            band_prior_sigma[b.name] = float(b.sigma_rel)
        line_meas_sq += (g.sigma_line_rel * tau_g_ref[:n_ch]) ** 2
        line_pred_sq += (g.sigma_line_rel * float(tau_g_ref[n_ch])) ** 2

    n_nodes_tot = int(sum(n_nodes_list))
    n_bands_tot = int(sum(n_bands_list))

    # Optional silica (aerosol) amplitude column: tau_silica(lambda_i) = x_sil *
    # silica_tau_ref(i), x_sil the loading multiplier.  Co-fitting it makes the
    # error budget degeneracy-aware (broad silica shape vs sharp gas lines).
    has_sil = setup.silica_tau_ref is not None
    if has_sil:
        K_sil = np.asarray(setup.silica_tau_ref, float).reshape(-1, 1)
        sil_blocks = [K_sil]
        sil_priors = [np.array([[setup.silica_prior_sigma ** 2]])]
    else:
        sil_blocks, sil_priors = [], []

    # State layout: [gas nodes | gas bands | (silica) | T]
    K_full = np.hstack(gas_blocks + band_blocks + sil_blocks + [K_T])
    i_sil = n_nodes_tot + n_bands_tot if has_sil else None
    iT = K_full.shape[1] - 1
    K_meas = K_full[:n_ch, :]

    # Measurement noise: photometric on the gas slant OD (small-signal silica) +
    # uncorrelated per-gas line scatter.
    if setup.photon_limited_noise:
        sigma_phot = setup.sigma_lnT * np.exp(tau_meas_total / 2.0)
    else:
        sigma_phot = np.full(n_ch, setup.sigma_lnT)
    S_eps = np.diag(sigma_phot ** 2 + line_meas_sq)

    # Block-diagonal prior: node priors..., band priors..., (silica), T.
    blocks = (list(node_priors) + list(band_priors) + sil_priors
              + [np.array([[setup.delta_T_K ** 2]])])
    S_a = blocks[0]
    for b in blocks[1:]:
        S_a = _block_diag(S_a, b)

    S_hat = posterior_covariance(K_meas, S_eps, S_a)

    # Gas-removal predictand (total gas OD at target): the silica column, if any,
    # is excluded -- it is a separate component, not part of the gas to remove.
    k_gas = K_full[n_ch, :].copy()
    if has_sil:
        k_gas[i_sil] = 0.0
    sigma_OE_sq = float(k_gas @ S_hat @ k_gas)
    sigma_line_pred = float(np.sqrt(line_pred_sq))
    sigma_total = float(np.sqrt(sigma_OE_sq + sigma_line_pred ** 2))

    # Per-gas state-contribution diagnostics (sub-blocks of the node region).
    sigma_OE_state_by_gas = {}
    off = 0
    for g, n_nodes in zip(setup.gases, n_nodes_list):
        sl = slice(off, off + n_nodes)
        sigma_OE_state_by_gas[g.name] = float(np.sqrt(
            k_gas[sl] @ S_hat[sl, sl] @ k_gas[sl]))
        off += n_nodes
    band_sl = slice(n_nodes_tot, n_nodes_tot + n_bands_tot)
    sigma_OE_band = float(np.sqrt(
        k_gas[band_sl] @ S_hat[band_sl, band_sl] @ k_gas[band_sl])) if n_bands_tot else 0.0
    sigma_OE_T = float(np.sqrt(k_gas[iT] ** 2 * S_hat[iT, iT]))

    # Band posteriors, in the band region in gas/band order.
    band_posterior_sigma = {}
    bcol = n_nodes_tot
    for g in setup.gases:
        for b in g.bands:
            band_posterior_sigma[b.name] = float(np.sqrt(S_hat[bcol, bcol]))
            bcol += 1
    T_posterior_K = float(np.sqrt(S_hat[iT, iT]))

    G = S_hat @ K_meas.T @ np.linalg.inv(S_eps)
    A = G @ K_meas
    dofs = float(np.trace(A))

    out = {
        "tau_target_ref":       tau_target_total,
        "tau_target_by_gas":    tau_target_by_gas,
        "tau_meas_ref":         tau_meas_total,
        "sigma_OE_state_by_gas": sigma_OE_state_by_gas,
        "sigma_OE_band":        sigma_OE_band,
        "sigma_OE_T":           sigma_OE_T,
        "sigma_OE":             float(np.sqrt(sigma_OE_sq)),
        "sigma_line_pred":      sigma_line_pred,
        "sigma_total":          sigma_total,
        "band_prior_sigma":     band_prior_sigma,
        "band_posterior_sigma": band_posterior_sigma,
        "T_prior_K":            float(setup.delta_T_K),
        "T_posterior_K":        T_posterior_K,
        "S_hat":                S_hat,
        "gain":                 G,
        "averaging_kernel":     A,
        "dofs":                 dofs,
        "K":                    K_full,
        "info":                 info,
        "setup":                setup,
    }

    # Silica detection limit: posterior error on the retrieved silica OD at the
    # target = silica_tau_ref(target) * sigma(x_sil), from the JOINT fit -- this
    # is the degeneracy-aware sigma(tau_silica) (cf. gas-removal sigma_total,
    # which assumed silica perfectly separable).
    if has_sil:
        sil_od_target = float(K_full[n_ch, i_sil])      # = silica_tau_ref(target)
        x_sil_post = float(np.sqrt(S_hat[i_sil, i_sil]))
        out["silica_od_target"] = sil_od_target
        out["x_sil_posterior"] = x_sil_post
        out["sigma_silica_od"] = sil_od_target * x_sil_post
    return out


def o3_setup_to_gases(setup: O3RemovalSetup) -> tuple:
    """The RetrievalGas list equivalent to an O3RemovalSetup (O3, +N2O if on)."""
    gases = [RetrievalGas(
        name="o3", nodes_km=setup.nodes_km, prior_sigma_rel=setup.prior_sigma_rel,
        prior_correlation_length_km=setup.prior_correlation_length_km,
        bands=setup.bands, sigma_line_rel=setup.sigma_line_rel)]
    if setup.coretrieve_n2o:
        gases.append(n2o_default_gas(setup))
    return tuple(gases)


def run_o3_removal_analysis(setup: O3RemovalSetup) -> dict:
    """O3-removal error for the silica 8.80 um channel -- now a thin wrapper over
    the config-agnostic run_gas_removal_analysis with gases = (O3 [, N2O]).

    With setup.coretrieve_n2o = False the predictand is sigma(tau_O3 @ target);
    with it True the state gains N2O and the predictand is the total O3+N2O OD.
    Returns the same legacy keys as before (sigma_total, sigma_OE, per-band
    posteriors, DOFS, ...) plus tau_O3_target / tau_n2o_target.
    """
    gsetup = GasRemovalSetup(
        target_wavelength_m=setup.target_wavelength_m,
        channels_m=tuple(setup.all_channels_m()),
        gases=o3_setup_to_gases(setup),
        h_tan_m=setup.h_tan_m, sigma_lnT=setup.sigma_lnT,
        photon_limited_noise=setup.photon_limited_noise,
        delta_T_K=setup.delta_T_K, fwhm_um_target=setup.fwhm_um_target,
        nu_pad_cm_ir=setup.nu_pad_cm_ir, altitude_grid_m=setup.altitude_grid_m)
    r = run_gas_removal_analysis(gsetup)

    # Map to legacy keys.
    r["tau_O3_target"] = r["tau_target_by_gas"]["o3"]
    r["tau_n2o_target"] = r["tau_target_by_gas"].get("n2o", 0.0)
    r["sigma_OE_state"] = r["sigma_OE_state_by_gas"]["o3"]
    r["sigma_OE_n2o_state"] = r["sigma_OE_state_by_gas"].get("n2o", 0.0)
    r["setup"] = setup
    return r
