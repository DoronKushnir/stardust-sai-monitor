"""uncertainty_model.py -- referee round 6 (paper 1; public port), items 2, 3, 4, 5 and 7:
one measurement model for the headline sensitivity, propagated as a full covariance through
the joint element-observable retrieval of reproduce/element_observable_closure.py (which
this script imports for its forward model, Jacobians, priors and covariance blocks).

What is computed (20-km tangent; the 47 design elements as SLANT optical-depth rows of the one
prescribed silica profile, the 15 window elements with the full gas state of round 70 and the 32
outer elements with the aerosol columns only; the visible/NIR triplet as per-shell extinction
rows; 38-parameter state and priors of round 70):

 1. The measurement covariance of the primary model, term by term (item 2 and 7):
      (a) radiometric noise: either the round-70 convention (per-sample transmission floor
          e^{y/2}/S with S = 2000) or the separated law of item 7, additive detector noise
          e^{y}/S_det (S_det the sourced transmission SNR at the exo-atmospheric reference),
          the systematic transmission floor 1/2000 per sample and element, and the shared
          exo-atmospheric reference (mean of n_ref samples, common to all tangents of an event);
      (b) the independent per-line database scatter through the element responses
          (window elements; round 70);
      (c) a model-discrepancy term: the measured sounding-to-sounding scatter of the ACE-FTS
          residual product on the design grid (Appendix E, unscaled: 1.86e-3 OD at 8.80 um),
          with an inter-element correlation R(|dl|) fitted to the measured neighbour
          correlations (0.61 at 0.1 um, 0.14 at 0.2 um): Gaussian exp(-(dl/l)^2) with
          l = 0.14 um matches both; the exponential form cannot (l = 0.20 from the first pair,
          0.10 from the second) and is kept as a bracket; vertical structure: independent
          between tangent heights (it is a scatter across soundings); physical content: the
          instrument-line-shape, calibration and atmospheric-variability residual of a product
          after gas removal, which the spectroscopic model of (a)-(b) does not contain.
    The marginal silica uncertainty (3 sigma, per 0.5-km shell at 20 km = slant amplitude
    x rho) of the joint fit for: window with the triplet, window alone, band with the triplet,
    band alone; for the primary model and for the scenarios (no discrepancy term; discrepancy
    at the 2.47e-3 anchor; independent discrepancy; exponential l = 0.1 / 0.25 um;
    noise-law variant; gray offset mode added as a nuisance), and the two-stage diagonal-floor
    procedure of the paper on the same per-element levels (the sensitivity scenario the paper
    quoted as its threshold until this round).
 2. Audit of the joint retrieval (item 3): the row Jacobians (units), the prior covariance
    (units), the measurement covariance, the gain row and the averaging-kernel row of the
    silica amplitude, and the sensitivity of the silica uncertainty to every prior width and
    to the gas correlation length.
 3. Response-parameter derivatives (item 4): the analytic derivative of y_i with respect to
    the wavelength registration and the edge softness (the two response-only terms of the
    general formula) against the central finite differences the code uses; continuity of y_i
    in the registration for a step-edged boxcar.
 4. Geometry (item 5): the silica row Jacobian at every tangent height 14-60 km (slant OD per
    Tg per element) and its onion-peel to a per-shell extinction, compared with the per-shell
    extinction per Tg of the forward model (the slant-to-shell factor rho as a consequence,
    not an input); a vertically coherent +2 % O3 database error carried through the element
    observable at successive tangents and peeled, against the f alpha_gas(z) identity of
    Appendix D9.
 5. The exponential-correlation scan of Appendix D6 restated as ratios on the common fine
    grid (item 2), and the correlation-length fits.

Run from the repo root:  python reproduce/uncertainty_model.py
Writes outputs/uncertainty_model.json and _audit.npz.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))

import element_observable_closure as C                           # noqa: E402  (module-level forward model)
import calibrated_background_thresholds as cbt                            # noqa: E402
from saimon.geometry import tangent_to_slant_paths                         # noqa: E402
from saimon.atmosphere import us_standard_atmosphere                       # noqa: E402
from saimon import trace_gases as tg                                        # noqa: E402
from saimon.onion_peel import M1_PER_OD, OD_PER_M1, G_ONION, R_EARTH_M, DZ_M  # noqa: E402

OUT = ROOT / "outputs"
ARCH = OUT / "uncertainty_model.json"
AUDIT = OUT / "uncertainty_model_audit.npz"
t_start = time.time()


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- inputs
thr = json.load(open(ROOT / "outputs/calibrated_background_thresholds.json"))
FLOOR_SCALE = float(thr["floor_anchor"]["floor_scale"])                    # 1.327: adopted floors / measured scatter
band_el = np.round(np.array(thr["band01_elements_um"], float), 2)          # 47 design elements
band_fl_m1 = np.array(thr["band01_floors_m1"], float)                     # adopted floors (2.47e-3-anchored), m^-1 per shell
s_adopt_od = band_fl_m1 * OD_PER_M1                                        # adopted floors in slant OD
s_meas_od = s_adopt_od / FLOOR_SCALE                                       # measured ACE scatter per element (unscaled)
win = np.array([k for k, c in enumerate(band_el) if c < 9.25]); outer = np.array([k for k, c in enumerate(band_el) if c > 9.25])
assert len(win) == C.n_el and np.allclose(band_el[win], C.EL_CENTRES_UM)
n_mir = len(band_el)
log(f"47 elements: {len(win)} window + {len(outer)} outer; measured scatter at 8.80 um {s_meas_od[C.i880]:.3e} OD "
    f"(adopted {s_adopt_od[C.i880]:.3e})")

# neighbour correlations of the ACE null-test residuals on the design band (Appendix D6 / F1)
resn = json.load(open(ROOT / "outputs/ace_v52/nulltest_resolution.json"))
corr_meas = {float(k): v["median_corr"] for k, v in resn["residual_correlation_band_0p1"].items()}
r1, r2 = corr_meas[0.10], corr_meas[0.20]
ell_gauss = 0.1 / np.sqrt(-np.log(r1))                                     # Gaussian length fitted to the 0.1-um pair
ell_exp_1, ell_exp_2 = -0.1 / np.log(r1), -0.2 / np.log(r2)                # exponential lengths from each pair
corr_fits = dict(measured={f"{k:.2f}": v for k, v in corr_meas.items()},
                 gaussian_l_um=float(ell_gauss), gaussian_predicts_0p2=float(np.exp(-(0.2 / ell_gauss) ** 2)),
                 gaussian_predicts_0p3=float(np.exp(-(0.3 / ell_gauss) ** 2)),
                 exponential_l_from_0p1=float(ell_exp_1), exponential_l_from_0p2=float(ell_exp_2),
                 exponential_0p25_predicts=[float(np.exp(-0.1 / 0.25)), float(np.exp(-0.2 / 0.25))],
                 exponential_0p1_predicts=[float(np.exp(-0.1 / 0.1)), float(np.exp(-0.2 / 0.1))])
log(f"correlation: measured {r1:.3f}/{r2:.3f} at 0.1/0.2 um; Gaussian l = {ell_gauss:.3f} um predicts {corr_fits['gaussian_predicts_0p2']:.3f} at 0.2, "
    f"{corr_fits['gaussian_predicts_0p3']:.3f} at 0.3; exponential l from the two pairs {ell_exp_1:.2f}/{ell_exp_2:.2f} um")


def corr_matrix(kind, ell=None):
    D = np.abs(band_el[:, None] - band_el[None, :])
    if kind == "indep":
        return np.eye(n_mir)
    if kind == "gauss":
        return np.exp(-(D / ell) ** 2)
    if kind == "exp":
        return np.exp(-D / ell)
    raise ValueError(kind)


# ---------------------------------------------------------------- rows: 47 slant-OD rows + triplet
# window rows: the round-70 Jacobian (full gas state); outer rows: aerosol columns only, slant convention
sl20_o = C.slant_at(C.H_TAN_M, C.prof_o)                                   # (7, n_lam_outer) slant OD per unit parameter
K_out = np.zeros((len(outer), C.N_STATE))
for n, k in enumerate(outer):
    c = band_el[k]
    m = (C.lam_outer >= c - C.EL_HALF_UM) & (C.lam_outer < c + C.EL_HALF_UM + 1e-9)
    K_out[n, C.I_AER] = sl20_o[:, m].mean(axis=1)
K_mir = np.zeros((n_mir, C.N_STATE)); K_mir[win] = C.K0; K_mir[outer] = K_out
# gas optical depth of the outer elements (photon term): -ln<T> over the landscape mean-OD curve (0.01-um steps;
# the outer elements carry no line-resolved model; the mean OD is an upper bound on the attenuation there)
land = np.load(ROOT / "outputs/slant_od_landscape_curves.npz")
wl_l, gas_l = land["p1_wl"], land["p1_gas"]
y_mir = np.zeros(n_mir); y_mir[win] = C.y0
for k in outer:
    m = (wl_l >= band_el[k] - 0.05 - 1e-9) & (wl_l <= band_el[k] + 0.05 + 1e-9)
    y_mir[k] = -np.log(np.mean(np.exp(-gas_l[m])))
K_tri, S_tri = C.triplet_rows()
K_all = np.vstack([K_mir, K_tri])
row_labels = [f"MIR {c:.2f} um (slant OD at 20 km)" for c in band_el] + [f"vis/NIR {w:.0f} nm (per-shell extinction, m^-1)" for w in C.WL_TRIPLET_NM]
RHO = C.RHO_SLANT_TO_SHELL

# ---------------------------------------------------------------- covariance terms
S_DET = 1.4e4         # sourced transmission SNR at the exo-atmospheric reference, 0.17-s dwell (Table D5, VOx microbolometer)
S_CAP = 2000.0        # systematic transmission floor of flown occultation instruments
N_REF = 100           # exo-atmospheric reference samples averaged


def noise_cov(law="round70", snr=S_CAP, s_det=S_DET):
    """radiometric noise on the 47 MIR slant rows (variances in OD^2)."""
    if law == "round70":
        return np.diag((np.exp(y_mir / 2.0) / snr) ** 2)
    if law == "separated":
        det = (np.exp(y_mir) / s_det) ** 2                 # additive detector noise: fractional error grows as e^y
        floor = (1.0 / S_CAP) ** 2                          # systematic transmission floor, per sample and element
        ref = (1.0 / (s_det * np.sqrt(N_REF))) ** 2         # shared exo-atmospheric reference (common to all tangents)
        return np.diag(det + floor + ref)
    if law == "detector_only":
        return np.diag((np.exp(y_mir) / s_det) ** 2 + (1.0 / (s_det * np.sqrt(N_REF))) ** 2)
    raise ValueError(law)


def line_cov():
    Cl = np.zeros((n_mir, n_mir)); Cl[np.ix_(win, win)] = C.line_covariance(C.SIGMA_ETA_BASE)
    return Cl


def disc_cov(level="measured", kind="gauss", ell=None):
    s = {"measured": s_meas_od, "adopted": s_adopt_od, "none": np.zeros(n_mir)}[level]
    R = corr_matrix(kind, ell if ell is not None else ell_gauss)
    return np.outer(s, s) * R


def full_cov(noise="round70", disc="measured", kind="gauss", ell=None, line=True):
    S = noise_cov(noise) + (line_cov() if line else 0.0) + disc_cov(disc, kind, ell)
    return S


AER_FLAT = 1e6         # round 74: a flat prior on the seven aerosol parameters (the round-70 value 1e3, in nm for the radii, is
                       # binding on the coarse-mode radius, nominal 1500 nm; see the prior scan)


def prior_inv(aer=AER_FLAT):
    Sa = C.S_a.copy()
    for k in range(C.I_AER.start, C.I_AER.stop):
        Sa[k, k] = aer ** 2
    return np.linalg.inv(Sa)


S_A_INV_PRIMARY = prior_inv(AER_FLAT)


def joint_sigma(S_mir, rows="band", triplet=True, S_a_inv=None, extra_nuis=None):
    """marginal silica sigma (slant Tg) of the joint fit on the chosen MIR rows (+ triplet); flat aerosol prior unless given."""
    S_a_inv = S_A_INV_PRIMARY if S_a_inv is None else S_a_inv
    idx = np.arange(n_mir) if rows == "band" else win
    K = K_mir[idx]; S = S_mir[np.ix_(idx, idx)]
    if extra_nuis is not None:                               # one extra nuisance column with a flat prior (gray offset)
        K = np.hstack([K, extra_nuis[idx][:, None]])
        Sa = np.zeros((C.N_STATE + 1,) * 2); Sa[:C.N_STATE, :C.N_STATE] = S_a_inv; Sa[-1, -1] = 1e-12
        S_a_inv = Sa
        Kt = np.hstack([K_tri, np.zeros((3, 1))])
    else:
        Kt = K_tri
    if triplet:
        K = np.vstack([K, Kt]); S = C._blockdiag(S, S_tri)
    Sinv = np.linalg.inv(S)
    S_hat = np.linalg.inv(K.T @ Sinv @ K + S_a_inv)
    return float(np.sqrt(S_hat[C.I_SIL, C.I_SIL])), S_hat, S_hat @ K.T @ Sinv, K, S


def two_stage(floors_od, rows="band", triplet=True):
    """the paper's two-stage procedure: per-element diagonal floors -> extinction (white gain) -> Fisher
    marginalisation on the per-shell columns (cbt.marginal), band or window, with or without the triplet."""
    idx = np.arange(n_mir) if rows == "band" else win
    sig_ext = floors_od[idx] * M1_PER_OD
    shell_rows = np.zeros((n_mir, 7))
    for n, c in enumerate(band_el):
        if c < 9.25:
            m = (C.lam_coarse >= c - C.EL_HALF_UM) & (C.lam_coarse < c + C.EL_HALF_UM)
            shell_rows[n] = C.shell_c[:, m].mean(axis=1)
        else:
            m = (C.lam_outer >= c - C.EL_HALF_UM) & (C.lam_outer < c + C.EL_HALF_UM + 1e-9)
            shell_rows[n] = C.shell_o[:, m].mean(axis=1)
    t_all = np.r_[C.shell_trip[0], shell_rows[idx, 0]]
    nuis = [np.r_[C.shell_trip[j], shell_rows[idx, j]] for j in range(1, 7)]
    sig_all = np.r_[np.sqrt(np.diag(S_tri)), sig_ext]
    ch = np.arange(len(t_all)) if triplet else np.arange(3, len(t_all))
    sa, _, R2 = cbt.marginal(ch, nuis, sig_all, t_all)
    return float(sa), float(1 - R2)


arch = dict(script="reproduce/uncertainty_model.py", round=74,
            elements_um=band_el.tolist(), window_index=win.tolist(), outer_index=outer.tolist(),
            s_measured_od=s_meas_od.tolist(), s_adopted_od=s_adopt_od.tolist(), floor_scale=FLOOR_SCALE,
            y_gas_elements=y_mir.tolist(), rho_slant_to_shell_20km=RHO, correlation=corr_fits,
            noise_constants=dict(S_cap=S_CAP, S_det_toa=S_DET, n_ref=N_REF),
            row_labels=row_labels, state=C.names)

# ---------------------------------------------------------------- 1. the models
log("[1] measurement models (3 sigma per 0.5-km shell at 20 km, Tg)")
models = {
    "A_spectroscopic": dict(noise="round70", disc="none", kind="indep", note="photon e^{y/2}/2000 + per-line scatter (window); outer elements photon only (no gas model): window numbers only are meaningful"),
    "B_primary": dict(noise="round70", disc="measured", kind="gauss", note="A + measured ACE scatter per element with Gaussian correlation l = 0.14 um (fitted to 0.61/0.14)"),
    "B2_primary_separated_noise": dict(noise="separated", disc="measured", kind="gauss", note="B with detector noise e^y/S_det + floor 1/2000 + shared reference"),
    "C_disc_independent": dict(noise="round70", disc="measured", kind="indep", note="B with the discrepancy term independent between elements"),
    "D_disc_exp_0p10": dict(noise="round70", disc="measured", kind="exp", ell=0.10, note="B with exponential correlation l = 0.10 um"),
    "E_disc_exp_0p25": dict(noise="round70", disc="measured", kind="exp", ell=0.25, note="B with exponential correlation l = 0.25 um"),
    "F_disc_exp_0p20": dict(noise="round70", disc="measured", kind="exp", ell=0.20, note="B with exponential correlation l = 0.20 um (the 1/e length of the measured profile)"),
    "G_disc_adopted_level": dict(noise="round70", disc="adopted", kind="gauss", note="B with the discrepancy term at the 2.47e-3-anchored level (x1.33)"),
    "H_disc_adopted_indep": dict(noise="round70", disc="adopted", kind="indep", note="adopted floors, independent, with the gas state and photon term (round-70 'paper floors as covariance')"),
    "I_detector_only_no_floor": dict(noise="detector_only", disc="measured", kind="gauss", note="B without the 1/2000 systematic floor (detector noise and reference only)"),
}
res_models = {}
for key, m in models.items():
    S = full_cov(m["noise"], m["disc"], m["kind"], m.get("ell"))
    r = {}
    for rows in ("window", "band"):
        for trip in (True, False):
            s, *_ = joint_sigma(S, rows, trip)
            r[f"{rows}_{'triplet' if trip else 'alone'}"] = 3 * s * RHO
    r["note"] = m["note"]
    r["diag_sd_od_880"] = float(np.sqrt(S[C.i880, C.i880]))
    r["window_triplet_aerosol_prior_1e3"] = 3 * joint_sigma(S, "window", True, S_a_inv=C.S_a_inv)[0] * RHO
    r["band_triplet_aerosol_prior_1e3"] = 3 * joint_sigma(S, "band", True, S_a_inv=C.S_a_inv)[0] * RHO
    res_models[key] = r
    log(f"   {key:28s} window {r['window_triplet']:.4f} / alone {r['window_alone']:.4f}; band {r['band_triplet']:.4f} / alone {r['band_alone']:.4f}; diag SD 8.80 {r['diag_sd_od_880']:.2e} OD"
        f" (aerosol prior 1e3: window {r['window_triplet_aerosol_prior_1e3']:.4f}, band {r['band_triplet_aerosol_prior_1e3']:.4f})")
# gray-offset mode on the primary model (one more nuisance column, flat prior)
S_B = full_cov("round70", "measured", "gauss")
off = np.ones(n_mir)
res_models["B_plus_gray_offset"] = {f"{rows}_triplet": 3 * joint_sigma(S_B, rows, True, extra_nuis=off)[0] * RHO for rows in ("window", "band")}
res_models["B_plus_gray_offset"]["note"] = "primary model with a constant slant-OD offset common to all MIR elements as a 48th parameter (flat prior)"
log(f"   B + gray offset: window {res_models['B_plus_gray_offset']['window_triplet']:.4f}, band {res_models['B_plus_gray_offset']['band_triplet']:.4f}")
# the two-stage diagonal-floor procedure on the same levels
ts = {}
for lev, fl in (("adopted_2p47", s_adopt_od), ("measured_1p86", s_meas_od)):
    for rows in ("band", "window"):
        for trip in (True, False):
            sa, omr2 = two_stage(fl, rows, trip)
            ts[f"{lev}_{rows}_{'triplet' if trip else 'alone'}"] = dict(mmin_3sig_shell=3 * sa, one_minus_R2=omr2)
res_models["two_stage_diagonal"] = ts
log("   two-stage diagonal floors: " + ", ".join(f"{k} {v['mmin_3sig_shell']:.4f}" for k, v in ts.items()))
# SNR scan under the primary model (window with the triplet)
scan = {}
for snr in (400.0, 1000.0, 2000.0, 1e4, 1e9):
    S = noise_cov("round70", snr) + line_cov() + disc_cov("measured", "gauss")
    scan[f"{snr:.0f}"] = 3 * joint_sigma(S, "window", True)[0] * RHO
res_models["B_snr_scan_window_triplet"] = scan
log("   primary SNR scan: " + ", ".join(f"S={k}: {v:.4f}" for k, v in scan.items()))
arch["models"] = res_models

# ---------------------------------------------------------------- 2. audit of the joint retrieval (primary model)
log("[2] audit: Jacobians, priors, covariance, gain and averaging kernel; prior-width scan")
s_B, S_hat, G, K_used, S_used = joint_sigma(S_B, "band", True)
A = G @ K_used
units = (["dimensionless (profile multiplier)"] * (C.NGAS * C.NN) + ["K"] + ["dimensionless (amplitude)"] + ["dimensionless (intensity scaling)"] * C.NG
         + ["Tg (mass of the prescribed silica profile)", "dimensionless (fine amplitude)", "dimensionless (coarse amplitude)",
            "nm (fine median radius)", "dimensionless (fine geometric width)", "nm (coarse median radius)", "dimensionless (coarse geometric width)",
            "um (wavelength registration)", "um (edge softness)"])
assert len(units) == C.N_STATE
nominal = [1.0] * (C.NGAS * C.NN) + [0.0, 1.0] + [0.0] * C.NG + [0.0, 1.0, 1.0, cbt.CAL_QUIET[0][0], cbt.CAL_QUIET[0][1], cbt.CAL_QUIET[1][0], cbt.CAL_QUIET[1][1], 0.0, C.RESP_SOFT0]
prior_sd = np.sqrt(np.diag(C.S_a))
audit = dict(state=C.names, units=units, nominal=nominal, prior_sd=prior_sd.tolist(),
             prior_note="gas nodes: Gaussian prior with 6-km correlation length; aerosol parameters: 1e3 in their own units (nominal 60/1500 nm radii, widths 1.6/1.8), i.e. unconstrained; silica 1e3 Tg",
             rows=row_labels, jacobian_units="MIR rows: slant OD per unit of the state parameter; triplet rows: m^-1 per unit",
             silica_column_mir_slant_od_per_tg=K_mir[:, C.I_SIL].tolist(), silica_column_triplet_m1_per_tg=K_tri[:, C.I_SIL].tolist(),
             measurement_sd_od=np.sqrt(np.diag(S_used)).tolist(),
             gain_row_silica=G[C.I_SIL].tolist(), averaging_kernel_row_silica=A[C.I_SIL].tolist(),
             averaging_kernel_silica_self=float(A[C.I_SIL, C.I_SIL]),
             posterior_sd_silica_slant_tg=s_B, posterior_sd_silica_shell_tg=s_B * RHO,
             posterior_sd=np.sqrt(np.diag(S_hat)).tolist(),
             dofs_total=float(np.trace(A)))
np.savez_compressed(AUDIT, K=K_used, S_eps=S_used, S_a=C.S_a, S_hat=S_hat, G=G, A=A, names=np.array(C.names), rows=np.array(row_labels),
                    K_window=C.K0, y_mir=y_mir, s_measured_od=s_meas_od, s_adopted_od=s_adopt_od)
# prior-width sensitivity (primary model; band and window with the triplet)
from saimon.o3_removal import gaussian_prior_covariance  # noqa: E402


def prior_variant(gas_scale=1.0, gas_L_km=6.0, T_K=C.T_PRIOR_K, cfc=C.CFC_PRIOR, grp_scale=1.0, resp_scale=1.0, aer=AER_FLAT, which_gas=None, aer_only=None):
    Sa = np.zeros((C.N_STATE, C.N_STATE))
    for g in C.GASES:
        s, L = C.GAS_PRIOR[g]
        sc = gas_scale if (which_gas is None or g == which_gas) else 1.0
        Sa[C.I_GAS[g], C.I_GAS[g]] = gaussian_prior_covariance(C.nodes_m, s * sc, gas_L_km * 1e3)
    Sa[C.I_T, C.I_T] = T_K ** 2; Sa[C.I_CFC, C.I_CFC] = cfc ** 2
    for k, (g, r_, m_, s) in enumerate(C.GROUPS):
        Sa[C.I_GRP.start + k, C.I_GRP.start + k] = (s * grp_scale) ** 2
    for k in range(C.I_AER.start, C.I_AER.stop):
        Sa[k, k] = aer ** 2
    if aer_only is not None:                                   # (name, value): one aerosol parameter at another prior
        Sa[C.I_AER.start + C.AER_NAMES.index(aer_only[0]), C.I_AER.start + C.AER_NAMES.index(aer_only[0])] = aer_only[1] ** 2
    Sa[C.I_SHIFT, C.I_SHIFT] = (C.RESP_PRIOR_UM * resp_scale) ** 2; Sa[C.I_SOFT, C.I_SOFT] = (0.002 * resp_scale) ** 2
    return np.linalg.inv(Sa)


variants = {
    "baseline": {}, "gas_priors_x0.5": dict(gas_scale=0.5), "gas_priors_x2": dict(gas_scale=2.0),
    "o3_prior_x0.5": dict(gas_scale=0.5, which_gas="o3"), "o3_prior_x2": dict(gas_scale=2.0, which_gas="o3"),
    "gas_corr_3km": dict(gas_L_km=3.0), "gas_corr_12km": dict(gas_L_km=12.0),
    "T_prior_1K": dict(T_K=1.0), "T_prior_4K": dict(T_K=4.0),
    "group_priors_x0.5": dict(grp_scale=0.5), "group_priors_x2": dict(grp_scale=2.0),
    "response_priors_x0.5": dict(resp_scale=0.5), "response_priors_x2": dict(resp_scale=2.0),
    "aerosol_prior_10": dict(aer=10.0), "aerosol_prior_1e3": dict(aer=1e3), "aerosol_prior_1e4": dict(aer=1e4),
    "aerosol_prior_1e5": dict(aer=1e5), "aerosol_prior_1e8": dict(aer=1e8),
    "r_c_prior_1e3_only": dict(aer_only=("r_c", 1e3)), "r_f_prior_1e3_only": dict(aer_only=("r_f", 1e3)),
    "s_c_prior_1e3_only": dict(aer_only=("s_c", 1e3)), "A_c_prior_1e3_only": dict(aer_only=("A_c", 1e3)),
}
pscan = {}
for key, kw in variants.items():
    Sai = prior_variant(**kw)
    pscan[key] = {f"{rows}_triplet": 3 * joint_sigma(S_B, rows, True, S_a_inv=Sai)[0] * RHO for rows in ("window", "band")}
    pscan[key]["window_triplet_model"] = 3 * joint_sigma(full_cov("round70", "none", "indep"), "window", True, S_a_inv=Sai)[0] * RHO
    pscan[key]["window_alone"] = 3 * joint_sigma(S_B, "window", False, S_a_inv=Sai)[0] * RHO
    log(f"   prior {key:22s}: window {pscan[key]['window_triplet']:.4f}, band {pscan[key]['band_triplet']:.4f} (round-70 model window {pscan[key]['window_triplet_model']:.4f})")
audit["prior_width_scan"] = pscan
arch["audit"] = audit

# ---------------------------------------------------------------- 3. response-parameter derivatives (item 4)
log("[3] response-parameter derivatives: analytic against the finite differences of the code")
lam = C.lam_um; wS = C.wS; T0 = np.exp(-np.maximum(C.tau0, -60.0))


def dw_dshift(shift, soft):
    sc = max(abs(soft), 1e-4) * np.sqrt(2)
    lo = C.EL_CENTRES_UM[:, None] + shift - C.EL_HALF_UM; hi = C.EL_CENTRES_UM[:, None] + shift + C.EL_HALF_UM
    # d/dshift of 0.5[erf((l-lo)/sc) - erf((l-hi)/sc)]: lo and hi move with shift
    return -0.5 * (2 / np.sqrt(np.pi)) * (np.exp(-((lam[None, :] - lo) / sc) ** 2) - np.exp(-((lam[None, :] - hi) / sc) ** 2)) / sc


def dw_dsoft(shift, soft):
    sc = max(abs(soft), 1e-4) * np.sqrt(2)
    lo = C.EL_CENTRES_UM[:, None] + shift - C.EL_HALF_UM; hi = C.EL_CENTRES_UM[:, None] + shift + C.EL_HALF_UM
    ulo, uhi = (lam[None, :] - lo) / sc, (lam[None, :] - hi) / sc
    # d/dsoft: d(erf(u))/dsoft = (2/sqrt(pi)) e^{-u^2} du/dsoft, du/dsoft = -u/soft (sc proportional to soft)
    return 0.5 * (2 / np.sqrt(np.pi)) * (np.exp(-ulo ** 2) * (-ulo / soft) - np.exp(-uhi ** 2) * (-uhi / soft))


def analytic_response_derivative(dw):
    prof = C.PROF0
    A_ = prof @ (wS * T0); B_ = prof @ wS
    return -(dw @ (wS * T0)) / A_ + (dw @ wS) / B_          # <dtau/dp>_T = 0 for a response-only parameter


d = 0.001
fd_shift = (C.element_obs(C.tau0, d, C.RESP_SOFT0) - C.element_obs(C.tau0, -d, C.RESP_SOFT0)) / (2 * d)
an_shift = analytic_response_derivative(dw_dshift(0.0, C.RESP_SOFT0))
fd_soft = (C.element_obs(C.tau0, 0.0, C.RESP_SOFT0 + d) - C.element_obs(C.tau0, 0.0, C.RESP_SOFT0 - d)) / (2 * d)
an_soft = analytic_response_derivative(dw_dsoft(0.0, C.RESP_SOFT0))
d2 = 1e-4
fd_shift2 = (C.element_obs(C.tau0, d2, C.RESP_SOFT0) - C.element_obs(C.tau0, -d2, C.RESP_SOFT0)) / (2 * d2)
fd_soft2 = (C.element_obs(C.tau0, 0.0, C.RESP_SOFT0 + d2) - C.element_obs(C.tau0, 0.0, C.RESP_SOFT0 - d2)) / (2 * d2)
# continuity of y in the registration for a step-edged boxcar (soft -> 1e-4) and for the declared response
shifts = np.linspace(-0.004, 0.004, 161)
y_step = np.array([C.element_obs(C.tau0, s, 1e-4)[C.i880] for s in shifts])
y_soft = np.array([C.element_obs(C.tau0, s, C.RESP_SOFT0)[C.i880] for s in shifts])
dy_step = np.gradient(y_step, shifts); dy_soft = np.gradient(y_soft, shifts)
resp = dict(
    formula="dy_i/dp = <dtau/dp>_T - int[d(w S)/dp T]/A + int[d(w S)/dp]/B, A = int w S T, B = int w S; for the registration and the edge softness the first term vanishes and the two response terms are evaluated with dw/dp analytic",
    shift_fd_code_od_per_um=fd_shift.tolist(), shift_analytic_od_per_um=an_shift.tolist(),
    shift_fd_1e_4_od_per_um=fd_shift2.tolist(),
    shift_max_rel_diff_fd_vs_analytic=float(np.max(np.abs(fd_shift - an_shift) / np.maximum(np.abs(an_shift), 1e-12))),
    shift_max_rel_diff_fd1e_3_vs_fd1e_4=float(np.max(np.abs(fd_shift - fd_shift2) / np.maximum(np.abs(fd_shift2), 1e-12))),
    soft_fd_code_od_per_um=fd_soft.tolist(), soft_analytic_od_per_um=an_soft.tolist(),
    soft_max_rel_diff_fd_vs_analytic=float(np.max(np.abs(fd_soft - an_soft) / np.maximum(np.abs(an_soft), 1e-12))),
    soft_max_rel_diff_fd1e_3_vs_fd1e_4=float(np.max(np.abs(fd_soft - fd_soft2) / np.maximum(np.abs(fd_soft2), 1e-12))),
    atmospheric_term_for_response_parameters="zero (the response parameters do not enter tau)",
    continuity_880=dict(shifts_um=shifts.tolist(), y_step_edge=y_step.tolist(), y_declared=y_soft.tolist(),
                        dy_dshift_step_edge_od_per_um=dy_step.tolist(), dy_dshift_declared_od_per_um=dy_soft.tolist(),
                        max_abs_dy_dshift_step=float(np.abs(dy_step).max()), max_abs_dy_dshift_declared=float(np.abs(dy_soft).max()),
                        rms_dy_dshift_step=float(np.sqrt(np.mean(dy_step ** 2))), rms_dy_dshift_declared=float(np.sqrt(np.mean(dy_soft ** 2))),
                        note="y_i is continuous in the registration for either response (a shifted integral of a bounded integrand); the step edge only makes the derivative rougher"))
resp["shift_max_rel_diff_analytic_vs_fd1e_4"] = float(np.max(np.abs(an_shift - fd_shift2) / np.maximum(np.abs(fd_shift2), 1e-12)))
resp["soft_max_rel_diff_analytic_vs_fd1e_4"] = float(np.max(np.abs(an_soft - fd_soft2) / np.maximum(np.abs(fd_soft2), 1e-12)))
K_save = K_mir.copy()
K_mir[win, C.I_SHIFT] = an_shift; K_mir[win, C.I_SOFT] = an_soft
resp["sigma_sil_with_analytic_response_columns"] = {f"{rows}_triplet": 3 * joint_sigma(S_B, rows, True)[0] * RHO for rows in ("window", "band")}
resp["sigma_sil_with_analytic_response_columns"]["window_triplet_model"] = 3 * joint_sigma(full_cov("round70", "none", "indep"), "window", True, S_a_inv=C.S_a_inv)[0] * RHO
K_mir[:] = K_save
resp["sigma_sil_with_code_fd_columns"] = {f"{rows}_triplet": 3 * joint_sigma(S_B, rows, True)[0] * RHO for rows in ("window", "band")}
resp["sigma_sil_with_code_fd_columns"]["window_triplet_model"] = 3 * joint_sigma(full_cov("round70", "none", "indep"), "window", True, S_a_inv=C.S_a_inv)[0] * RHO
arch["response_derivatives"] = resp
log(f"   analytic vs FD(1e-4): shift {resp['shift_max_rel_diff_analytic_vs_fd1e_4']:.2e}, soft {resp['soft_max_rel_diff_analytic_vs_fd1e_4']:.2e}; "
    f"silica 3 sigma with analytic response columns {resp['sigma_sil_with_analytic_response_columns']} vs code FD {resp['sigma_sil_with_code_fd_columns']}")
log(f"   shift: FD(code) vs analytic max rel diff {resp['shift_max_rel_diff_fd_vs_analytic']:.2e} (FD 1e-3 vs 1e-4: {resp['shift_max_rel_diff_fd1e_3_vs_fd1e_4']:.2e}); "
    f"soft: {resp['soft_max_rel_diff_fd_vs_analytic']:.2e} ({resp['soft_max_rel_diff_fd1e_3_vs_fd1e_4']:.2e}); 8.80-um dy/dshift {fd_shift[C.i880]:.3f} OD/um; "
    f"step-edge rms |dy/dshift| {resp['continuity_880']['rms_dy_dshift_step']:.2f} vs declared {resp['continuity_880']['rms_dy_dshift_declared']:.2f} OD/um")

# ---------------------------------------------------------------- 4. geometry: multi-tangent silica rows and the coherent O3 term (item 5)
log("[4] geometry: silica row Jacobians at successive tangents, their onion-peel, and a coherent O3 database error")
zb = np.arange(14_000.0, 60_000.0 + DZ_M, DZ_M)                           # tangent heights / shell boundaries, edge convention
n_t = len(zb) - 1
P = np.empty((n_t, n_t))
for i, zt in enumerate(zb[:-1]):
    r2 = (R_EARTH_M + zt) ** 2
    rb, rt = R_EARTH_M + zb[:-1], R_EARTH_M + zb[1:]
    P[i] = 2.0 * (np.sqrt(np.maximum(rt ** 2 - r2, 0)) - np.sqrt(np.maximum(rb ** 2 - r2, 0)))
Ginv = np.linalg.inv(P)
k20 = int(np.argmin(np.abs(zb[:-1] - C.H_TAN_M)))
xnu, xs = C.xnu, C.xs
cfc_x = np.interp(C.nu, xnu, xs, left=0.0, right=0.0) * 1e-4
y_sil = np.empty((n_t, C.n_el)); y_ref = np.empty((n_t, C.n_el)); y_pert = np.empty((n_t, C.n_el))
tau_o3_mean_el = np.empty((n_t, C.n_el)); y_gasonly = np.empty((n_t, C.n_el))
dnu_w = C.dlam_dnu
for i, h in enumerate(zb[:-1]):
    ch = tangent_to_slant_paths(np.array([h]), C.alt)[0]
    t_gas = {g: (ch * C.dens[g]) @ C.fine[g]["sig_z"] for g in C.GASES}
    t_cfc = cfc_x * float((ch * C.dens["cfc12"]).sum())
    sl = C.slant_at(h, C.prof_c)
    t_aer = np.array([np.interp(C.lam_um, C.lam_coarse, s) for s in sl])      # (7, n_nu)
    tau_bg = sum(t_gas.values()) + t_cfc + t_aer[1] + t_aer[2]
    y_ref[i] = C.element_obs(tau_bg)
    y_pert[i] = C.element_obs(tau_bg + 0.02 * t_gas["o3"])
    y_gasonly[i] = C.element_obs(sum(t_gas.values()) + t_cfc)
    # silica row: derivative of y_i with respect to the mass (transmission-weighted slant OD per Tg)
    y_sil[i] = C.element_mean(tau_bg, t_aer[0])
    for j in range(C.n_el):
        m = C.el_id == j
        tau_o3_mean_el[i, j] = np.average(t_gas["o3"][m], weights=dnu_w[m])
# onion-peel of the silica rows -> per-shell extinction per Tg (should reproduce the forward model's local extinction)
alpha_sil_peel = Ginv @ y_sil                                              # (n_t shells, n_el) m^-1 per Tg
t_shell_el = np.array([np.mean(C.shell_c[0][(C.lam_coarse >= c - C.EL_HALF_UM) & (C.lam_coarse < c + C.EL_HALF_UM)]) for c in C.EL_CENTRES_UM])
# the per-shell silica extinction of the forward model at every shell centre (prescribed profile) for the 8.80-um element
z_shell = 0.5 * (zb[:-1] + zb[1:])
prof_sil_880 = np.interp(z_shell, cbt.ALT, C.prof_c[0][:, np.argmin(np.abs(C.lam_coarse - 8.80))])
# coherent O3 database error: delta y at every tangent, peeled, against f alpha_O3(z)
dy = y_pert - y_ref
dalpha = Ginv @ dy                                                         # m^-1 (n_t, n_el)
zc = 0.5 * (zb[:-1] + zb[1:]); atm_c = us_standard_atmosphere(zc)
n_o3_c = tg.number_density_profiles(zc, atm_c.number_density_m3, gases=["o3"])["o3"]
masks = [C.el_id == j for j in range(C.n_el)]
sig_z_o3 = C.fine["o3"]["sig_z"]
alpha_o3_mean = np.empty((n_t, C.n_el))                                    # mean-cross-section local O3 extinction (n_t, n_el)
for i in range(n_t):
    iz_ = int(np.argmin(np.abs(C.alt - zc[i])))
    alpha_o3_mean[i] = n_o3_c[i] * np.array([np.average(sig_z_o3[iz_][m], weights=dnu_w[m]) for m in masks])
f_alpha = 0.02 * alpha_o3_mean
sel = (zc >= 17_500) & (zc <= 23_000)
geo = dict(
    tangents_km=(zb[:-1] / 1e3).tolist(), shell_centres_km=(zc / 1e3).tolist(),
    silica_row_slant_od_per_tg=y_sil.tolist(), silica_row_880_at_20km=float(y_sil[k20, C.i880]),
    silica_peeled_m1_per_tg_880=alpha_sil_peel[:, C.i880].tolist(), forward_model_shell_ext_880_m1_per_tg=prof_sil_880.tolist(),
    peel_over_forward_880_at_20km=float(alpha_sil_peel[k20, C.i880] / prof_sil_880[k20]),
    peel_over_forward_880_17p5_23km=(alpha_sil_peel[sel, C.i880] / prof_sil_880[sel]).tolist(),
    rho_from_rows_880=float(y_sil[k20, C.i880] / (t_shell_el[C.i880] * OD_PER_M1)), rho=RHO,
    coherent_o3_2pct=dict(
        dy_slant_od=dy.tolist(), dalpha_peeled_m1=dalpha.tolist(), f_alpha_o3_mean_m1=f_alpha.tolist(),
        ratio_peeled_over_f_alpha_880_17p5_23km=(dalpha[sel, C.i880] / f_alpha[sel, C.i880]).tolist(),
        ratio_peeled_over_f_alpha_at_20km_per_element=(dalpha[k20] / f_alpha[k20]).tolist(),
        dalpha_880_at_20km_m1=float(dalpha[k20, C.i880]), f_alpha_880_at_20km_m1=float(f_alpha[k20, C.i880]),
        white_conversion_of_dy_880_at_20km_m1=float(dy[k20, C.i880] * M1_PER_OD),
        dy_880_at_20km_od=float(dy[k20, C.i880]), y_ref_880_tangents=y_ref[:, C.i880].tolist(),
        note="a coherent +2 % error on every O3 line, carried through the nonlinear element observable at every tangent (14-60 km, 0.5-km steps) and peeled with the exact path matrix; f alpha_O3 is 0.02 times the mean-cross-section local O3 extinction of the element"))
arch["geometry"] = geo
log(f"   silica row at 20 km, 8.80 um: {geo['silica_row_880_at_20km']:.4f} OD/Tg; peeled/forward shell extinction {geo['peel_over_forward_880_at_20km']:.4f} "
    f"(17.5-23 km: {min(geo['peel_over_forward_880_17p5_23km']):.3f}-{max(geo['peel_over_forward_880_17p5_23km']):.3f}); rho from the rows {geo['rho_from_rows_880']:.3f} (round 70 {RHO:.3f})")
r_ = geo["coherent_o3_2pct"]
log(f"   coherent +2 % O3: dy(20 km, 8.80) = {r_['dy_880_at_20km_od']:.2e} OD; peeled {r_['dalpha_880_at_20km_m1']:.2e} m^-1 vs f alpha {r_['f_alpha_880_at_20km_m1']:.2e} "
    f"(ratio {r_['dalpha_880_at_20km_m1']/r_['f_alpha_880_at_20km_m1']:.2f}; 17.5-23 km {min(r_['ratio_peeled_over_f_alpha_880_17p5_23km']):.2f}-{max(r_['ratio_peeled_over_f_alpha_880_17p5_23km']):.2f}); "
    f"white conversion of the same dy would give {r_['white_conversion_of_dy_880_at_20km_m1']:.2e} m^-1")

# ---------------------------------------------------------------- 5. the exponential scan of Appendix D6 as fine-grid ratios
# the table values (Table F1 / Appendix D6) and their ratios, independent of the archive layout
tab = dict(band_indep=0.108, band_l0p1=0.124, band_l0p25=0.107, window_indep=0.150, window_l0p1=0.159, window_l0p25=0.131)
ratios = dict(band_l0p1_over_indep=tab["band_l0p1"] / tab["band_indep"] - 1, band_l0p25_over_indep=tab["band_l0p25"] / tab["band_indep"] - 1,
              window_l0p1_over_indep=tab["window_l0p1"] / tab["window_indep"] - 1, window_l0p25_over_indep=tab["window_l0p25"] / tab["window_indep"] - 1,
              band_l0p1_over_0p101=tab["band_l0p1"] / 0.101 - 1, band_l0p25_over_0p101=tab["band_l0p25"] / 0.101 - 1,
              window_l0p1_over_0p141=tab["window_l0p1"] / 0.141 - 1, window_l0p25_over_0p141=tab["window_l0p25"] / 0.141 - 1)
arch["exponential_scan_ratios"] = dict(table_values=tab, ratios=ratios,
                                       note="the paper's '+7 to +23 %' mixed the fine-grid correlated values with the design-grid baselines; on the common fine grid the ratios are those listed")
log("[5] exponential-scan ratios on the fine grid: " + ", ".join(f"{k} {100*v:+.1f} %" for k, v in ratios.items() if "indep" in k))

arch["runtime_s"] = time.time() - t_start
with open(ARCH, "w") as f:
    json.dump(arch, f, indent=1)
log("archived ->", ARCH, f"({arch['runtime_s']:.0f} s)")
