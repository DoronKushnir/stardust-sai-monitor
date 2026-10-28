"""gas_budget_conditioning.py -- referee round 7 (paper 1; public port), items 2 and 3: the conditioning of the
gas-removal budget and the decomposition of the posterior silica uncertainty.

 1. The gas optical depth predicted at the 8.80-um element: (a) the known-aerosol diagnostic of round 70
    (window rows, spectroscopic covariance, round-70 priors, the seven aerosol and two response parameters
    held known), the 5.12e-4 OD of the paper; (b) the same diagnostic with the aerosol and response
    parameters in the state (round-70 prior 1e3 on the aerosol); (c) the gas prediction through the gain of
    the FINAL retrieval (47 rows + triplet, adopted covariance, flat aerosol prior), i.e. the posterior
    uncertainty of the gas part of the 8.80-um element observable, decomposed into the propagated
    measurement covariance and the prior contribution.  Each is labelled by its conditioning.
 2. The posterior variance of the silica amplitude of the final retrieval, band and window, split into the
    propagated measurement covariance G S_eps G^T and the prior contribution (I - A) S_a (I - A)^T, and
    further into the three mid-infrared terms and the triplet.
 3. The gas-removal error at every window element through the final gain, against the discrepancy term, for
    the statement that the ACE scatter is a multiple of the co-retrieval error.

Run from the repo root (seconds):  python reproduce/gas_budget_conditioning.py
Writes outputs/gas_budget_conditioning.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))
import adopted_model_engine as E                                  # noqa: E402

C = E.C
OUT = ROOT / "outputs/gas_budget_conditioning.json"
REF = E.REF; RHO = REF.rho
log = E.log
res = dict(script="reproduce/gas_budget_conditioning.py", round=76)

# ---- 1. the gas budget under three conditionings
b_known = C.removal_budget(C.SNR_REF, C.SIGMA_ETA_BASE, with_aerosol_state=False)
b_joint70 = C.removal_budget(C.SNR_REF, C.SIGMA_ETA_BASE, with_aerosol_state=True)
res["gas_budget_880"] = dict(
    known_aerosol_diagnostic=dict(sigma_total_od=b_known["sigma_total"], photon=b_known["photon"], line=b_known["line"], prior=b_known["prior_smoothing"],
                                  conditioning="window rows, spectroscopic covariance, round-70 priors; the seven aerosol and two response parameters known"),
    joint_prior=dict(sigma_total_od=b_joint70["sigma_total"], photon=b_joint70["photon"], line=b_joint70["line"], prior=b_joint70["prior_smoothing"],
                             conditioning="window rows, spectroscopic covariance; aerosol (prior 1e3) and response parameters in the state"))
# (c) through the gain of the final retrieval: predictand = the gas part of y_880 (gas, T, CFC, group columns)
s_b, S_hat, G, K, S = E.joint_sigma(REF, rows="band", return_all=True)
A = G @ K
k = np.zeros(C.N_STATE); k[:C.I_SIL] = REF.K_mir[C.i880, :C.I_SIL]        # gas, T, CFC, groups at the 8.80-um element
IA = np.eye(C.N_STATE) - A
S_a = np.linalg.inv(E.S_A_INV)
v_meas = float(k @ G @ S @ G.T @ k); v_prior = float(k @ IA @ S_a @ IA.T @ k); v_post = float(k @ S_hat @ k)
# the line term's own contribution to the element (its covariance diagonal) enters the removal error as in round 70
Cl = E.np.zeros((E.n_mir, E.n_mir)); Cl[np.ix_(E.win, E.win)] = C.line_covariance(C.SIGMA_ETA_BASE)
e = np.zeros(K.shape[0]); e[C.i880] = 1.0
Gk = G.T @ k
v_line_full = float(Gk @ C._blockdiag(Cl, np.zeros((3, 3))) @ Gk + e[:E.n_mir] @ Cl @ e[:E.n_mir] - 2.0 * Gk[:E.n_mir] @ Cl @ e[:E.n_mir])
S_full = C._blockdiag(REF.cov_mir(), REF.S_tri)
S_noise = C._blockdiag(np.diag((np.exp(REF.y_mir / 2.0) / E.S_CAP) ** 2), np.zeros((3, 3)))
S_disc = C._blockdiag(REF.cov_mir(disc_level="measured") - REF.cov_mir(disc_level="none"), np.zeros((3, 3)))
S_trip = C._blockdiag(np.zeros((E.n_mir, E.n_mir)), REF.S_tri)
res["gas_budget_880"]["final_retrieval_gain"] = dict(
    sigma_posterior_od=float(np.sqrt(v_post)), from_measurement_covariance=float(np.sqrt(v_meas)), from_prior=float(np.sqrt(v_prior)),
    photon_term=float(np.sqrt(Gk @ S_noise @ Gk)), discrepancy_term=float(np.sqrt(Gk @ S_disc @ Gk)), triplet=float(np.sqrt(Gk @ S_trip @ Gk)),
    line_term_including_the_elements_own_error=float(np.sqrt(max(v_line_full, 0.0))),
    conditioning="47 design elements + triplet, adopted covariance (Table 2), flat aerosol prior; aerosol and response parameters co-retrieved",
    note="sigma of the gas part of the 8.80-um element observable predicted by the posterior state; the measurement part is G S G^T, the prior part (I-A) S_a (I-A)^T")
log("gas budget at 8.80 um [OD]: known-aerosol diagnostic %.3e (photon %.3e, line %.3e, prior %.3e); joint with the round-70 prior %.3e; "
    "through the final gain: posterior %.3e (measurement %.3e, prior %.3e; photon %.3e, discrepancy %.3e, triplet %.3e; line incl. own %.3e)" % (
        b_known["sigma_total"], b_known["photon"], b_known["line"], b_known["prior_smoothing"], b_joint70["sigma_total"],
        np.sqrt(v_post), np.sqrt(v_meas), np.sqrt(v_prior), res["gas_budget_880"]["final_retrieval_gain"]["photon_term"],
        res["gas_budget_880"]["final_retrieval_gain"]["discrepancy_term"], res["gas_budget_880"]["final_retrieval_gain"]["triplet"],
        res["gas_budget_880"]["final_retrieval_gain"]["line_term_including_the_elements_own_error"]))
# per-window-element gas prediction error through the final gain, against the discrepancy term
per_el = []
for j in E.win:
    kj = np.zeros(C.N_STATE); kj[:C.I_SIL] = REF.K_mir[j, :C.I_SIL]
    per_el.append(float(np.sqrt(kj @ S_hat @ kj)))
res["gas_budget_880"]["per_window_element_posterior_od"] = per_el
res["gas_budget_880"]["discrepancy_term_per_window_element_od"] = E.s_meas_od[E.win].tolist()
res["gas_budget_880"]["ratio_discrepancy_over_known_aerosol_880"] = float(E.s_meas_od[C.i880] / b_known["sigma_total"])
res["gas_budget_880"]["ratio_discrepancy_over_final_gain_880"] = float(E.s_meas_od[C.i880] / np.sqrt(v_post))

# ---- 2. decomposition of the silica posterior (band and window)
dec = {}
for rows in ("band", "window"):
    s_, S_hat_, G_, K_, S_ = E.joint_sigma(REF, rows=rows, return_all=True)
    A_ = G_ @ K_; IA_ = np.eye(C.N_STATE) - A_
    g = G_[C.I_SIL]                                                        # gain row of the silica amplitude
    n_m = len(E.win) if rows == "window" else E.n_mir
    idx = E.win if rows == "window" else np.arange(E.n_mir)
    Sn = C._blockdiag(np.diag((np.exp(REF.y_mir[idx] / 2.0) / E.S_CAP) ** 2), np.zeros((3, 3)))
    Sl = C._blockdiag(Cl[np.ix_(idx, idx)], np.zeros((3, 3)))
    Sd = C._blockdiag((REF.cov_mir(disc_level="measured") - REF.cov_mir(disc_level="none"))[np.ix_(idx, idx)], np.zeros((3, 3)))
    St = C._blockdiag(np.zeros((n_m, n_m)), REF.S_tri)
    v = dict(posterior=float(S_hat_[C.I_SIL, C.I_SIL]), measurement=float(g @ S_ @ g), prior=float(IA_[C.I_SIL] @ S_a @ IA_[C.I_SIL]),
             photon=float(g @ Sn @ g), line=float(g @ Sl @ g), discrepancy=float(g @ Sd @ g), triplet=float(g @ St @ g))
    dec[rows] = {k_: float(np.sqrt(max(x, 0.0)) * RHO) for k_, x in v.items()}
    dec[rows]["check_sum"] = float(np.sqrt(v["measurement"] + v["prior"]) * RHO)
    log(f"silica posterior {rows}: 1 sigma per shell {dec[rows]['posterior']:.4f} Tg = measurement {dec[rows]['measurement']:.4f} (photon {dec[rows]['photon']:.4f}, "
        f"line {dec[rows]['line']:.4f}, discrepancy {dec[rows]['discrepancy']:.4f}, triplet {dec[rows]['triplet']:.4f}) + prior {dec[rows]['prior']:.4f} in quadrature (check {dec[rows]['check_sum']:.4f})")
res["silica_posterior_decomposition_1sig_per_shell_tg"] = dec
with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
log("archived ->", OUT)
