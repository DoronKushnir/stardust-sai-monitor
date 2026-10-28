"""closure_final.py -- referee round 7 (paper 1; public port), item 1: the synthetic-truth closure test rerun on
the FINAL retrieval, i.e. the estimator that gives the quoted sensitivities (Table 2 model): the 47 design
elements as slant rows (the 15 window elements with the line-resolved gas state, the 32 outer elements with
the aerosol columns), the heritage triplet, the adopted covariance (radiometric term, per-line database
scatter, Gaussian-correlated discrepancy term) and the flat aerosol prior of reproduce/uncertainty_model.py.

The round-70/74 closure tests (reproduce/element_observable_closure.py, closure_variants.py)
used the window rows, the spectroscopic covariance and an aerosol prior of 1e3 in the parameters' own units,
which binds the coarse-mode radius; they are kept as a separately named diagnostic.

Generative model of each draw: gas profiles, temperature offset, CFC amplitude, source-group scalings,
sulfate state and response perturbations drawn as in round 70; the independent per-line scatter drawn
physically through the line profiles (its covariance is term 2 of the fit); the radiometric noise drawn
per element; the discrepancy term drawn from its declared covariance (measured per-element scatter,
Gaussian inter-element correlation of length 0.14 um), i.e. an implementation closure test of the
declared covariance, not a physically independent discrepancy model.  Retrieval: damped Gauss-Newton
optimal estimation with the nonlinear element model on the window rows and the linear outer and triplet
rows, from the prior state.

Tests: baseline (null, 0.05, 0.1, 0.2 Tg per shell; --ndraw draws, default 100); misspecification with
33 draws each at 0 and 0.1 Tg: truths at twice the prior widths, the Biermann 70 wt% member in the truth,
air-broadening widths x1.1 in the truth, three fixed database realizations held over the draws; and the
spectroscopic-only diagnostic (window rows, spectroscopic covariance, flat prior) for comparison with the
round-70 rows.  Clopper-Pearson 95 % intervals (two-sided) for every fraction; exact one-sided bounds where
stated.

Run from the repo root (about 40 min):  python reproduce/closure_final.py [--ndraw 100] [--seed 76]
Writes outputs/closure_final.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))

import adopted_model_engine as E                                  # noqa: E402
from closure_variants import perturbed_line_widths, alt_member_profiles   # noqa: E402

C, cbt = E.C, E.cbt
OUT = ROOT / "outputs/closure_final.json"
log = E.log
REF = E.REF
RHO = REF.rho
win, outer, n_mir = E.win, E.outer, E.n_mir


def cp(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


# ---------------------------------------------------------------- the final estimator
class Estimator:
    """rows: window (nonlinear), outer (linear aerosol columns), triplet (linear per-shell); covariance and prior given."""

    def __init__(self, S_mir, S_tri, S_a_inv, rows="band"):
        self.rows = rows
        self.idx_out = outer if rows == "band" else np.array([], int)
        self.K_out = REF.K_mir[self.idx_out]                        # (n_out, N_STATE), aerosol columns only
        self.K_tri = REF.K_tri; self.S_a_inv = S_a_inv
        idx = np.r_[win, self.idx_out]
        S = C._blockdiag(S_mir[np.ix_(idx, idx)], S_tri)
        self.Sinv = np.linalg.inv(S); self.S = S; self.idx = idx

    def model(self, x):
        K_win, tau = C.jacobian(x)
        y_win = C.element_obs(tau, shift=float(x[C.I_SHIFT]), soft=float(x[C.I_SOFT]))
        K = np.vstack([K_win, self.K_out, self.K_tri])
        y = np.r_[y_win, self.K_out @ x, self.K_tri @ x]
        return y, K

    def fit(self, y_obs, n_iter=25):
        x = C.x_ref.copy(); hist = []
        for it in range(n_iter):
            y, K = self.model(x)
            S_hat = np.linalg.inv(K.T @ self.Sinv @ K + self.S_a_inv)
            dx = S_hat @ (K.T @ self.Sinv @ (y_obs - y) - self.S_a_inv @ (x - C.x_ref))
            step = min(1.0, 0.5 / max(float(np.abs(dx[C.GAS_IDX]).max()), 1e-12), 10.0 / max(abs(float(dx[C.I_T])), 1e-12),
                       0.003 / max(abs(float(dx[C.I_SHIFT])), 1e-12))
            x = x + step * dx
            x[C.GAS_IDX] = np.clip(x[C.GAS_IDX], 0.05, 5.0); x[C.I_SOFT] = float(np.clip(x[C.I_SOFT], 0.001, 0.02))
            hist.append((step, float(np.abs(step * dx[C.GAS_IDX]).max()), float(np.abs(step * dx[C.I_SIL]))))
            if hist[-1][1] < 1e-5 and hist[-1][2] < 1e-6:
                break
        return x, S_hat, dict(n_iter=len(hist), converged=bool(hist[-1][1] < 1e-5 and hist[-1][2] < 1e-6), min_damping=float(min(h[0] for h in hist)))


S_tri = REF.S_tri
S_adopted = REF.cov_mir()                                                 # radiometric + line + discrepancy (Gaussian)
S_noise_disc = E.np.diag((np.exp(REF.y_mir / 2.0) / E.S_CAP) ** 2) + E.np.outer(E.s_meas_od, E.s_meas_od) * E.R_GAUSS   # drawn terms
S_spec = REF.cov_mir(disc_level="none")                                   # spectroscopic model (radiometric + line)
FINAL = Estimator(S_adopted, S_tri, E.S_A_INV, "band")
WINDOW_FINAL = Estimator(S_adopted, S_tri, E.S_A_INV, "window")
SPEC_FLAT = Estimator(S_spec, S_tri, E.S_A_INV, "window")               # spectroscopic-only diagnostic, flat prior
L_draw_band = np.linalg.cholesky(S_noise_disc)
L_draw_win = np.linalg.cholesky(S_noise_disc[np.ix_(win, win)])
L_draw_spec = np.linalg.cholesky(np.diag((np.exp(REF.y_mir[win] / 2.0) / E.S_CAP) ** 2))
Lsa = {g: np.linalg.cholesky(C.S_a[C.I_GAS[g], C.I_GAS[g]]) for g in C.GASES}
sig_trip = np.sqrt(np.diag(S_tri)); sig_grp = np.array([s for g, r, m, s in C.GROUPS])


def draw_truth(rng, prior_scale=1.0, T_sig=C.T_PRIOR_K, cfc_sig=C.CFC_PRIOR, grp_fixed=None, response_pert=True, resp_sigma_um=0.003):
    xt = C.x_ref.copy()
    for g in C.GASES:
        xt[C.I_GAS[g]] = np.clip(1.0 + prior_scale * (Lsa[g] @ rng.standard_normal(C.NN)), 0.2, 3.0)
    xt[C.I_T] = T_sig * rng.standard_normal(); xt[C.I_CFC] = 1.0 + cfc_sig * rng.standard_normal()
    xt[C.I_GRP] = sig_grp * rng.standard_normal(C.NG) if grp_fixed is None else grp_fixed
    xt[C.I_AER] = [0.0, np.exp(rng.uniform(-0.7, 0.7)), np.exp(rng.uniform(-0.7, 0.7)),
                   rng.uniform(-0.2, 0.2) * cbt.CAL_QUIET[0][0], rng.uniform(-0.1, 0.1) * cbt.CAL_QUIET[0][1],
                   rng.uniform(-0.2, 0.2) * cbt.CAL_QUIET[1][0], rng.uniform(-0.1, 0.1) * cbt.CAL_QUIET[1][1]]
    xt[C.I_SHIFT] = rng.normal(0.0, resp_sigma_um) if response_pert else 0.0
    xt[C.I_SOFT] = float(np.clip((C.RESP_SOFT0 + rng.normal(0.0, 0.001)) if response_pert else C.RESP_SOFT0, 0.001, 0.02))
    return xt


def draw_dtau(rng, sigma_eta=C.SIGMA_ETA_BASE):
    dtau = np.zeros(C.n_nu)
    for g in C.GASES:
        dtau += C.Pline[g].T @ (sigma_eta * rng.standard_normal(len(C.meta[g]["nu"]))).astype(np.float32)
    return dtau


def truth_rows(xt, dtau, tau_aer=None, Kt_true=None, dtau_shape=None):
    """true element observables of the window (nonlinear), the outer elements and the triplet (linear)."""
    tau_aer = C.tau_aer_fine if tau_aer is None else tau_aer
    tau_t = np.zeros(C.n_nu)
    for g in C.GASES:
        tau_t += (xt[C.I_GAS[g]] @ C.tau_nodes[g]) * (1.0 + xt[C.I_T] * C.a_T[g])
    tau_t = tau_t + xt[C.I_GRP] @ C.tau_group + xt[C.I_CFC] * C.tau_cfc + xt[C.I_AER] @ tau_aer + dtau + (0.0 if dtau_shape is None else dtau_shape)
    y_win = C.element_obs(tau_t, shift=float(xt[C.I_SHIFT]), soft=float(xt[C.I_SOFT]))
    Kt = REF.K_tri if Kt_true is None else Kt_true
    return y_win, REF.K_mir[outer] @ xt, Kt @ xt


def run(est, rng, ndraw, masses, L_mir, label, prior_scale=1.0, T_sig=C.T_PRIOR_K, cfc_sig=C.CFC_PRIOR, alt=None, dtau_fixed=None,
        grp_fixed=None, dtau_shape=None):
    """Monte Carlo on one estimator: returns per-mass summaries (Tg per shell at 20 km)."""
    band = est.rows == "band"
    tau_aer, Kt_true = None, None
    if alt is not None:
        prof_alt_c, prof_alt_t = alt
        slant_alt = C.slant_at(C.H_TAN_M, prof_alt_c)
        tau_aer = C.tau_aer_fine.copy(); tau_aer[1:] = np.array([np.interp(C.lam_um, C.lam_coarse, s) for s in slant_alt[1:]])
        Kt_true = np.zeros((3, C.N_STATE)); Kt_true[:, C.I_AER] = prof_alt_t[:, cbt.iz, :].T; Kt_true[:, C.I_SIL] = REF.K_tri[:, C.I_SIL]
    res = {m: dict(A=[], s=[], conv=[], it=[]) for m in masses}
    t0 = time.time()
    for d in range(ndraw):
        xt = draw_truth(rng, prior_scale, T_sig, cfc_sig, grp_fixed)
        dtau = draw_dtau(rng) if dtau_fixed is None else dtau_fixed
        for M_shell in masses:
            xt_m = xt.copy(); xt_m[C.I_SIL] = M_shell / RHO
            y_win, y_out, y_tri = truth_rows(xt_m, dtau, tau_aer, Kt_true, dtau_shape)
            eps = L_mir @ rng.standard_normal(L_mir.shape[0])
            if band:
                y_obs = np.r_[y_win + eps[win], y_out + eps[outer], y_tri + sig_trip * rng.standard_normal(3)]
            else:
                y_obs = np.r_[y_win + eps, y_tri + sig_trip * rng.standard_normal(3)]
            x_hat, S_hat, rep = est.fit(y_obs)
            res[M_shell]["A"].append(float(x_hat[C.I_SIL] * RHO)); res[M_shell]["s"].append(float(np.sqrt(S_hat[C.I_SIL, C.I_SIL]) * RHO))
            res[M_shell]["conv"].append(rep["converged"]); res[M_shell]["it"].append(rep["n_iter"])
    summ = {}
    for M_shell, r in res.items():
        A = np.array(r["A"]); s = np.array(r["s"]); z = (A - M_shell) / s; n = len(A)
        k1, k2, k3, k4 = int(np.sum(np.abs(z) < 1)), int(np.sum(np.abs(z) < 2)), int(np.sum(A / s > 3.0)), int(np.sum(A / s > 4.54))
        summ[str(M_shell)] = dict(n=n, bias_mean=float(np.mean(A - M_shell)), bias_median=float(np.median(A - M_shell)), sd=float(np.std(A, ddof=1)),
                                  robust_sd=float(1.4826 * np.median(np.abs(A - np.median(A)))), posterior_sigma_median=float(np.median(s)),
                                  sd_over_posterior=float(np.std(A, ddof=1) / np.median(s)),
                                  coverage_1sig=k1 / n, coverage_1sig_ci95=cp(k1, n), coverage_2sig=k2 / n, coverage_2sig_ci95=cp(k2, n),
                                  p_detect_3sig=k3 / n, p_detect_3sig_ci95=cp(k3, n), p_detect_4p54sig=k4 / n, p_detect_4p54sig_ci95=cp(k4, n),
                                  n_converged=int(np.sum(r["conv"])), median_iterations=float(np.median(r["it"])), masses_tg=A.tolist())
        if M_shell == 0.0:
            summ["0.0"]["null_exceed_3sig_one_sided_95_upper"] = float(beta.ppf(0.95, k3 + 1, n - k3)) if k3 < n else 1.0
        log(f"  {label:44s} M={M_shell:<5}: bias {summ[str(M_shell)]['bias_median']:+.4f} sd {summ[str(M_shell)]['sd']:.4f} post {summ[str(M_shell)]['posterior_sigma_median']:.4f} "
            f"cov1 {k1/n:.2f} cov2 {k2/n:.2f} P3 {k3/n:.2f} P4.54 {k4/n:.2f} conv {summ[str(M_shell)]['n_converged']}/{n} ({time.time()-t0:.0f} s)")
    return summ


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ndraw", type=int, default=100); ap.add_argument("--seed", type=int, default=76)
    ap.add_argument("--nvar", type=int, default=33); args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    arch = dict(script="reproduce/closure_final.py", round=76, seed=args.seed, ndraw=args.ndraw, nvar=args.nvar, rho=RHO,
                estimator=dict(rows="47 design elements (15 window with the gas state, 32 outer with the aerosol columns) + heritage triplet",
                               covariance="Table 2 primary model: radiometric e^{y/2}/2000, per-line scatter sigma_eta = 0.02, discrepancy term at the measured scatter with Gaussian correlation l = 0.142 um",
                               prior="round-70 gas, temperature, CFC, source-group and response priors; flat (1e6) on the seven aerosol parameters",
                               posterior_3sig_band_tg=3 * E.joint_sigma(REF) * RHO, posterior_3sig_window_tg=3 * E.joint_sigma(REF, rows="window") * RHO,
                               generative="line scatter drawn per line through the line profiles; radiometric noise per element; discrepancy term drawn from its declared Gaussian covariance (implementation closure, not an independent discrepancy model)"),
                tests={})
    t0 = time.time()
    masses = (0.0, 0.05, 0.1, 0.2)
    arch["tests"]["final_band"] = run(FINAL, rng, args.ndraw, masses, L_draw_band, "final estimator, band")
    arch["tests"]["final_window"] = run(WINDOW_FINAL, rng, args.ndraw, masses, L_draw_win, "final estimator, window")
    arch["tests"]["spectroscopic_flat_window"] = run(SPEC_FLAT, rng, args.ndraw, masses, L_draw_spec, "spectroscopic-only diagnostic, window, flat prior")
    log(f"baseline done, {time.time()-t0:.0f} s")
    # misspecification on the final band estimator
    v = {}
    v["out_of_prior_x2"] = run(FINAL, rng, args.nvar, (0.0, 0.1), L_draw_band, "truths at 2x the prior widths", prior_scale=2.0, T_sig=4.0, cfc_sig=0.06)
    v["alt_sulfate_member_B70T215"] = run(FINAL, rng, args.nvar, (0.0, 0.1), L_draw_band, "Biermann member in the truth", alt=alt_member_profiles("B70T215"))
    dshape = perturbed_line_widths("o3", 1.1) + perturbed_line_widths("n2o", 1.1)
    v["alt_line_widths_x1p1"] = run(FINAL, rng, args.nvar, (0.0, 0.1), L_draw_band, "air-broadening widths x1.1 in the truth", dtau_shape=dshape)
    dbs = {}
    for k in range(3):
        dtau = draw_dtau(rng); grp = sig_grp * rng.standard_normal(C.NG)
        dbs[f"db{k}"] = run(FINAL, rng, args.nvar, (0.0, 0.1), L_draw_band, f"fixed database {k+1}", dtau_fixed=dtau, grp_fixed=grp)
    means = [dbs[f"db{k}"]["0.0"]["bias_mean"] for k in range(3)]
    v["fixed_database"] = dict(per_database=dbs, spread_of_db_means_null=float(np.std(means, ddof=1)),
                               within_db_sd_median=float(np.median([dbs[f"db{k}"]["0.0"]["sd"] for k in range(3)])))
    arch["tests"]["misspecification_final_band"] = v
    arch["runtime_s"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(arch, f, indent=1)
    log("archived ->", OUT, f"({arch['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
