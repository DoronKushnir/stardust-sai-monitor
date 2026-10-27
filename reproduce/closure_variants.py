"""closure_variants.py -- referee round 6 (paper 1; public port), item 3: additions to the synthetic-truth
closure test of reproduce/element_observable_closure.py (imported for its forward model,
priors, covariance and iterated optimal-estimation fit).

 1. Fixed database realization: one draw of the independent per-line scatter and of the
    source-group scalings is held fixed across N soundings whose gas profiles, temperature,
    sulfate state, response errors and photon noise are drawn afresh; repeated for several
    databases.  The mean retrieved mass per database is the bias a mission would carry with
    that database; its spread across databases is the round-70 'database' scatter.
 2. Truths outside the priors: gas profiles drawn from twice the prior widths (O3 60 % per
    node), temperature offset 4 K, CFC 6 %.
 3. An alternative sulfate family in the truth: the Biermann 70 wt%, 215 K member optics in
    place of the Lund Myhre 65 wt%, 223 K member of the retrieval basis (the member-optics
    bracket of Table 6), at the drawn amplitudes and sizes.
 4. An alternative line-shape family in the truth: the air-broadening widths of every O3 and
    N2O line scaled by 1.1 (the truth's line profiles differ from the retrieval's).
 5. Convergence bookkeeping (iterations, step sizes, failures) and seeds for every variant,
    and Clopper-Pearson 95 % intervals for every coverage and detection fraction, including
    the round-70 tables (re-derived from the archived counts).

Masses are Tg per 0.5-km shell at 20 km (slant amplitude x rho).  Run from the repo root
(about 30 min):  python reproduce/closure_variants.py [--ndraw 33] [--seed 74]
Writes outputs/closure_variants.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import wofz
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))

import element_observable_closure as C                           # noqa: E402
import calibrated_background_thresholds as cbt                            # noqa: E402

OUT = ROOT / "outputs" / "closure_variants.json"


def log(*a):
    print(*a, flush=True)


def cp(k, n, alpha=0.05):
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return [float(lo), float(hi)]


# ---------------------------------------------------------------- a Gauss-Newton fit that also reports its convergence
def gn_fit_report(y_obs, K_tri, S_eps, n_iter=25):
    x = C.x_ref.copy()
    Sinv = np.linalg.inv(S_eps)
    hist = []
    for it in range(n_iter):
        K, tau = C.jacobian(x)
        y = C.element_obs(tau, shift=float(x[C.I_SHIFT]), soft=float(x[C.I_SOFT]))
        if K_tri is not None:
            K = np.vstack([K, K_tri]); yv = np.r_[y, K_tri @ x]
        else:
            yv = y
        S_hat = np.linalg.inv(K.T @ Sinv @ K + C.S_a_inv)
        dx = S_hat @ (K.T @ Sinv @ (y_obs - yv) - C.S_a_inv @ (x - C.x_ref))
        step = min(1.0, 0.5 / max(float(np.abs(dx[C.GAS_IDX]).max()), 1e-12), 10.0 / max(abs(float(dx[C.I_T])), 1e-12),
                   0.003 / max(abs(float(dx[C.I_SHIFT])), 1e-12))
        x = x + step * dx
        x[C.GAS_IDX] = np.clip(x[C.GAS_IDX], 0.05, 5.0)
        x[C.I_SOFT] = float(np.clip(x[C.I_SOFT], 0.001, 0.02))
        last = float(np.abs(step * dx[C.I_SIL]))
        hist.append((step, float(np.abs(step * dx[C.GAS_IDX]).max()), last))
        if np.abs(step * dx[C.GAS_IDX]).max() < 1e-5 and last < 1e-6:
            break
    converged = (hist[-1][1] < 1e-5 and hist[-1][2] < 1e-6)
    return x, S_hat, dict(n_iter=len(hist), converged=bool(converged), last_step_sil_slant=hist[-1][2], min_damping=float(min(h[0] for h in hist)))


# ---------------------------------------------------------------- alternative truths
def perturbed_line_widths(gas, scale):
    """tau(nu) difference (perturbed minus nominal) from scaling the air-broadening widths of every line of `gas`."""
    nu_l, sw, ga, na, el, de, code, ref = C.load_lines(gas)
    T, P = C.T_eff[gas], C.P_eff[gas]
    mol, iso, M = C.MOL[gas]
    Q = C.hapi.partitionSum(mol, iso, T) / C.hapi.partitionSum(mol, iso, 296.0)
    S_T = (sw / Q * np.exp(-C.C2 * el / T) / np.exp(-C.C2 * el / 296.0) * (1 - np.exp(-C.C2 * nu_l / T)) / (1 - np.exp(-C.C2 * nu_l / 296.0)))
    alpha_D = 3.5812e-7 * nu_l * np.sqrt(T / M)
    nu0 = nu_l + de * P
    off = np.arange(-C.LINE_WIN_CM, C.LINE_WIN_CM + C.STEP / 2, C.STEP); n_off = len(off); ioff = np.round(off / C.STEP).astype(int)
    dtau = np.zeros(C.n_nu)
    for s in range(0, len(nu_l), 1000):
        sl = slice(s, s + 1000)
        c0 = np.round((nu0[sl] - C.NU_LO) / C.STEP).astype(int)
        idx = c0[:, None] + ioff[None, :]
        ok = (idx >= 0) & (idx < C.n_nu)
        dnu_loc = (C.NU_LO + idx * C.STEP) - nu0[sl][:, None]
        sD = alpha_D[sl][:, None] / np.sqrt(2 * np.log(2))
        out = np.zeros_like(dnu_loc)
        for gsc, sign in ((scale, +1.0), (1.0, -1.0)):
            gam_L = ga[sl] * P * (296.0 / T) ** na[sl] * gsc
            z = (dnu_loc + 1j * gam_L[:, None]) / (sD * np.sqrt(2))
            V = np.real(wofz(z)) / (sD * np.sqrt(2 * np.pi))
            out += sign * (S_T[sl][:, None] * V) * C.col[gas] * 1e-4
        np.add.at(dtau, idx[ok], out[ok])
    return dtau


def alt_member_profiles(member="B70T215"):
    """per-shell extinction of the two sulfate modes (and their PSD derivatives) with another member optics,
    on the window fine grid and the triplet, in the same (7, nz, n_wl) layout as C.prof_c / C.prof_t."""
    ri, wt, Tm = cbt.member_ri(member)
    bg = cbt.TwoComponentBackground(cbt.glossac_2025N_quiet_profile(), cbt.CAL_QUIET, ri, wt, Tm)

    def prof(layer, wl_m):
        return layer.extinction_profile_m1(wl_m)

    def cols(wl_m):
        def jac(i):
            r0, s0, _ = cbt.CAL_QUIET[i]
            dr, ds = 0.05 * r0, 0.05 * s0
            return ((prof(bg.column(i, dr=+dr), wl_m) - prof(bg.column(i, dr=-dr), wl_m)) / (2 * dr),
                    (prof(bg.column(i, ds=+ds), wl_m) - prof(bg.column(i, ds=-ds), wl_m)) / (2 * ds))
        P = [np.zeros_like(prof(bg.column(0), wl_m)), prof(bg.column(0), wl_m), prof(bg.column(1), wl_m)]
        jrf, jsf = jac(0); jrc, jsc = jac(1)
        return np.array(P + [jrf, jsf, jrc, jsc])
    return cols(C.lam_coarse * 1e-6), cols(C.WL_TRIPLET_NM * 1e-9)


# ---------------------------------------------------------------- the Monte Carlo
def summarize(res, M_shell):
    A = np.array(res["A"]); s = np.array(res["s"]); z = (A - M_shell) / s
    n = len(A)
    k1, k2, k3 = int(np.sum(np.abs(z) < 1)), int(np.sum(np.abs(z) < 2)), int(np.sum(np.abs(z) < 3))
    d3, d454 = int(np.sum(A / s > 3.0)), int(np.sum(A / s > 4.54))
    conv = res["conv"]
    return dict(n=n, bias_mean=float(np.mean(A - M_shell)), bias_median=float(np.median(A - M_shell)),
                sd=float(np.std(A, ddof=1)), robust_sd=float(1.4826 * np.median(np.abs(A - np.median(A)))),
                posterior_sigma_median=float(np.median(s)), sd_over_posterior=float(np.std(A, ddof=1) / np.median(s)),
                coverage_1sig=k1 / n, coverage_1sig_ci95=cp(k1, n), coverage_2sig=k2 / n, coverage_2sig_ci95=cp(k2, n),
                coverage_3sig=k3 / n, coverage_3sig_ci95=cp(k3, n),
                p_detect_3sig=d3 / n, p_detect_3sig_ci95=cp(d3, n), p_detect_4p54sig=d454 / n, p_detect_4p54sig_ci95=cp(d454, n),
                n_converged=int(sum(c["converged"] for c in conv)), n_iter_median=float(np.median([c["n_iter"] for c in conv])),
                n_iter_max=int(max(c["n_iter"] for c in conv)), min_damping=float(min(c["min_damping"] for c in conv)),
                mean_z=float(np.mean(z)), sd_z=float(np.std(z, ddof=1)))


def run_variant(ndraw, rng, masses=(0.0, 0.1), prior_scale=1.0, T_sig=C.T_PRIOR_K, cfc_sig=C.CFC_PRIOR,
                alt_prof=None, dtau_fixed=None, grp_fixed=None, dtau_shape=None, snr=C.SNR_REF, sigma_eta=C.SIGMA_ETA_BASE,
                response_pert=True, resp_sigma_um=0.003):
    S_eps_el = C.photon_cov(C.y0, snr) + C.line_covariance(sigma_eta)
    Kt, St = C.triplet_rows()
    S_eps = C._blockdiag(S_eps_el, St)
    Lsa = {g: np.linalg.cholesky(C.S_a[C.I_GAS[g], C.I_GAS[g]]) for g in C.GASES}
    sig_trip = np.sqrt(np.diag(St)); sig_grp = np.array([s for g, r, m, s in C.GROUPS])
    if alt_prof is not None:                       # truth sulfate columns from another member (slant at 20 km on the fine grid)
        prof_alt_c, prof_alt_t = alt_prof
        slant_alt = C.slant_at(C.H_TAN_M, prof_alt_c)
        tau_aer_alt = C.tau_aer_fine.copy()
        tau_aer_alt[1:] = np.array([np.interp(C.lam_um, C.lam_coarse, s) for s in slant_alt[1:]])
        shell_trip_alt = prof_alt_t[:, cbt.iz, :]
        Kt_true = np.zeros((3, C.N_STATE)); Kt_true[:, C.I_AER] = shell_trip_alt.T; Kt_true[:, C.I_SIL] = Kt[:, C.I_SIL]
    else:
        tau_aer_alt, Kt_true = C.tau_aer_fine, Kt
    res = {m: dict(A=[], s=[], conv=[]) for m in masses}
    for d in range(ndraw):
        xt = C.x_ref.copy()
        for g in C.GASES:
            xt[C.I_GAS[g]] = np.clip(1.0 + prior_scale * (Lsa[g] @ rng.standard_normal(C.NN)), 0.2, 3.0)
        xt[C.I_T] = T_sig * rng.standard_normal()
        xt[C.I_CFC] = 1.0 + cfc_sig * rng.standard_normal()
        xt[C.I_GRP] = sig_grp * rng.standard_normal(C.NG) if grp_fixed is None else grp_fixed
        xt[C.I_AER] = [0.0, np.exp(rng.uniform(-0.7, 0.7)), np.exp(rng.uniform(-0.7, 0.7)),
                       rng.uniform(-0.2, 0.2) * cbt.CAL_QUIET[0][0], rng.uniform(-0.1, 0.1) * cbt.CAL_QUIET[0][1],
                       rng.uniform(-0.2, 0.2) * cbt.CAL_QUIET[1][0], rng.uniform(-0.1, 0.1) * cbt.CAL_QUIET[1][1]]
        if dtau_fixed is None:
            dtau = np.zeros(C.n_nu)
            for g in C.GASES:
                dtau += C.Pline[g].T @ (sigma_eta * rng.standard_normal(len(C.meta[g]["nu"]))).astype(np.float32)
        else:
            dtau = dtau_fixed.copy()
        if dtau_shape is not None:
            dtau = dtau + dtau_shape
        xt[C.I_SHIFT] = rng.normal(0.0, resp_sigma_um) if response_pert else 0.0
        xt[C.I_SOFT] = float(np.clip((C.RESP_SOFT0 + rng.normal(0.0, 0.001)) if response_pert else C.RESP_SOFT0, 0.001, 0.02))
        for M_shell in masses:
            xt_m = xt.copy(); xt_m[C.I_SIL] = M_shell / C.RHO_SLANT_TO_SHELL
            # truth tau with the (possibly alternative) sulfate columns
            tau_t = np.zeros(C.n_nu)
            for g in C.GASES:
                tau_t += (xt_m[C.I_GAS[g]] @ C.tau_nodes[g]) * (1.0 + xt_m[C.I_T] * C.a_T[g])
            tau_t = tau_t + xt_m[C.I_GRP] @ C.tau_group + xt_m[C.I_CFC] * C.tau_cfc + xt_m[C.I_AER] @ tau_aer_alt + dtau
            y_t = C.element_obs(tau_t, shift=float(xt_m[C.I_SHIFT]), soft=float(xt_m[C.I_SOFT]))
            y_obs = y_t + rng.standard_normal(C.n_el) * np.exp(y_t / 2.0) / snr
            y_obs = np.r_[y_obs, Kt_true @ xt_m + sig_trip * rng.standard_normal(3)]
            x_hat, S_hat, rep = gn_fit_report(y_obs, Kt, S_eps)
            res[M_shell]["A"].append(float(x_hat[C.I_SIL] * C.RHO_SLANT_TO_SHELL))
            res[M_shell]["s"].append(float(np.sqrt(S_hat[C.I_SIL, C.I_SIL]) * C.RHO_SLANT_TO_SHELL))
            res[M_shell]["conv"].append(rep)
    return {str(m): summarize(r, m) for m, r in res.items()}, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndraw", type=int, default=33)
    ap.add_argument("--seed", type=int, default=74)
    ap.add_argument("--ndb", type=int, default=5, help="number of fixed database realizations")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    arch = dict(script="reproduce/closure_variants.py", round=74, seed=args.seed, ndraw=args.ndraw,
                note="masses Tg per 0.5-km shell at 20 km; retrieval: damped Gauss-Newton optimal estimation from the prior "
                     "(round 70: 25 iterations max, convergence when the gas nodes move < 1e-5 and the silica amplitude < 1e-6 slant Tg)")
    t0 = time.time()
    # ---- 0. the round-70 archive with binomial intervals
    r70 = json.load(open(ROOT / "outputs/element_observable_closure.json"))["monte_carlo"]
    ci = {}
    for key, v in r70.items():
        ci[key] = {}
        for m, r in v.items():
            n = r["n"]
            ci[key][m] = dict(n=n, coverage_1sig=r["coverage_1sig"], coverage_1sig_ci95=cp(int(round(r["coverage_1sig"] * n)), n),
                              coverage_2sig=r["coverage_2sig"], coverage_2sig_ci95=cp(int(round(r["coverage_2sig"] * n)), n),
                              p_detect_3sig=r["p_detect_3sig"], p_detect_3sig_ci95=cp(int(round(r["p_detect_3sig"] * n)), n),
                              p_detect_4p54sig=r["p_detect_4p54sig"], p_detect_4p54sig_ci95=cp(int(round(r["p_detect_4p54sig"] * n)), n))
    ci["null_zero_of_100_upper95"] = cp(0, 100)[1]
    arch["round70_with_binomial_intervals"] = ci
    log(f"[0] round-70 intervals: tropical 0.1 Tg coverage 2 sigma {r70['tropical_o3_0p4']['0.1']['coverage_2sig']:.3f} "
        f"CI {ci['tropical_o3_0p4']['0.1']['coverage_2sig_ci95']}; 0/100 null -> upper {ci['null_zero_of_100_upper95']:.3f}")
    # ---- 1. fixed database realizations
    dbs = {}
    sig_grp = np.array([s for g, r, m, s in C.GROUPS])
    db_means = []
    for k in range(args.ndb):
        dtau = np.zeros(C.n_nu)
        for g in C.GASES:
            dtau += C.Pline[g].T @ (C.SIGMA_ETA_BASE * rng.standard_normal(len(C.meta[g]["nu"]))).astype(np.float32)
        grp = sig_grp * rng.standard_normal(C.NG)
        summ, raw = run_variant(args.ndraw, rng, masses=(0.0, 0.1), dtau_fixed=dtau, grp_fixed=grp)
        dbs[f"db{k}"] = dict(summary=summ, group_scalings=grp.tolist(),
                             dy_880_od=float(C.element_obs(C.tau0 + dtau)[C.i880] - C.y0[C.i880]))
        db_means.append([summ["0.0"]["bias_mean"], summ["0.1"]["bias_mean"]])
        log(f"[1] database {k}: null mean {summ['0.0']['bias_mean']:+.4f} +- {summ['0.0']['sd']/np.sqrt(args.ndraw):.4f} (sd {summ['0.0']['sd']:.4f}), "
            f"0.1 Tg mean bias {summ['0.1']['bias_mean']:+.4f}, converged {summ['0.1']['n_converged']}/{args.ndraw}, {time.time()-t0:.0f} s")
    db_means = np.array(db_means)
    arch["fixed_database"] = dict(per_database=dbs, mean_bias_null_per_db=db_means[:, 0].tolist(), mean_bias_0p1_per_db=db_means[:, 1].tolist(),
                                  spread_of_db_means_null=float(np.std(db_means[:, 0], ddof=1)), spread_of_db_means_0p1=float(np.std(db_means[:, 1], ddof=1)),
                                  within_db_sd_median=float(np.median([dbs[k]["summary"]["0.0"]["sd"] for k in dbs])),
                                  note="each database: one realization of the per-line scatter (2 %) and the source-group scalings, fixed over ndraw soundings; "
                                       "the across-database spread of the per-database means is the bias a single mission database can carry")
    log(f"    across-database spread of the null means: {arch['fixed_database']['spread_of_db_means_null']:.4f} Tg; within-database sd {arch['fixed_database']['within_db_sd_median']:.4f}")
    # ---- 2. out-of-prior truths
    summ, _ = run_variant(args.ndraw, rng, masses=(0.0, 0.1), prior_scale=2.0, T_sig=4.0, cfc_sig=0.06)
    arch["out_of_prior_x2"] = summ
    log(f"[2] truths at 2x the priors: 0.1 Tg bias {summ['0.1']['bias_median']:+.4f}, sd {summ['0.1']['sd']:.4f}, post {summ['0.1']['posterior_sigma_median']:.4f}, "
        f"cov1 {summ['0.1']['coverage_1sig']:.2f}, P3s {summ['0.1']['p_detect_3sig']:.2f}; null sd {summ['0.0']['sd']:.4f}; converged {summ['0.1']['n_converged']}/{args.ndraw}, {time.time()-t0:.0f} s")
    summ, _ = run_variant(args.ndraw, rng, masses=(0.0, 0.1), prior_scale=3.0, T_sig=6.0, cfc_sig=0.09)
    arch["out_of_prior_x3"] = summ
    log(f"    truths at 3x the priors: 0.1 Tg bias {summ['0.1']['bias_median']:+.4f}, sd {summ['0.1']['sd']:.4f}, cov1 {summ['0.1']['coverage_1sig']:.2f}, P3s {summ['0.1']['p_detect_3sig']:.2f}; converged {summ['0.1']['n_converged']}/{args.ndraw}")
    # ---- 3. alternative sulfate member in the truth
    log("[3] alternative sulfate member (Biermann 70 wt%, 215 K) in the truth ...")
    alt = alt_member_profiles("B70T215")
    summ, _ = run_variant(args.ndraw, rng, masses=(0.0, 0.1), alt_prof=alt)
    arch["alt_sulfate_member_B70T215"] = summ
    log(f"    0.1 Tg bias {summ['0.1']['bias_median']:+.4f}, sd {summ['0.1']['sd']:.4f}, cov1 {summ['0.1']['coverage_1sig']:.2f}, P3s {summ['0.1']['p_detect_3sig']:.2f}; "
        f"null bias {summ['0.0']['bias_median']:+.4f}; converged {summ['0.1']['n_converged']}/{args.ndraw}, {time.time()-t0:.0f} s")
    # ---- 4. alternative line widths in the truth
    log("[4] air-broadening widths x1.1 (O3, N2O) in the truth ...")
    dshape = perturbed_line_widths("o3", 1.1) + perturbed_line_widths("n2o", 1.1)
    arch["line_width_perturbation_dy_od"] = (C.element_obs(C.tau0 + dshape) - C.y0).tolist()
    summ, _ = run_variant(args.ndraw, rng, masses=(0.0, 0.1), dtau_shape=dshape)
    arch["alt_line_widths_x1p1"] = summ
    log(f"    dy at 8.80 um {arch['line_width_perturbation_dy_od'][C.i880]:+.2e} OD; 0.1 Tg bias {summ['0.1']['bias_median']:+.4f}, sd {summ['0.1']['sd']:.4f}, "
        f"cov1 {summ['0.1']['coverage_1sig']:.2f}, P3s {summ['0.1']['p_detect_3sig']:.2f}; null bias {summ['0.0']['bias_median']:+.4f}; converged {summ['0.1']['n_converged']}/{args.ndraw}, {time.time()-t0:.0f} s")
    arch["runtime_s"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(arch, f, indent=1)
    log("archived ->", OUT)


if __name__ == "__main__":
    main()
