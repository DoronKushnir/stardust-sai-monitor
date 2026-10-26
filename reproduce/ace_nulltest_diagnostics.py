"""
ace_nulltest_diagnostics.py -- referee round 5 (referee report 5),
items 4 and 5: diagnostics of the ACE null test, the injection-recovery
test and the ash positive control.

Everything is computed on the same 101 occultations, the same cached
bases (outputs/ace_v52/basis_spectra_cal.npz) and the same fit code as
reproduce/ace_fullspectrum_retrieval.py (imported, as the spike test does).

Parts (keys of the archive):
  1 'rows'        exact mask / weighting / parameter count per Table-3 row
                  (full, win2, win01; std for completeness), the
                  reproduced headline statistics and the three +- conventions
  2 'chi2_coverage'  the low chi^2 (per-bin residual rms, rescaled formal
                  error), empirical coverage of the formal and the robust
                  sigma, normality summary, SD vs robust scatter
  3 'svd'         singular values / condition numbers of the whitened
                  unit-normalised design, truncated-pseudoinverse solutions,
                  projected-template fraction, variance decomposition
  4 'nonlinear_backgrounds'  refits of the 101 with bases linearized around
                  the post-Hunga-Tonga calibrated background (CAL_POSTHT,
                  LM72T213) and around the quiet background with the fine /
                  coarse r_med scaled by 0.7 and 1.4, plus one iterated step
                  around the median fitted PSD (clipped to admissible ranges)
  5 'sample'      selection bookkeeping over the 143, years, lag-1
                  autocorrelation, persistence and tail checks
  6 'injection'   per-configuration incremental recovery from spike_test.json
  7 'ash'         paired residuals / bootstraps / agreement test on the nine
                  matched spectra; detection counts after the chi^2 rescaling

Outputs
  outputs/ace_v52/nulltest_diagnostics.json
  outputs/ace_v52/nulltest_diagnostics_perocc.csv
  outputs/ace_v52/basis_diag_<case>.npz   (cached bases of part 4)

Run from the repo root:
  python reproduce/ace_nulltest_diagnostics.py
"""
from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as sps

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "reproduce"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ace_fullspectrum_retrieval as F  # noqa: E402
from calibrated_background_thresholds import (  # noqa: E402
    ALT, CAL_QUIET, CAL_POSTHT, member_ri, glossac_2025N_quiet_profile,
    TwoComponentBackground)
from saimon.geometry import tangent_to_slant_paths  # noqa: E402

OUTDIR = _ROOT / "outputs" / "ace_v52"
OUT_JSON = OUTDIR / "nulltest_diagnostics.json"
OUT_CSV = OUTDIR / "nulltest_diagnostics_perocc.csv"
RNG = np.random.default_rng(20261025)
RCONDS_F = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14]      # on the normal matrix F
NAMES_FULL = F.VARIANTS["full"]
ISIL = len(NAMES_FULL) - 1
# admissible PSD ranges for the iterated step (nm, sigma)
ADMISSIBLE = {"fine": ((30.0, 150.0), (1.3, 2.0)),
              "coarse": ((800.0, 3000.0), (1.4, 2.2))}
# the nonlinear-background cases of part 4: (member, comps)
CASES = {
    "postHT_LM72T213": ("LM72T213", CAL_POSTHT),
    "quiet_rf_x0.7": ("LM65T223", [(0.7 * CAL_QUIET[0][0],) + CAL_QUIET[0][1:],
                                   CAL_QUIET[1]]),
    "quiet_rf_x1.4": ("LM65T223", [(1.4 * CAL_QUIET[0][0],) + CAL_QUIET[0][1:],
                                   CAL_QUIET[1]]),
    "quiet_rc_x0.7": ("LM65T223", [CAL_QUIET[0],
                                   (0.7 * CAL_QUIET[1][0],) + CAL_QUIET[1][1:]]),
    "quiet_rc_x1.4": ("LM65T223", [CAL_QUIET[0],
                                   (1.4 * CAL_QUIET[1][0],) + CAL_QUIET[1][1:]]),
}


def robust_sd(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


# ---------------------------------------------------------------------------
# part 4 helper: a basis linearized around arbitrary (member, comps); the
# silica column is taken from the nominal cache (it does not depend on the
# background).  Module-level so that ProcessPoolExecutor can pickle it.
def build_basis_case(args):
    case, member, comps, nu_grid, path = args
    path = Path(path)
    if path.exists():
        return case
    ri, wt, T = member_ri(member)
    bg = TwoComponentBackground(glossac_2025N_quiet_profile(), comps, ri, wt, T)
    wl_m = 1.0e-2 / nu_grid

    def prof(layer):
        return layer.extinction_profile_m1(wl_m)

    def jac(i):
        r0, s0, _ = comps[i]
        dr, ds = F.FD * r0, F.FD * s0
        jr = (prof(bg.column(i, dr=+dr)) - prof(bg.column(i, dr=-dr))) / (2 * dr)
        js = (prof(bg.column(i, ds=+ds)) - prof(bg.column(i, ds=-ds))) / (2 * ds)
        return jr, js

    jrf, jsf = jac(0)
    jrc, jsc = jac(1)
    C = np.array(tangent_to_slant_paths(F.H_GRID_KM * 1e3, ALT))
    bases = dict(f=C @ prof(bg.column(0)), c=C @ prof(bg.column(1)),
                 jrf=C @ jrf, jsf=C @ jsf, jrc=C @ jrc, jsc=C @ jsc)
    np.savez(path, nu=nu_grid, h_km=F.H_GRID_KM, member=member,
             comps=np.array(comps, float), **bases)
    return case


def load_case_basis(path, sil):
    d = np.load(path)
    b = {k: d[k] for k in ("f", "c", "jrf", "jsf", "jrc", "jsc")}
    b["sil"] = sil
    return b


# ---------------------------------------------------------------------------
def window_elements(tau, J_full, nu_grid):
    """Copy of the win01 element construction of F.fit_occ."""
    wl_um = 1e4 / nu_grid
    in_win = (wl_um >= F.WIN_UM[0]) & (wl_um <= F.WIN_UM[1])
    y_el, J_el, ok_el = [], [], []
    for c in F.WIN_EL_UM:
        mb = in_win & (wl_um >= c - F.WIN_EL_W / 2) & (wl_um < c + F.WIN_EL_W / 2)
        good = mb & np.isfinite(tau)
        if mb.sum() == 0 or good.sum() < 0.6 * mb.sum():
            y_el.append(np.nan); J_el.append(np.zeros(J_full.shape[1]))
            ok_el.append(False); continue
        y_el.append(float(-np.log(np.mean(np.exp(-tau[good])))))
        J_el.append(J_full[good].mean(axis=0)); ok_el.append(True)
    return np.array(y_el), np.array(J_el), np.array(ok_el)


def whitened_design(J, y, ok, sig, clip):
    """Replicates F._solve and returns (theta, cov, use, A, yw, scales) with
    A the whitened, unit-normalised design on the used rows."""
    theta, cov, use, Fn = F._solve(J, y, ok, sig, ISIL, clip=clip)
    sig = np.broadcast_to(np.asarray(sig, float), y.shape)
    scales = np.linalg.norm(J[ok] / sig[ok][:, None], axis=0)
    A = (J[use] / scales) / sig[use][:, None]
    yw = y[use] / sig[use]
    return theta, cov, use, A, yw, scales, Fn


def svd_block(A, yw, scales, theta_exact, sM_exact):
    """Singular values, truncated solutions and the projection diagnostics."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    cond_A = float(s[0] / s[-1])
    out = dict(sing_vals=s.tolist(), cond_A=cond_A, cond_F=cond_A**2)
    # exact via SVD (all directions) and truncated at rcond on F = s^2
    c = U.T @ yw
    trunc = {}
    for rc in [0.0] + RCONDS_F:
        keep = (s / s[0]) ** 2 > rc
        thn = (V[:, keep] / s[keep]) @ c[keep]
        covn = (V[:, keep] / s[keep]**2) @ V[:, keep].T
        M = float(thn[ISIL] / scales[ISIL])
        sM = float(np.sqrt(covn[ISIL, ISIL]) / scales[ISIL])
        trunc[rc] = dict(M=M, sM=sM, n_kept=int(keep.sum()),
                         dM_over_sM=(M - theta_exact[ISIL]) / sM_exact,
                         sM_ratio=sM / sM_exact)
    out["trunc"] = trunc
    # variance decomposition of the silica parameter over singular directions
    contrib = V[ISIL, :]**2 / s**2
    out["var_frac_by_direction"] = (contrib / contrib.sum()).tolist()
    out["esil_sq_by_direction"] = (V[ISIL, :]**2).tolist()
    # projected-template fraction: whitened silica column vs nuisance span
    t = A[:, ISIL]
    N = np.delete(A, ISIL, axis=1)
    t_par = N @ np.linalg.lstsq(N, t, rcond=None)[0]
    out["t_perp_frac"] = float(1.0 - (t_par @ t_par) / (t @ t))
    return out


# ---------------------------------------------------------------------------
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    nu = F.reference_grid()
    bases = F.build_bases(nu)
    thr = json.load(open(F._THR_JSON))
    win_sig_od = np.array(thr["window_floors_m1"]) * F.OD_PER_M1
    wl_um = 1e4 / nu
    full_rng = (nu >= F.NU_LO) & (nu <= F.NU_HI)
    in_win = (wl_um >= F.WIN_UM[0]) & (wl_um <= F.WIN_UM[1])

    # geolocation / dates
    dates = {}
    with open(F.DATA / "occultationlist.csv") as f:
        for row in csv.DictReader(f):
            dates[row["occultation_name"]] = datetime.strptime(
                row["occultation_datetime"][:19], "%Y-%m-%d %H:%M:%S")

    # ---- part 5a: selection bookkeeping over the 143 ----------------------
    ens = [str(o) for o in np.load(F.SIDE / "matrix_19_22.npz",
                                   allow_pickle=True)["occs"]]
    sel = dict(n_ensemble=len(ens), no_dir=0, no_near_20p5=0, lt2_above_30=0,
               both_rules=0, fit_failure_lt100_bins=0, passed=0)
    occs = []
    for occ in sorted(ens):
        d = F.DATA / "residual" / occ
        if not d.is_dir():
            sel["no_dir"] += 1
            continue
        alts = []
        for fn in d.iterdir():
            try:
                alts.append(float(fn.name.split(".", 1)[1].removesuffix(".gz")))   # shipped files are gzipped
            except ValueError:
                pass
        alts = np.array(alts)
        near = np.abs(alts - 20.5) <= 1.5
        hi = alts >= 30.0
        a, b = (not near.any()), (hi.sum() < 2)
        if a and b:
            sel["both_rules"] += 1
        elif a:
            sel["no_near_20p5"] += 1
        elif b:
            sel["lt2_above_30"] += 1
        if a or b:
            continue
        got = F.occultation_data(occ, nu)
        h, tau, h_ref = got
        if (np.isfinite(tau) & full_rng).sum() < 100:
            sel["fit_failure_lt100_bins"] += 1
            continue
        occs.append((occ, h, tau, h_ref))
    sel["passed"] = len(occs)
    print("selection:", sel)
    assert sel["passed"] == 101

    # ---- part 1: nominal refits + exact row facts -------------------------
    per = []           # per-occultation record
    svd_full, svd_win = [], []
    for occ, h, tau, h_ref in occs:
        r = F.fit_occ(h, tau, h_ref, bases, nu, win_sig_od)
        rec = dict(occ=occ, h=h, h_ref=h_ref, date=dates[occ].isoformat(),
                   year=dates[occ].year)
        for tag in F.ALL_TAGS:
            for k in ("M", "sM", "chi2", "nbin", "cond"):
                rec[f"{k}_{tag}"] = r[tag][k]
        J_full = F.basis_columns(h, h_ref, bases, nu, NAMES_FULL)
        # mask bookkeeping, full variant
        ok = np.isfinite(tau) & full_rng
        rec["n_bins_in_range_full"] = int(full_rng.sum())
        rec["n_invalid_full"] = int(full_rng.sum() - ok.sum())
        th, cov, use, A, yw, scales, Fn = whitened_design(J_full, tau, ok,
                                                           F.SIG_BIN, True)
        rec["n_clipped_full"] = int(ok.sum() - use.sum())
        assert abs(th[ISIL] - r["full"]["M"]) < 1e-12
        rec["rms_resid_full_od"] = float(np.sqrt(np.mean(
            (tau[use] - J_full[use] @ th)**2)))
        sv = svd_block(A, yw, scales, th, r["full"]["sM"])
        sv["occ"] = occ
        svd_full.append(sv)
        # implied PSD shifts (for the iterated step of part 4)
        with np.errstate(divide="ignore", invalid="ignore"):
            rec["dr_f_nm"] = float(th[2] / th[0]); rec["ds_f"] = float(th[3] / th[0])
            rec["dr_c_nm"] = float(th[4] / th[1]); rec["ds_c"] = float(th[5] / th[1])
        rec["theta_full"] = th.tolist()
        # win2 mask bookkeeping
        okw = np.isfinite(tau) & in_win
        rec["n_bins_in_range_win2"] = int(in_win.sum())
        rec["n_invalid_win2"] = int(in_win.sum() - okw.sum())
        rec["n_clipped_win2"] = int(okw.sum() - r["win2"]["nbin"])
        # win01
        y_el, J_el, ok_el = window_elements(tau, J_full, nu)
        th, cov, use, A, yw, scales, Fn = whitened_design(J_el, y_el, ok_el,
                                                           win_sig_od, False)
        assert abs(th[ISIL] - r["win01"]["M"]) < 1e-12
        sv = svd_block(A, yw, scales, th, r["win01"]["sM"])
        sv["occ"] = occ
        svd_win.append(sv)
        per.append(rec)
    n = len(per)

    def col(k):
        return np.array([p[k] for p in per], float)

    rows = {}
    for tag, mask_txt in (
            ("full", "760-1250 cm^-1 (8.00-13.16 um) native 2-cm^-1 bins"),
            ("std", "760-1250 cm^-1 (8.00-13.16 um) native 2-cm^-1 bins"),
            ("win2", "7.8-9.3 um (1075.3-1282.1 cm^-1) native 2-cm^-1 bins"),
            ("win01", "fifteen 0.1-um elements 7.85-9.25 um (-ln<T> per element)")):
        M, sM, chi2 = col(f"M_{tag}"), col(f"sM_{tag}"), col(f"chi2_{tag}")
        med = float(np.median(M))
        rows[tag] = dict(
            mask=mask_txt, n_params=(6 if tag == "std" else 8),
            params=("A_f, A_c, dr_f, ds_f, M_sil, const" if tag == "std" else
                    "A_f, A_c, dr_f, ds_f, dr_c, ds_c, M_sil, const"),
            weighting=("design per-element floors (window_floors_m1 x OD_PER_M1)"
                       if tag == "win01" else "constant 8e-3 OD per 2-cm^-1 bin"),
            clipping=("none" if tag == "win01" else
                      "4 sigma on |residual|, up to 3 passes (until the mask is stable)"),
            n_occ=n, median_M=med, mean_M=float(M.mean()),
            robust_scatter=robust_sd(M), sd=float(M.std(ddof=1)),
            median_formal=float(np.median(sM)),
            ratio_robust_over_formal=robust_sd(M) / float(np.median(sM)),
            median_chi2red=float(np.median(chi2)),
            median_bins_used=float(np.median(col(f"nbin_{tag}"))),
            median_cond_F=float(np.median(col(f"cond_{tag}"))))
    for tag in ("full", "win2"):
        rows[tag].update(
            median_bins_in_range=float(np.median(col(f"n_bins_in_range_{tag}"))),
            median_invalid_bins=float(np.median(col(f"n_invalid_{tag}"))),
            median_clipped_bins=float(np.median(col(f"n_clipped_{tag}"))))
    rows["win01"].update(
        elements_um=[float(x) for x in F.WIN_EL_UM],
        floors_od=[float(x) for x in win_sig_od],
        floors_od_min=float(win_sig_od.min()), floors_od_max=float(win_sig_od.max()),
        od_per_m1=float(F.OD_PER_M1))
    # origin of the 8e-3 OD per-bin assumption
    for lam in (8.74, 8.80):
        nb = 1e4 * 0.25 / lam**2 / 2.0
        rows[f"sig_bin_origin_{lam}"] = dict(
            bins_per_0p25um_element=float(nb),
            sig_bin_from_2em3_per_element=float(2e-3 * np.sqrt(nb)))
    print("\nrows:")
    for t in ("full", "win2", "win01"):
        r = rows[t]
        print(f"  {t:5s} med {r['median_M']:+.6f} robust {r['robust_scatter']:.6f} "
              f"sd {r['sd']:.6f} formal {r['median_formal']:.6f} "
              f"ratio {r['ratio_robust_over_formal']:.3f} chi2 {r['median_chi2red']:.3f}")

    # ---- part 2: low chi2, coverage, normality ----------------------------
    cc = {}
    for tag in ("full", "win2", "win01"):
        M, sM, chi2 = col(f"M_{tag}"), col(f"sM_{tag}"), col(f"chi2_{tag}")
        med = float(np.median(M))
        sig_emp = robust_sd(M)
        sd = float(M.std(ddof=1))
        sig_bin = F.SIG_BIN if tag != "win01" else None
        resc = sM * np.sqrt(chi2)
        d = dict(median_chi2red=float(np.median(chi2)),
                 median_formal=float(np.median(sM)),
                 median_formal_rescaled_sqrt_chi2=float(np.median(resc)),
                 robust_over_rescaled_formal=sig_emp / float(np.median(resc)),
                 robust_scatter=sig_emp, sd=sd, sd_over_robust=sd / sig_emp)
        if sig_bin:
            d["median_perbin_resid_rms_od"] = float(np.median(np.sqrt(chi2) * sig_bin))
            if tag == "full":
                d["median_perbin_resid_rms_od_direct"] = float(
                    np.median(col("rms_resid_full_od")))
        zf = (M - med) / sM
        ze = (M - med) / sig_emp
        zs = (M - med) / sd
        for lab, z in (("formal", zf), ("robust", ze), ("sd", zs)):
            d[f"coverage_{lab}"] = {f"lt{k}": float(np.mean(np.abs(z) < k))
                                    for k in (1, 2, 3)}
            d[f"frac_abs_z_gt2_{lab}"] = float(np.mean(np.abs(z) > 2))
            d[f"frac_abs_z_gt3_{lab}"] = float(np.mean(np.abs(z) > 3))
            d[f"max_abs_z_{lab}"] = float(np.max(np.abs(z)))
        d["coverage_gaussian_expected"] = {f"lt{k}": float(2 * sps.norm.cdf(k) - 1)
                                           for k in (1, 2, 3)}
        d["skewness"] = float(sps.skew(M))
        d["excess_kurtosis"] = float(sps.kurtosis(M))
        d["shapiro_W"], d["shapiro_p"] = map(float, sps.shapiro(M))
        d["robust_sigma_of_z_formal"] = robust_sd(zf)
        d["sd_of_z_formal"] = float(zf.std(ddof=1))
        # SD vs robust scatter: trimming the |z|>2 tail (robust units)
        core = np.abs(ze) <= 2
        d["sd_core_abs_z_le2"] = float(M[core].std(ddof=1))
        d["n_tail_abs_z_gt2"] = int((~core).sum())
        d["iqr_over_1p349"] = float((np.percentile(M, 75) - np.percentile(M, 25)) / 1.349)
        cc[tag] = d
    print("\nchi2/coverage full:", json.dumps(cc["full"], indent=None)[:600])

    # ---- part 3: SVD summary ---------------------------------------------
    svd = {}
    for tag, lst in (("full", svd_full), ("win01", svd_win)):
        sM = col(f"sM_{tag}")
        sv = np.array([x["sing_vals"] for x in lst])
        d = dict(median_sing_vals=np.median(sv, axis=0).tolist(),
                 median_cond_A=float(np.median([x["cond_A"] for x in lst])),
                 median_cond_F=float(np.median([x["cond_F"] for x in lst])),
                 median_cond_F_reported=float(np.median(col(f"cond_{tag}"))),
                 t_perp_frac_median=float(np.median([x["t_perp_frac"] for x in lst])),
                 t_perp_frac_min=float(np.min([x["t_perp_frac"] for x in lst])),
                 t_perp_frac_max=float(np.max([x["t_perp_frac"] for x in lst])),
                 var_frac_by_direction_median=np.median(
                     [x["var_frac_by_direction"] for x in lst], axis=0).tolist(),
                 esil_sq_by_direction_median=np.median(
                     [x["esil_sq_by_direction"] for x in lst], axis=0).tolist(),
                 trunc={})
        vf = np.array([x["var_frac_by_direction"] for x in lst])
        d["var_frac_smallest_1_median"] = float(np.median(vf[:, -1]))
        d["var_frac_smallest_2_median"] = float(np.median(vf[:, -2:].sum(axis=1)))
        d["var_frac_smallest_3_median"] = float(np.median(vf[:, -3:].sum(axis=1)))
        d["var_frac_smallest_3_max"] = float(np.max(vf[:, -3:].sum(axis=1)))
        for rc in [0.0] + RCONDS_F:
            dM = np.array([x["trunc"][rc]["dM_over_sM"] for x in lst])
            kept = np.array([x["trunc"][rc]["n_kept"] for x in lst])
            sr = np.array([x["trunc"][rc]["sM_ratio"] for x in lst])
            Ms = np.array([x["trunc"][rc]["M"] for x in lst])
            d["trunc"][f"rcond_F={rc:g}"] = dict(
                equiv_rcond_A=float(np.sqrt(rc)),
                n_kept_median=float(np.median(kept)), n_kept_min=int(kept.min()),
                max_abs_dM_over_sM=float(np.max(np.abs(dM))),
                median_abs_dM_over_sM=float(np.median(np.abs(dM))),
                median_sM_ratio=float(np.median(sr)),
                median_M=float(np.median(Ms)), robust_scatter=robust_sd(Ms))
        svd[tag] = d
        print(f"\nsvd {tag}: cond_F {d['median_cond_F']:.3e} (reported "
              f"{d['median_cond_F_reported']:.3e}), t_perp {d['t_perp_frac_median']:.4f}")
        for k, v in d["trunc"].items():
            print(f"  {k:14s} kept {v['n_kept_median']:.0f}  max|dM|/sM "
                  f"{v['max_abs_dM_over_sM']:.3e}  sM ratio {v['median_sM_ratio']:.4f} "
                  f"med M {v['median_M']:+.5f} rob {v['robust_scatter']:.5f}")

    # ---- part 4: nonlinear backgrounds ------------------------------------
    # iterated step: median implied PSD shift over the 101, clipped
    it = {}
    for mode, i in (("fine", 0), ("coarse", 1)):
        r0, s0, frac = CAL_QUIET[i]
        dr = col("dr_f_nm" if mode == "fine" else "dr_c_nm")
        ds = col("ds_f" if mode == "fine" else "ds_c")
        (rlo, rhi), (slo, shi) = ADMISSIBLE[mode]
        r_imp, s_imp = float(np.median(r0 + dr)), float(np.median(s0 + ds))
        it[mode] = dict(r0_nm=r0, s0=s0, median_implied_r_nm=r_imp,
                        median_implied_sigma=s_imp,
                        iqr_implied_r_nm=[float(x) for x in np.percentile(r0 + dr, [25, 75])],
                        iqr_implied_sigma=[float(x) for x in np.percentile(s0 + ds, [25, 75])],
                        r_used_nm=float(np.clip(r_imp, rlo, rhi)),
                        sigma_used=float(np.clip(s_imp, slo, shi)),
                        admissible_r_nm=[rlo, rhi], admissible_sigma=[slo, shi])
    comps_it = [(it["fine"]["r_used_nm"], it["fine"]["sigma_used"], CAL_QUIET[0][2]),
                (it["coarse"]["r_used_nm"], it["coarse"]["sigma_used"], CAL_QUIET[1][2])]
    cases = dict(CASES)
    cases["iterated_median_psd"] = ("LM65T223", comps_it)
    jobs = [(c, m, [tuple(map(float, x)) for x in comps], nu,
             str(OUTDIR / f"basis_diag_{c}.npz")) for c, (m, comps) in cases.items()]
    todo = [j for j in jobs if not Path(j[4]).exists()]
    if todo:
        print(f"\nbuilding {len(todo)} bases in parallel (Mie, ~2 min each)...")
        with ProcessPoolExecutor(max_workers=min(len(todo), 6)) as ex:
            for c in ex.map(build_basis_case, todo):
                print("  built", c)
    nl = dict(nominal=dict(member="LM65T223", comps=[list(map(float, c)) for c in CAL_QUIET]),
              iterated_psd=it, cases={})
    M0 = col("M_full"); sM0 = col("sM_full")
    M0w = col("M_win01")
    for c, (m, comps) in cases.items():
        b = load_case_basis(OUTDIR / f"basis_diag_{c}.npz", bases["sil"])
        Mc, sMc, chi2c, Mw, chi2w = [], [], [], [], []
        for occ, h, tau, h_ref in occs:
            r = F.fit_occ(h, tau, h_ref, b, nu, win_sig_od)
            Mc.append(r["full"]["M"]); sMc.append(r["full"]["sM"])
            chi2c.append(r["full"]["chi2"])
            Mw.append(r["win01"]["M"]); chi2w.append(r["win01"]["chi2"])
        Mc, sMc, chi2c, Mw = map(np.array, (Mc, sMc, chi2c, Mw))
        dM = Mc - M0
        nl["cases"][c] = dict(
            member=m, comps=[list(map(float, x)) for x in comps],
            full=dict(median_M=float(np.median(Mc)), robust_scatter=robust_sd(Mc),
                      sd=float(Mc.std(ddof=1)), median_formal=float(np.median(sMc)),
                      median_chi2red=float(np.median(chi2c)),
                      median_dM_vs_nominal=float(np.median(dM)),
                      robust_scatter_dM=robust_sd(dM),
                      max_abs_dM_over_sM=float(np.max(np.abs(dM) / sM0)),
                      corr_with_nominal=float(np.corrcoef(Mc, M0)[0, 1])),
            win01=dict(median_M=float(np.median(Mw)), robust_scatter=robust_sd(Mw),
                       median_chi2red=float(np.median(chi2w)),
                       median_dM_vs_nominal=float(np.median(Mw - M0w)),
                       robust_scatter_dM=robust_sd(Mw - M0w)))
        for k, v in zip(range(n), Mc):
            per[k][f"M_full_{c}"] = float(v)
        f_ = nl["cases"][c]["full"]
        print(f"  case {c:22s} full: med {f_['median_M']:+.5f} rob {f_['robust_scatter']:.5f} "
              f"formal {f_['median_formal']:.5f} chi2 {f_['median_chi2red']:.3f} "
              f"dM med {f_['median_dM_vs_nominal']:+.5f} rob(dM) {f_['robust_scatter_dM']:.5f}")

    # ---- part 5: sample, years, autocorrelation, persistence, tail --------
    order = np.argsort([p["date"] for p in per])
    years = col("year").astype(int)
    yc = {int(y): int((years == y).sum()) for y in sorted(set(years))}
    samp = dict(selection=sel, n_used=n, years=yc, n_distinct_years=len(yc),
                tangent_height_median_km=float(np.median(col("h"))),
                tangent_height_range_km=[float(col("h").min()), float(col("h").max())])

    def lag1(x):
        x = x - x.mean()
        return float((x[:-1] * x[1:]).sum() / (x * x).sum())

    def pairs_boot(x, nb=20000):
        a, b = x[:-1], x[1:]
        idx = RNG.integers(0, len(a), size=(nb, len(a)))
        r = np.array([np.corrcoef(a[i], b[i])[0, 1] for i in idx])
        return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]

    # expectation of max|z| for n Gaussian draws
    sim = np.max(np.abs(RNG.standard_normal((100000, n))), axis=1)
    samp["max_abs_z_gaussian_n101"] = dict(mean=float(sim.mean()),
                                            p2p5=float(np.percentile(sim, 2.5)),
                                            p97p5=float(np.percentile(sim, 97.5)))
    for tag in ("full", "win01"):
        M = col(f"M_{tag}")[order]
        yr = years[order]
        med, sig = float(np.median(M)), robust_sd(M)
        z = (M - med) / sig
        same = yr[:-1] == yr[1:]
        d = dict(lag1_all=lag1(M), lag1_ci95_pairs_bootstrap=pairs_boot(M),
                 lag1_gaussian_sd=float(1 / np.sqrt(n)),
                 lag1_same_season_pairs=float(np.corrcoef(M[:-1][same], M[1:][same])[0, 1]),
                 n_same_season_pairs=int(same.sum()), n_pairs=n - 1)
        for k in (1, 2):
            p1 = float(np.mean(z > k))
            both = (z[:-1] > k) & (z[1:] > k)
            d[f"frac_above_plus{k}sig"] = p1
            d[f"n_above_plus{k}sig"] = int((z > k).sum())
            d[f"frac_consecutive_both_above_plus{k}sig"] = float(both.mean())
            d[f"n_consecutive_both_above_plus{k}sig"] = int(both.sum())
            d[f"expected_frac_both_if_independent"] = p1**2
            d[f"expected_n_both_if_independent_{k}sig"] = float(p1**2 * (n - 1))
        # year-demeaned series: within-season persistence only
        ymed = {y: float(np.median(M[yr == y])) for y in set(yr)}
        Md = M - np.array([ymed[y] for y in yr])
        d["lag1_year_demeaned_all"] = lag1(Md)
        d["lag1_year_demeaned_same_season_pairs"] = float(
            np.corrcoef(Md[:-1][same], Md[1:][same])[0, 1])
        d["lag1_year_demeaned_ci95_pairs_bootstrap"] = pairs_boot(Md)
        # demeaning within groups of mean size m induces a lag-1 bias of about
        # -1/(m-1) for independent draws; Monte Carlo of the exact design
        m_bar = n / len(set(yr))
        d["lag1_demeaning_bias_formula"] = float(-1.0 / (m_bar - 1))
        sim_r = []
        for _ in range(4000):
            g = RNG.standard_normal(n)
            gd = g - np.array([np.median(g[yr == y]) for y in yr])
            sim_r.append(np.corrcoef(gd[:-1][same], gd[1:][same])[0, 1])
        sim_r = np.array(sim_r)
        d["lag1_year_demeaned_same_season_null_mc"] = dict(
            mean=float(sim_r.mean()), sd=float(sim_r.std()),
            p_two_sided=float(np.mean(np.abs(sim_r - sim_r.mean())
                                      >= abs(d["lag1_year_demeaned_same_season_pairs"] - sim_r.mean()))))
        # one-way random-effects variance components (years as clusters)
        groups = [M[yr == y] for y in sorted(set(yr))]
        k = len(groups); ni = np.array([len(g) for g in groups], float)
        gm = M.mean()
        ss_b = float(sum(len(g) * (g.mean() - gm)**2 for g in groups))
        ss_w = float(sum(((g - g.mean())**2).sum() for g in groups))
        ms_b, ms_w = ss_b / (k - 1), ss_w / (n - k)
        n0 = (n - ni @ ni / n) / (k - 1)
        var_b = max((ms_b - ms_w) / n0, 0.0)
        icc = var_b / (var_b + ms_w)
        m_bar = n / k
        d["variance_components"] = dict(
            n_years=k, between_year_sd=float(np.sqrt(var_b)), within_year_sd=float(np.sqrt(ms_w)),
            icc=float(icc), F_anova=float(ms_b / ms_w),
            p_anova=float(sps.f.sf(ms_b / ms_w, k - 1, n - k)),
            design_effect=float(1 + (m_bar - 1) * icc),
            n_eff=float(n / (1 + (m_bar - 1) * icc)),
            sd_of_year_medians=float(np.std(list(ymed.values()), ddof=1)))
        d["year_medians"] = {int(y): ymed[y] for y in sorted(ymed)}
        # secular drift: OLS slope vs decimal year, Spearman, epoch medians
        dec = np.array([datetime.fromisoformat(p["date"]).year
                        + (datetime.fromisoformat(p["date"]).timetuple().tm_yday - 0.5) / 365.25
                        for p in per])[order]
        lr_ = sps.linregress(dec, M)
        early, late = yr <= 2015, yr >= 2016
        d["trend"] = dict(slope_tg_per_decade=float(lr_.slope * 10),
                          slope_err_tg_per_decade=float(lr_.stderr * 10),
                          p_slope=float(lr_.pvalue), spearman_rho=float(sps.spearmanr(dec, M)[0]),
                          spearman_p=float(sps.spearmanr(dec, M)[1]),
                          median_2004_2015=float(np.median(M[early])), n_2004_2015=int(early.sum()),
                          median_2016_2024=float(np.median(M[late])), n_2016_2024=int(late.sum()),
                          robust_scatter_2004_2015=robust_sd(M[early]),
                          robust_scatter_2016_2024=robust_sd(M[late]),
                          robust_scatter_year_demeaned=robust_sd(Md))
        occ_sorted = np.array([p["occ"] for p in per])[order]
        imax = int(np.argmax(np.abs(z)))
        d["max_abs_z_occ"] = dict(occ=str(occ_sorted[imax]), M=float(M[imax]), z=float(z[imax]),
                                  year=int(yr[imax]))
        d["max_abs_z_robust"] = float(np.max(np.abs(z)))
        d["max_abs_z_formal"] = float(np.max(np.abs((M - med) / col(f"sM_{tag}")[order])))
        d["n_abs_z_gt3_robust"] = int((np.abs(z) > 3).sum())
        d["p_value_max_abs_z_vs_gaussian"] = float(np.mean(sim >= d["max_abs_z_robust"]))
        d["gaussian_tail_prob_4p5sig"] = float(sps.norm.sf(4.5))
        d["expected_n_beyond_4p5sig_in_101"] = float(n * sps.norm.sf(4.5))
        samp[tag] = d
    print("\nsample:", json.dumps({k: v for k, v in samp.items() if k != "selection"})[:900])

    # ---- part 6: injection recovery per configuration ---------------------
    sp = json.load(open(F.OUT / "spike_test.json"))
    inj = {}
    for tag in F.ALL_TAGS:
        c0 = sp["0.0"][tag]
        d = dict(zero_median=c0["median"], zero_mean=c0["mean"],
                 zero_robust_scatter=c0["sigma_emp"], zero_sd=c0["sd"], inc={})
        for m in ("0.05", "0.1", "0.2", "0.3"):
            s = sp[m][tag]
            d["inc"][m] = dict(median_M=s["median"],
                               increment_median=s["median"] - c0["median"] - float(m),
                               increment_mean=s["mean"] - c0["mean"] - float(m),
                               absolute_median_offset=s["median"] - float(m),
                               robust_scatter=s["sigma_emp"], sd=s["sd"])
        d["max_abs_increment_median"] = float(max(abs(v["increment_median"]) for v in d["inc"].values()))
        d["max_abs_increment_mean"] = float(max(abs(v["increment_mean"]) for v in d["inc"].values()))
        inj[tag] = d
        print(f"  inj {tag:5s} zero med {d['zero_median']:+.6f}  max|inc| med "
              f"{d['max_abs_increment_median']:.2e} mean {d['max_abs_increment_mean']:.2e}")

    # ---- part 7: ash positive control -------------------------------------
    rows_a = list(csv.DictReader(open(F.OUT / "ash_positive_control.csv")))

    def fl(r, k):
        try:
            return float(r[k])
        except (ValueError, KeyError):
            return np.nan

    A = [r for r in rows_a if r["sample"] == "ash"]
    N = [r for r in rows_a if r["sample"] == "null"]
    nine = [r for r in A if np.isfinite(fl(r, "lec_Nl_ash"))]
    ash = dict(n_ash=len(A), n_ash_occ=len(set(r["occ"] for r in A)),
               n_null=len(N), n_matched=len(nine),
               n_matched_occ=len(set(r["occ"] for r in nine)))
    theirs = np.array([fl(r, "lec_Nl_ash") for r in nine])
    terr = np.array([fl(r, "lec_Nl_ash_err") for r in nine])
    occ9 = np.array([r["occ"] for r in nine])
    pairs = {}
    for lab, kN, kS in (("fixed_psd", "N_ash", "sN_ash"),
                        ("matched_psd", "N_ash_match", "sN_ash_match")):
        ours = np.array([fl(r, kN) for r in nine])
        serr = np.array([fl(r, kS) for r in nine])
        comb = np.sqrt(serr**2 + terr**2)
        resid = ours - theirs
        z = resid / comb
        pos = (ours > 0) & (theirs > 0)
        with np.errstate(invalid="ignore"):
            lr_all = np.log(ours / theirs)
        lr = lr_all[pos]                      # log ratios: positive pairs only
        occ9p = occ9[pos]
        d = dict(n_positive_pairs=int(pos.sum()),
                 n_negative_ours=int((ours <= 0).sum()),per_spectrum=[dict(occ=o, alt=fl(r, "alt"), ours=float(a), s_ours=float(b),
                                    theirs=float(t), s_theirs=float(e), resid=float(x),
                                    comb_err=float(c), z=float(zz), log_ratio=float(l),
                                    chi2red_S=fl(r, "chi2red_S"),
                                    chi2red_ash=fl(r, "chi2red_ash"))
                               for o, r, a, b, t, e, x, c, zz, l in
                               zip(occ9, nine, ours, serr, theirs, terr, resid, comb, z, lr_all)],
                 median_ratio=float(np.median(ours / theirs)),
                 median_log_ratio=float(np.median(lr)), mean_log_ratio=float(lr.mean()),
                 sd_log_ratio=float(lr.std(ddof=1)),
                 sem_log_ratio_analytic=float(lr.std(ddof=1) / np.sqrt(len(lr))),
                 n_abs_z_lt1=int((np.abs(z) < 1).sum()), n_abs_z_lt2=int((np.abs(z) < 2).sum()),
                 n_abs_z_lt3=int((np.abs(z) < 3).sum()), median_abs_z=float(np.median(np.abs(z))),
                 median_s_ours_over_s_theirs=float(np.median(serr / terr)))
        d["note_log"] = ("log-ratio statistics over the positive pairs; the median "
                         "ratio and the z-scores use all pairs")
        # (a) per-spectrum bootstrap, (b) occultation-level bootstrap
        nb = 20000
        idx = RNG.integers(0, len(lr), size=(nb, len(lr)))
        bm, bmed = lr[idx].mean(axis=1), np.median(lr[idx], axis=1)
        d["boot_spectrum"] = dict(
            mean_log_ratio_sd=float(bm.std()), mean_log_ratio_ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
            median_log_ratio_sd=float(bmed.std()), median_log_ratio_ci95=[float(np.percentile(bmed, 2.5)), float(np.percentile(bmed, 97.5))],
            ratio_mean_ci95=[float(np.exp(np.percentile(bm, 2.5))), float(np.exp(np.percentile(bm, 97.5)))])
        uo = sorted(set(occ9p))
        groups = [np.where(occ9p == o)[0] for o in uo]
        bm2, bmed2 = [], []
        for _ in range(nb):
            pick = RNG.integers(0, len(groups), size=len(groups))
            ii = np.concatenate([groups[g] for g in pick])
            bm2.append(lr[ii].mean()); bmed2.append(np.median(lr[ii]))
        bm2, bmed2 = np.array(bm2), np.array(bmed2)
        d["boot_occultation"] = dict(
            n_occ=len(uo), mean_log_ratio_sd=float(bm2.std()),
            mean_log_ratio_ci95=[float(np.percentile(bm2, 2.5)), float(np.percentile(bm2, 97.5))],
            median_log_ratio_sd=float(bmed2.std()),
            median_log_ratio_ci95=[float(np.percentile(bmed2, 2.5)), float(np.percentile(bmed2, 97.5))],
            ratio_mean_ci95=[float(np.exp(np.percentile(bm2, 2.5))), float(np.exp(np.percentile(bm2, 97.5)))])
        # chi-square agreement tests, fixed weights w = 1/(s_ours^2 + s_theirs^2)
        w = 1.0 / comb**2

        def chi2_at(s):
            return float(np.sum(w * (ours - s * theirs)**2))

        s_wls = float(np.sum(w * ours * theirs) / np.sum(w * theirs**2))
        s_wls_err = float(1.0 / np.sqrt(np.sum(w * theirs**2)))
        c1, cmed, cb = chi2_at(1.0), chi2_at(d["median_ratio"]), chi2_at(s_wls)
        d["agreement"] = dict(
            weights="1/(s_ours^2 + s_theirs^2), fixed",
            chi2_ratio1=c1, dof_ratio1=len(ours), p_ratio1=float(sps.chi2.sf(c1, len(ours))),
            chi2_ratio_median=cmed, dof_ratio_median=len(ours) - 1,
            p_ratio_median=float(sps.chi2.sf(cmed, len(ours) - 1)),
            scale_wls=s_wls, scale_wls_err=s_wls_err, chi2_scale_wls=cb,
            dof_scale_wls=len(ours) - 1, p_scale_wls=float(sps.chi2.sf(cb, len(ours) - 1)))
        # the same without the one pair whose published error is ~3e13
        keep = terr < 1e12
        if keep.sum() < len(ours):
            o2, t2, w2 = ours[keep], theirs[keep], w[keep]
            s2 = float(np.sum(w2 * o2 * t2) / np.sum(w2 * t2**2))
            c1b = float(np.sum(w2 * (o2 - t2)**2))
            cbb = float(np.sum(w2 * (o2 - s2 * t2)**2))
            cmb = float(np.sum(w2 * (o2 - d["median_ratio"] * t2)**2))
            p2 = pos & keep
            d["agreement_excl_huge_err"] = dict(
                n=int(keep.sum()), chi2_ratio1=c1b, p_ratio1=float(sps.chi2.sf(c1b, keep.sum())),
                chi2_ratio_median=cmb, p_ratio_median=float(sps.chi2.sf(cmb, keep.sum() - 1)),
                scale_wls=s2, scale_wls_err=float(1.0 / np.sqrt(np.sum(w2 * t2**2))),
                chi2_scale_wls=cbb, p_scale_wls=float(sps.chi2.sf(cbb, keep.sum() - 1)),
                median_ratio=float(np.median(o2 / t2)),
                median_log_ratio=float(np.median(np.log(ours[p2] / theirs[p2]))),
                mean_log_ratio=float(np.mean(np.log(ours[p2] / theirs[p2]))),
                n_positive_pairs=int(p2.sum()))
        # the same with OUR errors inflated by sqrt(chi2red_ash) per spectrum
        cra = np.array([fl(r, "chi2red_ash") for r in nine])
        comb_r = np.sqrt(serr**2 * cra + terr**2)
        w_r = 1.0 / comb_r**2
        z_r = resid / comb_r
        s_r = float(np.sum(w_r * ours * theirs) / np.sum(w_r * theirs**2))
        d["agreement_our_errors_rescaled"] = dict(
            chi2_ratio1=float(np.sum(w_r * resid**2)),
            p_ratio1=float(sps.chi2.sf(np.sum(w_r * resid**2), len(ours))),
            scale_wls=s_r, chi2_scale_wls=float(np.sum(w_r * (ours - s_r * theirs)**2)),
            p_scale_wls=float(sps.chi2.sf(np.sum(w_r * (ours - s_r * theirs)**2), len(ours) - 1)),
            n_abs_z_lt1=int((np.abs(z_r) < 1).sum()), n_abs_z_lt2=int((np.abs(z_r) < 2).sum()),
            z=[float(x) for x in z_r])
        pairs[lab] = d
        print(f"  ash pairs {lab}: med ratio {d['median_ratio']:.3f} mean lr {d['mean_log_ratio']:+.3f} "
              f"sd {d['sd_log_ratio']:.3f} sem(spec) {d['boot_spectrum']['mean_log_ratio_sd']:.3f} "
              f"sem(occ) {d['boot_occultation']['mean_log_ratio_sd']:.3f}; chi2(1) {d['agreement']['chi2_ratio1']:.1f} "
              f"p {d['agreement']['p_ratio1']:.2e}; WLS scale {s_wls:.2f}+-{s_wls_err:.2f} chi2 {cb:.1f} p {d['agreement']['p_scale_wls']:.2e}")
    ash["pairs"] = pairs

    # fit inadequacy: rescale sN by sqrt(chi2red) of the relevant fit
    det = {}
    for lab, rs in (("ash", A), ("null", N)):
        zA = np.array([fl(r, "N_ash") / fl(r, "sN_ash") for r in rs])
        cA = np.array([fl(r, "chi2red_ash") for r in rs])
        cS = np.array([fl(r, "chi2red_S") for r in rs])
        cB = np.array([fl(r, "chi2red_both") for r in rs])
        zB = np.array([fl(r, "N_ash_b") / fl(r, "sN_ash_b") for r in rs])
        dA = np.array([fl(r, "dchi2_ash") for r in rs])
        dS = np.array([fl(r, "dchi2_sil") for r in rs])
        zA_s, zB_s = zA / np.sqrt(cA), zB / np.sqrt(cB)
        det[lab] = dict(
            n=len(rs), median_chi2red_S=float(np.median(cS)),
            median_chi2red_ash=float(np.median(cA)), median_chi2red_both=float(np.median(cB)),
            n_ash_gt3_unscaled=int((zA > 3).sum()), n_ash_gt3_scaled_by_chi2red_ash=int((zA_s > 3).sum()),
            n_ash_gt5_scaled_by_chi2red_ash=int((zA_s > 5).sum()),
            median_z_ash_unscaled=float(np.median(zA)), median_z_ash_scaled=float(np.median(zA_s)),
            n_ash_joint_gt3_unscaled=int((zB > 3).sum()), n_ash_joint_gt3_scaled_by_chi2red_both=int((zB_s > 3).sum()),
            median_z_ash_joint_unscaled=float(np.median(zB)), median_z_ash_joint_scaled=float(np.median(zB_s)),
            n_prefer_ash=int((dA > dS).sum()), frac_prefer_ash=float(np.mean(dA > dS)),
            note_prefer="dchi2_ash > dchi2_sil is invariant under a common per-spectrum sigma rescaling",
            median_dchi2_ash_minus_sil=float(np.median(dA - dS)),
            median_dchi2_ash_minus_sil_scaled_by_chi2red_S=float(np.median((dA - dS) / cS)),
            median_dchi2_ash_minus_sil_scaled_by_chi2red_both=float(np.median((dA - dS) / cB)),
            n_dchi2_diff_gt9_scaled_by_chi2red_S=int((((dA - dS) / cS) > 9).sum()))
        print(f"  det {lab}: >3sig {det[lab]['n_ash_gt3_unscaled']} -> scaled "
              f"{det[lab]['n_ash_gt3_scaled_by_chi2red_ash']}; joint {det[lab]['n_ash_joint_gt3_unscaled']} -> "
              f"{det[lab]['n_ash_joint_gt3_scaled_by_chi2red_both']}; median z {det[lab]['median_z_ash_unscaled']:.2f} -> "
              f"{det[lab]['median_z_ash_scaled']:.2f}")
    ash["detections"] = det

    # ---- archive -----------------------------------------------------------
    out = dict(script="reproduce/ace_nulltest_diagnostics.py",
               sig_bin_od=F.SIG_BIN, rconds_F=RCONDS_F,
               rows=rows, chi2_coverage=cc, svd=svd, nonlinear_backgrounds=nl,
               sample=samp, injection=inj, ash=ash)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    keys = [k for k in per[0] if k != "theta_full"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(per)
    print(f"\narchived -> {OUT_JSON}\n            {OUT_CSV}")


if __name__ == "__main__":
    main()
