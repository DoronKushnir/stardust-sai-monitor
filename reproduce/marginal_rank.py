"""
marginal_rank.py -- round 70 (referee round 5, item 6): re-evaluation, with the
fixed span-test marginal() (no parameter-count shortcut; SVD projector onto the nuisance
span at SVD_RTOL; zero-norm nuisance columns dropped), of every configuration the former
K < N + 1 shortcut excluded or that the paper calls singular / rank-deficient, plus a
direct check that the previously finite results (baseline band/window thresholds among
them) are unchanged.

PART 0  baseline reproduction from the cached 2-D engine
        (outputs/detectability_2d_calibrated/engine_band_quiet.npz: triplet + 47 band
        elements, calibrated quiet background, profile bit-identical to the thresholds
        script at 20 km): band 0.101039 Tg, window 0.141152 Tg, and the other archived
        band/window rows (5-param, 7-param, 7-param + MIR offset; triplet and MIR-alone
        rosters), old function vs new function vs archive.
PART 1  the single 8.80-um element (PART 1 of calibrated_background_thresholds.py,
        7-channel machinery, measured vis/NIR errors, element floor SIG_874_ABS):
        (a) the heritage triplet alone (448, 756, 1544 nm; K = 3) -- Sect. 2
            "with the heritage triplet the seven-parameter Fisher matrix is singular";
        (b) the triplet + one 0.1-um element at 8.80 um (K = 4) -- the Sect. 3 footnote's
            "rank-deficient fit for the triplet"; the fuller rosters for comparison.
        Nuisance sets: A1+rs (3), A2 (2), A2+rs (4), A2+rs+rc (5), full (6 columns).
PART 2  (c) the triplet + a 0.1-um continuum element at 3.8 um (and 4.0 um; K = 4) --
        Appendix D / Sect. 3 "rank-deficient against the six-parameter background family";
        the six-channel roster + 3.8 um (K = 7) as the archived comparison.
PART 3  (d) the log-spaced visible/NIR channel sets of calibrated_background_problem.py
        (385-1550 nm, phi = 4 %) at K = 3 ... 8 (the archived scan starts at K = 4 and
        returned None for the 7-param fit at K = 4, 5, 6 and the 5-param fit at K = 4
        through the shortcut).

For every configuration: K, the number of nuisance columns, their rank at SVD_RTOL, the
singular values of the whitened unit-normalised nuisance block, ||t_perp||/||t_hat||,
R^2, the old and new sigma_A and M_min = 3 sigma_A, and the verdict.

Run from the repo root:  python reproduce/marginal_rank.py
Output: outputs/marginal_rank.json (+ .txt summary table)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np

_THIS = Path(__file__).resolve()
_PARENT = _THIS.parent.parent
for p in (_PARENT, _THIS.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from calibrated_background_thresholds import (                       # noqa: E402
    ALT, iz, RMED_SIL_NM, SIGMA_SIL, ALPHA_FLOOR, CAL_QUIET, WL_NM, PHI, I874,
    SIG_874_ABS, SPECTRAL_RES_UM, SVD_RTOL, TPERP_RTOL,
    TwoComponentBackground, glossac_2025N_quiet_profile, marginal, nuisance_span,
    member_ri, banded_ext, noise)
from calibrated_background_problem import make_bext, GenBackground, PHI_SWEEP  # noqa: E402
from design_sensitivity_calibrated import ElementSet, VIS_PHI, N_VIS           # noqa: E402
from saimon.materials import SilicaRefractiveIndex                              # noqa: E402
from saimon.sai import create_silica_sai_layer                                  # noqa: E402

OUT_JSON = _PARENT / "outputs" / "marginal_rank.json"
OUT_TXT = OUT_JSON.with_suffix(".txt")
THR_JSON = _PARENT / "outputs" / "calibrated_background_thresholds.json"
PROB_JSON = _PARENT / "outputs" / "calibrated_background_problem.json"
ENGINE_DIR = _PARENT / "outputs" / "detectability_2d_calibrated"
ARCH_BAND, ARCH_WIN = 0.101039, 0.141152     # the quoted baselines (rounded in the paper to 0.101/0.141)


def marginal_old(idx, nuis_cols, sig, t):
    """The pre-round-70 function, verbatim, for the bit-identity comparison."""
    idx = np.asarray(idx)
    N = np.column_stack([c[idx] for c in nuis_cols])
    if len(idx) < N.shape[1] + 1:
        return np.inf, 0.0, np.nan
    J = np.column_stack([N, t[idx]]) / sig[idx][:, None]
    scales = np.linalg.norm(J, axis=0)
    if np.any(scales == 0):
        return np.inf, 0.0, np.nan
    Jn = J / scales
    t_hat = J[:, -1]
    Nw = Jn[:, :-1]
    t_par = Nw @ np.linalg.lstsq(Nw, t_hat, rcond=None)[0]
    t_perp = t_hat - t_par
    R2 = float(t_par @ t_par) / float(t_hat @ t_hat)
    if np.linalg.norm(t_perp) <= 1e-8 * np.linalg.norm(t_hat):
        return np.inf, 0.0, np.nan
    sig_a = 1.0 / float(np.linalg.norm(t_perp))
    return sig_a, 1.0 / sig_a, R2


ROWS = []


def evaluate(config, idx, cols, sig, t, archived=None, note=""):
    """One configuration: both functions, the span diagnostics, the verdict."""
    idx = np.asarray(idx)
    sa_new, tperp_new, R2_new = marginal(idx, cols, sig, t)
    sa_old, tperp_old, R2_old = marginal_old(idx, cols, sig, t)
    Ur, s, n_kept = nuisance_span(idx, cols, sig)
    t_hat = np.asarray(t, float)[idx] / np.asarray(sig, float)[idx]
    t_perp = t_hat - Ur @ (Ur.T @ t_hat)
    ratio = float(np.linalg.norm(t_perp) / np.linalg.norm(t_hat))
    K, n_nuis, rank = int(len(idx)), int(len(cols)), int(Ur.shape[1])
    if np.isfinite(sa_new):
        verdict = "identifiable"
    elif rank == K:
        verdict = "unidentifiable: nuisance span = whole space (rank = K), t_perp = 0 to round-off"
    else:
        verdict = "unidentifiable: target inside the nuisance span (rank < K)"
    old_excluded = (K < n_nuis + 1)
    row = dict(config=config, note=note, K=K, n_nuisance=n_nuis, n_nonzero=int(n_kept),
               rank_at_svd_rtol=rank, singular_values=[float(x) for x in s],
               tperp_over_that=ratio, R2=(None if not np.isfinite(R2_new) else float(R2_new)),
               one_minus_R2=(None if not np.isfinite(R2_new) else float(1 - R2_new)),
               sigma_a_tg_new=(None if not np.isfinite(sa_new) else float(sa_new)),
               mmin_tg_new=(None if not np.isfinite(sa_new) else float(3 * sa_new)),
               sigma_a_tg_old=(None if not np.isfinite(sa_old) else float(sa_old)),
               mmin_tg_old=(None if not np.isfinite(sa_old) else float(3 * sa_old)),
               old_shortcut_fired=bool(old_excluded),
               mmin_tg_archived=archived,
               verdict=verdict)
    if np.isfinite(sa_new) and np.isfinite(sa_old):
        row["rel_diff_new_vs_old"] = float(abs(sa_new / sa_old - 1))
    if archived is not None and np.isfinite(sa_new):
        row["rel_diff_new_vs_archive"] = float(abs(3 * sa_new / archived - 1))
    tag = f"{3 * sa_new:9.4f} Tg" if np.isfinite(sa_new) else "      inf   "
    tago = f"{3 * sa_old:9.4f}" if np.isfinite(sa_old) else "      inf"
    print(f"  {config:58s} K={K:2d} nuis={n_nuis} rank={rank} "
          f"|t_perp|/|t|={ratio:9.2e}  new {tag} old {tago}"
          + (f" arch {archived:9.4f}" if archived is not None else "")
          + f"  smin/smax={s[-1] / s[0]:.1e}" if len(s) else "")
    ROWS.append(row)
    return row


def main():
    t0 = time.time()
    res = {"tolerances": dict(SVD_RTOL=SVD_RTOL, TPERP_RTOL=TPERP_RTOL),
           "element_floor_880_m1": float(SIG_874_ABS), "alpha_floor_m1": float(ALPHA_FLOOR)}
    thr = json.load(open(THR_JSON))
    prob = json.load(open(PROB_JSON))
    cq = thr["results"]["CALIBRATED quiet (LM65T223, 2-comp)"]

    ri_kp = SilicaRefractiveIndex(str(_PARENT / "data/optics/SiO2_KitamuraPopova.yml"))
    sai = create_silica_sai_layer(ALT, 1.0, refractive_index=ri_kp,
                                  rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL)
    ri65, wt65, T65 = member_ri("LM65T223")
    alpha_cal = glossac_2025N_quiet_profile()

    # ---- PART 0: the baseline band / window from the cached engine ----------------------
    print("\nPART 0: baseline band / window rows from the cached 2-D engine (quiet bracket)")
    es_band = ElementSet("band 8-13 @0.1", thr["band01_elements_um"], 0.10, thr["band01_floors_m1"])
    es_win = ElementSet("window 7.8-9.3 @0.1", thr["window_elements_um"], 0.10, thr["window_floors_m1"])
    assert np.allclose(es_win.c, es_band.c[:15]) and np.allclose(es_win.floors, es_band.floors[:15])
    d = np.load(ENGINE_DIR / "engine_band_quiet.npz")
    cols_band = [d["ext0"][iz], d["ext1"][iz], d["er0"][iz], d["es0"][iz], d["er1"][iz], d["es1"][iz]]
    t_band = es_band.prof(sai)[iz]
    print(f"  silica band profile: {time.time() - t0:.0f} s")
    n_band = es_band.n
    i_win = np.r_[np.arange(N_VIS), N_VIS + np.arange(15)]      # window = the first 15 band elements
    res["baseline"] = {}
    for sname, es, isel, key, rkeys in (
            ("band", es_band, np.arange(n_band), "band 8-13 @0.1: measured floors",
             ("triplet + band01", "band01 alone")),
            ("window", es_win, i_win, "window 7.8-9.3 @0.1: measured floors",
             ("triplet + window", "window alone"))):
        cols = [c[isel] for c in cols_band]
        t = t_band[isel]
        s_tot = cols[0] + cols[1]
        sig = np.r_[np.maximum(VIS_PHI * s_tot[:N_VIS], ALPHA_FLOOR), es.floors]
        offs = (np.arange(len(isel)) >= N_VIS).astype(float)
        nsets = {'A2+rs (5-param)': cols[:4], 'full 7-param': cols,
                 '7-param + MIR offs': cols + [offs]}
        rosters = {rkeys[0]: np.arange(len(isel)), rkeys[1]: np.arange(N_VIS, len(isel))}
        for ns, cl in nsets.items():
            for rl, idx in rosters.items():
                arch = cq[key][ns][rl]["mmin_tg"]
                r = evaluate(f"{sname}: {ns}, {rl}", idx, cl, sig, t, archived=arch)
                res["baseline"][f"{sname} | {ns} | {rl}"] = r
    bl = res["baseline"]["band | full 7-param | triplet + band01"]["mmin_tg_new"]
    wl_ = res["baseline"]["window | full 7-param | triplet + window"]["mmin_tg_new"]
    print(f"  baseline check: band {bl:.6f} Tg (quoted {ARCH_BAND}), window {wl_:.6f} Tg (quoted {ARCH_WIN})")
    res["baseline_check"] = dict(band_tg=bl, window_tg=wl_, band_quoted=ARCH_BAND, window_quoted=ARCH_WIN,
                                 band_ok=bool(abs(bl - ARCH_BAND) < 5e-6), window_ok=bool(abs(wl_ - ARCH_WIN) < 5e-6))

    # ---- PART 1: the single 8.80-um element, 7-channel machinery -----------------------
    print("\nPART 1: single 0.1-um element at 8.80 um (7-channel machinery, measured vis/NIR errors)")
    bg = TwoComponentBackground(alpha_cal, CAL_QUIET, ri65, wt65, T65)
    exts = bg.component_ext()
    s_total = np.sum(exts, axis=0)
    jr_f, js_f = bg.psd_derivs(0)
    jr_c, js_c = bg.psd_derivs(1)
    t_sil = banded_ext(sai)
    print(f"  background columns: {time.time() - t0:.0f} s")
    nsets1 = {'A1+rs (4-param)': [s_total, jr_f, js_f],
              'A2 (3-param)': [exts[0], exts[1]],
              'A2+rs (5-param)': [exts[0], exts[1], jr_f, js_f],
              'A2+rs+rc (6-param)': [exts[0], exts[1], jr_f, js_f, jr_c],
              'full 7-param': [exts[0], exts[1], jr_f, js_f, jr_c, js_c]}
    rosters1 = {'(a) triplet alone 448/756/1544': [0, 1, 5],
                '(b) triplet + 8.80 um element': [0, 1, 5, I874],
                'vis/NIR six alone': [0, 1, 2, 3, 4, 5],
                'baseline roster (+1021, 1250) + element': [0, 1, 3, 4, 5, I874],
                'extended roster (+869) + element': [0, 1, 2, 3, 4, 5, I874]}
    arch_names = {'(b) triplet + 8.80 um element': 'minimal  (Wrana3 + 8.74)',
                  'baseline roster (+1021, 1250) + element': 'baseline (+1021, 1250)',
                  'extended roster (+869) + element': 'extended (+869)'}
    res["part1_single_element"] = {}
    res["part1_inputs"] = dict(wl_nm=WL_NM.tolist(), phi=[None if not np.isfinite(p) else float(p) for p in PHI],
                               s_fine=exts[0].tolist(), s_coarse=exts[1].tolist(), t_sil=t_sil.tolist(),
                               jr_f=jr_f.tolist(), js_f=js_f.tolist(), jr_c=jr_c.tolist(), js_c=js_c.tolist())
    for fl_lab, fl in (("fractional", 0.0), ("with SNR=2000 floor", ALPHA_FLOOR)):
        sig = noise(s_total, fl)
        print(f"  -- {fl_lab} --")
        for ns, cl in nsets1.items():
            for rl, idx in rosters1.items():
                arch = None
                if rl in arch_names and ns in cq[fl_lab]:
                    a = cq[fl_lab][ns][arch_names[rl]]["mmin_tg"]
                    arch = None if (a is None or not np.isfinite(a)) else a
                r = evaluate(f"element: {ns}, {rl} [{fl_lab}]", idx, cl, sig, t_sil, archived=arch)
                res["part1_single_element"][f"{fl_lab} | {ns} | {rl}"] = r

    # ---- PART 2: the continuum element -------------------------------------------------
    print("\nPART 2: continuum element (0.1-um boxcar) with the triplet (K = 4) and the six-channel roster (K = 7)")
    res["part2_continuum"] = {}
    for cont_um in (3.8, 4.0):
        wl_c = np.r_[WL_NM[:3], cont_um * 1000.0]
        bext_c = make_bext(wl_c, np.r_[np.zeros(3), SPECTRAL_RES_UM])
        bg_c = GenBackground(bext_c, alpha_cal, CAL_QUIET, ri65, wt65, T65)
        exts_c = bg_c.component_ext()
        s_tc = np.sum(exts_c, axis=0)
        jrf, jsf = bg_c.psd_derivs(0)
        jrc, jsc = bg_c.psd_derivs(1)
        t_c = bext_c(sai)
        nsets_c = {'A1+rs (4-param)': [s_tc, jrf, jsf], 'A2 (3-param)': [exts_c[0], exts_c[1]],
                   'A2+rs (5-param)': [exts_c[0], exts_c[1], jrf, jsf],
                   'A2+rs+rc (6-param)': [exts_c[0], exts_c[1], jrf, jsf, jrc],
                   'full 7-param': [exts_c[0], exts_c[1], jrf, jsf, jrc, jsc]}
        for sig_mir in (SIG_874_ABS, 1.5e-8):
            sig = np.r_[np.maximum(PHI[:3] * s_tc[:3], ALPHA_FLOOR), sig_mir]
            for ns, cl in nsets_c.items():
                arch = None
                if ns == 'A1+rs (4-param)' and sig_mir == 1.5e-8:
                    arch = prob["continuum"][f"wrana3+{cont_um}um sig1.5e-08 A1+rs"]["mmin_tg"]
                r = evaluate(f"(c) triplet + {cont_um} um: {ns} [floor {sig_mir:.3e}]", np.arange(4), cl, sig, t_c,
                             archived=arch)
                res["part2_continuum"][f"triplet + {cont_um} um | floor {sig_mir:.3e} | {ns}"] = r
    # the archived six-channel + 3.8-um comparison (K = 7)
    wl_e = np.r_[WL_NM[:I874], 3800.0]
    bext_e = make_bext(wl_e, np.r_[np.zeros(I874), SPECTRAL_RES_UM])
    bg_e = GenBackground(bext_e, alpha_cal, CAL_QUIET, ri65, wt65, T65)
    exts_e = bg_e.component_ext(); s_te = np.sum(exts_e, axis=0)
    jrf, jsf = bg_e.psd_derivs(0); jrc, jsc = bg_e.psd_derivs(1)
    t_e = bext_e(sai)
    sig_e = np.r_[np.maximum(PHI[:I874] * s_te[:I874], ALPHA_FLOOR), 1.5e-8]
    for ns, cl in (('A2+rs (5-param)', [exts_e[0], exts_e[1], jrf, jsf]),
                   ('full 7-param', [exts_e[0], exts_e[1], jrf, jsf, jrc, jsc])):
        arch = prob["continuum"][f"extended+3.8um {ns}"]["mmin_tg"]
        r = evaluate(f"six vis/NIR + 3.8 um: {ns} [floor 1.5e-8]", np.arange(7), cl, sig_e, t_e, archived=arch)
        res["part2_continuum"][f"six + 3.8 um | floor 1.500e-08 | {ns}"] = r
    print(f"  continuum: {time.time() - t0:.0f} s")

    # ---- PART 3: log-spaced vis/NIR sets, K = 3 ... 8 ---------------------------------
    print("\nPART 3: K log-spaced vis/NIR channels 385-1550 nm, phi = 4 % (K = 3 ... 8)")
    res["part3_ksweep"] = {}
    res["part3_note"] = ("calibrated_background_problem.py scans KS = [4, 5, 6, 7, 8, 12, ...]: K = 3 was never "
                         "scanned; the archive holds None for 'full 7-param' at K = 4, 5, 6 and for "
                         "'A2+rs (5-param)' at K = 4, all returned by the K < N + 1 shortcut")
    for K in range(3, 9):
        wl = np.geomspace(385.0, 1550.0, K)
        bext = make_bext(wl)
        bgk = GenBackground(bext, alpha_cal, CAL_QUIET, ri65, wt65, T65)
        ek = bgk.component_ext(); s_tk = np.sum(ek, axis=0)
        jrf, jsf = bgk.psd_derivs(0); jrc, jsc = bgk.psd_derivs(1)
        tk = bext(sai)
        sig = PHI_SWEEP * s_tk
        nsets_k = {'A2 (3-param)': [ek[0], ek[1]], 'A2+rs (5-param)': [ek[0], ek[1], jrf, jsf],
                   'A2+rs+rc (6-param)': [ek[0], ek[1], jrf, jsf, jrc],
                   'full 7-param': [ek[0], ek[1], jrf, jsf, jrc, jsc]}
        for ns, cl in nsets_k.items():
            arch = None
            if str(K) in prob["ksweep"] and ns in prob["ksweep"][str(K)]:
                arch = prob["ksweep"][str(K)][ns]["mmin_tg"]
            r = evaluate(f"(d) K={K} log-spaced vis/NIR: {ns}", np.arange(K), cl, sig, tk, archived=arch)
            res["part3_ksweep"][f"K={K} | {ns}"] = r
    print(f"  K-sweep: {time.time() - t0:.0f} s")

    # ---- summary ----------------------------------------------------------------------
    finite_both = [r for r in ROWS if r["mmin_tg_new"] is not None and r["mmin_tg_old"] is not None]
    newly_finite = [r for r in ROWS if r["mmin_tg_new"] is not None and r["mmin_tg_old"] is None]
    newly_inf = [r for r in ROWS if r["mmin_tg_new"] is None and r["mmin_tg_old"] is not None]
    arch_cmp = [r for r in ROWS if r.get("rel_diff_new_vs_archive") is not None]
    res["summary"] = dict(
        n_configurations=len(ROWS),
        n_finite_both=len(finite_both),
        max_rel_diff_new_vs_old=max((r["rel_diff_new_vs_old"] for r in finite_both), default=0.0),
        n_bit_identical_new_vs_old=sum(1 for r in finite_both if r["sigma_a_tg_new"] == r["sigma_a_tg_old"]),
        n_archive_comparisons=len(arch_cmp),
        max_rel_diff_new_vs_archive=max((r["rel_diff_new_vs_archive"] for r in arch_cmp), default=0.0),
        newly_finite=[r["config"] for r in newly_finite],
        newly_infinite=[r["config"] for r in newly_inf],
        shortcut_cases=[dict(config=r["config"], K=r["K"], n_nuisance=r["n_nuisance"], rank=r["rank_at_svd_rtol"],
                             tperp_over_that=r["tperp_over_that"], verdict=r["verdict"], mmin_tg_new=r["mmin_tg_new"])
                        for r in ROWS if r["old_shortcut_fired"]],
        runtime_s=time.time() - t0)
    res["rows"] = ROWS
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1)
    with open(OUT_TXT, "w") as f:
        f.write(f"# round 70 item 6: marginal() span test, SVD_RTOL={SVD_RTOL:g}, TPERP_RTOL={TPERP_RTOL:g}\n")
        f.write(f"{'configuration':70s} {'K':>2s} {'nuis':>4s} {'rank':>4s} {'|t_perp|/|t|':>12s} "
                f"{'smin/smax':>9s} {'M_min new':>10s} {'M_min old':>10s} {'archived':>10s}  verdict\n")
        for r in ROWS:
            s = r["singular_values"]
            f.write(f"{r['config']:70s} {r['K']:2d} {r['n_nuisance']:4d} {r['rank_at_svd_rtol']:4d} "
                    f"{r['tperp_over_that']:12.3e} {(s[-1] / s[0]) if s else float('nan'):9.1e} "
                    f"{(r['mmin_tg_new'] if r['mmin_tg_new'] is not None else float('inf')):10.5f} "
                    f"{(r['mmin_tg_old'] if r['mmin_tg_old'] is not None else float('inf')):10.5f} "
                    f"{(r['mmin_tg_archived'] if r['mmin_tg_archived'] is not None else float('nan')):10.5f}  "
                    f"{r['verdict']}\n")
    print("\nSUMMARY")
    for k, v in res["summary"].items():
        if k != "shortcut_cases":
            print(f"  {k}: {v}")
    print(f"archived -> {OUT_JSON}\n            {OUT_TXT}")


if __name__ == "__main__":
    main()
