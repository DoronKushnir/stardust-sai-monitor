"""adopted_robustness.py -- round 75 (paper 1, referee round 6 items 2 and 5, left open in
round 74): the robustness scans of Table 6 (Sect. 5.2, Appendix F) recomputed under the ADOPTED
measurement model (Table 2), i.e. the joint element-observable retrieval with the full covariance
(radiometric term, per-line database scatter, measured ACE discrepancy term with its Gaussian
inter-element correlation), the 46-parameter state with a flat aerosol prior and the heritage
triplet, at 20 km, through reproduce/adopted_model_engine.py.  Every row of Table 6 that is a
perturbation of the forward model (background microphysics, member optics, enlarged background
family, background loading, silica size distribution, optics, coagulation, sulfate coating,
visible/NIR noise, measurement altitude) is rebuilt with the same Mie machinery as round 70 and
evaluated (i) under the adopted model and (ii) in the diagonal-floor scenario on the same rows (the
two-stage procedure on the adopted floors), the latter as the reproduction check against the
round-53 to round-67 archives behind Table 6.  Rows that concern the measurement model itself (the
discrepancy level, its inter-element correlation, the gray offset, uniform floors) are evaluated as
covariance scenarios; the element-width row is not repeated (no measured scatter exists at 0.05
um) and the vertically coherent database row is superseded by the multi-tangent inverse of
reproduce/multitangent_retrieval.py.

Run from the repo root (about 30-60 min of Mie integrals in 5 processes):
  python reproduce/adopted_robustness.py
Writes outputs/adopted_robustness.json (+ _profiles.npz cache) and
tables/robust_summary.tex (two new columns beside the scenario columns).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))

OUT = ROOT / "outputs/adopted_robustness.json"
CACHE = ROOT / "outputs/adopted_robustness_profiles.npz"
TEX = ROOT / "tables/robust_summary.tex"
THIRD_MODE = dict(rmed_nm=150.0, sigma=1.4, fine_share=0.15)       # design_sensitivity_calibrated.py (round 53)
SECOND_MEMBER = "B70T215"
FN = {'1 Tg/yr': {1: 0.89, 2: 0.10, 4: 0.015, 8: 0.005, 16: 0.0},   # Lederer (2026) equilibrium mass fractions
      '8 Tg/yr': {1: 0.585, 2: 0.26, 4: 0.12, 8: 0.03, 16: 0.005}}
for _k in FN:
    _s = sum(FN[_k].values()); FN[_k] = {n: v / _s for n, v in FN[_k].items()}
MU_COAT = (0.1, 0.3)
WT, TK = 65.0, 223.0


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- Mie builds (workers)
def _profiles(layer, lam_c, lam_o, wl_t):
    return (layer.extinction_profile_m1(lam_c * 1e-6), layer.extinction_profile_m1(lam_o * 1e-6), layer.extinction_profile_m1(wl_t * 1e-9))


def _bg_columns(bg, comps, lam_c, lam_o, wl_t, modes=(0, 1)):
    """[A_i, ..., (r_i, s_i), ...] per-shell profiles of a TwoComponentBackground (round-70 finite differences, 5 %)."""
    amp, der = [], []
    for i in modes:
        amp.append(_profiles(bg.column(i), lam_c, lam_o, wl_t))
        r0, s0 = comps[i][0], comps[i][1]
        dr, ds = 0.05 * r0, 0.05 * s0
        pr = [_profiles(bg.column(i, dr=+dr), lam_c, lam_o, wl_t), _profiles(bg.column(i, dr=-dr), lam_c, lam_o, wl_t)]
        ps = [_profiles(bg.column(i, ds=+ds), lam_c, lam_o, wl_t), _profiles(bg.column(i, ds=-ds), lam_c, lam_o, wl_t)]
        der.append(([(a - b) / (2 * dr) for a, b in zip(pr[0], pr[1])], [(a - b) / (2 * ds) for a, b in zip(ps[0], ps[1])]))
    return amp, der


def build(job):
    """one variant -> dict of arrays; runs in a worker process."""
    name = job["name"]
    t0 = time.time()
    import element_observable_closure as C
    import calibrated_background_thresholds as cbt
    from saimon.sai import create_silica_sai_layer, SILICA_DENSITY_KG_M3
    from saimon.materials import SilicaRefractiveIndex
    lam_c, lam_o, wl_t = C.lam_coarse, C.lam_outer, C.WL_TRIPLET_NM
    out = {}

    def stack(cols):      # list of (c, o, t) -> three arrays (n_cols, nz, n_wl)
        return dict(c=np.array([x[0] for x in cols]), o=np.array([x[1] for x in cols]), t=np.array([x[2] for x in cols]))

    def sil(ri=None, rmed=cbt.RMED_SIL_NM, sg=cbt.SIGMA_SIL):
        ri = ri or SilicaRefractiveIndex(str(ROOT / "data/optics/SiO2_KitamuraPopova.yml"))
        return create_silica_sai_layer(cbt.ALT, 1.0, refractive_index=ri, rmed_nm=rmed, sigma=sg)

    if job["kind"] == "background":          # six sulfate columns of a (calibration, member) pair
        comps = job["comps"]; ri, wt, Tm = cbt.member_ri(job["member"])
        bg = cbt.TwoComponentBackground(cbt.glossac_2025N_quiet_profile(), comps, ri, wt, Tm)
        amp, der = _bg_columns(bg, comps, lam_c, lam_o, wl_t)
        cols = [amp[0], amp[1], der[0][0], der[0][1], der[1][0], der[1][1]]          # A_f, A_c, r_f, s_f, r_c, s_c
        out = stack(cols)
    elif job["kind"] == "member_column":     # the second member's total background as one column
        ri, wt, Tm = cbt.member_ri(job["member"])
        bg = cbt.TwoComponentBackground(cbt.glossac_2025N_quiet_profile(), cbt.CAL_QUIET, ri, wt, Tm)
        p = [_profiles(c, lam_c, lam_o, wl_t) for c in bg.columns()]
        out = stack([tuple(p[0][k] + p[1][k] for k in range(3))])
    elif job["kind"] == "third_mode":        # six base columns with the reduced fine share + three columns of the third mode
        f_fine = cbt.CAL_QUIET[0][2] * THIRD_MODE["fine_share"]
        comps3 = [(cbt.CAL_QUIET[0][0], cbt.CAL_QUIET[0][1], cbt.CAL_QUIET[0][2] - f_fine), cbt.CAL_QUIET[1],
                  (THIRD_MODE["rmed_nm"], THIRD_MODE["sigma"], f_fine)]
        ri, wt, Tm = cbt.member_ri("LM65T223")
        bg = cbt.TwoComponentBackground(cbt.glossac_2025N_quiet_profile(), comps3, ri, wt, Tm)
        amp, der = _bg_columns(bg, comps3, lam_c, lam_o, wl_t, modes=(0, 1, 2))
        out = stack([amp[0], amp[1], der[0][0], der[0][1], der[1][0], der[1][1], amp[2], der[2][0], der[2][1]])
    elif job["kind"] == "silica":
        kw = job.get("kw", {})
        if "franta" in kw:
            lay = sil(ri=SilicaRefractiveIndex(str(ROOT / "data/optics/SiO2_Franta2016.yml")))
        else:
            lay = sil(**kw)
        out = stack([_profiles(lay, lam_c, lam_o, wl_t)])
    elif job["kind"] == "agglomerate":
        fn = FN[job["case"]]
        acc = None
        for n, f in fn.items():
            if f <= 0:
                continue
            p = _profiles(sil(rmed=cbt.RMED_SIL_NM * n ** (1 / 3)), lam_c, lam_o, wl_t)
            acc = [f * x for x in p] if acc is None else [a + f * x for a, x in zip(acc, p)]
        out = stack([tuple(acc)])
    elif job["kind"] == "coating":
        from sensitivity_sulfate_coating import CoatedSilicaLayer
        from saimon.sulfate_background import sulfate_solution_density_kg_m3
        ri_kp = SilicaRefractiveIndex(str(ROOT / "data/optics/SiO2_KitamuraPopova.yml"))
        ri65, _, _ = cbt.member_ri("LM65T223")
        v = job["mu"] * SILICA_DENSITY_KG_M3 / sulfate_solution_density_kg_m3(WT, TK)
        lay = CoatedSilicaLayer(sil(), ri_kp, ri65, v, cbt.RMED_SIL_NM, cbt.SIGMA_SIL)
        out = stack([_profiles(lay, lam_c, lam_o, wl_t)])
    out["name"] = name
    out["seconds"] = time.time() - t0
    return out


JOBS = [
    dict(name="postHT", kind="background", comps=[(100.0, 1.5, 0.9561), (1500.0, 1.8, 0.0439)], member="LM72T213"),
    dict(name="biermann_bg", kind="background", comps=[(60.0, 1.6, 0.8211), (1500.0, 1.8, 0.1789)], member=SECOND_MEMBER),
    dict(name="biermann_column", kind="member_column", member=SECOND_MEMBER),
    dict(name="third_mode", kind="third_mode"),
    dict(name="sig1.5", kind="silica", kw=dict(sg=1.5)), dict(name="sig1.8", kind="silica", kw=dict(sg=1.8)),
    dict(name="rmed150", kind="silica", kw=dict(rmed=150.0)), dict(name="rmed600", kind="silica", kw=dict(rmed=600.0)),
    dict(name="franta", kind="silica", kw=dict(franta=True)),
    dict(name="agg1", kind="agglomerate", case="1 Tg/yr"), dict(name="agg8", kind="agglomerate", case="8 Tg/yr"),
    dict(name="coat0.1", kind="coating", mu=0.1), dict(name="coat0.3", kind="coating", mu=0.3),
]


def build_all():
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k].item() for k in d.files}
    t0 = time.time()
    with mp.get_context("spawn").Pool(5) as pool:
        res = pool.map(build, JOBS)
    prof = {r["name"]: {k: r[k] for k in ("c", "o", "t", "seconds")} for r in res}
    np.savez_compressed(CACHE, **{k: np.array(v, dtype=object) for k, v in prof.items()})
    log(f"profiles built in {time.time()-t0:.0f} s: " + ", ".join(f"{k} {v['seconds']:.0f} s" for k, v in prof.items()))
    return prof


# ---------------------------------------------------------------- evaluation
def main():
    prof = build_all()
    import adopted_model_engine as E
    C, cbt = E.C, E.cbt
    base = (C.prof_c, C.prof_o, C.prof_t)

    def with_bg(six):     # silica column of round 70 + six sulfate columns
        return (np.vstack([base[0][:1], six["c"][:6]]), np.vstack([base[1][:1], six["o"][:6]]), np.vstack([base[2][:1], six["t"][:6]]))

    def evaluate(tm, extra_cols=None, vis_scale=1.0, S_mir=None, label=""):
        r = {}
        for rows in ("band", "window"):
            r[rows] = 3 * E.joint_sigma(tm, S_mir=S_mir, rows=rows, extra_cols=extra_cols, vis_scale=vis_scale) * tm.rho
            r[rows + "_scenario"] = 3 * E.two_stage_sigma(tm, E.s_adopt_od, rows=rows, vis_scale=vis_scale) if extra_cols is None else None
        log(f"  {label:58s} adopted band {r['band']:.4f} window {r['window']:.4f}" +
            (f" | scenario {r['band_scenario']:.4f} / {r['window_scenario']:.4f}" if r['band_scenario'] is not None else ""))
        return r

    res = dict(script="reproduce/adopted_robustness.py", round=75, rows={})
    ref = E.REF
    res["rows"]["baseline"] = evaluate(ref, label="baseline (20 km)")
    # --- sulfate side
    res["rows"]["post_hunga_tonga"] = evaluate(E.TangentModel(20e3, bg_profiles=with_bg(prof["postHT"])), label="post-Hunga Tonga calibration (LM72T213)")
    res["rows"]["member_biermann"] = evaluate(E.TangentModel(20e3, bg_profiles=with_bg(prof["biermann_bg"])), label="member optics Biermann 70 wt%, 215 K")
    # enlarged family: third mode (9 sulfate columns) and the second member as a free column
    tmode = prof["third_mode"]
    tm3 = E.TangentModel(20e3, bg_profiles=with_bg(tmode))
    ex_c = np.zeros((E.n_mir, 3)); ex_t = np.zeros((3, 3))
    ch = E.tangent_to_slant_paths(np.array([20e3]), cbt.ALT)[0]
    for k in range(3):
        sl_c = ch @ tmode["c"][6 + k]; sl_o = ch @ tmode["o"][6 + k]
        tau_f = np.interp(C.lam_um, C.lam_coarse, sl_c)
        ex_c[E.win, k] = C.element_mean(tm3.tau0, tau_f)
        ex_c[E.outer, k] = [sl_o[m].mean() for m in E._out_masks]
        ex_t[:, k] = tmode["t"][6 + k][tm3.iz]
    res["rows"]["third_mode"] = evaluate(tm3, extra_cols=(ex_c, ex_t), label="third sulfate mode free (amplitude, r_med, sigma)")
    bc = prof["biermann_column"]
    sl_c = ch @ bc["c"][0]; sl_o = ch @ bc["o"][0]
    ex_c = np.zeros((E.n_mir, 1)); ex_c[E.win, 0] = C.element_mean(ref.tau0, np.interp(C.lam_um, C.lam_coarse, sl_c)); ex_c[E.outer, 0] = [sl_o[m].mean() for m in E._out_masks]
    ex_t = bc["t"][0][ref.iz][:, None]
    res["rows"]["second_member_column"] = evaluate(ref, extra_cols=(ex_c, ex_t), label="second member (Biermann) as a free amplitude")
    # --- measurement-model rows (covariance scenarios at 20 km)
    UM = E.UM["models"]
    res["rows"]["gray_offset"] = dict(band=UM["B_plus_gray_offset"]["band_triplet"], window=UM["B_plus_gray_offset"]["window_triplet"], source="uncertainty_model (Table 2)")
    res["rows"]["uniform_level"] = evaluate(ref, S_mir=ref.cov_mir(kind="uniform"), label="discrepancy term uniform at the 8.80-um level")
    res["rows"]["disc_x2"] = evaluate(ref, S_mir=ref.cov_mir(disc_mult=2.0), label="discrepancy term x2")
    res["rows"]["disc_x0.5"] = evaluate(ref, S_mir=ref.cov_mir(disc_mult=0.5), label="discrepancy term x0.5")
    res["rows"]["all_terms_x2"] = evaluate(ref, S_mir=4.0 * ref.cov_mir(), label="all three MIR terms x2")
    res["rows"]["all_terms_x0.5"] = evaluate(ref, S_mir=0.25 * ref.cov_mir(), label="all three MIR terms x0.5")
    res["rows"]["exp_corr"] = dict(band_l0p1=UM["D_disc_exp_0p10"]["band_triplet"], band_l0p25=UM["E_disc_exp_0p25"]["band_triplet"],
                                   window_l0p1=UM["D_disc_exp_0p10"]["window_triplet"], window_l0p25=UM["E_disc_exp_0p25"]["window_triplet"],
                                   band_indep=UM["C_disc_independent"]["band_triplet"], window_indep=UM["C_disc_independent"]["window_triplet"],
                                   source="uncertainty_model (Table 2)")
    # --- background loading: g = SAOD / SAOD_quiet (the quiet calibration SAOD_525 = 0.00446)
    saod0 = float(E.thr["saod_cal_20_25N_quiet"])
    res["rows"]["loading"] = {}
    for saod in (0.003, 0.010):
        g = saod / saod0
        res["rows"]["loading"][str(saod)] = evaluate(E.TangentModel(20e3, g_bg=g), label=f"background loading SAOD_525 = {saod} (g = {g:.2f})")
    # --- silica side
    for key, label in (("sig1.5", "silica PSD width 1.5"), ("sig1.8", "silica PSD width 1.8"), ("rmed150", "silica r_med 150 nm"), ("rmed600", "silica r_med 600 nm"),
                       ("franta", "silica optics Franta (2016)"), ("agg1", "coagulation 1 Tg/yr"), ("agg8", "coagulation 8 Tg/yr"),
                       ("coat0.1", "sulfate shell 0.1 of the silica mass"), ("coat0.3", "sulfate shell 0.3 of the silica mass")):
        p = prof[key]
        tm = E.TangentModel(20e3, template=p["c"][0], template_outer=p["o"][0], template_trip=p["t"][0])
        res["rows"][key] = evaluate(tm, label=label)
        res["rows"][key]["rho"] = tm.rho
    # --- instrument side
    for g in (0.5, 4.0):
        res["rows"][f"vis_noise_x{g:g}"] = evaluate(ref, vis_scale=g, label=f"visible/NIR noise x{g:g}")
    res["rows"]["altitude"] = {}
    for z in (16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0):
        tm = E.TangentModel(z * 1e3)
        r = evaluate(tm, label=f"altitude {z:.0f} km (discrepancy term constant)")
        r["band_atlas"] = 3 * E.joint_sigma(tm, S_mir=tm.cov_mir("atlas"), rows="band") * tm.rho
        r["window_atlas"] = 3 * E.joint_sigma(tm, S_mir=tm.cov_mir("atlas"), rows="window") * tm.rho
        r["rho"] = tm.rho; r["y880"] = float(tm.y0[C.i880])
        res["rows"]["altitude"][f"{z:.0f}"] = r
    res["atlas_ratio"] = E.ATLAS_RATIO
    res["note"] = ("3 sigma per 0.5-km shell at 20 km (slant amplitude x rho); 'adopted' = joint retrieval under the Table-2 model, flat aerosol prior; "
                   "'scenario' = two-stage diagonal-floor procedure on the adopted (2.47e-3-anchored) floors with the same rows, the check against Table 6; "
                   "altitude rows: discrepancy term constant with altitude, or scaled by the measured ACE 16-19/19-22/22-25-km bins (atlas)")
    # ratios to the baselines (the percentages quoted in Sect. 5.2) and the per-kilometre altitude factors
    b0 = res["rows"]["baseline"]
    def pct(r, rows):
        return 100.0 * (r[rows] / b0[rows] - 1.0)
    res["percent_change"] = {k: dict(band=pct(v, "band"), window=pct(v, "window")) for k, v in res["rows"].items()
                             if isinstance(v, dict) and "band" in v and "window" in v and k not in ("baseline",)}
    res["percent_change"]["loading"] = {k: dict(band=pct(v, "band"), window=pct(v, "window")) for k, v in res["rows"]["loading"].items()}
    res["percent_change"]["altitude"] = {k: dict(band=pct(v, "band"), window=pct(v, "window")) for k, v in res["rows"]["altitude"].items()}
    alt = res["rows"]["altitude"]
    res["altitude_factor_per_km"] = dict(
        band_below_20=(alt["20"]["band"] / alt["16"]["band"]) ** 0.25, band_above_20=(alt["24"]["band"] / alt["20"]["band"]) ** 0.25,
        window_below_20=(alt["20"]["window"] / alt["16"]["window"]) ** 0.25, window_above_20=(alt["24"]["window"] / alt["20"]["window"]) ** 0.25,
        band_16_over_24=alt["16"]["band"] / alt["24"]["band"])
    res["factors"] = {k: dict(band=res["rows"][k]["band"] / b0["band"], window=res["rows"][k]["window"] / b0["window"])
                      for k in ("third_mode", "second_member_column", "gray_offset", "disc_x2", "disc_x0.5", "coat0.1", "coat0.3")}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    log("archived ->", OUT)
    log("percent changes: " + json.dumps({k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in res["percent_change"].items() if "band" in v}))
    log("altitude factors per km: " + json.dumps({k: round(v, 3) for k, v in res["altitude_factor_per_km"].items()}))
    write_table(res)
    write_altitude_table(res)


def write_altitude_table(res):
    T = ROOT / "tables/altitude_scan.tex"
    old = T.read_text().split("\n")
    a = res["rows"]["altitude"]
    f = lambda v: f"{v:.3f}"      # noqa: E731
    L = ["% Moved from sections/appendix_sensitivity.tex in round 44 (was an inline tabular).",
         "% Round 75: adopted-model columns added by reproduce/adopted_robustness.py (outputs/adopted_robustness.json); the scenario columns are the round-35c values.",
         "\\begin{table}[t]",
         "\\caption{Single-sounding threshold $M_{\\min}^{(3\\sigma)}$ against the tangent altitude for the $7.8$--$13~\\mu$m band and its $7.8$--$9.3~\\mu$m window (heritage triplet, six background parameters free beside the silica amplitude, calibrated undisturbed background). Adopted model: the joint retrieval of Table~\\ref{tab:error_budget} re-evaluated at each tangent, with the discrepancy term constant with altitude (in parentheses, scaled by the measured $16$--$19$ and $22$--$25$-km bins of Appendix~\\ref{app:acefloor}). Scenario: the diagonal-floor scenario with the per-element floors re-derived per tangent altitude.}",
         "\\label{tab:altitude_scan}", "\\centering", "\\small", "\\begin{tabular}{ccccc}", "\\toprule",
         "$z$ [km] & Band, adopted [Tg] & Window, adopted [Tg] & Band, scenario [Tg] & Window, scenario [Tg] \\\\", "\\midrule"]
    scen = {"16": ("0.056", "0.078"), "18": ("0.077", "0.108"), "20": ("0.101", "0.141"), "22": ("0.125", "0.175"), "24": ("0.146", "0.204")}
    for z in ("16", "18", "20", "22", "24"):
        r = a[z]
        L.append(f"${z}$ & {f(r['band'])} ({f(r['band_atlas'])}) & {f(r['window'])} ({f(r['window_atlas'])}) & ${scen[z][0]}$ & ${scen[z][1]}$ \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", "", "% ---- round-74 version ----"] + ["% " + l for l in old if l.strip()]
    T.write_text("\n".join(L) + "\n")
    log("table ->", T)


def write_table(res):
    r = res["rows"]
    f = lambda v: f"{v:.3f}" if v < 1.0 else f"{v:.2f}"      # noqa: E731
    old = TEX.read_text().split("\n")
    lines = ["% Moved from sections/design.tex in round 44 (tables live in tables/).",
             "% Round 75: regenerated by reproduce/adopted_robustness.py (outputs/adopted_robustness.json): the two 'adopted model'",
             "% columns are new; the scenario columns are the round-53 to round-67 values. The round-74 version of this file is kept below as comments.",
             "\\begin{table*}[t]",
             "\\caption{Sensitivity of the single-sounding threshold $M_{\\min}^{(3\\sigma)}$ at $z=20$~km (heritage triplet, the six background parameters free, silica amplitude fitted, calibrated undisturbed background) for the $7.8$--$13~\\mu$m band at $0.1~\\mu$m sampling and for its $7.8$--$9.3~\\mu$m window subset. "
             "``Adopted'' is the joint retrieval under the measurement model of Table~\\ref{tab:error_budget} (baselines $0.119$ and $0.168$~Tg); ``scenario'' is the diagonal-floor scenario of that table (baselines $0.101$ and $0.141$~Tg), in which the scans of Appendix~\\ref{app:sensitivity} were made. In the adopted column the rows that scale the floors scale the discrepancy term (all three mid-infrared terms in parentheses), the constant offset and the exponential correlation are the scenarios of Table~\\ref{tab:error_budget}, the element-width row is not repeated (no measured scatter exists at $0.05~\\mu$m), and the vertically coherent database row is superseded by the multi-tangent retrieval of Appendix~\\ref{app:tracegas_geometry}. The altitude rows carry the discrepancy term constant with altitude (scaled by the measured $16$--$19$ and $22$--$25$-km bins of Appendix~\\ref{app:acefloor} in parentheses).}",
             "\\label{tab:robust_summary}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{adjustbox}{max width=\\textwidth}", "\\begin{tabular}{p{7.0cm}cccc}", "\\toprule",
             "Perturbation & \\multicolumn{2}{c}{Adopted model [Tg]} & \\multicolumn{2}{c}{Scenario [Tg]} \\\\",
             " & Band & Window & Band & Window \\\\", "\\midrule"]
    def row(label, a_b, a_w, s_b, s_w):
        lines.append(f"{label} & {a_b} & {a_w} & {s_b} & {s_w} \\\\")
    b = r["baseline"]
    row("Baseline", f(b["band"]), f(b["window"]), "$0.101$", "$0.141$")
    lines.append("\\midrule")
    row("Sulfate microphysics: post--Hunga Tonga calibration", f(r["post_hunga_tonga"]["band"]), f(r["post_hunga_tonga"]["window"]), "$0.100$", "$0.146$")
    row("Sulfate member optics: Biermann $70$~wt\\,\\%, $215$~K", f(r["member_biermann"]["band"]), f(r["member_biermann"]["window"]), "$0.103$", "$0.143$")
    row("Constant offset common to all MIR elements, free", f(r["gray_offset"]["band"]), f(r["gray_offset"]["window"]), "$0.102$", "$0.26$")
    row("Uniform floors (adopted: discrepancy term at the $8.80$-$\\mu$m level everywhere)", f(r["uniform_level"]["band"]), f(r["uniform_level"]["window"]), "$0.083$", "$0.113$")
    row("MIR floors $\\times2$ / $\\times0.5$ (adopted: discrepancy term; all terms in parentheses)",
        f"{f(r['disc_x2']['band'])} / {f(r['disc_x0.5']['band'])} ({f(r['all_terms_x2']['band'])} / {f(r['all_terms_x0.5']['band'])})",
        f"{f(r['disc_x2']['window'])} / {f(r['disc_x0.5']['window'])} ({f(r['all_terms_x2']['window'])} / {f(r['all_terms_x0.5']['window'])})",
        "$0.202$ / $0.051$", "$0.282$ / $0.071$")
    row("Database line-intensity term vertically coherent (Sect.~\\ref{sec:dz_scaling}; adopted: Appendix~\\ref{app:tracegas_geometry})", "--", "--", "$0.078$", "$0.109$")
    e = r["exp_corr"]
    row("Floors exponentially correlated, $\\ell=0.1$ / $0.25~\\mu$m (independent in parentheses)",
        f"{f(e['band_l0p1'])} / {f(e['band_l0p25'])} ({f(e['band_indep'])})", f"{f(e['window_l0p1'])} / {f(e['window_l0p25'])} ({f(e['window_indep'])})",
        "$0.124$ / $0.107$ ($0.108$)", "$0.159$ / $0.131$ ($0.150$)")
    lo, hi = r["loading"]["0.003"], r["loading"]["0.01"]
    row("Background loading $\\mathrm{SAOD}_{525}=0.003$--$0.010$", f"{f(lo['band'])}--{f(hi['band'])}", f"{f(lo['window'])}--{f(hi['window'])}", "$0.101$--$0.101$", "$0.141$--$0.141$")
    row("Element width $0.05$ / $0.25~\\mu$m", "--", "--", "$0.079$ / $0.176$", "$0.108$ / $0.279$")
    row("Enlarged background family: third sulfate mode free ($r_\\text{med}$, $\\sigma$, amplitude; ten aerosol parameters\\ClaudeText{, $49$ in the joint state})", f(r["third_mode"]["band"]), f(r["third_mode"]["window"]), "$0.183$", "$0.337$")
    row("Enlarged background family: second optics member (Biermann) as a free amplitude (eight aerosol parameters\\ClaudeText{, $47$ in the joint state})", f(r["second_member_column"]["band"]), f(r["second_member_column"]["window"]), "$0.128$", "$0.220$")
    lines.append("\\midrule")
    row("Silica PSD width $\\sigma_\\text{psd}=1.5$ / $1.8$", f"{f(r['sig1.5']['band'])} / {f(r['sig1.8']['band'])}", f"{f(r['sig1.5']['window'])} / {f(r['sig1.8']['window'])}", "$0.107$ / $0.143$", "$0.156$ / $0.244$")
    row("Silica $r_\\text{med}=150$ / $600$~nm", f"{f(r['rmed150']['band'])} / {f(r['rmed600']['band'])}", f"{f(r['rmed150']['window'])} / {f(r['rmed600']['window'])}", "$0.102$ / $0.128$", "$0.140$ / $0.219$")
    row("Silica optics: \\citet{Franta2016}", f(r["franta"]["band"]), f(r["franta"]["window"]), "$0.109$", "$0.150$")
    row("Coagulation: $1$ / $8$~Tg\\,yr$^{-1}$", f"{f(r['agg1']['band'])} / {f(r['agg8']['band'])}", f"{f(r['agg1']['window'])} / {f(r['agg8']['window'])}", "$0.101$ / $0.103$", "$0.142$ / $0.146$")
    row("Sulfate shell, $0.1$ / $0.3$ of the silica mass (Table~\\ref{tab:coating_scans})", f"{f(r['coat0.1']['band'])} / {f(r['coat0.3']['band'])}", f"{f(r['coat0.1']['window'])} / {f(r['coat0.3']['window'])}", "$0.116$ / $0.138$", "$0.173$ / $0.236$")
    lines.append("Reststrahlen peak, $r_\\text{med}=150\\to600$~nm & \\multicolumn{4}{l}{$8.79\\to8.84~\\mu$m} \\\\")
    lines.append("\\midrule")
    row("Visible/NIR noise $\\times0.5$ / $\\times4$", f"{f(r['vis_noise_x0.5']['band'])} / {f(r['vis_noise_x4']['band'])}", f"{f(r['vis_noise_x0.5']['window'])} / {f(r['vis_noise_x4']['window'])}", "$0.101$ / $0.102$", "$0.141$ / $0.141$")
    a16, a24 = r["altitude"]["16"], r["altitude"]["24"]
    row("Altitude $16$ / $24$~km", f"{f(a16['band'])} / {f(a24['band'])} ({f(a16['band_atlas'])} / {f(a24['band_atlas'])})",
        f"{f(a16['window'])} / {f(a24['window'])} ({f(a16['window_atlas'])} / {f(a24['window_atlas'])})", "$0.056$ / $0.146$", "$0.078$ / $0.204$")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table*}", ""]
    lines += ["% ---- round-74 version of this table (before the adopted-model columns) ----"] + ["% " + l for l in old if l.strip()]
    TEX.write_text("\n".join(lines) + "\n")
    log("table ->", TEX)


if __name__ == "__main__":
    main()
