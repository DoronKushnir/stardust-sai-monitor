"""
design_sensitivity_calibrated.py -- the design-configuration and sensitivity
scans of Paper 1 Sect. 5 / Appendix app:sensitivity, re-run on the
CALIBRATED two-component background with the nominal configuration
(round 32, Doron): the Wrana heritage triplet (448, 756, 1544 nm) plus
the whole 8-13 um band at 0.1-um sampling (45 elements; round 33) or its
minimal 15-element subset, the 7.8-9.3 um window, measured
per-element floors, full 7-parameter background fit, z = 20 km.

Machinery imported bit-identically from calibrated_background_thresholds.py
(member optics, GloSSAC 20-25N undisturbed profile, TwoComponentBackground,
marginalized bound by stable projection).  The element grids and floors are
read from that script's archive.  Baselines are asserted against the
archive (window 0.134 Tg, band 0.169 Tg).

Scans (each for the window and the band; a third set, the window at
0.25-um elements, isolates resolution from coverage):
  A. silica PSD width sigma_psd in {1.05, 1.2, 1.31*, 1.5, 1.8} at
     r_med = 268 nm, and median radius r_med in {150, 200, 268*, 400,
     600} nm at sigma 1.31 (* = the Segev D300o nominal); plus the
     monochromatic reststrahlen peak wavelength versus r_med.
  B. silica optics: Franta (2016) in place of Kitamura+Popova.
  C. coagulation/agglomeration: Lederer (2026) equilibrium mass fractions
     f_N over N-monomer agglomerates (volume-equivalent spheres,
     r_med -> 268 N^(1/3) nm), 1 Tg/yr and 8 Tg/yr cases (digitized values
     as in reproduce/sensitivity_agglomerate.py).
  D. visible/NIR noise: the triplet's fractional errors and SNR floor
     scaled by g in {0.5, 1, 2, 4} (MIR floors held: systematic).
  E. measurement altitude z = 16-24 km (round 35c, Doron: the per-element
     MIR floors follow the trace-gas OE budget re-derived per tangent
     altitude, outputs/detectability_2d/floor_altitude_scan.csv, normalized
     at 20 km -- the "held" convention is retired; the silica layer is
     constant mixing ratio, the sulfate the GloSSAC profile).
  F. (read from the archives) sulfate microphysics brackets, member-optics
     swap, correlated MIR offset, MIR floor scan.
  G. enlarged background family (round 53, referee RC1 M9): (i) a THIRD
     sulfate lognormal mode with its own amplitude, r_med and sigma free
     (10 nuisance parameters; nominal intermediate mode r_med = 150 nm,
     sigma = 1.4, carrying 15 % of the fine mode's 525-nm extinction
     share, THIRD_MODE below); (ii) the SECOND laboratory member's
     background spectrum (Biermann 70 wt%/215 K, the optics-swap member)
     as an additional free amplitude column beside the nominal member
     (8 nuisance parameters).  Since the threshold is set by the silica
     component orthogonal to the nuisance span, both can only raise it.
  --dry-run writes the archive to the scratchpad (no figure, archive
  untouched); --family-only evaluates the baseline and scan G alone.
The sulfate DENSITY prior of the old appendix has no counterpart: with
both component amplitudes free, the density enters only through the
amplitudes, which are marginalized; only the optics shape matters, and
that is the member swap.

Run from the repo root:
  python reproduce/design_sensitivity_calibrated.py          # compute + plot
  python reproduce/design_sensitivity_calibrated.py --plot   # plot from archive
Output: outputs/design_sensitivity_calibrated.json,
        figures/design_sensitivity_calibrated.png (copied to figures/)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from calibrated_background_thresholds import (  # noqa: E402
    ALT, iz, RMED_SIL_NM, SIGMA_SIL, ALPHA_FLOOR, CAL_QUIET,
    TwoComponentBackground, glossac_2025N_quiet_profile, marginal,
    member_ri)
from saimon.materials import SilicaRefractiveIndex  # noqa: E402
from saimon.sai import create_silica_sai_layer  # noqa: E402

VIS_NM = np.array([448., 756., 1544.])
VIS_PHI = np.array([0.040, 0.032, 0.088])
N_VIS = 3
SIGMAS = (1.05, 1.2, 1.31, 1.5, 1.8)
RMEDS = (150., 200., 268., 400., 600.)
G_VIS = (0.5, 1.0, 2.0, 4.0)
Z_KM = (16, 17, 18, 19, 20, 21, 22, 23, 24)
# Lederer (2026) Fig. 2 equilibrium MASS fractions (reproduce/sensitivity_agglomerate.py)
FN = {'1 Tg/yr': {1: 0.89, 2: 0.10, 4: 0.015, 8: 0.005, 16: 0.0},
      '8 Tg/yr': {1: 0.585, 2: 0.26, 4: 0.12, 8: 0.03, 16: 0.005}}
for k in FN:
    s = sum(FN[k].values())
    FN[k] = {n: v / s for n, v in FN[k].items()}
OUT_JSON = _ROOT / "outputs" / "design_sensitivity_calibrated.json"
# G. enlarged family: third mode (rmed_nm, sigma, share of the FINE mode's
# 525-nm extinction fraction moved to it) and the second member's optics key
THIRD_MODE = dict(rmed_nm=150.0, sigma=1.4, fine_share=0.15)
SECOND_MEMBER = "B70T215"
FLOOR_SCAN = _ROOT / "outputs" / "detectability_2d" / "floor_altitude_scan.csv"


def oe_floor_scale(z_km):
    """MIR floor factor vs tangent altitude from the trace-gas OE scan
    (scan_floor_altitude.py), normalized to 1 at 20 km."""
    scan = np.loadtxt(FLOOR_SCAN, delimiter=",", skiprows=1)
    return float(np.interp(z_km, scan[:, 0], scan[:, 2])
                 / np.interp(20.0, scan[:, 0], scan[:, 2]))
FIG = _ROOT / "figures" / "design_sensitivity_calibrated.png"
FIG_PAPER = FIG   # repo: figures/ is the paper figure directory


class ElementSet:
    """Channel list = triplet + MIR boxcar elements; evaluates full
    extinction PROFILES (nz x nchan) so that any altitude can be sliced."""

    def __init__(self, name, centers_um, width_um, floors_m1):
        self.name = name
        self.c = np.asarray(centers_um, float)
        self.w = float(width_um)
        self.floors = np.asarray(floors_m1, float)
        self.sub = np.linspace(-self.w / 2, self.w / 2, 41)
        self.wl_nm = np.r_[VIS_NM, self.c * 1000.0]
        self.n = len(self.wl_nm)

    def prof(self, layer):
        out = layer.extinction_profile_m1(self.wl_nm * 1e-9).copy()
        for j, c in enumerate(self.c):
            out[:, N_VIS + j] = layer.extinction_profile_m1(
                (c + self.sub) * 1e-6).mean(axis=1)
        return out                                          # (nz, n)

    def background(self, comps, ri, wt, T):
        bg = TwoComponentBackground(glossac_2025N_quiet_profile(), comps,
                                    ri, wt, T)
        cols = bg.columns()
        exts = [self.prof(c) for c in cols]
        ders = []
        for i, (r0, s0, f) in enumerate(comps):
            dr, ds = bg.fd * r0, bg.fd * s0
            er = (self.prof(bg.column(i, dr=+dr))
                  - self.prof(bg.column(i, dr=-dr))) / (2 * dr)
            es = (self.prof(bg.column(i, ds=+ds))
                  - self.prof(bg.column(i, ds=-ds))) / (2 * ds)
            ders.append((er, es))
        return exts, ders

    def mmin(self, exts, ders, t_prof, k=iz, g_vis=1.0, mir_scale=1.0,
             nset='full', extra_cols=()):
        """3-sigma threshold [Tg] at altitude index k.  nset='full': every
        mode's amplitude plus its (r_med, sigma) derivatives (2 modes = the
        7-parameter fit, 3 modes = 10 parameters); 'std': amplitudes plus
        the fine mode's derivatives only.  extra_cols: further nuisance
        profiles (nz x n), e.g. a second member's background spectrum."""
        s_tot = sum(e[k] for e in exts)
        if nset == 'full':
            cols = [e[k] for e in exts] + [d[k] for pair in ders for d in pair]
        else:
            cols = [e[k] for e in exts] + [ders[0][0][k], ders[0][1][k]]
        cols += [c[k] for c in extra_cols]
        sig = np.r_[g_vis * np.maximum(VIS_PHI * s_tot[:N_VIS], ALPHA_FLOOR),
                    mir_scale * self.floors]
        sa, tperp, R2 = marginal(np.arange(self.n), cols, sig, t_prof[k])
        return 3 * sa, tperp, (1 - R2)


def compute(dry_run=False, family_only=False):
    thr = json.load(open(_ROOT / "outputs" /
                         "calibrated_background_thresholds.json"))
    # round 33 (Doron): one resolution, 0.1 um, for the whole 8-13 um band
    # (PART 4 of the thresholds script); the window is its 15-element subset
    sets = [ElementSet("window 7.8-9.3 @0.1", thr["window_elements_um"], 0.10,
                       thr["window_floors_m1"]),
            ElementSet("band 8-13 @0.1", thr["band01_elements_um"], 0.10,
                       thr["band01_floors_m1"])]
    cq = thr["results"]["CALIBRATED quiet (LM65T223, 2-comp)"]
    ref = {"window 7.8-9.3 @0.1": cq["window 7.8-9.3 @0.1: measured floors"][
               "full 7-param"]["triplet + window"]["mmin_tg"],
           "band 8-13 @0.1": cq["band 8-13 @0.1: measured floors"][
               "full 7-param"]["triplet + band01"]["mmin_tg"]}

    ri65, wt65, T65 = member_ri("LM65T223")
    ri_kp = SilicaRefractiveIndex("data/optics/SiO2_KitamuraPopova.yml")
    ri_fr = SilicaRefractiveIndex("data/optics/SiO2_Franta2016.yml")

    def sil(ri=ri_kp, rmed=RMED_SIL_NM, sg=SIGMA_SIL):
        return create_silica_sai_layer(ALT, 1.0, refractive_index=ri,
                                       rmed_nm=rmed, sigma=sg)

    arch = {"z_km": list(Z_KM), "sigmas": list(SIGMAS), "rmeds": list(RMEDS),
            "g_vis": list(G_VIS), "sets": {}}
    # reststrahlen peak wavelength vs r_med (monochromatic, z = 20 km)
    lam = np.arange(8.3, 9.31, 0.01)
    arch["peak_um_vs_rmed"] = {}
    for r in RMEDS:
        e = sil(rmed=r).extinction_profile_m1(lam * 1e-6)[iz, :]
        arch["peak_um_vs_rmed"][str(r)] = float(lam[np.argmax(e)])
    print("reststrahlen peak [um] vs r_med [nm]:",
          {k: round(v, 3) for k, v in arch["peak_um_vs_rmed"].items()})

    for es in sets:
        print(f"\n=== {es.name}: {len(es.c)} elements, floors "
              f"{es.floors.min():.2e}-{es.floors.max():.2e} m^-1 ===")
        exts, ders = es.background(CAL_QUIET, ri65, wt65, T65)
        t_nom = es.prof(sil())
        m0, tperp, r2 = es.mmin(exts, ders, t_nom)
        m0_5, _, _ = es.mmin(exts, ders, t_nom, nset='std')
        print(f"  baseline full 7-param: {m0:.4f} Tg (||t_perp|| {tperp:.1f}, "
              f"1-R2 {r2:.3f}); 5-param {m0_5:.4f}"
              + (f"; archived {ref[es.name]:.4f}" if es.name in ref else ""))
        if es.name in ref:
            assert abs(m0 - ref[es.name]) < 1e-3, "baseline reproduction"
        res = dict(baseline_7p=m0, baseline_5p=m0_5, tperp=tperp,
                   one_minus_R2=r2, n_elements=len(es.c))
        # G. enlarged background family (round 53, RC1 M9)
        f_fine = CAL_QUIET[0][2] * THIRD_MODE["fine_share"]
        comps3 = [(CAL_QUIET[0][0], CAL_QUIET[0][1], CAL_QUIET[0][2] - f_fine),
                  CAL_QUIET[1],
                  (THIRD_MODE["rmed_nm"], THIRD_MODE["sigma"], f_fine)]
        exts3, ders3 = es.background(comps3, ri65, wt65, T65)
        m3, tperp3, r23 = es.mmin(exts3, ders3, t_nom)
        ri_b, wt_b, T_b = member_ri(SECOND_MEMBER)
        exts_b, _ = es.background(CAL_QUIET, ri_b, wt_b, T_b)
        bg_b = exts_b[0] + exts_b[1]
        m_b, tperp_b, r2_b = es.mmin(exts, ders, t_nom, extra_cols=[bg_b])
        res["third_mode"] = m3
        res["third_mode_tperp"] = tperp3
        res["second_member"] = m_b
        res["second_member_tperp"] = tperp_b
        print(f"  enlarged family: third mode (r_med {THIRD_MODE['rmed_nm']:.0f} nm, "
              f"sigma {THIRD_MODE['sigma']}, {THIRD_MODE['fine_share']:.0%} of the fine "
              f"share; 10 params) {m3:.4f} Tg (||t_perp|| {tperp3:.1f}); "
              f"second member {SECOND_MEMBER} as free column (8 params) {m_b:.4f} Tg "
              f"(||t_perp|| {tperp_b:.1f})")
        if family_only:
            arch["sets"][es.name] = res
            continue
        # A. silica PSD
        res["sigma_psd"] = {}
        for sg in SIGMAS:
            m, _, _ = es.mmin(exts, ders, es.prof(sil(sg=sg)))
            res["sigma_psd"][str(sg)] = m
        res["rmed"] = {}
        for r in RMEDS:
            m, _, _ = es.mmin(exts, ders, es.prof(sil(rmed=r)))
            res["rmed"][str(r)] = m
        print("  sigma_psd scan:", {k: round(v, 3) for k, v in res["sigma_psd"].items()})
        print("  r_med scan:   ", {k: round(v, 3) for k, v in res["rmed"].items()})
        # B. Franta
        m, _, _ = es.mmin(exts, ders, es.prof(sil(ri=ri_fr)))
        res["franta"] = m
        print(f"  Franta optics: {m:.4f} Tg")
        # C. agglomeration
        res["agglomerate"] = {}
        t_aggs = {}
        for case, fn in FN.items():
            t_agg = sum(f * es.prof(sil(rmed=RMED_SIL_NM * n ** (1 / 3)))
                        for n, f in fn.items() if f > 0)
            t_aggs[case] = t_agg
            m, _, _ = es.mmin(exts, ders, t_agg)
            res["agglomerate"][case] = m
        print("  agglomerate:  ", {k: round(v, 4) for k, v in res["agglomerate"].items()})
        # extractability of the agglomeration state: t(a) = (1-a) t_mono +
        # a t_agg(8 Tg/yr); at a = 0 the a-column is M (t_agg - t_mono), so
        # sigma_a = C / M[Tg] with C = 1/||d_perp||, d whitened and projected
        # out of the span of the seven other columns (background + t_mono)
        k = iz
        s_tot = sum(e[k] for e in exts)
        sig = np.r_[np.maximum(VIS_PHI * s_tot[:N_VIS], ALPHA_FLOOR), es.floors]
        N = np.column_stack([exts[0][k], exts[1][k], ders[0][0][k],
                             ders[0][1][k], ders[1][0][k], ders[1][1][k],
                             t_nom[k]]) / sig[:, None]
        d = (t_aggs['8 Tg/yr'][k] - t_nom[k]) / sig
        d_perp = d - N @ np.linalg.lstsq(N, d, rcond=None)[0]
        res["agglomerate_sigma_a_times_M_tg"] = float(1.0 / np.linalg.norm(d_perp))
        print(f"  agglomeration-state extractability: sigma_a = "
              f"{res['agglomerate_sigma_a_times_M_tg']:.3f} / M[Tg]")
        # D. vis/NIR noise scaling
        res["g_vis"] = {str(g): es.mmin(exts, ders, t_nom, g_vis=g)[0]
                        for g in G_VIS}
        print("  vis/NIR noise x g:", {k: round(v, 3) for k, v in res["g_vis"].items()})
        # E. altitude
        res["altitude"] = {}
        for z in Z_KM:
            k = int(round(z * 1000 / (ALT[1] - ALT[0])))
            m, _, _ = es.mmin(exts, ders, t_nom, k=k, mir_scale=oe_floor_scale(z))
            res["altitude"][str(z)] = m
        res["altitude_floor_scale"] = {str(z): oe_floor_scale(z) for z in Z_KM}
        print("  altitude [km]:", {k: round(v, 3) for k, v in res["altitude"].items()})
        arch["sets"][es.name] = res

    # F. from the archives
    ph = thr["results"]["bracket: post-HT calib (LM72T213)"]
    os_ = thr["results"]["bracket: optics swap (B70T215)"]
    prob = json.load(open(_ROOT / "outputs" / "calibrated_background_problem.json"))
    arch["archived"] = {
        "window": dict(
            post_ht=ph["window 7.8-9.3 @0.1: measured floors"]["full 7-param"]["triplet + window"]["mmin_tg"],
            optics_swap=os_["window 7.8-9.3 @0.1: measured floors"]["full 7-param"]["triplet + window"]["mmin_tg"],
            mir_offset=cq["window 7.8-9.3 @0.1: measured floors"]["7-param + MIR offs"]["triplet + window"]["mmin_tg"],
            uniform_floors=cq["window 7.8-9.3 @0.1: uniform 1.5e-8 floors"]["full 7-param"]["triplet + window"]["mmin_tg"],
            floor_scan=prob["floor_scan_triplet"],
            loading={k: v["mmin_7p_triplet"] for k, v in prob["loading"].items()}),
        "band": dict(
            floor_scan=prob["floor_scan_band01_triplet"],
            loading={k: v["mmin_7p_band01_triplet"] for k, v in prob["loading"].items()},
            post_ht=ph["band 8-13 @0.1: measured floors"]["full 7-param"]["triplet + band01"]["mmin_tg"],
            optics_swap=os_["band 8-13 @0.1: measured floors"]["full 7-param"]["triplet + band01"]["mmin_tg"],
            mir_offset=cq["band 8-13 @0.1: measured floors"]["7-param + MIR offs"]["triplet + band01"]["mmin_tg"],
            uniform_floors=cq["band 8-13 @0.1: uniform 1.5e-8 floors"]["full 7-param"]["triplet + band01"]["mmin_tg"]),
    }
    arch["enlarged_family"] = dict(third_mode=THIRD_MODE, second_member=SECOND_MEMBER)
    out = OUT_JSON
    if dry_run:
        import os
        out = Path(os.environ.get("DRY_RUN_DIR", "/tmp")) / OUT_JSON.name
    out.parent.mkdir(exist_ok=True, parents=True)
    with open(out, "w") as f:
        json.dump(arch, f, indent=1)
    print(f"\narchived -> {out}")
    return arch



def _paper_style(plt):
    """Match the manuscript: Times-like serif, 11-pt body (Copernicus
    manuscript layout), figures rendered at their printed width."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
        "axes.linewidth": 0.8, "lines.linewidth": 1.6})

def plot(arch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _paper_style(plt)
    W, B = "window 7.8-9.3 @0.1", "band 8-13 @0.1"
    nW, nB = arch["sets"][W]["n_elements"], arch["sets"][B]["n_elements"]
    lab = {W: rf"window $7.8$–$9.3\,\mu$m ({nW} elements)",
           B: rf"band $8$–$13\,\mu$m ({nB} elements)"}
    # CVD-validated categorical slots 1-3 (dataviz reference palette, light)
    # round 34 (Doron): window black, band red; printed width 15 cm
    col = {W: "black", B: "#c00000"}
    fig, ax = plt.subplots(2, 2, figsize=(15 / 2.54, 11.5 / 2.54))
    for name, res in arch["sets"].items():
        kw = dict(color=col[name], marker="o", ms=4, lw=1.6, label=lab[name])
        x = [float(k) for k in res["sigma_psd"]]
        ax[0, 0].plot(x, list(res["sigma_psd"].values()), **kw)
        x = [float(k) for k in res["rmed"]]
        ax[0, 1].plot(x, list(res["rmed"].values()), **kw)
        x = [float(k) for k in res["altitude"]]
        ax[1, 0].plot(x, list(res["altitude"].values()), **kw)
        x = [float(k) for k in res["g_vis"]]
        ax[1, 1].plot(x, list(res["g_vis"].values()), **kw)
    ax[0, 0].set_xlabel(r"silica PSD width $\sigma_\mathrm{psd}$")
    ax[0, 0].set_title("(a) injected PSD width")
    ax[0, 1].set_xlabel(r"silica median radius $r_\mathrm{med}$ [nm]")
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_title("(b) injected median radius")
    ax[1, 0].set_xlabel("tangent altitude $z$ [km]")
    ax[1, 0].set_title("(c) measurement altitude")
    ax[1, 1].set_xlabel("visible/NIR noise scale $g$")
    ax[1, 1].set_xscale("log", base=2)
    ax[1, 1].set_title("(d) visible/NIR channel noise")
    from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter
    dec = FuncFormatter(lambda v, _: f"{v:g}")
    for a in ax.flat:
        a.set_ylabel(r"$M_{\min}^{(3\sigma)}$ [Tg]")
        a.set_yscale("log")
        a.yaxis.set_major_locator(FixedLocator([0.05, 0.07, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.6]))
        a.yaxis.set_major_formatter(dec)
        a.yaxis.set_minor_formatter(NullFormatter())
        a.grid(True, which="major", alpha=0.25)
    ax[0, 1].xaxis.set_major_locator(FixedLocator([150, 200, 268, 400, 600]))
    ax[0, 1].xaxis.set_major_formatter(dec)
    ax[0, 1].xaxis.set_minor_formatter(NullFormatter())
    ax[1, 1].xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
    ax[1, 1].xaxis.set_major_formatter(dec)
    ax[1, 1].xaxis.set_minor_formatter(NullFormatter())
    for a in (ax[0, 0], ax[0, 1]):
        a.axvline({ax[0, 0]: 1.31, ax[0, 1]: 268.0}[a], color="0.5", ls=":",
                  lw=1)
    # legend in the empty middle of panel (d)
    h, l = ax[0, 0].get_legend_handles_labels()
    ax[1, 1].legend(h, l, loc="center", bbox_to_anchor=(0.55, 0.5), frameon=False, handlelength=1.6)
    ax[0, 0].xaxis.set_major_locator(FixedLocator([1.0, 1.2, 1.4, 1.6, 1.8]))
    ax[1, 0].xaxis.set_major_locator(FixedLocator([16, 18, 20, 22, 24]))
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=300, bbox_inches="tight")
    FIG_PAPER.parent.mkdir(exist_ok=True)
    shutil.copy(FIG, FIG_PAPER)
    print(f"figure -> {FIG} (copied to {FIG_PAPER})")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--plot" in args:
        plot(json.load(open(OUT_JSON)))
    elif "--dry-run" in args or "--family-only" in args:
        compute(dry_run=True, family_only="--family-only" in args)
    else:
        plot(compute())
