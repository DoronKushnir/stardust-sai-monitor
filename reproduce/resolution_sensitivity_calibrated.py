"""
resolution_sensitivity_calibrated.py -- the detection threshold as a
function of the spectral ELEMENT WIDTH dlam, for the 7.8-9.3 um window and
the 7.8-13 um band (47 elements since round 57), heritage triplet + mid-infrared elements, full
7-parameter calibrated background, z = 20 km (Paper 1 round 33, Doron: is
the 0.1 um / 0.25 um choice arbitrary?  can one resolution serve both?).

Machinery: calibrated_background_thresholds.py (imported).  To make the
element width a free parameter the Mie extinction of every background
column (two components, their four PSD-derivative columns) and of the
silica layer is evaluated ONCE on a fine 0.005-um grid over 7.75-13.30 um
at z = 20 km and then box-averaged into contiguous elements of width dlam
(the design's 41-point sub-sampling is the same operation; the 0.1-um
window and 0.25-um band baselines are reproduced to ~1 %).

Floor model (the one assumption everything hinges on; round 36): the
per-element systematic floor is the DIRECTLY measured 0.1-um-element atlas
SD (Appendix app:acefloor, 19-22 km, residual_floor_w0p1), interpolated to
the element center, times the measured width curve of the 8.80-um record (round 43; 8.74 before)
normalized at 0.1 um, w(dlam)/w(0.1) (w = 1 beyond 0.25 um), converted to
extinction by 2.04e-3 OD <-> 1.5e-8 m^-1.  Elements whose center falls in
the saturated O3 core (9.3-10.0 um) are dropped.

Nuisance conventions per (range, dlam):
  * independent per-element floors (the paper's convention);
  * + a fully correlated MIR offset column (the worst-case common mode);
  * exponentially correlated floors, C_ij = s_i s_j exp(-|c_i-c_j|/ell),
    for ell = 0.1, 0.25, 0.5 um -- the honest version of "finer elements
    are not independent"; the empirical correlation length is measured by
    reproduce/ace_nulltest_resolution.py.

Run from the repo root:  python reproduce/resolution_sensitivity_calibrated.py
Output: outputs/resolution_sensitivity_calibrated.json,
        figures/resolution_sensitivity_calibrated.png
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

DLAMS = (0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
RANGES = {"window": (7.8, 9.3), "band": (7.8, 13.25)}   # round 57 (RC2 S3, Doron): band from 7.8 um, 47 elements
GAP = (9.3, 10.0)
VIS_NM = np.array([448., 756., 1544.])
VIS_PHI = np.array([0.040, 0.032, 0.088])
ELLS = (0.1, 0.25, 0.5)
# round 53 (RC1 M1/M3, Doron): exact edge-grid gain and the design-budget anchor
from saimon.onion_peel import M1_PER_OD, OD_PER_M1, BUDGET_OD_880, G_ONION   # noqa: E402
OUT = _ROOT / "outputs" / "resolution_sensitivity_calibrated.json"
FIG = _ROOT / "figures" / "resolution_sensitivity_calibrated.png"
FIG_PAPER = FIG   # repo: figures/ is the paper figure directory

_atlas = json.load(open(_ROOT / "data/ace_floor/"
                        "w0p25/atlas.json"))
_rec = json.load(open(_ROOT / "data/ace_floor/"
                      "w0p25/19_22_8.8um.json"))   # round 43: the 8.80-um record (was 8.74)
_ac, _asd = zip(*sorted((r["center_um"], r["sd"]) for r in _atlas["19_22"]
                        if r.get("sd") is not None and 7.8 <= r["center_um"] <= 13.25))
_ac, _asd = np.array(_ac), np.array(_asd)
# Round 36 (2026-09-25, Doron): the floors are anchored to the DIRECT
# 0.1-um-element re-analysis of the record (residual_floor_w0p1), interpolated
# to the element centers, times the measured width curve normalized at 0.1 um
# (w(dlam)/w(0.1)); the 0.25-um atlas above is kept only for reference.
# Round 53: the direct 0.1-um atlas re-run on the x.x0 grid centred on 8.80 um
# (residual_floor_w0p1_c880), scaled so the 8.80-um element carries the design's
# R~100 budget (FLOOR_SCALE), OD -> m^-1 with the exact edge-grid gain.
_atlas01 = json.load(open(_ROOT / "data/ace_floor/"
                          "w0p1_c880/atlas.json"))
_ac01, _asd01 = zip(*sorted((r["center_um"], r["sd"]) for r in _atlas01["19_22"]
                            if r.get("sd") is not None and 7.3 <= r["center_um"] <= 13.3))
_ac01, _asd01 = np.array(_ac01), np.array(_asd01)
_sd880 = {round(r["center_um"], 3): r["sd"] for r in _atlas01["19_22"] if r.get("sd") is not None}[8.8]
FLOOR_SCALE = BUDGET_OD_880 / _sd880
_wk = sorted(float(k) for k in _rec["averaging_fixed_full_element_holdout"])
_wf = np.array([_rec["averaging_fixed_full_element_holdout"][
    (f"{k:g}")]["exact"]["sd"] for k in _wk]) / float(_rec["sd"])


def width_factor(dlam):
    return float(np.interp(dlam, _wk, _wf, right=1.0))


PHASES = (0.0, 0.2, 0.4, 0.6, 0.8)   # element-grid offsets in units of dlam


def element_centers(lo, hi, dlam, phase=0.0):
    """Contiguous elements of width dlam starting at lo + phase*dlam; the
    last element must fit inside [lo, hi].  Elements centered in the
    saturated O3 core are dropped."""
    # elements whose CENTER lies within [lo, hi] (the last one may extend
    # dlam/2 beyond hi), the same rule as PART 4 of the thresholds script
    # round 53 (Doron): grid centred on 8.80 um -- phase 0 puts an element CENTRE at lo
    # (window 7.80...9.20, band 8.00...13.20), the design grid of the thresholds script
    c = np.round(np.arange(lo + dlam * phase, hi + 1e-9, dlam), 6)   # round 53: no 9.2999... slipping past the core cut
    return c[(c < GAP[0] - 1e-9) | (c > GAP[1] + 1e-9)]


def floor_model_m1(centers, dlam):
    """per-element systematic floor [m^-1] at dz = 0.5 km: direct 0.1-um
    atlas at the centers x w(dlam)/w(0.1) (round 36)."""
    sd = np.interp(centers, _ac01, _asd01) * width_factor(dlam) / width_factor(0.10)
    return sd * FLOOR_SCALE * M1_PER_OD   # round 53: design-budget anchor, exact gain


def sigma_marg(cols, t, cov_or_sig):
    """marginalized sigma(A_sil) by stable projection with a general noise
    covariance (whitened by Cholesky)."""
    if np.ndim(cov_or_sig) == 1:
        Lw = np.diag(1.0 / cov_or_sig)
    else:
        Lw = np.linalg.inv(np.linalg.cholesky(cov_or_sig))
    N = Lw @ np.column_stack(cols)
    tw = Lw @ t
    tpar = N @ np.linalg.lstsq(N, tw, rcond=None)[0]
    tperp = tw - tpar
    return 1.0 / np.linalg.norm(tperp), 1 - float(tpar @ tpar) / float(tw @ tw)


def compute():
    from calibrated_background_thresholds import (
        ALT, iz, RMED_SIL_NM, SIGMA_SIL, ALPHA_FLOOR, CAL_QUIET,
        TwoComponentBackground, glossac_2025N_quiet_profile, member_ri)
    from saimon.materials import SilicaRefractiveIndex
    from saimon.sai import create_silica_sai_layer

    grid = np.arange(7.75, 13.3001, 0.005)
    wl_all = np.r_[VIS_NM * 1e-9, grid * 1e-6]
    cache = _ROOT / "outputs" / "resolution_fine_grid_cache.npz"
    if cache.exists() and np.array_equal(np.load(cache)["grid"], grid):
        d = np.load(cache)
        exts, ders, t_fine = list(d["exts"]), list(d["ders"]), d["t_fine"]
        print("fine-grid Mie rows loaded from cache", flush=True)
    else:
        def row(layer):
            return layer.extinction_profile_m1(wl_all)[iz, :]

        ri, wt, T = member_ri("LM65T223")
        bg = TwoComponentBackground(glossac_2025N_quiet_profile(), CAL_QUIET, ri, wt, T)
        print("fine-grid Mie: background columns ...", flush=True)
        exts = [row(c) for c in bg.columns()]
        ders = []
        for i, (r0, s0, f) in enumerate(CAL_QUIET):
            dr, ds = bg.fd * r0, bg.fd * s0
            er = (row(bg.column(i, dr=+dr)) - row(bg.column(i, dr=-dr))) / (2 * dr)
            es = (row(bg.column(i, ds=+ds)) - row(bg.column(i, ds=-ds))) / (2 * ds)
            ders += [er, es]
        sil = create_silica_sai_layer(ALT, 1.0, refractive_index=SilicaRefractiveIndex(
            "data/optics/SiO2_KitamuraPopova.yml"), rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL)
        t_fine = row(sil)
        np.savez(cache, grid=grid, exts=np.array(exts), ders=np.array(ders), t_fine=t_fine)
    print("scanning element widths", flush=True)
    s_tot_vis = (exts[0] + exts[1])[:3]
    sig_vis = np.maximum(VIS_PHI * s_tot_vis, ALPHA_FLOOR)

    def avg(v, centers, dlam):
        out = np.empty(len(centers))
        for j, c in enumerate(centers):
            m = (grid >= c - dlam / 2 - 1e-9) & (grid < c + dlam / 2 - 1e-9)
            out[j] = v[3:][m].mean()
        return np.r_[v[:3], out]

    arch = {"floor_anchor": dict(budget_od_880=float(BUDGET_OD_880), measured_sd_880_od=float(_sd880),
                                 floor_scale=float(FLOOR_SCALE), m1_per_od=float(M1_PER_OD), g_exact=float(G_ONION)),
            "dlams": list(DLAMS), "ells": list(ELLS), "width_factor":
            {f"{d}": width_factor(d) for d in DLAMS}, "results": {}}
    for rname, (lo, hi) in RANGES.items():
        arch["results"][rname] = {}
        for dlam in DLAMS:
            per_phase = []
            for ph in PHASES:
                centers = element_centers(lo, hi, dlam, ph)
                if 3 + len(centers) < 7 + 1:
                    # fewer data than the 7 parameters + silica: rank-deficient
                    per_phase.append(dict(phase=ph, n_elements=int(len(centers)),
                                          mmin=np.inf, one_minus_R2=np.nan,
                                          mmin_offset=np.inf,
                                          mmin_corr={f"{e}": np.inf for e in ELLS}))
                    continue
                fl = floor_model_m1(centers, dlam)
                cols = [avg(v, centers, dlam) for v in exts + ders]
                t = avg(t_fine, centers, dlam)
                sig = np.r_[sig_vis, fl]
                sa, r2 = sigma_marg(cols, t, sig)
                offs = np.r_[np.zeros(3), np.ones(len(centers))]
                sa_off, _ = sigma_marg(cols + [offs], t, sig)
                rec = dict(phase=ph, n_elements=int(len(centers)), mmin=3 * sa,
                           one_minus_R2=r2, mmin_offset=3 * sa_off, mmin_corr={})
                for ell in ELLS:
                    C = np.diag(sig ** 2).copy()
                    D = np.abs(centers[:, None] - centers[None, :])
                    C[3:, 3:] = np.outer(fl, fl) * np.exp(-D / ell)
                    sa_c, _ = sigma_marg(cols, t, C)
                    rec["mmin_corr"][f"{ell}"] = 3 * sa_c
                per_phase.append(rec)
            med = lambda key: float(np.median([r[key] for r in per_phase]))
            rng = lambda key: [float(min(r[key] for r in per_phase)), float(max(r[key] for r in per_phase))]
            out = dict(n_elements=per_phase[0]["n_elements"],
                       mmin=med("mmin"), mmin_range=rng("mmin"),
                       one_minus_R2=med("one_minus_R2"),
                       mmin_offset=med("mmin_offset"), mmin_offset_range=rng("mmin_offset"),
                       mmin_corr={f"{e}": float(np.median([r["mmin_corr"][f"{e}"] for r in per_phase])) for e in ELLS},
                       mmin_phase0=per_phase[0]["mmin"], per_phase=per_phase)
            arch["results"][rname][f"{dlam}"] = out
            print(f"{rname:7s} dlam={dlam:4.2f}: {out['n_elements']:3d} el, w={width_factor(dlam):.3f}, "
                  f"M_min={out['mmin']:.3f} [{out['mmin_range'][0]:.3f}-{out['mmin_range'][1]:.3f}] Tg "
                  f"(phase0 {out['mmin_phase0']:.3f}); +offset {out['mmin_offset']:.3f}; "
                  f"corr ell=0.1/0.25/0.5: " + "/".join(f"{out['mmin_corr'][str(e)]:.3f}" for e in ELLS),
                  flush=True)
    with open(OUT, "w") as f:
        json.dump(arch, f, indent=1)
    print(f"archived -> {OUT}")
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
    from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter
    _paper_style(plt)
    emp = None
    p = _ROOT / "outputs" / "ace_v52" / "nulltest_resolution.json"
    if p.exists():
        emp = json.load(open(p))
    # round 34 (Doron): window black, band red; printed width 15 cm; both
    # panels on identical axes; one shared legend below the panels
    from matplotlib.lines import Line2D
    col = {"window": "black", "band": "#c00000"}
    fig, ax = plt.subplots(1, 2, figsize=(15 / 2.54, 8.8 / 2.54))
    x = [float(d) for d in arch["dlams"]]
    fin = lambda v: v if np.isfinite(v) and v < 5 else np.nan
    for r in ("window", "band"):
        R = arch["results"][r]
        y = np.array([fin(R[str(d)]["mmin"]) for d in arch["dlams"]])
        lo_ = np.array([fin(R[str(d)]["mmin_range"][0]) for d in arch["dlams"]])
        hi_ = np.array([fin(R[str(d)]["mmin_range"][1]) for d in arch["dlams"]])
        ax[0].errorbar(x, y, yerr=[y - lo_, hi_ - y], fmt="o-", color=col[r],
                       ms=4, capsize=2)
        ax[0].plot(x, [fin(R[str(d)]["mmin_corr"]["0.25"]) for d in arch["dlams"]],
                   "s--", color=col[r], ms=3.5, lw=1.2)
        ax[0].plot(x, [fin(R[str(d)]["mmin_offset"]) for d in arch["dlams"]], ":",
                   color=col[r], lw=1.4)
        if emp:
            E = {d: v for d, v in emp["ranges"][r].items() if v.get("usable")}
            xe = [float(d) for d in E]
            ye = np.array([E[d]["mmin_emp_shell"] for d in E])
            lo_ = np.array([E[d]["mmin_emp_shell_range"][0] for d in E])
            hi_ = np.array([E[d]["mmin_emp_shell_range"][1] for d in E])
            ax[1].errorbar(xe, ye, yerr=[ye - lo_, hi_ - ye], fmt="o-", color=col[r],
                           ms=4, capsize=2)
            ax[1].plot(xe, [E[d]["mmin_formal_shell"] for d in E], "s--", color=col[r],
                       ms=3.5, lw=1.2)
    ax[0].set_title("(a) linearized, triplet + elements")
    ax[1].set_title("(b) ACE null test, MIR alone")
    dec = FuncFormatter(lambda v, _: f"{v:g}")
    for a in ax:
        a.set_xlabel(r"element width $\Delta\lambda$ [$\mu$m]")
        a.set_ylabel(r"$M_{\min}^{(3\sigma)}$ [Tg]")
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlim(0.042, 0.6); a.set_ylim(0.06, 3.0)
        a.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.3, 0.5]))
        a.xaxis.set_major_formatter(dec); a.xaxis.set_minor_formatter(NullFormatter())
        a.yaxis.set_major_locator(FixedLocator([0.07, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]))
        a.yaxis.set_major_formatter(dec); a.yaxis.set_minor_formatter(NullFormatter())
        a.grid(True, which="major", alpha=0.25)
    handles = [Line2D([], [], color="black", lw=1.6, label=r"window $7.8$–$9.3\,\mu$m"),
               Line2D([], [], color="#c00000", lw=1.6, label=r"band $7.8$–$13\,\mu$m"),   # round 59 (RC3 S2)
               Line2D([], [], color="0.35", marker="o", ms=4, lw=1.6,
                      label="(a) independent floors; (b) empirical scatter"),
               Line2D([], [], color="0.35", marker="s", ms=3.5, ls="--", lw=1.2,
                      label=r"(a) floors correlated, $\ell=0.25\,\mu$m; (b) formal error"),
               Line2D([], [], color="0.35", ls=":", lw=1.4, label="(a) + fully correlated offset")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handlelength=2.2, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.2, 1, 1), pad=0.4)
    fig.savefig(FIG, dpi=300, bbox_inches="tight")
    if "--paper" in sys.argv[1:]:
        FIG_PAPER.parent.mkdir(exist_ok=True)
        (shutil.copy(FIG, FIG_PAPER) if FIG_PAPER != FIG else None)
    print(f"figure -> {FIG}")


if __name__ == "__main__":
    if "--plot" in sys.argv[1:]:
        plot(json.load(open(OUT)))
    else:
        plot(compute())
