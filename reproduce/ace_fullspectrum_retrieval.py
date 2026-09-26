"""
ace_fullspectrum_retrieval.py -- silica null-test retrieval on real ACE
occultations using the FULL residual spectral shape (760-1250 cm^-1,
~250 bins of 2 cm^-1) instead of the 4 broadband channels of
scripts/ace_bplus_nulltest.py.

Purpose: the 4-channel null test finds an empirical sigma(M) several times
the formal noise -- dominated by day-to-day background-colour variability
that 4 channels cannot co-retrieve.  This script measures how much of that
geophysical variance collapses when the fit sees the full spectral shape,
i.e. it measures directly what added spectral information buys (what a
band-resolving instrument achieves).

Method (all slant-space, as before):
  * data: per occultation, the residual spectrum at the tangent height
    nearest 20.5 km (within 1.5 km), self-calibrated BIN-WISE by the median
    >=30 km residual spectrum of the same occultation;
  * model: slant basis spectra on the same wavenumber grid for the
    CALIBRATED two-component sulfate background of Paper 1 Sect. 2.2
    (round 27, Doron; imported bit-identically from
    scripts/calibrated_background_thresholds.py: GloSSAC 20-25N quiet
    profile, Lund-Myhre 65 wt%/223 K member optics, 60 nm s1.6 fine +
    1.5 um s1.8 coarse modes) -- the fine and coarse component spectra,
    their (r_med, sigma) Jacobians at fixed 525-nm share, and the per-Tg
    silica layer -- evaluated at the occultation's tangent height and
    offset-corrected with the same operator (basis(h) - basis(h_ref));
  * fit: robust weighted LSQ (columns unit-normalized for conditioning),
    sigma_bin = 8e-3 (per-2 cm^-1 single-spectrum noise implied by the
    measured 2e-3 per 0.25-um element), 2 passes of 4-sigma bin clipping.
    Two variants, silica column always LAST:
      full : [A_f, A_c, dr_f, ds_f, dr_c, ds_c, M_sil, const]  (8 params:
             the design's full 7-parameter background+silica fit + the
             self-calibration constant)
      std  : [A_f, A_c, dr_f, ds_f, M_sil, const]              (coarse shape
             held; check variant)
  * bins 760-1250 cm^-1; saturated (-1) and invalid bins dropped.
  * Round 28 (Doron): two WINDOW variants restricted to the design's
    7.8-9.3 um retrieval window (1075-1282 cm^-1), MIR alone (no visible
    anchor), full 8-parameter model:
      win2  : native 2-cm^-1 bins, sigma_bin = 8e-3 (resolution kept);
      win01 : the 15 design elements of 0.1 um (7.85-9.25 um), each the
              -ln of the mean residual transmittance over its bins, with
              the per-element measured floors of Appendix
              app:tracegas_measured (window_floors_m1 of the threshold
              archive converted back to OD by 2.04e-3/1.5e-8) as weights
              -- the closest real-data emulation of the R~100 design
              window, minus its visible/NIR channels.

Outputs
  outputs/ace_v52/basis_spectra_cal.npz  cached slant bases (shared w/ Ruang)
  outputs/ace_v52/fullspectrum.csv       per-occultation results
  outputs/ace_v52/fullspectrum_stats.json
  figures/ace_fullspectrum_nulltest.png
  printed: empirical robust sigma(M) vs the 4-channel value, M_min(3sigma)

Run AFTER scripts/ace_bplus_nulltest.py (reads its stats for comparison).
Run from the repo root:  python scripts/ace_fullspectrum_retrieval.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrated_background_thresholds import (  # noqa: E402
    ALT, CAL_QUIET, member_ri, glossac_2025N_quiet_profile,
    TwoComponentBackground)
from saimon.geometry import tangent_to_slant_paths  # noqa: E402
from saimon.materials import SilicaRefractiveIndex  # noqa: E402
from saimon.sai import create_silica_sai_layer  # noqa: E402

DATA = _HERE / "data" / "ace" / "v52_subset"
OUT = _HERE / "outputs" / "ace_v52"
FIG = _HERE / "figures" / "ace_fullspectrum_nulltest.png"
# the canonical 143-occultation ensemble (Appendix app:acefloor record); the
# local mirror now holds 1274 occultations of the release, so the scan must
# be filtered (as in scripts/ace_mir_aerosol.py)
SIDE = _HERE / "data" / "ace" / "revision20260909"
BASIS_CACHE = OUT / "basis_spectra_cal.npz"

NU_LO, NU_HI = 760.0, 1250.0          # full-spectrum fit range
NU_LOAD_HI = 1290.0                    # load up to here (window reaches 1282)
WIN_UM = (7.8, 9.3)                    # Sect.-3.3 retrieval window
WIN_EL_UM = np.round(np.arange(7.85, 9.2501, 0.10), 2)   # 15 elements
WIN_EL_W = 0.10
OD_PER_M1 = 2.04e-3 / 1.5e-8           # measured element OD <-> extinction
FD = 0.05
RMED_NM, SIGMA_PSD = 268.0, 1.31
H_GRID_KM = np.arange(16.0, 35.0, 0.5)
SIG_BIN = 8.0e-3
DESIGN_MMIN = 0.12       # Paper 1 Sect. 3.2: window mode, full 7-param fit
BASIS_NAMES = ("f", "c", "jrf", "jsf", "jrc", "jsc", "sil")
VARIANTS = {
    "full": ("f", "c", "jrf", "jsf", "jrc", "jsc", "sil"),
    "std":  ("f", "c", "jrf", "jsf", "sil"),
}
SIL_SCALE_TG = 0.2       # silica shape drawn at this mass for scale
ALL_TAGS = ("full", "std", "win2", "win01")


# ---------------------------------------------------------------------------
def reference_grid():
    """Wavenumber grid of the residual files, [NU_LO, NU_LOAD_HI]."""
    d = sorted((DATA / "residual").iterdir())[0]
    f = sorted(d.iterdir())[0]
    nu = np.loadtxt(f)[:, 0]
    return nu[(nu >= NU_LO) & (nu <= NU_LOAD_HI)]


def build_bases(nu_grid):
    if BASIS_CACHE.exists():
        dat = np.load(BASIS_CACHE)
        if np.array_equal(dat["nu"], nu_grid) and all(
                k in dat for k in BASIS_NAMES):
            return {k: dat[k] for k in BASIS_NAMES}
    print("building slant basis spectra (Mie on "
          f"{len(nu_grid)} wavelengths, calibrated two-component "
          "background)...")
    ri, wt, T = member_ri("LM65T223")
    bg = TwoComponentBackground(glossac_2025N_quiet_profile(), CAL_QUIET,
                                ri, wt, T)
    silica_ri = SilicaRefractiveIndex("data/optics/SiO2_KitamuraPopova.yml")
    sil = create_silica_sai_layer(ALT, 1.0, refractive_index=silica_ri,
                                  rmed_nm=RMED_NM, sigma=SIGMA_PSD)
    wl_m = 1.0e-2 / nu_grid

    def prof(layer):
        return layer.extinction_profile_m1(wl_m)          # (nz, n_nu)

    def jac(i):
        r0, s0, _ = CAL_QUIET[i]
        dr, ds = FD * r0, FD * s0
        jr = (prof(bg.column(i, dr=+dr)) - prof(bg.column(i, dr=-dr))) / (2 * dr)
        js = (prof(bg.column(i, ds=+ds)) - prof(bg.column(i, ds=-ds))) / (2 * ds)
        return jr, js

    jrf, jsf = jac(0)
    jrc, jsc = jac(1)
    C = np.array(tangent_to_slant_paths(H_GRID_KM * 1e3, ALT))  # (n_h, nz)
    bases = dict(f=C @ prof(bg.column(0)), c=C @ prof(bg.column(1)),
                 jrf=C @ jrf, jsf=C @ jsf, jrc=C @ jrc, jsc=C @ jsc,
                 sil=C @ prof(sil))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(BASIS_CACHE, nu=nu_grid, h_km=H_GRID_KM, **bases)
    return bases


# ---------------------------------------------------------------------------
def load_spectrum(path, nu_grid):
    dat = np.loadtxt(path)
    nu, t = dat[:, 0], dat[:, 1]
    m = (nu >= NU_LO - 1) & (nu <= NU_LOAD_HI + 1)
    nu, t = nu[m], t[m]
    t_g = np.interp(nu_grid, nu, t, left=np.nan, right=np.nan)
    t_g[t_g <= 0.01] = np.nan          # saturated (-1) or unphysical
    return t_g


def occultation_data(occ, nu_grid):
    """(tangent height, corrected tau spectrum, h_ref) or None."""
    d = DATA / "residual" / occ
    files = []
    for f in d.iterdir():
        try:
            files.append((float(f.name.split(".", 1)[1].removesuffix(".gz")), f))
        except ValueError:
            continue
    if not files:
        return None
    # deterministic order: nearest to 20.5 km first, the LOWER of two
    # equidistant heights first.  Three ensemble occultations have an exact
    # tie (19.8/21.2 km, 20.3/20.7 km twice); directory order must not
    # decide it.
    files.sort(key=lambda af: (abs(af[0] - 20.5), af[0]))
    alts = np.array([a for a, _ in files])
    near = np.abs(alts - 20.5) <= 1.5
    hi = alts >= 30.0
    if not near.any() or hi.sum() < 2:
        return None
    k = 0
    t20 = load_spectrum(files[k][1], nu_grid)
    tref = np.nanmedian(np.array([load_spectrum(f, nu_grid)
                                  for a, f in files if a >= 30.0]), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = -np.log(t20) + np.log(tref)
    return alts[k], tau, float(np.median(alts[hi]))


def basis_columns(h, h_ref, bases, nu_grid, names):
    cols = []
    for name in names:
        b = bases[name]                            # (n_h, n_nu)
        v = np.array([np.interp(h, H_GRID_KM, b[:, j]) -
                      np.interp(h_ref, H_GRID_KM, b[:, j])
                      for j in range(b.shape[1])])
        cols.append(v)
    cols.append(np.ones_like(nu_grid))             # constant nuisance
    return np.column_stack(cols)


def _solve(J_full, y, ok, sig, isil, clip=True):
    """Weighted LSQ with unit-normalized columns; sig is scalar or per-row.
    Returns theta, cov, used mask."""
    sig = np.broadcast_to(np.asarray(sig, float), y.shape)
    scales = np.linalg.norm(J_full[ok] / sig[ok][:, None], axis=0)
    Jn_full = J_full / scales
    use = ok.copy()
    theta = cov = None
    for _ in range(3):
        Jn, yy, ss = Jn_full[use], y[use], sig[use]
        Jw = Jn / ss[:, None]
        F = Jw.T @ Jw
        covn = np.linalg.inv(F)
        thn = covn @ (Jw.T @ (yy / ss))
        theta = thn / scales
        cov = covn / np.outer(scales, scales)
        if not clip:
            break
        resid = y - J_full @ theta
        new = ok & (np.abs(resid) < 4 * sig)
        if new.sum() == use.sum():
            break
        use = new
    return theta, cov, use, F


def fit_occ(h, tau, h_ref, bases, nu_grid, win_sig_od):
    full_rng = (nu_grid >= NU_LO) & (nu_grid <= NU_HI)
    ok = np.isfinite(tau) & full_rng
    if ok.sum() < 100:
        return None
    out = {}
    for tag, names in VARIANTS.items():
        J_full = basis_columns(h, h_ref, bases, nu_grid, names)
        isil = len(names) - 1                      # silica column index
        theta, cov, use, F = _solve(J_full, tau, ok, SIG_BIN, isil)
        dof = use.sum() - J_full.shape[1]
        chi2 = float(np.sum((tau[use] - J_full[use] @ theta)**2)
                     / SIG_BIN**2 / dof)
        out[tag] = dict(theta=theta, M=float(theta[isil]),
                        sM=float(np.sqrt(cov[isil, isil])), chi2=chi2,
                        nbin=int(use.sum()), cond=float(np.linalg.cond(F)),
                        Af=float(theta[0]), Ac=float(theta[1]),
                        names=names)

    # ---- round 28: window variants (7.8-9.3 um, MIR alone, full model) ----
    names = VARIANTS["full"]
    isil = len(names) - 1
    J_full = basis_columns(h, h_ref, bases, nu_grid, names)
    wl_um = 1e4 / nu_grid
    in_win = (wl_um >= WIN_UM[0]) & (wl_um <= WIN_UM[1])
    okw = np.isfinite(tau) & in_win
    if okw.sum() >= 40:
        theta, cov, use, F = _solve(J_full, tau, okw, SIG_BIN, isil)
        dof = use.sum() - J_full.shape[1]
        out["win2"] = dict(theta=theta, M=float(theta[isil]),
                           sM=float(np.sqrt(cov[isil, isil])),
                           chi2=float(np.sum((tau[use] - J_full[use] @ theta)**2)
                                      / SIG_BIN**2 / dof),
                           nbin=int(use.sum()), cond=float(np.linalg.cond(F)),
                           Af=float(theta[0]), Ac=float(theta[1]), names=names)
    # 0.1-um elements: -ln<T> over the bins of each element; basis columns
    # averaged linearly (small-OD linearization)
    y_el, J_el, ok_el = [], [], []
    for c in WIN_EL_UM:
        mb = in_win & (wl_um >= c - WIN_EL_W / 2) & (wl_um < c + WIN_EL_W / 2)
        good = mb & np.isfinite(tau)
        if mb.sum() == 0 or good.sum() < 0.6 * mb.sum():
            y_el.append(np.nan); J_el.append(np.zeros(J_full.shape[1]))
            ok_el.append(False); continue
        y_el.append(float(-np.log(np.mean(np.exp(-tau[good])))))
        J_el.append(J_full[good].mean(axis=0)); ok_el.append(True)
    y_el, J_el, ok_el = np.array(y_el), np.array(J_el), np.array(ok_el)
    if ok_el.sum() >= J_el.shape[1] + 2:
        theta, cov, use, F = _solve(J_el, y_el, ok_el, win_sig_od, isil,
                                    clip=False)
        dof = use.sum() - J_el.shape[1]
        out["win01"] = dict(theta=theta, M=float(theta[isil]),
                            sM=float(np.sqrt(cov[isil, isil])),
                            chi2=float(np.sum(((y_el[use] - J_el[use] @ theta)
                                               / win_sig_od[use])**2) / dof),
                            nbin=int(use.sum()), cond=float(np.linalg.cond(F)),
                            Af=float(theta[0]), Ac=float(theta[1]), names=names)
    return out


# ---------------------------------------------------------------------------
def main():
    nu_grid = reference_grid()
    bases = build_bases(nu_grid)
    # per-element window floors (OD) from the threshold archive
    thr = json.load(open(_HERE / "outputs" /
                         "calibrated_background_thresholds.json"))
    assert np.allclose(thr["window_elements_um"], WIN_EL_UM)
    win_sig_od = np.array(thr["window_floors_m1"]) * OD_PER_M1
    print(f"window element floors: {win_sig_od.min():.2e}-"
          f"{win_sig_od.max():.2e} OD")

    # 4-channel comparison value (scripts/ace_bplus_nulltest.py)
    try:
        st4 = json.load(open(OUT / "nulltest_stats.json"))
        sig4ch = float(st4["4p"]["sigma_emp"])
    except (OSError, KeyError):
        sig4ch = np.nan
        print("WARNING: nulltest_stats.json not found -- run "
              "ace_bplus_nulltest.py first")

    # geolocation (year) per occultation
    geo = {}
    with open(DATA / "occultationlist.csv") as f:
        for row in csv.DictReader(f):
            dt = row["occultation_datetime"]
            geo[row["occultation_name"]] = (int(dt[:4])
                                            + (int(dt[5:7]) - 1) / 12.0
                                            + int(dt[8:10]) / 365.25)

    keep = set(str(o) for o in np.load(SIDE / "matrix_19_22.npz",
                                        allow_pickle=True)["occs"])
    rows = []
    example = None
    for d in sorted((DATA / "residual").iterdir()):
        occ = d.name
        if occ not in keep:
            continue
        got = occultation_data(occ, nu_grid)
        if got is None:
            continue
        h, tau, h_ref = got
        r = fit_occ(h, tau, h_ref, bases, nu_grid, win_sig_od)
        if r is None:
            continue
        row = dict(occ=occ, year=geo.get(occ, np.nan), h=h)
        for tag in ALL_TAGS:
            for k in ("M", "sM", "chi2", "nbin", "cond", "Af", "Ac"):
                row[f"{k}_{tag}"] = r[tag][k] if tag in r else np.nan
        rows.append(row)
        if example is None and occ == "sr101445":
            example = (tau, h, h_ref, r)

    keys = list(rows[0])
    with open(OUT / "fullspectrum.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    stats = {"n": len(rows), "sigma_4channel": sig4ch}
    for tag in ALL_TAGS:
        M = np.array([r[f"M_{tag}"] for r in rows])
        sM = np.array([r[f"sM_{tag}"] for r in rows])
        fin = np.isfinite(M) & np.isfinite(sM)
        M, sM = M[fin], sM[fin]
        med = float(np.median(M))
        sig_emp = float(1.4826 * np.median(np.abs(M - med)))
        z = (M - med) / sM
        sig_z = float(1.4826 * np.median(np.abs(z - np.median(z))))
        stats[tag] = dict(median_M=med, sigma_emp=sig_emp,
                          sigma_formal=float(np.median(sM)),
                          ratio=sig_emp / float(np.median(sM)),
                          mmin_3sigma=3 * sig_emp, sigma_z=sig_z,
                          n_fit=int(fin.sum()),
                          median_chi2=float(np.nanmedian(
                              [r[f"chi2_{tag}"] for r in rows])),
                          median_nbin=float(np.nanmedian(
                              [r[f"nbin_{tag}"] for r in rows])),
                          median_cond=float(np.nanmedian(
                              [r[f"cond_{tag}"] for r in rows])),
                          median_Af=float(np.nanmedian(
                              [r[f"Af_{tag}"] for r in rows])),
                          median_Ac=float(np.nanmedian(
                              [r[f"Ac_{tag}"] for r in rows])))
        s = stats[tag]
        print(f"\n[{tag}] fitted {int(fin.sum())} occultations, median bins used "
              f"= {s['median_nbin']:.0f}, median chi2/dof = "
              f"{s['median_chi2']:.2f}, median cond(F) = "
              f"{s['median_cond']:.1e}")
        print(f"[{tag}] median M = {med:+.4f} Tg, empirical robust sigma = "
              f"{sig_emp:.4f} Tg, median formal sigma = "
              f"{s['sigma_formal']:.4f} Tg (ratio {s['ratio']:.2f}; robust "
              f"sigma of normalized nulls {sig_z:.2f})")
        print(f"[{tag}] empirical M_min(3sigma) = {3*sig_emp:.3f} Tg "
              f"(4-channel null test: {3*sig4ch:.3f} Tg; designed window "
              f"mode, full fit: {DESIGN_MMIN:.2f} Tg); median A_f = "
              f"{s['median_Af']:.2f}, A_c = {s['median_Ac']:.2f}")
        print(f"[{tag}] variance collapse vs 4-channel: sigma ratio = "
              f"{sig_emp/sig4ch:.2f}")
    with open(OUT / "fullspectrum_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(f"archived -> {OUT / 'fullspectrum_stats.json'}")

    # ---- figure (the 'full' variant) ----------------------------------------
    tag = "full"
    M = np.array([r[f"M_{tag}"] for r in rows])
    yr = np.array([r["year"] for r in rows])
    sM = np.array([r[f"sM_{tag}"] for r in rows])
    sig4ch = sig4ch if np.isfinite(sig4ch) else np.nan
    med, sig_emp = stats[tag]["median_M"], stats[tag]["sigma_emp"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    ax = axes[0, 0]
    if example is not None:
        tau, h, h_ref, r = example
        J = basis_columns(h, h_ref, bases, nu_grid, r[tag]["names"])
        model = J @ r[tag]["theta"]
        isil = len(r[tag]["names"]) - 1
        ax.plot(nu_grid, tau, lw=0.7, color="0.4", label="data (sr101445)")
        ax.plot(nu_grid, model, lw=1.4, color="tab:red",
                label=f"{J.shape[1]}-param fit (full background)")
        ax.plot(nu_grid, J[:, isil] * SIL_SCALE_TG, lw=1.0, color="magenta",
                ls="--",
                label=rf"silica shape $\times$ {SIL_SCALE_TG} Tg (for scale)")
        ax.set_xlabel("wavenumber [cm$^{-1}$]")
        ax.set_ylabel(r"$\tau_{\rm slant}$ (self-calibrated)")
        ax.set_title("(a) Example full-spectrum fit", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(M, bins=40, range=(-0.5, 0.5), color="tab:red", alpha=0.6,
            label=f"full-spectrum: σ={sig_emp:.3f} Tg")
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(med, color="tab:red", lw=1.2)
    if np.isfinite(sig4ch):
        ax.axvspan(med - sig4ch, med + sig4ch, color="tab:blue", alpha=0.12,
                   label=f"4-channel σ={sig4ch:.3f} Tg")
    ax.set_xlabel("retrieved silica mass $M$ [Tg]")
    ax.set_ylabel("occultations")
    ax.set_title("(b) Null-test mass distribution", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.errorbar(yr, M, yerr=sM, fmt="o", ms=4, lw=0.7, alpha=0.7,
                color="tab:red")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(med, color="tab:red", lw=0.8, ls=":")
    ax.set_xlabel("year")
    ax.set_ylabel("retrieved $M$ [Tg]")
    ax.set_title("(c) Null consistency vs time (full spectrum)", fontsize=10)
    ax.grid(alpha=0.3)

    # (d) normalized null: M/sigma_M should be ~N(0,1) if the retrieval is
    # noise-limited and unbiased.  (The background parameters are mutually
    # degenerate -- only their combination, and M, are meaningful.)
    ax = axes[1, 1]
    z = (M - med) / sM
    ax.hist(z, bins=25, range=(-4, 4), density=True, color="tab:gray",
            edgecolor="k", lw=0.3, label=f"(M - med)/σ_M, N={len(z)}")
    xg = np.linspace(-4, 4, 200)
    ax.plot(xg, np.exp(-xg**2 / 2) / np.sqrt(2 * np.pi), color="tab:red",
            lw=1.5, label="unit Gaussian")
    ax.set_xlabel(r"normalized null $(M - \mathrm{med})/\sigma_M$")
    ax.set_ylabel("density")
    ax.set_title(f"(d) Noise-limited closure: robust σ(z) = "
                 f"{stats[tag]['sigma_z']:.2f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Full-spectrum (760-1250 cm$^{-1}$) silica null test on "
                 "real ACE occultations, calibrated two-component background",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG, dpi=180, bbox_inches="tight")
    print(f"Saved -> {FIG}")


if __name__ == "__main__":
    main()
