"""
ace_ruang_bandshape.py -- composition-discriminator rehearsal against the
Ruang eruption (17 April 2024, 2N): does the post-eruption aerosol increment
seen by ACE-FTS have the sulfate band shape, or does it show any
silicate-ash reststrahlen signature?

This exercises, on a real stratospheric perturbation, exactly the band-shape
discrimination that Config B+ uses to separate silica SAI from sulfate.

Method (slant space, as everywhere in this series):
  * split the 2024 tropical occultations at the eruption date (2024.29);
    build median self-calibrated slant-OD spectra (760-1250 cm^-1) at
    tangent heights 19-22 km for the pre- and post-eruption groups;
  * form the increment Delta-tau(nu) = tau_post - tau_pre and fit it two
    ways: sulfate shape only [A_f, A_c, const], and sulfate + silica shapes
    [A_f, A_c, const, M_sil]; then with the PSD shape terms of both
    modes freed (shapes from the cached CALIBRATED two-component slant
    bases of scripts/ace_fullspectrum_retrieval.py -- round 27, Doron:
    the fits are linearized around the calibrated background of Paper 1
    Sect. 2.2, not the superseded single-mode elevated one);
  * report the retrieved 'ash-as-silica' mass and the chi2 improvement --
    a real ash signature would appear as a significant M_sil with the
    reststrahlen shape; pure sulfate predicts M_sil ~ 0;
  * timeline: per-occultation full-spectrum M and A around the eruption
    (from outputs/ace_v52/fullspectrum.csv).

Outputs
  figures/ace_ruang_bandshape.png
  printed: group sizes, Delta-tau band values, fitted A and M_sil +/- sigma,
  chi2 with/without the silica component.

Run AFTER ace_fullspectrum_retrieval.py (uses its basis cache + CSV):
  python scripts/ace_ruang_bandshape.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

DATA = _HERE / "data" / "ace" / "v52_subset"
OUT = _HERE / "outputs" / "ace_v52"
FIG = _HERE / "figures" / "ace_ruang_bandshape.png"

T_RUANG = 2024.29
NU_LO, NU_HI = 760.0, 1250.0
ALT_LO, ALT_HI = 19.0, 22.0
H_FIT = 20.5            # basis tangent height for the composite fit
H_REF = 31.0
SIG_BIN_COMPOSITE = 3.0e-3   # noise of a ~10-spectrum median composite bin


def decimal_year(dt):
    return int(dt[:4]) + (int(dt[5:7]) - 1) / 12.0 + int(dt[8:10]) / 365.25


def load_spectrum(path, nu_grid):
    dat = np.loadtxt(path)
    nu, t = dat[:, 0], dat[:, 1]
    m = (nu >= NU_LO - 1) & (nu <= NU_HI + 1)
    t_g = np.interp(nu_grid, nu[m], t[m], left=np.nan, right=np.nan)
    t_g[t_g <= 0.01] = np.nan
    return t_g


def occ_corrected_tau(occ, nu_grid):
    """Self-calibrated slant-OD spectra of one occultation at 19-22 km."""
    d = DATA / "residual" / occ
    files = []
    for f in d.iterdir():
        try:
            files.append((float(f.name.split(".", 1)[1].removesuffix(".gz")), f))
        except ValueError:
            continue
    ref_files = [f for a, f in files if a >= 30.0]
    mid_files = [f for a, f in files if ALT_LO <= a <= ALT_HI]
    if len(ref_files) < 2 or not mid_files:
        return []
    tref = np.nanmedian(np.array([load_spectrum(f, nu_grid)
                                  for f in ref_files]), axis=0)
    out = []
    for f in mid_files:
        t = load_spectrum(f, nu_grid)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(-np.log(t) + np.log(tref))
    return out


def main():
    dat = np.load(OUT / "basis_spectra_cal.npz")
    nu_grid, h_km = dat["nu"], dat["h_km"]

    def basis(name):
        b = dat[name]
        return np.array([np.interp(H_FIT, h_km, b[:, j]) -
                         np.interp(H_REF, h_km, b[:, j])
                         for j in range(len(nu_grid))])
    b_f, b_c, b_sil = basis("f"), basis("c"), basis("sil")
    b_jrf, b_jsf = basis("jrf"), basis("jsf")
    b_jrc, b_jsc = basis("jrc"), basis("jsc")

    # 2024 tropical occultations, split at the eruption
    geo = {}
    with open(DATA / "occultationlist.csv") as f:
        for row in csv.DictReader(f):
            geo[row["occultation_name"]] = decimal_year(
                row["occultation_datetime"])
    # all 2024 tropical occultations are post-eruption (Jun-Aug); the
    # pre-eruption baseline is the same-season group one year earlier
    # (Aug 2023).  The increment is therefore net change Aug 2023 ->
    # Jun-Aug 2024: Ruang plus any residual Hunga Tonga decay.
    keep = set(str(o) for o in np.load(
        _HERE / "data/ace/revision20260909/"
        "matrix_19_22.npz", allow_pickle=True)["occs"])
    all_occ = [d.name for d in sorted((DATA / "residual").iterdir())
               if d.name in geo and d.name in keep]
    pre = [o for o in all_occ if 2023.0 <= geo[o] < T_RUANG]
    post = [o for o in all_occ if T_RUANG <= geo[o] < 2025.0]
    print(f"tropical occultations: {len(pre)} pre (2023), "
          f"{len(post)} post-Ruang (2024)")

    spec_pre, spec_post = [], []
    for o in pre:
        spec_pre.extend(occ_corrected_tau(o, nu_grid))
    for o in post:
        spec_post.extend(occ_corrected_tau(o, nu_grid))
    print(f"spectra 19-22 km: {len(spec_pre)} pre, {len(spec_post)} post")
    tau_pre = np.nanmedian(np.array(spec_pre), axis=0)
    tau_post = np.nanmedian(np.array(spec_post), axis=0)
    dtau = tau_post - tau_pre

    ok = np.isfinite(dtau)
    print(f"usable bins: {ok.sum()}")
    # round 43: the paper's 0.1-um element at 8.80 um (8.75-8.85 um); was the
    # 0.25-um element at 8.74 um (1128.5-1160.8 cm^-1)
    B_LO, B_HI = 1e4 / 8.85, 1e4 / 8.75
    mB = ok & (nu_grid >= B_LO) & (nu_grid <= B_HI)
    print(f"Delta-tau at the B+ element: {dtau[mB].mean():+.4f}")

    # fits to the increment: rigid sulfate shape, then with PSD freedom,
    # each with and without a silica component.  The silica-mass column is
    # always LAST.
    one = np.ones_like(nu_grid)
    psd = [b_jrf, b_jsf, b_jrc, b_jsc]
    VARIANTS = {
        "sulfate-only":        [b_f, b_c, one],
        "sulfate+silica":      [b_f, b_c, one, b_sil],
        "sulfate+PSD":         [b_f, b_c] + psd + [one],
        "sulfate+PSD+silica":  [b_f, b_c] + psd + [one, b_sil],
    }
    results = {}
    for tag, cols in VARIANTS.items():
        Jf = np.column_stack(cols)
        J = Jf[ok]
        y = dtau[ok]
        scales = np.linalg.norm(J, axis=0)          # unit-normalize columns
        Jn = J / scales
        F = Jn.T @ Jn / SIG_BIN_COMPOSITE**2
        covn = np.linalg.inv(F)
        th = (covn @ (Jn.T @ y) / SIG_BIN_COMPOSITE**2) / scales
        cov = covn / np.outer(scales, scales)
        chi2 = float(np.sum((y - J @ th)**2) / SIG_BIN_COMPOSITE**2
                     / (ok.sum() - len(cols)))
        results[tag] = (th, cov, chi2, Jf)
        msg = f"{tag:20s}: A_f = {th[0]:+.3f}, A_c = {th[1]:+.3f}"
        if tag.endswith("silica"):
            msg += (f", M_ash-as-silica = {th[-1]:+.4f} "
                    f"+- {np.sqrt(cov[-1,-1]):.4f} Tg")
        print(msg + f", chi2/dof = {chi2:.2f} (cond {np.linalg.cond(F):.1e})")

    th1, _, chi2_1, J1 = results["sulfate+PSD"]
    th2, cov2, chi2_2, J2 = results["sulfate+PSD+silica"]
    th0, cov0, chi2_0, _ = results["sulfate+silica"]
    print(f"rigid-shape counterfactual (amplitudes only + silica): "
          f"M = {th0[-1]:+.4f} +- {np.sqrt(cov0[-1,-1]):.4f} Tg "
          f"({th0[-1]/np.sqrt(cov0[-1,-1]):+.1f} sigma), chi2/dof "
          f"{chi2_0:.2f} -> with PSD freedom {chi2_1:.2f}")

    # timeline from the full-spectrum per-occultation fits
    rows = [r for r in csv.DictReader(open(OUT / "fullspectrum.csv"))
            if 2023.0 <= float(r["year"]) < 2025.0]
    t_yr = np.array([float(r["year"]) for r in rows])
    M_t = np.array([float(r["M_full"]) for r in rows])
    sM_t = np.array([float(r["sM_full"]) for r in rows])

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    ax.plot(nu_grid, dtau, lw=0.8, color="0.4",
            label=r"$\Delta\tau$ = post $-$ pre (19-22 km)")
    ax.plot(nu_grid, J1 @ th1,
            lw=1.5, color="tab:blue",
            label=f"sulfate+PSD fit (χ²/dof={chi2_1:.2f})")
    ax.plot(nu_grid, J2 @ th2,
            lw=1.2, color="tab:red", ls="--",
            label=f"+silica: M={th2[-1]:+.3f}±{np.sqrt(cov2[-1,-1]):.3f} Tg "
                  f"(χ²/dof={chi2_2:.2f})")
    ax.axvspan(1e4 / 8.85, 1e4 / 8.75, color="magenta", alpha=0.15,
               label="8.80 µm element")
    ax.set_xlabel("wavenumber [cm$^{-1}$]")
    ax.set_ylabel(r"$\Delta\tau_{\rm slant}$")
    ax.set_title("(a) Ruang increment vs band shapes", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.errorbar(t_yr, M_t, yerr=sM_t, fmt="o", ms=4, lw=0.7,
                color="tab:red", alpha=0.8)
    ax.axvline(T_RUANG, color="gray", lw=1.2)
    ax.text(T_RUANG + 0.02, ax.get_ylim()[0] + 0.02, "Ruang", rotation=90,
            fontsize=8, color="gray")
    ax.axhline(np.median(M_t), color="tab:red", lw=0.8, ls=":")
    ax.set_xlabel("year")
    ax.set_ylabel("retrieved $M$ [Tg] (full-spectrum)")
    ax.set_title("(b) Silica-mass null across the eruption", fontsize=10)
    ax.grid(alpha=0.3)

    # (c) tau at the silica element (column tau874 = 8.80 um since round 43)
    # near 20 km, 2023-2024 timeline (from mir_bands.csv)
    ax = axes[2]
    tb = [r for r in csv.DictReader(open(OUT / "mir_bands.csv"))
          if r["tau874"] and 2023.0 <= float(r["year"]) < 2025.0
          and abs(float(r["alt"]) - 20.5) <= 1.5]
    ax.scatter([float(r["year"]) for r in tb],
               [float(r["tau874"]) for r in tb],
               s=18, alpha=0.8, color="tab:purple")
    ax.axvline(T_RUANG, color="gray", lw=1.2)
    ax.set_xlabel("year")
    ax.set_ylabel(r"$\tau_{\rm slant}$(8.80 µm) near 20 km")
    ax.set_title("(c) MIR aerosol OD across the eruption", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Band-shape discrimination rehearsal: Ruang (Apr 2024) in "
                 "ACE-FTS residuals (calibrated background)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG, dpi=180, bbox_inches="tight")
    print(f"Saved -> {FIG}")


if __name__ == "__main__":
    main()
