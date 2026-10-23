# Ported from the working repository's scripts/ace_ash_positive_control.py (paper 1 round 63); paths follow the repository layout.
"""
ace_ash_positive_control.py -- positive control of the paper-1 composition
discriminator on REAL solid aerosol: the ash-sulfate spectra of Lecours,
Boone & Bernath (2026, egusphere-2026-4424; Zenodo 10.5281/zenodo.21479502).

Question (Doron's reply to Peter Bernath, 2026-10-07): does the full
background marginalization of Sect. 4 recover a solid aerosol that is
really there, and does the 9.1-um volcanic-ash Si-O band separate from the
8.80-um silica reststrahlen band?

Sample
  * Lecours et al. flag 771 occultations (1825 spectra) mission-wide as
    ash-sulfate mixtures; 42 of them (57 spectra, tangent heights 9-14.5 km,
    2005-2015, SH) are in the PUBLIC v5.2 residual release
    (data/ace/ash_lecours2026/ash_manifest_v52.txt, fetched 2026-10-18).  10 of the 57
    carry their full ash+sulfate retrieval (8 PCC 2011, 2 Calbuco 2015:
    N_l^ash, r_m^ash, N_l^sul, r_m^sul); the other 47 are flagged 'other'.
  * Control: the 101 null-test occultations of
    reproduce/ace_fullspectrum_retrieval.py (20-25 N, Jun-Aug, ~20.5 km).

Method (slant space, the null test's machinery imported bit-identically)
  * data: residual spectrum at the flagged tangent height, self-calibrated
    bin by bin against the median >=30-km spectrum of the same occultation
    (>=25 km if fewer than two above 30 km);
  * background: the calibrated two-component sulfate background with both
    amplitudes and both modes' (r, sigma) terms free + a constant (the
    paper's seven background parameters + self-calibration constant), the
    spectral shapes at the null test's reference pair (20.5 km tangent,
    >=30 km reference) so the silica column is 'Tg-equivalent in the
    null-test slant convention' (0.209 OD per Tg at 8.80 um);
  * solid-aerosol columns: (i) silica, the paper's shape (Kitamura/Popova
    optics, r_med 268 nm, sigma 1.31); (ii) ash, the per-particle mean
    extinction cross-section [cm^2] of a lognormal (r_m = 0.085 um, S = 1.6,
    the Lecours fixed width and their median PCC radius) with the Deguine
    et al. (2020) Calbuco optical constants (the set Lecours use; the
    Puyehue-Cordon Caulle set as a sensitivity), so the ash amplitude is
    N_l [particles cm^-2], directly comparable with their column;
  * fits: S (sulfate only), S+sil, S+ash, S+sil+ash; robust weighted LSQ,
    sigma_bin = 8e-3, bins clipped at 4 sigma with the fullest model, then
    all four refit on that common mask so chi^2 differences are like for
    like;
  * matched-PSD variant for the 10 spectra with a published retrieval:
    ash shape at THEIR r_m^ash.

Outputs
  outputs/ace_v52/ash_positive_control.csv    per spectrum
  outputs/ace_v52/ash_positive_control.json   summary
  figures/ace_ash_positive_control.png
Run from the repo root (after ace_fullspectrum_retrieval.py, for the basis
cache):  python reproduce/ace_ash_positive_control.py
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

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ace_fullspectrum_retrieval as F  # noqa: E402
from saimon.refractive_index import TabulatedRI  # noqa: E402
from saimon.sulfate_background import (SulfateAerosolColumn,  # noqa: E402
                                      LognormalModeProfile)

ZEN = _ROOT / "data" / "ace" / "ash_lecours2026"
OUT = _ROOT / "outputs" / "ace_v52"
FIG = _ROOT / "figures" / "ace_ash_positive_control.png"
ASH_RI = {"calbuco": _ROOT / "data" / "optics" / "ash_deguine2020" / "calbuco_ash_Deguine_2020.ri",
          "pcc": _ROOT / "data" / "optics" / "ash_deguine2020" / "puyehue-cordon-caulle_ash_Deguine_2020.ri"}
H_SHAPE, H_REF_SHAPE = 20.5, 33.0      # null-test reference pair for the shapes
RM_ASH_UM, S_ASH = 0.085, 1.6          # Lecours: S fixed; r_m their PCC median
SIG = F.SIG_BIN
NAMES_BG = ("f", "c", "jrf", "jsf", "jrc", "jsc")


# ---------------------------------------------------------------------------
def load_ri(path):
    dat = np.loadtxt(path, comments="#")
    wavn, n, k = dat[:, 0], dat[:, 1], dat[:, 2]
    wl_nm = 1e7 / wavn
    o = np.argsort(wl_nm)
    return TabulatedRI(wl_nm[o], n[o], k[o])


def ash_cross_section_cm2(nu_grid, ri, rm_um, s):
    """Per-particle lognormal-mean extinction cross-section [cm^2]."""
    col = SulfateAerosolColumn(
        altitude_m=np.array([0.0, 1000.0]),
        modes=[LognormalModeProfile("ash", rm_um * 1e-6, s, np.array([1.0, 1.0]))],
        density_kg_m3=2400.0, weight_percent_h2so4=100.0,
        refractive_index=ri, name="ash")
    wl_m = 1e-2 / nu_grid
    return col.extinction_profile_m1(wl_m)[0] * 1e4


def self_calibrated(occ, alt, nu_grid):
    """tau(alt) self-calibrated against the occultation's high spectra."""
    d = F.DATA / "residual" / occ
    files = {}
    for f in d.iterdir():
        try:
            files[round(float(f.name.split(".", 1)[1].removesuffix(".gz")), 1)] = f   # repo: gzipped residuals
        except ValueError:
            continue
    if round(alt, 1) not in files:
        return None
    alts = np.array(sorted(files))
    for cut in (30.0, 25.0, 20.0):
        hi = alts[alts >= cut]
        if len(hi) >= 2:
            break
    else:
        return None
    t = F.load_spectrum(files[round(alt, 1)], nu_grid)
    tref = np.nanmedian(np.array([F.load_spectrum(files[a], nu_grid)
                                  for a in hi]), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = -np.log(t) + np.log(tref)
    return tau, float(np.median(hi))


def fit_family(tau, nu_grid, cols_bg, col_sil, col_ash):
    """S, S+sil, S+ash, S+sil+ash on a common 4-sigma-clipped mask."""
    rng = (nu_grid >= F.NU_LO) & (nu_grid <= F.NU_HI)
    ok = np.isfinite(tau) & rng
    if ok.sum() < 100:
        return None
    one = np.ones_like(nu_grid)
    J = {"S": np.column_stack(cols_bg + [one]),
         "Ssil": np.column_stack(cols_bg + [one, col_sil]),
         "Sash": np.column_stack(cols_bg + [one, col_ash]),
         "Sboth": np.column_stack(cols_bg + [one, col_sil, col_ash])}
    # clip once with the fullest model
    _, _, use, _ = F._solve(J["Sboth"], tau, ok, SIG, -1, clip=True)
    res = {"nbin": int(use.sum())}
    for tag, Jm in J.items():
        th, cov, _, _ = F._solve(Jm, tau, use, SIG, -1, clip=False)
        r = tau[use] - Jm[use] @ th
        chi2 = float(np.sum(r**2) / SIG**2)
        dof = use.sum() - Jm.shape[1]
        res[tag] = dict(theta=th, cov=cov, chi2=chi2, chi2red=chi2 / dof)
    nb = len(cols_bg) + 1
    out = dict(nbin=res["nbin"], chi2_S=res["S"]["chi2"],
               chi2red_S=res["S"]["chi2red"])
    # S+sil
    th, cov = res["Ssil"]["theta"], res["Ssil"]["cov"]
    out.update(M_sil=float(th[nb]), sM_sil=float(np.sqrt(cov[nb, nb])),
               dchi2_sil=res["S"]["chi2"] - res["Ssil"]["chi2"])
    # S+ash
    th, cov = res["Sash"]["theta"], res["Sash"]["cov"]
    out.update(N_ash=float(th[nb]), sN_ash=float(np.sqrt(cov[nb, nb])),
               dchi2_ash=res["S"]["chi2"] - res["Sash"]["chi2"],
               chi2red_ash=res["Sash"]["chi2red"])
    # S+sil+ash
    th, cov = res["Sboth"]["theta"], res["Sboth"]["cov"]
    out.update(M_sil_b=float(th[nb]), sM_sil_b=float(np.sqrt(cov[nb, nb])),
               N_ash_b=float(th[nb + 1]),
               sN_ash_b=float(np.sqrt(cov[nb + 1, nb + 1])),
               rho_sil_ash=float(cov[nb, nb + 1]
                                 / np.sqrt(cov[nb, nb] * cov[nb + 1, nb + 1])),
               dchi2_both=res["S"]["chi2"] - res["Sboth"]["chi2"],
               chi2red_both=res["Sboth"]["chi2red"])
    out["_models"] = {k: (J[k], res[k]["theta"]) for k in J}
    out["_use"] = use
    return out


# ---------------------------------------------------------------------------
def main():
    nu = F.reference_grid()
    bases = F.build_bases(nu)
    wl_um = 1e4 / nu

    # shapes at the null-test reference pair
    Jshape = F.basis_columns(H_SHAPE, H_REF_SHAPE, bases, nu,
                             NAMES_BG + ("sil",))
    cols_bg = [Jshape[:, i] for i in range(len(NAMES_BG))]
    col_sil = Jshape[:, len(NAMES_BG)]
    i880 = np.argmin(np.abs(wl_um - 8.80))
    print(f"silica shape: {col_sil[i880]:.3f} OD per Tg at 8.80 um "
          f"(null-test slant convention)")

    ri = {k: load_ri(v) for k, v in ASH_RI.items()}
    col_ash = {k: ash_cross_section_cm2(nu, r, RM_ASH_UM, S_ASH)
               for k, r in ri.items()}
    for k, c in col_ash.items():
        j = np.argmax(c)
        print(f"ash shape [{k}]: peak {c[j]:.3e} cm^2 at {wl_um[j]:.2f} um "
              f"({nu[j]:.0f} cm^-1); at 8.80 um {c[i880]/c[j]:.2f} of peak")
    # silica peak position for the record
    j = np.argmax(col_sil)
    print(f"silica shape: peak at {wl_um[j]:.2f} um ({nu[j]:.0f} cm^-1)")

    # geolocation
    geo = {}
    with open(F.DATA / "occultationlist.csv") as f:
        for row in csv.DictReader(f):
            dt = row["occultation_datetime"]
            geo[row["occultation_name"]] = (
                int(dt[:4]) + (int(dt[5:7]) - 1) / 12 + int(dt[8:10]) / 365.25,
                float(row["latitude"]))

    # Lecours retrievals (PCC + Calbuco) keyed by (occ, alt)
    lec = {}
    for fn, tag in (("data_pcc.txt", "puyehue"), ("data_calbuco.txt", "calbuco")):
        with open(ZEN / fn) as f:
            hdr = f.readline().split()
            for line in f:
                v = line.split()
                if not v:
                    continue
                d = dict(zip(hdr, v))
                lec[(d["occ"], round(float(d["alt"]), 1))] = dict(
                    tag=tag, Nl_ash=float(d["Nl_ash"]),
                    Nl_ash_err=float(d["Nl_ash_error"]),
                    rm_ash=float(d["rm_ash"]), Nl_sul=float(d["Nl_sul"]),
                    rm_sul=float(d["rm_sul"]), chi2=float(d["chisquared"]),
                    lat=float(d["lat"]), year=float(d["year"]),
                    month=float(d["month"]))

    # flagged spectra present locally
    flagged = []
    with open(ZEN / "ash-sulfate_missionwide.txt") as f:
        f.readline()
        for line in f:
            v = line.split()
            if len(v) < 3:
                continue
            occ, alt, tag = v[0], round(float(v[1]), 1), v[2]
            if (F.DATA / "residual" / occ).is_dir():
                flagged.append((occ, alt, tag))
    print(f"flagged ash-sulfate spectra with public residuals: {len(flagged)} "
          f"in {len(set(o for o, _, _ in flagged))} occultations; "
          f"{sum((o, a) in lec for o, a, _ in flagged)} with a Lecours retrieval")

    rows, examples = [], {}
    for occ, alt, tag in flagged:
        got = self_calibrated(occ, alt, nu)
        if got is None:
            print(f"  skip {occ} {alt}: no spectrum / no reference")
            continue
        tau, h_ref = got
        r = fit_family(tau, nu, cols_bg, col_sil, col_ash["calbuco"])
        if r is None:
            print(f"  skip {occ} {alt}: <100 usable bins")
            continue
        r2 = fit_family(tau, nu, cols_bg, col_sil, col_ash["pcc"])
        row = dict(sample="ash", occ=occ, alt=alt, tag=tag, h_ref=h_ref,
                   year=geo.get(occ, (np.nan, np.nan))[0],
                   lat=geo.get(occ, (np.nan, np.nan))[1])
        row.update({k: v for k, v in r.items() if not k.startswith("_")})
        row.update(N_ash_pcc=r2["N_ash"], sN_ash_pcc=r2["sN_ash"],
                   dchi2_ash_pcc=r2["dchi2_ash"])
        L = lec.get((occ, alt))
        if L:
            row.update(lec_Nl_ash=L["Nl_ash"], lec_Nl_ash_err=L["Nl_ash_err"],
                       lec_rm_ash=L["rm_ash"], lec_Nl_sul=L["Nl_sul"],
                       lec_rm_sul=L["rm_sul"], lec_chi2=L["chi2"])
            if not np.isfinite(row["year"]):
                row["year"] = L["year"] + (L["month"] - 0.5) / 12
                row["lat"] = L["lat"]
            # matched-PSD variant at THEIR median radius (Calbuco optics,
            # the set they use)
            cm = ash_cross_section_cm2(nu, ri["calbuco"], L["rm_ash"], S_ASH)
            rm = fit_family(tau, nu, cols_bg, col_sil, cm)
            row.update(N_ash_match=rm["N_ash"], sN_ash_match=rm["sN_ash"],
                       dchi2_ash_match=rm["dchi2_ash"])
        else:
            row.update(lec_Nl_ash=np.nan, lec_Nl_ash_err=np.nan,
                       lec_rm_ash=np.nan, lec_Nl_sul=np.nan, lec_rm_sul=np.nan,
                       lec_chi2=np.nan, N_ash_match=np.nan,
                       sN_ash_match=np.nan, dchi2_ash_match=np.nan)
        rows.append(row)
        examples[(occ, alt)] = (tau, r)

    # control: the 101 null-test occultations, own (h, h_ref) for the
    # silica column as in the null test; ash shape (Calbuco) added
    keep = set(str(o) for o in np.load(F.SIDE / "matrix_19_22.npz",
                                        allow_pickle=True)["occs"])
    nctrl = 0
    for d in sorted((F.DATA / "residual").iterdir()):
        if d.name not in keep:
            continue
        got = F.occultation_data(d.name, nu)
        if got is None:
            continue
        h, tau, h_ref = got
        Jh = F.basis_columns(h, h_ref, bases, nu, NAMES_BG + ("sil",))
        r = fit_family(tau, nu, [Jh[:, i] for i in range(len(NAMES_BG))],
                       Jh[:, len(NAMES_BG)], col_ash["calbuco"])
        if r is None:
            continue
        row = dict(sample="null", occ=d.name, alt=h, tag="null", h_ref=h_ref,
                   year=geo.get(d.name, (np.nan, np.nan))[0],
                   lat=geo.get(d.name, (np.nan, np.nan))[1])
        row.update({k: v for k, v in r.items() if not k.startswith("_")})
        rows.append(row)
        nctrl += 1
    print(f"control (null-test) spectra fitted: {nctrl}")

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(OUT / "ash_positive_control.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, np.nan) for k in keys})

    # ---- summary -----------------------------------------------------------
    A = [r for r in rows if r["sample"] == "ash"]
    N = [r for r in rows if r["sample"] == "null"]

    def arr(rs, k):
        return np.array([r[k] for r in rs], float)

    def robust(x):
        x = x[np.isfinite(x)]
        med = np.median(x)
        return float(med), float(1.4826 * np.median(np.abs(x - med)))

    summ = {"n_ash_spectra": len(A), "n_ash_occ": len(set(r["occ"] for r in A)),
            "n_lecours_retrieval": int(np.isfinite(arr(A, "lec_Nl_ash")).sum()),
            "n_null": len(N), "rm_ash_um": RM_ASH_UM, "S_ash": S_ASH,
            "silica_od_per_tg_880": float(col_sil[i880])}
    for lab, rs in (("ash", A), ("null", N)):
        z_ash = arr(rs, "N_ash") / arr(rs, "sN_ash")
        z_sil = arr(rs, "M_sil") / arr(rs, "sM_sil")
        d = {}
        d["N_ash_median"], d["N_ash_robust_sd"] = robust(arr(rs, "N_ash"))
        d["sN_ash_median"] = float(np.median(arr(rs, "sN_ash")))
        d["z_ash_median"] = float(np.median(z_ash))
        d["frac_ash_gt3"] = float(np.mean(z_ash > 3))
        d["frac_ash_gt5"] = float(np.mean(z_ash > 5))
        d["M_sil_median"], d["M_sil_robust_sd"] = robust(arr(rs, "M_sil"))
        d["z_sil_median"] = float(np.median(z_sil))
        d["frac_sil_gt3"] = float(np.mean(z_sil > 3))
        d["dchi2_ash_median"] = float(np.median(arr(rs, "dchi2_ash")))
        d["dchi2_sil_median"] = float(np.median(arr(rs, "dchi2_sil")))
        d["dchi2_ash_minus_sil_median"] = float(np.median(
            arr(rs, "dchi2_ash") - arr(rs, "dchi2_sil")))
        d["frac_prefer_ash"] = float(np.mean(
            arr(rs, "dchi2_ash") > arr(rs, "dchi2_sil")))
        d["rho_sil_ash_median"] = float(np.median(arr(rs, "rho_sil_ash")))
        d["M_sil_both_median"], d["M_sil_both_robust_sd"] = robust(arr(rs, "M_sil_b"))
        d["N_ash_both_median"], _ = robust(arr(rs, "N_ash_b"))
        d["chi2red_S_median"] = float(np.median(arr(rs, "chi2red_S")))
        d["chi2red_both_median"] = float(np.median(arr(rs, "chi2red_both")))
        summ[lab] = d
    # Lecours comparison
    m = np.isfinite(arr(A, "lec_Nl_ash"))
    if m.any():
        ours, theirs = arr(A, "N_ash")[m], arr(A, "lec_Nl_ash")[m]
        ours_m = arr(A, "N_ash_match")[m]
        summ["lecours_compare"] = dict(
            n=int(m.sum()),
            ratio_median_fixed_psd=float(np.median(ours / theirs)),
            ratio_median_matched_psd=float(np.median(ours_m / theirs)),
            theirs_median=float(np.median(theirs)),
            ours_median=float(np.median(ours)),
            ours_matched_median=float(np.median(ours_m)),
            their_rm_median=float(np.median(arr(A, "lec_rm_ash")[m])),
            pearson_log_fixed=float(np.corrcoef(np.log(ours.clip(1e6)),
                                                np.log(theirs))[0, 1]),
            pearson_log_matched=float(np.corrcoef(np.log(ours_m.clip(1e6)),
                                                  np.log(theirs))[0, 1]))
    with open(OUT / "ash_positive_control.json", "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))

    # ---- per-spectrum table (the 10 with a retrieval) -----------------------
    print("\nocc        alt  tag       N_ash(ours)   +-      N_ash(Lecours)  +-     "
          "rm_L   N_match     z_ash  z_sil  dchi2_ash dchi2_sil  M_sil(Tg-eq)")
    for r in sorted(A, key=lambda r: (r["tag"], r["year"])):
        if np.isfinite(r["lec_Nl_ash"]):
            print(f"{r['occ']:9s} {r['alt']:5.1f} {r['tag']:8s} "
                  f"{r['N_ash']:11.3e} {r['sN_ash']:8.1e} {r['lec_Nl_ash']:11.3e} "
                  f"{r['lec_Nl_ash_err']:8.1e} {r['lec_rm_ash']:.3f} "
                  f"{r['N_ash_match']:11.3e} {r['N_ash']/r['sN_ash']:6.1f} "
                  f"{r['M_sil']/r['sM_sil']:6.1f} {r['dchi2_ash']:9.1f} "
                  f"{r['dchi2_sil']:9.1f} {r['M_sil']:+.3f}")

    # ---- figure --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # (a) example: the flagged spectrum with the largest ash significance
    # among those with a Lecours retrieval
    cand = [r for r in A if np.isfinite(r["lec_Nl_ash"])]
    ex = max(cand, key=lambda r: r["N_ash"] / r["sN_ash"]) if cand else \
        max(A, key=lambda r: r["N_ash"] / r["sN_ash"])
    tau, r = examples[(ex["occ"], ex["alt"])]
    ax = axes[0, 0]
    use = r["_use"]
    ax.plot(nu, np.where(use, tau, np.nan), lw=0.8, color="0.35",
            label=f"{ex['occ']} {ex['alt']:.1f} km ({ex['tag']})")
    JS, thS = r["_models"]["S"]
    JA, thA = r["_models"]["Sash"]
    JL, thL = r["_models"]["Ssil"]
    ax.plot(nu, JS @ thS, color="tab:blue", lw=1.4,
            label=f"sulfate background only (χ²/dof {r['chi2red_S']:.2f})")
    ax.plot(nu, JA @ thA, color="tab:green", lw=1.4, ls="--",
            label=f"+ ash: N={r['N_ash']:.1e}±{r['sN_ash']:.0e} cm⁻² "
                  f"(χ²/dof {r['chi2red_ash']:.2f})")
    ax.plot(nu, JL @ thL, color="tab:red", lw=1.0, ls=":",
            label=f"+ silica instead: M={r['M_sil']:+.3f}±{r['sM_sil']:.3f} Tg-eq")
    ax.plot(nu, col_ash["calbuco"] * r["N_ash"], color="tab:green", lw=0.8,
            alpha=0.6, label="ash component")
    ax.set_xlabel("wavenumber [cm$^{-1}$]")
    ax.set_ylabel(r"$\tau_{\rm slant}$ (self-calibrated)")
    ax.set_title("(a) A flagged ash-sulfate spectrum and the three fits", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) detection significance: ash and null samples
    ax = axes[0, 1]
    zA = arr(A, "N_ash") / arr(A, "sN_ash")
    zN = arr(N, "N_ash") / arr(N, "sN_ash")
    bins = np.linspace(-6, max(12, np.nanpercentile(zA, 98)), 40)
    ax.hist(np.clip(zN, bins[0], bins[-1]), bins=bins, color="tab:gray",
            alpha=0.6, label=f"null sample (N={len(zN)}), tropical ~20.5 km")
    for tag, c in (("puyehue", "tab:orange"), ("calbuco", "tab:purple"),
                   ("other", "tab:green")):
        zz = np.array([r["N_ash"] / r["sN_ash"] for r in A if r["tag"] == tag])
        if len(zz):
            ax.hist(np.clip(zz, bins[0], bins[-1]), bins=bins, color=c,
                    alpha=0.6, label=f"Lecours-flagged: {tag} (N={len(zz)})")
    ax.axvline(3, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"ash column significance $N_{\rm ash}/\sigma$")
    ax.set_ylabel("spectra")
    ax.set_title("(b) Ash detected with the full background free", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (c) ours vs Lecours
    ax = axes[1, 0]
    if cand:
        x = arr(cand, "lec_Nl_ash"); xe = arr(cand, "lec_Nl_ash_err")
        y = arr(cand, "N_ash"); ye = arr(cand, "sN_ash")
        ym = arr(cand, "N_ash_match"); yme = arr(cand, "sN_ash_match")
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", color="tab:green",
                    ms=5, lw=0.8, label=f"fixed PSD r$_m$={RM_ASH_UM} µm, S={S_ASH}")
        ax.errorbar(x, ym, xerr=xe, yerr=yme, fmt="s", color="tab:olive",
                    ms=4, lw=0.8, alpha=0.8, label="matched PSD (their r$_m$)")
        lo, hi = 3e7, 1e10          # one Lecours fit has a 3e13 error bar
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls="--", label="1:1")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Lecours et al. (2026) $N_l^{\rm ash}$ [cm$^{-2}$]")
    ax.set_ylabel(r"this retrieval $N_l^{\rm ash}$ [cm$^{-2}$]")
    ax.set_title("(c) Ash column against the published retrieval", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # (d) silica-equivalent mass: does ash fool the 8.80-um channel?
    ax = axes[1, 1]
    MA, MN = arr(A, "M_sil"), arr(N, "M_sil")
    MAb = arr(A, "M_sil_b")
    b = np.linspace(-0.3, 0.3, 41)
    ax.hist(np.clip(MN, b[0], b[-1]), bins=b, color="tab:gray", alpha=0.6,
            label=f"null sample: σ={summ['null']['M_sil_robust_sd']:.3f} Tg")
    ax.hist(np.clip(MA, b[0], b[-1]), bins=b, color="tab:red", alpha=0.5,
            label="ash spectra, silica shape only (S+sil)")
    ax.hist(np.clip(MAb, b[0], b[-1]), bins=b, color="tab:blue", alpha=0.4,
            histtype="step", lw=1.6,
            label="ash spectra, silica with ash co-fitted (S+sil+ash)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("retrieved silica mass [Tg-equivalent, null-test convention]")
    ax.set_ylabel("spectra")
    ax.set_title("(d) The silica channel on real ash", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("Positive control: the paper-1 discriminator on the Lecours "
                 "et al. (2026) ash-sulfate spectra (public v5.2 residuals)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG, dpi=180, bbox_inches="tight")
    print(f"Saved -> {FIG}")


if __name__ == "__main__":
    main()
