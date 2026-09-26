"""
ace_nulltest_convention.py -- the convention gap between the ACE null test
(slant optical depth at ONE tangent height, through the whole SAI layer)
and the design threshold (local extinction of a 0.5-km retrieved shell at
z = 20 km).  Paper 1 round 30, Doron: "the narrow window without visible
anchor gives 0.12 Tg" (Sect. 4) versus "window + one channel gives
0.32 Tg" (Sect. 3 linearized sweep).

Both analyses use the same spectral shapes and the same per-element floors,
so the projection geometry (1 - R^2) is the same; what differs is the
silica SIGNAL per Tg against those floors:
  * design (shell) convention: the floors are mapped to a 0.5-km shell,
    sigma_alpha = 1.5e-8 m^-1 <-> 2.04e-3 OD (P_eff = 136 km including the
    onion-peel factor), and the signal is the LOCAL per-Tg extinction at
    20 km;
  * null test (slant) convention: the fit is on the slant OD at the
    occultation's own tangent height (median 20.5 km), and the per-Tg
    silica column is the chord through the entire constant-mixing-ratio
    layer (16 km to 30 hPa, ~8 km deep).
Computed here:
  1. per-Tg silica signal at the 8.75-um window element in both
     conventions and their ratio rho (at the median and quartile tangent
     heights of the 101-occultation null-test sample);
  2. like-for-like check: the linearized formal sigma(M) of the win01 /
     win2 / full variants rebuilt from the cached slant bases
     (outputs/ace_v52/basis_spectra_cal.npz) against the formal sigmas the
     null test itself reports (outputs/ace_v52/fullspectrum_stats.json);
  3. the null-test numbers expressed in the shell convention (x rho) and
     the design numbers in the slant convention (/ rho).

Run from the repo root:  python scripts/ace_nulltest_convention.py
Output: printed tables + outputs/ace_v52/nulltest_convention.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from calibrated_background_thresholds import (  # noqa: E402
    ALT, iz, RMED_SIL_NM, SIGMA_SIL, SIG_874_ABS)
from saimon.materials import SilicaRefractiveIndex  # noqa: E402
from saimon.sai import create_silica_sai_layer  # noqa: E402

OUT = _ROOT / "outputs" / "ace_v52"
OD_PER_M1 = 2.04e-3 / SIG_874_ABS          # 136 km effective shell path
NAMES = ("f", "c", "jrf", "jsf", "jrc", "jsc", "sil")


def sigma_sil(A, isil):
    """sigma of the last-column amplitude by stable projection: the silica
    column against the span of the nuisance columns (whitened A)."""
    t = A[:, isil]
    N = np.delete(A, isil, axis=1)
    t_par = N @ np.linalg.lstsq(N, t, rcond=None)[0]
    t_perp = t - t_par
    return 1.0 / np.linalg.norm(t_perp), 1 - float(t_par @ t_par) / float(t @ t)


def main():
    cache = np.load(OUT / "basis_spectra_cal.npz")
    nu, H = cache["nu"], cache["h_km"]
    wl = 1e4 / nu
    thr = json.load(open(_ROOT / "outputs" /
                         "calibrated_background_thresholds.json"))
    el = np.array(thr["window_elements_um"])
    sig_win_m1 = np.array(thr["window_floors_m1"])
    sig_win_od = sig_win_m1 * OD_PER_M1
    stats = json.load(open(OUT / "fullspectrum_stats.json"))
    rows = list(csv.DictReader(open(OUT / "fullspectrum.csv")))
    h_all = np.array([float(r["h"]) for r in rows])
    h_med = float(np.median(h_all))
    h_q = np.percentile(h_all, [25, 75])
    h_ref = 32.0     # typical median of the >=30 km reference spectra

    # 1. per-Tg silica signal, both conventions --------------------------
    silica_ri = SilicaRefractiveIndex("data/optics/SiO2_KitamuraPopova.yml")
    sai = create_silica_sai_layer(ALT, 1.0, refractive_index=silica_ri,
                                  rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL)
    sub = np.linspace(-0.05, 0.05, 41)
    t_local = np.array([sai.extinction_profile_m1((c + sub) * 1e-6)[iz, :]
                        .mean() for c in el])                # m^-1 per Tg
    t_shell_od = t_local * OD_PER_M1                        # OD per Tg

    def slant_cols(h):
        cols = []
        for n in NAMES:
            b = cache[n]
            v = np.array([np.interp(h, H, b[:, j]) - np.interp(h_ref, H, b[:, j])
                          for j in range(b.shape[1])])
            cols.append(v)
        return np.column_stack(cols)

    def element_avg(J):
        return np.array([J[(wl >= c - 0.05) & (wl < c + 0.05)].mean(axis=0)
                         for c in el])

    i875 = int(np.argmin(np.abs(el - 8.75)))
    conv = {}
    print("per-Tg silica signal at the 8.75-um window element")
    for lab, h in [("median", h_med), ("q25", h_q[0]), ("q75", h_q[1]),
                   ("20.0", 20.0)]:
        Jel = element_avg(slant_cols(h))
        t_slant = Jel[:, -1]
        rho = float(t_slant[i875] / t_shell_od[i875])
        # floor-weighted ratio over the whole window (what sigma(M) sees)
        rho_w = float(np.sqrt(np.sum((t_slant / sig_win_od) ** 2)
                              / np.sum((t_shell_od / sig_win_od) ** 2)))
        conv[lab] = dict(h_km=float(h), slant_od_per_tg=float(t_slant[i875]),
                         shell_od_per_tg=float(t_shell_od[i875]),
                         local_ext_per_tg_m1=float(t_local[i875]),
                         rho_875=rho, rho_window_weighted=rho_w,
                         effective_layer_chord_km=float(
                             t_slant[i875] / t_local[i875] / 1e3))
        print(f"  h = {h:4.1f} km ({lab:6s}): slant {t_slant[i875]:.4f} OD/Tg,"
              f" shell {t_shell_od[i875]:.4f} OD/Tg (local "
              f"{t_local[i875]:.2e} m^-1/Tg x {OD_PER_M1/1e3:.0f} km); "
              f"rho = {rho:.2f} (window-weighted {rho_w:.2f}); layer chord "
              f"{t_slant[i875]/t_local[i875]/1e3:.0f} km")
    rho = conv["median"]["rho_875"]

    # 2. like-for-like formal sigma from the cached slant bases -----------
    print(f"\nlinearized formal sigma(M) from the cached slant bases at "
          f"h = {h_med:.1f} km vs the null test's own median formal sigma")
    J = slant_cols(h_med)
    Jel = element_avg(J)
    like = {}
    full = np.ones(7, bool)
    std = np.array([1, 1, 1, 1, 0, 0, 1], bool)
    checks = [
        ("win01", Jel, sig_win_od, full, True),
        ("win01 (no constant, 7-param)", Jel, sig_win_od, full, False),
        ("win01 std 5-param + const", Jel, sig_win_od, std, True),
        ("win2", J[(wl >= 7.8) & (wl <= 9.3)], 8e-3, full, True),
        ("full", J[(nu >= 760) & (nu <= 1250)], 8e-3, full, True),
        ("std", J[(nu >= 760) & (nu <= 1250)], 8e-3, std, True),
    ]
    for lab, Jm, sig, use, const in checks:
        Jm = Jm[:, use]
        ok = np.all(np.isfinite(Jm), axis=1)
        sig_v = np.broadcast_to(np.asarray(sig, float), (len(Jm),))[ok]
        A = Jm[ok] / sig_v[:, None]
        if const:
            A = np.column_stack([A, 1.0 / sig_v])
        isil = int(use.sum()) - 1
        s, one_minus_r2 = sigma_sil(A, isil)
        key = lab.split()[0]
        rep = stats[key]["sigma_formal"] if lab in stats else None
        like[lab] = dict(sigma_formal_linearized=float(s),
                         one_minus_R2=one_minus_r2,
                         sigma_formal_reported=rep,
                         sigma_emp_reported=stats[key]["sigma_emp"]
                         if lab in stats else None)
        print(f"  {lab:32s}: sigma = {s:.4f} Tg (1-R2 = {one_minus_r2:.3f})"
              + (f"; null test reports formal {rep:.4f}, empirical "
                 f"{stats[key]['sigma_emp']:.4f}" if rep else ""))

    # 3. cross-convention table --------------------------------------------
    print(f"\nnull-test thresholds (slant, as quoted) -> shell convention "
          f"(x rho = {rho:.2f}); design thresholds (shell) -> slant (/ rho)")
    table = {}
    for tag, lab in [("full", "8-13 um, all bins"),
                     ("win2", "7.8-9.3 um, native bins"),
                     ("win01", "7.8-9.3 um, 15 x 0.1-um elements")]:
        s = stats[tag]
        emp, form = 3 * s["sigma_emp"], 3 * s["sigma_formal"]
        table[tag] = dict(mmin_emp_slant=emp, mmin_formal_slant=form,
                          mmin_emp_shell=emp * rho,
                          mmin_formal_shell=form * rho)
        print(f"  null test {lab:34s}: empirical {emp:.3f} -> {emp*rho:.2f} Tg;"
              f" formal {form:.3f} -> {form*rho:.2f} Tg")
    sweep = json.load(open(_ROOT / "outputs" / "window_vis_roster_sweep.json"))
    res = sweep["results"]["CALIBRATED undisturbed (LM65T223)"]
    design = {}
    for r_lab in ("full roster (448,756,869,1021,1250,1544)",
                  "heritage triplet (448,756,1544)", "521 only",
                  "window alone"):
        for ns in ("full 7-param", "A2+rs (5-param)"):
            m = res[r_lab][ns]["mmin_tg"]
            design[f"{r_lab} | {ns}"] = dict(
                mmin_shell=m, mmin_slant=(m / rho) if m else None)
            print(f"  design window + {r_lab:42s} {ns:16s}: "
                  + (f"{m:.3f} -> {m/rho:.3f} Tg" if m else "degenerate"))
    for tag in ("band 8-13 um @0.25", "band 8-13 @0.1"):
        band = thr["results"]["CALIBRATED quiet (LM65T223, 2-comp)"][
            f"{tag}: measured floors"]["full 7-param"]
        for k, v in band.items():
            m = v["mmin_tg"]
            if not m or not np.isfinite(m):
                continue
            design[f"{tag} | {k}"] = dict(mmin_shell=m, mmin_slant=m / rho)
            print(f"  design {tag}, {k:22s} full 7-param: "
                  f"{m:.3f} -> {m/rho:.3f} Tg")

    out = dict(od_per_m1=OD_PER_M1, h_ref_km=h_ref, h_median_km=h_med,
               h_quartiles_km=list(map(float, h_q)), n_occ=len(h_all),
               convention=conv, rho=rho, like_for_like=like,
               nulltest_cross_convention=table,
               design_cross_convention=design)
    with open(OUT / "nulltest_convention.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\narchived -> {OUT / 'nulltest_convention.json'}")


if __name__ == "__main__":
    main()
