"""
calibrated_background_problem.py -- the remaining Paper-1 Sect. 2/3 numbers
under the ACE-CALIBRATED two-component background (round-26: the major
revision of Sects. 2 and 3 onto the calibrated background model and the
window-mode detection method).

Companion to scripts/calibrated_background_thresholds.py (rounds 24-25b),
whose machinery (member optics, calibrated GloSSAC 20-25N quiet profile,
TwoComponentBackground, marginalized bound) is imported and reused
bit-identically.  That script carries the Sect.-3 threshold table; this one
computes what the rewritten Sect. 2 (the visible/NIR degeneracy) and the
Sect.-3 supporting statements need:

PART A  K-sweep: K log-spaced channels 385-1550 nm, uniform fractional
        error PHI = 4% (the Sect.-2.3 idealization), against the calibrated
        two-component background with nuisance sets
          A2      (two amplitudes; 3-param fit with silica)
          A2+rs   (+ fine-mode PSD; 5-param fit)
          full    (+ coarse-mode PSD; 7-param fit)
        plus the Franta-silica robustness check and the measured-phi
        variant at K = 16.
PART B  Forward-model input table (the tab:input replacement): per-channel
        fine/coarse/total sulfate extinction, silica extinction, adopted
        errors and silica/sulfate ratios at z = 20 km.
PART C  The mid-infrared continuum alternative (Sect. 3.4) under the
        calibrated background: Wrana3 + one continuum channel (3.8 and
        4.0 um, 0.25-um boxcar) at the 4-param convention (A1+rs, directly
        comparable to the reststrahlen 4-param row), and the modern
        comparison: extended vis/NIR + continuum channel vs + window,
        full 7-param.  Also K = 16 with one channel moved to 4.0 um.
PART D  Window-mode extras: reproduction check against the archived
        rounds-25b numbers (asserts), floor-scale scan (all measured MIR
        window floors x f), per-element silica extinction on the 0.1-um
        grid, sulfate/silica ratios at the 8.74 um element, ||t_perp||.
        Round 31 (Doron): PARTS D-F also evaluate the Wrana heritage
        triplet (448, 756, 1544 nm) + window, now the quoted design
        roster (keys 'window_triplet ...', 'floor_scan_triplet',
        'mmin_*_triplet', 'franta_window/triplet ...').
        Round 33 (Doron): the whole 8-13 um band at 0.1-um sampling
        ('band01', PART 4 of the thresholds script) is evaluated alongside
        the window in PARTS D-E (keys 'band01_triplet ...',
        'floor_scan_band01_triplet', 'mmin_7p_band01_triplet').
PART E  Loading scan: the calibrated microphysics on the same profile
        scaled to SAOD525 = 0.003 / 0.0045 (calibrated) / 0.010 / 0.0083*
        (*the post-HT matched anchor), full 7-param window mode.

Results (2026-09-15 run) are quoted in the script output and archived to
outputs/calibrated_background_problem.json.

Run from the repo root:  python scripts/calibrated_background_problem.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_PARENT = _THIS.parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

from calibrated_background_thresholds import (           # noqa: E402
    ALT, iz, RMED_SIL_NM, SIGMA_SIL, SPECTRAL_RES_UM, SIG_874_ABS,
    ALPHA_FLOOR, WL_NM, PHI, I874, ROSTERS, CAL_QUIET,
    member_ri, glossac_2025N_quiet_profile, TwoComponentBackground,
    marginal)
from saimon.materials import SilicaRefractiveIndex        # noqa: E402
from saimon.sai import create_silica_sai_layer            # noqa: E402

# CAL_QUIET is imported from the thresholds module (round 27); the window
# floors are read from that module's archive, so re-run it first.
PHI_SWEEP = 0.04
# measured SAGE III/ISS Level-2 fractional uncertainties (Table tab:combined)
PHI_MEAS_WL = np.array([448., 756., 869., 1021., 1544.])
PHI_MEAS = np.array([0.040, 0.032, 0.040, 0.045, 0.088])


def make_bext(centers_nm, widths_um=None):
    """Extinction evaluator at z = 20 km for an arbitrary channel list.
    widths_um: per-channel boxcar width in um (None/0 = monochromatic)."""
    centers_nm = np.asarray(centers_nm, dtype=float)
    if widths_um is None:
        widths_um = np.zeros(len(centers_nm))
    widths_um = np.asarray(widths_um, dtype=float)

    def bext(layer):
        out = layer.extinction_profile_m1(centers_nm * 1e-9)[iz, :].copy()
        for j, (c, w) in enumerate(zip(centers_nm, widths_um)):
            if w > 0:
                sub = np.linspace(-w / 2, w / 2, 41)
                out[j] = layer.extinction_profile_m1(
                    (c / 1000.0 + sub) * 1e-6)[iz, :].mean()
        return out
    return bext


class GenBackground(TwoComponentBackground):
    """TwoComponentBackground with a per-instance channel evaluator."""

    def __init__(self, bext, *a, **k):
        super().__init__(*a, **k)
        self._bext = bext

    def component_ext(self):
        return [self._bext(self._col(r, s, f)) for r, s, f in self.comps]

    def psd_derivs(self, i):
        r0, s0, f = self.comps[i]
        dr, ds = self.fd * r0, self.fd * s0
        er = (self._bext(self._col(r0 + dr, s0, f))
              - self._bext(self._col(r0 - dr, s0, f))) / (2 * dr)
        es = (self._bext(self._col(r0, s0 + ds, f))
              - self._bext(self._col(r0, s0 - ds, f))) / (2 * ds)
        return er, es


def nuisance_sets(bg):
    exts = bg.component_ext()
    s_total = np.sum(exts, axis=0)
    jr_f, js_f = bg.psd_derivs(0)
    jr_c, js_c = bg.psd_derivs(1)
    return s_total, {
        'A2 (3-param)':    [exts[0], exts[1]],
        'A2+rs (5-param)': [exts[0], exts[1], jr_f, js_f],
        'full 7-param':    [exts[0], exts[1], jr_f, js_f, jr_c, js_c],
    }, exts


def main():
    silica_ri = SilicaRefractiveIndex("data/optics/SiO2_KitamuraPopova.yml")
    silica_ri_franta = SilicaRefractiveIndex("data/optics/SiO2_Franta2016.yml")
    ri_lm65, wt65, T65 = member_ri("LM65T223")
    alpha_cal = glossac_2025N_quiet_profile()
    saod_cal = float(np.trapezoid(alpha_cal, ALT))
    arch = {"saod_cal": saod_cal}
    print(f"calibrated 20-25N quiet profile: SAOD525 = {saod_cal:.4f}")

    def sai_layer(ri, rmed=RMED_SIL_NM, sg=SIGMA_SIL):
        return create_silica_sai_layer(ALT, 1.0, refractive_index=ri,
                                       rmed_nm=rmed, sigma=sg)

    # ---- PART A: the visible/NIR K-sweep -----------------------------------
    print("\n" + "=" * 72)
    print("PART A: K log-spaced vis/NIR channels 385-1550 nm, PHI = 4%")
    KS = [4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 96]
    arch["ksweep"] = {}
    for K in KS:
        wl = np.geomspace(385.0, 1550.0, K)
        bext = make_bext(wl)
        bg = GenBackground(bext, alpha_cal, CAL_QUIET, ri_lm65, wt65, T65)
        s_total, nsets, _ = nuisance_sets(bg)
        t_sil = bext(sai_layer(silica_ri))
        sig = PHI_SWEEP * s_total
        idx = np.arange(K)
        row = {}
        line = f"  K ={K:3d}:"
        for lab, cols in nsets.items():
            sa, tperp, R2 = marginal(idx, cols, sig, t_sil)
            row[lab] = dict(mmin_tg=3 * sa if np.isfinite(sa) else None,
                            one_minus_R2=(1 - R2) if np.isfinite(R2)
                            else None)
            line += (f"  {lab.split()[0]:6s} "
                     + (f"{3 * sa:7.2f} Tg" if np.isfinite(sa)
                        else "    --- "))
        print(line)
        arch["ksweep"][K] = row

    # robustness variants at K = 16
    print("\n  -- K = 16 variants --")
    K = 16
    wl = np.geomspace(385.0, 1550.0, K)
    bext = make_bext(wl)
    bg = GenBackground(bext, alpha_cal, CAL_QUIET, ri_lm65, wt65, T65)
    s_total, nsets, _ = nuisance_sets(bg)
    idx = np.arange(K)
    arch["k16_variants"] = {}
    # (a) Franta silica optics
    t_fr = bext(sai_layer(silica_ri_franta))
    sig = PHI_SWEEP * s_total
    for lab in ('A2+rs (5-param)', 'full 7-param'):
        sa, _, R2 = marginal(idx, nsets[lab], sig, t_fr)
        print(f"  Franta silica    {lab:18s} M_min = {3 * sa:7.2f} Tg  "
              f"(1-R2 = {1 - R2:.2e})")
        arch["k16_variants"][f"franta {lab}"] = dict(
            mmin_tg=3 * sa, one_minus_R2=1 - R2)
    # (b) measured phi(lambda) interpolated through the clean channels
    phi_l = np.interp(np.log(wl), np.log(PHI_MEAS_WL), PHI_MEAS)
    sig_m = phi_l * s_total
    t_sil = bext(sai_layer(silica_ri))
    for lab in ('A2+rs (5-param)', 'full 7-param'):
        sa, _, R2 = marginal(idx, nsets[lab], sig_m, t_sil)
        print(f"  measured phi(l)  {lab:18s} M_min = {3 * sa:7.2f} Tg")
        arch["k16_variants"][f"measured-phi {lab}"] = dict(mmin_tg=3 * sa)

    # ---- PART B: forward-model input table ---------------------------------
    print("\n" + "=" * 72)
    print("PART B: per-channel inputs at z = 20 km, calibrated background")
    widths = np.zeros(len(WL_NM))
    widths[I874] = SPECTRAL_RES_UM
    bext7 = make_bext(WL_NM, widths)
    bg7 = GenBackground(bext7, alpha_cal, CAL_QUIET, ri_lm65, wt65, T65)
    exts7 = bg7.component_ext()
    s_tot7 = np.sum(exts7, axis=0)
    t7 = bext7(sai_layer(silica_ri))
    sig7 = np.maximum(PHI * s_tot7, ALPHA_FLOOR)
    sig7[I874] = SIG_874_ABS
    arch["input_table"] = []
    print(f"  {'lam [nm]':>9s} {'s_fine':>10s} {'s_coarse':>10s} "
          f"{'s_total':>10s} {'t_sil':>10s} {'sigma':>10s} {'t/s':>7s}")
    for j, lam in enumerate(WL_NM):
        print(f"  {lam:9.0f} {exts7[0][j]:10.3e} {exts7[1][j]:10.3e} "
              f"{s_tot7[j]:10.3e} {t7[j]:10.3e} {sig7[j]:10.3e} "
              f"{t7[j] / s_tot7[j]:7.3f}")
        arch["input_table"].append(dict(
            lam_nm=lam, s_fine=exts7[0][j], s_coarse=exts7[1][j],
            s_total=s_tot7[j], t_sil=t7[j], sigma=sig7[j],
            ratio=t7[j] / s_tot7[j]))

    # ---- PART C: the continuum alternative ---------------------------------
    print("\n" + "=" * 72)
    print("PART C: mid-infrared continuum channel, calibrated background")
    arch["continuum"] = {}
    for cont_um in (3.8, 4.0):
        wl_c = np.r_[WL_NM[:3], cont_um * 1000.0]     # Wrana3 + continuum
        w_c = np.r_[np.zeros(3), SPECTRAL_RES_UM]
        bext_c = make_bext(wl_c, w_c)
        bg_c = GenBackground(bext_c, alpha_cal, CAL_QUIET, ri_lm65,
                             wt65, T65)
        s_tc, nsets_c, exts_c = nuisance_sets(bg_c)
        jr_f, js_f = bg_c.psd_derivs(0)
        t_c = bext_c(sai_layer(silica_ri))
        for sig_mir in (1.5e-8, 1.0e-8):
            sig = np.maximum(PHI[:3] * s_tc[:3], ALPHA_FLOOR)
            sig = np.r_[sig, sig_mir]
            # 4-param convention: total amplitude at fixed split + fine PSD
            cols4 = [s_tc, jr_f, js_f]
            sa, _, R2 = marginal(np.arange(4), cols4, sig, t_c)
            print(f"  Wrana3 + {cont_um} um (sig={sig_mir:.1e}):  A1+rs "
                  f"4-param M_min = {3 * sa:7.3f} Tg  (1-R2 = "
                  f"{1 - R2:.2e})")
            arch["continuum"][f"wrana3+{cont_um}um sig{sig_mir:.1e} "
                              "A1+rs"] = dict(mmin_tg=3 * sa,
                                              one_minus_R2=1 - R2)
        # continuum channel gas OD context is in the tracegas appendix
    # modern comparison: extended vis/NIR + one continuum channel, full fit
    wl_e = np.r_[WL_NM[:I874], 3800.0]
    w_e = np.r_[np.zeros(I874), SPECTRAL_RES_UM]
    bext_e = make_bext(wl_e, w_e)
    bg_e = GenBackground(bext_e, alpha_cal, CAL_QUIET, ri_lm65, wt65, T65)
    s_te, nsets_e, _ = nuisance_sets(bg_e)
    t_e = bext_e(sai_layer(silica_ri))
    sig_e = np.maximum(PHI[:I874] * s_te[:I874], ALPHA_FLOOR)
    sig_e = np.r_[sig_e, 1.5e-8]
    for lab in ('A2+rs (5-param)', 'full 7-param'):
        sa, _, R2 = marginal(np.arange(len(wl_e)), nsets_e[lab], sig_e,
                             t_e)
        tagv = f"{3 * sa:7.3f}" if np.isfinite(sa) else "    ---"
        print(f"  extended vis/NIR + 3.8 um: {lab:18s} M_min = {tagv} Tg")
        arch["continuum"][f"extended+3.8um {lab}"] = dict(
            mmin_tg=3 * sa if np.isfinite(sa) else None)
    # K = 16 with one channel moved to 4.0 um (PHI = 4% throughout)
    wl_k = np.r_[np.geomspace(385.0, 1550.0, 15), 4000.0]
    w_k = np.r_[np.zeros(15), SPECTRAL_RES_UM]
    bext_k = make_bext(wl_k, w_k)
    bg_k = GenBackground(bext_k, alpha_cal, CAL_QUIET, ri_lm65, wt65, T65)
    s_tk, nsets_k, _ = nuisance_sets(bg_k)
    t_k = bext_k(sai_layer(silica_ri))
    sig_k = PHI_SWEEP * s_tk
    for lab in ('A2+rs (5-param)', 'full 7-param'):
        sa, _, R2 = marginal(np.arange(16), nsets_k[lab], sig_k, t_k)
        print(f"  K=16 w/ one 4.0-um channel: {lab:18s} M_min = "
              f"{3 * sa:7.2f} Tg  (1-R2 = {1 - R2:.2e})")
        arch["continuum"][f"k16+4um {lab}"] = dict(mmin_tg=3 * sa,
                                                   one_minus_R2=1 - R2)

    # ---- PART D: window-mode extras -----------------------------------------
    print("\n" + "=" * 72)
    print("PART D: window-mode reproduction, floor scan, band-shape data")
    prev = json.load(open(_PARENT / "outputs" /
                          "calibrated_background_thresholds.json"))
    win_c = np.array(prev["window_elements_um"])
    sig_win = np.array(prev["window_floors_m1"])
    wl_w = np.r_[WL_NM[:I874], win_c * 1000.0]
    w_w = np.r_[np.zeros(I874), np.full(len(win_c), 0.10)]
    bext_w = make_bext(wl_w, w_w)
    bg_w = GenBackground(bext_w, alpha_cal, CAL_QUIET, ri_lm65, wt65, T65)
    s_tw, nsets_w, exts_w = nuisance_sets(bg_w)
    t_w = bext_w(sai_layer(silica_ri))
    idx_all = np.arange(len(wl_w))
    sig_w = np.r_[np.maximum(PHI[:I874] * s_tw[:I874], ALPHA_FLOOR),
                  sig_win]
    # reproduction check against the archived rounds-25b numbers
    ref = prev["results"]["CALIBRATED quiet (LM65T223, 2-comp)"][
        "window 7.8-9.3 @0.1: measured floors"]
    for lab, refl in (('A2+rs (5-param)', 'A2+rs (5-param)'),
                      ('full 7-param', 'full 7-param')):
        sa, tperp, R2 = marginal(idx_all, nsets_w[lab], sig_w, t_w)
        want = ref[refl]['vis/NIR 6 + window']['mmin_tg']
        print(f"  reproduce {lab:18s}: M_min = {3 * sa:.4f} Tg "
              f"(archived {want:.4f}); ||t_perp|| = {tperp:.1f}")
        assert abs(3 * sa - want) < 1e-3, "window-mode reproduction failed"
        arch[f"window {lab}"] = dict(mmin_tg=3 * sa, tperp=tperp,
                                     one_minus_R2=1 - R2)
    # round 31 (Doron): the quoted roster is the Wrana heritage triplet
    # (448, 756, 1544 nm) + window; the six-channel roster is footnoted
    idx_tri = np.r_[[0, 1, 5], np.arange(I874, len(wl_w))]
    for lab in ('A2+rs (5-param)', 'full 7-param'):
        sa, tperp, R2 = marginal(idx_tri, nsets_w[lab], sig_w, t_w)
        print(f"  triplet + window {lab:18s}: M_min = {3 * sa:.4f} Tg; "
              f"||t_perp|| = {tperp:.1f}; 1-R2 = {1 - R2:.4f}")
        arch[f"window_triplet {lab}"] = dict(mmin_tg=3 * sa, tperp=tperp,
                                             one_minus_R2=1 - R2)
    # floor-scale scan (full 7-param), both rosters
    arch["floor_scan"] = {}
    arch["floor_scan_triplet"] = {}
    for f in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0):
        sig_f = np.r_[np.maximum(PHI[:I874] * s_tw[:I874], ALPHA_FLOOR),
                      f * sig_win]
        sa, _, _ = marginal(idx_all, nsets_w['full 7-param'], sig_f, t_w)
        sa_t, _, _ = marginal(idx_tri, nsets_w['full 7-param'], sig_f, t_w)
        print(f"  floors x {f:3.1f}: full 7-param M_min = {3 * sa:.3f} Tg "
              f"(six channels), {3 * sa_t:.3f} Tg (triplet)")
        arch["floor_scan"][f] = 3 * sa
        arch["floor_scan_triplet"][f] = 3 * sa_t

    # ---- round 33: the whole 8-13 um band at the same 0.1-um sampling
    # (PART 4 of calibrated_background_thresholds.py): triplet + band01
    # baseline, ||t_perp||, floor scan and Franta (loading scan in PART E)
    b01_c = np.array(prev["band01_elements_um"])
    sig_b01 = np.array(prev["band01_floors_m1"])
    wl_b = np.r_[WL_NM[:I874], b01_c * 1000.0]
    w_b = np.r_[np.zeros(I874), np.full(len(b01_c), 0.10)]
    bext_b = make_bext(wl_b, w_b)
    bg_b = GenBackground(bext_b, alpha_cal, CAL_QUIET, ri_lm65, wt65, T65)
    s_tb, nsets_b, _ = nuisance_sets(bg_b)
    t_b = bext_b(sai_layer(silica_ri))
    t_b_fr = bext_b(sai_layer(silica_ri_franta))
    idx_tri_b = np.r_[[0, 1, 5], np.arange(I874, len(wl_b))]
    sig_b = np.r_[np.maximum(PHI[:I874] * s_tb[:I874], ALPHA_FLOOR), sig_b01]
    for lab in ('A2+rs (5-param)', 'full 7-param'):
        sa, tperp, R2 = marginal(idx_tri_b, nsets_b[lab], sig_b, t_b)
        want = prev["results"]["CALIBRATED quiet (LM65T223, 2-comp)"][
            "band 8-13 @0.1: measured floors"][lab]["triplet + band01"]["mmin_tg"]
        print(f"  triplet + band01 {lab:18s}: M_min = {3 * sa:.4f} Tg (archived "
              f"{want:.4f}); ||t_perp|| = {tperp:.1f}; 1-R2 = {1 - R2:.4f}")
        assert abs(3 * sa - want) < 1e-3, "band01 reproduction failed"
        arch[f"band01_triplet {lab}"] = dict(mmin_tg=3 * sa, tperp=tperp,
                                             one_minus_R2=1 - R2)
        sa_f, _, R2f = marginal(idx_tri_b, nsets_b[lab], sig_b, t_b_fr)
        arch[f"band01_triplet franta {lab}"] = dict(mmin_tg=3 * sa_f,
                                                    one_minus_R2=1 - R2f)
        print(f"  triplet + band01 Franta {lab:18s}: M_min = {3 * sa_f:.4f} Tg")
    arch["floor_scan_band01_triplet"] = {}
    for f in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0):
        sig_f = np.r_[np.maximum(PHI[:I874] * s_tb[:I874], ALPHA_FLOOR),
                      f * sig_b01]
        sa_t, _, _ = marginal(idx_tri_b, nsets_b['full 7-param'], sig_f, t_b)
        print(f"  band01 floors x {f:3.1f}: full 7-param M_min = {3 * sa_t:.3f} Tg (triplet)")
        arch["floor_scan_band01_triplet"][f] = 3 * sa_t
    # per-element silica extinction and sulfate ratios on the window grid
    j874 = int(np.argmin(np.abs(win_c - 8.75))) + I874
    print(f"  silica t at 8.75 um (0.1-um element): {t_w[j874]:.3e} "
          f"m^-1/Tg; sulfate total there: {s_tw[j874]:.3e} m^-1 "
          f"(ratio {t_w[j874] / s_tw[j874]:.2f}); fine share of sulfate: "
          f"{exts_w[0][j874] / s_tw[j874]:.2f}")
    arch["window_874"] = dict(t_sil=t_w[j874], s_total=s_tw[j874],
                              s_fine=exts_w[0][j874],
                              ratio=t_w[j874] / s_tw[j874])
    arch["window_tsil"] = dict(zip([f"{c:.2f}" for c in win_c],
                                   [float(x) for x in t_w[I874:]]))
    # round 43: the reference element (WL_NM[I874] = 8.80 um) at the two
    # element widths, for the Sect.-3.1 boxcar sentence
    ref_nm = np.array([WL_NM[I874]])
    t_ref01 = float(make_bext(ref_nm, np.array([0.10]))(sai_layer(silica_ri))[0])
    t_ref25 = float(make_bext(ref_nm, np.array([0.25]))(sai_layer(silica_ri))[0])
    t_ref00 = float(make_bext(ref_nm, np.array([0.0]))(sai_layer(silica_ri))[0])
    print(f"  silica t at the {ref_nm[0]/1000:.2f} um reference element: "
          f"{t_ref01:.3e} (0.1 um), {t_ref25:.3e} (0.25 um), {t_ref00:.3e} (mono) m^-1/Tg")
    arch["reference_element"] = dict(center_um=ref_nm[0] / 1000, t_sil_0p1um=t_ref01,
                                     t_sil_0p25um=t_ref25, t_sil_mono=t_ref00)

    # ---- PART E: loading scan ----------------------------------------------
    print("\n" + "=" * 72)
    print("PART E: loading scan (window mode, full 7-param, measured "
          "floors)")
    arch["loading"] = {}
    for saod in (0.003, saod_cal, 0.0083, 0.010):
        scale = saod / saod_cal
        bg_s = GenBackground(bext_w, alpha_cal * scale, CAL_QUIET,
                             ri_lm65, wt65, T65)
        s_ts, nsets_s, _ = nuisance_sets(bg_s)
        t_s = t_w
        sig_s = np.r_[np.maximum(PHI[:I874] * s_ts[:I874], ALPHA_FLOOR),
                      sig_win]
        sa, _, _ = marginal(idx_all, nsets_s['full 7-param'], sig_s, t_s)
        sa5, _, _ = marginal(idx_all, nsets_s['A2+rs (5-param)'], sig_s,
                             t_s)
        sa_t, _, _ = marginal(idx_tri, nsets_s['full 7-param'], sig_s, t_s)
        sa5_t, _, _ = marginal(idx_tri, nsets_s['A2+rs (5-param)'], sig_s,
                               t_s)
        # round 33: band01 (triplet, full 7-param) at the same loading
        bg_sb = GenBackground(bext_b, alpha_cal * scale, CAL_QUIET,
                              ri_lm65, wt65, T65)
        s_tsb, nsets_sb, _ = nuisance_sets(bg_sb)
        sig_sb = np.r_[np.maximum(PHI[:I874] * s_tsb[:I874], ALPHA_FLOOR),
                       sig_b01]
        sa_b, _, _ = marginal(idx_tri_b, nsets_sb['full 7-param'], sig_sb, t_b)
        print(f"  SAOD525 = {saod:.4f} (x{scale:4.2f}): 5-param "
              f"{3 * sa5:.3f} Tg, full 7-param {3 * sa:.3f} Tg (six "
              f"channels); triplet: 5-param {3 * sa5_t:.3f}, full 7-param "
              f"{3 * sa_t:.3f} Tg")
        print(f"    band01 triplet full 7-param: {3 * sa_b:.3f} Tg")
        arch["loading"][f"{saod:.4f}"] = dict(mmin_5p=3 * sa5,
                                              mmin_7p=3 * sa,
                                              mmin_5p_triplet=3 * sa5_t,
                                              mmin_7p_triplet=3 * sa_t,
                                              mmin_7p_band01_triplet=3 * sa_b)

    out = _PARENT / "outputs" / "calibrated_background_problem.json"
    with open(out, "w") as f:
        json.dump(arch, f, indent=1, default=float)
    print(f"\narchived -> {out}")


def franta_window():
    """PART F (invoked as `... calibrated_background_problem.py franta`):
    silica-optics robustness of the window mode -- the Franta dataset in
    place of Kitamura+Popova, calibrated quiet background, measured
    floors.  Archived into the same JSON under 'franta_window'."""
    silica_ri_franta = SilicaRefractiveIndex("data/optics/SiO2_Franta2016.yml")
    ri_lm65, wt65, T65 = member_ri("LM65T223")
    alpha_cal = glossac_2025N_quiet_profile()
    prev = json.load(open(_PARENT / "outputs" /
                          "calibrated_background_thresholds.json"))
    win_c = np.array(prev["window_elements_um"])
    sig_win = np.array(prev["window_floors_m1"])
    wl_w = np.r_[WL_NM[:I874], win_c * 1000.0]
    w_w = np.r_[np.zeros(I874), np.full(len(win_c), 0.10)]
    bext_w = make_bext(wl_w, w_w)
    bg_w = GenBackground(bext_w, alpha_cal, CAL_QUIET, ri_lm65, wt65, T65)
    s_tw, nsets_w, _ = nuisance_sets(bg_w)
    t_fr = bext_w(create_silica_sai_layer(
        ALT, 1.0, refractive_index=silica_ri_franta,
        rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL))
    sig_w = np.r_[np.maximum(PHI[:I874] * s_tw[:I874], ALPHA_FLOOR),
                  sig_win]
    idx_all = np.arange(len(wl_w))
    res = {}
    idx_tri = np.r_[[0, 1, 5], np.arange(I874, len(wl_w))]
    for lab in ('A2+rs (5-param)', 'full 7-param'):
        sa, tperp, R2 = marginal(idx_all, nsets_w[lab], sig_w, t_fr)
        print(f"  Franta window {lab:18s}: M_min = {3 * sa:.4f} Tg  "
              f"(1-R2 = {1 - R2:.2e})")
        res[lab] = dict(mmin_tg=3 * sa, one_minus_R2=1 - R2)
        # round 31: the heritage-triplet roster
        sa, tperp, R2 = marginal(idx_tri, nsets_w[lab], sig_w, t_fr)
        print(f"  Franta triplet+window {lab:18s}: M_min = {3 * sa:.4f} Tg  "
              f"(1-R2 = {1 - R2:.2e})")
        res[f"triplet {lab}"] = dict(mmin_tg=3 * sa, one_minus_R2=1 - R2)
    # vis/NIR roster dependence in window mode (Kitamura silica)
    t_kt = bext_w(create_silica_sai_layer(
        ALT, 1.0, refractive_index=SilicaRefractiveIndex(
            "data/optics/SiO2_KitamuraPopova.yml"),
        rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL))
    rosters = {'wrana3 + window': np.r_[np.arange(3),
                                        np.arange(I874, len(wl_w))],
               'extended + window': idx_all}
    for rlab, idx in rosters.items():
        for lab in ('A2+rs (5-param)', 'full 7-param'):
            sa, tperp, R2 = marginal(idx, nsets_w[lab], sig_w, t_kt)
            tagv = f"{3 * sa:.4f}" if np.isfinite(sa) else "---"
            print(f"  {rlab:18s} {lab:18s}: M_min = {tagv} Tg")
            res[f"{rlab} {lab}"] = dict(
                mmin_tg=3 * sa if np.isfinite(sa) else None)
    out = _PARENT / "outputs" / "calibrated_background_problem.json"
    arch = json.load(open(out)) if out.exists() else {}
    arch["franta_window"] = res
    with open(out, "w") as f:
        json.dump(arch, f, indent=1, default=float)
    print(f"archived -> {out}")


if __name__ == "__main__":
    if "franta" in sys.argv[1:]:
        franta_window()
    else:
        main()
