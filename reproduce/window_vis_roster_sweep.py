"""
window_vis_roster_sweep.py -- how much of the visible/NIR roster does the
7.8-9.3 um window threshold actually need?  (Paper 1 round 29, Doron:
"In Sect. 4 a single optical constraint (GloSSAC 525) is sufficient.  Then
why use all the optical bands in Sect. 3?")

Same machinery, background and floors as PART 3 of
scripts/calibrated_background_thresholds.py (imported; the window element
centers and measured per-element floors are read from its archive so the
numbers are bit-identical), but the visible/NIR roster is swept:

  * the full six-channel roster of Sect. 3 (448, 756, 869, 1021, 1250,
    1544 nm);
  * the Wrana heritage triplet (448, 756, 1544 nm);
  * each channel alone, plus a 521-nm channel alone (SAGE III/ISS aerosol
    channel; observed L2 fractional uncertainty 5.7 % at 20 km,
    Table tab:combined) -- the closest thing to "one optical constraint at
    525 nm";
  * an idealized single 521-nm channel at the SNR=2000 transmission floor
    only (what a perfect single anchor would buy);
  * the window alone (no visible/NIR channel).

Calibrated undisturbed background (LM65T223, two-component), z = 20 km,
measured window floors; nuisance conventions: A2+rs (5-param) and the
paper's full 7-param fit.  Post-Hunga-Tonga calibration bracket printed
for the full 7-param rows.

Run from the repo root:  python scripts/window_vis_roster_sweep.py
Output: printed table + outputs/window_vis_roster_sweep.json
"""

from __future__ import annotations

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
    ALT, iz, ALPHA_FLOOR, CAL_QUIET, CAL_POSTHT, RMED_SIL_NM, SIGMA_SIL,
    TwoComponentBackground, glossac_2025N_quiet_profile, marginal,
    member_ri)
from saimon.materials import SilicaRefractiveIndex  # noqa: E402
from saimon.sai import create_silica_sai_layer  # noqa: E402

# visible/NIR channels [nm] and observed SAGE III/ISS L2 fractional
# extinction uncertainties at 20 km (Table tab:combined; 1250 nm inherits
# the SAGE-class extension value used in calibrated_background_thresholds)
VIS_NM = np.array([448., 521., 756., 869., 1021., 1250., 1544.])
VIS_PHI = np.array([0.040, 0.057, 0.032, 0.040, 0.045, 0.063, 0.088])
IDX = {int(w): i for i, w in enumerate(VIS_NM)}

ROSTERS = {
    'full roster (448,756,869,1021,1250,1544)': [448, 756, 869, 1021, 1250,
                                                 1544],
    'heritage triplet (448,756,1544)': [448, 756, 1544],
    '448 only': [448],
    '521 only': [521],
    '756 only': [756],
    '869 only': [869],
    '1021 only': [1021],
    '1250 only': [1250],
    '1544 only': [1544],
    '448 + 1544': [448, 1544],
    'window alone': [],
}


def main():
    arch_in = json.load(open(_ROOT / "outputs" /
                             "calibrated_background_thresholds.json"))
    win_c = np.array(arch_in["window_elements_um"])
    sig_win = np.array(arch_in["window_floors_m1"])
    n_vis = len(VIS_NM)
    n_el = len(win_c)
    sub = np.linspace(-0.05, 0.05, 41)
    wl_all = np.r_[VIS_NM, win_c * 1000.0]

    def bext(layer):
        out = layer.extinction_profile_m1(wl_all * 1e-9)[iz, :].copy()
        for j, c in enumerate(win_c):
            out[n_vis + j] = layer.extinction_profile_m1(
                (c + sub) * 1e-6)[iz, :].mean()
        return out

    class BandBackground(TwoComponentBackground):
        def component_ext(self):
            return [bext(self._col(r, s, f)) for r, s, f in self.comps]

        def psd_derivs(self, i):
            r0, s0, f = self.comps[i]
            dr, ds = self.fd * r0, self.fd * s0
            er = (bext(self._col(r0 + dr, s0, f))
                  - bext(self._col(r0 - dr, s0, f))) / (2 * dr)
            es = (bext(self._col(r0, s0 + ds, f))
                  - bext(self._col(r0, s0 - ds, f))) / (2 * ds)
            return er, es

    silica_ri = SilicaRefractiveIndex("data/optics/SiO2_KitamuraPopova.yml")
    sai = create_silica_sai_layer(ALT, 1.0, refractive_index=silica_ri,
                                  rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL)
    t_sil = bext(sai)
    alpha_cal = glossac_2025N_quiet_profile()

    ri65, wt65, T65 = member_ri("LM65T223")
    ri72, wt72, T72 = member_ri("LM72T213")
    models = {
        'CALIBRATED undisturbed (LM65T223)': (CAL_QUIET, ri65, wt65, T65),
        'bracket: post-HT calib (LM72T213)': (CAL_POSTHT, ri72, wt72, T72),
    }
    mir_idx = np.arange(n_vis, n_vis + n_el)

    out = {"window_elements_um": list(win_c), "window_floors_m1":
           list(sig_win), "vis_nm": list(VIS_NM), "vis_phi": list(VIS_PHI),
           "results": {}}
    for label, (comps, ri_m, wt, T) in models.items():
        print(f"\n=== {label}: 7.8-9.3 um window @0.1 um, measured floors, "
              f"z = 20 km ===")
        bg = BandBackground(alpha_cal, comps, ri_m, wt, T)
        exts = bg.component_ext()
        s_total = np.sum(exts, axis=0)
        jr_f, js_f = bg.psd_derivs(0)
        jr_c, js_c = bg.psd_derivs(1)
        nsets = {
            'A2+rs (5-param)': [exts[0], exts[1], jr_f, js_f],
            'full 7-param': [exts[0], exts[1], jr_f, js_f, jr_c, js_c],
        }
        sig = np.r_[np.maximum(VIS_PHI * s_total[:n_vis], ALPHA_FLOOR),
                    sig_win]
        # idealized 521-nm anchor: SNR-floor error only
        sig_ideal = sig.copy()
        sig_ideal[IDX[521]] = ALPHA_FLOOR
        print(f"  vis/NIR errors [m^-1]: " + ", ".join(
            f"{int(w)}:{s:.2e}" for w, s in zip(VIS_NM, sig[:n_vis])))
        res = {}
        hdr = f"    {'roster':44s}" + "".join(f"{k:>18s}" for k in nsets)
        print(hdr)
        rows = list(ROSTERS.items()) + [('521 only, SNR-floor error',
                                         [521])]
        for r_lab, chans in rows:
            idx = np.r_[[IDX[c] for c in chans], mir_idx].astype(int)
            s_use = sig_ideal if 'SNR-floor' in r_lab else sig
            line = f"    {r_lab:44s}"
            row = {}
            for ns_lab, cols in nsets.items():
                sa, tperp, R2 = marginal(idx, cols, s_use, t_sil)
                mmin = 3 * sa
                row[ns_lab] = dict(mmin_tg=mmin if np.isfinite(mmin)
                                   else None,
                                   one_minus_R2=(1 - R2) if np.isfinite(R2)
                                   else None)
                line += (f"{mmin:14.3f} Tg " if np.isfinite(mmin)
                         else f"{'degenerate':>17s} ")
            print(line)
            res[r_lab] = row
        out["results"][label] = res

    dst = _ROOT / "outputs" / "window_vis_roster_sweep.json"
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\narchived -> {dst}")


if __name__ == "__main__":
    main()
