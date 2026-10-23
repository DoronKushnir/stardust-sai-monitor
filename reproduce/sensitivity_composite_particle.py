# Ported from the working repository's scripts/sensitivity_composite_particle_round63.py (paper 1 rounds 62-63); paths follow the repository layout.
"""
sensitivity_composite_particle.py -- Doron (paper-1 round 63, on
Mark's coating question): "a different coating is being discussed, see the
Spector and Kislev papers": the Stardust composite particle (Spector et al.
2026, arXiv:2606.15955) is a calcium-carbonate core inside a spherical
amorphous-silica shell (plus a surface treatment), the silica sphere being
the other design.  What does the 7.8-13 um band see of a calcite-core /
silica-shell particle?

Same machinery as sensitivity_sulfate_coating.py (BHCOAT coated
sphere, the paper-1 element grids, measured floors, calibrated background,
full seven-parameter fit, z = 20 km), with the CORE = calcite (ray-mean
Querry optics, rho = 2710 kg m^-3) and the SHELL = amorphous silica
(Kitamura+Popova, rho = 2200), the OUTER radius lognormal at the measured
Segev-2026 dispersion (r_med = 268 nm, sigma = 1.31).  The silica volume
fraction F_SIL of the particle is scanned from 0 (bare calcite sphere) to
1 (the paper's silica sphere).  The injected spectrum is the composite
particle's own, and the threshold is quoted per teragram of TOTAL injected
mass.  N0(z) follows from the mass closure of the silica layer rescaled by
the ratio of mean particle masses (same outer PSD).

Run from the repo root:  python reproduce/sensitivity_composite_particle.py
Output: outputs/composite_particle.json (+ stdout)
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
    ALT, iz, RMED_SIL_NM, SIGMA_SIL, CAL_QUIET, member_ri)
from design_sensitivity_calibrated import ElementSet  # noqa: E402
from sensitivity_sulfate_coating import bhcoat, R_GRID  # noqa: E402
from saimon.materials import SilicaRefractiveIndex, CalciteRefractiveIndex  # noqa: E402
from saimon.sai import (create_silica_sai_layer, SILICA_DENSITY_KG_M3,  # noqa: E402
                       CALCITE_DENSITY_KG_M3)
from saimon.psd import LognormalPSD  # noqa: E402

F_SIL = (0.0, 0.25, 0.5, 0.75, 1.0)      # silica volume fraction of the particle
OUT = _ROOT / "outputs" / "composite_particle.json"


class CompositeLayer:
    """Calcite core, silica shell, outer radius lognormal (RMED_SIL_NM,
    SIGMA_SIL); n0(z) per Tg of TOTAL mass."""

    def __init__(self, base_layer, ri_core, ri_shell, f_sil):
        self.f = float(f_sil)
        rho_mean = self.f * SILICA_DENSITY_KG_M3 + (1 - self.f) * CALCITE_DENSITY_KG_M3
        # same outer PSD: particles per Tg scale as 1/mean particle mass
        self.n0 = np.asarray(base_layer.modes[0].n0_m3_profile, float) * SILICA_DENSITY_KG_M3 / rho_mean
        self.ri_core, self.ri_shell = ri_core, ri_shell
        self.psd = LognormalPSD(RMED_SIL_NM * 1e-9, SIGMA_SIL, 1.0)

    def cross_sections_m2(self, wl_m):
        wl_m = np.atleast_1d(np.asarray(wl_m, float))
        r_out = R_GRID
        r_core = r_out * (1 - self.f) ** (1 / 3)
        dn = self.psd.dn_dr(r_out)
        mc = np.asarray(self.ri_core(wl_m), complex)
        ms = np.asarray(self.ri_shell(wl_m), complex)
        out = np.empty(len(wl_m))
        for j, lam in enumerate(wl_m):
            y = 2 * np.pi * r_out / lam
            if self.f >= 1.0:
                qe, _ = bhcoat(y * (1 - 1e-6), y, complex(ms[j]), complex(ms[j]))
            elif self.f <= 0.0:
                qe, _ = bhcoat(y * (1 - 1e-6), y, complex(mc[j]), complex(mc[j]))
            else:
                x = 2 * np.pi * r_core / lam
                qe, _ = bhcoat(x, y, complex(mc[j]), complex(ms[j]))
            out[j] = np.trapezoid(qe * np.pi * r_out ** 2 * dn, r_out)
        return out

    def extinction_profile_m1(self, wl_m):
        return self.n0[:, None] * self.cross_sections_m2(wl_m)[None, :]


def main():
    thr = json.load(open(_ROOT / "outputs" / "calibrated_background_thresholds.json"))
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
    ri_sil = SilicaRefractiveIndex("data/optics/SiO2_KitamuraPopova.yml")
    ri_cal = CalciteRefractiveIndex()
    base = create_silica_sai_layer(ALT, 1.0, refractive_index=ri_sil,
                                   rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL)
    layers = {f: CompositeLayer(base, ri_cal, ri_sil, f) for f in F_SIL}
    arch = {"f_sil": list(F_SIL), "shell_nm_at_rmed": {}, "silica_mass_fraction": {},
            "peak_um_8_9": {}, "ext_8p80_rel_to_silica": {}, "ext_11p4_rel_to_silica": {}, "sets": {}}
    lam = np.arange(8.3, 9.31, 0.01) * 1e-6
    e_sil = base.extinction_profile_m1(np.array([8.8e-6, 11.4e-6]))[iz, :]
    for f in F_SIL:
        arch["shell_nm_at_rmed"][str(f)] = float(RMED_SIL_NM * (1 - (1 - f) ** (1 / 3)))
        m_sil = f * SILICA_DENSITY_KG_M3
        arch["silica_mass_fraction"][str(f)] = float(m_sil / (m_sil + (1 - f) * CALCITE_DENSITY_KG_M3))
        e = layers[f].extinction_profile_m1(lam)[iz, :]
        arch["peak_um_8_9"][str(f)] = float(lam[np.argmax(e)] * 1e6)
        e2 = layers[f].extinction_profile_m1(np.array([8.8e-6, 11.4e-6]))[iz, :]
        arch["ext_8p80_rel_to_silica"][str(f)] = float(e2[0] / e_sil[0])
        arch["ext_11p4_rel_to_silica"][str(f)] = float(e2[1] / e_sil[1])
    for k in ("shell_nm_at_rmed", "silica_mass_fraction", "peak_um_8_9",
              "ext_8p80_rel_to_silica", "ext_11p4_rel_to_silica"):
        print(f"{k:26s}", {kk: round(v, 3) for kk, v in arch[k].items()})
    for es in sets:
        exts, ders = es.background(CAL_QUIET, ri65, wt65, T65)
        res = {}
        for f in F_SIL:
            m, tperp, r2 = es.mmin(exts, ders, es.prof(layers[f]))
            res[str(f)] = {"mmin_tg_total": float(m), "tperp": float(tperp), "one_minus_R2": float(r2)}
        m1 = res["1.0"]["mmin_tg_total"]
        assert abs(m1 - ref[es.name]) < 2e-3, (m1, ref[es.name])   # silica endpoint reproduces the archive
        print(f"\n=== {es.name} (silica endpoint {m1:.4f} Tg, archived {ref[es.name]:.4f}) ===")
        for f in F_SIL:
            r = res[str(f)]
            print(f"  f_sil = {f:<5}: M_min = {r['mmin_tg_total']:.4f} Tg (total mass), "
                  f"||t_perp|| {r['tperp']:.1f}, 1-R2 {r['one_minus_R2']:.3f}")
        arch["sets"][es.name] = res
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(arch, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
