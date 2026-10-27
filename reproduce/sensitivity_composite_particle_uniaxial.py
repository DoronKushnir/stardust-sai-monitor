"""sensitivity_composite_particle_uniaxial.py -- round 71 (referee round 5, item 9
follow-up): the composite-particle test of sensitivity_composite_particle.py
(calcium-carbonate core inside an amorphous-silica shell, BHCOAT coated sphere, outer
radius at the reference dispersion, threshold per teragram of total particle mass)
repeated with the calcite core entered through the orientation-averaged optics of
randomly oriented uniaxial particles -- convention (b) of materials_uniaxial.py,
the Rayleigh-exact Clausius-Mossotti average of the ordinary and extraordinary rays --
which the paper adopts since round 70.  Convention (a), the ray-arithmetic mean of the
round-63 run, is evaluated alongside and must reproduce the round-63 archive; nothing
else changes (same CompositeLayer, element sets, floors, calibrated background, seven
free parameters, z = 20 km).

Run from the repo root:  python reproduce/sensitivity_composite_particle_uniaxial.py
Output: outputs/composite_particle_uniaxial.json (+ stdout)
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
from sensitivity_composite_particle import CompositeLayer, F_SIL  # noqa: E402
from materials_uniaxial import RayTables, ConventionRI  # noqa: E402
from saimon.materials import SilicaRefractiveIndex, CalciteRefractiveIndex  # noqa: E402
from saimon.sai import create_silica_sai_layer  # noqa: E402

OUT = _ROOT / "outputs" / "composite_particle_uniaxial.json"
ARCH63 = _ROOT / "outputs" / "composite_particle.json"
CONVS = ("a", "b")


def main():
    thr = json.load(open(_ROOT / "outputs" / "calibrated_background_thresholds.json"))
    sets = [ElementSet("window 7.8-9.3 @0.1", thr["window_elements_um"], 0.10, thr["window_floors_m1"]),
            ElementSet("band 8-13 @0.1", thr["band01_elements_um"], 0.10, thr["band01_floors_m1"])]
    cq = thr["results"]["CALIBRATED quiet (LM65T223, 2-comp)"]
    ref = {"window 7.8-9.3 @0.1": cq["window 7.8-9.3 @0.1: measured floors"]["full 7-param"]["triplet + window"]["mmin_tg"],
           "band 8-13 @0.1": cq["band 8-13 @0.1: measured floors"]["full 7-param"]["triplet + band01"]["mmin_tg"]}
    ri65, wt65, T65 = member_ri("LM65T223")
    ri_sil = SilicaRefractiveIndex("data/optics/SiO2_KitamuraPopova.yml")
    tabs = RayTables(CalciteRefractiveIndex)
    ri_cal = {"a": ConventionRI(tabs, "a"), "b": ConventionRI(tabs, "b")}
    # the library's ray mean and convention (a) coincide (round-70 check); verify on the band grid
    lib = CalciteRefractiveIndex()
    wl = np.arange(7.8, 13.3, 0.01) * 1e-6
    d_lib = float(np.max(np.abs(np.asarray(lib(wl), complex) - ri_cal["a"](wl))))
    base = create_silica_sai_layer(ALT, 1.0, refractive_index=ri_sil, rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL)
    arch63 = json.load(open(ARCH63)) if ARCH63.exists() else None
    arch = {"f_sil": list(F_SIL), "library_ray_mean_vs_conv_a_max_abs_diff": d_lib, "conventions": {}}
    lam = np.arange(8.3, 9.31, 0.01) * 1e-6
    e_sil = base.extinction_profile_m1(np.array([8.8e-6, 11.4e-6]))[iz, :]
    bgs = {es.name: es.background(CAL_QUIET, ri65, wt65, T65) for es in sets}
    for conv in CONVS:
        layers = {f: CompositeLayer(base, ri_cal[conv], ri_sil, f) for f in F_SIL}
        a = {"peak_um_8_9": {}, "ext_8p80_rel_to_silica": {}, "ext_11p4_rel_to_silica": {}, "sets": {}}
        for f in F_SIL:
            e = layers[f].extinction_profile_m1(lam)[iz, :]
            a["peak_um_8_9"][str(f)] = float(lam[np.argmax(e)] * 1e6)
            e2 = layers[f].extinction_profile_m1(np.array([8.8e-6, 11.4e-6]))[iz, :]
            a["ext_8p80_rel_to_silica"][str(f)] = float(e2[0] / e_sil[0])
            a["ext_11p4_rel_to_silica"][str(f)] = float(e2[1] / e_sil[1])
        print(f"\n##### convention ({conv}) #####")
        for k in ("peak_um_8_9", "ext_8p80_rel_to_silica", "ext_11p4_rel_to_silica"):
            print(f"{k:26s}", {kk: round(v, 4) for kk, v in a[k].items()})
        for es in sets:
            exts, ders = bgs[es.name]
            res = {}
            for f in F_SIL:
                m, tperp, r2 = es.mmin(exts, ders, es.prof(layers[f]))
                res[str(f)] = {"mmin_tg_total": float(m), "tperp": float(tperp), "one_minus_R2": float(r2)}
            m1 = res["1.0"]["mmin_tg_total"]
            assert abs(m1 - ref[es.name]) < 2e-3, (m1, ref[es.name])   # silica endpoint reproduces the archive
            print(f"=== {es.name} (silica endpoint {m1:.4f} Tg, archived {ref[es.name]:.4f}) ===")
            for f in F_SIL:
                r = res[str(f)]
                extra = ""
                if arch63 is not None:
                    r["round63_archive_a"] = arch63["sets"][es.name][str(f)]["mmin_tg_total"]
                    extra = f"  (round-63 archive {r['round63_archive_a']:.4f})"
                print(f"  f_sil = {f:<5}: M_min = {r['mmin_tg_total']:.4f} Tg (total mass), ||t_perp|| {r['tperp']:.1f}, 1-R2 {r['one_minus_R2']:.3f}{extra}")
            a["sets"][es.name] = res
        arch["conventions"][conv] = a
    if arch63 is not None:
        d = max(abs(arch["conventions"]["a"]["sets"][n][str(f)]["mmin_tg_total"] / arch63["sets"][n][str(f)]["mmin_tg_total"] - 1)
                for n in arch63["sets"] for f in F_SIL)
        arch["conv_a_reproduces_round63_max_rel_diff"] = float(d)
        print(f"\nconvention (a) vs round-63 archive: max rel diff {d:.1e}")
    arch["ratio_b_over_a"] = {n: {str(f): arch["conventions"]["b"]["sets"][n][str(f)]["mmin_tg_total"]
                                  / arch["conventions"]["a"]["sets"][n][str(f)]["mmin_tg_total"] for f in F_SIL}
                              for n in arch["conventions"]["a"]["sets"]}
    print("ratio (b)/(a):", {n: {k: round(v, 3) for k, v in r.items()} for n, r in arch["ratio_b_over_a"].items()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(arch, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
