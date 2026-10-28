"""farir_adopted.py -- round 75 (paper 1; public port, referee round 6 item 2, left open in round 74): the
far-infrared columns of Table 8 with the DESIGN BAND under the adopted measurement model (Table 2).

The design band (47 elements, slant rows, full gas state, the covariance of Table 2) and the heritage
triplet are the joint retrieval of reproduce/adopted_model_engine.py with the material's
column in place of silica's (orientation-averaged optics, convention b, as Table 8).  The far-infrared
band of each material (a 5-um-wide band of 0.1-um elements; silica 18-23, calcite 26-31, dolomite
22.5-27.5, alumina 18.25-23.25 um) enters as per-shell extinction rows at 20 km with the calculated
per-element gas-removal floors of round 37 (farir_band_floors_w0p1.json, bounded below by the
SNR-2000 floor), diagonal, times 1, 3 and 10, exactly as in Table 8; those rows carry the material's
and the six sulfate columns and no gas columns (their floors already contain the gas removal; no
measured scatter exists beyond 13.3 um, so no discrepancy term is assigned to them).  The far-IR-only
column (triplet + far-IR band, no design band) does not involve the design-band terms and is
unchanged from round 71; it is re-evaluated here on the same rows as the check.

Run from the repo root (about 1 h: the four far-IR backgrounds in parallel):
  python reproduce/farir_adopted.py
Writes outputs/farir_adopted.json (+ _bg.npz cache) and the far-IR columns of
tables/materials.tex are updated by hand from it.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))

OUT = ROOT / "outputs/farir_adopted.json"
CACHE = ROOT / "outputs/farir_adopted_bg.npz"
MATS = ("silica", "calcite", "dolomite", "alumina")
FLOOR_SCAN = (1.0, 3.0, 10.0)


def log(*a):
    print(*a, flush=True)


def build_far(mat):
    """far-IR background (exts, ders at every altitude) and the material's far-IR column; runs in a worker."""
    t0 = time.time()
    import materials_calibrated_thresholds as MCT
    from materials_calibrated_thresholds import FarSet, far_background, farir_spec, iz
    from materials_farir_uniaxial import layers_for
    import calibrated_background_thresholds as cbt
    spec = farir_spec()
    c_far, f_far, note = spec[mat]
    fs = FarSet(f"{mat} far-IR", c_far, MCT.FAR_WIDTH_UM, f_far)
    comps, member = MCT.BRACKETS["quiet"]
    exts, ders = far_background(fs, comps, member)
    lay = layers_for(mat)["b"]
    t_far = fs.prof(lay)[iz]
    return dict(mat=mat, centers=np.asarray(c_far), floors=np.asarray(f_far), note=note,
                cols=np.array([exts[0][iz], exts[1][iz], ders[0][0][iz], ders[0][1][iz], ders[1][0][iz], ders[1][1][iz]]),
                t_far=t_far, seconds=time.time() - t0)


def main():
    t0 = time.time()
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True); far = {k: d[k].item() for k in d.files}
    else:
        with mp.get_context("spawn").Pool(4) as pool:
            res = pool.map(build_far, MATS)
        far = {r["mat"]: r for r in res}
        np.savez_compressed(CACHE, **{k: np.array(v, dtype=object) for k, v in far.items()})
        log("far-IR backgrounds built: " + ", ".join(f"{k} {v['seconds']:.0f} s" for k, v in far.items()))
    import adopted_model_engine as E
    from materials_farir_uniaxial import layers_for
    from materials_uniaxial import RayTables, ConventionRI, MAT_CLS, MAT_FACTORY
    C, cbt = E.C, E.cbt
    arch71 = json.load(open(ROOT / "outputs/materials_farir_uniaxial.json"))
    out = dict(script="reproduce/farir_adopted.py", round=75, floor_scan=list(FLOOR_SCAN), materials={})
    for mat in MATS:
        if mat == "silica":
            lay = C.sil_layer
        else:
            lay = MAT_FACTORY[mat](cbt.ALT, 1.0, refractive_index=ConventionRI(RayTables(MAT_CLS[mat]), "b"))
        prof_c = lay.extinction_profile_m1(C.lam_coarse * 1e-6); prof_o = lay.extinction_profile_m1(C.lam_outer * 1e-6); prof_t = lay.extinction_profile_m1(C.WL_TRIPLET_NM * 1e-9)
        # per-shell convention for every row: the material's 20-km shell spectrum per Tg placed in the tangent shell alone
        # (slant rows see it through the one-shell path, the triplet and far-infrared rows per shell), so no slant-to-shell factor
        st = E.shell_template(20e3, prof=(prof_c[None], prof_o[None], prof_t[None]))
        tm = E.TangentModel(20e3, **st)
        tm_p = E.TangentModel(20e3, template=prof_c, template_outer=prof_o, template_trip=prof_t)   # prescribed profile (Table 8 convention)
        f = far[mat]
        n_far = len(f["centers"])
        K_far = np.zeros((n_far, C.N_STATE)); K_far[:, C.I_SIL] = f["t_far"]; K_far[:, C.I_SIL + 1:C.I_SIL + 7] = f["cols"].T
        r = dict(rho_prescribed=tm_p.rho, far_centers_um=f["centers"].tolist(), far_floors_m1=f["floors"].tolist(), far_note=f["note"],
                 peak_far_um=float(f["centers"][int(np.argmax(f["t_far"]))]), peak_far_ext_m1_per_tg=float(f["t_far"].max()),
                 design_band_adopted_shell=3 * E.joint_sigma(tm), design_band_adopted_prescribed_x_rho=3 * E.joint_sigma(tm_p) * tm_p.rho,
                 design_band_scenario=3 * E.two_stage_sigma(tm_p, E.s_adopt_od),
                 band_plus_farir_adopted={}, farir_only={}, band_plus_farir={}, farir_only={})
        for mult in FLOOR_SCAN:
            S_far = np.diag((f["floors"] * mult) ** 2)
            r["band_plus_farir_adopted"][str(mult)] = 3 * E.joint_sigma(tm, extra_rows=(K_far, S_far))
            # far-IR only: triplet + far-IR rows (per-shell, Fisher marginalisation on the aerosol columns; no design band)
            cols_t = np.vstack([tm.K_tri[:, C.I_SIL:C.I_SIL + 7], K_far[:, C.I_SIL:C.I_SIL + 7]])
            sig = np.r_[np.sqrt(np.diag(tm.S_tri)), f["floors"] * mult]
            sa, _, _ = cbt.marginal(np.arange(len(sig)), [cols_t[:, j] for j in range(1, 7)], sig, cols_t[:, 0])
            r["farir_only"][str(mult)] = 3 * float(sa)
            a71 = arch71["materials"][mat]["conventions"]["b"]["floor_scan"][str(mult)]
            r["band_plus_farir"][str(mult)] = a71["band_plus_farir"]; r["farir_only"][str(mult)] = a71["triplet_plus_farir"]
        out["materials"][mat] = r
        log(f"{mat:>8}: design band adopted (shell convention) {r['design_band_adopted_shell']:.4f} (prescribed x rho {r['design_band_adopted_prescribed_x_rho']:.4f}; scenario {r['design_band_scenario']:.4f}); band + far-IR adopted x1/x3/x10 "
            + "/".join(f"{r['band_plus_farir_adopted'][str(m)]:.4f}" for m in FLOOR_SCAN) + "; far-IR only "
            + "/".join(f"{r['farir_only'][str(m)]:.4f}" for m in FLOOR_SCAN) + f"; round 71 band+far {r['band_plus_farir']}, far-only {r['farir_only']}; {time.time()-t0:.0f} s")
    out["runtime_s"] = time.time() - t0
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    log("archived ->", OUT)


if __name__ == "__main__":
    main()
