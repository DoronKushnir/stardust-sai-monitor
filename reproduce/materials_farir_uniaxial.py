"""materials_farir_uniaxial.py -- round 71 (referee round 5, item 9 follow-up):
the far-infrared columns of Table tab:materials and the red-extension test (PART 2 and
PART 3 of materials_calibrated_thresholds.py) recomputed with the orientation-averaged
optics of the uniaxial materials, convention (b) of materials_uniaxial.py (the
Rayleigh-exact Clausius-Mossotti average of the ordinary and extraordinary rays), which
the paper adopts for the design band since round 70.  Convention (a), the ray-arithmetic
mean, is evaluated on the same backgrounds to reproduce the archived values of
outputs/materials_calibrated/results.json, and convention (c), the cross-section average
(2 Q_o + Q_e)/3, as the beyond-Rayleigh bracket.  Silica is isotropic and serves as the
reproduction check only.

Everything else is bit-identical to materials_calibrated_thresholds.py: the far-IR
element sets and their per-element OE floors (farir_band_floors_w0p1.json), the cached
design-band engines, the calibrated two-component background (undisturbed member), the
floor scan {1, 3, 10, 30}, the marginalization.  The background (exts / ders on the
far-IR elements) does not depend on the injected material, so it is built once per band
and the material enters only through its per-Tg extinction vector t.

Run from the repo root (one process per band; ~15 min each for the background build):
  python reproduce/materials_farir_uniaxial.py --mat calcite
  ... --mat silica | dolomite | alumina | red
  python reproduce/materials_farir_uniaxial.py --merge
Per-band archives: outputs/materials_farir_uniaxial_<mat>.json;
merged archive:    outputs/materials_farir_uniaxial.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import materials_calibrated_thresholds as MCT  # noqa: E402
from materials_calibrated_thresholds import (  # noqa: E402
    ALT, iz, ALPHA_FLOOR, RMED_SIL_NM, SIGMA_SIL, VIS_PHI, N_VIS, FAR_WIDTH_UM, FLOOR_SCAN,
    ElementSet, Engine, BRACKETS, FarSet, far_background, farir_spec, mmin_general,
    glossac_2025N_quiet_profile, SilicaRefractiveIndex, create_silica_sai_layer)
from materials_uniaxial import RayTables, ConventionRI, MAT_CLS, MAT_FACTORY  # noqa: E402

OUT_DIR = _ROOT / "outputs"
MERGED = OUT_DIR / "materials_farir_uniaxial.json"
ARCH_A = _ROOT / "outputs" / "materials_calibrated" / "results.json"
MATS = ("silica", "calcite", "dolomite", "alumina")
CONVS = ("a", "b", "c")


def layers_for(mat):
    """{conv: layer or (layer_o, layer_e)} -- convention (c) is evaluated at the layer
    level as (2 t_o + t_e)/3, exact because extinction is linear in the ensemble."""
    if mat == "silica":
        ri = SilicaRefractiveIndex(str(_ROOT / "data/optics/SiO2_KitamuraPopova.yml"))
        lay = create_silica_sai_layer(ALT, 1.0, refractive_index=ri, rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL)
        return {"a": lay, "b": lay, "c": lay}
    tabs = RayTables(MAT_CLS[mat])
    out = {c: MAT_FACTORY[mat](ALT, 1.0, refractive_index=ConventionRI(tabs, c)) for c in ("a", "b")}
    out["c"] = (MAT_FACTORY[mat](ALT, 1.0, refractive_index=ConventionRI(tabs, "o")),
                MAT_FACTORY[mat](ALT, 1.0, refractive_index=ConventionRI(tabs, "e")))
    return out


def t_vec(es, lay):
    if isinstance(lay, tuple):
        return (2 * es.prof(lay[0])[iz] + es.prof(lay[1])[iz]) / 3.0
    return es.prof(lay)[iz]


def setup():
    thr = json.load(open(MCT.THR_JSON))
    es_band = ElementSet("band 8-13 @0.1", thr["band01_elements_um"], 0.10, thr["band01_floors_m1"])
    alpha525 = glossac_2025N_quiet_profile()
    cache = MCT.ENGINE_DIR / "engine_band_quiet.npz"
    assert cache.exists(), cache
    comps, member = BRACKETS["quiet"]
    eng_b = Engine(es_band, comps, member, alpha525, None, cache=cache)
    return thr, es_band, eng_b, comps, member


def run_far(mat):
    t0 = time.time()
    thr, es_band, eng_b, comps, member = setup()
    c_far, f_far, note = farir_spec()[mat]
    fs = FarSet(f"{mat} far-IR", c_far, FAR_WIDTH_UM, f_far)
    print(f"[{mat}] building far-IR background ({len(c_far)} elements {c_far.min():.2f}-{c_far.max():.2f} um) ...", flush=True)
    exts_f, ders_f = far_background(fs, comps, member)
    print(f"[{mat}] background built in {time.time()-t0:.0f} s", flush=True)
    exts = [np.concatenate([eng_b.exts[i], exts_f[i]], axis=1) for i in range(2)]
    ders = [(np.concatenate([eng_b.ders[i][0], ders_f[i][0]], axis=1),
             np.concatenate([eng_b.ders[i][1], ders_f[i][1]], axis=1)) for i in range(2)]
    s_tot = sum(e[iz] for e in exts)
    sig = np.r_[np.maximum(VIS_PHI * s_tot[:N_VIS], ALPHA_FLOOR), es_band.floors, f_far]
    n_b = es_band.n
    idx_all = np.arange(len(sig))
    idx_far = np.r_[np.arange(N_VIS), np.arange(n_b, len(sig))]
    idx_band = np.arange(n_b)
    lays = layers_for(mat)
    res = dict(material=mat, far_centers_um=list(map(float, c_far)), far_floors_m1=list(map(float, f_far)),
               far_note=note, floor_scan=list(FLOOR_SCAN), conventions={})
    for conv in CONVS:
        t = np.concatenate([t_vec(es_band, lays[conv]), t_vec(fs, lays[conv])])
        m_band, _, _ = mmin_general(exts, ders, sig, t, idx_band)       # design band alone (check)
        m_all, _, r2_all = mmin_general(exts, ders, sig, t, idx_all)
        m_far, _, _ = mmin_general(exts, ders, sig, t, idx_far)
        scan = {}
        for fmul in FLOOR_SCAN:
            sig_m = np.r_[sig[:n_b], f_far * fmul]
            scan[str(fmul)] = dict(band_plus_farir=float(mmin_general(exts, ders, sig_m, t, idx_all)[0]),
                                   triplet_plus_farir=float(mmin_general(exts, ders, sig_m, t, idx_far)[0]))
        res["conventions"][conv] = dict(
            band_quiet=float(m_band), band_plus_farir_quiet=float(m_all), triplet_plus_farir_quiet=float(m_far),
            one_minus_R2_all=r2_all, floor_scan=scan,
            far_peak_wl_um=float(c_far[int(np.argmax(t[n_b:]))]), far_peak_ext_m1=float(t[n_b:].max()),
            band_peak_wl_um=float(es_band.c[int(np.argmax(t[N_VIS:n_b]))]), band_peak_ext_m1=float(t[N_VIS:n_b].max()),
            t_far_m1_per_Tg=list(map(float, t[n_b:])))
        print(f"[{mat}] ({conv}) band {m_band:.4f}; band+farIR {m_all:.4f}; triplet+farIR {m_far:.4f} Tg; "
              f"far peak {t[n_b:].max():.3e} at {c_far[int(np.argmax(t[n_b:]))]:.2f} um; "
              f"x3/x10/x30 band+far {scan['3.0']['band_plus_farir']:.4f}/{scan['10.0']['band_plus_farir']:.4f}/{scan['30.0']['band_plus_farir']:.4f}, "
              f"far-only {scan['3.0']['triplet_plus_farir']:.4f}/{scan['10.0']['triplet_plus_farir']:.4f}/{scan['30.0']['triplet_plus_farir']:.4f}", flush=True)
    # reproduction of the archived (a) values
    if ARCH_A.exists():
        a = json.load(open(ARCH_A))["materials"][mat]
        ra = res["conventions"]["a"]
        res["reproduces_round35_archive_a"] = dict(
            band_quiet_archive=a["band/quiet"], band_plus_farir_archive=a["farir"]["band_plus_farir_quiet"],
            triplet_plus_farir_archive=a["farir"]["triplet_plus_farir_quiet"],
            max_rel_diff=float(max(abs(ra["band_quiet"] / a["band/quiet"] - 1),
                                   abs(ra["band_plus_farir_quiet"] / a["farir"]["band_plus_farir_quiet"] - 1),
                                   abs(ra["triplet_plus_farir_quiet"] / a["farir"]["triplet_plus_farir_quiet"] - 1))))
        print(f"[{mat}] archive (a): band {a['band/quiet']:.4f}, band+far {a['farir']['band_plus_farir_quiet']:.4f}, "
              f"far-only {a['farir']['triplet_plus_farir_quiet']:.4f}; max rel diff {res['reproduces_round35_archive_a']['max_rel_diff']:.1e}", flush=True)
    out = OUT_DIR / f"materials_farir_uniaxial_{mat}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"[{mat}] archived -> {out} ({time.time()-t0:.0f} s)", flush=True)


def run_red():
    """PART 3: the design band extended to 13.35-13.95 um, all materials, all conventions."""
    t0 = time.time()
    thr, es_band, eng_b, comps, member = setup()
    c_red = np.round(np.arange(13.35, 13.951, 0.10), 2)
    f_red = np.full(len(c_red), float(np.asarray(thr["band01_floors_m1"])[-1]))
    rs = FarSet("red extension", c_red, 0.10, f_red)
    print(f"[red] design band + {len(c_red)} elements {c_red.min():.2f}-{c_red.max():.2f} um (floor {f_red[0]:.2e}) ...", flush=True)
    exts_r, ders_r = far_background(rs, comps, member)
    exts = [np.concatenate([eng_b.exts[i], exts_r[i]], axis=1) for i in range(2)]
    ders = [(np.concatenate([eng_b.ders[i][0], ders_r[i][0]], axis=1),
             np.concatenate([eng_b.ders[i][1], ders_r[i][1]], axis=1)) for i in range(2)]
    s_tot = sum(e[iz] for e in exts)
    sig = np.r_[np.maximum(VIS_PHI * s_tot[:N_VIS], ALPHA_FLOOR), es_band.floors, f_red]
    n_b = es_band.n
    res = dict(elements_um=list(map(float, c_red)), floor_m1=float(f_red[0]), materials={})
    arch = json.load(open(ARCH_A))["red_extension"]["quiet"] if ARCH_A.exists() else {}
    for mat in MATS:
        lays = layers_for(mat)
        r = {}
        for conv in CONVS:
            t = np.concatenate([t_vec(es_band, lays[conv]), t_vec(rs, lays[conv])])
            m_b, _, _ = mmin_general(exts, ders, sig, t, np.arange(n_b))
            m_r, _, _ = mmin_general(exts, ders, sig, t, np.arange(len(t)))
            r[conv] = dict(band_to_13p25=float(m_b), band_to_14=float(m_r), gain_percent=float(100 * (1 - m_r / m_b)))
            print(f"[red] {mat:>8} ({conv}): band to 13.25 {m_b:.4f} -> to 14 um {m_r:.4f} Tg ({r[conv]['gain_percent']:+.1f} %)", flush=True)
        if mat in arch:
            r["archive_a_band_to_14"] = arch[mat]
        res["materials"][mat] = r
    out = OUT_DIR / "materials_farir_uniaxial_red.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"[red] archived -> {out} ({time.time()-t0:.0f} s)", flush=True)


def merge():
    res = dict(note="Round 71: far-IR and red-extension thresholds of Table tab:materials under the three "
                    "orientation-averaging conventions ((a) ray mean = the pre-round-70 paper, (b) Clausius-Mossotti "
                    "= adopted, (c) cross-section average = bracket); same machinery as "
                    "materials_calibrated_thresholds.py PART 2/3.", materials={}, red_extension=None)
    for mat in MATS:
        f = OUT_DIR / f"materials_farir_uniaxial_{mat}.json"
        res["materials"][mat] = json.load(open(f))
    res["red_extension"] = json.load(open(OUT_DIR / "materials_farir_uniaxial_red.json"))
    MERGED.write_text(json.dumps(res, indent=1))
    print("merged ->", MERGED)
    print("\nmaterial  conv  band      far-only  band+far  [x3 far-only / band+far]  [x10]            far peak")
    for mat in MATS:
        for conv in CONVS:
            c = res["materials"][mat]["conventions"][conv]
            s = c["floor_scan"]
            print(f"{mat:>8}  ({conv})  {c['band_quiet']:.4f}  {c['triplet_plus_farir_quiet']:.4f}    {c['band_plus_farir_quiet']:.4f}    "
                  f"[{s['3.0']['triplet_plus_farir']:.3f} / {s['3.0']['band_plus_farir']:.3f}]  "
                  f"[{s['10.0']['triplet_plus_farir']:.3f} / {s['10.0']['band_plus_farir']:.3f}]  "
                  f"{c['far_peak_ext_m1']:.2e} at {c['far_peak_wl_um']:.2f}")
    print("\nred extension (band to 14 um), (a)/(b)/(c):")
    for mat, r in res["red_extension"]["materials"].items():
        print(f"{mat:>8}: " + "; ".join(f"({c}) {r[c]['band_to_13p25']:.4f} -> {r[c]['band_to_14']:.4f} ({r[c]['gain_percent']:+.1f} %)" for c in CONVS))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", choices=MATS + ("red",))
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
    elif a.mat == "red":
        run_red()
    elif a.mat:
        run_far(a.mat)
    else:
        ap.error("--mat or --merge")
