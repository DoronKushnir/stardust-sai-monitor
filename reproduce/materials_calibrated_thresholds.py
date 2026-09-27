"""
materials_calibrated_thresholds.py -- Paper 1 Sect. sec:materials re-based on
the calibrated two-component background and the resolved-band convention
(round 35d, 2026-09-24, Doron's comment in materials.tex).

Same machinery as Sects. 3-6 (calibrated_background_thresholds.py,
design_sensitivity_calibrated.py, detectability_2d_calibrated.py, imported
bit-identically; the cached background engines of the 2D analysis are
reused), with the injected material swapped for calcite, dolomite, or
alumina (saimon.sai factories: ray-arithmetic-mean optics, the silica
detection PSD r_med = 268 nm / sigma = 1.31, material densities):

  PART 1 -- the baseline design as flown for silica: heritage triplet
    (448/756/1544 nm, measured fractional errors, SNR-2000 floor) + the
    8-13 um band at 0.1-um sampling (46 elements, MEASURED per-element ACE
    floors) and its 7.8-9.3 um window subset; all seven background
    parameters free; undisturbed calibration and the post-Hunga-Tonga
    bracket.  No material-specific hardware: this is what the silica
    instrument sees of each material (calcite/dolomite nu2 at 11.3-11.4 um,
    alumina surface mode at 12.9 um, silica 8.74 um).

  PART 2 -- a far-infrared extension: a resolved 5-um-wide band of 0.1-um
    elements (round 35f, Doron: the paper's single sampling; 50 elements)
    around each material's long-wave resonance (silica 18-23 um; alumina
    18.25-23.25 um; dolomite 22.5-27.5 um; calcite 26-31 um), added to the
    design band, and also alone with the triplet ("far-IR-only instrument").
    Floors (round 37, Doron): the CALCULATED per-element trace-gas-removal
    (OE) floors of reproduce/farir_band_floors_w0p1.py -- every 0.1-um element
    of every band its own OE floor (archive
    outputs/materials_calibrated/farir_band_floors_w0p1.json); the former
    single-target 0.25-um floors x width factor 1.16 are retired.
    No detector-noise assumption enters (round 35f, Doron); instead the
    sensitivity of every far-IR result to a common floor multiplier
    {1, 3, 10, 30} is archived.
  PART 3 -- the design band extended to the red (13.35-13.95 um, floors
    held at the last measured 13.25-um value): does cutting at 13 um cost
    anything, in particular for the alumina surface mode?
    No measured floors exist beyond 13.3 um, so each far-IR element carries
    the trace-gas-removal (optimal-estimation) floor of that band from the
    material budgets (outputs/<mat>/*_band_floors_20km.csv, column
    sigma_alpha_1_m: the per-element gas-removal floor at the 0.25-um
    element, dz = 0.5 km; calcite interpolated in log between its 28.5-um
    peak and 29.2-um window values).  Undisturbed calibration only.
    Caveat: the sulfate member optics table ends at 26 um and is clamped
    beyond, so the calcite far-IR row is indicative.

Output: outputs/materials_calibrated/results.json (+ printed table).
Run from the repo root:  python reproduce/materials_calibrated_thresholds.py
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
    ALT, iz, CAL_QUIET, CAL_POSTHT, ALPHA_FLOOR, RMED_SIL_NM, SIGMA_SIL,
    TwoComponentBackground, glossac_2025N_quiet_profile, marginal, member_ri)
from design_sensitivity_calibrated import ElementSet, VIS_NM, VIS_PHI, N_VIS  # noqa: E402
from detectability_2d_calibrated import Engine, BRACKETS  # noqa: E402
from saimon.materials import SilicaRefractiveIndex  # noqa: E402
from saimon.sai import (create_silica_sai_layer, create_calcite_sai_layer,  # noqa: E402
                       create_dolomite_sai_layer, create_alumina_sai_layer)

OUT_DIR = _ROOT / "outputs" / "materials_calibrated"
THR_JSON = _ROOT / "outputs" / "calibrated_background_thresholds.json"
ENGINE_DIR = _ROOT / "outputs" / "detectability_2d_calibrated"
FAR_WIDTH_UM = 0.10          # round 35f: the paper's single sampling
FAR_BAND_UM = 5.0
SIG_204_CALC = 2.8e-9        # (retired, round 37) m^-1 at the 0.25-um element: the 20.4-um H2O+HNO3 floor
FARIR_JSON = _ROOT / "outputs" / "materials_calibrated" / "farir_band_floors_w0p1.json"
FLOOR_SCAN = (1.0, 3.0, 10.0, 30.0)


def band_floor_csv(mat, label_prefix):
    """(target_um, sigma_alpha_1_m) of one band row of the material budget CSV."""
    rows = list(csv.DictReader(open(_ROOT / "outputs" / mat / f"{mat}_band_floors_20km.csv")))
    for r in rows:
        if r["band"].startswith(label_prefix):
            return float(r["target_um"]), float(r["sigma_alpha_1_m"])
    raise KeyError(label_prefix)


def farir_spec(width_factor=None):
    """Far-IR element centers [um] and per-element floors [m^-1] per material.
    Round 37: the per-element 0.1-um OE floors of farir_band_floors_w0p1.py
    (width_factor=None); the legacy branch (a width factor given) keeps the
    former single-target 0.25-um floors x factor for comparison only."""
    spec = {}
    if width_factor is None:
        arch = json.load(open(FARIR_JSON))
        notes = {"silica": "silica 20.4-um band (18-23 um), per-element H2O+HNO3 OE floors",
                 "alumina": "alumina W band (18.25-23.25 um), per-element H2O+HNO3+O3 OE floors",
                 "dolomite": "dolomite lattice band (22.5-27.5 um), per-element H2O+O3+N2O OE floors",
                 "calcite": "calcite lattice band (26-31 um), per-element H2O+O3+N2O OE floors"}
        for mat, note in notes.items():
            a = arch[mat]
            assert abs(a["element_um"] - FAR_WIDTH_UM) < 1e-9
            # Each far-IR element carries the OE gas-removal floor of that element,
            # bounded below by the SNR=2000 transmission floor (ALPHA_FLOOR,
            # 5e-4 OD <-> 3.9e-9 m^-1) exactly as the visible/NIR channels are: a
            # gas-free element has no removal error but is still a measurement.
            floors = np.maximum(np.asarray(a["sigma_alpha_m1"]), ALPHA_FLOOR)
            spec[mat] = (np.asarray(a["centers_um"]), floors,
                         note + f"; bounded below by the SNR-2000 floor {ALPHA_FLOOR:.1e} m^-1")
        return spec
    n_el = int(round(FAR_BAND_UM / FAR_WIDTH_UM))
    def band(lo):
        return lo + FAR_WIDTH_UM / 2 + FAR_WIDTH_UM * np.arange(n_el)
    wf = width_factor
    c = band(18.0)
    spec["silica"] = (c, np.full(len(c), SIG_204_CALC * wf), "silica 20.4-um band (18-23 um), calculated H2O+HNO3 co-retrieval floor")
    _, f_w = band_floor_csv("alumina", "W")
    c = band(18.25)
    spec["alumina"] = (c, np.full(len(c), f_w * wf), "alumina W band (18.25-23.25 um), OE gas-removal floor")
    _, f_l = band_floor_csv("dolomite", "L")
    c = band(22.5)
    spec["dolomite"] = (c, np.full(len(c), f_l * wf), "dolomite lattice band (22.5-27.5 um), OE gas-removal floor")
    x1, f1 = band_floor_csv("calcite", "L (")
    x2, f2 = band_floor_csv("calcite", "L-win")
    c = band(26.0)
    fl = 10 ** np.interp(c, [x1, x2], np.log10([f1, f2])) * wf   # held at the ends
    spec["calcite"] = (c, fl, "calcite lattice band (26-31 um), OE gas-removal floor (log-interp 28.5/29.2, held outside)")
    return spec


class FarSet(ElementSet):
    """Far-IR elements only (no visible channels)."""

    def __init__(self, name, centers_um, width_um, floors_m1):
        super().__init__(name, centers_um, width_um, floors_m1)
        self.wl_nm = self.c * 1000.0
        self.n = len(self.wl_nm)

    def prof(self, layer):
        out = np.empty((len(ALT), self.n))
        for j, c in enumerate(self.c):
            out[:, j] = layer.extinction_profile_m1((c + self.sub) * 1e-6).mean(axis=1)
        return out


def far_background(fs: FarSet, comps, member):
    """exts / ders of the two-component background on the far-IR elements."""
    ri, wt, T = member_ri(member)
    bg = TwoComponentBackground(glossac_2025N_quiet_profile(), comps, ri, wt, T)
    cols = bg.columns()
    exts = [fs.prof(c) for c in cols]
    ders = []
    for i, (r0, s0, f) in enumerate(comps):
        dr, ds = bg.fd * r0, bg.fd * s0
        er = (fs.prof(bg.column(i, dr=+dr)) - fs.prof(bg.column(i, dr=-dr))) / (2 * dr)
        es_ = (fs.prof(bg.column(i, ds=+ds)) - fs.prof(bg.column(i, ds=-ds))) / (2 * ds)
        ders.append((er, es_))
    return exts, ders


def mmin_general(exts, ders, sig, t, idx):
    k = iz
    cols = [exts[0][k], exts[1][k], ders[0][0][k], ders[0][1][k], ders[1][0][k], ders[1][1][k]]
    sa, tperp, R2 = marginal(np.asarray(idx), cols, sig, t)
    return 3 * sa, tperp, (1 - R2 if np.isfinite(R2) else None)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    thr = json.load(open(THR_JSON))
    es_band = ElementSet("band 8-13 @0.1", thr["band01_elements_um"], 0.10, thr["band01_floors_m1"])
    es_win = ElementSet("window 7.8-9.3 @0.1", thr["window_elements_um"], 0.10, thr["window_floors_m1"])
    alpha525 = glossac_2025N_quiet_profile()
    ri_kp = SilicaRefractiveIndex(str(_ROOT / "data/optics/SiO2_KitamuraPopova.yml"))
    layers = {
        "silica": create_silica_sai_layer(ALT, 1.0, refractive_index=ri_kp, rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL),
        "calcite": create_calcite_sai_layer(ALT, 1.0),
        "dolomite": create_dolomite_sai_layer(ALT, 1.0),
        "alumina": create_alumina_sai_layer(ALT, 1.0),
    }
    res = {"materials": {}, "psd": dict(rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL),
           "design_band_elements_um": list(map(float, es_band.c))}

    # ---- PART 1: the baseline design (cached engines) -----------------------
    engines = {}
    for sname, es in (("band", es_band), ("window", es_win)):
        for bname, (comps, member) in BRACKETS.items():
            cache = ENGINE_DIR / f"engine_{sname}_{bname}.npz"
            assert cache.exists(), cache
            engines[(sname, bname)] = Engine(es, comps, member, alpha525, None, cache=cache)
    ref = thr["results"]["CALIBRATED quiet (LM65T223, 2-comp)"]["band 8-13 @0.1: measured floors"]["full 7-param"]["triplet + band01"]["mmin_tg"]
    for mat, lay in layers.items():
        r = {}
        for sname, es in (("band", es_band), ("window", es_win)):
            t = es.prof(lay)[iz]
            r[f"peak_tau_per_Tg_20km_{sname}"] = dict(
                wl_um=float(es.c[int(np.argmax(t[N_VIS:]))]), ext_m1=float(t[N_VIS:].max()))
            for bname in BRACKETS:
                eng = engines[(sname, bname)]
                eng.t20 = t
                sa = eng.sigma(iz)
                r[f"{sname}/{bname}"] = float(3 * sa)
        res["materials"][mat] = r
        print(f"PART 1 {mat:>8}: band {r['band/quiet']:.4f} (postHT {r['band/postHT']:.4f}) Tg; "
              f"window {r['window/quiet']:.4f} ({r['window/postHT']:.4f}) Tg; "
              f"peak in band {r['peak_tau_per_Tg_20km_band']['wl_um']:.2f} um", flush=True)
    assert abs(res["materials"]["silica"]["band/quiet"] - ref) < 1e-3, "silica design reproduction"

    # ---- PART 2: far-IR extension (undisturbed calibration) -----------------
    # round 37: per-element 0.1-um OE floors; the width factor is retired
    res["far_floor_source"] = str(FARIR_JSON.relative_to(_ROOT))
    res["far_width_factor"] = None
    res["far_floor_scan"] = list(FLOOR_SCAN)
    spec = farir_spec()
    comps, member = BRACKETS["quiet"]
    eng_b = engines[("band", "quiet")]
    for mat, lay in layers.items():
        c_far, f_far, note = spec[mat]
        fs = FarSet(f"{mat} far-IR", c_far, FAR_WIDTH_UM, f_far)
        print(f"PART 2 {mat:>8}: building far-IR background ({len(c_far)} elements "
              f"{c_far.min():.2f}-{c_far.max():.2f} um) ...", flush=True)
        exts_f, ders_f = far_background(fs, comps, member)
        # concatenate: [vis(3) + band01(46)] from the cached engine, then far-IR
        exts = [np.concatenate([eng_b.exts[i], exts_f[i]], axis=1) for i in range(2)]
        ders = [(np.concatenate([eng_b.ders[i][0], ders_f[i][0]], axis=1),
                 np.concatenate([eng_b.ders[i][1], ders_f[i][1]], axis=1)) for i in range(2)]
        t = np.concatenate([es_band.prof(lay)[iz], fs.prof(lay)[iz]])
        s_tot = sum(e[iz] for e in exts)
        sig = np.r_[np.maximum(VIS_PHI * s_tot[:N_VIS], ALPHA_FLOOR), es_band.floors, f_far]
        n_b = es_band.n
        idx_all = np.arange(len(t))
        idx_far = np.r_[np.arange(N_VIS), np.arange(n_b, len(t))]
        m_all, _, r2_all = mmin_general(exts, ders, sig, t, idx_all)
        m_far, _, _ = mmin_general(exts, ders, sig, t, idx_far)
        r = res["materials"][mat]
        scan = {}
        for fmul in FLOOR_SCAN:
            sig_m = np.r_[sig[:n_b], f_far * fmul]
            scan[str(fmul)] = dict(band_plus_farir=float(mmin_general(exts, ders, sig_m, t, idx_all)[0]),
                                   triplet_plus_farir=float(mmin_general(exts, ders, sig_m, t, idx_far)[0]))
        r["farir"] = dict(centers_um=list(map(float, c_far)), floors_m1=list(map(float, f_far)), note=note,
                          peak_wl_um=float(c_far[int(np.argmax(t[n_b:]))]), peak_ext_m1=float(t[n_b:].max()),
                          band_plus_farir_quiet=float(m_all), triplet_plus_farir_quiet=float(m_far),
                          one_minus_R2_all=r2_all, floor_scan=scan)
        print(f"          {mat:>8}: triplet+band+farIR {m_all:.4f} Tg; triplet+farIR {m_far:.4f} Tg; "
              f"far-IR peak {r['farir']['peak_ext_m1']:.3e} m^-1/Tg at {r['farir']['peak_wl_um']:.2f} um; "
              f"floor x3/x10/x30 -> band+far {scan['3.0']['band_plus_farir']:.4f}/{scan['10.0']['band_plus_farir']:.4f}/{scan['30.0']['band_plus_farir']:.4f}, "
              f"far-only {scan['3.0']['triplet_plus_farir']:.4f}/{scan['10.0']['triplet_plus_farir']:.4f}/{scan['30.0']['triplet_plus_farir']:.4f}", flush=True)

    # ---- PART 3: design band extended to the red (13.35-13.95 um) ----------
    c_red = np.round(np.arange(13.35, 13.951, 0.10), 2)
    f_red = np.full(len(c_red), float(np.asarray(thr["band01_floors_m1"])[-1]))
    rs = FarSet("red extension", c_red, 0.10, f_red)
    print(f"PART 3: design band + {len(c_red)} elements {c_red.min():.2f}-{c_red.max():.2f} um "
          f"(floor held at {f_red[0]:.2e}) ...", flush=True)
    exts_r, ders_r = far_background(rs, comps, member)
    exts = [np.concatenate([eng_b.exts[i], exts_r[i]], axis=1) for i in range(2)]
    ders = [(np.concatenate([eng_b.ders[i][0], ders_r[i][0]], axis=1),
             np.concatenate([eng_b.ders[i][1], ders_r[i][1]], axis=1)) for i in range(2)]
    res["red_extension"] = dict(elements_um=list(map(float, c_red)), floor_m1=float(f_red[0]), quiet={})
    for mat, lay in layers.items():
        t = np.concatenate([es_band.prof(lay)[iz], rs.prof(lay)[iz]])
        s_tot = sum(e[iz] for e in exts)
        sig = np.r_[np.maximum(VIS_PHI * s_tot[:N_VIS], ALPHA_FLOOR), es_band.floors, f_red]
        m, _, _ = mmin_general(exts, ders, sig, t, np.arange(len(t)))
        res["red_extension"]["quiet"][mat] = float(m)
        print(f"          {mat:>8}: band to 14 um -> {m:.4f} Tg (band to 13.25: {res['materials'][mat]['band/quiet']:.4f})", flush=True)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"archived -> {OUT_DIR/'results.json'}")


if __name__ == "__main__":
    main()
