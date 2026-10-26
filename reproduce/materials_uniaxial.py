"""
materials_uniaxial.py -- referee round 5, item 9 (paper 1, Sect. 7):
orientation averaging of the uniaxial materials, optical-constant provenance
and an uncertainty test adequate for the material ranking.

PART A  Three conventions for the extinction of a randomly oriented ensemble
        of uniaxial spheres (calcite, dolomite, alumina; ordinary/extraordinary
        ray tables of data/optics/):
          (a) ray-arithmetic mean  m = (m_o + m_e)/2 fed to Mie (the current
              paper convention, Lederer 2026; saimon.materials);
          (b) Rayleigh-exact random-orientation polarizability average: the
              Clausius-Mossotti inversion of
                 (eps_eff-1)/(eps_eff+2) = [2 (eps_o-1)/(eps_o+2) + (eps_e-1)/(eps_e+2)]/3,
              m_eff = sqrt(eps_eff) fed to Mie (exact in the Rayleigh limit for
              the absorption and scattering of a sphere with a uniaxial
              dielectric tensor: the polarizability tensor is diagonal with
              alpha_j = 4 pi a^3 (eps_j-1)/(eps_j+2) and the orientation
              average of the extinction of an unpolarized beam is the mean of
              the three);
          (c) cross-section average: Mie with m_o and with m_e separately,
              combined as (2 Q_o + Q_e)/3 (coincides with (b) in the Rayleigh
              limit; the usual approximation beyond it).
        For each: per-Tg band spectrum (47 elements, 20 km, reference PSD
        268 nm / 1.31), peak per-Tg extinction at the material's band element,
        and the baseline band threshold (heritage triplet + 47-element band,
        six background parameters free, measured per-element floors; cached
        engines of outputs/detectability_2d_calibrated/), quiet and post-HT.
        Rayleigh test at r = 10 nm and a monodisperse radius scan.
PART B  Uncertainty test for the ranking (convention (a), the paper's): k
        scaled by +-20 % across the design band and the table shifted by
        +-0.05 um in wavelength, thresholds for calcite, dolomite, alumina and
        silica.
PART C  Optical-constant provenance (files, ranges, interpolation,
        extrapolation, temperature, densities) -> JSON and the booktabs table
        tables/optical_constants_provenance.tex.

Archive: outputs/materials_uniaxial.json
Run from the repo root:  python reproduce/materials_uniaxial.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import interp1d

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from calibrated_background_thresholds import (  # noqa: E402
    ALT, iz, RMED_SIL_NM, SIGMA_SIL, MEMBERS, HRI)
from design_sensitivity_calibrated import ElementSet, VIS_NM, N_VIS  # noqa: E402
from detectability_2d_calibrated import Engine, BRACKETS  # noqa: E402
import materials_calibrated_thresholds as MCT  # noqa: E402
from saimon.materials import (SilicaRefractiveIndex, RayMeanUniaxialRefractiveIndex,  # noqa: E402
                             CalciteRefractiveIndex, DolomiteRefractiveIndex,
                             AluminaRefractiveIndex)
from saimon import sai as SAI  # noqa: E402
from saimon.sai import (create_silica_sai_layer, create_calcite_sai_layer,  # noqa: E402
                       create_dolomite_sai_layer, create_alumina_sai_layer)
from saimon.mie import mie_efficiencies  # noqa: E402

CSV = _ROOT / "data" / "optics"
OUT = _ROOT / "outputs" / "materials_uniaxial.json"
TEX = _ROOT / "tables" / "optical_constants_provenance.tex"
BAND_LO_UM, BAND_HI_UM = 7.75, 13.25     # design band incl. the half elements
K_SCALES = (1.2, 0.8)
SHIFTS_UM = (+0.05, -0.05)
PEAK_UM = {"silica": 8.80, "calcite": 11.4, "dolomite": 11.3, "alumina": 12.9}
MAT_CLS = {"calcite": CalciteRefractiveIndex, "dolomite": DolomiteRefractiveIndex,
           "alumina": AluminaRefractiveIndex}
MAT_FACTORY = {"calcite": create_calcite_sai_layer, "dolomite": create_dolomite_sai_layer,
               "alumina": create_alumina_sai_layer}
MAT_RHO = {"silica": SAI.SILICA_DENSITY_KG_M3, "calcite": SAI.CALCITE_DENSITY_KG_M3,
           "dolomite": SAI.DOLOMITE_DENSITY_KG_M3, "alumina": SAI.ALUMINA_DENSITY_KG_M3}


# ---------------------------------------------------------------------------
# ray tables and the three conventions
# ---------------------------------------------------------------------------
class RayTables:
    """Ordinary and extraordinary ray tables of one material, with the same
    cleaning as saimon.materials (Cauchy smoothing of n below the cutoff is
    linear in the data, so applying it per ray and averaging reproduces the
    library's smoothed ray mean).  Linear interpolation in wavelength, end
    values held outside the table (as the library)."""

    def __init__(self, cls):
        self.cls = cls
        self.rays = {}
        for ray, fn in (("o", cls._DEFAULT_ORDINARY), ("e", cls._DEFAULT_EXTRAORDINARY)):
            wl, n, k = RayMeanUniaxialRefractiveIndex._load_ray(CSV / fn)
            k = np.abs(k)
            cut = cls._CAUCHY_SMOOTH_BELOW_UM
            if cut is not None:
                sel = wl < cut
                x = 1.0 / wl[sel] ** 2
                n = n.copy()
                n[sel] = np.polyval(np.polyfit(x, n[sel], 2), x)
            wl_m = wl * 1e-6
            self.rays[ray] = (
                interp1d(wl_m, n, bounds_error=False, fill_value=(n[0], n[-1])),
                interp1d(wl_m, k, bounds_error=False, fill_value=(k[0], k[-1])),
                wl, n, k, fn)

    def m(self, ray, wl_m):
        fn_n, fn_k, *_ = self.rays[ray]
        wl = np.asarray(wl_m, float)
        return fn_n(wl) + 1j * fn_k(wl)


def cm_average(mo, me):
    """Rayleigh-exact effective index of a randomly oriented uniaxial sphere:
    Clausius-Mossotti inversion of the mean polarizability (2 o + 1 e)/3."""
    eo, ee = mo ** 2, me ** 2
    F = (2.0 * (eo - 1) / (eo + 2) + (ee - 1) / (ee + 2)) / 3.0
    eps = (1 + 2 * F) / (1 - F)
    m = np.sqrt(eps)
    m = np.where(m.imag < 0, -m, m)          # passive branch (k >= 0, n > 0)
    return m, F


class ConventionRI:
    """Callable refractive index on wavelengths [m] for one convention
    ('a' ray mean, 'b' Clausius-Mossotti, 'o', 'e' single ray), with optional
    perturbations: k scaled by `kscale` inside [BAND_LO, BAND_HI] um, and the
    whole table shifted by `shift_um` (m'(lambda) = m(lambda - shift)) above
    5 um (the visible/NIR channels keep the unshifted table)."""

    def __init__(self, tabs: RayTables, conv: str, kscale=1.0, shift_um=0.0):
        self.tabs, self.conv, self.kscale, self.shift = tabs, conv, kscale, shift_um

    def _rays(self, wl_m):
        wl = np.asarray(wl_m, float)
        q = wl.copy()
        if self.shift:
            q[wl > 5e-6] -= self.shift * 1e-6
        mo, me = self.tabs.m("o", q), self.tabs.m("e", q)
        if self.kscale != 1.0:
            band = (wl >= BAND_LO_UM * 1e-6) & (wl <= BAND_HI_UM * 1e-6)
            s = np.where(band, self.kscale, 1.0)
            mo = mo.real + 1j * mo.imag * s
            me = me.real + 1j * me.imag * s
        return mo, me

    def __call__(self, wl_m):
        mo, me = self._rays(wl_m)
        if self.conv == "a":
            return 0.5 * (mo + me)
        if self.conv == "b":
            return cm_average(mo, me)[0]
        if self.conv == "o":
            return mo
        if self.conv == "e":
            return me
        raise ValueError(self.conv)


class PerturbedRI:
    """Same perturbations on an isotropic (silica) index callable."""

    def __init__(self, base, kscale=1.0, shift_um=0.0):
        self.base, self.kscale, self.shift = base, kscale, shift_um

    def __call__(self, wl_m):
        wl = np.asarray(wl_m, float)
        q = wl.copy()
        if self.shift:
            q[wl > 5e-6] -= self.shift * 1e-6
        m = self.base(q)
        if self.kscale != 1.0:
            band = (wl >= BAND_LO_UM * 1e-6) & (wl <= BAND_HI_UM * 1e-6)
            m = m.real + 1j * m.imag * np.where(band, self.kscale, 1.0)
        return m


# ---------------------------------------------------------------------------
# threshold machinery (PART 1 of materials_calibrated_thresholds.py)
# ---------------------------------------------------------------------------
def setup():
    thr = json.load(open(MCT.THR_JSON))
    es = ElementSet("band 8-13 @0.1", thr["band01_elements_um"], 0.10, thr["band01_floors_m1"])
    alpha525 = None   # the cached engines do not need the profile
    engines = {}
    for bname, (comps, member) in BRACKETS.items():
        cache = MCT.ENGINE_DIR / f"engine_band_{bname}.npz"
        assert cache.exists(), cache
        engines[bname] = Engine(es, comps, member, alpha525, None, cache=cache)
    ref = thr["results"]["CALIBRATED quiet (LM65T223, 2-comp)"]["band 8-13 @0.1: measured floors"]["full 7-param"]["triplet + band01"]["mmin_tg"]
    return es, engines, ref


def thresholds(engines, t):
    out = {}
    for bname, eng in engines.items():
        eng.t20 = t
        out[bname] = float(3 * eng.sigma(iz))
    return out


def summarize(es, t, mat):
    """Per-Tg band vector at 20 km -> peak diagnostics."""
    tb = t[N_VIS:]
    j_peak = int(np.argmin(np.abs(es.c - PEAK_UM[mat])))
    return dict(peak_element_um=float(es.c[j_peak]), peak_ext_m1_per_Tg=float(tb[j_peak]),
                argmax_element_um=float(es.c[int(np.argmax(tb))]), max_ext_m1_per_Tg=float(tb.max()),
                t_vis_m1_per_Tg=list(map(float, t[:N_VIS])), t_band_m1_per_Tg=list(map(float, tb)))


def prof20(es, layer, label):
    t0 = time.time()
    t = es.prof(layer)[iz]
    print(f"    prof {label}: {time.time() - t0:.0f} s", flush=True)
    return t


# ---------------------------------------------------------------------------
def part_a(es, engines, res):
    print("PART A: uniaxial conventions", flush=True)
    res["conventions"] = {
        "a": "ray-arithmetic mean m=(m_o+m_e)/2 -> Mie (paper, Lederer 2026)",
        "b": "Clausius-Mossotti inversion of the mean Rayleigh polarizability (2 o + 1 e)/3 -> Mie (Rayleigh-exact)",
        "c": "cross-section average (2 Q_o + Q_e)/3 of Mie with m_o and with m_e",
    }
    res["materials"] = {}
    for mat, cls in MAT_CLS.items():
        print(f"  {mat}", flush=True)
        tabs = RayTables(cls)
        lib = cls()
        wl_chk = es.c * 1e-6
        ri_a = ConventionRI(tabs, "a")
        d = np.abs(ri_a(wl_chk) - lib(wl_chk)).max()
        r = {"ray_mean_reproduces_library_max_abs_diff": float(d)}
        t = {}
        for conv in ("a", "b", "o", "e"):
            lay = MAT_FACTORY[mat](ALT, 1.0, refractive_index=ConventionRI(tabs, conv))
            t[conv] = prof20(es, lay, f"{mat} {conv}")
        t["c"] = (2 * t["o"] + t["e"]) / 3.0
        for conv in ("a", "b", "c", "o", "e"):
            r[conv] = summarize(es, t[conv], mat)
            r[conv]["mmin_tg"] = thresholds(engines, t[conv])
            print(f"    {conv}: peak {r[conv]['peak_ext_m1_per_Tg']:.3e} m^-1/Tg at {r[conv]['peak_element_um']:.2f} um; "
                  f"threshold quiet {r[conv]['mmin_tg']['quiet']:.4f} Tg (postHT {r[conv]['mmin_tg']['postHT']:.4f})", flush=True)
        for conv in ("b", "c"):
            r[f"ratio_{conv}_over_a"] = dict(
                mmin_quiet=r[conv]["mmin_tg"]["quiet"] / r["a"]["mmin_tg"]["quiet"],
                mmin_postHT=r[conv]["mmin_tg"]["postHT"] / r["a"]["mmin_tg"]["postHT"],
                peak_ext=r[conv]["peak_ext_m1_per_Tg"] / r["a"]["peak_ext_m1_per_Tg"])
        r["ratio_c_over_b"] = dict(mmin_quiet=r["c"]["mmin_tg"]["quiet"] / r["b"]["mmin_tg"]["quiet"],
                                   peak_ext=r["c"]["peak_ext_m1_per_Tg"] / r["b"]["peak_ext_m1_per_Tg"])
        # effective indices at the peak element, for the record
        lam_pk = PEAK_UM[mat] * 1e-6
        mo, me = tabs.m("o", lam_pk), tabs.m("e", lam_pk)
        mb, F = cm_average(mo, me)
        r["index_at_peak_element"] = dict(wl_um=PEAK_UM[mat],
                                          m_o=[float(mo.real), float(mo.imag)], m_e=[float(me.real), float(me.imag)],
                                          m_a=[float((mo + me).real / 2), float((mo + me).imag / 2)],
                                          m_b=[float(mb.real), float(mb.imag)], ImF_mean_polarizability=float(F.imag))
        # --- Rayleigh test: monodisperse radius scan at the peak element
        scan = {"r_nm": [10, 50, 100, 268, 500, 1000, 2000], "wl_um": PEAK_UM[mat]}
        qa, qb, qc, qabs_b, ray_b = [], [], [], [], []
        ma = 0.5 * (mo + me)
        for r_nm in scan["r_nm"]:
            x = 2 * np.pi * r_nm * 1e-9 / lam_pk
            Qa = mie_efficiencies(np.array([x]), complex(ma))[0][0]
            Qb, _, Qb_abs = (v[0] for v in mie_efficiencies(np.array([x]), complex(mb)))
            Qo = mie_efficiencies(np.array([x]), complex(mo))[0][0]
            Qe = mie_efficiencies(np.array([x]), complex(me))[0][0]
            qa.append(float(Qa)); qb.append(float(Qb)); qc.append(float((2 * Qo + Qe) / 3)); qabs_b.append(float(Qb_abs))
            ray_b.append(float(4 * x * F.imag))     # Rayleigh absorption 4 x Im(mean polarizability)
        scan.update(Q_ext_a=qa, Q_ext_b=qb, Q_ext_c=qc, Q_abs_b_mie=qabs_b, Q_abs_rayleigh_4xImF=ray_b,
                    ratio_b_over_a=[b / a for a, b in zip(qa, qb)], ratio_c_over_a=[c / a for a, c in zip(qa, qc)],
                    ratio_c_over_b=[c / b for b, c in zip(qb, qc)],
                    x=[float(2 * np.pi * rn * 1e-9 / lam_pk) for rn in scan["r_nm"]])
        r["monodisperse_scan_at_peak"] = scan
        # --- Rayleigh test over the whole band at r = 10 nm
        x10 = 2 * np.pi * 10e-9 / (es.c * 1e-6)
        mo_b, me_b = tabs.m("o", es.c * 1e-6), tabs.m("e", es.c * 1e-6)
        mb_b, F_b = cm_average(mo_b, me_b)
        ma_b = 0.5 * (mo_b + me_b)
        Qa = np.array([mie_efficiencies(np.array([x]), complex(m))[0][0] for x, m in zip(x10, ma_b)])
        Qb = np.array([mie_efficiencies(np.array([x]), complex(m))[0][0] for x, m in zip(x10, mb_b)])
        Qo = np.array([mie_efficiencies(np.array([x]), complex(m))[0][0] for x, m in zip(x10, mo_b)])
        Qe = np.array([mie_efficiencies(np.array([x]), complex(m))[0][0] for x, m in zip(x10, me_b)])
        Qc = (2 * Qo + Qe) / 3
        j_peak = int(np.argmin(np.abs(es.c - PEAK_UM[mat])))
        r["rayleigh_test_r10nm_band"] = dict(
            max_abs_rel_dev_c_vs_b=float(np.max(np.abs(Qc / Qb - 1))),
            max_abs_rel_dev_a_vs_b=float(np.max(np.abs(Qa / Qb - 1))),
            a_over_b_at_peak=float(Qa[j_peak] / Qb[j_peak]), c_over_b_at_peak=float(Qc[j_peak] / Qb[j_peak]),
            a_over_b_band=list(map(float, Qa / Qb)), c_over_b_band=list(map(float, Qc / Qb)))
        print(f"    Rayleigh r=10 nm: max |c/b-1| = {r['rayleigh_test_r10nm_band']['max_abs_rel_dev_c_vs_b']:.2e}, "
              f"a/b at peak = {r['rayleigh_test_r10nm_band']['a_over_b_at_peak']:.4f}, "
              f"max |a/b-1| = {r['rayleigh_test_r10nm_band']['max_abs_rel_dev_a_vs_b']:.3f}", flush=True)
        res["materials"][mat] = r


def part_b(es, engines, res, ref):
    print("PART B: k +-20 % and lambda +-0.05 um (convention a)", flush=True)
    ri_kp = SilicaRefractiveIndex(str(CSV / "SiO2_KitamuraPopova.yml"))
    res["perturbations"] = {"k_scales": list(K_SCALES), "shifts_um": list(SHIFTS_UM),
                            "k_scale_band_um": [BAND_LO_UM, BAND_HI_UM],
                            "shift_applied_above_um": 5.0, "materials": {}}
    for mat in ("silica", "calcite", "dolomite", "alumina"):
        print(f"  {mat}", flush=True)
        if mat == "silica":
            def make(**kw):
                return create_silica_sai_layer(ALT, 1.0, refractive_index=PerturbedRI(ri_kp, **kw),
                                               rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL)
        else:
            tabs = RayTables(MAT_CLS[mat])

            def make(**kw):
                return MAT_FACTORY[mat](ALT, 1.0, refractive_index=ConventionRI(tabs, "a", **kw))
        r = {}
        t0 = prof20(es, make(), f"{mat} base")
        base = summarize(es, t0, mat)
        base["mmin_tg"] = thresholds(engines, t0)
        r["base"] = base
        if mat == "silica":
            assert abs(base["mmin_tg"]["quiet"] - ref) < 1e-3, "silica design reproduction"
        cases = [(f"k_x{s}", dict(kscale=s)) for s in K_SCALES] + \
                [(f"shift_{d:+.2f}um", dict(shift_um=d)) for d in SHIFTS_UM]
        for lab, kw in cases:
            t = prof20(es, make(**kw), f"{mat} {lab}")
            s = summarize(es, t, mat)
            s["mmin_tg"] = thresholds(engines, t)
            s["mmin_ratio_to_base"] = {k: s["mmin_tg"][k] / base["mmin_tg"][k] for k in s["mmin_tg"]}
            s["peak_ratio_to_base"] = s["peak_ext_m1_per_Tg"] / base["peak_ext_m1_per_Tg"]
            r[lab] = s
            print(f"    {lab:>14}: threshold {s['mmin_tg']['quiet']:.4f} Tg (x{s['mmin_ratio_to_base']['quiet']:.3f}); "
                  f"peak x{s['peak_ratio_to_base']:.3f}", flush=True)
        res["perturbations"]["materials"][mat] = r
    # ranking robustness: every pairwise ordering under every perturbation
    mats = ("silica", "alumina", "dolomite", "calcite")
    pm = res["perturbations"]["materials"]
    envelope = {m: dict(min=min(v["mmin_tg"]["quiet"] for k, v in pm[m].items()),
                        max=max(v["mmin_tg"]["quiet"] for k, v in pm[m].items())) for m in mats}
    res["perturbations"]["threshold_envelope_quiet_tg"] = envelope
    order_ok = all(envelope[mats[i]]["max"] < envelope[mats[i + 1]]["min"] for i in range(2))
    res["perturbations"]["ranking_silica_lt_alumina_lt_carbonates_preserved_under_all"] = bool(order_ok)
    res["perturbations"]["carbonate_pair_envelopes_overlap"] = bool(
        envelope["dolomite"]["max"] >= envelope["calcite"]["min"])


# ---------------------------------------------------------------------------
def _yaml_range(fn):
    data = yaml.safe_load(open(CSV / fn))
    wl = []
    for block in data["DATA"]:
        if "tabulated" in block.get("type", ""):
            for line in block["data"].strip().split("\n"):
                parts = line.split()
                if len(parts) >= 2:
                    wl.append(float(parts[0]))
    wl = np.asarray(wl)
    return dict(file=fn, wl_min_um=float(wl.min()), wl_max_um=float(wl.max()), n_points=int(len(wl)))


def _csv_range(fn):
    wl, n, k = RayMeanUniaxialRefractiveIndex._load_ray(CSV / fn)
    j = int(np.argmin(np.abs(wl - 0.5893)))
    return dict(file=fn, wl_min_um=float(wl.min()), wl_max_um=float(wl.max()), n_points=int(len(wl)),
                n_at_0p5893um=float(n[j]), k_at_0p5893um=float(k[j]),
                min_k=float(k.min()))


def _member_range(name):
    wt, T, f, col = MEMBERS[name]
    rows = []
    for line in (HRI / f).read_text().splitlines():
        try:
            v = [float(x) for x in line.split()]
        except ValueError:
            continue
        if len(v) >= 3 + col:
            rows.append(v[1])
    rows = np.asarray(rows)
    return dict(file=str(Path("data/optics/h2so4_members") / f), wt_percent=wt, T_K=T,
                table_wl_min_um=float(rows.min()), table_wl_max_um=float(rows.max()),
                used_range_um=[2.05, 26.0], note="linear interpolation in wavelength; used only 2.05-26 um; "
                "below 2.05 um Palmer-Williams (300 K) + Lorentz-Lorenz to 215 K")


def part_c(res):
    print("PART C: provenance", flush=True)
    prov = {}
    prov["silica"] = dict(
        primary=_yaml_range("SiO2_KitamuraPopova.yml"), alternative=_yaml_range("SiO2_Franta2016.yml"),
        popova=_yaml_range("Popova.yml"),
        source="Kitamura et al. (2007) 0.1-20 um stitched with Popova et al. (1972) 7-50 um at 20 um (Lederer 2026 set); "
               "Franta et al. (2016) as the alternative",
        temperature="room temperature (refractiveindex.info headers: Kitamura 'near room temperature', Popova/Franta 'room temperature')",
        interpolation="linear in wavelength, n and k separately (scipy interp1d; saimon.materials.SilicaRefractiveIndex)",
        extrapolation="linear extrapolation beyond the table ends (fill_value='extrapolate'); never exercised: the table covers 0.1-50 um",
        density_kg_m3=MAT_RHO["silica"], density_source="saimon.sai.SILICA_DENSITY_KG_M3 = 2200 (amorphous silica; code constant, no citation in the code)")
    prov["sulfuric_acid"] = dict(
        LM65T223=_member_range("LM65T223"), LM72T213=_member_range("LM72T213"), B70T215=_member_range("B70T215"),
        visible="Palmer & Williams (1975) 300 K tables (data/optics/palmer_williams_75_combined_300K.csv), "
                "Lorentz-Lorenz corrected to 215 K (saimon.sulfate_background.sulfuric_acid_at_temperature), 0.30-2.04 um",
        temperature="Lund Myhre 65 wt% at 223 K (quiet calibration), 72 wt% at 213 K (post-HT bracket), Biermann 70 wt% at 215 K (optics-swap bracket)",
        interpolation="linear in wavelength (np.interp on the wavenumber-ordered table), n and k separately; k clipped >= 0",
        extrapolation="TabulatedRI holds the end values beyond 26 um (tables used only 2.05-26 um); far-IR calcite row (26-31 um) extrapolates",
        density="H2SO4 density from the Myhre (1998) polynomial for the Lorentz-Lorenz correction; droplet density ~1700 kg m^-3 in the number-density conversions")
    for mat, cls in MAT_CLS.items():
        o = _csv_range(cls._DEFAULT_ORDINARY)
        e = _csv_range(cls._DEFAULT_EXTRAORDINARY)
        prov[mat] = dict(ordinary=o, extraordinary=e, density_kg_m3=MAT_RHO[mat],
                         interpolation="linear in wavelength, n and k separately, on the ray tables (saimon.materials.RayMeanUniaxialRefractiveIndex)",
                         extrapolation="end values held (clamped) beyond the table ends",
                         temperature="room-temperature laboratory reflectance tabulations (no low-temperature tables exist)")
    prov["calcite"].update(
        source="Long et al. (1993) Lorentz-oscillator fits (o and e rays) evaluated 0.1-1000 um on a 20000-point log grid "
               "(the Stardust calcite optical set, Lederer 2026 convention); visible n from the oscillator model",
        density_source="saimon.sai.CALCITE_DENSITY_KG_M3 = 2710 ('standard literature value'; code comment)",
        handbook_visible_indices_for_reference=dict(n_o=1.658, n_e=1.486, note="Handbook of Mineralogy calcite Na-D; NOT what the file holds (see n_at_0p5893um)"))
    prov["dolomite"].update(
        source="Querry (1987, CRDEC-CR-88009) o/e tables via refractiveindex.info (data/optics/Dolomite_Querry-{o,e}.yml), 2.5-40 (o) / 2.5-50 um (e); "
               "reproduce/convert_querry_optics.py: k clamped >= 0, flat visible/NIR anchors at the Handbook-of-Mineralogy Na-D indices "
               "(n_o = 1.679, n_e = 1.500, k = 0) at 0.20, 0.5893, 0.6563 um, linear bridge 0.6563-2.5 um",
        yaml=dict(o=_yaml_range("Dolomite_Querry-o.yml"), e=_yaml_range("Dolomite_Querry-e.yml")),
        density_source="saimon.sai.DOLOMITE_DENSITY_KG_M3 = 2860 (Handbook of Mineralogy, D(meas.) = 2.86 g/cm^3; code comment)")
    prov["alumina"].update(
        source="Querry (1985, CRDC-CR-85034) sapphire o/e tables via refractiveindex.info (data/optics/Al2O3_Querry-{o,e}.yml), 0.21-55.6 um; "
               "reproduce/convert_querry_optics.py: k clamped >= 0 and set to zero below 6 um; ray-mean n below 5 um replaced by a Cauchy fit",
        yaml=dict(o=_yaml_range("Al2O3_Querry-o.yml"), e=_yaml_range("Al2O3_Querry-e.yml")),
        density_source="saimon.sai.ALUMINA_DENSITY_KG_M3 = 3980 ('standard sapphire value'; code comment)")
    res["provenance"] = prov

    # ---- booktabs table -------------------------------------------------------
    def rng(d):
        return f"{d['wl_min_um']:.3g}--{d['wl_max_um']:.4g}"
    rows = [
        ("Silica (amorphous)", r"\citet{Kitamura2007} $0.1$--$20$ + \citet{Popova1972} $7$--$50$, stitched at $20$",
         r"\texttt{SiO2\_KitamuraPopova.yml}", rng(prov["silica"]["primary"]), "room", "linear; linear beyond ends (unused)", "2200"),
        ("", r"alternative: \citet{Franta2016}", r"\texttt{SiO2\_Franta2016.yml}", rng(prov["silica"]["alternative"]), "room", "linear", "--"),
        ("Sulfuric acid", r"\citet{LundMyhre2003} $65$~wt\,\% (quiet); $72$~wt\,\% (post-HT bracket)",
         r"\texttt{myhreh2so4set4.dat}, \texttt{set3.dat}",
         f"{prov['sulfuric_acid']['LM65T223']['used_range_um'][0]:.2f}--{prov['sulfuric_acid']['LM65T223']['used_range_um'][1]:.0f} (used)",
         "223; 213", "linear; held beyond $26~\\mu$m", "--"),
        ("", r"\citet{Biermann2000} $70$~wt\,\% (optics bracket)", r"\texttt{h2so4T215.biermann}",
         f"{prov['sulfuric_acid']['B70T215']['used_range_um'][0]:.2f}--{prov['sulfuric_acid']['B70T215']['used_range_um'][1]:.0f} (used)", "215", "linear", "--"),
        ("", r"\citet{PalmerWilliams1975}, Lorentz--Lorenz to $215$~K", r"\texttt{palmer\_williams\_75\_*.csv}", "0.30--2.04", r"300$\to$215", "linear", "--"),
        ("Calcite", r"\citet{Long1993} oscillator fits, $o$ and $e$ rays", r"\texttt{calcite\_optical\_*Ray.csv}",
         rng(prov["calcite"]["ordinary"]), "room", "linear; held beyond ends", "2710"),
        ("Dolomite", r"\citet{Querry1987}, $o$ ($2.5$--$40$) and $e$ ($2.5$--$50$) rays; flat visible at handbook indices",
         r"\texttt{dolomite\_optical\_*Ray.csv}", f"{rng(prov['dolomite']['ordinary'])} ($o$), {rng(prov['dolomite']['extraordinary'])} ($e$)",
         "room", "linear; held beyond ends", "2860"),
        ("Alumina", r"\citet{Querry1985} sapphire, $o$ and $e$ rays; $k=0$ below $6~\mu$m",
         r"\texttt{alumina\_optical\_*Ray.csv}", rng(prov["alumina"]["ordinary"]), "room", "linear; held beyond ends", "3980"),
    ]
    lines = [
        "% Generated by reproduce/materials_uniaxial.py (round 70, referee item 9).",
        "% Archive: outputs/materials_uniaxial.json ['provenance'].",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Optical constants and densities of the forward model: source tabulation, file in the",
        r"repository, tabulated wavelength range, measurement temperature, interpolation/extrapolation",
        r"as implemented, and the solid density used in the mass closure. All mineral tabulations are",
        r"room-temperature laboratory data; the three uniaxial minerals enter through their ordinary ($o$)",
        r"and extraordinary ($e$) ray tables (Sect.~\ref{sec:materials}). Interpolation is linear in",
        r"wavelength on $n$ and $k$ separately; ``held'' means the end value is used outside the table.}",
        r"\label{tab:optical_provenance}",
        r"\footnotesize",
        r"\begin{tabular}{l p{5.6cm} p{3.0cm} p{2.2cm} l p{2.4cm} r}",
        r"\toprule",
        r"Material & Source & File (\texttt{data/optics/}) & Range [$\mu$m] & $T$ [K] & Interp./extrap. & $\rho$ [kg\,m$^{-3}$] \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    TEX.write_text("\n".join(lines))
    print(f"  wrote {TEX.relative_to(_ROOT)}", flush=True)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    es, engines, ref = setup()
    res = {"script": "reproduce/materials_uniaxial.py", "round": 70, "referee_item": 9,
           "psd": dict(rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL), "z_km": 20.0,
           "band_elements_um": list(map(float, es.c)), "band_floors_m1": list(map(float, es.floors)),
           "paper_reference_values": dict(calcite_tg=0.78, dolomite_tg=0.75, alumina_tg=0.150, silica_tg=ref,
                                          peak_e7=dict(calcite=0.97, dolomite=0.94, alumina=8.3))}
    part_c(res)
    part_a(es, engines, res)
    part_b(es, engines, res, ref)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"archived -> {OUT.relative_to(_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
