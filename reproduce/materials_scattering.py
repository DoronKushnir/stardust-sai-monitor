"""materials_scattering.py -- referee round 6 (paper 1; public port), item 6: the anisotropic-sphere
optics of the uniaxial materials (calcite, dolomite, alumina) stated correctly, and the
optical-constant perturbation envelopes repeated under the adopted orientation average.

 A. Rayleigh limit, per material and band element: the Clausius-Mossotti factors F_o and F_e,
    the orientation-averaged absorption factor Im(2F_o+F_e)/3 (reproduced exactly by the
    effective index of round 70), the orientation-averaged scattering factor
    (2|F_o|^2+|F_e|^2)/3 against the effective index's |(2F_o+F_e)/3|^2, and the missing
    term 2|F_o-F_e|^2/9 (the referee's counterexample).
 B. At the reference size distribution (268 nm, 1.31): the absorption and scattering parts of
    the extinction per Tg at every band element, under the effective index (convention b)
    and under the cross-section average (2 Q_o + Q_e)/3 (convention c), so that the size of
    the scattering part and the difference between the two prescriptions are stated on the
    actual distribution; the Rayleigh scattering factors are weighted by the PSD's sixth
    moment to show what fraction of the scattering the effective index misses.
 C. The +-20 % k and +-0.05 um perturbation envelopes of Appendix F4 recomputed with the
    adopted optics (convention b) for calcite, dolomite and alumina (silica is isotropic and
    its round-70 envelope stands), with the thresholds of the baseline configuration (heritage
    triplet + 47-element band, six background parameters free, measured per-element floors;
    the cached engines of outputs/detectability_2d_calibrated/).

Run from the repo root (about 30 min):  python reproduce/materials_scattering.py
Writes outputs/materials_scattering.json
"""
from __future__ import annotations

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

import materials_uniaxial as U                                    # noqa: E402
from materials_uniaxial import (RayTables, ConventionRI, cm_average, MAT_CLS, MAT_FACTORY,  # noqa: E402
                                        PEAK_UM, K_SCALES, SHIFTS_UM, setup, thresholds, summarize, prof20)
from calibrated_background_thresholds import ALT, iz, RMED_SIL_NM, SIGMA_SIL  # noqa: E402
from design_sensitivity_calibrated import N_VIS  # noqa: E402
from saimon.mie import mie_efficiencies  # noqa: E402

OUT = _ROOT / "outputs" / "materials_scattering.json"


def log(*a):
    print(*a, flush=True)


def lognormal_weights(r, rmed, sg):
    w = np.exp(-0.5 * (np.log(r / rmed) / np.log(sg)) ** 2) / r
    return w / np.trapezoid(w, r)


def psd_cross_sections(m, lam_m, r_grid, w):
    """<sigma_ext>, <sigma_sca>, <sigma_abs> [m^2] over the PSD for index m at wavelength lam_m."""
    x = 2 * np.pi * r_grid / lam_m
    qe, qs, qa = mie_efficiencies(x, complex(m))
    area = np.pi * r_grid ** 2
    return (np.trapezoid(qe * area * w, r_grid), np.trapezoid(qs * area * w, r_grid), np.trapezoid(qa * area * w, r_grid))


def main():
    t_start = time.time()
    es, engines, ref = setup()
    lam_el = es.c * 1e-6
    res = dict(script="reproduce/materials_scattering.py", round=74, referee_item=6,
               psd=dict(rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL), band_elements_um=list(map(float, es.c)), materials={})
    r_grid = np.logspace(np.log10(1e-9), np.log10(1e-5), 600)
    w = lognormal_weights(r_grid, RMED_SIL_NM * 1e-9, SIGMA_SIL)
    r6 = np.trapezoid(r_grid ** 6 * w, r_grid); r3 = np.trapezoid(r_grid ** 3 * w, r_grid)
    for mat, cls in MAT_CLS.items():
        log(f"--- {mat}")
        tabs = RayTables(cls)
        mo, me = tabs.m("o", lam_el), tabs.m("e", lam_el)
        eo, ee = mo ** 2, me ** 2
        Fo, Fe = (eo - 1) / (eo + 2), (ee - 1) / (ee + 2)
        Fm = (2 * Fo + Fe) / 3
        mb, _ = cm_average(mo, me)
        # A. Rayleigh factors
        sca_true = (2 * np.abs(Fo) ** 2 + np.abs(Fe) ** 2) / 3
        sca_eff = np.abs(Fm) ** 2
        missing = 2 * np.abs(Fo - Fe) ** 2 / 9
        abs_true = Fm.imag
        j = int(np.argmin(np.abs(es.c - PEAK_UM[mat])))
        A = dict(elements_um=list(map(float, es.c)),
                 F_o=[[float(v.real), float(v.imag)] for v in Fo], F_e=[[float(v.real), float(v.imag)] for v in Fe],
                 abs_factor_ImFmean=abs_true.tolist(), sca_factor_true=sca_true.tolist(), sca_factor_effective_index=sca_eff.tolist(),
                 sca_missing_term=missing.tolist(), sca_ratio_eff_over_true=(sca_eff / sca_true).tolist(),
                 at_peak=dict(element_um=float(es.c[j]), abs_factor=float(abs_true[j]), sca_true=float(sca_true[j]), sca_eff=float(sca_eff[j]),
                              ratio=float(sca_eff[j] / sca_true[j]), missing_over_true=float(missing[j] / sca_true[j]),
                              rayleigh_sca_over_abs_at_x_ref=float(8.0 / 3.0 * (2 * np.pi * RMED_SIL_NM * 1e-9 / lam_el[j]) ** 3 * sca_true[j] / abs_true[j] * (r6 / r3) / (RMED_SIL_NM * 1e-9) ** 3),
                              note="Rayleigh: Q_abs = 4 x Im F, Q_sca = (8/3) x^4 |F|^2; the PSD ratio uses <r^6>/<r^3> for the scattering-to-absorption weighting"))
        log(f"  Rayleigh at {es.c[j]:.2f} um: Im<F> {abs_true[j]:.4f}; scattering factor true {sca_true[j]:.4f}, effective index {sca_eff[j]:.4f} "
            f"(ratio {sca_eff[j]/sca_true[j]:.3f}; missing term {missing[j]/sca_true[j]:.1%} of the true)")
        # B. absorption / scattering at the reference PSD, conventions b and c, every element
        B = dict(elements_um=list(map(float, es.c)), b=dict(ext=[], sca=[], abs=[]), c=dict(ext=[], sca=[], abs=[]))
        for k, lam in enumerate(lam_el):
            eb, sb, ab = psd_cross_sections(mb[k], lam, r_grid, w)
            eo_, so_, ao_ = psd_cross_sections(mo[k], lam, r_grid, w)
            ee_, se_, ae_ = psd_cross_sections(me[k], lam, r_grid, w)
            for key, v in (("ext", eb), ("sca", sb), ("abs", ab)):
                B["b"][key].append(float(v))
            B["c"]["ext"].append(float((2 * eo_ + ee_) / 3)); B["c"]["sca"].append(float((2 * so_ + se_) / 3)); B["c"]["abs"].append(float((2 * ao_ + ae_) / 3))
        for conv in ("b", "c"):
            e, s, a = (np.array(B[conv][k]) for k in ("ext", "sca", "abs"))
            B[conv]["sca_fraction"] = (s / e).tolist()
        eb, sb = np.array(B["b"]["ext"]), np.array(B["b"]["sca"]); ec, sc = np.array(B["c"]["ext"]), np.array(B["c"]["sca"])
        B["ext_ratio_c_over_b"] = (ec / eb).tolist(); B["sca_ratio_c_over_b"] = (sc / sb).tolist()
        B["at_peak"] = dict(element_um=float(es.c[j]), sca_fraction_b=float(sb[j] / eb[j]), sca_fraction_c=float(sc[j] / ec[j]),
                            ext_ratio_c_over_b=float(ec[j] / eb[j]), sca_ratio_c_over_b=float(sc[j] / sb[j]),
                            abs_ratio_c_over_b=float(B["c"]["abs"][j] / B["b"]["abs"][j]))
        B["band_max_abs_ext_ratio_c_over_b_minus_1"] = float(np.max(np.abs(ec / eb - 1)))
        B["band_mean_sca_fraction_b"] = float(np.mean(sb / eb)); B["band_max_sca_fraction_b"] = float(np.max(sb / eb))
        B["note"] = ("PSD-integrated Mie cross sections [m^2 per particle] at the reference distribution: b = effective index "
                     "(Clausius-Mossotti average), c = cross-section average (2 Q_o + Q_e)/3 of Mie with each ray index; "
                     "c carries the orientation-averaged Rayleigh scattering exactly where b does not")
        log(f"  reference PSD at {es.c[j]:.2f} um: scattering fraction of the extinction {sb[j]/eb[j]:.3f} (b) / {sc[j]/ec[j]:.3f} (c); "
            f"extinction c/b {ec[j]/eb[j]:.4f}, scattering c/b {sc[j]/sb[j]:.3f}; band: max |ext c/b - 1| {B['band_max_abs_ext_ratio_c_over_b_minus_1']:.3f}, "
            f"mean/max scattering fraction {B['band_mean_sca_fraction_b']:.3f}/{B['band_max_sca_fraction_b']:.3f}")
        res["materials"][mat] = dict(rayleigh=A, reference_psd=B)
    # C. perturbation envelopes under the adopted optics (convention b)
    log("--- perturbation envelopes under convention b")
    pert = {"k_scales": list(K_SCALES), "shifts_um": list(SHIFTS_UM), "materials": {}}
    for mat in ("calcite", "dolomite", "alumina"):
        tabs = RayTables(MAT_CLS[mat])

        def make(**kw):
            return MAT_FACTORY[mat](ALT, 1.0, refractive_index=ConventionRI(tabs, "b", **kw))
        r = {}
        t0 = prof20(es, make(), f"{mat} base (b)")
        base = summarize(es, t0, mat); base["mmin_tg"] = thresholds(engines, t0); r["base"] = base
        log(f"  {mat} base: {base['mmin_tg']['quiet']:.4f} Tg")
        for lab, kw in [(f"k_x{s}", dict(kscale=s)) for s in K_SCALES] + [(f"shift_{d:+.2f}um", dict(shift_um=d)) for d in SHIFTS_UM]:
            t = prof20(es, make(**kw), f"{mat} {lab} (b)")
            s = summarize(es, t, mat); s["mmin_tg"] = thresholds(engines, t)
            s["mmin_ratio_to_base"] = {k: s["mmin_tg"][k] / base["mmin_tg"][k] for k in s["mmin_tg"]}
            s["peak_ratio_to_base"] = s["peak_ext_m1_per_Tg"] / base["peak_ext_m1_per_Tg"]
            r[lab] = s
            log(f"    {lab:>14}: {s['mmin_tg']['quiet']:.4f} Tg (x{s['mmin_ratio_to_base']['quiet']:.3f}); peak x{s['peak_ratio_to_base']:.3f}")
        pert["materials"][mat] = r
    r70 = json.load(open(_ROOT / "outputs/materials_uniaxial.json"))
    sil = r70["perturbations"]["materials"]["silica"]
    env = {m: dict(min=min(v["mmin_tg"]["quiet"] for v in pert["materials"][m].values()), max=max(v["mmin_tg"]["quiet"] for v in pert["materials"][m].values()))
           for m in pert["materials"]}
    env["silica"] = dict(min=min(v["mmin_tg"]["quiet"] for v in sil.values()), max=max(v["mmin_tg"]["quiet"] for v in sil.values()), note="round-70 (isotropic, unchanged)")
    pert["threshold_envelope_quiet_tg"] = env
    pert["carbonate_pair_envelopes_overlap"] = bool(env["dolomite"]["max"] >= env["calcite"]["min"] and env["calcite"]["max"] >= env["dolomite"]["min"])
    pert["alumina_below_calcite_under_all"] = bool(env["alumina"]["max"] < env["calcite"]["min"])
    pert["round70_envelopes_convention_a"] = r70["perturbations"]["threshold_envelope_quiet_tg"]
    res["perturbations_convention_b"] = pert
    log("envelopes (b): " + ", ".join(f"{m} {v['min']:.3f}-{v['max']:.3f}" for m, v in env.items()))
    res["runtime_s"] = time.time() - t_start
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    log("archived ->", OUT)


if __name__ == "__main__":
    main()
