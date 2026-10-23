# Ported from the working repository's scripts/sensitivity_sulfate_coating_round62.py (paper 1 rounds 62-63); paths follow the repository layout.
"""
sensitivity_sulfate_coating.py -- Mark Schoeberl (paper-1 review,
22 Oct 2026): "you mention reststrahlen as lattice vibration bands, but what
if the particle is coated with sulfate ... would that make the particle turn
into a sulfate aerosol from a remote sensing perspective?"

Answer by computation: the injected silica particles are given a concentric
sulfuric-acid shell (the calibrated background's own laboratory member,
65 wt% / 223 K) and the marginalized 3-sigma threshold of the paper-1
baseline (heritage triplet + the 7.8-13 um band at 0.1 um, or its 7.8-9.3 um
window; full 7-parameter background fit; z = 20 km) is recomputed with the
coated-particle spectrum in place of the bare-silica spectrum.  The sulfate
mass carried on the silica is scanned as a fraction MU of the silica mass
(MU = 0 is the baseline).  The threshold is still quoted per Tg of SILICA.

Coated-sphere optics: Bohren & Huffman (1983) Appendix B, BHCOAT, vectorized
over the core-radius grid; validated here against the homogeneous Mie code
of saimon (shell optics = core optics, and a vanishing shell).

Everything else (element grids, measured per-element floors, calibrated
background, marginalization) is imported from design_sensitivity_calibrated
/ calibrated_background_thresholds so the baseline reproduces the archive.

Run from the repo root:  python reproduce/sensitivity_sulfate_coating.py
Output: outputs/sulfate_coating.json (+ stdout)
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
from saimon.materials import SilicaRefractiveIndex  # noqa: E402
from saimon.sai import create_silica_sai_layer, SILICA_DENSITY_KG_M3  # noqa: E402
from saimon.sulfate_background import sulfate_solution_density_kg_m3  # noqa: E402
from saimon.mie import BohrenHuffmanMie  # noqa: E402
from saimon.psd import LognormalPSD  # noqa: E402

MU = (0.0, 0.1, 0.3, 1.0, 3.0)        # sulfate mass / silica mass
WT, TK = 65.0, 223.0                  # the calibrated member (LM65T223)
R_GRID = np.logspace(np.log10(30e-9), np.log10(3e-6), 400)
OUT = _ROOT / "outputs" / "sulfate_coating.json"


def bhcoat(x, y, m1, m2, nmax=None, tol=1e-8):
    """Q_ext, Q_sca of a coated sphere (BH83 BHCOAT) for arrays x (core
    size parameter) and y (outer size parameter) with scalar refractive
    indices m1 (core), m2 (shell) relative to the medium."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    y = np.where(y <= x, x * (1 + 1e-9), y)
    nstop = np.floor(y + 4 * y ** (1 / 3) + 2)      # per-element series length (BH83)
    if nmax is None:
        nmax = int(np.max(nstop)) + 1
    x1, x2, y2 = m1 * x, m2 * x, m2 * y
    d0x1 = np.cos(x1) / np.sin(x1)
    d0x2 = np.cos(x2) / np.sin(x2)
    d0y2 = np.cos(y2) / np.sin(y2)
    psi0y, psi1y = np.cos(y), np.sin(y)
    chi0y, chi1y = -np.sin(y), np.cos(y)
    xi1y = psi1y - 1j * chi1y
    chi0y2, chi1y2 = -np.sin(y2), np.cos(y2)
    chi0x2, chi1x2 = -np.sin(x2), np.cos(x2)
    qext = np.zeros_like(x); qsca = np.zeros_like(x)
    active = np.ones(x.shape, bool)       # IFLAG == 0
    d1x1 = d1x2 = np.zeros_like(x1)
    err = np.seterr(all="ignore")       # orders beyond an element's own nstop are frozen below
    for n in range(1, nmax + 1):
        rn = float(n)
        upd = nstop >= n                 # elements still summing at this order
        psiy = (2 * rn - 1) * psi1y / y - psi0y
        chiy = (2 * rn - 1) * chi1y / y - chi0y
        xiy = psiy - 1j * chiy
        d1y2 = 1 / (rn / y2 - d0y2) - rn / y2
        d1x1 = 1 / (rn / x1 - d0x1) - rn / x1
        d1x2 = 1 / (rn / x2 - d0x2) - rn / x2
        chix2 = (2 * rn - 1) * chi1x2 / x2 - chi0x2
        chiy2 = (2 * rn - 1) * chi1y2 / y2 - chi0y2
        chipx2 = chi1x2 - rn * chix2 / x2
        chipy2 = chi1y2 - rn * chiy2 / y2
        ancap = (m2 * d1x1 - m1 * d1x2)
        ancap = ancap / (m2 * d1x1 * chix2 - m1 * chipx2)
        ancap = ancap / (chix2 * d1x2 - chipx2)
        brack = ancap * (chiy2 * d1y2 - chipy2)
        bncap = (m2 * d1x2 - m1 * d1x1)
        bncap = bncap / (m2 * chipx2 - m1 * d1x1 * chix2)
        bncap = bncap / (chix2 * d1x2 - chipx2)
        crack = bncap * (chiy2 * d1y2 - chipy2)
        a1, a2 = brack * chipy2, brack * chiy2
        a3, a4 = crack * chipy2, crack * chiy2
        conv = ((np.abs(a1) < tol * np.abs(d1y2)) & (np.abs(a2) < tol)
                & (np.abs(a3) < tol * np.abs(d1y2)) & (np.abs(a4) < tol))
        bad = ~(np.isfinite(a1) & np.isfinite(a2) & np.isfinite(a3) & np.isfinite(a4))
        active = active & ~conv & ~bad
        brack = np.where(active, brack, 0.0)
        crack = np.where(active, crack, 0.0)
        dnbar = (d1y2 - brack * chipy2) / (1 - brack * chiy2)
        gnbar = (d1y2 - crack * chipy2) / (1 - crack * chiy2)
        an = ((dnbar / m2 + rn / y) * psiy - psi1y) / ((dnbar / m2 + rn / y) * xiy - xi1y)
        bn = ((m2 * gnbar + rn / y) * psiy - psi1y) / ((m2 * gnbar + rn / y) * xiy - xi1y)
        qsca = np.where(upd, qsca + (2 * rn + 1) * (np.abs(an) ** 2 + np.abs(bn) ** 2), qsca)
        qext = np.where(upd, qext + (2 * rn + 1) * (an + bn).real, qext)
        psi0y, psi1y = np.where(upd, psi1y, psi0y), np.where(upd, psiy, psi1y)
        chi0y, chi1y = np.where(upd, chi1y, chi0y), np.where(upd, chiy, chi1y)
        xi1y = psi1y - 1j * chi1y
        chi0x2, chi1x2 = np.where(upd, chi1x2, chi0x2), np.where(upd, chix2, chi1x2)
        chi0y2, chi1y2 = np.where(upd, chi1y2, chi0y2), np.where(upd, chiy2, chi1y2)
        d0x1 = np.where(upd, d1x1, d0x1); d0x2 = np.where(upd, d1x2, d0x2); d0y2 = np.where(upd, d1y2, d0y2)
    np.seterr(**err)
    assert np.all(np.isfinite(qext)), "BHCOAT produced non-finite Q_ext"
    return qext * 2 / y ** 2, qsca * 2 / y ** 2


class CoatedSilicaLayer:
    """The paper's silica SAI layer (same N0(z) per Tg of silica) with each
    core of radius r carrying a sulfate shell of volume v_shell = V_RATIO *
    V_core; exposes extinction_profile_m1(wl_m) like a SulfateAerosolColumn."""

    def __init__(self, base_layer, ri_core, ri_shell, v_ratio, rmed_nm, sigma):
        self.n0 = np.asarray(base_layer.modes[0].n0_m3_profile, float)
        self.ri_core, self.ri_shell, self.v = ri_core, ri_shell, float(v_ratio)
        self.psd = LognormalPSD(rmed_nm * 1e-9, sigma, 1.0)

    def cross_sections_m2(self, wl_m):
        wl_m = np.atleast_1d(np.asarray(wl_m, float))
        r = R_GRID
        r_out = r * (1 + self.v) ** (1 / 3)
        dn = self.psd.dn_dr(r)
        out = np.empty(len(wl_m))
        mc = np.asarray(self.ri_core(wl_m), complex)
        ms = np.asarray(self.ri_shell(wl_m), complex)
        for j, lam in enumerate(wl_m):
            x = 2 * np.pi * r / lam
            y = 2 * np.pi * r_out / lam
            qe, _ = bhcoat(x, y, complex(mc[j]), complex(ms[j]))
            out[j] = np.trapezoid(qe * np.pi * r_out ** 2 * dn, r)
        return out

    def extinction_profile_m1(self, wl_m):
        c = self.cross_sections_m2(wl_m)
        return self.n0[:, None] * c[None, :]


def validate(ri_sil, ri_sulf):
    """BHCOAT against the homogeneous Mie code: identical optics in core and
    shell must give the homogeneous Q_ext of the outer sphere; a vanishing
    shell must give the core's."""
    mie = BohrenHuffmanMie()
    worst = 0.0
    for lam in (0.448e-6, 1.544e-6, 8.8e-6, 12.0e-6):
        for m in (complex(ri_sil(np.array([lam]))[0]), complex(ri_sulf(np.array([lam]))[0])):
            x = 2 * np.pi * R_GRID / lam
            y = x * 1.3
            q_ref_y, _, _ = mie.efficiencies(y, m)
            q_ref_x, _, _ = mie.efficiencies(x, m)
            q_same, _ = bhcoat(x, y, m, m)
            q_thin, _ = bhcoat(x, x * (1 + 1e-7), m, m * 1.1)
            worst = max(worst, float(np.max(np.abs(q_same / q_ref_y - 1))),
                        float(np.max(np.abs(q_thin / q_ref_x - 1))))
    return worst


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
    ri_kp = SilicaRefractiveIndex("data/optics/SiO2_KitamuraPopova.yml")
    base = create_silica_sai_layer(ALT, 1.0, refractive_index=ri_kp,
                                   rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL)
    rho_sulf = sulfate_solution_density_kg_m3(WT, TK)
    worst = validate(ri_kp, ri65)
    print(f"BHCOAT validation against homogeneous Mie: worst relative deviation {worst:.2e}")
    assert worst < 1e-3, worst   # series truncated at each element's own BH83 nstop

    arch = {"mu_sulfate_over_silica_mass": list(MU), "rho_sulfate_kg_m3": rho_sulf,
            "rho_silica_kg_m3": SILICA_DENSITY_KG_M3, "sets": {}, "peak_um": {},
            "shell_nm_at_rmed": {}, "ext_8p80_per_tg_rel": {}}
    lam_peak = np.arange(8.3, 9.31, 0.01) * 1e-6
    layers = {}
    for mu in MU:
        v = mu * SILICA_DENSITY_KG_M3 / rho_sulf
        lay = CoatedSilicaLayer(base, ri_kp, ri65, v, RMED_SIL_NM, SIGMA_SIL) if mu > 0 else base
        layers[mu] = lay
        e = lay.extinction_profile_m1(lam_peak)[iz, :]
        arch["peak_um"][str(mu)] = float(lam_peak[np.argmax(e)] * 1e6)
        arch["shell_nm_at_rmed"][str(mu)] = float(RMED_SIL_NM * ((1 + v) ** (1 / 3) - 1))
    e0 = base.extinction_profile_m1(np.array([8.8e-6]))[iz, 0]
    e0_525 = base.extinction_profile_m1(np.array([525e-9]))[iz, 0]
    for mu in MU:
        e = layers[mu].extinction_profile_m1(np.array([8.8e-6, 525e-9]))[iz, :]
        arch["ext_8p80_per_tg_rel"][str(mu)] = float(e[0] / e0)
        arch.setdefault("ext_525_per_tg_rel", {})[str(mu)] = float(e[1] / e0_525)
    print("shell thickness at r_med [nm]:", {k: round(v, 1) for k, v in arch["shell_nm_at_rmed"].items()})
    print("reststrahlen peak [um]:        ", {k: round(v, 3) for k, v in arch["peak_um"].items()})
    print("8.80-um extinction per Tg silica, relative to bare:", {k: round(v, 3) for k, v in arch["ext_8p80_per_tg_rel"].items()})
    print("525-nm extinction per Tg silica, relative to bare: ", {k: round(v, 3) for k, v in arch["ext_525_per_tg_rel"].items()})

    for es in sets:
        exts, ders = es.background(CAL_QUIET, ri65, wt65, T65)
        res = {}
        for mu in MU:
            m, tperp, r2 = es.mmin(exts, ders, es.prof(layers[mu]))
            res[str(mu)] = {"mmin_tg": float(m), "tperp": float(tperp), "one_minus_R2": float(r2)}
        m0 = res["0.0"]["mmin_tg"]
        assert abs(m0 - ref[es.name]) < 1e-3, (m0, ref[es.name])
        print(f"\n=== {es.name} (baseline {m0:.4f} Tg, archived {ref[es.name]:.4f}) ===")
        for mu in MU:
            r = res[str(mu)]
            print(f"  mu = {mu:<4}: M_min = {r['mmin_tg']:.4f} Tg  ({r['mmin_tg']/m0-1:+.1%}), "
                  f"||t_perp|| {r['tperp']:.1f}, 1-R2 {r['one_minus_R2']:.3f}")
        arch["sets"][es.name] = res
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(arch, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
