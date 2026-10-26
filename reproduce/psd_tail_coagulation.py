"""
psd_tail_coagulation.py -- referee round 5, item 9 (paper 1):
(A) the large-particle tail removed by the 4-um truncation of the measured
    Segev et al. (2026) dispersions (D300o 1.4 %, D300 10 %, D500 11 % of the
    mass): its effect on the per-Tg 8.80-um extinction and on the band
    threshold at the same total mass;
(B) the settling velocity of the tail particles at 20 km (slip-corrected
    Stokes, the formula of reproduce/settling_velocity_stratosphere.py, which
    archives no numbers);
(C) the Brownian self-coagulation bookkeeping of
    reproduce/verify_particle_number_density.py item 7 (Fuchs-interpolated
    kernel, Seinfeld & Pandis 2016 Eq. 13.56 / Table 13.1; Fuchs 1964),
    reproduced and extended to local release concentrations, the dilution
    history under which the uniform estimate applies, and heterocoagulation
    with the sulfate background (silica 268 nm against 60-nm droplets).

Tail representation (A): the measured cumulative mass distribution above
4 um, binned at the measured points (D300/D500: consecutive digitized points
of data/digitized/Segev2026_Fig2b.csv; D300o: the bins of Segev2026_CDF_best.xlsx), mass
uniform in ln D within a bin (3 sub-points); the alternative puts the whole
removed fraction flat in ln D over 4-10 um.  Sphere extinction per unit mass
3 Q_ext / (4 r rho); Q_ext = 2 for x > 300 (geometric limit; the Mie series is
not evaluated there).  The mixture per-Tg extinction at 20 km is
(1-f) t_LN + f C20 kappa_tail with C20 the 1-Tg layer's mass concentration
at 20 km and t_LN the band engine's per-Tg vector of the truncated lognormal.

Archive: outputs/psd_tail_coagulation.json
Run from the repo root:  python reproduce/psd_tail_coagulation.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from calibrated_background_thresholds import (  # noqa: E402
    ALT, iz, RMED_SIL_NM, SIGMA_SIL, CAL_QUIET, TwoComponentBackground,
    glossac_2025N_quiet_profile, member_ri)
from design_sensitivity_calibrated import ElementSet, VIS_NM, N_VIS  # noqa: E402
from detectability_2d_calibrated import Engine, BRACKETS  # noqa: E402
import materials_calibrated_thresholds as MCT  # noqa: E402
from saimon.materials import SilicaRefractiveIndex  # noqa: E402
from saimon.sai import SILICA_DENSITY_KG_M3, create_silica_sai_layer  # noqa: E402
from saimon.mie import mie_efficiencies  # noqa: E402
from saimon.atmosphere import us_standard_atmosphere  # noqa: E402
from saimon.psd import lognormal_volume_per_particle  # noqa: E402

OUT = _ROOT / "outputs" / "psd_tail_coagulation.json"
CSV = _ROOT / "data" / "digitized" / "Segev2026_Fig2b.csv"
XLSX = _ROOT / "data" / "digitized" / "Segev2026_CDF_best.xlsx"
DCUT = 4.0           # um, truncation diameter (fit_segev_psd.py)
D_FLAT = (4.0, 10.0)  # um, the alternative tail placement
X_GEOM = 300.0        # Q_ext = 2 beyond this size parameter
RHO_DROP = 1700.0     # kg m^-3, sulfate droplet (paper App. A)
R_SULF_NM = 60.0      # fine-mode median of the quiet calibration (CAL_QUIET)
YR = 365.25 * 86400.0
DAY = 86400.0
PLUME_FACTORS = (1e3, 1e4, 1e5, 1e6)


# ---------------------------------------------------------------------------
# measured dispersions (loaders copied from reproduce/fit_segev_psd.py, which
# cannot be imported without re-plotting the paper figure)
# ---------------------------------------------------------------------------
def load_csv(col_x, col_y):
    raw = np.genfromtxt(CSV, delimiter=',', skip_header=2)
    x, y = raw[:, col_x], raw[:, col_y]
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    o = np.argsort(x)
    return x[o], np.maximum.accumulate(y[o])


def load_xlsx_bins(path):
    """D300o bins: (dmin, dmax, fraction) rows and the CDF at dmax."""
    M = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    z = zipfile.ZipFile(path)
    root = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))

    def colnum(ref):
        mm = re.match(r'([A-Z]+)(\d+)', ref); col = 0
        for ch in mm.group(1):
            col = col * 26 + (ord(ch) - 64)
        return col - 1, int(mm.group(2))
    rows = {}
    for c in root.iter(M + 'c'):
        v = c.find(M + 'v')
        if v is None or c.get('t') == 's':
            continue
        ci, ri = colnum(c.get('r'))
        rows.setdefault(ri, {})[ci] = float(v.text)
    rr = sorted(r for r in rows if r > 1)
    dmin = np.array([rows[r][0] for r in rr]); dmax = np.array([rows[r][1] for r in rr])
    frac = np.array([rows[r][2] for r in rr]); F = np.array([rows[r][3] for r in rr])
    o = np.argsort(dmax)
    return dmin[o], dmax[o], frac[o], np.maximum.accumulate(F[o])


def lognormal_cdf(D, Dmed, sigma_g):
    return norm.cdf(np.log(D / Dmed) / np.log(sigma_g))


def truncated_fit(D, F):
    F = F / F[-1]
    cdf_cut = float(np.interp(DCUT, D, F))
    sel = D <= DCUT
    p, _ = curve_fit(lognormal_cdf, D[sel], F[sel] / cdf_cut, p0=[0.5, 1.4], maxfev=20000)
    return dict(rmed_nm=float(p[0] / 2 * 1000), sigma=float(p[1]), mass_removed=1.0 - cdf_cut)


def tail_bins_from_points(D, F):
    """Bins [D_i, D_i+1] above DCUT from a digitized CDF (normalized), mass = dF."""
    F = F / F[-1]
    F_cut = np.interp(DCUT, D, F)
    sel = D > DCUT
    edges = np.r_[DCUT, D[sel]]
    Fe = np.r_[F_cut, F[sel]]
    dm = np.diff(Fe)
    keep = dm > 0
    return edges[:-1][keep], edges[1:][keep], dm[keep]


def tail_bins_from_xlsx(dmin, dmax, frac):
    lo, hi, m = [], [], []
    for a, b, f in zip(dmin, dmax, frac):
        if b <= DCUT or f <= 0:
            continue
        if a < DCUT:   # straddling bin: split uniformly in ln D
            f = f * np.log(b / DCUT) / np.log(b / a)
            a = DCUT
        lo.append(a); hi.append(b); m.append(f)
    return np.asarray(lo), np.asarray(hi), np.asarray(m)


# ---------------------------------------------------------------------------
# tail optics
# ---------------------------------------------------------------------------
def kappa_tail(lo_um, hi_um, mass, wl_m, m_of_wl, rho, nsub=3):
    """Mass-specific extinction [m^2 kg^-1] of the tail (mass fractions `mass`
    summing to the tail fraction; returned per unit tail mass) on wavelengths
    wl_m with complex index m_of_wl (array like wl_m)."""
    mass = np.asarray(mass, float)
    w_tot = mass.sum()
    kap = np.zeros(len(wl_m))
    n_mie = 0
    for a, b, f in zip(lo_um, hi_um, mass):
        d_sub = np.exp(np.linspace(np.log(a), np.log(b), 2 * nsub + 1))[1::2]   # bin centres in ln D
        for d in d_sub:
            r = 0.5 * d * 1e-6
            x = 2 * np.pi * r / wl_m
            q = np.full(len(wl_m), 2.0)
            sm = x <= X_GEOM
            if sm.any():
                for j in np.where(sm)[0]:
                    q[j] = mie_efficiencies(np.array([x[j]]), complex(m_of_wl[j]))[0][0]
                    n_mie += 1
            kap += (f / nsub / w_tot) * 3.0 * q / (4.0 * r * rho)
    return kap, n_mie


def band_vector_tail(es, lo, hi, mass, ri, rho, nsub_wl=9):
    """Per-unit-mass band vector (triplet + boxcar elements) of the tail."""
    sub = np.linspace(-es.w / 2, es.w / 2, nsub_wl)
    wl_vis = VIS_NM * 1e-9
    k_vis, n1 = kappa_tail(lo, hi, mass, wl_vis, ri(wl_vis), rho)
    wl_mir = (es.c[:, None] + sub[None, :]).ravel() * 1e-6
    k_mir, n2 = kappa_tail(lo, hi, mass, wl_mir, ri(wl_mir), rho)
    k_mir = k_mir.reshape(len(es.c), nsub_wl).mean(axis=1)
    return np.r_[k_vis, k_mir], n1 + n2


# ---------------------------------------------------------------------------
def part_a(res):
    print("PART A: PSD tail", flush=True)
    thr = json.load(open(MCT.THR_JSON))
    es = ElementSet("band 8-13 @0.1", thr["band01_elements_um"], 0.10, thr["band01_floors_m1"])
    engines = {}
    for bname, (comps, member) in BRACKETS.items():
        cache = MCT.ENGINE_DIR / f"engine_band_{bname}.npz"
        engines[bname] = Engine(es, comps, member, None, None, cache=cache)
    ref = thr["results"]["CALIBRATED quiet (LM65T223, 2-comp)"]["band 8-13 @0.1: measured floors"]["full 7-param"]["triplet + band01"]["mmin_tg"]
    ri = SilicaRefractiveIndex(str(_ROOT / "data/optics/SiO2_KitamuraPopova.yml"))
    j880 = int(np.argmin(np.abs(es.c - 8.80)))
    rho = SILICA_DENSITY_KG_M3

    cases = {}
    D, F = load_csv(0, 1); cases["D300"] = dict(fit=truncated_fit(D, F), bins=tail_bins_from_points(D, F), Dmax=float(D.max()))
    D, F = load_csv(2, 3); cases["D500"] = dict(fit=truncated_fit(D, F), bins=tail_bins_from_points(D, F), Dmax=float(D.max()))
    dmin, dmax, frac, Fx = load_xlsx_bins(XLSX)
    cases["D300o"] = dict(fit=truncated_fit(dmax, Fx), bins=tail_bins_from_xlsx(dmin, dmax, frac), Dmax=float(dmax.max()))

    def thresholds(t):
        out = {}
        for bname, eng in engines.items():
            eng.t20 = t
            out[bname] = float(3 * eng.sigma(iz))
        return out

    res["tail"] = dict(method=__doc__.split("Tail representation (A):")[1].split("Archive:")[0].strip(),
                       x_geometric_cutoff=X_GEOM, element_880_um=float(es.c[j880]), cases={})
    for name in ("D300o", "D300", "D500"):
        c = cases[name]
        fit = c["fit"]; lo, hi, mass = c["bins"]
        f_tail = float(mass.sum())
        f_4_10 = float(sum(mi * (np.log(min(b, 10.0) / a) / np.log(b / a)) for a, b, mi in zip(lo, hi, mass) if a < 10.0))
        print(f"  {name}: truncated fit r_med {fit['rmed_nm']:.0f} nm, sigma {fit['sigma']:.3f}; removed {fit['mass_removed']*100:.2f} % "
              f"(binned {f_tail*100:.2f} %; of which 4-10 um {f_4_10*100:.2f} %, >10 um {(f_tail-f_4_10)*100:.2f} %); "
              f"{len(lo)} bins to {c['Dmax']:.0f} um", flush=True)
        lay = create_silica_sai_layer(ALT, 1.0, refractive_index=ri, rmed_nm=fit["rmed_nm"], sigma=fit["sigma"])
        t0 = time.time()
        t_ln = es.prof(lay)[iz]
        print(f"    lognormal band vector: {time.time() - t0:.0f} s", flush=True)
        C20 = float(lay.mass_concentration_kg_m3()[iz])          # kg m^-3 per Tg at 20 km
        # consistency: t = C20 * <sigma>/(rho <V>)  (per-Tg extinction = mass concentration x mass-specific extinction)
        thr_ln = thresholds(t_ln)
        r = dict(fit=fit, mass_removed_binned=f_tail, mass_4_10um=f_4_10, mass_above_10um=f_tail - f_4_10,
                 n_bins=int(len(lo)), D_max_um=c["Dmax"], C20_kg_m3_per_Tg=C20,
                 bins=dict(lo_um=list(map(float, lo)), hi_um=list(map(float, hi)), mass=list(map(float, mass))),
                 lognormal=dict(t880_m1_per_Tg=float(t_ln[N_VIS + j880]), t_band=list(map(float, t_ln[N_VIS:])),
                                t_vis=list(map(float, t_ln[:N_VIS])), mmin_tg=thr_ln))
        for variant, (vlo, vhi, vm) in (("measured_tail", (lo, hi, mass)),
                                       ("flat_4_10um", (np.array([D_FLAT[0]]), np.array([D_FLAT[1]]), np.array([f_tail])))):
            t0 = time.time()
            nsub = 3 if variant == "measured_tail" else 12
            kap, n_mie = band_vector_tail(es, vlo, vhi, vm, ri, rho) if variant == "measured_tail" else \
                band_vector_tail_flat(es, vlo[0], vhi[0], ri, rho, nsub)
            t_tail = C20 * kap                                     # per Tg if ALL the mass were in the tail
            t_mix = (1 - f_tail) * t_ln + f_tail * t_tail
            thr_mix = thresholds(t_mix)
            d = dict(kappa_880_m2_per_kg=float(kap[N_VIS + j880]), kappa_vis_m2_per_kg=list(map(float, kap[:N_VIS])),
                     kappa_band_m2_per_kg=list(map(float, kap[N_VIS:])),
                     t880_tail_only_m1_per_Tg=float(t_tail[N_VIS + j880]),
                     t880_mix_m1_per_Tg=float(t_mix[N_VIS + j880]),
                     t880_change_percent=float((t_mix[N_VIS + j880] / t_ln[N_VIS + j880] - 1) * 100),
                     t_band_mix=list(map(float, t_mix[N_VIS:])), t_vis_mix=list(map(float, t_mix[:N_VIS])),
                     first_order_mmin_tg={k: v * t_ln[N_VIS + j880] / t_mix[N_VIS + j880] for k, v in thr_ln.items()},
                     mmin_tg=thr_mix, mmin_ratio_to_lognormal={k: thr_mix[k] / thr_ln[k] for k in thr_mix},
                     n_mie_evaluations=int(n_mie), seconds=float(time.time() - t0))
            r[variant] = d
            print(f"    {variant:>14}: t880 {t_ln[N_VIS+j880]:.3e} -> {t_mix[N_VIS+j880]:.3e} m^-1/Tg ({d['t880_change_percent']:+.2f} %); "
                  f"threshold {thr_ln['quiet']:.4f} -> {thr_mix['quiet']:.4f} Tg (x{d['mmin_ratio_to_lognormal']['quiet']:.4f}); "
                  f"first-order {d['first_order_mmin_tg']['quiet']:.4f} Tg", flush=True)
        res["tail"]["cases"][name] = r
    res["tail"]["silica_reference_mmin_tg"] = ref
    res["tail"]["D300o_reproduces_baseline"] = bool(abs(res["tail"]["cases"]["D300o"]["lognormal"]["mmin_tg"]["quiet"] - ref) < 1e-3)
    # mass-specific extinction of the baseline lognormal at 8.80 um, for comparison with the tail's
    V = lognormal_volume_per_particle(RMED_SIL_NM * 1e-9, SIGMA_SIL)
    c = res["tail"]["cases"]["D300o"]
    res["tail"]["kappa_880_lognormal_D300o_m2_per_kg"] = c["lognormal"]["t880_m1_per_Tg"] / c["C20_kg_m3_per_Tg"]
    res["tail"]["mean_volume_m3_D300o"] = float(V)


def band_vector_tail_flat(es, lo_um, hi_um, ri, rho, nsub):
    """Flat-in-ln-D tail between lo and hi (all of the removed mass)."""
    edges = np.exp(np.linspace(np.log(lo_um), np.log(hi_um), nsub + 1))
    lo, hi = edges[:-1], edges[1:]
    mass = np.full(nsub, 1.0 / nsub)
    return band_vector_tail(es, lo, hi, mass, ri, rho)


# ---------------------------------------------------------------------------
def air(z_km):
    atm = us_standard_atmosphere(np.array([z_km * 1000.0]))
    T = float(atm.temperature_k[0]); P = float(atm.pressure_pa[0])
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
    lam = (mu / P) * np.sqrt(np.pi * 8.314 * T / (2 * 0.028964))
    g = 9.80665 * (6.371e6 / (6.371e6 + z_km * 1000.0)) ** 2
    return T, P, mu, lam, g


def v_terminal(Dp, mu, lam, g, rho_p=SILICA_DENSITY_KG_M3, rho_a=0.09):
    Kn = 2.0 * lam / Dp
    Cc = 1.0 + Kn * (1.257 + 0.4 * np.exp(-1.1 / Kn))
    return Dp ** 2 * (rho_p - rho_a) * g * Cc / (18.0 * mu)


def part_b(res):
    print("PART B: settling of the tail (reproduce/settling_velocity_stratosphere.py formula)", flush=True)
    T, P, mu, lam, g = air(20.0)
    zs = np.linspace(16.0, 20.0, 401)
    props = [air(z) for z in zs]
    out = dict(formula="slip-corrected Stokes, Seinfeld & Pandis 2016: v = D^2 rho g Cc/(18 mu), Cc = 1 + Kn (1.257 + 0.4 exp(-1.1/Kn)), Kn = 2 lambda/D; "
               "US Standard Atmosphere T, P; rho_p = 2200 kg m^-3",
               T20_K=T, P20_Pa=P, mu_Pa_s=mu, lambda_20km_um=lam * 1e6, diameters={})
    for D_um in (0.536, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0, 50.0):
        v20 = float(v_terminal(D_um * 1e-6, mu, lam, g))
        vz = np.array([v_terminal(D_um * 1e-6, m_, l_, g_) for (_, _, m_, l_, g_) in props])
        t_16_20 = float(np.trapezoid(1.0 / vz, zs * 1000.0))       # s to fall from 20 to 16 km
        out["diameters"][f"{D_um:g}"] = dict(v20_m_per_s=v20, v20_m_per_day=v20 * DAY, v20_km_per_yr=v20 * YR / 1e3,
                                             t_1km_shell_days=1000.0 / v20 / DAY, t_20_to_16km_days=t_16_20 / DAY)
        print(f"  D = {D_um:5g} um: v(20 km) = {v20*DAY:8.1f} m/day; 1-km shell {1000/v20/DAY:9.2f} d; 20->16 km {t_16_20/DAY:9.2f} d", flush=True)
    res["settling"] = out


# ---------------------------------------------------------------------------
def fuchs_kernel(r1, r2, rho1, rho2, T, p):
    """Fuchs-interpolated Brownian coagulation kernel [m^3 s^-1] between
    spheres r1 (density rho1) and r2 (rho2), arrays broadcastable; the form
    of verify_particle_number_density.py item 7 (Seinfeld & Pandis 2016,
    Eq. 13.56 / Table 13.1; Fuchs 1964), generalized to two densities."""
    k_b = 1.380649e-23
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
    lam = 0.0651e-6 * (101325.0 / p) * (T / 293.15)

    def single(r, rho):
        kn = lam / r
        cc = 1.0 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn))
        diff = k_b * T * cc / (6 * np.pi * mu * r)
        mass = rho * 4.0 / 3.0 * np.pi * r ** 3
        cbar = np.sqrt(8 * k_b * T / (np.pi * mass))
        ell = 8 * diff / (np.pi * cbar)
        gg = ((2 * r + ell) ** 3 - (4 * r ** 2 + ell ** 2) ** 1.5) / (6 * r * ell) - 2 * r
        return diff, cbar, gg
    d1, c1, g1 = single(r1, rho1); d2, c2, g2 = single(r2, rho2)
    R = r1 + r2; Dsum = d1 + d2
    beta = 1.0 / (R / (R + np.sqrt(g1 ** 2 + g2 ** 2)) + 4 * Dsum / (np.sqrt(c1 ** 2 + c2 ** 2) * R))
    return 4 * np.pi * Dsum * R * beta, dict(mu=mu, lam=lam)


def part_c(res):
    print("PART C: coagulation bookkeeping (item 7 of verify_particle_number_density.py)", flush=True)
    z = np.arange(0.0, 60_000.0 + 1.0, 10.0)
    atm = us_standard_atmosphere(z)
    i_ref = int(np.argmin(np.abs(z - 20_000.0)))
    T20 = float(np.asarray(atm.temperature_k)[i_ref]); p20 = float(np.asarray(atm.pressure_pa)[i_ref])
    # item 2 of that script: the number density at the 0.1-Tg threshold at 20 km
    # from the 1D layer built on the same 10-m altitude grid (the mass closure
    # of saimon.sai is a trapezoid on the layer's own grid; on the 500-m ALT
    # grid of the threshold engines the 20-km number density is ~2 % lower)
    ri_kp = SilicaRefractiveIndex(str(_ROOT / "data/optics/SiO2_KitamuraPopova.yml"))
    lay = create_silica_sai_layer(z, 1.0, rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL, refractive_index=ri_kp)
    n0_20_per_tg = float(lay.number_density_m3()[i_ref])          # m^-3 per Tg (item 1 of that script)
    n_01 = 0.1 * n0_20_per_tg * 1e-6                                # cm^-3 at 0.1 Tg
    lay_alt = create_silica_sai_layer(ALT, 1.0, rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL, refractive_index=ri_kp)
    n0_alt_grid = float(lay_alt.number_density_m3()[iz])
    # PSD grid and weights exactly as item 7
    r_grid = np.exp(np.linspace(np.log(RMED_SIL_NM * 1e-9) - 4 * np.log(SIGMA_SIL),
                                np.log(RMED_SIL_NM * 1e-9) + 4 * np.log(SIGMA_SIL), 161))
    w = np.exp(-0.5 * (np.log(r_grid / (RMED_SIL_NM * 1e-9)) / np.log(SIGMA_SIL)) ** 2); w /= w.sum()
    K, aux = fuchs_kernel(r_grid[:, None], r_grid[None, :], SILICA_DENSITY_KG_M3, SILICA_DENSITY_KG_M3, T20, p20)
    K_mean = float((w[:, None] * w[None, :] * K).sum())
    i_med = int(np.argmin(np.abs(r_grid - RMED_SIL_NM * 1e-9)))
    t_half_yr = 2.0 / (K_mean * n_01 * 1e6) / YR
    out = dict(kernel="Fuchs-interpolated Brownian kernel (Seinfeld & Pandis 2016, Eq. 13.56 / Table 13.1; Fuchs 1964), "
                      "Cunningham slip 1.257/0.4/1.1, Sutherland viscosity, mean free path 65.1 nm x (101325/p)(T/293.15); "
                      "number-weighted average over the D300o lognormal (161-point grid, +-4 ln sigma)",
               T20_K=T20, p20_Pa=p20, mu_Pa_s=aux["mu"], lambda_um=aux["lam"] * 1e6,
               K_rmed_cm3_s=float(K[i_med, i_med]) * 1e6, K_psd_mean_cm3_s=K_mean * 1e6,
               n0_20km_per_Tg_m3=n0_20_per_tg, n0_20km_per_Tg_m3_on_ALT_grid=n0_alt_grid,
               n_threshold_0p1Tg_cm3=n_01,
               halving_time_uniform_yr=t_half_yr,
               item7_archive_values=dict(K_psd_mean_cm3_s=2.9361050202586667e-09, n_threshold_cm3=0.09323448347053553, halving_yr=231.51432797936056,
                                         source="outputs/particle_number_density.csv"),
               halving_rule="dN/dt = -(1/2) K N^2 -> t_1/2 = 2/(K N); the mass is conserved, the number halves")
    print(f"  T = {T20:.2f} K, p = {p20:.0f} Pa; K_psd = {K_mean*1e6:.3e} cm^3/s; N(0.1 Tg) = {n_01:.4f} cm^-3; t_1/2 = {t_half_yr:.1f} yr", flush=True)
    # local release concentrations
    plume = {}
    for fac in PLUME_FACTORS:
        N = n_01 * fac
        th = 2.0 / (K_mean * N * 1e6)
        tau_d_10pct = th / 9.0        # exponential dilution e-folding time for < 10 % number loss
        plume[f"{fac:.0e}"] = dict(N_cm3=N, halving_time_s=th, halving_time_hours=th / 3600.0, halving_time_days=th / DAY,
                                   dilution_efolding_for_10pct_loss_hours=tau_d_10pct / 3600.0,
                                   dilution_efolding_for_10pct_loss_days=tau_d_10pct / DAY)
        print(f"  plume x{fac:.0e}: N = {N:9.1f} cm^-3; t_1/2 = {th/3600:10.2f} h = {th/DAY:8.3f} d; "
              f"dilution e-folding < {tau_d_10pct/3600:8.2f} h for < 10 % loss", flush=True)
    out["local_plume"] = plume
    out["dilution_rule"] = ("for N(t) = N0 exp(-t/tau_d) the number fraction lost to self-coagulation is "
                            "(K N0 tau_d/2)/(1 + K N0 tau_d/2), i.e. < 10 % requires tau_d < t_1/2(N0)/9; "
                            "the uniform-dilution estimate applies only if the plume dilutes to the global density within "
                            "a time short compared with the local halving time")
    # heterocoagulation with the sulfate background
    r_s = R_SULF_NM * 1e-9
    K_het_grid, _ = fuchs_kernel(r_grid, r_s, SILICA_DENSITY_KG_M3, RHO_DROP, T20, p20)
    K_het_mean = float((w * K_het_grid).sum())
    K_het_med = float(K_het_grid[i_med])
    # the sulfate fine mode's own width (CAL_QUIET: 60 nm, sigma 1.6): PSD-averaged kernel over both
    rs_grid = np.exp(np.linspace(np.log(r_s) - 4 * np.log(1.6), np.log(r_s) + 4 * np.log(1.6), 161))
    ws = np.exp(-0.5 * (np.log(rs_grid / r_s) / np.log(1.6)) ** 2); ws /= ws.sum()
    K2, _ = fuchs_kernel(r_grid[:, None], rs_grid[None, :], SILICA_DENSITY_KG_M3, RHO_DROP, T20, p20)
    K_het_both = float((w[:, None] * ws[None, :] * K2).sum())
    het = dict(sulfate_radius_nm=R_SULF_NM, sulfate_density_kg_m3=RHO_DROP,
               K_268nm_60nm_cm3_s=K_het_med * 1e6, K_psd_mean_silica_cm3_s=K_het_mean * 1e6,
               K_psd_mean_both_sigma1p6_cm3_s=K_het_both * 1e6, lifetimes={})
    # the calibrated quiet background's own number densities at 20 km
    n_bg = []
    try:
        ri, wt, T = member_ri("LM65T223")
        bg = TwoComponentBackground(glossac_2025N_quiet_profile(), CAL_QUIET, ri, wt, T)
        cols = bg.columns()
        nd = [float(c.number_density_m3()[iz]) * 1e-6 for c in cols]
        het["calibrated_quiet_background_number_density_20km_cm3"] = dict(fine_60nm=nd[0], coarse_1500nm=nd[1], total=sum(nd))
        n_bg = [nd[0]]
        print(f"  calibrated quiet background at 20 km: fine {nd[0]:.2f}, coarse {nd[1]:.4f} cm^-3", flush=True)
    except Exception as exc:  # noqa: BLE001
        het["calibrated_quiet_background_number_density_20km_cm3"] = f"not computed: {exc!r}"
    for N_s in [1.0, 10.0] + n_bg:
        tau = 1.0 / (K_het_med * N_s * 1e6)
        het["lifetimes"][f"N_sulfate_{N_s:g}_cm3"] = dict(
            tau_mono_268_60_days=tau / DAY, tau_mono_268_60_yr=tau / YR,
            tau_psd_silica_days=1.0 / (K_het_mean * N_s * 1e6) / DAY,
            tau_psd_both_days=1.0 / (K_het_both * N_s * 1e6) / DAY,
            rule="1/(K N_sulfate): mean time for a silica particle to collect one sulfate droplet")
        print(f"  heterocoagulation N_sulf = {N_s:4g} cm^-3: K(268,60) = {K_het_med*1e6:.3e} cm^3/s -> tau = {tau/DAY:8.1f} d "
              f"({tau/YR:.2f} yr); PSD-avg both: {1/(K_het_both*N_s*1e6)/DAY:.1f} d", flush=True)
    out["heterocoagulation"] = het
    res["coagulation"] = out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res = {"script": "reproduce/psd_tail_coagulation.py", "round": 70, "referee_item": 9}
    part_c(res)
    part_b(res)
    part_a(res)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"archived -> {OUT.relative_to(_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
