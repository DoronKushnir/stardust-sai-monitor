"""highres_gas_removal_budget.py -- what does ACE-class spectral resolution buy
for the trace-gas-removal floors of every mineral band?

The existing floors (material_trace_gas_removal.py, Notes Sec. 9 / App. C-E)
assume an R~100 instrument (0.25 um elements at 0.1 um steps).  There the gas
and the mineral are BOTH smooth within an element, so a band-coherent HITRAN
line-intensity error (~0.2 % of tau_gas) is irreducible and sets a floor that
does not improve with SNR.

At line-resolving resolution (ACE-FTS: 0.02 cm^-1) the logic changes:

  * the gas columns are pinned by the sharp line structure itself, so any
    band-coherent intensity error is absorbed into the retrieved amplitude
    (the product S*N is what the lines measure) and cancels in the removal;
  * what is left is (i) per-line uncorrelated intensity scatter, which now
    averages down over thousands of resolved lines, and (ii) the SMOOTH part
    of the gas model that is degenerate with the aerosol continuum: Voigt/far
    wings (scaled by the air-broadened-width uncertainty) and, for H2O, the
    MT_CKD continuum (scaled by its published uncertainty).

This script quantifies that budget per band with a generalized least-squares
(GLS) fit on a 0.02 cm^-1 grid at a 20 km tangent:

  state:  [mineral amplitude | per-gas column amplitudes | T_eff | quadratic
           Legendre baseline (absorbs sulfate + calibration continuum)]
  noise:  sigma_i^2 = (exp(tau_tot,i)/SNR)^2 + (f_line,g * tau_lines,i)^2
          (photon term + per-line-uncorrelated HITRAN intensity scatter)
  systematics (projected through the GLS estimator onto the mineral):
     - width term: d tau for a fractional air-width error, computed exactly by
       re-running the Voigt grid at P*(1+d) (Doppler unchanged, first order);
     - continuum term: delta_C * tau_MT_CKD (H2O windows);
     - CFC term: absolute cross-section error * tau_CFC (measured .xsc shape).

Outputs per band: sigma_stat(SNR), the coherent systematic floor, the SNR at
which they cross, and the onion-peel extinction floors, next to the R~100
floors from the notes.

Usage: python scripts/highres_gas_removal_budget.py [silica|calcite|dolomite|alumina|all]
First run is slow (fine Voigt grids, incl. perturbed T and P); all cached.
"""

import sys
import csv
from pathlib import Path
import numpy as np

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from saimon.sai import (create_silica_sai_layer, create_calcite_sai_layer,
                       create_dolomite_sai_layer, create_alumina_sai_layer)
from saimon.geometry import tangent_to_slant_paths
from saimon.atmosphere import us_standard_atmosphere
from saimon.trace_gases import (HITRAN_GASES, CFC_GASES, _voigt_native,
                               cfc_cross_section, mt_ckd_h2o_continuum,
                               number_density_profiles)

TANGENT_KM = 20.0
R_EARTH_M = 6.371e6
G_ONION = np.sqrt(5.0 - 2.0 * np.sqrt(3.0))   # white-noise onion-peel gain
DZ_M = 500.0
ATM_PER_PA = 1.0 / 101_325.0

DNU_FINE = 0.005          # cm^-1, Voigt computation grid
DNU_SAMP = 0.02           # cm^-1, ACE-FTS-class sample spacing
SNR_REF = 300.0           # ACE-FTS per-0.02 cm^-1-sample SNR (HgCdTe band)
SNR_GRID = (300.0, 1000.0, 3000.0)
TAU_SAT = 8.0             # drop samples blacker than this (no photons)
TAU_AER_BG = 0.1          # smooth background aerosol OD folded into photon term
WIDTH_PERTURB = 0.02      # fractional P perturbation used to build d tau/d gamma

# Per-gas spectroscopic error model.
#   width_err : fractional uncertainty of air-broadened widths (smooth leak),
#               taken from the intensity-weighted HITRAN ierr gamma_air codes
#               in the relevant windows (parsed from the cached line lists):
#               O3 7.8-9.3um 0.9-1.0%; HNO3 19.5-21um 15% (code 2, 99% of
#               intensity); H2O 27-30um 3.5% (code 4), 19.5-21um 0.6%
#               (mostly code 7 post-Mlawer-2019); CO2 13-15um 0.5%.
#   line_scatter : per-line uncorrelated intensity scatter (HITRAN ierr;
#                  O3 8.74/9.6 um bands are 95 % ierr=5 -> 1.5 %)
GAS_ERR = {
    "o3":   dict(width_err=0.010, line_scatter=0.015),
    "h2o":  dict(width_err=0.035, line_scatter=0.03),
    "co2":  dict(width_err=0.005, line_scatter=0.005),
    "n2o":  dict(width_err=0.02, line_scatter=0.02),
    "ch4":  dict(width_err=0.03, line_scatter=0.03),
    "hno3": dict(width_err=0.15, line_scatter=0.10),
    "so2":  dict(width_err=0.05, line_scatter=0.05),
}
CFC_ABS_ERR = 0.03        # absolute .xsc cross-section scale error (Harrison)

# MT_CKD H2O continuum uncertainty vs wavenumber (Mlawer+2019; REFIR-PAD Dome C
# closures: <~10 % for 400-600 cm^-1, degrading toward the 300 cm^-1 edge).
def continuum_err(nu_cm: float) -> float:
    if nu_cm >= 400.0:
        return 0.10
    if nu_cm >= 330.0:
        return 0.15
    return 0.20

BANDS = {
    "silica": dict(
        make_layer=create_silica_sai_layer,
        bands=[
            dict(label="B+ (8.80um)", target=8.80, lo=7.8, hi=9.3,
                 gases=("o3", "n2o", "h2o", "cfc12"), pad=dict(o3=300.0)),
            dict(label="C (20.40um)", target=20.40, lo=19.5, hi=21.0,
                 gases=("h2o", "hno3", "o3"), width_err=dict(h2o=0.006)),
        ]),
    "calcite": dict(
        make_layer=create_calcite_sai_layer,
        bands=[
            dict(label="nu3 (6.9um)", target=6.9, lo=6.3, hi=7.6,
                 gases=("h2o", "ch4", "n2o", "o3")),
            dict(label="nu2 (11.4um)", target=11.4, lo=10.5, hi=12.0,
                 gases=("hno3", "h2o", "cfc11", "cfc12")),
            dict(label="nu4 (14.0um)", target=14.0, lo=13.2, hi=14.6,
                 gases=("co2", "o3"), pad=dict(co2=150.0)),
            dict(label="L (28.5um)", target=28.5, lo=27.0, hi=30.0,
                 gases=("h2o", "o3", "n2o")),
            dict(label="L-win (29.2um)", target=29.2, lo=28.0, hi=30.5,
                 gases=("h2o", "o3", "n2o")),
        ]),
    "dolomite": dict(
        make_layer=create_dolomite_sai_layer,
        bands=[
            dict(label="nu3 (6.4um)", target=6.4, lo=5.8, hi=7.2,
                 gases=("h2o", "ch4", "n2o", "o3")),
            dict(label="nu2 (11.3um)", target=11.3, lo=10.5, hi=12.0,
                 gases=("hno3", "h2o", "cfc11", "cfc12")),
            dict(label="nu4 (13.7um)", target=13.7, lo=13.2, hi=14.6,
                 gases=("co2", "o3"), pad=dict(co2=150.0)),
            dict(label="L (24.9um)", target=24.9, lo=23.5, hi=26.5,
                 gases=("h2o", "o3", "n2o")),
        ]),
    "alumina": dict(
        make_layer=create_alumina_sai_layer,
        bands=[
            dict(label="R (12.9um)", target=12.9, lo=12.0, hi=13.8,
                 gases=("co2", "h2o", "o3", "hno3", "cfc11"),
                 pad=dict(co2=150.0, o3=150.0)),
            dict(label="W (20.7um)", target=20.7, lo=19.8, hi=21.6,
                 gases=("h2o", "hno3", "o3"), width_err=dict(h2o=0.006)),
        ]),
}

DEFAULT_PAD = 60.0


def slant_atmosphere():
    alt_m = np.arange(14_000.0, 80_001.0, 500.0)
    chord = tangent_to_slant_paths(np.array([TANGENT_KM * 1e3]), alt_m)[0]
    prof = us_standard_atmosphere(alt_m)
    dens = number_density_profiles(alt_m, prof.number_density_m3)
    return alt_m, chord, prof, dens


def gas_slant(chord, prof, dens, gas):
    """Slant column [m^-2] and path-weighted T [K], P [Pa], H2O vmr."""
    n = dens[gas]
    w = chord * n
    col = float(np.sum(w))
    if col <= 0:
        return 0.0, 220.0, 5000.0, 0.0
    t_rep = float(np.sum(w * prof.temperature_k) / col)
    p_rep = float(np.sum(w * prof.pressure_pa) / col)
    n_air = prof.pressure_pa / (1.380649e-23 * prof.temperature_k)
    vmr = float(np.sum(w * (n / n_air)) / col)
    return col, t_rep, p_rep, vmr


def voigt_od(gas, nu, t_k, p_pa, col_m2, pad):
    """Line-only slant OD of `gas` on fine grid `nu` (cm^-1)."""
    g = HITRAN_GASES[gas]
    nu_lo, nu_hi = nu[0] - pad, nu[-1] + pad
    gnu, coef = _voigt_native(g, nu_lo, nu_hi, t_k, p_pa * ATM_PER_PA, DNU_FINE)
    sig = np.interp(nu, gnu, coef) * 1e-4          # cm^2 -> m^2 / molecule
    return sig * col_m2


def wing_pedestal_od(gas, nu, t_k, p_pa, col_m2, pad):
    """Smooth inter-line pedestal from the 0.2-25 cm^-1 Lorentz wings that the
    fine Voigt grid truncates (HAPI cuts at 50 half-widths ~ 0.2 cm^-1 at 20 km
    pressure).  Computed with full 25 cm^-1 wings (the MT_CKD line/continuum
    partition boundary) on a coarse grid, then rolling-minimum filtered so line
    cores drop out and only the smooth floor survives."""
    import os
    from saimon.trace_gases import XSEC_CACHE, HITRAN_CACHE, _ensure_hapi
    g = HITRAN_GASES[gas]
    step = 0.25
    nu_lo, nu_hi = nu[0] - pad, nu[-1] + pad
    key = (f"wing25_{g.name}_{int(round(nu_lo))}_{int(round(nu_hi))}"
           f"_T{t_k:.1f}_P{p_pa * ATM_PER_PA:.6f}_s{step:g}")
    path = os.path.join(XSEC_CACHE, key + ".npz")
    if os.path.exists(path):
        d = np.load(path)
        gnu, coef = d["nu"], d["coef"]
    else:
        hapi = _ensure_hapi()
        table = f"cont_{g.name}_{int(nu_lo)}_{int(nu_hi)}"
        data_file = os.path.join(HITRAN_CACHE, table + ".data")
        if not (os.path.exists(data_file) and os.path.getsize(data_file) > 0):
            hapi.fetch(table, g.mol_id, g.iso_id, nu_lo, nu_hi)
        gnu, coef = hapi.absorptionCoefficient_Voigt(
            SourceTables=table, Environment={"T": t_k, "p": p_pa * ATM_PER_PA},
            WavenumberStep=step, OmegaWing=25.0, OmegaWingHW=50.0,
            HITRAN_units=True)
        np.savez_compressed(path, nu=gnu, coef=coef)
    # rolling minimum over +-1.5 cm^-1 (13 coarse samples): line cores drop out
    n_w = 6
    cmin = np.array([coef[max(0, i - n_w): i + n_w + 1].min()
                     for i in range(len(coef))])
    sig = np.interp(nu, gnu, cmin) * 1e-4
    return sig * col_m2


def bin_to_samples(nu_fine, y_fine, n_bin):
    m = (len(nu_fine) // n_bin) * n_bin
    nu_s = nu_fine[:m].reshape(-1, n_bin).mean(axis=1)
    y_s = y_fine[:m].reshape(-1, n_bin).mean(axis=1)
    return nu_s, y_s


def run_band(mat, band, layer, alt_m, chord, prof, dens):
    lo_um, hi_um, tgt_um = band["lo"], band["hi"], band["target"]
    pads = band.get("pad", {})
    nu_lo, nu_hi = 1e4 / hi_um, 1e4 / lo_um
    nu = np.arange(nu_lo, nu_hi, DNU_FINE)
    wl_m = 1e4 / nu * 1e-6
    n_bin = int(round(DNU_SAMP / DNU_FINE))

    # --- mineral shape on a coarse wavelength grid, interpolated to fine
    wl_coarse = np.linspace(lo_um, hi_um, 121) * 1e-6
    tau_min_coarse = chord @ layer.extinction_profile_m1(wl_coarse)
    tau_min = np.interp(wl_m * 1e6, wl_coarse * 1e6, tau_min_coarse)
    i_tgt = np.argmin(np.abs(wl_m * 1e6 - tgt_um))
    tau_min_tgt = float(tau_min[i_tgt])

    # --- per-gas fine OD + derivative spectra
    tau_gas, tau_sharp, dtau_width, dtau_T = {}, {}, {}, {}
    tau_cont = np.zeros_like(nu)
    cont_err_val = continuum_err(1e4 / tgt_um)
    for gas in band["gases"]:
        col, t_rep, p_rep, vmr = gas_slant(chord, prof, dens, gas)
        pad = pads.get(gas, DEFAULT_PAD)
        if gas in CFC_GASES:
            sig = cfc_cross_section(gas, wl_m)
            tau_gas[gas] = sig * col
            tau_sharp[gas] = np.zeros_like(nu)
            # an absolute .xsc scale error is absorbed by the co-retrieved
            # amplitude (projects to ~0); .xsc SHAPE errors are not modeled
            dtau_width[gas] = CFC_ABS_ERR * tau_gas[gas]
            dtau_T[gas] = np.zeros_like(nu)
            continue
        tau0 = voigt_od(gas, nu, t_rep, p_rep, col, pad)
        tau_w = voigt_od(gas, nu, t_rep, p_rep * (1.0 + WIDTH_PERTURB), col, pad)
        tau_t = voigt_od(gas, nu, t_rep + 2.0, p_rep, col, pad)
        ped = wing_pedestal_od(gas, nu, t_rep, p_rep, col, pad)
        tau_gas[gas] = tau0 + ped
        tau_sharp[gas] = tau0
        werr = band.get("width_err", {}).get(gas, GAS_ERR[gas]["width_err"])
        # near-wing redistribution (exact, from the perturbed-P grid) plus the
        # far-wing pedestal, which scales linearly with the Lorentz width
        dtau_width[gas] = (tau_w - tau0) * (werr / WIDTH_PERTURB) + werr * ped
        dtau_T[gas] = (tau_t - tau0) / 2.0            # per K
        if gas == "h2o":
            tau_cont = mt_ckd_h2o_continuum(wl_m, t_rep, p_rep, vmr) * col

    # --- bin everything to the 0.02 cm^-1 sample grid
    nu_s, tau_min_s = bin_to_samples(nu, tau_min, n_bin)
    tau_gas_s = {g: bin_to_samples(nu, t, n_bin)[1] for g, t in tau_gas.items()}
    dtw_s = {g: bin_to_samples(nu, d, n_bin)[1] for g, d in dtau_width.items()}
    dtT_s = sum(bin_to_samples(nu, d, n_bin)[1] for d in dtau_T.values())
    tau_cont_s = bin_to_samples(nu, tau_cont, n_bin)[1]

    tau_tot_s = tau_min_s + tau_cont_s + TAU_AER_BG + sum(tau_gas_s.values())
    ok = tau_tot_s < TAU_SAT
    n_ok = int(ok.sum())
    if n_ok < 50:
        return dict(label=band["label"], dead=True, n_ok=n_ok,
                    tau_min_tgt=tau_min_tgt)

    # --- GLS design: [mineral | gas amplitudes | T_eff | Legendre 0..2]
    x = np.linspace(-1.0, 1.0, len(nu_s))
    cols = [tau_min_s / tau_min_tgt]
    names = ["mineral"]
    for g in band["gases"]:
        cols.append(tau_gas_s[g] + (tau_cont_s if g == "h2o" else 0.0))
        names.append(g)
    cols.append(dtT_s); names.append("T_eff")
    for k in range(3):
        cols.append(np.polynomial.legendre.Legendre.basis(k)(x))
        names.append(f"L{k}")
    A_full = np.stack(cols, axis=1)[ok]
    # drop empty/degenerate columns (a gas with no lines in the window would
    # make the normal matrix singular); the mineral column (0) is always kept
    norms = np.sqrt((A_full ** 2).sum(axis=0))
    keep = norms > 1e-14 * norms.max()
    keep[0] = True
    dropped = [names[j] for j in range(len(names)) if not keep[j]]
    A = A_full[:, keep] / np.where(norms[keep] > 0, norms[keep], 1.0)
    scale0 = norms[0]

    # per-line-uncorrelated intensity scatter (noise-like at this resolution;
    # applies to the sharp line structure, not the smooth pedestal)
    tau_sharp_s = {g: bin_to_samples(nu, t, n_bin)[1] for g, t in tau_sharp.items()}
    line_var = sum((GAS_ERR.get(g, dict(line_scatter=0.0))["line_scatter"]
                    * tau_sharp_s[g]) ** 2
                   for g in band["gases"] if g not in CFC_GASES)[ok]

    def sigma_stat(snr):
        var = (np.exp(np.minimum(tau_tot_s[ok], TAU_SAT)) / snr) ** 2 + line_var
        W = 1.0 / var
        Cinv = A.T @ (A * W[:, None])
        C = np.linalg.pinv(Cinv, rcond=1e-12, hermitian=True)
        return float(np.sqrt(C[0, 0])) / scale0, C, W

    sig_stats = {snr: sigma_stat(snr)[0] for snr in SNR_GRID}
    _, C, W = sigma_stat(SNR_REF)

    # --- coherent systematics projected through the GLS estimator
    def project(dtau):
        b = C @ (A.T @ (W * dtau[ok]))
        return float(abs(b[0])) / scale0

    sys_terms = {}
    for g in band["gases"]:
        lbl = "cfc_abs" if g in CFC_GASES else "width"
        sys_terms[f"{lbl}:{g}"] = project(dtw_s[g])
    if "h2o" in band["gases"]:
        sys_terms["continuum:h2o"] = project(cont_err_val * tau_cont_s)
    sig_sys = float(np.sqrt(sum(v ** 2 for v in sys_terms.values())))

    # SNR where photon-only statistical error equals the systematic floor
    sig_ref_photon = sigma_stat(SNR_REF)[0]
    snr_cross = (SNR_REF * sig_ref_photon / sig_sys) if sig_sys > 0 else np.inf

    P_kk = 2.0 * np.sqrt(2.0 * R_EARTH_M * DZ_M)
    out = dict(
        label=band["label"], dead=False, target_um=tgt_um, n_ok=n_ok, dropped=dropped,
        n_samp=len(nu_s), tau_min_tgt=tau_min_tgt,
        tau_gas_tgt={g: float(np.interp(1e4 / tgt_um, nu_s, tau_gas_s[g]))
                     for g in band["gases"]},
        tau_cont_tgt=float(np.interp(1e4 / tgt_um, nu_s, tau_cont_s)),
        sig_stats=sig_stats, sig_sys=sig_sys, sys_terms=sys_terms,
        snr_cross=snr_cross,
        sigma_alpha_stat={snr: G_ONION * s / P_kk for snr, s in sig_stats.items()},
        sigma_alpha_sys=sig_sys / P_kk,
    )
    tot = {snr: np.hypot(s, sig_sys) for snr, s in sig_stats.items()}
    out["sig_total"] = tot
    out["min_tg"] = {snr: t / tau_min_tgt for snr, t in tot.items()}
    return out


def main(which):
    alt_m, chord, prof, dens = slant_atmosphere()
    mats = list(BANDS) if which == "all" else [which]
    rows = []
    for mat in mats:
        cfg = BANDS[mat]
        layer = cfg["make_layer"](alt_m, target_mass_tg=1.0)
        print(f"\n================ {mat.upper()} (1 Tg, {TANGENT_KM:.0f} km tangent, "
              f"{DNU_SAMP} cm^-1 samples) ================")
        for band in cfg["bands"]:
            print(f"\n--- {band['label']}")
            r = run_band(mat, band, layer, alt_m, chord, prof, dens)
            r["material"] = mat
            rows.append(r)
            if r.get("dead"):
                print(f"    DEAD: only {r['n_ok']} unsaturated samples")
                continue
            gs = ", ".join(f"{g}={v:.3g}" for g, v in r["tau_gas_tgt"].items())
            print(f"    tau_mineral(tgt) = {r['tau_min_tgt']:.3e} OD/Tg")
            print(f"    tau_gas(tgt)     = {gs}; continuum={r['tau_cont_tgt']:.3g}")
            print(f"    usable samples   = {r['n_ok']}/{r['n_samp']}"
                  + (f"  (dropped cols: {r['dropped']})" if r["dropped"] else ""))
            for snr, s in r["sig_stats"].items():
                print(f"    sigma_stat(SNR={snr:.0f})  = {s:.3e} OD")
            print(f"    sigma_sys (coherent) = {r['sig_sys']:.3e} OD")
            for k, v in sorted(r["sys_terms"].items(), key=lambda kv: -kv[1]):
                print(f"        {k:<18} {v:.3e}")
            print(f"    SNR crossover (stat=sys) ~ {r['snr_cross']:.0f}")
            print(f"    sigma_alpha stat@300 = {r['sigma_alpha_stat'][300.0]:.3e} 1/m"
                  f" ; sys = {r['sigma_alpha_sys']:.3e} 1/m")
            print(f"    total sigma_od @SNR300/1000/3000 = "
                  + " / ".join(f"{r['sig_total'][s]:.2e}" for s in SNR_GRID))
            print(f"    1-sigma mass @SNR300/1000/3000   = "
                  + " / ".join(f"{r['min_tg'][s]:.4f}" for s in SNR_GRID) + " Tg")

    outdir = _PARENT / "outputs" / "highres"
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "highres_gas_removal_budget.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["material", "band", "target_um", "tau_min_1Tg",
                    "sigma_stat_snr300", "sigma_stat_snr1000", "sigma_stat_snr3000",
                    "sigma_sys", "snr_crossover", "sigma_alpha_sys_1m",
                    "min_tg_snr300", "min_tg_snr1000", "min_tg_snr3000",
                    "sys_breakdown"])
        for r in rows:
            if r.get("dead"):
                w.writerow([r["material"], r["label"], "", f"{r['tau_min_tgt']:.4e}",
                            "DEAD", "", "", "", "", "", "", "", "", ""])
                continue
            w.writerow([r["material"], r["label"], r["target_um"],
                        f"{r['tau_min_tgt']:.4e}",
                        *(f"{r['sig_stats'][s]:.4e}" for s in SNR_GRID),
                        f"{r['sig_sys']:.4e}", f"{r['snr_cross']:.0f}",
                        f"{r['sigma_alpha_sys']:.4e}",
                        *(f"{r['min_tg'][s]:.5f}" for s in SNR_GRID),
                        "; ".join(f"{k}={v:.2e}" for k, v in r["sys_terms"].items())])
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
