"""radiometric_budget_observable.py -- referee round 6 (paper 1; public port), item 7: the radiometric chain of
round 70 (reproduce/radiometric_budget.py, imported) carried through with the same element
observable as the retrieval, and its noise terms separated.

 * The 8.80-um attenuation uses the line-resolved element observable of Appendix D4
   (-ln<T> = 0.406 for the gases, 0.44 with the sulfate background) instead of the mean-OD
   landscape value (1.2); at 11.40 and 20.40 um the landscape -ln<exp(-tau)> is kept (no
   line-resolved model there; the gas load is small).
 * Three noise terms are reported separately at the attenuated sample: shot noise (fractional
   transmission error growing as e^{y/2}), additive detector noise (growing as e^{y}; the
   microbolometer term that dominates), and the systematic transmission floor 1/2000 (per
   sample and element, covariance not known); plus the shared exo-atmospheric reference.
 * The calibration tolerances (offset, nonlinearity, gain drift, focal-plane temperature) are
   recomputed at the element observable, and the gain-drift rate is stated for the longest
   reference-to-sample interval (70 s at 1 km/s).
 * The warm-background row is defined relative to the attenuated solar power per pixel.

Run from the repo root:  python reproduce/radiometric_budget_observable.py
Writes outputs/radiometric_budget_observable.json and tables/radiometric_budget.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))
import radiometric_budget as R                                     # noqa: E402

OUT_JSON = ROOT / "outputs/radiometric_budget_observable.json"
OUT_TEX = ROOT / "tables/radiometric_budget.tex"
CLOSURE = json.load(open(ROOT / "outputs/element_observable_closure.json"))
i880 = 10
Y_GAS_880 = float(CLOSURE["observable"]["y_gas_880"])
Y_TOT_880 = float(CLOSURE["observable"]["y_total"][i880])


def main():
    instr = R.Instrument()
    dets = R.sourced_detectors()
    geo, g = R.geometry(instr, pitch_um=dets["vox_microbolometer"]["adopted"]["pitch_um"])
    instr.f_number = g["F_num"]
    od = R.optical_depths()
    od["8.80"]["tau_gas_element_landscape_meanOD"] = od["8.80"]["tau_gas_element"]
    od["8.80"]["tau_total_element_landscape_meanOD"] = od["8.80"]["tau_total_element"]
    od["8.80"]["tau_gas_element"] = R.r(Y_GAS_880)
    od["8.80"]["tau_total_element"] = R.r(Y_TOT_880)
    od["8.80"]["source"] = "line-resolved element observable -ln<T> of Appendix D4 (element_observable_closure.json)"
    E_scale = 0.24 / R.solar_spectral_irradiance_W_m2_um(10.0)
    det = dict(label="VOx microbolometer, sourced D* 1e9, 17 um", dstar_cm_rtHz_W=1.0e9, pitch_um=17.0, nep_W_rtHz=17e-4 / 1e9)
    floors = {"8.80": 1.692e-8, "11.40": 1.2e-8, "20.40": 3.4e-9}
    chain = {}
    for lam in (8.80, 11.40, 20.40):
        key = f"{lam:.2f}"
        chain[key] = {}
        for dw in g["dwell"]:
            res = R.chain(lam, od[key]["tau_total_element"], det, g, instr, dw, E_scale=E_scale, floor_m1=floors[key])
            # separated terms at the attenuated sample (fractional error of the ratio)
            T = np.exp(-od[key]["tau_total_element"])
            snr_shot = res["noise"]["snr_shot_atm"]; snr_nep = res["noise"]["snr_nep_atm"]
            n_ref = 100
            frac_shot = 1.0 / snr_shot
            frac_det = 1.0 / snr_nep
            frac_ref = 1.0 / (res["result"]["SNR_single_TOA"] * np.sqrt(n_ref))
            frac_floor = 1.0 / 2000.0
            res["separated_terms_attenuated_sample"] = dict(
                T=R.r(T), y=R.r(float(od[key]["tau_total_element"])),
                shot_fractional=R.r(frac_shot), detector_fractional=R.r(frac_det), reference_fractional=R.r(frac_ref) if frac_ref else None,
                systematic_floor_fractional=frac_floor,
                detector_over_shot=R.r(frac_det / frac_shot),
                scaling="shot ~ e^{y/2}, detector ~ e^{y} (additive), floor: per sample and element, covariance not known",
                total_random_fractional=R.r(np.sqrt(frac_shot ** 2 + frac_det ** 2 + (frac_ref or 0.0) ** 2)),
                total_with_floor_fractional=R.r(np.sqrt(frac_shot ** 2 + frac_det ** 2 + (frac_ref or 0.0) ** 2 + frac_floor ** 2)))
            chain[key][f"dwell_{dw:.2g}"] = res
    # calibration tolerances at the element observable
    canc = R.ratio_cancellation(g, instr, E_scale, dets["vox_microbolometer"]["adopted"])
    canc_obs = dict(canc)
    tau = Y_TOT_880; T = np.exp(-tau)
    E = R.solar_spectral_irradiance_W_m2_um(8.80) * E_scale
    P_toa = E * instr.dlam_um * g["A_ap"] * instr.tau_optics * g["f_strip"]
    dlam_pix = instr.dlam_um / g["n_spec"]
    P_bkg_elem = R.planck_W_m2_sr_um(8.80, instr.T_optics_K) * dlam_pix * g["A_pix"] * g["omega_pix"] * g["N_pix"]
    budget = 5e-4
    canc_obs.update(tau=R.r(tau), T=R.r(T), note="recomputed at the element observable of the 8.80-um element (0.44 with the sulfate background)",
                    offset_requirement=dict(P_element_TOA_W=R.r(P_toa), P_element_atm_W=R.r(P_toa * T), P_bkg_element_W=R.r(P_bkg_elem),
                                            dB_allowed_W=R.r(budget * P_toa * T), fraction_of_background_to_know=R.r(budget * P_toa * T / P_bkg_elem),
                                            background_over_attenuated_solar=R.r(P_bkg_elem / (P_toa * T))),
                    nonlinearity_requirement=dict(quadratic_coefficient_e2_max=R.r(budget / (1 - T))),
                    drift_requirement=dict(dt_ref_to_sample_s=[R.r(70.0 / 3.0), 70.0], gain_drift_rate_max_per_s=[R.r(budget / (70.0 / 3.0)), R.r(budget / 70.0)],
                                           statement="the whole 5e-4 allowance over the longest reference-to-sample interval (70 s at 1 km/s); "
                                                     "a requirement, not a demonstration; with other terms allocated it is tighter"))
    out = dict(script="reproduce/radiometric_budget_observable.py", round=74, based_on="reproduce/radiometric_budget.py (functions imported)",
               optical_depths=od, chain=chain, ratio_cancellation_observable=canc_obs, ratio_cancellation=canc,
               solar=dict(scale=R.r(E_scale)), geometry=geo, floors_m1=floors)
    OUT_JSON.write_text(json.dumps(out, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    # ---- the table
    lams = [8.80, 11.40, 20.40]
    c = {l: chain[f"{l:.2f}"] for l in lams}
    L = []
    L.append("% Round 74 (referee round 6, item 7): first-order radiometric chain at the element observable, generated by")
    L.append("% reproduce/radiometric_budget_observable.py (functions of reproduce/radiometric_budget.py); numbers in outputs/radiometric_budget_observable.json.")
    L.append("\\begin{table*}[t]")
    L.append("\\caption{First-order radiometric chain of the baseline MIR spectrograph for one $0.5$-km sample at a $20$-km tangent, "
             "with the sourced uncooled VO$_x$ microbolometer ($D^{*}=10^{9}$~cm\\,Hz$^{1/2}$\\,W$^{-1}$, $17$-$\\mu$m pitch, $12$-ms time constant). "
             "The solar irradiance is the $5772$-K disk model anchored to the ASTM~E490 value at $10~\\mu$m; the element signal is spread over "
             "$2\\times" + f"{int(round(geo.get('solar_diameter_pixels', g['M_cols'])))}" + "$ pixels (two spectral pixels per $0.1$-$\\mu$m element, the solar diameter across the slit); "
             "detector noise is $\\sqrt{A_\\text{tot}}/D^{*}$ over those pixels with a noise bandwidth $1/(2t)$; the exo-atmospheric reference is the mean of $100$ samples above $80$~km. "
             "The attenuation at $8.80~\\mu$m is the element observable $-\\ln\\langle T\\rangle$ of Appendix~\\ref{app:tracegas_element} with the sulfate background included; "
             "at $11.40$ and $20.40~\\mu$m it is the element average of the mean optical depth, an upper bound on the attenuation. "
             "The random terms are given separately as fractional errors of the ratio at the attenuated sample: shot noise grows as $e^{y/2}$ with the element optical depth $y$, "
             "the additive detector noise as $e^{y}$, and the systematic transmission floor of $5\\times10^{-4}$ is a per-sample, per-element term whose covariance is not established. "
             "``SNR'' is the per-sample transmission SNR of the ratio, $T/\\sigma_T$; the systematics cap of $2000$ is applied afterwards. "
             "The two dwell values bracket the $1$--$3$~km\\,s$^{-1}$ descent of the tangent point.}")
    L.append("\\label{tab:radiometric_budget}")
    L.append("\\centering\\footnotesize")
    L.append("\\begin{tabular}{l" + "c" * len(lams) + "}")
    L.append("\\toprule")
    L.append("Quantity & " + " & ".join(f"${l:.2f}~\\mu$m" for l in lams) + " \\\\")
    L.append("\\midrule")

    def fmt(v):
        if v is None:
            return "--"
        a = abs(v)
        if a == 0:
            return "0"
        if 0.01 <= a < 1e4:
            return f"${v:.3g}$" if a < 100 else f"${v:.0f}$"
        e = int(np.floor(np.log10(a))); m = v / 10 ** e
        return f"${m:.1f}\\times10^{{{e}}}$"
    rows = []
    rows.append(("Solar irradiance used (W\\,m$^{-2}\\mu$m$^{-1}$)", [c[l]["dwell_0.5"]["inputs"]["E_used_W_m2_um"] for l in lams]))
    rows.append(("Element optical depth $y$ at $20$~km (gas; with sulfate)", [f"${od[f'{l:.2f}']['tau_gas_element']:.3g}$; ${od[f'{l:.2f}']['tau_total_element']:.3g}$" for l in lams]))
    rows.append(("Power on the element, exo-atmospheric ($\\mu$W)", [c[l]["dwell_0.5"]["signal"]["P_element_TOA_W"] * 1e6 for l in lams]))
    rows.append(("Power on the element, attenuated ($\\mu$W)", [c[l]["dwell_0.5"]["signal"]["P_element_atm_W"] * 1e6 for l in lams]))
    rows.append(("Warm-optics ($300$~K) power per pixel relative to the attenuated solar power per pixel", [c[l]["dwell_0.5"]["signal"]["bkg_over_signal_pixel_atm"] for l in lams]))
    rows.append(("Detector noise per sample, $t=0.5$~s (pW)", [c[l]["dwell_0.5"]["noise"]["sigma_det_per_sample_W"] * 1e12 for l in lams]))
    rows.append(("Shot-noise fractional error, attenuated sample, $t=0.5$~s ($\\propto e^{y/2}$)", [c[l]["dwell_0.5"]["separated_terms_attenuated_sample"]["shot_fractional"] for l in lams]))
    rows.append(("Detector-noise fractional error, attenuated sample, $t=0.5$~s ($\\propto e^{y}$)", [c[l]["dwell_0.5"]["separated_terms_attenuated_sample"]["detector_fractional"] for l in lams]))
    rows.append(("Shared-reference fractional error ($100$ samples)", [c[l]["dwell_0.5"]["separated_terms_attenuated_sample"]["reference_fractional"] for l in lams]))
    rows.append(("Systematic transmission floor (per sample and element)", [5e-4 for l in lams]))
    rows.append(("Transmission SNR $T/\\sigma_T$ of the ratio, random terms, $t=0.5$~s", [c[l]["dwell_0.5"]["result"]["SNR_T_ratio"] for l in lams]))
    rows.append(("Transmission SNR $T/\\sigma_T$ of the ratio, random terms, $t=0.17$~s", [c[l]["dwell_0.17"]["result"]["SNR_T_ratio"] for l in lams]))
    rows.append(("Net SNR with the systematic floor, $t=0.17$~s", [c[l]["dwell_0.17"]["result"]["SNR_net_with_cap2000"] for l in lams]))
    rows.append(("Random $\\sigma_\\alpha$, $t=0.17$~s ($10^{-9}$~m$^{-1}$)", [c[l]["dwell_0.17"]["result"]["sigma_alpha_random_m1"] * 1e9 for l in lams]))
    rows.append(("Gas-removal floor adopted in the diagonal-floor scenario ($10^{-9}$~m$^{-1}$)", [floors[f"{l:.2f}"] * 1e9 for l in lams]))
    for name, vals in rows:
        L.append(name + " & " + " & ".join(v if isinstance(v, str) else fmt(v) for v in vals) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table*}")
    OUT_TEX.write_text("\n".join(L) + "\n")
    for l in lams:
        s = c[l]["dwell_0.17"]["separated_terms_attenuated_sample"]
        print(f"{l:.2f} um: y {s['y']}, shot {s['shot_fractional']:.2e}, detector {s['detector_fractional']:.2e} (x{s['detector_over_shot']:.0f}), ref {s['reference_fractional']}, "
              f"SNR_T 0.17 s {c[l]['dwell_0.17']['result']['SNR_T_ratio']:.3g}, 0.5 s {c[l]['dwell_0.5']['result']['SNR_T_ratio']:.3g}")
    print("calibration at the observable:", json.dumps({k: canc_obs[k] for k in ("tau", "T", "offset_requirement", "nonlinearity_requirement", "drift_requirement")}, indent=1))
    print("wrote", OUT_JSON, OUT_TEX)


if __name__ == "__main__":
    main()
