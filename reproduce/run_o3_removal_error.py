"""
run_o3_removal_error.py -- propagate the O3-removal uncertainty into the silica
SAI 8.80 um channel at h_tan = 20 km.

Runs the augmented-state Rodgers-OE analysis from saimon.o3_removal for two
scenarios:

  (A) Chappuis-only: SAGE-III 521 + 602 + 676 nm channels.  Constrains the
      O3 slant column tightly, but cannot pin down the IR band-scaling
      systematic -- so the prediction at 8.80 um inherits the full HITRAN
      absolute line-intensity prior in the IR band.

  (B) Chappuis + 9.6 um: adds a direct IR channel inside the SAME vibrational
      band as 8.80 um.  The band-coherent HITRAN systematic now constrained by
      data largely cancels in the 9.6 um -> 8.80 um ratio.

Defaults:
  delta_T_K       = 2 K      (met-field accuracy; was 10 K in the first pass)
  sigma_line_rel  = 0.002    (justified by HITRAN ierr distribution -- see
                              scripts/analyze_hitran_o3_line_uncertainty.py;
                              ~0.1% intensity-weighted scatter at 8.80 um is
                              the upper bound when band-averaging over ~200
                              independent lines, taken with a x2 safety margin.)

A small sensitivity table at the end varies these defaults so the dominant
lever can be read off at a glance.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from saimon.o3_removal import O3RemovalSetup, run_o3_removal_analysis


def _fmt(x: float) -> str:
    return f"{x:.3e}"


def _report(label: str, res: dict, setup: O3RemovalSetup) -> None:
    info = res["info"]
    ch_nm = [f"{w*1e9:.0f}" if w < 1e-6 else f"{w*1e6:.2f}µm"
             for w in info["channels_m"]]
    print()
    print(f"=== {label} ===")
    print(f"  h_tan                     : {setup.h_tan_m/1e3:.1f} km")
    print(f"  channels                  : {ch_nm}")
    print(f"  target                    : {info['target_wavelength_m']*1e6:.2f} µm")
    print(f"  tau_O3(target)  reference : {_fmt(res['tau_target_ref'])}  OD")
    for nm, t in zip(ch_nm, res["tau_meas_ref"]):
        print(f"  tau_O3({nm}) reference : {_fmt(float(t))}  OD")
    print(f"  --")
    print(f"  sigma_OE_state            : {_fmt(res['sigma_OE_state'])}  OD"
          f"   (O3-profile retrieval)")
    print(f"  sigma_OE_band             : {_fmt(res['sigma_OE_band'])}  OD"
          f"   (posterior on band scalings)")
    print(f"  sigma_OE_T                : {_fmt(res['sigma_OE_T'])}  OD"
          f"   (posterior on T_eff, see below)")
    print(f"  sigma_OE (joint)          : {_fmt(res['sigma_OE'])}  OD"
          f"   (state + bands + T together)")
    print(f"  sigma_line_pred           : {_fmt(res['sigma_line_pred'])}  OD"
          f"   ({setup.sigma_line_rel*100:.2f}% line scatter @ target)")
    print(f"  --")
    print(f"  sigma_total               : {_fmt(res['sigma_total'])}  OD"
          f"   <- input to SAI pipeline")
    print(f"  DOFS                      : {res['dofs']:.2f}"
          f"  (state has {len(setup.nodes_km)} O3 nodes + {len(setup.bands)} bands)")
    print(f"  band-scaling posteriors   :")
    for name, post in res["band_posterior_sigma"].items():
        prior = res["band_prior_sigma"][name]
        shrink = prior / post if post > 0 else float("inf")
        print(f"      {name:>4s}   sigma = {post*100:5.2f}%   "
              f"(prior {prior*100:.1f}%, shrink ×{shrink:.1f})")
    shrink_T = res["T_prior_K"] / res["T_posterior_K"] if res["T_posterior_K"] > 0 else float("inf")
    print(f"  T-offset posterior         : sigma = {res['T_posterior_K']:.3f} K   "
          f"(prior {res['T_prior_K']:.1f} K, shrink ×{shrink_T:.1f})")


def _write_csv(path: Path, res: dict, setup: O3RemovalSetup) -> None:
    info = res["info"]
    ch_nm = [f"{w*1e9:.0f}" if w < 1e-6 else f"{w*1e6:.2f}um"
             for w in info["channels_m"]]
    with open(path, "w") as fh:
        fh.write("metric,value,units\n")
        fh.write(f"h_tan_km,{setup.h_tan_m/1e3},km\n")
        fh.write(f"delta_T_K,{setup.delta_T_K},K\n")
        fh.write(f"sigma_line_rel,{setup.sigma_line_rel},fraction\n")
        fh.write(f"target_wavelength_um,{info['target_wavelength_m']*1e6},um\n")
        fh.write(f"channels,\"{';'.join(ch_nm)}\",\n")
        fh.write(f"tau_target_ref,{res['tau_target_ref']},OD\n")
        fh.write(f"sigma_OE,{res['sigma_OE']},OD\n")
        fh.write(f"sigma_OE_state,{res['sigma_OE_state']},OD\n")
        fh.write(f"sigma_OE_band,{res['sigma_OE_band']},OD\n")
        fh.write(f"sigma_OE_T,{res['sigma_OE_T']},OD\n")
        fh.write(f"sigma_line_pred,{res['sigma_line_pred']},OD\n")
        fh.write(f"sigma_total,{res['sigma_total']},OD\n")
        fh.write(f"T_prior_K,{res['T_prior_K']},K\n")
        fh.write(f"T_posterior_K,{res['T_posterior_K']},K\n")
        fh.write(f"dofs,{res['dofs']},\n")
        for name, post in res["band_posterior_sigma"].items():
            fh.write(f"b_{name}_posterior_sigma,{post},fraction\n")


def main():
    out_dir = _HERE / "outputs" / "o3_removal"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Updated defaults: T uncertainty 2 K (was 10 K), line scatter 0.002 (was 0.02).
    common_kwargs = dict(h_tan_m=20_000.0, delta_T_K=2.0, sigma_line_rel=0.002)

    setup_chappuis = O3RemovalSetup(**common_kwargs)
    res_chappuis = run_o3_removal_analysis(setup_chappuis)
    _report("Scenario A: Chappuis-only (3 channels)", res_chappuis, setup_chappuis)
    _write_csv(out_dir / "scenario_A_chappuis_only_20km.csv",
               res_chappuis, setup_chappuis)

    # 9.2 um sits on the O3 ν3 band shoulder where tau ~ 1 (not saturated).
    # Confirmed optimum by scripts/scan_ir_o3_channel.py.
    setup_plus92 = O3RemovalSetup(ir_o3_channels_m=(9.2e-6,), **common_kwargs)
    res_plus92 = run_o3_removal_analysis(setup_plus92)
    _report("Scenario B: Chappuis + 9.2 µm (unsaturated band shoulder)",
            res_plus92, setup_plus92)
    _write_csv(out_dir / "scenario_B_chappuis_plus_9.2um_20km.csv",
               res_plus92, setup_plus92)

    print()
    print("=== Bottom line (default scenario) ===")
    a = res_chappuis["sigma_total"]
    b = res_plus92["sigma_total"]
    print(f"  sigma_total (A: Chappuis only)         : {_fmt(a)}  OD")
    print(f"  sigma_total (B: Chappuis + 9.2 µm)     : {_fmt(b)}  OD")
    print(f"  improvement                             : x{a/b:.2f}")

    # ---- Sensitivity sweep: how much does each lever move sigma_total in B? ----
    print()
    print("=== Sensitivity sweep (Scenario B, h_tan = 20 km) ===")
    print(f"  varying delta_T_K and sigma_line_rel; sigma_total in OD\n")
    print(f"   {'delta_T':>10s}  {'sigma_line':>11s}  {'sigma_total':>13s}"
          f"  {'sigma_OE':>10s}  {'sigma_OE_T':>11s}  {'sigma_line':>11s}  {'T_post(K)':>10s}")
    rows = []
    for dT in (1.0, 2.0, 5.0, 10.0):
        for sl in (0.001, 0.002, 0.005, 0.02):
            setup = O3RemovalSetup(h_tan_m=20_000.0,
                                   ir_o3_channels_m=(9.2e-6,),
                                   delta_T_K=dT, sigma_line_rel=sl)
            r = run_o3_removal_analysis(setup)
            rows.append((dT, sl, r))
            print(f"   {dT:>10.1f}  {sl:>11.4f}  {r['sigma_total']:>13.3e}"
                  f"  {r['sigma_OE']:>10.3e}  {r['sigma_OE_T']:>11.3e}"
                  f"  {r['sigma_line_pred']:>11.3e}  {r['T_posterior_K']:>10.2f}")

    # Save sweep
    with open(out_dir / "sensitivity_sweep_B_20km.csv", "w") as fh:
        fh.write("delta_T_K,sigma_line_rel,sigma_total,sigma_OE,sigma_OE_T,sigma_line_pred,T_posterior_K\n")
        for dT, sl, r in rows:
            fh.write(f"{dT},{sl},{r['sigma_total']},{r['sigma_OE']},"
                     f"{r['sigma_OE_T']},{r['sigma_line_pred']},{r['T_posterior_K']}\n")

    print(f"\n  -> wrote scenario_A_*.csv, scenario_B_*.csv, "
          f"sensitivity_sweep_B_20km.csv")


if __name__ == "__main__":
    main()
