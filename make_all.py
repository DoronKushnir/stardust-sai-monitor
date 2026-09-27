#!/usr/bin/env python3
"""Regenerate the paper's figures, tables and archived numbers.

    python make_all.py --fast     # figures + tables from the shipped archives (minutes)
    python make_all.py --full     # also rerun the heavy analyses (hours; see the table)
    python make_all.py --list     # show the steps
    python make_all.py --only fig4 fig5 ...

Every step is a script in reproduce/; run one directly to see its options.
--fast reads the archived intermediate results in outputs/ (the state they
had when the paper was compiled) and redraws every figure and table from
them.  --full recomputes those archives first, in dependency order, from the
data in data/.  Steps marked HITRAN need the HITRAN-derived cross-section
cache (data/trace_gases/xsec_cache/), which is shipped for the bands used.
"""
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
R = ROOT / "reproduce"

# (key, script and args, produces, runtime note, heavy?)
STEPS = [
    ("peak",   ["silica_peak_wavelength.py"],            "outputs/silica_peak_wavelength.txt (Sect. 3.1: 8.80 um)", "s", False),
    ("psd",    ["silica_element_psd_variation.py"],      "outputs/silica_element_psd_variation.txt (Sect. 3.1)", "s", False),
    ("mass",   ["reservoir_mass_saod.py"],               "outputs/reservoir_mass_saod.json (Appendix A)", "1 min", False),
    ("figA1",  ["fit_segev_psd.py"],                     "figures/segev_psd_fit.png (Fig. A1)", "s", False),
    ("figA2",  ["reproduce_glossac_tropical_saod.py", "--paper"], "figures/glossac_tropical_saod.png (Fig. A2)", "s", False),
    ("cbt",    ["calibrated_background_thresholds.py"],  "outputs/calibrated_background_thresholds.json (Sect. 3, Table 1)", "30-60 min", True),
    ("cbp",    ["calibrated_background_problem.py"],     "outputs/calibrated_background_problem.json (Sect. 2, Table C1)", "~1 h", True),
    ("fig1",   ["plot_slant_od_landscape_paper.py"],     "figures/slant_od_landscape.png (Fig. 1)", "1 min", False),
    ("fig2",   ["plot_refractive_index_paper.py"],       "figures/refractive_index_problem.png (Fig. 2)", "1 min", False),
    ("fig3",   ["plot_spectral_shapes_paper.py"],        "figures/spectral_shapes_problem.png (Fig. 3)", "1 min", False),
    ("fig4",   ["plot_ace_atlas_paper.py"],              "figures/scisat_ace_atlas_874.png (Fig. 4)", "1 min", False),
    ("fig5",   ["ace_mir_aerosol.py"],                   "figures/ace_mir_aerosol.png (Fig. 5), outputs/ace_v52/*", "3 min", False),
    ("null",   ["ace_fullspectrum_retrieval.py"],        "outputs/ace_v52/fullspectrum*.json (Sect. 4.3 null test)", "5 min", False),
    ("roster", ["window_vis_roster_sweep.py"],           "outputs/window_vis_roster_sweep.json (Sect. 3.2 roster variants)", "10 min", True),
    ("nullc",  ["ace_nulltest_convention.py"],           "outputs/ace_v52/nulltest_convention.json (per-shell convention)", "2 min", False),
    ("spike",  ["ace_spike_test.py"],                    "outputs/ace_v52/spike_test.json (Sect. 4.3 injection-recovery test)", "1 min", False),
    ("ruang",  ["ace_ruang_bandshape.py"],               "Sect. 4.3 Ruang numbers", "2 min", False),
    ("elem",   ["ace_element_sampling.py"],              "outputs/ace_v52/element_sampling.json (Sect. 4.1)", "s", False),
    ("epoch",  ["ace_floor_epoch_check.py"],             "Sect. 4.2 epoch floors", "s", False),
    ("floor1", ["ace_floor/residual_floor_pipeline.py", "--sample", "data/ace_floor/sample.csv", "--output", "data/ace_floor/w0p1", "--config", "data/ace_floor/w0p1/config.json"], "data/ace_floor/w0p1 (Appendix E, 0.1-um element)", "30 min", True),
    ("floor25",["ace_floor/residual_floor_pipeline.py", "--sample", "data/ace_floor/sample.csv", "--output", "data/ace_floor/w0p25", "--config", "data/ace_floor/w0p25/config.json"], "data/ace_floor/w0p25 (Appendix E, 0.25-um check)", "30 min", True),
    ("floorc", ["ace_floor/residual_floor_pipeline.py", "--sample", "data/ace_floor/sample.csv", "--output", "data/ace_floor/w0p1_c880", "--config", "data/ace_floor/w0p1_c880/config.json"], "data/ace_floor/w0p1_c880 (the design grid centred on 8.80 um; floors of Sects. 3-6)", "25 min", True),
    ("tabE1",  ["ace_floor/residual_floor_paper1.py"],   "tables/ace_residual_floors.tex, figures/ace_floor_atlas.png (Table E1, Fig. E1)", "s", False),
    ("design", ["design_sensitivity_calibrated.py"],     "outputs/design_sensitivity_calibrated.json, figures/design_sensitivity_calibrated.png (Table 2, Fig. F2)", "20 min", True),
    ("resol",  ["resolution_sensitivity_calibrated.py"], "outputs/resolution_sensitivity_calibrated.json, figures/resolution_sensitivity_calibrated.png (Table F1, Fig. F1)", "20 min", True),
    ("nullr",  ["ace_nulltest_resolution.py"],           "outputs/ace_v52/nulltest_resolution.json (Table F1, ACE column)", "10 min", True),
    ("mat",    ["materials_calibrated_thresholds.py"],   "outputs/materials_calibrated/results.json (Table 3)", "20 min", True),
    ("fig6",   ["detectability_2d_calibrated.py"],       "outputs/detectability_2d_calibrated/results.json, figures/detectability_2d_mass_contours.png (Sect. 6, Fig. 6)", "35 min", True),
    ("oe",     ["run_o3_removal_error.py"],              "outputs/o3_removal/* (Appendix D budget scenario)  [HITRAN]", "20 min", True),
    ("floorw", ["gas_removal_floor_880_widths.py"],      "outputs/round43_tracegas/gas_removal_floor_widths.csv (Appendix D budget)  [HITRAN]", "1 min", True),
    ("snr",    ["snr_integration_time_trade.py"],        "outputs/snr_trade/* (Appendix D SNR behaviour)  [HITRAN]", "1 min", True),
    ("det",    ["mir_detector_etc.py"],                  "Appendix D detector numbers", "s", False),
    ("onion",  ["onion_peel_geometry_exact.py"],         "outputs/onion_peel_geometry_exact.txt (Appendix B: exact onion-peel gain, edge vs centre grid)", "s", False),
    ("m2",     ["m2_vertical_correlation_floor.py"],     "outputs/m2_vertical_correlation_floor.json (Appendix D.6: coherent vs independent conversion of the floor)", "s", False),
    ("far",    ["false_alarm_trials.py"],                "outputs/false_alarm_trials.txt (Sect. 5: trials factor and false-alarm rate)", "s", False),
    ("o3lat",  ["o3_slant_latitude_proxy.py"],           "outputs/o3_slant_latitude_proxy.txt (AFGL cross-check of the slant ozone load vs latitude; superseded in the text by o3meas)", "s", False),
    ("o3meas", ["o3_slant_latitude_measured.py"],        "outputs/o3_slant_latitude_measured.{json,txt} (Sect. 6: slant ozone load vs latitude from the ACE v5.2 O3 retrievals; RC2 S1)", "s", False),
    ("fig6o3", ["detectability_2d_calibrated.py", "--o3-latitude", "outputs/o3_slant_latitude_measured.json"], "outputs/detectability_2d_calibrated/results_o3lat.json, figures/detectability_2d_mass_contours_o3lat.png (Sect. 6: the map with the latitude-dependent floor; RC2 S1)", "21 min", True),
    ("tab53",  ["paper1_tables_round53.py"],             "tables/element_width_scan.tex, tables/nulltest_conventions.tex (Table F1, the null-test convention table)", "s", False),
    ("limb",   ["check_limb_emission_baseline.py"],      "Appendix D limb-emission ratio  [HITRAN]", "1 min", True),
    ("cont",   ["mir_continuum_gas_od.py"],              "Sect. 3.4 continuum optical depths  [HITRAN: needs the full cache, fetch/]", "1 min", True),
    ("cfc",    ["verify_cfc_contribution.py"],           "Appendix D CFC-12 residual  [HITRAN: needs the full cache, fetch/]", "1 min", True),
    ("highres",["highres_gas_removal_budget.py", "silica"], "Sect. 4 line-resolved budget  [HITRAN]", "1 min", True),
    ("stT",    ["analyze_st_temperature_sensitivity.py"], "Appendix D S(T) table  [HITRAN line lists]", "8 min", True),
    ("lines",  ["analyze_hitran_o3_line_uncertainty.py"], "Appendix D line-scatter check  [HITRAN line lists]", "5 min", True),
]
FAST = [k for k, *_ in STEPS if not [s for s in STEPS if s[0] == k][0][4]]


def run(step):
    key, args, produces, note, heavy = step
    print(f"\n=== {key}: {args[0]} -> {produces}  ({note})", flush=True)
    t = time.time()
    r = subprocess.run([PY, str(R / args[0]), *args[1:]], cwd=ROOT)
    print(f"=== {key}: exit {r.returncode} in {time.time()-t:.0f} s", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fast", action="store_true", help="figures/tables from the shipped archives (default)")
    g.add_argument("--full", action="store_true", help="recompute every archive first")
    g.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="KEY")
    a = ap.parse_args()
    if a.list:
        for k, args, prod, note, heavy in STEPS:
            print(f"{k:8s} {'FULL ' if heavy else 'fast '} {note:10s} {args[0]:42s} {prod}")
        return
    keys = a.only or ([s[0] for s in STEPS] if a.full else FAST)
    failed = [s[0] for s in STEPS if s[0] in keys and run(s) != 0]
    print("\nDONE" + (f" (failed: {failed})" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
