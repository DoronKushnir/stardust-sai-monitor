# Manifest: every paper object, its inputs, command, output and tier

For Kushnir et al. (2026), *Design of a dedicated solar-occultation satellite
for monitoring stratospheric aerosol injection*, Atmos. Meas. Tech. (in
review). This file links each figure, table and quoted number of the paper to
the script that produces it, the shipped inputs it reads (with their
provenance), the files it writes, whether `make_all.py --fast` regenerates it
or only `--full` does, and which of its inputs are themselves derived products
shipped with the repository.

**Software state described here.** Commit
`c523d6c77ca7eb8ea260be7756673e9a22782a1e` (the HEAD of this repository on
2026-10-25 and the commit checked in referee round 5); the rows marked
"round 70" describe the port of the round-70 analyses (the reply to referee
round 5), tagged `rc5-round70`. Annotated tags in the
history: `round51`, `round53-54`, `rc2-round56`, `rc2-round57`, `rc3-round59`.
The round-75 state is tagged `rc6-round75` (Sect. 11), the round-74 state `rc6-round74` (Sect. 10); the release tag and the Zenodo deposit are
listed in Sect. 8 (to be completed at submission). Python >= 3.10 with
`requirements.txt` (numpy, scipy, matplotlib, netCDF4, pyyaml; `hitran-api`
only for the full HITRAN chain).

**Paper numbering.** As compiled on 2026-10-25 (round 70): Figs. 1-5 and A1-A3, E1,
F1, F2; Tables 1-8 and A1, B1, C1, D1-D5, E1, F1-F3 (before round 70 the
detector table was D2; it is now D4, and D2, D3, D5 and A1 are new). `make_all.py --list` and
`guide/GUIDE.md` still print the figure numbers of an earlier compilation
(landscape = "Fig. 1", refractive index = "Fig. 2", spectral shapes =
"Fig. 3", atlas = "Fig. 4", ACE aerosol = "Fig. 5", 2-D map = "Fig. 6");
the figure *file names* are unambiguous and are what this manifest keys on.

## 1. Vocabulary

Tier (what regenerates the object):

| Tier | Meaning |
|---|---|
| **FAST** | `python make_all.py --fast` runs the step; the object is regenerated in minutes from the shipped `data/` and the archived `outputs/`. |
| **FULL** | only `python make_all.py --full` runs the step (hours); it recomputes the archive in `outputs/` from `data/`. Under `--fast` the archived output and, for figures, the shipped PNG are what the reader gets. |
| **SHIPPED (figure)** | the PNG in `figures/` is the deliverable under `--fast`; the script that draws it is a FULL step. Each of these scripts has a `--plot` mode that redraws the figure from its archive in seconds, but `make_all.py --fast` does not call it. |
| **TYPED (table)** | the `.tex` in the manuscript is typed from archive values; no script in this repository writes the table file. The archive and the keys holding each value are given. |
| **HITRAN** | the step needs the shipped element-resolution cross-section grids `data/trace_gases/xsec_cache/` (90 `.npz`, our computation from HITRAN2020 line lists); since round 70 also the 32 line-resolved 0.0025-cm^-1 grids and the four carved line lists (D-XSEC-FINE, D-LINES) of step `closure`, which imports `hapi` (`hitran-api`) for the partition sums only. |
| **HITRAN-full** | the step needs the untrimmed HITRAN line-list cache `data/trace_gases/hitran_cache/`, which is not shipped (`fetch/build_hitran_cache.py`; hours; hitran.org). |

Input identifiers (all under `data/`; source, reduction and regeneration
script per directory in `data/PROVENANCE.md`):

| ID | Files | Source and nature |
|---|---|---|
| D-OPT-SiO2 | `optics/SiO2_KitamuraPopova.yml`, `SiO2_Kitamura2007.yml`, `Popova.yml`, `SiO2_Franta2016.yml` | silica optical constants (Kitamura 2007 + Popova 1972; Franta 2016); file copies / digitizations |
| D-OPT-MIN | `optics/Al2O3_Querry-{e,o}.yml`, `Dolomite_Querry-{e,o}.yml`, `{alumina,calcite,dolomite}_optical_{Ordinary,ExtraOrdinary}Ray.csv` | alumina, dolomite (Querry), calcite (Long 1993) optical constants |
| D-OPT-H2SO4 | `optics/palmer_williams_75_*.csv`, `optics/h2so4_members/{myhre_h2so4,biermann_h2so4}/` | Palmer & Williams (1975); Lund Myhre et al. (2003) and Biermann et al. (2000) members from the HITRAN aerosol collection |
| D-OPT-ASH | `optics/ash_deguine2020/` | Deguine et al. (2020) volcanic-ash optical constants (ARIA) |
| D-GLO | `glossac/GloSSAC_V2.23_subset.nc` | NASA GloSSAC V2.23, 525 and 1020 nm, <= 40 km; subset by `fetch/reduce_glossac.py` |
| D-ACE143 | `ace/v52_subset/residual/<occ>/*.gz` for the 143 occultations of `ace_floor/sample.csv`; `ace/v52_subset/occultationlist.csv` | ACE-FTS v5.2 Level-2 residual release (FRDR, doi:10.20383/103.01245), registered access; bit-identical subset by `fetch/reduce_ace_v52.py` (Sect. 7) |
| D-ACE42 | `ace/v52_subset/residual/<occ>/*.gz` for the 42 occultations of `ace/ash_lecours2026/ash_manifest_v52.txt` | same release, same reduction (Sect. 7) |
| D-ACEO3 | `ace/o3_l2_v52/o3_profiles_v52.csv.gz` (192 occultations) | ACE-FTS v5.2 Level-2 O3 profiles; column extract by `fetch/reduce_ace_o3_l2.py` |
| D-ACEREV | `ace/revision20260909/{matrix_19_22,basis,optics}.npz` | **derived product**: the 2-cm^-1 common-grid residual matrix of the 143 occultations at 19-22 km, the two-mode ensemble basis and the sulfate optics library, built from D-ACE143 and D-OPT-H2SO4 by the `reproduce/ace_floor/revision_*.py` scripts (not a `make_all` step) |
| D-FLOOR | `ace_floor/w0p1/`, `ace_floor/w0p25/`, `ace_floor/w0p1_c880/` (`atlas.json`, per-element `.json/.csv`, `manifest.json` with hashes, `matrix_*.npz`), `ace_floor/sample.csv` | **derived product**: the ACE residual-floor analysis records, built from D-ACE143 by FULL steps `floor1`, `floor25`, `floorc` |
| D-ATLAS | `ace_atlas/Tropics_016-020km.txt.gz`, `Tropics_020-024km.txt.gz` | ACE Atmospheric Atlas (Hughes et al. 2014, public), 950-1300 cm^-1 window by `fetch/reduce_ace_atlas.py` (950-1250 before round 70) |
| D-XSEC | `trace_gases/xsec_cache/*.npz` (90 grids) | **derived product**: Voigt cross-section grids computed by `saimon.trace_gases` from HITRAN2020 line lists for the paper's bands, gases and (T, P); `fetch/build_hitran_cache.py` |
| D-XSEC-FINE | `trace_gases/xsec_cache/{o3,n2o,ch4,h2o}_1067_1294_T*_P*_s0.0025.npz` (32 grids, 27 MB) | **derived product** (round 70): line-resolved 0.0025-cm^-1 Voigt grids of the four co-retrieved gases over 1067-1294 cm^-1 at the five budget nodes and at (T_eff, T_eff +- 5 K, P_eff); `fetch/build_finegrid_window.py` from D-LINES |
| D-LINES | `trace_gases/hitran_cache/cont_{o3,n2o,ch4,h2o}_1042_1319.{data,header}` (10 MB) | HITRAN2020 line lists of the four gases over 1042-1319 cm^-1 (the closure window plus the 25-cm^-1 Voigt wing) in HAPI format, carved from the full-band lists; read directly by `element_observable_closure.py` (positions, intensities, widths, lower-state energies, uncertainty codes, source references) |
| D-TG | `trace_gases/{afgl,mpi_mainz,mt_ckd,cfc_xsec,pseudolines,hitran_mw}/` | AFGL profiles, MPI-Mainz UV/Vis, MT_CKD H2O continuum, HITRAN CFC `.xsc`, ClONO2 pseudo-lines, molecule metadata; file copies |
| D-DIG | `digitized/Segev2026_Fig2b.csv`, `Segev2026_CDF_best.xlsx`, `Wrana2021_Fig2a.csv`, `Wrana2021_Fig3.csv`, `Lederer2026_Fig2_massfraction.csv` | digitized / author-provided tables |
| D-MOD | `models/Silica_05_no_coagulation_with2D.nc` | Lederer et al. (2026) 2-D transport-model field, +-41 deg / 51 hPa injection |
| D-LEC | `ace/ash_lecours2026/{ash-sulfate_missionwide,data_pcc,data_calbuco}.txt`, `README.md`; `ash_manifest_v52.txt` (ours) | Lecours, Boone & Bernath (2026), Zenodo doi:10.5281/zenodo.21479502 |

Archived derived products in `outputs/` that other steps read (all
shipped; "A-" identifiers):

| ID | Files | Produced by | Tier of the producer |
|---|---|---|---|
| A-CBT | `outputs/calibrated_background_thresholds.json` | step `cbt` (`reproduce/calibrated_background_thresholds.py`) | FULL, 30-60 min. Holds the element grids, the per-element floors and the baselines; read by nearly every other step |
| A-CBP | `outputs/calibrated_background_problem.json` | step `cbp` | FULL, ~1 h |
| A-FLOORW | `outputs/round43_tracegas/gas_removal_floor_widths.csv` | step `floorw` | FULL, HITRAN, 1 min; the 8.80-um budget row |
| A-ALT | `outputs/detectability_2d/floor_altitude_scan.csv`, `floor_altitude_scan_w0p1.csv` | `reproduce/scan_floor_altitude.py` — **not a `make_all` step** (logs `outputs/round53_tracegas/scan_floor_altitude*.log`) | HITRAN; the floor versus tangent altitude read by `design` and `fig6` |
| A-ENG | `outputs/detectability_2d_calibrated/engine_{band,window}_{quiet,postHT}.npz` | step `fig6` | FULL; cached background engines that step `mat` requires (`assert cache.exists()`) |
| A-MAP | `outputs/detectability_2d_calibrated/map_pm41*.npz`, `results.json`, `results_o3lat.json`, `mmin_lat_*.csv` | steps `fig6`, `fig6o3` | FULL; read by `tablat` |
| A-FARIR | `outputs/materials_calibrated/farir_band_floors_w0p1.json` | `reproduce/farir_band_floors_w0p1.py` — **not a `make_all` step** (log `outputs/round53_tracegas/farir_band_floors_w0p1.log`) | HITRAN-full (HNO3); the far-infrared per-element floors of Table 8 |
| A-SCISAT | `outputs/scisat_atlas/taus_h{18.0,20.0,21.9}.npz` | `reproduce/analyze_ace_atlas.py --rebuild` — **not a `make_all` step** | HITRAN; the gas model drawn against the atlas in Fig. 3 |
| A-LAND | `outputs/slant_od_landscape_curves.npz` | step `fig1` with `--recompute` (included in `fetch/build_hitran_cache.py`) | HITRAN; the gas, Rayleigh, sulfate and material curves of Fig. 2 |
| A-ACEV52 | `outputs/ace_v52/{basis_spectra_cal.npz, fullspectrum.csv, fullspectrum_stats.json, nulltest.csv, nulltest_stats.json, mir_bands.csv, anchor_constrained_fits.json, glossac_matched_saod525.npz, sulfate_shape_grid.npz, element_sampling.json, spike_test.json, nulltest_convention.json, nulltest_resolution.json, ash_positive_control.{csv,json}}` | steps `fig5`, `null`, `nullc`, `spike`, `elem`, `ash` (FAST) and `nullr` (FULL) | recomputed from D-ACE143 / D-ACE42 by `--fast`, except `nulltest_resolution.json` |
| A-RESOLC | `outputs/resolution_fine_grid_cache.npz` | step `resol` | FULL; fine-grid Mie cache |
| A-TG | `outputs/o3_removal/*`, `outputs/snr_trade/*`, `outputs/highres/*`, `outputs/{alumina,calcite,dolomite}/*.csv`, `outputs/round43_tracegas/*.log`, `outputs/round53_tracegas/*.log` | steps `oe`, `snr`, `highres`, `limb`, `det` and the scripts `material_trace_gas_removal.py`, `calcite_trace_gas_removal.py` (not steps) | HITRAN / HITRAN-full; the Appendix D archives and the logs of every HITRAN run |
| A-CLOSURE | `outputs/element_observable_closure.json`, `.npz` | step `closure` (`reproduce/element_observable_closure.py`; round 70) | FULL, HITRAN (D-XSEC-FINE, D-LINES), ~45 min with `--ndraw 100` (the shipped archive; the Monte Carlo dominates); read by `tabD2` and `fig4` |
| A-RANK | `outputs/marginal_rank.{json,txt}` | step `rank` (round 70) | FAST, 3 min; A-CBT, A-CBP, A-ENG |
| A-LAT70 | `outputs/latitude_declared_level.json` | step `lat70` (round 70) | FAST, 1 min; A-MAP, D-MOD, `outputs/m2_vertical_correlation_floor.json`, `outputs/ace_v52/fullspectrum.csv` |
| A-RADIO | `outputs/radiometric_budget.{json,txt}`, `tables/radiometric_budget.tex` | step `radio` (round 70) | FAST, seconds; A-LAND (20-km optical depths), `reproduce/mir_detector_etc.py` |
| A-UNIAX | `outputs/materials_uniaxial.json`, `tables/optical_constants_provenance.tex` | step `uniax` (round 70) | FULL, ~30 min; A-ENG, A-CBT, D-OPT-MIN, D-OPT-SiO2 |
| A-TAIL | `outputs/psd_tail_coagulation.json` | step `tail` (round 70) | FAST, ~5 min; D-DIG (Segev tables), A-ENG, A-CBT, D-OPT-SiO2 |
| A-NULLD | `outputs/ace_v52/nulltest_diagnostics.json`, `nulltest_diagnostics_perocc.csv`, `basis_diag_<case>.npz` (caches) | step `nulld` (round 70) | FULL, ~15 min; D-ACE143, D-ACEREV, A-ACEV52 (`basis_spectra_cal.npz`, `spike_test.json`, `ash_positive_control.csv`), A-CBT |

In every row below the software commit is `c523d6c` (Sect. 0 above); the
column is kept so that a later release can record a different commit per
row if a step is re-run.

## 2. Figures

| Paper object | Script / `make_all` step | Inputs | Output file(s) | Tier | Key assumptions | Commit |
|---|---|---|---|---|---|---|
| Fig. 1, `figures/spectral_shapes_problem.png` (`fig:spectral_shapes_hires`, Sect. 2) | `reproduce/plot_spectral_shapes_paper.py`, step `fig3` | D-OPT-SiO2 (Kitamura+Popova, Franta 2016); D-OPT-H2SO4 (member LM65T223) and D-GLO through the imported `calibrated_background_thresholds` background module | `figures/spectral_shapes_problem.png` | FAST (full recomputation; Mie only) | normalized extinction at z = 20 km; calibrated two-component sulfate background (CAL_QUIET, LM65T223); silica r_med = 268 nm, sigma = 1.31 | c523d6c |
| Fig. 2, `figures/slant_od_landscape.png` (`fig:landscape`, Sect. 3) | `reproduce/plot_slant_od_landscape_paper.py`, step `fig1` | A-LAND (archived curves); when recomputed (`--recompute`): D-XSEC / HITRAN line lists, D-TG (AFGL US-standard profiles, CFC `.xsc`, MT_CKD, MPI-Mainz), D-OPT-SiO2, D-OPT-MIN, D-OPT-H2SO4, D-GLO | `figures/slant_od_landscape.png` | FAST replays the archived curves; the curves themselves are HITRAN (`fetch/build_hitran_cache.py` lists `fig1`) | 20-km tangent; US-standard atmosphere and AFGL profiles; total trace gas only; calibrated background; the four materials at 1 Tg. The script docstring says "HITRAN2024 lines" while `data/PROVENANCE.md` says HITRAN2020 for the shipped grids — to confirm which edition the archived curves used | c523d6c |
| Fig. 3, `figures/scisat_ace_atlas_874.png` (`fig:ace_atlas`, Sect. 4.1; round 70) | `reproduce/plot_ace_atlas_element.py`, step `fig4`, from A-CLOSURE (`atlas_test/fine_boxcar`); the archive itself is written by step `closure` | A-CLOSURE; when recomputed: D-ATLAS (16-20 km tropical bin, 1067-1294 cm^-1), D-XSEC-FINE, D-LINES, A-CBT, D-OPT-SiO2, D-OPT-H2SO4 | `figures/scisat_ace_atlas_874.png` | FAST replays the archive; `closure` (FULL, HITRAN) recomputes it | both sides in the element observable -ln<T> on 0.02-um boxcars at the 18-km tangent; line-resolved O3, N2O, CH4, H2O, CFC-12 and the calibrated sulfate background with one O3 amplitude fitted on the 9.2-um element (0.75; 0.99 gas-only); the 8.80-um element predicted to -0.037 OD of the measured 0.453 OD. The superseded mean-optical-depth version (0.02-um-sampled <tau>, A-SCISAT) is still drawn by `reproduce/plot_ace_atlas_paper.py` (no step) | rc5-round70 |
| Fig. 4, `figures/ace_mir_aerosol.png` (`fig:ace_mir_aerosol`, Sect. 4.2) | `reproduce/ace_mir_aerosol.py`, step `fig5` | D-ACE143 (all tangent heights) + `occultationlist.csv`; D-ACEREV; D-FLOOR (`w0p1/` scores); D-GLO (matched SAOD525 anchors); D-OPT-SiO2. The script also names `data/ace_floor/w0p1_874/19_22_8.74um_scores.csv`, which is not shipped — the referee's `--fast` run succeeded, so the path must be optional; to confirm | `figures/ace_mir_aerosol.png`; `outputs/ace_v52/{mir_bands.csv, anchor_constrained_fits.json, glossac_matched_saod525.npz}` | FAST — full recomputation from the shipped ACE extracts (3 min) | slant space (no onion peel; ACE tangent heights refraction-corrected); 8.80-um element 8.75-8.85 um and 4.0-um continuum element; tau = -ln<T_resid> | c523d6c |
| Fig. 5, `figures/detectability_2d_mass_contours.png` (`fig:2d_mass_contours`, Sect. 6) | `reproduce/detectability_2d_calibrated.py`, step `fig6` (35 min); `--plot` redraws from the archive | A-CBT; D-FLOOR (`w0p1/atlas.json`: the 16-19 / 22-25 km floor ratios); A-ALT; D-GLO; D-MOD; D-OPT-SiO2; D-OPT-H2SO4 | `figures/detectability_2d_mass_contours.png`; `outputs/detectability_2d_calibrated/{results.json, map_pm41.npz, map_pm41_103hPa.npz, mmin_lat_*.csv, engine_*.npz}` | SHIPPED (figure) under `--fast`; FULL recomputes | +-41 deg / 51 hPa sustained-injection field (lifetime 1.11 yr) on the calibrated background; three floor scenarios (held, OE per altitude, atlas ratio); orbits ISS-like 51.6 deg and dedicated 70 deg; matched-filter campaign gain. Variant step `fig6o3` (`--o3-latitude outputs/o3_slant_latitude_measured.json`) writes `results_o3lat.json` and `figures/detectability_2d_mass_contours_o3lat.png` (figure not in the paper; feeds Table 7) | c523d6c |
| Fig. A1, `figures/segev_psd_fit.png` (`fig:segev_psd`) | `reproduce/fit_segev_psd.py`, step `figA1` | D-DIG (`Segev2026_Fig2b.csv`, `Segev2026_CDF_best.xlsx`) | `figures/segev_psd_fit.png` | FAST (full recomputation) | single-mode lognormal fits of the measured CDFs, truncated at 4 um; the D300o case gives the baseline r_med = 268 nm, sigma = 1.31 | c523d6c |
| Fig. A2, `figures/refractive_index_problem.png` (`fig:ri_compare`) | `reproduce/plot_refractive_index_paper.py`, step `fig2` | D-OPT-SiO2 (`SiO2_Kitamura2007.yml`, `Popova.yml`, `SiO2_Franta2016.yml`); D-OPT-H2SO4 (LM65T223 member, as the forward model uses it) | `figures/refractive_index_problem.png` | FAST (file plot) | sulfate curve = the calibrated background's member optics; NIR/MIR split at 3 um | c523d6c |
| Fig. A3, `figures/glossac_tropical_saod.png` (`fig:glossac_saod`) | `reproduce/reproduce_glossac_tropical_saod.py --paper`, step `figA2` | D-GLO | `figures/glossac_tropical_saod.png` | FAST (full recomputation from the subset) | tropical SAOD at 525 and 1020 nm from the shipped GloSSAC slice | c523d6c |
| Fig. E1, `figures/ace_floor_atlas.png` (`fig:ace_floor_atlas`) | `reproduce/ace_floor/residual_floor_paper1.py`, step `tabE1` | D-FLOOR `w0p1/` (`atlas.json`, `manifest.json`, `paired_sd_reductions.json`, `sensitivity.json`) and `w0p25/` (comparison column) | `figures/ace_floor_atlas.png`, `tables/ace_residual_floors.tex`, `ace_floor_export.json` (hashes of the archive, the exporter and both outputs) | FAST exports from the archive after verifying its hashes; the archive is FULL (steps `floor1`, `floor25` from D-ACE143, 30 min each) | 0.1-um elements at 19-22 km; one native tangent spectrum per occultation; complete calendar-year holdout; 500 bootstrap draws, seed 20260909 (`w0p1/config.json`) | c523d6c |
| Fig. F1, `figures/resolution_sensitivity_calibrated.png` (`fig:resolution_sensitivity`) | `reproduce/resolution_sensitivity_calibrated.py`, step `resol` (20 min); `--plot` redraws | A-CBT; D-FLOOR (`w0p1_c880/atlas.json`, `w0p25/atlas.json`, `w0p25/19_22_8.8um.json`); D-OPT-SiO2; A-RESOLC | `figures/resolution_sensitivity_calibrated.png`, `outputs/resolution_sensitivity_calibrated.json` | SHIPPED (figure) under `--fast`; FULL recomputes | Mie columns on a 0.005-um grid box-averaged into elements of width dlam; floors interpolated from the measured 0.1-um atlas SDs scaled to the 8.80-um budget; correlation lengths ell; five grid placements | c523d6c |
| Fig. F2, `figures/design_sensitivity_calibrated.png` (`fig:design_sensitivity`) | `reproduce/design_sensitivity_calibrated.py`, step `design` (20 min); `--plot` redraws | A-CBT; A-CBP; A-ALT; D-OPT-SiO2 (Kitamura+Popova, Franta 2016); D-DIG `Lederer2026_Fig2_massfraction.csv` for the agglomerate rows (to confirm) | `figures/design_sensitivity_calibrated.png`, `outputs/design_sensitivity_calibrated.json` | SHIPPED (figure) under `--fast`; FULL recomputes | z = 20 km; triplet + band / window; seven-parameter fit; scans of sigma_psd, r_med, visible noise, altitude, floor scale, background loading, enlarged background families | c523d6c |

Figures shipped in `figures/` that are **not** in the paper: `ace_ash_positive_control.png` (step `ash`; its `\includegraphics` is commented out in Sect. 4.3), `ace_fullspectrum_nulltest.png` (step `null`), `ace_ruang_bandshape.png` (step `ruang`), `detectability_2d_mass_contours_o3lat.png` (step `fig6o3`), `snr_integration_time_trade.png` (step `snr`).

## 3. Tables

| Paper object | Script / `make_all` step | Inputs | Output file(s) | Tier | Key assumptions | Commit |
|---|---|---|---|---|---|---|
| Table 1, `tables/parameter_counts.tex` (`tab:parameter_counts`, Sect. 2) | none | — | — | TYPED; definitional (the count of fitted parameters per configuration), no computation | the six background parameters, the material amplitude, the self-calibration constant of the ACE fits, the gray offset of the robustness test | c523d6c |
| Table 2, `tables/error_budget.tex` (`tab:error_budget`, Sect. 3; rewritten in round 70 on the element observable: the eight rows are typed from A-CLOSURE `removal_budget_880` (photon 4.08e-4, line 2.29e-4, prior 2.06e-4, total 5.12e-4 OD at S = 2000; 2.10e-4 at S -> inf), `joint_vs_two_stage_window`, `joint_band` and `snr_scan`; the archived mean-optical-depth budget 2.466e-3 OD (A-FLOORW) is quoted for comparison) | steps `closure` (round 70) and, for the retained rows, `onion` (`onion_peel_geometry_exact.py`), `floorw`, `floorc`, `cbt`, `design`, `det` / `snr` | `saimon/onion_peel.py` (g = 1.097, P_kk = 159.9 km, 1/|G_k| = 145.7 km); A-FLOORW row (8.8, 0.1): `sigma_removal_od` = 2.4656e-3; D-FLOOR `w0p1_c880/atlas.json` 19-22 km (8.80-um SD 1.858e-3; per-element range); A-CBT `floor_anchor/floor_scale` = 1.327; `outputs/design_sensitivity_calibrated.json` `archived/{band,window}/mir_offset` (band 0.1018); photon term 6.9e-6/S m^-1 from the SNR-trade / detector steps (`outputs/round43_tracegas/{snr_integration_time_trade,mir_detector_etc}.log`; exact source to confirm) | — | TYPED from the archives named | OD slant per 0.1-um element at a 20-km tangent; edge-grid onion peel | c523d6c |
| Table 3, `tables/nulltest_conventions.tex` (`tab:nulltest_conventions`, Sect. 4.3) | `reproduce/paper1_tables_round53.py`, step `tab53`, from `outputs/ace_v52/nulltest_convention.json` (step `nullc`), which reads `fullspectrum.csv` / `fullspectrum_stats.json` / `basis_spectra_cal.npz` (step `null`), A-CBT and `outputs/window_vis_roster_sweep.json` (step `roster`, FULL 10 min) | D-ACE143 (101 occultations with a tangent within 1.5 km of 20.5 km), D-ACEREV, D-OPT-SiO2, A-CBT | `tables/nulltest_conventions.tex`, `outputs/ace_v52/nulltest_convention.json` | FAST (the `.tex` is generated; `null` -> `nullc` -> `tab53` recompute from the shipped ACE extracts) — except the linearized rows that read the roster archive (FULL) | tangent nearest 20.5 km, ties to the lower height; bin-wise self-calibration by the >= 30 km median; rho = 2.33 at the 20.5-km median tangent | c523d6c |
| Table 4, `tables/ace_control_summary.tex` (`tab:ace_control`, Sect. 4.3) | steps `null`, `spike`, `nullc`, `ruang`, `ash` | `outputs/ace_v52/fullspectrum_stats.json` (`n` 101; `full/sigma_emp` 0.02594, `full/median_M` 0.01125, `full/mmin_3sigma` 0.07783); `nulltest_convention.json` (`rho` 2.335 -> 0.18 Tg per shell); `spike_test.json` (`0.1/full/median` 0.1112, `sigma_emp` 0.02594); `ash_positive_control.json` (`n_ash_spectra` 56, `n_ash_occ` 41, `n_null` 101, `ash/frac_ash_gt3` 0.946 = 53/56, `null/frac_ash_gt3` 0.0099 = 1/101, `ash/frac_prefer_ash` 0.964 = 54/56, `lecours_compare/ratio_median_fixed_psd` 1.84 with `n` 9, `ash/M_sil_both_median` -0.012 +- 0.102). **Ruang row**: the numbers are printed by `ace_ruang_bandshape.py` (step `ruang`) and are not archived to a JSON — to confirm | — | TYPED; all steps FAST | see the row scripts | c523d6c |
| Table 5, `tables/design_baseline.tex` (`tab:design`, Sect. 5) | none writes it; numbers from steps `cbt`, `det`, `snr`, `fig6`, `far` | A-CBT (`band01_elements_um`, `window_elements_um`, `band01_floors_converted_m1`: the floor range 1.7-9.5e-8 m^-1 — to confirm against the array), `outputs/detectability_2d_calibrated/results.json` `orbits/ISS-like` (`per_day` 30.22, `lat_min/max` -70.3 / +69.9), `outputs/false_alarm_trials.txt` (decision threshold 1.51 x 0.101 = 0.15 Tg at one false alarm per year), Table B1 for the visible/NIR errors | — | TYPED; requirements table | — | c523d6c |
| Table 6, `tables/robust_summary.tex` (`tab:robust_summary`, Sect. 5) | steps `design`, `resol`, `m2`, `coat`, `cbt` | `outputs/design_sensitivity_calibrated.json`: `sets/{band 8-13 @0.1, window 7.8-9.3 @0.1}/{sigma_psd, rmed, franta, g_vis, altitude, third_mode, second_member, agglomerate}` and `archived/{band,window}/{post_ht, optics_swap, mir_offset, uniform_floors, floor_scan, loading}`; `outputs/resolution_sensitivity_calibrated.json` (element width 0.05 / 0.25 um; correlated floors ell = 0.1 / 0.25 um); `outputs/m2_vertical_correlation_floor.json` `thresholds_tg/{band,window}/mixed_measured` (0.0782 / 0.1093); `outputs/sulfate_coating.json` `sets/*/{0.1,0.3}` ; `peak_um_vs_rmed` (8.79 -> 8.84 um). The coagulation row (1 / 8 Tg yr^-1) comes from the `agglomerate` set — to confirm | — | TYPED; `design`, `resol`, `coat` are FULL, `m2` FAST | perturbations one at a time around the 0.101 / 0.141 Tg baselines | c523d6c |
| Table 7, `tables/latitude_scenarios.tex` (`tab:latitude_scenarios`, Sect. 6) | `reproduce/latitude_scenarios.py`, step `tablat` | A-MAP (`map_pm41.npz`, `map_pm41_o3lat.npz`) | `tables/latitude_scenarios.tex`, `outputs/latitude_scenarios.json` | FAST (table from the archived maps); the maps are FULL (`fig6`, `fig6o3`) | fixed 20.5-km bin versus best height; coverage = latitudes with threshold < 0.1 Tg; four scenarios (reference floor / measured altitude floor / measured ozone / both) | c523d6c |
| Table 8, `tables/materials.tex` (`tab:materials`, Sect. 7; design-band columns since round 70 from A-UNIAX `materials/<m>/b/{mmin_tg/{quiet,postHT}, peak_ext_m1_per_Tg}` under the Clausius-Mossotti average: calcite 0.796/0.791, dolomite 1.162/1.160, alumina 0.157/0.158 Tg; the ray-mean values below are reproduced by `materials/<m>/a`) | `reproduce/materials_uniaxial.py`, step `uniax` (round 70, 30 min); `reproduce/materials_calibrated_thresholds.py`, step `mat` (20 min) for the window and far-infrared columns. Note (round 70 port): `saimon/materials.py` now carries the working repository's Cauchy smoothing of the alumina ray-mean index below 5 um and the end-value hold outside the tables (October 2026); `mat` re-run with it leaves silica, calcite and dolomite unchanged to 1e-9 and moves alumina by -0.2 % in the band (0.1499 -> 0.1496 Tg, 0.150 as quoted), -0.15 % in the red extension (0.1461 -> 0.1459) and +12 % in the window (13.5 -> 15.2 Tg quiet, 11.4 -> 12.9 post-HT; the text's "13-25 Tg" lower end becomes 15 Tg); the shipped archive and `reference_outputs/outputs/materials_results.json` are the re-run | A-CBT; A-ENG (asserted present); A-FARIR; D-OPT-SiO2, D-OPT-MIN via `saimon.sai` factories | `outputs/materials_calibrated/results.json`: `materials/<m>/band/{quiet,postHT}` (0.101/0.100, 0.782/0.780, 0.746/0.750, 0.150/0.151), `window/quiet` (13-25 Tg for the three alternatives), `farir/{triplet_plus_farir_quiet, band_plus_farir_quiet, floor_scan, floors_m1, peak_wl_um}`, `peak_tau_per_Tg_20km_band/wl_um` | TYPED from the archive; `mat` is FULL and depends on `fig6` (A-ENG) and on A-FARIR, which no `make_all` step regenerates (HITRAN-full) | ray-arithmetic-mean optics for the uniaxial crystals; the silica detection PSD for every material; far-IR band of fifty 0.1-um elements with per-element OE floors bounded below by the SNR-2000 floor 3.4e-9 m^-1; brackets at 3x and 10x the floors | c523d6c |
| Table B1, `tables/sage3_level2_errors.tex` (`tab:combined`, App. B) | `reproduce/tableB1_predicted.py`, step `tabB1` (rewrites the "Predicted" column in place) | `saimon/onion_peel.py` constants; SNR = 2000, eps_R = 0.4 %, P_kk = 160 km, g = 1.10. The "Observed" column is Wrana et al. (2021) v5.1 values typed in the manuscript (D-DIG holds the digitized Wrana figures, not this column — to confirm) | `tables/sage3_level2_errors.tex` | FAST (predicted column generated); observed column TYPED from the literature | ATBD extinction profile with Angstrom exponent 1.73 at 20 km; full spherical Rayleigh limb integral | c523d6c |
| Table C1, `tables/fisher_inputs.tex` (`tab:input`, App. C) | step `cbp` (`calibrated_background_problem.py`, ~1 h) | `outputs/calibrated_background_problem.json` key `input_table`; A-CBT `floor_anchor` for the 8800-nm sigma_k (1.692e-8 m^-1) | — | TYPED from the archive; `cbp` is FULL | z = 20 km; calibrated two-component background; visible/NIR errors floored at SNR = 2000 | c523d6c |
| Table D1, `tables/st_sensitivity.tex` (`tab:st_sensitivity`, App. D) | step `stT` (`analyze_st_temperature_sensitivity.py`, 8 min) | HITRAN line lists (marked "[HITRAN line lists]" in `make_all.py`; the shipped grids do not suffice); D-TG | printed; archived log `outputs/round43_tracegas/analyze_st_temperature_sensitivity.log` | TYPED from the printed output; the step is HITRAN-full; the shipped log is the record | +-15 K at fixed column and pressure; line-intensity S(T) only | c523d6c |
| Table A1, `tables/optical_constants_provenance.tex` (`tab:optical_provenance`, App. A; round 70) | `reproduce/materials_uniaxial.py`, step `uniax` (written from the `provenance` block of A-UNIAX) | D-OPT-SiO2, D-OPT-MIN, D-OPT-H2SO4 (file headers and the loaders of `saimon.materials` / `saimon.refractive_index`) | `tables/optical_constants_provenance.tex` | FULL (`uniax`) | source, range, temperature, interpolation and density per material as the code applies them | rc5-round70 |
| Table D2, `tables/element_closure.tex` first part (`tab:element_closure`, App. D; round 70) | `reproduce/element_closure_table.py`, step `tabD2`, from A-CLOSURE | A-CLOSURE (`setup`, `removal_budget_880`, `joint_vs_two_stage_window`, `joint_band`, `snr_scan`, `response_knowledge`, `observable`, `rho_slant_to_shell_20km`) | `tables/element_closure.tex` | FAST (table); `closure` FULL for the archive | joint 38-parameter state and priors; window 7.8-9.3 um, 20-km tangent, S = 2000, sigma_eta = 2 %; masses 3 sigma per 0.5-km shell (slant x 2.61) | rc5-round70 |
| Table D3, `tables/element_closure.tex` second part (`tab:element_mc`, App. D; round 70) | same script and step | A-CLOSURE `monte_carlo/{baseline,<variant>}/<mass>/{bias_median, sd, posterior_sigma_median, coverage_1sig, coverage_2sig, p_detect_3sig, p_detect_4p54sig}` (100 draws per mass, 33 per variant) | `tables/element_closure.tex` | FAST (table); `closure` FULL for the archive (`--ndraw 100`, the shipped archive; the binomial resolution of the fractions is +-5 points) | truths drawn from the priors with line, response and photon perturbations; damped iterated optimal estimation from the prior | rc5-round70 |
| Table D4, `tables/mir_detectors.tex` (`tab:mir_detectors`, App. D; Table D2 before round 70) | none writes it; step `det` (`mir_detector_etc.py`) supports the NEP / SNR statements of the text; step `radio` (round 70) the sourced D* values | literature (detector heritage); log `outputs/round43_tracegas/mir_detector_etc.log`; A-RADIO `sourced_detectors` | — | TYPED; literature table | — | c523d6c |
| Table D5, `tables/radiometric_budget.tex` (`tab:radiometric_budget`, App. D; round 70) | `reproduce/radiometric_budget.py`, step `radio` | A-LAND (the 20-km gas, sulfate and silica optical depths at 8.80, 11.40 and 20.40 um), `reproduce/mir_detector_etc.py` (K_GEOM, element geometry), the sourced detector D* values in the script | `tables/radiometric_budget.tex`, `outputs/radiometric_budget.{json,txt}` | FAST | VOx microbolometer D* 1e9 (sourced; 2e10 is the ideal 300-K limit), 0.5-s and 0.17-s dwells, 100 reference samples, absorptance 0.8; transmission SNR 1.4-2.4e4 at 8.80 um; the systematics cap S = 2000 limits every chain | rc5-round70 |
| Table E1, `tables/ace_residual_floors.tex` (`tab:ace_floors`, App. E) | `reproduce/ace_floor/residual_floor_paper1.py`, step `tabE1` (see Fig. E1) | D-FLOOR `w0p1/` and `w0p25/` | `tables/ace_residual_floors.tex` (the exporter asserts every rounded value quoted in App. E against the archive; `tests/test_package.py::test_residual_floor_archive_quotes`) | FAST export; archive FULL (`floor1`, `floor25`) | as Fig. E1 | c523d6c |
| Table F1, `tables/element_width_scan.tex` (`tab:element_width`, App. F) | `reproduce/paper1_tables_round53.py`, step `tab53` | `outputs/resolution_sensitivity_calibrated.json` (step `resol`, FULL) and `outputs/ace_v52/nulltest_resolution.json` (step `nullr`, FULL 10 min; reads D-ACE143, D-ACEREV, D-FLOOR `w0p1/atlas.json`, A-CBT, `nulltest_convention.json`) | `tables/element_width_scan.tex` | FAST (table) over FULL archives | medians over five placements of the element grid; ACE null in the per-shell convention | c523d6c |
| Table F2, `tables/coating_scans.tex` (`tab:coating_scans`, App. F) | steps `coat` (`sensitivity_sulfate_coating.py`, 8 min) and `comp` (`sensitivity_composite_particle.py`, 8 min) | A-CBT; D-OPT-SiO2; D-OPT-H2SO4 (member); D-OPT-MIN (calcite) | `outputs/sulfate_coating.json` (`sets/*/<mu>/mmin_tg`, `shell_nm_at_rmed`, `peak_um`), `outputs/composite_particle.json` (`sets/*/<f_sil>/mmin_tg_total`, `shell_nm_at_rmed`, `silica_mass_fraction`, `peak_um_8_9`) | TYPED from the archives; both steps FULL. The manuscript's table comment cites the working-repository scripts (`scripts/sensitivity_sulfate_coating_round62.py`, `scripts/sensitivity_composite_particle_round63.py`); the public scripts are their ports and reproduce the same archives (rounds 62-63 commit) | Bohren-Huffman coated spheres (BHCOAT, validated against `saimon.mie`); sulfate density 1620.9 kg m^-3 (the member), silica 2200 kg m^-3; outer radius at the reference dispersion | c523d6c |
| Table F3, `tables/altitude_scan.tex` (`tab:altitude_scan`, App. F) | step `design` | `outputs/design_sensitivity_calibrated.json` `sets/{band 8-13 @0.1, window 7.8-9.3 @0.1}/altitude/{16,18,20,22,24}` (band 0.0561 / 0.0773 / 0.1010 / 0.1252 / 0.1460) with `altitude_floor_scale` from A-ALT | — | TYPED from the archive; `design` is FULL; A-ALT (the per-altitude floors) is not regenerated by any step (HITRAN, `scan_floor_altitude.py`) | per-element floors re-derived per tangent altitude by the OE budget | c523d6c |

## 4. Headline numbers

| Number as quoted | Archive and key | Step (tier) | Inputs | Assumptions |
|---|---|---|---|---|
| Band threshold 0.101 Tg; window 0.141 Tg (Sects. 3, 5; Tables 5, 6) | `outputs/calibrated_background_thresholds.json` `results/"CALIBRATED quiet (LM65T223, 2-comp)"/"band 8-13 @0.1: measured floors"/"full 7-param"/"triplet + band01"/mmin_tg` = 0.10104 and `.../"window 7.8-9.3 @0.1: measured floors"/"full 7-param"/"triplet + window"/mmin_tg` = 0.14115 (asserted by `tests/test_package.py::test_thresholds_archive`) | `cbt` (FULL, 30-60 min) | D-GLO (20-25 N quiet-epoch 525-nm profile), D-OPT-SiO2, D-OPT-H2SO4, D-FLOOR (`w0p1_c880/atlas.json`, `w0p25/`), the 8.80-um budget (`saimon/onion_peel.py` `BUDGET_OD_880`; A-FLOORW) | 47 elements 7.80 ... 13.20 um (O3 core 9.3-10.0 um excluded); heritage triplet 448/756/1544 nm; per-shell convention at 20 km with g = 1.097; floors = measured atlas SDs x 1.327 |
| ACE full-spectrum null test: sigma(M) = 0.026 Tg, M_min = 0.078 Tg slant, 0.18 Tg per shell (Sect. 4.3, Tables 3, 4) | `outputs/ace_v52/fullspectrum_stats.json` `full/sigma_emp` 0.02594, `full/median_M` 0.01125, `full/mmin_3sigma` 0.07783, `n` 101; `outputs/ace_v52/nulltest_convention.json` `rho` 2.3346 (0.0778 x 2.33 = 0.18) | `null` (FAST, 5 min), `nullc` (FAST, 2 min) | D-ACE143, D-ACEREV (`basis.npz`, `matrix_19_22.npz`), A-CBT, D-OPT-SiO2 | 101 occultations at the tangent nearest 20.5 km; bin-wise self-calibration; empirical sigma = robust scatter of the fitted masses |
| 8.80-um design floor 2.466e-3 OD = 1.692e-8 m^-1 | A-FLOORW `outputs/round43_tracegas/gas_removal_floor_widths.csv` row `target_um` 8.8, `width_um` 0.1, `sigma_removal_od` 2.4656e-3; `saimon/onion_peel.py` `BUDGET_OD_880`, `SIG_880_ABS` (asserted by `test_onion_peel_constants`); A-CBT `floor_anchor/budget_od_880` | `floorw` (FULL, HITRAN, 1 min); `onion` (FAST) | D-XSEC, D-TG (AFGL US-standard), D-OPT-SiO2 | R ~ 100 optimal-estimation co-retrieval of O3 + N2O over the window; tau_O3 = 1.154 at the element |
| Measured 8.80-um floor 1.86e-3 OD (1.858e-3), 95 % interval [1.63, 2.26] (Table E1; App. E) | D-FLOOR `w0p1_c880/atlas.json` and `w0p1/atlas.json` (19-22 km, 8.80-um element; bit-identical between the two archives, `test_c880_archive_anchor`); `tables/ace_residual_floors.tex` row "Silica 8.80"; A-CBT `floor_anchor/measured_sd_880_od` 0.00186, `floor_scale` 1.327 | `floorc`, `floor1` (FULL, 25-30 min each); `tabE1` (FAST) | D-ACE143 | held-year prediction error after the ensemble fit; one native spectrum per occultation; 500 bootstrap draws |
| Onion-peel gain g = 1.097, P_kk = 159.9 km, |G_k| = 6.863e-6 m^-1 per OD (App. B, D) | `saimon/onion_peel.py`; `outputs/onion_peel_geometry_exact.txt` | `onion` (FAST) | none (geometry) | edge-bounded 0.5-km shells, R = 6371 km, z = 20 km |
| Latitude coverages of the 20-km bin and of the best height; worst best bin (Table 7; Sect. 6) | `outputs/latitude_scenarios.json`: `fixed20_cov` = [-63, 63] / [-61, 63] / [-69, 69] / [-67, 69] deg, `best_cov` = [-89, 89] / [-69, 73] / [-89, 89] / [-77, 87] deg, `best_worst` = 0.086 / 0.209 / 0.082 / 0.135 Tg, `best_min` = 0.026 Tg, for the four scenarios in order. Related map keys: `results.json` `maps/pm41/band/quiet/oe/lat_range_20km_below_0p1` = [-69, 73]; `results_o3lat.json` same key = [-77, 85] (the values quoted in the README and guide). The scenario definitions differ between the table and those keys — to confirm that each sentence of Sect. 6 cites the intended one | `tablat` (FAST) over `fig6`, `fig6o3` (FULL) | A-MAP | coverage = threshold below 0.1 Tg; symmetric, year-round extension of the measured ozone ratios |
| Campaign gains 1.89 (ISS-like orbit) / 1.56 (dedicated 70 deg), 1.8 / 1.4 with the measured ozone floor; per-sounding peak gain 3.9 (Sect. 6) | `outputs/detectability_2d_calibrated/results.json` `scenarios/combined/band/quiet/gain_ISS-like` 1.8915, `gain_dedicated 70` 1.5637; `orbits/ISS-like/per_day` 30.22, `lat_min/max` -70.3 / 69.9; `orbits/dedicated 70/per_day` 25.35, +-86.8; `results_o3lat.json` 1.8301 / 1.4183. Peak gain 3.9 = `ref_20km/band` 0.10104 / `maps/pm41/band/quiet/oe/best_mmin` 0.0261 (to confirm) | `fig6`, `fig6o3` (FULL) | A-MAP inputs as Fig. 5 | matched filter over the year-one sampling; systematic floors not averaged as sqrt(N) |
| Materials: calcite 0.78, dolomite 0.75, alumina 0.150 Tg (silica 0.101) in the design band; far-IR columns (Table 8; Sect. 7) | `outputs/materials_calibrated/results.json` `materials/<m>/band/quiet` = 0.7818 / 0.7459 / 0.1499 / 0.1010; `band/postHT`; `farir/triplet_plus_farir_quiet` = 0.0329 / 0.0408 / 0.1306 / 0.0475; `farir/band_plus_farir_quiet` = 0.0241 / 0.0270 / 0.0765 / 0.0362; `farir/floor_scan` for the 3x / 10x brackets | `mat` (FULL) | A-CBT, A-ENG, A-FARIR, D-OPT-MIN | as Table 8 |
| Ash positive control: 56 spectra of 41 occultations; ash above 3 sigma in 53/56 against 1/101 in the null sample; ash shape preferred in 54/56; silica channel not fooled; column 1.8x the published value for 9 matched spectra (Sect. 4.3, Table 4) | `outputs/ace_v52/ash_positive_control.json` `n_ash_spectra` 56, `n_ash_occ` 41, `n_null` 101, `ash/frac_ash_gt3` 0.9464, `null/frac_ash_gt3` 0.0099, `ash/frac_prefer_ash` 0.9643, `ash/frac_sil_gt3` 0.0, `ash/M_sil_both_median` -0.0121 (robust SD 0.1019), `lecours_compare/ratio_median_fixed_psd` 1.841, `n_lecours_retrieval` 9 (asserted by `test_ash_positive_control_archive`). 42 flagged occultations are shipped (`ash_manifest_v52.txt`, `test_ash_occultations_shipped`); 41 contribute spectra at 9-14.5 km — to confirm the reason for the one difference | `ash` (FAST, 5 min) | D-ACE42, D-LEC, D-ACEREV, D-OPT-ASH, A-CBT | ash optics of Deguine et al. (2020), r_m = 0.085 um, S = 1.6; the same eight-parameter fit as the null test with an ash amplitude added |
| False alarms: 3.64e5 tests per year (33 bins x 30.2 events per day); 4.54 sigma per test for one false alarm per year, decision threshold 1.51 x M_min (0.15 Tg), P_det 0.06 at M_min and 0.93 at 2 M_min; 1.33 at one per month; 1.35 bare-3-sigma false positives per day (Sect. 5) | `outputs/false_alarm_trials.txt`, block "all 33 bins per event" | `far` (FAST) | none (Gaussian statistics; the event rate from `results.json` `orbits/ISS-like`) | Gaussian, independent errors |
| Silica extinction peak 8.80 um (Sect. 3.1) | `outputs/silica_peak_wavelength.txt` ("KitamuraPopova rmed= 268 sg=1.31: mono peak 8.800 um"; Franta 2016: 8.760 um); `test_silica_extinction_peak_is_at_8p80_um` | `peak` (FAST) | D-OPT-SiO2 | lognormal PSD r_med = 268 nm, sigma = 1.31 |
| Measured slant tau_O3 ratios 0.92 / 0.94 / 0.50 at a 20-km tangent (floor factors 0.93 / 0.95 / 0.60) at 20-25 N, 35-55 N, 60-90 S (Sect. 6) | `outputs/o3_slant_latitude_measured.json` `bands/<band>/ratio_median[16]` (h_tan 20.0 km) = 0.9185 / 0.9376 / 0.5024, `floor_factor[16]` = 0.9313 / 0.9473 / 0.6044, `n` = 138 / 17 / 27 of `per_occultation_n` 192 (`test_o3_latitude_measured_extract`) | `o3meas` (FAST) | D-ACEO3, A-CBT, A-FLOORW | reference = US Standard 1976 density x AFGL US-standard O3 VMR; floor line term linear in tau_O3 |
| Vertically coherent database term (Table 6 row; App. D.6): band 0.078, window 0.109 Tg | `outputs/m2_vertical_correlation_floor.json` `thresholds_tg/{band,window}/mixed_measured` = 0.0782 / 0.1093 (`mixed` = 0.0432 / 0.0603; `ratios/white_over_coherent_line` 5.18) | `m2` (FAST) | A-CBT, A-FLOORW | the line-intensity term coherent between tangent heights, the measured scatter white |
| 0.1 Tg of silica in observables: 0.091 cm^-3 at 20 km, 8.3e4 cm^-2, 9 % of the background's particles above 0.15 um, +16 % / +33 % of the vertical OD at 525 nm / 8.80 um (Sects. 1, 8) | `outputs/signal_vs_background.json` `silica/0.1/{n20_cm3 0.0915, column_cm2 8.34e4}`, `ratios_0p1Tg_over_background/{n20_cm3_over_bg_above_015um 0.0919, vert_525 0.165, vert_880 0.334}` (`test_signal_vs_background_archive`) | `obs` (FAST, 1 min) | D-OPT-SiO2, D-GLO, D-OPT-H2SO4 via the background module | calibrated two-component background (modes 60 nm / sigma 1.6 and 1500 nm / sigma 1.8) |
| GloSSAC anchor at 20-25 N: 0.00451 (median SAOD525, 2016.5-2018.5, unchanged on the SAGE III/ISS-only epoch; Sect. 2.2) | `outputs/glossac_epoch_check.txt` (median 0.00451 both epochs; ratio 1.000). A-CBT `saod_cal_20_25N_quiet` = 0.00446 is a different statistic of the same slice — to confirm which one each sentence quotes | `gepoch` (FAST); `cbt` | D-GLO | — |
| Reservoir masses, App. A: quiet 0.40, elevated 1.00, post-eruption 24.2 Tg H2SO4 | `outputs/reservoir_mass_saod.json` `presets/{quiet,elevated,post_eruption}/M_H2SO4_Tg` (`test_reservoir_mass_archive`) | `mass` (FAST, 1 min) | D-GLO, D-OPT-H2SO4 | — |
| Sulfate shell / composite particle thresholds (Table F2; Table 6) | `outputs/sulfate_coating.json` `sets/"band 8-13 @0.1"/{0.1,0.3,1.0,3.0}/mmin_tg` = 0.116 / 0.138 / 0.175 / 0.204; `outputs/composite_particle.json` `sets/"band 8-13 @0.1"/{0.0,0.25,0.5,0.75,1.0}/mmin_tg_total` = 0.78 / 0.77 / 0.52 / 0.34 / 0.10 (`test_coated_sphere_archives_share_the_bare_silica_endpoint`) | `coat`, `comp` (FULL, 8 min each) | A-CBT, D-OPT-SiO2, D-OPT-H2SO4, D-OPT-MIN | BHCOAT coated spheres |
| Injection-recovery (spike) test: 0.1 Tg injected returns 0.111 Tg median (the null's +0.011 bias), scatter unchanged 0.026 Tg (Table 4) | `outputs/ace_v52/spike_test.json` `0.1/full/{median 0.1112, sigma_emp 0.0259}` | `spike` (FAST, 1 min) | D-ACE143, D-ACEREV, A-CBT | synthetic layer multiplied into the residual transmittance before the fit |
| Trace-gas budget scenarios and SNR trade, App. D | `outputs/o3_removal/*.csv` (step `oe`), `outputs/snr_trade/*.csv` (step `snr`), `outputs/highres/highres_gas_removal_budget.csv` (step `highres`), `outputs/round43_tracegas/*.log` (all HITRAN steps of round 43), `outputs/round53_tracegas/*.log` (re-runs with the exact gain) | `oe`, `snr`, `highres`, `limb` (FULL, HITRAN); `cont`, `cfc` (HITRAN-full); `stT`, `lines` (HITRAN line lists) | D-XSEC, D-TG, D-OPT-SiO2 | the logs are the archived record of every HITRAN-dependent run |
| Round 70 (referee round 5): gas optical depth of the 8.80-um element -ln<T> = 0.41 against <tau> = 1.19; element-observable removal error 5.1e-4 OD (photon 4.1, line 2.3, prior 2.1 e-4); joint window 0.068 / band 0.057 Tg per shell; two-stage on the same floors 0.023 Tg; paper floors as covariance with the gas state free 0.184 Tg; plateau 0.047 Tg, crossover S ~ 2200; unretrieved registration error 0.003 um -> +0.021 Tg, retrieved to 0.35 nm; vertical dz* 23.6 km (Sects. 3, 4.1, App. D) | A-CLOSURE `observable/{y_gas_880 0.4062, mean_tau_gas_880 1.1856}`, `removal_budget_880/0.02/snr2000/{sigma_total 5.115e-4, photon, line, prior_smoothing}`, `joint_vs_two_stage_window/0.02/{joint_window_triplet_shell, two_stage_window_triplet_shell}` (x3), `joint_vs_two_stage_window/paper_floors_joint/window_triplet_shell`, `joint_band/per_eta/0.02/joint_band_shell`, `snr_scan/{plateau_shell_triplet, snr_crossover_sqrt2}`, `response_knowledge/shift_0.003/silica_error_shell_tg`, `response_knowledge/posterior_sigma_shift_um`, `vertical_conversion` (`test_element_observable_closure_archive`) | `closure` (FULL, HITRAN, 45 min) | D-XSEC-FINE, D-LINES, D-ATLAS, A-CBT, A-FLOORW, D-OPT-SiO2, D-OPT-H2SO4 | boxcar response 0.1 um with 0.005-um Gaussian-softened edges; 20-km midlatitude state; S = 2000 per sample; HITRAN code midpoints as source-group priors (O3 code 5 = 5-10 %); independent per-line scatter 2 % |
| Monte Carlo closure: bias -0.001 to +0.004 Tg, scatter 0.020-0.024 against posterior 0.022 Tg, 67-75 % within 1 sigma, 0.1 Tg detected above 3 sigma in 94 % (48 % above 4.54 sigma), no null draw above 3 sigma (App. D, Table D3, Sect. 9) | A-CLOSURE `monte_carlo/baseline/{0.0,0.05,0.1,0.2}/...` and `monte_carlo/<variant>/{0.0,0.1}/...` | `closure` | as above | 100 draws per mass, 33 per variant (binomial resolution +-5 points) |
| marginal() rank re-evaluation: 108 configurations, none newly finite or infinite, max relative change 6e-12 (App. C) | A-RANK `summary/{n_configurations, newly_finite, newly_infinite, max_rel_diff_new_vs_old, max_rel_diff_new_vs_archive 1.7e-5}` (`test_marginal_rank_archive`) | `rank` (FAST, 3 min) | A-CBT, A-CBP, A-ENG | SVD_RTOL 1e-10, TPERP_RTOL 1e-8 |
| Latitude on the declared 20.47-km level: f = 3.78 in the cores, 0.23-0.29 at the poles, 0.026 Tg core threshold, 0.43 Tg at 87 S, coverage 63 S-63 N; campaign common-error rho_c 0.18 -> sigma 2.5e-3 Tg; core vertical response: 1-km / 2-km smoothing costs 16 / 37 % (Sect. 6, Table 7) | A-LAT70 `quoted_number_map[*]/{quoted, value, level_km}`, `vertical_response`, `campaign/table_level_reference_floor` (`test_latitude_declared_level_archive`) | `lat70` (FAST) | A-MAP, D-MOD, `outputs/m2_vertical_correlation_floor.json`, A-ACEV52 `fullspectrum.csv` | the model level nearest 20 km is 20.4656 km (0.88 km thick); exact-20-km values are interpolated |
| Radiometric budget: transmission SNR 1.4-2.4e4 at 8.80 um for a VOx microbolometer with sourced D* 1e9 (Table D5) | A-RADIO `chain/8.80/vox_microbolometer/dwell_{0.5,0.17}/result/SNR_T_ratio` (23640, 13650), `sourced_detectors` (`test_radiometric_budget_archive`) | `radio` (FAST) | A-LAND, `mir_detector_etc.py` | see Table D5 row |
| Uniaxial optics (Table 8): dolomite 0.75 -> 1.16 Tg, calcite 0.78 -> 0.80, alumina 0.150 -> 0.157 under the Clausius-Mossotti average; (b)-(c) bracket <= 2.4 %; ray mean overestimates the Rayleigh peak absorption by 3.0 / 4.4 / 3.1; k +-20 % moves alumina to 0.10-0.88 Tg | A-UNIAX `materials/<m>/{ratio_b_over_a, ratio_c_over_b, b/mmin_tg, rayleigh_test_r10nm_band}`, `perturbations/materials/alumina` (`test_materials_uniaxial_archive`) | `uniax` (FULL, 30 min) | A-ENG, A-CBT, D-OPT-MIN, D-OPT-SiO2 | Rayleigh-exact orientation average of the polarizability (Bohren & Huffman 1983, Ch. 5); the far-infrared and composite-particle numbers are a separate re-computation (round 71, in progress in the working repository) |
| PSD tail: truncating the measured distribution above 4 um understates the threshold by 1.5 % (D300o; 11-12 % for D300 / D500); > 10-um particles hold 94-97 % of the removed mass and leave a 1-km shell within 1.2 days; coagulation halving time 231 yr at the threshold density 0.093 cm^-3 (App. A, Sect. 8) | A-TAIL `tail/cases/<case>/{measured_tail,flat_4_10um}/mmin_ratio_to_lognormal`, `settling/diameters`, `coagulation/{K_psd_mean_cm3_s, n_threshold_0p1Tg_cm3, halving_time_uniform_yr}` (`test_psd_tail_coagulation_archive`) | `tail` (FAST, 5 min) | D-DIG, A-ENG, A-CBT, D-OPT-SiO2 | Fuchs kernel, slip-corrected Stokes settling, US-standard 20-km state |
| Null-test diagnostics (Tables 3-4, Sect. 4.3): full-band weighting 8e-3 OD per 2-cm^-1 bin on 8.00-13.16 um, chi2 0.30, robust 0.0259 / SD 0.0218 / formal 0.0232 Tg, median +0.0112; 42 of 143 fail the two-spectra-above-30-km rule; drift +0.0228 +- 0.0026 Tg per decade; n_eff 36 (ICC 0.48); ash log ratio 0.85 +- 0.21 (factor 2.3), chi2 50 for 9 d.o.f. | A-NULLD `rows/full/*`, `sample/{selection, full/trend, full/variance_components}`, `chi2_coverage`, `svd`, `nonlinear_backgrounds`, `ash` (`test_nulltest_diagnostics_archive`) | `nulld` (FULL, 15 min) | D-ACE143, D-ACEREV, A-ACEV52, A-CBT | the same fit code as step `null` (`ace_fullspectrum_retrieval.py` imported) |
| Null-test convention conversion rho = 2.33 at 20.5 km (Table 3) | `outputs/ace_v52/nulltest_convention.json` `rho` 2.3346, `convention/median/{slant_od_per_tg 0.2093, shell_od_per_tg 0.0896}` | `nullc` (FAST) | A-CBT, `outputs/ace_v52/fullspectrum.csv` | per-Tg layer 16 km-30 hPa seen at the occultation's tangent |

## 5. Supplied artifacts

Every figure in the paper is drawn by a script in `reproduce/`; there is
no paper figure that exists only as a file. What `--fast` does **not**
redraw, so that the shipped PNG is the deliverable in that tier:

| Figure | File | Drawn by | Redraw from the archive | Regenerate the archive |
|---|---|---|---|---|
| Fig. 5 | `figures/detectability_2d_mass_contours.png` | `reproduce/detectability_2d_calibrated.py` (step `fig6`, FULL) | `python reproduce/detectability_2d_calibrated.py --plot` (seconds) | `python make_all.py --only fig6` (35 min) |
| Fig. F1 | `figures/resolution_sensitivity_calibrated.png` | `reproduce/resolution_sensitivity_calibrated.py` (step `resol`, FULL) | `... resolution_sensitivity_calibrated.py --plot` | `--only resol` (20 min) |
| Fig. F2 | `figures/design_sensitivity_calibrated.png` | `reproduce/design_sensitivity_calibrated.py` (step `design`, FULL) | `... design_sensitivity_calibrated.py --plot` | `--only design` (20 min) |

The working-repository scripts that originally produced every figure have
the same file names under `scripts/` of the analysis repository
(`plot_slant_od_landscape_paper.py`, `plot_refractive_index_paper.py`,
`plot_spectral_shapes_paper.py`, `plot_ace_atlas_paper.py`,
`ace_mir_aerosol.py`, `detectability_2d_calibrated.py`, `fit_segev_psd.py`,
`reproduce_glossac_tropical_saod.py`, `resolution_sensitivity_calibrated.py`,
`design_sensitivity_calibrated.py`; the floor atlas by
`SideStudies/ace_mir_sulfate/scripts/residual_floor_paper1.py`); the
`reproduce/` versions are their ports onto the `saimon` package and the
`data/` layout, and `reproduce/compare_reference.py` checks that they
regenerate the archived numbers (JSON to 1e-3 relative) and figure files.

Figures under `--fast` whose *drawing* is live but whose *physics input* is
an archived HITRAN product (replay of the gas model, recomputation of
everything else): Fig. 2 (A-LAND) and Fig. 3 (A-SCISAT); see Sect. 6.

## 6. Fast replay versus full recomputation

`make_all.py` has two tiers (`python make_all.py --list` prints every step
with its tier and runtime):

**`--fast` (about 25 min).** Runs the 40 steps not marked heavy (34 before round 70; `tabD2`, `rank`, `ranktest`, `lat70`, `radio` and `tail` are new, and `fig4` now redraws Fig. 3 from the closure archive). It
*recomputes* from the shipped `data/`: the ACE-FTS demonstration (Fig. 4,
the null, injection-recovery, convention, element-sampling, Ruang, epoch and
ash-control steps: `fig5`, `null`, `nullc`, `spike`, `elem`, `ruang`,
`epoch`, `ash`), the reservoir masses, the silica peak and PSD checks, the
GloSSAC figure and epoch check, the Segev fit, the refractive-index and
spectral-shape figures, the onion-peel geometry, the vertical-correlation
bracket, the false-alarm statistics, the measured-ozone latitude ratios, the
observables of 0.1 Tg, the Table B1 predicted column, and the detector
numbers. It *replays* from archives: Fig. 2 (gas curves from A-LAND),
Fig. 3 (gas model from A-SCISAT), Table E1 / Fig. E1 (export from the
verified floor archive D-FLOOR), Tables 3, F1 and 7 (generated from the
archived JSON / maps). It does *not* touch Figs. 5, F1, F2 (Sect. 5).

**`--full` (several hours).** Additionally runs the 24 heavy steps (21 before round 70; `closure`, `uniax` and `nulld` are new), in
dependency order, which recompute every archive in `outputs/` and the floor
archives in `data/ace_floor/` from the shipped raw-side inputs: `cbt`, `cbp`
(thresholds and Fisher inputs from D-GLO, D-OPT-*, D-FLOOR), `roster`,
`floor1`, `floor25`, `floorc` (the residual-floor pipeline from D-ACE143;
each run re-verifies its own score tables and hashes), `design`, `resol`,
`nullr`, `mat`, `fig6`, `fig6o3`, `coat`, `comp`, and the HITRAN steps `oe`,
`floorw`, `snr`, `limb`, `highres` (from D-XSEC) plus `cont`, `cfc`, `stT`,
`lines` (which need the unshipped line lists and will fail without
`fetch/build_hitran_cache.py`). After a full run,
`python reproduce/compare_reference.py` compares 22 archived JSON files with
`reference_outputs/outputs/` number by number and lists the figures.

**What `--full` does not recompute** (shipped as derived products, each
with its regeneration script):

| Derived input | Regenerated by | Needs |
|---|---|---|
| D-XSEC, the HITRAN cross-section grids | `fetch/build_hitran_cache.py` | `hitran-api`, hitran.org (hours); fetches that stall can leave an empty table (delete and rerun) |
| A-LAND, A-SCISAT, A-ALT, A-FARIR, `outputs/{alumina,calcite,dolomite}/` | `reproduce/plot_slant_od_landscape_paper.py --recompute`, `reproduce/analyze_ace_atlas.py --rebuild`, `reproduce/scan_floor_altitude.py`, `reproduce/farir_band_floors_w0p1.py`, `reproduce/material_trace_gas_removal.py`, `reproduce/calcite_trace_gas_removal.py` | D-XSEC or the full HITRAN cache; none of these is a `make_all` step (to confirm whether they should be added as a third tier) |
| D-ACEREV (common-grid matrix, ensemble basis, optics library) | `reproduce/ace_floor/revision_*.py` | D-ACE143, D-OPT-H2SO4 |
| D-GLO | `fetch/reduce_glossac.py` | the 530-MB GloSSAC V2.23 file (NASA Earthdata login) |
| D-ACE143, D-ACE42 | `fetch/reduce_ace_v52.py` | the registered ACE-FTS v5.2 release (Sect. 7) |
| D-ACEO3 | `fetch/reduce_ace_o3_l2.py` | the same release (Level-2 ascii files) |
| D-ATLAS | `fetch/reduce_ace_atlas.py` | the public ACE atlas files |

Not reproducible from this repository at any tier, because the inputs are
not ours to redistribute or not data at all: the SAGE III/ISS Level-2
uncertainties of Table B1 (literature values, Wrana et al. 2021), the
detector heritage of Table D2 (literature), the ACE Level-1-to-residual
step (the ACE team's retrieval; we start from the released residuals), the
GloSSAC gridding (NASA; we start from the released file), and the HITRAN
line-list download itself.

## 7. Data access and redistribution

* **ACE-FTS v5.2 Level-2 residual product.** FRDR, doi:10.20383/103.01245
  (Bernath et al., 2005; Boone et al., 2020). The ACE team distributes it on
  registration (https://ace.scisat.ca). Registration gates *access* to the
  full release; it does not gate the reproducibility of this paper, because
  every ACE-dependent result starts from the extracts shipped here.
* **What is shipped.** `data/ace/v52_subset/`: the residual spectra at all
  tangent heights of the 143 occultations at 20.1-25.0 N, June-August
  2004-2024 (`data/ace_floor/sample.csv`) and of the 42 occultations of the
  release flagged as ash-sulfate mixtures by Lecours et al. (2026)
  (`data/ace/ash_lecours2026/ash_manifest_v52.txt`), 185 occultation
  directories in all, plus the matching rows of `occultationlist.csv` (184
  rows: one ash occultation has no row in the release's list). The files
  are the released residual files rewritten with shortest-round-trip number
  formatting and gzip compression — **bit-identical values** — by
  `fetch/reduce_ace_v52.py` (`tests/test_package.py::test_ace_subset_round_trip`
  checks one file against the release's values). `data/ace/o3_l2_v52/` is a
  column extract of the Level-2 ozone profiles of 192 occultations made the
  same way (`fetch/reduce_ace_o3_l2.py`). The extracts are shipped with the
  ACE team's knowledge (see the Code and data availability statement of the
  paper and `data/PROVENANCE.md`); they are released here under CC BY 4.0 as
  derived data, and the ACE-FTS data policy and citations apply to any reuse.
* **What a reader without registration can recompute.** Everything
  downstream of the shipped extracts: the residual-floor pipeline (steps
  `floor1`, `floor25`, `floorc`, from the raw residual spectra), the
  demonstration of Sect. 4 (Fig. 4, the null, spike, convention,
  element-width and ash-control tests), the measured-ozone latitude ratios,
  and everything that depends on them (every threshold, table and map).
* **What a reader without registration cannot recompute.** The extraction
  itself — selecting the 143 + 42 occultations from the 5460 of the release
  and the 192 ozone profiles from the Level-2 ascii files — and the
  metadata audit that `residual_floor_pipeline.py` runs when the ACE
  `data_issues.csv` is present (that file is never packaged; the pipeline
  skips the audit without it). A registered reader can repeat the extraction
  with `fetch/reduce_ace_v52.py /path/to/ace_v52` and verify that the
  result is byte-identical to the shipped files with `CHECKSUMS.sha256`.
* **ACE Atmospheric Atlas** (`data/ace_atlas/`): public download
  (https://ace.scisat.ca), spectral-window extract.
* **GloSSAC V2.23** (`data/glossac/`): NASA ASDC, free Earthdata login;
  a 7-MB subset of the 530-MB file with the original variable names.
* **HITRAN2020**: line lists are fetched from hitran.org by HAPI and are not
  redistributed; the shipped `xsec_cache/` grids are our computed products.
* **Lecours et al. (2026) tables** (`data/ace/ash_lecours2026/`): Zenodo
  doi:10.5281/zenodo.21479502, file copies with their README.
* **Optical constants, digitizations, the 2-D model field**: see
  `data/PROVENANCE.md` (refractiveindex.info CC0 files, cited papers,
  ARIA, this group's own work).

## 8. Persistent archive

* **Zenodo deposit**: DOI to be inserted at submission (placeholder:
  `10.5281/zenodo.XXXXXXX`), capturing the tagged release of this
  repository (tag to be created at HEAD `c523d6c` or at the final revision
  commit; the paper's Code and data availability statement will cite the
  DOI). The deposit will contain the repository tree (code, `data/`,
  `outputs/`, `figures/`, `tables/`, `reference_outputs/`) with this
  manifest and `CHECKSUMS.sha256`.
* **Licences**: code MIT (`LICENSE`, Doron Kushnir, Stardust Solutions
  Ltd.); derived data in `data/` and the figures CC BY 4.0 (`data/LICENSE`),
  with the third-party sources' own terms and citations as listed in
  `data/PROVENANCE.md`.
* **Provenance**: `data/PROVENANCE.md` (per directory: source, reduction,
  regeneration script), `fetch/README.md` (how to obtain the originals),
  `data/ace_floor/*/manifest.json` (hashes of every file of the floor
  archives, verified by `residual_floor_pipeline.py --finish-only` and by
  the exporter), `ace_floor_export.json` (hashes of the archive manifests,
  the exporter and the exported table and figure).
* **Checksums**: `CHECKSUMS.sha256` at the repository root lists the SHA-256
  of every file tracked under `data/` (3835 files) and `reference_outputs/`
  (40 files), 3875 lines, paths relative to the repository root. Verify
  with

  ```bash
  shasum -a 256 -c CHECKSUMS.sha256          # prints one OK per file
  shasum -a 256 -c CHECKSUMS.sha256 --quiet  # silent when everything matches
  ```

  (`sha256sum -c CHECKSUMS.sha256` on Linux). Regenerate after changing a
  shipped file with `git ls-files -z data reference_outputs | sort -z |
  xargs -0 shasum -a 256 > CHECKSUMS.sha256`. Local-only, git-ignored files
  (`data/trace_gases/hitran_cache/`, `data/ace_floor/w0p25/matrix_*.npz`)
  are not listed because they are not part of the repository.

## 9. Open items marked "to confirm" in this manifest

Collected here so that they can be closed one by one: the HITRAN edition
behind A-LAND (Fig. 2); the optional `w0p1_874` path in `ace_mir_aerosol.py`
(Fig. 4); the source of the photon-noise term in Table 2; the Ruang row of
Table 4 (printed, not archived); the floor range 1.7-9.5e-8 m^-1 of
Table 5; the coagulation row of Table 6 (`agglomerate` set and the Lederer
mass fractions); the observed column of Table B1; the per-sounding peak
gain 3.9 (ratio of archive keys); the 41-versus-42 ash occultations; the
scenario behind each coverage sentence of Sect. 6; the 0.00451 versus
0.00446 GloSSAC anchors; whether the non-step producers of A-ALT, A-FARIR,
A-SCISAT, A-LAND and the material trace-gas floors should become a third
`make_all` tier; the release tag and the Zenodo DOI. Round 70 port: the
shipped closure archive is the working repository's `--ndraw 100` run; the
port was verified in this tree with `--ndraw 5` (every deterministic block
identical, the Monte Carlo statistically consistent), see
`tests/test_package.py` and the port note of referee round 5; the far-infrared
and composite-particle columns under the orientation-averaged optics (round
71) are not yet ported.

## 10. Rounds 71 and 74 (referee round 6; tag `rc6-round74`)

The commits of the round-70 port (`2cdb0f1`, `b846684`, `cb13365`, tag `rc5-round70`) were made on 25-26 October 2026 but not pushed until this round; the public HEAD seen by the referee on 27 October was `c523d6c`. The state tagged `rc6-round74` contains them and the rows below. Every archive listed here reproduces the working repository's to the printed precision; the FULL steps were verified by a short run where noted.

| Archive / object | Script and step | Inputs | Outputs | Tier | Verified |
|---|---|---|---|---|---|
| A-FARIR71 `outputs/materials_farir_uniaxial{,_silica,_calcite,_dolomite,_alumina,_red}.json` (Table 8 far-infrared and red-extension columns, Sect. 7, App. F4; round 71) | `reproduce/materials_farir_uniaxial.py`, step `farir71` | A-CBT, A-ENG, A-FARIR floors (`outputs/{calcite,dolomite,alumina}/*_band_floors_20km.csv`), D-OPT-MIN, D-OPT-SiO2 | per-material and merged archives | FULL (~1 h per material) | archives shipped from the working run; not rerun here |
| A-COMP71 `outputs/composite_particle_uniaxial.json` (Table F2 calcite-core rows; round 71) | `reproduce/sensitivity_composite_particle_uniaxial.py`, step `comp71` | A-CBT, `outputs/composite_particle.json`, D-OPT-MIN (calcite rays), D-OPT-SiO2 | archive | FULL (8 min) | rerun here in round 75: every value within 1e-12 of the shipped archive |
| A-NLREFIT `outputs/ace_nulltest_nonlinear_refit{,_narrow,_narrow_nonneg,_nonneg,_wide,_wide_nonneg}.json`, `_perocc*.csv`, `mie_qext_tables.npz` (Sect. 4.3: null median -0.009 to +0.010 Tg over the background representations, robust scatter 0.015-0.022 Tg, size parameters at a box boundary in 97-100 % of fits; round 71) | `reproduce/ace_nulltest_nonlinear_refit.py`, step `nlrefit` | D-ACE143, A-ACEV52 (`basis_spectra_cal.npz`), A-CBT, D-OPT-H2SO4 | archives | FULL (hours, multiprocess) | archives shipped |
| A-FPC `outputs/closure_flat_prior_check.json` (App. D4, Table D2: the round-70 joint model under a flat aerosol prior; window 0.068 -> 0.094 Tg, hybrid band 0.057 -> 0.061 Tg) | `reproduce/closure_flat_prior_check.py`, step `fpc` | A-CLOSURE machinery (D-XSEC-FINE, D-LINES), A-CBT | archive | FAST (10 s) | rerun here, identical |
| A-UMODEL `outputs/uncertainty_model.json`, `uncertainty_model_audit.npz` (Table 2 all rows; Sect. 3.2-3.3; App. D4: primary model band 0.1187 / window 0.1678 Tg with the triplet, 0.1727 / 0.3363 alone; scenarios; prior scan; response derivatives; App. D9 multi-tangent rows and the coherent-O3 ratio 0.22; the exponential-scan ratios) | `reproduce/uncertainty_model.py`, step `umodel` | A-CLOSURE machinery, A-CBT (`window_floors_m1`, `band01_floors_m1`, `floor_anchor/floor_scale`), `outputs/ace_v52/nulltest_resolution.json` (neighbour correlations 0.609 / 0.140), A-LAND (`slant_od_landscape_curves.npz`, outer-element gas OD) | archive + audit npz (Jacobians, covariances, gain, averaging kernel) | FAST (10 s) | rerun here, identical |
| A-CVAR `outputs/closure_variants.json` (Table D3 lower rows: five fixed databases, out-of-prior x2/x3, Biermann-member truth, line widths x1.1; binomial intervals of the round-70 rows) | `reproduce/closure_variants.py`, step `cvar` (`--ndraw 33 --seed 74`) | A-CLOSURE machinery and archive, A-CBT, D-OPT-H2SO4 (Biermann member) | archive | FULL (15 min) | rerun here in round 75 with the shipped arguments: every value within 2e-9 of the shipped archive (seeded draws) |
| A-GCONV `outputs/grid_convergence.{json,_s0.005.npz,_s0.0025.npz,_s0.00125.npz}` (App. D4: mass error of the 0.005 and 0.0025 grids against 0.00125, +0.0141 and -0.0006 Tg per shell) | `reproduce/grid_convergence.py`, step `gconv`; the 0.00125 grids by `python fetch/build_finegrid_window.py 0.00125` (32 grids, 54 MB, not shipped; ~25 min) and the 0.005 grids by subsampling (done by the script) | D-XSEC-FINE (+ the finer grids), D-LINES | archive; the three driver dumps | FULL, HITRAN | the comparison replays from the shipped dumps; the grids were rebuilt in the working repository |
| A-MSCAT `outputs/materials_scattering.json` (Sect. 7, App. F4: Rayleigh scattering factors true vs effective index, scattering fractions on the reference PSD, perturbation envelopes under convention b: calcite 0.758-0.928, dolomite 0.753-1.162, alumina 0.150-0.368 Tg) | `reproduce/materials_scattering.py`, step `mscat` | A-ENG, A-CBT, D-OPT-MIN, A-UNIAX (silica envelope) | archive | FULL (30 min) | rerun here in round 75: identical to the shipped archive |
| A-MPRIM `outputs/materials_primary.json` (Table 8 'adopted model' column: silica 0.119, calcite 0.410, dolomite 0.783, alumina 0.180 Tg) | `reproduce/materials_primary.py`, step `mprim` | A-UMODEL, A-CLOSURE machinery, A-CBT, D-OPT-MIN, D-OPT-SiO2 | archive | FAST (1 min) | rerun here, identical |
| A-RADOBS `outputs/radiometric_budget_observable.json`, `tables/radiometric_budget.tex` (Table D5 at the element observable: 8.80-um y = 0.406 gas / 0.491 total; SNR_T 2.9e4 / 5.1e4; separated shot, detector, reference and floor terms; calibration tolerances; drift 7.1e-6 per second) | `reproduce/radiometric_budget_observable.py`, step `radobs` (uses the functions of `radiometric_budget.py`) | A-LAND, A-CLOSURE (`observable`), `reproduce/mir_detector_etc.py` | archive, Table D5 | FAST | rerun here, identical |
| A-NULLB `outputs/ace_null_baseline.json` (Sects. 4.3, 5.3: coverage 60/94/99 % about zero, 69/98/100 % about the median; drift 0.0228 +- 0.0026 (OLS) / 0.0028 (year blocks); prospective split 9/54 above 3 sigma; secular ramps absorbed 82 % by a rolling 3-season median) | `reproduce/ace_null_baseline.py`, step `nullb` | A-NULLD (`nulltest_diagnostics_perocc.csv`) | archive | FAST (10 s) | rerun here, identical |
| Tables D2-D3 `tables/element_closure.tex` | `reproduce/element_closure_table.py` (rewritten), step `tabD2` | A-CLOSURE, A-UMODEL, A-FPC, A-CVAR, A-GCONV | `tables/element_closure.tex` | FAST | rerun here |
| Table 2 `tables/error_budget.tex` (TYPED) | from A-UMODEL `models/*`, `correlation`, `audit/prior_width_scan`, A-FPC | — | — | TYPED | values asserted by `test_round74_archives` |
| Table 8 'adopted model' column (TYPED from A-MPRIM) | — | — | — | TYPED | asserted |

Headline numbers of round 74 (Sects. 1, 3.2, 5.3, 9, abstract): sensitivity 0.12 Tg (A-UMODEL `models/B_primary/band_triplet` = 0.1187), window 0.17 Tg (`window_triplet` = 0.1678); decision threshold 0.18 Tg (1.5 x 0.1187); 90 % completeness at 0.23 Tg (1.94 x 0.1187); alumina 0.18, calcite 0.41, dolomite 0.78 Tg (A-MPRIM). The round-70 numbers superseded by them (0.101 / 0.141 Tg; 0.068 / 0.057 Tg) remain in the archives as the diagonal-floor scenario and the spectroscopic model, and are quoted as such.

## 11. Round 75 (completion of referee round 6; tag `rc6-round75`)

The items the round-74 reply listed as not done. One engine (`reproduce/adopted_model_engine.py`) evaluates the adopted measurement model of Table 2 at any tangent height, background amplitude, ozone load and aerosol template (line-resolved gas model at the chord, per-line responses, exact Jacobian of the element observable; at 20 km it reproduces A-UMODEL to 0.1 %; the per-shell template uses the edge-bounded one-shell path over the onion-peel gain, and the sulfate background below 16 km is the full GloSSAC profile, cached in `outputs/adopted_engine_bg_full.npz`). The heavy round-71/74 archives shipped in `rc6-round74` without a public rerun were rerun in this checkout: A-CVAR within 2e-9 (seeded draws), A-MSCAT and A-COMP71 identical, A-GCONV identical after the 0.00125-cm^-1 grids were rebuilt by `fetch/build_finegrid_window.py 0.00125`, A-FARIR71 silica within 3e-15 (the other materials and A-NLREFIT were still running when the tag was made; see the git log for their completion).

| Archive / object | Script and step | Inputs | Outputs | Tier | Verified |
|---|---|---|---|---|---|
| A-ROBUST75 `outputs/adopted_robustness.json`, `_profiles.npz` (Table 6 'adopted model' columns, Table F3 adopted columns, Sect. 5.2: baseline 0.1186 / 0.1676 Tg; third mode x1.51, second member x1.12, width 1.8 +70 %, r_med 600 nm +55 %, coating +24/+68 %, discrepancy term x2 +66 %; altitude 0.076 (16 km) to 0.205 Tg (24 km), 0.170/0.155 with the measured bins; the scenario column reproduces Table 6 of rounds 53-67 to rounding) | `reproduce/adopted_robustness.py`, step `robust` | engine, A-CBT, A-UMODEL, D-OPT-SiO2 (Kitamura-Popova, Franta), D-OPT-H2SO4 (LM65T223, LM72T213, B70T215), `reproduce/sensitivity_sulfate_coating.py` (coated-sphere optics) | archive, Mie-profile cache, `tables/robust_summary.tex`, `tables/altitude_scan.tex` | FULL (4 min; 3 min of Mie builds, cached) | rerun here |
| A-LAT75 `outputs/adopted_latitude.json`, `outputs/detectability_2d_calibrated/map_pm41_adopted.npz` (Sect. 6, Table 7, Fig. 5: 20.47-km level 0.117 Tg, cores 0.031, coverage 61S-61N, best-height worst 0.103 / 0.211 Tg (measured bins), gains 1.89 / 1.56; measured ozone -0.6 / -4 % at 20 km) | `reproduce/adopted_latitude.py`, step `lat75` | engine, A-2D (`map_pm41.npz`: f2d, q_obsc), GloSSAC quiet medians (D-GLOSSAC), `outputs/o3_slant_latitude_measured.json`, A-FLOOR atlas (altitude bins), `saimon.sai_2d` orbits | archive, map, `tables/latitude_scenarios.tex`, `figures/detectability_2d_mass_contours_adopted.png` | FULL (10 min) | rerun here |
| A-MTAN75 `outputs/multitangent_retrieval.json` (App. D9: shells 14-40 km, 199 parameters, 2600 rows; 20-km shell 0.057 Tg with per-shell sulfate amplitudes, 0.072 / 0.092 Tg with 0.5 / 1-km fields of view, 0.063 Tg per 1-km shell, 0.030 Tg with the sulfate profile prescribed; floor common 0.052, discrepancy correlated over 1 / 2 km 0.046 / 0.038 Tg) | `reproduce/multitangent_retrieval.py`, step `mtan` | engine | archive | FULL (5 min) | rerun here |
| A-FARAD75 `outputs/farir_adopted.json`, `_bg.npz` (Table 8 last column: band + far-IR under the adopted model 0.038 / 0.026 / 0.031 / 0.075 Tg at the calculated floors, x3 and x10 in brackets; far-IR only reproduces A-FARIR71 to 1e-6) | `reproduce/farir_adopted.py`, step `farad` | engine, A-FARIR floors (`farir_band_floors_w0p1.json`), `reproduce/materials_calibrated_thresholds.py` (far-IR element sets, backgrounds), D-OPT-MIN | archive, far-IR background cache | FULL (1 h of far-IR backgrounds, cached) | rerun here |
| A-DDA75 `outputs/dda_uniaxial.json`, `dda_uniaxial_convergence.json` (App. F4: discrete-dipole solution for the uniaxial spheres, 7 elements around each band peak, 9 radii, anchored to the exact Rayleigh average at 20 nm; the two scalar prescriptions tested against it) | `reproduce/dda_uniaxial.py`, step `dda` | D-OPT-MIN ray tables, `saimon.mie`, engine (threshold re-evaluation) | archives | FULL (30 min, 4 processes) | rerun here |
| Tables D2-D3 `tables/element_closure.tex` (four floats since round 75; state count 46 parameters, 15 source groups) | `reproduce/element_closure_table.py`, step `tabD2` | A-CLOSURE, A-UMODEL, A-FPC, A-CVAR, A-GCONV | table | FAST | rerun here |

Headline numbers are unchanged (0.12 / 0.17 Tg at 20 km); the multi-tangent inverse shows the single-tangent shell-equivalent convention to be conservative by a factor of about two for a point field of view.
