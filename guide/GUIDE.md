# Guide

This guide explains what the repository computes, where each result of the
paper comes from, what the shipped data are, and how to verify a run.
Equations and derivations are in the paper (Appendices A–F); this guide
points to them rather than repeating them.

## 1. The chain in one paragraph

A solid aerosol layer (silica by default; calcite, dolomite, alumina in
Sect. 7) is placed in the stratosphere as a constant-mixing-ratio layer
between 16 and 24 km and its extinction spectrum is computed by Mie theory
over a lognormal size distribution from tabulated optical constants
(Appendix A). The natural sulfate background is a two-component (fine +
coarse) lognormal model whose 525-nm amplitude and profile follow GloSSAC at
20–25°N and whose partition and optical constants were fitted to the
ACE-FTS mid-infrared record (Appendix A6, E7). A solar-occultation
instrument sees both through a limb chord (Appendix C1). The detection
threshold is the Cramér–Rao bound on the injected mass after marginalizing
over the six background parameters (Appendix C4), with the visible/NIR
channel errors from the SAGE III/ISS product (Appendix B) and the
mid-infrared per-element floors measured on the ACE-FTS residual record
(Appendix E). The trace gases under the band are co-retrieved in an
optimal-estimation step whose budget is Appendix D. Sect. 4 runs the whole
measurement on real ACE-FTS spectra; Sect. 6 folds in a two-dimensional
silica field and orbit sampling.

## 2. Package map (`saimon/`)

| Module | Role | Paper |
|---|---|---|
| `atmosphere.py` | US Standard Atmosphere 1976; Rayleigh cross sections | App. A3 |
| `geometry.py` | spherical-shell chord matrix `tangent_to_slant_paths` | App. C1, Eq. C1 |
| `mie.py`, `psd.py`, `aerosol_mie.py` | Bohren–Huffman Mie series, lognormal PSDs, size-integrated extinction (cached) | App. A4, Eq. A7–A8 |
| `materials.py`, `refractive_index.py`, `density.py` | optical constants (silica: Kitamura + Popova, Franta; uniaxial crystals: ray mean; H2SO4: Lund Myhre / Biermann members spliced to Palmer–Williams with Lorentz–Lorenz) | App. A4–A5, Eq. A14 |
| `sai.py`, `sai_2d.py` | injected layers (`create_*_sai_layer`), the zonal 2-D field and orbit sampling | App. A4, Sect. 6 |
| `sulfate_background.py`, `backgrounds.py` | sulfate columns anchored to a 525-nm profile (`make_profile_column`) | App. A5–A6, Eq. A13 |
| `trace_gases.py` | HITRAN line-by-line cross sections resolution-matched to the element, MT_CKD continuum, CFC `.xsc`, MPI-Mainz UV/Vis, AFGL profiles | App. D |
| `o3_removal.py` | the optimal-estimation gas co-retrieval and its error budget | App. D2–D5 |
| `glossac.py`, `glossac_profiles.py` | GloSSAC access, tropical SAOD and latitude-bin profiles | App. A6, Fig. A2 |

## 3. Results map (`reproduce/`)

`python make_all.py --list` prints the table; in short:

| Paper | Script | Notes |
|---|---|---|
| Fig. 1 | `plot_slant_od_landscape_paper.py` | the gas curves come from the archived `outputs/slant_od_landscape_curves.npz` (HITRAN); rebuilt by the full chain |
| Fig. 2, 3 | `plot_refractive_index_paper.py`, `plot_spectral_shapes_paper.py` | |
| Sect. 2 numbers, Table C1 | `calibrated_background_problem.py` | K-channel scan, Franta comparison, Fisher inputs |
| Sect. 3 thresholds, Table 1 | `calibrated_background_thresholds.py` | window/band modes, brackets, floor and loading scans |
| Fig. 4, Sect. 4.1 | `analyze_ace_atlas.py`, `plot_ace_atlas_paper.py` | atlas bin vs gas model; element-averaged gas load |
| Fig. 5, Sect. 4.2 | `ace_mir_aerosol.py` | composites, anchor-constrained fits, matched GloSSAC anchors |
| Sect. 4.3 null tests, Ruang | `ace_fullspectrum_retrieval.py`, `ace_nulltest_convention.py`, `ace_ruang_bandshape.py` | |
| Table 2, Fig. F2 | `design_sensitivity_calibrated.py` | |
| Table F1, Fig. F1 | `resolution_sensitivity_calibrated.py`, `ace_nulltest_resolution.py` | |
| Table 3 | `materials_calibrated_thresholds.py` | |
| Sect. 6, Fig. 6 | `detectability_2d_calibrated.py` | `--o3-latitude outputs/o3_slant_latitude_measured.json` re-runs the map with the latitude-dependent floor (round 56; `results_o3lat.json`, `figures/detectability_2d_mass_contours_o3lat.png`) |
| Sect. 6 ozone load vs latitude | `o3_slant_latitude_measured.py` (measured, ACE v5.2 O3), `o3_slant_latitude_proxy.py` (AFGL cross-check) | slant τ_O3 ratios and floor factors per latitude band |
| Appendix A masses, Fig. A1–A2 | `reservoir_mass_saod.py`, `fit_segev_psd.py`, `reproduce_glossac_tropical_saod.py` | |
| Appendix D | `run_o3_removal_error.py`, `gas_removal_floor_880_widths.py`, `snr_integration_time_trade.py`, `mir_detector_etc.py`, `check_limb_emission_baseline.py`, `analyze_st_temperature_sensitivity.py`, `verify_cfc_contribution.py`, `analyze_hitran_o3_line_uncertainty.py`, `highres_gas_removal_budget.py` | |
| Appendix E, Table E1, Fig. E1 | `ace_floor/residual_floor_pipeline.py`, `ace_floor/residual_floor_paper1.py` | self-verifying archive (hashes in `manifest.json`) |

## 4. Conventions that matter

* The silica reference element is 8.80 µm (8.75–8.85 µm), the
  aerosol-extinction peak of the adopted optical constants; the design's
  retrieval band is 8–13 µm at 0.1 µm sampling (45 elements centred at
  8.00 … 13.20 µm, one of them on the resonance; O₃ core 9.3–10.0 µm
  excluded), its minimal subset the 7.8–9.3 µm window (15 elements centred
  at 7.80 … 9.20 µm). The floors on that grid are the archive
  `data/ace_floor/w0p1_c880/` (the Appendix E scan, `w0p1/`, is on the
  x.x5 grid and shares the 8.80 µm element bit for bit).
* Element optical depths are `-ln <T>` over the element; thresholds are
  per 0.5-km retrieved shell at a 20-km tangent; the ACE null tests are
  quoted in the slant convention and converted (Sect. 4.3).
* Floors: the per-element floors of the thresholds carry the design's own
  R~100 optimal-estimation budget at the 8.80 µm element (2.466e-3 OD,
  `outputs/round43_tracegas/gas_removal_floor_widths.csv`) with the
  wavelength dependence of the measured ACE residual SDs
  (`data/ace_floor/w0p1_c880/atlas.json`, 19–22 km; scale 1.327 on the
  measured 1.858e-3 OD at 8.80 µm), converted to a per-shell extinction with
  the exact gain of the edge-bounded onion peel, |G_k| = g / P_kk with
  g = 1.097 (`saimon/onion_peel.py`; the paper's Eq. D8). The single source
  of these constants is `saimon/onion_peel.py`; the vertical-correlation
  bracket of the floor (coherent database term vs independent noise) is
  `m2_vertical_correlation_floor.py`.

## 5. Data (see `data/PROVENANCE.md`)

Nothing shipped is a third-party database. The ACE subset is 143
occultations of the public v5.2 residual release (with the ACE team's
knowledge); GloSSAC is a 2-MB slice of the NASA file; the HITRAN-derived
cross-section grids are our computed products for the paper's bands.
`fetch/README.md` explains how to rebuild all of them from the originals.

## 6. Verifying a run

* `pytest -q` checks the package against numbers quoted in the paper and
  the shipped archives against the values quoted in Appendix E.
* `python reproduce/compare_reference.py` compares `outputs/` and
  `figures/` with `reference_outputs/` (the versions used in the paper):
  archived JSON numbers to the rounding of the text, figures by size.
* The residual-floor pipeline verifies every score table and hash of its
  own archive (`residual_floor_pipeline.py --finish-only`).

## 7. Known differences from the submitted manuscript

* Table F1 (element-width scan) in the paper was corrected in round 50 to
  the values this repository produces (differences ≤ 2 %): the manuscript's
  earlier run had used the 8.74 µm record for the averaging-width scaling.
* The ACE null test (Sect. 4.3, Table F1 ACE columns) selects, per
  occultation, the tangent height nearest to 20.5 km.  Five ensemble
  occultations have two heights exactly equidistant (19.8/21.2 km,
  20.3/20.7 km, ...) and the paper's runs up to round 50 had let the
  directory listing order decide; `ace_fullspectrum_retrieval.py` now breaks
  the tie deterministically (lower height).  The manuscript's null-test
  numbers were updated to this deterministic run in round 51 (window
  0.033 -> 0.032 Tg, element-averaged 0.040 -> 0.038 Tg, per shell
  0.29 -> 0.28 Tg, fixed-coarse 0.022 -> 0.021 Tg; the full-spectrum null,
  0.026 / 0.08 / 0.19 Tg, is unchanged).  The ACE columns of Table F1 had
  last been run in round 33 (before the round-36 floor model) and were re-run
  in round 51 as well (band at 0.1 um: 0.18 -> 0.15 Tg per shell).
* Rounds 53–54 (referee report RC1): the whole chain was re-run on the
  edge-grid exact gain (0.885 of the former 1.24/P_kk), the design-budget
  floor anchor (1.327 on the measured atlas) and the 8.80-centred grid.
  Headline thresholds 0.081 -> 0.102 Tg (band) and 0.115 -> 0.141 Tg
  (window); per-shell ACE conversion rho 2.46 -> 2.33; Sect. 6 best bin
  0.021 -> 0.026 Tg; materials 0.64/0.60/0.125 -> 0.78/0.75/0.153 Tg. New
  steps: `floorc`, `spike`, `onion`, `m2`, `far`, `o3lat`, `tab53`; new
  Table 2 rows (third sulfate mode free, second optics member free, database
  term coherent). The round-51 reference outputs are in the git history.
* Round 56 (referee report RC2, S1): the AFGL proxy for the latitude
  dependence of the ozone load is replaced by the ACE v5.2 Level-2 O3
  retrievals of the same occultations (`o3meas`; extract
  `data/ace/o3_l2_v52/`): slant ratios 0.92 / 0.94 / 0.50 at 20 km, floor
  factors 0.93 / 0.95 / 0.60, and the map with the latitude-dependent floor
  (`fig6o3`) archived beside the baseline (`results_o3lat.json`). The
  baseline map (standard climatology) remains the paper's reference.
* The reference archives in `reference_outputs/` are the versions this
  repository regenerates; `outputs/calibrated_background_thresholds.json`,
  the Fisher inputs, the materials thresholds, the design-sensitivity
  scans (Table 2, Appendix F), the 2D study, the null tests and the floor
  records reproduce the manuscript's numbers exactly.
