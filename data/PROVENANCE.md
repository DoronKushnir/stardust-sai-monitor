# Data provenance

Every file under `data/` is a *derived* product: an extract, subset,
reduction or digitization made for this study.  Each entry gives the
original source (which must be cited when the file is used), what was done
to it, and the script in `fetch/` that regenerates it from the original.
No original database is redistributed here; `fetch/README.md` explains how
to obtain the originals for the full (raw-data) chain.

| Directory | Contents | Source | Reduction | Regenerate with |
|---|---|---|---|---|
| `optics/` | Silica (Kitamura 2007 + Popova 1972; Franta 2016), alumina and dolomite (Querry 1985/1987) optical constants as `.yml`; calcite (Long 1993) and the three uniaxial-crystal ray tables as `.csv`; Palmer & Williams (1975) 75 wt% H2SO4 tables (`palmer_williams_75_*.csv`) | refractiveindex.info (CC0) and the cited papers, digitized by us | file copies / digitizations | — |
| `optics/h2so4_members/` | Lund Myhre et al. (2003) and Biermann et al. (2000) H2SO4/H2O optical-constant members (`myhre_h2so4/`, `biermann_h2so4/`) | HITRAN aerosol refractive-index collection (hitran.org), which redistributes the published tables | file copies of the members used | — |
| `digitized/` | Segev et al. (2026) dispersed-silica size distributions (`Segev2026_Fig2b.csv`, `Segev2026_CDF_best.xlsx`); Wrana et al. (2021) SAGE III/ISS PSD figures (`Wrana2021_Fig2a.csv`, `Wrana2021_Fig3.csv`); Lederer et al. (2026) coagulation mass fractions (`Lederer2026_Fig2_massfraction.csv`) | the cited papers (Segev and Lederer: this group's own work) | digitized / provided by the authors | — |
| `models/` | `Silica_05_no_coagulation_with2D.nc`: the steady-state zonal silica field of the two-dimensional transport model of Lederer et al. (2026) for the ±41°, 51 hPa injection | this group's transport model | model output (6 MB) | — |
| `glossac/` | `GloSSAC_V2.23_subset.nc`: GloSSAC V2.23 restricted to 525 and 1020 nm, ≤ 40 km, all latitudes and months, same variable names and conventions as the original | NASA GloSSAC V2.23 (Thomason et al., 2018), https://asdc.larc.nasa.gov/project/GloSSAC | subset (7 MB of 530 MB) | `fetch/reduce_glossac.py` |
| `trace_gases/` | AFGL standard gas profiles (`afgl/`: the US-standard and mid-latitude-summer files as `.csv` used by the trace-gas budget, and the six libRadtran `afgl*.dat` files used only by `o3_slant_latitude_proxy.py`), MPI-Mainz UV/Vis cross sections (`mpi_mainz/`), MT_CKD 4.x H2O continuum tables (`mt_ckd/`), HITRAN `.xsc` CFC bands (`cfc_xsec/`), ClONO2 pseudo-linelist (`pseudolines/`), HITRAN molecule metadata (`hitran_mw/`) | the cited public tables (as in the companion `stardust-slant-od` repository) | file copies | — |
| `trace_gases/xsec_cache/` | element-resolution Voigt cross-section grids (66 files, 45 MB) for the bands, gases and (T, P) of the trace-gas budget steps, computed from HITRAN2020 line lists (Gordon et al., 2022) | computed by `saimon.trace_gases` from HITRAN2020 via HAPI | our computation; the line lists themselves are not shipped, nor the whole-spectrum grids of two auxiliary checks | `fetch/build_hitran_cache.py` |
| `ace/v52_subset/` | ACE-FTS v5.2 public residual spectra of the 143 occultations at 20.1–25.0°N, June–August 2004–2024 (all tangent heights; `residual/<occ>/<occ>.<height>.gz`) and the matching rows of `occultationlist.csv` | ACE-FTS v5.2 Level-2 release (FRDR, doi:10.20383/103.01245; Bernath et al., 2005; Boone et al., 2020), registered access | subset (143 of 5460 occultations), bit-identical values, gzip | `fetch/reduce_ace_v52.py` |
| `ace/revision20260909/` | the 2-cm⁻¹ common-grid residual matrix of the 143 occultations at 19–22 km (`matrix_19_22.npz`), the two-mode ensemble basis (`basis.npz`) and the sulfate optics library (`optics.npz`) of the ACE side study | derived from `ace/v52_subset/` and the H2SO4 members | our analysis products | `reproduce/ace_floor/` (revision scripts) |
| `ace_floor/w0p1/`, `ace_floor/w0p25/` | the archived residual-floor analysis records (0.1-µm and 0.25-µm elements): per-element summaries, bootstrap intervals, score tables, common-grid matrices, manifest with hashes | derived from `ace/v52_subset/` | our analysis products; the bulky pair-bootstrap tables are omitted (the manifest lists them) | `reproduce/ace_floor/residual_floor_pipeline.py` |
| `ace_floor/w0p1_c880/` | the same residual-floor analysis on the design's element grid centred on 8.80 µm (0.1-µm elements at 7.30, 7.40, … 13.20 µm plus the material and control bands; `config.json` differs from `w0p1/config.json` only in `scan_centers_um`); the 8.80 µm element is bit-identical to `w0p1/` | derived from `ace/v52_subset/` | our analysis products; pair-bootstrap tables omitted as for `w0p1/` | `reproduce/ace_floor/residual_floor_pipeline.py` (step `floorc`) |
| `ace_floor/sample.csv` | the 143-occultation sample definition | — | — | — |

| `ace_atlas/` | the tropical 16–20 and 20–24 km bins of the public ACE-FTS Atmospheric Atlas cut to 950–1250 cm⁻¹ (`Tropics_0xx-0yy km.txt.gz`, headers kept) | Hughes, Bernath and Boone (2014), https://ace.scisat.ca (public) | spectral window extract (0.6 MB each of 40 MB) | `fetch/reduce_ace_atlas.py` |

The gas model against which the atlas bin is compared (Sect. 4.1) is
archived in `outputs/scisat_atlas/` (HITRAN-derived; rebuilt by the full
chain).
