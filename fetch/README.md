# Obtaining the original data (full chain)

Everything in the paper regenerates from the shipped derived data in `data/`
(see `data/PROVENANCE.md`).  The scripts here let you rebuild those derived
files from the original sources, which are not redistributed:

| Source | Where | Then run |
|---|---|---|
| GloSSAC V2.23 (`GloSSAC_V2.23_NC4.nc`, 530 MB) | https://asdc.larc.nasa.gov/project/GloSSAC (free NASA Earthdata login) | `python fetch/reduce_glossac.py /path/to/GloSSAC_V2.23_NC4.nc` |
| ACE-FTS v5.2 Level-2 release incl. residual spectra (FRDR doi:10.20383/103.01245) | registration with the ACE team (https://ace.scisat.ca) | `python fetch/reduce_ace_v52.py /path/to/ace_v52` (expects `residual/<occ>/…` and `occultationlist.csv`); place `data_issues.csv` next to it to enable the pipeline's metadata audit |
| ACE Atmospheric Atlas (Hughes et al., 2014), tropical 16–20 km bin | https://ace.scisat.ca (public) | `reproduce/analyze_ace_atlas.py --rebuild` (needs the HITRAN cache) |
| HITRAN2020 line lists | fetched by HAPI from hitran.org (`pip install hitran-api`) | `python fetch/build_hitran_cache.py` (hours; fills `data/trace_gases/hitran_cache/` and `data/trace_gases/xsec_cache/`) |

Set `SAIMON_ROOT` to point the package at another copy of the repository
layout if you keep raw data elsewhere.
