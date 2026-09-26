# MPI-Mainz UV/VIS gas-phase absorption cross sections

Gas-phase absorption cross sections (cm^2/molecule vs wavelength in nm) from
the MPI-Mainz UV/VIS Spectral Atlas of Gaseous Molecules of Atmospheric
Interest. Atlas reference: Keller-Rudek, Moortgat, Schermann, Burrows,
Earth Syst. Sci. Data 5, 365 (2013), DOI 10.5194/essd-5-365-2013.
Atlas site: https://uv-vis-spectral-atlas-mainz.org

All CSVs have two columns and a multi-line `#` comment header documenting the
exact source file, dataset citation, temperature, and wavelength range:

| column            | units            |
|-------------------|------------------|
| wavelength_nm     | nm               |
| cross_section_cm2 | cm^2 / molecule  |

Data files were downloaded from the atlas data path
`https://uv-vis-spectral-atlas-mainz.org/uvvis_data/cross_sections/<Category>/<file>.txt`
(two whitespace-separated columns), then converted to CSV. A handful of small
negative cross-section values appear near each dataset's noise floor in
weakly-absorbing regions; these are present in the original atlas data and were
left unchanged.

## Files obtained

### O3 (ozone) — PRIORITY, obtained
- `o3_serdyuchenko_223K.csv`
  - Serdyuchenko et al. (2014), **223 K**, 2013 published version.
  - 88,668 points, **213-1100 nm**, peak ~1.16e-17 cm^2 (Hartley band).
  - Ref: Serdyuchenko, Gorshelev, Weber, Chehade, Burrows, AMT 7, 625 (2014).
  - (The atlas also offers 193-293 K in 10 K steps if other temperatures are
    needed; same source path, filename pattern
    `O3_Serdyuchenko(2014)_<T>K_213-1100nm(2013 version).txt`.)

### NO2 — obtained (both temperatures)
- `no2_vandaele_220K.csv` — Vandaele et al. (1998), **220 K**, 27,993 pts, **238-667 nm**.
- `no2_vandaele_294K.csv` — Vandaele et al. (1998), **294 K**, 27,993 pts, **238-667 nm**.
  - Ref: Vandaele et al., JQSRT 59, 171 (1998).
  - (Vandaele 1998 in this atlas covers 238-667 nm, not the full 250-1000 nm;
    667 nm is the long-wavelength end of the archived file.)

### O2 — obtained (UV continuum region)
- `o2_johnston_herzberg_298K.csv` — Johnston et al. (1984), **298 K**, analytic
  fit, 16 pts on a coarse 5-nm grid, **170-242 nm** (the **Herzberg continuum**).
  Ref: Johnston, Paige, Yao, JGR 89, 11661 (1984).
- `o2_bogumil_293K.csv` — Bogumil et al. (2003), **293 K** (SCIAMACHY), 1,374
  pts, **235-389 nm** (Herzberg-continuum tail / near-UV). Ref: Bogumil et al.,
  J. Photochem. Photobiol. A 157, 167 (2003).
  - NOTE on Schumann-Runge: the **Schumann-Runge continuum (~130-175 nm)** is
    NOT included. The atlas has it only as narrow fragmentary files (e.g.
    Lu 2010 115-180 nm, Gibson 1983 140-174 nm, Ogawa sets) rather than one
    clean broad table. Add one of those if deep-UV (<175 nm) is required;
    the Herzberg continuum (170-260 nm) is the dominant O2 UV absorption for
    most stratospheric work and is covered by the Johnston + Bogumil files.

## Molecule NOT obtained

### SO2 — MISSING (server-side gap at the atlas)
**No SO2 cross-section data file could be downloaded.** The atlas landing/
dataset pages list SO2 datasets (Bogumil 2003 293/223 K 239-395 nm,
Vandaele-Hermans-Fally 2009, Rufus 2003, Vandaele 1994, Manatt-Lane 1993,
etc.) as downloadable, but the entire `uvvis_data/cross_sections/Sulfur
compounds/` raw-data folder returns **HTTP 404** on every file and on all
three host names (`uv-vis-spectral-atlas-mainz.org`, `www.`,
`uvvis.mpch-mainz.gwdg.de`). This was confirmed with the identical
download recipe that succeeds for O3, NO2, and O2 on the same server, ruling
out an encoding error. It is a genuine gap in the served data, not a tooling
problem.
- To obtain SO2 UV cross sections, use an alternate source, e.g. the
  IUP-Bremen SCIAMACHY reference spectra (Bogumil et al. 2003) or the
  Vandaele-Hermans-Fally (2009) supplementary data, or contact the atlas
  maintainers about the missing sulfur folder.
