# AFGL model-atmosphere constituent profiles

Standard AFGL (Air Force Geophysics Laboratory) atmospheric constituent
profiles, AFGL-TR-86-0110. Vertical grid: 50 levels from z = 120 km down to
0 km (top-of-atmosphere first).

## Source / provenance
- Files `afglus.dat` (US standard 1976) and `afglms.dat` (midlatitude summer)
  from the libRadtran distribution `data/atmmod/`, obtained from the public
  mirror https://github.com/ns-bak/libradtran (path `data/atmmod/`).
- The libRadtran main `.dat` carries 9 columns:
  z, p, T, air, O3, O2, H2O, CO2, NO2 (number densities in cm^-3).
- N2O, CO, CH4 are NOT in the main file in this libRadtran layout; they are
  shipped as volume-mixing-ratio sidecars `afglus_{n2o,co,ch4}_vmr.dat`
  (libRadtran source: http://www.atm.ox.ac.uk/RFM/atm/). These VMR profiles
  are atmosphere-independent in libRadtran. We converted them to number
  density via  n_gas = VMR * air_cm3, interpolating each VMR profile onto the
  AFGL altitude grid (linear in z). The same VMR sidecars were applied to BOTH
  afglus and afglms (no ms-specific VMR sidecars exist in libRadtran).

## Files
- `afglus.csv` — US Standard Atmosphere 1976
- `afglms.csv` — Midlatitude Summer

## Columns (both files)
| column   | units  | description                         |
|----------|--------|-------------------------------------|
| z_km     | km     | geometric altitude                  |
| p_mb     | mb=hPa | pressure                            |
| T_K      | K      | temperature                         |
| air_cm3  | cm^-3  | total air number density            |
| h2o_cm3  | cm^-3  | H2O number density                  |
| co2_cm3  | cm^-3  | CO2 number density                  |
| o3_cm3   | cm^-3  | O3 number density                   |
| n2o_cm3  | cm^-3  | N2O number density (VMR*air, see above) |
| co_cm3   | cm^-3  | CO number density (VMR*air)         |
| ch4_cm3  | cm^-3  | CH4 number density (VMR*air)        |
| o2_cm3   | cm^-3  | O2 number density                   |
| no2_cm3  | cm^-3  | NO2 number density                  |

To get a VMR for any gas: VMR = <gas>_cm3 / air_cm3.

## Gases NOT obtained as profiles  (and typical stratospheric values)
The libRadtran mirror used here ships no HNO3, CFC-11 (F11) or CFC-12 (F12)
profile files, so they are NOT included. Typical lower-stratospheric values
for use as constant or simple profiles:
- HNO3: peaks ~8-10 ppbv near 22-25 km; ~ few ppbv in lower stratosphere.
- CFC-11 (CCl3F): ~ 200-250 pptv in the lower stratosphere, decreasing
  rapidly with altitude above ~20 km (photolysis sink).
- CFC-12 (CCl2F2): ~ 500 pptv in the lower stratosphere, decreasing with
  altitude (more slowly than CFC-11).
(These are representative early-2000s/era values; scale to the desired epoch.)
