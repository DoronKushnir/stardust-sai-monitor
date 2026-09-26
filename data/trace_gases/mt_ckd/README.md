# MT_CKD H2O continuum coefficients (MT_CKD 4.3)

Water-vapor self- and foreign-broadened continuum coefficients from AER's
MT_CKD model, version 4.3.

## Source / provenance
- File: `data/absco-ref_wv-mt-ckd.nc` from
  https://github.com/AER-RC/MT_CKD_H2O (master branch).
- Reference: Mlawer et al. (2012), doi:10.1098/rsta.2011.0295.
- Converted from netCDF to CSV with python netCDF4 (no data altered).

## File
- `mt_ckd_h2o.csv`

## Columns
| column            | units                | description                                   |
|-------------------|----------------------|-----------------------------------------------|
| wavenumber_cm-1   | cm^-1                | spectral grid, 0..20000 cm^-1, 10 cm^-1 step (2003 pts; first point is -20) |
| self_coeff        | cm^2/molecule * cm-1 | self-continuum coefficient at 296 K (`self_absco_ref`)        |
| foreign_coeff     | cm^2/molecule * cm-1 | foreign-continuum coefficient at 296 K (`for_absco_ref`)      |
| for_closure_coeff | cm^2/molecule * cm-1 | alternative "closure" foreign continuum at 296 K (`for_closure_absco_ref`; see note) |
| self_texp         | dimensionless        | temperature exponent for the self continuum (`self_texp`)    |

Reference pressure = 1013 mb, reference temperature T0 = 296 K.
Spectral coverage 0-20000 cm^-1 fully includes the requested 0-3000 cm^-1
(8-30 um window). Spacing is 10 cm^-1.

## How to turn these into an absorption coefficient (cm^2/molecule)
The stored coefficients must be (a) scaled for density/temperature and
(b) multiplied by the "radiation term" (per the netCDF global attribute:
"All continuum coefficients need to be multiplied by the radiation term to
get an absorption coefficient in cm2/molecule").

Let nu = wavenumber [cm^-1], T = local temperature [K], T0 = 296 K,
P = local pressure, p_h2o = H2O partial pressure, p_frgn = P - p_h2o.

1. Radiation term (a dimensionless factor that carries the cm-1 dimension of
   the stored coefficient; it is the spectral line-shape normalization used by
   LBLRTM/MT_CKD):
       RADTERM(nu, T) = nu * tanh( h*c*nu / (2*k*T) )
   With c2 = h*c/k = 1.4387769 cm*K:
       RADTERM(nu, T) = nu * tanh( c2 * nu / (2*T) )

2. Self continuum: scale by H2O density (relative to the reference number
   density at T0, 1013 mb) and apply the temperature exponent:
       C_self(nu,T) = self_coeff(nu) * (T0/T)^self_texp(nu)
   The density scaling multiplies by the local H2O column amount; in the
   coefficient form, the per-molecule self contribution is:
       k_self = C_self(nu,T) * RADTERM(nu,T)
                 * (p_h2o/P0) * (T0/T)            [P0 = 1013 mb]

3. Foreign continuum (no MT_CKD T-dependence beyond density scaling):
       k_frgn = foreign_coeff(nu) * RADTERM(nu,T)
                 * (p_frgn/P0) * (T0/T)

4. Total H2O continuum absorption coefficient per H2O molecule:
       k_cont(nu,T) = k_self + k_frgn          [cm^2/molecule]
   Multiply by the H2O number density [cm^-3] and path [cm] for optical depth,
   then ADD to the local line-by-line H2O absorption (HITRAN lbl) for the
   total H2O absorption.

Notes:
- This is the standard CKD/MT_CKD recipe; the authoritative implementation is
  the FORTRAN driver in the AER repo (drive_mt_ckd_h2o.f90), which also does
  spectral interpolation from this 10 cm^-1 grid to the user's fine grid.
- `for_closure_coeff` is an alternative foreign continuum (v4.2+) that attains
  radiative closure with AERI 780-1250 cm^-1 measurements (presumed to absorb
  an aerosol effect). Use `foreign_coeff` for the standard MT_CKD continuum;
  use `for_closure_coeff` only if you specifically want the closure variant.
