"""
calibrated_background_thresholds.py -- detection thresholds under the

ROUND 43 (2026-09-25, Doron): the silica REFERENCE ELEMENT moved from
8.74 um (8.69-8.79) to 8.80 um (8.75-8.85), the aerosol-extinction peak of
the adopted Kitamura+Popova optics (8.74 um was the Franta-2016 peak).
WL_NM[I874] = 8800 nm; the internal names (I874, SIG_874_ABS, sd874, the
0.25-um archive record 19_22_8.8um.json) keep their historical spelling.
The window/band element grids (7.85 ... 9.25 / 13.25 um) are unchanged,
so PARTS 2-4 are unchanged; PART 1 (single element) and the calibration
closure at the element are re-evaluated at 8.80 um.  Docstring numbers
below this note are the pre-round-43 (8.74 um) values.

ACE-CALIBRATED background model (round-24 agreement, Doron), against the
nominal single-mode background of solution_thresholds_fractional.py.

The calibrated background
-------------------------
* 525-nm amplitude and altitude profile: GloSSAC v2.23 at the 22.5N bin
  (20-25N, the ACE ensemble band -- closer to the 30-50 deg concentration
  of the steady-state combined-injection SAI field than the deep tropics,
  and the band where the microphysics is calibrated, so nothing is
  extrapolated), median profile over the quiet 2016.5-2018.5 epoch, ALL
  months (the ACE June-August sampling is a 4% detail there;
  scripts/glossac_saod_20_25N.py).
* Microphysics: the anchor-constrained two-component fit of the quiet
  epoch (outputs/ace_v52/anchor_constrained_fits.json; Paper 1 Appendix
  app:acefloor_anchor): Lund-Myhre 65 wt%/223 K member, fine mode 60 nm
  sigma 1.6 carrying 82.11% of the 525-nm extinction, coarse mode 1.5 um
  sigma 1.8 carrying 17.89%.
* Member optics spliced to the Palmer-Williams+Lorentz-Lorenz visible
  table below 2.05 um (the ACE side-study convention, revision_optics.py),
  so the visible/NIR channels stay on the established optics.

Microphysics brackets (all on the same quiet 20-25N profile, so the
differences isolate the microphysics; the background amplitude cancels in
the linearized threshold up to the absolute-floor and SNR-floor terms):
* post-HT calibration: Lund-Myhre 72 wt%/213 K, 100 nm s1.5 (95.61%) +
  1.5 um s1.8 (4.39%);
* optics swap: Biermann 70 wt%/215 K with the quiet modes/split;
* the nominal single-mode "Wrana" PSD (100 nm, s1.6, PW 75 wt%/215 K)
  on the same profile.

Threshold machinery identical to solution_thresholds_fractional.py
(fractional per-channel errors PHI * s_k on the aerosol channels, absolute
1.5e-8 m^-1 systematic floor at 8.74 um, optional SNR=2000 transmission
floor), generalized to arbitrary nuisance-column sets:
  A1+rs  : [A_total (fixed split), r_fine, sigma_fine]   (4-param fit,
           directly comparable to the paper's 4-parameter convention)
  A2     : [A_fine, A_coarse]                            (3-param fit)
  A2+rs  : [A_fine, A_coarse, r_fine, sigma_fine]        (5-param fit)
  A2+rs+rc: + r_coarse                                   (6-param fit)
PSD-derivative columns are evaluated at fixed mode 525-nm extinction
(the convention of the existing script: the 525 anchor is held).

Sanity checks: (i) reproduces the Sect. 3 nominal-background numbers with
the original machinery; (ii) the calibrated model's 8.74-um slant OD at a
20.5-km tangent is compared against the ACE anchor-constrained fit it was
calibrated to.

Results (2026-09-14 run), M_min(3sigma) [Tg]
--------------------------------------------
Sanity: OLD nominal on the elevated profile reproduces the Sect. 3 ladder
exactly (0.176 / 0.172 / 0.168 minimal/baseline/extended); calibration
closure at 20.5 km: slant OD525 0.2172 (matched anchor 0.2057, +6% from
the median-profile vs matched-median convention), 8.74-um element 0.0817
-- MIR-per-visible color within 3% of the ACE anchor-constrained fit.

PART 1 (broadband: one 8.74-um element), calibrated quiet model:
  A1+rs 4-param (fixed split): 0.242 / 0.195 / 0.191
  A2 amplitudes-only:          0.119 / 0.113 / 0.106
  A2+rs 5-param:                 --- / 4.11  / 3.65   <- collapse: with ONE
    MIR datum the coarse mode is a free MIR pedestal (round-25, Doron)
  naive SINGLE-mode 60 nm, 4-param: 5.3 / 3.5 / 2.5 (the fine-PSD runaway
    corner; the calibrated coarse component rescues broadband by ~x20)
  post-HT bracket 4-param: 0.208 / 0.193 / 0.190; optics swap (B70T215):
    within 1% of the calibrated rows.

PART 2 (R~100 band shape: 18 contiguous 0.25-um MIR elements, measured
per-element ACE floors, saturated O3 core excluded), calibrated quiet:
  A2+rs 5-param:      0.149 (vis/NIR+band)   0.267 (MIR band alone)
  full 7-param:       0.167                  0.349
  7-param + correlated MIR offset: 0.168     0.349
  (uniform 1.5e-8 floors: 0.127-0.140 with vis/NIR)
  post-HT bracket, full 7-param: 0.166 / 0.388.
So the band shape removes the nuisance-convention sensitivity entirely:
under the calibrated two-component background a FULL 7-parameter
background fit costs only ~0.17 Tg -- the broadband 5-param collapse is
an artifact of a single MIR element, and even a fully-correlated common
MIR floor mode adds nothing.  Caveat: per-element floors are treated as
independent between elements (each is a measured element SD); the
correlated-offset variant bounds the leading common mode, and the
empirical full-spectrum null (0.065 Tg, noise-limited, real data) bounds
reality from below the linearized figures.

PART 3 (the Sect.-3.3 window: 7.8-9.3 um at 0.1-um sampling, 15 elements;
atlas floors interpolated to the centers and inflated by the measured
narrow-width factor 2.33/2.04), calibrated quiet, with vis/NIR:
  A2+rs 5-param 0.101, full 7-param 0.121, +correlated offset 0.122
  (uniform 1.5e-8 floors: 0.077/0.096/0.097); post-HT bracket 0.117-0.120.
  Window alone: 0.25-0.32; the quiet 7-param window-alone case is
  genuinely degenerate (no visible anchor and no long-wave lever for the
  coarse mode) -- the vis/NIR channels are load-bearing here.
The Sect.-3.3 window at 0.1-um sampling slightly BEATS the full 8-13 um
band at 0.25-um elements (0.12 vs 0.17 Tg, full 7-param, measured
floors): denser sampling across the reststrahlen band where the floors
are lowest.  Caveat: at 0.1-um sampling the element-independence
assumption carries more weight (the width factor corrects the
per-element SD, not inter-element correlation; the correlated-offset
variant bounds the leading common mode at +0.001 Tg).

Round 33 (2026-09-23, Doron): PART 4 adds the whole 8-13 um band at the
SAME 0.1-um sampling (45 elements; the window is its minimal 15-element
subset), the design configuration after the element-width study of
scripts/resolution_sensitivity_calibrated.py.  Archive keys
'band 8-13 @0.1: ...', 'band01_elements_um', 'band01_floors_m1'.

Round 31 (2026-09-22, Doron): the quoted design roster becomes the Wrana
heritage triplet (448, 756, 1544 nm) + band/window; run_band_mode gains a
'triplet + ...' roster in PARTS 2 and 3 (the six-channel roster stays as
the footnoted variant).  Existing archive keys are unchanged.

Round 30 (2026-09-22): marginal() now uses the stable projection instead of
the cond(F) > 1e12 cutoff; the 7-param window-alone case, previously
reported "degenerate", is 0.345 Tg (near-singular among the background
parameters only).  All previously finite archive values are unchanged.
The slant-vs-shell convention gap between this analysis and the ACE null
test (factor rho = 2.46 in signal per Tg) is quantified in
scripts/ace_nulltest_convention.py.

Round 27 (2026-09-22, Doron's Appendix-E consistency question): PART 3 now
uses the DIRECTLY MEASURED 0.1-um averaging-width factor from the archived
8.74-um record (2.37/2.04 = 1.160) instead of the 0.04-um proxy
(2.33/2.04 = 1.142); all window-mode floors rise by 1.6% and the window
thresholds by ~1.4% (see the printed tables / JSON for the current values).
CAL_QUIET / CAL_POSTHT and TwoComponentBackground.column() are exported for
the demonstration scripts, which now linearize around this background.

Round 36 (2026-09-25, Doron): PARTS 3 and 4 now take the per-element floors
from the DIRECT 0.1-um-element re-analysis of the ACE record
(data/ace_floor/w0p1/atlas.json, 19-22 km)
instead of the 0.25-um atlas times the 8.74-um width factor.  Archive keys
'window_floors_m1' / 'band01_floors_m1' are therefore the direct floors;
'band01_floors_converted_m1' keeps the former values for comparison.

Run from the repo root:  python scripts/calibrated_background_thresholds.py
Output: printed tables + outputs/calibrated_background_thresholds.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

_THIS = Path(__file__).resolve()
_PARENT = _THIS.parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from scipy.interpolate import PchipInterpolator

from saimon.backgrounds import make_empirical_column, make_profile_column
from saimon.geometry import tangent_to_slant_paths
from saimon.glossac import GloSSACLoader
from saimon.materials import SilicaRefractiveIndex
from saimon.refractive_index import sulfuric_acid_at_temperature, TabulatedRI
from saimon.sai import create_silica_sai_layer

ALT = np.linspace(0, 60_000, 121)
iz = 40  # z = 20 km
RMED_SIL_NM, SIGMA_SIL = 268.0, 1.31
SPECTRAL_RES_UM = 0.10   # round 37 (Doron): the paper's single 0.1-um element convention;
                         # was 0.25 (the single-element legacy analyses now use 0.1 um;
                         # the legacy 0.25-um band run below keeps a literal 0.25)
SIG_874_ABS = 1.5e-8       # m^-1, systematic trace-gas-removal floor
ALPHA_FLOOR = 3.9e-9       # m^-1, SNR = 2000, dz = 0.5 km (app:validation)

WL_NM = np.array([448., 756., 869., 1021., 1250., 1544., 8800.])   # round 43: 8800 (was 8740)
PHI = np.array([0.040, 0.032, 0.040, 0.045, 0.063, 0.088, np.nan])
I874 = 6
ROSTERS = {
    'minimal  (Wrana3 + 8.74)':        [0, 1, 5, I874],
    'baseline (+1021, 1250)':          [0, 1, 3, 4, 5, I874],
    'extended (+869)':                 [0, 1, 2, 3, 4, 5, I874],
}

QUIET_YEARS = (2016.5, 2018.5)
HRI = _PARENT / "data/optics/h2so4_members"
# (label, wt%, T[K], file, column) -- the members of the ACE side-study
# library used here (revision_optics.py MEMBERS convention)
MEMBERS = {
    "LM65T223": (65, 223, "myhre_h2so4/myhreh2so4set4.dat", 3),
    "LM72T213": (72, 213, "myhre_h2so4/myhreh2so4set3.dat", 3),
    "B70T215":  (70, 215, "biermann_h2so4/h2so4T215.biermann", 0),
}

_SUB = np.linspace(-SPECTRAL_RES_UM / 2, SPECTRAL_RES_UM / 2, 41)

# The anchor-constrained two-component calibrations (Paper 1 Appendix
# app:acefloor_anchor; outputs/ace_v52/anchor_constrained_fits.json), as
# (rmed_nm, sigma, fraction of the 525-nm extinction).  Module-level since
# round 27 so that the demonstration scripts (null test, full-spectrum
# retrieval, Ruang, atlas and Fig.-4 plots) linearize around the SAME
# background bit-identically.
CAL_QUIET = [(60.0, 1.6, 0.8211), (1500.0, 1.8, 0.1789)]
CAL_POSTHT = [(100.0, 1.5, 0.9561), (1500.0, 1.8, 0.0439)]

# PART-3 width factor for the 0.1-um window elements.  Rounds 25b-26 used
# the 0.04-um averaging result of the Appendix-app:acefloor record
# (2.33/2.04) as a proxy; the same archived record also holds the
# DIRECTLY MEASURED 0.1-um averaging-width SD at 8.74 um (19-22 km, full
# 0.25-um exclusion held fixed), which round 27 (Doron's App.-E
# consistency question) adopts instead: 2.37/2.04 (+1.6%).
_FLOOR_874 = (_PARENT / "data/ace_floor/w0p25/"
              "19_22_8.8um.json")            # round 43: the 8.80-um record
if not _FLOOR_874.exists():                     # pipeline re-run still pending: fall back to the
    _FLOOR_874 = (_PARENT / "data/ace_floor/w0p25_874/"   # archived 8.74-um record
                  "19_22_8.74um.json")
    print(f"WARNING: 8.80-um 0.25-um record not found; width factor from {_FLOOR_874}")


def width_factor_0p1um():
    """(factor, sd at 0.1-um width, sd at the 0.25-um element) [OD]."""
    rec = json.load(open(_FLOOR_874))
    sd_full = float(rec["sd"])
    sd_0p1 = float(rec["averaging_fixed_full_element_holdout"]["0.1"]
                   ["exact"]["sd"])
    return sd_0p1 / sd_full, sd_0p1, sd_full


def member_ri(name):
    """Laboratory H2SO4 member optics spliced with the PW+LL visible table
    below 2.05 um -- ported from the ACE side-study (revision_optics.ri_for)
    so the calibrated background uses bit-equivalent member optics."""
    wt, T, f, col = MEMBERS[name]
    rows = []
    for line in (HRI / f).read_text().splitlines():
        try:
            v = [float(x) for x in line.split()]
        except ValueError:
            continue
        if len(v) >= 3 + col:
            rows.append([v[0], v[2 + col]])
    rows = np.array(rows)
    split = np.argmax(np.abs(np.diff(rows[:, 0]))) + 1

    def order(block):
        w = 1e7 / np.maximum(block[:, 0], 1e-3)
        o = np.argsort(w)
        return w[o], block[o, 1]

    wr, n = order(rows[:split])
    wi, k = order(rows[split:])
    wl = np.unique(np.r_[wr, wi])
    wl = wl[(wl >= 2050) & (wl <= 26000)]
    nn = np.interp(wl, wr, n)
    kk = np.clip(np.interp(wl, wi, k, left=0, right=np.nan), 0, None)
    ok = np.isfinite(kk)
    vis = np.linspace(300, 2040, 200)
    vv = sulfuric_acid_at_temperature(215)(vis * 1e-9)
    return TabulatedRI(np.r_[vis, wl[ok]], np.r_[vv.real, nn[ok]],
                       np.r_[vv.imag, kk[ok]], T), wt, T


def glossac_2025N_quiet_profile():
    """Median GloSSAC 525-nm extinction profile [1/m on ALT] at the 22.5N
    bin over the quiet 2016.5-2018.5 epoch, all months; pchip onto ALT,
    zero below 16 km and above the 40-km GloSSAC ceiling."""
    d = GloSSACLoader(str(_PARENT / "data/glossac/GloSSAC_V2.23_subset.nc")
                      ).load_raw_data(525)
    years = (d["time"] // 100) + (d["time"] % 100 - 0.5) / 12.0
    li = int(np.argmin(np.abs(d["lat"] - 22.5)))
    m = (years >= QUIET_YEARS[0]) & (years <= QUIET_YEARS[1])
    prof_km = np.ma.median(np.ma.masked_invalid(d["ext"][m, li, :]), axis=0)
    alt_km, prof_km = d["alt"], np.ma.filled(prof_km, 0.0)     # 1/km
    interp = PchipInterpolator(alt_km, prof_km, extrapolate=False)
    alpha = interp(ALT / 1000.0)
    alpha = np.where(np.isfinite(alpha), alpha, 0.0) / 1000.0  # 1/m
    alpha = np.maximum(alpha, 0.0)
    alpha[ALT < 16_000.0] = 0.0
    return alpha


def banded_ext(layer):
    """Per-channel extinction at z = 20 km; 8.74 um boxcar-averaged."""
    out = layer.extinction_profile_m1(WL_NM * 1e-9)[iz, :].copy()
    out[I874] = layer.extinction_profile_m1(
        (WL_NM[I874] / 1000.0 + _SUB) * 1e-6)[iz, :].mean()
    return out


class TwoComponentBackground:
    """Calibrated two-component background: each mode is a
    make_profile_column carrying a prescribed fraction of the 525-nm
    extinction profile.  Provides per-channel component extinctions and the
    fixed-525 PSD-derivative columns."""

    def __init__(self, alpha525, comps, ri, wt, T, fd=0.05):
        # comps: list of (rmed_nm, sigma, fraction)
        self.alpha525, self.comps, self.ri = alpha525, comps, ri
        self.wt, self.T, self.fd = wt, T, fd

    def _col(self, rmed, sg, frac):
        return make_profile_column(ALT, frac * self.alpha525, rmed, sg,
                                   weight_percent_h2so4=self.wt,
                                   temperature_k=self.T,
                                   refractive_index=self.ri)

    def column(self, i, dr=0.0, ds=0.0):
        """SulfateAerosolColumn of component i with its PSD perturbed by
        (dr [nm], ds) at fixed 525-nm extinction share (round 27: shared
        with the demonstration scripts)."""
        r0, s0, f = self.comps[i]
        return self._col(r0 + dr, s0 + ds, f)

    def columns(self):
        return [self._col(r, s, f) for r, s, f in self.comps]

    def component_ext(self):
        return [banded_ext(self._col(r, s, f)) for r, s, f in self.comps]

    def psd_derivs(self, i):
        """(d ext/d rmed, d ext/d sigma) of component i at fixed 525-nm
        extinction share (finite differences, +-fd fractional steps)."""
        r0, s0, f = self.comps[i]
        dr, ds = self.fd * r0, self.fd * s0
        er = (banded_ext(self._col(r0 + dr, s0, f))
              - banded_ext(self._col(r0 - dr, s0, f))) / (2 * dr)
        es = (banded_ext(self._col(r0, s0 + ds, f))
              - banded_ext(self._col(r0, s0 - ds, f))) / (2 * ds)
        return er, es

    def slant_od(self, wl_m, htan_m=20_500.0):
        """Slant OD spectrum of the full model at a tangent height."""
        chord = tangent_to_slant_paths(np.array([htan_m]), ALT)[0]
        tot = np.zeros(len(wl_m))
        for r, s, f in self.comps:
            a = self._col(r, s, f).extinction_profile_m1(np.asarray(wl_m))
            tot += np.array([float(chord @ a[:, j])
                             for j in range(len(wl_m))])
        return tot


def marginal(idx, nuis_cols, sig, t):
    """Marginalized sigma(A_sil) [Tg], ||t_perp||, R^2 for channels idx and
    the given list of nuisance columns.

    The Fisher matrix is formed on unit-normalized whitened columns so the
    conditioning test is meaningful across mixed parameter units (amplitudes
    vs per-nm PSD derivatives); sigma(A_sil) is scale-invariant and is
    recovered by undoing the silica-column normalization."""
    idx = np.asarray(idx)
    N = np.column_stack([c[idx] for c in nuis_cols])
    if len(idx) < N.shape[1] + 1:
        return np.inf, 0.0, np.nan
    J = np.column_stack([N, t[idx]]) / sig[idx][:, None]   # whitened
    scales = np.linalg.norm(J, axis=0)
    if np.any(scales == 0):
        return np.inf, 0.0, np.nan
    Jn = J / scales
    # Round 30 (Doron's Sect.-3/Sect.-4 consistency question): the
    # marginalized sigma is computed by the numerically stable projection
    # of the whitened silica column onto the complement of the nuisance
    # span (sigma_A = 1/||t_perp||, identical to sqrt([F^-1]_AA) when F
    # is invertible) instead of inverting F behind a cond(F) > 1e12
    # cutoff.  The cutoff had reported "degenerate" for cases whose
    # near-singularity lies entirely among the background parameters
    # (e.g. the 7-parameter window-alone fit), where the silica amplitude
    # itself is finite; only an exact null direction (t_perp = 0 to
    # working precision) or K < N is reported as degenerate.
    t_hat = J[:, -1]
    Nw = Jn[:, :-1]
    t_par = Nw @ np.linalg.lstsq(Nw, t_hat, rcond=None)[0]
    t_perp = t_hat - t_par
    R2 = float(t_par @ t_par) / float(t_hat @ t_hat)
    if np.linalg.norm(t_perp) <= 1e-8 * np.linalg.norm(t_hat):
        return np.inf, 0.0, np.nan
    sig_a = 1.0 / float(np.linalg.norm(t_perp))
    return sig_a, 1.0 / sig_a, R2


def noise(s_total, floor=0.0):
    sig = np.maximum(PHI * s_total, floor)
    sig[I874] = SIG_874_ABS
    return sig


def main():
    ri_pw = sulfuric_acid_at_temperature(215.0)
    silica_ri = SilicaRefractiveIndex("data/optics/SiO2_KitamuraPopova.yml")
    sai = create_silica_sai_layer(ALT, 1.0, refractive_index=silica_ri,
                                  rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL)
    t_sil = banded_ext(sai)

    alpha_cal = glossac_2025N_quiet_profile()
    saod_cal = float(np.trapezoid(alpha_cal, ALT))
    print(f"calibrated 20-25N quiet profile: SAOD525 = {saod_cal:.4f}, "
          f"alpha(20 km) = {alpha_cal[iz]:.3e} m^-1")

    # ---- background models -------------------------------------------------
    # single-mode references (old machinery, via make_empirical_column /
    # make_profile_column with PW 75wt%/215K optics)
    def single_mode(profile, r0, s0):
        def ext(rr, ss):
            return banded_ext(make_profile_column(
                ALT, profile, rr, ss, refractive_index=ri_pw))
        s = ext(r0, s0)
        dr, ds = 0.05 * r0, 0.05 * s0
        jr = (ext(r0 + dr, s0) - ext(r0 - dr, s0)) / (2 * dr)
        js = (ext(r0, s0 + ds) - ext(r0, s0 - ds)) / (2 * ds)
        return s, jr, js

    # elevated preset profile (for the sanity reproduction)
    elev_col = make_empirical_column(ALT, 'elevated', 100.0, 1.6,
                                     refractive_index=ri_pw)
    alpha_elev = elev_col.extinction_profile_m1(
        np.array([525e-9]))[:, 0]

    ri_lm65, wt65, T65 = member_ri("LM65T223")
    ri_lm72, wt72, T72 = member_ri("LM72T213")
    ri_b70, wt70, T70 = member_ri("B70T215")

    # CAL_QUIET / CAL_POSTHT are module-level (round 27)
    models = {}
    # 1. the paper's current fiducial (sanity: minimal roster = 0.176 Tg)
    models['OLD nominal on elevated profile'] = ('single', alpha_elev,
                                                 (100.0, 1.6))
    # 2. same PSD/optics on the calibrated profile (profile isolation)
    models['OLD nominal on 20-25N quiet profile'] = ('single', alpha_cal,
                                                     (100.0, 1.6))
    # 2b. naive single-mode reading of the calibrated fine mode: what the
    # old machinery would do with r_med = 60 nm and no coarse component
    models['single-mode 60 nm on 20-25N profile'] = ('single', alpha_cal,
                                                     (60.0, 1.6))
    # 3. the calibrated model (LEADING)
    models['CALIBRATED quiet (LM65T223, 2-comp)'] = ('two', alpha_cal,
                                                     CAL_QUIET, ri_lm65,
                                                     wt65, T65)
    # 4. microphysics bracket: post-HT calibration on the same profile
    models['bracket: post-HT calib (LM72T213)'] = ('two', alpha_cal,
                                                   CAL_POSTHT, ri_lm72,
                                                   wt72, T72)
    # 5. optics swap bracket
    models['bracket: optics swap (B70T215)'] = ('two', alpha_cal,
                                                CAL_QUIET, ri_b70,
                                                wt70, T70)

    arch = {"saod_cal_20_25N_quiet": saod_cal, "results": {}}
    for label, spec in models.items():
        print(f"\n=== {label} ===")
        res = {}
        if spec[0] == 'single':
            _, profile, (r0, s0) = spec
            s, jr, js = single_mode(profile, r0, s0)
            nsets = {'A1+rs (4-param)': [s, jr, js]}
            s_total = s
        else:
            _, profile, comps, ri_m, wt, T = spec
            bg = TwoComponentBackground(profile, comps, ri_m, wt, T)
            exts = bg.component_ext()
            s_total = np.sum(exts, axis=0)
            jr_f, js_f = bg.psd_derivs(0)
            jr_c, _ = bg.psd_derivs(1)
            nsets = {
                'A1+rs (4-param)':   [s_total, jr_f, js_f],
                'A2 (3-param)':      [exts[0], exts[1]],
                'A2+rs (5-param)':   [exts[0], exts[1], jr_f, js_f],
                'A2+rs+rc (6-param)': [exts[0], exts[1], jr_f, js_f, jr_c],
            }
            # calibration closure: slant OD at the 8.74 um element, 20.5 km
            wl_el = (WL_NM[I874] / 1000.0 + _SUB) * 1e-6
            tau874 = float(np.mean(bg.slant_od(wl_el)))
            tau525 = float(bg.slant_od(np.array([525e-9]))[0])
            print(f"  slant OD at 20.5 km: 525 nm {tau525:.4f}, {WL_NM[I874]/1000:.2f} um "
                  f"element {tau874:.4f} (ACE anchor-constrained fit: "
                  f"sulfate element ~0.070-0.075 at the 0.206 anchor)")
            res['slant_od_20p5km'] = dict(od525=tau525, od874=tau874)
        for fl_lab, fl in [("fractional", 0.0),
                           ("with SNR=2000 floor", ALPHA_FLOOR)]:
            sig = noise(s_total, fl)
            print(f"  -- {fl_lab} --")
            for ns_lab, cols in nsets.items():
                row = {}
                for r_lab, idx in ROSTERS.items():
                    sa, tperp, R2 = marginal(idx, cols, sig, t_sil)
                    row[r_lab] = dict(mmin_tg=3 * sa, one_minus_R2=1 - R2
                                      if np.isfinite(R2) else None)
                    tag = (f"{3 * sa:8.3f}" if np.isfinite(sa)
                           else "     ---")
                    print(f"    {ns_lab:22s} {r_lab:28s} M_min ={tag} Tg"
                          + (f"  (1-R2 = {1 - R2:.2e})"
                             if np.isfinite(R2) else ""))
                res.setdefault(fl_lab, {})[ns_lab] = row
        arch["results"][label] = res

    # ---- PART 2: the R~100 band-shape mode --------------------------------
    # Round-25 (Doron): the 5-parameter collapse above is an artifact of
    # giving the MIR exactly ONE boxcar element -- the coarse mode is then a
    # free MIR pedestal.  The design flies an R~100 spectrograph over
    # 8-13 um, so the honest representation is the full set of contiguous
    # 0.25-um elements, each with its own MEASURED ACE floor (Appendix
    # app:acefloor atlas, 19-22 km), scaled to extinction units by the
    # paper's 2.04e-3 OD <-> 1.5e-8 m^-1 conversion at 8.74 um.  Elements in
    # the saturated O3 core (9.4-9.9 um) have no measured floor and are
    # excluded, exactly as the trace-gas analysis excludes them.  A
    # fully-correlated MIR offset nuisance (flat extinction error across all
    # MIR elements) is available as the conservative check on inter-element
    # floor correlation, which the per-element treatment otherwise ignores.
    print("\n" + "=" * 72)
    print("PART 2: R~100 band-shape mode (contiguous 0.25-um MIR elements, "
          "measured floors)")
    atlas = json.load(open(_PARENT / "data/ace_floor/"
                           "w0p25/atlas.json"))
    # regular (non-overlapping) element grid only: centers x.12/x.38/x.62/
    # x.88 within the 8-13.25 um design band; drop the inserted material
    # centers (8.74, 11.30, 11.40, 12.90), which overlap their neighbors
    mir_c, mir_sd = [], []
    for r in atlas["19_22"]:
        c = r["center_um"]
        if 8.0 < c < 13.25 and r["sd"] is not None:
            frac = round((c * 100) % 25)
            if frac in (12, 13):  # .12/.38/.62/.88 grid (x100 mod 25 = 12/13)
                mir_c.append(c)
                mir_sd.append(r["sd"])
    mir_c, mir_sd = np.array(mir_c), np.array(mir_sd)
    sd874 = 0.00204
    sig_mir = SIG_874_ABS * mir_sd / sd874
    print(f"MIR elements: {len(mir_c)} ({mir_c.min():.2f}-{mir_c.max():.2f} "
          f"um, gap at the saturated O3 core); floors "
          f"{sig_mir.min():.2e}-{sig_mir.max():.2e} m^-1")

    band_models = {
        'CALIBRATED quiet (LM65T223, 2-comp)': (CAL_QUIET, ri_lm65,
                                                wt65, T65),
        'bracket: post-HT calib (LM72T213)':   (CAL_POSTHT, ri_lm72,
                                                wt72, T72),
        # round 27: the member-optics swap in the band/window modes too
        # (the paper quotes it for the window threshold)
        'bracket: optics swap (B70T215)':      (CAL_QUIET, ri_b70,
                                                wt70, T70),
    }

    def run_band_mode(tag, centers_um, width_um, sig_el, roster_labels):
        """Threshold analysis with a set of MIR spectral elements.

        centers_um/width_um define the boxcar elements; sig_el their
        absolute extinction floors [m^-1].  Rosters: vis/NIR + band, and
        the band alone."""
        n_vis = I874
        n_el = len(centers_um)
        sub = np.linspace(-width_um / 2, width_um / 2, 41)
        wl_all = np.r_[WL_NM[:n_vis], np.asarray(centers_um) * 1000.0]
        is_mir = np.arange(len(wl_all)) >= n_vis

        def bext(layer):
            out = layer.extinction_profile_m1(wl_all * 1e-9)[iz, :].copy()
            for j, c in enumerate(centers_um):
                out[n_vis + j] = layer.extinction_profile_m1(
                    (c + sub) * 1e-6)[iz, :].mean()
            return out

        class BandBackground(TwoComponentBackground):
            def component_ext(self):
                return [bext(self._col(r, s, f)) for r, s, f in self.comps]

            def psd_derivs(self, i):
                r0, s0, f = self.comps[i]
                dr, ds = self.fd * r0, self.fd * s0
                er = (bext(self._col(r0 + dr, s0, f))
                      - bext(self._col(r0 - dr, s0, f))) / (2 * dr)
                es = (bext(self._col(r0, s0 + ds, f))
                      - bext(self._col(r0, s0 - ds, f))) / (2 * ds)
                return er, es

        t_sil_band = bext(sai)
        offset_mir = is_mir.astype(float)   # fully-correlated floor mode
        rosters = {
            roster_labels[0]: np.arange(len(wl_all)),
            roster_labels[1]: np.arange(n_vis, len(wl_all)),
            # round 31 (Doron): the quoted design roster is the Wrana
            # heritage triplet (448, 756, 1544 nm); the six-channel roster
            # is kept as the footnoted variant
            roster_labels[2]: np.r_[[0, 1, 5], np.arange(n_vis, len(wl_all))],
        }
        for label, (comps, ri_m, wt, T) in band_models.items():
            print(f"\n=== {label}, {tag} ===")
            bg = BandBackground(alpha_cal, comps, ri_m, wt, T)
            exts = bg.component_ext()
            s_total = np.sum(exts, axis=0)
            jr_f, js_f = bg.psd_derivs(0)
            jr_c, js_c = bg.psd_derivs(1)
            nsets = {
                'A2+rs (5-param)':    [exts[0], exts[1], jr_f, js_f],
                'full 7-param':       [exts[0], exts[1], jr_f, js_f, jr_c,
                                       js_c],
                '7-param + MIR offs': [exts[0], exts[1], jr_f, js_f, jr_c,
                                       js_c, offset_mir],
            }
            res = arch["results"].setdefault(label, {})
            for fl_lab, mir_sigma in [("measured floors", sig_el),
                                      ("uniform 1.5e-8 floors",
                                       np.full(n_el, SIG_874_ABS))]:
                sig = np.r_[np.maximum(PHI[:n_vis] * s_total[:n_vis],
                                       ALPHA_FLOOR), mir_sigma]
                print(f"  -- {fl_lab} (vis/NIR with SNR floor) --")
                for ns_lab, colsn in nsets.items():
                    row = {}
                    for r_lab, idx in rosters.items():
                        sa, tperp, R2 = marginal(idx, colsn, sig,
                                                 t_sil_band)
                        row[r_lab] = dict(mmin_tg=3 * sa,
                                          one_minus_R2=1 - R2
                                          if np.isfinite(R2) else None)
                        tag2 = (f"{3 * sa:8.3f}" if np.isfinite(sa)
                                else "     ---")
                        print(f"    {ns_lab:22s} {r_lab:22s} "
                              f"M_min ={tag2} Tg"
                              + (f"  (1-R2 = {1 - R2:.2e})"
                                 if np.isfinite(R2) else ""))
                    res.setdefault(f"{tag}: {fl_lab}", {})[ns_lab] = row

    run_band_mode("band 8-13 um @0.25", mir_c, 0.25, sig_mir,   # legacy comparison, literal 0.25
                  ('vis/NIR 6 + MIR band', 'MIR band alone',
                   'triplet + MIR band'))

    # ---- PART 3: the Sect. 3.3 window: 7.8-9.3 um at 0.1-um sampling ------
    # Round-25b (Doron): the validated O3+N2O joint-retrieval window of
    # Sect. 3.3, sampled at 0.1 um (15 elements, 7.85-9.25).  Floors:
    # measured atlas SDs interpolated to the element centers (available
    # 7.88-9.12 um; flat-extended at the edges), inflated by the measured
    # narrow-width factor 2.33/2.04 (Appendix app:acefloor: the element SD
    # rises from 2.04e-3 to 2.33e-3 when the averaging width narrows from
    # 0.25 to 0.04 um -- spectrally correlated errors do not average down),
    # and scaled to extinction by the same 2.04e-3 <-> 1.5e-8 conversion.
    print("\n" + "=" * 72)
    print("PART 3: Sect.-3.3 window mode (7.8-9.3 um at 0.1-um sampling)")
    win_c = np.round(np.arange(7.85, 9.2501, 0.10), 2)
    atl_c, atl_sd = [], []
    for r in atlas["19_22"]:
        if r.get("sd") is not None and 7.5 < r["center_um"] < 9.3:
            atl_c.append(r["center_um"])
            atl_sd.append(r["sd"])
    o = np.argsort(atl_c)
    atl_c, atl_sd = np.array(atl_c)[o], np.array(atl_sd)[o]
    # round 27: the directly measured 0.1-um width factor (see the
    # module-level note); rounds 25b-26 used the 0.04-um proxy 2.33/2.04
    WIDTH_FACTOR, sd_0p1, sd_full = width_factor_0p1um()
    # Round 36 (2026-09-25, Doron): the floors are the DIRECT 0.1-um-element
    # re-analysis of the ACE record (data/residual_floor_w0p1, every element
    # with its own exclusion/training/bootstrap) at the element centers, no
    # width factor; the 0.25-um atlas x width factor is kept only for the
    # printed comparison.  OD -> extinction by the same 2.04e-3 <-> 1.5e-8
    # conversion (onion-peel geometry at dz = 0.5 km).
    atlas01 = json.load(open(_PARENT / "data/ace_floor/"
                             "w0p1/atlas.json"))
    sd01 = {round(r["center_um"], 3): r["sd"] for r in atlas01["19_22"]
            if r.get("sd") is not None}
    sd_win_conv = np.interp(win_c, atl_c, atl_sd) * WIDTH_FACTOR
    sd_win = np.array([sd01[round(c, 3)] for c in win_c])
    sig_win = SIG_874_ABS * sd_win / sd874
    print(f"window floors: direct 0.1-um SDs {sd_win.min():.5f}-{sd_win.max():.5f} OD "
          f"(converted would be {sd_win_conv.min():.5f}-{sd_win_conv.max():.5f})")
    print(f"window elements: {len(win_c)} ({win_c.min():.2f}-"
          f"{win_c.max():.2f} um); floors {sig_win.min():.2e}-"
          f"{sig_win.max():.2e} m^-1 (atlas x{WIDTH_FACTOR:.3f} width "
          f"factor = measured 0.1-um SD {sd_0p1:.5f} / 0.25-um SD "
          f"{sd_full:.5f} OD at 8.74 um)")
    arch["width_factor_0p1um"] = dict(factor=WIDTH_FACTOR, sd_0p1um=sd_0p1,
                                      sd_0p25um=sd_full)
    arch["floor_source"] = ("direct 0.1-um-element ACE re-analysis "
                            "(data/ace_floor/w0p1/atlas.json, 19-22 km)")
    arch["sd874_0p1um_direct"] = sd01[8.8] if 8.8 in sd01 else sd01[round(8.75, 3)]   # round 43: the 8.80-um element (8.75 if the re-run is pending)
    arch["reference_element_um"] = WL_NM[I874] / 1000.0
    run_band_mode("window 7.8-9.3 @0.1", win_c, 0.10, sig_win,
                  ('vis/NIR 6 + window', 'window alone', 'triplet + window'))

    # ---- PART 4: the whole 8-13 um band at the SAME 0.1-um sampling -------
    # Round 33 (Doron, after the element-width study of
    # scripts/resolution_sensitivity_calibrated.py): one resolution, 0.1 um,
    # for the whole band; the 7.8-9.3 um window is its minimal 15-element
    # subset.  Elements 8.05-13.25 um, those centered in the saturated O3
    # core (9.3-10.0 um) dropped; floors = atlas SD interpolated to the
    # element centers (measured 7.875-13.125 um; flat-extended) x the same
    # measured 0.1-um width factor.
    print("\n" + "=" * 72)
    print("PART 4: band 8-13 um at 0.1-um sampling (one resolution for all)")
    b01_c = np.round(np.arange(8.05, 13.2501, 0.10), 2)
    b01_c = b01_c[(b01_c < 9.3) | (b01_c > 10.0)]
    all_c, all_sd = [], []
    for r in atlas["19_22"]:
        if r.get("sd") is not None and 7.8 <= r["center_um"] <= 13.25:
            all_c.append(r["center_um"]); all_sd.append(r["sd"])
    o = np.argsort(all_c)
    all_c, all_sd = np.array(all_c)[o], np.array(all_sd)[o]
    sd_b01_conv = np.interp(b01_c, all_c, all_sd) * WIDTH_FACTOR
    sd_b01 = np.array([sd01[round(c, 3)] for c in b01_c])      # round 36: direct 0.1-um floors
    sig_b01 = SIG_874_ABS * sd_b01 / sd874
    print(f"band@0.1 floors: direct 0.1-um SDs {sd_b01.min():.5f}-{sd_b01.max():.5f} OD "
          f"(converted would be {sd_b01_conv.min():.5f}-{sd_b01_conv.max():.5f}); "
          f"median direct/converted {np.median(sd_b01 / sd_b01_conv):.3f}")
    arch["band01_floors_converted_m1"] = list(np.round(SIG_874_ABS * sd_b01_conv / sd874, 12))
    print(f"band@0.1 elements: {len(b01_c)} ({b01_c.min():.2f}-{b01_c.max():.2f} "
          f"um, gap {9.3}-{10.0}); floors {sig_b01.min():.2e}-{sig_b01.max():.2e} m^-1")
    run_band_mode("band 8-13 @0.1", b01_c, 0.10, sig_b01,
                  ('vis/NIR 6 + band01', 'band01 alone', 'triplet + band01'))
    arch["band01_elements_um"] = list(np.round(b01_c, 2))
    arch["band01_floors_m1"] = list(np.round(sig_b01, 12))

    arch["mir_elements_um"] = list(np.round(mir_c, 2))
    arch["mir_floors_m1"] = list(np.round(sig_mir, 12))
    arch["window_elements_um"] = list(np.round(win_c, 2))
    arch["window_floors_m1"] = list(np.round(sig_win, 12))
    out = _PARENT / "outputs" / "calibrated_background_thresholds.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(arch, f, indent=1)
    print(f"\narchived -> {out}")


if __name__ == "__main__":
    main()
