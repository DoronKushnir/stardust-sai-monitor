"""
ace_mir_aerosol.py -- direct demonstration that the stratospheric sulfate
aerosol is visible in the MIR in real ACE-FTS occultations after gas removal,
using the public v5.2 residual spectra (data/ace/v52_subset/, 143 tropical
occultations, 2004-2024).

Everything stays in SLANT space: the residual files give measured/gas-fit
transmittance per tangent height, and ACE tangent heights are already
refraction-corrected (N2-continuum analysis; Boone & Bernath 2019), so no
onion-peel inversion -- and no refraction modelling -- is needed.

For each occultation and tangent height we band-average the residual
transmittance over the two MIR channels of Config B/B+:
    8.80 um element: 8.75-8.85 um (1129.9-1142.9 cm^-1; silica reststrahlen / sulfate band; round 43, was 8.74)
    4.0  um element: 3.95-4.05 um (2469.1-2531.6 cm^-1; continuum channel)
    (round 37: 0.1-um elements, the paper's convention; were 0.25 um)
and form the slant aerosol OD  tau = -ln<T_resid>.

Outputs
  figures/ace_mir_aerosol.png   4 panels (round-22, Doron: the former
  time-series panel is removed and the rest relabeled):
    (a) median residual spectrum at 19-22 km, quiet vs post-Hunga Tonga,
        drawn only on the trusted (strict-mask) bins (round-22), with
        the anchor-constrained model, its sulfate-only part, and the
        nominal sulfate at the matched anchor.  Round-14
        (Doron): drawn as slant OD versus wavelength [um] (rather than
        transmittance versus wavenumber), so it compares directly with
        the atlas figure scisat_ace_atlas_874.png.  Round-15 (Doron):
        x range matched to that figure (8.0-9.5 um).  The elevated-model
        peak here (~0.070) is lower than in the atlas figure (~0.094)
        purely through tangent height: the atlas bin is at 18 km, this
        panel's composites at 19-22 km (model drawn at 20 km), and the
        chord through the 16-24 km layer is 1.34x longer at 18 km
        (checked with model_sulfate_slant: 8.74 um element OD 0.0729 at
        18 km vs 0.0543 at 20 km);
    (b) tau_slant(8.74 um) profiles vs tangent height (tall right-hand
        panel), quiet vs post-HT, with the matched-anchor nominal model
        and (round-22) the anchor-constrained model distributed along
        the epoch-median matched GloSSAC 525-nm profile shape;
    (c) the held-year full-element prediction errors of Appendix
        app:acefloor (the measured floor);
    (d) round-18 (Doron): revised per the Appendix app:acefloor analysis.
        Full ACE range (2.2-13.35 um, 750-4546 cm^-1): the canonical
        height-averaged 19-22 km epoch composites of the ACE side-study
        record against the two-mode ensemble decomposition -- the
        empirical intercept A(nu) (which absorbs the epoch-shared
        removal systematics, including the O3-edge product artifact; no
        frozen template) plus a two-component sulfate member fit.
        Round-21 (Doron): the LEADING fit variant is now the
        GloSSAC-ANCHOR-CONSTRAINED two-component fit -- the member
        library's amplitude convention (revision_optics.py: unit
        amplitude == the 525-nm slant OD of the elevated reference
        profile at a 20.5-km tangent, BASE_COLUMN) makes "match the
        GloSSAC anchor" a linear equality constraint on the amplitude
        sum; the anchor is the epoch-median GloSSAC v2.23 525-nm slant
        OD computed from each occultation's own matched (month,
        latitude) extinction profile chorded at its own stored 19-22 km
        tangent heights.  The scan covers all within-member PSD pairs
        of the 14-member x 17-PSD library; the drawn leading fit is the
        best constrained pair among members at stratospherically
        relevant temperatures (T <= 223 K).  The MIR-only
        (unconstrained) fits -- including the round-18 canonical
        Biermann 70 wt%/215 K, 60 nm + 1 um pair -- are retained for
        comparison, together with the amplitude-sum "valley" profile
        rms_min(c) that quantifies how weakly the MIR spectra alone
        constrain the implied 525-nm amplitude.  All quotable numbers
        are printed and archived to
        outputs/ace_v52/anchor_constrained_fits.json.  Inputs are the
        verified caches in
        data/ace/revision20260909/ (basis.npz,
        matrix_19_22.npz, optics.npz); trusted (strict-mask) bins are
        marked.
  outputs/ace_v52/mir_bands.csv  per-(occultation, tangent height) band ODs.

Round 27 (Doron): panels reordered -- the full-range decomposition is now
panel (a) (top, full width), the 8-9.5 um zoom (b), the profiles (c, tall
right column) and the floor histogram (d); the zoom panel carries the slant
OD of a 0.2-Tg silica injection (20-km tangent) for comparison; and the
superseded nominal single-mode (100 nm, 1.6) sulfate overlays are removed
from the zoom and profile panels (their matched-anchor ratios are still
printed for Appendix app:acefloor_anchor).

Run from the repo root:  python scripts/ace_mir_aerosol.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from saimon.backgrounds import make_empirical_column
from saimon.geometry import tangent_to_slant_paths
from saimon.refractive_index import sulfuric_acid_at_temperature
from saimon.materials import SilicaRefractiveIndex
from saimon.sai import create_silica_sai_layer

SILICA_COMPARE_TG = 0.2     # round 27: silica signal drawn in the zoom panel

DATA = _HERE / "data" / "ace" / "v52_subset"
OUT_DIR = _HERE / "outputs" / "ace_v52"
FIG = _HERE / "figures" / "ace_mir_aerosol.png"
# canonical ensemble record behind Paper 1 Appendix app:acefloor
SIDE = _HERE / "data" / "ace" / "revision20260909"

# Round 37 (Doron): the paper's single 0.1-um element convention (was the
# 0.25-um element: 1128.5-1160.8 and 2422-2578 cm^-1).
# Round 43 (Doron): the silica element centre moved from 8.74 to 8.80 um
# (8.75-8.85 um, 1129.9-1142.9 cm^-1).  The key "874" and the tau874 column
# of mir_bands.csv are KEPT for downstream readers; they now denote 8.80 um.
SILICA_UM = 8.80
ELEMENT_UM = 0.10
BANDS = {
    "874":  (1e4 / (SILICA_UM + ELEMENT_UM / 2), 1e4 / (SILICA_UM - ELEMENT_UM / 2)),   # 8.80 um +- 0.05 um
    "400":  (1e4 / (4.00 + ELEMENT_UM / 2), 1e4 / (4.00 - ELEMENT_UM / 2)),   # 4.0  um +- 0.05 um
}
# spectral window for the shape panel
WIN = (750.0, 1600.0)   # round-17: extended from 1300 for the wide-fit panel

# tropical-relevant eruptions to mark (decimal year)
ERUPTIONS = {
    "Soufrière Hills": 2006.4,
    "Nabro": 2011.45,
    "Kelud": 2014.12,
    "Ambae": 2018.55,
    "Ulawun/Raikoke": 2019.5,
    "La Soufrière": 2021.3,
    "Hunga Tonga": 2022.04,
    "Ruang": 2024.3,
}
QUIET_YEARS = (2013.2, 2014.1)      # post-Nabro decay, pre-Kelud
QUIET_YEARS2 = (2016.5, 2018.5)     # long quiet stretch
HT_YEARS = (2022.1, 2023.5)         # post-Hunga Tonga


def decimal_year(dt_str):
    y = int(dt_str[:4]); m = int(dt_str[5:7]); d = int(dt_str[8:10])
    return y + ((m - 1) * 30.4 + d) / 365.25


def load_geoloc():
    geo = {}
    with open(DATA / "occultationlist.csv") as f:
        for row in csv.DictReader(f):
            geo[row["occultation_name"]] = (
                decimal_year(row["occultation_datetime"]),
                float(row["latitude"]))
    return geo


def parse_residual(path):
    dat = np.loadtxt(path)
    nu, t = dat[:, 0], dat[:, 1]
    ok = t > -0.5                      # -1 = saturated, filter
    return nu[ok], t[ok]


# minimum number of ~2 cm^-1 residual bins inside an element for a valid element
# OD: 8 at the former 0.25-um element (~32 cm^-1), scaled with the element width
# (round 37: 3 at the 0.1-um element, ~13 cm^-1).
MIN_BINS = max(3, int(round(8 * ELEMENT_UM / 0.25)))


def band_od(nu, t, lo, hi):
    m = (nu >= lo) & (nu <= hi)
    if m.sum() < MIN_BINS:
        return np.nan
    tm = float(t[m].mean())
    if tm <= 0:
        return np.nan
    return -np.log(tm)


def ensemble_occs():
    """The canonical 143-occultation ensemble (20.1-25.0 N, June-August,
    19-22 km coverage), pinned to the Appendix app:acefloor analysis record
    so this figure cannot silently drift as more of the released residual
    set is synced into data/ace/v52_subset/ (round-18: the local mirror now
    holds 1274 occultations of the full release, so an unfiltered scan no
    longer reproduces the ensemble)."""
    z = np.load(SIDE / "matrix_19_22.npz", allow_pickle=True)
    return set(str(o) for o in z["occs"])


def collect():
    geo = load_geoloc()
    keep = ensemble_occs()
    rows = []
    for d in sorted((DATA / "residual").iterdir()):
        occ = d.name
        if occ not in geo or occ not in keep:
            continue
        year, lat = geo[occ]
        for f in sorted(d.iterdir()):
            try:
                alt = float(f.name.split(".", 1)[1].removesuffix(".gz"))
            except ValueError:
                continue
            nu, t = parse_residual(f)
            row = dict(occ=occ, year=year, lat=lat, alt=alt)
            for k, (lo, hi) in BANDS.items():
                row[f"tau{k}"] = band_od(nu, t, lo, hi)
            rows.append(row)
    return rows


def spectrum_composite(names, alt_lo=19.0, alt_hi=22.0):
    """Median residual spectrum over occultations in `names`, 19-22 km."""
    grid = np.arange(WIN[0], WIN[1] + 1, 2.0)
    specs = []
    for occ in names:
        d = DATA / "residual" / occ
        if not d.is_dir():
            continue
        for f in d.iterdir():
            try:
                alt = float(f.name.split(".", 1)[1].removesuffix(".gz"))
            except ValueError:
                continue
            if alt_lo <= alt <= alt_hi:
                nu, t = parse_residual(f)
                m = (nu >= WIN[0]) & (nu <= WIN[1])
                specs.append(np.interp(grid, nu[m], t[m],
                                       left=np.nan, right=np.nan))
    return grid, np.nanmedian(np.array(specs), axis=0), len(specs)


def model_sulfate_slant(bg, r0, s0, htan_km):
    """Slant OD of a model sulfate background at the silica element."""
    ALT = np.linspace(0, 60_000, 121)
    ri = sulfuric_acid_at_temperature(215.0)
    col = make_empirical_column(ALT, bg, r0, s0, refractive_index=ri)
    sub = np.linspace(-ELEMENT_UM / 2, ELEMENT_UM / 2, 41)
    wl = (SILICA_UM + sub) * 1e-6
    alpha = col.extinction_profile_m1(wl).mean(axis=1)      # (nz,)
    chords = tangent_to_slant_paths(np.asarray(htan_km) * 1e3, ALT)
    return np.array([float(np.sum(c * alpha)) for c in chords])


def model_sulfate_spectrum(bg, r0, s0, htan_km=20.0):
    """Slant transmittance spectrum of a model sulfate background."""
    ALT = np.linspace(0, 60_000, 121)
    ri = sulfuric_acid_at_temperature(215.0)
    col = make_empirical_column(ALT, bg, r0, s0, refractive_index=ri)
    grid = np.arange(WIN[0], WIN[1] + 1, 10.0)
    wl_m = 1e-2 / grid
    alpha = col.extinction_profile_m1(wl_m)                 # (nz, nwl)
    chord = tangent_to_slant_paths(np.array([htan_km * 1e3]), ALT)[0]
    tau = np.array([float(np.sum(chord * alpha[:, j]))
                    for j in range(len(grid))])
    return grid, np.exp(-tau)


def model_silica_slant_od(mass_tg, htan_km=20.0, step_cm=2.0):
    """Slant OD spectrum of a silica injection (nominal PSD 268 nm, 1.31;
    Kitamura+Popova optics) at a tangent height, on a step_cm wavenumber
    grid over WIN."""
    ALT = np.linspace(0, 60_000, 121)
    sil = create_silica_sai_layer(
        ALT, mass_tg, rmed_nm=268.0, sigma=1.31,
        refractive_index=SilicaRefractiveIndex(
            "data/optics/SiO2_KitamuraPopova.yml"))
    grid = np.arange(WIN[0], WIN[1] + 1, step_cm)
    alpha = sil.extinction_profile_m1(1e-2 / grid)          # (nz, nwl)
    chord = tangent_to_slant_paths(np.array([htan_km * 1e3]), ALT)[0]
    return grid, chord @ alpha


# ---- two-component member fits: GloSSAC-anchor machinery (round 21) --------

# Member metadata for the optics.npz library of the ACE side-study
# (revision_optics.py MEMBERS): laboratory H2SO4 optical-constant datasets,
# member id -> (source, wt%, T[K]).  B*/Bt* = Biermann et al. (2000),
# L*/Lt* = Lund Myhre et al. (2003).
MEMBER_META = {
    "B57T213": ("Biermann", 57, 213), "B60T213": ("Biermann", 60, 213),
    "B64T213": ("Biermann", 64, 213), "B70T215": ("Biermann", 70, 215),
    "B75T215": ("Biermann", 75, 215), "B80T215": ("Biermann", 80, 215),
    "L48T213": ("Lund Myhre", 48, 213), "L58T233": ("Lund Myhre", 58, 233),
    "L65T243": ("Lund Myhre", 65, 243), "L72T233": ("Lund Myhre", 72, 233),
    "L76T213": ("Lund Myhre", 76, 213), "Bt64T203": ("Biermann", 64, 203),
    "Lt65T223": ("Lund Myhre", 65, 223), "Lt72T213": ("Lund Myhre", 72, 213),
}
# stratospherically relevant member temperatures (20-km tropics ~195-220 K):
# the drawn leading constrained fit is restricted to T <= COLD_TMAX
COLD_TMAX = 223.0
# the round-18 canonical (MIR-only) pair, kept for continuity
CANONICAL_PAIR = ("B70T215_r60s1.5", "B70T215_r1000s1.8")


def _reference_norm():
    """The amplitude normalization of the member library, replicated from
    reproduce/ace_floor/revision_optics.py: each library
    spectrum is cs(MIR)/cs(525) * BASE_COLUMN, where BASE_COLUMN is the
    525-nm slant OD of the 'elevated' empirical reference profile at a
    20.5-km tangent.  A fitted amplitude is therefore the member's 525-nm
    slant OD in units of BASE_COLUMN, and amplitudes sum linearly across
    modes -- which makes 'match a 525-nm anchor' a linear equality
    constraint on the amplitude sum."""
    from scipy.interpolate import PchipInterpolator
    from saimon.backgrounds import _EMPIRICAL_ALT_KM, _EMPIRICAL_EXT_525
    alt = np.arange(0.0, 60001.0, 500.0)
    prof = np.maximum(PchipInterpolator(
        _EMPIRICAL_ALT_KM, _EMPIRICAL_EXT_525["elevated"])(alt / 1000), 0)
    prof[alt < 16000] = 0
    base_column = float(tangent_to_slant_paths(
        np.array([20500.0]), alt)[0] @ prof)
    saod_ref = float(np.trapezoid(prof, alt))
    return dict(alt=alt, prof=prof, base_column=base_column,
                saod_ref=saod_ref)


def matched_glossac_slant525(occ_names, heights, alt_grid):
    """GloSSAC v2.23 525-nm slant OD 'anchor' per occultation: the matched
    (month, latitude) extinction profile (same matching as
    matched_glossac_saod525) chorded at the occultation's own stored
    19-22 km tangent heights [km] and averaged over them -- the same
    height-averaging convention as the canonical composites.  Masked
    profile bins are treated as zero extinction; values above the 40-km
    GloSSAC ceiling contribute nothing along these chords.

    Returns (slant, betas): per-occultation slant OD and the matched
    525-nm extinction profiles [1/m on alt_grid] (used for the
    constrained-model profile curves of the figure's profile panel)."""
    from saimon.glossac import GloSSACLoader
    geo = {}
    with open(DATA / "occultationlist.csv") as f:
        for r in csv.DictReader(f):
            geo[r["occultation_name"]] = r
    d = GloSSACLoader(str(_HERE / "data/glossac/GloSSAC_V2.23_subset.nc")
                      ).load_raw_data(525)
    out = np.full(len(occ_names), np.nan)
    betas = np.zeros((len(occ_names), len(alt_grid)))
    for i, (occ, hh) in enumerate(zip(occ_names, heights)):
        row = geo[str(occ)]
        ym = int(row["occultation_datetime"][:7].replace("-", ""))
        lat = float(row["latitude"])
        ti = np.argmin(np.abs(d["time"] - ym))
        li = np.argmin(np.abs(d["lat"] - lat))
        prof = np.ma.filled(d["ext"][ti, li, :], 0.0)          # 1/km
        beta = np.interp(alt_grid / 1000.0, d["alt"], prof,
                         left=0.0, right=0.0) / 1000.0          # 1/m
        betas[i] = beta
        chords = tangent_to_slant_paths(
            np.asarray(hh, float) * 1e3, alt_grid)
        out[i] = float(np.mean([c @ beta for c in chords]))
    return out, betas


def _fit_pair_constrained(S1, S2, d, c):
    """min_t ||t*S1 + (c-t)*S2 - d|| with t in [0, c] (both amplitudes
    nonnegative, amplitude sum pinned to c).  Closed form in t."""
    u = S1 - S2
    v = d - c * S2
    denom = float(u @ u)
    t = float(u @ v) / denom if denom > 0 else 0.0
    t = float(np.clip(t, 0.0, c))
    r = t * S1 + (c - t) * S2 - d
    return t, float(np.sqrt(np.mean(r ** 2)))


def _two_component_scans(d_st, lib_st, pair_names, c_anchor, c_grid):
    """All within-member two-component fits on the strict-bin data d_st.

    Returns (records, valley): records has one entry per PSD pair with the
    unconstrained nnls fit and the anchor-constrained fit; valley is
    rms_min(c) over the pair scan on c_grid (the amplitude-sum profile
    that quantifies how weakly the MIR spectra constrain the implied
    525-nm amplitude)."""
    from scipy.optimize import nnls
    records = []
    valley = np.full(len(c_grid), np.inf)
    for n1, n2 in pair_names:
        S1, S2 = lib_st[n1], lib_st[n2]
        coef = nnls(np.column_stack([S1, S2]), d_st)[0]
        rms_u = float(np.sqrt(np.mean(
            (coef[0] * S1 + coef[1] * S2 - d_st) ** 2)))
        t_c, rms_c = _fit_pair_constrained(S1, S2, d_st, c_anchor)
        records.append(dict(pair=(n1, n2), amps_unc=(float(coef[0]),
                            float(coef[1])), rms_unc=rms_u,
                            amps_con=(t_c, c_anchor - t_c), rms_con=rms_c))
        # amplitude-sum valley, vectorized over c (closed form is linear
        # in c before clipping)
        u = S1 - S2
        denom = float(u @ u)
        t = (float(u @ d_st) - c_grid * float(u @ S2)) / denom
        t = np.clip(t, 0.0, c_grid)
        resid = (t[:, None] * S1[None, :]
                 + (c_grid - t)[:, None] * S2[None, :] - d_st[None, :])
        valley = np.minimum(valley, np.sqrt(np.mean(resid ** 2, axis=1)))
    return records, valley


_ENSEMBLE_CACHE = None


def panel_e_ensemble():
    """Two-mode ensemble decomposition + two-component sulfate member fits.

    Loads the canonical 19-22 km record of the ACE side-study (the same
    analysis as Paper 1 Appendix app:acefloor): per-occultation
    height-averaged residual spectra M on the full 750-4546 cm^-1 grid,
    the ensemble basis (m1, m2, A) and strict trust mask, and the Mie
    member library.  For each epoch composite (median over the epoch's
    occultations) the empirical intercept A is removed and two-component
    (fine + tail lognormal modes, one member's optics, nonnegative
    amplitudes) fits are run on the strict-mask bins in three variants:

      lead      -- LEADING: anchor-constrained (amplitude sum pinned to
                   the epoch's matched GloSSAC 525-nm slant OD in
                   BASE_COLUMN units), best pair among T <= 223 K
                   members;
      best_all  -- anchor-constrained, best pair over all 14 members;
      best_unc  -- MIR-only (unconstrained), best pair over all members;
      canonical -- the round-18 MIR-only Biermann 70 wt%/215 K,
                   60 nm s1.5 + 1 um s1.8 fit, for continuity.

    Plus the amplitude-sum valley rms_min(c) over the full pair scan.
    """
    global _ENSEMBLE_CACHE
    if _ENSEMBLE_CACHE is not None:
        return _ENSEMBLE_CACHE
    from itertools import combinations
    from scipy.optimize import nnls
    b = np.load(SIDE / "basis.npz")
    z = np.load(SIDE / "matrix_19_22.npz", allow_pickle=True)
    lib = np.load(SIDE / "optics.npz")
    grid, C = b["grid"], b["C"]
    strict = b["strict"].astype(bool)
    M, yrs, occs, hts = z["M"], z["years"], z["occs"], z["heights"]
    A = C[-1]

    ref = _reference_norm()
    base = ref["base_column"]
    # matched GloSSAC 525-nm slant anchors at the stored tangent heights
    slant525, betas = matched_glossac_slant525(occs, hts, ref["alt"])

    spec_names = sorted(k[:-5] for k in lib.keys() if k.endswith("_spec"))
    members = sorted({n.split("_")[0] for n in spec_names})
    pair_names = [p for mem in members for p in combinations(
        [n for n in spec_names if n.split("_")[0] == mem], 2)]
    cold = {mem for mem in members if MEMBER_META[mem][2] <= COLD_TMAX}
    c_grid = np.linspace(0.05, 1.4, 55)

    out = {"grid": grid, "A": A, "strict": strict, "base_column": base,
           "saod_ref": ref["saod_ref"], "c_grid": c_grid}
    S_can = np.column_stack([lib[CANONICAL_PAIR[0] + "_spec"],
                             lib[CANONICAL_PAIR[1] + "_spec"]])
    for key, lo, hi in [("q", QUIET_YEARS2[0], QUIET_YEARS2[1]),
                        ("h", HT_YEARS[0], HT_YEARS[1])]:
        sel = (yrs >= lo) & (yrs <= hi)
        tau = np.nanmedian(M[sel], axis=0)
        d = tau - A
        st = strict & np.isfinite(d)
        d_st = d[st]
        anchor = float(np.nanmedian(slant525[sel]))
        c_anchor = anchor / base
        # geometry consistency of the constraint: the library amplitude is
        # normalized at a single 20.5-km tangent (BASE_COLUMN) while the
        # anchor averages each occultation's stored 19-22 km heights; for
        # the reference shape the two conventions agree to this ratio
        geom = [float(np.mean([c @ ref["prof"] for c in
                               tangent_to_slant_paths(
                                   np.asarray(hh, float) * 1e3,
                                   ref["alt"])])) / base
                for hh in hts[sel]]
        geom_ratio = float(np.median(geom))

        # canonical round-18 fit (continuity)
        coef = nnls(S_can[st], d_st)[0]
        model_can = A + S_can @ coef
        rms_can = float(np.sqrt(np.mean((tau - model_can)[st] ** 2)))

        # full within-member pair scans
        lib_st = {n: lib[n + "_spec"][st] for n in spec_names}
        recs, valley = _two_component_scans(d_st, lib_st, pair_names,
                                            c_anchor, c_grid)
        best_all = min(recs, key=lambda r: r["rms_con"])
        best_cold = min((r for r in recs
                         if r["pair"][0].split("_")[0] in cold),
                        key=lambda r: r["rms_con"])
        best_unc = min(recs, key=lambda r: r["rms_unc"])

        def _model(rec, which):
            amps = rec["amps_" + which]
            return A + (amps[0] * lib[rec["pair"][0] + "_spec"]
                        + amps[1] * lib[rec["pair"][1] + "_spec"])

        lead_model = _model(best_cold, "con")
        # fine-mode share of the leading model's MIR signal (mean over
        # strict bins, and over the 8.74-um element bins): the constraint
        # is NOT absorbed by an MIR-invisible small-particle reservoir
        el = (grid >= BANDS["874"][0]) & (grid <= BANDS["874"][1])
        parts = [a * lib[n + "_spec"]
                 for n, a in zip(best_cold["pair"], best_cold["amps_con"])]
        i_fine = int(np.argmax(
            [float(n.split("_r")[1].split("s")[0]) < 500
             for n in best_cold["pair"]]))
        tot = parts[0] + parts[1]
        fine_share = float(np.mean(parts[i_fine][st]) / np.mean(tot[st]))
        fine_share_el = float(np.mean(parts[i_fine][st & el])
                              / np.mean(tot[st & el]))
        # round-22 (Doron): constrained-model slant-OD profile at the
        # 8.74-um element for the figure's profile panel -- the sulfate
        # part of the leading model, distributed in height along the
        # epoch-median matched GloSSAC 525-nm extinction profile and
        # normalized so its average over the record's stored 19-22 km
        # tangent heights equals the fitted element OD
        sulf874 = float(-np.log(np.mean(np.exp(-tot[st & el]))))
        beta_med = np.median(betas[sel], axis=0)
        h_pool = np.concatenate([np.asarray(hh, float) for hh in hts[sel]])
        h_prof = np.arange(16.0, 33.01, 0.5)
        P = np.array([c @ beta_med for c in
                      tangent_to_slant_paths(h_prof * 1e3, ref["alt"])])
        P_avg = float(np.mean([c @ beta_med for c in
                               tangent_to_slant_paths(h_pool * 1e3,
                                                      ref["alt"])]))
        prof874 = sulf874 * P / P_avg
        # flat range of the amplitude-sum valley (within 10% of its min)
        vmin = float(valley.min())
        flat = c_grid[valley <= 1.1 * vmin]
        ep = dict(
            tau=tau, n=int(sel.sum()), st=st,
            anchor_slant525=anchor, c_anchor=c_anchor,
            geom_ratio=geom_ratio,
            canonical=dict(pair=CANONICAL_PAIR,
                           amps=(float(coef[0]), float(coef[1])),
                           rms=rms_can, ampsum=float(coef.sum()),
                           model=model_can),
            lead=dict(pair=best_cold["pair"], amps=best_cold["amps_con"],
                      rms=best_cold["rms_con"], model=lead_model,
                      fine_share=fine_share, fine_share_el=fine_share_el,
                      sulf874_el=sulf874),
            profile874=(h_prof, prof874),
            best_all=dict(pair=best_all["pair"],
                          amps=best_all["amps_con"],
                          rms=best_all["rms_con"]),
            best_unc=dict(pair=best_unc["pair"],
                          amps=best_unc["amps_unc"],
                          rms=best_unc["rms_unc"],
                          ampsum=float(sum(best_unc["amps_unc"])),
                          model=_model(best_unc, "unc")),
            valley=valley, valley_flat=(float(flat.min()),
                                        float(flat.max())),
        )
        # keep panel compatibility: "model" is the LEADING (constrained)
        # fit; the canonical MIR-only fit lives under "canonical"
        ep["model"] = lead_model
        out[key] = ep

        mq = ep  # short alias for the print block
        print(f"panel (e) fits [{key}]: n_occ={ep['n']}, strict bins "
              f"{int(st.sum())}")
        print(f"  matched GloSSAC slant OD525 anchor = {anchor:.4f} "
              f"(amplitude sum {c_anchor:.3f} in BASE_COLUMN={base:.4f} "
              f"units; implied SAOD525 {c_anchor * ref['saod_ref']:.4f}; "
              f"20.5-km vs height-avg geometry ratio {geom_ratio:.3f})")
        print(f"  LEAD (anchor-constrained, T<={COLD_TMAX:.0f}K): "
              f"{best_cold['pair'][0]}+{best_cold['pair'][1]}, amps "
              f"{best_cold['amps_con'][0]:.3f}/"
              f"{best_cold['amps_con'][1]:.3f}, "
              f"rms {best_cold['rms_con']:.5f} OD; fine-mode MIR share "
              f"{fine_share:.2f} (strict mean) / {fine_share_el:.2f} "
              f"({SILICA_UM:g} um element)")
        print(f"  constrained best (all members): "
              f"{best_all['pair'][0]}+{best_all['pair'][1]}, "
              f"rms {best_all['rms_con']:.5f}")
        print(f"  MIR-only best: {best_unc['pair'][0]}+"
              f"{best_unc['pair'][1]}, rms {best_unc['rms_unc']:.5f}, "
              f"amplitude sum {mq['best_unc']['ampsum']:.3f} "
              f"({mq['best_unc']['ampsum'] / c_anchor:.2f} of anchor)")
        print(f"  canonical (round-18, MIR-only) "
              f"{CANONICAL_PAIR[0]}+{CANONICAL_PAIR[1]}: fine/tail "
              f"{coef[0]:.4f}/{coef[1]:.4f}, rms {rms_can:.5f}, "
              f"amplitude sum {coef.sum():.3f} "
              f"({coef.sum() / c_anchor:.2f} of anchor)")
        print(f"  amplitude-sum valley: min rms {vmin:.5f}, within 10% "
              f"for sum in [{ep['valley_flat'][0]:.2f}, "
              f"{ep['valley_flat'][1]:.2f}] "
              f"([{ep['valley_flat'][0] / c_anchor:.2f}, "
              f"{ep['valley_flat'][1] / c_anchor:.2f}] of anchor)")
    _ENSEMBLE_CACHE = out
    return out


def matched_glossac_saod525(occ_names):
    """GloSSAC v2.23 SAOD525 at each occultation's own (month, latitude) --
    the file's precomputed stratospheric OD at the nearest month and
    latitude bin.  This is the fair anchor for the ensemble (round 20,
    Doron): the broad tropical SAOD is not, because the released
    occultations all sit at 20.1-25.0 N in June-August while e.g. the
    Hunga plume stayed south of that band."""
    from saimon.glossac import GloSSACLoader
    geo = {}
    with open(DATA / "occultationlist.csv") as f:
        for r in csv.DictReader(f):
            geo[r["occultation_name"]] = r
    d = GloSSACLoader(str(_HERE / "data/glossac/GloSSAC_V2.23_subset.nc")
                      ).load_raw_data(525)
    sa = np.full(len(occ_names), np.nan)
    for i, occ in enumerate(occ_names):
        row = geo[str(occ)]
        ym = int(row["occultation_datetime"][:7].replace("-", ""))
        lat = float(row["latitude"])
        ti = np.argmin(np.abs(d["time"] - ym))
        li = np.argmin(np.abs(d["lat"] - lat))
        v = d["od_file"][ti, li]
        if v is not np.ma.masked:
            sa[i] = float(v)
    return sa


def draw_panel_e(axE, title=None):
    """Full-range panel, round-19 styling (Doron): paper (Times/serif)
    fonts, quiet epoch black / post-HT red, no proxy-band or artifact
    shading, and the sulfate-only part of each model
    (model minus the intercept A) dotted.  Round-21 (Doron): the dashed
    model is the LEADING anchor-constrained fit; residual dots refer to
    it.  Round-22 (Doron): the thin MIR-only (unconstrained) overlay is
    removed -- the constrained fit is the only fit drawn.  Round 27: an
    optional panel title (it is now panel (a) of the paper figure)."""
    ens = panel_e_ensemble()
    if title:
        axE.set_title(title, fontsize=13)
    wlE = 1e4 / ens["grid"]
    strictE = ens["strict"]
    for key, color, lab in [("q", "k", "2016.5-2018.5"),
                            ("h", "r", "2022-2023.5")]:
        tau = ens[key]["tau"]
        model = ens[key]["model"]
        # plot with NaNs left in place so lines break at coverage gaps
        axE.plot(wlE, tau, lw=0.8, color=color, label=f"{lab} data")
        axE.plot(wlE, np.where(np.isfinite(tau), model, np.nan), lw=1.0,
                 ls="--", color=color, alpha=0.9,
                 label=f"{lab} model")   # round 45 (Doron): no "anchor-constrained" in legends
        # sulfate-only part of the model (intercept A removed)
        axE.plot(wlE, model - ens["A"], lw=1.0, ls=":", color=color,
                 alpha=0.9, label="_nolegend_")
        res = np.where(strictE, tau - model, np.nan)
        axE.plot(wlE, res, lw=0.0, marker=".", ms=1.6, color=color,
                 alpha=0.55, label="_nolegend_")
    axE.axhline(0.0, color="k", lw=0.6)
    # trusted (strict-mask) bins of the ensemble analysis
    axE.plot(wlE[strictE], np.full(strictE.sum(), -0.038), "|", ms=5,
             color="k", alpha=0.55, lw=0.4, label="_nolegend_")
    axE.axvspan(1e4 / BANDS["874"][1], 1e4 / BANDS["874"][0],
                color="magenta", alpha=0.12)
    axE.set_xlim(2.2, 13.35)
    axE.set_ylim(-0.045, 0.16)
    axE.set_xticks(np.arange(3.0, 14.0, 1.0))
    axE.set_xlabel(r"wavelength [$\mu$m]")
    axE.set_ylabel("slant optical depth")
    axE.grid(alpha=0.3)
    axE.legend(fontsize=11, loc="upper right")


# Match the Copernicus (Times) text of the paper; STIX math gives an
# italic mu in the axis labels.
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                   "STIXGeneral"],
    "mathtext.fontset": "stix",
    # round-20 (Doron, caption comment): larger fonts throughout
    "font.size": 16,        # round 44 (Doron): larger fonts, heavier lines
    "axes.titlesize": 15,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
}


def panel_e_standalone():
    """Fast iteration on panel (e) alone: skips the residual-directory scan
    of main() and renders only the ensemble panel with the paper fonts.
    Output: figures/ace_mir_aerosol_panel_e.png"""
    out = _HERE / "figures" / "ace_mir_aerosol_panel_e.png"
    with plt.rc_context(PAPER_RC):
        fig, axE = plt.subplots(figsize=(13, 4.2))
        draw_panel_e(axE)
        fig.tight_layout()
        fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved -> {out}")


def main():
    rows = collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = ["occ", "year", "lat", "alt"] + [f"tau{k}" for k in BANDS]
    with open(OUT_DIR / "mir_bands.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"parsed {len(rows)} (occultation, tangent-height) spectra "
          f"from {len(set(r['occ'] for r in rows))} occultations")

    arr = {k: np.array([r[k] for r in rows]) for k in keys[1:]}
    occs = np.array([r["occ"] for r in rows])

    # ---- panel data -------------------------------------------------------
    near20 = np.abs(arr["alt"] - 20.0) <= 1.5
    quiet_occ = sorted(set(occs[(arr["year"] >= QUIET_YEARS2[0])
                               & (arr["year"] <= QUIET_YEARS2[1])]))
    ht_occ = sorted(set(occs[(arr["year"] >= HT_YEARS[0])
                             & (arr["year"] <= HT_YEARS[1])]))
    print(f"quiet composite: {len(quiet_occ)} occultations "
          f"({QUIET_YEARS2[0]}-{QUIET_YEARS2[1]})")
    print(f"post-HT composite: {len(ht_occ)} occultations "
          f"({HT_YEARS[0]}-{HT_YEARS[1]})")

    # ---- Appendix app:acefloor machinery (round 20, Doron): every panel
    # now uses the canonical ensemble record + matched GloSSAC anchors ----
    ens = panel_e_ensemble()
    # archive the round-21 anchor-constrained fit numbers (the quotables of
    # Paper 1 Appendix app:acefloor, two-component-fit subsection)
    import json
    arch = dict(base_column=ens["base_column"], saod_ref=ens["saod_ref"],
                cold_tmax=COLD_TMAX, canonical_pair=list(CANONICAL_PAIR))
    for key, tag in [("q", "quiet"), ("h", "post_hunga_tonga")]:
        e = ens[key]
        arch[tag] = dict(
            n_occ=e["n"], anchor_slant525=e["anchor_slant525"],
            c_anchor=e["c_anchor"], geom_ratio=e["geom_ratio"],
            implied_saod525=e["c_anchor"] * ens["saod_ref"],
            lead=dict(pair=list(e["lead"]["pair"]),
                      amps=list(e["lead"]["amps"]),
                      rms=e["lead"]["rms"],
                      fine_share=e["lead"]["fine_share"],
                      fine_share_el=e["lead"]["fine_share_el"]),
            best_all_constrained=dict(pair=list(e["best_all"]["pair"]),
                                      amps=list(e["best_all"]["amps"]),
                                      rms=e["best_all"]["rms"]),
            best_unconstrained=dict(pair=list(e["best_unc"]["pair"]),
                                    amps=list(e["best_unc"]["amps"]),
                                    rms=e["best_unc"]["rms"],
                                    ampsum=e["best_unc"]["ampsum"],
                                    anchor_fraction=e["best_unc"]["ampsum"]
                                    / e["c_anchor"]),
            canonical=dict(amps=list(e["canonical"]["amps"]),
                           rms=e["canonical"]["rms"],
                           ampsum=e["canonical"]["ampsum"],
                           anchor_fraction=e["canonical"]["ampsum"]
                           / e["c_anchor"]),
            valley=dict(c_grid=list(np.round(ens["c_grid"], 4)),
                        rms_min=list(np.round(e["valley"], 6)),
                        flat10=list(e["valley_flat"]),
                        flat10_anchor_frac=[e["valley_flat"][0]
                                            / e["c_anchor"],
                                            e["valley_flat"][1]
                                            / e["c_anchor"]]),
        )
    with open(OUT_DIR / "anchor_constrained_fits.json", "w") as f:
        json.dump(arch, f, indent=1)
    print(f"archived -> {OUT_DIR / 'anchor_constrained_fits.json'}")
    zM = np.load(SIDE / "matrix_19_22.npz", allow_pickle=True)
    M, yrsM, occsM = zM["M"], zM["years"], zM["occs"]
    gridM = ens["grid"]
    wlM = 1e4 / gridM
    el = (gridM >= BANDS["874"][0]) & (gridM <= BANDS["874"][1])
    # per-occultation 8.74 um element OD, flat-response convention
    # q = -ln<exp(-y)> (height-averaged 19-22 km canonical spectra)
    with np.errstate(invalid="ignore"):
        q_el = -np.log(np.nanmean(np.exp(-M[:, el]), axis=1))
    sel_q = (yrsM >= QUIET_YEARS2[0]) & (yrsM <= QUIET_YEARS2[1])
    sel_h = (yrsM >= HT_YEARS[0]) & (yrsM <= HT_YEARS[1])
    med_q = float(np.nanmedian(q_el[sel_q]))
    med_h = float(np.nanmedian(q_el[sel_h]))
    print(f"{SILICA_UM:g} element OD (canonical 19-22 km): quiet median {med_q:.4f} "
          f"(n={sel_q.sum()}), post-HT {med_h:.4f} (n={sel_h.sum()})")

    # matched GloSSAC SAOD525 at each occultation's own (month, latitude)
    sa = matched_glossac_saod525(occsM)
    np.savez(OUT_DIR / "glossac_matched_saod525.npz",
             occs=occsM, saod525=sa, years=yrsM)
    saq = (float(np.nanmedian(sa[sel_q])), float(np.nanmin(sa[sel_q])),
           float(np.nanmax(sa[sel_q])))
    sah = (float(np.nanmedian(sa[sel_h])), float(np.nanmin(sa[sel_h])),
           float(np.nanmax(sa[sel_h])))
    print(f"matched GloSSAC SAOD525: quiet {saq[0]:.4f} "
          f"[{saq[1]:.4f},{saq[2]:.4f}], post-HT {sah[0]:.4f} "
          f"[{sah[1]:.4f},{sah[2]:.4f}]")

    # nominal-PSD sulfate model (elevated shape, 100 nm, 1.6), scaled per
    # epoch so its vertical 525-nm column equals the matched SAOD525
    ALT = np.linspace(0, 60_000, 121)
    ri = sulfuric_acid_at_temperature(215.0)
    col = make_empirical_column(ALT, "elevated", 100, 1.6,
                                refractive_index=ri)
    base525 = float(np.trapezoid(
        col.extinction_profile_m1(np.array([525e-9]))[:, 0], ALT))
    scale_q, scale_h = saq[0] / base525, sah[0] / base525
    gm2, tm_elev = model_sulfate_spectrum("elevated", 100, 1.6)
    h_grid = np.arange(16.0, 33.0, 1.0)
    od_elev = model_sulfate_slant("elevated", 100, 1.6, h_grid)
    print(f"model vertical OD(525) = {base525:.4f}; matched scale factors "
          f"quiet {scale_q:.2f}, post-HT {scale_h:.2f}")
    # measured-to-model MIR ratio at the element, matched anchors (20 km
    # model against the 19-22 km composites, the panel-(a) convention)
    mel = (gm2 >= BANDS["874"][0]) & (gm2 <= BANDS["874"][1])
    el_model = float(-np.log(tm_elev[mel]).mean())
    print(f"MIR-per-anchor ratio at {SILICA_UM:g} um: quiet "
          f"{med_q / (scale_q * el_model):.2f}, post-HT "
          f"{med_h / (scale_h * el_model):.2f}")

    # held-year full-element prediction errors (Appendix app:acefloor;
    # verified archive record) for panel (d)
    # round 37: the direct 0.1-um-element archive (was residual_floor/, 0.25 um)
    _scores = (_HERE / "data/ace_floor/w0p1"
               / f"19_22_{SILICA_UM:g}um_scores.csv")
    if not _scores.exists():   # round 43: pipeline at 8.80 um still running
        _scores = (_HERE / "data/ace_floor/w0p1_874"
                   / "19_22_8.74um_scores.csv")
        print(f"WARNING: {SILICA_UM:g}-um floor record not yet archived; panel (d) "
              f"uses the 8.74-um record {_scores}")
    sc = np.genfromtxt(_scores, delimiter=",", names=True, dtype=None, encoding=None)
    err = sc["error_od"]
    sd_floor = float(np.std(err - err.mean(), ddof=1))
    print(f"held-year element prediction errors: n={len(err)}, "
          f"sd = {sd_floor:.6f} OD")

    # empirical high-altitude noise floor
    hi = arr["alt"] >= 28.0
    floor = arr["tau874"][hi]
    floor = floor[np.isfinite(floor)]
    print(f"\nhigh-altitude (>=28 km) tau_874: N={len(floor)}, "
          f"median={np.median(floor):+.2e}, "
          f"robust sigma={1.4826*np.median(np.abs(floor-np.median(floor))):.2e}")

    # per-measurement noise, isolated from real profile structure: robust
    # scatter of adjacent-tangent-height differences at >=30 km (the smooth
    # aerosol profile cancels in the difference; /sqrt(2) for two samples)
    diffs = []
    t874 = arr["tau874"]
    for o in set(occs):
        m = (occs == o) & (arr["alt"] >= 30.0) & np.isfinite(t874)
        if m.sum() >= 2:
            t = t874[m][np.argsort(arr["alt"][m])]
            diffs.extend(np.diff(t))
    diffs = np.array(diffs)
    sig_meas = 1.4826 * np.median(np.abs(diffs - np.median(diffs))) / np.sqrt(2)
    print(f"per-measurement noise from adjacent-height differences "
          f"(>=30 km): sigma = {sig_meas:.4f}")

    # headline numbers near 20 km
    for tag, sel_occ in [("quiet", quiet_occ), ("post-HT", ht_occ)]:
        sel = near20 & np.isin(occs, sel_occ)
        v = arr["tau874"][sel]
        print(f"tau_874 near 20 km, {tag}: N={sel.sum()}, "
              f"median={np.median(v[np.isfinite(v)]):.4f}")

    # ---- figure -----------------------------------------------------------
    # round-20 (Doron): all panels rebuilt on the Appendix app:acefloor
    # machinery -- canonical height-averaged spectra, two-mode ensemble
    # decomposition, the appendix's best two-component member, GloSSAC
    # anchors sub-sampled at the ensemble's own months/latitudes, and the
    # held-year full-element floor; epoch colors black/red as in panel (e);
    # larger fonts (round-19/20 caption comment).
    plt.rcParams.update(PAPER_RC)
    # 13.5 (not 13): the serif fonts crop tighter horizontally, and at 12 cm
    # column width the taller aspect pushed the float past a page.
    # Round-22 (Doron): the time-series panel is removed; the profile panel
    # moves to a tall right-hand column (natural for a height axis).
    # Round 27 (Doron): panel order (a,b,c,d) -> the full-range panel leads
    # as (a) (top, full width); zoom (b), profiles (c), floor (d).
    fig = plt.figure(figsize=(13.5, 12.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.85, 1.0, 1.0])
    axE = fig.add_subplot(gs[0, :])
    axA = fig.add_subplot(gs[1, 0])
    axB = fig.add_subplot(gs[1:3, 1])
    axC = fig.add_subplot(gs[2, 0])

    # (b) zoom of the panel-(a) decomposition into the silica region;
    # round-21: the dashed model is the LEADING anchor-constrained fit;
    # round-22 (Doron): the data are drawn only on the trusted (strict-mask)
    # bins; round-23 (Doron): the data are drawn as circles, no connecting
    # line; round 27 (Doron): the nominal single-mode overlay is removed and
    # the slant OD of a 0.2-Tg silica injection is added for comparison
    ax = axA
    strictM = ens["strict"]
    for key, sa_e, color, lab in [("q", saq, "k", "2016.5-2018.5"),
                                  ("h", sah, "r", "2022-2023.5")]:
        tau = ens[key]["tau"]
        model = ens[key]["model"]
        ax.plot(wlM, np.where(strictM, tau, np.nan), lw=0.0, marker="o",
                ms=3.2, mew=0, color=color, alpha=0.8,
                label=f"{lab} data")   # round 45 (Doron): SAOD numbers dropped from the legend
        ax.plot(wlM, np.where(np.isfinite(tau), model, np.nan), lw=1.8,
                ls="--", color=color,
                label=f"{lab} model")
        ax.plot(wlM, model - ens["A"], lw=1.8, ls=":", color=color,
                label="_nolegend_")
    g_sil, tau_sil = model_silica_slant_od(SILICA_COMPARE_TG, htan_km=20.0)
    ax.plot(1e4 / g_sil, tau_sil, lw=2.4, color="tab:blue",
            label=f"silica, {SILICA_COMPARE_TG:g} Tg (20 km tangent)")
    el_s = (g_sil >= BANDS["874"][0]) & (g_sil <= BANDS["874"][1])
    print(f"{SILICA_COMPARE_TG:g}-Tg silica slant OD at the {SILICA_UM:g} um element, "
          f"20 km tangent: {float(np.mean(tau_sil[el_s])):.4f} (peak "
          f"{float(np.max(tau_sil)):.4f} at {1e4 / g_sil[np.argmax(tau_sil)]:.2f} um)")
    ax.axvspan(1e4 / BANDS["874"][1], 1e4 / BANDS["874"][0], color="magenta",
               alpha=0.15)
    ax.set_xlabel(r"wavelength [$\mu$m]")
    ax.set_ylabel("slant optical depth")
    ax.set_title(f"(b) Composites and decomposition, {SILICA_UM:.2f} $\\mu$m region",
                 fontsize=15)
    ax.set_xlim(8.0, 9.5)   # match the atlas figure scisat_ace_atlas_874
    ax.set_ylim(0.0, 0.15)
    ax.legend(fontsize=12, loc="upper right")   # round 27: clear of silica
    ax.grid(alpha=0.3)

    # (c) profiles against the anchor-constrained model of panels (a)/(b)
    # (round-22, Doron), distributed in height along the epoch's matched
    # GloSSAC profile shape; round 27: nominal single-mode overlay removed
    ax = axB
    for tag, sel_occ, c in [("2016.5-2018.5", quiet_occ, "k"),
                            ("2022-2023.5", ht_occ, "r")]:
        sel = np.isin(occs, sel_occ) & np.isfinite(arr["tau874"])
        ax.scatter(arr["tau874"][sel], arr["alt"][sel], s=10, alpha=0.5,
                   color=c, label=f"{tag} data")
    for key, c in [("q", "k"), ("h", "r")]:
        h_prof, prof874 = ens[key]["profile874"]
        ax.plot(prof874, h_prof, color=c, ls="--", lw=2.2,
                label="model (matched GloSSAC shape)"
                      if c == "k" else "_nolegend_")
    ax.set_xlabel(rf"$\tau_{{{SILICA_UM:.2f}}}$")
    ax.set_ylabel("tangent height [km]")
    ax.set_title("(c) Slant OD profiles vs the constrained model",
                 fontsize=15)
    ax.set_xlim(-0.01, 0.25)
    ax.set_ylim(15, 34)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    # (d) the Appendix app:acefloor floor: held-year full-element
    # prediction errors (verified archive record)
    ax = axC
    ax.hist(err * 1e3, bins=30, color="0.75", edgecolor="k", lw=0.3)
    BUDGET_MOD = 2.47   # round 45: the calculated OE budget at the 8.80-um element (1e-3 OD; was 2.0 at 8.74)
    ax.axvline(-BUDGET_MOD, color="red", lw=2.0, ls="--",   # round 44 (Doron): magenta -> red
               label=r"trace-gas budget $\pm\sigma(\tau_{\rm gas})"
                     r"=2.5\times10^{-3}$")
    ax.axvline(BUDGET_MOD, color="red", lw=2.0, ls="--", label="_nolegend_")
    ax.set_xlabel(r"held-year element prediction error [$10^{-3}$ OD]")
    ax.set_ylabel("count")
    ax.set_title(f"(d) {SILICA_UM:.2f} $\\mu$m element held-year errors: "
                 f"SD = {sd_floor*1e3:.2f}$\\times10^{{-3}}$ OD",
                 fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    # panel (a), round-18/19: full ACE range, two-mode ensemble
    # decomposition (Appendix app:acefloor) + the anchor-constrained
    # two-component sulfate fit (leads the figure since round 27)
    draw_panel_e(axE, title="(a) Full-range composites and decomposition, "
                            "2.2$-$13.35 $\\mu$m")

    fig.tight_layout()
    fig.savefig(FIG, dpi=180, bbox_inches="tight")
    print(f"\nSaved -> {FIG}")


if __name__ == "__main__":
    if "--panel-e" in sys.argv:
        panel_e_standalone()
    else:
        main()
