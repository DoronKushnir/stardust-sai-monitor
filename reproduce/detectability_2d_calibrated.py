"""
detectability_2d_calibrated.py -- Paper 1 Sect. sec:2d re-based on the
CALIBRATED two-component background and the current design configurations
(round 35b, 2026-09-24, Doron's comment in latitude.tex).

Replaces the single-mode / single-element machinery of
run_detectability_2d.py and run_detectability_2d_altitude.py with the
Sect.-3/5 pipeline (calibrated_background_thresholds.py,
design_sensitivity_calibrated.py, imported bit-identically):

  * background: GloSSAC 20-25N undisturbed-epoch (2016.5-2018.5) 525-nm
    profile carrying the ACE-calibrated two-component microphysics
    (CAL_QUIET, LM65T223 optics); the post-Hunga-Tonga calibration
    (CAL_POSTHT, LM72T213) is the microphysics bracket;
  * configurations: the heritage triplet (448/756/1544 nm, measured
    fractional Level-2 errors, SNR-2000 floor) + the 8-13 um band at
    0.1-um sampling (46 elements, measured per-element floors) -- the
    design -- and its 7.8-9.3 um window subset (15 elements);
  * fit: all seven background parameters free (marginalized bound).

What is new relative to the 1D analysis:
  1. Background AMPLITUDE varies with latitude and altitude: g(phi, z) =
     GloSSAC quiet-epoch median extinction at (phi, z) / the 22.5N value at
     the same z.  With fractional visible/NIR errors the amplitude no
     longer cancels exactly (only the absolute MIR floors and the SNR
     floor are amplitude-independent); the dependence is computed.
  2. Altitude: the noise model is evaluated per tangent altitude
     (12-28 km): visible/NIR fractional errors follow the local background,
     the SNR floor inflates with the slant Rayleigh self-attenuation
     exp(+dtau/2) relative to 20 km, and the MIR element floors follow one
     of three models -- HELD at their measured 20-km values (the Sect.-5
     convention), scaled by the trace-gas OE floor scan (model:
     scan_floor_altitude.py, flat below 20 km, better above), or scaled by
     the measured ACE atlas altitude bins (16-19 / 19-22 / 22-25 km:
     x3.06 / x1 / x0.72 at 8.74 um, piecewise constant over the bins;
     round 43: read at the 8.80-um reference element).
  3. Silica: the 20-km spectral shape (altitude independent) in fixed
     20-km-reference units, so M_min(phi, z) = 3 sigma_A(phi, z) / f(phi, z)
     with f the local-concentration enhancement of the Lederer (2026)
     transport field relative to the 1-Tg homogeneous layer at 20 km
     (saimon.sai_2d.load_silica_zonal_2d, unchanged).

Products
  outputs/detectability_2d_calibrated/results.json    (all headline numbers)
  outputs/detectability_2d_calibrated/mmin_lat_<scenario>.csv
  outputs/detectability_2d_calibrated/map_pm41[_103hPa].npz
  figures/detectability_2d_mass_contours.png   (round 35c: the
      single two-panel figure of the combined +-41 deg case; the former
      distributions / mmin_lat figures are no longer produced)

Run from the repo root:  python scripts/detectability_2d_calibrated.py
                         python scripts/detectability_2d_calibrated.py --plot
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import PchipInterpolator

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from calibrated_background_thresholds import (  # noqa: E402
    ALT, iz, CAL_QUIET, CAL_POSTHT, ALPHA_FLOOR, RMED_SIL_NM, SIGMA_SIL,
    QUIET_YEARS, TwoComponentBackground, glossac_2025N_quiet_profile,
    marginal, member_ri)
from design_sensitivity_calibrated import ElementSet, VIS_NM, VIS_PHI, N_VIS  # noqa: E402
from saimon.atmosphere import rayleigh_cross_section_m2, us_standard_atmosphere  # noqa: E402
from saimon.geometry import tangent_to_slant_paths  # noqa: E402
from saimon.glossac import GloSSACLoader  # noqa: E402
from saimon.materials import SilicaRefractiveIndex  # noqa: E402
from saimon.sai import create_silica_sai_layer  # noqa: E402
from saimon.sai_2d import (load_silica_zonal_2d, obscuration_probability,  # noqa: E402
                          simulate_occultation_latitudes, smooth_over_footprint)

OUT_DIR = _ROOT / "outputs" / "detectability_2d_calibrated"
FIG_DIR = _ROOT / "figures"
THR_JSON = _ROOT / "outputs" / "calibrated_background_thresholds.json"
FLOOR_SCAN = _ROOT / "outputs" / "detectability_2d" / "floor_altitude_scan.csv"
ATLAS = _ROOT / "data/ace_floor/w0p1/atlas.json"   # round 43: the design's 0.1-um element atlas (was the 0.25-um archive)
GLOSSAC = _ROOT / "data/glossac/GloSSAC_V2.23_subset.nc"

Z_KM = np.arange(12.0, 28.001, 0.5)          # tangent altitudes
K_OF_Z = lambda z: int(round(z * 1000.0 / (ALT[1] - ALT[0])))  # noqa: E731
LAT_ANCHOR = 22.5                            # GloSSAC bin of the calibration
FOOT_DEG = 0.7                               # median LOS latitudinal half-width
# round 35c (Doron): the paper shows only the combined +-41 deg case; the
# single-latitude scenarios are kept in the archive for reference
SCENARIOS = {"combined": [-41.0, 41.0], "equatorial": 1.0, "subtropical": 31.0,
             "high-latitude": 61.0}
INJ_PRES_HPA = 51.0
MAP_SCEN = {"pm41": ([-41.0, 41.0], 51.0), "pm41_103hPa": ([-41.0, 41.0], 103.0)}
ORBITS = {"ISS-like": dict(inclination_deg=51.6, altitude_km=420.0),
          "dedicated 70": dict(inclination_deg=70.0, altitude_km=650.0)}
G_SCAN = (0.25, 0.5, 1.0, 2.0, 4.0)
CONTOURS_TG = [0.03, 0.05, 0.1, 0.3, 1.0]
BRACKETS = {"quiet": (CAL_QUIET, "LM65T223"), "postHT": (CAL_POSTHT, "LM72T213")}
FLOOR_MODELS = ("held", "oe", "atlas")
BASELINE = "oe"   # round 35c (Doron): OE-scaled floors are the baseline; "held" retired from the paper
FLOOR_LABEL = {"held": "floors held at 20-km values",
               "oe": "floors scaled by the trace-gas OE scan",
               "atlas": "floors scaled by the measured ACE altitude bins"}


# ---------------------------------------------------------------------------
# GloSSAC quiet-epoch medians: the anchor profile without the 16-km layer cut,
# and the zonal/vertical amplitude factor g(phi, z)
# ---------------------------------------------------------------------------
def glossac_quiet_median():
    """(lat, alt_km, median ext [1/km] over the undisturbed epoch, all months)."""
    d = GloSSACLoader(str(GLOSSAC)).load_raw_data(525)
    years = (d["time"] // 100) + (d["time"] % 100 - 0.5) / 12.0
    m = (years >= QUIET_YEARS[0]) & (years <= QUIET_YEARS[1])
    med = np.ma.median(np.ma.masked_invalid(d["ext"][m]), axis=0)   # (lat, alt)
    return np.asarray(d["lat"]), np.asarray(d["alt"]), np.ma.filled(med, np.nan)


def anchor_profile_full(lat, alt_km, med):
    """22.5N quiet median on ALT [1/m]: bit-identical to
    glossac_2025N_quiet_profile at >= 16 km (so the 20-km design numbers are
    reproduced exactly), extended below 16 km -- where that function zeroes
    the profile for the 1-D layer convention -- by the pchip of the valid
    GloSSAC levels, held at the lowest valid value further down."""
    a_ref = glossac_2025N_quiet_profile()
    li = int(np.argmin(np.abs(lat - LAT_ANCHOR)))
    prof = med[li]
    ok = np.isfinite(prof) & (prof > 0)
    interp = PchipInterpolator(alt_km[ok], prof[ok], extrapolate=False)
    a = interp(ALT / 1000.0)
    a = np.where(ALT / 1000.0 < alt_km[ok].min(), prof[ok][0], a)
    a = np.maximum(np.where(np.isfinite(a), a, 0.0) / 1000.0, 0.0)
    return np.where(ALT >= 16_000.0, a_ref, a)


def amplitude_factor(lat, alt_km, med):
    """g(phi, z) = median ext(phi, z) / median ext(22.5N, z); NaN (no data)
    filled by the nearest valid level in altitude, then 1."""
    li = int(np.argmin(np.abs(lat - LAT_ANCHOR)))
    ref = med[li]
    g = med / ref[None, :]
    g = np.where(np.isfinite(g) & (g > 0), g, np.nan)
    for i in range(g.shape[0]):
        row = g[i]
        ok = np.isfinite(row)
        if ok.any():
            g[i] = np.interp(alt_km, alt_km[ok], row[ok])   # holds at the ends
        else:
            g[i] = 1.0
    return g


# ---------------------------------------------------------------------------
# Noise-floor altitude models
# ---------------------------------------------------------------------------
def vis_floor_inflation(z_km):
    """SNR-floor inflation exp(+d tau_Rayleigh,slant / 2) vs 20 km, per
    triplet channel: (n_z, 3)."""
    z_grid = np.arange(0.0, 80_001.0, 500.0)
    prof = us_standard_atmosphere(z_grid)
    n_air = np.asarray(prof.number_density_m3)
    alpha = n_air[:, None] * rayleigh_cross_section_m2(VIS_NM * 1e-9)[None, :]
    paths = tangent_to_slant_paths(np.asarray(z_km) * 1e3, z_grid)
    tau = paths @ alpha
    tau20 = tangent_to_slant_paths(np.array([20_000.0]), z_grid) @ alpha
    return np.exp(0.5 * (tau - tau20))


def mir_floor_scale(z_km):
    """Per-model multiplicative factor on the 20-km element floors."""
    out = {"held": np.ones_like(z_km)}
    scan = np.loadtxt(FLOOR_SCAN, delimiter=",", skiprows=1)
    s = np.interp(z_km, scan[:, 0], scan[:, 2])
    out["oe"] = s / np.interp(20.0, scan[:, 0], scan[:, 2])
    atlas = json.load(open(ATLAS))

    def sd874(bin_key):
        rows = [r for r in atlas[bin_key] if r.get("sd") is not None]
        # round 43: the 8.80-um reference element (exact atlas row; 8.74 before)
        exact = [r for r in rows if abs(r["center_um"] - 8.8) < 1e-6]
        if not exact:
            raise RuntimeError(f"no 8.8-um row in atlas bin {bin_key}: re-run the ACE floor pipeline first")
        return exact[0]["sd"]
    ratio = np.array([sd874("16_19"), sd874("19_22"), sd874("22_25")]) / sd874("19_22")
    # piecewise constant over the measured bins (16-19 / 19-22 / 22-25 km),
    # held beyond the outer bins; 20 km therefore keeps the design floors
    out["atlas"] = np.where(z_km < 19.0, ratio[0], np.where(z_km < 22.0, ratio[1], ratio[2]))
    out["_atlas_ratio"] = dict(zip(["16_19", "19_22", "22_25"], ratio.tolist()))
    return out


# ---------------------------------------------------------------------------
# Threshold engine
# ---------------------------------------------------------------------------
class Engine:
    """One element set on one background bracket: profiles once (cached on
    disk under outputs/, ~15 min per build), thresholds at any (altitude
    index, amplitude factor, floor vector)."""

    def __init__(self, es: ElementSet, comps, member, alpha525, t20, cache: Path | None = None):
        self.es, self.t20 = es, t20
        if cache is not None and cache.exists():
            d = np.load(cache)
            self.exts = [d["ext0"], d["ext1"]]
            self.ders = [(d["er0"], d["es0"]), (d["er1"], d["es1"])]
            return
        ri, wt, T = member_ri(member)
        bg = TwoComponentBackground(alpha525, comps, ri, wt, T)
        cols = bg.columns()
        self.exts = [es.prof(c) for c in cols]
        self.ders = []
        for i, (r0, s0, f) in enumerate(comps):
            dr, ds = bg.fd * r0, bg.fd * s0
            er = (es.prof(bg.column(i, dr=+dr)) - es.prof(bg.column(i, dr=-dr))) / (2 * dr)
            es_ = (es.prof(bg.column(i, ds=+ds)) - es.prof(bg.column(i, ds=-ds))) / (2 * ds)
            self.ders.append((er, es_))
        if cache is not None:
            np.savez(cache, ext0=self.exts[0], ext1=self.exts[1], er0=self.ders[0][0],
                     es0=self.ders[0][1], er1=self.ders[1][0], es1=self.ders[1][1])

    def sigma(self, k, g=1.0, mir_scale=1.0, vis_infl=1.0):
        s_tot = sum(e[k] for e in self.exts)
        cols = [self.exts[0][k], self.exts[1][k], self.ders[0][0][k], self.ders[0][1][k],
                self.ders[1][0][k], self.ders[1][1][k]]
        sig = np.r_[np.maximum(VIS_PHI * g * s_tot[:N_VIS], ALPHA_FLOOR * vis_infl),
                    mir_scale * self.es.floors]
        sa, _, _ = marginal(np.arange(self.es.n), cols, sig, self.t20)
        return sa


def compute():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    thr = json.load(open(THR_JSON))
    sets = {"band": ElementSet("band 8-13 @0.1", thr["band01_elements_um"], 0.10,
                               thr["band01_floors_m1"]),
            "window": ElementSet("window 7.8-9.3 @0.1", thr["window_elements_um"], 0.10,
                                 thr["window_floors_m1"])}
    cq = thr["results"]["CALIBRATED quiet (LM65T223, 2-comp)"]
    ref = {"band": cq["band 8-13 @0.1: measured floors"]["full 7-param"]["triplet + band01"]["mmin_tg"],
           "window": cq["window 7.8-9.3 @0.1: measured floors"]["full 7-param"]["triplet + window"]["mmin_tg"]}

    lat_g, alt_g, med = glossac_quiet_median()
    alpha525 = anchor_profile_full(lat_g, alt_g, med)
    g_grid = amplitude_factor(lat_g, alt_g, med)                 # (lat_g, alt_g)

    ri_kp = SilicaRefractiveIndex(str(_ROOT / "data/optics/SiO2_KitamuraPopova.yml"))
    sai = create_silica_sai_layer(ALT, 1.0, refractive_index=ri_kp,
                                  rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL)

    vis_infl = vis_floor_inflation(Z_KM)                          # (n_z, 3)
    fscale = mir_floor_scale(Z_KM)
    k20 = K_OF_Z(20.0)

    engines = {}
    for sname, es in sets.items():
        t20 = es.prof(sai)[iz]
        for bname, (comps, member) in BRACKETS.items():
            print(f"building {sname} / {bname} ...", flush=True)
            engines[(sname, bname)] = Engine(es, comps, member, alpha525, t20,
                                             cache=OUT_DIR / f"engine_{sname}_{bname}.npz")
    # sanity: the archived 20-km design numbers
    for sname in sets:
        m = 3 * engines[(sname, "quiet")].sigma(k20)
        print(f"  {sname}: 20 km, g=1, held -> {m:.4f} Tg (archived {ref[sname]:.4f})")
        assert abs(m - ref[sname]) < 1e-3, "baseline reproduction"

    res = {"z_km": Z_KM.tolist(), "atlas_ratio_874": fscale["_atlas_ratio"],
           "floor_scale": {m: fscale[m].tolist() for m in FLOOR_MODELS},
           "vis_floor_inflation": vis_infl.tolist(), "ref_20km": ref}

    # -- 1. amplitude dependence at 20 km (replaces the exact cancellation) --
    res["amplitude_scan"] = {}
    for (sname, bname), eng in engines.items():
        m1 = 3 * eng.sigma(k20)
        res["amplitude_scan"][f"{sname}/{bname}"] = {
            str(g): 3 * eng.sigma(k20, g=g) / m1 for g in G_SCAN}
    print("amplitude scan M(g)/M(1):", json.dumps(res["amplitude_scan"], indent=None)[:400])

    # -- 2. sigma(phi, z) on the GloSSAC latitude grid, per model -----------
    # thresholds in fixed 20-km-reference units [Tg]; (n_lat_g, n_z)
    sig_map = {}
    for (sname, bname), eng in engines.items():
        for fm in FLOOR_MODELS:
            arr = np.full((len(lat_g), len(Z_KM)), np.nan)
            for j, z in enumerate(Z_KM):
                k = K_OF_Z(z)
                for i in range(len(lat_g)):
                    g = float(np.interp(z, alt_g, g_grid[i]))
                    arr[i, j] = eng.sigma(k, g=g, mir_scale=fscale[fm][j],
                                          vis_infl=vis_infl[j])
            sig_map[(sname, bname, fm)] = arr
            print(f"  sigma map {sname}/{bname}/{fm}: 20 km at anchor "
                  f"{3*arr[int(np.argmin(np.abs(lat_g-LAT_ANCHOR))), int(np.argmin(np.abs(Z_KM-20)))]:.4f} Tg", flush=True)
    j20 = int(np.argmin(np.abs(Z_KM - 20.0)))
    ia = int(np.argmin(np.abs(lat_g - LAT_ANCHOR)))

    # altitude behaviour at the anchor latitude (homogeneous 20-km units)
    res["altitude_anchor"] = {}
    for (sname, bname, fm), arr in sig_map.items():
        res["altitude_anchor"][f"{sname}/{bname}/{fm}"] = {
            str(z): 3 * arr[ia, int(np.argmin(np.abs(Z_KM - z)))] for z in (12, 14, 16, 18, 20, 22, 24, 26, 28)}
    # latitude behaviour at 20 km (homogeneous layer, band/quiet/held)
    res["lat_20km_homogeneous"] = {f"{sname}/{bname}": {
        "lat": lat_g.tolist(), "mmin": (3 * sig_map[(sname, bname, "held")][:, j20]).tolist(),
        "g": [float(np.interp(20.0, alt_g, g_grid[i])) for i in range(len(lat_g))]}
        for sname in sets for bname in BRACKETS}

    # -- 3. scenarios at 20 km: per-latitude thresholds and campaign gains --
    scen = {n: load_silica_zonal_2d(inj_lat_deg=il, inj_pres_hpa=INJ_PRES_HPA)
            for n, il in SCENARIOS.items()}
    events = {o: simulate_occultation_latitudes(mission_days=365.0, **kw) for o, kw in ORBITS.items()}
    res["orbits"] = {o: dict(events_per_year=int(len(ev.day)), per_day=len(ev.day) / 365.0,
                             lat_min=float(ev.lat_deg.min()), lat_max=float(ev.lat_deg.max()),
                             median_footprint_deg=float(np.median(ev.footprint_half_deg)))
                     for o, ev in events.items()}
    res["scenarios"] = {}
    for n, sz in scen.items():
        f_eff = smooth_over_footprint(sz.lat_deg, sz.enhancement, FOOT_DEG)
        r = dict(inj_lat=sz.inj_lat_deg, f_peak=float(np.nanmax(f_eff)),
                 f_min=float(np.nanmin(f_eff)), lat_f_peak=float(sz.lat_deg[np.nanargmax(f_eff)]),
                 lifetime_yr=sz.lifetime_yr, steady_mass_tg=sz.total_mass_kg / 1e9)
        cols = {"lat_deg": sz.lat_deg, "f_enhancement": f_eff}
        for (sname, bname) in engines:
            s20 = np.interp(sz.lat_deg, lat_g, sig_map[(sname, bname, "held")][:, j20])
            m_lat = 3 * s20 / f_eff
            cols[f"Mmin3_{sname}_{bname}_Tg"] = m_lat
            cols[f"Mmin3_{sname}_{bname}_homogeneous_Tg"] = 3 * s20
            r[f"{sname}/{bname}"] = dict(
                best_lat=float(sz.lat_deg[np.nanargmin(m_lat)]), best_mmin=float(np.nanmin(m_lat)),
                homogeneous_at_best_lat=float(3 * s20[np.nanargmin(m_lat)]),
                homogeneous_anchor=float(3 * sig_map[(sname, bname, "held")][ia, j20]),
                worst_mmin=float(np.nanmax(m_lat)))
            for o, ev in events.items():
                f_e = np.interp(ev.lat_deg, sz.lat_deg, f_eff)
                s_e = np.interp(ev.lat_deg, lat_g, sig_map[(sname, bname, "held")][:, j20])
                gain = float(np.sqrt(np.sum(f_e**2 / s_e**2) / np.sum(1.0 / s_e**2)))
                gain_f_only = float(np.sqrt(np.mean(f_e**2)))
                r[f"{sname}/{bname}"][f"gain_{o}"] = gain
                r[f"{sname}/{bname}"][f"gain_f_only_{o}"] = gain_f_only
                r[f"{sname}/{bname}"][f"campaign_sigma_{o}_Tg"] = float(
                    1.0 / np.sqrt(np.sum(f_e**2 / s_e**2)))
                r[f"{sname}/{bname}"][f"best_event_mmin_{o}"] = float(np.min(3 * s_e / f_e))
        res["scenarios"][n] = r
        hdr = ",".join(cols)
        np.savetxt(OUT_DIR / f"mmin_lat_{n}.csv", np.column_stack(list(cols.values())),
                   delimiter=",", header=hdr, comments="")
        print(f"scenario {n}: f_peak {r['f_peak']:.2f}; band/quiet best {r['band/quiet']['best_mmin']:.4f} Tg "
              f"at {r['band/quiet']['best_lat']:+.0f}; gains " +
              ", ".join(f"{o} {r['band/quiet'][f'gain_{o}']:.2f}" for o in ORBITS), flush=True)

    # -- 4. injection-latitude scan (14 scenarios, band/quiet/held) ----------
    import netCDF4
    ds = netCDF4.Dataset(str(_ROOT / "data/models/Silica_05_no_coagulation_with2D.nc"))
    inj_lats = ds["inj_lat"][:].filled(np.nan)
    ds.close()
    res["injection_scan"] = []
    s20_band = sig_map[("band", "quiet", "held")][:, j20]
    for il in inj_lats:
        szi = load_silica_zonal_2d(inj_lat_deg=float(il), inj_pres_hpa=INJ_PRES_HPA)
        fi = smooth_over_footprint(szi.lat_deg, szi.enhancement, FOOT_DEG)
        s_l = np.interp(szi.lat_deg, lat_g, s20_band)
        m_l = 3 * s_l / fi
        row = dict(inj_lat=float(il), f_peak=float(np.nanmax(fi)), best_mmin=float(np.nanmin(m_l)))
        for o, ev in events.items():
            f_e = np.interp(ev.lat_deg, szi.lat_deg, fi)
            s_e = np.interp(ev.lat_deg, lat_g, s20_band)
            row[f"gain_{o}"] = float(np.sqrt(np.sum(f_e**2 / s_e**2) / np.sum(1.0 / s_e**2)))
        res["injection_scan"].append(row)

    # -- 5. altitude-resolved maps for the +-41 deg scenarios ---------------
    res["maps"] = {}
    for tag, (ils, pres) in MAP_SCEN.items():
        sz = load_silica_zonal_2d(inj_lat_deg=ils, inj_pres_hpa=pres)
        f2d = sz.enhancement_2d                                     # (n_lat, n_lev)
        zlev = sz.z_lev_km
        in_rng = (zlev >= Z_KM[0]) & (zlev <= Z_KM[-1])
        ilev20 = int(np.argmin(np.abs(zlev - 20.0)))
        save = dict(lat_deg=sz.lat_deg, z_lev_km=zlev, f2d=f2d, conc_kg_m3=sz.conc_kg_m3,
                    total_mass_kg=sz.total_mass_kg)
        rmap = dict(inj_pres_hpa=sz.inj_pres_hpa, lifetime_yr=sz.lifetime_yr,
                    f2d_max=float(np.nanmax(f2d)))
        for (sname, bname, fm), arr in sig_map.items():
            # sigma on (model lat, model level)
            s_lat = np.array([np.interp(sz.lat_deg, lat_g, arr[:, j]) for j in range(len(Z_KM))]).T  # (n_lat, n_z)
            s_lev = np.array([np.interp(zlev, Z_KM, s_lat[i], left=np.nan, right=np.nan)
                              for i in range(len(sz.lat_deg))])                             # (n_lat, n_lev)
            with np.errstate(divide="ignore", invalid="ignore"):
                m = 3 * s_lev / f2d
            save[f"mmin_{sname}_{bname}_{fm}"] = m
            mm = np.where(np.isfinite(m) & in_rng[None, :], m, np.inf)
            m_best = mm.min(axis=1)
            z_best = zlev[np.argmin(mm, axis=1)]
            m20 = m[:, ilev20]
            ib = int(np.argmin(m_best))
            # 0.1-Tg coverage: latitudes with some bin < 0.1 Tg
            cov = np.isfinite(m_best) & (m_best < 0.1)
            rmap[f"{sname}/{bname}/{fm}"] = dict(
                best_mmin=float(m_best.min()), best_lat=float(sz.lat_deg[ib]), best_z=float(z_best[ib]),
                mmin_20km_at_best_lat=float(m20[ib]),
                best_20km_bin=float(np.nanmin(np.where(np.isfinite(m20), m20, np.inf))),
                frac_lat_below_0p1=float(np.mean(cov)),
                lat_range_below_0p1=[float(sz.lat_deg[cov].min()), float(sz.lat_deg[cov].max())] if cov.any() else None,
                worst_best_bin=float(np.max(m_best[np.isfinite(m_best)])),
                z_best_median=float(np.median(z_best[np.isfinite(m_best)])))
        # LOS obscuration for the headline model
        # per-model latitude statistics (round 35c)
        for (sname, bname, fm) in sig_map:
            mmat = save[f"mmin_{sname}_{bname}_{fm}"]
            mm = np.where(np.isfinite(mmat) & in_rng[None, :], mmat, np.inf)
            m_best = mm.min(axis=1); z_best = zlev[np.argmin(mm, axis=1)]
            m20 = mmat[:, ilev20]
            mid, pol = np.abs(sz.lat_deg) <= 70.0, np.abs(sz.lat_deg) > 70.0
            ok20 = np.isfinite(m20) & (m20 < 0.1)
            st = dict(worst_best_bin_le70=float(m_best[mid].max()),
                      worst_best_bin_polar=float(m_best[pol].max()),
                      lat_range_20km_below_0p1=[float(sz.lat_deg[ok20].min()), float(sz.lat_deg[ok20].max())] if ok20.any() else None,
                      frac_lat_20km_below_0p1=float(np.mean(ok20)),
                      z_best_median_by_band={f"{a}-{b}": float(np.median(z_best[(np.abs(sz.lat_deg) >= a) & (np.abs(sz.lat_deg) < b)]))
                                             for a, b in ((0, 20), (20, 40), (40, 55), (55, 70), (70, 90))})
            rmap[f"{sname}/{bname}/{fm}"].update(st)
        m = save[f"mmin_band_quiet_{BASELINE}"]
        zsel = (zlev >= 10.0) & (zlev <= 22.0)
        P = obscuration_probability(sz.lat_deg, zlev[zsel], events=events["ISS-like"])
        Pfull = np.zeros_like(m)
        Pfull[:, zsel] = np.nan_to_num(P)
        save["q_obsc"] = Pfull
        det = np.isfinite(m) & (m < CONTOURS_TG[-1])
        rmap["frac_detectable_area_pobs_gt_50"] = float(np.mean((Pfull > 0.5) & det) / max(np.mean(det), 1e-12))
        np.savez(OUT_DIR / f"map_{tag}.npz", **save)
        res["maps"][tag] = rmap
        h = rmap["band/quiet/held"]
        print(f"map {tag}: band/quiet/held best {h['best_mmin']:.4f} Tg at ({h['best_lat']:+.0f}, {h['best_z']:.1f} km); "
              f"0.1-Tg coverage {100*h['frac_lat_below_0p1']:.0f}% of latitudes; 20-km best {h['best_20km_bin']:.3f}", flush=True)

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"archived -> {OUT_DIR/'results.json'}")
    return res


# ---------------------------------------------------------------------------
# Figures (paper style, print width)
# ---------------------------------------------------------------------------
def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
        "axes.linewidth": 0.8, "lines.linewidth": 1.5})


def plot(res):
    """Round 35c (Doron): one figure for the combined +-41 deg / 51 hPa case ---
    (a) the field with detectable-mass contours (baseline OE-scaled floors),
    (b) per-latitude thresholds: 20-km bin, best bin (baseline), best bin with
    the measured-atlas floors, and the homogeneous-layer reference."""
    _style()
    d = np.load(OUT_DIR / "map_pm41.npz")
    lat, zlev = d["lat_deg"], d["z_lev_km"]
    conc = d["conc_kg_m3"] * 1e9 / (d["total_mass_kg"] / 1e9)
    in_rng = (zlev >= Z_KM[0]) & (zlev <= Z_KM[-1])
    ilev20 = int(np.argmin(np.abs(zlev - 20.0)))

    def best(mmat):
        mm = np.where(np.isfinite(mmat) & in_rng[None, :], mmat, np.inf)
        return mm.min(axis=1), zlev[np.argmin(mm, axis=1)]

    m_base = d[f"mmin_band_quiet_{BASELINE}"]
    mb_base, zb_base = best(m_base)
    mb_atlas, _ = best(d["mmin_band_quiet_atlas"])
    mb_win, _ = best(d[f"mmin_window_quiet_{BASELINE}"])
    m20 = m_base[:, ilev20]
    hom = res["ref_20km"]["band"]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15 / 2.54, 13.5 / 2.54),
                                  gridspec_kw=dict(height_ratios=[1.35, 1.0]), sharex=True)
    pc = ax.pcolormesh(lat, zlev, conc.T, cmap="Greys", shading="nearest")
    cb = fig.colorbar(pc, ax=ax, pad=0.015, fraction=0.04)
    cb.set_label("silica [$\\mu$g m$^{-3}$ per Tg]", fontsize=9)
    cs = ax.contour(lat, zlev, m_base.T, levels=CONTOURS_TG, colors="red", linewidths=1.2)
    ax.clabel(cs, fmt=lambda v: f"{v:g} Tg", fontsize=8)
    plt.rcParams["hatch.color"] = "0.35"
    Q = d["q_obsc"]
    ax.contourf(lat, zlev, Q.T, levels=[0.1, 0.5, 0.9, 1.0001], colors="none",
                hatches=["..", "//", "xx"])
    show = mb_base < 5.0
    ax.plot(np.where(show, lat, np.nan), zb_base, color="black", lw=1.2, ls="--",
            label="optimal tangent altitude")
    ax.axhline(20.0, color="black", lw=0.9, ls=":", label="20-km reference bin")
    ax.set_ylim(10, 30); ax.set_xlim(-80, 80)
    ax.set_ylabel("Altitude [km]")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([0], [0], color="red", lw=1.2),
                Patch(facecolor="none", edgecolor="0.35", hatch="//")]
    labels += ["$M_{\\min}^{(3\\sigma)}$ contours", "$q_\\mathrm{obsc}>50\\%$"]
    ax.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    ax.text(0.99, 0.97, "(a)", transform=ax.transAxes, ha="right", va="top")

    ax2.plot(lat, m20, color="red", ls=":", lw=1.3, label="20-km bin")
    ax2.plot(lat, mb_base, color="red", lw=1.6, label="best bin, baseline floors")
    ax2.plot(lat, mb_atlas, color="red", ls="--", lw=1.2, label="best bin, measured-atlas floors")
    ax2.plot(lat, mb_win, color="black", lw=1.0, label="best bin, window subset")
    ax2.axhline(hom, color="0.4", lw=0.8, ls="-.", label=f"homogeneous layer ({hom:.3f} Tg)")
    ax2.axhline(0.1, color="0.7", lw=0.6)
    ax2.set_yscale("log"); ax2.set_ylim(0.015, 3)
    ax2.set_xlabel("Latitude [$^\\circ$]"); ax2.set_ylabel("$M_{\\min}^{(3\\sigma)}$ [Tg]")
    ax2.grid(True, which="both", alpha=0.25, lw=0.5)
    ax2.legend(loc="upper center", fontsize=8, ncol=2, framealpha=0.9)
    ax2.text(0.99, 0.97, "(b)", transform=ax2.transAxes, ha="right", va="top")
    # keep the colorbar from offsetting panel (b): give it the same right margin
    fig.tight_layout(h_pad=0.5)
    pos_a, pos_b = ax.get_position(), ax2.get_position()
    ax2.set_position([pos_b.x0, pos_b.y0, pos_a.width, pos_b.height])
    fig.savefig(FIG_DIR / "detectability_2d_mass_contours.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {FIG_DIR / 'detectability_2d_mass_contours.png'}")


if __name__ == "__main__":
    if "--plot" in sys.argv[1:]:
        plot(json.load(open(OUT_DIR / "results.json")))
    else:
        plot(compute())
