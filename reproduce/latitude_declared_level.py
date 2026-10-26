"""
latitude_declared_level.py -- referee round 5, item 7 (Sect. 6, latitude-resolved
performance): level bookkeeping of the 2-D maps, transport-field provenance,
horizontal gradients along the limb path, finite vertical response, campaign
combination with a shared (common-mode) error, the operational decision rule,
and a quick persistence check of the ACE null sample.

Everything is read from the archived maps of detectability_2d_calibrated.py
(outputs/detectability_2d_calibrated/map_pm41.npz, map_pm41_o3lat.npz,
results.json, results_o3lat.json, mmin_lat_combined*.csv); nothing is
re-derived from the Mie / background engines.  The orbit event samples are
regenerated with the same function the map script calls
(saimon.sai_2d.simulate_occultation_latitudes, mission_days=365) and cached
in the archive.

Run from the repo root:
    python reproduce/latitude_declared_level.py [--recompute-orbits]
Archive: outputs/latitude_declared_level.json   (keys documented inline)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from saimon.geometry import tangent_to_slant_paths            # noqa: E402
from saimon.sai_2d import (R_EARTH_M, load_silica_zonal_2d,    # noqa: E402
                          simulate_occultation_latitudes, smooth_over_footprint)

D2D = _ROOT / "outputs" / "detectability_2d_calibrated"
OUT = _ROOT / "outputs" / "latitude_declared_level.json"
NC = _ROOT / "data" / "models" / "Silica_05_no_coagulation_with2D.nc"
FLOOR_JSON = _ROOT / "outputs" / "m2_vertical_correlation_floor.json"
NULL_CSV = _ROOT / "outputs" / "ace_v52" / "fullspectrum.csv"

Z_TABLE = 20.47          # the level Table 7 uses (paper1_latitude_scenarios_round67.py: Z20)
FOOT_DEG = 0.7           # the map script's LOS footprint half-width (FOOT_DEG)
ORBITS = {"ISS-like": dict(inclination_deg=51.6, altitude_km=420.0),
          "dedicated 70": dict(inclination_deg=70.0, altitude_km=650.0)}
SCENARIOS = (  # (row label of Table 7, map file, key) -- same order as the round-67 script
    ("reference-atmosphere floor", "map_pm41.npz", "mmin_band_quiet_oe"),
    ("measured altitude-dependent floor", "map_pm41.npz", "mmin_band_quiet_atlas"),
    ("measured-ozone rescaled floor", "map_pm41_o3lat.npz", "mmin_band_quiet_oe"),
    ("measured-ozone + measured floor", "map_pm41_o3lat.npz", "mmin_band_quiet_atlas"),
    ("window, reference-atmosphere floor", "map_pm41.npz", "mmin_window_quiet_oe"),
)
SIGMA_EMP_TG = 0.0259    # empirical sigma of the 101-occultation null (Sect. 7)
M_MIN_BAND = 0.101       # Tg, 3 sigma single-sounding band threshold
EVENTS_PER_DAY, BINS = 30.2, 33   # false_alarm_trials.py constants (copied, not imported)


def cov(lat, m):
    """0.1-Tg coverage of a per-latitude threshold: (lat_min, lat_max, fraction)."""
    ok = np.isfinite(m) & (m <= 0.1)
    if not ok.any():
        return None
    return dict(lat_min=float(lat[ok].min()), lat_max=float(lat[ok].max()), frac=float(ok.mean()))


def at(lat, arr, phi):
    return float(arr[int(np.argmin(np.abs(lat - phi)))])


def loglin_z(zlev, arr2d, z0):
    """log-linear-in-z interpolation of a positive (lat, z) field to z0 (zlev descending)."""
    k_hi = int(np.where(zlev >= z0)[0][-1]); k_lo = k_hi + 1       # zlev[k_hi] >= z0 > zlev[k_lo]
    w = (z0 - zlev[k_lo]) / (zlev[k_hi] - zlev[k_lo])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.exp((1 - w) * np.log(arr2d[:, k_lo]) + w * np.log(arr2d[:, k_hi]))


def level_block(lat, f, m):
    """f and M_min on one level: cores (+-41), poles (+-89), extremes, coverage."""
    jmin, jmax = int(np.nanargmin(m)), int(np.nanargmax(m))
    return dict(f_core_p41=at(lat, f, 41), f_core_m41=at(lat, f, -41), f_pole_p89=at(lat, f, 89),
                f_pole_m89=at(lat, f, -89), f_max=float(np.nanmax(f)), f_min=float(np.nanmin(f)),
                M_core_p41=at(lat, m, 41), M_core_m41=at(lat, m, -41), M_pole_p89=at(lat, m, 89),
                M_pole_m89=at(lat, m, -89), M_min=float(m[jmin]), M_min_lat=float(lat[jmin]),
                M_max=float(m[jmax]), M_max_lat=float(lat[jmax]), coverage_0p1=cov(lat, m))


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    recompute = "--recompute-orbits" in sys.argv
    old = json.load(open(OUT)) if OUT.exists() else {}
    res = {"meta": dict(script="reproduce/latitude_declared_level.py", date="2026-10-25",
                        inputs=[str(p.relative_to(_ROOT)) for p in
                                (D2D / "map_pm41.npz", D2D / "map_pm41_o3lat.npz", D2D / "results.json",
                                 D2D / "results_o3lat.json", D2D / "mmin_lat_combined.csv",
                                 D2D / "mmin_lat_combined_o3lat.csv", NC, FLOOR_JSON, NULL_CSV)])}

    maps = {fn: np.load(D2D / fn) for fn in ("map_pm41.npz", "map_pm41_o3lat.npz")}
    lat = maps["map_pm41.npz"]["lat_deg"]; zlev = maps["map_pm41.npz"]["z_lev_km"]
    f2d = maps["map_pm41.npz"]["f2d"]
    assert np.allclose(f2d, maps["map_pm41_o3lat.npz"]["f2d"])
    kT = int(np.argmin(np.abs(zlev - Z_TABLE)))          # 20.4656 km
    kB = kT + 1                                           # 19.5835 km (next level down)
    k20_results = int(np.argmin(np.abs(zlev - 20.0)))     # what results.json's ilev20 picks
    zT, zB = float(zlev[kT]), float(zlev[kB])
    res["levels"] = dict(table_level_km=zT, lower_level_km=zB, exact_km=20.0,
                         results_json_ilev20_km=float(zlev[k20_results]),
                         note="Table 7 (round-67 script, Z20=20.47) sits on the 20.4656-km model level; "
                              "results.json 'mmin_20km'/'lat_range_20km_below_0p1' use argmin|z-20| = the 19.5835-km level; "
                              "the campaign gains and the 2.8/0.35/0.036-Tg prose use the exact 20.0-km interpolation of "
                              "load_silica_zonal_2d (analysis_alt_km=20.0).")
    r_main = json.load(open(D2D / "results.json")); r_o3 = json.load(open(D2D / "results_o3lat.json"))

    # ---- 1. level bookkeeping ------------------------------------------------
    f_T, f_B, f_20 = f2d[:, kT], f2d[:, kB], loglin_z(zlev, f2d, 20.0)
    book = {}
    for label, fn, key in SCENARIOS:
        m = maps[fn][key]
        s3_T, s3_B = m[:, kT] * f_T, m[:, kB] * f_B                 # 3 sigma on the level (M = 3 sigma / f)
        with np.errstate(invalid="ignore"):
            s3_20 = loglin_z(zlev, m * f2d, 20.0)                    # 3 sigma interpolated log-linearly to 20.0 km
        book[label] = {"archive": f"{fn}:{key}",
                       f"{zT:.4f}_km": level_block(lat, f_T, m[:, kT]),
                       f"{zB:.4f}_km": level_block(lat, f_B, m[:, kB]),
                       "20.0_km_loglinear": level_block(lat, f_20, s3_20 / f_20),
                       f"{zlev[k20_results]:.4f}_km_results_json_ilev20": level_block(lat, f2d[:, k20_results], m[:, k20_results]),
                       "three_sigma_homogeneous_Tg": dict(table_level_at_41=at(lat, s3_T, 41), lower_level_at_41=at(lat, s3_B, 41),
                                                          exact20_at_41=at(lat, s3_20, 41))}
    sel = (zlev > 14.0) & (zlev < 27.0)
    book["f_vertical_profiles"] = dict(z_km=zlev[sel].tolist(),
                                       **{f"phi_{p:+d}": f2d[int(np.argmin(np.abs(lat - p))), sel].tolist() for p in (41, -41, 20, 1, 70, 89)},
                                       note="the +-41 deg core is a single-level spike at the 51-hPa injection level (20.4656 km): the level above "
                                            "(21.35 km) holds 7 % of it and the level below (19.58 km) 49 %")
    res["level_bookkeeping"] = book

    # the map script's own exact-20-km quantities (scenarios/combined; mmin_lat_combined*.csv)
    sz = load_silica_zonal_2d(nc_path=str(NC), inj_lat_deg=[-41.0, 41.0], inj_pres_hpa=51.0, analysis_alt_km=20.0)
    assert np.allclose(sz.enhancement_2d, f2d) and np.allclose(sz.lat_deg, lat)
    f_ex = sz.enhancement; f_ex_sm = smooth_over_footprint(lat, f_ex, FOOT_DEG)
    ex = {}
    for tag, fn, rj in (("reference", "mmin_lat_combined.csv", r_main), ("measured_ozone", "mmin_lat_combined_o3lat.csv", r_o3)):
        c = np.genfromtxt(D2D / fn, delimiter=",", names=True)
        assert np.allclose(c["f_enhancement"], f_ex_sm, atol=1e-9)
        mb, mw = c["Mmin3_band_quiet_Tg"], c["Mmin3_window_quiet_Tg"]
        sc = rj["scenarios"]["combined"]
        ex[tag] = dict(csv=fn, f_raw_core_p41=at(lat, f_ex, 41), f_raw_pole_p89=at(lat, f_ex, 89), f_raw_min=float(f_ex.min()),
                       f_smoothed_core_p41=at(lat, f_ex_sm, 41), f_smoothed_min=float(f_ex_sm.min()), f_smoothed_min_lat=float(lat[np.argmin(f_ex_sm)]),
                       smoothing_note="+-0.7 deg boxcar on the 2-deg model grid contains only the centre point: smoothing is a no-op",
                       f_smoothed_max=float(f_ex_sm.max()), results_f_peak=sc["f_peak"], results_f_min=sc["f_min"],
                       band=level_block(lat, f_ex_sm, mb), window=level_block(lat, f_ex_sm, mw),
                       results_band_best_mmin=sc["band/quiet"]["best_mmin"], results_band_worst_mmin=sc["band/quiet"]["worst_mmin"],
                       results_window_best_mmin=sc["window/quiet"]["best_mmin"],
                       homogeneous_3sigma_lat_range=[float(np.min(c["Mmin3_band_quiet_homogeneous_Tg"])),
                                                     float(np.max(c["Mmin3_band_quiet_homogeneous_Tg"]))])
    res["exact20_map_script"] = ex

    # mapping "number in the text -> level"
    B = book["reference-atmosphere floor"]; BO = book["measured-ozone rescaled floor"]; TL = f"{zT:.4f}_km"
    res["quoted_number_map"] = [
        dict(quoted="f = 3.8 at 20.5 km (cores)", level_km=zT, value=B[TL]["f_core_p41"], archive="map_pm41.npz:f2d[:,k(20.4656)] / results.json maps/pm41/f2d_max"),
        dict(quoted="f = 2.8 over the cores 'at the fixed 20-km level'", level_km=20.0, value=ex["reference"]["f_smoothed_core_p41"],
             archive="results.json scenarios/combined/f_peak (exact 20.0 km, +-0.7 deg boxcar)", on_table_level=B[TL]["f_core_p41"],
             on_table_level_smoothed=float(smooth_over_footprint(lat, f_T, FOOT_DEG)[np.argmin(np.abs(lat - 41))])),
        dict(quoted="f = 0.35 at the poles", level_km=20.0, value=ex["reference"]["f_smoothed_min"],
             archive="results.json scenarios/combined/f_min (exact 20.0 km)", on_table_level=B[TL]["f_min"], on_table_level_lat=float(lat[np.argmin(f_T)])),
        dict(quoted="0.036 Tg over the cores (= 0.101/2.8)", level_km=20.0, value=ex["reference"]["band"]["M_min"],
             archive="results.json scenarios/combined/band/quiet/best_mmin", on_table_level=B[TL]["M_core_p41"]),
        dict(quoted="0.050 Tg window over the cores", level_km=20.0, value=ex["reference"]["window"]["M_min"],
             archive="results.json scenarios/combined/window/quiet/best_mmin", on_table_level=book["window, reference-atmosphere floor"][TL]["M_core_p41"]),
        dict(quoted="0.43 Tg at the poles (Table 7)", level_km=zT, value=B[TL]["M_max"], value_lat=B[TL]["M_max_lat"],
             archive="latitude_scenarios_round67.json fixed20_max = map_pm41.npz:mmin_band_quiet_oe[:,k(20.4656)]",
             on_exact20=ex["reference"]["band"]["M_max"], f_on_table_level_at_that_lat=B[TL]["f_min"]),
        dict(quoted="0.026 Tg best bin (+-41, 20.5 km)", level_km=zT, value=B[TL]["M_min"], archive="results.json maps/pm41/band/quiet/oe/best_mmin"),
        dict(quoted="63S-63N coverage of the fixed 20-km bin (Table 7)", level_km=zT, value=B[TL]["coverage_0p1"],
             archive="latitude_scenarios_round67.json fixed20_cov", on_exact20=ex["reference"]["band"]["coverage_0p1"],
             on_results_json_ilev20=B[f"{zlev[k20_results]:.4f}_km_results_json_ilev20"]["coverage_0p1"]),
        dict(quoted="0.025 Tg core under measured ozone", level_km=zT, value=BO[TL]["M_min"], archive="latitude_scenarios_round67.json row 3 fixed20_min",
             on_exact20=ex["measured_ozone"]["band"]["M_min"]),
        dict(quoted="69S-69N under measured ozone (Table 7)", level_km=zT, value=BO[TL]["coverage_0p1"], on_exact20=ex["measured_ozone"]["band"]["coverage_0p1"]),
        dict(quoted="campaign gains 1.89 / 1.56 (1.8 / 1.4 measured ozone)", level_km=20.0,
             value=[r_main["scenarios"]["combined"]["band/quiet"]["gain_ISS-like"], r_main["scenarios"]["combined"]["band/quiet"]["gain_dedicated 70"],
                    r_o3["scenarios"]["combined"]["band/quiet"]["gain_ISS-like"], r_o3["scenarios"]["combined"]["band/quiet"]["gain_dedicated 70"]],
             archive="results[_o3lat].json scenarios/combined/band/quiet/gain_*  (exact 20.0 km, f smoothed +-0.7 deg, sigma held at 20 km)"),
        dict(quoted="best-height numbers (0.086, 0.21, 0.082, 0.135 Tg worst best bin; 'all latitudes'; 69S-73N; 77S-87N)", level_km="min over levels 12-28 km",
             archive="latitude_scenarios_round67.json best_worst / best_cov", note="level-independent by construction"),
        dict(quoted="window 0.037 Tg at the cores; 0.1 Tg over 78 % of latitudes", level_km="best height (20.4656 at the cores)",
             archive="results.json maps/pm41/window/quiet/oe/best_mmin, frac_lat_below_0p1"),
        dict(quoted="Fig. 6a dotted '20-km reference bin' / Fig. 6b dotted '20-km bin'", level_km="see facts file (plot() of detectability_2d_calibrated.py)",
             archive="detectability_2d_calibrated.py plot()"),
    ]

    # ---- 2. transport-field provenance ---------------------------------------
    import netCDF4
    ds = netCDF4.Dataset(str(NC))
    inj_lats = ds["inj_lat"][:].filled(np.nan); inj_pres = ds["inj_pres"][:].filled(np.nan)
    ip = int(np.argmin(np.abs(inj_pres - 51.0))); ils = [int(np.argmin(np.abs(inj_lats - v))) for v in (-41.0, 41.0)]
    attrs = {a: str(ds.getncattr(a)) for a in ds.ncattrs()}
    lifetimes = {str(float(inj_lats[il])): float(ds["lifetime"][ip, il]) for il in ils}
    cols = {str(float(inj_lats[il])): float(np.nansum(ds["column_density"][ip, il, 0, :].filled(np.nan) * sz.area_m2)) for il in ils}
    lev = ds["lev"][:].filled(np.nan)
    ds.close()
    inj_rate = float(attrs["injection_rate"].split()[0])
    from saimon.atmosphere import us_standard_atmosphere
    zf = np.arange(0.0, 60_000.0, 10.0); prof = us_standard_atmosphere(zf)
    p_hpa = np.asarray(prof.pressure_pa) / 100.0
    z30 = float(np.interp(np.log(30.0), np.log(p_hpa[::-1]), zf[::-1])) / 1000.0
    p20 = float(np.interp(20_000.0, zf, p_hpa))
    res["transport_field"] = dict(
        file=str(NC.relative_to(_ROOT)), global_attributes=attrs,
        variables=dict(tracer_density="kg kg-1 aerosol mass mixing ratio (inj_pres, inj_lat, cores, lat, lev), yearly mean over last 12 months",
                       column_density="kg m-2 (inj_pres, inj_lat, cores, lat)", lifetime="yr (inj_pres, inj_lat)",
                       lat="90 model latitudes, -89..89 deg, 2-deg spacing", lev="50 model pressure levels, 1.07..935 hPa (top->bottom)"),
        inj_lat_grid=inj_lats.tolist(), inj_pres_grid=inj_pres.tolist(), lev_hpa=lev.tolist(), z_lev_km=zlev.tolist(),
        selection=dict(inj_pres_hpa=float(inj_pres[ip]), inj_lats_deg=[float(inj_lats[i]) for i in ils], cores_index=0,
                       combination="sum of the two single-latitude fields (linear transport), equal injection rates"),
        injection_rate_kg_per_yr_each=inj_rate, lifetime_yr_each=lifetimes, lifetime_yr_mean=sz.lifetime_yr,
        steady_mass_kg_each=cols, steady_mass_kg_total=sz.total_mass_kg, steady_mass_tg_total=sz.total_mass_kg / 1e9,
        check_mass_over_rate_yr=sz.total_mass_kg / (2 * inj_rate),
        altitude_assignment="z(lev) by linear-in-ln(p) interpolation of the US Standard Atmosphere (saimon.atmosphere.us_standard_atmosphere on a 10-m grid); "
                            "conc = q * rho_air(z_lev); column = trapz(q, p)/g; total = sum(column * band_area)",
        normalisation=dict(formula="f(phi,z) = [conc(phi,z)/M_tot] / [rho_air(20 km) / (A_earth * int_{16 km}^{z(30 hPa)} rho_air dz)]",
                           meaning="f is the local concentration per unit resident mass divided by that of a globally uniform constant-mixing-ratio "
                                   "layer between 16 km and the 30-hPa level evaluated at 20 km; 1 Tg is set by dividing by M_tot, i.e. the "
                                   "steady-state field is scaled linearly to the quoted resident mass (the model's own steady mass is the "
                                   "lifetime times the injection rate)",
                           layer_bottom_km=16.0, layer_top_hpa=30.0, layer_top_km=z30, p_at_20km_hpa=p20,
                           reference_altitude_km=20.0, f2d_reference_fixed_at_20km=True),
        interpolation=dict(exact_20km="q interpolated linearly in ln(p) to p(20 km) and multiplied by rho_air(20 km) (load_silica_zonal_2d: conc_a); "
                                      "this is what sz.enhancement / results.json scenarios/* use",
                           map_levels="no vertical interpolation: f2d is on the 50 model levels; sigma(phi,z) is interpolated linearly in z from "
                                      "the 0.5-km tangent grid (12-28 km) and linearly in latitude from the GloSSAC 5-deg grid (-77.5..77.5, held beyond)",
                           campaign="f at exact 20 km boxcar-smoothed over +-0.7 deg (smooth_over_footprint) then np.interp to event latitudes"))

    # ---- 3. horizontal gradients along the limb path --------------------------
    zs = zlev[::-1]; fz = f2d[:, ::-1]                       # ascending z
    interp = RegularGridInterpolator((lat, zs), fz, method="linear", bounds_error=False, fill_value=0.0)

    def fold(phi):
        phi = np.where(phi > 90.0, 180.0 - phi, phi); phi = np.where(phi < -90.0, -180.0 - phi, phi)
        return np.clip(phi, lat.min(), lat.max())

    def ray_shell_integrals(phi0, z_t, cos_az, dx_km=0.25, z_max=45.0):
        """Per-shell path integrals of f (km) along the ray tangent at z_t, both branches."""
        R = R_EARTH_M / 1e3; r_t = R + z_t
        X = np.sqrt((R + z_max) ** 2 - r_t ** 2)
        x = np.arange(dx_km / 2, X, dx_km); h = np.sqrt(r_t ** 2 + x ** 2) - R
        dphi = np.rad2deg(x / R) * cos_az
        fp = interp(np.column_stack([fold(phi0 + dphi), h])); fm = interp(np.column_stack([fold(phi0 - dphi), h]))
        # shell boundaries as tangent_to_slant_paths builds them (midpoints of the level centres)
        mid = 0.5 * (zs[:-1] + zs[1:]); bnd = np.concatenate([[zs[0] - 0.5 * (zs[1] - zs[0])], mid, [zs[-1] + 0.5 * (zs[-1] - zs[-2])]])
        j = np.clip(np.searchsorted(bnd, h, side="right") - 1, 0, len(zs) - 1)
        I = np.bincount(j, weights=(fp + fm) * dx_km, minlength=len(zs))
        return I

    L = tangent_to_slant_paths(zs * 1000.0, zs * 1000.0) / 1000.0   # (n_tangent, n_shell) km, tangent at every level centre
    jT = int(np.argmin(np.abs(zs - zT)))
    tan_idx = [j for j in range(jT, len(zs)) if zs[j] <= 45.0]

    def peel(phi0, cos_az):
        """Onion peel in f units: first-order (tangent-shell perturbation only) and full recursive."""
        I_az = {j: ray_shell_integrals(phi0, zs[j], cos_az) for j in tan_idx}
        I_zo = {j: ray_shell_integrals(phi0, zs[j], 0.0) for j in tan_idx}
        f_loc = float(interp([[phi0, zT]])[0])
        first = f_loc + (I_az[jT].sum() - I_zo[jT].sum()) / L[jT, jT]
        fhat = {}
        for j in reversed(tan_idx):
            fhat[j] = {}
            for tag, I in (("az", I_az), ("zo", I_zo)):
                upper = sum(L[j, k] * fhat[k][tag] for k in tan_idx if k > j)
                fhat[j][tag] = (I[j].sum() - upper) / L[j, j]
        full = f_loc + fhat[jT]["az"] - fhat[jT]["zo"]
        return f_loc, first, full, fhat[jT]["zo"]

    az_cases = {"meridional": 1.0, "azimuth_45deg": np.cos(np.deg2rad(45.0)), "zonal": 0.0}
    lp = dict(method="ray tangent at the 20.4656-km level centre, geometric (no refraction) straight line, sampled every 0.25 km up to 45 km altitude; "
                     "latitude along the ray phi0 +- (x/R) cos(azimuth), folded over the pole; f bilinear in (lat, z) from the archived f2d; shells = "
                     "the model levels with boundaries at the mid-points (saimon.geometry.tangent_to_slant_paths); "
                     "first-order: f_eff = f(phi0, z_t) + [tau_f(az) - tau_f(zonal)] / L_tt, i.e. the per-shell f perturbation enters with the shell's chord "
                     "weight relative to the tangent-shell chord (the retrieved tangent-shell extinction is the slant OD minus the upper shells, with the "
                     "upper shells at their homogeneous value); full: the recursive onion peel run on the gradient and on the zonal slant integrals, "
                     "difference added to f(phi0, z_t)",
              tangent_level_km=zT, tangent_shell_chord_km=float(L[jT, jT]),
              chord_half_length_km_above_tangent={f"h={h}": float(np.sqrt(2 * R_EARTH_M / 1e3 * h)) for h in (1, 2, 3, 5)},
              shell_weights_L_j_over_L_t=[dict(z_km=float(zs[j]), w=float(L[jT, j] / L[jT, jT])) for j in tan_idx[:8]])
    per_lat = {}
    fx = {}
    for name, ca in az_cases.items():
        rows = []
        for phi0 in lat:
            f_loc, first, full, zo = peel(float(phi0), ca)
            rows.append((f_loc, first, full, zo))
        rows = np.array(rows)
        fx[name] = rows
        per_lat[name] = dict(f_local=rows[:, 0].tolist(), f_eff_first_order=rows[:, 1].tolist(), f_eff_full_peel=rows[:, 2].tolist())
    lp["per_latitude"] = dict(lat_deg=lat.tolist(), **per_lat)
    m_ref = maps["map_pm41.npz"]["mmin_band_quiet_oe"][:, kT]
    summ = {}
    for name in ("meridional", "azimuth_45deg"):
        rows = fx[name]; rat1 = rows[:, 1] / rows[:, 0]; rat2 = rows[:, 2] / rows[:, 0]
        d = {}
        for phi in (41, -41, 20, -20, 1, -1, 70, -70):
            i = int(np.argmin(np.abs(lat - phi)))
            d[f"phi_{phi:+d}"] = dict(f=float(rows[i, 0]), ratio_first_order=float(rat1[i]), ratio_full_peel=float(rat2[i]))
        d["core_threshold_change_percent_first_order"] = float((1 / rat1[np.argmin(np.abs(lat - 41))] - 1) * 100)
        d["core_threshold_change_percent_full_peel"] = float((1 / rat2[np.argmin(np.abs(lat - 41))] - 1) * 100)
        d["core_threshold_Tg_first_order"] = float(m_ref[np.argmin(np.abs(lat - 41))] / rat1[np.argmin(np.abs(lat - 41))])
        d["ratio_first_order_minmax"] = [float(rat1.min()), float(rat1.max())]
        d["ratio_full_peel_minmax"] = [float(rat2.min()), float(rat2.max())]
        d["coverage_0p1_with_gradients_first_order"] = cov(lat, m_ref / rat1)
        d["coverage_0p1_with_gradients_full_peel"] = cov(lat, m_ref / rat2)
        d["coverage_0p1_reference"] = cov(lat, m_ref)
        summ[name] = d
    zo = fx["zonal"]
    summ["zonal_check"] = dict(first_order_equals_local=bool(np.allclose(zo[:, 1], zo[:, 0])), full_equals_local=bool(np.allclose(zo[:, 2], zo[:, 0])),
                               peel_of_zonal_slant_over_local_at_41=float(zo[np.argmin(np.abs(lat - 41)), 3] / zo[np.argmin(np.abs(lat - 41)), 0]),
                               note="the zonal-ray peel itself differs from the point value only through the within-shell vertical variation "
                                    "(a finite-vertical-response effect, see vertical_response), which is removed by differencing")
    lp["summary"] = summ
    res["limb_path"] = lp

    # ---- 4. finite vertical response -----------------------------------------
    zfine = np.arange(zT - 1.5, zT + 1.5001, 0.005)
    f_fine = np.array([np.interp(zfine, zs, fz[i]) for i in range(len(lat))])     # linear in z
    vr = dict(method="triangular response of full width W centred on the 20.4656-km level, f linear in z between model levels", center_km=zT, cases={})
    for W in (1.0, 2.0):
        w = np.clip(1 - np.abs(zfine - zT) / (W / 2), 0, None); w /= w.sum()
        f_v = f_fine @ w; rat = f_v / f_T
        c41 = int(np.argmin(np.abs(lat - 41)))
        vr["cases"][f"W_{W:.0f}km"] = dict(ratio_core_p41=float(rat[c41]), ratio_core_m41=float(rat[np.argmin(np.abs(lat + 41))]),
                                          ratio_pole_p89=float(rat[-1]), ratio_minmax=[float(rat.min()), float(rat.max())],
                                          core_threshold_change_percent=float((1 / rat[c41] - 1) * 100),
                                          core_threshold_Tg=float(m_ref[c41] / rat[c41]),
                                          coverage_0p1=cov(lat, m_ref / rat), ratio_per_lat=rat.tolist())
    vr["level_spacing_km"] = float(zT - zB)
    res["vertical_response"] = vr

    # ---- 5. campaign combination with a shared error --------------------------
    ev = {}
    if not recompute and "campaign" in old and "events" in old["campaign"]:
        for o in ORBITS:
            ev[o] = np.array(old["campaign"]["events"][o]["lat_deg"])
        print("orbit events: cached", {o: len(v) for o, v in ev.items()})
    else:
        for o, kw in ORBITS.items():
            e = simulate_occultation_latitudes(mission_days=365.0, **kw); ev[o] = e.lat_deg
            print(f"orbit {o}: {len(e.day)} events/yr  ({time.time() - t0:.0f} s)")
    for o in ORBITS:
        assert len(ev[o]) == r_main["orbits"][o]["events_per_year"], (o, len(ev[o]))
    sig_ref = r_main["ref_20km"]["band"] / 3.0
    floor = json.load(open(FLOOR_JSON))
    rho_floor = floor["conversion_m1"]["coherent_line"] / floor["conversion_m1"]["white_total"]
    camp = dict(events={o: dict(n_per_year=int(len(v)), per_day=float(len(v) / 365.0), lat_min=float(v.min()), lat_max=float(v.max()),
                                lat_deg=np.round(v, 4).tolist()) for o, v in ev.items()},
                sigma_ref_Tg=sig_ref, rho_c_floor=rho_floor, rho_c_floor_source="m2_vertical_correlation_floor.json conversion_m1 coherent_line / white_total",
                formula="sigma_camp = (f^T C^-1 f)^(-1/2), C = diag(sigma_e^2) + sigma_c^2 1 1^T, sigma_c = rho_c * sigma_ref; Sherman-Morrison: "
                        "f^T C^-1 f = S_ff - sigma_c^2 S_f1^2 / (1 + sigma_c^2 S_11), S_ab = sum a_e b_e / sigma_e^2; the homogeneous reference "
                        "(f = 1) is treated identically; gain = sigma_camp(f=1) / sigma_camp(f).  NOTE: with a strictly common additive "
                        "component the GLS estimator fits the offset out through the latitude pattern (f varies 0.23-3.8 across events), so the "
                        "homogeneous reference floors at sigma_c while the structured case does not; the 'eq6_weights_true_cov' entries give the "
                        "Eq.-(6) independent-weight estimator's actual uncertainty under the same covariance, which does not exploit the pattern "
                        "against the offset")

    def gains(f_lat, s3_lat_grid, lat_grid, tag, s3_lat_is_model_grid=False):
        out = {}
        f_sm = smooth_over_footprint(lat, f_lat, FOOT_DEG)
        for o, phi_e in ev.items():
            f_e = np.interp(phi_e, lat, f_sm)
            s_e = np.interp(phi_e, lat_grid, np.asarray(s3_lat_grid) / 3.0)
            Sff, Sf1, S11 = np.sum(f_e ** 2 / s_e ** 2), np.sum(f_e / s_e ** 2), np.sum(1 / s_e ** 2)
            d = {}
            for rho in (0.0, rho_floor, 0.5, 1.0):
                sc2 = (rho * sig_ref) ** 2
                info_f = Sff - sc2 * Sf1 ** 2 / (1 + sc2 * S11)
                info_1 = S11 - sc2 * S11 ** 2 / (1 + sc2 * S11)
                # the Eq.-(6) estimator (independent weights) evaluated under the true covariance:
                # Var = 1/S_ff + sigma_c^2 (S_f1/S_ff)^2 ; for f = 1: 1/S_11 + sigma_c^2
                var_eq6_f = 1 / Sff + sc2 * (Sf1 / Sff) ** 2; var_eq6_1 = 1 / S11 + sc2
                d[f"rho_{rho:.3f}"] = dict(sigma_camp_Tg=float(info_f ** -0.5), sigma_camp_1D_Tg=float(info_1 ** -0.5),
                                           gain=float(np.sqrt(info_f / info_1)), sigma_c_Tg=float(np.sqrt(sc2)),
                                           eq6_weights_true_cov=dict(sigma_camp_Tg=float(np.sqrt(var_eq6_f)), sigma_camp_1D_Tg=float(np.sqrt(var_eq6_1)),
                                                                     gain=float(np.sqrt(var_eq6_1 / var_eq6_f))),
                                           mean_f_over_mean_f2=float(Sf1 / Sff))
            # direct-solve check of Sherman-Morrison on a 400-event subsample, rho = 0.5
            idx = np.linspace(0, len(phi_e) - 1, 400).astype(int); sc2 = (0.5 * sig_ref) ** 2
            C = np.diag(s_e[idx] ** 2) + sc2; fe = f_e[idx]
            direct = float((fe @ np.linalg.solve(C, fe)) ** -0.5)
            Sff_, Sf1_, S11_ = np.sum(fe ** 2 / s_e[idx] ** 2), np.sum(fe / s_e[idx] ** 2), np.sum(1 / s_e[idx] ** 2)
            sm = float((Sff_ - sc2 * Sf1_ ** 2 / (1 + sc2 * S11_)) ** -0.5)
            d["sherman_morrison_check_400events_rho0.5"] = dict(direct=direct, sherman_morrison=sm)
            d["independent_gain_f_only"] = float(np.sqrt(np.mean(f_e ** 2)))
            out[o] = d
        return out

    h_main, h_o3 = r_main["lat_20km_homogeneous"]["band/quiet"], r_o3["lat_20km_homogeneous"]["band/quiet"]
    camp["exact20_reference_floor"] = gains(f_ex, h_main["mmin"], np.array(h_main["lat"]), "ref")
    camp["exact20_measured_ozone"] = gains(f_ex, h_o3["mmin"], np.array(h_o3["lat"]), "o3")
    camp["archived_gains"] = dict(reference=[r_main["scenarios"]["combined"]["band/quiet"]["gain_ISS-like"], r_main["scenarios"]["combined"]["band/quiet"]["gain_dedicated 70"]],
                                  reference_sigma_camp_Tg=[r_main["scenarios"]["combined"]["band/quiet"]["campaign_sigma_ISS-like_Tg"], r_main["scenarios"]["combined"]["band/quiet"]["campaign_sigma_dedicated 70_Tg"]],
                                  measured_ozone=[r_o3["scenarios"]["combined"]["band/quiet"]["gain_ISS-like"], r_o3["scenarios"]["combined"]["band/quiet"]["gain_dedicated 70"]])
    # the same on the declared (table) level: f2d on the level, sigma from the archived map (3 sigma = M * f) on the model latitude grid
    s3_T_ref = maps["map_pm41.npz"]["mmin_band_quiet_oe"][:, kT] * f_T
    s3_T_o3 = maps["map_pm41_o3lat.npz"]["mmin_band_quiet_oe"][:, kT] * f_T
    camp["table_level_reference_floor"] = gains(f_T, s3_T_ref, lat, "refT")
    camp["table_level_measured_ozone"] = gains(f_T, s3_T_o3, lat, "o3T")
    camp["referee_illustration"] = dict(n=20, sigma_e=1.0, sigma_c=1.0, mean_uncertainty_general=float(np.sqrt(1 + 1 / 20)),
                                        naive_independent=float(np.sqrt(2 / 20)),
                                        check_sherman_morrison=float((20 - 1.0 * 20 ** 2 / (1 + 20)) ** -0.5))
    res["campaign"] = camp

    # ---- 6. decision rule and completeness (false_alarm_trials.py logic, copied) -----
    dr_all = {}
    for nlabel, N in (("paper_design_tex_30x365x33", 30.0 * 365.0 * BINS), ("false_alarm_trials_py_30.2x365.25x33", EVENTS_PER_DAY * 365.25 * BINS)):
        zc = float(norm.isf(1.0 / N)); p = float(norm.sf(zc))
        dr_all[nlabel] = dict(tests_per_year=N, per_test_tail_p=p, z_c_one_expected_false_alarm=zc, M_thr_over_Mmin=zc / 3, M_thr_Tg=zc / 3 * M_MIN_BAND,
              sigma_M_Tg=M_MIN_BAND / 3,
              detection_probability={f"{k:.1f}xMmin": float(norm.sf(zc - 3 * k)) for k in (1.0, 1.5, 2.0, 2.5, 3.0)},
              mass_for_50pct_detection_Tg=zc / 3 * M_MIN_BAND, mass_for_90pct_detection_Tg=(zc + norm.isf(0.1)) / 3 * M_MIN_BAND,
              mass_for_95pct_detection_Tg=(zc + norm.isf(0.05)) / 3 * M_MIN_BAND,
              expected_false_alarms_per_year=N * p,
              familywise_false_alarm_probability=dict(independent=float(1 - (1 - p) ** N), union_bound=float(min(1.0, N * p))),
              per_test_z_for_familywise={f"{a:.2f}": dict(independent=float(norm.isf(1 - (1 - a) ** (1 / N))), bonferroni=float(norm.isf(a / N)),
                                                           detection_probability_at_3sigma_signal_independent=float(norm.sf(norm.isf(1 - (1 - a) ** (1 / N)) - 3)))
                                         for a in (0.05, 0.5)})
    dr = dict(**dr_all,
              statement="The expected number of false alarms E[count] = N p is a linear functional of the per-test tail probability and holds for any "
                        "dependence between the N tests; the familywise probability P(at least one) = 1 - (1-p)^N assumes independence (the union "
                        "bound N p holds for any dependence but is loose, here saturating at 1).")
    res["decision_rule"] = dr

    # ---- 7. persistence check of the null sample --------------------------------
    c = np.genfromtxt(NULL_CSV, delimiter=",", names=True, dtype=None, encoding="utf-8")
    yr = np.asarray(c["year"], float); M = np.asarray(c["M_full"], float); order = np.argsort(yr)
    yr, M = yr[order], M[order]
    cal = np.floor(yr).astype(int); years, counts = np.unique(cal, return_counts=True)
    gaps_d = np.diff(yr) * 365.25
    pers = dict(n_occultations=int(len(M)), per_calendar_year={str(y): int(n) for y, n in zip(years, counts)}, n_distinct_years=int(len(years)),
                sigma_emp_Tg=SIGMA_EMP_TG, median_gap_days=float(np.median(gaps_d)), gap_days_minmax=[float(gaps_d.min()), float(gaps_d.max())],
                n_consecutive_pairs=int(len(M) - 1), time_ordering="by the decimal year column; the sample mixes the three latitude bands of Sect. 7")
    pers["M_full_stats_Tg"] = dict(mean=float(M.mean()), std_ddof1=float(M.std(ddof=1)), median=float(np.median(M)),
                                   mean_over_sigma_emp=float(M.mean() / SIGMA_EMP_TG))
    for k in (1.0, 2.0):
        for tag, Mx in (("", M), ("_mean_subtracted", M - M.mean())):
            ex_ = Mx > k * SIGMA_EMP_TG
            both = ex_[:-1] & ex_[1:]
            pers[f"exceed_plus{k:.0f}sigma{tag}"] = dict(n_single=int(ex_.sum()), frac_single=float(ex_.mean()), n_consecutive_both=int(both.sum()),
                                                         frac_consecutive_both=float(both.mean()), expected_both_if_independent=float(ex_.mean() ** 2),
                                                         expected_count_if_independent=float(ex_.mean() ** 2 * (len(M) - 1)),
                                                         gaussian_single_tail=float(norm.sf(k)))
    pers["note"] = "quick persistence check only; the lag-1 autocorrelation is in facts_item45.md"
    res["persistence"] = pers

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT}  ({time.time() - t0:.0f} s)")
    # console digest
    for label, fn, key in SCENARIOS:
        b = book[label]
        for lv in (f"{zT:.4f}_km", f"{zB:.4f}_km", "20.0_km_loglinear"):
            x = b[lv]; cv = x["coverage_0p1"]
            print(f"{label:36s} {lv:18s} f41={x['f_core_p41']:.3f} f89={x['f_pole_p89']:.3f} fmin={x['f_min']:.3f} M41={x['M_core_p41']:.4f} "
                  f"M89={x['M_pole_p89']:.3f} Mmax={x['M_max']:.3f}@{x['M_max_lat']:+.0f} cov={cv and (cv['lat_min'], cv['lat_max'])}")
    print("exact20 (script):", {k: (v["f_smoothed_core_p41"], v["f_smoothed_min"], v["band"]["M_min"], v["band"]["M_max"], v["band"]["coverage_0p1"]) for k, v in ex.items()})
    for name in ("meridional", "azimuth_45deg"):
        print(name, {k: (round(v["ratio_first_order"], 4), round(v["ratio_full_peel"], 4)) for k, v in summ[name].items() if k.startswith("phi")},
              "core dM% ", summ[name]["core_threshold_change_percent_first_order"], summ[name]["core_threshold_change_percent_full_peel"])
    print("vertical:", {k: (v["ratio_core_p41"], v["core_threshold_change_percent"], v["coverage_0p1"]) for k, v in vr["cases"].items()})
    for k in ("exact20_reference_floor", "exact20_measured_ozone", "table_level_reference_floor", "table_level_measured_ozone"):
        print(k, {o: {r: (round(d[r]["gain"], 3), f"{d[r]['sigma_camp_Tg']:.3e}") for r in d if r.startswith("rho")} for o, d in camp[k].items()})
    print("decision:", json.dumps({k: v for k, v in dr.items() if k != "statement"}, indent=None))
    for k in ("exact20_reference_floor", "table_level_reference_floor"):
        print(k, "eq6 true-cov", {o: {r: (round(d[r]["eq6_weights_true_cov"]["gain"], 3), f"{d[r]['eq6_weights_true_cov']['sigma_camp_Tg']:.3e}",
                                         f"{d[r]['eq6_weights_true_cov']['sigma_camp_1D_Tg']:.3e}") for r in d if r.startswith("rho")} for o, d in camp[k].items()})
    print("persistence:", json.dumps({k: v for k, v in pers.items() if k != "note"}, indent=None))


if __name__ == "__main__":
    main()
