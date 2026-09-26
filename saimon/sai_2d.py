"""
sai_2d.py -- Two-dimensional (latitude-resolved) extension of the silica
SAI detection analysis.

Provides the four ingredients the 2D study needs on top of the 1D
(z = 20 km, single-column) Fisher machinery of the detection-threshold
sections:

1. The zonal-mean steady-state silica field of an injection scenario
   (Nahliel's climlab transport runs). Two loaders are provided: the
   column-only file (CSVFiles/Silica_05_no_coagulation.nc, kept for
   comparison) and the full latitude x pressure-level file
   (CSVFiles/Silica_05_no_coagulation_with2D.nc), from which the
   enhancement factor f(lat) is defined as the LOCAL mass concentration
   at the analysis altitude relative to the 1D forward-model layer of
   the same total mass.

2. The zonal structure of the sulfate background from GloSSAC (525 nm
   extinction at a chosen altitude and epoch), expressed as an amplitude
   factor g(lat) relative to the tropical mean that anchors the 1D
   background presets.

3. A solar-occultation sampling model: a circular orbit (with J2 nodal
   precession) plus a circular-ecliptic Sun, returning the tangent-point
   latitude of every sunrise/sunset occultation over a mission, and the
   latitudinal footprint of each limb line of sight.

4. Small Fisher helpers (2-parameter and 4-parameter marginal silica
   errors) shared with the 1D configuration-hierarchy analysis.

The central simplification the module exploits (verified numerically in
scripts/run_detectability_2d.py): in the linearised retrieval with
absolute per-channel noise floors, multiplying all sulfate nuisance
columns by a common latitude factor g(lat) leaves the marginal silica
error unchanged, so the latitude dependence of the detection threshold
comes only from the silica distribution f(lat) and from the sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import MODELS, GLOSSAC_NC
import numpy as np

R_EARTH_M = 6_371_000.0
J2_EARTH = 1.08263e-3
MU_EARTH = 3.986004418e14  # [m^3 s^-2]
OBLIQUITY_DEG = 23.44


# ---------------------------------------------------------------------------
# 1. Silica zonal distribution (Nahliel transport runs)
# ---------------------------------------------------------------------------

@dataclass
class SilicaZonalDistribution:
    """Zonal-mean steady-state silica column for one injection scenario."""
    lat_deg: np.ndarray          # (n_lat,) model latitudes
    column_kg_m2: np.ndarray     # (n_lat,) yearly-mean column density
    enhancement: np.ndarray      # (n_lat,) f(lat) = column / uniform column
    area_m2: np.ndarray          # (n_lat,) band areas
    total_mass_kg: float         # steady-state global burden
    lifetime_yr: float           # e-folding lifetime from the file
    inj_lat_deg: float
    inj_pres_hpa: float

    def enhancement_at(self, lat_deg) -> np.ndarray:
        """f interpolated to arbitrary latitudes."""
        return np.interp(lat_deg, self.lat_deg, self.enhancement)


def band_areas_m2(lat_deg: np.ndarray) -> np.ndarray:
    """Areas of the latitude bands centred on `lat_deg` (edges at midpoints)."""
    lat = np.asarray(lat_deg, dtype=float)
    edges = np.concatenate([[-90.0], 0.5 * (lat[:-1] + lat[1:]), [90.0]])
    return 2.0 * np.pi * R_EARTH_M**2 * (
        np.sin(np.deg2rad(edges[1:])) - np.sin(np.deg2rad(edges[:-1]))
    )


def load_silica_zonal_distribution(
    nc_path: str = None,
    inj_lat_deg: float = 1.0,
    inj_pres_hpa: float = 51.0,
) -> SilicaZonalDistribution:
    """Load one injection scenario from Nahliel's steady-state transport file.

    The file holds the yearly-mean total column density [kg m^-2] versus
    latitude for a 12 (injection pressure) x 14 (injection latitude) grid of
    continuous-injection runs (0.5 um particles, no coagulation, single
    cores bin).  The enhancement factor is normalised so that a globally
    uniform layer of the same total mass has f == 1 everywhere.
    """
    if nc_path is None:
        nc_path = str(MODELS / "Silica_05_no_coagulation.nc")
    import netCDF4

    ds = netCDF4.Dataset(nc_path)
    try:
        inj_lats = ds["inj_lat"][:].filled(np.nan)
        inj_pres = ds["inj_pres"][:].filled(np.nan)
        il = int(np.argmin(np.abs(inj_lats - inj_lat_deg)))
        ip = int(np.argmin(np.abs(inj_pres - inj_pres_hpa)))
        lat = ds["lat"][:].filled(np.nan)
        col = ds["column_density"][ip, il, 0, :].filled(np.nan)
        life = float(ds["lifetime"][ip, il])
    finally:
        ds.close()

    area = band_areas_m2(lat)
    total = float(np.nansum(col * area))
    uniform_col = total / area.sum()
    return SilicaZonalDistribution(
        lat_deg=lat,
        column_kg_m2=col,
        enhancement=col / uniform_col,
        area_m2=area,
        total_mass_kg=total,
        lifetime_yr=life,
        inj_lat_deg=float(inj_lats[il]),
        inj_pres_hpa=float(inj_pres[ip]),
    )


@dataclass
class SilicaZonal2D:
    """Full (latitude x pressure-level) steady-state silica field for one
    injection scenario, with the enhancement factor evaluated at the
    detection-analysis altitude from the LOCAL mass concentration."""
    lat_deg: np.ndarray            # (n_lat,)
    lev_hpa: np.ndarray            # (n_lev,) model pressure levels
    z_lev_km: np.ndarray           # (n_lev,) US-std altitudes of the levels
    q_kg_kg: np.ndarray            # (n_lat, n_lev) mass mixing ratio
    conc_kg_m3: np.ndarray         # (n_lat, n_lev) mass concentration
    column_kg_m2: np.ndarray       # (n_lat,) integral q dp/g
    area_m2: np.ndarray            # (n_lat,) band areas
    total_mass_kg: float
    lifetime_yr: float
    inj_lat_deg: float
    inj_pres_hpa: float
    analysis_alt_km: float
    enhancement: np.ndarray        # (n_lat,) f(lat) at analysis_alt_km
    enhancement_column: np.ndarray  # (n_lat,) column-based f (v1 definition)
    peak_alt_km: np.ndarray        # (n_lat,) altitude of max concentration
    enhancement_2d: np.ndarray = None  # (n_lat, n_lev) f(lat, z): local
    #   concentration per Tg total, divided by the 1D-layer concentration per
    #   Tg AT analysis_alt_km (fixed reference, so t_k(lat,z) = f * t_k(20 km))

    def enhancement_at(self, lat_deg) -> np.ndarray:
        return np.interp(lat_deg, self.lat_deg, self.enhancement)


def load_silica_zonal_2d(
    nc_path: str = None,
    inj_lat_deg: float = 1.0,
    inj_pres_hpa: float = 51.0,
    analysis_alt_km: float = 20.0,
    layer_bottom_km: float = 16.0,
    layer_top_hpa: float = 30.0,
) -> SilicaZonal2D:
    """Load one scenario from the transport file WITH vertical structure.

    The enhancement factor is now defined from the local mass concentration
    at the analysis altitude, relative to the 1D forward-model layer
    (constant mixing ratio between layer_bottom_km and layer_top_hpa) of the
    same total mass:

        f(lat) = [conc(lat, z_a) / M_tot] / [conc_1D(z_a) / M_1D] ,

    so that t_k(lat) = f(lat) * t_k exactly as in the column-based analysis,
    but with the modelled vertical profile instead of the assumed one.
    All altitude/density conversions use the US Standard Atmosphere, the
    same atmosphere used by the 1D forward model.
    """
    if nc_path is None:
        nc_path = str(MODELS / "Silica_05_no_coagulation_with2D.nc")
    import netCDF4

    from .atmosphere import AVOGADRO, G_0, M_AIR, us_standard_atmosphere

    ds = netCDF4.Dataset(nc_path)
    try:
        inj_lats = ds["inj_lat"][:].filled(np.nan)
        inj_pres = ds["inj_pres"][:].filled(np.nan)
        # inj_lat_deg may be a scalar or a sequence: a sequence means a
        # combined scenario (equal injection rates at each latitude; the
        # transport is linear, so the fields simply add).
        req_lats = np.atleast_1d(np.asarray(inj_lat_deg, dtype=float))
        ils = [int(np.argmin(np.abs(inj_lats - v))) for v in req_lats]
        ip = int(np.argmin(np.abs(inj_pres - inj_pres_hpa)))
        lat = ds["lat"][:].filled(np.nan)
        lev = ds["lev"][:].filled(np.nan)          # hPa, top -> bottom
        q = sum(ds["tracer_density"][ip, il, 0, :, :].filled(np.nan)
                for il in ils)                     # (lat, lev)
        # combined lifetime = total steady mass / total injection rate
        life = float(np.mean([ds["lifetime"][ip, il] for il in ils]))
        inj_lat_out = float(np.mean([inj_lats[il] for il in ils]))
    finally:
        ds.close()

    # US-std atmosphere on a fine grid: p(z), air mass density rho(z)
    z_fine = np.arange(0.0, 60_000.0, 10.0)
    prof = us_standard_atmosphere(z_fine)
    p_fine_hpa = np.asarray(prof.pressure_pa) / 100.0
    rho_fine = np.asarray(prof.number_density_m3) * M_AIR / AVOGADRO

    # altitude of each model level (log-p interpolation)
    z_lev_km = np.interp(np.log(lev), np.log(p_fine_hpa[::-1]),
                         z_fine[::-1]) / 1000.0
    rho_lev = np.interp(z_lev_km * 1000.0, z_fine, rho_fine)
    conc = q * rho_lev[None, :]                    # kg m^-3

    # column and total mass from the pressure integral (hydrostatic)
    p_pa = lev * 100.0
    column = np.trapezoid(q, p_pa, axis=1) / G_0       # kg m^-2 (lev ascending in p)
    area = band_areas_m2(lat)
    total = float(np.nansum(column * area))

    # local concentration per unit total mass, at the analysis altitude
    p_a = np.interp(analysis_alt_km * 1000.0, z_fine, p_fine_hpa)
    rho_a = np.interp(analysis_alt_km * 1000.0, z_fine, rho_fine)
    q_a = np.array([np.interp(np.log(p_a), np.log(lev), q[i, :])
                    for i in range(len(lat))])
    conc_a = q_a * rho_a

    # 1D reference: constant-mixing-ratio layer, same atmosphere
    z_top = np.interp(np.log(layer_top_hpa), np.log(p_fine_hpa[::-1]),
                      z_fine[::-1])
    in_layer = (z_fine >= layer_bottom_km * 1000.0) & (z_fine <= z_top)
    area_earth = 4.0 * np.pi * R_EARTH_M**2
    conc_1d_per_kg = rho_a / (area_earth * np.trapezoid(rho_fine[in_layer],
                                                    z_fine[in_layer]))

    f = (conc_a / total) / conc_1d_per_kg
    f_col = column / (total / area.sum())
    peak_alt = z_lev_km[np.nanargmax(np.where(np.isfinite(conc), conc, -1.0),
                                     axis=1)]
    # full 2D field, same fixed 20-km 1D reference in the denominator
    f_2d = (conc / total) / conc_1d_per_kg

    return SilicaZonal2D(
        lat_deg=lat, lev_hpa=lev, z_lev_km=z_lev_km, q_kg_kg=q,
        conc_kg_m3=conc, column_kg_m2=column, area_m2=area,
        total_mass_kg=total, lifetime_yr=life,
        inj_lat_deg=inj_lat_out, inj_pres_hpa=float(inj_pres[ip]),
        analysis_alt_km=analysis_alt_km, enhancement=f,
        enhancement_column=f_col, peak_alt_km=peak_alt,
        enhancement_2d=f_2d,
    )


def smooth_over_footprint(
    lat_deg: np.ndarray, f: np.ndarray, half_width_deg: float
) -> np.ndarray:
    """Boxcar-average f(lat) over +/- half_width_deg (limb LOS footprint)."""
    if half_width_deg <= 0:
        return f.copy()
    out = np.empty_like(f)
    for i, lc in enumerate(lat_deg):
        m = np.abs(lat_deg - lc) <= half_width_deg
        out[i] = np.nanmean(f[m])
    return out


# ---------------------------------------------------------------------------
# 2. GloSSAC zonal background factor
# ---------------------------------------------------------------------------

def glossac_zonal_factor(
    nc_path: str = None,
    wavelength_nm: float = 525.0,
    epoch_center_yyyymm: int = 200105,
    months_half_window: int = 2,
    alt_km: float = 20.0,
    norm_lat_range: tuple = (-20.0, 20.0),
):
    """Zonal amplitude factor g(lat) of the sulfate background at one epoch.

    Returns (lat_deg, g) where g is the GloSSAC extinction at
    (alt_km, wavelength_nm), time-averaged over the epoch window, divided
    by its cosine-weighted mean over `norm_lat_range` -- the same tropical
    band (20S-20N) used to anchor the 1D empirical background profiles.
    """
    if nc_path is None:
        nc_path = str(GLOSSAC_NC)
    import netCDF4

    ds = netCDF4.Dataset(nc_path)
    try:
        wls = ds["wavelengths_glossac"][:].filled(np.nan)
        iw = int(np.argmin(np.abs(wls - wavelength_nm)))
        alt = ds["alt"][:].filled(np.nan)
        iz = int(np.argmin(np.abs(alt - alt_km)))
        time = ds["time"][:].filled(-1).astype(int)
        # month arithmetic on YYYYMM
        y0, m0 = divmod(int(epoch_center_yyyymm), 100)
        idx0 = y0 * 12 + (m0 - 1)
        tidx = time // 100 * 12 + (time % 100 - 1)
        tmask = np.abs(tidx - idx0) <= months_half_window
        ext = ds["Glossac_Aerosol_Extinction_Coefficient"][tmask, :, iz, iw]
        ext = np.ma.masked_less(ext, 0.0)
        lat = ds["lat"][:].filled(np.nan)
    finally:
        ds.close()

    prof = ext.mean(axis=0)  # (lat,) masked mean over the window
    prof = prof.filled(np.nan)
    w = np.cos(np.deg2rad(lat))
    nmask = (lat >= norm_lat_range[0]) & (lat <= norm_lat_range[1]) & np.isfinite(prof)
    ref = np.nansum(prof[nmask] * w[nmask]) / np.nansum(w[nmask])
    return lat, prof / ref


# ---------------------------------------------------------------------------
# 3. Solar-occultation sampling (circular orbit + circular-ecliptic Sun)
# ---------------------------------------------------------------------------

@dataclass
class OccultationEvents:
    """Tangent-point sampling of a solar-occultation mission."""
    day: np.ndarray            # (n_events,) time of event [days from start]
    lat_deg: np.ndarray        # (n_events,) tangent-point latitude
    is_sunrise: np.ndarray     # (n_events,) True = sunrise (h_t increasing)
    footprint_half_deg: np.ndarray  # (n_events,) latitudinal half-extent of LOS
    s_north_abs: np.ndarray = None  # (n_events,) |LOS . north| at tangent point


def simulate_occultation_latitudes(
    inclination_deg: float = 51.6,
    altitude_km: float = 420.0,
    mission_days: float = 365.0,
    dt_s: float = 20.0,
    tangent_alt_km: float = 20.0,
    layer_top_km: float = 24.0,
    raan0_deg: float = 0.0,
    sun_lon0_deg: float = 0.0,
) -> OccultationEvents:
    """Tangent-point latitudes of all sunrise/sunset solar occultations.

    Model: satellite on a circular orbit (Keplerian rate + J2 secular nodal
    precession), Sun on a circular ecliptic orbit.  An occultation event is
    the instant the solar line of sight crosses the tangent altitude
    `tangent_alt_km` on the far side of the Earth; its tangent point is the
    point of closest approach of that line to the Earth's centre.  One
    sunrise and one sunset event occur per orbit (except near full-sun
    beta angles, where the satellite never enters eclipse).

    The latitudinal footprint of each event is the half-length of the limb
    chord through the aerosol layer (tangent_alt_km .. layer_top_km)
    projected on the local north direction, in degrees of latitude.
    """
    a = R_EARTH_M + altitude_km * 1e3
    n = np.sqrt(MU_EARTH / a**3)                       # [rad/s]
    inc = np.deg2rad(inclination_deg)
    raan_dot = -1.5 * n * J2_EARTH * (R_EARTH_M / a) ** 2 * np.cos(inc)
    eps = np.deg2rad(OBLIQUITY_DEG)

    t = np.arange(0.0, mission_days * 86400.0, dt_s)
    u = n * t                                          # argument of latitude
    raan = np.deg2rad(raan0_deg) + raan_dot * t

    cu, su = np.cos(u), np.sin(u)
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    # ECI position of the satellite (unit sphere, scaled by a)
    x = a * (cO * cu - sO * su * ci)
    y = a * (sO * cu + cO * su * ci)
    z = a * (su * si)

    lam = np.deg2rad(sun_lon0_deg) + 2.0 * np.pi * t / (365.25 * 86400.0)
    sx = np.cos(lam)
    sy = np.sin(lam) * np.cos(eps)
    sz = np.sin(lam) * np.sin(eps)

    proj = x * sx + y * sy + z * sz                    # r_s . s_hat
    px, py, pz = x - proj * sx, y - proj * sy, z - proj * sz
    pnorm = np.sqrt(px**2 + py**2 + pz**2)
    h_t = np.where(proj < 0.0, pnorm - R_EARTH_M, np.inf)

    target = tangent_alt_km * 1e3
    dsign = np.diff(np.sign(h_t - target))
    idx = np.nonzero(np.isfinite(h_t[:-1]) & np.isfinite(h_t[1:]) & (dsign != 0))[0]

    # linear interpolation to the crossing
    f1, f2 = h_t[idx] - target, h_t[idx + 1] - target
    w = f1 / (f1 - f2)
    lat_tp = np.rad2deg(np.arcsin(
        (pz[idx] + w * (pz[idx + 1] - pz[idx]))
        / (pnorm[idx] + w * (pnorm[idx + 1] - pnorm[idx]))
    ))
    day = (t[idx] + w * dt_s) / 86400.0
    is_sunrise = f2 > f1

    # LOS footprint: half-chord through the layer, projected northward
    r_t = R_EARTH_M + target
    half_chord = np.sqrt((R_EARTH_M + layer_top_km * 1e3) ** 2 - r_t**2)
    # north unit vector at tangent point p: N = (z_hat - (z_hat.p_hat) p_hat)/|...|
    phat = np.stack([px[idx], py[idx], pz[idx]]) / pnorm[idx]
    s_hat = np.stack([sx[idx], sy[idx], sz[idx]])
    sinlat = phat[2]
    north = np.stack([-phat[0] * sinlat, -phat[1] * sinlat, 1 - phat[2] * sinlat])
    north /= np.sqrt((north**2).sum(axis=0))
    s_north = np.abs((s_hat * north).sum(axis=0))
    foot_half_deg = half_chord * s_north / (np.pi * R_EARTH_M / 180.0)

    return OccultationEvents(
        day=day, lat_deg=lat_tp, is_sunrise=is_sunrise,
        footprint_half_deg=foot_half_deg, s_north_abs=s_north,
    )


# ---------------------------------------------------------------------------
# 3b. Line-of-sight tropopause clearance (along-path obscuration)
# ---------------------------------------------------------------------------
#
# A tangent point above the *local* tropopause can still have its limb ray
# dip below the tropopause at other latitudes along the line of sight.  The
# ray climbs as h(x) = sqrt((R+z_t)^2 + x^2) - R ~ x^2/2R (+1 km within
# 113 km, +5 km within 250 km of the tangent point), so with the ISS-like
# LOS azimuth statistics (median meridional cosine ~0.3 away from the
# turning latitudes) the effect is confined to a ~2 deg strip just poleward
# of the subtropical tropopause break, and only if the break is sharp.

KM_PER_DEG_LAT = np.pi * (R_EARTH_M / 1e3) / 180.0


def tropopause_break_km(lat_deg):
    """Sharp-break zonal-mean tropopause height [km]: 16.5 km inside +-30
    deg, dropping to 11.5 km across a 2-deg-wide subtropical break, then
    slowly to 9.5 km at high latitude.  Conservative (worst-case) sharpness
    for the along-path clearance question."""
    a = np.abs(np.asarray(lat_deg, dtype=float))
    return np.where(a <= 30.0, 16.5,
                    np.where(a <= 32.0, 16.5 - 5.0 * (a - 30.0) / 2.0,
                             np.maximum(11.5 - 2.0 * (a - 32.0) / 38.0, 9.5)))


def los_obscured(lat_ev_deg, s_north_abs, z_tan_km, margin_km=1.0,
                 z_trop_fun=tropopause_break_km,
                 x_max_km=800.0, dx_km=5.0):
    """True where the limb ray comes within `margin_km` of the tropopause
    anywhere along the line of sight (the x=0 term reproduces the local
    tangent-point criterion).  Vectorized over events."""
    lat = np.abs(np.atleast_1d(np.asarray(lat_ev_deg, dtype=float)))
    s_n = np.atleast_1d(np.asarray(s_north_abs, dtype=float))
    x = np.arange(0.0, x_max_km + 1e-9, dx_km)
    r_km = R_EARTH_M / 1e3 + z_tan_km
    h = np.sqrt(r_km**2 + x**2) - R_EARTH_M / 1e3           # (nx,) ray altitude
    bad = np.zeros(lat.shape, dtype=bool)
    for sgn in (1.0, -1.0):                                  # both LOS branches
        phi = lat[:, None] + sgn * s_n[:, None] * x[None, :] / KM_PER_DEG_LAT
        bad |= (h[None, :] - z_trop_fun(phi) < margin_km).any(axis=1)
    return bad


def obscuration_probability(lat_grid_deg, z_grid_km, events=None,
                            margin_km=1.0, z_trop_fun=tropopause_break_km,
                            lat_halfwidth_deg=1.0, **orbit_kw):
    """P_obs(lat, z): fraction of occultation events with tangent latitude
    within +-lat_halfwidth_deg of each grid latitude whose LOS crosses
    within `margin_km` of the tropopause at tangent altitude z.  Events
    default to one year of the ISS-like orbit."""
    if events is None:
        events = simulate_occultation_latitudes(**orbit_kw)
    lam, s_n = events.lat_deg, events.s_north_abs
    P = np.full((len(lat_grid_deg), len(z_grid_km)), np.nan)
    for i, lat0 in enumerate(np.asarray(lat_grid_deg, dtype=float)):
        m = np.abs(lam - lat0) <= lat_halfwidth_deg
        if m.sum() < 5:      # sparse band: fold the hemispheres
            m = np.abs(np.abs(lam) - abs(lat0)) <= lat_halfwidth_deg
        if m.sum() < 5:
            continue
        for j, z in enumerate(np.asarray(z_grid_km, dtype=float)):
            P[i, j] = float(np.mean(los_obscured(
                lam[m], s_n[m], z, margin_km=margin_km,
                z_trop_fun=z_trop_fun)))
    return P


# ---------------------------------------------------------------------------
# 4. Fisher helpers (shared with the 1D configuration hierarchy)
# ---------------------------------------------------------------------------

def fisher_sigma2(s, t, sigma) -> float:
    """Marginal silica error, 2-parameter fit (sulfate amplitude + silica)."""
    w = 1.0 / np.asarray(sigma) ** 2
    J = np.column_stack([s, t])
    F = J.T @ np.diag(w) @ J
    if np.linalg.cond(F) > 1e12:
        return np.inf
    return float(np.sqrt(np.linalg.inv(F)[-1, -1]))


def fisher_sigma4(s, jr, js, t, sigma) -> float:
    """Marginal silica error, 4-parameter fit (amplitude, r_med, sigma_psd free)."""
    w = 1.0 / np.asarray(sigma) ** 2
    J = np.column_stack([s, jr, js, t])
    F = J.T @ np.diag(w) @ J
    if np.linalg.cond(F) > 1e12:
        return np.inf
    return float(np.sqrt(np.linalg.inv(F)[-1, -1]))
