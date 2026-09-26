"""
trace_gases.py — Trace-gas absorption cross sections and column optical depth.

Goal
----
Provide trace-gas extinction across 100 nm – 30 µm so we can verify that gases
do not contaminate the proposed 8.8 µm silica band (config B+).  8–12 µm is the
atmospheric window, so the expectation is small — but O3 (9.6 µm band), the H2O
continuum, HNO3 (11.3 µm) and CFC-12 (~8.6 µm) all sit nearby and must be checked.

Database split (see memory: trace-gas-database-choice)
------------------------------------------------------
  * IR lines (1–30 µm)   : HITRAN2024 via HAPI, line-by-line Voigt cross sections.
  * IR heavy molecules   : HITRAN absorption cross-section (.xsc) files (CFC-11/12).
  * IR H2O continuum     : MT_CKD self + foreign continuum (not in the line lists).
  * UV/Vis (100 nm–1 µm) : MPI-Mainz UV/VIS Atlas — Serdyuchenko (O3), Vandaele (NO2),
                           O2, SO2.

Units convention (matches the rest of sage3)
--------------------------------------------
  * Wavelengths passed in metres; internal HITRAN work in wavenumber [cm^-1].
  * Cross sections returned in m^2 / molecule.
  * Column number densities in m^-2 (integrated from us_standard_atmosphere +
    AFGL volume-mixing-ratio profiles).

Column approximation (v1)
-------------------------
Line shapes depend on (T, P), which vary strongly from 16 km (~100 hPa) to the
upper stratosphere.  For an overview "spectral-components" figure we compute each
gas cross section at a single representative (T, P) equal to the *gas-column-
weighted mean* above 16 km (the dominant absorbing layer), then multiply by the
full 16-km-up column.  This reproduces band positions and band strengths to
overview accuracy; full per-layer integration can be added later.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass

import numpy as np

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
from .config import TRACE_GASES
DATA_DIR = str(TRACE_GASES)
HITRAN_CACHE = os.path.join(DATA_DIR, "hitran_cache")
XSEC_CACHE = os.path.join(DATA_DIR, "xsec_cache")   # cached native Voigt (nu, coef)
CFC_DIR = os.path.join(DATA_DIR, "cfc_xsec")
MPI_DIR = os.path.join(DATA_DIR, "mpi_mainz")
AFGL_DIR = os.path.join(DATA_DIR, "afgl")
MTCKD_DIR = os.path.join(DATA_DIR, "mt_ckd")

CM2_TO_M2 = 1.0e-4          # cm^2 -> m^2
ATM_PER_PA = 1.0 / 101325.0  # Pa -> atm


# ---------------------------------------------------------------------------
# Gas registry
# ---------------------------------------------------------------------------
# For HITRAN line gases we list the molecule id, principal isotopologue id, and
# the band windows [cm^-1] that matter within 333–10000 cm^-1 (1–30 µm).  Lines
# are only fetched/evaluated inside these windows; the cross section is zero
# elsewhere for this overview.  This bounds the HAPI download and Voigt cost.

@dataclass(frozen=True)
class HitranGas:
    name: str            # display / species key
    mol_id: int          # HITRAN molecule id
    iso_id: int          # principal isotopologue id (1)
    windows: tuple       # tuple of (nu_min, nu_max) in cm^-1
    nu_max_cm: float = 11_000.0   # cap for resolution-matched fetch (avoids huge
                                  # UV fetches for gases with no Vis/UV lines)


HITRAN_GASES = {
    # name        id iso  band windows [cm^-1]                         features
    "h2o":  HitranGas("h2o",  1, 1, ((333, 2300), (3400, 4300), (5000, 5600), (6800, 7600))),
    "co2":  HitranGas("co2",  2, 1, ((550, 820), (900, 1150), (1900, 2450), (3400, 3800))),
    "o3":   HitranGas("o3",   3, 1, ((600, 1300), (1900, 2300), (2800, 3300))),
    "n2o":  HitranGas("n2o",  4, 1, ((500, 680), (1100, 1350), (2100, 2600))),
    "co":   HitranGas("co",   5, 1, ((1900, 2250), (3900, 4350))),
    "ch4":  HitranGas("ch4",  6, 1, ((1100, 1400), (2700, 3200), (4000, 4600))),
    "no2":  HitranGas("no2", 10, 1, ((750, 850), (1550, 1650), (2850, 2950))),
    "hno3": HitranGas("hno3", 12, 1, ((760, 920), (1280, 1360), (1680, 1760))),
    # SO2 (IR lines, not xsc): nu2 ~518 (19.3 µm, near config C), nu1 ~1151
    # (8.69 µm, on config B+), nu3 ~1362 (7.34 µm), nu1+nu3 ~2500 (4 µm).
    "so2":  HitranGas("so2",  9, 1, ((450, 600), (1050, 1400), (2450, 2550))),
    # O2: 1.27 µm singlet, B-band (~688 nm), A-band (~762 nm) — needs a Vis cap.
    "o2":   HitranGas("o2",  7, 1, ((7700, 8100), (14300, 14600), (12900, 13200)),
                      nu_max_cm=15_500.0),
}

# HITRAN absorption cross-section (.xsc) molecules (no practical line list).
# CFC-12 (CCl2F2) has a strong band at ~8.6–9 µm, right on config B+.
# We download a representative low-T / low-P xsc file for each.  These ids are
# HITRAN cross-section dataset aliases used in the xsec download URLs.
CFC_GASES = {
    # name : (HITRAN xsec molecule alias, human note)
    "cfc11": ("CFC-11", "CCl3F  ~9.2, 11.8 µm"),
    "cfc12": ("CFC-12", "CCl2F2 ~8.6, 10.9 µm"),
}

# MPI-Mainz UV/Vis species expected from the acquisition step (file stems are
# resolved at load time by scanning MPI_DIR for a matching prefix).
UVVIS_GASES = ("o3", "no2", "o2", "so2")


# ---------------------------------------------------------------------------
# HITRAN line cross sections (via HAPI)
# ---------------------------------------------------------------------------
_HAPI_STARTED = False


def _ensure_hapi():
    """Import HAPI and point db_begin at the cache dir (once)."""
    global _HAPI_STARTED
    import hapi  # noqa: import here so the module imports without HAPI present
    if not _HAPI_STARTED:
        os.makedirs(HITRAN_CACHE, exist_ok=True)
        hapi.db_begin(HITRAN_CACHE)
        _HAPI_STARTED = True
    return hapi


def _fetch_gas_lines(gas: HitranGas):
    """Ensure HITRAN lines for every band window of `gas` are in the cache."""
    hapi = _ensure_hapi()
    for k, (nu0, nu1) in enumerate(gas.windows):
        table = f"{gas.name}_w{k}"
        data_file = os.path.join(HITRAN_CACHE, table + ".data")
        if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
            continue
        _log.info("Fetching HITRAN %s window %d: %d–%d cm^-1", gas.name, k, nu0, nu1)
        hapi.fetch(table, gas.mol_id, gas.iso_id, nu0, nu1)
    return hapi


def _voigt_native(gas: HitranGas, nu_lo: float, nu_hi: float,
                  temperature_k: float, p_atm: float, omega_step: float):
    """Native HITRAN Voigt cross section (nu [cm^-1], sigma [cm^2/molec]),
    cached to disk so the expensive line-by-line step runs only once per
    (gas, T, P, range, step)."""
    os.makedirs(XSEC_CACHE, exist_ok=True)
    key = (f"{gas.name}_{int(round(nu_lo))}_{int(round(nu_hi))}"
           f"_T{temperature_k:.1f}_P{p_atm:.6f}_s{omega_step:g}")
    path = os.path.join(XSEC_CACHE, key + ".npz")
    if os.environ.get("SAIMON_XSEC_LOG"):   # record which cached grids a run touches
        with open(os.environ["SAIMON_XSEC_LOG"], "a") as _f:
            _f.write(key + ".npz\n")
    if os.path.exists(path):
        d = np.load(path)
        return d["nu"], d["coef"]
    hapi = _ensure_hapi()
    table = f"cont_{gas.name}_{int(nu_lo)}_{int(nu_hi)}"
    data_file = os.path.join(HITRAN_CACHE, table + ".data")
    if not (os.path.exists(data_file) and os.path.getsize(data_file) > 0):
        _log.info("Fetching continuous %s %d-%d cm^-1", gas.name, nu_lo, nu_hi)
        hapi.fetch(table, gas.mol_id, gas.iso_id, nu_lo, nu_hi)
    nu, coef = hapi.absorptionCoefficient_Voigt(
        SourceTables=table, Environment={"T": temperature_k, "p": p_atm},
        WavenumberStep=omega_step, HITRAN_units=True)       # cm^2/molecule
    np.savez_compressed(path, nu=nu, coef=coef)
    return nu, coef


def resolution_matched_xsec(gas_name: str,
                            wavelengths_m: np.ndarray,
                            temperature_k: float,
                            pressure_pa: float,
                            fwhm_um: float = 0.25,
                            omega_step: float = 0.01,
                            nu_pad_cm: float = 40.0) -> np.ndarray:
    """
    HITRAN line cross section [m^2/molecule] computed over a CONTINUOUS
    wavenumber range (no band-window gaps), then averaged to an instrument
    resolution element (Gaussian, `fwhm_um`).

    This is the resolution-correct quantity: stratospheric lines (~0.004 cm^-1)
    are far narrower than any instrument element, so the single-wavenumber Voigt
    value is meaningless on its own — only the average over the element is.  At
    fine omega_step the band average is converged and consistent with the line
    intensities validated against Mark's scan (integral sigma dnu = sum S).
    Use this (not the band-windowed `hitran_cross_section`) for spectral figures.
    """
    gas = HITRAN_GASES[gas_name]
    wl_m = np.asarray(wavelengths_m, dtype=float)
    nu_lo = max(1.0e-2 / wl_m.max() - nu_pad_cm, 1.0)
    nu_hi = 1.0e-2 / wl_m.min() + nu_pad_cm
    try:
        nu, coef = _voigt_native(gas, nu_lo, nu_hi, float(temperature_k),
                                 float(pressure_pa) * ATM_PER_PA, omega_step)
    except Exception as exc:
        _log.warning("resolution_matched_xsec failed for %s: %s", gas_name, exc)
        return np.zeros_like(wl_m)
    lam_um = 1.0e4 / nu
    order = np.argsort(lam_um)
    lam_s, coef_s = lam_um[order], coef[order]
    if lam_s.size < 2:
        return np.zeros_like(wl_m)
    # Boxcar average over a resolution element of width fwhm_um, evaluated on the
    # NATIVE fine grid via the cumulative integral.  This averages (not samples)
    # the dense line spectrum, so it is independent of line spacing and robust to
    # gases whose lines cover only part of the range.
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (coef_s[1:] + coef_s[:-1])
                                           * np.diff(lam_s))))
    half = fwhm_um / 2.0
    wl_um = wl_m * 1e6
    hi = np.interp(wl_um + half, lam_s, cum, left=cum[0], right=cum[-1])
    lo = np.interp(wl_um - half, lam_s, cum, left=cum[0], right=cum[-1])
    sigma_cm2 = (hi - lo) / (2.0 * half)
    return sigma_cm2 * CM2_TO_M2   # -> m^2/molecule


def gas_xsec_rm(gas: str, wavelengths_m: np.ndarray, temperature_k: float,
                pressure_pa: float, fwhm_um: float = 0.25,
                vmr_h2o: float = 0.0, hitran_nu_max: float = 11_000.0,
                nu_pad_cm: float = 40.0) -> np.ndarray:
    """
    Resolution-matched TOTAL cross section [m^2/molecule] for one gas over an
    arbitrary wavelength grid at element width `fwhm_um`, combining every source:
      * resolution-matched HITRAN lines (only where wl >= 1e-2/hitran_nu_max, so
        UV/Vis grids don't trigger huge line fetches),
      * MPI-Mainz UV/Vis cross sections (O3 Chappuis, NO2, O2, ...),
      * MT_CKD H2O continuum (H2O only), CFC fallback bands.
    Used by the slant-OD spectral figures so VIS/SWIR and MIR panels share one
    consistent, resolution-correct cross-section path.

    `nu_pad_cm` widens the wavenumber range fetched for the Voigt calculation so
    that wings of strong lines outside the requested band still contribute to
    the local sigma.  The default 40 cm^-1 is safe for line-core channels but
    underestimates band-wing absorption (e.g. O3 at 8.74 um, ~100 cm^-1 from
    the 9.6 um band core) -- pass nu_pad_cm=300 or more in that regime.
    """
    wl = np.asarray(wavelengths_m, dtype=float)
    if gas in CFC_GASES:
        return cfc_cross_section(gas, wl, fwhm_um=fwhm_um)
    sigma = np.zeros_like(wl)
    if gas in HITRAN_GASES:
        m = wl >= (1.0e-2 / HITRAN_GASES[gas].nu_max_cm)
        if m.any():
            sigma[m] += resolution_matched_xsec(gas, wl[m], temperature_k,
                                                pressure_pa, fwhm_um=fwhm_um,
                                                nu_pad_cm=nu_pad_cm)
    if gas in UVVIS_GASES:
        sigma += uvvis_cross_section(gas, wl)
    if gas == "h2o" and vmr_h2o > 0.0:
        sigma += mt_ckd_h2o_continuum(wl, temperature_k, pressure_pa, vmr_h2o)
    return sigma


def hitran_cross_section(gas_name: str,
                         wavelengths_m: np.ndarray,
                         temperature_k: float,
                         pressure_pa: float,
                         omega_step: float = 0.1) -> np.ndarray:
    """
    HITRAN line-by-line cross section [m^2/molecule] for one gas at (T, P).

    Computed with a Voigt profile inside the gas's band windows and interpolated
    onto `wavelengths_m`; zero outside the windows.  HITRAN_units=True makes HAPI
    return cm^2/molecule (a true cross section, density-independent).
    """
    gas = HITRAN_GASES[gas_name]
    hapi = _fetch_gas_lines(gas)
    wl_m = np.asarray(wavelengths_m, dtype=float)
    sigma = np.zeros_like(wl_m)
    p_atm = pressure_pa * ATM_PER_PA

    for k, (nu0, nu1) in enumerate(gas.windows):
        table = f"{gas.name}_w{k}"
        try:
            nu, coef = hapi.absorptionCoefficient_Voigt(
                SourceTables=table,
                Environment={"T": float(temperature_k), "p": float(p_atm)},
                WavenumberStep=omega_step,
                HITRAN_units=True,        # -> cm^2 / molecule
            )
        except Exception as exc:  # empty table / no lines in window
            _log.warning("HAPI failed for %s window %d: %s", gas_name, k, exc)
            continue
        # nu [cm^-1] -> wavelength [m]; interpolate (in wavelength) onto target.
        lam_win_m = 1.0e-2 / nu          # 1/cm -> m
        order = np.argsort(lam_win_m)
        lam_s = lam_win_m[order]
        coef_s = coef[order] * CM2_TO_M2
        in_win = (wl_m >= lam_s[0]) & (wl_m <= lam_s[-1])
        sigma[in_win] += np.interp(wl_m[in_win], lam_s, coef_s)
    return sigma


# ---------------------------------------------------------------------------
# HITRAN cross-section (.xsc) files for CFCs
# ---------------------------------------------------------------------------
def _download_cfc_xsc(alias: str) -> str | None:
    """
    Download a representative HITRAN .xsc file for a CFC if not cached.
    Returns the local path or None on failure.  We pick the lowest available
    temperature/pressure set (stratosphere-like).  Files are listed at
    hitran.org/data/xsec/ ; the naming is <Alias>_<T>.<P>_<numin>-<numax>_..xsc.
    """
    os.makedirs(CFC_DIR, exist_ok=True)
    local = os.path.join(CFC_DIR, f"{alias}.xsc")
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local
    # Known representative files on hitran.org (216 K, low pressure where avail).
    candidates = {
        "CFC-11": [
            "CFC-11_207.0K-0.0Torr_760.0-1200.0_0.03_N2.xsc",
            "CFC-11_216.0K-0.0Torr_810.0-1120.0_00.xsc",
        ],
        "CFC-12": [
            "CFC-12_190.0K-0.0Torr_800.0-1270.0_0.03_N2.xsc",
            "CFC-12_216.0K-0.0Torr_850.0-1200.0_00.xsc",
        ],
    }
    base = "https://hitran.org/data/xsec/"
    for fname in candidates.get(alias, []):
        url = base + fname
        try:
            urllib.request.urlretrieve(url, local)
            if os.path.getsize(local) > 0:
                _log.info("Downloaded CFC xsc %s", fname)
                return local
        except Exception as exc:
            _log.debug("CFC xsc %s not available: %s", fname, exc)
            continue
    _log.warning("Could not download an xsc file for %s", alias)
    if os.path.exists(local):
        os.remove(local)
    return None


def _parse_xsc(path: str):
    """
    Parse a HITRAN .xsc cross-section file.

    Header (first line, fixed format): molecule, nu_min, nu_max, npts, T[K],
    P[Torr], max_xsec, resolution, name, ... .  Data: whitespace-separated
    cross sections [cm^2/molecule] on a uniform wavenumber grid.
    Returns (nu [cm^-1], sigma [cm^2/molecule]).
    """
    with open(path) as fh:
        header = fh.readline()
        body = fh.read().split()
    parts = header.split()
    nu_min = float(parts[1]); nu_max = float(parts[2]); npts = int(parts[3])
    vals = np.array(body[:npts], dtype=float)
    nu = np.linspace(nu_min, nu_max, npts)
    return nu, vals


def _find_cfc_xsc(name: str) -> str | None:
    """Locate a local HITRAN .xsc cross-section file for a CFC.

    Accepts any file in CFC_DIR whose name starts with the HITRAN alias
    (e.g. "CFC-12") or the chemical formula (e.g. "CCl2F2"), so a file
    downloaded directly from hitran.org/xsc -- named like
    "CCl2F2_216.1K-37.3Torr_800.0-1270.0_00.xsc" -- is used without renaming.
    Falls back to the legacy fixed "<alias>.xsc" name and finally to a network
    download.  If several files match, the lexicographically first is chosen.
    """
    alias = CFC_GASES[name][0]                 # "CFC-12"
    formula = CFC_GASES[name][1].split()[0]    # "CCl2F2"
    if os.path.isdir(CFC_DIR):
        matches = sorted(
            f for f in os.listdir(CFC_DIR)
            if f.lower().endswith(".xsc") and f.startswith((alias, formula)))
        if matches:
            return os.path.join(CFC_DIR, matches[0])
    legacy = os.path.join(CFC_DIR, f"{alias}.xsc")
    if os.path.exists(legacy) and os.path.getsize(legacy) > 0:
        return legacy
    return _download_cfc_xsc(alias)


def cfc_cross_section(name: str, wavelengths_m: np.ndarray,
                      fwhm_um: float = 0.25) -> np.ndarray:
    """CFC cross section [m^2/molecule] from a HITRAN .xsc file, band-averaged
    to an instrument resolution element of width `fwhm_um` (boxcar).

    The .xsc native grid (~0.01 cm^-1) resolves the Q-branch structure far more
    finely than any instrument element, so a single-wavelength sample is not the
    observable -- only the average over the element is.  At 8.74 um a 0.25 um
    element spans ~1128-1161 cm^-1, reaching the nu8 Q-branch peak, so this
    averaging is essential to get the CFC slant-OD contribution right (a point
    sample on the band wing badly underestimates it).

    If no .xsc file is available, falls back to a documented Gaussian-band
    approximation so figures still render; drop a real file (downloaded from
    hitran.org/xsc) into CSVFiles/trace_gases/cfc_xsec/ to use measured data.
    The file is at a single (T, P); we use it T-independently here, which is
    adequate because the slant column at a 20 km tangent is dominated by the
    ~216 K layers the file was chosen to match.
    """
    wl_m = np.asarray(wavelengths_m, dtype=float)
    path = _find_cfc_xsc(name)
    if path is not None:
        nu, vals = _parse_xsc(path)                       # cm^-1, cm^2/molecule
        lam_um = 1.0e4 / nu
        order = np.argsort(lam_um)
        lam_s, vals_s = lam_um[order], vals[order]
        if lam_s.size >= 2:
            # Boxcar average over an element of width fwhm_um via the cumulative
            # integral on the native grid (mirrors resolution_matched_xsec).
            cum = np.concatenate(([0.0], np.cumsum(
                0.5 * (vals_s[1:] + vals_s[:-1]) * np.diff(lam_s))))
            half = fwhm_um / 2.0
            wl_um = wl_m * 1e6
            hi = np.interp(wl_um + half, lam_s, cum, left=cum[0], right=cum[-1])
            lo = np.interp(wl_um - half, lam_s, cum, left=cum[0], right=cum[-1])
            sigma_cm2 = (hi - lo) / (2.0 * half)
            sigma_cm2 = np.where((wl_um >= lam_s[0]) & (wl_um <= lam_s[-1]),
                                 sigma_cm2, 0.0)
            return sigma_cm2 * CM2_TO_M2                   # -> m^2/molecule
    sigma = np.zeros_like(wl_m)
    # ---- fallback band model (peak cross sections from HITRAN/IUP lab data) ----
    # Each band: (center_um, FWHM_um, peak_sigma_cm2). Values are literature
    # band peaks; used only when a measured .xsc is absent.
    bands = {
        "cfc12": [(8.60, 0.30, 1.1e-17), (10.93, 0.45, 1.6e-17)],   # CCl2F2
        "cfc11": [(9.22, 0.30, 1.5e-17), (11.82, 0.55, 2.1e-17)],   # CCl3F
    }
    lam_um = wl_m * 1e6
    for c_um, fwhm_um, peak_cm2 in bands.get(name, []):
        s = fwhm_um / 2.3548
        sigma += (peak_cm2 * CM2_TO_M2) * np.exp(-0.5 * ((lam_um - c_um) / s) ** 2)
    _log.warning("CFC %s: using fallback band model (no .xsc file present)", name)
    return sigma


# ---------------------------------------------------------------------------
# Representative (T, P) for the column and column-density helper
# ---------------------------------------------------------------------------
def representative_tp(altitude_m: np.ndarray,
                      n_gas_m3: np.ndarray,
                      pressure_pa: np.ndarray,
                      temperature_k: np.ndarray,
                      alt_min_m: float = 16_000.0):
    """
    Gas-column-weighted mean (T, P) above `alt_min_m`, plus the column density.

    The cross section is evaluated once at this representative (T, P) — the
    pressure/temperature of the layer that dominates this gas's column
    absorption — then scaled by the full column.  See module docstring.

    Returns (T_rep [K], P_rep [Pa], column [m^-2]).
    """
    altitude_m = np.asarray(altitude_m, float)
    w = np.where(altitude_m >= alt_min_m, np.asarray(n_gas_m3, float), 0.0)
    column = np.trapezoid(w, altitude_m)               # molecules m^-2
    if column <= 0:
        return float(np.mean(temperature_k)), float(np.mean(pressure_pa)), 0.0
    t_rep = float(np.trapezoid(w * temperature_k, altitude_m) / column)
    p_rep = float(np.trapezoid(w * pressure_pa, altitude_m) / column)
    return t_rep, p_rep, column


# ---------------------------------------------------------------------------
# MPI-Mainz UV/Vis cross sections (from the acquisition step)
# ---------------------------------------------------------------------------
def _read_two_column(path: str):
    """Read a 2-column (x, y) ASCII/CSV file, skipping any header lines.

    Returns (x, y) float arrays.  Auto-detects comma/whitespace delimiters and
    discards non-numeric leading lines (headers / provenance).
    """
    xs, ys = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#;%":
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue  # header row
            xs.append(x); ys.append(y)
    return np.asarray(xs), np.asarray(ys)


def _find_mpi_file(gas: str) -> str | None:
    if not os.path.isdir(MPI_DIR):
        return None
    cands = [f for f in os.listdir(MPI_DIR)
             if f.lower().startswith(gas.lower())
             and f.lower().endswith((".csv", ".txt", ".spc", ".dat"))]
    if not cands:
        return None
    # Prefer a file naming a stratospheric temperature (~220–230 K).
    def temp_key(f):
        import re
        m = re.search(r"(\d{3})\s*k", f.lower())
        return abs(int(m.group(1)) - 225) if m else 999
    cands.sort(key=temp_key)
    return os.path.join(MPI_DIR, cands[0])


def uvvis_cross_section(gas: str, wavelengths_m: np.ndarray) -> np.ndarray:
    """UV/Vis cross section [m^2/molecule] from an MPI-Mainz file (wavelength_nm, cm^2)."""
    wl_m = np.asarray(wavelengths_m, float)
    sigma = np.zeros_like(wl_m)
    path = _find_mpi_file(gas)
    if path is None:
        return sigma
    wl_nm, xs_cm2 = _read_two_column(path)
    if wl_nm.size == 0:
        return sigma
    lam_m = wl_nm * 1e-9
    order = np.argsort(lam_m)
    lam_s, xs_s = lam_m[order], xs_cm2[order] * CM2_TO_M2
    in_win = (wl_m >= lam_s[0]) & (wl_m <= lam_s[-1])
    sigma[in_win] = np.clip(np.interp(wl_m[in_win], lam_s, xs_s), 0.0, None)
    return sigma


# ---------------------------------------------------------------------------
# MT_CKD H2O continuum  (finalised against the acquired coefficient table)
# ---------------------------------------------------------------------------
_MTCKD_CACHE = None
_C2_CM_K = 1.4387769     # h c / k  [cm·K]
_MTCKD_T0 = 296.0        # reference temperature [K]
_MTCKD_P0_MB = 1013.0    # reference pressure [mb]


def _load_mt_ckd():
    """Load and cache the MT_CKD H2O continuum coefficient table."""
    global _MTCKD_CACHE
    if _MTCKD_CACHE is not None:
        return _MTCKD_CACHE
    path = os.path.join(MTCKD_DIR, "mt_ckd_h2o.csv")
    if not os.path.exists(path):
        _MTCKD_CACHE = None
        return None
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            parts = line.replace(",", " ").split()
            try:
                rows.append([float(parts[0]), float(parts[1]),
                             float(parts[2]), float(parts[4])])  # nu, self, frgn, texp
            except (ValueError, IndexError):
                continue
    arr = np.array(rows, float)
    order = np.argsort(arr[:, 0])
    _MTCKD_CACHE = arr[order]
    return _MTCKD_CACHE


def mt_ckd_h2o_continuum(wavelengths_m: np.ndarray,
                         temperature_k: float,
                         pressure_pa: float,
                         vmr_h2o: float) -> np.ndarray:
    """
    MT_CKD H2O self+foreign continuum as an *effective cross section per H2O
    molecule* [m^2], so it adds to the line cross section and scales with the
    H2O column.  Recipe (MT_CKD README; result in cm^2/molecule):

        RADTERM = nu * tanh(c2*nu/(2T))
        k_self  = self_coeff*(T0/T)^texp * RADTERM * (p_h2o/P0)*(T0/T)
        k_frgn  = foreign_coeff          * RADTERM * (p_frgn/P0)*(T0/T)

    with p in mb, P0 = 1013 mb, T0 = 296 K.  The (p/P0)(T0/T) factor carries one
    power of number density; multiplying by the H2O column supplies the second
    (so the self term is correctly quadratic in water vapour).
    """
    wl_m = np.asarray(wavelengths_m, float)
    out = np.zeros_like(wl_m)
    tab = _load_mt_ckd()
    if tab is None:
        return out
    nu_t, self_t, frgn_t, texp_t = tab[:, 0], tab[:, 1], tab[:, 2], tab[:, 3]
    nu = 1.0e-2 / wl_m                         # cm^-1
    in_rng = (nu >= nu_t[0]) & (nu <= nu_t[-1]) & (nu > 0)
    nu_in = nu[in_rng]
    sc = np.interp(nu_in, nu_t, self_t)
    fc = np.interp(nu_in, nu_t, frgn_t)
    te = np.interp(nu_in, nu_t, texp_t)
    radterm = nu_in * np.tanh(_C2_CM_K * nu_in / (2.0 * temperature_k))
    p_mb = pressure_pa / 100.0
    p_h2o = max(vmr_h2o, 0.0) * p_mb
    p_frgn = max(p_mb - p_h2o, 0.0)
    t_ratio = _MTCKD_T0 / temperature_k
    k_self = sc * t_ratio ** te * radterm * (p_h2o / _MTCKD_P0_MB) * t_ratio
    k_frgn = fc * radterm * (p_frgn / _MTCKD_P0_MB) * t_ratio
    out[in_rng] = (k_self + k_frgn) * CM2_TO_M2   # cm^2 -> m^2 per H2O molecule
    return out


# ---------------------------------------------------------------------------
# High-level: total cross section and column optical depth per gas
# ---------------------------------------------------------------------------
def gas_total_cross_section(gas: str,
                            wavelengths_m: np.ndarray,
                            temperature_k: float,
                            pressure_pa: float,
                            vmr_h2o: float = 0.0) -> np.ndarray:
    """
    Full-spectrum cross section [m^2/molecule] for one gas, summing every source
    that applies to it across 0.1–30 µm:
      * HITRAN IR lines      (if gas in HITRAN_GASES)
      * MPI-Mainz UV/Vis     (if gas in UVVIS_GASES)  — e.g. O3 gets both
      * MT_CKD continuum      (H2O only)
      * HITRAN .xsc           (CFCs)
    """
    wl_m = np.asarray(wavelengths_m, float)
    sigma = np.zeros_like(wl_m)
    if gas in CFC_GASES:
        return cfc_cross_section(gas, wl_m)
    if gas in HITRAN_GASES:
        sigma = sigma + hitran_cross_section(gas, wl_m, temperature_k, pressure_pa)
    if gas in UVVIS_GASES:
        sigma = sigma + uvvis_cross_section(gas, wl_m)
    if gas == "h2o":
        sigma = sigma + mt_ckd_h2o_continuum(wl_m, temperature_k, pressure_pa, vmr_h2o)
    return sigma


def gas_column_od(gas: str,
                  wavelengths_m: np.ndarray,
                  altitude_m: np.ndarray,
                  n_gas_m3: np.ndarray,
                  pressure_pa: np.ndarray,
                  temperature_k: np.ndarray,
                  alt_min_m: float = 16_000.0) -> np.ndarray:
    """
    Column optical depth spectrum for one gas, integrated from `alt_min_m` up.

    Cross section is evaluated once at the gas-column-weighted representative
    (T, P) (see module docstring) and scaled by the column above alt_min_m.
    """
    t_rep, p_rep, column = representative_tp(
        altitude_m, n_gas_m3, pressure_pa, temperature_k, alt_min_m)
    if column <= 0:
        return np.zeros_like(np.asarray(wavelengths_m, float))
    # Representative H2O VMR above alt_min (column-weighted) for the continuum.
    vmr_h2o = 0.0
    if gas == "h2o":
        n_air = np.asarray(pressure_pa, float) / (1.380649e-23 * np.asarray(temperature_k, float))
        vmr_prof = np.divide(np.asarray(n_gas_m3, float), n_air,
                             out=np.zeros_like(n_air), where=n_air > 0)
        w = np.where(np.asarray(altitude_m, float) >= alt_min_m, np.asarray(n_gas_m3, float), 0.0)
        vmr_h2o = float(np.trapezoid(w * vmr_prof, altitude_m) / column) if column > 0 else 0.0
    sigma = gas_total_cross_section(gas, wavelengths_m, t_rep, p_rep, vmr_h2o)
    return sigma * column


# ---------------------------------------------------------------------------
# AFGL volume-mixing-ratio profiles -> number densities on the model grid
# ---------------------------------------------------------------------------
# Representative stratospheric volume mixing ratios, used only for gases the
# AFGL file does not provide (or if no AFGL file was acquired).  Order-of-
# magnitude lower-stratosphere values; documented so they can be refined.
FALLBACK_VMR = {
    "h2o": 5.0e-6, "co2": 420e-6, "o3": 5.0e-6, "n2o": 250e-9, "co": 30e-9,
    "ch4": 1.2e-6, "no2": 5e-9, "hno3": 6e-9, "so2": 1e-10, "o2": 0.2095,
    "cfc11": 230e-12, "cfc12": 520e-12,
}

# Map our gas keys to the exact column *base token* (text before the first '_',
# e.g. 'co2_cm3' -> 'co2') that may appear in AFGL CSV column names.  Base-token
# matching avoids 'o2' spuriously matching 'co2_cm3'/'no2_cm3'.
_AFGL_ALIASES = {
    "h2o": ("h2o", "water"), "co2": ("co2",), "o3": ("o3", "ozone"),
    "n2o": ("n2o",), "co": ("co",), "ch4": ("ch4", "methane"),
    "no2": ("no2",), "hno3": ("hno3",), "o2": ("o2", "oxygen"), "so2": ("so2",),
    "cfc11": ("f11", "cfc11", "ccl3f"),
    "cfc12": ("f12", "cfc12", "ccl2f2"),
}


def _find_afgl_file() -> str | None:
    if not os.path.isdir(AFGL_DIR):
        return None
    cands = [f for f in os.listdir(AFGL_DIR) if f.lower().endswith((".csv", ".dat", ".txt"))]
    if not cands:
        return None
    # Prefer the US-standard file (consistent with us_standard_atmosphere).
    cands.sort(key=lambda f: (0 if "us" in f.lower() else 1, f))
    return os.path.join(AFGL_DIR, cands[0])


def load_afgl_columns():
    """
    Load the AFGL profile file as (z_m, header_names, data_columns_dict).
    Number densities are converted to VMR using the file's own air column.
    Returns dict: {'z_m':…, 'air_cm3':…, '<gas>_vmr': vmr_array, …} or None.
    """
    path = _find_afgl_file()
    if path is None:
        return None
    with open(path) as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    # Locate header (first line with any alphabetic token), data below.
    header = lines[0].replace(",", " ").split()
    rows = []
    for ln in lines[1:]:
        parts = ln.replace(",", " ").split()
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    arr = np.array(rows, dtype=float)
    cols = {name.lower(): arr[:, i] for i, name in enumerate(header) if i < arr.shape[1]}

    def col_like(*subs):
        for key in cols:
            if any(s in key for s in subs):
                return cols[key]
        return None

    def col_base(*tokens):
        """Match by exact base token (text before first '_')."""
        for key in cols:
            if key.split("_")[0] in tokens:
                return cols[key]
        return None

    z = col_like("z_km", "z", "alt", "height")
    z_m = z * 1000.0 if z is not None and np.nanmax(z) < 200 else z  # km->m if needed
    air = col_like("air", "ntot", "density")
    out = {"z_m": z_m, "air_cm3": air, "_path": path, "_header": header}
    if air is not None:
        for gas, subs in _AFGL_ALIASES.items():
            g = col_base(*subs)
            if g is not None:
                out[f"{gas}_vmr"] = np.divide(g, air, out=np.zeros_like(g),
                                              where=air > 0)
    return out


def number_density_profiles(altitude_m: np.ndarray,
                            air_n_m3: np.ndarray,
                            gases=None) -> dict:
    """
    Build {gas: number_density[m^-3]} on `altitude_m`, using AFGL VMRs where
    available and FALLBACK_VMR otherwise.  n_gas = VMR(z) * air_n_m3(z), so the
    profiles are consistent with the supplied (model) air density.
    """
    if gases is None:
        gases = list(FALLBACK_VMR.keys())
    afgl = load_afgl_columns()
    out = {}
    for gas in gases:
        vmr_arr = None
        if afgl is not None and f"{gas}_vmr" in afgl and afgl["z_m"] is not None:
            zc = afgl["z_m"]; vc = afgl[f"{gas}_vmr"]
            order = np.argsort(zc)
            vmr = np.interp(altitude_m, zc[order], vc[order],
                            left=vc[order][0], right=0.0)
            vmr_arr = vmr
        if vmr_arr is None:
            vmr_arr = np.full_like(altitude_m, FALLBACK_VMR.get(gas, 0.0), dtype=float)
        out[gas] = vmr_arr * air_n_m3
    return out
