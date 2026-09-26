"""
analyze_ace_atlas.py -- compare the public ACE-FTS Atmospheric Atlas (Hughes,
Bernath & Boone 2014; databace.scisat.ca/aceatlas/) against the sage3 forward
model around the 8.74 um silica reststrahlen channel (Config B+).

The atlas provides mission-averaged limb *transmittance* spectra in 4-km
tangent-height bins for five climate zones.  We use the Tropical bins that
bracket the z = 20 km analysis altitude:
    16--20 km (3967 spectra, <h_tan> = 18.0 km, <T> = 199 K)
    20--24 km (3146 spectra, <h_tan> = 21.9 km, <T> = 211 K)

Outputs
  figures/scisat_ace_atlas_874.png
      Measured tropical limb transmittance 8--10.5 um at the two bins, with
      the sage3 gas-only slant-transmittance model overlaid (climatological
      US-standard profiles, and with a single fitted O3 amplitude) and the
      B+ 0.25-um element marked.
  printed numbers:
      * band-averaged transmittance over the B+ element (data vs model),
      * the fitted O3 amplitude s and the post-fit residual OD -- a
        1-parameter demonstration of the Sec.-5 co-retrieval: the
        midlatitude climatological O3 profile overestimates the tropical
        band OD by ~0.6, and one retrieved O3 scalar closes the gap,
      * slant OD attributed to the sulfate aerosol continuum (quiet bg).

Note: a noise/SNR estimate from the atlas itself is NOT attempted -- the
high-frequency structure of the averaged spectra is real (resolved lines),
not noise, so any rms-based estimate is dominated by spectroscopy.  The
per-sample SNR ~300 comes from the literature (Bernath 2017).

Run from the repo root:  python scripts/analyze_ace_atlas.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from saimon.atmosphere import us_standard_atmosphere
from saimon.geometry import tangent_to_slant_paths
from saimon import trace_gases as tg
from saimon.backgrounds import make_empirical_column
from saimon.refractive_index import sulfuric_acid_at_temperature

ATLAS_DIR = _HERE / "data" / "ace_atlas"   # 950-1250 cm^-1 extract of the public atlas (fetch/reduce_ace_atlas.py)

# Wavenumber window: 8--10.5 um
NU_MIN, NU_MAX = 950.0, 1250.0
# The design resolution element: boxcar centred on the silica reference
# element.  Round 37 (Doron): 0.1 um, the paper's single element convention
# (was 0.25).  Round 43 (Doron): centre moved from 8.74 to 8.80 um, the
# extinction peak of the adopted (Kitamura+Popova) optical constants.
SILICA_UM = 8.80
ELEMENT_UM = 0.10
BAND_LO_UM, BAND_HI_UM = SILICA_UM - ELEMENT_UM / 2, SILICA_UM + ELEMENT_UM / 2
# The O3 co-retrieval element (Sec. 5 uses a 9.2-um IR channel)
O3FIT_LO_UM, O3FIT_HI_UM = 9.2 - ELEMENT_UM / 2, 9.2 + ELEMENT_UM / 2

# Gases and mean tangent conditions of the two tropical atlas bins
GASES = ["o3", "h2o", "co2", "n2o", "ch4", "hno3", "no2", "so2", "co",
         "cfc11", "cfc12"]
BINS = {
    "16-20 km": dict(fname="Tropics_016-020km.txt", htan_km=18.0, nspec=3967),
    "20-24 km": dict(fname="Tropics_020-024km.txt", htan_km=21.9, nspec=3146),
}
# ACE-FTS true spectral point spacing (0.02 cm^-1 resolution); atlas files are
# oversampled to 0.0025 cm^-1, so adjacent atlas points are NOT independent.
ACE_RES_CM = 0.02
ATLAS_STEP_CM = 0.0025


def load_atlas(fname):
    p = ATLAS_DIR / fname
    if not p.exists() and (ATLAS_DIR / (fname + ".gz")).exists():
        p = ATLAS_DIR / (fname + ".gz")
    dat = np.loadtxt(p, skiprows=9)
    nu, tr = dat[:, 0], dat[:, 1]
    m = (nu >= NU_MIN) & (nu <= NU_MAX)
    return nu[m], tr[m]


def model_tau_by_gas(htan_km, wl_um):
    """Per-gas slant OD on wl_um at 0.02-um resolution elements.

    Returns dict gas -> tau(wl); climatological (US-standard) profiles.
    Cached to outputs/scisat_atlas/ (the Voigt evaluation costs ~10 min)."""
    cache = _HERE / "outputs" / "scisat_atlas" / f"taus_h{htan_km:.1f}.npz"
    if cache.exists():
        dat = np.load(cache)
        if np.array_equal(dat["wl_um"], wl_um):
            return {g: dat[g] for g in GASES}
    alt_m = np.arange(14_000.0, 80_001.0, 500.0)
    atm = us_standard_atmosphere(alt_m)
    chord = tangent_to_slant_paths(np.array([htan_km * 1e3]), alt_m)[0]
    n_prof = tg.number_density_profiles(alt_m, atm.number_density_m3,
                                        gases=GASES)
    node_km = np.array([16.0, 20.0, 26.0, 36.0])
    node_m = node_km * 1e3
    node_atm = us_standard_atmosphere(node_m)
    wl_m = wl_um * 1e-6

    taus = {}
    for gas in GASES:
        sig_nodes = np.zeros((len(node_km), len(wl_um)))
        for i in range(len(node_km)):
            T = float(node_atm.temperature_k[i])
            P = float(node_atm.pressure_pa[i])
            k = int(np.argmin(np.abs(alt_m - node_m[i])))
            vmr = (float(n_prof["h2o"][k] / atm.number_density_m3[k])
                   if gas == "h2o" else 0.0)
            sig_nodes[i] = tg.gas_xsec_rm(gas, wl_m, T, P, fwhm_um=0.02,
                                          vmr_h2o=vmr, nu_pad_cm=300.0)
        tau = np.zeros_like(wl_um)
        # interpolate sigma(z) per wavelength, then slant-integrate
        for j in range(len(wl_um)):
            sig_z = np.interp(alt_m, node_m, sig_nodes[:, j],
                              left=sig_nodes[0, j], right=sig_nodes[-1, j])
            tau[j] = float(np.sum(chord * sig_z * n_prof[gas]))
        taus[gas] = tau
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, wl_um=wl_um, **taus)
    return taus


def fit_o3_at_element(wl_um, tau_o3, tau_other, nu_data, tr_data,
                      lo_um, hi_um):
    """Solve for the O3 amplitude s that reproduces the measured
    band-averaged transmittance over the element [lo_um, hi_um]:
        <exp(-(tau_other + s*tau_o3))> = <T_data>.
    Monotonic in s -> bisection.  This is the 1-parameter analogue of the
    Sec.-5 co-retrieval: O3 is constrained by an O3-dominated element and
    then *predicts* the B+ element out-of-band."""
    t_target = band_average(nu_data, tr_data, lo_um, hi_um)
    m = (wl_um >= lo_um) & (wl_um <= hi_um)

    def band_model(s):
        return float(np.exp(-(tau_other[m] + s * tau_o3[m])).mean())

    s_lo, s_hi = 0.0, 3.0
    if band_model(s_hi) > t_target:   # even s=3 too transparent (shouldn't be)
        return s_hi
    for _ in range(60):
        s_mid = 0.5 * (s_lo + s_hi)
        if band_model(s_mid) > t_target:
            s_lo = s_mid
        else:
            s_hi = s_mid
    return 0.5 * (s_lo + s_hi)


def sulfate_continuum_od(htan_km, wl_um):
    """Slant OD of the quiet-background sulfate layer at wl_um."""
    ALT = np.linspace(0, 60_000, 121)
    ri = sulfuric_acid_at_temperature(215.0)
    col = make_empirical_column(ALT, "quiet", 80.0, 1.6, refractive_index=ri)
    alpha = col.extinction_profile_m1(wl_um * 1e-6)  # (nz, nwl)
    chord = tangent_to_slant_paths(np.array([htan_km * 1e3]), ALT)[0]
    return np.array([float(np.sum(chord * alpha[:, j]))
                     for j in range(len(wl_um))])


def boxcar_um(nu_cm, y, width_um):
    """Boxcar-average y(nu) over a wavelength width width_um at each point."""
    wl = 1e4 / nu_cm
    out = np.full_like(y, np.nan)
    # walk in coarse steps to keep this cheap
    step = max(1, int(len(nu_cm) / 4000))
    idx = np.arange(0, len(nu_cm), step)
    for i in idx:
        m = np.abs(wl - wl[i]) <= width_um / 2
        out[i] = y[m].mean()
    return idx, out[idx]


def band_average(nu_cm, y, lo_um, hi_um):
    wl = 1e4 / nu_cm
    m = (wl >= lo_um) & (wl <= hi_um)
    return float(y[m].mean())


def main():
    wl_um_model = np.arange(8.0, 10.531, 0.01)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)

    print("=== ACE-FTS Atmospheric Atlas (tropics) vs sage3 forward model ===")
    for ax, (label, info) in zip(axes, BINS.items()):
        nu, tr = load_atlas(info["fname"])
        wl_data = 1e4 / nu

        # data: raw (light) + 0.25-um boxcar (heavy)
        ax.plot(wl_data, tr, lw=0.3, color="0.75",
                label=f"ACE atlas {label} ({info['nspec']} spectra)")
        idx, tr_box = boxcar_um(nu, tr, ELEMENT_UM)
        ax.plot(wl_data[idx], tr_box, lw=1.8, color="tab:blue",
                label=f"data, {ELEMENT_UM:g} µm boxcar")

        # model: climatological, and with one O3 amplitude retrieved from
        # the O3-dominated 9.2-um element (the Sec.-5 co-retrieval channel)
        taus = model_tau_by_gas(info["htan_km"], wl_um_model)
        tau_o3 = taus["o3"]
        tau_other = sum(t for g, t in taus.items() if g != "o3")
        tr_clim = np.exp(-(tau_other + tau_o3))
        s_o3 = fit_o3_at_element(wl_um_model, tau_o3, tau_other, nu, tr,
                                 O3FIT_LO_UM, O3FIT_HI_UM)
        tr_fit = np.exp(-(tau_other + s_o3 * tau_o3))
        ax.plot(wl_um_model, tr_clim, lw=1.1, color="tab:red", alpha=0.6,
                ls="--",
                label="sage3 model, climatological (midlat) O$_3$")
        ax.plot(wl_um_model, tr_fit, lw=1.4, color="tab:red",
                label=f"sage3 model, O$_3$ retrieved at 9.2 µm "
                      f"(s = {s_o3:.2f})")

        ax.axvspan(BAND_LO_UM, BAND_HI_UM, color="magenta", alpha=0.15,
                   label=f"design {ELEMENT_UM:g} µm element @ {SILICA_UM:g} µm")
        ax.axvspan(O3FIT_LO_UM, O3FIT_HI_UM, color="green", alpha=0.10,
                   label="O$_3$ co-retrieval element @ 9.2 µm")
        ax.set_ylabel("limb transmittance")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title(f"Tropical {label} tangent bin "
                     f"(mean tangent height {info['htan_km']} km)",
                     fontsize=10)

        # numbers
        t_data = band_average(nu, tr, BAND_LO_UM, BAND_HI_UM)
        t_data_92 = band_average(nu, tr, O3FIT_LO_UM, O3FIT_HI_UM)
        mwl = (wl_um_model >= BAND_LO_UM) & (wl_um_model <= BAND_HI_UM)
        t_clim = float(tr_clim[mwl].mean())
        t_fit = float(tr_fit[mwl].mean())
        tau_sulf = sulfate_continuum_od(info["htan_km"],
                                        np.array([SILICA_UM]))[0]
        print(f"\n--- {label} (h_tan = {info['htan_km']} km) ---")
        print(f"  9.2-um element (O3 co-retrieval): data T = {t_data_92:.4f}"
              f"  ->  fitted O3 amplitude s = {s_o3:.3f} "
              f"(climatological midlat = 1.0)")
        print(f"  {SILICA_UM:g}-um B+ element, band-avg transmittance:")
        print(f"    data = {t_data:.4f}   climatological model = {t_clim:.4f}"
              f"   O3-retrieved model = {t_fit:.4f}")
        print(f"  implied slant OD:  data = {-np.log(t_data):.4f}   "
              f"clim = {-np.log(t_clim):.4f}   "
              f"O3-retrieved = {-np.log(t_fit):.4f}")
        print(f"  out-of-band prediction residual at {SILICA_UM:g} um: "
              f"{-np.log(t_data)+np.log(t_fit):+.4f} OD")
        print(f"  quiet-bg sulfate continuum slant OD at {SILICA_UM:g} um: "
              f"{tau_sulf:.4f}")

    axes[-1].set_xlabel("wavelength [µm]")
    axes[-1].set_xlim(8.0, 10.5)
    fig.suptitle("ACE-FTS measured tropical limb transmittance vs sage3 model "
                 f"around the B$^+$ {SILICA_UM:g} µm channel", fontsize=12)
    fig.tight_layout()
    out = _HERE / "figures" / "scisat_ace_atlas_874.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
