"""
plot_ace_atlas_paper.py — Paper-1 version of the ACE-FTS atlas figure.

Round-10: one panel only (the 16-20 km tropical bin) instead of two.
Round-11 (Doron): recast in the spirit of the Notes figure
slant_od_trace_gases.png --- slant optical depth rather than
transmittance, at the Delta lambda = 0.02 um sampling of paper Fig. 2
rather than the 0.25 um design element, with the individual trace gases
labelled, and with the 1-Tg silica and elevated-sulfate signals overlaid
so the measurement can be read against what it must detect.  The
unscaled climatology is dropped.

Round-12 (Doron): the native-sampling atlas spectrum is drawn underneath
as well, so the individual lines are visible, and the axes are cut to
8.0-9.5 um and OD <= 10 to show the silica and O3 elements at scale.

Physics and data identical to analyze_ace_atlas.py, which keeps producing
the internal two-panel Notes version; this script reuses that module's
loader, cached forward model (already computed at 0.02 um elements), and
O3-element fit.

Numbers printed (2026-08-27 run), 16-20 km tropical bin, 18 km tangent:
  9.2 um element   measured T = 0.5649 -> O3 scale s = 0.361
  8.74 um element  measured OD 0.459, model(scaled) 0.497, residual -0.038
  8.74 um element  1-Tg silica OD 0.288; elevated sulfate 0.073;
                   quiet sulfate 0.025
  resolution note  at the 8.74 um element the measured <tau> = 0.531 but
                   -ln<T> = 0.459, i.e. the element average and the
                   transmission-effective OD differ by ~14% even here --
                   the micro-window effect a line-resolving instrument
                   exploits (Sect. 2.4, Sect. 3.3, Sect. 4).

Round-13 (Doron, "the scaling does not look perfect by eye"): additional
diagnostics for the visible offset between the red model and the black
measured curve INSIDE the green 9.2 um scaling band.  The fit matches the
element-mean transmittance <T> exactly (that is the observable the
co-retrieval works in); the figure's y axis is the element-average OD at
0.02 um sampling, where the measured curve sits ~0.08 OD above the
gas-only model.  Decomposition (2026-08-31 run): ~0.05 OD is the real
aerosol continuum (elevated-sulfate slant OD in the band), present in the
measurement and deliberately absent from the gas-only model; the rest is
the <T>-vs-<tau> averaging gap on a co-added atlas (line cores are
flattened in a mean of 3967 variable atmospheres, so matching <T> leaves
the OD-average low).  Also printed: the atlas' effective transmittance
floor above 9.45 um (1st-percentile T ~ 2.1e-3 -> OD ceiling ~ 6.2, where
the measured black curve flattens in the saturated O3 core), and the
measured 5th-percentile OD of the resolved micro-windows at 8.15-8.35 um
(0.11, against gas-model quasi-continuum 0.04 + elevated-sulfate 0.09):
the lower envelope of the resolved lines is the physical continuum
(aerosol + gas far wings), three orders of magnitude above the
instrumental floor.

Round-27 (Doron): the aerosol background overlaid is the CALIBRATED
two-component sulfate background of Paper 1 Sect. 2.2 (imported
bit-identically from scripts/calibrated_background_thresholds.py) at the
atlas bin's 18-km tangent, replacing the superseded single-mode elevated
curve; its element OD is printed together with the fine/coarse split.
The atlas is distributed in five climate zones only (Arctic summer/winter,
mid-latitude summer/winter, tropics), so a 20-25N bin cannot be formed.

Writes figures/scisat_ace_atlas_874.png (file name kept;
round 43: the element is 8.80 um, see analyze_ace_atlas.SILICA_UM).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_ace_atlas import (  # noqa: E402
    SILICA_UM,
    load_atlas, model_tau_by_gas, fit_o3_at_element, band_average,
    sulfate_continuum_od, BAND_LO_UM, BAND_HI_UM, O3FIT_LO_UM, O3FIT_HI_UM,
    BINS,
)
from saimon.geometry import tangent_to_slant_paths  # noqa: E402
from saimon.sai import create_silica_sai_layer  # noqa: E402
from saimon.materials import SilicaRefractiveIndex  # noqa: E402
from saimon.backgrounds import make_empirical_column  # noqa: E402
from saimon.refractive_index import sulfuric_acid_at_temperature  # noqa: E402
from calibrated_background_thresholds import (  # noqa: E402
    ALT as ALT_CAL, CAL_QUIET, member_ri, glossac_2025N_quiet_profile,
    TwoComponentBackground)

plt.rcParams.update({   # round 44 (Doron): larger fonts, heavier lines
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 13,
})

ELEM_UM = 0.02          # sampling of the figure (paper Fig. 2)
WL_LO, WL_HI = 8.0, 9.5
LABEL = "16-20 km"      # the only bin shown in the paper
# Gases worth naming in this window; the rest are below the frame.
GAS_LABEL = {"o3": r"O$_3$", "n2o": r"N$_2$O", "cfc12": "CFC-12",
             "co2": r"CO$_2$"}
GAS_COLOR = {"o3": "tab:orange", "n2o": "tab:purple", "cfc12": "tab:brown",
             "h2o": "tab:cyan", "co2": "tab:olive", "hno3": "tab:pink"}


def aerosol_slant_od(htan_km, wl_um):
    """Slant OD of a 1-Tg silica layer and of the CALIBRATED two-component
    sulfate background (total, fine, coarse; round 27), plus the superseded
    elevated single-mode curve for the printed comparison only, on the same
    chord geometry as the gas model."""
    alt = ALT_CAL
    chord = tangent_to_slant_paths(np.array([htan_km * 1e3]), alt)[0]
    sil = create_silica_sai_layer(
        alt, target_mass_tg=1.0, rmed_nm=268.0, sigma=1.31,
        refractive_index=SilicaRefractiveIndex(
            "data/optics/SiO2_KitamuraPopova.yml"))
    ri, wt, T = member_ri("LM65T223")
    bg = TwoComponentBackground(glossac_2025N_quiet_profile(), CAL_QUIET,
                                ri, wt, T)
    a_sil = sil.extinction_profile_m1(wl_um * 1e-6)
    a_f = bg.column(0).extinction_profile_m1(wl_um * 1e-6)
    a_c = bg.column(1).extinction_profile_m1(wl_um * 1e-6)
    elev = make_empirical_column(
        alt, "elevated", 100.0, 1.6,
        refractive_index=sulfuric_acid_at_temperature(215.0))
    a_el = elev.extinction_profile_m1(wl_um * 1e-6)
    return (chord @ a_sil), (chord @ a_f + chord @ a_c), (chord @ a_f), \
        (chord @ a_c), (chord @ a_el)


def bin_to_elements(wl, y, lo, hi, width):
    """Average y(wl) into contiguous elements of the given width."""
    edges = np.arange(lo, hi + width / 2, width)
    cen = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(len(cen), np.nan)
    order = np.argsort(wl)
    wl, y = np.asarray(wl)[order], np.asarray(y)[order]
    j = np.searchsorted(wl, edges)
    for i in range(len(cen)):
        if j[i + 1] > j[i]:
            out[i] = float(np.mean(y[j[i]:j[i + 1]]))
    return cen, out


def main():
    info = BINS[LABEL]
    htan = info["htan_km"]
    # exactly the grid analyze_ace_atlas.py caches the Voigt step on
    wl_model = np.arange(8.0, 10.531, 0.01)

    nu, tr = load_atlas(info["fname"])
    wl_data = 1e4 / nu

    taus = model_tau_by_gas(htan, wl_model)
    tau_o3 = taus["o3"]
    tau_other = sum(t for g, t in taus.items() if g != "o3")
    s_o3 = fit_o3_at_element(wl_model, tau_o3, tau_other, nu, tr,
                             O3FIT_LO_UM, O3FIT_HI_UM)

    # measured slant OD: native sampling and figure sampling
    tau_data = -np.log(np.clip(tr, 1e-12, None))
    wl_e, tau_e = bin_to_elements(wl_data, tau_data, WL_LO, WL_HI, ELEM_UM)

    # model: scaled O3 + the other gases, same sampling
    tau_tot = s_o3 * tau_o3 + tau_other
    _, tau_tot_e = bin_to_elements(wl_model, tau_tot, WL_LO, WL_HI, ELEM_UM)

    tau_sil, tau_sulf, tau_f, tau_c, tau_elev = aerosol_slant_od(htan,
                                                                 wl_model)
    _, sil_e = bin_to_elements(wl_model, tau_sil, WL_LO, WL_HI, ELEM_UM)
    _, sulf_e = bin_to_elements(wl_model, tau_sulf, WL_LO, WL_HI, ELEM_UM)

    fig, ax = plt.subplots(figsize=(11, 6.0))

    # raw atlas at native sampling, so the individual lines are visible
    ax.plot(wl_data, tau_data, lw=0.25, color="0.72", zorder=1,
            label=r"ACE-FTS atlas, native 0.02 cm$^{-1}$ sampling")

    ax.axvspan(BAND_LO_UM, BAND_HI_UM, color="magenta", alpha=0.13, zorder=0)
    ax.axvspan(O3FIT_LO_UM, O3FIT_HI_UM, color="green", alpha=0.09, zorder=0)

    # individual gases, labelled at their in-window maxima
    for g, lab in GAS_LABEL.items():
        if g not in taus:
            continue
        _, ge = bin_to_elements(wl_model, taus[g] * (s_o3 if g == "o3" else 1.0),
                                WL_LO, WL_HI, ELEM_UM)
        if np.nanmax(ge) < 3e-3:
            continue
        ax.plot(wl_e, ge, lw=2.0, ls=":", color=GAS_COLOR[g], alpha=0.9,
                zorder=2)
        # round 45 (Doron): labels at prescribed wavelengths, below the line
        # round 46 (Doron): O3 a little right; N2O a little left and above the silica line
        LABEL_AT = {"cfc12": (9.0, (0, -18))}
        if g == "o3":    # round 48 (Doron): ~0.05 OD higher than the previous placement
            ax.text(9.06, 0.19, lab, ha="center", va="bottom", fontsize=14,
                    fontweight="bold", color=GAS_COLOR[g], zorder=6)
            continue
        if g == "n2o":   # explicit placement: left of the N2O peak, just above the silica line
            # round 47 (Doron): in the white window between the silica/sulfate crossing and the native atlas
            # round 48 (Doron): the window around 8.5 um
            ax.text(8.50, 0.135, lab, ha="center", va="bottom", fontsize=14,
                    fontweight="bold", color=GAS_COLOR[g], zorder=6)
            continue
        if g in LABEL_AT:
            k = int(np.nanargmin(np.abs(wl_e - LABEL_AT[g][0])))
            xy_off = LABEL_AT[g][1]
        else:
            k = int(np.nanargmax(np.where(wl_e < 9.05, ge, np.nan)))
            xy_off = (0, 4)
        ax.annotate(lab, (wl_e[k], ge[k]), textcoords="offset points",
                    xytext=xy_off, ha="center", va="top" if xy_off[1] < 0 else "bottom",
                    fontsize=14, fontweight="bold", color=GAS_COLOR[g], zorder=6)

    ax.plot(wl_e, tau_tot_e, lw=2.2, color="tab:red", zorder=3,
            label=rf"model, trace gases (O$_3$ scaled at 9.2 $\mu$m, "
                  rf"$s={s_o3:.2f}$)")
    ax.plot(wl_e, tau_e, lw=2.3, color="black", zorder=4,
            label=f"ACE-FTS atlas, measured ({info['nspec']} co-added "
                  "spectra)")
    ax.plot(wl_e, sil_e, lw=2.6, color="tab:blue", zorder=5,
            label="silica, 1 Tg")
    ax.plot(wl_e, sulf_e, lw=2.2, color="tab:green", ls=(0, (5, 2)),
            zorder=5, label="sulfate, calibrated background")

    ax.set_yscale("log")
    ax.set_xlim(WL_LO, WL_HI)
    ax.set_ylim(4e-3, 10)
    ax.set_xlabel(r"Wavelength [$\mu$m]")
    ax.set_ylabel(f"Slant optical depth at a {htan:.0f} km tangent")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower left", framealpha=0.95)
    ax.text(np.mean([BAND_LO_UM, BAND_HI_UM]), 4.0,   # round 45 (Doron): inside the box
            rf"{SILICA_UM:.2f} $\mu$m" "\n" "silica", ha="center", va="top", fontsize=14,
            color="magenta")
    ax.text(np.mean([O3FIT_LO_UM, O3FIT_HI_UM]), 4.0,
            r"9.2 $\mu$m" "\n" r"O$_3$ scaling", ha="center", va="top", fontsize=14,
            color="green")

    fig.tight_layout()
    out = _HERE / "figures" / "scisat_ace_atlas_874.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved -> {out}")

    # ── numbers quoted in the text ────────────────────────────────────────
    t_data = band_average(nu, tr, BAND_LO_UM, BAND_HI_UM)
    t_data_92 = band_average(nu, tr, O3FIT_LO_UM, O3FIT_HI_UM)
    mb = (wl_model >= BAND_LO_UM) & (wl_model <= BAND_HI_UM)
    t_fit = float(np.exp(-tau_tot[mb]).mean())
    tau_sulf_quiet = sulfate_continuum_od(htan, np.array([SILICA_UM]))[0]
    print(f"\n--- {LABEL} (h_tan = {htan} km, {info['nspec']} spectra) ---")
    print(f"  9.2 um element: measured T = {t_data_92:.4f} -> O3 scale "
          f"s = {s_o3:.3f}")
    print(f"  {SILICA_UM:g} um element: measured OD = {-np.log(t_data):.4f}, "
          f"model(scaled) = {-np.log(t_fit):.4f}, "
          f"residual = {-np.log(t_data) + np.log(t_fit):+.4f}")
    print(f"  {SILICA_UM:g} um element: 1-Tg silica OD = "
          f"{float(np.mean(tau_sil[mb])):.4f}; calibrated two-component "
          f"sulfate OD = {float(np.mean(tau_sulf[mb])):.4f} (fine "
          f"{float(np.mean(tau_f[mb])):.4f} + coarse "
          f"{float(np.mean(tau_c[mb])):.4f}); silica/background = "
          f"{float(np.mean(tau_sil[mb]))/float(np.mean(tau_sulf[mb])):.2f}; "
          f"[superseded: elevated single-mode {float(np.mean(tau_elev[mb])):.4f}, "
          f"quiet preset {tau_sulf_quiet:.4f}]")
    # resolution-dependence: band-average vs transmission-effective OD
    for lo, hi, name in [(BAND_LO_UM, BAND_HI_UM, f"{SILICA_UM:g} um")]:
        m = (wl_data >= lo) & (wl_data <= hi)
        tau_bar = float(np.mean(tau_data[m]))
        tau_eff = float(-np.log(np.mean(tr[m])))
        print(f"  {name} element, measured: <tau> = {tau_bar:.3f} but "
              f"-ln<T> = {tau_eff:.3f} (micro-window effect)")

    # ── round-13 diagnostics (see docstring) ─────────────────────────────
    # (a) inside the green scaling band: <T> matches by construction, but
    # the plotted element-average OD does not -- decompose the gap.
    mg = (wl_model >= O3FIT_LO_UM) & (wl_model <= O3FIT_HI_UM)
    wl_ge, tau_ge_dat = bin_to_elements(wl_data, tau_data,
                                        O3FIT_LO_UM, O3FIT_HI_UM, ELEM_UM)
    _, tau_ge_mod = bin_to_elements(wl_model, tau_tot,
                                    O3FIT_LO_UM, O3FIT_HI_UM, ELEM_UM)
    ok = ~np.isnan(tau_ge_dat) & ~np.isnan(tau_ge_mod)
    print(f"  9.2 um band: model <T> = "
          f"{float(np.exp(-tau_tot[mg]).mean()):.4f} (= measured, fitted); "
          f"element-average OD measured {np.mean(tau_ge_dat[ok]):.3f} vs "
          f"gas-only model {np.mean(tau_ge_mod[ok]):.3f}")
    print(f"  9.2 um band: calibrated-background continuum OD = "
          f"{float(np.mean(tau_sulf[mg])):.4f} (in the measurement, "
          "not in the gas-only model)")
    # (b) the saturated-core ceiling: the atlas' effective T floor
    msat = wl_data > 9.45
    t1 = float(np.percentile(tr[msat], 1))
    print(f"  above 9.45 um: 1st-percentile measured T = {t1:.2e} "
          f"-> OD ceiling ~ {-np.log(t1):.1f}")
    # (c) the lower envelope of the resolved lines: physical continuum
    for lo, hi in [(8.15, 8.35)]:
        md = (wl_data >= lo) & (wl_data <= hi)
        mm = (wl_model >= lo) & (wl_model <= hi)
        print(f"  {lo}-{hi} um micro-windows: measured 5th-pct OD "
              f"{np.percentile(tau_data[md], 5):.3f}; gas-model "
              f"quasi-continuum min {tau_tot[mm].min():.3f}; "
              f"calibrated sulfate {float(np.mean(tau_sulf[mm])):.3f}")


if __name__ == "__main__":
    main()
