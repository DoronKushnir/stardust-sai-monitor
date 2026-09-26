"""
plot_slant_od_landscape_paper.py -- Paper-1 introduction figure: the slant-path
optical-depth landscape of the SAI monitoring problem at a 20 km tangent.

Round 35 (2026-09-24, the "de-silica-bias" restructure): a modified version of
plot_slant_od_trace_gases.py (Notes / former Appendix app:tracegas figure),
per Doron's request:
  * sulfate background = the CALIBRATED two-component background of
    Sect. sec:calibrated_background (CAL_QUIET modes on the GloSSAC 20-25N
    quiet-epoch 525-nm profile, LM65T223 member optics) -- identical assembly
    to plot_spectral_shapes_paper.py, on this script's 14-80 km grid;
  * trace gases = the total only (no per-species curves, no peak labels),
    from the same saimon.trace_gases path as every gas budget in the paper
    (HITRAN2024 lines, measured CFC-11/12 cross sections, MT_CKD H2O
    continuum, MPI-Mainz UV/Vis), AFGL US-standard profiles;
  * all four candidate SAI materials at 1 Tg: silica (Kitamura/Popova,
    r_med = 268 nm, sigma = 1.31), and calcite / dolomite / alumina at the
    same PSD (ray-arithmetic-mean optics; the saimon.sai factories);
  * the panel split moved from 2 to 3 um (visible/NIR: 0.25-3 um at a 2 nm
    element; MIR: 3-30 um at a 0.1 um element), panels stacked at the
    12 cm figure* print width;
  * the Wrana et al. (2021) size-retrieval triple (448/756/1544 nm)
    highlighted (dashed lines, labelled in-panel; round 35b: no legend entry,
    all four SAI materials drawn solid); the ACE-FTS (SCISAT) mid-infrared coverage (to 13.3 um)
    marked; no band shading, no annotation text, no titles.

Writes figures/slant_od_landscape.png and prints the per-
material band peaks (center, 1 Tg slant OD, gas OD at the peak) for the
caption/text.  The Voigt cross sections are cached on disk (first run of a
new grid is slow).

Run from the repo root:  python scripts/plot_slant_od_landscape_paper.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

_THIS = Path(__file__).resolve()
_PARENT = _THIS.parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

from saimon.atmosphere import us_standard_atmosphere, rayleigh_cross_section_m2  # noqa: E402
from saimon.geometry import tangent_to_slant_paths  # noqa: E402
from saimon import trace_gases as tg  # noqa: E402
from saimon.sai import (create_silica_sai_layer, create_calcite_sai_layer,  # noqa: E402
                       create_dolomite_sai_layer, create_alumina_sai_layer)
from saimon.backgrounds import make_profile_column  # noqa: E402
from saimon.materials import SilicaRefractiveIndex  # noqa: E402
from calibrated_background_thresholds import (  # noqa: E402
    ALT as CAL_ALT, CAL_QUIET, member_ri, glossac_2025N_quiet_profile)

TANGENT_KM = 20.0
NODE_KM = [16.0, 20.0, 26.0, 35.0, 55.0]
CFC_KEYS = ["cfc11", "cfc12"]
P1_GASES = ["h2o", "co2", "o3", "ch4", "n2o", "co", "no2", "o2"]
P2_GASES = ["h2o", "co2", "o3", "ch4", "n2o", "co", "no2", "hno3", "so2"] + CFC_KEYS
SPLIT_UM = 3.0
# Computation grids are the ones of the former appendix figure (0.25-2 um at
# 2 nm; 2-30 um at 0.1 um), which are fully Voigt-cached, plus one new 2-3 um
# strip at 2 nm; the panels DISPLAY 0.25-3 um and 3-30 um.
# (compute ranges [(lo, hi), ...], step_um, fwhm_um, gas_keys, display range)
PANELS = [
    ([(0.25, 2.0), (2.002, SPLIT_UM)], 0.002, 0.002, P1_GASES, (0.25, SPLIT_UM)),
    ([(2.0, 30.0)], 0.01, 0.10, P2_GASES, (SPLIT_UM, 30.0)),
]
WRANA_NM = [448, 756, 1544]          # Wrana et al. (2021) size-retrieval triple
ACE_RANGE_UM = (2.2, 13.3)           # ACE-FTS 750-4400 cm^-1
RMED_SIL_NM, SIGMA_SIL = 268.0, 1.31

OUT = _PARENT / "figures/slant_od_landscape.png"
# Computed curves are archived here; re-plotting reads them back unless
# --recompute is given (HAPI's table load alone takes ~10 min on this cache).
CURVES = _PARENT / "outputs/slant_od_landscape_curves.npz"

# Paper style (round 34): Times-compatible serif at main-text size.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.0,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6})

# Curve styles: silica red / sulfate blue as elsewhere in the paper.
STYLE = {
    "gas":      dict(color="black", lw=1.3, ls="-", zorder=4, label="trace gases (total)"),
    "rayleigh": dict(color="0.45", lw=1.6, ls=":", zorder=2, label="Rayleigh"),  # round 35c: thicker
    "sulfate":  dict(color="tab:blue", lw=1.7, ls="-", zorder=6,
                     label="sulfate background (calibrated)"),
    "silica":   dict(color="red", lw=1.8, ls="-", zorder=9, label="silica, 1 Tg"),
    "calcite":  dict(color="darkorange", lw=1.4, ls="-", zorder=8, label="calcite, 1 Tg"),
    "dolomite": dict(color="green", lw=1.4, ls="-", zorder=8, label="dolomite, 1 Tg"),
    "alumina":  dict(color="purple", lw=1.4, ls="-", zorder=8, label="alumina, 1 Tg"),
}


def compute():
    alt_m = np.arange(14_000.0, 80_001.0, 500.0)
    atm = us_standard_atmosphere(alt_m)
    chord = tangent_to_slant_paths(np.array([TANGENT_KM * 1e3]), alt_m)[0]  # (n_layer,)
    node_m = np.array(NODE_KM) * 1e3
    node_atm = us_standard_atmosphere(node_m)
    node_idx = [int(np.argmin(np.abs(alt_m - z))) for z in node_m]
    all_keys = sorted(set(P1_GASES + P2_GASES))
    n_prof = tg.number_density_profiles(alt_m, atm.number_density_m3, gases=all_keys)

    # --- aerosol layers -----------------------------------------------------
    silica_ri = SilicaRefractiveIndex(str(_PARENT / "data/optics/SiO2_KitamuraPopova.yml"))
    layers = {
        "silica": create_silica_sai_layer(alt_m, target_mass_tg=1.0,
                                          rmed_nm=RMED_SIL_NM, sigma=SIGMA_SIL,
                                          refractive_index=silica_ri),
        "calcite": create_calcite_sai_layer(alt_m, target_mass_tg=1.0),
        "dolomite": create_dolomite_sai_layer(alt_m, target_mass_tg=1.0),
        "alumina": create_alumina_sai_layer(alt_m, target_mass_tg=1.0),
    }
    # Calibrated two-component sulfate background (Sect. sec:calibrated_background):
    # GloSSAC 20-25N quiet-epoch 525-nm profile (on CAL_ALT) split into the
    # CAL_QUIET modes, LM65T223 member optics -- as in plot_spectral_shapes_paper.py.
    ri_lm65, wt65, T65 = member_ri("LM65T223")
    alpha_cal = np.interp(alt_m, CAL_ALT, glossac_2025N_quiet_profile(),
                          left=0.0, right=0.0)
    sulfate_modes = [make_profile_column(alt_m, frac * alpha_cal, rmed, sg,
                                         weight_percent_h2so4=wt65,
                                         temperature_k=T65,
                                         refractive_index=ri_lm65)
                     for rmed, sg, frac in CAL_QUIET]

    def slant(alpha_zl):
        return chord @ alpha_zl

    def sigma_over_nodes(key, wl_m, fwhm, nu_pad_cm=500.0):
        s_nodes = np.zeros((len(NODE_KM), len(wl_m)))
        for i, k in enumerate(node_idx):
            Tn, Pn = float(node_atm.temperature_k[i]), float(node_atm.pressure_pa[i])
            vmr = float(n_prof["h2o"][k] / atm.number_density_m3[k]) if key == "h2o" else 0.0
            s_nodes[i] = tg.gas_xsec_rm(key, wl_m, Tn, Pn, fwhm_um=fwhm, vmr_h2o=vmr,
                                        nu_pad_cm=nu_pad_cm)
        return np.vstack([np.interp(alt_m, node_m, s_nodes[:, j],
                                    left=s_nodes[0, j], right=s_nodes[-1, j])
                          for j in range(len(wl_m))]).T

    def gas_total(rng, step, fwhm, gas_keys):
        """Total gas slant OD on the grid, plus the per-species curves (round
        38, Doron: the figure labels the major contributor at each peak)."""
        wl_um = np.arange(rng[0], rng[1] + 1e-9, step)
        wl_m = wl_um * 1e-6
        total = np.zeros_like(wl_um)
        per_gas = {}
        for key in gas_keys:
            print(f"    gas {key} on {rng} ...", flush=True)
            if key in CFC_KEYS:
                alpha = (tg.cfc_cross_section(key, wl_m, fwhm_um=fwhm)[None, :]
                         * n_prof[key][:, None])
            else:
                alpha = sigma_over_nodes(key, wl_m, fwhm) * n_prof[key][:, None]
            per_gas[key] = slant(alpha)
            total += per_gas[key]
        return wl_um, total, per_gas

    def compute_panel(ranges, step, fwhm, gas_keys):
        parts = [gas_total(r, step, fwhm, gas_keys) for r in ranges]
        wl_um = np.concatenate([p[0] for p in parts])
        total = np.concatenate([p[1] for p in parts])
        order = np.argsort(wl_um)
        wl_um, total = wl_um[order], total[order]
        wl_m = wl_um * 1e-6
        curves = {"gas": total}
        for key in gas_keys:      # per-species curves, archived as gas_<key>
            curves[f"gas_{key}"] = np.concatenate([p[2][key] for p in parts])[order]
        curves["rayleigh"] = slant(rayleigh_cross_section_m2(wl_m)[None, :]
                                   * atm.number_density_m3[:, None])
        curves["sulfate"] = sum(slant(m.extinction_profile_m1(wl_m)) for m in sulfate_modes)
        for name, lay in layers.items():
            curves[name] = slant(lay.extinction_profile_m1(wl_m))
        return wl_um, curves

    results = []
    for (ranges, step, fwhm, gas_keys, rng) in PANELS:
        print(f"Computing panel {rng} ...", flush=True)
        wl_um, curves = compute_panel(ranges, step, fwhm, gas_keys)
        results.append((wl_um, curves))
    CURVES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CURVES, **{f"p{i}_wl": r[0] for i, r in enumerate(results)},
                        **{f"p{i}_{k}": v for i, r in enumerate(results) for k, v in r[1].items()})
    print(f"Curves archived -> {CURVES}")
    return results


def load_curves():
    d = np.load(CURVES)
    results = []
    for i in range(len(PANELS)):
        wl = d[f"p{i}_wl"]
        curves = {k[len(f"p{i}_"):]: d[k] for k in d.files
                  if k.startswith(f"p{i}_") and k != f"p{i}_wl"}
        results.append((wl, curves))
    return results


def main():
    if CURVES.exists() and "--recompute" not in sys.argv:
        print(f"Loading archived curves from {CURVES}")
        results = load_curves()
    else:
        results = compute()

    fig, axes = plt.subplots(2, 1, figsize=(12.0 / 2.54, 15.0 / 2.54))
    ymin, ymax = 1e-3, 1e3
    for ax, (ranges, step, fwhm, gas_keys, rng), (wl_um, curves) in zip(axes, PANELS, results):
        for name in ("gas", "rayleigh", "sulfate", "calcite", "dolomite", "alumina", "silica"):
            st = dict(STYLE[name]); st.pop("label")
            ax.plot(wl_um, curves[name], **st)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(*rng); ax.set_ylim(ymin, ymax)
        ax.set_ylabel("Slant-path optical depth")
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    # --- round 38 (Doron): label the major gas contributor at the main peaks ----
    # of the total gas curve (both panels).  A first pass with an automatic
    # peak finder (prominence 0.5 dex) found 50 peaks and was unreadable, so
    # the peaks to label are curated below (wavelength, expected species); for
    # each, the local maximum of the total within +-1.5 % is located, the species
    # carrying the largest share of the total there is looked up in the
    # per-species curves (must equal the expected one, else the script stops),
    # and the label is set just above the curve, without a box, so that nothing
    # is hidden.  The H2O pure-rotation forest beyond 18 um gets one label.
    GAS_TEX = {"h2o": r"H$_2$O", "co2": r"CO$_2$", "o3": r"O$_3$", "ch4": r"CH$_4$",
               "n2o": r"N$_2$O", "co": "CO", "no2": r"NO$_2$", "o2": r"O$_2$",
               "hno3": r"HNO$_3$", "so2": r"SO$_2$", "cfc11": "CFC-11", "cfc12": "CFC-12"}
    LABEL_PEAKS = {
        # optional third/fourth items: horizontal alignment and a height
        # multiplier, for labels that would otherwise touch another element
        # (the Wrana line label at 756 nm, the CO2 wall at 2.7 um, the ACE bar)
        # round 39 (Doron): O3 label added at 0.30 um (Huggins band; 96 % of the
        # total there), the H2O label near 0.94 um raised clear of the aerosol
        # curves, the CH4 label near 1.7 um and the N2O label near 3.9 um
        # moved to the left of their peaks.
        # round 40 (Doron): O3 at 0.30 um to the right, CH4 near 1.6 um and
        # CO2 near 1.6 um and H2O near 1.8 um raised, N2O near 3.9 um centred
        0: [(0.30, "o3", "left", 1.6), (0.602, "o3"), (0.762, "o2", "left", 1.6), (0.938, "h2o", "center", 4.0),
            (1.134, "h2o"), (1.362, "h2o"), (1.572, "co2", "left", 3.0), (1.666, "ch4", "right", 2.5),
            (1.848, "h2o", "center", 3.0), (2.006, "co2"), (2.370, "ch4"), (2.594, "h2o", "right", 1.6), (2.72, "co2")],
        1: [(3.27, "ch4"), (3.91, "n2o", "center", 1.6), (4.26, "co2"), (4.75, "o3"),
            (6.3, "h2o"), (7.70, "ch4"), (9.6, "o3", "right", 0.7), (11.2, "hno3"),
            (14.94, "co2"), (24.0, "h2o")],
    }
    for ip, (ax, (ranges, step, fwhm, gas_keys, rng), (wl_um, curves)) in enumerate(
            zip(axes, PANELS, results)):
        per = {k[4:]: v for k, v in curves.items() if k.startswith("gas_")}
        if not per:
            print("WARNING: no per-species curves in the archive; run with --recompute")
            break
        y = curves["gas"]
        for entry in LABEL_PEAKS[ip]:
            wl0, expect = entry[:2]
            ha = entry[2] if len(entry) > 2 else "center"
            ymul = entry[3] if len(entry) > 3 else 1.6
            m = np.abs(np.log(wl_um / wl0)) < 0.015
            p = np.flatnonzero(m)[np.argmax(y[m])]
            shares = {k: per[k][p] / y[p] for k in per}
            key = max(shares, key=shares.get)
            assert key == expect, (wl0, expect, key, shares[key])
            txt = GAS_TEX[key] + (" (rotation)" if wl0 > 18 else "")
            if y[p] * ymul < ymax / 2.5:
                xt = wl_um[p] * {"center": 1.0, "left": 1.02, "right": 0.98}[ha]
                ax.text(xt, y[p] * ymul, txt, fontsize=7.5, ha=ha,
                        va="bottom", color="black", zorder=12)
            else:   # saturated peak (runs off the frame): label above the axis edge
                ax.text(wl_um[p], ymax * 1.08, txt, fontsize=7.5, ha="center",
                        va="bottom", color="black", zorder=12, clip_on=False)
            print(f"  label {wl_um[p]:7.3f} um: total {y[p]:.3g}, {key} {shares[key]*100:.0f} %")
    axes[0].set_xticks([0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    axes[1].set_xticks([3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30])
    axes[1].set_xlabel(r"Wavelength [$\mu$m]")

    # --- Wrana triple (visible/NIR panel) -----------------------------------
    ytext = 10 ** (np.log10(ymin) + 0.97 * (np.log10(ymax) - np.log10(ymin)))
    for nm in WRANA_NM:
        um = nm / 1000.0
        axes[0].axvline(um, color="0.35", lw=1.0, ls="--", zorder=1)
        axes[0].text(um, ytext, f"{nm} nm", fontsize=9, rotation=90,
                     ha="right", va="top", color="0.35")

    # --- ACE-FTS (SCISAT) mid-infrared coverage (MIR panel) -----------------
    ybar = 10 ** (np.log10(ymin) + 0.93 * (np.log10(ymax) - np.log10(ymin)))
    x0, x1 = SPLIT_UM, ACE_RANGE_UM[1]
    axes[1].annotate("", xy=(x1, ybar), xytext=(x0, ybar),
                     arrowprops=dict(arrowstyle="|-|", color="0.35", lw=1.0,
                                     shrinkA=0, shrinkB=0, mutation_scale=3))
    axes[1].text(np.sqrt(5.0 * x1), ybar * 1.4, r"ACE-FTS (SCISAT), to 13.3 $\mu$m",
                 fontsize=9, color="0.35", ha="center", va="bottom", zorder=10,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))

    # --- legend (visible/NIR panel, bottom) ---------------------------------
    order = ["gas", "rayleigh", "sulfate", "silica", "calcite", "dolomite", "alumina"]
    handles = [Line2D([0], [0], color=STYLE[n]["color"], lw=STYLE[n]["lw"], ls=STYLE[n]["ls"])
               for n in order]
    labels = [STYLE[n]["label"] for n in order]
    # round 35b (Doron): all SAI materials solid; the Wrana triple keeps its
    # labelled dashed lines in the panel but has no legend entry
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               handlelength=2.6, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(h_pad=1.8, rect=(0, 0.075, 1, 0.985))   # room for the above-axis labels
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Saved -> {OUT}")

    # --- diagnostics for the caption/text -----------------------------------
    wl2, c2 = results[1]
    print("\nMIR band peaks (3-30 um, 0.1 um element, 20 km tangent, 1 Tg):")
    for name in ("silica", "calcite", "dolomite", "alumina"):
        y = c2[name]
        pk, _ = find_peaks(np.log10(np.clip(y, 1e-12, None)), prominence=0.15)
        pk = sorted(pk, key=lambda p: -y[p])[:5]
        for p in sorted(pk):
            print(f"  {name:>8}: {wl2[p]:6.2f} um  tau_mat = {y[p]:.3f}  "
                  f"tau_gas = {c2['gas'][p]:.3g}  tau_sulf = {c2['sulfate'][p]:.3g}")
    wl1, c1 = results[0]
    print("\nVisible/NIR values at the Wrana triple:")
    for nm in WRANA_NM:
        i = int(np.argmin(np.abs(wl1 - nm / 1000.0)))
        print(f"  {nm} nm: gas {c1['gas'][i]:.3g}  Rayleigh {c1['rayleigh'][i]:.3g}  "
              f"sulfate {c1['sulfate'][i]:.3g}  silica {c1['silica'][i]:.3g}  "
              f"calcite {c1['calcite'][i]:.3g}  dolomite {c1['dolomite'][i]:.3g}  "
              f"alumina {c1['alumina'][i]:.3g}")
    for name in ("sulfate", "silica", "calcite", "dolomite", "alumina"):
        bad = ~np.isfinite(c2[name])
        if bad.any():
            print(f"  WARNING: {name} has {bad.sum()} non-finite MIR values "
                  f"(first at {wl2[bad][0]:.2f} um)")


if __name__ == "__main__":
    main()
