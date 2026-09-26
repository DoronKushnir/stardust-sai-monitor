import pathlib
"""
plot_refractive_index_paper.py — Paper-1 version of the refractive-index
comparison panel (Doron's round-2 caption comments in problem1.tex):

  - no figure/panel titles;
  - nominal silica (Kitamura+Popova) black, Franta dashed red, sulfate blue;
  - NIR/MIR panel split at 3 um (ISO 20473 border) instead of 2 um;
  - channel markers only at the Wrana channels (448/756/1544 nm) and the
    two reststrahlen channels (8.80/20.4 um; round 43, was 8.74);
  - bottom (k) panels y-minimum 1e-2;
  - larger, Times-compatible (STIX) fonts to match the Copernicus main text.

Round 3: channel labels in the top-left (n, vis/NIR) panel at mid panel
height; in the top-right (n, MIR) panel at the top.

Round 27 (Doron, caption comment): the sulfate curve is the optics of the
CALIBRATED two-component background of Paper 1 Sect. 2.2 exactly as the
forward model uses them (scripts/calibrated_background_thresholds.py,
member_ri("LM65T223")): the Lund Myhre et al. (2003) 65 wt%, 223 K
laboratory tabulation above 2.05 um (tabulated to 26 um), spliced to the
Palmer-Williams 75 wt% + Lorentz-Lorenz(215 K) visible table below
2.05 um; the splice is marked by a dotted line in the left panels.
Replaces the 75 wt%, 215 K Palmer-Williams curve of the earlier rounds.

Silica physics/data identical to plot_refractive_index_comparison.py (which
keeps producing the internal Notes version). Writes
figures/refractive_index_problem.png.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from saimon.materials import SilicaRefractiveIndex
from saimon.refractive_index import sulfuric_acid_at_temperature
from calibrated_background_thresholds import member_ri  # noqa: E402

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.labelsize": 18,   # round 44 (Doron): larger fonts, heavier lines
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

ri_franta = SilicaRefractiveIndex("data/optics/SiO2_Franta2016.yml")
ri_h2so4_vis = sulfuric_acid_at_temperature(215.0)     # PW + LL(215 K)
ri_h2so4_lm, LM_WT, LM_T = member_ri("LM65T223")        # spliced member
SPLICE_UM = 2.05          # below: PW+LL visible table; above: LM 65wt%/223K
LM_MAX_UM = 26.0          # extent of the Lund Myhre tabulation
ri_kit = SilicaRefractiveIndex(str(pathlib.Path(__file__).resolve().parent.parent / "data" / "optics" / "SiO2_Kitamura2007.yml"))   # Kitamura (2007) alone, 0.1-20 um
ri_pop = SilicaRefractiveIndex("data/optics/Popova.yml")
STITCH_UM = 20.0


def eval_kitpop(wl_um):
    wl_um = np.asarray(wl_um, dtype=float)
    mk = ri_kit(wl_um * 1e-6)
    mp = ri_pop(wl_um * 1e-6)
    far = wl_um > STITCH_UM
    n = np.where(far, np.real(mp), np.real(mk))
    k = np.where(far, np.abs(np.imag(mp)), np.abs(np.imag(mk)))
    return n, k


def eval_ri(ri, wl_um):
    m = ri(wl_um * 1e-6)
    return np.real(m), np.abs(np.imag(m))


def eval_h2so4(wl_um):
    """The calibrated background's sulfate optics: PW+LL(215 K) below the
    2.05-um splice, the Lund Myhre 65 wt%/223 K member above it (masked
    beyond its 26-um tabulation)."""
    wl_um = np.asarray(wl_um, dtype=float)
    n_v, k_v = eval_ri(ri_h2so4_vis, wl_um)
    n_m, k_m = eval_ri(ri_h2so4_lm, wl_um)
    vis = wl_um < SPLICE_UM
    n = np.where(vis, n_v, n_m)
    k = np.where(vis, k_v, k_m)
    far = wl_um > LM_MAX_UM
    n = np.where(far, np.nan, n)
    k = np.where(far, np.nan, k)
    return n, k


SPLIT_UM = 3.0  # ISO 20473 NIR/MIR border
wl_vis = np.geomspace(0.10, SPLIT_UM, 1800)
wl_mir = np.linspace(SPLIT_UM, 30.0, 2800)

n_fr_vis, k_fr_vis = eval_ri(ri_franta, wl_vis)
n_fr_mir, k_fr_mir = eval_ri(ri_franta, wl_mir)
n_h_vis, k_h_vis = eval_h2so4(wl_vis)
n_h_mir, k_h_mir = eval_h2so4(wl_mir)
n_kp_vis, k_kp_vis = eval_kitpop(wl_vis)
n_kp_mir, k_kp_mir = eval_kitpop(wl_mir)

CHANNELS_UM = [0.448, 0.756, 1.544, 8.80, 20.4]   # round 43: 8.80 (was 8.74)
CH_LABELS = {0.448: "448 nm", 0.756: "756 nm", 1.544: "1544 nm",
             8.80: r"8.80 $\mu$m", 20.4: r"20.4 $\mu$m"}

STYLES = {
    "KitPop": dict(color="black", lw=2.6, ls="-",
                   label=r"SiO$_2$ (Kitamura + Popova, nominal)"),
    "Franta": dict(color="red", lw=2.2, ls=(0, (4, 2)),
                   label=r"SiO$_2$ (Franta 2016)"),
    "H2SO4": dict(color="tab:blue", lw=2.6, ls="-",
                  # round 28 (Doron): member details in the caption only
                  label=r"H$_2$SO$_4$, calibrated background"),
}

K_FLOOR, K_TOP = 1e-2, 10.0

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

panels = [
    (axes[0, 0], wl_vis, n_kp_vis, n_fr_vis, n_h_vis, False),
    (axes[0, 1], wl_mir, n_kp_mir, n_fr_mir, n_h_mir, False),
    (axes[1, 0], wl_vis, k_kp_vis, k_fr_vis, k_h_vis, True),
    (axes[1, 1], wl_mir, k_kp_mir, k_fr_mir, k_h_mir, True),
]

for ax, wl, y_kp, y_fr, y_h, is_k in panels:
    if is_k:
        ax.semilogy(wl, np.maximum(y_kp, K_FLOOR * 0.5), **STYLES["KitPop"])
        ax.semilogy(wl, np.maximum(y_fr, K_FLOOR * 0.5), **STYLES["Franta"])
        ax.semilogy(wl, np.maximum(y_h, K_FLOOR * 0.5), **STYLES["H2SO4"])
        ax.set_ylabel(r"$k$  (imaginary index)")
        ax.set_ylim(K_FLOOR, K_TOP)
    else:
        ax.plot(wl, y_kp, **STYLES["KitPop"])
        ax.plot(wl, y_fr, **STYLES["Franta"])
        ax.plot(wl, y_h, **STYLES["H2SO4"])
        ax.set_ylabel(r"$n$  (real index)")

    vis_panel = wl[0] < 1.0
    ax.set_xlim(0.10, SPLIT_UM) if vis_panel else ax.set_xlim(SPLIT_UM, 30.0)
    ax.set_xlabel(r"Wavelength [$\mu$m]")
    if vis_panel:
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0])
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        # suppress default minor-tick labels (they collide with the majors)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, which="both", alpha=0.3, lw=0.5)
    if vis_panel:
        # the visible-table / laboratory-member splice of the sulfate optics
        ax.axvline(SPLICE_UM, color="tab:blue", lw=0.9, ls=":", alpha=0.8,
                   zorder=0)

    # Channel markers: Wrana channels (left panels), reststrahlen (right)
    ylo, yhi = ax.get_ylim()
    for ch in CHANNELS_UM:
        if ax.get_xlim()[0] <= ch <= ax.get_xlim()[1]:
            ax.axvline(ch, color="gray", lw=0.9, ls="--", alpha=0.75, zorder=0)
            if is_k:
                ytext = 10 ** (np.log10(ylo)
                               + 0.94 * (np.log10(yhi) - np.log10(ylo)))
                ax.text(ch, ytext, CH_LABELS[ch], fontsize=13, rotation=90,
                        ha="right", va="top", color="gray")
            elif vis_panel:
                # top-left n panel: labels at mid panel height (round 3)
                ax.text(ch, 0.5 * (ylo + yhi), CH_LABELS[ch],
                        fontsize=13, rotation=90, ha="right", va="center",
                        color="gray")
            else:
                # top-right n panel: labels at the top (round 3)
                ax.text(ch, yhi - 0.02 * (yhi - ylo), CH_LABELS[ch],
                        fontsize=13, rotation=90, ha="right", va="top",
                        color="gray")

axes[0, 0].legend(loc="upper right", framealpha=0.95)

fig.tight_layout()
out_path = "figures/refractive_index_problem.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved -> {out_path}")
