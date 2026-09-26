"""
plot_spectral_shapes_paper.py — Paper-1 version of the spectral-shapes panel.

Round 1 (Doron's caption comment in problem1.tex): descriptive channel
labels instead of the internal A/A+/B/B+/C codenames, elevated background
only. Round 2: only three curves — nominal silica (black), Franta silica
(dashed red), elevated sulfate (blue); no legend titles; no channels
legend (markers keep their rotated wavelength labels); larger,
Times-compatible (STIX) fonts to match the Copernicus main text.
Round 3: channel markers reduced to the same five as the
refractive-index figure (448/756/1544 nm, 8.80/20.4 um; round 43, was 8.74), all dashed
gray with labels at the top at a common height in gray; legend moved
to the bottom left.
Round 6 (2026-09-15, the Sect.-2/3 calibrated-background revision): the
single-mode elevated sulfate curve is replaced by the CALIBRATED
two-component background (Sect. sec:calibrated_background): total (solid
blue) plus its fine and coarse modes (dotted / dash-dot blue), all
normalized to the total's 525-nm value; sulfate optics = LM65T223 member
spliced to PW+LL below 2.05 um, profile = GloSSAC 20-25N quiet median
(the normalization makes the profile irrelevant here except through the
mode split).

Identical physics setup to plot_4param_hierarchy.py (which keeps producing
the internal Notes version); writes figures/
spectral_shapes_problem.png.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_THIS = Path(__file__).resolve()
_PARENT = _THIS.parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from saimon.backgrounds import make_empirical_column, make_profile_column
from saimon.sai import create_silica_sai_layer
from saimon.materials import SilicaRefractiveIndex
from saimon.refractive_index import sulfuric_acid_at_temperature

if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))
from calibrated_background_thresholds import (  # noqa: E402
    member_ri, glossac_2025N_quiet_profile)
from calibrated_background_problem import CAL_QUIET  # noqa: E402

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.labelsize": 18,   # round 44 (Doron): larger fonts, heavier lines
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

ALT = np.linspace(0, 60_000, 121)
iz = 40  # z = 20 km

ri = sulfuric_acid_at_temperature(215.0)
silica_ri = SilicaRefractiveIndex("data/optics/SiO2_KitamuraPopova.yml")
silica_ri_franta = SilicaRefractiveIndex("data/optics/SiO2_Franta2016.yml")
RMED_NM, SIGMA_PSD = 268.0, 1.31
sai = create_silica_sai_layer(ALT, target_mass_tg=1.0,
                              refractive_index=silica_ri,
                              rmed_nm=RMED_NM, sigma=SIGMA_PSD)
sai_franta = create_silica_sai_layer(ALT, target_mass_tg=1.0,
                                     refractive_index=silica_ri_franta,
                                     rmed_nm=RMED_NM, sigma=SIGMA_PSD)

print("Computing dense wavelength grid (Kit+Pop nominal, Franta comparison)...")
wl_visnir_um = np.geomspace(0.3, 2.0, 150)
wl_mir_um = np.arange(2.0, 30.0001, 0.02)
wl_um = np.unique(np.concatenate([wl_visnir_um, wl_mir_um]))
wl_m = wl_um * 1e-6

alpha_sil = sai.extinction_profile_m1(wl_m)[iz, :]
alpha_sil_franta = sai_franta.extinction_profile_m1(wl_m)[iz, :]

# Round 6: the calibrated two-component background
ri_lm65, wt65, T65 = member_ri("LM65T223")
alpha_cal = glossac_2025N_quiet_profile()
comps = []
for rmed, sg, frac in CAL_QUIET:
    colm = make_profile_column(ALT, frac * alpha_cal, rmed, sg,
                               weight_percent_h2so4=wt65,
                               temperature_k=T65,
                               refractive_index=ri_lm65)
    comps.append(colm.extinction_profile_m1(wl_m)[iz, :])
alpha_fine, alpha_coarse = comps
alpha_sulf = alpha_fine + alpha_coarse

i525 = int(np.argmin(np.abs(wl_um - 0.525)))
norm = lambda arr: arr / arr[i525]
norm_tot = lambda arr: arr / alpha_sulf[i525]   # components share the
                                                # total's 525-nm norm

fig, ax = plt.subplots(figsize=(12, 6.2))

ax.plot(wl_um, norm(alpha_sil), color='black', lw=3.0,
        label=r'Silica (nominal: $r_\mathrm{med}=268$ nm, $\sigma_\mathrm{psd}=1.31$)')
ax.plot(wl_um, norm(alpha_sil_franta), color='red', lw=2.2, ls=(0, (4, 2)),
        label=r'Silica (same PSD, Franta 2016 optical constants)')
ax.plot(wl_um, norm(alpha_sulf), color='tab:blue', lw=2.6, ls='-',
        label='Sulfate (calibrated two-component background)')
ax.plot(wl_um, norm_tot(alpha_fine), color='tab:blue', lw=1.8, ls=':',
        label=r'fine mode (60 nm, $\sigma=1.6$; 82\% of 525 nm)'
        .replace(r'\%', '%'))
ax.plot(wl_um, norm_tot(alpha_coarse), color='tab:blue', lw=1.8,
        ls=(0, (5, 2, 1, 2)),
        label=r'coarse mode (1.5 $\mu$m, $\sigma=1.8$; 18% of 525 nm)')

# Channel markers: same five channels as the refractive-index figure
# (round 3): three Wrana channels + two reststrahlen channels, all dashed
# gray; labels at the top, common height, gray.
CHANNELS_UM = [0.448, 0.756, 1.544, 8.80, 20.4]   # round 43: 8.80 (was 8.74)
CH_LABELS = {0.448: '448 nm', 0.756: '756 nm', 1.544: '1544 nm',
             8.80: r'8.80 $\mu$m', 20.4: r'20.4 $\mu$m'}

ymin, ymax = 1e-4, 50.0
ax.set_ylim(ymin, ymax)
ytext = 10 ** (np.log10(ymin) + 0.97 * (np.log10(ymax) - np.log10(ymin)))
for ch in CHANNELS_UM:
    ax.axvline(ch, color='gray', lw=0.9, ls='--', alpha=0.75, zorder=0)
    ax.text(ch, ytext, CH_LABELS[ch], fontsize=13, rotation=90,
            ha='right', va='top', color='gray')

ax.legend(loc='lower left', framealpha=0.93)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.3, 30.0)
ax.set_xlabel(r'Wavelength [$\mu$m]')
ax.set_ylabel(r'Normalised extinction at $z = 20$ km  ($\div$ value at 525 nm)')
ax.grid(True, which='both', alpha=0.3)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
# Round 4: no xtick at 20.4 (the channel line carries its own gray label);
# round 5: a plain tick at 20 instead.
ax.set_xticks([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0])

fig.tight_layout()
out = Path('figures/spectral_shapes_problem.png')
fig.savefig(out, dpi=180, bbox_inches='tight')
print(f"Saved -> {out}")
plt.close(fig)
