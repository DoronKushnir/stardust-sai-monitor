"""
silica_element_psd_variation.py -- per-Tg silica extinction collected by the
reference element (0.1-um boxcar) across the injected PSD range quoted in
Paper 1 Sect. 3.1 (r_med 100-500 nm, sigma_psd 1.05-1.8), at the old
(8.74 um) and new (8.80 um, round 43) element centres, plus the monochromatic
peak position of each PSD.  Behind the "band position is PSD-independent"
sentence.  Run from the repo root; output: outputs/silica_element_psd_variation.txt
"""
import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from saimon.sai import create_silica_sai_layer
from saimon.materials import SilicaRefractiveIndex
ALT = np.linspace(0, 60_000, 121); iz = 40
ri = SilicaRefractiveIndex("data/optics/SiO2_KitamuraPopova.yml")
sub = np.linspace(-0.05, 0.05, 41)
lam = np.arange(8.3, 9.31, 0.01)
for c in (8.74, 8.80):
    vals = {}; peaks = {}
    for r in (100., 268., 500.):
        for s in (1.05, 1.31, 1.8):
            lay = create_silica_sai_layer(ALT, 1.0, refractive_index=ri, rmed_nm=r, sigma=s)
            vals[(r, s)] = lay.extinction_profile_m1((c + sub) * 1e-6)[iz, :].mean()
            e = lay.extinction_profile_m1(lam * 1e-6)[iz, :]; peaks[(r, s)] = lam[np.argmax(e)]
    v = np.array(list(vals.values())); nom = vals[(268., 1.31)]
    print(f"element {c:.2f} um (0.1 boxcar): nominal t = {nom:.3e}; range {v.min():.3e}-{v.max():.3e}; "
          f"max deviation from nominal {100*max(abs(v/nom-1)):.1f}%; peak positions {min(peaks.values()):.2f}-{max(peaks.values()):.2f} um")
    for k, x in vals.items(): print(f"   r={k[0]:.0f} s={k[1]}: {x:.3e} ({100*(x/nom-1):+.1f}%) peak {peaks[k]:.2f}")
