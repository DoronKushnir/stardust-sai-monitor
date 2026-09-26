"""
silica_peak_wavelength.py -- where does the silica reststrahlen *aerosol extinction* peak sit?

Round 42 (Doron, paper1 Figs. 4 and 5b): the 1 Tg silica curve visibly peaks near
8.8 um, not at the 8.74 um used as the band label.  This script evaluates the
monochromatic peak and the 0.10 / 0.25 um boxcar peaks for the adopted
Kitamura+Popova optical constants and for the Franta 2016 table used in the
early rounds (which is where 8.74 um came from), for several PSDs, and the
fraction of the peak extinction carried at 8.74 and 8.80 um.

Run from the repo root: python scripts/silica_peak_wavelength.py
Output: outputs/silica_peak_wavelength.txt
"""
import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from saimon.sai import create_silica_sai_layer
from saimon.materials import SilicaRefractiveIndex
ALT = np.linspace(0, 60_000, 121); iz = 40
lam = np.arange(8.0, 9.5001, 0.005)
for name, f in [("KitamuraPopova", "data/optics/SiO2_KitamuraPopova.yml"), ("Franta2016", "data/optics/SiO2_Franta2016.yml")]:
    ri = SilicaRefractiveIndex(f)
    for rmed, sg in [(268., 1.31), (250., 1.05), (150., 1.31), (400., 1.31)]:
        lay = create_silica_sai_layer(ALT, 1.0, refractive_index=ri, rmed_nm=rmed, sigma=sg)
        e = lay.extinction_profile_m1(lam*1e-6)[iz, :]
        ip = np.argmax(e); pk = lam[ip]
        # boxcar 0.25 and 0.1 um
        out = []
        for dl in (0.10, 0.25):
            half = int(round(dl/2/0.005))
            eb = np.array([e[max(0,i-half):i+half+1].mean() for i in range(len(e))])
            out.append((lam[np.argmax(eb)], np.interp(8.74, lam, eb)/eb.max(), np.interp(8.80, lam, eb)/eb.max()))
        print(f"{name:15s} rmed={rmed:4.0f} sg={sg}: mono peak {pk:.3f} um; ext(8.74)/max={np.interp(8.74,lam,e)/e.max():.3f} ext(8.80)/max={np.interp(8.80,lam,e)/e.max():.3f}; "
              f"box0.10 peak {out[0][0]:.3f} (8.74:{out[0][1]:.3f}, 8.80:{out[0][2]:.3f}); box0.25 peak {out[1][0]:.3f} (8.74:{out[1][1]:.3f}, 8.80:{out[1][2]:.3f})")
