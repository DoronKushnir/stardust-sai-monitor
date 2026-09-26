"""Example: slant optical depth of an injected silica layer at the 8.80 um element.

Builds the paper's baseline silica layer (1 Tg, r_med = 268 nm, sigma = 1.31,
constant mixing ratio 16-24 km), the calibrated two-component sulfate
background, and prints their slant optical depths through a 20-km tangent
ray for the 8.75-8.85 um element and for the visible/NIR heritage channels.

    python examples/slant_od_silica_element.py [--mass-tg 1] [--rmed-nm 268] [--sigma 1.31]
"""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from saimon.config import OPTICS                              # noqa: E402
from saimon.materials import SilicaRefractiveIndex            # noqa: E402
from saimon.sai import create_silica_sai_layer                # noqa: E402
from saimon.geometry import tangent_to_slant_paths            # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--mass-tg", type=float, default=1.0)
ap.add_argument("--rmed-nm", type=float, default=268.0)
ap.add_argument("--sigma", type=float, default=1.31)
ap.add_argument("--tangent-km", type=float, default=20.0)
a = ap.parse_args()

alt = np.arange(0.0, 60_000.1, 250.0)                         # altitude grid [m]
ri = SilicaRefractiveIndex(str(OPTICS / "SiO2_KitamuraPopova.yml"))
layer = create_silica_sai_layer(alt, a.mass_tg, refractive_index=ri, rmed_nm=a.rmed_nm, sigma=a.sigma)

# element average over 8.75-8.85 um plus the heritage channels
wl_el = np.linspace(8.75e-6, 8.85e-6, 21)
ext_el = layer.extinction_profile_m1(wl_el).mean(axis=1)      # [m^-1] per altitude, boxcar over the element
wl_vis = np.array([448e-9, 756e-9, 1544e-9])
ext_vis = layer.extinction_profile_m1(wl_vis)

P = tangent_to_slant_paths(np.array([a.tangent_km * 1e3]), alt)   # chord matrix [m], (n_tangent, n_layer)
tau_el = float((P @ ext_el)[0])
tau_vis = (P @ ext_vis).ravel()
print(f"silica {a.mass_tg:g} Tg, r_med {a.rmed_nm:g} nm, sigma {a.sigma:g}, tangent {a.tangent_km:g} km")
print(f"  extinction at 20 km, 8.80 um element : {ext_el[np.argmin(np.abs(alt-20e3))]:.3e} m^-1")
print(f"  slant optical depth, 8.80 um element : {tau_el:.3f}")
for w, t in zip(wl_vis, tau_vis):
    print(f"  slant optical depth, {w*1e9:6.0f} nm       : {t:.3f}")
