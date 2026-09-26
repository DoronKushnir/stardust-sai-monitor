"""Example: the marginalized 3-sigma detection threshold for a custom silica PSD.

Evaluates the baseline design of the paper (heritage triplet 448/756/1544 nm
plus the 8-13 um band at 0.1 um sampling with its measured per-element
floors, all seven parameters of the calibrated two-component background
free) for an injected silica size distribution of your choice -- the
calculation behind one row of Table 2.  About two minutes.

    python examples/threshold_custom_psd.py --rmed-nm 268 --sigma 1.31      # baseline: 0.081 Tg
    python examples/threshold_custom_psd.py --rmed-nm 600 --sigma 1.8
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))
from saimon.config import OPTICS, OUT                                    # noqa: E402
from saimon.materials import SilicaRefractiveIndex                       # noqa: E402
from saimon.sai import create_silica_sai_layer                           # noqa: E402
from calibrated_background_thresholds import CAL_QUIET, member_ri, ALT   # noqa: E402
from design_sensitivity_calibrated import ElementSet                     # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--rmed-nm", type=float, default=268.0)
ap.add_argument("--sigma", type=float, default=1.31)
ap.add_argument("--window", action="store_true", help="use the 7.8-9.3 um window subset instead of the whole band")
a = ap.parse_args()

thr = json.load(open(OUT / "calibrated_background_thresholds.json"))
if a.window:
    es = ElementSet("window 7.8-9.3 @0.1", thr["window_elements_um"], 0.10, thr["window_floors_m1"])
else:
    es = ElementSet("band 8-13 @0.1", thr["band01_elements_um"], 0.10, thr["band01_floors_m1"])
ri65, wt65, T65 = member_ri("LM65T223")                       # the calibrated sulfate member
exts, ders = es.background(CAL_QUIET, ri65, wt65, T65)        # background columns + PSD derivatives
ri = SilicaRefractiveIndex(str(OPTICS / "SiO2_KitamuraPopova.yml"))
layer = create_silica_sai_layer(ALT, 1.0, refractive_index=ri, rmed_nm=a.rmed_nm, sigma=a.sigma)
m, tperp, one_minus_r2 = es.mmin(exts, ders, es.prof(layer))
print(f"{es.name}: silica r_med {a.rmed_nm:g} nm, sigma {a.sigma:g}")
print(f"  M_min(3 sigma) = {m:.3f} Tg per sounding at 20 km   (||t_perp|| = {tperp:.1f}, 1 - R^2 = {one_minus_r2:.3f})")
