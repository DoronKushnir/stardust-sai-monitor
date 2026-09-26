"""gas_removal_floor_880_widths.py -- the O3+N2O gas-removal floor and the
degeneracy-aware silica-amplitude error at the flown per-sample precision
(SNR 2000) for the 0.10-um design element and the 0.25-um legacy element,
at the 8.80-um reference element (paper1 round 43) and, for reference, at
the former 8.74-um element.  Same machinery as snr_integration_time_trade.py
(7.8-9.3 um window at 0.1-um steps, O3+N2O co-retrieval, 20-km tangent).

Run from the repo root (through scripts/round43_tracegas_driver.py so the
trimmed HITRAN cache is used):  python scripts/gas_removal_floor_880_widths.py
Writes outputs/round43_tracegas/gas_removal_floor_widths.csv
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import snr_integration_time_trade as sit
from saimon.geometry import tangent_to_slant_paths
from saimon.sai import create_silica_sai_layer
from saimon.materials import SilicaRefractiveIndex

alt_m = np.arange(14_000.0, 80_001.0, 500.0)
chord = tangent_to_slant_paths(np.array([sit.TANGENT_KM * 1e3]), alt_m)[0]
layer = create_silica_sai_layer(alt_m, target_mass_tg=1.0, rmed_nm=268.0, sigma=1.31,
                                refractive_index=SilicaRefractiveIndex(str(ROOT / "data/optics/SiO2_KitamuraPopova.yml")))
rows = []
for target in (8.80, 8.74):
    band = dict(sit.BANDS["B+ (8.80um)"]); band["target"] = target
    for width in (0.10, 0.25):
        sit.ELEMENT_UM = width
        gr = sit.run_band_at_snr(band, layer, chord, 2000.0, with_silica=False)
        da = sit.run_band_at_snr(band, layer, chord, 2000.0, with_silica=True)
        flat = sit.run_band_at_snr(band, layer, chord, 1e9, with_silica=False)
        r = dict(target_um=target, width_um=width,
                 tau_gas_target=gr["tau_gas_target"], tau_sil_1Tg=da["tau_sil_target"],
                 sigma_removal_od=gr["sigma_removal_od"], sigma_line_pred=gr["sigma_line_pred"],
                 sigma_removal_snr_inf=flat["sigma_removal_od"],
                 sigma_sil_od_snr2000=da["sigma_sil_od"], sigma_alpha_sil_snr2000=da["sigma_alpha"],
                 sigma_alpha_removal=sit.sigma_alpha_from_od(gr["sigma_removal_od"]))
        rows.append(r)
        print({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
import csv
out = ROOT / "outputs/round43_tracegas/gas_removal_floor_widths.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", out)
