# Ported from the working repository's scripts/glossac_epoch_check_round62.py (paper 1 rounds 62-63); paths follow the repository layout.
"""
glossac_epoch_check.py -- Mark Schoeberl (paper-1 review, Oct 2026):
"GloSSAC before mid-2017 is worthless ... SAGE III/ISS data was only added
after mid 2017; you should use a different period."

The calibrated background takes ONLY its 525-nm amplitude and vertical
profile from GloSSAC V2.23 at 20-25N over 2016.5-2018.5 (no size/colour-ratio
information is used).  This script prints the median SAOD525 at 20-25N and
the cos-weighted tropical mean for that epoch and for SAGE III/ISS-only
alternatives (2017.5-2018.5, 2017.5-2019.0, 2017.5-2019.5), plus the
per-month values across the SAGE III/ISS entry, so the paper can state how
much the anchor would move if the pre-SAGE-III year were dropped.

Run from the repo root:  python reproduce/glossac_epoch_check.py
Output: outputs/glossac_epoch_check.txt (+ stdout)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from saimon.glossac import GloSSACLoader

EPOCHS = {"paper 2016.5-2018.5": (2016.5, 2018.5),
          "SAGE III only 2017.5-2018.5": (2017.5, 2018.5),
          "SAGE III only 2017.5-2019.0": (2017.5, 2019.0),
          "SAGE III only 2017.5-2019.5": (2017.5, 2019.5),
          "pre-SAGE III 2016.5-2017.5": (2016.5, 2017.5)}

def main():
    d = GloSSACLoader(str(_HERE / "data/glossac/GloSSAC_V2.23_subset.nc")).load_raw_data(525)
    time, lat = d["time"], d["lat"]
    od = np.ma.masked_invalid(d["od_file"])
    years = (time // 100) + (time % 100 - 0.5) / 12.0
    li = int(np.argmin(np.abs(lat - 22.5)))
    assert abs(lat[li] - 22.5) < 0.1
    od_2025 = od[:, li]
    trop = (lat >= -20) & (lat <= 20)
    w = np.cos(np.deg2rad(lat[trop]))
    od_trop = np.ma.average(od[:, trop], axis=1, weights=w)
    lines = []
    lines.append(f"{'epoch':32s} {'20-25N median':>14s} {'20-25N mean':>12s} {'tropical median':>16s}  n")
    for name, (lo, hi) in EPOCHS.items():
        m = (years >= lo) & (years < hi)
        lines.append(f"{name:32s} {np.ma.median(od_2025[m]):14.5f} {np.ma.mean(od_2025[m]):12.5f} "
                     f"{np.ma.median(od_trop[m]):16.5f}  {int(m.sum())}")
    lines.append("")
    lines.append("monthly SAOD525 at 20-25N, 2016.5-2018.5:")
    m = (years >= 2016.5) & (years < 2018.5)
    for t, y, v in zip(time[m], years[m], od_2025[m]):
        lines.append(f"  {int(t):6d} {y:8.3f} {float(v):.5f}")
    ref = float(np.ma.median(od_2025[(years >= 2016.5) & (years < 2018.5)]))
    alt = float(np.ma.median(od_2025[(years >= 2017.5) & (years < 2018.5)]))
    lines.append("")
    lines.append(f"ratio (2017.5-2018.5)/(2016.5-2018.5) of the 20-25N median: {alt/ref:.3f}")
    out = _HERE / "outputs/glossac_epoch_check.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
