"""Example: the measured ACE-FTS residual floor at a 0.1-um element (Appendix E).

    python examples/floor_at_element.py 8.8 11.4 12.9
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
atlas = json.loads((ROOT / "data" / "ace_floor" / "w0p1" / "atlas.json").read_text())
for c in (sys.argv[1:] or ["8.8"]):
    c = float(c)
    for alt, rows in atlas.items():
        r = min((r for r in rows if r.get("sd") is not None), key=lambda r: abs(r["center_um"] - c))
        ci = r["sd_bootstrap_95"]
        print(f"{alt.replace('_','-')} km, element at {r['center_um']:.2f} um: SD {1e3*r['sd']:.2f}e-3 OD "
              f"[{1e3*ci['lower']:.2f}, {1e3*ci['upper']:.2f}], n = {r['n']}")
