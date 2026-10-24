# Ported from the working repository's scripts/paper1_tableB1_predicted.py (paper 1 round 67).
"""
paper1_tableB1_predicted.py -- referee round 4, T2: the "Predicted (%)" column
of Table B1 (tables/sage3_level2_errors.tex) was computed with the former
onion-peel gain 1.24; the paper has used the exact edge-grid gain
g = 1.097 since round 53.  This script recomputes the column from the
table's own inputs (alpha_k, tau_Ray at 20 km; SNR = 2000, eps_R = 0.4 %,
P_kk = 160 km) with Eq. (B2)-(B3) and Eq. (B8): sigma_alpha/alpha =
g * sqrt((1/SNR)^2 + (eps_R tau_Ray)^2) / (P_kk alpha_k), prints both the
old-gain and the current-gain columns, and rewrites the table.

Run from the repo root:  python reproduce/tableB1_predicted.py
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from saimon.onion_peel import G_ONION, P_KK_M   # noqa: E402

SNR, EPS_R = 2000.0, 0.004
TABLE = _ROOT / "tables/sage3_level2_errors.tex"
rows = [(384, 1.0e-6, 1.83), (448, 7.8e-7, 0.96), (521, 6.0e-7, 0.52), (602, 4.7e-7, 0.29),
        (676, 3.8e-7, 0.18), (756, 3.2e-7, 0.11), (869, 2.5e-7, 0.065), (1021, 1.9e-7, 0.034),
        (1544, 9.1e-8, 0.006)]

def pred(g, alpha, tau):
    return 100 * g * np.hypot(1 / SNR, EPS_R * tau) / (P_KK_M * alpha)

print(f"g = {G_ONION:.4f}, P_kk = {P_KK_M/1e3:.2f} km")
print(f"{'nm':>6} {'old g=1.24':>11} {'g=1.097':>9}")
new = {}
for nm, a, t in rows:
    new[nm] = pred(G_ONION, a, t)
    print(f"{nm:6d} {pred(1.24, a, t):11.2f} {new[nm]:9.2f}")
s = TABLE.read_text()
for nm, a, t in rows:
    pat = re.compile(r"(\n\s*%d\s*&[^&]*&[^&]*&\s*)([0-9.]+)(\s*&)" % nm)
    m = pat.search(s); assert m, nm
    s = s[:m.start(2)] + f"{new[nm]:.1f}" + s[m.end(2):]
s = s.replace("$\\mathrm{SNR}=2000$, $\\varepsilon_R=0.4\\%$, $P_{kk}=160$~km)",
              "$\\mathrm{SNR}=2000$, $\\varepsilon_R=0.4\\%$, $P_{kk}=160$~km, $g=1.10$)") if "$g=1.10$" not in s else s
TABLE.write_text(s); print("table rewritten:", TABLE)
