"""Onion-peel geometry of the retrieval grid and the gas-removal floor anchor
shared by the whole threshold chain (round 53, referee RC1 M1/M3).

Convention (confirmed by Doron, round 53): the retrieval shells are bounded by
successive tangent heights (the operational onion peel); the ray tangent at z_k
enters shell k at its lower boundary, so P_kk = 2 sqrt(2 R dz) ~ 160 km and the
neighbour ratio is sqrt(2) - 1.  The white-noise amplification g is the EXACT
row norm of the inverted path matrix, |G_k| P_kk = 1.097 (not the two-term
1.24 of the centre-grid formula that the paper carried before RC1).

Floor anchor: every mid-infrared per-element floor is the measured ACE-FTS
atlas shape scaled so that the 8.80-um 0.1-um element carries the DESIGN's own
R~100 optimal-estimation budget (gas_removal_floor_880_widths.py), not the
measured ACE value -- "use the R~100 floor everywhere" (Doron, round 53).
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

R_EARTH_M = 6.371e6
DZ_M = 500.0
Z_REF_M = 20_000.0
P_KK_M = 2.0 * np.sqrt(2.0 * R_EARTH_M * DZ_M)          # 159.9 km, edge chord


def exact_gain(dz_m: float = DZ_M, z_ref_m: float = Z_REF_M, z_top_m: float = 60_000.0):
    """|G_k| (m^-1 per unit optical depth) and g = |G_k| P_kk for the edge grid."""
    zb = np.arange(0.0, z_top_m + dz_m, dz_m)             # shell boundaries = tangent heights
    rb, rt = R_EARTH_M + zb[:-1], R_EARTH_M + zb[1:]
    P = np.empty((len(zb) - 1, len(zb) - 1))
    for i, zt in enumerate(zb[:-1]):
        r2 = (R_EARTH_M + zt) ** 2
        P[i] = 2.0 * (np.sqrt(np.maximum(rt**2 - r2, 0.0)) - np.sqrt(np.maximum(rb**2 - r2, 0.0)))
    G = np.linalg.inv(P)
    k = int(np.argmin(np.abs(zb[:-1] - z_ref_m)))
    Gk = float(np.sqrt(np.sum(G[k] ** 2)))
    return Gk, Gk * P[k, k]


M1_PER_OD, G_ONION = exact_gain()          # 6.863e-6 m^-1 per OD, 1.097
OD_PER_M1 = 1.0 / M1_PER_OD                # 145.7 km "effective path"


def budget_od_880(width_um: float = 0.1) -> float:
    """Design R~100 gas-removal budget at the 8.80-um element [OD], archived by
    reproduce/gas_removal_floor_880_widths.py (outputs/round43_tracegas)."""
    p = Path(__file__).resolve().parents[1] / "outputs/round43_tracegas/gas_removal_floor_widths.csv"
    for r in csv.DictReader(p.open()):
        if float(r["target_um"]) == 8.8 and float(r["width_um"]) == width_um:
            return float(r["sigma_removal_od"])
    raise KeyError("8.80-um budget not found in " + str(p))


BUDGET_OD_880 = budget_od_880()             # 2.466e-3 OD
SIG_880_ABS = BUDGET_OD_880 * M1_PER_OD    # 1.692e-8 m^-1: the design floor at 8.80 um

if __name__ == "__main__":
    print(f"P_kk = {P_KK_M/1e3:.1f} km; g_exact = {G_ONION:.4f}; |G_k| = {M1_PER_OD:.4e} m^-1/OD; "
          f"OD_PER_M1 = {OD_PER_M1/1e3:.1f} km; budget(8.80, 0.1 um) = {BUDGET_OD_880:.4e} OD; "
          f"SIG_880_ABS = {SIG_880_ABS:.4e} m^-1")
