"""Exact onion-peel gain in both tangent-grid conventions (referee RC1, M1) and
coherent-vs-white propagation of a 0.2 % O3 line-intensity error (RC1, M2).

Writes outputs/onion_peel_geometry_exact.txt.  Run: python reproduce/onion_peel_geometry_exact.py
"""
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from saimon.atmosphere import us_standard_atmosphere
from saimon import trace_gases as tg

R = 6_371_000.0; dz = 500.0
zb = np.arange(0.0, 60_000.0 + dz, dz)          # shell boundaries
zc = 0.5 * (zb[:-1] + zb[1:])                    # shell centres
def chords(zt):
    rt2 = (R + zt) ** 2
    rb, rtp = R + zb[:-1], R + zb[1:]
    return 2 * (np.sqrt(np.maximum(rtp**2 - rt2, 0)) - np.sqrt(np.maximum(rb**2 - rt2, 0)))
def analyse(name, ztan):
    P = np.array([chords(z) for z in ztan])      # rows = rays (top-down order irrelevant)
    G = np.linalg.inv(P)
    k = np.argmin(abs(ztan - 20_000.0)) if name == "edge" else np.argmin(abs(ztan - 20_000.0))
    Pkk, Pk1 = P[k, k], P[k, k + 1]
    ratio = Pk1 / Pkk
    g2 = np.sqrt(1 + ratio**2)
    row = G[k]
    Gk = np.sqrt(np.sum(row**2))
    twoterm = np.sqrt(row[k]**2 + row[k+1]**2)
    print(f"\n[{name}] tangent of shell k at z={ztan[k]/1e3:.2f} km; shell {zb[k]/1e3:.2f}-{zb[k+1]/1e3:.2f} km")
    print(f"  P_kk = {Pkk/1e3:.1f} km   P_k,k+1/P_kk = {ratio:.4f}   two-term g = {g2:.4f}")
    print(f"  exact |G_k| = {Gk:.4e} m^-1/OD   (two-term {twoterm:.4e}; 1.24/P_kk = {1.2408/Pkk:.4e}; g2/P_kk = {g2/Pkk:.4e})")
    print(f"  off-diagonal remainder (beyond k,k+1) = {1 - twoterm**2/Gk**2:.4f} of variance")
    print(f"  G_kk = {row[k]:.4e}  G_k,k+1 = {row[k+1]:.4e}  ratio {row[k+1]/row[k]:.4f}")
    for st, lab in [(5e-4, "SNR2000 5e-4"), (2.47e-3, "OE budget 2.47e-3"), (1.86e-3, "ACE 8.80um 1.86e-3"), (2.04e-3, "ACE 8.74 0.25um 2.04e-3"), (9.4e-4, "line-resolved 9.4e-4")]:
        print(f"    sigma_tau={st:.2e} ({lab}): exact {Gk*st:.3e}  paper(1.24/160km) {st*1.2408/160e3:.3e}")
    return P, G, k
# centre convention: rays tangent at shell centres (paper Eq. B4 / saimon.geometry usage)
Pc, Gc, kc = analyse("centre", zc)
# edge convention: rays tangent at lower shell edges (P_kk = 2 sqrt(2 R dz))
Pe, Ge, ke = analyse("edge", zb[:-1])
print(f"\nratio exact_edge/paper = {np.sqrt((Ge[ke]**2).sum())/(1.2408/160e3):.3f};  exact_centre/paper = {np.sqrt((Gc[kc]**2).sum())/(1.2408/160e3):.3f}")
print(f"thresholds x: band 0.081 -> edge {0.081*np.sqrt((Ge[ke]**2).sum())/(1.2408/160e3):.3f}, centre {0.081*np.sqrt((Gc[kc]**2).sum())/(1.2408/160e3):.3f} Tg;"
      f" window 0.115 -> edge {0.115*np.sqrt((Ge[ke]**2).sum())/(1.2408/160e3):.3f}, centre {0.115*np.sqrt((Gc[kc]**2).sum())/(1.2408/160e3):.3f} Tg")

# ---- M2: coherent 0.2% error on O3 slant OD, onion-peeled exactly -------------
atm = us_standard_atmosphere(zc)
n = tg.number_density_profiles(zc, atm.number_density_m3, gases=["o3"])["o3"]
for name, P, G, k in [("centre", Pc, Gc, kc), ("edge", Pe, Ge, ke)]:
    tau = P @ n                                  # arbitrary xsec units
    scale = 1.15 / tau[k]                        # normalise tau_O3(20 km tangent) = 1.15
    tau *= scale; alpha = n * scale              # local O3 extinction, m^-1
    f = 0.002
    d_coh = G @ (f * tau)                        # coherent fractional error, exact peel
    print(f"\n[M2 {name}] tau_O3(20km)={tau[k]:.3f}; alpha_O3(20km)={alpha[k]:.3e} m^-1; f*alpha={f*alpha[k]:.3e}")
    print(f"  coherent-peeled error at 20 km = {d_coh[k]:.3e} m^-1 (== f*alpha? {np.isclose(d_coh[k], f*alpha[k])})")
    white = np.sqrt((G[k]**2).sum()) * f * tau[k]
    print(f"  white propagation of the same sigma_tau=f*tau={f*tau[k]:.2e}: exact {white:.3e}; paper 1.24/160km -> {f*tau[k]*1.2408/160e3:.3e}")
    print(f"  ratio white(paper)/coherent = {f*tau[k]*1.2408/160e3/d_coh[k]:.2f}")
    print(f"  tau/P_kk = {tau[k]/P[k,k]:.3e} vs alpha = {alpha[k]:.3e}: chord-weighted mean O3 above tangent is {tau[k]/P[k,k]/alpha[k]:.2f}x the local value")
