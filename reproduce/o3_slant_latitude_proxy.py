"""Referee RC1 M4: how much does the slant ozone optical depth above a tangent
height vary between the six AFGL standard atmospheres?  The trace-gas floors of
the paper are evaluated with the AFGL US-standard 1976 profile (the file
saimon.trace_gases prefers); the mid-infrared line-intensity floor scales with
tau_O3, so the ratio of slant tau_O3 to the US-standard value is the factor by
which that floor moves with latitude.  Straight-line chord, no refraction (the
ratio is insensitive to it).  Writes outputs/o3_slant_latitude_proxy.txt."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AFGL = ROOT / "data/trace_gases/afgl"
R_E = 6371.0  # km
PROFILES = [("us", "afglus.dat.libradtran", "US standard 1976"), ("t", "afglt.dat", "tropical"),
            ("ms", "afglms.dat", "midlat summer"), ("mw", "afglmw.dat", "midlat winter"),
            ("ss", "afglss.dat", "subarctic summer"), ("sw", "afglsw.dat", "subarctic winter")]

def load(fname):
    d = np.loadtxt(AFGL / fname, comments="#")
    z, o3 = d[:, 0], d[:, 4]           # km, cm^-3 (file is top-down)
    o = np.argsort(z)
    return z[o], o3[o]

def slant_column(z, n, h, dz=0.05):
    """Straight-chord slant column above tangent height h (arbitrary units), both halves."""
    zz = np.arange(h, 60.0, dz)
    nn = np.interp(zz, z, n)
    ds_dz = (R_E + zz) / np.sqrt(np.maximum((R_E + zz) ** 2 - (R_E + h) ** 2, 1e-9))
    ds_dz[0] = ds_dz[1]                # remove the integrable singularity at the tangent
    return 2.0 * np.trapezoid(nn * ds_dz, zz)

prof = {k: load(f) for k, f, _ in PROFILES}
lines = []
def out(s=""):
    print(s); lines.append(s)
H = [16, 18, 20, 22, 24]
out("slant tau_O3 above the tangent, ratio to US standard 1976")
out("h_tan [km] | " + " | ".join(lab for _, _, lab in PROFILES[1:]))
for h in H:
    ref = slant_column(*prof["us"], h)
    out(f"{h:10d} | " + " | ".join(f"{slant_column(*prof[k], h)/ref:.2f}" for k, _, _ in PROFILES[1:]))
out("")
out("local O3 number density at 20 km [cm^-3] (for comparison: the referee's 'factor two' is a local/column statement)")
for k, _, lab in PROFILES:
    z, n = prof[k]
    out(f"  {lab:20s} {np.interp(20.0, z, n):.2e}")
out("")
out("floor factor if the 8.80-um budget is line term 2.3e-3 x R plus OE remainder 0.9e-3 in quadrature:")
for Rr in [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
    out(f"  tau ratio {Rr:.1f}: floor x {np.hypot(2.3e-3*Rr, 0.9e-3)/np.hypot(2.3e-3, 0.9e-3):.2f}")
(ROOT / "outputs/o3_slant_latitude_proxy.txt").write_text("\n".join(lines) + "\n")
