"""Regenerate the shipped 0.0025-cm^-1 grids (data/trace_gases/xsec_cache/*_1067_1294_*_s0.0025.npz)
of O3, N2O, CH4 and H2O over the 7.75-9.35-um window (round 70, element-observable closure). Run from the repo
root: python fetch/build_finegrid_window.py (needs hitran-api; ~1-10 min per grid, 32 grids). Original docstring:
Cache native Voigt grids of O3 and N2O over the 7.75-9.35-um window at 0.0025 cm^-1
(4x finer than the 0.01 budget grids; the lines are ~0.003 cm^-1 HWHM) at the five budget
nodes and at (T_eff, T_eff +- 5 K, P_eff) of each gas, using the same cache keys as
saimon.trace_gases._voigt_native so the closure script reads them back."""
import sys, os, time; sys.path.insert(0,'.')
import numpy as np
from saimon.atmosphere import us_standard_atmosphere
from saimon.geometry import tangent_to_slant_paths
from saimon import trace_gases as tg
from saimon.o3_removal import DEFAULT_NODES_KM
tg.HITRAN_CACHE = os.path.abspath("data/trace_gases/hitran_cache")   # the carved 1042-1319 cm^-1 lists shipped with the repository
import hapi
hapi.fetch = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fetch blocked"))
t0=time.time(); hapi.db_begin(tg.HITRAN_CACHE); print("db_begin", time.time()-t0, flush=True)
alt = np.arange(14_000.0, 80_001.0, 500.0); atm = us_standard_atmosphere(alt)
chord = tangent_to_slant_paths(np.array([20000.0]), alt)[0]
dens = tg.number_density_profiles(alt, atm.number_density_m3)
nodes_m = np.asarray(DEFAULT_NODES_KM)*1e3; na = us_standard_atmosphere(nodes_m)
STEP=float(sys.argv[1]) if len(sys.argv) > 1 else 0.0025; NU_LO, NU_HI = 1067.0, 1294.0   # round 74: python fetch/build_finegrid_window.py 0.00125 for the convergence grids
for gas in ("o3","n2o","ch4","h2o"):
    g = tg.HITRAN_GASES[gas]
    w = chord*dens[gas]; Te=float((w*atm.temperature_k).sum()/w.sum()); Pe=float((w*atm.pressure_pa).sum()/w.sum())/101325
    TP = [(float(T), float(P)/101325) for T,P in zip(na.temperature_k, na.pressure_pa)] + [(Te,Pe),(Te+5,Pe),(Te-5,Pe)]
    for T,P in TP:
        key=f"{g.name}_{int(round(NU_LO))}_{int(round(NU_HI))}_T{T:.1f}_P{P:.6f}_s{STEP:g}"
        path=os.path.join(tg.XSEC_CACHE, key+".npz")
        if os.path.exists(path): print("have", key, flush=True); continue
        t0=time.time()
        nu, coef = hapi.absorptionCoefficient_Voigt(SourceTables=f'cont_{gas}_1042_1319', Environment={'T':T,'p':P},
                    OmegaRange=(NU_LO,NU_HI), WavenumberStep=STEP, OmegaWing=25.0, OmegaWingHW=50.0, HITRAN_units=True)
        np.savez_compressed(path, nu=nu, coef=coef); print("wrote", key, len(nu), f"{time.time()-t0:.0f} s", flush=True)
print("DONE")
