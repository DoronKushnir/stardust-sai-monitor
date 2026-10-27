"""closure_flat_prior_check.py -- referee round 6 (paper 1; public port), items 2-3: the round-70 joint
element-observable model (reproduce/element_observable_closure.py, imported) re-evaluated with a
flat prior on the seven aerosol parameters.  The round-70 prior of 1e3 in the parameters' own units
(nm for the radii) binds the coarse-mode radius of nominal 1500 nm and lowered the window sensitivity.

Run from the repo root:  python reproduce/closure_flat_prior_check.py
Writes outputs/closure_flat_prior_check.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))
import element_observable_closure as C                           # noqa: E402

OUT = ROOT / "outputs/closure_flat_prior_check.json"
thr = json.load(open(ROOT / "outputs/calibrated_background_thresholds.json"))
band_el = np.round(np.array(thr["band01_elements_um"]), 2); band_fl = np.array(thr["band01_floors_m1"])
outer = [k for k, c in enumerate(band_el) if c > 9.25]
K_out = np.zeros((len(outer), C.N_STATE))
for n, c in enumerate(band_el[outer]):
    K_out[n, C.I_AER] = [np.mean(C.shell_o[jj][(C.lam_outer >= c - 0.05) & (C.lam_outer < c + 0.05 + 1e-9)]) for jj in range(7)]
S_out = np.diag(band_fl[outer] ** 2)
paper_od = np.array(thr["window_floors_m1"]) * C.OD_PER_M1
res = {}
res["round70_hybrid_band_3sig_prior1e3"] = 3 * C.joint_sigma_sil(C.SNR_REF, C.SIGMA_ETA_BASE, extra_rows=(K_out, S_out)) * C.RHO_SLANT_TO_SHELL
res["round70_window_3sig_prior1e3"] = 3 * C.joint_sigma_sil(C.SNR_REF, C.SIGMA_ETA_BASE) * C.RHO_SLANT_TO_SHELL
Sa = C.S_a.copy()
for k in range(C.I_AER.start, C.I_AER.stop):
    Sa[k, k] = 1e12
C.S_a_inv = np.linalg.inv(Sa)                                              # flat aerosol prior (1e6 in the parameters' units)
res["round70_hybrid_band_3sig_flat"] = 3 * C.joint_sigma_sil(C.SNR_REF, C.SIGMA_ETA_BASE, extra_rows=(K_out, S_out)) * C.RHO_SLANT_TO_SHELL
res["round70_window_3sig_flat"] = 3 * C.joint_sigma_sil(C.SNR_REF, C.SIGMA_ETA_BASE) * C.RHO_SLANT_TO_SHELL
res["round70_window_alone_3sig_flat"] = 3 * C.joint_sigma_sil(C.SNR_REF, C.SIGMA_ETA_BASE, with_triplet=False) * C.RHO_SLANT_TO_SHELL
res["paper_floors_joint_window_flat"] = 3 * C.joint_sigma_sil(C.SNR_REF, 0.0, override_cov=np.diag(paper_od ** 2)) * C.RHO_SLANT_TO_SHELL
res["note"] = "3 sigma per 0.5-km shell at 20 km; 'hybrid band' = window with the gas state + 32 outer elements on the adopted floors (round-70 construction); flat = aerosol prior 1e6"
print(json.dumps(res, indent=1))
json.dump(res, open(OUT, "w"), indent=1)
print("archived ->", OUT)
