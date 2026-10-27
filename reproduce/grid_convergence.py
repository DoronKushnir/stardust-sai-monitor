"""grid_convergence.py -- referee round 6 (paper 1; public port), item 3: convergence of the element
observables, the Jacobians, the covariance and the retrieved mass of the joint element-observable
model with the spectral grid.  reproduce/element_observable_closure.py is imported in three
subprocesses with CLOSURE_STEP = 0.005, 0.0025 (the archive) and 0.00125 cm^-1 (grids built by
reproduce/finegrid_window.py); each dumps y_i, the window Jacobian, the removal budget and
the joint silica uncertainty.  The forward-model error of a grid, dy = y(grid) - y(finest), is
propagated through the gain of the finest model to a mass error dM = G_sil dy (Tg per shell), the
quantity that decides whether the grid is adequate for the claimed sensitivity; the Richardson
estimate of the residual error of the finest grid assumes second-order convergence.

Run from the repo root (a few minutes once the 0.00125 grids exist):
  python reproduce/grid_convergence.py
Writes outputs/grid_convergence.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PY = sys.executable
STEPS = ("0.005", "0.0025", "0.00125")

DRIVER = r'''
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(sys.argv[1]); step = sys.argv[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))
import element_observable_closure as C
Kt, St = C.triplet_rows()
S_eps = C._blockdiag(C.photon_cov(C.y0, C.SNR_REF) + C.line_covariance(C.SIGMA_ETA_BASE), St)
K = np.vstack([C.K0, Kt])
Sinv = np.linalg.inv(S_eps)
def post(Sa_inv):
    S_hat = np.linalg.inv(K.T @ Sinv @ K + Sa_inv); return S_hat, S_hat @ K.T @ Sinv
S_hat, G = post(C.S_a_inv)
Sa_flat = C.S_a.copy()
for k in range(C.I_AER.start, C.I_AER.stop): Sa_flat[k, k] = 1e12
S_hat_f, G_f = post(np.linalg.inv(Sa_flat))
b = C.removal_budget(C.SNR_REF, C.SIGMA_ETA_BASE)
np.savez_compressed(ROOT / "outputs" / f"grid_convergence_s{step}.npz",
    y0=C.y0, y_gas0=C.y_gas0, y_mean=C.y_mean, K0=C.K0, G_sil=G[C.I_SIL], G_sil_flat=G_f[C.I_SIL],
    sigma_sil_slant=np.sqrt(S_hat[C.I_SIL, C.I_SIL]), sigma_sil_slant_flat=np.sqrt(S_hat_f[C.I_SIL, C.I_SIL]),
    rho=C.RHO_SLANT_TO_SHELL, removal=np.array([b["sigma_total"], b["photon"], b["line"], b["prior_smoothing"]]),
    n_nu=C.n_nu, step=float(step), tau_slant_880=C.tau_slant_880, S_hat_diag=np.sqrt(np.diag(S_hat)))
print("dumped", step, C.n_nu, C.y_gas0[C.i880], flush=True)
'''


def main():
    # the 0.005-cm^-1 grids are the 0.0025 grids subsampled (every other point), under the cache key the closure script expects
    import glob
    for f in glob.glob(str(ROOT / "data/trace_gases/xsec_cache/*_1067_1294_*_s0.0025.npz")):
        g = f.replace("_s0.0025.npz", "_s0.005.npz")
        if not os.path.exists(g):
            d = np.load(f); np.savez_compressed(g, nu=d["nu"][::2], coef=d["coef"][::2])
    drv = OUT / "grid_convergence_driver.py"
    drv.write_text(DRIVER)
    for step in STEPS:
        f = OUT / f"grid_convergence_s{step}.npz"
        if f.exists():
            continue
        env = dict(os.environ, CLOSURE_STEP=step)
        print(f"=== CLOSURE_STEP={step}", flush=True)
        r = subprocess.run([PY, str(drv), str(ROOT), step], env=env, cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit(f"driver failed for step {step}")
    d = {s: np.load(OUT / f"grid_convergence_s{s}.npz") for s in STEPS}
    fine = d["0.00125"]
    res = dict(script="reproduce/grid_convergence.py", round=74, steps_cm=[float(s) for s in STEPS],
               n_nu={s: int(d[s]["n_nu"]) for s in STEPS},
               y_gas_880={s: float(d[s]["y_gas0"][10]) for s in STEPS}, y_total_880={s: float(d[s]["y0"][10]) for s in STEPS},
               rho={s: float(d[s]["rho"]) for s in STEPS},
               removal_880={s: d[s]["removal"].tolist() for s in STEPS},
               sigma_sil_shell_3sig_window_triplet={s: float(3 * d[s]["sigma_sil_slant"] * d[s]["rho"]) for s in STEPS},
               sigma_sil_shell_3sig_window_triplet_flat_prior={s: float(3 * d[s]["sigma_sil_slant_flat"] * d[s]["rho"]) for s in STEPS})
    comp = {}
    for s in ("0.005", "0.0025"):
        dy = d[s]["y0"] - fine["y0"]
        dK = d[s]["K0"] - fine["K0"]
        dM = float(fine["G_sil"] @ np.r_[dy, np.zeros(3)] * fine["rho"])           # mass error (Tg per shell) if the retrieval used grid s on data from the finest model
        dM_f = float(fine["G_sil_flat"] @ np.r_[dy, np.zeros(3)] * fine["rho"])
        rel_K = np.abs(dK) / np.maximum(np.abs(fine["K0"]), 1e-12)
        big = np.abs(fine["K0"]) > 1e-3 * np.abs(fine["K0"]).max()
        comp[s] = dict(max_abs_dy_od=float(np.abs(dy).max()), dy_880_od=float(dy[10]), dy_od=dy.tolist(),
                       mass_error_tg_per_shell=dM, mass_error_tg_per_shell_flat_prior=dM_f,
                       max_rel_dK_significant_entries=float(rel_K[big].max()), median_rel_dK_significant_entries=float(np.median(rel_K[big])),
                       sigma_sil_rel_change=float(d[s]["sigma_sil_slant"] / fine["sigma_sil_slant"] - 1),
                       removal_rel_change=float(d[s]["removal"][0] / fine["removal"][0] - 1),
                       posterior_sd_max_rel_change=float(np.max(np.abs(d[s]["S_hat_diag"] / fine["S_hat_diag"] - 1))))
    # Richardson: second-order convergence -> error of the finest grid ~ (change from 0.0025 to 0.00125)/3
    comp["richardson_estimate_finest"] = dict(
        dy_880_od=float(comp["0.0025"]["dy_880_od"] / 3), max_abs_dy_od=float(comp["0.0025"]["max_abs_dy_od"] / 3),
        mass_error_tg_per_shell=float(comp["0.0025"]["mass_error_tg_per_shell"] / 3),
        convergence_order_880=float(np.log2(abs(comp["0.005"]["dy_880_od"] - comp["0.0025"]["dy_880_od"]) / max(abs(comp["0.0025"]["dy_880_od"]), 1e-15))) if comp["0.0025"]["dy_880_od"] != 0 else None,
        note="order from the ratio of successive differences at the 8.80-um element; the 0.001-cm^-1 single-element check of round 70 is superseded by this window-wide comparison")
    res["comparison_to_finest"] = comp
    print(json.dumps({k: v for k, v in res.items() if k != "comparison_to_finest"}, indent=1))
    for s in ("0.005", "0.0025"):
        c = comp[s]
        print(f"grid {s}: max |dy| {c['max_abs_dy_od']:.2e} OD (8.80: {c['dy_880_od']:+.2e}); mass error {c['mass_error_tg_per_shell']:+.4f} Tg/shell "
              f"(flat prior {c['mass_error_tg_per_shell_flat_prior']:+.4f}); sigma_sil rel change {c['sigma_sil_rel_change']:+.3f}; removal {c['removal_rel_change']:+.3f}; "
              f"K rel change (median/max over significant entries) {c['median_rel_dK_significant_entries']:.2e}/{c['max_rel_dK_significant_entries']:.2e}")
    print("Richardson residual of the finest grid:", comp["richardson_estimate_finest"])
    with open(OUT / "grid_convergence.json", "w") as f:
        json.dump(res, f, indent=1)
    print("archived ->", OUT / "grid_convergence.json")


if __name__ == "__main__":
    main()
