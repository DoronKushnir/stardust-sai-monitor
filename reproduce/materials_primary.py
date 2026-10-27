"""materials_primary.py -- referee round 6 (paper 1; public port), item 2 (last paragraph): the alternative
materials (calcite, dolomite, alumina; orientation-averaged optics of round 70) and silica under
the primary measurement model of reproduce/uncertainty_model.py, i.e. the joint
element-observable retrieval with the full covariance (photon term, per-line database scatter on
the window elements, measured ACE scatter per element with its Gaussian inter-element
correlation), the 38-parameter state with a flat aerosol prior, and the heritage triplet; and,
for reference, the two-stage diagonal-floor scenario on the same rows (Table 8).

Each material replaces the silica column: its slant optical depth per Tg of the prescribed
profile (constant mixing ratio 16-24 km, the material's density) on the line-resolved window
grid (transmission-weighted element means) and on the outer elements, and its per-shell
extinction at 20 km in the triplet rows.  Thresholds are 3 sigma per 0.5-km shell at 20 km
(slant amplitude x the material's own slant-to-shell factor, identical to silica's within
round-off because the profile shape is common).

Run from the repo root (about 10 min of Mie integrals):  python reproduce/materials_primary.py
Writes outputs/materials_primary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))

import element_observable_closure as C                           # noqa: E402
import calibrated_background_thresholds as cbt                            # noqa: E402
from materials_uniaxial import RayTables, ConventionRI, MAT_CLS, MAT_FACTORY  # noqa: E402
from saimon.onion_peel import M1_PER_OD, OD_PER_M1                         # noqa: E402

OUT = ROOT / "outputs/materials_primary.json"
UM = json.load(open(ROOT / "outputs/uncertainty_model.json"))
thr = json.load(open(ROOT / "outputs/calibrated_background_thresholds.json"))
band_el = np.round(np.array(thr["band01_elements_um"], float), 2)
s_meas = np.array(UM["s_measured_od"]); s_adopt = np.array(UM["s_adopted_od"]); y_mir = np.array(UM["y_gas_elements"])
win = np.array(UM["window_index"]); outer = np.array(UM["outer_index"]); n_mir = len(band_el)
ell = UM["correlation"]["gaussian_l_um"]
AER_FLAT = 1e6


def log(*a):
    print(*a, flush=True)


# ---- covariance of the primary model (copied from uncertainty_model.py)
D = np.abs(band_el[:, None] - band_el[None, :])
R_gauss = np.exp(-(D / ell) ** 2)
S_noise = np.diag((np.exp(y_mir / 2.0) / C.SNR_REF) ** 2)
S_line = np.zeros((n_mir, n_mir)); S_line[np.ix_(win, win)] = C.line_covariance(C.SIGMA_ETA_BASE)
S_B = S_noise + S_line + np.outer(s_meas, s_meas) * R_gauss
Sa = C.S_a.copy()
for k in range(C.I_AER.start, C.I_AER.stop):
    Sa[k, k] = AER_FLAT ** 2
S_A_INV = np.linalg.inv(Sa)
K_tri0, S_tri = C.triplet_rows()
sl20_o = C.slant_at(C.H_TAN_M, C.prof_o)
K_mir0 = np.zeros((n_mir, C.N_STATE)); K_mir0[win] = C.K0
for n, k in enumerate(outer):
    m = (C.lam_outer >= band_el[k] - C.EL_HALF_UM) & (C.lam_outer < band_el[k] + C.EL_HALF_UM + 1e-9)
    K_mir0[k, C.I_AER] = sl20_o[:, m].mean(axis=1)


def joint_sigma(K_mir, K_tri, rows="band", triplet=True):
    idx = np.arange(n_mir) if rows == "band" else win
    K = K_mir[idx]; S = S_B[np.ix_(idx, idx)]
    if triplet:
        K = np.vstack([K, K_tri]); S = C._blockdiag(S, S_tri)
    Sinv = np.linalg.inv(S)
    S_hat = np.linalg.inv(K.T @ Sinv @ K + S_A_INV)
    return float(np.sqrt(S_hat[C.I_SIL, C.I_SIL]))


def two_stage(floors_od, shell_rows, shell_trip, rows="band", triplet=True):
    idx = np.arange(n_mir) if rows == "band" else win
    t_all = np.r_[shell_trip[0], shell_rows[idx, 0]]
    nuis = [np.r_[shell_trip[j], shell_rows[idx, j]] for j in range(1, 7)]
    sig_all = np.r_[np.sqrt(np.diag(S_tri)), floors_od[idx] * M1_PER_OD]
    ch = np.arange(len(t_all)) if triplet else np.arange(3, len(t_all))
    sa, _, R2 = cbt.marginal(ch, nuis, sig_all, t_all)
    return float(sa)


def material_rows(layer):
    """slant rows (window: T-weighted element means on the fine grid; outer: boxcar means), triplet per-shell row,
    per-shell band rows (for the two-stage reference) and the slant-to-shell factor of this material."""
    prof_c = layer.extinction_profile_m1(C.lam_coarse * 1e-6)              # (nz, n_wl)
    prof_o = layer.extinction_profile_m1(C.lam_outer * 1e-6)
    prof_t = layer.extinction_profile_m1(C.WL_TRIPLET_NM * 1e-9)
    ch20 = C.tangent_to_slant_paths(np.array([C.H_TAN_M]), cbt.ALT)[0]
    sl_c = ch20 @ prof_c; sl_o = ch20 @ prof_o
    tau_fine = np.interp(C.lam_um, C.lam_coarse, sl_c)
    col_win = C.element_mean(C.tau0, tau_fine)
    col_out = np.array([sl_o[(C.lam_outer >= band_el[k] - C.EL_HALF_UM) & (C.lam_outer < band_el[k] + C.EL_HALF_UM + 1e-9)].mean() for k in outer])
    shell_rows = np.zeros(n_mir)
    for n, c in enumerate(band_el):
        if c < 9.25:
            m = (C.lam_coarse >= c - C.EL_HALF_UM) & (C.lam_coarse < c + C.EL_HALF_UM); shell_rows[n] = prof_c[cbt.iz, m].mean()
        else:
            m = (C.lam_outer >= c - C.EL_HALF_UM) & (C.lam_outer < c + C.EL_HALF_UM + 1e-9); shell_rows[n] = prof_o[cbt.iz, m].mean()
    col = np.zeros(n_mir); col[win] = col_win; col[outer] = col_out
    rho = float(col[C.i880] / (shell_rows[C.i880] * OD_PER_M1)) if shell_rows[C.i880] > 0 else float("nan")
    return col, prof_t[cbt.iz], shell_rows, rho


def main():
    t0 = time.time()
    res = dict(script="reproduce/materials_primary.py", round=74, correlation_length_um=ell, aerosol_prior=AER_FLAT, materials={})
    # the paper's shell rows for the sulfate columns (two-stage reference)
    shell_bg = np.zeros((n_mir, 7))
    for n, c in enumerate(band_el):
        if c < 9.25:
            m = (C.lam_coarse >= c - C.EL_HALF_UM) & (C.lam_coarse < c + C.EL_HALF_UM); shell_bg[n] = C.shell_c[:, m].mean(axis=1)
        else:
            m = (C.lam_outer >= c - C.EL_HALF_UM) & (C.lam_outer < c + C.EL_HALF_UM + 1e-9); shell_bg[n] = C.shell_o[:, m].mean(axis=1)
    layers = {"silica": C.sil_layer}
    for mat, cls in MAT_CLS.items():
        layers[mat] = MAT_FACTORY[mat](cbt.ALT, 1.0, refractive_index=ConventionRI(RayTables(cls), "b"))
    for mat, layer in layers.items():
        log(f"--- {mat}")
        col, trip, shell_rows, rho = material_rows(layer)
        K_mir = K_mir0.copy(); K_mir[:, C.I_SIL] = col
        K_tri = K_tri0.copy(); K_tri[:, C.I_SIL] = trip
        sb = shell_bg.copy(); sb[:, 0] = shell_rows
        j = int(np.argmax(np.abs(col[outer]))) if mat != "silica" else C.i880
        r = dict(rho_slant_to_shell=rho,
                 primary=dict(band_triplet=3 * joint_sigma(K_mir, K_tri, "band", True) * rho, band_alone=3 * joint_sigma(K_mir, K_tri, "band", False) * rho,
                              window_triplet=3 * joint_sigma(K_mir, K_tri, "window", True) * rho),
                 two_stage_adopted_floors=dict(band_triplet=3 * two_stage(s_adopt, sb, np.vstack([trip[None, :], C.shell_trip[1:]]), "band", True),
                                               window_triplet=3 * two_stage(s_adopt, sb, np.vstack([trip[None, :], C.shell_trip[1:]]), "window", True)),
                 two_stage_measured_floors=dict(band_triplet=3 * two_stage(s_meas, sb, np.vstack([trip[None, :], C.shell_trip[1:]]), "band", True)),
                 slant_od_per_tg_elements=col.tolist(), shell_ext_m1_per_tg_elements=shell_rows.tolist(), triplet_shell_ext_m1_per_tg=trip.tolist(),
                 peak_element_um=float(band_el[np.argmax(shell_rows)]), peak_shell_ext_m1_per_tg=float(shell_rows.max()))
        res["materials"][mat] = r
        log(f"  rho {rho:.3f}; primary band {r['primary']['band_triplet']:.4f} (alone {r['primary']['band_alone']:.4f}), window {r['primary']['window_triplet']:.4f}; "
            f"two-stage adopted {r['two_stage_adopted_floors']['band_triplet']:.4f} (window {r['two_stage_adopted_floors']['window_triplet']:.3f}), "
            f"measured {r['two_stage_measured_floors']['band_triplet']:.4f}; peak {r['peak_element_um']:.2f} um {r['peak_shell_ext_m1_per_tg']:.3e}; {time.time()-t0:.0f} s")
    res["ratios_primary_over_two_stage_adopted"] = {m: v["primary"]["band_triplet"] / v["two_stage_adopted_floors"]["band_triplet"] for m, v in res["materials"].items()}
    res["runtime_s"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    log("archived ->", OUT)


if __name__ == "__main__":
    main()
