"""multitangent_retrieval.py -- round 75 (paper 1, referee round 6 item 5, left open in
round 74): a multi-tangent forward and inverse model under the adopted measurement model, with the
field of view, the tangent sampling, the common-reference errors and the cross-height covariance,
so that the per-shell silica uncertainty is RETRIEVED from independently varying shells instead of
being converted from a single-tangent amplitude by the slant-to-shell factor.

Forward model.  Tangent heights 14-40 km (the 0.5-km shell grid of the paper, edge convention),
each tangent evaluated with reproduce/adopted_model_engine.py (line-resolved gas optical
depths at that chord, aerosol slant columns, the exact Jacobian of the element observable, the
per-line responses).  The silica state is one amplitude per 0.5-km shell, a_j, in units of the
20-km extinction per Tg of the homogeneous layer (the units of Sect. 6: a_j = 1 is a shell holding
the extinction of the 1-Tg layer's 20-km shell), with the 20-km spectral shape in every shell; the
silica column of tangent i for shell j is P_ij <alpha_20(lambda)>_T,i, the exact path matrix times
the transmission-weighted element mean at that tangent.  The gas state (20 nodes), the temperature
offset, the CFC-12 amplitude, the 15 source groups, the six sulfate parameters and the two response
parameters are common to all tangents (the profiles are the state).  The heritage triplet enters
as per-shell rows at every shell.  Rows: 52 tangents x (47 elements + 3 channels) = 2600;
state: 45 + 52 = 97 parameters.

Field of view and sampling.  The element observable of a tangent integrates the transmitted radiance
over a boxcar field of view of full width W (0 = point tangent; 0.5 km; 1 km): y_F = -ln sum_h w_h exp(-y_h)
over sub-rays on a 0.25-km grid (trapezoid weights; a 0.125-km grid checks the quadrature), and every
Jacobian column is the radiance-weighted mean of the sub-ray columns (round 76, referee round 7 item 4;
round 75 averaged the optical depths and Jacobians with the geometric weights).  The sampling is every
0.5 km (every shell) or every 1 km.

Covariance.  (a) radiometric: the round-70 law e^{y/2}/2000 per sample, independent, or the
separated law of item 7 (detector e^y/S_det, systematic floor 1/2000, shared reference), with the
floor either independent per sample or COMMON to every tangent of the event per element (the
common-reference scenario); (b) the per-line database scatter, fully coherent across tangents (one
database for the mission); (c) the discrepancy term, Gaussian across elements (l = 0.142 um) and
independent between tangents (as declared) or vertically correlated with a Gaussian length of 1 or
2 km (scenario).  Flat priors on the shell amplitudes and the sulfate parameters, round-70 priors
otherwise.

Results per scenario: 3 sigma of every shell (16-24 km) in 20-km units, the 20-km value against
the single-tangent shell-equivalent 0.119 Tg, the correlation of adjacent shells, the averaging
kernel of the 20-km shell and the degrees of freedom of the silica profile, the uncertainty of the
single profile amplitude of the prescribed layer retrieved from all tangents, and the O3 node
uncertainties against the single-tangent fit.

Run from the repo root (about 10 min):  python reproduce/multitangent_retrieval.py
Writes outputs/multitangent_retrieval.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))

import adopted_model_engine as E                                  # noqa: E402

C, cbt = E.C, E.cbt
OUT = ROOT / "outputs/multitangent_retrieval.json"
log = E.log
Z_SHELL = np.arange(14_000.0, 40_000.0, E.DZ_M)                           # shell bottoms = tangent heights (edge convention)
N_SH = len(Z_SHELL)
FINE = np.arange(13_500.0, 40_500.1, 125.0)                               # fine tangents for the field of view (125-m sub-rays)
FOV = {"point": {0.0: 1.0}, "0.5 km": {-250.0: 0.25, 0.0: 0.5, 250.0: 0.25},
       "1 km": {-500.0: 0.125, -250.0: 0.25, 0.0: 0.25, 250.0: 0.25, 500.0: 0.125},
       "0.5 km (5 sub-rays)": {-250.0: 0.125, -125.0: 0.25, 0.0: 0.25, 125.0: 0.25, 250.0: 0.125},
       "1 km (9 sub-rays)": {-500.0: 1 / 16, -375.0: 1 / 8, -250.0: 1 / 8, -125.0: 1 / 8, 0.0: 1 / 8, 125.0: 1 / 8, 250.0: 1 / 8, 375.0: 1 / 8, 500.0: 1 / 16}}
S_DET, N_REF, S_CAP = 1.4e4, 100, 2000.0
n_mir, win, outer = E.n_mir, E.win, E.outer
N_FIX = C.N_STATE - 1                                                     # the round-70 state without the silica amplitude


def path_row(h, zb=Z_SHELL):
    """path length [m] of the ray with tangent h through every shell [zb_j, zb_j + dz]."""
    R = E.R_EARTH_M; r2 = (R + h) ** 2
    rb, rt = R + zb, R + zb + E.DZ_M
    return 2.0 * (np.sqrt(np.maximum(rt ** 2 - r2, 0)) - np.sqrt(np.maximum(rb ** 2 - r2, 0)))


def path_row_dz(h, zb, dz):
    R = E.R_EARTH_M; r2 = (R + h) ** 2
    rb, rt = R + zb, R + zb + dz
    return 2.0 * (np.sqrt(np.maximum(rt ** 2 - r2, 0)) - np.sqrt(np.maximum(rb ** 2 - r2, 0)))


def expand(v):
    """map a 46-vector of the round-70 state onto the 97-state (silica slot -> shells)."""
    out = np.zeros(NP); out[:C.I_SIL] = v[:C.I_SIL]; out[C.I_SIL + N_SH:] = v[C.I_SIL + 1:]
    return out


def main():
    t0 = time.time()
    # ---- the 20-km reference silica spectrum (per-shell extinction per Tg of the homogeneous layer at 20 km)
    a20_fine = np.interp(C.lam_um, C.lam_coarse, C.shell_c[0])             # m^-1 per Tg on the fine grid
    a20_out = np.array([C.shell_o[0][m].mean() for m in E._out_masks])
    a20_trip = C.shell_trip[0]
    # ---- fine-tangent models
    # sulfate modes: per-shell spectra (shape common to all shells, amplitude c_f(z), c_c(z) of the prescribed background)
    i880c = int(np.argmin(np.abs(C.lam_coarse - 8.80)))
    c_f = C.prof_c[1][:, i880c]; c_c = C.prof_c[2][:, i880c]
    jref = int(np.argmax(c_f))
    shape_f_fine = np.interp(C.lam_um, C.lam_coarse, C.prof_c[1][jref] / c_f[jref]); shape_c_fine = np.interp(C.lam_um, C.lam_coarse, C.prof_c[2][jref] / c_c[jref])
    shape_f_out = np.array([(C.prof_o[1][jref] / c_f[jref])[m].mean() for m in E._out_masks]); shape_c_out = np.array([(C.prof_o[2][jref] / c_c[jref])[m].mean() for m in E._out_masks])
    fine = {}
    for k, h in enumerate(FINE):
        tm = E.TangentModel(h)
        s_el = np.zeros(n_mir); s_el[win] = C.element_mean(tm.tau0, a20_fine); s_el[outer] = a20_out
        sf = np.zeros(n_mir); sf[win] = C.element_mean(tm.tau0, shape_f_fine); sf[outer] = shape_f_out
        sc = np.zeros(n_mir); sc[win] = C.element_mean(tm.tau0, shape_c_fine); sc[outer] = shape_c_out
        fine[h] = dict(y=tm.y_mir, K=tm.K_mir, Kt=tm.K_tri, St=np.diag(tm.S_tri), A={g: tm.A_line[g] for g in C.GASES},
                       s_el=s_el, sulf_el=(sf, sc), iz=tm.iz, h=h)
        if k % 40 == 0:
            log(f"  fine tangent {h/1e3:.2f} km ({k+1}/{len(FINE)}), {time.time()-t0:.0f} s")
    # ---- assemble per (FOV, sampling)
    def assemble(fov, step_m, sulfate_per_shell=False):
        """rows of all tangents on the sampling grid (shells of thickness step_m); state = fixed block (gas, T, CFC, groups) +
        silica per shell + [sulfate: 6 global, or per-shell A_f, A_c and 4 global PSD parameters] + response."""
        tangents = Z_SHELL[::int(round(step_m / E.DZ_M))]
        shells = tangents                                            # shell bottoms = tangent heights, thickness step_m
        n_t, n_s = len(tangents), len(shells)
        n_sulf = 2 * n_s + 4 if sulfate_per_shell else 6
        n_p = C.I_SIL + n_s + n_sulf + 2
        i_sh = slice(C.I_SIL, C.I_SIL + n_s); i_af = C.I_SIL + n_s; i_ac = i_af + (n_s if sulfate_per_shell else 1)
        i_psd = C.I_SIL + n_s + (2 * n_s if sulfate_per_shell else 2); i_resp = n_p - 2
        iz_sh = [E.iz_of(z) for z in shells]
        K = np.zeros((n_t * 50, n_p)); y = np.zeros(n_t * 47); St = np.zeros(n_t * 3)
        A = {g: np.zeros((C.A_line[g].shape[0], n_t * 15)) for g in C.GASES}

        def path(h):
            return path_row(h, shells) if step_m == E.DZ_M else path_row_dz(h, shells, step_m)

        for i, h in enumerate(tangents):
            r0 = i * 50
            subs = [(fine[h + d], w) for d, w in FOV[fov].items()]
            # radiance averaging: the field integrates exp(-y_h) over the sub-rays; the observable is -ln of the mean and the
            # derivative of any column is the transmission-weighted mean of the sub-ray derivatives (per element)
            T_sub = np.array([w * np.exp(-f["y"]) for f, w in subs])                         # (n_sub, 47)
            T_tot = T_sub.sum(axis=0)
            W_sub = T_sub / T_tot[None, :]                                                   # radiance weights per element
            y[i * 47:(i + 1) * 47] = -np.log(T_tot)
            for (f, w), Wr in zip(subs, W_sub):
                Kf = f["K"]; d = f["h"] - h
                K[r0:r0 + 47, :C.I_SIL] += Wr[:, None] * Kf[:, :C.I_SIL]
                K[r0:r0 + 47, i_sh] += Wr[:, None] * np.outer(f["s_el"], path(h + d))        # silica shells: P_ij <alpha_20>_T,i
                K[r0:r0 + 47, i_psd:i_psd + 4] += Wr[:, None] * Kf[:, C.I_SIL + 3:C.I_SIL + 7]
                K[r0:r0 + 47, i_resp:] += Wr[:, None] * Kf[:, C.I_SHIFT:]
                if sulfate_per_shell:
                    P = path(h + d)
                    K[r0:r0 + 47, i_af:i_af + n_s] += Wr[:, None] * np.outer(np.ones(47), P) * f["sulf_el"][0][:, None] * np.array([c_f[j] for j in iz_sh])[None, :]
                    K[r0:r0 + 47, i_ac:i_ac + n_s] += Wr[:, None] * np.outer(np.ones(47), P) * f["sulf_el"][1][:, None] * np.array([c_c[j] for j in iz_sh])[None, :]
                else:
                    K[r0:r0 + 47, i_af] += Wr * Kf[:, C.I_SIL + 1]; K[r0:r0 + 47, i_ac] += Wr * Kf[:, C.I_SIL + 2]
                for g in C.GASES:
                    A[g][:, i * 15:(i + 1) * 15] += Wr[win][None, :] * f["A"][g]
            f0 = fine[h]; Kt = f0["Kt"]
            j = int(np.argmin(np.abs(shells - h)))
            K[r0 + 47:r0 + 50, C.I_SIL + j] = a20_trip
            K[r0 + 47:r0 + 50, i_psd:i_psd + 4] = Kt[:, C.I_SIL + 3:C.I_SIL + 7]
            if sulfate_per_shell:
                K[r0 + 47:r0 + 50, i_af + j] = C.prof_t[1][iz_sh[j]]; K[r0 + 47:r0 + 50, i_ac + j] = C.prof_t[2][iz_sh[j]]
            else:
                K[r0 + 47:r0 + 50, i_af] = Kt[:, C.I_SIL + 1]; K[r0 + 47:r0 + 50, i_ac] = Kt[:, C.I_SIL + 2]
            St[i * 3:(i + 1) * 3] = f0["St"]
        layout = dict(n_p=n_p, n_s=n_s, i_sh=i_sh, i_sulf=slice(C.I_SIL + n_s, C.I_SIL + n_s + n_sulf), i_resp=i_resp, shells=shells, sulfate_per_shell=sulfate_per_shell)
        return tangents, K, y, St, A, layout

    def covariance(tangents, y, St, A, law="round70", floor_common=False, disc_vert_km=None):
        n_t = len(tangents); n = n_t * 50
        S = np.zeros((n, n))
        mir = np.array([i * 50 + e for i in range(n_t) for e in range(47)]); tri = np.array([i * 50 + 47 + c for i in range(n_t) for c in range(3)])
        if law == "round70":
            S[mir, mir] += (np.exp(y / 2.0) / S_CAP) ** 2
        else:
            S[mir, mir] += (np.exp(y) / S_DET) ** 2
            S[mir, mir] += 0.0 if floor_common else (1.0 / S_CAP) ** 2
            for e in range(47):                                  # shared reference (and the common floor) per element across tangents
                idx = mir[e::47]
                S[np.ix_(idx, idx)] += (1.0 / (S_DET * np.sqrt(N_REF))) ** 2 + ((1.0 / S_CAP) ** 2 if floor_common else 0.0)
        S[tri, tri] += St
        # per-line scatter, coherent across tangents (window elements)
        wrows = np.array([i * 50 + e for i in range(n_t) for e in win])
        Cl = sum(E.SIGMA_ETA ** 2 * (A[g].T @ A[g]) for g in C.GASES)
        S[np.ix_(wrows, wrows)] += Cl
        # discrepancy term: Gaussian across elements, independent (or Gaussian-correlated) across tangents
        s = E.s_meas_od
        V = np.eye(n_t) if disc_vert_km is None else np.exp(-((tangents[:, None] - tangents[None, :]) / (disc_vert_km * 1e3)) ** 2)
        for i in range(n_t):
            for k in range(n_t):
                if V[i, k] != 0.0:
                    S[np.ix_(mir[i * 47:(i + 1) * 47], mir[k * 47:(k + 1) * 47])] += V[i, k] * np.outer(s, s) * E.R_GAUSS
        return S

    def prior_inv_for(L):
        n_p = L["n_p"]
        Sa = np.zeros((n_p, n_p))
        Sa[:C.I_SIL, :C.I_SIL] = C.S_a[:C.I_SIL, :C.I_SIL]
        for k in range(C.I_SIL, n_p - 2):
            Sa[k, k] = E.AER_FLAT ** 2
        Sa[n_p - 2, n_p - 2] = C.S_a[C.I_SHIFT, C.I_SHIFT]; Sa[n_p - 1, n_p - 1] = C.S_a[C.I_SOFT, C.I_SOFT]
        return np.linalg.inv(Sa)
    # the prescribed profile in shell units (for the single-amplitude retrieval): a_j = c_j / c_20
    c_prof = np.array([float(np.mean(C.prof_c[0][E.iz_of(z), :][E._coarse_masks[C.i880]])) for z in Z_SHELL])
    prof_units = c_prof / c_prof[int(np.argmin(np.abs(Z_SHELL - 20e3)))]

    def solve(K, S, L, label):
        Sinv = np.linalg.inv(S)
        S_hat = np.linalg.inv(K.T @ Sinv @ K + prior_inv_for(L))
        G = S_hat @ K.T @ Sinv; Av = G @ K
        sd = np.sqrt(np.diag(S_hat))
        shells = L["shells"]; i_sh = L["i_sh"]; n_s = L["n_s"]
        j20 = int(np.argmin(np.abs(shells - 20e3))); jj = C.I_SIL + j20
        corr = S_hat[jj, jj + 1] / (sd[jj] * sd[jj + 1]) if j20 + 1 < n_s else float("nan")
        corr_m = S_hat[jj, jj - 1] / (sd[jj] * sd[jj - 1])
        sel = (shells >= 16e3) & (shells <= 24e3)
        # one profile amplitude of the prescribed layer from all tangents (silica shells tied to the prescribed shape)
        pu = np.interp(shells, Z_SHELL, prof_units)
        Kp = np.hstack([K[:, :C.I_SIL], (K[:, i_sh] @ pu)[:, None], K[:, C.I_SIL + n_s:]])
        Lp = dict(n_p=Kp.shape[1]); Sp = np.linalg.inv(Kp.T @ Sinv @ Kp + prior_inv_for(Lp))
        r = dict(label=label, sulfate_per_shell=L["sulfate_per_shell"], shell_thickness_km=float((shells[1] - shells[0]) / 1e3), n_state=int(L["n_p"]), n_rows=int(K.shape[0]),
                 mmin_3sig_20km_tg=3 * sd[jj], mmin_3sig_shells_16_24=(3 * sd[i_sh][sel]).tolist(),
                 shells_km=(shells[sel] / 1e3).tolist(), corr_adjacent_above=float(corr), corr_adjacent_below=float(corr_m),
                 averaging_kernel_20km_row=Av[jj, i_sh].tolist(), ak_self_20km=float(Av[jj, jj]), dofs_silica_profile=float(np.trace(Av[i_sh, i_sh])),
                 dofs_total=float(np.trace(Av)), profile_amplitude_3sig_tg=3 * float(np.sqrt(Sp[C.I_SIL, C.I_SIL])),
                 o3_node_sd=sd[C.I_GAS["o3"]].tolist(), resp_sd=sd[L["n_p"] - 2:].tolist())
        if L["sulfate_per_shell"]:
            r["sulfate_amplitude_sd_20km"] = [float(sd[L["i_sulf"].start + j20]), float(sd[L["i_sulf"].start + n_s + j20])]
        else:
            r["sulfate_sd"] = sd[L["i_sulf"]].tolist()
        log(f"  {label:60s}: 20-km shell 3 sigma {r['mmin_3sig_20km_tg']:.4f} Tg; adjacent corr {corr:+.2f}/{corr_m:+.2f}; AK self {r['ak_self_20km']:.2f}; "
            f"DOF silica {r['dofs_silica_profile']:.1f}; profile amplitude {r['profile_amplitude_3sig_tg']:.4f} Tg; 16-24 km: " +
            " ".join(f"{v:.3f}" for v in r['mmin_3sig_shells_16_24'][::2]))
        return r

    res = dict(script="reproduce/multitangent_retrieval.py", round=75, shells_km=(Z_SHELL / 1e3).tolist(), n_rows_per_tangent=50,
               single_tangent_20km_shell_equivalent_tg=float(E.UM["models"]["B_primary"]["band_triplet"]), scenarios={})
    for per_shell, tag in ((False, "sulfate_prescribed"), (True, "sulfate_per_shell")):
        lab = "sulfate profile prescribed (6 parameters)" if not per_shell else "sulfate amplitudes free per shell (2 per shell + 4)"
        t_, K_, y_, St_, A_, L_ = assemble("point", 500.0, per_shell)
        base = covariance(t_, y_, St_, A_)
        res["scenarios"][f"{tag}_base"] = solve(K_, base, L_, f"{lab}; point tangent, 0.5-km sampling")
        res["scenarios"][f"{tag}_separated_floor_indep"] = solve(K_, covariance(t_, y_, St_, A_, law="separated"), L_, f"{lab}; separated law, floor independent")
        res["scenarios"][f"{tag}_separated_floor_common"] = solve(K_, covariance(t_, y_, St_, A_, law="separated", floor_common=True), L_, f"{lab}; separated law, floor common to all tangents")
        res["scenarios"][f"{tag}_disc_vertical_1km"] = solve(K_, covariance(t_, y_, St_, A_, disc_vert_km=1.0), L_, f"{lab}; discrepancy term vertically correlated 1 km")
        res["scenarios"][f"{tag}_disc_vertical_2km"] = solve(K_, covariance(t_, y_, St_, A_, disc_vert_km=2.0), L_, f"{lab}; discrepancy term vertically correlated 2 km")
        S_li = base.copy()
        wrows = np.array([i * 50 + e for i in range(len(t_)) for e in win])
        Cl = sum(E.SIGMA_ETA ** 2 * (A_[g].T @ A_[g]) for g in C.GASES)
        S_li[np.ix_(wrows, wrows)] -= Cl
        for i in range(len(t_)):
            idx = wrows[i * 15:(i + 1) * 15]; S_li[np.ix_(idx, idx)] += Cl[np.ix_(np.arange(i * 15, (i + 1) * 15), np.arange(i * 15, (i + 1) * 15))]
        res["scenarios"][f"{tag}_line_scatter_independent"] = solve(K_, S_li, L_, f"{lab}; database scatter independent between tangents (limit)")
        for fov in ("0.5 km", "1 km", "0.5 km (5 sub-rays)", "1 km (9 sub-rays)"):
            t2, K2, y2, St2, A2, L2 = assemble(fov, 500.0, per_shell)
            key = fov.replace(" ", "").replace("(", "_").replace(")", "").replace("-", "")
            res["scenarios"][f"{tag}_fov_{key}"] = solve(K2, covariance(t2, y2, St2, A2), L2, f"{lab}; field of view {fov}, 0.5-km sampling")
        t2, K2, y2, St2, A2, L2 = assemble("1 km", 1000.0, per_shell)
        res["scenarios"][f"{tag}_fov_1km_1km_shells"] = solve(K2, covariance(t2, y2, St2, A2), L2, f"{lab}; field of view 1 km, 1-km sampling and shells")
    res["runtime_s"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    log("archived ->", OUT, f"({res['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
