"""adopted_model_engine.py -- round 75 (paper 1; public port, completion of the referee-round-6 items left
open in round 74): the adopted measurement model of reproduce/uncertainty_model.py (Table 2),
evaluated at ANY tangent height, background amplitude, ozone load, silica (or other material)
template and element covariance, so that the robustness scans (Table 6), the altitude scan, the
latitude maps (Sect. 6, Table 7, Fig. 5), the far-infrared columns (Table 8) and a multi-tangent
inverse (Appendix D9) can all be evaluated under the one model the paper quotes.

The module imports the round-70 element-observable forward model (line-resolved 0.0025-cm^-1 grids,
15 window elements, 46-parameter state: 4 gases x 5 nodes, T_eff, CFC-12, 15 source groups, 7
aerosol parameters, 2 response parameters) and rebuilds, per tangent height h:

  * the gas slant optical depths per node (exact: the per-altitude cross sections of the cached
    node grids with the chord of h), the CFC-12 slant OD, T_eff and P_eff of the chord;
  * the per-line Voigt profiles at (T_eff(h), P_eff(h)) and the chord column (exact), hence the
    transmission-weighted line responses and the per-line scatter covariance at h;
  * the temperature-derivative column a_T: reused from the 20-km reference (the relative derivative
    d ln sigma / dT varies slowly with T_eff; the column is a nuisance with a 2-K prior);
  * the aerosol slant columns (silica or another material template, the two sulfate modes and their
    four PSD derivatives) from the cached per-shell Mie profiles and the chord of h; the background
    amplitude may be scaled (g) and the ozone profile scaled (o3);
  * the element observable y_i(h) and the exact Jacobian (transmission-weighted element means at h;
    the two response columns by central differences as in round 70);
  * the 32 outer elements (9.3-13.3 um) as aerosol-only slant rows; their gas optical depth (which
    enters only the radiometric term e^{y/2}/S) is the 20-km landscape value scaled by the ratio of
    the O3 slant column at h to that at 20 km (the outer elements carry no line-resolved model; the
    radiometric term there is 5e-4 against a discrepancy term of 2-10e-3);
  * the heritage triplet as per-shell extinction rows at the shell of h, with the Level-2
    fractional errors on the local background and the SNR-2000 floor inflated by the Rayleigh
    self-attenuation relative to 20 km (as the 2-D maps do).

The adopted covariance at h (Table 2): radiometric e^{y(h)/2}/2000 per element; the per-line
database scatter sigma_eta = 0.02 through the line responses at h (window elements); the
model-discrepancy term, the measured ACE scatter per element (19-22-km bin, unscaled) with the
Gaussian inter-element correlation of length 0.142 um, either CONSTANT with altitude (the model as
stated in Table 2: the term is a scatter across soundings, declared independent between tangent
heights, with no altitude dependence specified) or SCALED by the measured ACE altitude bins
(16-19 / 19-22 / 22-25 km at the 8.80-um element, piecewise constant, as the 'measured
altitude-dependent floor' scenario of Table 7).  Flat aerosol prior (1e6), round-70 priors
otherwise.  Thresholds are 3 sigma per 0.5-km shell: the slant amplitude (Tg of the prescribed
profile) times the slant-to-shell factor rho(h) read from the silica row at the 8.80-um element
and the per-shell extinction of the prescribed profile at h (rho(20 km) = 2.61, round 70).

This is a library; the round-75 scripts import it.  Importing it imports the round-70 module
(3 s) and builds the 20-km reference (a further ~3 s).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import wofz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))

import element_observable_closure as C                           # noqa: E402
import calibrated_background_thresholds as cbt                            # noqa: E402
from saimon.atmosphere import us_standard_atmosphere, rayleigh_cross_section_m2   # noqa: E402
from saimon.geometry import tangent_to_slant_paths                         # noqa: E402
from saimon import trace_gases as tg                                        # noqa: E402
from saimon.onion_peel import M1_PER_OD, OD_PER_M1, R_EARTH_M, DZ_M, G_ONION   # noqa: E402

OUT = ROOT / "outputs"
UM = json.load(open(OUT / "uncertainty_model.json"))
thr = json.load(open(ROOT / "outputs/calibrated_background_thresholds.json"))
band_el = np.round(np.array(thr["band01_elements_um"], float), 2)
n_mir = len(band_el)
win = np.array(UM["window_index"]); outer = np.array(UM["outer_index"])
s_meas_od = np.array(UM["s_measured_od"]); s_adopt_od = np.array(UM["s_adopted_od"])
y_mir_20 = np.array(UM["y_gas_elements"])
ELL = float(UM["correlation"]["gaussian_l_um"])
S_CAP = 2000.0
AER_FLAT = 1e6
SIGMA_ETA = C.SIGMA_ETA_BASE
ATLAS = ROOT / "data/ace_floor/w0p1/atlas.json"
N_STATE = C.N_STATE
I_SIL, I_AER, I_GAS, I_T, I_CFC, I_GRP, I_SHIFT, I_SOFT = C.I_SIL, C.I_AER, C.I_GAS, C.I_T, C.I_CFC, C.I_GRP, C.I_SHIFT, C.I_SOFT


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- fixed pieces
D_EL = np.abs(band_el[:, None] - band_el[None, :])
R_GAUSS = np.exp(-(D_EL / ELL) ** 2)
_cfc_x = np.interp(C.nu, C.xnu, C.xs, left=0.0, right=0.0) * 1e-4          # m^2 per molecule on the fine grid
_out_masks = [(C.lam_outer >= c - C.EL_HALF_UM) & (C.lam_outer < c + C.EL_HALF_UM + 1e-9) for c in band_el[outer]]
_coarse_masks = [(C.lam_coarse >= c - C.EL_HALF_UM) & (C.lam_coarse < c + C.EL_HALF_UM) for c in band_el]
_z_ray = np.arange(0.0, 80_001.0, 500.0)
_n_ray = np.asarray(us_standard_atmosphere(_z_ray).number_density_m3)
_alpha_ray = _n_ray[:, None] * rayleigh_cross_section_m2(C.WL_TRIPLET_NM * 1e-9)[None, :]
_tau_ray20 = (tangent_to_slant_paths(np.array([20_000.0]), _z_ray) @ _alpha_ray)[0]


def atlas_altitude_ratio():
    """measured ACE scatter at the 8.80-um element in the 16-19 / 19-22 / 22-25-km bins, relative to 19-22."""
    atlas = json.load(open(ATLAS))

    def sd(b):
        rows = [r for r in atlas[b] if r.get("sd") is not None and abs(r["center_um"] - 8.8) < 1e-6]
        return rows[0]["sd"]
    r = np.array([sd("16_19"), sd("19_22"), sd("22_25")]) / sd("19_22")
    return dict(zip(["16_19", "19_22", "22_25"], r.tolist()))


ATLAS_RATIO = atlas_altitude_ratio()


def disc_scale(z_km, mode="const"):
    if mode == "const":
        return 1.0
    if mode == "atlas":
        return ATLAS_RATIO["16_19"] if z_km < 19.0 else (ATLAS_RATIO["19_22"] if z_km < 22.0 else ATLAS_RATIO["22_25"])
    raise ValueError(mode)


def vis_floor_inflation(h_m):
    tau = (tangent_to_slant_paths(np.array([h_m]), _z_ray) @ _alpha_ray)[0]
    return np.exp(0.5 * (tau - _tau_ray20))


def iz_of(h_m):
    return int(round(h_m / (cbt.ALT[1] - cbt.ALT[0])))


# ---------------------------------------------------------------- per-line profiles at a tangent
def line_profiles_at(gas, T, P, col):
    """sparse (n_lines, n_nu) slant OD per line at (T_eff, P_eff) and the column col [m^-2] (round-70 code)."""
    nu_l, sw, ga, na, el, de, code, ref = C.load_lines(gas)
    mol, iso, M = C.MOL[gas]
    Q = C.hapi.partitionSum(mol, iso, T) / C.hapi.partitionSum(mol, iso, 296.0)
    S_T = (sw / Q * np.exp(-C.C2 * el / T) / np.exp(-C.C2 * el / 296.0)
           * (1 - np.exp(-C.C2 * nu_l / T)) / (1 - np.exp(-C.C2 * nu_l / 296.0)))
    gam_L = ga * P * (296.0 / T) ** na
    alpha_D = 3.5812e-7 * nu_l * np.sqrt(T / M)
    nu0 = nu_l + de * P
    STEP = C.STEP
    off = np.arange(-C.LINE_WIN_CM, C.LINE_WIN_CM + STEP / 2, STEP); n_off = len(off); ioff = np.round(off / STEP).astype(int)
    rows, cols_, vals = [], [], []
    for s in range(0, len(nu_l), 1000):
        sl = slice(s, s + 1000)
        c0 = np.round((nu0[sl] - C.NU_LO) / STEP).astype(int)
        idx = c0[:, None] + ioff[None, :]
        ok = (idx >= 0) & (idx < C.n_nu)
        dnu_loc = (C.NU_LO + idx * STEP) - nu0[sl][:, None]
        sD = alpha_D[sl][:, None] / np.sqrt(2 * np.log(2))
        z = (dnu_loc + 1j * gam_L[sl][:, None]) / (sD * np.sqrt(2))
        V = np.real(wofz(z)) / (sD * np.sqrt(2 * np.pi))
        tau_l = (S_T[sl][:, None] * V) * col * 1e-4
        r = np.repeat(np.arange(s, s + len(c0)), n_off).reshape(-1, n_off)
        rows.append(r[ok]); cols_.append(idx[ok]); vals.append(tau_l[ok].astype(np.float32))
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols_))), shape=(len(nu_l), C.n_nu)), ref


# ---------------------------------------------------------------- the tangent model
class TangentModel:
    """rows, observable, Jacobian and covariance pieces of the adopted model at one tangent height."""

    def __init__(self, h_m, g_bg=1.0, o3_scale=1.0, gas_scales=None, template=None, template_outer=None, template_trip=None,
                 bg_profiles=None, with_lines=True, line_cache=None):
        self.h = float(h_m); self.z_km = self.h / 1e3
        self.chord = tangent_to_slant_paths(np.array([self.h]), C.alt)[0]
        gs = {g: 1.0 for g in C.GASES}
        if gas_scales:
            gs.update(gas_scales)
        gs["o3"] *= o3_scale
        self.gas_scales = gs
        # gases: node optical depths at this chord (exact), T_eff / P_eff / column of the chord
        self.tau_nodes, self.T_eff, self.P_eff, self.col = {}, {}, {}, {}
        for g in C.GASES:
            w = self.chord * C.dens[g] * gs[g]
            self.tau_nodes[g] = np.einsum("z,zn,zj->jn", w, C.fine[g]["sig_z"], C.W)
            self.T_eff[g] = float((w * C.atm.temperature_k).sum() / w.sum())
            self.P_eff[g] = float((w * C.atm.pressure_pa).sum() / w.sum()) * C.ATM_PER_PA
            self.col[g] = float(w.sum())
        self.tau_cfc = _cfc_x * float((self.chord * C.dens["cfc12"]).sum())
        # aerosol: per-shell profiles (7, nz, n_wl) -> slant OD at this chord; template = silica (or material) column
        prof_c, prof_o, prof_t = (C.prof_c, C.prof_o, C.prof_t) if bg_profiles is None else bg_profiles
        ch121 = tangent_to_slant_paths(np.array([self.h]), cbt.ALT)[0]
        self.iz = iz_of(self.h)
        self.g_bg = float(g_bg)
        scale = np.array([1.0, g_bg, g_bg, g_bg, g_bg, g_bg, g_bg])
        sl_c = np.einsum("z,czw->cw", ch121, prof_c) * scale[:, None]
        sl_o = np.einsum("z,czw->cw", ch121, prof_o) * scale[:, None]
        self.shell_c = prof_c[:, self.iz, :] * scale[:, None]; self.shell_o = prof_o[:, self.iz, :] * scale[:, None]
        self.shell_t = prof_t[:, self.iz, :] * scale[:, None]
        if template is not None:                      # (nz, n_coarse) per-shell extinction per Tg of another template
            sl_c[0] = ch121 @ template; self.shell_c[0] = template[self.iz]
            sl_o[0] = ch121 @ template_outer; self.shell_o[0] = template_outer[self.iz]
            self.shell_t[0] = template_trip[self.iz]
        self.tau_aer_fine = np.array([np.interp(C.lam_um, C.lam_coarse, s) for s in sl_c])     # (7, n_nu)
        self.sl_o = sl_o
        # reference state and observable (the source-group scalings are zero at the reference state, so tau0 does
        # not need the group columns; those are built from the per-line profiles at this tangent below)
        x = C.x_ref.copy()
        self.x_ref = x
        self.tau0 = self.tau_total(x, groups=False)
        self.y0 = C.element_obs(self.tau0)
        self.with_lines = with_lines
        if with_lines:
            wT0 = C.PROF0 * (C.wS * np.exp(-np.maximum(self.tau0, -60.0)))[None, :]
            Wmat = (wT0 / wT0.sum(axis=1)[:, None]).T
            self.A_line, tg_, self.Pm = {}, np.empty_like(C.tau_group), {}
            for g in C.GASES:
                if line_cache is not None and g in line_cache:
                    Pm = line_cache[g]
                else:
                    Pm, _ = line_profiles_at(g, self.T_eff[g], self.P_eff[g], self.col[g])
                self.Pm[g] = Pm
                self.A_line[g] = np.asarray(Pm @ Wmat)
                for k, (gg, r, m, s_) in enumerate(C.GROUPS):
                    if gg == g:
                        tg_[k] = np.asarray(Pm[m].sum(axis=0)).ravel()      # exact group columns at this tangent
            self._tau_group = tg_
            self.C_line_win = sum(SIGMA_ETA ** 2 * (A.T @ A) for A in self.A_line.values())
        self.K0 = self.jacobian(x)
        # outer rows (aerosol columns only, slant convention) and their gas OD for the radiometric term
        K_out = np.zeros((len(outer), N_STATE))
        for n, m in enumerate(_out_masks):
            K_out[n, I_AER] = sl_o[:, m].mean(axis=1)
        self.K_mir = np.zeros((n_mir, N_STATE)); self.K_mir[win] = self.K0; self.K_mir[outer] = K_out
        ratio_o3 = self.col["o3"] / (C.col["o3"] * 1.0)
        self.y_mir = np.zeros(n_mir); self.y_mir[win] = self.y0; self.y_mir[outer] = y_mir_20[outer] * ratio_o3
        # triplet rows at the shell of h
        self.K_tri = np.zeros((3, N_STATE)); self.K_tri[:, I_AER] = self.shell_t.T
        s_tot = self.shell_t[1] + self.shell_t[2]
        self.S_tri = np.diag(np.maximum(C.PHI_TRIPLET * s_tot, cbt.ALPHA_FLOOR * vis_floor_inflation(self.h)) ** 2)
        # slant-to-shell factor of the template at the 8.80-um element
        t_shell_880 = float(np.mean(self.shell_c[0][_coarse_masks[C.i880]]))
        self.rho = float(self.K_mir[C.i880, I_SIL] / (t_shell_880 * OD_PER_M1)) if t_shell_880 > 0 else float("nan")

    # --- forward model at this tangent
    def tau_total(self, x, groups=True):
        t = np.zeros(C.n_nu)
        for g in C.GASES:
            t += (x[I_GAS[g]] @ self.tau_nodes[g]) * (1.0 + x[I_T] * C.a_T[g])
        if groups:
            t = t + x[I_GRP] @ self.tau_group()
        return t + x[I_CFC] * self.tau_cfc + x[I_AER] @ self.tau_aer_fine

    def tau_group(self):
        if not hasattr(self, "_tau_group"):
            # without the per-line build: the round-70 group profiles scaled by this chord's column ratio per gas
            tg_ = np.empty_like(C.tau_group)
            for k, (g, r, m, s) in enumerate(C.GROUPS):
                tg_[k] = C.tau_group[k] * (self.col[g] / C.col[g])
            self._tau_group = tg_
        return self._tau_group

    def jacobian(self, x):
        tau = self.tau_total(x)
        shift, soft = float(x[I_SHIFT]), float(x[I_SOFT])
        dt = np.empty((I_SHIFT, C.n_nu)); dt[I_T] = 0.0
        for g in C.GASES:
            tg_ = x[I_GAS[g]] @ self.tau_nodes[g]
            dt[I_GAS[g]] = self.tau_nodes[g] * (1.0 + x[I_T] * C.a_T[g]); dt[I_T] += C.a_T[g] * tg_
        dt[I_CFC] = self.tau_cfc; dt[I_GRP] = self.tau_group(); dt[I_AER] = self.tau_aer_fine
        K = np.empty((C.n_el, N_STATE))
        prof = C.PROF0 if (shift == 0.0 and soft == C.RESP_SOFT0) else C.response_profiles(shift, soft)
        K[:, :I_SHIFT] = C.element_mean(tau, dt, prof).T
        d = 0.001
        K[:, I_SHIFT] = (C.element_obs(tau, shift + d, soft) - C.element_obs(tau, shift - d, soft)) / (2 * d)
        K[:, I_SOFT] = (C.element_obs(tau, shift, soft + d) - C.element_obs(tau, shift, max(soft - d, 1e-4))) / (soft + d - max(soft - d, 1e-4))
        return K

    # --- covariance of the adopted model on the 47 MIR rows
    def cov_mir(self, disc_mode="const", disc_level="measured", disc_mult=1.0, snr=S_CAP, kind="gauss", ell=None, line=True):
        S = np.diag((np.exp(self.y_mir / 2.0) / snr) ** 2)
        if line and self.with_lines:
            S[np.ix_(win, win)] += self.C_line_win
        s = {"measured": s_meas_od, "adopted": s_adopt_od, "none": np.zeros(n_mir)}[disc_level] * disc_scale(self.z_km, disc_mode) * disc_mult
        if kind == "gauss":
            R = R_GAUSS if ell is None else np.exp(-(D_EL / ell) ** 2)
        elif kind == "indep":
            R = np.eye(n_mir)
        elif kind == "exp":
            R = np.exp(-D_EL / ell)
        elif kind == "uniform":       # uniform level: every element at the 8.80-um value
            s = np.full(n_mir, s[C.i880]); R = R_GAUSS
        else:
            raise ValueError(kind)
        return S + np.outer(s, s) * R


# ---------------------------------------------------------------- priors and the joint solve
def prior_inv(aer=AER_FLAT, extra=0):
    Sa = C.S_a.copy()
    for k in range(I_AER.start, I_AER.stop):
        Sa[k, k] = aer ** 2
    if extra:
        Z = np.zeros((N_STATE + extra,) * 2); Z[:N_STATE, :N_STATE] = Sa
        for k in range(extra):
            Z[N_STATE + k, N_STATE + k] = AER_FLAT ** 2
        Sa = Z
    return np.linalg.inv(Sa)


S_A_INV = prior_inv()


def joint_sigma(tm: TangentModel, S_mir=None, rows="band", triplet=True, S_a_inv=None, extra_cols=None, extra_rows=None,
                vis_scale=1.0, return_all=False, target=I_SIL):
    """marginal sigma of the target amplitude (slant Tg) of the joint fit at this tangent.
    extra_cols: (n_mir_cols (47, k), trip_cols (3, k)) additional nuisance columns with flat priors;
    extra_rows: (K_e (m, N_STATE + k), S_e (m, m)) additional per-shell rows (e.g. far-infrared elements)."""
    S_mir = tm.cov_mir() if S_mir is None else S_mir
    idx = np.arange(n_mir) if rows == "band" else win
    K = tm.K_mir[idx]; S = S_mir[np.ix_(idx, idx)]
    Kt = tm.K_tri; St = tm.S_tri * vis_scale ** 2
    k_extra = 0
    if extra_cols is not None:
        cm, ct = extra_cols; k_extra = cm.shape[1]
        K = np.hstack([K, cm[idx]]); Kt = np.hstack([Kt, ct])
    if S_a_inv is None:
        S_a_inv = S_A_INV if k_extra == 0 else prior_inv(extra=k_extra)
    if triplet:
        K = np.vstack([K, Kt]); S = C._blockdiag(S, St)
    if extra_rows is not None:
        Ke, Se = extra_rows
        if Ke.shape[1] < K.shape[1]:
            Ke = np.hstack([Ke, np.zeros((Ke.shape[0], K.shape[1] - Ke.shape[1]))])
        K = np.vstack([K, Ke]); S = C._blockdiag(S, Se)
    Sinv = np.linalg.inv(S)
    S_hat = np.linalg.inv(K.T @ Sinv @ K + S_a_inv)
    s = float(np.sqrt(S_hat[target, target]))
    return (s, S_hat, S_hat @ K.T @ Sinv, K, S) if return_all else s


def threshold(tm, **kw):
    """3 sigma per 0.5-km shell at the tangent (slant amplitude x rho(h))."""
    return 3.0 * joint_sigma(tm, **kw) * tm.rho


def two_stage_sigma(tm, floors_od, rows="band", triplet=True, vis_scale=1.0):
    """the diagonal-floor scenario at this tangent (per-element OD floors -> white extinction -> Fisher marginalisation)."""
    idx = np.arange(n_mir) if rows == "band" else win
    shell_rows = np.zeros((n_mir, 7))
    for n in range(n_mir):
        if band_el[n] < 9.25:
            shell_rows[n] = tm.shell_c[:, _coarse_masks[n]].mean(axis=1)
        else:
            shell_rows[n] = tm.shell_o[:, _out_masks[list(outer).index(n)]].mean(axis=1)
    t_all = np.r_[tm.shell_t[0], shell_rows[idx, 0]]
    nuis = [np.r_[tm.shell_t[j], shell_rows[idx, j]] for j in range(1, 7)]
    sig_all = np.r_[np.sqrt(np.diag(tm.S_tri)) * vis_scale, floors_od[idx] * M1_PER_OD]
    ch = np.arange(len(t_all)) if triplet else np.arange(3, len(t_all))
    sa, _, R2 = cbt.marginal(ch, nuis, sig_all, t_all)
    return float(sa)


# ---------------------------------------------------------------- per-shell templates and the full background profile
def edge_path(z_m, dz=DZ_M):
    """exact path [m] of the ray with tangent z through the edge-bounded shell [z, z + dz]."""
    return 2.0 * np.sqrt((R_EARTH_M + z_m + dz) ** 2 - (R_EARTH_M + z_m) ** 2)


def shell_template(z_m, prof=None):
    """the 20-km reference silica spectrum (per-shell extinction per Tg of the homogeneous layer at 20 km) placed in the tangent
    shell alone, in the paper's per-shell convention: the slant rows see it through the edge-bounded one-shell path divided by the
    white onion-peel gain (the conversion of Appendix B used by the two-stage procedure and by rho), the triplet rows per shell.
    On the level-centred 121-level grid the chord of the tangent level covers half a layer, so the level value is scaled to give
    exactly that slant optical depth.  The amplitude is then the per-shell mass of Sect. 2 directly (no slant-to-shell factor)."""
    prof_c, prof_o, prof_t = (C.prof_c, C.prof_o, C.prof_t) if prof is None else prof
    iz20 = iz_of(20_000.0); iz = iz_of(z_m)
    ch = tangent_to_slant_paths(np.array([float(z_m)]), cbt.ALT)[0]
    scale = (edge_path(z_m) / G_ONION) / ch[iz]
    tc = np.zeros_like(prof_c[0]); to = np.zeros_like(prof_o[0]); tt = np.zeros_like(prof_t[0])
    tc[iz] = prof_c[0][iz20] * scale; to[iz] = prof_o[0][iz20] * scale; tt[iz] = prof_t[0][iz20]
    return dict(template=tc, template_outer=to, template_trip=tt)


BG_FULL = OUT / "adopted_engine_bg_full.npz"


def bg_profiles_full():
    """the seven per-shell aerosol profiles with the sulfate background on the full GloSSAC anchor profile (extended below 16 km
    by the quiet-epoch median, as the 2-D maps do; the round-70 profile is zero below 16 km, which a tangent above 16 km never
    sees).  Built once (about 2 min of Mie integrals) and cached."""
    if BG_FULL.exists():
        d = np.load(BG_FULL)
        return d["prof_c"], d["prof_o"], d["prof_t"]
    from detectability_2d_calibrated import glossac_quiet_median, anchor_profile_full
    lat_g, alt_g, med = glossac_quiet_median()
    alpha525 = anchor_profile_full(lat_g, alt_g, med)
    ri, wt, Tm = cbt.member_ri("LM65T223")
    bg = cbt.TwoComponentBackground(alpha525, cbt.CAL_QUIET, ri, wt, Tm)

    def cols(wl_m):
        def prof(layer):
            return layer.extinction_profile_m1(wl_m)

        def jac(i):
            r0, s0, _ = cbt.CAL_QUIET[i]
            dr, ds = 0.05 * r0, 0.05 * s0
            return ((prof(bg.column(i, dr=+dr)) - prof(bg.column(i, dr=-dr))) / (2 * dr),
                    (prof(bg.column(i, ds=+ds)) - prof(bg.column(i, ds=-ds))) / (2 * ds))
        P = [prof(bg.column(0)), prof(bg.column(1))]
        jrf, jsf = jac(0); jrc, jsc = jac(1)
        return np.array(P + [jrf, jsf, jrc, jsc])
    log("building the full-profile background (7 columns x 3 wavelength sets) ...")
    pc = np.vstack([C.prof_c[:1], cols(C.lam_coarse * 1e-6)]); po = np.vstack([C.prof_o[:1], cols(C.lam_outer * 1e-6)]); pt = np.vstack([C.prof_t[:1], cols(C.WL_TRIPLET_NM * 1e-9)])
    np.savez_compressed(BG_FULL, prof_c=pc, prof_o=po, prof_t=pt)
    return pc, po, pt


# ---------------------------------------------------------------- the 20-km reference
REF = TangentModel(20_000.0)
if __name__ == "__main__":
    b = threshold(REF); w = threshold(REF, rows="window")
    log(f"adopted model at 20 km: band {b:.4f} Tg, window {w:.4f} Tg (round 74: {UM['models']['B_primary']['band_triplet']:.4f} / "
        f"{UM['models']['B_primary']['window_triplet']:.4f}); rho {REF.rho:.3f} (round 70 {C.RHO_SLANT_TO_SHELL:.3f}); "
        f"y(8.80) {REF.y0[C.i880]:.4f}; atlas ratio {ATLAS_RATIO}")
    ts = TangentModel(20_000.0, **shell_template(20_000.0))
    log(f"shell template at 20 km: 3 sigma {3*joint_sigma(ts):.4f} Tg per shell (prescribed x rho {b:.4f}); conv check {ts.K_mir[C.i880, I_SIL] / (OD_PER_M1 * float(np.mean(C.shell_c[0][_coarse_masks[C.i880]]))):.4f}")
    for z in (16.0, 18.0, 20.0, 22.0, 24.0):
        tm = TangentModel(z * 1e3)
        log(f"  {z:.0f} km: band {threshold(tm):.4f} (atlas-scaled discrepancy {threshold(tm, S_mir=tm.cov_mir('atlas')):.4f}), "
            f"window {threshold(tm, rows='window'):.4f}; rho {tm.rho:.3f}; y880 {tm.y0[C.i880]:.3f}; T_eff O3 {tm.T_eff['o3']:.1f} K")
