"""element_observable_closure.py -- referee round 5, items 1-3 (and the
synthetic-injection part of item 5): the trace-gas budget, the joint state and
covariance, and a closure test, all formulated on the ELEMENT OBSERVABLE of the
declared boxcar response,

    y_i = -ln [ int w_i(lambda) S(lambda) T(lambda) dlambda
                / int w_i(lambda) S(lambda) dlambda ],

with T(lambda) the line-resolved transmitted fraction at the tangent height, S
the exo-atmospheric solar spectral irradiance (5772-K disk; featureless at this
resolution) and w_i the 0.1-um boxcar of element i (uniform in wavelength; the
wavenumber integration carries the dlambda/dnu weight).  The earlier budget
(saimon.o3_removal; outputs/round43_tracegas/gas_removal_floor_widths.csv)
averaged the cross sections over the element before applying Beer's law, i.e.
it modelled <tau>_i, which is not the measured quantity when lines within the
element are optically thick.  Here every Jacobian is the exact derivative of
y_i,  d y_i / d p = < d tau / d p >_i  with <g>_i = int w S T g / int w S T,
the transmission-weighted element mean, so line saturation enters the measured
quantity exactly.

Grid: the cached native Voigt grids of O3 and N2O at 0.0025 cm^-1 (the lines
are ~0.003 cm^-1 HWHM at 20 km; a 0.001-cm^-1 check at the 8.80-um element
gives -ln<T> = 0.384 against 0.402 on the 0.01-cm^-1 budget grid and <tau> =
1.16 on both), at the five altitude nodes of the budget (hat basis, as
saimon.o3_removal) and at (T_eff, T_eff +- 5 K, P_eff) for the temperature
derivative; CFC-12 from the HITRAN .xsc file.

What is computed (20-km tangent, 7.8-9.3 um window = 15 elements of 0.1 um
centred 7.80 ... 9.20 um; US-standard atmosphere, AFGL profiles):
 1. the observable against the mean optical depth, element by element;
 2. the per-line error model from the actual HITRAN2020 uncertainty codes
    (ierr intensity digit) and source references (iref) of every O3 and N2O
    line in the window: a coherent scaling per (gas, source group) enters
    the STATE with the code midpoint as prior (for the one dominant group it is
    degenerate with the retrieved amount and cancels in the removal, as the
    band scaling of the earlier budget did); an independent per-line scatter
    sigma_eta, which the codes do not fix, enters the measurement covariance
    through the transmission-weighted element response a_li of every line,
    C_line = sigma_eta^2 sum_l a_l a_l^T; sigma_eta is scanned;
 3. the joint optimal-estimation state x = [O3 nodes (5), N2O nodes (5),
    T_eff, CFC-12 amplitude, source-group scalings (7), silica amplitude (Tg,
    slant convention), six sulfate parameters], its prior, the measurement
    covariance (photon + C_line on the window elements; the visible/NIR
    triplet as per-shell extinction rows with the Level-2 fractional errors
    and no gas columns), the marginal posterior of the silica amplitude; the
    equivalence test against the paper's two-stage procedure; the
    decomposition of the gas-removal error at 8.80 um by its propagation; the
    SNR scan on non-overlapping elements; the band configuration (the 32
    elements beyond the window carry the paper's empirical floors);
 4. the vertical conversion of each part and the corrected crossover (Eq. D7);
 5. the ACE atlas test: the tropical 16-20-km bin (18-km tangent) formed into
    the same element observable from the published 0.0025-cm^-1 transmittance,
    the O3 amplitude fitted on the 9.2-um element and the 8.80-um element
    predicted, in the -ln<T> convention (the earlier comparison matched a
    mean-cross-section model to the measured transmittance, which is why it
    needed 0.37 of the climatology);
 6. a synthetic-truth closure test: independently drawn gas profiles, T
    offset, CFC amplitude, sulfate state, line errors (group + independent),
    photon noise, injected silica (0 ... 0.3 Tg per shell) and response
    perturbations (wavelength shift and softened edges unknown to the
    retrieval), generated at native resolution through the nonlinear element
    forward model and retrieved with an iterated (Gauss-Newton)
    optimal-estimation fit; bias, scatter, coverage and detection probability.

Run from the repo root:
  python reproduce/element_observable_closure.py [--ndraw 100]
Writes outputs/element_observable_closure.json (+ .npz) and figures/scisat_ace_atlas_874.png (Fig. 3).
No HITRAN fetch; hapi (hitran-api) is imported only for the TIPS partition sums. Inputs shipped with the
repository: the 0.0025-cm^-1 grids data/trace_gases/xsec_cache/*_1067_1294_*_s0.0025.npz, the carved line lists
data/trace_gases/hitran_cache/cont_{o3,n2o,ch4,h2o}_1042_1319.data, the atlas extract data/ace_atlas/ (950-1300 cm^-1),
outputs/calibrated_background_thresholds.json and outputs/round43_tracegas/gas_removal_floor_widths.csv.
The shipped archive is the --ndraw 100 run of the working repository (round 70, ~45 min); --ndraw 5 reproduces every
deterministic block in ~15 min. The --coarse check mode (0.01-cm^-1 budget grids over 575-1782 cm^-1) needs grids that
are only partly shipped (fetch/build_hitran_cache.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import wofz, erf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))

from saimon.atmosphere import us_standard_atmosphere                      # noqa: E402
from saimon.geometry import tangent_to_slant_paths                         # noqa: E402
from saimon import trace_gases as tg                                        # noqa: E402
from saimon.o3_removal import DEFAULT_NODES_KM, _hat_basis, gaussian_prior_covariance  # noqa: E402
from saimon.onion_peel import M1_PER_OD, OD_PER_M1, G_ONION, R_EARTH_M, DZ_M, BUDGET_OD_880  # noqa: E402
from saimon.sai import create_silica_sai_layer                              # noqa: E402
from saimon.materials import SilicaRefractiveIndex                          # noqa: E402
import calibrated_background_thresholds as cbt                             # noqa: E402

OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
ARCH = OUT / "element_observable_closure.json"
SPEC = OUT / "element_observable_closure.npz"

# ---------------------------------------------------------------- constants
H_TAN_M = 20_000.0
# grid: the 0.0025-cm^-1 grids cached by fetch/build_finegrid_window.py over 1067-1294 cm^-1,
# or (--coarse) the 0.01-cm^-1 budget grids over 575-1782 cm^-1 (the lines are undersampled there by ~3x;
# -ln<T> at 8.80 um is 0.402 against 0.384 on a 0.001-cm^-1 check)
_COARSE = "--coarse" in sys.argv
STEP = 0.01 if _COARSE else 0.0025
CACHE_LO, CACHE_HI = (575.3, 1782.1) if _COARSE else (1067.0, 1294.0)
NU_LO, NU_HI = 1069.0, 1292.0            # working range (7.74-9.35 um)
EL_CENTRES_UM = np.round(np.arange(7.80, 9.2001, 0.1), 2)   # the 15 window elements
EL_HALF_UM = 0.05
SNR_REF = 2000.0                          # per-sample transmission SNR at the exo-atmospheric reference
ATM_PER_PA = 1.0 / 101_325.0
C2 = 1.4387769                            # cm K
T_SUN = 5772.0
HC_K = 1.438776877e-2                     # m K
# HITRAN intensity uncertainty codes (hitran.org/docs/uncertainties): range midpoints
IERR_INT = {"3": 0.25, "4": 0.15, "5": 0.075, "6": 0.035, "7": 0.015, "8": 0.0055, "9": 0.0005}
GAS_PRIOR = {"o3": (0.30, 6.0), "n2o": (0.05, 6.0), "ch4": (0.10, 6.0), "h2o": (0.30, 6.0)}   # (relative sigma per node, correlation length km)
T_PRIOR_K = 2.0
CFC_PRIOR = 0.03
AER_PRIOR = 1.0e3
SIGMA_ETA_SCAN = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075)
SIGMA_ETA_BASE = 0.02                      # the paper's working assumption (1-3 % per line)
WL_TRIPLET_NM = np.array([448.0, 756.0, 1544.0])
PHI_TRIPLET = np.array([0.040, 0.032, 0.088])
LINE_WIN_CM = 0.6                         # half-window of the per-line profiles (200 HWHM)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- geometry
alt = np.arange(14_000.0, 80_001.0, 500.0)
atm = us_standard_atmosphere(alt)
chord = tangent_to_slant_paths(np.array([H_TAN_M]), alt)[0]
dens = tg.number_density_profiles(alt, atm.number_density_m3)
nodes_m = np.asarray(DEFAULT_NODES_KM, float) * 1e3
node_atm = us_standard_atmosphere(nodes_m)
W = _hat_basis(nodes_m, alt)                                  # (n_z, n_nodes)

nu = np.arange(NU_LO, NU_HI + STEP / 2, STEP)
lam_um = 1e4 / nu
n_nu = len(nu)
dlam_dnu = 1e4 / nu**2                                        # um per cm^-1


def planck_lambda(lam_um_arr, T=T_SUN):
    lam = lam_um_arr * 1e-6
    return 1.0 / (lam**5 * np.expm1(HC_K / (lam * T)))


S_sun = planck_lambda(lam_um)
n_el = len(EL_CENTRES_UM)
el_id = np.full(n_nu, -1, int)
for i, c in enumerate(EL_CENTRES_UM):
    el_id[(lam_um >= c - EL_HALF_UM) & (lam_um < c + EL_HALF_UM)] = i
in_el = el_id >= 0
wS = dlam_dnu * S_sun                                         # S(lambda) dlambda/dnu
# the declared response: a 0.1-um boxcar whose edges are softened by a Gaussian of RESP_SOFT0 (5 % of the
# width; a pixel or slit response is never a step).  A sharp step would make the element observable
# discontinuous in the wavelength registration, because each edge crosses individual saturated lines.
RESP_SOFT0 = 0.005


def response_profiles(shift=0.0, soft=RESP_SOFT0):
    sc = max(abs(soft), 1e-4) * np.sqrt(2)
    lo = EL_CENTRES_UM[:, None] + shift - EL_HALF_UM; hi = EL_CENTRES_UM[:, None] + shift + EL_HALF_UM
    return 0.5 * (erf((lam_um[None, :] - lo) / sc) - erf((lam_um[None, :] - hi) / sc))   # (n_el, n_nu)


PROF0 = response_profiles()
DEN0 = PROF0 @ wS


def cached_grid(gas, T, P_atm):
    g = tg.HITRAN_GASES[gas]
    key = f"{g.name}_{int(round(CACHE_LO))}_{int(round(CACHE_HI))}_T{T:.1f}_P{P_atm:.6f}_s{STEP:g}"
    path = os.path.join(tg.XSEC_CACHE, key + ".npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing cached grid {key}: the shipped data/trace_gases/xsec_cache holds the 0.0025-cm^-1 grids of the default mode (regenerate with fetch/build_finegrid_window.py); the --coarse check mode needs the full 0.01-cm^-1 budget grids over 575-1782 cm^-1, of which only part are shipped (fetch/build_hitran_cache.py)")
    d = np.load(path)
    return np.interp(nu, d["nu"], d["coef"]) * 1e-4           # m^2


def gas_fine(gas):
    sig_nodes = np.array([cached_grid(gas, float(T), float(P) * ATM_PER_PA)
                          for T, P in zip(node_atm.temperature_k, node_atm.pressure_pa)])
    idx = np.clip(np.searchsorted(nodes_m, alt, side="right") - 1, 0, len(nodes_m) - 2)
    f = np.clip((alt - nodes_m[idx]) / (nodes_m[idx + 1] - nodes_m[idx]), 0.0, 1.0)
    sig_z = (1 - f)[:, None] * sig_nodes[idx] + f[:, None] * sig_nodes[idx + 1]
    weight = chord * dens[gas]
    tau_nodes = np.einsum("z,zn,zj->jn", weight, sig_z, W)
    T_eff = float((weight * atm.temperature_k).sum() / weight.sum())
    P_eff = float((weight * atm.pressure_pa).sum() / weight.sum()) * ATM_PER_PA
    s0 = cached_grid(gas, T_eff, P_eff); sp = cached_grid(gas, T_eff + 5.0, P_eff); sm = cached_grid(gas, T_eff - 5.0, P_eff)
    a_T = np.where(s0 > 0, (sp - sm) / (10.0 * np.maximum(s0, 1e-300)), 0.0)
    return dict(sig_z=sig_z, tau_nodes=tau_nodes, tau_ref=tau_nodes.sum(axis=0), T_eff=T_eff, P_eff=P_eff,
                a_T=a_T, col=float(weight.sum()))


log("line-resolved forward model (0.0025 cm^-1 cached grids) ...")
t0 = time.time()
GASES = ("o3", "n2o", "ch4", "h2o")
fine = {g: gas_fine(g) for g in GASES}
tau_nodes = {g: fine[g]["tau_nodes"] for g in GASES}
tau_ref = {g: fine[g]["tau_ref"] for g in GASES}
a_T = {g: fine[g]["a_T"] for g in GASES}
T_eff = {g: fine[g]["T_eff"] for g in GASES}; P_eff = {g: fine[g]["P_eff"] for g in GASES}
col = {g: fine[g]["col"] for g in GASES}
xnu, xs = tg._parse_xsc(tg._find_cfc_xsc("cfc12"))
tau_cfc = np.interp(nu, xnu, xs, left=0.0, right=0.0) * 1e-4 * float((chord * dens["cfc12"]).sum())
i880 = int(np.argmin(np.abs(EL_CENTRES_UM - 8.80)))
log(f"  {time.time()-t0:.0f} s; <tau> at the 8.80-um element: " + ", ".join(f"{g} {tau_ref[g][el_id==i880].mean():.4f}" for g in GASES)
    + f", CFC-12 {tau_cfc[el_id==i880].mean():.4f}; at 7.80 um: " + ", ".join(f"{g} {tau_ref[g][el_id==0].mean():.3f}" for g in GASES))

# ---------------------------------------------------------------- aerosol columns
ALT121 = cbt.ALT
ri, wt, Tm = cbt.member_ri("LM65T223")
bg = cbt.TwoComponentBackground(cbt.glossac_2025N_quiet_profile(), cbt.CAL_QUIET, ri, wt, Tm)
sil_layer = create_silica_sai_layer(ALT121, 1.0, refractive_index=SilicaRefractiveIndex(
    str(ROOT / "data/optics/SiO2_KitamuraPopova.yml")), rmed_nm=cbt.RMED_SIL_NM, sigma=cbt.SIGMA_SIL)
AER_NAMES = ("sil", "A_f", "A_c", "r_f", "s_f", "r_c", "s_c")
AER_CACHE = OUT / "element_observable_closure_aerosol.npz"


def aerosol_profiles(wl_m):
    """per-shell extinction profiles (7, nz, n_wl) of the seven aerosol columns."""
    def prof(layer):
        return layer.extinction_profile_m1(wl_m)

    def jac(i):
        r0, s0, _ = cbt.CAL_QUIET[i]
        dr, ds = 0.05 * r0, 0.05 * s0
        return ((prof(bg.column(i, dr=+dr)) - prof(bg.column(i, dr=-dr))) / (2 * dr),
                (prof(bg.column(i, ds=+ds)) - prof(bg.column(i, ds=-ds))) / (2 * ds))
    P = [prof(sil_layer), prof(bg.column(0)), prof(bg.column(1))]
    jrf, jsf = jac(0); jrc, jsc = jac(1)
    return np.array(P + [jrf, jsf, jrc, jsc])


lam_coarse = np.arange(7.70, 9.4001, 0.005)
lam_outer = np.arange(9.30, 13.3001, 0.01)
if AER_CACHE.exists():
    d = np.load(AER_CACHE)
    prof_c, prof_o, prof_t = d["prof_c"], d["prof_o"], d["prof_t"]
else:
    log("Mie profiles of the silica layer and the calibrated background (cached afterwards) ...")
    t0 = time.time()
    prof_c = aerosol_profiles(lam_coarse * 1e-6); prof_o = aerosol_profiles(lam_outer * 1e-6)
    prof_t = aerosol_profiles(WL_TRIPLET_NM * 1e-9)
    np.savez_compressed(AER_CACHE, prof_c=prof_c, prof_o=prof_o, prof_t=prof_t)
    log(f"  done in {time.time()-t0:.0f} s")


def slant_at(h_m, prof):
    ch = tangent_to_slant_paths(np.array([h_m]), ALT121)[0]
    return np.einsum("z,czw->cw", ch, prof)


slant_c = slant_at(H_TAN_M, prof_c)                              # (7, n_coarse)
shell_c = prof_c[:, cbt.iz, :]; shell_o = prof_o[:, cbt.iz, :]; shell_trip = prof_t[:, cbt.iz, :]
tau_aer_fine = np.array([np.interp(lam_um, lam_coarse, s) for s in slant_c])   # (7, n_nu)
t_shell_880 = float(np.mean(shell_c[0][(lam_coarse >= 8.75) & (lam_coarse < 8.85)]))
tau_slant_880 = float(tau_aer_fine[0][el_id == i880].mean())
RHO_SLANT_TO_SHELL = tau_slant_880 / (t_shell_880 * OD_PER_M1)
log(f"  silica per Tg: shell extinction {t_shell_880:.3e} m^-1 (paper 6.15e-7), slant OD {tau_slant_880:.4f}; "
    f"rho(20 km) = {RHO_SLANT_TO_SHELL:.3f}")

# ---------------------------------------------------------------- per-line model
import hapi  # noqa: E402  (partition sums only; no db_begin)
MOL = {"o3": (3, 1, 47.998), "n2o": (4, 1, 44.013), "ch4": (6, 1, 16.043), "h2o": (1, 1, 18.015)}


def load_lines(gas):
    fn = ROOT / f"data/trace_gases/hitran_cache/cont_{gas}_1042_1319.data"
    rec = []
    with open(fn) as f:
        for line in f:
            v = float(line[3:15])
            if NU_LO - 1.0 <= v <= NU_HI + 1.0:
                rec.append((v, float(line[15:25]), float(line[35:40]), float(line[55:59]), float(line[45:55]),
                            float(line[59:67]), line[128], line[135:137]))
    nu_l, sw, ga, na, el, de = (np.array([r[k] for r in rec]) for k in range(6))
    code = np.array([r[6] for r in rec]); ref = np.array([r[7] for r in rec])
    return nu_l, sw, ga, na, el, de, code, ref


def line_profiles(gas):
    nu_l, sw, ga, na, el, de, code, ref = load_lines(gas)
    T, P = T_eff[gas], P_eff[gas]
    mol, iso, M = MOL[gas]
    Q = hapi.partitionSum(mol, iso, T) / hapi.partitionSum(mol, iso, 296.0)
    S_T = (sw / Q * np.exp(-C2 * el / T) / np.exp(-C2 * el / 296.0)
           * (1 - np.exp(-C2 * nu_l / T)) / (1 - np.exp(-C2 * nu_l / 296.0)))
    gam_L = ga * P * (296.0 / T) ** na
    alpha_D = 3.5812e-7 * nu_l * np.sqrt(T / M)
    nu0 = nu_l + de * P
    off = np.arange(-LINE_WIN_CM, LINE_WIN_CM + STEP / 2, STEP); n_off = len(off); ioff = np.round(off / STEP).astype(int)
    rows, cols_, vals = [], [], []
    for s in range(0, len(nu_l), 1000):
        sl = slice(s, s + 1000)
        c0 = np.round((nu0[sl] - NU_LO) / STEP).astype(int)
        idx = c0[:, None] + ioff[None, :]
        ok = (idx >= 0) & (idx < n_nu)
        dnu_loc = (NU_LO + idx * STEP) - nu0[sl][:, None]
        sD = alpha_D[sl][:, None] / np.sqrt(2 * np.log(2))
        z = (dnu_loc + 1j * gam_L[sl][:, None]) / (sD * np.sqrt(2))
        V = np.real(wofz(z)) / (sD * np.sqrt(2 * np.pi))
        tau_l = (S_T[sl][:, None] * V) * col[gas] * 1e-4
        r = np.repeat(np.arange(s, s + len(c0)), n_off).reshape(-1, n_off)
        rows.append(r[ok]); cols_.append(idx[ok]); vals.append(tau_l[ok].astype(np.float32))
    Pm = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols_))), shape=(len(nu_l), n_nu))
    return Pm, dict(nu=nu_l, S=S_T, code=code, ref=ref, T=T, P=P)


log("per-line profiles, HITRAN codes and source groups ...")
t0 = time.time()
Pline, meta = {}, {}
for g in GASES:
    Pline[g], meta[g] = line_profiles(g)
GROUPS = []                                                   # (gas, ref, mask, sigma_code)
for g in GASES:
    for r in sorted(np.unique(meta[g]["ref"])):
        m = meta[g]["ref"] == r
        Sg = meta[g]["S"][m]
        s_g = float(sum(IERR_INT.get(c, 0.25) * Sg[k] for k, c in enumerate(meta[g]["code"][m])) / Sg.sum())
        GROUPS.append((g, r, m, s_g))
tau_group = np.array([np.asarray(Pline[g][m].sum(axis=0)).ravel() for g, r, m, s in GROUPS])   # (n_groups, n_nu)
NG = len(GROUPS)
code_share = {}
for g in GASES:
    tot = meta[g]["S"].sum()
    m880 = (meta[g]["nu"] >= 1e4 / 8.85) & (meta[g]["nu"] <= 1e4 / 8.75)
    code_share[g] = dict(
        n_lines=int(len(meta[g]["nu"])),
        intensity_code_share_window={c: float(meta[g]["S"][meta[g]["code"] == c].sum() / tot) for c in np.unique(meta[g]["code"])},
        intensity_code_share_880={c: float(meta[g]["S"][m880 & (meta[g]["code"] == c)].sum() / meta[g]["S"][m880].sum())
                                  for c in np.unique(meta[g]["code"][m880])},
        source_group_share_window={r: float(meta[g]["S"][meta[g]["ref"] == r].sum() / tot) for r in np.unique(meta[g]["ref"])},
        source_group_share_880={r: float(meta[g]["S"][m880 & (meta[g]["ref"] == r)].sum() / meta[g]["S"][m880].sum())
                                for r in np.unique(meta[g]["ref"][m880])},
        n_lines_880=int(m880.sum()), n_eff_880=float(meta[g]["S"][m880].sum() ** 2 / (meta[g]["S"][m880] ** 2).sum()),
        T_eff_K=meta[g]["T"], P_eff_atm=meta[g]["P"], slant_column_m2=col[g])
log(f"  {sum(len(meta[g]['nu']) for g in GASES)} lines, {NG} source groups, {time.time()-t0:.0f} s")

# ---------------------------------------------------------------- state layout
NN = len(nodes_m); NGAS = len(GASES)
names = ([f"{g}_{int(z/1e3)}" for g in GASES for z in nodes_m] + ["T_eff", "cfc12"]
         + [f"grp_{g}_{r}" for g, r, m, s in GROUPS] + list(AER_NAMES) + ["resp_shift_um", "resp_soft_um"])
I_GAS = {g: slice(k * NN, (k + 1) * NN) for k, g in enumerate(GASES)}
I_O3 = I_GAS["o3"]; I_N2O = I_GAS["n2o"]
I_T = NGAS * NN; I_CFC = NGAS * NN + 1
I_GRP = slice(NGAS * NN + 2, NGAS * NN + 2 + NG)
I_SIL = NGAS * NN + 2 + NG; I_AER = slice(I_SIL, I_SIL + 7)
I_SHIFT = I_SIL + 7; I_SOFT = I_SIL + 8
N_STATE = len(names)
RESP_PRIOR_UM = 0.005                      # prior on the wavelength registration (and 0.002 on the edge softness)
x_ref = np.zeros(N_STATE); x_ref[:NGAS * NN] = 1.0; x_ref[I_CFC] = 1.0
x_ref[I_AER] = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]; x_ref[I_SOFT] = RESP_SOFT0


def tau_total(x, dtau_extra=None):
    t = np.zeros(n_nu)
    for g in GASES:
        t += (x[I_GAS[g]] @ tau_nodes[g]) * (1.0 + x[I_T] * a_T[g])
    t = t + x[I_GRP] @ tau_group + x[I_CFC] * tau_cfc + x[I_AER] @ tau_aer_fine
    if dtau_extra is not None:
        t = t + dtau_extra
    return t


def element_obs(tau, shift=0.0, soft=RESP_SOFT0, prof=None):
    """y_i = -ln <T>_i for the declared response (or a shifted / differently softened one)."""
    if prof is None:
        prof = PROF0 if (shift == 0.0 and soft == RESP_SOFT0) else response_profiles(shift, soft)
    den = DEN0 if prof is PROF0 else prof @ wS
    return -np.log((prof @ (wS * np.exp(-np.maximum(tau, -60.0)))) / den)


def resp_obs(tau, shift, soft):
    return element_obs(tau, shift, soft)


def element_mean(tau_state, g, prof=None):
    """< g >_i: transmission-weighted element means at the state tau_state (g: (..., n_nu))."""
    if prof is None:
        prof = PROF0
    wT = prof * (wS * np.exp(-np.maximum(tau_state, -60.0)))[None, :]     # (n_el, n_nu)
    den = wT.sum(axis=1)
    g2 = np.atleast_2d(g)
    out = (g2 @ wT.T) / den
    return out if np.ndim(g) == 2 else out[0]


def jacobian(x):
    """exact Jacobian of the element observable with respect to the spectroscopic and aerosol state
    (transmission-weighted means at the state's own response) and finite differences for the two
    response parameters; returns (K (n_el, N_STATE), tau)."""
    tau = tau_total(x)
    shift, soft = float(x[I_SHIFT]), float(x[I_SOFT])
    dt = np.empty((I_SHIFT, n_nu))
    dt[I_T] = 0.0
    for g in GASES:
        tg_ = x[I_GAS[g]] @ tau_nodes[g]
        dt[I_GAS[g]] = tau_nodes[g] * (1.0 + x[I_T] * a_T[g])
        dt[I_T] += a_T[g] * tg_
    dt[I_CFC] = tau_cfc
    dt[I_GRP] = tau_group
    dt[I_AER] = tau_aer_fine
    K = np.empty((n_el, N_STATE))
    prof = PROF0 if (shift == 0.0 and soft == RESP_SOFT0) else response_profiles(shift, soft)
    K[:, :I_SHIFT] = element_mean(tau, dt, prof).T
    d = 0.001
    K[:, I_SHIFT] = (element_obs(tau, shift + d, soft) - element_obs(tau, shift - d, soft)) / (2 * d)
    K[:, I_SOFT] = (element_obs(tau, shift, soft + d) - element_obs(tau, shift, max(soft - d, 1e-4))) / (soft + d - max(soft - d, 1e-4))
    return K, tau


tau0 = tau_total(x_ref)
y0 = element_obs(tau0)
y_mean = np.array([np.average(tau0[el_id == i], weights=dlam_dnu[el_id == i]) for i in range(n_el)])
K0, _ = jacobian(x_ref)
x_gas = x_ref.copy(); x_gas[I_AER] = 0.0
tau_gas0 = tau_total(x_gas); y_gas0 = element_obs(tau_gas0)
y_gas_mean = np.array([np.average(tau_gas0[el_id == i], weights=dlam_dnu[el_id == i]) for i in range(n_el)])
log(f"8.80-um element: gas -ln<T> = {y_gas0[i880]:.4f} vs <tau> = {y_gas_mean[i880]:.4f}; with aerosol {y0[i880]:.4f}")

# element responses of the lines (transmission-weighted at the reference state)
wT0 = PROF0 * (wS * np.exp(-tau0))[None, :]
Wmat = (wT0 / wT0.sum(axis=1)[:, None]).T                                  # (n_nu, n_el) dense
A_line = {g: np.asarray(Pline[g] @ Wmat) for g in GASES}                   # (n_lines, n_el)
check = {g: float(A_line[g].sum(axis=0)[i880]) for g in GASES}
elem_indep_880 = {g: float(np.sqrt((A_line[g][:, i880] ** 2).sum())) for g in GASES}
log(f"  sum of line responses at 8.80 um: O3 {check['o3']:.4f} (T-weighted model {element_mean(tau0, tau_ref['o3'])[i880]:.4f}); "
    f"independent per-line scatter -> element SD per unit sigma_eta: O3 {elem_indep_880['o3']:.4f} OD "
    f"({elem_indep_880['o3']/check['o3']*100:.2f} % of the T-weighted tau_O3), N2O {elem_indep_880['n2o']:.5f}")


def line_covariance(sigma_eta, naive_codes=False):
    C = np.zeros((n_el, n_el))
    for g in GASES:
        A = A_line[g]
        if naive_codes:
            s = np.array([IERR_INT.get(c, 0.25) for c in meta[g]["code"]])
            C += (A * s[:, None] ** 2).T @ A
        elif sigma_eta > 0:
            C += sigma_eta ** 2 * (A.T @ A)
    return C


# ---------------------------------------------------------------- OE machinery
def prior_cov():
    Sa = np.zeros((N_STATE, N_STATE))
    for g in GASES:
        s, L = GAS_PRIOR[g]
        Sa[I_GAS[g], I_GAS[g]] = gaussian_prior_covariance(nodes_m, s, L * 1e3)
    Sa[I_T, I_T] = T_PRIOR_K ** 2
    Sa[I_CFC, I_CFC] = CFC_PRIOR ** 2
    for k, (g, r, m, s) in enumerate(GROUPS):
        Sa[I_GRP.start + k, I_GRP.start + k] = s ** 2
    for k in range(I_AER.start, I_AER.stop):
        Sa[k, k] = AER_PRIOR ** 2
    Sa[I_SHIFT, I_SHIFT] = RESP_PRIOR_UM ** 2; Sa[I_SOFT, I_SOFT] = 0.002 ** 2
    return Sa


S_a = prior_cov()
S_a_inv = np.linalg.inv(S_a)


def photon_cov(y, snr):
    return np.diag((np.exp(y / 2.0) / snr) ** 2)


def triplet_rows():
    K = np.zeros((3, N_STATE)); K[:, I_AER] = shell_trip.T
    s_tot = shell_trip[1] + shell_trip[2]
    sig = np.maximum(PHI_TRIPLET * s_tot, cbt.ALPHA_FLOOR)
    return K, np.diag(sig ** 2)


def _blockdiag(A, B):
    Z = np.zeros((A.shape[0] + B.shape[0],) * 2)
    Z[:A.shape[0], :A.shape[0]] = A; Z[A.shape[0]:, A.shape[0]:] = B
    return Z


def posterior(K, S_eps):
    Sinv = np.linalg.inv(S_eps)
    S_hat = np.linalg.inv(K.T @ Sinv @ K + S_a_inv)
    return S_hat, S_hat @ K.T @ Sinv


def joint_sigma_sil(snr, sigma_eta, with_triplet=True, extra_rows=None, naive_codes=False, return_all=False,
                    override_cov=None):
    S_eps = photon_cov(y0, snr) + line_covariance(sigma_eta, naive_codes) if override_cov is None else override_cov
    K = K0.copy()
    if with_triplet:
        Kt, St = triplet_rows(); K = np.vstack([K, Kt]); S_eps = _blockdiag(S_eps, St)
    if extra_rows is not None:
        Ke, Se = extra_rows; K = np.vstack([K, Ke]); S_eps = _blockdiag(S_eps, Se)
    S_hat, G = posterior(K, S_eps)
    s = float(np.sqrt(S_hat[I_SIL, I_SIL]))
    return (s, S_hat, G, K, S_eps) if return_all else s


def removal_budget(snr, sigma_eta, naive_codes=False, with_aerosol_state=False):
    """error of the gas optical depth predicted at the 8.80-um element, decomposed by propagation:
    photon-driven, line-driven (fitted elements and the element's own database error, with their
    covariance), prior-driven (smoothing, including the source-group priors).  The gas-only state
    (aerosol known) is the predictand convention of the archived budget; with_aerosol_state adds the
    seven aerosol parameters to the state (then the silica is co-fitted, but the predictand is still
    the gas OD)."""
    C_line = line_covariance(sigma_eta, naive_codes)
    keep = np.arange(N_STATE) if with_aerosol_state else np.arange(I_SIL)
    Kg = K0[:, keep]; Sa_g = S_a[np.ix_(keep, keep)]
    S_phot = photon_cov(y0, snr)
    S_eps = S_phot + C_line; Sinv = np.linalg.inv(S_eps)
    S_hat = np.linalg.inv(Kg.T @ Sinv @ Kg + np.linalg.inv(Sa_g)); G = S_hat @ Kg.T @ Sinv; A = G @ Kg
    k = Kg[i880].copy()
    if with_aerosol_state:
        k[I_SIL:] = 0.0                                  # predictand = the gas part only
    e = np.zeros(n_el); e[i880] = 1.0
    IA = np.eye(len(keep)) - A
    v_phot = float(k @ G @ S_phot @ G.T @ k)
    v_line = float(k @ G @ C_line @ G.T @ k + e @ C_line @ e - 2.0 * k @ G @ C_line @ e)
    v_prior = float(k @ IA @ Sa_g @ IA.T @ k)
    return dict(sigma_total=float(np.sqrt(v_phot + v_line + v_prior)), photon=float(np.sqrt(v_phot)),
                line=float(np.sqrt(max(v_line, 0.0))), prior_smoothing=float(np.sqrt(v_prior)), dofs=float(np.trace(A)))


def per_element_floors(snr, sigma_eta):
    """gas-removal error at every window element (the two-stage input) and its covariance."""
    C_line = line_covariance(sigma_eta); S_phot = photon_cov(y0, snr)
    keep = np.arange(I_SIL); Kg = K0[:, keep]; Sa_g = S_a[np.ix_(keep, keep)]
    S_eps = S_phot + C_line; Sinv = np.linalg.inv(S_eps)
    S_hat = np.linalg.inv(Kg.T @ Sinv @ Kg + np.linalg.inv(Sa_g)); G = S_hat @ Kg.T @ Sinv; A = G @ Kg
    M = Kg @ G; IA = np.eye(len(keep)) - A
    cov = M @ S_phot @ M.T + (M - np.eye(n_el)) @ C_line @ (M - np.eye(n_el)).T + Kg @ IA @ Sa_g @ IA.T @ Kg.T
    return np.sqrt(np.diag(cov)), cov


def two_stage(floors_od):
    """the paper's procedure on the window: per-element OD floors -> extinction (white gain) ->
    Fisher marginalisation with the triplet on the per-shell columns."""
    sig_ext = floors_od * M1_PER_OD
    def boxc(j):
        return np.array([np.mean(shell_c[j][(lam_coarse >= c - EL_HALF_UM) & (lam_coarse < c + EL_HALF_UM)]) for c in EL_CENTRES_UM])
    t_all = np.r_[shell_trip[0], boxc(0)]
    nuis = [np.r_[shell_trip[j], boxc(j)] for j in range(1, 7)]
    sig_all = np.r_[np.maximum(PHI_TRIPLET * (shell_trip[1] + shell_trip[2]), cbt.ALPHA_FLOOR), sig_ext]
    sa, _, R2 = cbt.marginal(np.arange(18), nuis, sig_all, t_all)
    sa_m, _, _ = cbt.marginal(np.arange(3, 18), nuis, sig_all, t_all)
    return sa, sa_m, 1 - R2


def coherent_peel_factor():
    zb = np.arange(0.0, 60_000.0 + DZ_M, DZ_M); rb, rt = R_EARTH_M + zb[:-1], R_EARTH_M + zb[1:]
    P = np.empty((len(zb) - 1, len(zb) - 1))
    for i, zt in enumerate(zb[:-1]):
        r2 = (R_EARTH_M + zt) ** 2
        P[i] = 2.0 * (np.sqrt(np.maximum(rt ** 2 - r2, 0)) - np.sqrt(np.maximum(rb ** 2 - r2, 0)))
    k = int(np.argmin(np.abs(zb[:-1] - H_TAN_M)))
    zc = 0.5 * (zb[:-1] + zb[1:]); a2 = us_standard_atmosphere(zc)
    n_o3 = tg.number_density_profiles(zc, a2.number_density_m3, gases=["o3"])["o3"]
    tau = P @ n_o3
    return float(n_o3[k] / tau[k])


# ---------------------------------------------------------------- ACE atlas test
def atlas_test():
    """the tropical 16-20-km atlas bin (18-km tangent, 3967 co-added spectra) in the element
    observable: fit one O3 amplitude on the 9.2-um element, predict the 8.80-um element."""
    p = ROOT / "data/ace_atlas/Tropics_016-020km.txt.gz"
    dat = np.loadtxt(p, skiprows=9)
    anu, atr = dat[:, 0], dat[:, 1]
    m = (anu >= NU_LO) & (anu <= NU_HI); anu, atr = anu[m], atr[m]
    tr = np.interp(nu, anu, atr)
    # measured element observable (same weights) and the mean of -ln T (the atlas's own OD average)
    y_meas = element_obs(-np.log(np.clip(tr, 1e-6, None)))
    tau_atlas = -np.log(np.clip(tr, 1e-6, None))
    meanlnT = np.array([np.average(tau_atlas[el_id == i], weights=dlam_dnu[el_id == i]) for i in range(n_el)])
    # model at an 18-km tangent: same sig_z, different chord; scale on the O3 profile
    h = 18_000.0
    ch = tangent_to_slant_paths(np.array([h]), alt)[0]
    t_o3 = (ch * dens["o3"]) @ fine["o3"]["sig_z"]
    t_n2o = sum((ch * dens[g]) @ fine[g]["sig_z"] for g in GASES if g != "o3")     # N2O, CH4, H2O at climatology
    t_cfc = np.interp(nu, xnu, xs, left=0.0, right=0.0) * 1e-4 * float((ch * dens["cfc12"]).sum())
    sl18 = slant_at(h, prof_c)
    t_sulf = np.interp(lam_um, lam_coarse, sl18[1] + sl18[2])
    i92 = int(np.argmin(np.abs(EL_CENTRES_UM - 9.20)))

    def y_model(s, with_aer=True):
        return element_obs(s * t_o3 + t_n2o + t_cfc + (t_sulf if with_aer else 0.0))

    def fit_s(with_aer):
        lo, hi = 0.05, 3.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if y_model(mid, with_aer)[i92] < y_meas[i92]:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    out = {}
    for with_aer in (True, False):
        s = fit_s(with_aer)
        ym = y_model(s, with_aer)
        # the same fit in the mean-cross-section convention (the earlier comparison)
        tau_mean = lambda s_: s_ * t_o3 + t_n2o + t_cfc + (t_sulf if with_aer else 0.0)
        def ymean(s_):
            tt = tau_mean(s_); return np.array([np.average(tt[el_id == i], weights=dlam_dnu[el_id == i]) for i in range(n_el)])
        lo, hi = 0.05, 3.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if ymean(mid)[i92] < y_meas[i92]:
                lo = mid
            else:
                hi = mid
        s_mean = 0.5 * (lo + hi)
        out["with_sulfate" if with_aer else "gas_only"] = dict(
            s_o3_element_observable=s, y_meas=y_meas.tolist(), y_model=ym.tolist(),
            residual_880=float(y_meas[i880] - ym[i880]), residual_all=(y_meas - ym).tolist(),
            s_o3_mean_tau_convention=s_mean, mean_tau_model_880=float(ymean(s_mean)[i880]),
            atlas_mean_minus_lnT_880=float(meanlnT[i880]), y_meas_880=float(y_meas[i880]), y_model_880=float(ym[i880]))
    # figure: measured and modelled -ln<T> on 0.02-um boxcars (one convention on both sides), with the
    # 1-Tg silica and the sulfate background, and the 15 design elements as markers
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        s_fit = out["with_sulfate"]["s_o3_element_observable"]
        tau_mod = s_fit * t_o3 + t_n2o + t_cfc + t_sulf
        centres = np.arange(7.80, 9.30, 0.02)
        def box_obs(tau_, cs, half=0.01):
            yy = np.empty(len(cs))
            for k, c in enumerate(cs):
                mm = (lam_um >= c - half) & (lam_um < c + half)
                yy[k] = -np.log(np.average(np.exp(-np.maximum(tau_[mm], -60.0)), weights=wS[mm]))
            return yy
        y_m_f = box_obs(tau_mod, centres); y_a_f = box_obs(tau_atlas, centres)
        y_sil_f = box_obs(np.interp(lam_um, lam_coarse, sl18[0]), centres); y_sul_f = box_obs(t_sulf, centres)
        ym_el = out["with_sulfate"]["y_model"]
        fig, ax = plt.subplots(figsize=(17 / 2.54, 9.0 / 2.54))
        ax.plot(centres, y_a_f, color="black", lw=1.0, label="ACE-FTS atlas, tropical 16--20 km (3967 spectra), $-\\ln\\langle T\\rangle$ per 0.02 $\\mu$m")
        ax.plot(centres, y_m_f, color="red", lw=1.0, label=f"line-resolved model, O$_3$ amplitude {s_fit:.2f} fitted at 9.2 $\\mu$m, same convention")
        ax.plot(centres, y_sil_f, color="tab:blue", lw=1.0, label="1 Tg silica")
        ax.plot(centres, y_sul_f, color="green", ls="--", lw=1.0, label="calibrated sulfate background")
        ax.plot(EL_CENTRES_UM, y_meas, "o", color="black", ms=5, mfc="none", label="measured 0.1-$\\mu$m elements")
        ax.plot(EL_CENTRES_UM, ym_el, "s", color="red", ms=4, mfc="none", label="modelled 0.1-$\\mu$m elements")
        ax.axvspan(8.75, 8.85, color="magenta", alpha=0.15); ax.axvspan(9.15, 9.25, color="green", alpha=0.15)
        ax.set_xlabel("wavelength [$\\mu$m]"); ax.set_ylabel("element optical depth $-\\ln\\langle T\\rangle$", fontsize=9)
        ax.set_xlim(7.8, 9.3); ax.set_ylim(0, 2.0); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(ROOT / "figures/scisat_ace_atlas_874.png", dpi=200); plt.close(fig)
        out["figure"] = "figures/scisat_ace_atlas_874.png (round 70: both sides in the element observable on 0.02-um boxcars)"
        out["fine_boxcar"] = dict(centres_um=centres.tolist(), y_atlas=y_a_f.tolist(), y_model=y_m_f.tolist(),
                                  y_silica_1tg=y_sil_f.tolist(), y_sulfate=y_sul_f.tolist())
    except Exception as exc:  # the archive is the deliverable; the figure is a convenience
        out["figure_error"] = repr(exc)
    out["note"] = ("tropical 16-20 km bin, 18-km tangent, US-standard profiles (N2O, CH4, H2O, CFC-12 at climatology) scaled by one O3 amplitude fitted on the 9.2-um element; "
                   "the atlas transmittance (0.02 cm^-1 resolution, oversampled at 0.0025 cm^-1) is formed into the same boxcar element "
                   "observable; residual = measured minus predicted at 8.80 um")
    return out


# ---------------------------------------------------------------- Monte Carlo closure
GAS_IDX = np.arange(NGAS * NN)


def gn_fit(y_obs, K_tri, S_eps, n_iter=25):
    """iterated optimal estimation (Gauss-Newton) with the nonlinear element forward model; the step is
    damped so that no gas-node multiplier moves by more than 0.5 and the registration by more than
    0.003 um per iteration, the multipliers stay in [0.05, 5], and the iteration stops when the gas
    nodes move by less than 1e-5 and the silica amplitude by less than 1e-6 (slant Tg)."""
    x = x_ref.copy()
    Sinv = np.linalg.inv(S_eps)
    last = np.inf
    for _ in range(n_iter):
        K, tau = jacobian(x)
        y = element_obs(tau, shift=float(x[I_SHIFT]), soft=float(x[I_SOFT]))
        if K_tri is not None:
            K = np.vstack([K, K_tri]); yv = np.r_[y, K_tri @ x]
        else:
            yv = y
        S_hat = np.linalg.inv(K.T @ Sinv @ K + S_a_inv)
        dx = S_hat @ (K.T @ Sinv @ (y_obs - yv) - S_a_inv @ (x - x_ref))
        step = min(1.0, 0.5 / max(float(np.abs(dx[GAS_IDX]).max()), 1e-12), 10.0 / max(abs(float(dx[I_T])), 1e-12),
                   0.003 / max(abs(float(dx[I_SHIFT])), 1e-12))
        x = x + step * dx
        x[GAS_IDX] = np.clip(x[GAS_IDX], 0.05, 5.0)
        x[I_SOFT] = float(np.clip(x[I_SOFT], 0.001, 0.02))
        last = float(np.abs(step * dx[I_SIL]))
        if np.abs(step * dx[GAS_IDX]).max() < 1e-5 and last < 1e-6:
            break
    return x, S_hat, last


def resp_obs(tau, shift, soft):
    base = dlam_dnu * S_sun * np.exp(-np.maximum(tau, -60.0)); base0 = dlam_dnu * S_sun
    y = np.empty(n_el)
    for i, c in enumerate(EL_CENTRES_UM):
        lo, hi = c + shift - EL_HALF_UM, c + shift + EL_HALF_UM
        sc = soft * np.sqrt(2) + 1e-9
        prof = 0.5 * (erf((lam_um - lo) / sc) - erf((lam_um - hi) / sc))
        y[i] = -np.log((prof * base).sum() / (prof * base0).sum())
    return y


def mc_closure(ndraw, sigma_eta, rng, snr=SNR_REF, masses_shell=(0.0, 0.05, 0.1, 0.2, 0.3),
               response_pert=True, line_errors=True, tropical=False, with_triplet=True, resp_sigma_um=0.003):
    S_eps_el = photon_cov(y0, snr) + (line_covariance(sigma_eta) if line_errors else 0.0)
    Kt, St = triplet_rows()
    S_eps = _blockdiag(S_eps_el, St) if with_triplet else S_eps_el
    Lsa = {g: np.linalg.cholesky(S_a[I_GAS[g], I_GAS[g]]) for g in GASES}
    sig_trip = np.sqrt(np.diag(St)); sig_grp = np.array([s for g, r, m, s in GROUPS])
    res = {m: dict(A=[], s=[], ge=[], conv=[]) for m in masses_shell}
    for d in range(ndraw):
        xt = x_ref.copy()
        for g in GASES:
            xt[I_GAS[g]] = np.clip(1.0 + Lsa[g] @ rng.standard_normal(NN), 0.2, 3.0)
        if tropical:
            xt[I_O3] *= 0.4
        xt[I_T] = T_PRIOR_K * rng.standard_normal()
        xt[I_CFC] = 1.0 + CFC_PRIOR * rng.standard_normal()
        xt[I_GRP] = sig_grp * rng.standard_normal(NG) if line_errors else 0.0
        xt[I_AER] = [0.0, np.exp(rng.uniform(-0.7, 0.7)), np.exp(rng.uniform(-0.7, 0.7)),
                     rng.uniform(-0.2, 0.2) * cbt.CAL_QUIET[0][0], rng.uniform(-0.1, 0.1) * cbt.CAL_QUIET[0][1],
                     rng.uniform(-0.2, 0.2) * cbt.CAL_QUIET[1][0], rng.uniform(-0.1, 0.1) * cbt.CAL_QUIET[1][1]]
        dtau = np.zeros(n_nu)
        if line_errors and sigma_eta > 0:
            for g in GASES:
                dtau += Pline[g].T @ (sigma_eta * rng.standard_normal(len(meta[g]["nu"]))).astype(np.float32)
        xt[I_SHIFT] = rng.normal(0.0, resp_sigma_um) if response_pert else 0.0
        xt[I_SOFT] = (RESP_SOFT0 + rng.normal(0.0, 0.001)) if response_pert else RESP_SOFT0
        xt[I_SOFT] = float(np.clip(xt[I_SOFT], 0.001, 0.02))
        for M_shell in masses_shell:
            xt_m = xt.copy(); xt_m[I_SIL] = M_shell / RHO_SLANT_TO_SHELL
            tau_t = tau_total(xt_m, dtau)
            y_t = element_obs(tau_t, shift=float(xt_m[I_SHIFT]), soft=float(xt_m[I_SOFT]))
            y_obs = y_t + rng.standard_normal(n_el) * np.exp(y_t / 2.0) / snr
            if with_triplet:
                y_obs = np.r_[y_obs, Kt @ xt_m + sig_trip * rng.standard_normal(3)]
            x_hat, S_hat, last = gn_fit(y_obs, Kt if with_triplet else None, S_eps)
            res[M_shell]["A"].append(float(x_hat[I_SIL] * RHO_SLANT_TO_SHELL))
            res[M_shell]["s"].append(float(np.sqrt(S_hat[I_SIL, I_SIL]) * RHO_SLANT_TO_SHELL))
            res[M_shell]["conv"].append(last * RHO_SLANT_TO_SHELL)
            xg_t = xt_m.copy(); xg_t[I_AER] = 0.0; xg_h = x_hat.copy(); xg_h[I_AER] = 0.0
            res[M_shell]["ge"].append(float(element_obs(tau_total(xg_h))[i880] - element_obs(tau_total(xg_t, dtau))[i880]))
            res[M_shell]["shift_err"] = res[M_shell].get("shift_err", []) + [float(x_hat[I_SHIFT] - xt_m[I_SHIFT])]
    summary = {}
    for M_shell, r in res.items():
        A = np.array(r["A"]); s = np.array(r["s"]); ge = np.array(r["ge"]); z = (A - M_shell) / s
        conv = np.array(r["conv"]); sh = np.array(r.get("shift_err", [0.0]))
        summary[str(M_shell)] = dict(
            n=int(len(A)), bias_mean=float(np.mean(A - M_shell)), bias_median=float(np.median(A - M_shell)),
            sd=float(np.std(A, ddof=1)), robust_sd=float(1.4826 * np.median(np.abs(A - np.median(A)))),
            posterior_sigma_median=float(np.median(s)), sd_over_posterior=float(np.std(A, ddof=1) / np.median(s)),
            coverage_1sig=float(np.mean(np.abs(z) < 1)), coverage_2sig=float(np.mean(np.abs(z) < 2)),
            coverage_3sig=float(np.mean(np.abs(z) < 3)),
            p_detect_3sig=float(np.mean(A / s > 3.0)), p_detect_4p54sig=float(np.mean(A / s > 4.54)),
            gas880_err_rms=float(np.sqrt(np.mean(ge ** 2))), gas880_err_mean=float(np.mean(ge)),
            M_min_3sig_posterior=float(3.0 * np.median(s)),
            frac_last_step_below_0p1sigma=float(np.mean(conv < 0.1 * s)), shift_err_rms_um=float(np.sqrt(np.mean(sh ** 2))))
    return summary


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndraw", type=int, default=100)
    ap.add_argument("--seed", type=int, default=70)
    ap.add_argument("--coarse", action="store_true", help="use the 0.01-cm^-1 budget grids")
    args = ap.parse_args()
    global ARCH, SPEC
    if args.coarse:
        ARCH = OUT / "element_observable_closure_coarse.json"; SPEC = OUT / "element_observable_closure_coarse.npz"
    rng = np.random.default_rng(args.seed)
    arch = dict(
        setup=dict(h_tan_km=H_TAN_M / 1e3, grid_cm=[NU_LO, NU_HI, STEP], coarse_grid=_COARSE, elements_um=EL_CENTRES_UM.tolist(),
                   element_half_width_um=EL_HALF_UM, snr_ref=SNR_REF, nodes_km=list(DEFAULT_NODES_KM),
                   priors=dict(gases=GAS_PRIOR, T_K=T_PRIOR_K, cfc12=CFC_PRIOR, aerosol=AER_PRIOR, response_um=[RESP_PRIOR_UM, 0.002],
                               groups={f"{g}:{r}": s for g, r, m, s in GROUPS}),
                   state=names, n_state=N_STATE, solar="5772-K Planck per unit wavelength", response=f"0.1-um boxcar in wavelength, edges softened by a Gaussian of {RESP_SOFT0} um",
                   ierr_code_midpoints=IERR_INT, sigma_eta_base=SIGMA_ETA_BASE, line_window_cm=LINE_WIN_CM),
        line_statistics=code_share, elem_indep_880_per_unit_sigma_eta=elem_indep_880, line_response_sum_880=check,
        rho_slant_to_shell_20km=RHO_SLANT_TO_SHELL, silica_shell_ext_880=t_shell_880, silica_slant_od_880=tau_slant_880)

    # ---- 1. observable vs mean optical depth
    arch["observable"] = dict(y_total=y0.tolist(), y_gas=y_gas0.tolist(), mean_tau_total=y_mean.tolist(),
                              mean_tau_gas=y_gas_mean.tolist(), y_gas_880=float(y_gas0[i880]), mean_tau_gas_880=float(y_gas_mean[i880]),
                              o3_T_weighted_880=float(element_mean(tau0, tau_ref["o3"])[i880]),
                              archived_budget_tau_o3_880=1.1538)
    log("[1] gas-only -ln<T> / <tau> per element: " + ", ".join(f"{a:.3f}/{b:.3f}" for a, b in zip(y_gas0, y_gas_mean)))

    # ---- 2. removal budget at 8.80 um
    bud = {str(eta): dict(snr2000=removal_budget(SNR_REF, eta), snr_inf=removal_budget(1e9, eta),
                          snr2000_with_aerosol_state=removal_budget(SNR_REF, eta, with_aerosol_state=True))
           for eta in SIGMA_ETA_SCAN}
    bud["naive_all_codes_independent"] = removal_budget(SNR_REF, 0.0, naive_codes=True)
    arch["removal_budget_880"] = bud
    b = bud[str(SIGMA_ETA_BASE)]["snr2000"]
    log(f"[2] gas-removal error at 8.80 um, sigma_eta {SIGMA_ETA_BASE}: total {b['sigma_total']:.3e} OD (photon {b['photon']:.2e}, "
        f"line {b['line']:.2e}, prior {b['prior_smoothing']:.2e}); archived <tau> budget {BUDGET_OD_880:.3e}")
    for eta in SIGMA_ETA_SCAN:
        bb = bud[str(eta)]["snr2000"]; bi = bud[str(eta)]["snr_inf"]; ba = bud[str(eta)]["snr2000_with_aerosol_state"]
        log(f"      eta {eta:6.3f}: total {bb['sigma_total']:.3e} (S->inf {bi['sigma_total']:.3e}; aerosol co-fitted {ba['sigma_total']:.3e}); "
            f"line {bb['line']:.2e} photon {bb['photon']:.2e} prior {bb['prior_smoothing']:.2e}")
    log(f"      all codes independent (naive): {bud['naive_all_codes_independent']['sigma_total']:.3e} OD")

    # ---- 3. joint vs two-stage (window)
    joint = {}
    for eta in SIGMA_ETA_SCAN:
        s_w = joint_sigma_sil(SNR_REF, eta); s_w_alone = joint_sigma_sil(SNR_REF, eta, with_triplet=False)
        fl, covfl = per_element_floors(SNR_REF, eta)
        sa, sa_m, omr2 = two_stage(fl)
        joint[str(eta)] = dict(joint_window_triplet_slant=s_w, joint_window_triplet_shell=s_w * RHO_SLANT_TO_SHELL,
                               joint_window_alone_shell=s_w_alone * RHO_SLANT_TO_SHELL,
                               two_stage_window_triplet_shell=sa, two_stage_window_alone_shell=sa_m, two_stage_one_minus_R2=omr2,
                               per_element_floors_od=fl.tolist(), per_element_floors_m1=(fl * M1_PER_OD).tolist(),
                               floor_corr_neighbour=float(np.median([covfl[i, i + 1] / np.sqrt(covfl[i, i] * covfl[i + 1, i + 1])
                                                                     for i in range(n_el - 1)])))
    joint["naive_codes"] = dict(joint_window_triplet_shell=joint_sigma_sil(SNR_REF, 0.0, naive_codes=True) * RHO_SLANT_TO_SHELL)
    # the paper's adopted per-element floors as the measurement covariance, with the gas state retained
    thr_ = json.load(open(ROOT / "outputs/calibrated_background_thresholds.json"))
    paper_od = np.array(thr_["window_floors_m1"], float) * OD_PER_M1
    joint["paper_floors_joint"] = dict(
        window_triplet_shell=joint_sigma_sil(SNR_REF, 0.0, override_cov=np.diag(paper_od ** 2)) * RHO_SLANT_TO_SHELL,
        window_alone_shell=joint_sigma_sil(SNR_REF, 0.0, with_triplet=False, override_cov=np.diag(paper_od ** 2)) * RHO_SLANT_TO_SHELL,
        note="measurement covariance = the adopted per-element floors (diagonal, in OD); gas, group, T, CFC and response states kept with their priors")
    # the paper's own floors on the same two-stage machinery (reference point)
    thr = json.load(open(ROOT / "outputs/calibrated_background_thresholds.json"))
    paper_fl = np.array(thr["window_floors_m1"], float) * OD_PER_M1
    sa_p, sa_pm, _ = two_stage(paper_fl)
    joint["paper_floors_two_stage"] = dict(window_triplet_shell=sa_p, window_alone_shell=sa_pm,
                                           archived=dict(window=0.1411502613954138 / 3, window_alone=0.3609855339800253 / 3))
    arch["joint_vs_two_stage_window"] = joint
    j = joint[str(SIGMA_ETA_BASE)]
    log(f"      paper floors as covariance, gas state kept: joint 3 sigma {3*joint['paper_floors_joint']['window_triplet_shell']:.4f} "
        f"(MIR alone {3*joint['paper_floors_joint']['window_alone_shell']:.4f}) Tg per shell")
    log(f"[3] window, eta {SIGMA_ETA_BASE}: joint 3 sigma {3*j['joint_window_triplet_shell']:.4f} Tg per shell (MIR alone "
        f"{3*j['joint_window_alone_shell']:.4f}); two-stage with the element-observable floors {3*j['two_stage_window_triplet_shell']:.4f} "
        f"({3*j['two_stage_window_alone_shell']:.4f}); paper's floors on this machinery {3*sa_p:.4f} (archived 0.1412); "
        f"neighbour floor correlation {j['floor_corr_neighbour']:.2f}; floors {np.min(j['per_element_floors_od']):.2e}-{np.max(j['per_element_floors_od']):.2e} OD")

    # ---- 4. band configuration
    band_el = np.round(np.array(thr["band01_elements_um"], float), 2); band_fl = np.array(thr["band01_floors_m1"], float)
    outer = [k for k, c in enumerate(band_el) if c > 9.25]
    K_out = np.zeros((len(outer), N_STATE))
    for n, c in enumerate(band_el[outer]):
        K_out[n, I_AER] = [np.mean(shell_o[jj][(lam_outer >= c - EL_HALF_UM) & (lam_outer < c + EL_HALF_UM + 1e-9)]) for jj in range(7)]
    S_out = np.diag(band_fl[outer] ** 2)
    band = {str(eta): dict(joint_band_shell=joint_sigma_sil(SNR_REF, eta, extra_rows=(K_out, S_out)) * RHO_SLANT_TO_SHELL)
            for eta in SIGMA_ETA_SCAN}
    arch["joint_band"] = dict(per_eta=band, paper_two_stage_band_sigma=0.1010373253040234 / 3,
                              note="window: gas state + line covariance; 32 outer elements: the paper's empirical floors as per-shell rows")
    log(f"[4] band, eta {SIGMA_ETA_BASE}: joint 3 sigma {3*band[str(SIGMA_ETA_BASE)]['joint_band_shell']:.4f} Tg per shell (paper 0.101)")

    # ---- 5. SNR scan
    scan = {}
    for snr in (100., 200., 400., 700., 1000., 2000., 4000., 7000., 10000., 20000., 1e5, 1e9):
        scan[str(snr)] = dict(sigma_sil_shell_alone=joint_sigma_sil(snr, SIGMA_ETA_BASE, with_triplet=False) * RHO_SLANT_TO_SHELL,
                              sigma_sil_shell_triplet=joint_sigma_sil(snr, SIGMA_ETA_BASE) * RHO_SLANT_TO_SHELL,
                              removal_880=removal_budget(snr, SIGMA_ETA_BASE)["sigma_total"])
    plateau = scan["1000000000.0"]["sigma_sil_shell_triplet"]; s2000 = scan["2000.0"]["sigma_sil_shell_triplet"]
    snrs = np.array([float(k) for k in scan]); vals = np.array([scan[k]["sigma_sil_shell_triplet"] for k in scan])
    cross = float(np.exp(np.interp(np.log(np.sqrt(2) * plateau), np.log(vals[::-1]), np.log(snrs[::-1])))) if vals[0] > np.sqrt(2) * plateau else None
    arch["snr_scan"] = dict(per_snr=scan, plateau_shell_triplet=plateau, ratio_2000_to_plateau=s2000 / plateau, snr_crossover_sqrt2=cross,
                            note="plateau = fixed database term (independent per-line scatter at sigma_eta_base), not reducible by averaging occultations")
    log(f"[5] SNR scan (triplet+window, 3 sigma per shell): " + ", ".join(f"S={float(k):.0f}: {3*v['sigma_sil_shell_triplet']:.4f}" for k, v in scan.items())
        + f"; crossover (sqrt2 x plateau) at S ~ {cross}")

    # ---- 6. vertical conversion and crossover
    cf = coherent_peel_factor(); white = M1_PER_OD
    photon_white = b["photon"] * white
    line_coh = b["line"] * cf; prior_coh = b["prior_smoothing"] * cf
    def dz_star(sig_white_od, ext_coh):
        return G_ONION ** 2 * sig_white_od ** 2 / (8.0 * R_EARTH_M * ext_coh ** 2)
    arch["vertical_conversion"] = dict(
        coherent_peel_m1_per_od=cf, white_m1_per_od=white, ratio_white_over_coherent=white / cf,
        photon_white_m1=photon_white, line_coherent_m1=line_coh, prior_coherent_m1=prior_coh,
        total_white_m1=b["sigma_total"] * white, mixed_m1=float(np.sqrt(photon_white ** 2 + line_coh ** 2 + prior_coh ** 2)),
        dz_star_m_photon_vs_coherent=dz_star(b["photon"], np.hypot(line_coh, prior_coh)),
        dz_star_m_referee_inputs=dz_star(0.00096, 3.0e-9),
        archived_budget_white_m1=BUDGET_OD_880 * white,
        formula="dz* = g^2 sigma_white^2 / (8 R_E (coherent extinction term)^2)")
    v = arch["vertical_conversion"]
    log(f"[6] conversion white {white:.3e} vs coherent {cf:.3e} m^-1/OD (ratio {white/cf:.2f}); photon {photon_white:.2e} (white), "
        f"line {line_coh:.2e} + prior {prior_coh:.2e} (coherent); mixed {v['mixed_m1']:.2e} vs all-white {v['total_white_m1']:.2e} m^-1; "
        f"dz* {v['dz_star_m_photon_vs_coherent']/1e3:.2f} km (referee inputs {v['dz_star_m_referee_inputs']/1e3:.2f} km)")

    # ---- 7. ACE atlas test
    arch["atlas_test"] = atlas_test()
    a = arch["atlas_test"]["with_sulfate"]; a2 = arch["atlas_test"]["gas_only"]
    log(f"[7] atlas 18 km: s_O3 = {a['s_o3_element_observable']:.3f} (element observable, sulfate included; gas-only {a2['s_o3_element_observable']:.3f}; "
        f"mean-tau convention {a['s_o3_mean_tau_convention']:.3f}); 8.80 um measured {a['y_meas_880']:.4f}, predicted {a['y_model_880']:.4f}, "
        f"residual {a['residual_880']:+.4f} OD (gas-only residual {a2['residual_880']:+.4f}); atlas mean(-lnT) {a['atlas_mean_minus_lnT_880']:.4f}")

    # ---- 7b. response knowledge: the effect of an unfitted wavelength shift / edge softening
    Kt, St = triplet_rows(); S_eps_r = _blockdiag(photon_cov(y0, SNR_REF) + line_covariance(SIGMA_ETA_BASE), St)
    resp = {}
    S_a_inv_save = S_a_inv.copy()
    # unfitted: pin the response parameters with a tight prior
    S_a_inv[I_SHIFT, I_SHIFT] = 1e12; S_a_inv[I_SOFT, I_SOFT] = 1e12
    for shift in (0.0005, 0.001, 0.003, 0.01):
        y_t = resp_obs(tau0, shift, RESP_SOFT0)
        x_hat, S_hat, _ = gn_fit(np.r_[y_t, Kt @ x_ref], Kt, S_eps_r)
        resp[f"shift_{shift}"] = dict(max_abs_dy_od=float(np.abs(y_t - y0).max()), dy_880_od=float(y_t[i880] - y0[i880]),
                                      silica_error_shell_tg=float(x_hat[I_SIL] * RHO_SLANT_TO_SHELL))
    for soft in (0.003, 0.007, 0.01):
        y_t = resp_obs(tau0, 0.0, soft)
        x_hat, S_hat, _ = gn_fit(np.r_[y_t, Kt @ x_ref], Kt, S_eps_r)
        resp[f"soft_{soft}"] = dict(max_abs_dy_od=float(np.abs(y_t - y0).max()), silica_error_shell_tg=float(x_hat[I_SIL] * RHO_SLANT_TO_SHELL))
    S_a_inv[:] = S_a_inv_save
    # fitted: the same perturbations with the response parameters free (prior 0.005 um)
    for key in list(resp):
        kind, val = key.split("_"); val = float(val)
        y_t = resp_obs(tau0, val if kind == "shift" else 0.0, val if kind == "soft" else RESP_SOFT0)
        x_hat, S_hat, _ = gn_fit(np.r_[y_t, Kt @ x_ref], Kt, S_eps_r)
        resp[key]["silica_error_shell_tg_response_fitted"] = float(x_hat[I_SIL] * RHO_SLANT_TO_SHELL)
        resp[key]["recovered_parameter_um"] = float(x_hat[I_SHIFT] if kind == "shift" else x_hat[I_SOFT])
    resp["posterior_sigma_shift_um"] = float(np.sqrt(np.linalg.inv(np.vstack([K0, Kt]).T @ np.linalg.inv(S_eps_r) @ np.vstack([K0, Kt]) + S_a_inv)[I_SHIFT, I_SHIFT]))
    resp["sigma_sil_shell_response_fixed"] = float(np.sqrt(np.linalg.inv(np.vstack([K0, Kt])[:, :I_SHIFT].T @ np.linalg.inv(S_eps_r) @ np.vstack([K0, Kt])[:, :I_SHIFT] + S_a_inv[:I_SHIFT, :I_SHIFT])[I_SIL, I_SIL]) * RHO_SLANT_TO_SHELL)
    arch["response_knowledge"] = resp
    log("[7b] unfitted response: " + "; ".join(f"{k}: dy_max {v['max_abs_dy_od']:.4f} OD, silica {v['silica_error_shell_tg']:+.4f} -> fitted {v['silica_error_shell_tg_response_fitted']:+.4f} Tg"
                                              for k, v in resp.items() if isinstance(v, dict)) + f"; posterior sigma(shift) {resp['posterior_sigma_shift_um']*1e3:.2f} nm")

    # ---- 8. Monte Carlo closure
    mc = {}; t0 = time.time()
    mc["baseline"] = mc_closure(args.ndraw, SIGMA_ETA_BASE, rng, masses_shell=(0.0, 0.05, 0.1, 0.2))
    log(f"[8] MC baseline ({args.ndraw} draws, {time.time()-t0:.0f} s): " + "; ".join(
        f"M={m}: bias {r['bias_median']:+.4f}, sd {r['sd']:.4f}, post {r['posterior_sigma_median']:.4f}, cov1 {r['coverage_1sig']:.2f}, "
        f"P3s {r['p_detect_3sig']:.2f}, P4.54 {r['p_detect_4p54sig']:.2f}" for m, r in mc["baseline"].items()))
    n2 = max(args.ndraw // 3, 30)
    for key, kw in (("no_line_errors", dict(line_errors=False)), ("no_response_pert", dict(response_pert=False)),
                    ("response_sigma_0p01", dict(resp_sigma_um=0.01)),
                    ("tropical_o3_0p4", dict(tropical=True)), ("sigma_eta_0p05", dict(sigma_eta=0.05)),
                    ("sigma_eta_0p01", dict(sigma_eta=0.01)), ("mir_alone", dict(with_triplet=False))):
        kw2 = dict(sigma_eta=SIGMA_ETA_BASE); kw2.update(kw)
        mc[key] = mc_closure(n2, kw2.pop("sigma_eta"), rng, masses_shell=(0.0, 0.1), **kw2)
        r = mc[key]["0.1"]; r0 = mc[key]["0.0"]
        log(f"      {key:18s}: M=0.1 bias {r['bias_median']:+.4f}, sd {r['sd']:.4f}, post {r['posterior_sigma_median']:.4f}, "
            f"cov1 {r['coverage_1sig']:.2f}, P3s {r['p_detect_3sig']:.2f}; null P(z>3) {r0['p_detect_3sig']:.3f}")
    arch["monte_carlo"] = mc
    arch["monte_carlo_note"] = ("truth: O3 nodes from the 30 %/6-km prior (clipped 0.2-3), N2O 5 %, T 2 K, CFC 3 %, source-group "
                                "scalings at the code midpoints, independent per-line scatter sigma_eta, sulfate amplitudes x0.5-2 "
                                "(log-uniform), PSD r_med +-20 %, sigma +-10 %, photon noise at S=2000 (exp(y/2) growth), element "
                                "response shifted by N(0, 0.003 um) and edges softened by |N(0, 0.003 um)| in the truth, both co-retrieved "
                                "with a 0.005-um prior; retrieval: 8 damped Gauss-Newton iterations of the nonlinear element model from the prior. Masses in Tg per 0.5-km "
                                "shell at 20 km (slant amplitude x rho).")
    with open(ARCH, "w") as f:
        json.dump(arch, f, indent=1)
    np.savez_compressed(SPEC, nu=nu, tau_gas=tau_gas0, tau_total=tau0, tau_sil_per_tg=tau_aer_fine[0],
                        tau_sulf=tau_aer_fine[1] + tau_aer_fine[2], y=y0, y_gas=y_gas0, y_mean=y_mean, el_centres=EL_CENTRES_UM)
    log("archived ->", ARCH)


if __name__ == "__main__":
    main()
