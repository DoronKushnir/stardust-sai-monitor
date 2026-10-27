"""
ace_nulltest_nonlinear_refit.py -- referee round 5, item 4, round 71:
a CONVERGED per-occultation nonlinear (Mie-in-the-loop) refit of the ACE
full-band null test, replacing the fixed linearization points of
reproduce/ace_nulltest_diagnostics.py part 4.

Same 101 occultations, mask (760-1250 cm^-1), constant 8e-3 OD per-bin
weight, 4-sigma clipping (up to 3 passes) and silica column as the nominal
linear fit (reproduce/ace_fullspectrum_retrieval.py).  The background is the
calibrated two-component sulfate model of calibrated_background_thresholds
(each mode carries a prescribed share of the GloSSAC 525-nm extinction
profile), but here the two modes' (r_med, sigma) are free nonlinear
parameters inside the admissible boxes of round 70 (fine 30-150 nm x
1.3-2.0, coarse 800-3000 nm x 1.4-2.2) and the Mie extinction of each mode
is recomputed exactly at every iteration.

Exactness.  The engine's mode column is separable, basis(h, nu) =
[C @ (frac alpha525)](h) x k(nu; r, sigma)/k(525 nm; r, sigma), with
k(lambda) = trapz(Q_ext(2 pi r/lambda, m(lambda)) pi r^2 dN/dr, r) on the
engine's 700-point 1 nm-10 um radius grid (checked against the cached
nominal basis to 1e-14).  Q_ext is therefore tabulated once per member
optics on that radius grid and every PSD evaluation is the exact engine
quadrature (a weighted sum), with analytic dN/dr derivatives in
(ln r_med, sigma); no interpolation anywhere.

Fit.  theta = [A_f, A_c, M_sil, const | ln r_f, sigma_f, ln r_c, sigma_c].
Variable projection: for given PSD parameters the four linear amplitudes are
solved exactly; the PSD parameters take damped Gauss-Newton
(Levenberg-Marquardt) steps on the reduced (Kaufman) Jacobian, projected
onto the admissible box; a step is accepted only if chi^2 decreases.
Convergence: |Delta M_sil| < 0.01 sigma_M (marginal) and relative chi^2
change < 1e-5 after an accepted step; MAX_ITER iterations; the iteration at
which the criterion is first met is recorded ("within 10" is reported).
Three starting points per member optics: the quiet calibrated PSD, the
post-Hunga-Tonga calibrated PSD, and the best of a 400-point grid over the
box (global check); the best of the three by chi^2 is the reported fit,
and the start dependence is archived.  Two member optics: LM65T223 (quiet,
nominal) and LM72T213 (post-HT).

Errors.  sigma_M marginal: from the full 8-column whitened Jacobian at the
solution, with PSD parameters sitting on a box boundary treated as fixed
(active set); sigma_M free: the same with all 8 free (pseudo-inverse);
sigma_M conditional: PSD fixed at the solution (4 linear parameters).

Outputs
  outputs/ace_nulltest_nonlinear_refit.json
  outputs/ace_nulltest_nonlinear_refit_perocc.csv
  outputs/mie_qext_tables.npz           (Q_ext tables, cached)
  outputs/log_ace_nonlinear_refit.txt

Run from the repo root:
  python reproduce/ace_nulltest_nonlinear_refit.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "reproduce"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ace_fullspectrum_retrieval as F  # noqa: E402
from calibrated_background_thresholds import (  # noqa: E402
    ALT, CAL_QUIET, CAL_POSTHT, member_ri, glossac_2025N_quiet_profile)
from saimon.geometry import tangent_to_slant_paths  # noqa: E402
from saimon.mie import BohrenHuffmanMie  # noqa: E402

OUTDIR = _ROOT / "outputs"
OUT_JSON = OUTDIR / "ace_nulltest_nonlinear_refit.json"
OUT_CSV = OUTDIR / "ace_nulltest_nonlinear_refit_perocc.csv"
QTAB = OUTDIR / "mie_qext_tables.npz"
LOG = OUTDIR / "log_ace_nonlinear_refit.txt"
R70_JSON = OUTDIR / "ace_nulltest_diagnostics.json"
R70_CSV = OUTDIR / "ace_nulltest_diagnostics_perocc.csv"

R_GRID = np.logspace(np.log10(1e-9), np.log10(10e-6), 700)   # the engine's grid
ADMISSIBLE = {"fine": ((30.0, 150.0), (1.3, 2.0)),
              "coarse": ((800.0, 3000.0), (1.4, 2.2))}
BOXES = {   # ln r_f [nm], sigma_f, ln r_c [nm], sigma_c
    "admissible": (np.array([np.log(30.0), 1.3, np.log(800.0), 1.4]),
                   np.array([np.log(150.0), 2.0, np.log(3000.0), 2.2])),
    "wide": (np.array([np.log(30.0), 1.3, np.log(500.0), 1.2]),
             np.array([np.log(300.0), 2.5, np.log(5000.0), 2.2])),
    "narrow": (np.array([np.log(50.0), 1.5, np.log(1200.0), 1.7]),
               np.array([np.log(72.0), 1.7, np.log(1800.0), 1.9])),
}
MEMBERS = {"LM65T223": CAL_QUIET, "LM72T213": CAL_POSTHT}
STARTS = {"quiet_psd": CAL_QUIET, "postHT_psd": CAL_POSTHT}
MAX_ITER = 40
TOL_M = 0.01          # |dM| < TOL_M * sigma_M (marginal)
TOL_CHI2 = 1e-5
GRID_N = (5, 4, 5, 4)  # ln r_f, sigma_f, ln r_c, sigma_c
ISIL = 2               # index of M_sil in the linear block
_LOG_LINES = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LOG_LINES.append(s)


def robust_sd(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


# ---------------------------------------------------------------------------
# Q_ext tables: Qw[j, i] = Q_ext(2 pi r_i / lambda_j, m_j) pi r_i^2 w_i with w the
# trapezoid weights of the engine's radius grid; row 0 is 525 nm.
def trapz_weights(r):
    w = np.empty_like(r)
    w[0] = 0.5 * (r[1] - r[0]); w[-1] = 0.5 * (r[-1] - r[-2])
    w[1:-1] = 0.5 * (r[2:] - r[:-2])
    return w


def build_qtable(member, wl_m):
    ri, wt, T = member_ri(member)
    mie = BohrenHuffmanMie()
    lams = np.r_[525e-9, wl_m]
    m = np.r_[complex(ri(np.array([525e-9]))[0]), ri(wl_m)]
    w = trapz_weights(R_GRID)
    Qw = np.empty((len(lams), len(R_GRID)))
    for j, lam in enumerate(lams):
        x = 2.0 * np.pi * R_GRID / float(lam)
        q, _, _ = mie.efficiencies(x, complex(m[j]))
        Qw[j] = q * np.pi * R_GRID**2 * w
    return Qw


def psd_and_derivs(lnr, sg):
    """dN/dr (n0 = 1) on R_GRID and its derivatives wrt (ln r_med [nm], sigma)."""
    L = np.log(sg)
    u = np.log(R_GRID * 1e9) - lnr          # lnr = ln(r_med [nm]); R_GRID in m
    n = np.exp(-u**2 / (2 * L**2)) / (np.sqrt(2 * np.pi) * R_GRID * L)
    dn_dlnr = n * u / L**2
    dn_ds = n * (u**2 / L**2 - 1.0) / (L * sg)
    return n, dn_dlnr, dn_ds


def mode_shape(Qw, lnr, sg):
    """s(nu) = k(nu)/k(525) and its derivatives wrt (ln r_med, sigma)."""
    n, dn1, dn2 = psd_and_derivs(lnr, sg)
    k = Qw @ n; k1 = Qw @ dn1; k2 = Qw @ dn2
    k0, k01, k02 = k[0], k1[0], k2[0]
    s = k[1:] / k0
    ds1 = (k1[1:] * k0 - k[1:] * k01) / k0**2
    ds2 = (k2[1:] * k0 - k[1:] * k02) / k0**2
    return s, ds1, ds2


# ---------------------------------------------------------------------------
class OccFit:
    """One occultation, one member optics."""

    def __init__(self, y, ok, sil, dS, Qf, Qc, sig, box, nonneg=False):
        self.y, self.ok, self.sil, self.dS = y, ok, sil, dS   # dS = (dS_f, dS_c)
        self.Qf, self.Qc, self.sig = Qf, Qc, sig
        self.box_lo, self.box_hi = box
        self.nonneg = nonneg       # A_f, A_c >= 0 (physically admissible amplitudes)

    def lstsq(self, A, yy):
        if not self.nonneg:
            return np.linalg.lstsq(A, yy, rcond=None)[0]
        res = lsq_linear(A, yy, bounds=([0.0, 0.0, -np.inf, -np.inf], [np.inf] * 4),
                         method="bvls", tol=1e-12)
        return res.x

    def columns(self, p, use):
        sf, df1, df2 = mode_shape(self.Qf, p[0], p[1])
        sc, dc1, dc2 = mode_shape(self.Qc, p[2], p[3])
        A = np.column_stack([self.dS[0] * sf, self.dS[1] * sc, self.sil,
                             np.ones_like(self.sil)])[use]
        D = np.column_stack([self.dS[0] * df1, self.dS[0] * df2,
                             self.dS[1] * dc1, self.dS[1] * dc2])[use]
        return A, D

    def linear_solve(self, p, use):
        A, D = self.columns(p, use)
        yy = self.y[use]
        lin = self.lstsq(A, yy)
        r = yy - A @ lin
        return lin, r, A, D

    def chi2(self, p, use):
        lin, r, A, D = self.linear_solve(p, use)
        return float(r @ r) / self.sig**2, lin, r, A, D

    def full_jacobian(self, lin, A, D):
        """Whitened 8-column Jacobian [A_f, A_c, M, c, lnr_f, s_f, lnr_c, s_c]."""
        Jn = D * np.array([lin[0], lin[0], lin[1], lin[1]])
        return np.column_stack([A, Jn]) / self.sig

    def sigmas(self, p, lin, A, D):
        J = self.full_jacobian(lin, A, D)
        at_lo = p <= self.box_lo + 1e-9; at_hi = p >= self.box_hi - 1e-9
        active = at_lo | at_hi
        lin_active = np.r_[(lin[:2] <= 1e-12) & self.nonneg, False, False]
        keep = np.r_[~lin_active, ~active]
        cov_cond = np.linalg.pinv(A[:, ~lin_active].T @ A[:, ~lin_active]) * self.sig**2
        isil_c = int((~lin_active)[:ISIL].sum())
        try:
            cov_act = np.linalg.inv(J[:, keep].T @ J[:, keep])
            s_act = float(np.sqrt(cov_act[isil_c, isil_c]))
        except np.linalg.LinAlgError:
            s_act = float("nan")
        cov_free = np.linalg.pinv(J.T @ J)
        return dict(sM_marg=s_act, sM_free=float(np.sqrt(cov_free[ISIL, ISIL])),
                    sM_cond=float(np.sqrt(cov_cond[isil_c, isil_c])),
                    at_bound=active.tolist(), amp_at_zero=lin_active[:2].tolist(),
                    cond_full=float(np.linalg.cond(J.T @ J)))

    def lm(self, p0, use):
        """Variable-projection Levenberg-Marquardt inside the box."""
        p = np.clip(np.asarray(p0, float), self.box_lo, self.box_hi)
        chi2, lin, r, A, D = self.chi2(p, use)
        lam = 1e-3
        hist = [dict(it=0, chi2=chi2, M=float(lin[ISIL]))]
        converged, it_conv = False, None
        for it in range(1, MAX_ITER + 1):
            # reduced (Kaufman) Jacobian of the residual wrt p
            Jn = D * np.array([lin[0], lin[0], lin[1], lin[1]])
            Q, _ = np.linalg.qr(A[:, ~(np.r_[(lin[:2] <= 1e-12) & self.nonneg, False, False])])
            Jr = -(Jn - Q @ (Q.T @ Jn))        # d r / d p with linear params re-solved
            g = Jr.T @ r; H = Jr.T @ Jr
            accepted = False
            for _try in range(12):
                try:
                    dp = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-30), g)
                except np.linalg.LinAlgError:
                    dp = -g / (np.diag(H) + lam)
                pn = np.clip(p + dp, self.box_lo, self.box_hi)
                chi2n, linn, rn, An, Dn = self.chi2(pn, use)
                if chi2n < chi2:
                    accepted = True
                    break
                lam *= 10.0
            if not accepted:
                hist.append(dict(it=it, chi2=chi2, M=float(lin[ISIL]), stalled=True))
                sg = self.sigmas(p, lin, A, D)
                sM = sg["sM_marg"] if np.isfinite(sg["sM_marg"]) else sg["sM_cond"]
                converged, it_conv = True, it   # no admissible decrease exists
                break
            dM = float(linn[ISIL] - lin[ISIL]); dchi = (chi2 - chi2n) / max(chi2, 1e-30)
            p, chi2, lin, r, A, D = pn, chi2n, linn, rn, An, Dn
            lam = max(lam / 3.0, 1e-12)
            sg = self.sigmas(p, lin, A, D)
            sM = sg["sM_marg"] if np.isfinite(sg["sM_marg"]) else sg["sM_cond"]
            hist.append(dict(it=it, chi2=chi2, M=float(lin[ISIL]), dM_over_sM=abs(dM) / sM,
                             dchi2_rel=dchi))
            if abs(dM) < TOL_M * sM and dchi < TOL_CHI2:
                converged, it_conv = True, it
                break
        sg = self.sigmas(p, lin, A, D)
        return dict(p=p, lin=lin, chi2=chi2, n_iter=hist[-1]["it"], converged=converged,
                    it_conv=it_conv, hist=hist, last_dM_over_sM=hist[-1].get("dM_over_sM", 0.0),
                    **sg)

    def fit(self, p0):
        use = self.ok.copy()
        passes = 0
        for passes in range(1, 4):
            res = self.lm(p0 if passes == 1 else res["p"], use)
            # model on all bins for clipping
            sf = mode_shape(self.Qf, res["p"][0], res["p"][1])[0]
            sc = mode_shape(self.Qc, res["p"][2], res["p"][3])[0]
            model = (res["lin"][0] * self.dS[0] * sf + res["lin"][1] * self.dS[1] * sc
                     + res["lin"][2] * self.sil + res["lin"][3])
            new = self.ok & (np.abs(self.y - model) < 4 * self.sig)
            if new.sum() == use.sum():
                break
            use = new
        res["nbin"] = int(use.sum()); res["n_passes"] = passes
        res["chi2red"] = res["chi2"] / (use.sum() - 8)
        return res

    def grid_start(self):
        g = [np.linspace(lo, hi, n) for lo, hi, n in zip(self.box_lo, self.box_hi, GRID_N)]
        use = self.ok
        sfs = {(a, b): mode_shape(self.Qf, a, b)[0] for a in g[0] for b in g[1]}
        scs = {(c, d): mode_shape(self.Qc, c, d)[0] for c in g[2] for d in g[3]}
        best = (np.inf, None)
        for (a, b), sf in sfs.items():
            for (c, d), sc in scs.items():
                A = np.column_stack([self.dS[0] * sf, self.dS[1] * sc, self.sil,
                                     np.ones_like(self.sil)])[use]
                lin = self.lstsq(A, self.y[use])
                r = self.y[use] - A @ lin
                c2 = float(r @ r)
                if c2 < best[0]:
                    best = (c2, np.array([a, b, c, d]))
        return best[1], best[0] / self.sig**2


# ---------------------------------------------------------------------------
_G = {}


def _init_worker(qtab_path, nu, sil_basis, dS_tables, box):
    d = np.load(qtab_path)
    _G["Q"] = {m: d[m] for m in MEMBERS}
    _G["nu"] = nu; _G["sil"] = sil_basis; _G["dS"] = dS_tables; _G["box"] = box[0]
    _G["nonneg"] = box[1]


def fit_occ_worker(args):
    occ, h, tau, h_ref = args
    nu = _G["nu"]
    full_rng = (nu >= F.NU_LO) & (nu <= F.NU_HI)
    ok = np.isfinite(tau) & full_rng
    y = np.where(ok, tau, 0.0)
    sil_b = _G["sil"]
    sil = np.array([np.interp(h, F.H_GRID_KM, sil_b[:, j]) - np.interp(h_ref, F.H_GRID_KM, sil_b[:, j])
                    for j in range(sil_b.shape[1])])
    out = dict(occ=occ, h=h, h_ref=h_ref, nbin_ok=int(ok.sum()))
    for member in MEMBERS:
        S = _G["dS"][member]            # (2, n_h): slant 525-nm share per mode
        dS = tuple(float(np.interp(h, F.H_GRID_KM, S[i]) - np.interp(h_ref, F.H_GRID_KM, S[i]))
                   for i in range(2))
        Q = _G["Q"][member]
        of = OccFit(y, ok, sil, dS, Q, Q, F.SIG_BIN, _G["box"], nonneg=_G["nonneg"])
        fits = {}
        for sname, comps in STARTS.items():
            p0 = np.array([np.log(comps[0][0]), comps[0][1], np.log(comps[1][0]), comps[1][1]])
            fits[sname] = of.fit(p0)
        pg, chi2g = of.grid_start()
        fits["grid"] = of.fit(pg)
        fits["grid"]["grid_p0"] = pg.tolist(); fits["grid"]["grid_chi2_start"] = chi2g
        best = min(fits, key=lambda k: fits[k]["chi2"])
        rec = {}
        for sname, r in fits.items():
            rec[sname] = dict(M=float(r["lin"][ISIL]), Af=float(r["lin"][0]), Ac=float(r["lin"][1]),
                              const=float(r["lin"][3]), r_f_nm=float(np.exp(r["p"][0])),
                              s_f=float(r["p"][1]), r_c_nm=float(np.exp(r["p"][2])), s_c=float(r["p"][3]),
                              chi2=r["chi2"], chi2red=r["chi2red"], nbin=r["nbin"], n_passes=r["n_passes"],
                              n_iter=r["n_iter"], converged=bool(r["converged"]), it_conv=r["it_conv"],
                              last_dM_over_sM=float(r["last_dM_over_sM"]),
                              sM_marg=r["sM_marg"], sM_free=r["sM_free"], sM_cond=r["sM_cond"],
                              at_bound=r["at_bound"], amp_at_zero=r["amp_at_zero"], cond_full=r["cond_full"],
                              M_hist=[hh["M"] for hh in r["hist"]], chi2_hist=[hh["chi2"] for hh in r["hist"]])
        rec["best"] = best
        out[member] = rec
    return out


# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", choices=list(BOXES), default="admissible",
                    help="PSD box; non-default boxes write *_<box>.json/.csv/.txt")
    ap.add_argument("--nonneg", action="store_true",
                    help="also constrain the mode amplitudes A_f, A_c >= 0 (suffix _nonneg)")
    args = ap.parse_args()
    box = BOXES[args.box]
    global OUT_JSON, OUT_CSV, LOG
    suffix = ("" if args.box == "admissible" else f"_{args.box}") + ("_nonneg" if args.nonneg else "")
    if suffix:
        OUT_JSON = OUT_JSON.with_name(OUT_JSON.stem + suffix + ".json")
        OUT_CSV = OUT_CSV.with_name(OUT_CSV.stem + suffix + ".csv")
        LOG = LOG.with_name(LOG.stem + suffix + ".txt")
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log(f"amplitudes {'constrained >= 0' if args.nonneg else 'free in sign'}; PSD box '{args.box}': r_f {np.exp(box[0][0]):.0f}-{np.exp(box[1][0]):.0f} nm, sigma_f {box[0][1]}-{box[1][1]}; "
        f"r_c {np.exp(box[0][2]):.0f}-{np.exp(box[1][2]):.0f} nm, sigma_c {box[0][3]}-{box[1][3]}")
    nu = F.reference_grid()
    wl_m = 1.0e-2 / nu
    bases = F.build_bases(nu)
    # Q_ext tables per member optics (cached)
    if not QTAB.exists():
        tabs = {}
        for m in MEMBERS:
            t1 = time.time()
            tabs[m] = build_qtable(m, wl_m)
            log(f"Q_ext table {m}: {tabs[m].shape}, {time.time() - t1:.0f} s")
        np.savez(QTAB, nu=nu, r_grid=R_GRID, **tabs)
    d = np.load(QTAB)
    assert np.array_equal(d["nu"], nu)
    # exactness check of the separable engine column against the cached nominal basis
    a525 = glossac_2025N_quiet_profile()
    C = np.array(tangent_to_slant_paths(F.H_GRID_KM * 1e3, ALT))
    dS_tables = {}
    for m, comps in MEMBERS.items():
        dS_tables[m] = np.array([C @ (comps[i][2] * a525) for i in range(2)])
    sf = mode_shape(d["LM65T223"], np.log(60.0), 1.6)[0]
    sc = mode_shape(d["LM65T223"], np.log(1500.0), 1.8)[0]
    chk_f = np.max(np.abs(np.outer(dS_tables["LM65T223"][0], sf) - bases["f"])) / np.abs(bases["f"]).max()
    chk_c = np.max(np.abs(np.outer(dS_tables["LM65T223"][1], sc) - bases["c"])) / np.abs(bases["c"]).max()
    log(f"separable engine column vs cached nominal basis: max rel diff f {chk_f:.2e}, c {chk_c:.2e}")
    assert chk_f < 1e-10 and chk_c < 1e-10
    # finite-difference check of the analytic PSD derivatives
    p = np.array([np.log(60.0), 1.6]); eps = 1e-5
    s, d1, d2 = mode_shape(d["LM65T223"], *p)
    fd1 = (mode_shape(d["LM65T223"], p[0] + eps, p[1])[0] - mode_shape(d["LM65T223"], p[0] - eps, p[1])[0]) / (2 * eps)
    fd2 = (mode_shape(d["LM65T223"], p[0], p[1] + eps)[0] - mode_shape(d["LM65T223"], p[0], p[1] - eps)[0]) / (2 * eps)
    log(f"analytic vs FD derivatives: max rel diff d/dlnr {np.max(np.abs(d1 - fd1)) / np.max(np.abs(d1)):.2e}, "
        f"d/dsigma {np.max(np.abs(d2 - fd2)) / np.max(np.abs(d2)):.2e}")

    # the 101 occultations, selected exactly as in round 70 (and the nominal linear fit)
    thr = json.load(open(F._THR_JSON))
    win_sig_od = np.array(thr["window_floors_m1"]) * F.OD_PER_M1
    ens = [str(o) for o in np.load(F.SIDE / "matrix_19_22.npz", allow_pickle=True)["occs"]]
    full_rng = (nu >= F.NU_LO) & (nu <= F.NU_HI)
    occs, nominal = [], {}
    for occ in sorted(ens):
        dd = F.DATA / "residual" / occ
        if not dd.is_dir():
            continue
        got = F.occultation_data(occ, nu)
        if got is None:
            continue
        h, tau, h_ref = got
        if (np.isfinite(tau) & full_rng).sum() < 100:
            continue
        occs.append((occ, h, tau, h_ref))
        r = F.fit_occ(h, tau, h_ref, bases, nu, win_sig_od)
        nominal[occ] = dict(M=r["full"]["M"], sM=r["full"]["sM"], chi2red=r["full"]["chi2"],
                            nbin=r["full"]["nbin"], Af=r["full"]["Af"], Ac=r["full"]["Ac"])
    assert len(occs) == 101, len(occs)
    log(f"{len(occs)} occultations; nominal linear full-band fit reproduced: median M "
        f"{np.median([v['M'] for v in nominal.values()]):+.5f}, robust {robust_sd([v['M'] for v in nominal.values()]):.5f}")
    r70 = json.load(open(R70_JSON))["rows"]["full"]
    assert abs(np.median([v["M"] for v in nominal.values()]) - r70["median_M"]) < 1e-9

    t1 = time.time()
    with ProcessPoolExecutor(max_workers=6, initializer=_init_worker,
                             initargs=(str(QTAB), nu, bases["sil"], dS_tables, (box, args.nonneg))) as ex:
        results = list(ex.map(fit_occ_worker, occs, chunksize=4))
    log(f"nonlinear refits: {time.time() - t1:.0f} s for {len(results)} occultations x 2 members x 3 starts")

    # ---- summaries ---------------------------------------------------------
    M0 = np.array([nominal[r["occ"]]["M"] for r in results])
    sM0 = np.array([nominal[r["occ"]]["sM"] for r in results])
    summary = dict(
        script="reproduce/ace_nulltest_nonlinear_refit.py",
        method=__doc__.split("Outputs")[0],
        n_occ=len(results), max_iter=MAX_ITER, tol_M_over_sM=TOL_M, tol_chi2_rel=TOL_CHI2,
        box_name=args.box, nonneg_amplitudes=bool(args.nonneg), box=dict(r_f_nm=[float(np.exp(box[0][0])), float(np.exp(box[1][0]))],
                                    sigma_f=[float(box[0][1]), float(box[1][1])],
                                    r_c_nm=[float(np.exp(box[0][2])), float(np.exp(box[1][2]))],
                                    sigma_c=[float(box[0][3]), float(box[1][3])]),
        admissible=ADMISSIBLE, grid_n=GRID_N,
        checks=dict(separable_vs_cached_f=chk_f, separable_vs_cached_c=chk_c),
        nominal_linear=dict(median_M=float(np.median(M0)), robust_scatter=robust_sd(M0),
                            sd=float(M0.std(ddof=1)), median_formal=float(np.median(sM0)),
                            Af_median=float(np.median([nominal[r["occ"]]["Af"] for r in results])),
                            Ac_median=float(np.median([nominal[r["occ"]]["Ac"] for r in results])),
                            frac_Af_neg=float(np.mean([nominal[r["occ"]]["Af"] < 0 for r in results])),
                            frac_Ac_neg=float(np.mean([nominal[r["occ"]]["Ac"] < 0 for r in results])),
                            round70_rows_full=r70),
        members={})

    def arr(member, start, key):
        return np.array([r[member][start if start != "best" else r[member]["best"]][key]
                         for r in results], float)

    for member in MEMBERS:
        ms = {}
        for start in ("best", "quiet_psd", "postHT_psd", "grid"):
            M = arr(member, start, "M"); sMm = arr(member, start, "sM_marg")
            sMf = arr(member, start, "sM_free"); sMc = arr(member, start, "sM_cond")
            chi = arr(member, start, "chi2red"); nit = arr(member, start, "n_iter")
            conv = arr(member, start, "converged"); itc = np.array(
                [r[member][start if start != "best" else r[member]["best"]]["it_conv"] or MAX_ITER + 1
                 for r in results], float)
            ab = np.array([r[member][start if start != "best" else r[member]["best"]]["at_bound"]
                           for r in results], bool)
            Af_ = arr(member, start, "Af"); Ac_ = arr(member, start, "Ac")
            rf = arr(member, start, "r_f_nm"); sf_ = arr(member, start, "s_f")
            rc = arr(member, start, "r_c_nm"); sc_ = arr(member, start, "s_c")
            dM = M - M0
            ms[start] = dict(
                median_M=float(np.median(M)), mean_M=float(M.mean()), robust_scatter=robust_sd(M),
                sd=float(M.std(ddof=1)), mmin_3sig_empirical=3 * robust_sd(M),
                median_sM_marginal=float(np.nanmedian(sMm)), median_sM_free=float(np.median(sMf)),
                median_sM_conditional=float(np.median(sMc)),
                mmin_3sig_formal_marginal=3 * float(np.nanmedian(sMm)),
                median_chi2red=float(np.median(chi)),
                median_n_iter=float(np.median(nit)), max_n_iter=int(nit.max()),
                frac_converged=float(conv.mean()), frac_converged_within_10=float((itc <= 10).mean()),
                median_it_conv=float(np.median(itc)),
                frac_any_bound=float(ab.any(axis=1).mean()),
                frac_bound_per_param=dict(zip(("lnr_f", "s_f", "lnr_c", "s_c"), ab.mean(axis=0).tolist())),
                amplitudes=dict(Af_median=float(np.median(Af_)), Ac_median=float(np.median(Ac_)),
                                frac_Af_neg_or_zero=float((Af_ <= 1e-12).mean()),
                                frac_Ac_neg_or_zero=float((Ac_ <= 1e-12).mean())),
                psd_median=dict(r_f_nm=float(np.median(rf)), s_f=float(np.median(sf_)),
                                r_c_nm=float(np.median(rc)), s_c=float(np.median(sc_))),
                psd_iqr=dict(r_f_nm=np.percentile(rf, [25, 75]).tolist(), s_f=np.percentile(sf_, [25, 75]).tolist(),
                             r_c_nm=np.percentile(rc, [25, 75]).tolist(), s_c=np.percentile(sc_, [25, 75]).tolist()),
                vs_nominal=dict(median_dM=float(np.median(dM)), robust_scatter_dM=robust_sd(dM),
                                sd_dM=float(dM.std(ddof=1)),
                                median_abs_dM_over_sM0=float(np.median(np.abs(dM) / sM0)),
                                max_abs_dM_over_sM0=float(np.max(np.abs(dM) / sM0)),
                                frac_abs_dM_gt_1sM0=float((np.abs(dM) > sM0).mean()),
                                frac_abs_dM_gt_0p5sM0=float((np.abs(dM) > 0.5 * sM0).mean()),
                                corr_with_nominal=float(np.corrcoef(M, M0)[0, 1])))
        # start dependence
        Mq, Mp, Mg = (arr(member, s, "M") for s in ("quiet_psd", "postHT_psd", "grid"))
        cq, cp, cg = (arr(member, s, "chi2") for s in ("quiet_psd", "postHT_psd", "grid"))
        sMb = arr(member, "best", "sM_marg")
        sMb = np.where(np.isfinite(sMb), sMb, arr(member, "best", "sM_cond"))
        best_names = [r[member]["best"] for r in results]
        spread = np.max(np.c_[Mq, Mp, Mg], axis=1) - np.min(np.c_[Mq, Mp, Mg], axis=1)
        chi_spread = np.max(np.c_[cq, cp, cg], axis=1) - np.min(np.c_[cq, cp, cg], axis=1)
        ms["start_dependence"] = dict(
            best_start_counts={k: best_names.count(k) for k in ("quiet_psd", "postHT_psd", "grid")},
            median_M_spread_over_sM=float(np.median(spread / sMb)),
            max_M_spread_over_sM=float(np.max(spread / sMb)),
            frac_spread_gt_0p1sM=float((spread > 0.1 * sMb).mean()),
            frac_spread_gt_1sM=float((spread > sMb).mean()),
            median_chi2_spread=float(np.median(chi_spread)), max_chi2_spread=float(np.max(chi_spread)),
            frac_chi2_spread_gt_1=float((chi_spread > 1.0).mean()),
            median_abs_Mq_minus_Mp_over_sM=float(np.median(np.abs(Mq - Mp) / sMb)),
            max_abs_Mq_minus_Mp_over_sM=float(np.max(np.abs(Mq - Mp) / sMb)))
        summary["members"][member] = ms
        b = ms["best"]; sd_ = ms["start_dependence"]
        log(f"\n{member}: best-of-starts: median M {b['median_M']:+.5f} (nominal {summary['nominal_linear']['median_M']:+.5f}), "
            f"robust {b['robust_scatter']:.5f} (nominal {summary['nominal_linear']['robust_scatter']:.5f}), SD {b['sd']:.5f}, "
            f"3sig emp {b['mmin_3sig_empirical']:.4f}, formal marg {b['median_sM_marginal']:.5f} free {b['median_sM_free']:.5f} "
            f"cond {b['median_sM_conditional']:.5f}; chi2red {b['median_chi2red']:.3f}; iter median {b['median_n_iter']:.0f} max {b['max_n_iter']}, "
            f"converged {100 * b['frac_converged']:.0f} % (within 10: {100 * b['frac_converged_within_10']:.0f} %); "
            f"at a bound {100 * b['frac_any_bound']:.0f} % {b['frac_bound_per_param']}; amplitudes {b['amplitudes']}; "
            f"dM vs nominal: median {b['vs_nominal']['median_dM']:+.5f}, robust {b['vs_nominal']['robust_scatter_dM']:.5f}, "
            f"median |dM|/sM0 {b['vs_nominal']['median_abs_dM_over_sM0']:.3f}, max {b['vs_nominal']['max_abs_dM_over_sM0']:.3f}, "
            f">1 sM0 in {100 * b['vs_nominal']['frac_abs_dM_gt_1sM0']:.0f} %; PSD medians {b['psd_median']}; "
            f"start dependence: counts {sd_['best_start_counts']}, median spread/sM {sd_['median_M_spread_over_sM']:.3f}, "
            f"max {sd_['max_M_spread_over_sM']:.3f}, spread > 1 sM in {100 * sd_['frac_spread_gt_1sM']:.0f} %, "
            f"chi2 spread median {sd_['median_chi2_spread']:.2f} max {sd_['max_chi2_spread']:.2f}")
        for start in ("quiet_psd", "postHT_psd", "grid"):
            s = ms[start]
            log(f"   start {start:10s}: median M {s['median_M']:+.5f} robust {s['robust_scatter']:.5f} SD {s['sd']:.5f} "
                f"chi2red {s['median_chi2red']:.3f} conv {100 * s['frac_converged']:.0f} % bound {100 * s['frac_any_bound']:.0f} %")
    # epochs and drift (best of starts), as in round 70 part 5
    from datetime import datetime
    dates = {}
    with open(F.DATA / "occultationlist.csv") as f:
        for row in csv.DictReader(f):
            dates[row["occultation_name"]] = datetime.strptime(row["occultation_datetime"][:19], "%Y-%m-%d %H:%M:%S")
    tdec = np.array([dates[r["occ"]].year + (dates[r["occ"]].timetuple().tm_yday - 0.5) / 365.25 for r in results])
    early = tdec < 2016.0

    def drift(M):
        A = np.column_stack([np.ones_like(tdec), (tdec - tdec.mean()) / 10.0])
        th, res, *_ = np.linalg.lstsq(A, M, rcond=None)
        r = M - A @ th
        cov = np.linalg.inv(A.T @ A) * (r @ r) / (len(M) - 2)
        return dict(slope_tg_per_decade=float(th[1]), slope_se=float(np.sqrt(cov[1, 1])),
                    median_2004_2015=float(np.median(M[early])), median_2016_2024=float(np.median(M[~early])),
                    robust_2004_2015=robust_sd(M[early]), robust_2016_2024=robust_sd(M[~early]),
                    n_early=int(early.sum()), n_late=int((~early).sum()))
    summary["drift"] = dict(nominal_linear=drift(M0),
                            **{m: drift(arr(m, "best", "M")) for m in MEMBERS})
    for k, v in summary["drift"].items():
        log(f"drift {k:14s}: {v['slope_tg_per_decade']:+.4f} +- {v['slope_se']:.4f} Tg/decade; medians {v['median_2004_2015']:+.4f} (2004-2015, n={v['n_early']}) / "
            f"{v['median_2016_2024']:+.4f} (2016-2024, n={v['n_late']}); robust {v['robust_2004_2015']:.4f} / {v['robust_2016_2024']:.4f}")
    for r in results:
        r["date"] = dates[r["occ"]].isoformat()
    # member comparison (best of starts)
    Mq = arr("LM65T223", "best", "M"); Mp = arr("LM72T213", "best", "M")
    cq = arr("LM65T223", "best", "chi2"); cp = arr("LM72T213", "best", "chi2")
    summary["member_comparison"] = dict(
        median_dM_postHT_minus_quiet=float(np.median(Mp - Mq)), robust_dM=robust_sd(Mp - Mq),
        frac_postHT_lower_chi2=float((cp < cq).mean()), median_dchi2=float(np.median(cp - cq)))
    log(f"\nmember comparison (best of starts): median M(postHT optics) - M(quiet optics) "
        f"{summary['member_comparison']['median_dM_postHT_minus_quiet']:+.5f}, robust {summary['member_comparison']['robust_dM']:.5f}; "
        f"post-HT optics lower chi2 in {100 * summary['member_comparison']['frac_postHT_lower_chi2']:.0f} %")
    summary["runtime_s"] = time.time() - t0

    # ---- per-occultation csv + json --------------------------------------
    fields = ["occ", "h", "h_ref", "M_nominal", "sM_nominal", "chi2red_nominal"]
    for member in MEMBERS:
        for start in ("best", "quiet_psd", "postHT_psd", "grid"):
            for k in ("M", "sM_marg", "sM_cond", "chi2red", "r_f_nm", "s_f", "r_c_nm", "s_c", "n_iter",
                      "converged", "it_conv", "Af", "Ac"):
                fields.append(f"{k}_{member}_{start}")
        fields.append(f"best_start_{member}")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = dict(occ=r["occ"], h=r["h"], h_ref=r["h_ref"], M_nominal=nominal[r["occ"]]["M"],
                       sM_nominal=nominal[r["occ"]]["sM"], chi2red_nominal=nominal[r["occ"]]["chi2red"])
            for member in MEMBERS:
                for start in ("best", "quiet_psd", "postHT_psd", "grid"):
                    src = r[member][start if start != "best" else r[member]["best"]]
                    for k in ("M", "sM_marg", "sM_cond", "chi2red", "r_f_nm", "s_f", "r_c_nm", "s_c", "n_iter",
                              "converged", "it_conv", "Af", "Ac"):
                        row[f"{k}_{member}_{start}"] = src[k]
                row[f"best_start_{member}"] = r[member]["best"]
            w.writerow(row)
    summary["per_occultation"] = results
    summary["per_occultation_nominal"] = nominal
    json.dump(summary, open(OUT_JSON, "w"), indent=1, default=float)
    log(f"archived -> {OUT_JSON}\n            {OUT_CSV}\nruntime {summary['runtime_s']:.0f} s")
    LOG.write_text("\n".join(_LOG_LINES) + "\n")


if __name__ == "__main__":
    main()
