"""dda_uniaxial.py -- round 75 (paper 1; public port, referee round 6 item 6, left open in round 74): an
anisotropic solver for the uniaxial materials (calcite, dolomite, alumina), so that the two scalar
orientation-average prescriptions of the paper (the effective index of the mean polarizability,
convention b, and the cross-section average (2 sigma_o + sigma_e)/3, convention c) are tested
against a solution of the full anisotropic problem on the reference size distribution instead of
against each other.

Solver: the discrete dipole approximation (Draine 1988; Draine and Flatau 1994) with a diagonal
dielectric tensor per dipole, diag(eps_o, eps_o, eps_e) in the crystal frame rotated to the lab
frame (the uniaxial axis c at polar angle theta from the propagation direction and azimuth phi
from the polarization), the lattice dispersion relation polarizability applied per principal axis,
about 1000 dipoles per sphere (a cubic lattice scaled so that the dipole volume equals the sphere
volume), dense LU solve.  The extinction is C_ext = 4 pi k sum Im(E_inc* . P) and the absorption
C_abs = 4 pi k sum [Im(P . alpha^-1* P*) - (2/3) k^3 |P|^2].  The orientation average over the axis
direction for unpolarized light is a Gauss-Legendre quadrature over cos(theta) in [0, 1] (4 nodes)
and phi in [0, pi/2] (3 nodes), the octant that the mirror symmetries of the problem map onto the
whole sphere.  The DDA's own discretization error is measured on the same lattice with an isotropic
index against the Mie solution.  Near the resonances (eps close to -2 for the extraordinary ray of calcite at
11.4 um) a 1000-dipole lattice misplaces the electrostatic surface mode and its ABSOLUTE cross
sections are off by tens of per cent, whereas its RELATIVE size dependence is not: the exact
anisotropic result is therefore built as C_exact(r) = C_R(r0) x C_DDA,aniso(r) / C_DDA,aniso(r0),
the exact orientation-averaged Rayleigh cross section at r0 = 20 nm, (2 C_o + C_e)/3 from Mie,
times the DDA's own finite-size factor, and the method is validated on the same lattice for the
three isotropic indices (o, e, b), for which g_DDA(r) = C_DDA(r)/C_DDA(r0) is compared with the Mie
factor C_Mie(r)/C_Mie(r0); the largest deviation over o, e and b is reported as the method's error.

Scope: 7 elements around each material's band peak (the elements that carry the ranking), 9 radii
from 100 to 700 nm (the reference distribution, r_med = 268 nm, sigma = 1.31, carries 0.3 % of its
volume beyond 700 nm), the ratio interpolated in ln r over the PSD, and the per-Tg extinction under
the exact average against conventions b and c at each element; the thresholds of Table 8 under the
adopted model re-evaluated with the material column corrected at those elements.

Run from the repo root (about 15-30 min in 4 processes):  python reproduce/dda_uniaxial.py
Writes outputs/dda_uniaxial.json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))

OUT = ROOT / "outputs/dda_uniaxial.json"
PEAK_UM = {"calcite": 11.4, "dolomite": 11.3, "alumina": 12.9}
R_NM = np.array([100.0, 140.0, 190.0, 250.0, 320.0, 400.0, 500.0, 600.0, 700.0])
R0_NM = 20.0
N_TARGET = 1000
B1, B2, B3 = -1.891531, 0.1648469, -1.7700004
RMED_NM, SIG = 268.0, 1.31


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- DDA
def lattice(n_target):
    """cubic lattice points inside a sphere, about n_target of them; returns unit-spacing coordinates (N, 3)."""
    a = (3 * n_target / (4 * np.pi)) ** (1 / 3)
    n = int(np.ceil(a)) + 1
    g = np.arange(-n, n) + 0.5
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return pts[np.sum(pts ** 2, axis=1) <= a ** 2]


def offdiag(pts, d, k):
    """the interaction blocks A_jk (j != k) for a lattice with spacing d [m] and wavenumber k [1/m]: (3N, 3N) complex."""
    r = (pts[:, None, :] - pts[None, :, :]) * d                       # (N, N, 3)
    dist = np.linalg.norm(r, axis=2)
    np.fill_diagonal(dist, 1.0)
    rh = r / dist[:, :, None]
    rr = rh[:, :, :, None] * rh[:, :, None, :]                        # (N, N, 3, 3)
    I3 = np.eye(3)[None, None]
    ph = np.exp(1j * k * dist) / dist
    A = ph[:, :, None, None] * (k ** 2 * (rr - I3) + ((1j * k * dist - 1) / dist ** 2)[:, :, None, None] * (3 * rr - I3))
    N = len(pts)
    idx = np.arange(N)
    A[idx, idx] = 0.0
    return A.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)


def alpha_ldr(m, d, k, S=0.0):
    """lattice-dispersion-relation polarizability for index m (scalar) on spacing d."""
    eps = m ** 2
    a_cm = 3 * d ** 3 / (4 * np.pi) * (eps - 1) / (eps + 2)
    kd = k * d
    return a_cm / (1 + (a_cm / d ** 3) * ((B1 + eps * B2 + eps * B3 * S) * kd ** 2 - (2.0 / 3.0) * 1j * kd ** 3))


def solve_dda(A_off, N, alpha_inv_blocks, E0=np.array([1.0, 0.0, 0.0])):
    """P for a plane wave along z polarized along x (phase e^{ikz} applied by the caller through E_inc)."""
    A = A_off.copy()
    for j in range(N):
        A[3 * j:3 * j + 3, 3 * j:3 * j + 3] += alpha_inv_blocks[j]
    return A


def cross_sections(pts, d, k, m_o, m_e, axis, A_off):
    """C_ext, C_abs [m^2] for the uniaxial sphere with axis `axis` (unit vector) in the lab frame, E along x, k along z."""
    N = len(pts)
    a_o, a_e = alpha_ldr(m_o, d, k), alpha_ldr(m_e, d, k)
    c = np.asarray(axis, float); c = c / np.linalg.norm(c)
    P_par = np.outer(c, c); P_perp = np.eye(3) - P_par
    alpha = a_o * P_perp + a_e * P_par                                 # the same tensor at every dipole
    ainv = np.linalg.inv(alpha)
    A = A_off.copy()
    for j in range(N):
        A[3 * j:3 * j + 3, 3 * j:3 * j + 3] += ainv
    z = pts[:, 2] * d
    E_inc = np.zeros((N, 3), complex); E_inc[:, 0] = np.exp(1j * k * z)
    P = np.linalg.solve(A, E_inc.ravel()).reshape(N, 3)
    c_ext = 4 * np.pi * k * np.sum(np.imag(np.sum(np.conj(E_inc) * P, axis=1)))
    c_abs = 4 * np.pi * k * np.sum(np.imag(np.einsum("ni,ij,nj->n", P, np.conj(ainv), np.conj(P))) - (2.0 / 3.0) * k ** 3 * np.sum(np.abs(P) ** 2, axis=1))
    return float(c_ext), float(c_abs)


def iso_cross_sections(pts, d, k, m, A_off):
    return cross_sections(pts, d, k, m, m, (0.0, 0.0, 1.0), A_off)


def quad_axes():
    """Gauss-Legendre nodes on cos(theta) in [0, 1] and phi in [0, pi/2]; weights normalized to 1."""
    xc, wc = np.polynomial.legendre.leggauss(4); xp, wp = np.polynomial.legendre.leggauss(3)
    ct = 0.5 * (xc + 1); wct = 0.5 * wc
    ph = 0.25 * np.pi * (xp + 1); wph = 0.25 * np.pi * wp / (0.5 * np.pi)
    axes, w = [], []
    for c_, wc_ in zip(ct, wct):
        st = np.sqrt(1 - c_ ** 2)
        for p_, wp_ in zip(ph, wph):
            axes.append((st * np.cos(p_), st * np.sin(p_), c_)); w.append(wc_ * wp_)
    return np.array(axes), np.array(w) / np.sum(w)


def job(args):
    """one (material, wavelength): per-radius DDA (anisotropic average and the three isotropic indices on the same lattice),
    anchored at r0 = 20 nm, and Mie (o, e, b)."""
    mat, lam_um = args
    from materials_uniaxial import RayTables, MAT_CLS, cm_average
    from saimon.mie import mie_efficiencies
    t0 = time.time()
    tabs = RayTables(MAT_CLS[mat])
    lam = lam_um * 1e-6; k = 2 * np.pi / lam
    m_o = complex(tabs.m("o", np.array([lam]))[0]); m_e = complex(tabs.m("e", np.array([lam]))[0])
    m_b = complex(cm_average(np.array([m_o]), np.array([m_e]))[0][0])
    pts = lattice(N_TARGET); N = len(pts)
    axes, w = quad_axes()

    def dda_at(a):
        d = a * (4 * np.pi / (3 * N)) ** (1 / 3)
        A_off = offdiag(pts, d, k)
        ce_an = ca_an = 0.0
        for ax, wi in zip(axes, w):
            ce, ca = cross_sections(pts, d, k, m_o, m_e, ax, A_off)
            ce_an += wi * ce; ca_an += wi * ca
        iso = {X: iso_cross_sections(pts, d, k, m, A_off) for X, m in (("o", m_o), ("e", m_e), ("b", m_b))}
        return dict(an_ext=ce_an, an_abs=ca_an, **{f"{X}_ext": v[0] for X, v in iso.items()}, **{f"{X}_abs": v[1] for X, v in iso.items()})

    def mie_at(a):
        x = 2 * np.pi * a / lam; area = np.pi * a ** 2
        out = {}
        for X, m in (("o", m_o), ("e", m_e), ("b", m_b)):
            qe, qs, qa = mie_efficiencies(np.array([x]), m)
            out[f"{X}_ext"] = float(qe[0] * area); out[f"{X}_abs"] = float(qa[0] * area); out[f"{X}_sca"] = float(qs[0] * area)
        return out
    a0 = R0_NM * 1e-9
    d0, M0 = dda_at(a0), mie_at(a0)
    C_R_ext = (2 * M0["o_ext"] + M0["e_ext"]) / 3; C_R_abs = (2 * M0["o_abs"] + M0["e_abs"]) / 3     # exact Rayleigh anisotropic average at r0
    rows = []
    for r_nm in R_NM:
        a = r_nm * 1e-9
        dd, mm = dda_at(a), mie_at(a)
        g_an_ext = dd["an_ext"] / d0["an_ext"]; g_an_abs = dd["an_abs"] / d0["an_abs"]
        val = {X: (dd[f"{X}_ext"] / d0[f"{X}_ext"]) / (mm[f"{X}_ext"] / M0[f"{X}_ext"]) - 1 for X in ("o", "e", "b")}
        val_abs = {X: (dd[f"{X}_abs"] / d0[f"{X}_abs"]) / (mm[f"{X}_abs"] / M0[f"{X}_abs"]) - 1 for X in ("o", "e", "b")}
        ext_exact = C_R_ext * g_an_ext; abs_exact = C_R_abs * g_an_abs
        ext_c = (2 * mm["o_ext"] + mm["e_ext"]) / 3
        rows.append(dict(r_nm=float(r_nm), x=float(2 * np.pi * a / lam), ext_exact=ext_exact, abs_exact=abs_exact, ext_b=mm["b_ext"], ext_c=ext_c, abs_b=mm["b_abs"],
                         ratio_exact_over_b=ext_exact / mm["b_ext"], ratio_exact_over_c=ext_exact / ext_c, ratio_abs_exact_over_b=abs_exact / mm["b_abs"],
                         g_dda_aniso_ext=g_an_ext, validation_g_error_ext=val, validation_g_error_abs=val_abs,
                         dda_abs_error_ext={X: dd[f"{X}_ext"] / mm[f"{X}_ext"] - 1 for X in ("o", "e", "b")},
                         dda_aniso_raw_ext=dd["an_ext"], dda_aniso_raw_abs=dd["an_abs"]))
    out = dict(material=mat, lam_um=float(lam_um), m_o=[m_o.real, m_o.imag], m_e=[m_e.real, m_e.imag], m_b=[m_b.real, m_b.imag], n_dipoles=int(N),
               r0_nm=R0_NM, rayleigh_anchor=dict(ext=C_R_ext, abs=C_R_abs, dda_aniso_ext_r0=d0["an_ext"], dda_aniso_error_r0=d0["an_ext"] / C_R_ext - 1,
                                                  dda_b_error_r0=d0["b_ext"] / M0["b_ext"] - 1, dda_e_error_r0=d0["e_ext"] / M0["e_ext"] - 1),
               rows=rows, seconds=time.time() - t0)
    log(f"  {mat} {lam_um:.2f} um: m_o {m_o:.3f} m_e {m_e:.3f}; exact/b " + " ".join(f"{r['ratio_exact_over_b']:.3f}" for r in rows)
        + "; validation |g err| max " + " ".join(f"{max(abs(v) for v in r['validation_g_error_ext'].values()):.3f}" for r in rows) + f"; {out['seconds']:.0f} s")
    return out


def main():
    t0 = time.time()
    jobs = [(mat, round(PEAK_UM[mat] + 0.1 * k, 2)) for mat in PEAK_UM for k in range(-3, 4)]
    jobs = [(m, l) for m, l in jobs if l <= 13.25]
    if "--resummarize" in sys.argv:                      # recompute the PSD integration and the thresholds from the archived DDA rows
        src = OUT if OUT.exists() else OUT.with_suffix(".raw.json")
        res = json.load(open(src))["per_wavelength"]
    else:
        with mp.get_context("spawn").Pool(4) as pool:
            res = pool.map(job, jobs)
    if "--resummarize" not in sys.argv:
        with open(OUT.with_suffix(".raw.json"), "w") as f:
            json.dump(dict(per_wavelength=res), f)
    # PSD integration: C_exact(r) = C_Mie,b(r) x ratio(r), the ratio interpolated in ln r over the fine PSD grid
    from saimon.mie import mie_efficiencies
    from materials_uniaxial import RayTables, MAT_CLS, cm_average
    r_grid = np.logspace(np.log10(1e-8), np.log10(2e-6), 400)
    wpsd = np.exp(-0.5 * (np.log(r_grid / (RMED_NM * 1e-9)) / np.log(SIG)) ** 2) / r_grid; wpsd /= np.trapezoid(wpsd, r_grid)
    summary = {}
    for mat in PEAK_UM:
        summary[mat] = dict(elements_um=[], ext_exact=[], ext_b=[], ext_c=[], ratio_exact_over_b=[], ratio_exact_over_c=[], abs_ratio_exact_over_b=[],
                            validation_error_max=[], validation_error_psd_weighted=[], sca_fraction_exact=[], sca_fraction_b=[], sca_fraction_c=[])
        for r_ in sorted([x for x in res if x["material"] == mat], key=lambda x: x["lam_um"]):
            lam = r_["lam_um"] * 1e-6; x = 2 * np.pi * r_grid / lam
            m_o = complex(*r_["m_o"]); m_e = complex(*r_["m_e"]); m_b = complex(*r_["m_b"])
            qe_b, qs_b, _ = mie_efficiencies(x, m_b); qe_o, qs_o, _ = mie_efficiencies(x, m_o); qe_e, qs_e, _ = mie_efficiencies(x, m_e)
            area = np.pi * r_grid ** 2
            rr = np.array([row["r_nm"] for row in r_["rows"]]) * 1e-9
            rat = np.array([row["ratio_exact_over_b"] for row in r_["rows"]]); rat_a = np.array([row["ratio_abs_exact_over_b"] for row in r_["rows"]])
            ratio_r = np.interp(np.log(r_grid), np.log(rr), rat, left=rat[0], right=rat[-1])
            ratio_ra = np.interp(np.log(r_grid), np.log(rr), rat_a, left=rat_a[0], right=rat_a[-1])
            ext_b = np.trapezoid(qe_b * area * wpsd, r_grid); ext_c = np.trapezoid((2 * qe_o + qe_e) / 3 * area * wpsd, r_grid)
            ext_x = np.trapezoid(qe_b * ratio_r * area * wpsd, r_grid)
            abs_b = np.trapezoid((qe_b - qs_b) * area * wpsd, r_grid); abs_x = np.trapezoid((qe_b - qs_b) * ratio_ra * area * wpsd, r_grid)
            abs_c = np.trapezoid(((2 * (qe_o - qs_o) + (qe_e - qs_e)) / 3) * area * wpsd, r_grid)
            s_ = summary[mat]
            s_["elements_um"].append(r_["lam_um"]); s_["ext_exact"].append(float(ext_x)); s_["ext_b"].append(float(ext_b)); s_["ext_c"].append(float(ext_c))
            s_["ratio_exact_over_b"].append(float(ext_x / ext_b)); s_["ratio_exact_over_c"].append(float(ext_x / ext_c)); s_["abs_ratio_exact_over_b"].append(float(abs_x / abs_b))
            s_["validation_error_max"].append(float(max(max(abs(v) for v in row["validation_g_error_ext"].values()) for row in r_["rows"])))
            # the validation error weighted as the PSD integration weights the radii (extinction x PSD), the error of the integrated ratio
            verr = np.array([max(abs(v) for v in row["validation_g_error_ext"].values()) for row in r_["rows"]])
            verr_r = np.interp(np.log(r_grid), np.log(rr), verr, left=verr[0], right=verr[-1])
            s_["validation_error_psd_weighted"].append(float(np.trapezoid(verr_r * qe_b * area * wpsd, r_grid) / ext_b))
            s_["sca_fraction_exact"].append(float(1 - abs_x / ext_x)); s_["sca_fraction_b"].append(float(1 - abs_b / ext_b)); s_["sca_fraction_c"].append(float(1 - abs_c / ext_c))
        log(f"{mat}: exact/b per element " + " ".join(f"{v:.4f}" for v in summary[mat]["ratio_exact_over_b"]) + "; exact/c " + " ".join(f"{v:.4f}" for v in summary[mat]["ratio_exact_over_c"])
            + "; validation max " + " ".join(f"{v:.3f}" for v in summary[mat]["validation_error_max"]) + "; PSD-weighted " + " ".join(f"{v:.3f}" for v in summary[mat]["validation_error_psd_weighted"]))
    # thresholds under the adopted model with the corrected elements
    import adopted_model_engine as E
    from materials_uniaxial import ConventionRI, MAT_FACTORY
    C, cbt = E.C, E.cbt
    thresholds = {}
    for mat in PEAK_UM:
        lay = MAT_FACTORY[mat](cbt.ALT, 1.0, refractive_index=ConventionRI(RayTables(MAT_CLS[mat]), "b"))
        prof_c = lay.extinction_profile_m1(C.lam_coarse * 1e-6); prof_o = lay.extinction_profile_m1(C.lam_outer * 1e-6); prof_t = lay.extinction_profile_m1(C.WL_TRIPLET_NM * 1e-9)
        tm_b = E.TangentModel(20e3, template=prof_c, template_outer=prof_o, template_trip=prof_t)
        m_b = 3 * E.joint_sigma(tm_b) * tm_b.rho
        # corrected outer profile: scale the 0.1-um element bands around the computed elements by the exact/b ratio (interpolated between elements)
        els = np.array(summary[mat]["elements_um"]); rat = np.array(summary[mat]["ratio_exact_over_b"])
        f_o = np.interp(C.lam_outer, els, rat, left=1.0, right=1.0)
        f_o[(C.lam_outer < els.min() - 0.05) | (C.lam_outer > els.max() + 0.05)] = 1.0
        tm_x = E.TangentModel(20e3, template=prof_c, template_outer=prof_o * f_o[None, :], template_trip=prof_t)
        m_x = 3 * E.joint_sigma(tm_x) * tm_x.rho
        thresholds[mat] = dict(adopted_b=m_b, adopted_exact_corrected=m_x, change_percent=100 * (m_x / m_b - 1), elements_corrected_um=els.tolist())
        log(f"{mat}: threshold (adopted model) convention b {m_b:.4f} Tg -> exact-corrected elements {m_x:.4f} Tg ({thresholds[mat]['change_percent']:+.2f} %)")
    out = dict(script="reproduce/dda_uniaxial.py", round=75, n_dipoles=res[0]["n_dipoles"], radii_nm=R_NM.tolist(), psd=dict(rmed_nm=RMED_NM, sigma=SIG),
               quadrature="Gauss-Legendre 4 x 3 on cos(theta) in [0,1], phi in [0, pi/2]", method="C_exact(r) = C_R(r0) C_DDA,aniso(r)/C_DDA,aniso(r0), r0 = 20 nm", per_wavelength=res, summary=summary, thresholds=thresholds, runtime_s=time.time() - t0)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    log("archived ->", OUT, f"({out['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
