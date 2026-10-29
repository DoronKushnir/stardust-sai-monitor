"""Replay all seed-76 draws and audit/refit closure convergence without dropping draws.

The unchanged round-76 driver supplies identical truths and noise. Every draw is
archived with its uncertainty, legacy flag, full-state stationarity check, and
bounded least-squares refinement when the old fit failed or the check fails.
Run with --ndraw 100 --nvar 33 (the defaults). BLAS threads may be limited to one.
"""
from pathlib import Path
import json
import inspect
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares
import closure_final as F

OUT = F.ROOT / 'outputs/closure_stability.json'
LEGACY = F.Estimator.fit
RECORDS = []
TOL = 1e-3


def setup(est):
    if not hasattr(est, 'white'):
        est.white = solve_triangular(np.linalg.cholesky(est.S), np.eye(len(est.S)), lower=True)
        est.prior_white = np.linalg.cholesky(est.S_a_inv).T
        est.lower = np.full(F.C.N_STATE, -np.inf)
        est.upper = np.full(F.C.N_STATE, np.inf)
        est.lower[F.C.GAS_IDX], est.upper[F.C.GAS_IDX] = .05, 5.
        est.lower[F.C.I_SOFT], est.upper[F.C.I_SOFT] = .001, .02


def residual_jac(est, obs, x):
    Kwin, tau = F.C.jacobian(x)
    shift, soft = float(x[F.C.I_SHIFT]), float(x[F.C.I_SOFT])
    d = 1e-5
    Kwin[:, F.C.I_SHIFT] = (F.C.element_obs(tau, shift+d, soft)-F.C.element_obs(tau, shift-d, soft))/(2*d)
    Kwin[:, F.C.I_SOFT] = (F.C.element_obs(tau, shift, soft+d)-F.C.element_obs(tau, shift, soft-d))/(2*d)
    K = np.vstack([Kwin, est.K_out, est.K_tri])
    y = np.r_[F.C.element_obs(tau, shift, soft), est.K_out @ x, est.K_tri @ x]
    r = np.r_[est.white @ (y - obs), est.prior_white @ (x - F.C.x_ref)]
    J = np.vstack([est.white @ K, est.prior_white])
    return r, J


def audit(est, obs, x):
    r, J = residual_jac(est, obs, x)
    grad = J.T @ r
    active = ((x <= est.lower + 1e-10) & (grad > 0)) | ((x >= est.upper - 1e-10) & (grad < 0))
    scaled = np.where(active, 0., grad) / np.maximum(np.linalg.norm(J, axis=0), 1e-30)
    return dict(objective=float(r @ r), scaled_projected_gradient=float(np.max(np.abs(scaled)))), J


def fit(est, y_obs, n_iter=25):
    setup(est)
    x, S, rep = LEGACY(est, y_obs, n_iter)
    before, _ = audit(est, y_obs, x)
    old_mass = float(x[F.C.I_SIL] * F.RHO)
    old_sigma = float(np.sqrt(S[F.C.I_SIL, F.C.I_SIL]) * F.RHO)
    refine = not rep['converged'] or before['scaled_projected_gradient'] > TOL
    result = None
    xr = x.copy()
    if refine:
        # Cache the expensive model shared by the residual and Jacobian calls.
        cache = {}
        def evaluate(q):
            if 'x' not in cache or not np.array_equal(q, cache['x']):
                cache['x'] = q.copy()
                cache['r'], cache['J'] = residual_jac(est, y_obs, q)
            return cache['r'], cache['J']
        result = least_squares(lambda q: evaluate(q)[0], xr, jac=lambda q: evaluate(q)[1],
                               bounds=(est.lower, est.upper), x_scale='jac',
                               ftol=1e-11, xtol=1e-11, gtol=1e-9, max_nfev=200)
        xr = result.x
    after, J = audit(est, y_obs, xr)
    # A scaled information inverse evaluated at the returned state.
    scale = np.maximum(np.linalg.norm(J, axis=0), 1e-30)
    Js = J / scale
    cov = np.linalg.inv(Js.T @ Js) / scale[:, None] / scale[None, :]
    RECORDS.append(dict(legacy_mass_tg=old_mass, legacy_sigma_tg=old_sigma,
                        legacy_converged=rep['converged'], legacy_iterations=rep['n_iter'],
                        before=before, refitted=refine,
                        mass_tg=float(xr[F.C.I_SIL] * F.RHO),
                        sigma_tg=float(np.sqrt(cov[F.C.I_SIL, F.C.I_SIL]) * F.RHO),
                        after=after, converged=after['scaled_projected_gradient'] <= TOL,
                        optimizer_status=None if result is None else int(result.status),
                        nfev=0 if result is None else int(result.nfev)))
    # Return the original values so that the old published summary is replayed exactly.
    return x, S, rep


def summarize(draws, truth, prefix=''):
    A = np.array([d[prefix + 'mass_tg'] for d in draws])
    s = np.array([d[prefix + 'sigma_tg'] for d in draws])
    z = (A-truth)/s
    return dict(n=len(A), bias_mean=float(np.mean(A-truth)), bias_median=float(np.median(A-truth)),
                sd=float(np.std(A, ddof=1)), posterior_sigma_median=float(np.median(s)),
                coverage_1sig=float(np.mean(np.abs(z)<1)), coverage_2sig=float(np.mean(np.abs(z)<2)),
                p_detect_3sig=float(np.mean(A/s>3)), p_detect_4p54sig=float(np.mean(A/s>4.54)))


def worker(est, obs):
    x, S, rep = fit(est, obs)
    return x, S, rep, RECORDS.pop()


if __name__ == '__main__':
    # Generate truths/noise in the original serial RNG order, then fit independently.
    # Each worker initializes its immutable line profiles; stochastic work stays in the parent.
    pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn'))
    F.POOL, F.WORKER, F.RECORDS = pool, worker, RECORDS
    source = inspect.getsource(F.run)
    source = source.replace('    t0 = time.time()', '    jobs = []\n    t0 = time.time()')
    begin = source.index('            x_hat, S_hat, rep = est.fit(y_obs)')
    end = source.index('    summ = {}', begin)
    source = source[:begin] + """            jobs.append((M_shell, POOL.submit(WORKER, est, y_obs)))
    for j, (M_shell, future) in enumerate(jobs):
        x_hat, S_hat, rep, record = future.result()
        RECORDS.append(record)
        res[M_shell]["A"].append(float(x_hat[C.I_SIL] * RHO))
        res[M_shell]["s"].append(float(np.sqrt(S_hat[C.I_SIL, C.I_SIL]) * RHO))
        res[M_shell]["conv"].append(rep["converged"])
        res[M_shell]["it"].append(rep["n_iter"])
        if (j+1) % 50 == 0:
            log(label, j+1, '/', len(jobs), 'fits audited', int(time.time()-t0), 's')
""" + source[end:]
    exec(source, F.__dict__)
    # Keep the published ensemble archive intact.
    F.OUT = OUT.with_suffix('.replay.json')
    F.main()
    pool.shutdown()
    replay = json.loads(F.OUT.read_text())
    rows = []
    offset = 0
    def add_group(name, group):
        global offset
        masses = list(group)
        n = group[masses[0]]['n']
        block = RECORDS[offset:offset+n*len(masses)]
        assert len(block) == n*len(masses)
        for i, m in enumerate(masses):
            draws = block[i::len(masses)]
            old = summarize(draws, float(m), 'legacy_')
            for key in old:
                assert np.isclose(old[key], group[m][key], rtol=1e-8, atol=1e-12), (name, m, key)
            rows.append(dict(case=name, truth_tg=float(m), legacy=old, refined=summarize(draws,float(m)),
                             n_legacy_unconverged=sum(not d['legacy_converged'] for d in draws),
                             n_refitted=sum(d['refitted'] for d in draws),
                             n_refined_unconverged=sum(not d['converged'] for d in draws), draws=draws))
        offset += len(block)
    tests = replay['tests']
    for name in ('final_band','final_window','spectroscopic_flat_window'):
        add_group(name,tests[name])
    variants = tests['misspecification_final_band']
    for name in ('out_of_prior_x2','alt_sulfate_member_B70T215','alt_line_widths_x1p1'):
        add_group(name,variants[name])
    for name, group in variants['fixed_database']['per_database'].items():
        add_group('fixed_database_'+name,group)
    assert offset == len(RECORDS)
    OUT.write_text(json.dumps(dict(seed=replay['seed'], n_fits=len(RECORDS),
        convergence='max abs projected objective gradient / whitened Jacobian column norm <= 1e-3, all state coordinates; objective=sum squared whitened residuals including prior',
        refinement='response derivatives at 1e-5 um; bounded trust-region least_squares; ftol=xtol=1e-11, gtol=1e-9; max_nfev=200; no draws discarded',
        rows=rows), indent=1)+'\n')
    print('Stability archive:',OUT,flush=True)
    from closure_stability_summary import main as summarize_audit
    summarize_audit()
