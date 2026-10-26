"""
marginal_rank_tests.py -- round 70 (referee round 5, item 6): the sparse /
redundant-column checks of marginal() against the WORKING-repo function
(reproduce/calibrated_background_thresholds.py).  The same checks are the four
round-70 tests of ../stardust-sai-monitor/tests/test_package.py (public copy).

Checks
  1. referee counterexample J = [[1,1,0],[0,0,1]], unit errors, last column the
     target: sigma = 1 exactly (full Fisher matrix singular, K = 2 < N = 3);
  2. duplicated / linearly dependent nuisance column: span and sigma unchanged;
  3. zero nuisance column: dropped, sigma unchanged; only-zero nuisance -> sigma = 1/||t/sig||;
  4. target in the nuisance span, zero target, or K generic nuisance columns on K
     channels: (inf, 0.0, nan); one more channel makes it finite.

Run from the repo root:  python reproduce/marginal_rank_tests.py
Exit status 0 when every check passes.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

_THIS = Path(__file__).resolve()
_PARENT = _THIS.parent.parent
for p in (_PARENT, _THIS.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from calibrated_background_thresholds import marginal, nuisance_span, SVD_RTOL, TPERP_RTOL  # noqa: E402


def main():
    fails = 0

    def check(name, cond, detail=""):
        nonlocal fails
        print(f"  [{'ok' if cond else 'FAIL'}] {name}  {detail}")
        fails += 0 if cond else 1

    print(f"marginal() from {sys.modules['calibrated_background_thresholds'].__file__}")
    print(f"SVD_RTOL = {SVD_RTOL:g}, TPERP_RTOL = {TPERP_RTOL:g}")

    print("\n1. referee counterexample J = [[1,1,0],[0,0,1]], unit errors, last column target")
    c1 = np.array([1.0, 0.0]); c2 = np.array([1.0, 0.0]); t = np.array([0.0, 1.0]); sig = np.ones(2)
    sa, tperp, R2 = marginal([0, 1], [c1, c2], sig, t)
    Ur, s, n_kept = nuisance_span([0, 1], [c1, c2], sig)
    check("sigma == 1.0 exactly", sa == 1.0, f"sigma = {sa!r}, ||t_perp|| = {tperp!r}, R2 = {R2!r}")
    check("nuisance rank 1 of 2 columns", Ur.shape == (2, 1) and n_kept == 2,
          f"singular values {s}, kept {Ur.shape[1]}")

    print("\n2. duplicated / dependent nuisance columns")
    rng = np.random.default_rng(7)
    a, b, t = rng.normal(size=(3, 6)); sig = rng.uniform(0.5, 2.0, 6); idx = np.arange(6)
    ref = marginal(idx, [a, b], sig, t)
    print(f"  reference [a, b]: sigma = {ref[0]:.15g}")
    for lab, cols in (("[a, a, b]", [a, a, b]), ("[a, b, 2.5a-b]", [a, b, 2.5 * a - b]),
                      ("[b, a, a, b]", [b, a, a, b])):
        got = marginal(idx, cols, sig, t)
        rk = nuisance_span(idx, cols, sig)[0].shape[1]
        check(f"{lab}: sigma unchanged, rank 2", np.allclose(got, ref, rtol=1e-12, atol=0) and rk == 2,
              f"sigma = {got[0]:.15g}, rank {rk}")

    print("\n3. zero nuisance column")
    rng = np.random.default_rng(11)
    a, b, t = rng.normal(size=(3, 6)); sig = rng.uniform(0.5, 2.0, 6)
    ref = marginal(idx, [a, b], sig, t)
    got = marginal(idx, [a, np.zeros(6), b], sig, t)
    check("[a, 0, b] == [a, b]", np.allclose(got, ref, rtol=1e-12, atol=0),
          f"{got[0]:.15g} vs {ref[0]:.15g}; kept {nuisance_span(idx, [a, np.zeros(6), b], sig)[2]} of 3")
    sa, tperp, R2 = marginal(idx, [np.zeros(6)], sig, t)
    check("[0] only: sigma = 1/||t/sig||, R2 = 0",
          abs(sa * np.linalg.norm(t / sig) - 1) < 1e-14 and R2 == 0.0, f"sigma = {sa:.15g}")

    print("\n4. unidentifiable target")
    rng = np.random.default_rng(3)
    a, b, c, t = rng.normal(size=(4, 6)); sig = rng.uniform(0.5, 2.0, 6)
    for lab, cols, tt in (("t = 0.3a - 2b in span", [a, b], 0.3 * a - 2.0 * b),
                          ("zero target", [a, b], np.zeros(6))):
        sa, tperp, R2 = marginal(idx, cols, sig, tt)
        check(f"{lab}: (inf, 0, nan)", sa == np.inf and tperp == 0.0 and np.isnan(R2),
              f"got {(sa, tperp, R2)}; rank {nuisance_span(idx, cols, sig)[0].shape[1]}")
    # six nuisance columns of rank 3 (a + b, c - a and a - 2b dependent on a, b, c) on six
    # channels leave a generic target finite: the count (6 columns, K = 6) would have said inf
    six_dep = [a, b, c, a + b, c - a, a - 2 * b]
    sa, tperp, R2 = marginal(idx, six_dep, sig, t)
    check("6 dependent nuisance columns (rank 3) on 6 channels: finite",
          np.isfinite(sa) and nuisance_span(idx, six_dep, sig)[0].shape[1] == 3, f"sigma = {sa:.6g}")
    # the whole-space case needs six independent columns
    six_ind = [a, b, c] + [rng.normal(size=6) for _ in range(3)]
    sa, tperp, R2 = marginal(idx, six_ind, sig, t)
    check("6 independent nuisance columns on 6 channels: (inf, 0, nan)",
          sa == np.inf and tperp == 0.0 and np.isnan(R2),
          f"got {(sa, tperp, R2)}; ||t_perp||/||t_hat|| would be round-off")
    cols7 = [np.r_[x, rng.normal()] for x in six_ind]
    sa7, _, _ = marginal(np.arange(7), cols7, np.r_[sig, 1.0], np.r_[t, rng.normal()])
    check("one more channel: finite", np.isfinite(sa7), f"sigma = {sa7:.6g}")

    print(f"\n{'ALL CHECKS PASSED' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
