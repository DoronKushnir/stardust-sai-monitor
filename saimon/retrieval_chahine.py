"""
retrieval_chahine.py — Chahine iterative nonlinear extinction retrieval.

Implements the Chahine (1972) / LOA method for retrieving vertical extinction
profiles from limb occultation slant-path optical depths.

Reference: Chu et al. (1989) Eq. (22).
"""

import numpy as np
from typing import Optional


def retrieve_extinction_chahine(
    delta_aer: np.ndarray,
    sigma_delta: np.ndarray,
    P: np.ndarray,
    max_iter: int = 100,
    epsilon: float = 1.0,
    sigma_init: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Chahine iterative retrieval of vertical extinction profile.

    Parameters
    ----------
    delta_aer   : (n_tangent,)  aerosol slant-path OD, top-down ordered
    sigma_delta : (n_tangent,)  1-sigma error on delta_aer
    P           : (n_tangent, n_layers)  path length matrix [m], lower-triangular
    max_iter    : int, maximum iterations
    epsilon     : float, convergence threshold (in units of sigma_delta)
    sigma_init  : (n_layers,) or None.  Initial guess.
                  If None: sigma_init[k] = delta_aer[k] / P[k,k]
                  (diagonal estimate = onion-peel first guess).

    Returns
    -------
    sigma_ext    : (n_layers,)   retrieved extinction [m^-1]
    n_iterations : int           number of iterations to convergence
    residuals    : (n_iterations,)  rms residual at each iteration
    """
    n = len(delta_aer)

    if sigma_init is None:
        diag_P = np.diag(P)
        sigma = np.where(diag_P > 0, delta_aer / np.where(diag_P > 0, diag_P, 1.0), 0.0)
    else:
        sigma = np.array(sigma_init, dtype=float)

    sigma = np.maximum(sigma, 0.0)

    residuals = []
    n_iterations = max_iter

    for iteration in range(max_iter):
        delta_computed = P @ sigma
        r = (delta_computed - delta_aer) / sigma_delta
        residuals.append(float(np.sqrt(np.mean(r ** 2))))

        if np.max(np.abs(r)) < epsilon:
            n_iterations = iteration + 1
            break

        # Chahine multiplicative update with diagonal pairing (layer k updated by ray k)
        FLOOR = 1e-12  # m^-1, prevents division by zero in wings
        sigma_new = sigma.copy()
        mask = delta_computed >= 1e-30
        sigma_new[mask] = sigma[mask] * delta_aer[mask] / delta_computed[mask]
        sigma = np.maximum(sigma_new, FLOOR)

    return sigma, n_iterations, np.array(residuals)


def chahine_error_estimate(
    sigma_ext: np.ndarray,
    sigma_delta: np.ndarray,
    P: np.ndarray,
    n_mc: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    Monte Carlo error estimate for Chahine retrieval.

    Runs n_mc realisations with noise added to delta_aer, retrieves each,
    returns std across realisations.

    Parameters
    ----------
    sigma_ext   : (n_layers,)  retrieved profile used as truth for MC
    sigma_delta : (n_tangent,) 1-sigma error on delta_aer
    P           : (n_tangent, n_layers)
    n_mc        : int, number of MC realisations
    seed        : int, RNG seed

    Returns
    -------
    sigma_err_mc : (n_layers,)  empirical 1-sigma error from MC
    """
    rng = np.random.default_rng(seed)
    delta_aer_true = P @ sigma_ext

    n_layers = len(sigma_ext)
    n_tangent = len(sigma_delta)
    mc_results = np.zeros((n_mc, n_layers))

    for i in range(n_mc):
        noise = rng.normal(0.0, 1.0, n_tangent) * sigma_delta
        delta_noisy = delta_aer_true + noise
        sigma_ret, _, _ = retrieve_extinction_chahine(
            delta_noisy, sigma_delta, P, max_iter=200
        )
        mc_results[i] = sigma_ret

    return np.std(mc_results, axis=0, ddof=1)
