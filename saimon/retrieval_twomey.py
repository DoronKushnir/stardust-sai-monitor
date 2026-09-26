"""
retrieval_twomey.py — Twomey (1963) regularized vertical extinction retrieval.

Implements the LaRC method (Chu et al. 1989, Eq. 18):
    sigma = (P^T W P + gamma H)^{-1} P^T W delta

with automatic gamma selection via L-curve and GCV criteria.
"""

import numpy as np
from typing import Optional


def second_difference_matrix(n: int) -> np.ndarray:
    """
    Build the n x n second-difference regularization matrix H.

    Interior: H[i,i]=2, H[i,i±1]=-1
    Boundaries: H[0,0]=1, H[n-1,n-1]=1
    """
    H = np.zeros((n, n))
    for i in range(n):
        if i == 0 or i == n - 1:
            H[i, i] = 1.0
        else:
            H[i, i] = 2.0
        if i > 0:
            H[i, i - 1] = -1.0
        if i < n - 1:
            H[i, i + 1] = -1.0
    return H


def retrieve_extinction_twomey(
    delta_aer: np.ndarray,
    sigma_delta: np.ndarray,
    P: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Twomey regularized retrieval.

    Parameters
    ----------
    delta_aer   : (n_tangent,)  aerosol slant-path OD, top-down ordered
    sigma_delta : (n_tangent,)  1-sigma error on delta_aer
    P           : (n_tangent, n_layers)  path length matrix [m]
    gamma       : float, regularization parameter

    Returns
    -------
    sigma_ext  : (n_layers,)  retrieved extinction [m^-1]
    sigma_err  : (n_layers,)  1-sigma error
    cov_sigma  : (n_layers, n_layers)  full error covariance matrix
    """
    n_layers = P.shape[1]
    W = np.diag(1.0 / sigma_delta ** 2)
    H = second_difference_matrix(n_layers)

    A = P.T @ W @ P + gamma * H
    b = P.T @ W @ delta_aer

    sigma_ext = np.linalg.solve(A, b)

    A_inv = np.linalg.inv(A)
    cov_sigma = A_inv @ (P.T @ W @ P) @ A_inv
    sigma_err = np.sqrt(np.diag(cov_sigma))

    return sigma_ext, sigma_err, cov_sigma


def choose_gamma_lcurve(
    delta_aer: np.ndarray,
    sigma_delta: np.ndarray,
    P: np.ndarray,
    gamma_range: Optional[tuple] = None,
    n_points: int = 50,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Select optimal gamma using the L-curve criterion (maximum curvature).

    Returns
    -------
    gamma_opt   : float
    gammas      : (n_points,) array of gamma values tested
    residuals   : (n_points,) residual norms (log scale x-axis)
    roughness   : (n_points,) solution roughness norms (log scale y-axis)
    corner_idx  : int, index of selected gamma
    """
    if gamma_range is None:
        gamma_range = (1e-20, 1e10)

    gammas = np.logspace(np.log10(gamma_range[0]), np.log10(gamma_range[1]), n_points)
    n_layers = P.shape[1]
    H = second_difference_matrix(n_layers)
    W = np.diag(1.0 / sigma_delta ** 2)
    PtWP = P.T @ W @ P
    PtWd = P.T @ W @ delta_aer

    residuals = np.zeros(n_points)
    roughness = np.zeros(n_points)

    for idx, g in enumerate(gammas):
        A = PtWP + g * H
        sigma = np.linalg.solve(A, PtWd)
        res = P @ sigma - delta_aer
        # weighted residual norm
        residuals[idx] = np.sqrt(np.sum((res / sigma_delta) ** 2))
        roughness[idx] = np.sqrt(sigma @ H @ sigma)

    # L-curve corner: maximum curvature in log-log space
    log_r = np.log(np.maximum(residuals, 1e-300))
    log_s = np.log(np.maximum(roughness, 1e-300))

    # Finite differences for curvature
    dr = np.gradient(log_r)
    ds = np.gradient(log_s)
    d2r = np.gradient(dr)
    d2s = np.gradient(ds)

    # Curvature κ = (r' s'' - r'' s') / (r'^2 + s'^2)^(3/2)
    denom = (dr ** 2 + ds ** 2) ** 1.5
    with np.errstate(divide='ignore', invalid='ignore'):
        curvature = np.where(denom > 0, (dr * d2s - d2r * ds) / denom, 0.0)

    corner_idx = int(np.argmax(curvature))
    gamma_opt = float(gammas[corner_idx])

    return gamma_opt, gammas, residuals, roughness, corner_idx


def choose_gamma_gcv(
    delta_aer: np.ndarray,
    sigma_delta: np.ndarray,
    P: np.ndarray,
    gamma_range: Optional[tuple] = None,
    n_points: int = 50,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Select optimal gamma using Generalized Cross-Validation (GCV).

    GCV(gamma) = ||P sigma(gamma) - delta||^2 / [trace(I - P A^{-1} P^T W)]^2

    Returns
    -------
    gamma_opt  : float
    gammas     : (n_points,) array
    gcv_values : (n_points,) GCV scores
    """
    if gamma_range is None:
        gamma_range = (1e-20, 1e10)

    gammas = np.logspace(np.log10(gamma_range[0]), np.log10(gamma_range[1]), n_points)
    n_tangent = len(delta_aer)
    n_layers = P.shape[1]
    H = second_difference_matrix(n_layers)
    W = np.diag(1.0 / sigma_delta ** 2)
    PtWP = P.T @ W @ P
    PtWd = P.T @ W @ delta_aer

    gcv_values = np.zeros(n_points)

    for idx, g in enumerate(gammas):
        A = PtWP + g * H
        A_inv = np.linalg.inv(A)
        sigma = A_inv @ PtWd
        res = P @ sigma - delta_aer
        numerator = np.sum(res ** 2)
        # influence matrix: P A^{-1} P^T W
        influence = P @ A_inv @ P.T @ W
        denom = np.trace(np.eye(n_tangent) - influence) ** 2
        gcv_values[idx] = numerator / max(denom, 1e-300)

    gamma_opt = float(gammas[np.argmin(gcv_values)])
    return gamma_opt, gammas, gcv_values
