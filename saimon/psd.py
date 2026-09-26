"""
psd.py — Particle size distributions implementing PSDProtocol.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Analytic moment helpers (Wrana Eqs. 5–7)
# ---------------------------------------------------------------------------

def effective_radius(rmed_m: float, sigma: float) -> float:
    """Geometric mean of the area-weighted distribution [Wrana Eq. 5].

    r_eff = r_med * exp(2 * ln^2(sigma))
    This is the median of dA/d(ln r), consistent with Wrana §5 values.
    """
    return rmed_m * np.exp(2.0 * np.log(sigma) ** 2)


def mode_radius(rmed_m: float, sigma: float) -> float:
    """r_mod = exp(ln(r_med) - ln^2(sigma))  [Wrana Eq. 6]"""
    return np.exp(np.log(rmed_m) - np.log(sigma) ** 2)


def absolute_mode_width(rmed_m: float, sigma: float) -> float:
    """omega = r_med * ln(sigma) * exp(0.5 * ln^2(sigma))  [Wrana Eq. 7]"""
    s = np.log(sigma)
    return rmed_m * s * np.exp(0.5 * s ** 2)


def lognormal_volume_per_particle(rmed_m: float, sigma: float) -> float:
    """Mean particle volume for a lognormal PSD per unit number density [m^3 particle^-1].

    Derived from the analytic third moment of the lognormal distribution:
        <V> = (4/3) pi * integral_0^inf r^3 * f(r) dr
            = (4/3) pi * r_med^3 * exp(9/2 * ln^2(sigma))

    This is the per-particle volume to use for mass concentration:
        rho_aer(z) = N0(z) * lognormal_volume_per_particle(r_med, sigma) * rho_p
    """
    s = np.log(sigma)
    return (4.0 / 3.0) * np.pi * float(rmed_m) ** 3 * np.exp(4.5 * s ** 2)


# ---------------------------------------------------------------------------
# Monomodal lognormal (Wrana Eq. 1)
# ---------------------------------------------------------------------------

class LognormalPSD:
    """Monomodal lognormal PSD, implements PSDProtocol.

    dN/dr = N0 / (sqrt(2*pi) * r * ln(sigma))
            * exp( - ln(r/r_med)^2 / (2 * ln(sigma)^2) )

    Parameters
    ----------
    rmed_m : float   median radius [m]
    sigma  : float   mode width (dimensionless, >1)
    n0_m3  : float   total number density [m^-3], default 1.0
    """

    def __init__(self, rmed_m: float, sigma: float, n0_m3: float = 1.0):
        self._rmed = rmed_m
        self._sigma = sigma
        self._n0 = n0_m3
        self._ln_sigma = np.log(sigma)

    def dn_dr(self, r_grid_m: np.ndarray) -> np.ndarray:
        """Return dN/dr [m^-4] on r_grid_m."""
        r = np.asarray(r_grid_m, dtype=float)
        norm = self._n0 / (np.sqrt(2.0 * np.pi) * r * self._ln_sigma)
        exponent = -(np.log(r / self._rmed) ** 2) / (2.0 * self._ln_sigma ** 2)
        return norm * np.exp(exponent)

    @property
    def total_number_density_m3(self) -> float:
        return self._n0


# ---------------------------------------------------------------------------
# Future-form stubs
# ---------------------------------------------------------------------------

class BimodalLognormalPSD:
    """Two lognormal modes. Raises NotImplementedError for now."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("BimodalLognormalPSD is not yet implemented.")

    def dn_dr(self, r_grid_m):
        raise NotImplementedError("BimodalLognormalPSD is not yet implemented.")

    @property
    def total_number_density_m3(self):
        raise NotImplementedError("BimodalLognormalPSD is not yet implemented.")


class GammaPSD:
    """Gamma distribution (Nyaku et al. 2020). Raises NotImplementedError."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("GammaPSD is not yet implemented.")

    def dn_dr(self, r_grid_m):
        raise NotImplementedError("GammaPSD is not yet implemented.")

    @property
    def total_number_density_m3(self):
        raise NotImplementedError("GammaPSD is not yet implemented.")
