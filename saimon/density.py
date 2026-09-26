"""
density.py — Aqueous H2SO4 density models for SAGE III aerosol optics.

All classes implement DensityModel: callable(weight_percent, temperature_k) -> kg/m³.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class DensityModel:
    """Protocol: rho(weight_percent, temperature_k) -> density [kg m^-3]."""
    def __call__(self, weight_percent: float, temperature_k: float) -> float: ...


# ---------------------------------------------------------------------------
# Myhre et al. (1998) — default implementation
# ---------------------------------------------------------------------------

# Coefficient matrix C[i, j] for:
#   rho = sum_{i,j} C[i,j] * w^i * (T - 273.15)^j
# where w = weight_percent / 100 (mass fraction) and T is in Kelvin.
#
# Source: Myhre et al. (1998), J. Chem. Eng. Data 43, 617, Table 2.
# Valid range: w in [0.10, 0.90], T in [210, 323] K.
#
# IMPORTANT: The j=4 column appears as ×10^-7 in the printed paper, but that
# is a TYPO — it must be ×10^-9 for the polynomial to reproduce the paper's
# own Table 1 (verified numerically; see test_density.py::test_myhre_j4_typo).
_MYHRE_C = np.zeros((11, 5))
_MYHRE_C[0, 0] = 999.8426;    _MYHRE_C[0, 1] = 334.5402e-4;   _MYHRE_C[0, 2] = -569.1304e-5
_MYHRE_C[1, 0] = 547.2659;    _MYHRE_C[1, 1] = -530.0445e-2;  _MYHRE_C[1, 2] = 118.7671e-4;  _MYHRE_C[1, 3] = 599.0008e-6
_MYHRE_C[2, 0] = 526.2950e1;  _MYHRE_C[2, 1] = 372.0445e-1;   _MYHRE_C[2, 2] = 120.1909e-3;  _MYHRE_C[2, 3] = -414.8594e-5; _MYHRE_C[2, 4] = 119.7973e-9
_MYHRE_C[3, 0] = -621.3958e2; _MYHRE_C[3, 1] = -287.7670;     _MYHRE_C[3, 2] = -406.4638e-3; _MYHRE_C[3, 3] = 111.9488e-4;  _MYHRE_C[3, 4] = 360.7768e-9
_MYHRE_C[4, 0] = 409.0293e3;  _MYHRE_C[4, 1] = 127.0854e1;    _MYHRE_C[4, 2] = 326.9710e-3;  _MYHRE_C[4, 3] = -137.7435e-4; _MYHRE_C[4, 4] = -263.3585e-9
_MYHRE_C[5, 0] = -159.6989e4; _MYHRE_C[5, 1] = -306.2836e1;   _MYHRE_C[5, 2] = 136.6499e-3;  _MYHRE_C[5, 3] = 637.3031e-5
_MYHRE_C[6, 0] = 385.7411e4;  _MYHRE_C[6, 1] = 408.3714e1;    _MYHRE_C[6, 2] = -192.7785e-3
_MYHRE_C[7, 0] = -580.8064e4; _MYHRE_C[7, 1] = -284.4401e1
_MYHRE_C[8, 0] = 530.1976e4;  _MYHRE_C[8, 1] = 809.1053
_MYHRE_C[9, 0] = -268.2616e4
_MYHRE_C[10, 0] = 576.4288e3


class MyhreDensity:
    """Aqueous H2SO4 density (Myhre 1998, Table 2).

    rho(w, T) = sum_{i=0..10} sum_{j=0..4} C[i,j] * w^i * (T-273.15)^j

    where w = weight_percent / 100 is the mass fraction.
    Valid: w in [0.10, 0.90], T in [210, 323] K.
    """

    def __repr__(self) -> str:
        return "MyhreDensity()"

    def __call__(self, weight_percent: float, temperature_k: float) -> float:
        w = weight_percent / 100.0
        t = temperature_k - 273.15

        if not (0.10 <= w <= 0.90) or not (210 <= temperature_k <= 323):
            logger.warning(
                "MyhreDensity: (w=%.4f, T=%.2f K) outside valid range "
                "[0.10, 0.90] × [210, 323 K]; result is an extrapolation.",
                w, temperature_k,
            )

        w_powers = np.array([w ** i for i in range(11)])
        t_powers = np.array([t ** j for j in range(5)])
        return float(w_powers @ _MYHRE_C @ t_powers)


# ---------------------------------------------------------------------------
# Timmermans (1960) — placeholder for later comparison
# ---------------------------------------------------------------------------

class TimmermansDensity:
    """Density from Timmermans (1960), Vol. 4, pp. 568–569.

    Will implement DensityModel once the data table is entered from the
    physical copy. Intended for comparison against MyhreDensity by swapping
    into lorentz_lorenz_to_temperature(density_model=TimmermansDensity()).
    """

    def __call__(self, weight_percent: float, temperature_k: float) -> float:
        raise NotImplementedError(
            "TimmermansDensity: data table not yet entered. "
            "Supply the CSV and call from_csv()."
        )

    @classmethod
    def from_csv(cls, path: str) -> "TimmermansDensity":
        """Load the Timmermans density table from a CSV file (stub)."""
        raise NotImplementedError(
            "TimmermansDensity.from_csv: not yet implemented."
        )
