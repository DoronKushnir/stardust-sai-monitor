"""
refractive_index.py — Complex refractive index providers for SAGE III aerosol Mie.

All classes implement RefractiveIndexProtocol: callable with a wavelength_m array,
returns a complex ndarray of the same shape.
"""

from __future__ import annotations

import csv as _csv
import hashlib
import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

from .config import OPTICS
_CSVFILES_DIR = str(OPTICS)


@runtime_checkable
class RefractiveIndexProtocol(Protocol):
    """Protocol for complex refractive index providers."""
    def __call__(self, wavelength_m: np.ndarray) -> np.ndarray:
        ...


class ConstantRI:
    """Wavelength-independent complex refractive index, RefractiveIndexProtocol.

    Default m = 1.4 + 0.0j (current placeholder for stratospheric sulfate).
    """

    def __init__(self, n: float = 1.4, k: float = 0.0):
        self._m = complex(n, k)

    def __repr__(self) -> str:
        return f"ConstantRI(n={self._m.real!r}, k={self._m.imag!r})"

    def __call__(self, wavelength_m) -> np.ndarray:
        wavelength_m = np.asarray(wavelength_m, dtype=float)
        return np.full(wavelength_m.shape, self._m, dtype=complex)


class TabulatedRI:
    """Wavelength-dependent m from a table, RefractiveIndexProtocol.

    Parameters
    ----------
    wavelength_nm : 1-D array of tabulated wavelengths [nm]
    n             : 1-D array of real parts at those wavelengths
    k             : 1-D array of imaginary parts (k >= 0), default zeros

    __call__ interpolates n and k linearly in λ to the requested wavelengths
    and returns n + i k as a complex ndarray. Out-of-range requests clamp to
    the table ends with a logged warning.
    """

    def __init__(self, wavelength_nm, n, k=None, temperature_k=None):
        self._wl_nm = np.asarray(wavelength_nm, dtype=float)
        self._n = np.asarray(n, dtype=float)
        self._k = np.zeros_like(self._n) if k is None else np.asarray(k, dtype=float)
        self.temperature_k = temperature_k

    def __repr__(self) -> str:
        h = hashlib.sha256(
            self._wl_nm.tobytes() + self._n.tobytes() + self._k.tobytes()
        )
        t = self.temperature_k
        t_part = f", T={t}K" if t is not None else ""
        return f"TabulatedRI(sha256={h.hexdigest()[:32]}{t_part})"

    def __call__(self, wavelength_m) -> np.ndarray:
        wl_nm = np.asarray(wavelength_m, dtype=float) * 1e9
        lo, hi = self._wl_nm[0], self._wl_nm[-1]
        if np.any(wl_nm < lo) or np.any(wl_nm > hi):
            logger.warning(
                "Wavelength(s) outside table range [%.1f, %.1f] nm; clamping.", lo, hi
            )
        wl_clamped = np.clip(wl_nm, lo, hi)
        n_out = np.interp(wl_clamped, self._wl_nm, self._n)
        k_out = np.interp(wl_clamped, self._wl_nm, self._k)
        return (n_out + 1j * k_out).astype(complex)

    @classmethod
    def from_csv(cls, path, temperature_k=300.0) -> "TabulatedRI":
        """Load a wavelength_nm, n, k table (blank k → 0). Returns a TabulatedRI.

        Parameters
        ----------
        path          : CSV file with columns wavelength_nm, n[, k].
        temperature_k : Stored in repr/cache-key so callers can distinguish tables
                        at different temperatures.

        Notes
        -----
        k below the table's first non-zero entry → 0.0 (transparent), because
        the CSV encodes transparent wavelengths with a blank k column; those are
        read as 0.0 and linear interpolation between them stays at 0.0.  n is
        clamped to the nearest tabulated value outside the table range (warning
        is logged).
        """
        wavelengths, n_vals, k_vals = [], [], []
        with open(path, newline='') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                wavelengths.append(float(row['wavelength_nm']))
                n_vals.append(float(row['n']))
                k_str = row.get('k', '').strip()
                k_vals.append(float(k_str) if k_str else 0.0)
        return cls(wavelengths, n_vals, k_vals, temperature_k=temperature_k)


class SulfuricAcidRI:
    """Palmer & Williams (1975) + Lorentz-Lorenz T-correction. Future work.
    Raises NotImplementedError; left here to mark the intended extension point."""

    def __call__(self, wavelength_m) -> np.ndarray:
        raise NotImplementedError(
            "SulfuricAcidRI is not yet implemented. Use ConstantRI or TabulatedRI."
        )


class _LorentzLorentzRI:
    """RI wrapper that applies the Lorentz–Lorenz temperature correction.

    For each requested wavelength:
        n300 = ri_300k(λ).real
        f300 = (n300² − 1) / (n300² + 2)
        f(T) = f300 · ρ(w, T) / ρ(w, T_ref)
        n(T) = sqrt((1 + 2f) / (1 − f))
    k is passed through unchanged (L–L acts on the real part only).
    """

    def __init__(self, ri_300k, target_temperature_k, density_model,
                 weight_percent, reference_temperature_k):
        self._ri_base = ri_300k
        self.temperature_k = target_temperature_k
        self._density_model = density_model
        self._weight_percent = weight_percent
        self._ref_temp = reference_temperature_k

        rho_t   = density_model(weight_percent, target_temperature_k)
        rho_ref = density_model(weight_percent, reference_temperature_k)
        self._density_ratio = rho_t / rho_ref

    def __repr__(self) -> str:
        return (
            f"LorentzLorentzRI("
            f"base={self._ri_base!r}, "
            f"T={self.temperature_k}K, "
            f"density={self._density_model!r}, "
            f"w%={self._weight_percent})"
        )

    def __call__(self, wavelength_m) -> np.ndarray:
        m_base = self._ri_base(wavelength_m)
        n_base = m_base.real
        k_base = m_base.imag

        f_ref = (n_base ** 2 - 1.0) / (n_base ** 2 + 2.0)
        f_t   = f_ref * self._density_ratio
        n_t   = np.sqrt((1.0 + 2.0 * f_t) / (1.0 - f_t))

        return (n_t + 1j * k_base).astype(complex)


def lorentz_lorenz_to_temperature(
    ri_300k,
    target_temperature_k: float,
    density_model=None,
    weight_percent: float = 75.0,
    reference_temperature_k: float = 300.0,
) -> _LorentzLorentzRI:
    """Wrap a 300 K RI and return a new RI corrected to target_temperature_k.

    Uses the Lorentz–Lorenz relation (Steele & Hamill 1981, Eq. 6):
        f(n(T)) = f(n_ref) · ρ(w, T) / ρ(w, T_ref),   f(n) = (n²−1)/(n²+2).

    Parameters
    ----------
    ri_300k               : RefractiveIndexProtocol at reference temperature.
    target_temperature_k  : Temperature to correct to [K].
    density_model         : DensityModel callable; default MyhreDensity().
    weight_percent        : H2SO4 weight percent; default 75.0.
    reference_temperature_k : Temperature at which ri_300k was measured; default 300.0 K.
    """
    if density_model is None:
        from .density import MyhreDensity
        density_model = MyhreDensity()
    return _LorentzLorentzRI(
        ri_300k, target_temperature_k, density_model,
        weight_percent, reference_temperature_k,
    )


def sulfuric_acid_at_temperature(
    temperature_k: float,
    density_model=None,
    csvfiles_dir=None,
) -> _LorentzLorentzRI:
    """Palmer & Williams 75 wt% H2SO4 corrected to temperature_k via Lorentz–Lorenz.

    Works for any T within Myhre's valid range (210–323 K), not just 215 K.
    Pass density_model to substitute an alternative density model (e.g. Timmermans).
    """
    return lorentz_lorenz_to_temperature(
        sulfuric_acid_300k(csvfiles_dir),
        temperature_k,
        density_model=density_model,
    )


def sulfuric_acid_300k(csvfiles_dir=None) -> TabulatedRI:
    """Palmer & Williams (1975) 75 wt% H2SO4 at 300 K.

    Returns a TabulatedRI interpolated from the combined table in
    palmer_williams_75_combined_300K.csv (360 nm – 25 µm, k=0 below ~700 nm).

    Parameters
    ----------
    csvfiles_dir : Directory containing palmer_williams_75_combined_300K.csv.
                   Defaults to <project>/CSVFiles.
    """
    if csvfiles_dir is None:
        csvfiles_dir = _CSVFILES_DIR
    path = os.path.join(csvfiles_dir, 'palmer_williams_75_combined_300K.csv')
    return TabulatedRI.from_csv(path, temperature_k=300.0)
