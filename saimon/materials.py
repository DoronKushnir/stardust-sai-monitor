"""
materials.py -- Refractive index material parsers.

Parses YAML material files (like silica_stardust.yml) and implements
RefractiveIndexProtocol.
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path

import yaml
import numpy as np
from scipy.interpolate import interp1d

from .refractive_index import RefractiveIndexProtocol

from .config import OPTICS as _CSVFILES_DIR


class SilicaRefractiveIndex:
    """Refractive index model for Silica based on a stardust YAML file."""

    def __init__(self, yaml_path: str = str(_CSVFILES_DIR / "SiO2_Kitamura2007.yml")):
        """Loads and prepares interpolation for Silica refractive index."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # The structure of the yaml has 'DATA' which is a list of datasets.
        # We look for tabulated n and k.
        
        # We need to find the correct data structure from the yaml
        # Typically:
        # DATA:
        #   - type: tabulated n
        #     data: |
        #       wl n
        #   - type: tabulated k
        #     data: |
        #       wl k
        
        wl_n, n_vals = [], []
        wl_k, k_vals = [], []
        
        for dataset in data.get('DATA', []):
            dtype = dataset.get('type', '')
            raw_data = dataset.get('data', '')
            
            if not raw_data:
                continue
                
            # Parse the string block of numbers
            lines = raw_data.strip().split('\n')
            parsed = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    parsed.append([float(x) for x in parts])
            parsed = np.array(parsed)
            
            if 'tabulated n' in dtype and 'k' not in dtype:
                wl_n = parsed[:, 0]  # usually in micrometers
                n_vals = parsed[:, 1]
            elif 'tabulated k' in dtype and 'n' not in dtype:
                wl_k = parsed[:, 0]
                k_vals = parsed[:, 1]
            elif 'tabulated nk' in dtype:
                wl_n = parsed[:, 0]
                n_vals = parsed[:, 1]
                wl_k = parsed[:, 0]
                k_vals = parsed[:, 2]
                
        if len(wl_n) == 0 or len(wl_k) == 0:
            raise ValueError(f"Could not parse n or k from {yaml_path}")

        # The YAML usually has wavelength in micrometers. Convert to meters for the protocol.
        wl_n_m = wl_n * 1e-6
        wl_k_m = wl_k * 1e-6

        # Create interpolators (linear in wavelength is typical)
        # Use fill_value="extrapolate" just in case we query slightly out of bounds
        self._interp_n = interp1d(wl_n_m, n_vals, bounds_error=False, fill_value="extrapolate")
        
        # Ensure k is treated as positive for Bohren-Huffman backend
        self._interp_k = interp1d(wl_k_m, np.abs(k_vals), bounds_error=False, fill_value="extrapolate")

    def __call__(self, wavelengths_m: np.ndarray) -> np.ndarray:
        """
        Evaluate refractive index at the specified wavelengths [m].

        Parameters
        ----------
        wavelengths_m : np.ndarray
            1-D array of wavelengths in meters.

        Returns
        -------
        m : np.ndarray
            Complex refractive index array (n + i k), where k is positive.
        """
        wls = np.asarray(wavelengths_m, dtype=float)
        n = self._interp_n(wls)
        k = self._interp_k(wls)

        return n + 1j * k


class RayMeanUniaxialRefractiveIndex:
    """Ray-arithmetic-mean refractive index for an optically uniaxial crystal.

    Birefringent (uniaxial) crystals have distinct ordinary (o) and
    extraordinary (e) complex indices.  For randomly-oriented small particles
    we follow Lederer (2026) and use the *ray-arithmetic-mean* convention --
    the arithmetic mean of the ordinary and extraordinary complex indices,

        m(lambda) = 1/2 [ m_o(lambda) + m_e(lambda) ],

    with n and k averaged separately.  (This is the "1/3 e + 2/3 o" family's
    equal-weight variant used by Lederer; see the calcite appendix.)  The two
    rays are supplied as CSV tables with columns
    ``Wavelength (um), Real Part, Imaginary Part`` (the Stardust calcite
    optical-set convention, also used for the Querry-derived dolomite and
    alumina tables built by scripts/convert_querry_optics.py).

    Implements RefractiveIndexProtocol: callable on wavelengths [m], returns a
    complex ndarray n + i k with k >= 0.  Material subclasses set the default
    ray tables via ``_DEFAULT_ORDINARY`` / ``_DEFAULT_EXTRAORDINARY``.
    """

    _DEFAULT_ORDINARY: str | None = None
    _DEFAULT_EXTRAORDINARY: str | None = None

    def __init__(self, ordinary_csv=None, extraordinary_csv=None):
        """Load the two rays and build ray-mean n, k interpolators.

        Parameters
        ----------
        ordinary_csv, extraordinary_csv : path-like or None
            CSV tables for the ordinary and extraordinary rays.  Default to
            the class's ``_DEFAULT_*`` files in ``CSVFiles/``.
        """
        if ordinary_csv is None:
            ordinary_csv = _CSVFILES_DIR / self._DEFAULT_ORDINARY
        if extraordinary_csv is None:
            extraordinary_csv = _CSVFILES_DIR / self._DEFAULT_EXTRAORDINARY

        wl_o, n_o, k_o = self._load_ray(ordinary_csv)
        wl_e, n_e, k_e = self._load_ray(extraordinary_csv)

        # Ray mean requires a common wavelength grid; interpolate e onto o's grid
        # (the Stardust tables share the same grid, but do not assume it).
        if wl_o.shape == wl_e.shape and np.allclose(wl_o, wl_e):
            wl = wl_o
            n = 0.5 * (n_o + n_e)
            k = 0.5 * (k_o + k_e)
        else:
            wl = wl_o
            n = 0.5 * (n_o + np.interp(wl, wl_e, n_e))
            k = 0.5 * (k_o + np.interp(wl, wl_e, k_e))

        self.wavelength_um = wl
        self.n_mean = n
        self.k_mean = np.abs(k)  # k >= 0 for the Bohren-Huffman backend

        wl_m = wl * 1e-6
        self._interp_n = interp1d(wl_m, self.n_mean, bounds_error=False,
                                  fill_value="extrapolate")
        self._interp_k = interp1d(wl_m, self.k_mean, bounds_error=False,
                                  fill_value="extrapolate")

    @staticmethod
    def _load_ray(path):
        """Read a ``Wavelength (um), Real Part, Imaginary Part`` CSV -> (um, n, k)."""
        wl, n, k = [], [], []
        with open(path, newline='') as f:
            reader = _csv.reader(f)
            next(reader)  # header
            for row in reader:
                if len(row) < 3 or not row[0].strip():
                    continue
                wl.append(float(row[0]))
                n.append(float(row[1]))
                k.append(float(row[2]))
        order = np.argsort(wl)
        return (np.asarray(wl)[order], np.asarray(n)[order], np.asarray(k)[order])

    def __call__(self, wavelengths_m: np.ndarray) -> np.ndarray:
        wls = np.asarray(wavelengths_m, dtype=float)
        return self._interp_n(wls) + 1j * self._interp_k(wls)


class CalciteRefractiveIndex(RayMeanUniaxialRefractiveIndex):
    """Ray-arithmetic-mean calcite (CaCO3) index from the Stardust calcite
    optical set (0.1-1000 um)."""

    _DEFAULT_ORDINARY = "calcite_optical_OrdinaryRay.csv"
    _DEFAULT_EXTRAORDINARY = "calcite_optical_ExtraOrdinaryRay.csv"


class DolomiteRefractiveIndex(RayMeanUniaxialRefractiveIndex):
    """Ray-arithmetic-mean dolomite (CaMg(CO3)2) index.

    Rays from Querry (1987, CRDEC-CR-88009; refractiveindex.info), 2.5-40 um
    (o) / 2.5-50 um (e), extended flat through the visible at the Handbook of
    Mineralogy Na-D indices (n_o=1.679, n_e=1.500, k=0); negative
    Kramers-Kronig k artifacts clamped to zero.  Tables built by
    scripts/convert_querry_optics.py.
    """

    _DEFAULT_ORDINARY = "dolomite_optical_OrdinaryRay.csv"
    _DEFAULT_EXTRAORDINARY = "dolomite_optical_ExtraOrdinaryRay.csv"


class AluminaRefractiveIndex(RayMeanUniaxialRefractiveIndex):
    """Ray-arithmetic-mean alumina (alpha-Al2O3, corundum/sapphire) index.

    Rays from Querry (1985, CRDC-CR-85034; refractiveindex.info), 0.21-55.6 um;
    k set to zero below 6 um where sapphire is transparent (Querry's ~2e-2
    Kramers-Kronig k floor is noise there) and negative k artifacts clamped.
    Tables built by scripts/convert_querry_optics.py.
    """

    _DEFAULT_ORDINARY = "alumina_optical_OrdinaryRay.csv"
    _DEFAULT_EXTRAORDINARY = "alumina_optical_ExtraOrdinaryRay.csv"
