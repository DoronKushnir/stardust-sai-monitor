"""
geometry.py — Solar occultation ray tracing and air mass factor computation.

Implements:
  - Spherical shell atmosphere geometry
  - Tangent height to slant path length mapping (each atmospheric layer)
  - Air mass factor matrix (used by retrieval as the A matrix)
  - Optional first-order refraction correction

All distances in metres. Earth radius R_EARTH_M is the equatorial value;
for high-precision work the geometry engine should accept a latitude-dependent
effective radius.

AGENT TASK: Implement all functions marked TODO.
The stubs define the expected signatures and array shapes — do not change them.
"""

import numpy as np
from .interfaces import OccultationGeometry, AtmosphereProfile

R_EARTH_M = 6_371_000.0          # mean Earth radius [m]
R_EARTH_EQUATOR_M = 6_378_137.0  # WGS84 equatorial radius [m]


def build_layer_grid(alt_min_m: float, alt_max_m: float,
                     n_layers: int) -> np.ndarray:
    """
    Return layer-centre altitudes on a uniform grid.

    Returns shape (n_layers,), units metres.
    """
    return np.linspace(alt_min_m, alt_max_m, n_layers)


def tangent_to_slant_paths(
    tangent_altitudes_m: np.ndarray,
    layer_altitudes_m: np.ndarray,
    earth_radius_m: float = R_EARTH_M,
) -> np.ndarray:
    """
    Compute the geometric (no refraction) path length through each shell layer
    for each ray defined by its tangent altitude.

    Parameters
    ----------
    tangent_altitudes_m : (n_tangent,) array
    layer_altitudes_m   : (n_layers,) array of layer-centre altitudes,
                          strictly increasing
    earth_radius_m      : scalar

    Returns
    -------
    path_lengths : (n_tangent, n_layers) array [m]
        path_lengths[i, j] = chord length of ray i through layer j.
        Zero if the ray's tangent altitude is above the layer.

    Algorithm
    ---------
    For a spherical shell layer bounded by [r_bot, r_top] (with r = R + z):
        half_chord = sqrt(max(r_top^2 - r_t^2, 0)) - sqrt(max(r_bot^2 - r_t^2, 0))
    where r_t = R + tangent_alt.  The factor of 2 accounts for both limbs.
    """
    n_tangent = len(tangent_altitudes_m)
    n_layers = len(layer_altitudes_m)

    # Build layer boundaries: midpoints between centres, plus extrapolated edges
    midpoints = 0.5 * (layer_altitudes_m[:-1] + layer_altitudes_m[1:])
    dz_bot = layer_altitudes_m[1] - layer_altitudes_m[0]
    dz_top = layer_altitudes_m[-1] - layer_altitudes_m[-2]
    boundaries = np.concatenate([
        [layer_altitudes_m[0] - 0.5 * dz_bot],
        midpoints,
        [layer_altitudes_m[-1] + 0.5 * dz_top],
    ])  # shape (n_layers + 1,)

    r_bot = (earth_radius_m + boundaries[:-1])[np.newaxis, :]  # (1, n_layers)
    r_top = (earth_radius_m + boundaries[1:])[np.newaxis, :]   # (1, n_layers)
    r_t = (earth_radius_m + tangent_altitudes_m)[:, np.newaxis]  # (n_tangent, 1)

    r_t2 = r_t ** 2
    half_chord = (np.sqrt(np.maximum(r_top ** 2 - r_t2, 0.0))
                  - np.sqrt(np.maximum(r_bot ** 2 - r_t2, 0.0)))

    return 2.0 * half_chord


def compute_air_mass_factors(
    path_lengths_m: np.ndarray,
    layer_thicknesses_m: np.ndarray,
) -> np.ndarray:
    """
    Air mass factor = path_length / layer_thickness for each (ray, layer) pair.

    Parameters
    ----------
    path_lengths_m      : (n_tangent, n_layers)
    layer_thicknesses_m : (n_layers,)

    Returns
    -------
    amf : (n_tangent, n_layers)
    """
    thickness = layer_thicknesses_m[np.newaxis, :]
    return np.where(thickness > 0, path_lengths_m / thickness, 0.0)


def apply_refraction_correction(
    tangent_altitudes_m: np.ndarray,
    atmosphere: AtmosphereProfile,
) -> np.ndarray:
    """
    First-order refraction correction to tangent altitudes.

    Uses the simple Edlén formula for the refractive index of air.
    Returns corrected tangent altitudes (same shape as input).

    This is a stub for future implementation — returns input unchanged.
    """
    # TODO: implement Edlen refractivity and ray-bending integral
    return tangent_altitudes_m.copy()


def build_occultation_geometry(
    tangent_altitudes_m: np.ndarray,
    layer_altitudes_m: np.ndarray,
    atmosphere: AtmosphereProfile | None = None,
    apply_refraction: bool = False,
    earth_radius_m: float = R_EARTH_M,
) -> OccultationGeometry:
    """
    Top-level geometry builder.  Returns a fully populated OccultationGeometry.

    Parameters
    ----------
    tangent_altitudes_m : (n_tangent,) measurement tangent altitudes
    layer_altitudes_m   : (n_layers,) retrieval layer-centre altitudes
    atmosphere          : needed only if apply_refraction=True
    apply_refraction    : whether to apply first-order refraction correction
    """
    if apply_refraction and atmosphere is not None:
        tangent_altitudes_m = apply_refraction_correction(
            tangent_altitudes_m, atmosphere)

    # Layer boundaries (assume uniform spacing for thickness)
    dz = layer_altitudes_m[1] - layer_altitudes_m[0]
    layer_thicknesses_m = np.full_like(layer_altitudes_m, dz)

    path_lengths = tangent_to_slant_paths(
        tangent_altitudes_m, layer_altitudes_m, earth_radius_m)
    amf = compute_air_mass_factors(path_lengths, layer_thicknesses_m)

    return OccultationGeometry(
        tangent_altitudes_m=tangent_altitudes_m,
        slant_path_lengths_m=path_lengths,
        air_mass_factors=amf,
        refraction_correction=apply_refraction,
    )


def global_shell_volume_integration(
    altitude_m: np.ndarray,
    profile: np.ndarray,
    earth_radius_m: float = R_EARTH_M,
) -> float:
    """Integrate a per-unit-volume quantity over global spherical shells.

    Computes Integral[ 4 * pi * (R + z)^2 * profile(z) dz ].
    """
    altitude_m = np.asarray(altitude_m, dtype=float)
    profile = np.asarray(profile, dtype=float)
    area_m2 = 4.0 * np.pi * (earth_radius_m + altitude_m) ** 2
    return float(np.trapezoid(profile * area_m2, altitude_m))
