"""
analyze_hitran_o3_line_uncertainty.py -- compute a defensible per-channel
"line-by-line scatter" sigma_line_rel for O3 cross sections from HITRAN error
codes.

Motivation
----------
The O3 cross section at any wavelength is a band-averaged sum over many HITRAN
lines.  HITRAN provides a per-line intensity error code (ierr digit 1; the
2nd char of the 6-char ierr string).  Convention (HITRAN 2020):

    ierr=0 unreported          ierr=4 2-5% known
    ierr=1 default (>20%)      ierr=5 1-2% known
    ierr=2 average (10-20%)    ierr=6 <1% known
    ierr=3 above average (5-10%)

The OE error model in saimon.o3_removal decomposes the spectroscopic uncertainty
into (a) a band-coherent multiplier b_band that cancels in same-band ratios,
and (b) an uncorrelated line-by-line residual sigma_line that survives band-
averaging.  This script computes the BAND-AVERAGED uncorrelated scatter at a
chosen channel from the actual per-line ierr distribution:

    sigma_avg^2 = sum_i [ (S_i * delta_i)^2 ] / ( sum_i S_i )^2

where delta_i is the representative relative intensity uncertainty for line i
(midpoint of the ierr-code range).  The sqrt(sum S_i^2 / (sum S_i)^2) factor
is the "intensity-weighted effective inverse line count": a band dominated by
a few strong lines gives a larger uncorrelated scatter than one with many
weak lines of equal strength.

We report two estimates:
  - "uncorrelated" : assume all of each ierr-code uncertainty is independent
                     per line (upper bound on sigma_line; conservative).
  - "half-coherent": assume half of each ierr-code uncertainty is already
                     captured in b_band (a more realistic split, since
                     HITRAN intensity calibrations are typically collective
                     within a band).
"""

from __future__ import annotations

import sys, io, contextlib
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# Midpoints of the HITRAN ierr-code intensity-uncertainty ranges.
# 0/1: unreported or >20% (treat as ~30% pessimistically)
IERR_MIDPOINT = {0: 0.30, 1: 0.30, 2: 0.15, 3: 0.075,
                 4: 0.035, 5: 0.015, 6: 0.005, 7: 0.005}

# Windows of interest -- a 0.25 um resolution element around each channel.
WINDOWS_UM = {
    "8.80 um (target wing, 0.25 um)": (8.675, 8.925),
    "8.80 um (target wing, 0.10 um design element)": (8.75, 8.85),
    "8.74 um (former target wing, 0.25 um)": (8.615, 8.865),
    "9.6 um (band core)"   : (9.475, 9.725),
}


def um_to_cm(lam_um: float) -> float:
    return 1.0e4 / lam_um


def load_o3_lines(table: str = "cont_o3_541_1644"):
    """Read the cached O3 HAPI table and return (nu, sw, ierr_digit_intensity)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        import hapi
        import os
        hapi.db_begin(os.environ.get("SAGE_HITRAN_CACHE", str(_HERE / "data" / "trace_gases" / "hitran_cache")))
    nu = np.asarray(hapi.getColumn(table, "nu"), float)
    sw = np.asarray(hapi.getColumn(table, "sw"), float)
    ierr = hapi.getColumn(table, "ierr")
    # 6-char string; digit 1 is the intensity error code.
    digit = np.array([int(s[1]) for s in ierr], dtype=int)
    return nu, sw, digit


def band_averaged_scatter(nu, sw, digit, nu_lo_cm, nu_hi_cm,
                          factor_coherent: float = 0.0) -> dict:
    """Compute the intensity-weighted uncorrelated scatter for lines in
    [nu_lo_cm, nu_hi_cm].

    factor_coherent in [0, 1]: fraction of each line's ierr-code uncertainty
    that is already captured in the band-coherent multiplier b_band (and so
    REMOVED from the uncorrelated residual).  factor=0 -> fully uncorrelated;
    factor=0.5 -> half coherent, half line-specific; factor=1 -> all coherent.
    """
    mask = (nu >= nu_lo_cm) & (nu <= nu_hi_cm)
    S = sw[mask]
    d_per_line = np.array([IERR_MIDPOINT[k] for k in digit[mask]])
    # Subtract the band-coherent fraction in quadrature -- everything below 0 clipped to 0.
    d_resid = np.sqrt(np.clip(d_per_line ** 2 * (1.0 - factor_coherent ** 2), 0.0, None))

    S_total = S.sum()
    sigma_avg = np.sqrt(np.sum((S * d_resid) ** 2)) / S_total if S_total > 0 else 0.0

    # Compute intensity-weighted average per-line uncertainty (for reference).
    d_weighted = (S * d_per_line).sum() / S_total if S_total > 0 else 0.0

    # Effective inverse line count (Herfindahl on S)
    herf = np.sum((S / S_total) ** 2) if S_total > 0 else 0.0

    # ierr-code distribution by intensity weight
    code_weight = {k: float(S[digit[mask] == k].sum() / S_total) if S_total > 0 else 0.0
                   for k in sorted(set(digit[mask].tolist()))}

    return {
        "n_lines": int(mask.sum()),
        "S_total": float(S_total),
        "intensity_weighted_per_line_uncert": float(d_weighted),
        "intensity_weighted_herfindahl": float(herf),
        "intensity_weighted_effective_n_lines": float(1.0 / herf) if herf > 0 else float("inf"),
        "sigma_avg_uncorrelated": float(sigma_avg),
        "code_weight_fraction": code_weight,
    }


def main():
    nu, sw, digit = load_o3_lines()
    print(f"Loaded O3 line list: {len(nu)} lines in [{nu.min():.0f}, {nu.max():.0f}] cm^-1")
    print()

    for label, (lam_lo, lam_hi) in WINDOWS_UM.items():
        if label.startswith('8.74'):
            pass
        nu_hi = um_to_cm(lam_lo)   # shorter wavelength -> higher wavenumber
        nu_lo = um_to_cm(lam_hi)
        print(f"=== {label}  ({lam_lo}-{lam_hi} um  =  {nu_lo:.1f}-{nu_hi:.1f} cm^-1) ===")
        for factor_label, factor in [("0 % band-coherent (upper bound)", 0.0),
                                     ("50% band-coherent (realistic)",    0.5),
                                     ("90% band-coherent (lower bound)",  0.9)]:
            stat = band_averaged_scatter(nu, sw, digit, nu_lo, nu_hi,
                                         factor_coherent=factor)
            tag = f"  factor_coherent={factor:.2f}  ({factor_label})"
            print(tag)
            if factor == 0.0:
                # Print the line-counting diagnostics only once per window.
                print(f"    n_lines (in window)          : {stat['n_lines']}")
                print(f"    sum(S) [cm/molecule]         : {stat['S_total']:.3e}")
                print(f"    intensity-weighted <delta>   : {stat['intensity_weighted_per_line_uncert']*100:5.2f}%"
                      f"   (mean per-line ierr-mapped uncertainty)")
                print(f"    intensity-weighted Herfindahl: {stat['intensity_weighted_herfindahl']:.3e}"
                      f"   (effective n_lines = {stat['intensity_weighted_effective_n_lines']:.1f})")
                print(f"    ierr-code distribution (by intensity fraction):")
                for k, w in stat['code_weight_fraction'].items():
                    print(f"        ierr={k}: {w*100:5.1f}% of band intensity")
            print(f"    sigma_avg_uncorrelated        : {stat['sigma_avg_uncorrelated']*100:5.2f}%"
                  f"   <- input to sigma_line_rel")
        print()

    print("=== Recommended sigma_line_rel for saimon.o3_removal ===")
    # Use the larger of the two windows' realistic-case estimates, with a safety factor.
    sigmas = []
    for lam_lo, lam_hi in WINDOWS_UM.values():
        stat = band_averaged_scatter(nu, sw, digit,
                                     um_to_cm(lam_hi), um_to_cm(lam_lo),
                                     factor_coherent=0.5)
        sigmas.append(stat["sigma_avg_uncorrelated"])
    sigma_rec = max(sigmas) * 1.5   # 50% safety margin
    print(f"  max over windows (50% coherent)        : {max(sigmas)*100:.2f}%")
    print(f"  recommended (with 50% safety margin)    : {sigma_rec*100:.2f}%   "
          f"<-- set O3RemovalSetup.sigma_line_rel ~ {sigma_rec:.3f}")


if __name__ == "__main__":
    main()
