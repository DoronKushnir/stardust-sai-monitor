"""ace_element_sampling.py -- how many independent ACE-FTS samples fall in one
design element, and the element SNR that binning alone gives (Paper 1,
Sect. sec:demo, the random-noise-margin paragraph; round 37: quoted at the
paper's 0.1-um element, formerly at the 0.25-um element).

ACE-FTS: 0.02 cm^-1 spectral resolution / point spacing (Bernath et al. 2005),
single-sample transmission SNR ~300 over most of the 8-13 um band (Bernath
2017; Boone et al. 2005).  The silica reststrahlen band spans ~8.61-8.87 um
(the ~0.26-um FWHM quoted in the paper).

Archive: outputs/ace_v52/element_sampling.json
"""
import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
OUT = _ROOT / "outputs" / "ace_v52" / "element_sampling.json"
ACE_SPACING_CM = 0.02
SNR_SAMPLE = 300.0
CENTER_UM = 8.80          # round 43 (Doron): was 8.74
BAND_UM = (8.61, 8.87)          # ~0.26-um reststrahlen band


def element(center_um, width_um):
    lo, hi = center_um - width_um / 2, center_um + width_um / 2
    nu_lo, nu_hi = 1e4 / hi, 1e4 / lo
    n = (nu_hi - nu_lo) / ACE_SPACING_CM
    return dict(width_um=width_um, lo_um=lo, hi_um=hi, nu_lo_cm=nu_lo, nu_hi_cm=nu_hi,
                span_cm=nu_hi - nu_lo, n_samples=n, snr_binned=SNR_SAMPLE * np.sqrt(n),
                n_elements_in_band=(BAND_UM[1] - BAND_UM[0]) / width_um)


def main():
    arch = {"ace_spacing_cm": ACE_SPACING_CM, "snr_per_sample": SNR_SAMPLE,
            "band_um": list(BAND_UM), "band_n_samples":
            (1e4 / BAND_UM[0] - 1e4 / BAND_UM[1]) / ACE_SPACING_CM,
            "elements": {f"{w:g}": element(CENTER_UM, w) for w in (0.25, 0.10)}}
    for k, e in arch["elements"].items():
        print(f"{k} um element at {CENTER_UM} um: {e['lo_um']:.2f}-{e['hi_um']:.2f} um = "
              f"{e['nu_lo_cm']:.1f}-{e['nu_hi_cm']:.1f} cm^-1 ({e['span_cm']:.1f} cm^-1), "
              f"{e['n_samples']:.0f} samples -> binned SNR {e['snr_binned']:.0f}; "
              f"{e['n_elements_in_band']:.1f} elements across the {BAND_UM} um band")
    print(f"band {BAND_UM} um: {arch['band_n_samples']:.0f} samples")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(arch, open(OUT, "w"), indent=1)
    print(f"archived -> {OUT}")


if __name__ == "__main__":
    main()
