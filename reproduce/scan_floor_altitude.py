"""
scan_floor_altitude.py — Trace-gas-removal noise floor of the 8.74 µm silica
channel versus tangent altitude.

Reruns the canonical B+ gas-removal OE (Chappuis 521/602/676 nm + the
validated 7.8–9.3 µm @ 0.1 µm MIR window; O3 + N2O co-retrieved;
delta_T = 2 K, sigma_line_rel = 0.2%) for tangent altitudes 12–28 km, with
the altitude grid and gas nodes extended down to 10 km so low tangents are
bracketed.  At 20 km this reproduces the published sigma(tau) = 1.96e-3 OD
(sigma_alpha = 1.5e-8 m^-1 at dz = 0.5 km) to <0.1%.

A second variant adds H2O to the co-retrieved state (band systematics 5%,
per-node prior 30%, line scatter 0.2% — same conventions as O3), to check
whether water vapour changes the floor at low tangent altitudes where the
H2O lines/continuum grow.

Output: outputs/detectability_2d/floor_altitude_scan.csv with columns
  h_tan_km, sigma_tau_OD, sigma_alpha_m1, tau_o3, tau_n2o,
  sigma_tau_h2o_OD, sigma_alpha_h2o_m1, tau_h2o
where sigma_alpha = g * sigma_tau / P_kk, g = sqrt(5 - 2 sqrt(3)) ~ 1.24
(onion-peeling amplification), P_kk = 2 sqrt(2 R dz) (0.5-km shell chord).
"""

import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_PARENT = _THIS.parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from saimon.o3_removal import (
    GasRemovalSetup,
    O3RemovalSetup,
    RetrievalGas,
    O3Band,
    o3_setup_to_gases,
    run_gas_removal_analysis,
    run_o3_removal_analysis,
)

# Round 37 (paper1): optional target-element width, e.g. `--fwhm 0.10` (the
# paper's 0.1-um element); the default 0.25 reproduces the archived scan, a
# non-default width writes a suffixed CSV (floor_altitude_scan_w0p1.csv).
import sys
FWHM_UM = 0.25
if "--fwhm" in sys.argv:
    FWHM_UM = float(sys.argv[sys.argv.index("--fwhm") + 1])
_SUF = "" if abs(FWHM_UM - 0.25) < 1e-9 else f"_w{FWHM_UM:g}".replace(".", "p")
OUT = Path(f"outputs/detectability_2d/floor_altitude_scan{_SUF}.csv")

H_TAN_KM = np.arange(12.0, 28.001, 1.0)
NODES_KM = (10.0, 14.0, 18.0, 22.0, 28.0, 40.0)
ALT_GRID = np.arange(10_000.0, 80_001.0, 500.0)

from saimon.onion_peel import G_ONION  # round 53 (RC1 M1): exact edge-grid gain from saimon.onion_peel (1.097)
P_KK = 2.0 * np.sqrt(2.0 * 6.371e6 * 500.0)          # 0.5-km shell chord [m]


def chans(lo, hi, step):
    return tuple(round(x, 3) * 1e-6 for x in np.arange(lo, hi + 1e-9, step))


IR_CHANNELS = chans(7.8, 9.3, 0.1)


def base_setup(h_tan_km: float) -> O3RemovalSetup:
    return O3RemovalSetup(
        h_tan_m=h_tan_km * 1e3,
        ir_o3_channels_m=IR_CHANNELS,
        delta_T_K=2.0,
        sigma_line_rel=0.002,
        fwhm_um_target=FWHM_UM,
        coretrieve_n2o=True,
        nodes_km=NODES_KM,
        n2o_nodes_km=NODES_KM,
        altitude_grid_m=ALT_GRID.copy(),
    )


def run_with_h2o(setup: O3RemovalSetup) -> dict:
    """Same OE with H2O added to the co-retrieved state."""
    h2o = RetrievalGas(
        name="h2o", nodes_km=NODES_KM,
        prior_sigma_rel=0.30, prior_correlation_length_km=6.0,
        bands=(O3Band("h2o", 0.05, (1e-6, 3e-5)),),
        sigma_line_rel=0.002,
    )
    gsetup = GasRemovalSetup(
        target_wavelength_m=setup.target_wavelength_m,
        channels_m=tuple(setup.all_channels_m()),
        gases=tuple(o3_setup_to_gases(setup)) + (h2o,),
        h_tan_m=setup.h_tan_m, sigma_lnT=setup.sigma_lnT,
        photon_limited_noise=setup.photon_limited_noise,
        delta_T_K=setup.delta_T_K, fwhm_um_target=setup.fwhm_um_target,
        nu_pad_cm_ir=setup.nu_pad_cm_ir, altitude_grid_m=setup.altitude_grid_m)
    return run_gas_removal_analysis(gsetup)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in H_TAN_KM:
        s = base_setup(h)
        r = run_o3_removal_analysis(s)
        rh = run_with_h2o(s)
        row = dict(
            h_tan_km=h,
            sigma_tau_OD=r["sigma_total"],
            sigma_alpha_m1=G_ONION * r["sigma_total"] / P_KK,
            tau_o3=r["tau_O3_target"],
            tau_n2o=r["tau_n2o_target"],
            sigma_tau_h2o_OD=rh["sigma_total"],
            sigma_alpha_h2o_m1=G_ONION * rh["sigma_total"] / P_KK,
            tau_h2o=rh["tau_target_by_gas"].get("h2o", 0.0),
        )
        rows.append(row)
        print(f"h_tan={h:5.1f} km: sigma_tau={row['sigma_tau_OD']:.3e} OD "
              f"-> sigma_alpha={row['sigma_alpha_m1']:.3e} m^-1 | "
              f"+H2O: {row['sigma_tau_h2o_OD']:.3e} OD "
              f"(tau_h2o={row['tau_h2o']:.3e})")

    hdr = list(rows[0])
    np.savetxt(OUT, np.array([[r[k] for k in hdr] for r in rows]),
               delimiter=",", header=",".join(hdr), comments="")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
