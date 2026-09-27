"""Referee RC2 S1 (round 56): latitude dependence of the trace-gas floor from
MEASURED ozone -- the ACE-FTS v5.2 Level-2 O3 retrievals (Bernath et al. 2025,
FRDR doi:10.20383/103.01245) of the very occultations the paper's null tests
use -- instead of the six AFGL reference atmospheres of
o3_slant_latitude_proxy.py.

The profiles are read from the shipped extract
data/ace/o3_l2_v52/o3_profiles_v52.csv.gz (fetch/reduce_ace_o3_l2.py: O3 number
density = VMR x air density on the 1-km Level-2 grid, filled values dropped,
192 occultations; see data/PROVENANCE.md).  For every occultation the profile
is integrated along a straight chord above each tangent height (both halves,
the same construction as the proxy script and as the budget's onion-peel path)
and expressed as a ratio to the same integral through the profile the
trace-gas budget uses (US Standard 1976 density x AFGL US-standard O3 VMR,
saimon.atmosphere + saimon.trace_gases).  Bands are the paper's null-test
samples: 20-25N Jun-Aug (the 143-occultation floor sample, of which 101 carry
the full-spectrum null), 35-55N Jun-Sep, and 60-90S Aug-Sep (the Antarctic
vortex season).

The floor at the 8.80-um element follows the budget's split
(gas_removal_floor_880_widths.py): sigma = sqrt((sigma_line R)^2 + sigma_OE^2),
R = slant tau_O3 ratio; thresholds scale with the floor (unit slope, Table 5).

Writes outputs/o3_slant_latitude_measured.json and .txt.
"""
from __future__ import annotations
import csv, gzip, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from saimon.atmosphere import us_standard_atmosphere
from saimon import trace_gases as tg
from saimon.config import ACE, OUT

EXTRACT = ACE / "o3_l2_v52" / "o3_profiles_v52.csv.gz"
R_E = 6371.0
H_TAN = np.arange(12.0, 28.001, 0.5)
Z_FINE = np.arange(12.0, 80.0, 0.05)
BANDS = {"tropical 20-25N Jun-Aug": (20.0, 25.0, (6, 7, 8)),
         "midlat 35-55N Jun-Sep": (35.0, 55.0, (6, 7, 8, 9)),
         "antarctic 60-90S Aug-Sep": (-90.0, -60.0, (8, 9))}
BAND_LAT = {"tropical 20-25N Jun-Aug": 22.5, "midlat 35-55N Jun-Sep": 45.0, "antarctic 60-90S Aug-Sep": 75.0}


def read_extract():
    """{occ: (lat, month, z [km], O3 number density [m^-3])} from the shipped extract."""
    prof = {}
    with gzip.open(EXTRACT, "rt") as fh:
        for r in csv.DictReader(fh):
            d = prof.setdefault(r["occultation"], [float(r["latitude_deg"]), int(r["month"]), [], []])
            d[2].append(float(r["z_km"])); d[3].append(float(r["o3_number_density_m3"]))
    return {o: (lat, month, np.array(z), np.array(n)) for o, (lat, month, z, n) in prof.items()}


def slant_column(z, n, h):
    """straight-chord slant column above tangent height h [km], both halves, units of n x km"""
    zz = Z_FINE[Z_FINE >= h]
    nn = np.interp(zz, z, n, left=np.nan, right=0.0)
    ok = np.isfinite(nn)
    zz, nn = zz[ok], nn[ok]
    ds = (R_E + zz) / np.sqrt(np.maximum((R_E + zz) ** 2 - (R_E + h) ** 2, 1e-9))
    ds[0] = ds[1] if len(ds) > 1 else ds[0]
    return 2.0 * np.trapezoid(nn * ds, zz)


def main():
    # reference: the profile the budget uses
    atm = us_standard_atmosphere(Z_FINE * 1e3)
    n_ref = tg.number_density_profiles(Z_FINE * 1e3, atm.number_density_m3, gases=["o3"])["o3"]
    tau_ref = np.array([slant_column(Z_FINE, n_ref, h) for h in H_TAN])
    n_ref_local = np.interp(H_TAN, Z_FINE, n_ref)

    per_occ = {}
    for occ, (lat, month, z, n) in sorted(read_extract().items()):
        # tangent heights below the retrieved profile's bottom (cloud cut-off) are left NaN
        tau = np.array([slant_column(z, n, h) if h >= z.min() else np.nan for h in H_TAN])
        per_occ[occ] = dict(lat=lat, month=month, ratio=(tau / tau_ref).tolist(),
                            local_ratio=(np.interp(H_TAN, z, n) / n_ref_local).tolist())

    # budget split at the 8.80-um element (round 43 archive)
    row = next(r for r in csv.DictReader((OUT / "round43_tracegas/gas_removal_floor_widths.csv").open())
               if float(r["target_um"]) == 8.8 and float(r["width_um"]) == 0.1)
    s_tot, s_line = float(row["sigma_removal_od"]), float(row["sigma_line_pred"])
    s_oe = float(np.sqrt(max(s_tot**2 - s_line**2, 0.0)))
    tau_budget_20 = float(row["tau_gas_target"])
    thr = json.load(open(OUT / "calibrated_background_thresholds.json"))
    cq = thr["results"]["CALIBRATED quiet (LM65T223, 2-comp)"]
    base = {"band": cq["band 8-13 @0.1: measured floors"]["full 7-param"]["triplet + band01"]["mmin_tg"],
            "window": cq["window 7.8-9.3 @0.1: measured floors"]["full 7-param"]["triplet + window"]["mmin_tg"]}

    def floor_factor(R):
        return float(np.hypot(s_line * R, s_oe) / s_tot)

    out = dict(source="ACE-FTS v5.2 Level-2 O3 (Bernath et al. 2025, doi:10.20383/103.01245), external/ace_v52/l2_asc",
               reference="US Standard 1976 density x AFGL US-standard O3 VMR (sage3), the budget's profile",
               budget_od_880=dict(total=s_tot, line=s_line, oe=s_oe, tau_o3_20km=tau_budget_20),
               baselines_tg=base, h_tan_km=H_TAN.tolist(), bands={}, per_occultation_n=len(per_occ))
    lines = [f"ACE v5.2 Level-2 O3 slant ratios to the budget's US-standard profile ({len(per_occ)} occultations with L2 files)",
             "band | n | h_tan: " + " ".join(f"{h:g}" for h in H_TAN[::4])]
    for name, (lo, hi, months) in BANDS.items():
        sel = [o for o, d in per_occ.items() if lo <= d["lat"] <= hi and d["month"] in months]
        if not sel:
            continue
        rat = np.array([per_occ[o]["ratio"] for o in sel])
        loc = np.array([per_occ[o]["local_ratio"] for o in sel])
        med, q25, q75 = np.nanmedian(rat, 0), np.nanpercentile(rat, 25, 0), np.nanpercentile(rat, 75, 0)
        n_h = np.isfinite(rat).sum(0)
        ff = np.array([floor_factor(r) for r in med])
        out["bands"][name] = dict(n=len(sel), lat_deg=BAND_LAT[name],
                                  lat_range=[float(min(per_occ[o]["lat"] for o in sel)), float(max(per_occ[o]["lat"] for o in sel))],
                                  months=sorted({per_occ[o]["month"] for o in sel}),
                                  ratio_median=med.tolist(), ratio_q25=q25.tolist(), ratio_q75=q75.tolist(),
                                  local_ratio_median=np.nanmedian(loc, 0).tolist(), n_per_altitude=n_h.tolist(),
                                  floor_factor=ff.tolist(),
                                  threshold_band_tg=(base["band"] * ff).tolist(),
                                  threshold_window_tg=(base["window"] * ff).tolist())
        lines.append(f"{name} | {len(sel)} | ratio: " + " ".join(f"{r:.2f}" for r in med[::4]) +
                     " | floor: " + " ".join(f"{f:.2f}" for f in ff[::4]))
    # headline rows for the text
    lines.append("")
    lines.append("h_tan  " + "  ".join(f"{n[:8]:>12s}" for n in out["bands"]))
    for j, h in enumerate(H_TAN):
        if h in (16, 18, 20, 22, 24):
            lines.append(f"{h:5.1f}  " + "  ".join(f"{b['ratio_median'][j]:5.2f} [{b['ratio_q25'][j]:.2f}-{b['ratio_q75'][j]:.2f}] n={b['n_per_altitude'][j]}"
                                                    for b in out["bands"].values()))
    lines.append("floor factor at 20 km: " + ", ".join(f"{n}: {b['floor_factor'][list(H_TAN).index(20.0)]:.2f}" for n, b in out["bands"].items()))
    lines.append("band threshold at 20 km [Tg]: " + ", ".join(f"{n}: {b['threshold_band_tg'][list(H_TAN).index(20.0)]:.3f}" for n, b in out["bands"].items()))
    lines.append("window threshold at 20 km [Tg]: " + ", ".join(f"{n}: {b['threshold_window_tg'][list(H_TAN).index(20.0)]:.3f}" for n, b in out["bands"].items()))
    j18 = list(H_TAN).index(18.0)
    trop = out["bands"]["tropical 20-25N Jun-Aug"]
    lines.append(f"tropical slant ratio at 18 km: {trop['ratio_median'][j18]:.2f} (local density ratio {trop['local_ratio_median'][j18]:.2f}); Sect. 4.1 atlas fit s = 0.37 at the 16-20 km bin")
    (OUT / "o3_slant_latitude_measured.json").write_text(json.dumps(out, indent=1))
    (OUT / "o3_slant_latitude_measured.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
