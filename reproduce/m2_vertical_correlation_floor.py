"""Referee RC1 M2 (round 54): the design's 8.80-um floor under an explicit
vertical error-correlation model, instead of the white-noise conversion of the
whole optical-depth budget.

The calculated R~100 budget at the 8.80-um 0.1-um element (gas_removal_floor_880_widths.py)
is sigma_removal = sqrt(sigma_line^2 + sigma_OE^2) in slant optical depth.  Its two
parts convert to a per-shell extinction differently:
  * sigma_OE (photon / measurement noise, independent between tangent heights):
      white-noise onion peel, sigma_alpha = |G_k| sigma_OE   (|G_k| = g / P_kk, saimon.onion_peel)
  * sigma_line (HITRAN line-intensity scatter carried by the same lines at every
      tangent height, i.e. vertically coherent): exact peel of a coherent fractional
      error f on every slant tau_O3 gives f * alpha_O3(z) -- computed here by inverting
      the edge-grid path matrix on the US-standard O3 profile normalised to the
      budget's tau_O3 at the 20-km tangent (same construction as onion_peel_geometry_exact.py).
The measured ACE-FTS floor (across-occultation scatter at fixed tangent bins) contains
no coherent database term by construction; it bounds the non-coherent part of any
instrument's floor from below and is converted white.

Writes outputs/m2_vertical_correlation_floor.json and prints the numbers quoted in
Appendix D of Paper 1 (round 54).
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from saimon.onion_peel import M1_PER_OD, G_ONION, P_KK_M, R_EARTH_M, DZ_M, BUDGET_OD_880
from saimon.atmosphere import us_standard_atmosphere
from saimon import trace_gases as tg

# --- the budget's parts (archived CSV of gas_removal_floor_880_widths.py) ------------
row = next(r for r in csv.DictReader((ROOT / "outputs/round43_tracegas/gas_removal_floor_widths.csv").open())
           if float(r["target_um"]) == 8.8 and float(r["width_um"]) == 0.1)
sig_tot = float(row["sigma_removal_od"]); sig_line = float(row["sigma_line_pred"]); tau_o3 = float(row["tau_gas_target"])
sig_oe = float(np.sqrt(max(sig_tot**2 - sig_line**2, 0.0)))
f_line = sig_line / tau_o3

# --- exact coherent peel on the edge grid ---------------------------------------------
zb = np.arange(0.0, 60_000.0 + DZ_M, DZ_M); rb, rt = R_EARTH_M + zb[:-1], R_EARTH_M + zb[1:]
P = np.empty((len(zb) - 1, len(zb) - 1))
for i, zt in enumerate(zb[:-1]):
    r2 = (R_EARTH_M + zt) ** 2
    P[i] = 2.0 * (np.sqrt(np.maximum(rt**2 - r2, 0)) - np.sqrt(np.maximum(rb**2 - r2, 0)))
G = np.linalg.inv(P); k = int(np.argmin(np.abs(zb[:-1] - 20_000.0)))
zc = 0.5 * (zb[:-1] + zb[1:])
atm = us_standard_atmosphere(zc)
n_o3 = tg.number_density_profiles(zc, atm.number_density_m3, gases=["o3"])["o3"]
tau = P @ n_o3; scale = tau_o3 / tau[k]; tau *= scale; alpha_o3 = n_o3 * scale
coh = (G @ (f_line * tau))[k]                     # == f_line * alpha_o3[k]
white_line = M1_PER_OD * sig_line
white_oe = M1_PER_OD * sig_oe
white_total = M1_PER_OD * sig_tot                 # what the design floors adopt (SIG_880_ABS)
mixed = float(np.hypot(white_oe, coh))

# --- the measured ACE floor (non-coherent by construction) -------------------------------
thr = json.load(open(ROOT / "outputs/calibrated_background_thresholds.json"))
sd_meas = thr["floor_anchor"]["measured_sd_880_od"]
meas_white = M1_PER_OD * sd_meas
mixed_meas = float(np.hypot(meas_white, coh))     # measured non-coherent part + coherent database term

# --- thresholds scale linearly with the floors (Table 5 floor scan, unit slope) ---------
base = {"band": thr["results"]["CALIBRATED quiet (LM65T223, 2-comp)"]["band 8-13 @0.1: measured floors"]["full 7-param"]["triplet + band01"]["mmin_tg"],
        "window": thr["results"]["CALIBRATED quiet (LM65T223, 2-comp)"]["window 7.8-9.3 @0.1: measured floors"]["full 7-param"]["triplet + window"]["mmin_tg"]}
out = dict(budget_od=dict(total=sig_tot, line=sig_line, oe=sig_oe, tau_o3=tau_o3, f_line=f_line),
           conversion_m1=dict(white_total=white_total, white_line=white_line, white_oe=white_oe,
                              coherent_line=float(coh), mixed_oe_white_line_coherent=mixed,
                              measured_white=meas_white, mixed_measured_white_plus_coherent=mixed_meas,
                              g=float(G_ONION), m1_per_od=float(M1_PER_OD)),
           ratios=dict(mixed_over_adopted=mixed / white_total, mixed_meas_over_adopted=mixed_meas / white_total,
                       white_over_coherent_line=white_line / coh),
           thresholds_tg={k2: dict(adopted=v, mixed=v * mixed / white_total, mixed_measured=v * mixed_meas / white_total) for k2, v in base.items()})
(ROOT / "outputs/m2_vertical_correlation_floor.json").write_text(json.dumps(out, indent=1))
print(f"budget at 8.80 um: total {sig_tot:.3e} OD = line {sig_line:.3e} (+) OE {sig_oe:.3e}; tau_O3 {tau_o3:.3f}, f = {f_line:.4%}")
print(f"white conversion of the whole budget (adopted design floor): {white_total:.3e} m^-1")
print(f"  OE part, white: {white_oe:.3e};  line part, white: {white_line:.3e};  line part, coherent (f alpha_O3(20 km)): {coh:.3e}  (white/coherent = {white_line/coh:.1f})")
print(f"mixed conversion (OE white + line coherent): {mixed:.3e} m^-1 = {mixed/white_total:.3f} x adopted")
print(f"measured ACE floor {sd_meas:.3e} OD (non-coherent) white: {meas_white:.3e}; + coherent database term: {mixed_meas:.3e} = {mixed_meas/white_total:.3f} x adopted")
for k2, v in out["thresholds_tg"].items():
    print(f"  {k2}: adopted {v['adopted']:.3f} Tg -> mixed {v['mixed']:.3f} Tg; measured-plus-coherent {v['mixed_measured']:.3f} Tg")
