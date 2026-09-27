"""snr_integration_time_trade.py -- the SNR <-> integration-time trade for the
silica MIR detection channels (AirPhoton action item 4, 2026-07-14 meeting).

Vanderlei's question: affordable MIR detectors won't deliver the requested
per-sample SNR directly, only by co-adding -- so how long can the instrument
stare at a given region, and what does a lower per-sample SNR actually cost on
the retrieved silica extinction?

The two halves, coupled here for the first time:

  (A) sigma_alpha(silica) as a FUNCTION of per-sample SNR.
      The degeneracy-aware OE engine (saimon.o3_removal, silica co-fit) is run
      over a grid of sigma_lnT = 1/SNR for the two silica configs:
        B+ : target 8.80 um, channels 7.8-9.3 um @ 0.1 um, O3+N2O co-retrieval
        C  : target 20.4 um, channels 19.5-21.0 um @ 0.1 um, H2O+HNO3 co-retrieval
      The curve has a noise-limited branch (sigma ~ 1/SNR) and a systematic
      plateau (spectroscopy: line scatter + band scaling + T_eff).  The
      statistical-systematic crossover SNR_x (noise = plateau, i.e. total =
      sqrt(2) x plateau) is THE per-sample SNR requirement: photons beyond it
      buy nothing.

      TWO DISTINCT QUANTITIES (do not conflate; cf. Notes Sec. 5):
      * The GAS-REMOVAL floor -- the error on the gas OD to subtract AT the
        target channel.  It is dominated by the target-channel line-intensity
        scatter (0.2% x tau_O3 = 1.74e-3 OD at B+), is essentially SNR-flat
        (1.96e-3 at flown precision -> 1.88e-3 at zero noise), and is the
        Notes-adopted sigma_alpha = 1.5e-8 m^-1.  For THIS quantity "SNR buys
        nothing" is exactly right.
      * The DEG-AWARE SILICA AMPLITUDE error (what this script scans, and what
        detection actually uses) -- the joint band-shape fit does NOT inherit
        the target-channel line scatter: per-channel line-intensity errors are
        uncorrelated ACROSS channels (different lines in each element), so the
        broad silica shape averages them down (~x3.3 over the 16-channel B+
        band).  Its plateau is therefore lower (5.3e-4 OD = 4.1e-9 m^-1) and
        it keeps improving with per-sample precision up to SNR_x ~ 9500.
      At the flown precision (SNR 2000) the two nearly coincide (1.73e-3 vs
      1.96e-3 OD -- Notes Sec. 5 quotes both), which is why they were
      interchangeable until this scan separated them.

  (B) per-sample SNR as a function of integration time, per detector.
      The occultation geometry fixes the dwell: the tangent point descends at
      1-3 km/s, so a 0.5-km sample gets t = 0.17-0.5 s -- you cannot stare
      longer at one altitude without vertical smearing.  The ETC
      (reproduce/mir_detector_etc.py) gives each candidate detector's
      statistical SNR at that dwell; combining with (A) yields the achieved
      sigma_alpha per detector and the co-add factor (if any) needed to reach
      SNR_x.

Co-adding beyond SNR_x does not help: the plateau is spectroscopic, common to
all samples and all events, and does not average down.  Conversely a
noise-starved detector has a strong lever in vertical binning: dwell ~ dz and
the onion-peel factor ~ 1/sqrt(dz), so the noise-limited sigma_alpha ~ 1/dz
while the plateau only ~ 1/sqrt(dz).

Outputs
  outputs/snr_trade/snr_trade_<band>.csv      sigma vs SNR tables
  outputs/snr_trade/detector_dwell_880.csv    per-detector dwell-time SNR (8.80 um)
  figures/snr_integration_time_trade.png      two-panel summary figure

Run from the repo root:  python reproduce/snr_integration_time_trade.py
First run may be slow (Voigt grids); subsequent runs use the disk cache.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from saimon.sai import create_silica_sai_layer
from saimon.materials import SilicaRefractiveIndex
from saimon.geometry import tangent_to_slant_paths
from saimon.o3_removal import (GasRemovalSetup, RetrievalGas, O3Band,
                              run_gas_removal_analysis)
import mir_detector_etc as etc

TANGENT_KM = 20.0
ELEMENT_UM = 0.25
DZ_M = 500.0
R_EARTH_M = 6.371e6
from saimon.onion_peel import G_ONION  # round 53 (RC1 M1): exact edge-grid gain from saimon.onion_peel

# Per-sample SNR grid (transmission SNR per 0.1-um channel, quoted at TOA).
SNR_GRID = np.array([100., 150., 220., 320., 470., 700., 1000., 1500.,
                     2200., 3200., 4700., 7000., 10_000., 30_000.])
SNR_PLATEAU = 1e9                                  # "infinite photons" reference

# Occultation kinematics: limb descent rate envelope (Specs R7).
DESCENT_KM_S = (1.0, 2.0, 3.0)                     # -> dwell = dz / v per sample


def sigma_alpha_from_od(sigma_od, dz_m=DZ_M):
    return G_ONION * sigma_od / (2.0 * np.sqrt(2.0 * R_EARTH_M * dz_m))


def _gas(name, prior, line, band_sigma=0.05):
    return RetrievalGas(name, prior_sigma_rel=prior,
                        bands=(O3Band(name, band_sigma, (1e-6, 3e-5)),),
                        sigma_line_rel=line)


BANDS = {
    "B+ (8.80um)": dict(
        target=8.80, lo=7.8, hi=9.3, step=0.1,
        gases=(_gas("o3", 0.30, 0.002), _gas("n2o", 0.05, 0.002)),
        note="O3 9.6um blue wing + N2O; the silica/sulfate discriminator"),
    "C (20.4um)": dict(
        target=20.4, lo=19.5, hi=21.0, step=0.1,
        gases=(_gas("h2o", 0.30, 0.002), _gas("hno3", 0.30, 0.005)),
        note="silica reststrahlen peak on the H2O rotation forest + HNO3 nu9"),
}


def run_band_at_snr(band, layer, chord, snr, with_silica=True):
    lo, hi, step = band["lo"], band["hi"], band["step"]
    chans_m = tuple(round(x, 3) * 1e-6 for x in np.arange(lo, hi + 1e-9, step))
    target_m = band["target"] * 1e-6
    all_m = np.array(list(chans_m) + [target_m])
    tau_sil_all = chord @ layer.extinction_profile_m1(all_m)

    s = GasRemovalSetup(
        target_wavelength_m=target_m, channels_m=chans_m, gases=band["gases"],
        h_tan_m=TANGENT_KM * 1e3, delta_T_K=2.0, fwhm_um_target=ELEMENT_UM,
        sigma_lnT=1.0 / snr,
        silica_tau_ref=tau_sil_all if with_silica else None,
        silica_prior_sigma=1.0e3)
    r = run_gas_removal_analysis(s)
    sig_sil = r["sigma_silica_od"] if with_silica else np.nan
    return dict(
        snr=snr,
        tau_sil_target=float(tau_sil_all[-1]),
        tau_gas_target=r["tau_target_ref"],
        sigma_removal_od=r["sigma_total"],
        sigma_line_pred=r["sigma_line_pred"],
        sigma_sil_od=sig_sil,
        sigma_alpha=sigma_alpha_from_od(sig_sil),
        dofs=r["dofs"],
    )


def crossover_snr(snrs, sig, plateau):
    """SNR where the noise part equals the plateau (total = sqrt(2)*plateau)."""
    tgt = np.sqrt(2.0) * plateau
    s, y = np.asarray(snrs, float), np.asarray(sig, float)
    if y.min() > tgt or y.max() < tgt:
        return np.nan
    # sigma is decreasing in SNR: interpolate log(SNR) against log(sigma)
    order = np.argsort(y)
    return float(np.exp(np.interp(np.log(tgt), np.log(y[order]),
                                  np.log(s[order]))))


def scan_bands():
    alt_m = np.arange(14_000.0, 80_001.0, 500.0)
    chord = tangent_to_slant_paths(np.array([TANGENT_KM * 1e3]), alt_m)[0]
    # Kitamura-Popova optics (valid past 20 um), Wrana-derived PSD -- the same
    # layer the config-B+/C detection-limit studies used.
    layer = create_silica_sai_layer(
        alt_m, target_mass_tg=1.0, rmed_nm=268.0, sigma=1.31,
        refractive_index=SilicaRefractiveIndex("data/optics/SiO2_KitamuraPopova.yml"))

    results = {}
    for label, band in BANDS.items():
        print(f"\n=== {label}: {band['note']} ===")
        ref = run_band_at_snr(band, layer, chord, SNR_PLATEAU)
        plateau = ref["sigma_sil_od"]
        # The Notes-adopted gas-removal floor: predictand = gas OD at the target
        # (no silica co-fit), at the flown per-sample precision.  Line-scatter
        # dominated and SNR-flat -- the quantity for which "SNR buys nothing".
        gr = run_band_at_snr(band, layer, chord, 2000.0, with_silica=False)
        print(f"  1 Tg silica signal        : {ref['tau_sil_target']:.3e} OD"
              f"   (gas bg {ref['tau_gas_target']:.3e} OD)")
        print(f"  gas-removal floor (Notes) : {gr['sigma_removal_od']:.3e} OD"
              f"  (sigma_alpha {sigma_alpha_from_od(gr['sigma_removal_od']):.3e} 1/m;"
              f" line-scatter part {gr['sigma_line_pred']:.3e}, SNR-flat)")
        print(f"  silica-amplitude plateau  : {plateau:.3e} OD"
              f"  (sigma_alpha {sigma_alpha_from_od(plateau):.3e} 1/m;"
              f" band-shape fit averages the per-channel scatter)")
        rows = []
        for snr in SNR_GRID:
            r = run_band_at_snr(band, layer, chord, snr)
            noise = np.sqrt(max(r["sigma_sil_od"] ** 2 - plateau ** 2, 0.0))
            r["sigma_noise_od"] = noise
            rows.append(r)
            print(f"    SNR {snr:8.0f}: sigma_sil = {r['sigma_sil_od']:.3e} OD "
                  f"(noise {noise:.3e})  sigma_alpha = {r['sigma_alpha']:.3e} 1/m"
                  f"  [x{r['sigma_sil_od']/plateau:5.2f} plateau]")
        snr_x = crossover_snr([r["snr"] for r in rows],
                              [r["sigma_sil_od"] for r in rows], plateau)
        print(f"  -> statistical-systematic crossover SNR_x ~= {snr_x:.0f} "
              f"(total = sqrt(2) x plateau)")
        results[label] = dict(rows=rows, plateau=plateau, snr_x=snr_x, ref=ref,
                              gas_removal_floor=gr["sigma_removal_od"])
    return results


def detector_dwell_table(results):
    """Part B: per-detector statistical SNR at the geometric dwell (8.80 um).

    The practical per-sample requirement is min(SNR_x, systematic cap): the
    flown transmission-systematics cap (~2000: pointing, solar-disk structure,
    linearity) binds before the spectroscopic crossover when SNR_x > cap, and
    no amount of photons pushes a single sample past it.
    """
    lam = 8.80
    snr_x = results["B+ (8.80um)"]["snr_x"]
    instr = etc.Instrument()
    snr_req = min(snr_x, instr.snr_sys_cap)
    print(f"\n=== Detector SNR at the occultation dwell, {lam} um, "
          f"D={instr.D_aperture_m*100:.0f} cm, dlam={instr.dlam_um} um ===")
    print(f"  dwell per {DZ_M/1e3:.1f}-km sample: "
          + ", ".join(f"{DZ_M/1e3/v:.2f}s @{v:.0f}km/s" for v in DESCENT_KM_S))
    print(f"  spectroscopic crossover SNR_x = {snr_x:.0f}; flown transmission-"
          f"systematics cap = {instr.snr_sys_cap:.0f}")
    print(f"  -> practical per-sample requirement: detector statistical SNR >= "
          f"{snr_req:.0f} in one dwell\n")

    hdr = (f"  {'detector':34s} {'T_op':>5s} " +
           "".join(f"{'SNR@'+format(DZ_M/1e3/v,'.2f')+'s':>11s}" for v in DESCENT_KM_S)
           + f" {'t_req[s]':>9s}  verdict")
    print(hdr)
    rows_out = []
    for det in etc.DETECTORS:
        if not det.covers(lam):
            continue
        # statistical SNR at t=1 s (shot + NEP, no systematic cap); ~ sqrt(t)
        instr.t_int_s = 1.0
        r = etc.snr_for(det, lam, instr)
        snr1 = 1.0 / np.sqrt(1.0 / r["shot"] ** 2 + 1.0 / r["nep"] ** 2)
        snr_at = {v: snr1 * np.sqrt(DZ_M / 1e3 / v) for v in DESCENT_KM_S}
        t_req = (snr_req / snr1) ** 2                # time to reach snr_req
        worst = snr_at[max(DESCENT_KM_S)]
        if worst >= snr_req:
            verdict = "reaches the cap in a single fastest-case dwell"
        elif snr_at[min(DESCENT_KM_S)] >= snr_req:
            verdict = "reaches the cap only at slow descent (low-beta)"
        else:
            n_co = int(np.ceil(t_req / (DZ_M / 1e3 / max(DESCENT_KM_S))))
            verdict = f"needs x{n_co} co-add (events or dz-binning)"
        print(f"  {det.name:34s} {det.top_K:5.0f} " +
              "".join(f"{snr_at[v]:11.0f}" for v in DESCENT_KM_S) +
              f" {t_req:9.3f}  {verdict}")
        rows_out.append(dict(name=det.name, top_K=det.top_K, snr_1s=snr1,
                             t_req_s=t_req, verdict=verdict,
                             **{f"snr_{v:.0f}kms": snr_at[v] for v in DESCENT_KM_S}))
    return rows_out, snr_x, snr_req


def make_figure(results, det_rows, snr_x_874, snr_req, outpath):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # --- Panel 1: sigma_alpha vs per-sample SNR ------------------------------
    colors = {"B+ (8.80um)": "tab:red", "C (20.4um)": "tab:blue"}
    for label, res in results.items():
        snrs = np.array([r["snr"] for r in res["rows"]])
        sig = np.array([r["sigma_alpha"] for r in res["rows"]])
        pl = sigma_alpha_from_od(res["plateau"])
        gr = sigma_alpha_from_od(res["gas_removal_floor"])
        c = colors[label]
        ax1.loglog(snrs, sig, "o-", color=c, lw=1.8, ms=4,
                   label=f"{label} silica-amplitude error")
        ax1.axhline(pl, color=c, ls="--", lw=1.0, alpha=0.6)
        ax1.axhline(gr, color=c, ls="-.", lw=1.0, alpha=0.6)
        ax1.annotate("gas-removal floor (Notes §5: line-scatter, SNR-flat)",
                     (SNR_GRID[0] * 1.05, gr * 1.08), color=c, fontsize=7)
        ax1.annotate("silica-amplitude plateau (band-shape fit)",
                     (SNR_GRID[0] * 1.05, pl * 1.08), color=c, fontsize=7)
        if np.isfinite(res["snr_x"]):
            ax1.axvline(res["snr_x"], color=c, ls=":", lw=1.2, alpha=0.8)
            ax1.annotate(f"SNR$_x$={res['snr_x']:.0f}",
                         (res["snr_x"], pl * 2.6), color=c, fontsize=9,
                         rotation=90, ha="right", va="bottom")
    # Single samples cannot beat the flown transmission-systematics cap; the
    # region beyond it is reachable only by event-averaging (if the per-event
    # systematics decorrelate), down to the spectroscopic plateau.
    ax1.axvspan(2000, SNR_GRID.max() * 1.3, color="grey", alpha=0.12)
    ax1.annotate("beyond flown per-sample cap:\nevent-averaging territory",
                 (2600, ax1.get_ylim()[1] * 0.5), fontsize=8, color="dimgrey")
    for s, tag in ((1100, "R5 floor"), (2000, "R5 baseline / flown cap")):
        ax1.axvline(s, color="grey", lw=0.8, alpha=0.6)
        ax1.annotate(tag, (s, ax1.get_ylim()[0] * 1.15), color="grey",
                     fontsize=8, rotation=90, ha="right", va="bottom")
    ax1.set_xlabel("per-sample transmission SNR (per 0.1 µm channel)")
    ax1.set_ylabel(r"degeneracy-aware $\sigma_\alpha$(silica) [m$^{-1}$]"
                   f"  (dz={DZ_M/1e3:.1f} km)")
    ax1.set_title("(A) silica extinction error vs per-sample SNR\n"
                  "(dash-dot: gas-removal floor, SNR-flat; dashed: silica-"
                  "amplitude spectroscopic plateau)", fontsize=10)
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=9)

    # --- Panel 2: detector statistical SNR vs integration time (8.80 um) ----
    t = np.logspace(-2, 1.5, 100)
    for row in det_rows:
        ax2.loglog(t, row["snr_1s"] * np.sqrt(t), lw=1.6,
                   label=f"{row['name']} ({row['top_K']:.0f} K)")
    ax2.axhline(snr_x_874, color="tab:red", ls=":", lw=1.2, alpha=0.7)
    ax2.annotate(f"spectroscopic crossover SNR$_x$ = {snr_x_874:.0f}\n"
                 "(single sample can't get there — needs event-averaging)",
                 (0.012, snr_x_874 * 1.2), color="tab:red", fontsize=8)
    ax2.axhline(snr_req, color="k", ls="--", lw=1.6)
    ax2.annotate(f"practical per-sample requirement = {snr_req:.0f}\n"
                 "(flown transmission-systematics cap)",
                 (0.012, snr_req * 0.42), color="k", fontsize=8)
    dw_lo, dw_hi = DZ_M / 1e3 / max(DESCENT_KM_S), DZ_M / 1e3 / min(DESCENT_KM_S)
    ax2.axvspan(dw_lo, dw_hi, color="gold", alpha=0.25)
    ax2.annotate("dwell per 0.5-km sample\n(1–3 km/s descent)",
                 (np.sqrt(dw_lo * dw_hi), 6e5), fontsize=8, ha="center",
                 color="darkgoldenrod")
    ax2.set_xlabel("integration time [s]")
    ax2.set_ylabel("statistical transmission SNR at 8.80 µm")
    ax2.set_title("(B) detector SNR vs integration time, 8.80 µm\n"
                  "(10 cm aperture, 0.1 µm element; SNR ∝ √t)", fontsize=10)
    ax2.set_ylim(10, 4e6)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=7.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    print(f"\nSaved -> {outpath}")


def write_csvs(results, det_rows, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for label, res in results.items():
        tag = label.split()[0].replace("+", "plus").lower()
        p = outdir / f"snr_trade_{tag}.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["snr_per_sample", "sigma_silica_od", "sigma_noise_od",
                        "sigma_alpha_1_m", "sigma_removal_od", "dofs",
                        "plateau_od", "snr_crossover", "gas_removal_floor_od"])
            for r in res["rows"]:
                w.writerow([f"{r['snr']:.0f}", f"{r['sigma_sil_od']:.6e}",
                            f"{r['sigma_noise_od']:.6e}", f"{r['sigma_alpha']:.6e}",
                            f"{r['sigma_removal_od']:.6e}", f"{r['dofs']:.3f}",
                            f"{res['plateau']:.6e}", f"{res['snr_x']:.1f}",
                            f"{res['gas_removal_floor']:.6e}"])
        print(f"wrote {p}")
    p = outdir / "detector_dwell_880.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(det_rows[0].keys()))
        w.writeheader()
        w.writerows(det_rows)
    print(f"wrote {p}")


def main():
    results = scan_bands()
    det_rows, snr_x_874, snr_req = detector_dwell_table(results)
    write_csvs(results, det_rows, _PARENT / "outputs" / "snr_trade")
    make_figure(results, det_rows, snr_x_874, snr_req,
                _PARENT / "figures" / "snr_integration_time_trade.png")

    print("\n=== Bottom line for the Specs (R5/R7) ===")
    sig_at = {}                       # {label: {snr: sigma_alpha}} interpolants
    for label, res in results.items():
        pl_a = sigma_alpha_from_od(res["plateau"])
        s_arr = np.array([r["snr"] for r in res["rows"]])
        a_arr = np.array([r["sigma_alpha"] for r in res["rows"]])
        at = {s: float(np.exp(np.interp(np.log(s), np.log(s_arr),
                                        np.log(a_arr)))) for s in (1100., 2000.)}
        sig_at[label] = at
        print(f"  {label}: sigma_alpha = {at[2000.]:.2e} @SNR2000, "
              f"{at[1100.]:.2e} @SNR1100 (x{at[1100.]/at[2000.]:.2f}); "
              f"spectroscopic plateau {pl_a:.2e} 1/m (SNR_x={res['snr_x']:.0f})")
    bp = "B+ (8.80um)"
    headroom = sig_at[bp][2000.] / sigma_alpha_from_od(results[bp]["plateau"])
    print(f"""
  1. TWO quantities (Notes Sec. 5): the GAS-REMOVAL floor (line-scatter
     dominated, SNR-flat -- the Notes-adopted 1.5e-8 at B+; for it "SNR buys
     nothing" holds) vs the DEG-AWARE SILICA-AMPLITUDE error (this scan): the
     band-shape fit averages the per-channel line scatter, so it keeps
     improving ~1/SNR up to the flown per-sample cap (~2000) and plateaus
     ~x3 lower.  They nearly coincide at the flown precision.
  2. Dwell per 0.5-km sample is geometry-fixed at
     {DZ_M/1e3/max(DESCENT_KM_S):.2f}-{DZ_M/1e3/min(DESCENT_KM_S):.2f} s (limb descent 1-3 km/s) -- 'staring longer' at one
     altitude is not an option, but the detector only needs to reach the cap
     STATISTICALLY within that dwell; the table shows the margins.
  3. A noise-starved detector has two levers: vertical binning (noise
     sigma_alpha ~ 1/dz vs plateau ~ 1/sqrt(dz); 0.5->1 km buys x2 on
     noise for x1.4 on the floor, x2.8-4 if the FOV is binned optically
     so the element also collects more light) and event co-adding.
  4. If per-event transmission systematics decorrelate event-to-event,
     campaign averaging pushes below the cap toward the spectroscopic
     plateau (x{headroom:.1f} headroom at 8.80 um, ~{(snr_x_874/2000.)**2:.0f} events to exhaust it).""")


if __name__ == "__main__":
    main()
