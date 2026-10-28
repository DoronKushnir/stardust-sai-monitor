"""adopted_latitude.py -- round 75 (paper 1; public port, referee round 6 items 2, 5 and 9, left open in
round 74): the altitude scan, the latitude-resolved maps (Sect. 6, Fig. 5, Table 7) and the campaign
gains recomputed under the ADOPTED measurement model (Table 2) through
reproduce/adopted_model_engine.py, so that the maps inherit the covariance of the primary
sensitivity calculation instead of the diagonal-floor scenario.

Construction (the 2-D machinery of detectability_2d_calibrated.py, with the threshold engine replaced):
  * sigma_A(z) of the joint retrieval at tangent heights 12-28 km (0.5-km steps; the line-resolved
    model, the aerosol columns, the triplet rows and the covariance all re-evaluated per tangent);
  * the background amplitude g(phi, z) of GloSSAC (median quiet-epoch extinction relative to 22.5 N)
    enters the forward model (the sulfate optical depth, the Level-2 fractional errors): the ratio
    sigma(z, g)/sigma(z, 1) is tabulated on g = 0.25 ... 4 and interpolated in ln g;
  * the measured ozone load (ACE-FTS v5.2 slant ratios of outputs/o3_slant_latitude_measured.json:
    20-25 N, 35-55 N, 60-90 S bands, per tangent height) scales the O3 profile in the forward model
    (the gas optical depth, the line responses and the radiometric term follow); applied
    symmetrically in latitude, interpolated between the band latitudes and held beyond them, as the
    'measured-ozone rescaled' scenario of round 56;
  * the discrepancy term constant with altitude (the model as stated) or scaled by the measured ACE
    altitude bins (the 'measured altitude-dependent floor' scenario);
  * the map M(phi, z) = 3 sigma_A(z, phi) rho(z) [t_shell(z)/t_shell(20 km)] / f(phi, z), i.e. the
    per-shell uncertainty at z in units of the 20-km extinction per Tg of the homogeneous layer,
    divided by the local enhancement of the Lederer (2026) transport field (f2d of map_pm41.npz);
  * Table 7 (fixed 20.47-km level and best height, 0.1-Tg coverage) for the four scenarios under the
    adopted model beside the diagonal-floor rows; the independent-weight campaign gains of the two
    orbits; Fig. 5 redrawn from the adopted map.

Run from the repo root (about 10 min):  python reproduce/adopted_latitude.py
Writes outputs/adopted_latitude.json, outputs/detectability_2d_calibrated/map_pm41_adopted.npz,
tables/latitude_scenarios.tex and figures/detectability_2d_mass_contours_adopted.png.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reproduce"))

import adopted_model_engine as E                                  # noqa: E402
from detectability_2d_calibrated import (glossac_quiet_median, amplitude_factor, LAT_ANCHOR, FOOT_DEG, ORBITS,   # noqa: E402
                                         CONTOURS_TG, OUT_DIR as D2D, _style)
from saimon.sai_2d import load_silica_zonal_2d, smooth_over_footprint, simulate_occultation_latitudes   # noqa: E402

C, cbt = E.C, E.cbt
OUT = ROOT / "outputs/adopted_latitude.json"
MAP = D2D / "map_pm41_adopted.npz"
TEX = ROOT / "tables/latitude_scenarios.tex"
FIG = ROOT / "figures/detectability_2d_mass_contours_adopted.png"
O3 = json.load(open(ROOT / "outputs/o3_slant_latitude_measured.json"))
Z_KM = np.arange(12.0, 28.001, 0.5)
G_SCAN = (0.25, 0.5, 1.0, 2.0, 4.0)
Z_TABLE = 20.47
log = E.log


def o3_bands():
    h = np.array(O3["h_tan_km"], float)
    nodes = sorted((float(b["lat_deg"]), np.array(b["ratio_median"], float)) for b in O3["bands"].values())
    return h, np.array([n[0] for n in nodes]), np.array([n[1] for n in nodes])     # (n_band,), (n_band, n_h)


def main():
    t0 = time.time()
    h_o3, lat_o3, rat_o3 = o3_bands()
    # ---- 1. sigma(z) and its g- and O3-ratios ---------------------------------------------------
    t20 = float(np.mean(C.shell_c[0][E._coarse_masks[C.i880]]))                  # 20-km shell extinction per Tg at 8.80 um
    tab = dict(z_km=Z_KM.tolist(), g_scan=list(G_SCAN), o3_band_lat=lat_o3.tolist())
    sig = {k: np.zeros(len(Z_KM)) for k in ("band_const", "band_atlas", "window_const", "window_atlas")}
    conv = np.zeros(len(Z_KM))                                                  # rho(z) t_shell(z)/t20: slant Tg -> 20-km-reference per-shell Tg
    rho = np.zeros(len(Z_KM)); y880 = np.zeros(len(Z_KM))
    g_ratio = {k: np.zeros((len(Z_KM), len(G_SCAN))) for k in ("band", "window")}
    o3_ratio = {k: np.zeros((len(Z_KM), len(lat_o3))) for k in ("band", "window")}
    BG = E.bg_profiles_full()                                                   # sulfate below 16 km as the 2-D maps (tangents below 16 km)

    def shell_template(z):
        return dict(bg_profiles=BG, **E.shell_template(z * 1e3))
    for j, z in enumerate(Z_KM):
        tm = E.TangentModel(z * 1e3, **shell_template(z))
        # the silica column is the 20-km reference spectrum in the tangent shell in the paper's per-shell convention (engine
        # shell_template), so sigma_A is the per-shell mass directly; conv records the check K_880 / (OD_PER_M1 t20) = 1 up to the
        # transmission weighting of the element mean
        conv[j] = 1.0; rho[j] = float(tm.K_mir[C.i880, E.I_SIL]) / (E.OD_PER_M1 * t20); y880[j] = tm.y0[C.i880]
        for rows in ("band", "window"):
            sig[f"{rows}_const"][j] = E.joint_sigma(tm, rows=rows)
            sig[f"{rows}_atlas"][j] = E.joint_sigma(tm, S_mir=tm.cov_mir("atlas"), rows=rows)
            for k, g in enumerate(G_SCAN):
                g_ratio[rows][j, k] = 1.0 if g == 1.0 else E.joint_sigma(E.TangentModel(z * 1e3, g_bg=g, line_cache=tm.Pm, **shell_template(z)), rows=rows) / sig[f"{rows}_const"][j]
            for b in range(len(lat_o3)):
                s = float(np.interp(z, h_o3, rat_o3[b]))
                tmo = E.TangentModel(z * 1e3, o3_scale=s, line_cache={g: (tm.Pm[g] * s if g == "o3" else tm.Pm[g]) for g in tm.Pm}, **shell_template(z))
                o3_ratio[rows][j, b] = E.joint_sigma(tmo, rows=rows) / sig[f"{rows}_const"][j]
        log(f"  z {z:4.1f} km: band 3 sigma per shell {3*sig['band_const'][j]*rho[j]:.4f} Tg (atlas {3*sig['band_atlas'][j]*rho[j]:.4f}), "
            f"20-km units {3*sig['band_const'][j]*conv[j]:.4f} (atlas {3*sig['band_atlas'][j]*conv[j]:.4f}); g-ratios {np.round(g_ratio['band'][j], 3)}; O3 ratios {np.round(o3_ratio['band'][j], 3)}; {time.time()-t0:.0f} s")
    tab.update(sigma_slant_tg={k: v.tolist() for k, v in sig.items()}, conv_to_20km_units=conv.tolist(), conv_check=rho.tolist(), y880=y880.tolist(),
               mmin_per_shell_at_z={k: (3 * v * rho).tolist() for k, v in sig.items()},
               mmin_20km_units={k: (3 * v * conv).tolist() for k, v in sig.items()},
               g_ratio={k: v.tolist() for k, v in g_ratio.items()}, o3_ratio={k: v.tolist() for k, v in o3_ratio.items()})
    # separability check at 20 km: sigma(g=2, o3 band 0) against the product of the ratios
    j20 = int(np.argmin(np.abs(Z_KM - 20.0))); tm = E.TangentModel(20e3)
    s_b = float(np.interp(20.0, h_o3, rat_o3[0]))
    both = E.joint_sigma(E.TangentModel(20e3, g_bg=2.0, o3_scale=s_b, line_cache={g: (tm.Pm[g] * s_b if g == "o3" else tm.Pm[g]) for g in tm.Pm}, **shell_template(20.0)))
    tab["check_20km_prescribed_vs_shell_template"] = dict(shell_template_tg=3 * sig["band_const"][j20] * conv[j20], prescribed_profile_tg=float(E.threshold(E.REF)),
                                                           shell_template_profile_tg=3 * E.joint_sigma(E.TangentModel(20e3, **E.shell_template(20e3))))
    tab["separability_20km"] = dict(joint=both / sig["band_const"][j20], product=g_ratio["band"][j20, 3] * o3_ratio["band"][j20, 0])

    # ---- 2. sigma(phi, z) on the GloSSAC latitude grid ----------------------------------------
    lat_g, alt_g, med = glossac_quiet_median()
    g_grid = amplitude_factor(lat_g, alt_g, med)
    lng = np.log(np.array(G_SCAN))

    def sigma_map(rows, disc, o3lat):
        arr = np.empty((len(lat_g), len(Z_KM)))
        for j, z in enumerate(Z_KM):
            base = sig[f"{rows}_{disc}"][j] * conv[j]
            for i, lat in enumerate(lat_g):
                g = float(np.interp(z, alt_g, g_grid[i]))
                rg = float(np.interp(np.log(np.clip(g, G_SCAN[0], G_SCAN[-1])), lng, g_ratio[rows][j]))
                ro = float(np.interp(abs(lat), lat_o3, o3_ratio[rows][j])) if o3lat else 1.0
                arr[i, j] = base * rg * ro
        return arr                                                                   # slant sigma in 20-km-reference per-shell units (Tg)
    models = {("band", "const", False): "band_adopted_const", ("band", "atlas", False): "band_adopted_atlas",
              ("band", "const", True): "band_adopted_const_o3lat", ("band", "atlas", True): "band_adopted_atlas_o3lat",
              ("window", "const", False): "window_adopted_const", ("window", "const", True): "window_adopted_const_o3lat"}
    smap = {name: sigma_map(*key) for key, name in models.items()}

    # ---- 3. maps on the transport field (f2d of the archived +-41 deg map) -------------------
    d0 = np.load(D2D / "map_pm41.npz")
    lat, zlev, f2d = d0["lat_deg"], d0["z_lev_km"], d0["f2d"]
    in_rng = (zlev >= Z_KM[0]) & (zlev <= Z_KM[-1])
    save = dict(lat_deg=lat, z_lev_km=zlev, f2d=f2d, conc_kg_m3=d0["conc_kg_m3"], total_mass_kg=d0["total_mass_kg"], q_obsc=d0["q_obsc"])
    maps = {}
    for name, arr in smap.items():
        s_lat = np.array([np.interp(lat, lat_g, arr[:, j]) for j in range(len(Z_KM))]).T
        s_lev = np.array([np.interp(zlev, Z_KM, s_lat[i], left=np.nan, right=np.nan) for i in range(len(lat))])
        with np.errstate(divide="ignore", invalid="ignore"):
            m = 3 * s_lev / f2d
        save[f"mmin_{name}"] = m
        maps[name] = m
    np.savez(MAP, **save)

    def table_row(m):
        k = int(np.argmin(np.abs(zlev - Z_TABLE)))
        m20 = m[:, k]
        mm = np.where(np.isfinite(m) & in_rng[None, :], m, np.inf); best = mm.min(axis=1)

        def cov(v):
            ok = np.isfinite(v) & (v <= 0.1)
            return None if not ok.any() else (float(lat[ok].min()), float(lat[ok].max()), float(ok.mean()))
        return dict(z20_level=float(zlev[k]), fixed20_min=float(np.nanmin(m20)), fixed20_max=float(np.nanmax(m20)), fixed20_cov=cov(m20),
                    best_min=float(best.min()), best_worst=float(best[np.isfinite(best)].max()), best_cov=cov(best),
                    core_lat=float(lat[np.nanargmin(m20)]))
    rows_adopted = {name: table_row(m) for name, m in maps.items()}
    scen_old = {"reference-atmosphere floor": ("map_pm41.npz", "mmin_band_quiet_oe"), "measured altitude-dependent floor": ("map_pm41.npz", "mmin_band_quiet_atlas"),
                "measured-ozone rescaled floor": ("map_pm41_o3lat.npz", "mmin_band_quiet_oe"), "measured-ozone + measured floor": ("map_pm41_o3lat.npz", "mmin_band_quiet_atlas")}
    rows_scen = {}
    for scen, (fn, key) in scen_old.items():
        z_ = np.load(D2D / fn); m = z_[key]
        if m.shape[0] != len(z_["lat_deg"]):
            m = m.T
        rows_scen[scen] = table_row(m)

    # ---- 4. campaign gains (independent weights) at the table level ---------------------------
    sz = load_silica_zonal_2d(inj_lat_deg=[-41.0, 41.0], inj_pres_hpa=51.0)
    f_eff = smooth_over_footprint(sz.lat_deg, sz.enhancement, FOOT_DEG)
    kz = int(np.argmin(np.abs(zlev - Z_TABLE)))
    events = {o: simulate_occultation_latitudes(mission_days=365.0, **kw) for o, kw in ORBITS.items()}
    gains = {}
    for name in ("band_adopted_const", "band_adopted_const_o3lat", "band_adopted_atlas", "band_adopted_atlas_o3lat"):
        arr = smap[name]
        s20 = np.interp(sz.lat_deg, lat_g, np.array([np.interp(Z_TABLE, Z_KM, arr[i]) for i in range(len(lat_g))]))
        m_lat = 3 * s20 / f_eff
        g_ = dict(best_mmin=float(np.nanmin(m_lat)), best_lat=float(sz.lat_deg[np.nanargmin(m_lat)]), f_peak=float(np.nanmax(f_eff)),
                  homogeneous_at_anchor=float(3 * np.interp(LAT_ANCHOR, lat_g, np.array([np.interp(Z_TABLE, Z_KM, arr[i]) for i in range(len(lat_g))]))))
        for o, ev in events.items():
            f_e = np.interp(ev.lat_deg, sz.lat_deg, f_eff); s_e = np.interp(ev.lat_deg, sz.lat_deg, s20)
            g_[f"gain_{o}"] = float(np.sqrt(np.sum(f_e ** 2 / s_e ** 2) / np.sum(1.0 / s_e ** 2)))
            g_[f"campaign_sigma_{o}_Tg"] = float(1.0 / np.sqrt(np.sum(f_e ** 2 / s_e ** 2)))
            g_[f"events_{o}"] = int(len(ev.day))
        gains[name] = g_
    res = dict(script="reproduce/adopted_latitude.py", round=75, altitude=tab, table7_adopted=rows_adopted, table7_scenario=rows_scen,
               campaign=gains, atlas_ratio=E.ATLAS_RATIO, map=str(MAP.relative_to(ROOT)), runtime_s=time.time() - t0,
               note="M(phi,z) = 3 sigma_A(z,phi) / f(phi,z), sigma_A the per-shell silica uncertainty of the joint retrieval under the adopted model at each tangent (20-km reference spectrum in the tangent shell, edge-bounded path over the onion gain), sulfate background on the full GloSSAC profile; "
                    "g(phi,z) and the measured O3 ratios enter the forward model, tabulated as ratios and interpolated")
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    log("archived ->", OUT)
    write_table(rows_adopted, rows_scen)
    plot(maps, save, rows_adopted)


def rng(c):
    return "all latitudes" if c and c[2] >= 0.999 else (f"{-c[0]:.0f}$^\\circ$S--{c[1]:.0f}$^\\circ$N" if c else "none")


def write_table(ra, rs):
    old = TEX.read_text().split("\n")
    sup = {"reference-atmosphere floor": "US-standard ozone at every latitude; floors re-derived per tangent altitude",
           "measured altitude-dependent floor": "ACE 20--25$^\\circ$N Jun--Aug, 16--19 / 19--22 / 22--25 km",
           "measured-ozone rescaled floor": "ACE O$_3$ at 20--25$^\\circ$N, 35--55$^\\circ$N, 60--90$^\\circ$S; symmetric, year-round",
           "measured-ozone + measured floor": "both of the above"}
    sup_a = {"band_adopted_const": ("reference atmosphere, discrepancy term constant with altitude", "US-standard ozone at every latitude; the spectroscopic terms re-evaluated per tangent"),
             "band_adopted_atlas": ("reference atmosphere, discrepancy term at the measured altitude bins", "ACE 20--25$^\\circ$N Jun--Aug, 16--19 / 19--22 / 22--25 km"),
             "band_adopted_const_o3lat": ("measured ozone, discrepancy term constant", "ACE O$_3$ at 20--25$^\\circ$N, 35--55$^\\circ$N, 60--90$^\\circ$S in the forward model; symmetric, year-round"),
             "band_adopted_atlas_o3lat": ("measured ozone + measured altitude bins", "both of the above")}
    L = ["% Round 75: regenerated by reproduce/adopted_latitude.py (outputs/adopted_latitude.json): adopted-model rows added above the",
         "% diagonal-floor rows (the latter re-read from the archived maps as in paper1_latitude_scenarios_round67.py). The round-74 file is kept below as comments.",
         "\\begin{table*}[t]", "\\centering",
         "\\caption{Single-sounding $3\\sigma$ sensitivity to $0.1$~Tg of silica in the $\\pm41^\\circ$ sustained-injection field (band configuration). ``Fixed $20$~km'' is the model level at $51$~hPa ($20.47$~km, the injection level, ``the $20.5$-km level'' of the text) alone; ``best height'' takes the most sensitive bin at each latitude. Coverage is the latitude range over which the threshold is below $0.1$~Tg; ``worst'' is the largest best-height threshold on the map. The latitude and season of the measurements behind each scenario are given in the second column; outside them the scenario is an assumption, not a measurement. Whether the orbit observes a latitude is a separate question (Sect.~\\ref{sec:2d}).\\ClaudeText{ Top block: the adopted measurement model of Table~\\ref{tab:error_budget} evaluated at every tangent height, with the background amplitude and the ozone load in the forward model ($0.119$~Tg at $20$~km). Bottom block: the diagonal-floor scenario ($0.101$~Tg at $20$~km), in which the floors are scaled per altitude and latitude.}}",
         "\\label{tab:latitude_scenarios}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{2.8cm}>{\\raggedright\\arraybackslash}p{4.2cm}>{\\raggedright\\arraybackslash}p{1.9cm}>{\\raggedright\\arraybackslash}p{2.0cm}>{\\raggedright\\arraybackslash}p{1.9cm}>{\\raggedright\\arraybackslash}p{2.0cm}}",
         "\\toprule", "Scenario & Measurement support & Fixed $20$~km: range (Tg) & Fixed $20$~km: $<0.1$~Tg & Best height: worst (Tg) & Best height: $<0.1$~Tg \\\\", "\\midrule",
         "\\multicolumn{6}{l}{\\ClaudeText{Adopted measurement model (Table~\\ref{tab:error_budget})}} \\\\"]
    for key, (lab, s) in sup_a.items():
        r = ra[key]
        L.append(f"\\ClaudeText{{{lab}}} & \\ClaudeText{{{s}}} & \\ClaudeText{{{r['fixed20_min']:.3f}--{r['fixed20_max']:.2f}}} & \\ClaudeText{{{rng(r['fixed20_cov'])}}} & \\ClaudeText{{{r['best_worst']:.3f}}} & \\ClaudeText{{{rng(r['best_cov'])}}} \\\\")
    L += ["\\midrule", "\\multicolumn{6}{l}{\\ClaudeText{Diagonal-floor scenario}} \\\\"]
    for scen, r in rs.items():
        L.append(f"{scen} & {sup[scen]} & {r['fixed20_min']:.3f}--{r['fixed20_max']:.2f} & {rng(r['fixed20_cov'])} & {r['best_worst']:.3f} & {rng(r['best_cov'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", "", "% ---- round-74 version ----"] + ["% " + l for l in old if l.strip()]
    TEX.write_text("\n".join(L) + "\n")
    log("table ->", TEX)


def plot(maps, save, ra):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    _style()
    lat, zlev = save["lat_deg"], save["z_lev_km"]
    conc = save["conc_kg_m3"] * 1e9 / (save["total_mass_kg"] / 1e9)
    in_rng = (zlev >= Z_KM[0]) & (zlev <= Z_KM[-1])
    ilev20 = int(np.argmin(np.abs(zlev - Z_TABLE)))

    def best(mmat):
        mm = np.where(np.isfinite(mmat) & in_rng[None, :], mmat, np.inf)
        return mm.min(axis=1), zlev[np.argmin(mm, axis=1)]
    m_base = maps["band_adopted_const"]; mb_base, zb_base = best(m_base)
    mb_atlas, _ = best(maps["band_adopted_atlas"]); mb_win, _ = best(maps["window_adopted_const"])
    m20 = m_base[:, ilev20]
    hom = float(E.UM["models"]["B_primary"]["band_triplet"])
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15 / 2.54, 13.5 / 2.54), gridspec_kw=dict(height_ratios=[1.35, 1.0]), sharex=True)
    pc = ax.pcolormesh(lat, zlev, conc.T, cmap="Greys", shading="nearest")
    cb = fig.colorbar(pc, ax=ax, pad=0.015, fraction=0.04); cb.set_label("silica [$\\mu$g m$^{-3}$ per Tg]", fontsize=9)
    cs = ax.contour(lat, zlev, m_base.T, levels=CONTOURS_TG, colors="red", linewidths=1.2)
    ax.clabel(cs, fmt=lambda v: f"{v:g} Tg", fontsize=8)
    plt.rcParams["hatch.color"] = "0.35"
    ax.contourf(lat, zlev, save["q_obsc"].T, levels=[0.1, 0.5, 0.9, 1.0001], colors="none", hatches=["..", "//", "xx"])
    show = mb_base < 5.0
    ax.plot(np.where(show, lat, np.nan), zb_base, color="black", lw=1.2, ls="--", label="optimal tangent altitude")
    ax.axhline(float(zlev[ilev20]), color="black", lw=0.9, ls=":", label="20.5-km level (51 hPa)")
    ax.set_ylim(10, 30); ax.set_xlim(-80, 80); ax.set_ylabel("Altitude [km]")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([0], [0], color="red", lw=1.2), Patch(facecolor="none", edgecolor="0.35", hatch="//")]
    labels += ["$M_{\\min}^{(3\\sigma)}$ contours", "$q_\\mathrm{obsc}>50\\%$"]
    ax.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    ax.text(0.99, 0.97, "(a)", transform=ax.transAxes, ha="right", va="top")
    ax2.plot(lat, m20, color="red", ls=":", lw=1.3, label="20.5-km level")
    ax2.plot(lat, mb_base, color="red", lw=1.6, label="best bin, adopted model")
    ax2.plot(lat, mb_atlas, color="red", ls="--", lw=1.2, label="best bin, measured altitude bins")
    ax2.plot(lat, mb_win, color="black", lw=1.0, label="best bin, window subset")
    ax2.axhline(hom, color="0.4", lw=0.8, ls="-.", label=f"homogeneous layer ({hom:.3f} Tg)")
    ax2.axhline(0.1, color="0.7", lw=0.6)
    ax2.set_yscale("log"); ax2.set_ylim(0.015, 3)
    ax2.set_xlabel("Latitude [$^\\circ$]"); ax2.set_ylabel("$M_{\\min}^{(3\\sigma)}$ [Tg]")
    ax2.grid(True, which="both", alpha=0.25, lw=0.5); ax2.legend(loc="upper center", fontsize=8, ncol=2, framealpha=0.9)
    ax2.text(0.99, 0.97, "(b)", transform=ax2.transAxes, ha="right", va="top")
    fig.tight_layout(h_pad=0.5)
    pos_a, pos_b = ax.get_position(), ax2.get_position(); ax2.set_position([pos_b.x0, pos_b.y0, pos_a.width, pos_b.height])
    fig.savefig(FIG, dpi=300, bbox_inches="tight"); plt.close(fig)
    log("figure ->", FIG)


if __name__ == "__main__":
    main()
