"""Fast checks of the package and the shipped data (pytest -q; ~1 min).

Regression targets are numbers quoted in the paper; they are stated with the
tolerance appropriate to the rounding used in the text.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saimon.config import DATA, OUT, OPTICS, GLOSSAC_NC   # noqa: E402


def test_layout():
    for p in [OPTICS / "SiO2_KitamuraPopova.yml", GLOSSAC_NC, DATA / "ace" / "v52_subset" / "occultationlist.csv",
              DATA / "ace_floor" / "w0p1" / "atlas.json", DATA / "trace_gases" / "afgl", OUT / "calibrated_background_thresholds.json"]:
        assert p.exists(), p


def test_chord_geometry():
    from saimon.geometry import tangent_to_slant_paths  # noqa: F401
    R = 6371e3; dz = 500.0
    assert abs(2 * np.sqrt(2 * R * dz) / 1e3 - 159.6) < 0.5      # P_kk ~ 160 km (Appendix C)


def test_us_standard_atmosphere():
    from saimon.atmosphere import us_standard_atmosphere
    atm = us_standard_atmosphere(np.array([0.0, 20000.0]))
    T = np.asarray(atm.temperature_k); P = np.asarray(atm.pressure_pa)
    assert 285 < T[0] < 290 and 210 < T[1] < 222 and abs(P[1] / 5529 - 1) < 0.02   # 55.3 hPa at 20 km


def test_silica_extinction_peak_is_at_8p80_um():
    from saimon.materials import SilicaRefractiveIndex
    from saimon.sai import create_silica_sai_layer
    alt = np.linspace(0, 60_000, 121)
    lam = np.arange(8.6, 9.0001, 0.005)
    lay = create_silica_sai_layer(alt, 1.0, refractive_index=SilicaRefractiveIndex(str(OPTICS / "SiO2_KitamuraPopova.yml")),
                                  rmed_nm=268.0, sigma=1.31)
    e = lay.extinction_profile_m1(lam * 1e-6)[40, :]
    assert abs(lam[np.argmax(e)] - 8.80) < 0.011                    # Sect. 3.1 / Appendix F
    assert abs(np.max(e) / 6.55e-7 - 1) < 0.05                       # ~6.6e-7 m^-1 per Tg (monochromatic)


def test_glossac_subset_reproduces_the_anchors():
    from saimon.glossac import GloSSACLoader, compute_tropical_saod
    d = GloSSACLoader().load_raw_data(525)
    t, s, _ = compute_tropical_saod(d)
    i = int(np.argmin(np.abs(t - 200105))); j = int(np.argmin(np.abs(t - 199112)))
    assert abs(s[i] - 0.00325) < 2e-4 and abs(s[j] - 0.1615) < 2e-3   # Appendix A / Fig. A2


def test_reservoir_mass_archive():
    r = json.loads((OUT / "reservoir_mass_saod.json").read_text())
    assert abs(r["presets"]["quiet"]["M_H2SO4_Tg"] - 0.40) < 0.01
    assert abs(r["presets"]["post_eruption"]["M_H2SO4_Tg"] - 24.2) < 0.2


def test_residual_floor_archive_quotes():
    """The exporter asserts every rounded value quoted in Appendix E against the archive."""
    sys.path.insert(0, str(ROOT / "reproduce" / "ace_floor"))
    import residual_floor_paper1 as exp
    import residual_floor_manuscript as base
    atlas = json.loads((DATA / "ace_floor" / "w0p1" / "atlas.json").read_text())
    silica = json.loads((DATA / "ace_floor" / "w0p1" / "19_22_8.8um.json").read_text())
    sens = json.loads((DATA / "ace_floor" / "w0p1" / "sensitivity.json").read_text())
    red = json.loads((DATA / "ace_floor" / "w0p1" / "paired_sd_reductions.json").read_text())
    exp.verify_appendix_quotes(atlas, silica, sens, red)
    assert base.fmt(silica["sd"]) == "1.86"


def test_thresholds_archive():
    d = json.loads((OUT / "calibrated_background_thresholds.json").read_text())
    q = d["results"]["CALIBRATED quiet (LM65T223, 2-comp)"]
    assert abs(q["window 7.8-9.3 @0.1: measured floors"]["full 7-param"]["triplet + window"]["mmin_tg"] - 0.141) < 0.002
    assert abs(q["band 8-13 @0.1: measured floors"]["full 7-param"]["triplet + band01"]["mmin_tg"] - 0.101) < 0.002
    assert d["window_elements_um"][10] == 8.8 and len(d["band01_elements_um"]) == 47 and d["band01_elements_um"][0] == 7.8   # grid centred on 8.80 um (round 53), band from 7.80 um (round 57)


def test_onion_peel_constants():
    """Edge-bounded retrieval grid, exact gain (Appendix B, referee round): g = 1.10, |G_k| = 6.86e-6 m^-1 per OD."""
    from saimon.onion_peel import G_ONION, M1_PER_OD, OD_PER_M1, BUDGET_OD_880, SIG_880_ABS
    assert abs(G_ONION - 1.097) < 0.002 and abs(M1_PER_OD / 6.863e-6 - 1) < 0.002
    assert abs(OD_PER_M1 / 1e3 - 145.7) < 0.2
    assert abs(BUDGET_OD_880 - 2.466e-3) < 2e-6 and abs(SIG_880_ABS / 1.692e-8 - 1) < 0.002


def test_c880_archive_anchor():
    """The design-grid atlas (centred on 8.80 um) carries the same 8.80-um element as the Appendix E scan."""
    a = json.loads((DATA / "ace_floor" / "w0p1_c880" / "atlas.json").read_text())
    b = json.loads((DATA / "ace_floor" / "w0p1" / "atlas.json").read_text())
    sa = {round(r["center_um"], 3): r["sd"] for r in a["19_22"] if r.get("sd") is not None}
    sb = {round(r["center_um"], 3): r["sd"] for r in b["19_22"] if r.get("sd") is not None}
    assert sa[8.8] == sb[8.8] and all(round(c * 10) % 1 == 0 for c in sa if 7.3 <= c <= 13.3)


def test_ace_subset_round_trip():
    import gzip
    occ = ROOT / "data" / "ace" / "v52_subset" / "residual" / "sr101445"
    files = sorted(occ.iterdir())
    assert len(files) == 21
    dat = np.loadtxt(files[0])
    assert dat.shape == (1740, 2) and abs(dat[0, 0] - 750.99939) < 1e-6


def test_o3_latitude_measured_extract():
    """RC2 S1: the ACE v5.2 O3 extract carries 192 profiles and the archived slant ratios at 20 km read 0.92/0.94/0.50."""
    import csv, gzip
    with gzip.open(ROOT / "data" / "ace" / "o3_l2_v52" / "o3_profiles_v52.csv.gz", "rt") as fh:
        occs = {r["occultation"] for r in csv.DictReader(fh)}
    assert len(occs) == 192
    d = json.loads((OUT / "o3_slant_latitude_measured.json").read_text())
    j20 = d["h_tan_km"].index(20.0)
    r = {k: round(v["ratio_median"][j20], 2) for k, v in d["bands"].items()}
    assert r["tropical 20-25N Jun-Aug"] == 0.92 and r["midlat 35-55N Jun-Sep"] == 0.94 and r["antarctic 60-90S Aug-Sep"] == 0.50


# ---- paper 1 rounds 62-63 (co-author review): the new archives --------------

def test_signal_vs_background_archive():
    """0.1 Tg of silica in observables (Sects. 1, 8): 0.09 cm^-3 at 20 km, 8e4 cm^-2, 9 % of the
    calibrated background's particles above 0.15 um, +16 % / +33 % of the vertical optical depth."""
    d = json.loads((OUT / "signal_vs_background.json").read_text())
    s = d["silica"]["0.1"]; r = d["ratios_0p1Tg_over_background"]
    assert abs(s["n20_cm3"] - 0.091) < 0.002 and abs(s["column_cm2"] / 8.3e4 - 1) < 0.02
    assert abs(r["n20_cm3_over_bg_above_015um"] - 0.092) < 0.003
    assert abs(r["vert_525"] - 0.165) < 0.003 and abs(r["vert_880"] - 0.334) < 0.003


def test_coated_sphere_archives_share_the_bare_silica_endpoint():
    """Appendix F coatings: the mu = 0 (sulfate shell) and f_sil = 1 (composite) cases are the
    band and window baselines of the thresholds archive."""
    t = json.loads((OUT / "calibrated_background_thresholds.json").read_text())
    q = t["results"]["CALIBRATED quiet (LM65T223, 2-comp)"]
    band = q["band 8-13 @0.1: measured floors"]["full 7-param"]["triplet + band01"]["mmin_tg"]
    win = q["window 7.8-9.3 @0.1: measured floors"]["full 7-param"]["triplet + window"]["mmin_tg"]
    for name, key, sub in (("sulfate_coating.json", "0.0", "mmin_tg"),
                           ("composite_particle.json", "1.0", "mmin_tg_total")):
        p = OUT / name
        if not p.exists():
            pytest.skip(f"{name} not present (run make_all.py --only coat comp)")
        d = json.loads(p.read_text())
        assert abs(d["sets"]["band 8-13 @0.1"][key][sub] - band) < 2e-3
        assert abs(d["sets"]["window 7.8-9.3 @0.1"][key][sub] - win) < 2e-3
    p = OUT / "sulfate_coating.json"
    if p.exists():
        d = json.loads(p.read_text())
        assert abs(d["sets"]["band 8-13 @0.1"]["0.1"]["mmin_tg"] - 0.116) < 0.002   # +15 % at a tenth of the silica mass
        assert abs(d["peak_um"]["3.0"] - 8.93) < 0.011
    p = OUT / "composite_particle.json"
    if p.exists():
        d = json.loads(p.read_text())
        assert abs(d["sets"]["band 8-13 @0.1"]["0.0"]["mmin_tg_total"] - 0.78) < 0.01   # bare calcite sphere
        assert abs(d["sets"]["band 8-13 @0.1"]["0.5"]["mmin_tg_total"] - 0.52) < 0.01


def test_ash_positive_control_archive():
    """Sect. 4.3: the discriminator on the Lecours et al. (2026) ash-sulfate spectra of the release."""
    d = json.loads((OUT / "ace_v52" / "ash_positive_control.json").read_text())
    assert d["n_ash_spectra"] == 56 and d["n_ash_occ"] == 41 and d["n_null"] == 101
    assert d["ash"]["frac_ash_gt3"] > 0.94 and d["null"]["frac_ash_gt3"] < 0.02
    assert d["ash"]["frac_sil_gt3"] == 0.0 and abs(d["ash"]["M_sil_both_median"]) < 0.05
    assert abs(d["lecours_compare"]["ratio_median_fixed_psd"] - 1.84) < 0.02


def test_ash_occultations_shipped():
    man = (DATA / "ace" / "ash_lecours2026" / "ash_manifest_v52.txt").read_text().split()
    assert len(man) == 42 and all((DATA / "ace" / "v52_subset" / "residual" / o).is_dir() for o in man)
    assert (OPTICS / "ash_deguine2020" / "calbuco_ash_Deguine_2020.ri").exists()


# ---- paper 1 round 70 (referee round 5, item 6): target identifiability in marginal() ----

def _marginal():
    sys.path.insert(0, str(ROOT / "reproduce"))
    import calibrated_background_thresholds as cbt
    return cbt


def test_marginal_referee_counterexample_is_identifiable():
    """J = [[1, 1, 0], [0, 0, 1]], unit errors, last column the target: sigma = 1 exactly although
    the full Fisher matrix is singular (K = 2 < N = 3) -- no parameter-count shortcut."""
    cbt = _marginal()
    c1 = np.array([1.0, 0.0]); c2 = np.array([1.0, 0.0]); t = np.array([0.0, 1.0]); sig = np.ones(2)
    sa, tperp, R2 = cbt.marginal([0, 1], [c1, c2], sig, t)
    assert sa == 1.0 and tperp == 1.0 and R2 == 0.0
    Ur, s, n_kept = cbt.nuisance_span([0, 1], [c1, c2], sig)
    assert Ur.shape == (2, 1) and n_kept == 2 and s[1] <= cbt.SVD_RTOL * s[0]


def test_marginal_duplicated_and_dependent_nuisance_columns():
    """A duplicated (or linearly dependent) nuisance column changes neither the span nor sigma."""
    cbt = _marginal()
    rng = np.random.default_rng(7)
    a, b, t = rng.normal(size=(3, 6)); sig = rng.uniform(0.5, 2.0, 6); idx = np.arange(6)
    ref = cbt.marginal(idx, [a, b], sig, t)
    assert np.isfinite(ref[0])
    for cols in ([a, a, b], [a, b, 2.5 * a - b], [b, a, a, b]):
        got = cbt.marginal(idx, cols, sig, t)
        assert np.allclose(got, ref, rtol=1e-12, atol=0)
        assert cbt.nuisance_span(idx, cols, sig)[0].shape[1] == 2


def test_marginal_zero_nuisance_column_is_dropped():
    """A nuisance column that vanishes on the chosen channels spans nothing: sigma unchanged."""
    cbt = _marginal()
    rng = np.random.default_rng(11)
    a, b, t = rng.normal(size=(3, 6)); sig = rng.uniform(0.5, 2.0, 6); idx = np.arange(6)
    ref = cbt.marginal(idx, [a, b], sig, t)
    got = cbt.marginal(idx, [a, np.zeros(6), b], sig, t)
    assert np.allclose(got, ref, rtol=1e-12, atol=0)
    assert cbt.nuisance_span(idx, [a, np.zeros(6), b], sig)[2] == 2
    # only zero nuisance columns: the target is fitted alone, sigma = 1/||t/sig||
    sa, tperp, R2 = cbt.marginal(idx, [np.zeros(6)], sig, t)
    assert abs(sa * np.linalg.norm(t / sig) - 1) < 1e-14 and R2 == 0.0


def test_marginal_target_in_nuisance_span_is_unidentifiable():
    """Target inside the nuisance span (t_perp = 0 to round-off) and a zero target: (inf, 0, nan);
    K generic nuisance columns on K channels span everything, so the target is unidentifiable."""
    cbt = _marginal()
    rng = np.random.default_rng(3)
    a, b, c, t = rng.normal(size=(4, 6)); sig = rng.uniform(0.5, 2.0, 6); idx = np.arange(6)
    six = [a, b, c] + [rng.normal(size=6) for _ in range(3)]      # six independent columns
    for cols, tt in (([a, b], 0.3 * a - 2.0 * b), ([a, b], np.zeros(6)), (six, t)):
        sa, tperp, R2 = cbt.marginal(idx, cols, sig, tt)
        assert sa == np.inf and tperp == 0.0 and np.isnan(R2)
    # six columns of rank three (a + b, c - a, a - 2b dependent) on six channels leave the target finite
    dep = [a, b, c, a + b, c - a, a - 2 * b]
    assert np.isfinite(cbt.marginal(idx, dep, sig, t)[0]) and cbt.nuisance_span(idx, dep, sig)[0].shape[1] == 3
    # and one more generic channel makes the six-column case finite again
    cols = [np.r_[x, rng.normal()] for x in six]
    sa, _, _ = cbt.marginal(np.arange(7), cols, np.r_[sig, 1.0], np.r_[t, rng.normal()])
    assert np.isfinite(sa)


# ---- paper 1 round 70 (referee round 5): the shipped inputs and archives of the new steps ----

def test_round70_fine_grids_and_line_lists_shipped():
    """32 line-resolved 0.0025-cm^-1 grids (four gases x 8 (T, P)) and the four carved HITRAN line lists
    of the 1042-1319 cm^-1 window, with the row counts stated in data/PROVENANCE.md."""
    grids = sorted((DATA / "trace_gases" / "xsec_cache").glob("*_1067_1294_*_s0.0025.npz"))
    assert len(grids) == 32 and {g.name.split("_")[0] for g in grids} == {"o3", "n2o", "ch4", "h2o"}
    g = np.load(grids[0]); assert g["nu"][0] >= 1067.0 and g["nu"][-1] <= 1294.0 and abs(np.diff(g["nu"])[0] - 0.0025) < 1e-9
    rows = {"o3": 26707, "n2o": 21298, "ch4": 14511, "h2o": 1742}
    for gas, n in rows.items():
        p = DATA / "trace_gases" / "hitran_cache" / f"cont_{gas}_1042_1319"
        hdr = json.load(open(p.with_suffix(".header")))
        assert hdr["number_of_rows"] == n and hdr["table_name"] == p.name
        with open(p.with_suffix(".data")) as f:
            k = sum(1 for _ in f)
        assert k == n


def test_atlas_extract_covers_the_closure_window():
    import gzip
    last = None
    with gzip.open(DATA / "ace_atlas" / "Tropics_016-020km.txt.gz", "rt") as f:
        for i, line in enumerate(f):
            if i >= 9: last = line
    assert float(last.split()[0]) >= 1294.0


def test_element_observable_closure_archive():
    """Appendix D (round 70): -ln<T> against <tau> at the 8.80-um element, the removal error, the joint
    window/band values, the plateau, and the Monte Carlo closure of the 0.1-Tg draws."""
    d = json.load(open(OUT / "element_observable_closure.json"))
    eta = str(d["setup"]["sigma_eta_base"])
    assert abs(d["observable"]["y_gas_880"] - 0.406) < 2e-3 and abs(d["observable"]["mean_tau_gas_880"] - 1.186) < 2e-3
    b = d["removal_budget_880"][eta]["snr2000"]
    assert abs(b["sigma_total"] / 5.115e-4 - 1) < 2e-3
    assert abs(3 * d["joint_vs_two_stage_window"][eta]["joint_window_triplet_shell"] - 0.0680) < 5e-4
    assert abs(3 * d["joint_band"]["per_eta"][eta]["joint_band_shell"] - 0.0572) < 5e-4
    assert abs(3 * d["snr_scan"]["plateau_shell_triplet"] - 0.0466) < 5e-4
    mc = d["monte_carlo"]["baseline"]
    assert mc["0.1"]["n"] >= 100 and mc["0.1"]["p_detect_3sig"] >= 0.85 and mc["0.0"]["p_detect_3sig"] <= 0.02
    assert 0.55 <= mc["0.1"]["coverage_1sig"] <= 0.85 and abs(mc["0.1"]["bias_median"]) < 0.01


def test_marginal_rank_archive():
    """Every archived configuration re-evaluated with the SVD marginalization: nothing newly finite or
    infinite, the baselines reproduced to the archive's own rounding."""
    d = json.load(open(OUT / "marginal_rank.json"))
    s = d["summary"]
    assert s["n_configurations"] == 108 and s["newly_finite"] == [] and s["newly_infinite"] == []
    assert s["max_rel_diff_new_vs_old"] < 1e-9 and s["max_rel_diff_new_vs_archive"] < 1e-4
    assert abs(d["baseline"]["band | full 7-param | triplet + band01"]["mmin_tg_new"] - 0.10104) < 1e-5
    assert abs(d["baseline"]["window | full 7-param | triplet + window"]["mmin_tg_new"] - 0.14115) < 1e-5


def test_materials_uniaxial_archive():
    """Table 8 under the Clausius-Mossotti average (b) against the ray mean (a): dolomite x1.56,
    calcite +2 %, alumina +5 %; (b) and (c) agree within 2.5 %; the ray mean reproduces the library."""
    d = json.load(open(OUT / "materials_uniaxial.json"))["materials"]
    for mat, r_ba, m_b in (("calcite", 1.018, 0.7956), ("dolomite", 1.558, 1.1621), ("alumina", 1.046, 0.1565)):
        m = d[mat]
        assert abs(m["ratio_b_over_a"]["mmin_quiet"] - r_ba) < 0.005 and abs(m["ratio_c_over_b"]["mmin_quiet"] - 1) < 0.025
        assert abs(m["b"]["mmin_tg"]["quiet"] - m_b) < 5e-4
        assert m["ray_mean_reproduces_library_max_abs_diff"] < 1e-9


def test_psd_tail_coagulation_archive():
    d = json.load(open(OUT / "psd_tail_coagulation.json"))
    assert d["tail"]["D300o_reproduces_baseline"] is True
    c = d["tail"]["cases"]["D300o"]
    assert abs(c["measured_tail"]["mmin_ratio_to_lognormal"]["quiet"] - 1.0146) < 1e-3 and abs(c["mass_removed_binned"] - 0.0145) < 5e-4
    k = d["coagulation"]
    assert abs(k["K_psd_mean_cm3_s"] / 2.936e-9 - 1) < 1e-3 and abs(k["n_threshold_0p1Tg_cm3"] - 0.0932) < 5e-4


def test_latitude_declared_level_archive():
    d = json.load(open(OUT / "latitude_declared_level.json"))
    assert abs(d["levels"]["table_level_km"] - 20.4656) < 1e-3
    q = {r["quoted"]: r for r in d["quoted_number_map"]}
    assert abs(q["f = 3.8 at 20.5 km (cores)"]["value"] - 3.779) < 2e-3


def test_radiometric_budget_archive():
    d = json.load(open(OUT / "radiometric_budget.json"))
    r = d["chain"]["8.80"]["vox_microbolometer"]["dwell_0.5"]
    assert r["inputs"]["dstar_cm_rtHz_W"] == 1e9 and abs(r["result"]["SNR_T_ratio"] / 23640 - 1) < 2e-3
    assert r["result"]["limited_by"] == "systematics cap"


def test_nulltest_diagnostics_archive():
    """The referee's four numbers for the full-band null test reproduce (Sect. 4.3)."""
    d = json.load(open(OUT / "ace_v52" / "nulltest_diagnostics.json"))
    f = d["rows"]["full"]
    assert f["n_occ"] == 101 and abs(f["robust_scatter"] - 0.025943) < 2e-6 and abs(f["sd"] - 0.021847) < 2e-6
    assert abs(f["median_formal"] - 0.023208) < 2e-6 and abs(f["median_M"] - 0.011245) < 2e-6
    assert d["sample"]["selection"]["passed"] == 101 and d["sample"]["selection"]["lt2_above_30"] == 42
    assert abs(d["sample"]["full"]["trend"]["slope_tg_per_decade"] - 0.0228) < 5e-4


def test_round74_archives():
    """Referee round 6 (round 74): the adopted measurement model (Table 2), its scenarios, the closure-test
    variants, the grid convergence, the materials under the adopted model, the radiometric chain at the
    element observable and the null-baseline procedure."""
    u = json.load(open(OUT / "uncertainty_model.json"))
    m = u["models"]
    assert abs(m["B_primary"]["band_triplet"] - 0.1187) < 5e-4 and abs(m["B_primary"]["window_triplet"] - 0.1678) < 5e-4
    assert abs(m["B_primary"]["band_alone"] - 0.1727) < 5e-4 and abs(m["B_primary"]["window_alone"] - 0.3363) < 1e-3
    assert abs(m["two_stage_diagonal"]["adopted_2p47_band_triplet"]["mmin_3sig_shell"] - 0.1008) < 5e-4
    assert abs(m["two_stage_diagonal"]["adopted_2p47_window_triplet"]["mmin_3sig_shell"] - 0.1405) < 5e-4
    assert abs(u["correlation"]["gaussian_l_um"] - 0.142) < 2e-3
    assert u["response_derivatives"]["shift_max_rel_diff_analytic_vs_fd1e_4"] < 1e-3
    g = u["geometry"]
    assert abs(g["rho_from_rows_880"] - 2.607) < 5e-3 and 0.15 < g["coherent_o3_2pct"]["dalpha_880_at_20km_m1"] / g["coherent_o3_2pct"]["f_alpha_880_at_20km_m1"] < 0.3
    f = json.load(open(OUT / "closure_flat_prior_check.json"))
    assert abs(f["round70_window_3sig_flat"] - 0.0941) < 5e-4 and abs(f["round70_hybrid_band_3sig_flat"] - 0.0612) < 5e-4
    v = json.load(open(OUT / "closure_variants.json"))
    assert abs(v["fixed_database"]["spread_of_db_means_null"] - 0.0082) < 2e-3
    assert 0.025 < v["alt_sulfate_member_B70T215"]["0.1"]["bias_median"] < 0.045
    assert v["round70_with_binomial_intervals"]["null_zero_of_100_upper95"] < 0.04
    gc = json.load(open(OUT / "grid_convergence.json"))
    assert abs(gc["comparison_to_finest"]["0.0025"]["mass_error_tg_per_shell"]) < 1e-3 and gc["comparison_to_finest"]["0.0025"]["max_abs_dy_od"] < 1e-4
    mp = json.load(open(OUT / "materials_primary.json"))["materials"]
    for mat, val in (("silica", 0.119), ("calcite", 0.41), ("dolomite", 0.78), ("alumina", 0.18)):
        assert abs(mp[mat]["primary"]["band_triplet"] - val) < 0.01
    r = json.load(open(OUT / "radiometric_budget_observable.json"))
    assert abs(r["optical_depths"]["8.80"]["tau_gas_element"] - 0.406) < 2e-3
    assert 2.5e4 < r["chain"]["8.80"]["dwell_0.17"]["result"]["SNR_T_ratio"] < 3.5e4
    assert abs(r["ratio_cancellation_observable"]["drift_requirement"]["gain_drift_rate_max_per_s"][1] - 7.14e-6) < 1e-7
    n = json.load(open(OUT / "ace_null_baseline.json"))
    assert abs(n["drift"]["slope_tg_per_decade"] - 0.0228) < 1e-3 and n["drift"]["se_year_block_bootstrap"] < 0.004
    assert n["prospective_split"]["2004-2015 -> 2016-2024"]["exceed_3.0sigma"]["count"] == 9
    assert n["secular_injection"]["cases"]["ramp_0.10_per_decade_rolling"]["first_season_seasonal_mean_above_4p54"] is None
    assert n["secular_injection"]["cases"]["step_0.10_rolling"]["first_season_seasonal_mean_above_4p54"] == 2010
    s = json.load(open(OUT / "materials_scattering.json"))
    env = s["perturbations_convention_b"]["threshold_envelope_quiet_tg"]
    assert env["alumina"]["max"] < env["calcite"]["min"] and s["perturbations_convention_b"]["carbonate_pair_envelopes_overlap"]


def test_round71_archives():
    """Round 71 (follow-up to referee round 5): far-infrared thresholds under the orientation-averaged
    optics and the converged nonlinear ACE refit."""
    for mat, far_only, band_far in (("calcite", 0.037, 0.026), ("dolomite", 0.063, 0.031), ("alumina", 0.10, 0.071)):
        d = json.load(open(OUT / f"materials_farir_uniaxial_{mat}.json"))
        txt = json.dumps(d)
        assert f"{far_only:.3f}"[:4] in txt and f"{band_far:.3f}"[:4] in txt
    d = json.load(open(OUT / "ace_nulltest_nonlinear_refit.json"))
    assert (OUT / "ace_nulltest_nonlinear_refit_perocc.csv").exists() and len(json.dumps(d)) > 1000


def test_round75_archives():
    """Round 75 (completion of referee round 6): the adopted model carried through the robustness scans, the altitude and
    latitude maps, the far-infrared columns, a multi-tangent inverse and an anisotropic solver."""
    r = json.load(open(OUT / "adopted_robustness.json"))["rows"]
    assert abs(r["baseline"]["band"] - 0.1186) < 5e-4 and abs(r["baseline"]["window"] - 0.1676) < 5e-4
    assert abs(r["baseline"]["band_scenario"] - 0.1008) < 5e-4 and abs(r["sig1.8"]["band"] - 0.2017) < 1e-3
    assert abs(r["altitude"]["16"]["band"] - 0.0758) < 5e-4 and abs(r["altitude"]["24"]["band"] - 0.2048) < 5e-4
    m = json.load(open(OUT / "multitangent_retrieval.json"))["scenarios"]
    assert abs(m["sulfate_per_shell_base"]["mmin_3sig_20km_tg"] - 0.0574) < 1e-3
    assert abs(m["sulfate_per_shell_fov_1km"]["mmin_3sig_20km_tg"] - 0.0921) < 1e-3
    assert abs(m["sulfate_prescribed_base"]["mmin_3sig_20km_tg"] - 0.0300) < 1e-3
    f = json.load(open(OUT / "farir_adopted.json"))["materials"]
    assert abs(f["silica"]["band_plus_farir_adopted"]["1.0"] - 0.0377) < 5e-4 and abs(f["calcite"]["band_plus_farir_adopted"]["1.0"] - 0.0260) < 5e-4
    assert abs(f["silica"]["farir_only"]["1.0"] - f["silica"]["farir_only_round71"]["1.0"]) < 1e-6
    lat = json.load(open(OUT / "adopted_latitude.json"))
    t = lat["table7_adopted"]["band_adopted_const"]
    assert abs(t["z20_level"] - 20.4656) < 1e-3 and 0.02 < t["fixed20_min"] < 0.05
    d = json.load(open(OUT / "dda_uniaxial.json"))
    assert max(d["summary"]["dolomite"]["validation_error_psd_weighted"]) < 0.05       # validated on the isotropic rays
    assert max(d["summary"]["alumina"]["validation_error_psd_weighted"]) > 0.05       # the surface mode does not converge on this lattice
    assert 0.0 < d["thresholds"]["dolomite"]["change_percent"] < 5.0 and -10.0 < d["thresholds"]["calcite"]["change_percent"] < 0.0
