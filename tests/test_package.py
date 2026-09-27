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
