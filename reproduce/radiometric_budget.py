"""
radiometric_budget.py -- round 70 (referee round 5, item 8): a consistent,
sourced first-order radiometric chain for the MIR solar-occultation channels.

The referee asked for one model connecting aperture, accepted solar solid angle,
spectral response, etendue, pixel count, dwell, attenuation, reference noise,
detector noise, warm emission, nonlinearity and saturation, ending in the
TRANSMISSION precision after the same-event ratio (not an incident-power/NEP
SNR), and for the detector specifications to be reconciled and sourced.

Everything printed is written to outputs/radiometric_budget.json
(every input and every intermediate), a text log next to it, and a booktabs
table tables/radiometric_budget.tex (the lead decides whether to
include it).  Sources are listed in SOURCES with the bibliographic data verified
(Crossref API + full texts) on 2026-10-25; see facts_item8.md.

Conventions
-----------
* "SNR" is always the per-sample transmission SNR of the ratio
  T = S_atm / S_ref (atmospheric sample over the exo-atmospheric reference),
  i.e. SNR_T = T / sigma_T.  Two single-measurement SNRs are also given:
  SNR_TOA (one unattenuated sample, random terms only) and SNR_atm (one
  attenuated sample).  The systematics cap (2000, flight heritage) is applied
  AFTER the random chain, as in the paper.
* dz = 0.5 km sample; the geometric constant K_GEOM = sigma_alpha * SNR of
  reproduce/mir_detector_etc.py (onion-peel conversion, saimon.onion_peel) is reused.
* The adopted tangent range is the ETC's 3000 km (paper); the geometric value
  for the ISS-like orbit is also computed and the difference reported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(ROOT / "reproduce") not in sys.path:
    sys.path.insert(0, str(ROOT / "reproduce"))
from mir_detector_etc import (  # noqa: E402
    H, C, KB, T_SUN, R_SUN, AU, K_GEOM, DETECTORS, Instrument,
    solar_spectral_irradiance_W_m2_um, planck_W_sr_m2_um,
)
from saimon.onion_peel import G_ONION  # noqa: E402

OUT_DIR = ROOT / "outputs"
OUT_JSON = OUT_DIR / "radiometric_budget.json"
OUT_LOG = OUT_DIR / "radiometric_budget.txt"
OUT_TEX = ROOT / "tables" / "radiometric_budget.tex"
NPZ = ROOT / "outputs" / "slant_od_landscape_curves.npz"

SIGMA_SB = 5.670374419e-8  # W m^-2 K^-4
GM_EARTH = 3.986004418e14  # m^3 s^-2
R_EARTH_M = 6.371e6

# ----------------------------------------------------------------------------
# Sources (verified 2026-10-25; Crossref metadata + full text where stated)
# ----------------------------------------------------------------------------
SOURCES = {
    "Christensen2018_OTES": dict(
        cite="Christensen, P. R., Hamilton, V. E., Mehall, G. L., Pelham, D., O'Donnell, W., Anwar, S., "
             "Bowles, H., Chase, S., Fahlgren, J., Farkas, Z., Fisher, T., James, O., Kubik, I., Lazbin, I., "
             "Miner, M., Rassas, M., Schulze, L., Shamordola, K., Tourville, T., West, G., Woodward, R., and "
             "Lauretta, D.: The OSIRIS-REx Thermal Emission Spectrometer (OTES) Instrument, Space Sci. Rev., "
             "214, 87, 2018, doi:10.1007/s11214-018-0513-6",
        verified="full text (Springer PDF), Sect. 2.3.2 and Table 2",
        facts="uncooled DLATGS pyroelectric detector (Selex-Galileo); measured D* incl. diamond lens "
              "1.2e9 cm Hz^1/2 W^-1 at 10 Hz, 22 C; AC-coupled; FTS (moving mirror supplies the modulation); "
              "mass 6.27 kg; power 10.8 W average, 15.99 W peak; 15.2-cm Cassegrain; 5.71-100 um"),
    "Christensen2024_LTES": dict(
        cite="Christensen, P. R., Hamilton, V. E., Mehall, G. L., Anwar, S., Bowles, H., Chase, S., Farkas, Z., "
             "Fisher, T., Holmes, A., Kubik, I., Lazbin, I., O'Donnell, W., Ortiz, C., Pelham, D., Rogers, S., "
             "Shamordola, K., Tourville, T., and Woodward, R.: The Lucy Thermal Emission Spectrometer (L'TES) "
             "Instrument, Space Sci. Rev., 220, 1, 2024 (published online 19 Dec 2023), "
             "doi:10.1007/s11214-023-01029-y",
        verified="full text (Springer PDF; = PMC10730683), Table 1 and detector section",
        facts="build-to-print copy of OTES; uncooled DLATGS pyroelectric; D* 1.2e9 cm Hz^1/2 W^-1 at 10 Hz, 22 C; "
              "mass 6.47 kg; power 12.6 W average, 18.0 W peak"),
    "Christensen2024_ETHEMIS": dict(
        cite="Christensen, P. R., Spencer, J. R., Mehall, G. L., et al.: The Europa Thermal Emission Imaging "
             "System (E-THEMIS) Investigation for the Europa Clipper Mission, Space Sci. Rev., 220, 38, 2024, "
             "doi:10.1007/s11214-024-01074-1 (already Christensen2024 in references.bib)",
        verified="full text (Springer PDF), abstract, Sects. 3.x and 4.3",
        facts="uncooled VOx microbolometer (Raytheon Vision Systems), 1280x1200 at 12-um pitch, 60 Hz frames, "
              "14-bit ADC, detector time constant 0.012 s; measured FPM NEDT 0.042 +/- 0.006 K viewing a 300-K "
              "scene at f/1; flight-detector average NEDT 0.33 C in an f/1 system at 300 K giving NEP 4.95e-12 W; "
              "TEC-stabilized FPA; bands 7-14, 14-28, 28-80 um; sensor 11.4 kg + vault electronics 1.8 kg + "
              "harness 3.1 kg; 34.8 W average"),
    "Christensen2004_THEMIS": dict(
        cite="Christensen, P. R., Jakosky, B. M., Kieffer, H. H., Malin, M. C., McSween, H. Y., Nealson, K., "
             "Mehall, G. L., Silverman, S. H., Ferry, S., Caplinger, M., and Ravine, M.: The Thermal Emission "
             "Imaging System (THEMIS) for the Mars 2001 Odyssey Mission, Space Sci. Rev., 110, 85-130, 2004, "
             "doi:10.1023/B:SPAC.0000021008.16305.94",
        verified="full text (PDF), abstract and Table I",
        facts="320x240 uncooled microbolometer, 50-um pitch, f/1.6, 12-cm aperture; NEdT 0.5 K at 245 K and "
              "10 um (1-um bands); FPA temperature stabilized to +/-0.001 K; calibration flag protects the "
              "detectors from direct Sun; mass 11.2 kg; orbital-average power 14 W"),
    "Paige2010_Diviner": dict(
        cite="Paige, D. A., Foote, M. C., Greenhagen, B. T., et al.: The Lunar Reconnaissance Orbiter Diviner "
             "Lunar Radiometer Experiment, Space Sci. Rev., 150, 125-160, 2010, doi:10.1007/s11214-009-9529-2",
        verified="full text (Caltech open-access PDF), Table 1",
        facts="nine 21-element linear arrays of uncooled thermopiles, pixel 240 x 480 um, f/1.7, 4-cm apertures, "
              "0.128-s integration; mass/power NOT given in the paper"),
    "McCleese2007_MCS": dict(
        cite="McCleese, D. J., Schofield, J. T., Taylor, F. W., Calcutt, S. B., Foote, M. C., Kass, D. M., "
             "Leovy, C. B., Paige, D. A., Read, P. L., and Zurek, R. W.: Mars Climate Sounder: An investigation "
             "of thermal and water vapor structure, dust and condensate distributions in the atmosphere, and "
             "energy balance of the polar regions, J. Geophys. Res., 112, E05S06, 2007, doi:10.1029/2006JE002790",
        verified="Crossref metadata only (heritage statement; detector numbers taken from Foote2003)",
        facts="nine 21-element thermopile arrays (heritage for Diviner)"),
    "Foote2003_thermopile": dict(
        cite="Foote, M. C., Krueger, T. R., Schofield, J. T., McCleese, D. J., McCann, T. A., Jones, E. W., and "
             "Dickie, M. R.: Space science applications of thermopile detector arrays, Proc. SPIE, 4999, 443-447, "
             "2003, doi:10.1117/12.482474",
        verified="Crossref metadata; numbers from the JPL Thermal Detectors Workshop 2003 version of the same "
                 "paper (NASA NTRS 20040068195, full text)",
        facts="MCS thermopile pixels 240 x 480 um; D* requirement 1e9 cm Hz^1/2 W^-1 (0.03-0.25 Hz, 0.3-45 um); "
              "engineering-model focal plane 8e8 with readout, 1.7e9 without; typical JPL thermopiles 1-2e9 and "
              "response time < 100 ms; dynamic range 13,000 for 2-s integration; linearity 1 %; Johnson-noise "
              "limited (80 kOhm, 36 nV Hz^-1/2)"),
    "Mizrahi2013_VOx17": dict(
        cite="Mizrahi, U., Argaman, N., Elkind, S., Giladi, A., Hirsh, Y., Labilov, M., Pivnik, I., Shiloah, N., "
             "Singer, M., Tuito, A., Ben-Ezra, M., and Shtrichman, I.: Large-format 17 um high-end VOx "
             "u-bolometer infrared detector, Proc. SPIE, 8704, 87041H, 2013, doi:10.1117/12.2015653",
        verified="Crossref metadata; abstract value as indexed by web search (full text not accessed)",
        facts="1024x768 VOx, 17-um pitch, NETD better than 35 mK at F/1, 30 Hz"),
    "Murphy2007_VOx17": dict(
        cite="Murphy, D., Ray, M., Wyles, J., Hewitt, C., Wyles, R., Gordon, E., Almada, K., Sessler, T., "
             "Baur, S., Van Lue, D., and Black, S.: 640 x 512 17 um microbolometer FPA and sensor development, "
             "Proc. SPIE, 6542, 65421Z, 2007, doi:10.1117/12.724345",
        verified="Crossref metadata; abstract value as indexed by web search (full text not accessed)",
        facts="17-um VOx; design target < 50 mK at f/1, 30 Hz with 8-ms time constant (target, not measured)"),
    "Celik2022_VOx17": dict(
        cite="Celik, O., Aydin, O., Aydemir, K., Yidizak, C., Keskin, S., Nuzumlali, O. L., and Yuce, F. G.: "
             "Electro-optical characterization of TEC-less ASELSAN SAFIR640 microbolometer FPA, Proc. SPIE, 12107, "
             "1210715, 2022, doi:10.1117/12.2618225",
        verified="Crossref metadata; abstract value as indexed by web search (full text not accessed)",
        facts="640x480 VOx, 17-um pitch, TEC-less; NETD < 30 mK at f/1; time constant < 12 ms"),
    "Niklaus2007_review": dict(
        cite="Niklaus, F., Vieider, C., and Jakobsen, H.: MEMS-based uncooled infrared bolometer arrays: a review, "
             "Proc. SPIE, 6836, 68360D, 2007, doi:10.1117/12.755128",
        verified="Crossref metadata only (full text not accessed; no number taken from it)",
        facts="review of uncooled bolometer arrays (materials VOx, a-Si, SiGe, metals)"),
    "Wang2025_review": dict(
        cite="Wang, B., Zhao, X., Dong, T., Li, B., Zhang, F., Su, J., Ren, Y., Duan, X., Lin, H., Miao, Y., and "
             "Radamson, H. H.: Convergence of Thermistor Materials and Focal Plane Arrays in Uncooled "
             "Microbolometers: Trends and Perspectives, Nanomaterials, 15, 1316, 2025, doi:10.3390/nano15171316",
        verified="full text (PMC12430745), abstract and Sect. 2",
        facts="state-of-the-art VOx FPAs: NETD below 30 mK and D* near 2e10 cm Hz^1/2 W^-1 (abstract; no "
              "conditions given); Sect. 2.2.5: ~2e10 is the maximum detectivity achievable for a thermal detector "
              "at room temperature (theoretical limit, citing Radamson and Kolahdouz 2012); typical bolometer "
              "response time ~10 ms (citing Rogalski 2022)"),
    "Rogalski2022": dict(
        cite="Rogalski, A.: Scaling infrared detectors - status and outlook, Rep. Prog. Phys., 85, 126501, 2022, "
             "doi:10.1088/1361-6633/ac97a8",
        verified="Crossref metadata only (cited by Wang2025 for the ~10 ms bolometer response time)",
        facts="review"),
    "Cisewski2014_SAGE3ISS": dict(
        cite="Cisewski, M., Zawodny, J., Gasbarre, J., Eckman, R., Topiwala, N., Rodriguez-Alvarez, O., Cheek, D., "
             "and Hall, S.: The Stratospheric Aerosol and Gas Experiment (SAGE III) on the International Space "
             "Station (ISS) Mission, Proc. SPIE, 9241, 924107, 2014, doi:10.1117/12.2073131",
        verified="Crossref metadata + NTRS full text (no instrument mass/power sentence found in it)",
        facts="mission/payload description; instrument is one of three SAGE III copies (Meteor-3M heritage)"),
    "WMO_OSCAR_SAGE3": dict(
        cite="WMO OSCAR instrument record 'SAGE-III', https://space.oscar.wmo.int/instruments/view/sage_iii "
             "(accessed 2026-10-25)",
        verified="web page",
        facts="mass 76 kg; power 80 W; data rate 115 kbps; flown on Meteor-3M (2002-2006) and ISS (2017-)"),
    "NASA_SAGE3_basics": dict(
        cite="NASA SAGE III/ISS education sheet 'SAGE III Basics' (S. Odenwald), "
             "https://sage.nasa.gov/wp/wp-content/uploads/2015/05/SAGE-III-Basics.pdf (accessed 2026-10-25)",
        verified="PDF text",
        facts="'The 76-kilogram instrument'; 700 kWh per year (= 80 W average)"),
    "FLIR_SDR": dict(
        cite="Teledyne FLIR OEM knowledge base, 'What is the scene dynamic range (SDR) of FLIR's OEM modules?' "
             "(vendor page; values as indexed by web search 2026-10-25, page itself returned 404 on fetch)",
        verified="search snippet only -- treated as a PARAMETRIZATION, not a measurement",
        facts="Tau2 640 (17-um VOx): scene range -25 to +135 C in high gain, -40 to +550 C in low gain"),
    "Bernath2005": dict(
        cite="already in references.bib (Bernath et al. 2005, GRL 32, L15S01)",
        verified="paper's own bib",
        facts="ACE-FTS: InSb and HgCdTe detectors, passive radiator; no D* used from it here"),
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def planck_W_m2_sr_um(lam_um, T):
    return planck_W_sr_m2_um(lam_um, T)  # W m^-2 sr^-1 um^-1


def band_exitance_derivative_W_cm2_K(lam1_um, lam2_um, T, n=2000):
    """d/dT of the hemispherical exitance pi*int B dlam over the band, W cm^-2 K^-1."""
    lam = np.linspace(lam1_um, lam2_um, n)
    dT = 0.05
    M1 = np.pi * np.trapezoid(planck_W_m2_sr_um(lam, T + dT), lam)
    M0 = np.pi * np.trapezoid(planck_W_m2_sr_um(lam, T - dT), lam)
    return (M1 - M0) / (2 * dT) * 1e-4  # W m^-2 K^-1 -> W cm^-2 K^-1


def band_radiance_W_m2_sr(lam1_um, lam2_um, T, n=2000):
    lam = np.linspace(lam1_um, lam2_um, n)
    return float(np.trapezoid(planck_W_m2_sr_um(lam, T), lam))


def dstar_from_netd(netd_K, F, pitch_um, frame_Hz, lam1, lam2, T_scene=300.0, tau_opt=1.0):
    """Standard NETD relation with Omega = pi/(4F^2+1), noise bandwidth = frame/2.
    NETD = (4F^2+1) sqrt(df) / (sqrt(A_d) D* tau_opt dM/dT)  ->  D*."""
    df = frame_Hz / 2.0
    A_cm2 = (pitch_um * 1e-4) ** 2
    dMdT = band_exitance_derivative_W_cm2_K(lam1, lam2, T_scene)
    dstar = (4 * F ** 2 + 1) * np.sqrt(df) / (netd_K * np.sqrt(A_cm2) * tau_opt * dMdT)
    return float(dstar), float(dMdT), df


def thermal_detector_limit_dstar(T=300.0, eps=1.0):
    """Radiative-exchange (temperature-fluctuation) limit of an ideal thermal detector,
    D* = sqrt(eps / (16 sigma k T^5)), 2pi FOV, in cm Hz^1/2 W^-1."""
    return float(np.sqrt(eps / (16 * SIGMA_SB * KB * T ** 5)) * 1e2)


def element_od(wl, tau, lam0, half=0.05):
    """-ln<exp(-tau)> over the 0.1-um element centred on lam0 (curve at 0.01-um steps)."""
    m = (wl >= lam0 - half - 1e-9) & (wl <= lam0 + half + 1e-9)
    return float(-np.log(np.mean(np.exp(-tau[m])))), float(np.mean(tau[m])), int(m.sum())


def r(x, n=4):
    """round floats for the json"""
    if isinstance(x, (float, np.floating)):
        return float(f"{x:.{n}g}")
    return x


# ----------------------------------------------------------------------------
# 1. detector catalogue reconciliation (NEP <-> D* for every entry of the ETC)
# ----------------------------------------------------------------------------
def reconcile_catalogue():
    rows = []
    for det in DETECTORS:
        a_cm2 = (det.pitch_um * 1e-4) ** 2
        dstar_implied = np.sqrt(a_cm2) / det.nep
        rows.append(dict(name=det.name, kind=det.kind, pitch_um=det.pitch_um,
                         nep_W_rtHz=r(det.nep), dstar_implied_cm_rtHz_W=r(dstar_implied)))
    flags = [
        "Uncooled microbolometer (VOx): catalogue NEP 1e-11 W/rtHz at 17 um pitch implies D* = 1.7e8; the "
        "Appendix text quotes D* ~ 2e10, which is the radiative-exchange LIMIT of an ideal 300-K thermal "
        "detector (computed below: 1.8e10), not a measured array value; measured arrays give ~1e9 (below). "
        "The two paper numbers are therefore inconsistent by a factor ~100 and neither is the sourced value.",
        "Uncooled thermopile array: catalogue NEP 1e-9 W/rtHz at 100 um implies D* = 1e7; the sourced MCS "
        "thermopiles are 8e8 (with readout) to 1.7e9 (detector alone) -- the catalogue was ~100x too pessimistic. "
        "The Appendix statement that the microbolometer D* is '~20 times the thermopile's' is NOT supported by "
        "the sourced values (both ~1e9).",
        "Pyroelectric (DLaTGS): catalogue NEP 3e-10 W/rtHz at 100 um implies D* = 3.3e7; OTES/L'TES measure "
        "1.2e9 at 10 Hz. AC-only remains true (needs modulation).",
        "Photon detectors (InSb, HgCdTe, Si:As, Si:Sb, T2SL): NEP is derived from D* and pitch by construction, "
        "so they are self-consistent; their D* values are representative (not flight-measured here) and are "
        "flagged as such.",
    ]
    return rows, flags


# ----------------------------------------------------------------------------
# 2. sourced detector set
# ----------------------------------------------------------------------------
def sourced_detectors():
    out = {}
    # (a) VOx microbolometer, 17-um pitch, from measured NETD (and E-THEMIS NEP)
    band = (7.5, 13.5)
    d_miz, dMdT, df_miz = dstar_from_netd(0.035, 1.0, 17.0, 30.0, *band)
    d_cel, _, _ = dstar_from_netd(0.030, 1.0, 17.0, 30.0, *band)   # frame rate not stated; 30 Hz assumed
    d_mur, _, _ = dstar_from_netd(0.050, 1.0, 17.0, 30.0, *band)   # design target
    # E-THEMIS: NEP 4.95e-12 W quoted for the flight detectors (12-um pitch, 60-Hz frames -> df = 30 Hz)
    nep_eth_W = 4.95e-12
    df_eth = 30.0
    nep_eth_density = nep_eth_W / np.sqrt(df_eth)
    d_eth = (12e-4) / nep_eth_density
    d_eth_netd, _, _ = dstar_from_netd(0.042, 1.0, 12.0, 60.0, 7.0, 14.0)  # FPM NEDT 0.042 K at 300 K, f/1
    d_lim = thermal_detector_limit_dstar(300.0)
    adopted_dstar = 1.0e9
    out["vox_microbolometer"] = dict(
        adopted=dict(dstar_cm_rtHz_W=adopted_dstar, pitch_um=17.0,
                     nep_W_rtHz=r(17e-4 / adopted_dstar), tau_thermal_s=0.012, T_op_K=300.0,
                     cooling="none; TEC stabilization of the FPA (E-THEMIS, THEMIS +/-1 mK)",
                     frame_Hz=60.0, adc_bits=14,
                     basis="conservative round value of the measured range below; tau from E-THEMIS"),
        derived_from_measurements=dict(
            Mizrahi2013_NETD35mK_f1_30Hz_17um=r(d_miz),
            Celik2022_NETD30mK_f1_17um_30Hz_assumed=r(d_cel),
            Murphy2007_target_NETD50mK_f1_30Hz_17um=r(d_mur),
            ETHEMIS_NEP4p95em12W_12um_df30Hz=r(d_eth),
            ETHEMIS_FPM_NEDT0p042K_f1_60Hz_12um_7to14um=r(d_eth_netd),
            dM_dT_7p5_13p5um_300K_W_cm2_K=r(dMdT),
            netd_formula="NETD = (4F^2+1) sqrt(df)/(sqrt(A_d) D* tau_o dM/dT), df = frame/2, tau_o = 1",
            Wang2025_abstract_claim=2.0e10,
            ideal_thermal_detector_limit_300K_2pi=r(d_lim),
        ),
        inconsistency_statement=(
            f"Paper text D* ~2e10 = ideal-thermal-detector limit ({d_lim:.2g}); ETC catalogue 1e-11 W/rtHz at "
            f"17 um = {17e-4/1e-11:.2g}; measured arrays {min(d_mur,d_miz,d_cel,d_eth):.2g}-"
            f"{max(d_mur,d_miz,d_cel,d_eth):.2g}. Adopted {adopted_dstar:.1g}."),
        sources=["Mizrahi2013_VOx17", "Celik2022_VOx17", "Murphy2007_VOx17", "Christensen2024_ETHEMIS",
                 "Wang2025_review", "Christensen2004_THEMIS"],
    )
    # (b) thermopile (MCS/Diviner class)
    out["thermopile_linear_array"] = dict(
        adopted=dict(dstar_cm_rtHz_W=8.0e8, pitch_um=None, pixel_um=[240.0, 480.0],
                     nep_W_rtHz=r(np.sqrt(240e-4 * 480e-4) / 8.0e8), tau_thermal_s=0.1, T_op_K=300.0,
                     cooling="none", dstar_detector_alone=1.7e9, dstar_band_Hz=[0.03, 0.25],
                     dynamic_range_2s=13000, linearity=0.01,
                     basis="MCS engineering-model focal plane with readout (Foote2003); Diviner uses the same class"),
        caveats=["D* specified at 0.03-0.25 Hz (2-s integrations); response time <= 100 ms, so a 0.17-s dwell "
                 "is only ~2 time constants; linear 21-element arrays, not 2-D formats"],
        sources=["Foote2003_thermopile", "Paige2010_Diviner", "McCleese2007_MCS"],
    )
    # (c) DLATGS pyroelectric (OTES/L'TES)
    out["dlatgs_pyroelectric"] = dict(
        adopted=dict(dstar_cm_rtHz_W=1.2e9, pitch_um=None, nep_W_rtHz=None, T_op_K=295.0,
                     cooling="none", modulation_required=True,
                     basis="OTES/L'TES measured D* 1.2e9 at 10 Hz, 22 C, single element, AC-coupled; in an FTS "
                           "the moving mirror modulates the signal; a staring spectrograph would need a chopper"),
        sources=["Christensen2018_OTES", "Christensen2024_LTES"],
    )
    # (d) HgCdTe (ACE-FTS heritage) -- representative catalogue value, flagged
    hg = [d for d in DETECTORS if d.name == "HgCdTe LWIR"][0]
    out["hgcdte_lwir"] = dict(
        adopted=dict(dstar_cm_rtHz_W=5.0e10, pitch_um=hg.pitch_um, nep_W_rtHz=r(hg.nep), T_op_K=77.0,
                     cooling="passive radiator 80-100 K (ACE-FTS, Bernath2005)",
                     basis="REPRESENTATIVE catalogue value; no flight-measured D* sourced in this round"),
        sources=["Bernath2005"],
    )
    si = [d for d in DETECTORS if d.name == "Si:As IBC/BIB"][0]
    out["si_as_bib"] = dict(
        adopted=dict(dstar_cm_rtHz_W=1.0e11, pitch_um=si.pitch_um, nep_W_rtHz=r(si.nep), T_op_K=7.0,
                     basis="REPRESENTATIVE catalogue value (JWST/MIRI heritage); flagged"),
        sources=[],
    )
    return out


# ----------------------------------------------------------------------------
# 3. geometry, optics mapping
# ----------------------------------------------------------------------------
def geometry(instr: Instrument, pitch_um=17.0, n_rows=2, n_spec=2):
    h_orb = 420e3
    z_t = 20e3
    R = R_EARTH_M + h_orb
    v_orb = np.sqrt(GM_EARTH / R)
    sinphi = (R_EARTH_M + z_t) / R
    cosphi = np.sqrt(1 - sinphi ** 2)
    range_geom = R * cosphi                       # satellite -> tangent point
    rate_inplane = v_orb * cosphi                 # dz_t/dt for a Sun in the orbit plane (maximum)
    range_adopted = instr.tangent_range_m         # ETC / paper value 3000 km
    theta_v = instr.dz_m / range_adopted          # angular height of one 0.5-km sample
    theta_sun = 2 * R_SUN / AU                    # full solar diameter
    omega_sun = np.pi * (R_SUN / AU) ** 2
    f_strip = instr.disk_fraction()
    omega_strip = f_strip * omega_sun
    A_ap = instr.A_aperture_m2
    etendue = A_ap * omega_strip
    # focal length so that the 0.5-km sample maps on n_rows pixel rows
    f_len = n_rows * pitch_um * 1e-6 / theta_v
    F_num = f_len / instr.D_aperture_m
    M_cols = theta_sun * f_len / (pitch_um * 1e-6)       # solar diameter in pixels (central strip)
    N_pix = n_spec * M_cols
    omega_pix = np.pi / (4 * F_num ** 2)
    A_pix = (pitch_um * 1e-6) ** 2
    etendue_pix_total = N_pix * A_pix * omega_pix
    # smaller-geometry variant
    theta_v_geom = instr.dz_m / range_geom
    inst_geom = Instrument(); inst_geom.tangent_range_m = range_geom
    f_strip_geom = inst_geom.disk_fraction()
    rates = np.array([1.0e3, 3.0e3])             # adopted descent-rate span (m/s)
    dwell = instr.dz_m / rates                   # 0.5 km / rate
    diff = {lam: 2.44 * lam * 1e-6 * F_num / (pitch_um * 1e-6) for lam in (8.80, 11.40, 20.40)}
    return dict(
        orbit_altitude_km=h_orb / 1e3, tangent_height_km=z_t / 1e3, v_orbit_km_s=r(v_orb / 1e3),
        tangent_range_geometric_km=r(range_geom / 1e3), tangent_range_adopted_km=range_adopted / 1e3,
        descent_rate_inplane_max_km_s=r(rate_inplane / 1e3), descent_rate_adopted_km_s=[1.0, 3.0],
        dwell_per_sample_s=[r(dwell[1]), r(dwell[0])],
        strip_angular_height_mrad=r(theta_v * 1e3), strip_angular_height_geom_mrad=r(theta_v_geom * 1e3),
        solar_diameter_mrad=r(theta_sun * 1e3), solar_disk_sr=r(omega_sun),
        strip_fraction_of_disk=r(f_strip), strip_fraction_geom_range=r(f_strip_geom),
        strip_solid_angle_sr=r(omega_strip), aperture_area_m2=r(A_ap), etendue_m2_sr=r(etendue),
        pitch_um=pitch_um, rows_per_sample=n_rows, pixels_per_element_spectral=n_spec,
        focal_length_m=r(f_len), f_number=r(F_num), solar_diameter_pixels=r(M_cols),
        pixels_per_element_total=r(N_pix), pixel_solid_angle_sr=r(omega_pix),
        etendue_check_pixels_m2_sr=r(etendue_pix_total),
        slit="narrow dimension = altitude = n_rows pixels = 0.1-um element width in the spectral direction; "
             "long dimension along the solar diameter (M columns)",
        array_format_needed=f"{int(np.ceil(2*47))} spectral x {int(np.ceil(M_cols))} spatial pixels for 47 elements "
                            f"(a 320x240 array suffices)",
        diffraction_spot_pixels={f"{k:.2f}": r(v) for k, v in diff.items()},
        diffraction_note="2.44 lambda F/pitch; > 2 pixels means the 0.1-um element (2 pixels) is blurred by "
                         "diffraction at this F-number -- a design constraint, strongest at 20.4 um",
    ), dict(theta_v=theta_v, theta_sun=theta_sun, f_strip=f_strip, A_ap=A_ap, f_len=f_len, F_num=F_num,
            M_cols=M_cols, N_pix=N_pix, omega_pix=omega_pix, A_pix=A_pix, dwell=dwell, n_spec=n_spec,
            range_geom=range_geom, f_strip_geom=f_strip_geom)


# ----------------------------------------------------------------------------
# 4. atmospheric optical depths at the 20-km tangent
# ----------------------------------------------------------------------------
def optical_depths():
    d = np.load(NPZ)
    wl, gas = d["p1_wl"], d["p1_gas"]
    tot = gas + d["p1_sulfate"] + d["p1_rayleigh"]
    out = {"source": str(NPZ.relative_to(ROOT)) + " (20-km tangent, reproduce/plot_slant_od_landscape_paper.py)"}
    for lam in (8.80, 11.40, 20.40):
        tg, mg, n = element_od(wl, gas, lam)
        tt, _, _ = element_od(wl, tot, lam)
        out[f"{lam:.2f}"] = dict(tau_gas_element=r(tg), tau_gas_mean=r(mg), tau_total_element=r(tt), n_grid=n)
    # window edge: largest element OD among the 15 window elements 7.80..9.20
    edge = {}
    for lam in np.round(np.arange(7.80, 9.2001, 0.1), 2):
        tg, _, _ = element_od(wl, gas, lam)
        edge[f"{lam:.2f}"] = r(tg)
    kmax = max(edge, key=edge.get)
    out["window_elements_tau_gas"] = edge
    out["window_edge_max"] = dict(element_um=kmax, tau_gas=edge[kmax])
    return out


# ----------------------------------------------------------------------------
# 5. the chain
# ----------------------------------------------------------------------------
def chain(lam_um, tau_atm, det, g, instr: Instrument, dwell_s, n_ref=100, E_scale=1.0, qe=0.8,
          snr_cap=2000.0, floor_m1=None):
    lam_m = lam_um * 1e-6
    e_ph = H * C / lam_m
    E_disk = solar_spectral_irradiance_W_m2_um(lam_um)           # 5772-K disk model
    E = E_disk * E_scale                                          # E490-anchored
    P_toa = E * instr.dlam_um * g["A_ap"] * instr.tau_optics * g["f_strip"]   # W on the element, unattenuated
    T = np.exp(-tau_atm)
    P_atm = P_toa * T
    N_pix = g["N_pix"]
    # warm optics on one pixel (its 0.05-um slice, within Omega_pix)
    dlam_pix = instr.dlam_um / g["n_spec"]
    P_bkg_pix = (instr.eps_optics * planck_W_m2_sr_um(lam_um, instr.T_optics_K) * dlam_pix
                 * g["A_pix"] * g["omega_pix"])
    P_bkg_elem = P_bkg_pix * N_pix
    # photon counts per sample
    n_toa = qe * P_toa * dwell_s / e_ph
    n_atm = qe * P_atm * dwell_s / e_ph
    n_bkg = qe * P_bkg_elem * dwell_s / e_ph
    # detector noise over the N_pix pixels, noise bandwidth 1/(2 dwell) (white-noise averaging assumption)
    df = 1.0 / (2.0 * dwell_s)
    if det.get("nep_W_rtHz") is not None and det.get("pitch_um") is not None:
        # array detector: pixel NEP density, summed over N_pix pixels; A_pix of the design pitch
        nep_pix = np.sqrt(g["A_pix"] * 1e4) / det["dstar_cm_rtHz_W"]    # W/rtHz for the design pixel
        nep_elem = nep_pix * np.sqrt(N_pix)                                # = sqrt(A_total)/D*
        note = "sqrt(A_total)/D*, A_total = N_pix A_pix (design pitch)"
    else:
        # single-element thermal detector matched to the element etendue (same A_total)
        nep_elem = np.sqrt(N_pix * g["A_pix"] * 1e4) / det["dstar_cm_rtHz_W"]
        note = "single element of area A_total = N_pix A_pix, sqrt(A_total)/D*"
    sig_det = nep_elem * np.sqrt(df)                                       # W rms per sample
    # single-sample SNRs (random terms)
    def snr_single(P, n_sig):
        inv2 = (n_sig + n_bkg) / n_sig ** 2 + (sig_det / P) ** 2
        return 1.0 / np.sqrt(inv2), n_sig / np.sqrt(n_sig + n_bkg), P / sig_det
    snr_toa, shot_toa, nep_toa = snr_single(P_toa, n_toa)
    snr_atm, shot_atm, nep_atm = snr_single(P_atm, n_atm)
    snr_ref = snr_toa * np.sqrt(n_ref)
    sigT_T = np.sqrt(1 / snr_atm ** 2 + 1 / snr_ref ** 2)
    snr_T = 1.0 / sigT_T
    snr_net = 1.0 / np.sqrt(1 / snr_T ** 2 + 1 / snr_cap ** 2)
    sig_alpha_random = K_GEOM / snr_T
    sig_alpha_net = K_GEOM / snr_net
    out = dict(
        inputs=dict(lambda_um=lam_um, tau_atm=r(tau_atm), T=r(T), dwell_s=r(dwell_s), n_ref_samples=n_ref,
                    E_disk_5772K_W_m2_um=r(E_disk), E_scale_to_E490=r(E_scale), E_used_W_m2_um=r(E),
                    qe_absorptance=qe, noise_bandwidth_Hz=r(df), detector=det.get("label"),
                    dstar_cm_rtHz_W=det["dstar_cm_rtHz_W"]),
        signal=dict(P_element_TOA_W=r(P_toa), P_element_atm_W=r(P_atm), P_pixel_TOA_W=r(P_toa / N_pix),
                    P_pixel_atm_W=r(P_atm / N_pix), P_bkg_pixel_W=r(P_bkg_pix), P_bkg_element_W=r(P_bkg_elem),
                    bkg_over_signal_pixel_atm=r(P_bkg_pix / (P_atm / N_pix)),
                    photons_TOA_per_sample=r(n_toa), photons_atm_per_sample=r(n_atm),
                    photons_bkg_per_sample=r(n_bkg)),
        noise=dict(nep_element_W_rtHz=r(nep_elem), nep_note=note, sigma_det_per_sample_W=r(sig_det),
                   snr_shot_TOA=r(shot_toa), snr_nep_TOA=r(nep_toa), snr_shot_atm=r(shot_atm),
                   snr_nep_atm=r(nep_atm)),
        result=dict(SNR_single_TOA=r(snr_toa), SNR_single_atm=r(snr_atm), SNR_reference_100samples=r(snr_ref),
                    sigmaT_over_T=r(sigT_T), SNR_T_ratio=r(snr_T), SNR_net_with_cap2000=r(snr_net),
                    sigma_alpha_random_m1=r(sig_alpha_random), sigma_alpha_net_m1=r(sig_alpha_net),
                    limited_by=("systematics cap" if snr_T > snr_cap else
                                ("detector NEP" if nep_atm < shot_atm else "photon shot")),
                    floor_m1=floor_m1, floor_ratio_random=(r(sig_alpha_random / floor_m1) if floor_m1 else None)),
    )
    return out


# ----------------------------------------------------------------------------
# 6. saturation / dynamic range, ratio cancellation, drift requirement
# ----------------------------------------------------------------------------
def saturation(g, instr, E_scale, det_vox):
    lam = 8.80
    E = solar_spectral_irradiance_W_m2_um(lam) * E_scale
    # per-pixel solar power in the spectrograph (pixel passband 0.05 um, F = design F-number, throughput)
    dlam_pix = instr.dlam_um / g["n_spec"]
    B_sun = E / (np.pi * (R_SUN / AU) ** 2)                      # solar radiance W m^-2 sr^-1 um^-1
    P_sun_pix = B_sun * dlam_pix * g["A_pix"] * g["omega_pix"] * instr.tau_optics
    # datasheet reference scene: f/1 imaging, 7.5-13.5 um, tau = 1, emissivity 1
    omega_f1 = np.pi / (4 * 1.0 ** 2 + 1)
    def P_scene(T):
        return band_radiance_W_m2_sr(7.5, 13.5, T) * g["A_pix"] * omega_f1
    P300 = P_scene(300.0)
    P_sat_hi = P_scene(273.15 + 135.0)
    P_sat_lo = P_scene(273.15 + 550.0)
    # broadband imaging of the Sun (no spectrograph): pixel filled by the solar disk, 7.5-13.5 um, f/1
    P_sun_broad = band_radiance_W_m2_sr(7.5, 13.5, T_SUN) * E_scale * g["A_pix"] * omega_f1
    # equivalent scene temperature of the spectrograph per-pixel solar power (f/1 reference band)
    Ts = np.linspace(150, 1500, 5000)
    Ps = np.array([P_scene(t) for t in Ts[::50]])
    T_equiv = float(np.interp(P_sun_pix, Ps, Ts[::50]))
    att_hi = max(1.0, P_sun_pix / (0.5 * P_sat_hi))
    att_lo = max(1.0, P_sun_pix / (0.5 * P_sat_lo))
    # per-frame, per-pixel SNR and ADC headroom
    nep_pix = np.sqrt(g["A_pix"] * 1e4) / det_vox["dstar_cm_rtHz_W"]
    sig_frame = nep_pix * np.sqrt(det_vox["frame_Hz"] / 2)
    snr_pix_frame_toa = P_sun_pix / sig_frame
    return dict(
        wavelength_um=lam, pixel_passband_um=dlam_pix, F_number=r(g["F_num"]), throughput=instr.tau_optics,
        P_sun_pixel_spectrograph_W=r(P_sun_pix), P_300K_scene_pixel_f1_7p5_13p5um_W=r(P300),
        ratio_sun_pixel_to_300K_scene=r(P_sun_pix / P300), equivalent_scene_T_K=r(T_equiv),
        P_sun_pixel_broadband_imaging_f1_W=r(P_sun_broad), ratio_broadband_sun_to_300K=r(P_sun_broad / P300),
        saturation_parametrization=dict(
            source="FLIR_SDR (vendor; parametrization)", T_sat_high_gain_C=135.0, T_sat_low_gain_C=550.0,
            P_sat_high_gain_W=r(P_sat_hi), P_sat_low_gain_W=r(P_sat_lo),
            attenuation_to_half_well_high_gain=r(att_hi), attenuation_to_half_well_low_gain=r(att_lo)),
        per_pixel_per_frame=dict(nep_pixel_W_rtHz=r(nep_pix), sigma_frame_W=r(sig_frame),
                                 SNR_pixel_frame_TOA=r(snr_pix_frame_toa),
                                 bits_needed_TOA_over_noise=r(np.log2(snr_pix_frame_toa)),
                                 adc_bits_ETHEMIS=14),
        statement=(f"In an R~100 spectrograph each pixel receives a 0.05-um slice of the solar spectrum at the "
                   f"design F-number: {P_sun_pix/P300:.2f} of a 300-K f/1 scene (the configuration in which "
                   f"microbolometer NETDs are measured; equivalent scene temperature {T_equiv:.0f} K), so the Sun "
                   f"does not saturate the array; a broadband 7.5-13.5-um imager pointed at the Sun receives "
                   f"{P_sun_broad/P300:.0f}x a 300-K scene and does saturate, which is why imagers carry a Sun "
                   f"flag (THEMIS). The margin depends on the gain state; no attenuator is required at the adopted "
                   f"parameters, but the saturation figure is a vendor parametrization, not a measured flight value."),
    )


def ratio_cancellation(g, instr, E_scale, det_vox, budget=5e-4):
    lam = 8.80
    tau = 1.2
    T = np.exp(-tau)
    E = solar_spectral_irradiance_W_m2_um(lam) * E_scale
    P_toa = E * instr.dlam_um * g["A_ap"] * instr.tau_optics * g["f_strip"]
    dlam_pix = instr.dlam_um / g["n_spec"]
    P_bkg_pix = planck_W_m2_sr_um(lam, instr.T_optics_K) * dlam_pix * g["A_pix"] * g["omega_pix"]
    P_bkg_elem = P_bkg_pix * g["N_pix"]
    # additive offset: T_meas = (S T + dB)/S -> dT/T = dB/(S T)
    dB_allowed = budget * P_toa * T
    frac_bkg_knowledge = dB_allowed / P_bkg_elem
    # quadratic nonlinearity R(S) = S (1 + e2 S/S_toa): T_meas/T = (1 + e2 T)/(1 + e2) ~ 1 - e2 (1 - T)
    e2_allowed = budget / (1 - T)
    # gain drift between reference block (80-100 km, centre 90 km) and the 20-km sample
    dz_km = 90.0 - 20.0
    dt = dz_km / np.array([3.0, 1.0])             # s, for 3 and 1 km/s
    t_event = 100.0 / np.array([3.0, 1.0])        # 0-100 km
    drift_rate = budget / dt                      # per second
    # FPA temperature stability from TCR
    tcr = np.array([0.02, 0.042])                 # /K, VOx typical to Al-doped (Wang2025)
    dT_fpa = budget / tcr
    return dict(
        budget_fractional=budget, lambda_um=lam, tau=tau, T=r(T),
        cancelled_by_ratio=["absolute radiometric calibration", "optical throughput (mirrors, grating, filters)",
                            "detector responsivity and electronic gain, if constant over the event",
                            "the solar spectrum itself (same disk strip, same spectral slice)",
                            "slow degradation of optics/detector between events (Chu1989)"],
        not_cancelled=["additive offsets (warm-optics emission, dark level, stray light) -- must be measured "
                       "(space look before/after the event) and subtracted",
                       "nonlinearity between the TOA level and the attenuated level",
                       "gain/responsivity drift within the event (FPA temperature, 1/f noise)",
                       "pointing-dependent solar-disk structure (limb darkening) if the strip moves on the disk",
                       "spectral registration drift (grating/thermal)"],
        offset_requirement=dict(P_element_TOA_W=r(P_toa), P_bkg_element_W=r(P_bkg_elem),
                                dB_allowed_W=r(dB_allowed),
                                fraction_of_background_to_know=r(frac_bkg_knowledge),
                                statement="the 300-K background on the element must be known/subtracted to this "
                                          "fraction of itself"),
        nonlinearity_requirement=dict(quadratic_coefficient_e2_max=r(e2_allowed),
                                      statement="response R(S)=S(1+e2 S/S_TOA): |e2| below this over the "
                                                "TOA-to-30% range; MCS thermopile spec is 1 % (Foote2003), "
                                                "so a 1 %-class linearity would need characterization/correction "
                                                "by a factor ~14"),
        drift_requirement=dict(dt_ref_to_sample_s=[r(dt[0]), r(dt[1])], event_0_100km_s=[r(t_event[0]), r(t_event[1])],
                               gain_drift_rate_max_per_s=[r(drift_rate[0]), r(drift_rate[1])],
                               gain_drift_rate_max_per_min=[r(drift_rate[0] * 60), r(drift_rate[1] * 60)],
                               statement="a requirement, not a demonstration"),
        fpa_temperature_requirement=dict(TCR_per_K=[0.02, 0.042], dT_fpa_max_K=[r(dT_fpa[0]), r(dT_fpa[1])],
                                         heritage="THEMIS stabilizes the IR FPA to +/-0.001 K (Christensen2004)",
                                         source_tcr="Wang2025 (Al-doped VOx -4.2 %/K; typical VOx ~ -2 %/K)"),
    )


def vertical_resolution(g, instr):
    theta_v = g["theta_v"]
    rng = instr.tangent_range_m
    rates = np.array([1.0, 3.0])  # km/s
    dwell = 0.5 / rates
    smear_km = rates * dwell  # = 0.5 km by construction
    # FOV boxcar (0.5 km) convolved with smear boxcar (0.5 km): triangle, base 1.0 km, FWHM 0.5 km
    fov_km = theta_v * rng / 1e3
    base_km = fov_km + smear_km
    fwhm_km = np.maximum(fov_km, smear_km)
    # pointing knowledge to hold 0.1 km (20 % of the sample) at the adopted and geometric ranges
    dz_req = 0.1e3
    for_ranges = {"adopted_3000km": rng, "geometric_2296km": g["range_geom"]}
    pointing = {k: dict(urad=r(dz_req / v * 1e6), arcsec=r(dz_req / v * 206265)) for k, v in for_ranges.items()}
    diff_blur_rows = 2.44 * 8.80e-6 * g["F_num"] / (17e-6)  # pixels at 8.80 um
    return dict(
        fov_per_sample_km=r(fov_km), fov_mrad=r(theta_v * 1e3), rows_per_sample=2,
        integration_smear_km=[r(smear_km[0]), r(smear_km[1])], dwell_s=[r(dwell[0]), r(dwell[1])],
        combined_response="boxcar(FOV) * boxcar(smear): triangle",
        combined_base_km=[r(base_km[0]), r(base_km[1])], combined_fwhm_km=[r(fwhm_km[0]), r(fwhm_km[1])],
        diffraction_blur_8p80um_pixels=r(diff_blur_rows),
        diffraction_blur_8p80um_km=r(diff_blur_rows / 2 * fov_km),
        pointing_knowledge_for_0p1km=pointing,
        onion_peel="shell k is bounded by successive tangent heights z_k, z_k+1 (0.5 km apart); the retrieved "
                   "shell extinction is the path-weighted average over the net response above, so the retrieval "
                   "resolution is >= the 0.5-km grid and ~0.5-1 km FWHM-to-base for these terms; the onion-peel "
                   f"gain g = {G_ONION:.3f} (saimon.onion_peel) is already in K_GEOM",
        statement="0.5 km is the SAMPLING; the resolution budget (FOV, smear, diffraction, pointing, retrieval "
                  "response) gives ~0.5 km FWHM with a 1-km base for the adopted dwell = 0.5 km / descent rate; "
                  "none of this is demonstrated -- these are the requirement-level numbers",
    )


MASS_POWER = [
    dict(instrument="SAGE III/ISS (solar-occultation grating spectrometer, UV-NIR)", mass_kg=76.0, power_W=80.0,
         scope="instrument (OSCAR); the ISS payload with Nadir Viewing Platform and hexapod is 527 kg (eoportal)",
         sources=["WMO_OSCAR_SAGE3", "NASA_SAGE3_basics", "Cisewski2014_SAGE3ISS"]),
    dict(instrument="THEMIS / Mars Odyssey (uncooled microbolometer imager, 6.8-14.9 um + VIS)", mass_kg=11.2,
         power_W=14.0, scope="instrument, orbital-average power", sources=["Christensen2004_THEMIS"]),
    dict(instrument="E-THEMIS / Europa Clipper (uncooled microbolometer imager, 7-80 um)",
         mass_kg=11.4 + 1.8 + 3.1, power_W=34.8, scope="sensor 11.4 + vault electronics 1.8 + harness 3.1 kg; "
         "average operating power", sources=["Christensen2024_ETHEMIS"]),
    dict(instrument="OTES / OSIRIS-REx (uncooled DLATGS FTS, 5.7-100 um)", mass_kg=6.27, power_W=10.8,
         scope="instrument; 15.99 W peak", sources=["Christensen2018_OTES"]),
    dict(instrument="L'TES / Lucy (uncooled DLATGS FTS, 5.7-100 um)", mass_kg=6.47, power_W=12.6,
         scope="instrument; 18.0 W peak", sources=["Christensen2024_LTES"]),
    dict(instrument="Diviner / LRO (thermopile radiometer)", mass_kg=None, power_W=None,
         scope="not given in Paige2010; not quoted", sources=["Paige2010_Diviner"]),
]

TABLE_D2_CORRECTIONS = dict(
    thermopile_row="Uncooled thermopile array | 2--45 | ~300 K; none | 10--100 (Diviner, MCS)  "
                   "[remove OTES and L'TES from this row]",
    pyroelectric_row="Pyroelectric (DLaTGS) | 2--40 | ~300 K | high-R in FTS; AC-only (Mini-TES, OTES, L'TES)",
    caption_ethemis="E-THEMIS on Europa Clipper (Christensen2024) flies uncooled microbolometer bands at 14--28 and "
                    "28--80 um, which demonstrates microbolometer spectral response across the long-wave bands "
                    "considered here; it does not verify the SNR of a solar-occultation instrument.",
    caption_photon="Among the candidates surveyed in this table, the only photon detectors reaching 20--29 um are "
                   "the extrinsic-silicon devices (Si:As, Si:Sb), which need sub-10 K cooling; this is a statement "
                   "about the surveyed candidates, not a survey of all photon-detector technologies.",
    text_heritage="OTES and L'TES use a single uncooled DLATGS pyroelectric detector (Christensen et al. 2018, "
                  "Sect. 2.3.2; Christensen et al. 2024), not thermopiles; thermopile heritage is Diviner and MCS "
                  "(Paige et al. 2010; McCleese et al. 2007; Foote et al. 2003).",
    text_dstar="Replace 'D* (~2e10) is ~20 times the thermopile's' by the sourced values: VOx microbolometer "
               "arrays ~1e9 cm Hz^1/2 W^-1 (NETD 30-50 mK at f/1, 17-um pitch; E-THEMIS flight NEP 4.95e-12 W at "
               "12 um), MCS thermopiles 0.8-1.7e9, OTES/L'TES DLATGS 1.2e9 at 10 Hz; 2e10 is the ideal "
               "300-K thermal-detector limit. The microbolometer is preferred for its staring 2-D format and "
               "~12-ms time constant, not for a D* advantage.",
)

BIBTEX = r"""
@article{Christensen2018,
  author  = {Christensen, P. R. and Hamilton, V. E. and Mehall, G. L. and Pelham, D. and O'Donnell, W. and Anwar, S. and Bowles, H. and Chase, S. and Fahlgren, J. and Farkas, Z. and Fisher, T. and James, O. and Kubik, I. and Lazbin, I. and Miner, M. and Rassas, M. and Schulze, L. and Shamordola, K. and Tourville, T. and West, G. and Woodward, R. and Lauretta, D.},
  title   = {The {OSIRIS-REx} {Thermal} {Emission} {Spectrometer} ({OTES}) Instrument},
  journal = {Space Sci. Rev.},
  volume  = {214},
  pages   = {87},
  year    = {2018},
  doi     = {10.1007/s11214-018-0513-6}
}

@article{Christensen2024ltes,
  author  = {Christensen, P. R. and Hamilton, V. E. and Mehall, G. L. and Anwar, S. and Bowles, H. and Chase, S. and Farkas, Z. and Fisher, T. and Holmes, A. and Kubik, I. and Lazbin, I. and O'Donnell, W. and Ortiz, C. and Pelham, D. and Rogers, S. and Shamordola, K. and Tourville, T. and Woodward, R.},
  title   = {The {Lucy} {Thermal} {Emission} {Spectrometer} ({L'TES}) Instrument},
  journal = {Space Sci. Rev.},
  volume  = {220},
  pages   = {1},
  year    = {2024},
  doi     = {10.1007/s11214-023-01029-y}
}

@article{Christensen2004,
  author  = {Christensen, P. R. and Jakosky, B. M. and Kieffer, H. H. and Malin, M. C. and McSween, H. Y. and Nealson, K. and Mehall, G. L. and Silverman, S. H. and Ferry, S. and Caplinger, M. and Ravine, M.},
  title   = {The {Thermal} {Emission} {Imaging} {System} ({THEMIS}) for the {Mars} 2001 {Odyssey} Mission},
  journal = {Space Sci. Rev.},
  volume  = {110},
  pages   = {85--130},
  year    = {2004},
  doi     = {10.1023/B:SPAC.0000021008.16305.94}
}

@article{Paige2010,
  author  = {Paige, D. A. and Foote, M. C. and Greenhagen, B. T. and Schofield, J. T. and Calcutt, S. and Vasavada, A. R. and Preston, D. J. and Taylor, F. W. and Allen, C. C. and Snook, K. J. and Jakosky, B. M. and Murray, B. C. and Soderblom, L. A. and Jau, B. and Loring, S. and Bulharowski, J. and Bowles, N. E. and Thomas, I. R. and Sullivan, M. T. and Avis, C. and De Jong, E. M. and Hartford, W. and McCleese, D. J.},
  title   = {The {Lunar} {Reconnaissance} {Orbiter} {Diviner} {Lunar} {Radiometer} Experiment},
  journal = {Space Sci. Rev.},
  volume  = {150},
  pages   = {125--160},
  year    = {2010},
  doi     = {10.1007/s11214-009-9529-2}
}

@article{McCleese2007,
  author  = {McCleese, D. J. and Schofield, J. T. and Taylor, F. W. and Calcutt, S. B. and Foote, M. C. and Kass, D. M. and Leovy, C. B. and Paige, D. A. and Read, P. L. and Zurek, R. W.},
  title   = {{Mars} {Climate} {Sounder}: An investigation of thermal and water vapor structure, dust and condensate distributions in the atmosphere, and energy balance of the polar regions},
  journal = {J. Geophys. Res.},
  volume  = {112},
  eid     = {E05S06},
  year    = {2007},
  doi     = {10.1029/2006JE002790}
}

@inproceedings{Foote2003,
  author    = {Foote, M. C. and Krueger, T. R. and Schofield, J. T. and McCleese, D. J. and McCann, T. A. and Jones, E. W. and Dickie, M. R.},
  title     = {Space science applications of thermopile detector arrays},
  booktitle = {Quantum Sensing: Evolution and Revolution from Past to Future},
  series    = {Proc. SPIE},
  volume    = {4999},
  pages     = {443--447},
  year      = {2003},
  doi       = {10.1117/12.482474}
}

@inproceedings{Mizrahi2013,
  author    = {Mizrahi, U. and Argaman, N. and Elkind, S. and Giladi, A. and Hirsh, Y. and Labilov, M. and Pivnik, I. and Shiloah, N. and Singer, M. and Tuito, A. and Ben-Ezra, M. and Shtrichman, I.},
  title     = {Large-format 17~$\mu$m high-end {VOx} $\mu$-bolometer infrared detector},
  booktitle = {Infrared Technology and Applications XXXIX},
  series    = {Proc. SPIE},
  volume    = {8704},
  pages     = {87041H},
  year      = {2013},
  doi       = {10.1117/12.2015653}
}

@inproceedings{Murphy2007,
  author    = {Murphy, D. and Ray, M. and Wyles, J. and Hewitt, C. and Wyles, R. and Gordon, E. and Almada, K. and Sessler, T. and Baur, S. and Van Lue, D. and Black, S.},
  title     = {640 $\times$ 512 17~$\mu$m microbolometer {FPA} and sensor development},
  booktitle = {Infrared Technology and Applications XXXIII},
  series    = {Proc. SPIE},
  volume    = {6542},
  pages     = {65421Z},
  year      = {2007},
  doi       = {10.1117/12.724345}
}

@inproceedings{Celik2022,
  author    = {Celik, O. and Aydin, O. and Aydemir, K. and Yidizak, C. and Keskin, S. and Nuzumlali, O. L. and Yuce, F. G.},
  title     = {Electro-optical characterization of {TEC}-less {ASELSAN} {SAFIR640} microbolometer {FPA}},
  booktitle = {Infrared Technology and Applications XLVIII},
  series    = {Proc. SPIE},
  volume    = {12107},
  pages     = {1210715},
  year      = {2022},
  doi       = {10.1117/12.2618225}
}

@article{Wang2025,
  author  = {Wang, B. and Zhao, X. and Dong, T. and Li, B. and Zhang, F. and Su, J. and Ren, Y. and Duan, X. and Lin, H. and Miao, Y. and Radamson, H. H.},
  title   = {Convergence of Thermistor Materials and Focal Plane Arrays in Uncooled Microbolometers: Trends and Perspectives},
  journal = {Nanomaterials},
  volume  = {15},
  pages   = {1316},
  year    = {2025},
  doi     = {10.3390/nano15171316}
}

@inproceedings{Niklaus2007,
  author    = {Niklaus, F. and Vieider, C. and Jakobsen, H.},
  title     = {{MEMS}-based uncooled infrared bolometer arrays: a review},
  booktitle = {MEMS/MOEMS Technologies and Applications III},
  series    = {Proc. SPIE},
  volume    = {6836},
  pages     = {68360D},
  year      = {2007},
  doi       = {10.1117/12.755128}
}

@inproceedings{Cisewski2014,
  author    = {Cisewski, M. and Zawodny, J. and Gasbarre, J. and Eckman, R. and Topiwala, N. and Rodriguez-Alvarez, O. and Cheek, D. and Hall, S.},
  title     = {The {Stratospheric} {Aerosol} and {Gas} {Experiment} ({SAGE III}) on the {International} {Space} {Station} ({ISS}) Mission},
  booktitle = {Sensors, Systems, and Next-Generation Satellites XVIII},
  series    = {Proc. SPIE},
  volume    = {9241},
  pages     = {924107},
  year      = {2014},
  doi       = {10.1117/12.2073131}
}
"""


def paper_statements(ch, solar, geo, dets):
    v = ch["8.80"]["vox_microbolometer"]; w = ch["20.40"]["vox_microbolometer"]
    def rng(key, lam):
        c = ch[lam]["vox_microbolometer"]
        return [c["dwell_0.17"]["result"][key], c["dwell_0.5"]["result"][key]]
    return {
        "collected power 8.80 um '~2.5 uW'": dict(
            new_W=v["dwell_0.5"]["signal"]["P_element_TOA_W"],
            note="E490-anchored disk model; the unscaled 5772-K model gives the paper's 2.5 uW"),
        "collected power 20.4 um '~90 nW'": dict(new_W=w["dwell_0.5"]["signal"]["P_element_TOA_W"]),
        "'~2 % of the solar disk per 0.5-km sample'": dict(strip_fraction=geo["strip_fraction_of_disk"],
            geometric_ISS_value=geo["strip_fraction_geom_range"], strip_mrad=geo["strip_angular_height_mrad"],
            solar_diameter_mrad=geo["solar_diameter_mrad"]),
        "'photon shot-noise SNR ceiling of order 1e6'": dict(
            shot_SNR_single_atm_sample_8p80=[v["dwell_0.17"]["noise"]["snr_shot_atm"], v["dwell_0.5"]["noise"]["snr_shot_atm"]],
            shot_SNR_single_atm_sample_20p4=[w["dwell_0.17"]["noise"]["snr_shot_atm"], w["dwell_0.5"]["noise"]["snr_shot_atm"]],
            note="true as an order of magnitude for a single sample; it is not the transmission precision"),
        "'microbolometer D* ~2e10, ~20x the thermopile'": dict(
            sourced_D_star=dets["vox_microbolometer"]["adopted"]["dstar_cm_rtHz_W"],
            thermopile_D_star=dets["thermopile_linear_array"]["adopted"]["dstar_cm_rtHz_W"],
            ideal_limit=dets["vox_microbolometer"]["derived_from_measurements"]["ideal_thermal_detector_limit_300K_2pi"],
            note="2e10 is the ideal-detector limit; measured arrays ~1e9; ratio to the thermopile ~1, not 20"),
        "'NEP-limited SNR exceeds 1e5' (8.80 um)": dict(
            NEP_SNR_single_atm_sample=[v["dwell_0.17"]["noise"]["snr_nep_atm"], v["dwell_0.5"]["noise"]["snr_nep_atm"]],
            transmission_SNR_after_ratio=rng("SNR_T_ratio", "8.80"),
            note="with the sourced D* and the signal spread over 2 x 112 pixels: 1.4e4-2.4e4, not >1e5; "
                 "still 7-12x above the systematics cap"),
        "'thermopile falls to SNR ~100 at 20.4 um'": dict(
            thermopile_SNR_T_20p4=[ch["20.40"]["thermopile"]["dwell_0.17"]["result"]["SNR_T_ratio"],
                                   ch["20.40"]["thermopile"]["dwell_0.5"]["result"]["SNR_T_ratio"]],
            microbolometer_SNR_T_20p4=rng("SNR_T_ratio", "20.40"),
            note="with sourced D* both are ~1.3e3-3e3 at 20.4 um; the microbolometer's margin over the cap at "
                 "20.4 um is <1.4x and its D* at 20 um is not separately sourced"),
        "systematics cap 2000 and crossover S~400": dict(net_SNR_8p80=rng("SNR_net_with_cap2000", "8.80"),
            random_sigma_alpha_8p80=rng("sigma_alpha_random_m1", "8.80")),
        "'uncooled microbolometer ... thermoelectric stabilization only'": dict(
            supported="yes: E-THEMIS and THEMIS fly TEC-stabilized uncooled VOx arrays; the gain-drift requirement "
                      "translates to 12-25 mK FPA stability over the event against THEMIS's +/-1 mK heritage"),
    }


# ----------------------------------------------------------------------------
# LaTeX table
# ----------------------------------------------------------------------------
def write_table(res, lams, dets_for_table):
    def fmt(x, nd=2):
        if x is None:
            return "--"
        ax = abs(x)
        if ax >= 1e4 or ax < 1e-2:
            m, e = f"{x:.{nd-1}e}".split("e")
            return f"${m}\\times10^{{{int(e)}}}$"
        return f"${x:.{nd}g}$" if ax < 100 else f"${x:.0f}$"
    g = res["geometry"]
    L = []
    L.append("% Round 70 (referee round 5, item 8): first-order radiometric chain, generated by")
    L.append("% reproduce/radiometric_budget.py; numbers in outputs/radiometric_budget.json.")
    L.append("\\begin{table*}[t]")
    L.append("\\caption{\\Doron{Mark - for your review} First-order radiometric chain of the baseline MIR spectrograph "
             "for one $0.5$-km sample at a $20$-km tangent, with the sourced uncooled VO$_x$ microbolometer "
             "($D^{*}=10^{9}$~cm\\,Hz$^{1/2}$\\,W$^{-1}$, $17$-$\\mu$m pitch, $12$-ms time constant). "
             "The solar irradiance is the $5772$-K disk model anchored to the ASTM~E490 value at $10~\\mu$m; the "
             "element signal is spread over $2\\times" + f"{int(round(g['solar_diameter_pixels']))}" +
             "$ pixels (two spectral pixels per $0.1$-$\\mu$m element, the solar diameter across the slit); "
             "detector noise is $\\sqrt{A_\\text{tot}}/D^{*}$ over those pixels with a noise bandwidth $1/(2t)$; "
             "the exo-atmospheric reference is the mean of $100$ samples above $80$~km. ``SNR'' is the per-sample "
             "transmission SNR of the ratio, $T/\\sigma_T$; the systematics cap of $2000$ is applied afterwards. "
             "The two dwell values bracket the $1$--$3$~km\\,s$^{-1}$ descent of the tangent point.}")
    L.append("\\label{tab:radiometric_budget}")
    L.append("\\centering\\footnotesize")
    L.append("\\begin{tabular}{l" + "c" * len(lams) + "}")
    L.append("\\toprule")
    L.append("Quantity & " + " & ".join(f"${l:.2f}~\\mu$m" for l in lams) + " \\\\")
    L.append("\\midrule")
    rows = []
    c = {l: res["chain"][f"{l:.2f}"] for l in lams}
    od = res["optical_depths"]
    rows.append(("Solar irradiance used (W\\,m$^{-2}\\mu$m$^{-1}$)",
                 [c[l]["vox_microbolometer"]["dwell_0.5"]["inputs"]["E_used_W_m2_um"] for l in lams]))
    rows.append(("Gas optical depth of the element at $20$~km",
                 [od[f"{l:.2f}"]["tau_gas_element"] for l in lams]))
    rows.append(("Power on the element, exo-atmospheric ($\\mu$W)",
                 [c[l]["vox_microbolometer"]["dwell_0.5"]["signal"]["P_element_TOA_W"] * 1e6 for l in lams]))
    rows.append(("Power on the element, attenuated ($\\mu$W)",
                 [c[l]["vox_microbolometer"]["dwell_0.5"]["signal"]["P_element_atm_W"] * 1e6 for l in lams]))
    rows.append(("Warm-optics ($300$~K) power per pixel / solar power per pixel",
                 [c[l]["vox_microbolometer"]["dwell_0.5"]["signal"]["bkg_over_signal_pixel_atm"] for l in lams]))
    rows.append(("Detector noise per sample, $t=0.5$~s (pW)",
                 [c[l]["vox_microbolometer"]["dwell_0.5"]["noise"]["sigma_det_per_sample_W"] * 1e12 for l in lams]))
    rows.append(("Shot-noise SNR, attenuated sample, $t=0.5$~s",
                 [c[l]["vox_microbolometer"]["dwell_0.5"]["noise"]["snr_shot_atm"] for l in lams]))
    rows.append(("NEP-limited SNR, attenuated sample, $t=0.5$~s",
                 [c[l]["vox_microbolometer"]["dwell_0.5"]["noise"]["snr_nep_atm"] for l in lams]))
    rows.append(("Transmission SNR $T/\\sigma_T$ after the ratio, $t=0.5$~s",
                 [c[l]["vox_microbolometer"]["dwell_0.5"]["result"]["SNR_T_ratio"] for l in lams]))
    rows.append(("Transmission SNR $T/\\sigma_T$ after the ratio, $t=0.17$~s",
                 [c[l]["vox_microbolometer"]["dwell_0.17"]["result"]["SNR_T_ratio"] for l in lams]))
    rows.append(("Net SNR with the systematics cap $2000$, $t=0.17$~s",
                 [c[l]["vox_microbolometer"]["dwell_0.17"]["result"]["SNR_net_with_cap2000"] for l in lams]))
    rows.append(("Random $\\sigma_\\alpha$, $t=0.17$~s ($10^{-9}$~m$^{-1}$)",
                 [c[l]["vox_microbolometer"]["dwell_0.17"]["result"]["sigma_alpha_random_m1"] * 1e9 for l in lams]))
    rows.append(("Gas-removal floor ($10^{-9}$~m$^{-1}$)",
                 [(res["floors_m1"].get(f"{l:.2f}") or 0) * 1e9 or None for l in lams]))
    for name, vals in rows:
        L.append(name + " & " + " & ".join(fmt(v) for v in vals) + " \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{" + str(len(lams) + 1) + "}{l}{Transmission SNR after the ratio, $t=0.17$~s, other detectors "
             "(sourced $D^{*}$; cap not applied):} \\\\")
    for key, label in dets_for_table:
        vals = [c[l][key]["dwell_0.17"]["result"]["SNR_T_ratio"] if key in c[l] else None for l in lams]
        L.append(label + " & " + " & ".join(fmt(v) for v in vals) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table*}")
    OUT_TEX.write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    instr = Instrument()
    instr.f_number = None  # set from the mapping below
    log = []
    def P(*a):
        s = " ".join(str(x) for x in a); print(s); log.append(s)

    P("Round 70 item 8 -- radiometric budget")
    P(f"K_GEOM = {K_GEOM:.4e} m^-1 (sigma_alpha = K_GEOM/SNR, dz = 0.5 km, g = {G_ONION:.4f})")

    cat_rows, cat_flags = reconcile_catalogue()
    P("\n== catalogue reconciliation (NEP <-> D*) ==")
    for row in cat_rows:
        P(f"  {row['name']:34s} pitch {row['pitch_um']:>6} um  NEP {row['nep_W_rtHz']:.1e} W/rtHz  -> D* {row['dstar_implied_cm_rtHz_W']:.1e}")
    for f in cat_flags:
        P("  FLAG:", f)

    dets = sourced_detectors()
    P("\n== sourced detector set ==")
    for k, v in dets.items():
        P(f"  {k}: {json.dumps(v['adopted'])}")
        if "derived_from_measurements" in v:
            P("     derived:", json.dumps(v["derived_from_measurements"]))

    geo, g = geometry(instr, pitch_um=dets["vox_microbolometer"]["adopted"]["pitch_um"])
    instr.f_number = g["F_num"]
    P("\n== geometry / mapping ==")
    P(json.dumps(geo, indent=1))

    od = optical_depths()
    P("\n== optical depths at the 20-km tangent (element -ln<T>) ==")
    P(json.dumps(od, indent=1))

    # solar irradiance anchor: E490 0.24 W m^-2 um^-1 at 10 um (paper, ASTM_E490)
    E490_10 = 0.24
    E_disk_10 = solar_spectral_irradiance_W_m2_um(10.0)
    E_scale = E490_10 / E_disk_10
    solar = dict(model="5772-K blackbody disk (ETC) scaled to ASTM E490 at 10 um",
                 E490_at_10um_W_m2_um=E490_10, disk_model_at_10um_W_m2_um=r(E_disk_10), scale=r(E_scale),
                 disk_model_values={f"{l:.2f}": r(solar_spectral_irradiance_W_m2_um(l)) for l in (8.80, 11.40, 20.40)},
                 used_values={f"{l:.2f}": r(solar_spectral_irradiance_W_m2_um(l) * E_scale) for l in (8.80, 11.40, 20.40)},
                 note="the paper's 2.5 uW at 8.80 um follows from the unscaled disk model; the E490-anchored value "
                      "is lower by the scale factor")
    P("\n== solar irradiance ==", json.dumps(solar))

    floors = {"8.80": 1.692e-8, "11.40": 1.2e-8, "20.40": 3.4e-9}
    det_list = {
        "vox_microbolometer": dict(label="VOx microbolometer, sourced D* 1e9, 17 um", dstar_cm_rtHz_W=1.0e9, pitch_um=17.0,
                                   nep_W_rtHz=17e-4 / 1e9),
        "vox_catalogue_1e-11": dict(label="VOx microbolometer, ETC catalogue NEP 1e-11 (D* 1.7e8)", dstar_cm_rtHz_W=1.7e8,
                                    pitch_um=17.0, nep_W_rtHz=1e-11),
        "vox_ideal_limit": dict(label="ideal 300-K thermal detector limit D* 1.8e10 (the paper's '2e10')",
                                dstar_cm_rtHz_W=thermal_detector_limit_dstar(), pitch_um=17.0, nep_W_rtHz=1.0),
        "thermopile": dict(label="thermopile, MCS class D* 8e8 (single element of the element area)", dstar_cm_rtHz_W=8.0e8,
                           pitch_um=None, nep_W_rtHz=None),
        "dlatgs": dict(label="DLATGS pyroelectric D* 1.2e9 at 10 Hz (needs a chopper in a staring spectrograph)",
                       dstar_cm_rtHz_W=1.2e9, pitch_um=None, nep_W_rtHz=None),
        "hgcdte": dict(label="HgCdTe LWIR, representative D* 5e10 (flagged)", dstar_cm_rtHz_W=5.0e10, pitch_um=17.0,
                       nep_W_rtHz=1.0),
        "si_as": dict(label="Si:As BIB, representative D* 1e11 (flagged)", dstar_cm_rtHz_W=1.0e11, pitch_um=17.0,
                      nep_W_rtHz=1.0),
    }
    coverage = {"8.80": ["vox_microbolometer", "vox_catalogue_1e-11", "vox_ideal_limit", "thermopile", "dlatgs", "hgcdte"],
                "11.40": ["vox_microbolometer", "vox_catalogue_1e-11", "thermopile", "dlatgs", "hgcdte"],
                "20.40": ["vox_microbolometer", "vox_catalogue_1e-11", "thermopile", "dlatgs", "si_as"]}
    chain_out = {}
    P("\n== chain ==")
    for lam in (8.80, 11.40, 20.40):
        key = f"{lam:.2f}"
        tau = od[key]["tau_total_element"]
        chain_out[key] = {}
        for dk in coverage[key]:
            chain_out[key][dk] = {}
            for dw in g["dwell"]:
                res = chain(lam, tau, det_list[dk], g, instr, dw, E_scale=E_scale, floor_m1=floors[key])
                chain_out[key][dk][f"dwell_{dw:.2g}"] = res
                rr = res["result"]
                P(f"  {key} um  {dk:22s} t={dw:.2f}s  P_toa={res['signal']['P_element_TOA_W']:.2e} W  "
                  f"P_atm={res['signal']['P_element_atm_W']:.2e} W  shot_atm={res['noise']['snr_shot_atm']:.1e}  "
                  f"nep_atm={res['noise']['snr_nep_atm']:.1e}  SNR_T={rr['SNR_T_ratio']:.3g}  "
                  f"net={rr['SNR_net_with_cap2000']:.0f}  sig_a_rand={rr['sigma_alpha_random_m1']:.2e}  [{rr['limited_by']}]")
        # window-edge variant for the microbolometer at 8.80 (tau ~ 2)
    edge_tau = od["window_edge_max"]["tau_gas"]
    chain_out["window_edge"] = {"element_um": od["window_edge_max"]["element_um"], "tau_gas": edge_tau}
    for dw in g["dwell"]:
        res = chain(float(od["window_edge_max"]["element_um"]), edge_tau, det_list["vox_microbolometer"], g, instr, dw,
                    E_scale=E_scale)
        chain_out["window_edge"][f"dwell_{dw:.2g}"] = res
        P(f"  window edge {od['window_edge_max']['element_um']} um tau={edge_tau:.2f} t={dw:.2f}s  "
          f"SNR_T={res['result']['SNR_T_ratio']:.3g}")

    # parametric window-edge case requested by the lead: tau = 2 at an element of the O3 shoulder (8.90 um)
    chain_out["window_edge_tau2_parametric"] = {"element_um": 8.90, "tau": 2.0,
        "note": "paper: 'the 7.8-9.3 um shoulder with tau_O3 <~ 2 in the element average'; the 8.90 element reads "
                f"{od['window_elements_tau_gas']['8.90']} in the archive; tau = 2 taken as the edge case"}
    for dw in g["dwell"]:
        res = chain(8.90, 2.0, det_list["vox_microbolometer"], g, instr, dw, E_scale=E_scale)
        chain_out["window_edge_tau2_parametric"][f"dwell_{dw:.2g}"] = res
        P(f"  window edge parametric 8.90 um tau=2.0 t={dw:.2f}s  SNR_T={res['result']['SNR_T_ratio']:.3g}  "
          f"net={res['result']['SNR_net_with_cap2000']:.0f}")

    # crossover with the gas-removal floor
    cross = dict(K_GEOM_m1=r(K_GEOM), floor_8p80_m1=1.692e-8, S_crossover=r(K_GEOM / 1.692e-8),
                 sigma_alpha_at_cap2000_m1=r(K_GEOM / 2000), quadrature_add_at_2000=r(np.sqrt(1 + (K_GEOM / 2000 / 1.692e-8) ** 2) - 1))
    P("\n== crossover ==", json.dumps(cross))

    sat = saturation(g, instr, E_scale, dets["vox_microbolometer"]["adopted"])
    P("\n== saturation ==", json.dumps(sat, indent=1))
    canc = ratio_cancellation(g, instr, E_scale, dets["vox_microbolometer"]["adopted"])
    P("\n== ratio cancellation / requirements ==", json.dumps(canc, indent=1))
    vres = vertical_resolution(g, instr)
    P("\n== vertical resolution bookkeeping ==", json.dumps(vres, indent=1))
    P("\n== mass/power analogues (no budget exists for this design) ==")
    for m in MASS_POWER:
        P("  ", json.dumps(m))

    out = dict(
        script="reproduce/radiometric_budget.py", date="2026-10-25",
        definitions=dict(SNR_T="per-sample transmission SNR of the ratio T/sigma_T, sigma_T/T = sqrt(1/SNR_atm^2 + 1/SNR_ref^2)",
                         SNR_single_TOA="one unattenuated sample, random terms (shot + background shot + NEP)",
                         SNR_single_atm="one attenuated sample, random terms",
                         SNR_reference="SNR_single_TOA * sqrt(n_ref), n_ref = 100 samples above 80 km",
                         cap="systematics cap 2000 applied in quadrature after the random chain (paper convention)",
                         K_GEOM=f"{K_GEOM:.4e} m^-1 from reproduce/mir_detector_etc.py (sigma_alpha = K_GEOM/SNR)"),
        instrument=dict(D_aperture_m=instr.D_aperture_m, tau_optics=instr.tau_optics, dlam_um=instr.dlam_um,
                        T_optics_K=instr.T_optics_K, eps_optics=instr.eps_optics, qe_absorptance=0.8,
                        snr_sys_cap=instr.snr_sys_cap),
        solar=solar, geometry=geo, optical_depths=od, floors_m1=floors,
        catalogue_reconciliation=dict(rows=cat_rows, flags=cat_flags),
        sourced_detectors=dets, detectors_in_chain=det_list, chain=chain_out, crossover=cross,
        saturation=sat, ratio_cancellation=canc, vertical_resolution=vres,
        mass_power=dict(statement="No mass, power, thermal or duty-cycle budget exists in this analysis; the "
                                  "'small-satellite class' description is an unverified design objective supported "
                                  "only by the flown analogues below.", analogues=MASS_POWER),
        table_d2_corrections=TABLE_D2_CORRECTIONS, sources=SOURCES, bibtex=BIBTEX,
        paper_statements_reconciled=paper_statements(chain_out, solar, geo, dets),
    )
    OUT_JSON.write_text(json.dumps(out, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    OUT_LOG.write_text("\n".join(log) + "\n")
    write_table(out, [8.80, 11.40, 20.40],
                [("vox_catalogue_1e-11", "VO$_x$ with the former catalogue NEP $10^{-11}$~W\\,Hz$^{-1/2}$ ($D^{*}=1.7\\times10^{8}$)"),
                 ("thermopile", "Thermopile, MCS class ($D^{*}=8\\times10^{8}$)"),
                 ("dlatgs", "DLATGS pyroelectric ($D^{*}=1.2\\times10^{9}$; chopper needed)"),
                 ("hgcdte", "HgCdTe LWIR (representative $D^{*}=5\\times10^{10}$)"),
                 ("si_as", "Si:As BIB (representative $D^{*}=10^{11}$)")])
    P(f"\nwrote {OUT_JSON}\n      {OUT_LOG}\n      {OUT_TEX}")


if __name__ == "__main__":
    main()
