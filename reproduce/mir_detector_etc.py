"""
mir_detector_etc.py -- radiometric exposure-time / noise calculator (ETC) for
the MIR silica-detection channels of a SAGE-III-class SOLAR OCCULTATION
instrument, and a comparison of candidate MIR detectors.

Purpose
-------
The note (Secs. 4-5) adopts MIR extinction noise floors sigma_alpha ~ 1e-8 m^-1
(8.80, 20.4 um) but treats the *detector* side only qualitatively ("a sub-10 K
detector is itself noisy").  This script makes it quantitative: for a given
detector (NEP / D* or NETD), aperture, spectral element and integration time,
it computes the achievable per-sample transmission SNR and converts it to
sigma_alpha, then compares against each band's systematic (gas-removal) floor.

Key mapping (grounded in the note, sec:master / Wrana2021 rescaling)
--------------------------------------------------------------------
The VIS/NIR channels use sigma_alpha = 8.19e-9 m^-1 at SNR = 2000, dz = 0.5 km,
with sigma_alpha ∝ SNR^-1 (dz)^-1/2.  Because the limb geometry and onion
retrieval are wavelength-independent, the product

    K_GEOM = sigma_alpha * SNR = 8.19e-9 * 2000 = 1.638e-5  [m^-1]   (dz=0.5 km)

is a fixed geometric/retrieval constant.  So a channel whose per-sample
transmission SNR is S has an extinction noise
    sigma_alpha(S) = K_GEOM / S
and the channel's realised floor is max(systematic_floor, K_GEOM/S).

Self-calibration / bright-source point
--------------------------------------
This is a *transmission* measurement against the Sun: the signal is the bright
solar disk and the reference is the exo-atmospheric solar spectrum, so gain and
DC offsets cancel.  The question is whether a detector's *temporal* noise (NEP)
is small vs the collected solar power.  We therefore compute the solar photon
signal per resolution element and the SNR each detector delivers.

All instrument parameters are explicit and adjustable; every number the script
prints is traceable to them.  Detector specs are first-order representative
values (refined against the literature survey) -- see DETECTORS below.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------------
H = 6.62607015e-34     # J s
C = 2.99792458e8       # m/s
KB = 1.380649e-23      # J/K
T_SUN = 5772.0         # K  (solar effective temperature)
R_SUN = 6.957e8        # m
AU = 1.495978707e11    # m

# ----------------------------------------------------------------------------
# Geometry / retrieval constant (from the note)
# ----------------------------------------------------------------------------
# Round 43 (paper1): 8.19e-9 was the MEASURED 1544-nm channel error (0.09 x 9.1e-8),
# not the SNR-2000 transmission floor.  The floor is the onion-peel conversion
# 1.24/(SNR * P_kk), P_kk = 2 sqrt(2 R_earth dz) ~ 160 km (paper1 Eq. alpha_floor),
# i.e. 3.9e-9 m^-1 at SNR = 2000, dz = 0.5 km -> K_GEOM = 7.75e-6 m^-1.
SNR_VIS = 2000.0
_R_EARTH_M = 6.371e6
_P_KK_M = 2.0 * np.sqrt(2.0 * _R_EARTH_M * 500.0)
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from saimon.onion_peel import G_ONION  # round 53 (RC1 M1): exact edge-grid gain from saimon.onion_peel
SIGMA_VIS = G_ONION / (SNR_VIS * _P_KK_M)   # 3.44e-9 m^-1 (was 3.875e-9 with g = 1.24)
K_GEOM = SIGMA_VIS * SNR_VIS      # = 7.75e-6 m^-1 ; sigma_alpha = K_GEOM / SNR
DZ_KM = 0.5

# Band systematic (gas-removal) floors from Sec. 5, at dz = 0.5 km [m^-1].
BAND_FLOOR = {
    4.00:  1.0e-8,    # smooth stretch, assigned
    8.80:  1.5e-8,    # O3 9.6 um wing line-intensity scatter (round 43: 8.74 -> 8.80 um)
    11.40: 1.2e-8,    # calcite nu2 (appendix C)
    20.40: 2.8e-9,    # H2O/HNO3 co-retrieval floor (nominal adopted 1.0e-8)
    28.50: 3.4e-9,    # calcite lattice far-IR (appendix C)
}


# ----------------------------------------------------------------------------
# Instrument parameters (nominal; adjust here)
# ----------------------------------------------------------------------------
class Instrument:
    D_aperture_m = 0.10        # entrance aperture diameter [m]
    tau_optics = 0.30          # end-to-end MIR optical throughput (grating + surfaces)
    dlam_um = 0.10             # spectral resolution element [um]
    # Vertical sampling: dz=0.5 km at the tangent, tangent range ~ 3000 km from
    # a LEO platform -> vertical IFOV that resolves the layer.
    tangent_range_m = 3.0e6
    dz_m = DZ_KM * 1e3
    # Integration time per 0.5 km sample: set by the limb-scan rate.  The tangent
    # point descends at ~2 km/s for an ISS-like orbit; per 0.5 km that is ~0.25 s,
    # but the Sun is re-scanned and co-added, so 0.5 s is a fair per-sample value.
    t_int_s = 0.5
    T_optics_K = 300.0         # temperature of the (warm or cooled) fore-optics
    f_number = 2.0             # detector-side focal ratio (sets pixel etendue)
    eps_optics = 1.0           # emissivity of the warm optics seen by a pixel
    qe = 0.7                   # photon-detector quantum efficiency
    # Systematic transmission-precision floor (pointing, detector linearity,
    # solar-disk structure).  SAGE III's VIS SNR=2000 is exactly this design
    # floor, not a photon limit -- so no channel does better than this however
    # bright the source.  We cap the net per-sample transmission SNR here.
    snr_sys_cap = 2000.0

    @property
    def A_aperture_m2(self):
        return np.pi * (self.D_aperture_m / 2.0) ** 2

    @property
    def solar_disk_omega_sr(self):
        return np.pi * (R_SUN / AU) ** 2      # 6.79e-5 sr

    def disk_fraction(self):
        """Fraction of the solar-disk irradiance falling in one vertical
        resolution element (a horizontal strip of angular height theta_v across
        the disk).  theta_v = dz / tangent_range."""
        theta_v = self.dz_m / self.tangent_range_m          # rad
        d_sun = 2.0 * (R_SUN / AU)                           # angular diameter [rad]
        if theta_v >= d_sun:
            return 1.0
        # area of a central strip of half-height h in a disk of radius a:
        a = d_sun / 2.0
        h = min(theta_v / 2.0, a)
        strip = 2.0 * (h * np.sqrt(a * a - h * h) + a * a * np.arcsin(h / a))
        return strip / (np.pi * a * a)


def solar_spectral_irradiance_W_m2_um(lam_um):
    """Top-of-atmosphere solar spectral irradiance [W/m^2/um] from a 5772 K
    blackbody solar disk.  Reproduces TSI=1361 W/m^2 on integration; good to
    ~10-30% in the MIR where the solar brightness temperature departs from the
    effective value."""
    lam = np.asarray(lam_um, float) * 1e-6
    B = (2.0 * H * C ** 2 / lam ** 5) / (np.exp(H * C / (lam * KB * T_SUN)) - 1.0)  # W/sr/m^3
    omega = np.pi * (R_SUN / AU) ** 2
    return B * omega * 1e-6      # W/m^2/um


# ----------------------------------------------------------------------------
# Candidate detectors.  NEP is referred to the detector; where a D* and pixel
# pitch are given, NEP = sqrt(A_det)/D*.  'tmax_um' is the useful long-wave
# cutoff; 'top_K' the required operating temperature.  These are representative
# first-order numbers to be reconciled with the literature survey.
# ----------------------------------------------------------------------------
class Detector:
    def __init__(self, name, kind, lam_min, lam_max, top_K, cooler,
                 nep_W_rtHz=None, dstar=None, pitch_um=None, tau_resp_s=None,
                 R_typ="", heritage="", note=""):
        self.name, self.kind = name, kind
        self.lam_min, self.lam_max = lam_min, lam_max
        self.top_K, self.cooler = top_K, cooler
        self.pitch_um, self.tau_resp_s = pitch_um, tau_resp_s
        self.R_typ, self.heritage, self.note = R_typ, heritage, note
        if nep_W_rtHz is not None:
            self.nep = nep_W_rtHz
        else:
            a_cm2 = (pitch_um * 1e-4) ** 2      # detector area [cm^2]
            self.nep = np.sqrt(a_cm2) / dstar   # W/Hz^1/2

    def covers(self, lam_um):
        return self.lam_min <= lam_um <= self.lam_max


# Representative specs reconciled with the 2025-2026 literature survey.  NEP for
# the thermal detectors is order-of-magnitude (flagged); the SNR conclusion is
# robust across the plausible range because the solar signal is 10^4-10^6x NEP.
# R_typ = spectral resolving power the technology DELIVERS in real instruments
# (set by the spectrometer, but bounded by detector speed/format).
DETECTORS = [
    Detector("Uncooled microbolometer (VOx)", "thermal", 7.5, 40.0, 300, "none / TEC-trim",
             nep_W_rtHz=1e-11, pitch_um=17.0, tau_resp_s=1e-2,
             R_typ="R~10-300 (disperser); slow (~10ms) -> poor for high-OPD FTS",
             heritage="THEMIS/Odyssey 6.8-14.9um; E-THEMIS/Europa Clipper 14-28,28-80um",
             note="broadband-absorber variant reaches 20-29um (E-THEMIS); NEP degrades "
                  ">28um; saturates on Sun -> attenuate"),
    Detector("Uncooled thermopile array", "thermal", 2.0, 45.0, 300, "none",
             nep_W_rtHz=1e-9, pitch_um=100.0, tau_resp_s=1e-2,
             R_typ="R~10-100 (radiometer/disperser); DC-coupled, no chopper",
             heritage="LRO Diviner 0.3-400um; MCS/MRO; OSIRIS-REx OTES; Lucy L'TES",
             note="flat broadband to ~45um, highly linear -> ideal absolute radiometry at 20-29um"),
    Detector("Pyroelectric (DLaTGS/LiTaO3)", "thermal", 2.0, 40.0, 300, "none / TE-stab",
             nep_W_rtHz=3e-10, pitch_um=100.0, tau_resp_s=1e-2,
             R_typ="high R in FTS (mirror supplies modulation)",
             heritage="Mini-TES(MER); OSIRIS-REx OTES; Lucy L'TES; ESA FORUM",
             note="AC-ONLY: needs chopping/modulation (an FTS provides it; a staring imager cannot)"),
    Detector("InSb", "photon", 1.0, 5.5, 30, "passive/Stirling",
             dstar=1.0e11, pitch_um=30.0,
             R_typ="R up to ~10^5 (FTS) / ~few 1000 (grating)",
             heritage="Spitzer IRAC 3.6/4.5um @15K; ACE-FTS (with MCT)",
             note="MWIR only; covers 4 um"),
    Detector("HgCdTe LWIR", "photon", 3.0, 12.0, 77, "passive/Stirling",
             dstar=5.0e10, pitch_um=30.0,
             R_typ="R~10^4-10^5 (FTS: MIPAS,ACE); ~2700 (JWST NIRSpec-class)",
             heritage="ACE-FTS; MIPAS; IASI; CrIS; JWST NIRCam/NIRSpec",
             note="covers 8.74, 11.4 um; ~40K for low-background astronomy"),
    Detector("HgCdTe VLWIR", "photon", 3.0, 16.0, 50, "Stirling/pulse-tube",
             dstar=2.0e10, pitch_um=30.0,
             R_typ="R~10^3-10^4",
             heritage="WISE ~12um @7.8K; sounder 15um bands ~50K",
             note="cutoff ~16 um -> does NOT reach 20.4 um"),
    Detector("Si:As IBC/BIB", "photon", 5.0, 28.0, 7, "pulse-tube+JT / stored",
             dstar=1.0e11, pitch_um=25.0,
             R_typ="R~100 (MIRI LRS) to ~3500 (MIRI MRS); ~60-600 (Spitzer IRS)",
             heritage="JWST MIRI (6.4K); Spitzer IRS+MIPS-24um; WISE; AKARI",
             note="THE photon option at 20.4 & 28.5 um; needs ~6-7 K"),
    Detector("Si:Sb BIB", "photon", 14.0, 40.0, 8, "pulse-tube+JT / stored",
             dstar=8.0e10, pitch_um=25.0,
             R_typ="R~60-600 (Spitzer IRS Long-High/Long-Low)",
             heritage="Spitzer IRS LL/LH only",
             note="extends photon route to 40 um; needs ~4-10 K"),
    Detector("Type-II superlattice (T2SL)", "photon", 3.0, 15.0, 80, "Stirling/pulse-tube",
             dstar=1.0e11, pitch_um=30.0,
             R_typ="R~10^3 (grating/imaging)",
             heritage="MWIR maturing; NO operational VLWIR space heritage (2025)",
             note="LWIR ~12um production; VLWIR to ~18um research; still needs <=90K"),
]


def planck_W_sr_m2_um(lam_um, T):
    lam = np.asarray(lam_um, float) * 1e-6
    B = (2.0 * H * C ** 2 / lam ** 5) / (np.exp(H * C / (lam * KB * T)) - 1.0)
    return B * 1e-6


def snr_for(det: Detector, lam_um, instr: Instrument):
    """Per-sample transmission SNR for detector `det` at `lam_um`, decomposed
    into the limiting mechanisms.  Returns a dict of component SNRs and the net.

    - shot : signal photon shot noise (+ warm-optics background photons)
    - nep  : detector NEP (Johnson/phonon/read), from D* or NEP
    - sys  : systematic transmission floor (design limit; caps everything)
    net = 1/sqrt(sum 1/SNR_i^2)."""
    lam_m = lam_um * 1e-6
    e_ph = H * C / lam_m                                     # photon energy [J]
    E = solar_spectral_irradiance_W_m2_um(lam_um)           # W/m^2/um
    P_sig = (E * instr.dlam_um * instr.A_aperture_m2 * instr.tau_optics
             * instr.disk_fraction())                        # W on the element

    # --- warm-optics thermal background on one pixel, over one spectral element
    A_det = (det.pitch_um * 1e-6) ** 2 if det.pitch_um else (25e-6) ** 2
    omega_pix = np.pi / (4.0 * instr.f_number ** 2)          # sr (cold/warm stop)
    P_bkg = (instr.eps_optics * planck_W_sr_m2_um(lam_um, instr.T_optics_K)
             * instr.dlam_um * A_det * omega_pix)            # W per pixel

    # --- signal + background photon shot noise (ideal co-adding over t_int)
    n_sig = instr.qe * P_sig * instr.t_int_s / e_ph
    n_bkg = instr.qe * P_bkg * instr.t_int_s / e_ph
    snr_shot = n_sig / np.sqrt(n_sig + n_bkg) if n_sig > 0 else 0.0

    # --- detector NEP-limited SNR
    df = 1.0 / (2.0 * instr.t_int_s)
    snr_nep = P_sig / (det.nep * np.sqrt(df))

    # --- combine (systematic floor caps the result)
    inv2 = 1.0 / snr_shot ** 2 + 1.0 / snr_nep ** 2 + 1.0 / instr.snr_sys_cap ** 2
    snr_net = 1.0 / np.sqrt(inv2)
    return {"P_sig": P_sig, "P_bkg": P_bkg, "shot": snr_shot,
            "nep": snr_nep, "net": snr_net}


def main():
    instr = Instrument()
    print("MIR solar-occultation ETC")
    print(f"  aperture D={instr.D_aperture_m*100:.0f} cm (A={instr.A_aperture_m2:.2e} m^2), "
          f"tau_opt={instr.tau_optics}, dlam={instr.dlam_um} um, t_int={instr.t_int_s}s")
    print(f"  disk fraction per {DZ_KM} km element = {instr.disk_fraction():.4f}")
    print(f"  K_GEOM = sigma_alpha*SNR = {K_GEOM:.3e} m^-1  (dz={DZ_KM} km)")
    print("=" * 92)

    bands = sorted(BAND_FLOOR)
    print("\nSolar spectral irradiance (5772 K disk model) and required resolving power:")
    print(f"  (spectral element dlam={instr.dlam_um} um -> R = lam/dlam; a LOW-resolution requirement)")
    for lam in bands:
        R_need = lam / instr.dlam_um
        print(f"  {lam:6.2f} um : E = {solar_spectral_irradiance_W_m2_um(lam):.3e} W/m^2/um"
              f"   R_needed = {R_need:5.0f}")

    print("\nDetector catalogue (spectral coverage, operating T, cooler, delivered R):")
    for det in DETECTORS:
        print(f"  {det.name}")
        print(f"      {det.lam_min:.1f}-{det.lam_max:.1f} um | T_op~{det.top_K:.0f} K | "
              f"cooler: {det.cooler} | NEP~{det.nep:.0e} W/rtHz")
        print(f"      resolution: {det.R_typ}")
        print(f"      heritage:   {det.heritage}")

    print(f"\nOptics temperature {instr.T_optics_K:.0f} K, f/{instr.f_number:.0f}, "
          f"QE={instr.qe}, systematic SNR cap={instr.snr_sys_cap:.0f}")
    for lam in bands:
        floor = BAND_FLOOR[lam]
        snr_need = K_GEOM / floor
        print(f"\n--- {lam:.2f} um  (systematic floor {floor:.2e} m^-1 -> "
              f"needs SNR >= {snr_need:.0f}) ---")
        print(f"  {'detector':34s} {'Top[K]':>6s} {'SNR_shot':>9s} {'SNR_nep':>9s} "
              f"{'SNR_net':>8s} {'sig_a[m^-1]':>12s}  limited-by")
        for det in DETECTORS:
            if not det.covers(lam):
                continue
            r = snr_for(det, lam, instr)
            sig_a = K_GEOM / r["net"]
            if r["net"] >= 0.999 * instr.snr_sys_cap:
                limited = "systematic(design)"
            elif r["nep"] < r["shot"]:
                limited = "DETECTOR-NEP"
            else:
                limited = "shot/background"
            print(f"  {det.name:34s} {det.top_K:6.0f} {r['shot']:9.1e} "
                  f"{r['nep']:9.1e} {r['net']:8.0f} {sig_a:12.2e}  {limited}")


if __name__ == "__main__":
    main()
