"""Referee RC1 M7: spike (injection-recovery) test on the real ACE null-test sample.
Same 101 occultations, same bases, same fits as reproduce/ace_fullspectrum_retrieval.py;
a synthetic silica layer of M_inj Tg (the null test's own slant/layer convention,
per-Tg layer 16 km-30 hPa seen at the occultation's tangent height, self-calibration
operator applied) is multiplied into the residual transmittance before the fit."""
import sys, json, csv
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import ace_fullspectrum_retrieval as F

nu = F.reference_grid()
bases = F.build_bases(nu)
thr = json.load(open(ROOT / "outputs/calibrated_background_thresholds.json"))
win_sig_od = np.array(thr["window_floors_m1"]) * F.OD_PER_M1
keep = set(str(o) for o in np.load(F.SIDE / "matrix_19_22.npz", allow_pickle=True)["occs"])
occs = []
for d in sorted((F.DATA / "residual").iterdir()):
    if d.name in keep:
        got = F.occultation_data(d.name, nu)
        if got is not None:
            occs.append((d.name,) + got)
print("occultations:", len(occs))
M_INJ = [0.0, 0.05, 0.1, 0.2, 0.3]
res = {}
for Minj in M_INJ:
    rows = {t: [] for t in F.ALL_TAGS}
    for occ, h, tau, h_ref in occs:
        sil = np.interp(h, F.H_GRID_KM, bases["sil"].T.T[:, 0]) if False else None
        # slant per-Tg silica OD at h minus at h_ref (same operator as the basis column)
        b = bases["sil"]
        sig = np.array([np.interp(h, F.H_GRID_KM, b[:, j]) - np.interp(h_ref, F.H_GRID_KM, b[:, j]) for j in range(b.shape[1])])
        tau_i = tau + Minj * sig            # == -ln(T * exp(-tau_sil)) + ln(T_ref)
        r = F.fit_occ(h, tau_i, h_ref, bases, nu, win_sig_od)
        if r is None: continue
        for t in F.ALL_TAGS:
            if t in r: rows[t].append((r[t]["M"], r[t]["sM"]))
    res[Minj] = {}
    for t in F.ALL_TAGS:
        M = np.array([m for m, s in rows[t]]); sM = np.array([s for m, s in rows[t]])
        med = float(np.median(M)); sig_emp = float(1.4826 * np.median(np.abs(M - med)))
        res[Minj][t] = dict(n=len(M), mean=float(M.mean()), median=med, sigma_emp=sig_emp,
                            sd=float(M.std(ddof=1)), sigma_formal=float(np.median(sM)))
    print(f"\nM_inj = {Minj:.2f} Tg")
    for t in F.ALL_TAGS:
        s = res[Minj][t]
        print(f"  [{t:5s}] n={s['n']:3d}  median M = {s['median']:+.4f}  mean = {s['mean']:+.4f}  "
              f"robust sigma = {s['sigma_emp']:.4f}  sd = {s['sd']:.4f}  formal = {s['sigma_formal']:.4f}")
# recovery table relative to the control
print("\nRecovery relative to the M_inj=0 control (median M - control median - M_inj):")
for t in F.ALL_TAGS:
    c = res[0.0][t]["median"]
    line = "  " + f"{t:5s} " + "  ".join(f"M={m:.2f}: bias {res[m][t]['median']-c-m:+.4f}, sig {res[m][t]['sigma_emp']:.4f}" for m in M_INJ[1:])
    print(line)
json.dump({str(k): v for k, v in res.items()}, open(ROOT / "outputs/ace_v52/spike_test.json", "w"), indent=1)
