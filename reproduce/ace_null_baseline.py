"""ace_null_baseline.py -- referee round 6 (paper 1; public port), item 8: the operational null
correction of Sects. 4.3 and 5.3 tested as a procedure, on the shipped per-occultation masses of
the full-band ACE-FTS null test (outputs/ace_v52/nulltest_diagnostics_perocc.csv,
101 occultations near 20.5 km, 2004-2024; masses in the slant convention of the fit).

 1. Coverage of the formal errors about zero and about the fitted centre (median), with
    Clopper-Pearson 95 % intervals.
 2. The null drift: ordinary least squares with its nominal error, and with year-level
    dependence (block bootstrap over seasons, and a fit to the seasonal medians).
 3. A prospective split: the centre and scatter estimated on 2004-2015 and applied to
    2016-2024 (and the reverse), with the exceedance fractions at 3 sigma and 4.54 sigma.
 4. A slow secular injection passed through the whole baseline-estimation and detection
    procedure: a linear ramp of r Tg per decade (and a step) is added to the measured masses
    from a start season on, the baseline is the median of the preceding N seasons (rolling; a
    frozen variant uses the first N seasons only), the decision statistic is
    (M - baseline)/sigma with sigma the robust scatter, and we record the fraction of the
    injected mass absorbed by the rolling baseline, the first season in which the seasonal
    mean exceeds the decision threshold, and the single-sounding exceedance fraction.

Run from the repo root:  python reproduce/ace_null_baseline.py
Writes outputs/ace_null_baseline.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, norm

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs/ace_v52/nulltest_diagnostics_perocc.csv"
OUT = ROOT / "outputs/ace_null_baseline.json"
RNG = np.random.default_rng(74)


def cp_interval(k, n, alpha=0.05):
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def robust_sd(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


df = pd.read_csv(CSV)
df["t"] = pd.to_datetime(df["date"]).map(lambda d: d.year + (d.dayofyear - 0.5) / 365.25)
df = df.sort_values("t").reset_index(drop=True)
M, sM, t, yr = df["M_full"].to_numpy(), df["sM_full"].to_numpy(), df["t"].to_numpy(), df["year"].to_numpy()
n = len(M)
years = np.unique(yr)
out = dict(script="reproduce/ace_null_baseline.py", round=74, n=int(n), years=years.tolist(),
           median=float(np.median(M)), mean=float(np.mean(M)), robust_sd=robust_sd(M), sd=float(np.std(M, ddof=1)),
           formal_sigma_median=float(np.median(sM)))
SIG = out["robust_sd"]

# ---------------------------------------------------------------- 1. coverage about zero and about the centre
cov = {}
for centre_name, c in (("zero", 0.0), ("median", float(np.median(M))), ("mean", float(np.mean(M)))):
    z = (M - c) / sM
    d = {}
    for k in (1, 2, 3):
        m = int(np.sum(np.abs(z) < k))
        d[f"within_{k}sigma"] = dict(fraction=m / n, count=m, ci95=cp_interval(m, n))
    d["max_abs_z"] = float(np.abs(z).max())
    cov[centre_name] = d
out["coverage_formal_errors"] = cov
print("coverage 1/2/3 sigma about zero:", [round(cov["zero"][f"within_{k}sigma"]["fraction"], 3) for k in (1, 2, 3)],
      "about the median:", [round(cov["median"][f"within_{k}sigma"]["fraction"], 3) for k in (1, 2, 3)])

# ---------------------------------------------------------------- 2. drift with year-level dependence
X = np.c_[np.ones(n), (t - 2014.0) / 10.0]                      # slope in Tg per decade
beta_hat, res, *_ = np.linalg.lstsq(X, M, rcond=None)
resid = M - X @ beta_hat
s2 = resid @ resid / (n - 2)
cov_ols = s2 * np.linalg.inv(X.T @ X)
drift = dict(slope_tg_per_decade=float(beta_hat[1]), se_ols=float(np.sqrt(cov_ols[1, 1])))
# block bootstrap over seasons (years)
B = 4000
slopes = np.empty(B)
for b in range(B):
    ys = RNG.choice(years, size=len(years), replace=True)
    idx = np.concatenate([np.flatnonzero(yr == y) for y in ys])
    Xb, Mb = X[idx], M[idx]
    slopes[b] = np.linalg.lstsq(Xb, Mb, rcond=None)[0][1]
drift["se_year_block_bootstrap"] = float(np.std(slopes, ddof=1))
drift["ci95_year_block_bootstrap"] = [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]
# fit to the seasonal medians (one point per season, equal weight)
med_y = np.array([np.median(M[yr == y]) for y in years]); ty = np.array([np.mean(t[yr == y]) for y in years])
Xy = np.c_[np.ones(len(years)), (ty - 2014.0) / 10.0]
by, *_ = np.linalg.lstsq(Xy, med_y, rcond=None)
ry = med_y - Xy @ by; s2y = ry @ ry / (len(years) - 2)
drift["seasonal_medians_slope_tg_per_decade"] = float(by[1])
drift["seasonal_medians_se"] = float(np.sqrt(s2y * np.linalg.inv(Xy.T @ Xy)[1, 1]))
drift["n_seasons"] = int(len(years))
# lag-one correlation (for the record)
drift["lag1"] = float(np.corrcoef(M[:-1], M[1:])[0, 1])
out["drift"] = drift
print(f"drift {drift['slope_tg_per_decade']:+.4f} Tg/decade; se OLS {drift['se_ols']:.4f}, year-block bootstrap {drift['se_year_block_bootstrap']:.4f}, "
      f"seasonal medians {drift['seasonal_medians_slope_tg_per_decade']:+.4f} +- {drift['seasonal_medians_se']:.4f}")

# ---------------------------------------------------------------- 3. prospective split
split = {}
for name, train, test in (("2004-2015 -> 2016-2024", yr <= 2015, yr >= 2016), ("2016-2024 -> 2004-2015", yr >= 2016, yr <= 2015)):
    c = float(np.median(M[train])); s = robust_sd(M[train])
    z = (M[test] - c) / s
    d = dict(n_train=int(train.sum()), n_test=int(test.sum()), centre_train=c, robust_sd_train=s,
             centre_test=float(np.median(M[test])), robust_sd_test=robust_sd(M[test]),
             shift_of_centre_in_train_sigma=float((np.median(M[test]) - c) / s))
    for thr_ in (3.0, 4.54):
        k = int(np.sum(z > thr_)); d[f"exceed_{thr_}sigma"] = dict(count=k, fraction=k / int(test.sum()), ci95_upper=cp_interval(k, int(test.sum()))[1])
    d["gaussian_expected_3sigma_fraction_with_shift"] = float(norm.sf(3.0 - d["shift_of_centre_in_train_sigma"]))
    split[name] = d
out["prospective_split"] = split
for k, v in split.items():
    print(f"split {k}: centre {v['centre_train']:+.4f} -> test median {v['centre_test']:+.4f} ({v['shift_of_centre_in_train_sigma']:+.2f} sigma); "
          f"exceed 3 sigma {v['exceed_3.0sigma']['count']}/{v['n_test']}, 4.54 sigma {v['exceed_4.54sigma']['count']}/{v['n_test']}")

# ---------------------------------------------------------------- 4. a secular injection through the procedure
def run_procedure(M_obs, n_base=3, frozen=False, thr=4.54):
    """baseline = median of the preceding n_base seasons (rolling) or of the first n_base seasons (frozen);
    statistic z = (M - baseline)/SIG; per season: the seasonal mean of z, and the single-sounding exceedances."""
    base, zs, seas = {}, np.full(len(M_obs), np.nan), {}
    for i, y in enumerate(years):
        prev = years[(years < y)][-n_base:] if not frozen else years[:n_base]
        if len(prev) < n_base or (frozen and y in years[:n_base]):
            continue
        b = float(np.median(np.concatenate([M_obs[yr == p] for p in prev])))
        base[int(y)] = b
        m = yr == y
        zs[m] = (M_obs[m] - b) / SIG
        seas[int(y)] = dict(baseline=b, mean_z=float(np.mean(zs[m])), n=int(m.sum()),
                            seasonal_mean_z=float(np.mean(zs[m]) * np.sqrt(m.sum())),   # z of the seasonal mean (independent errors)
                            exceed=int(np.sum(zs[m] > thr)))
    return base, zs, seas


inj = dict(sigma_used=SIG, threshold_sigma=4.54, baseline_seasons=3, start_year=2010, cases={})
t0 = 2010.0
for case, rate, step in (("ramp_0.02_per_decade", 0.02, 0.0), ("ramp_0.05_per_decade", 0.05, 0.0), ("ramp_0.10_per_decade", 0.10, 0.0),
                         ("ramp_0.20_per_decade", 0.20, 0.0), ("step_0.10", 0.0, 0.10), ("step_0.15", 0.0, 0.15), ("null", 0.0, 0.0)):
    signal = np.where(t >= t0, rate * (t - t0) / 10.0 + step, 0.0)
    M_obs = M + signal
    for mode in ("rolling", "frozen"):
        base, zs, seas = run_procedure(M_obs, frozen=(mode == "frozen"))
        # fraction of the injected signal absorbed by the baseline, season by season (where the signal is non-zero)
        absorbed = {}
        base_null, _, _ = run_procedure(M, frozen=(mode == "frozen"))        # the same procedure on the measured (null) record
        for y, d in seas.items():
            if y >= t0 and signal[yr == y].mean() > 0:
                true_mean = float(signal[yr == y].mean())
                absorbed[y] = float((d["baseline"] - base_null[y]) / true_mean)   # baseline rise caused by the signal, per unit signal
        # first season whose seasonal-mean statistic exceeds the threshold, and the single-sounding count
        first = next((y for y, d in seas.items() if d["seasonal_mean_z"] > 4.54 and y >= t0), None)
        first3 = next((y for y, d in seas.items() if d["seasonal_mean_z"] > 3.0 and y >= t0), None)
        n_exc = int(sum(d["exceed"] for y, d in seas.items() if y >= t0)); n_tot = int(sum(d["n"] for y, d in seas.items() if y >= t0))
        # the same statistic with the true pre-injection null as the fixed centre (what a frozen pre-SAI baseline would give)
        z_fixed = (M_obs - float(np.median(M[yr <= 2009]))) / SIG
        seas_fixed = {int(y): float(np.mean(z_fixed[yr == y]) * np.sqrt(np.sum(yr == y))) for y in years if y >= t0}
        first_fixed = next((y for y, v in seas_fixed.items() if v > 4.54), None)
        inj["cases"][f"{case}_{mode}"] = dict(
            rate_tg_per_decade=rate, step_tg=step, mode=mode,
            seasons={int(y): dict(baseline=round(d["baseline"], 4), seasonal_mean_z=round(d["seasonal_mean_z"], 2), single_exceed=d["exceed"], n=d["n"]) for y, d in seas.items()},
            fraction_of_signal_absorbed_by_baseline={int(y): round(v, 3) for y, v in absorbed.items()},
            first_season_seasonal_mean_above_4p54=first, first_season_seasonal_mean_above_3=first3,
            single_sounding_exceedances_after_start=[n_exc, n_tot],
            first_season_above_4p54_with_fixed_pre2010_centre=first_fixed,
            signal_in_last_season_tg=float(signal[yr == years[-1]].mean()))
out["secular_injection"] = inj
for k, v in inj["cases"].items():
    ab = v["fraction_of_signal_absorbed_by_baseline"]
    print(f"{k:28s}: first season > 4.54 (seasonal mean) {v['first_season_seasonal_mean_above_4p54']} (fixed centre {v['first_season_above_4p54_with_fixed_pre2010_centre']}), "
          f"> 3: {v['first_season_seasonal_mean_above_3']}; single exceed {v['single_sounding_exceedances_after_start']}; absorbed (last) {list(ab.values())[-1] if ab else None}")

with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print("archived ->", OUT)
