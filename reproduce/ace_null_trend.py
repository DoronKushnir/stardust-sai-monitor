"""ace_null_trend.py -- referee round 7 (paper 1; public port), item 7: the trend correction of the operational null
level defined as a rule and tested.

Rule (frozen pre-deployment trend).  The null level is a linear trend M_0(t) = a + b (t - t_ref) fitted by
ordinary least squares to the retrieved masses of the pre-deployment training record, frozen at the
deployment date T_0 and not updated with monitored data (an update with data that may contain the signal
would absorb a secular injection as the rolling baseline did).  Its forecast at a later time t carries the
covariance of (a, b) estimated by a block bootstrap over seasons (year-level dependence), so that the
variance of the predicted null at t is var(a) + 2 (t - t_ref) cov(a, b) + (t - t_ref)^2 var(b).  The
decision statistic of a sounding is z = (M - M_0(t)) / sqrt(sigma^2 + var(M_0(t))), with sigma the robust
per-sounding scatter of the training record; the seasonal statistic is the seasonal mean of z times
sqrt(n).  The centre uncertainty of a median is taken from the same bootstrap, not from sigma/sqrt(N).

Mass convention: the ACE masses are slant-fit masses at the 20.5-km tangent (Sect. 4.3); the design
sensitivities are per 0.5-km shell; the conversion is the factor 2.33 of Sect. 4.3 and is stated with
every number that mixes the two.

Tests: the ramps (0.02, 0.05, 0.10, 0.20 Tg per decade) and steps (0.10, 0.15 Tg) added from T_0 = 2016.0
to the measured record (training 2004-2015, 12 seasons) and passed through the rule; the null record
itself (false alarms); the same with the training record 2004-2009 (6 seasons, T_0 = 2010) to show the
dependence on the training length; the forecast uncertainty at 1, 5 and 10 years after T_0.

Run from the repo root (seconds):  python reproduce/ace_null_trend.py
Writes outputs/ace_null_trend.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, beta

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs/ace_v52/nulltest_diagnostics_perocc.csv"
OUT = ROOT / "outputs/ace_null_trend.json"
RNG = np.random.default_rng(76)
SLANT_TO_SHELL = 2.33          # Sect. 4.3: slant fit at 20.5 km -> per 0.5-km shell (design convention)
B = 4000

df = pd.read_csv(CSV)
df["t"] = pd.to_datetime(df["date"]).map(lambda d: d.year + (d.dayofyear - 0.5) / 365.25)
df = df.sort_values("t").reset_index(drop=True)
M, t, yr = df["M_full"].to_numpy(), df["t"].to_numpy(), df["year"].to_numpy()
years = np.unique(yr)


def robust_sd(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def cp_upper(k, n, alpha=0.05):
    return 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))


def fit_rule(train, t_ref):
    """(a, b) by OLS on the training record, covariance by a year-block bootstrap; sigma = robust scatter of the training residuals."""
    X = np.c_[np.ones(train.sum()), (t[train] - t_ref) / 10.0]
    ab = np.linalg.lstsq(X, M[train], rcond=None)[0]
    tr_years = np.unique(yr[train]); boots = np.empty((B, 2)); meds = np.empty(B)
    for i in range(B):
        ys = RNG.choice(tr_years, size=len(tr_years), replace=True)
        idx = np.concatenate([np.flatnonzero((yr == y) & train) for y in ys])
        boots[i] = np.linalg.lstsq(X[np.searchsorted(np.flatnonzero(train), idx)], M[idx], rcond=None)[0]
        meds[i] = np.median(M[idx])
    cov_ab = np.cov(boots.T)
    sig = robust_sd(M[train] - X @ ab)
    return dict(a=float(ab[0]), b_per_decade=float(ab[1]), cov_ab=cov_ab.tolist(), se_a=float(np.sqrt(cov_ab[0, 0])), se_b_per_decade=float(np.sqrt(cov_ab[1, 1])),
                sigma_train=sig, n_train=int(train.sum()), n_seasons=int(len(tr_years)), median_train=float(np.median(M[train])),
                se_median_bootstrap=float(np.std(meds, ddof=1)), se_median_gaussian_formula=float(np.sqrt(np.pi / 2) * robust_sd(M[train]) / np.sqrt(train.sum())),
                t_ref=t_ref)


def forecast_var(rule, tt):
    c = np.array(rule["cov_ab"]); u = (tt - rule["t_ref"]) / 10.0
    return c[0, 0] + 2 * u * c[0, 1] + u ** 2 * c[1, 1]


def apply_rule(rule, M_obs, test, thr=4.54):
    """per-sounding z with the forecast null and its uncertainty; seasonal statistics after T_0."""
    tt = t[test]; m0 = rule["a"] + rule["b_per_decade"] * (tt - rule["t_ref"]) / 10.0
    z = (M_obs[test] - m0) / np.sqrt(rule["sigma_train"] ** 2 + forecast_var(rule, tt))
    seas = {}
    for y in np.unique(yr[test]):
        m = yr[test] == y
        seas[int(y)] = dict(n=int(m.sum()), seasonal_mean_z=float(np.mean(z[m]) * np.sqrt(m.sum())), single_exceed=int(np.sum(z[m] > thr)),
                            forecast_null=float(np.mean(m0[m])), forecast_sd=float(np.sqrt(np.mean(forecast_var(rule, tt[m])))))
    return z, seas


out = dict(script="reproduce/ace_null_trend.py", round=76, slant_to_shell=SLANT_TO_SHELL, bootstrap_draws=B,
           rule="linear trend fitted by OLS to the pre-deployment training record, frozen at T_0, forecast covariance of (a, b) from a year-block bootstrap; z = (M - M_0(t)) / sqrt(sigma^2 + var M_0(t))",
           cases={})
for name, t0, train in (("train_2004_2015", 2016.0, yr <= 2015), ("train_2004_2009", 2010.0, yr <= 2009)):
    rule = fit_rule(train, t_ref=t0)
    test = yr >= t0
    case = dict(rule=rule, T0=t0, forecast_sd_tg={f"{dt:.0f}_yr": float(np.sqrt(forecast_var(rule, t0 + dt))) for dt in (1.0, 5.0, 10.0)},
                forecast_sd_per_shell_tg={f"{dt:.0f}_yr": float(np.sqrt(forecast_var(rule, t0 + dt)) * SLANT_TO_SHELL) for dt in (1.0, 5.0, 10.0)}, signals={})
    for sname, rate, step in (("null", 0.0, 0.0), ("ramp_0.02_per_decade", 0.02, 0.0), ("ramp_0.05_per_decade", 0.05, 0.0), ("ramp_0.10_per_decade", 0.10, 0.0),
                              ("ramp_0.20_per_decade", 0.20, 0.0), ("step_0.10", 0.0, 0.10), ("step_0.15", 0.0, 0.15)):
        signal = np.where(t >= t0, rate * (t - t0) / 10.0 + step, 0.0)
        z, seas = apply_rule(rule, M + signal, test)
        first = next((y for y, d in seas.items() if d["seasonal_mean_z"] > 4.54), None)
        first3 = next((y for y, d in seas.items() if d["seasonal_mean_z"] > 3.0), None)
        n_exc = int(np.sum(z > 4.54)); n_tot = int(len(z))
        case["signals"][sname] = dict(rate_tg_per_decade=rate, step_tg=step, first_season_seasonal_mean_above_4p54=first, first_season_above_3=first3,
                                      single_sounding_exceedances=[n_exc, n_tot], exceedance_fraction_upper95=cp_upper(n_exc, n_tot),
                                      seasons={y: dict(z_mean=round(d["seasonal_mean_z"], 2), exceed=d["single_exceed"], null=round(d["forecast_null"], 4), null_sd=round(d["forecast_sd"], 4)) for y, d in seas.items()},
                                      signal_last_season_tg=float(signal[yr == years[-1]].mean()))
    out["cases"][name] = case
    r = rule
    print(f"{name}: a {r['a']:+.4f} +- {r['se_a']:.4f} Tg, b {r['b_per_decade']:+.4f} +- {r['se_b_per_decade']:.4f} Tg/decade, sigma {r['sigma_train']:.4f}; "
          f"median SE bootstrap {r['se_median_bootstrap']:.4f} (Gaussian formula {r['se_median_gaussian_formula']:.4f}); forecast sd at +1/+5/+10 yr "
          + "/".join(f"{v:.4f}" for v in case["forecast_sd_tg"].values()) + " Tg slant")
    for sname, d in case["signals"].items():
        print(f"   {sname:22s}: first season > 4.54 {d['first_season_seasonal_mean_above_4p54']}, > 3 {d['first_season_above_3']}; single exceed {d['single_sounding_exceedances']}")
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print("archived ->", OUT)
