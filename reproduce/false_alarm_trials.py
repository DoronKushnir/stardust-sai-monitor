"""Referee RC1 M8: what the 3-sigma single-test threshold means for a monitoring
system that performs many tests.  Gaussian, independent tests; the per-test
z-value for a given global false-alarm rate, the mass threshold relative to
M_3sigma = 3 sigma(M), and the detection probabilities at M_3sigma and 2 M_3sigma.
Event rate: 30.2 events/day (51.6 deg, 420 km orbit; Sect. design); tangent grid
12-28 km at 0.5 km = 33 bins.  Writes outputs/false_alarm_trials.txt."""
from pathlib import Path
from scipy.stats import norm

EVENTS_PER_DAY = 30.2
BINS = 33
lines = []
def out(s=""):
    print(s); lines.append(s)
out("tests/yr | FAR | z_c | M_thr/M_3sigma | P_det(M_3sigma) | P_det(2 M_3sigma)")
for label, tests_per_event in [("one 20-km test per event", 1), ("all 33 bins per event", BINS), ("33 bins x 2 fits (band, window)", 2 * BINS)]:
    n_per_year = EVENTS_PER_DAY * 365.25 * tests_per_event
    out(f"# {label}: {n_per_year:.3g} tests per year")
    for far_label, per_year in [("1/yr", 1.0), ("1/month", 12.0), ("1/day", 365.25)]:
        zc = norm.isf(per_year / n_per_year)
        out(f"  {n_per_year:9.3g} | {far_label:7s} | {zc:.2f} | {zc/3:.2f} | {norm.sf(zc-3):.2f} | {norm.sf(zc-6):.3f}")
    fp_per_day = norm.sf(3.0) * EVENTS_PER_DAY * tests_per_event
    out(f"  bare 3-sigma per test: expected false positives {fp_per_day:.2f} per day")
Path(__file__).resolve().parents[1].joinpath("outputs/false_alarm_trials.txt").write_text("\n".join(lines) + "\n")
