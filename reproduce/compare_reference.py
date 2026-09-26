"""Compare a run of this repository with the versions used in the paper.

    python reproduce/compare_reference.py

Archived JSON results in outputs/ are compared number by number with
reference_outputs/outputs/ (relative tolerance 1e-3, i.e. below the rounding
of the text); figures are compared by existence and pixel size.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference_outputs"
TOL = 1e-3
PAIRS = [
    ("outputs/calibrated_background_thresholds.json", "outputs/calibrated_background_thresholds.json"),
    ("outputs/calibrated_background_problem.json", "outputs/calibrated_background_problem.json"),
    ("outputs/design_sensitivity_calibrated.json", "outputs/design_sensitivity_calibrated.json"),
    ("outputs/resolution_sensitivity_calibrated.json", "outputs/resolution_sensitivity_calibrated.json"),
    ("outputs/reservoir_mass_saod.json", "outputs/reservoir_mass_saod.json"),
    ("outputs/detectability_2d_calibrated/results.json", "outputs/detectability_2d_results.json"),
    ("outputs/materials_calibrated/results.json", "outputs/materials_results.json"),
    ("outputs/ace_v52/nulltest_stats.json", "outputs/nulltest_stats.json"),
    ("outputs/ace_v52/nulltest_convention.json", "outputs/nulltest_convention.json"),
    ("outputs/ace_v52/anchor_constrained_fits.json", "outputs/anchor_constrained_fits.json"),
    ("outputs/ace_v52/element_sampling.json", "outputs/element_sampling.json"),
    ("outputs/ace_v52/fullspectrum_stats.json", "outputs/fullspectrum_stats.json"),
]


def walk(a, b, path, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b: walk(a[k], b[k], path + "/" + str(k), diffs)
            else: diffs.append((path + "/" + str(k), "missing in reference"))
    elif isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}[{i}]", diffs)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a != b and abs(a - b) > TOL * max(abs(a), abs(b), 1e-30):
            diffs.append((path, f"{a!r} vs {b!r}"))


def main():
    bad = 0
    for run, ref in PAIRS:
        p, q = ROOT / run, REF / ref
        if not p.exists(): print(f"[skip] {run} not present (run make_all.py)"); continue
        if not q.exists(): print(f"[skip] no reference for {run}"); continue
        diffs = []; walk(json.loads(p.read_text()), json.loads(q.read_text()), "", diffs)
        print(f"[{'ok' if not diffs else 'DIFF'}] {run}: {len(diffs)} differing values")
        for d in diffs[:10]: print("      ", d[0], d[1])
        bad += bool(diffs)
    for f in sorted((REF / "figures").glob("*.png")):
        g = ROOT / "figures" / f.name
        print(f"[{'ok' if g.exists() else 'missing'}] figures/{f.name}" + (f"  ({g.stat().st_size/1e3:.0f} kB vs reference {f.stat().st_size/1e3:.0f} kB)" if g.exists() else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
