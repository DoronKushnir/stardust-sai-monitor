"""Compare scientific outputs with the reference archive; see COMPARISON.md.

Missing products, schema changes, changed flags/counts and scientific differences
fail. A hash-backed make_all receipt distinguishes regeneration from a comparison
of shipped archives. Explicit numerical diagnostics are reported separately.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference_outputs"
TOL, ATOL = 1e-3, 1e-8
SKIP = ("runtime_s", "seconds", "elapsed")

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
    ("outputs/ace_v52/nulltest_resolution.json", "outputs/nulltest_resolution.json"),
    ("outputs/ace_v52/spike_test.json", "outputs/spike_test.json"),
    ("outputs/window_vis_roster_sweep.json", "outputs/window_vis_roster_sweep.json"),
    ("outputs/m2_vertical_correlation_floor.json", "outputs/m2_vertical_correlation_floor.json"),
    ("outputs/o3_slant_latitude_measured.json", "outputs/o3_slant_latitude_measured.json"),
    ("outputs/detectability_2d_calibrated/results_o3lat.json", "outputs/detectability_2d_results_o3lat.json"),
    ("outputs/signal_vs_background.json", "outputs/signal_vs_background.json"),
    ("outputs/sulfate_coating.json", "outputs/sulfate_coating.json"),
    ("outputs/composite_particle.json", "outputs/composite_particle.json"),
    ("outputs/ace_v52/ash_positive_control.json", "outputs/ash_positive_control.json"),
    # round 70 (referee round 5)
    ("outputs/element_observable_closure.json", "outputs/element_observable_closure.json"),
    ("outputs/marginal_rank.json", "outputs/marginal_rank.json"),
    ("outputs/latitude_declared_level.json", "outputs/latitude_declared_level.json"),
    ("outputs/radiometric_budget.json", "outputs/radiometric_budget.json"),
    ("outputs/materials_uniaxial.json", "outputs/materials_uniaxial.json"),
    ("outputs/psd_tail_coagulation.json", "outputs/psd_tail_coagulation.json"),
    ("outputs/ace_v52/nulltest_diagnostics.json", "outputs/nulltest_diagnostics.json"),
    # rounds 71, 74 and 75 (referee round 6 and its completion)
    ("outputs/composite_particle_uniaxial.json", "outputs/composite_particle_uniaxial.json"),
    ("outputs/materials_farir_uniaxial.json", "outputs/materials_farir_uniaxial.json"),
    ("outputs/ace_nulltest_nonlinear_refit.json", "outputs/ace_nulltest_nonlinear_refit.json"),
    ("outputs/uncertainty_model.json", "outputs/uncertainty_model.json"),
    ("outputs/closure_variants.json", "outputs/closure_variants.json"),
    ("outputs/closure_flat_prior_check.json", "outputs/closure_flat_prior_check.json"),
    ("outputs/grid_convergence.json", "outputs/grid_convergence.json"),
    ("outputs/materials_scattering.json", "outputs/materials_scattering.json"),
    ("outputs/materials_primary.json", "outputs/materials_primary.json"),
    ("outputs/radiometric_budget_observable.json", "outputs/radiometric_budget_observable.json"),
    ("outputs/ace_null_baseline.json", "outputs/ace_null_baseline.json"),
    ("outputs/adopted_robustness.json", "outputs/adopted_robustness.json"),
    ("outputs/adopted_latitude.json", "outputs/adopted_latitude.json"),
    ("outputs/multitangent_retrieval.json", "outputs/multitangent_retrieval.json"),
    ("outputs/farir_adopted.json", "outputs/farir_adopted.json"),
    ("outputs/dda_uniaxial.json", "outputs/dda_uniaxial.json"),
    ("outputs/ace_null_trend.json", "outputs/ace_null_trend.json"),
    ("outputs/gas_budget_conditioning.json", "outputs/gas_budget_conditioning.json"),
    ("outputs/closure_final.json", "outputs/closure_final.json"),
    ("outputs/closure_stability.json", "outputs/closure_stability.json"),
    ("outputs/closure_stability_summary.json", "outputs/closure_stability_summary.json"),
    ("outputs/closure_stability_summary.csv", "outputs/closure_stability_summary.csv"),
]

def diagnostic_reason(product, path):
    if product == "outputs/marginal_rank.json" and path == "/summary/n_bit_identical_new_vs_old":
        return "Bit identity depends on floating-point library/platform; numerical rank and thresholds remain checked."
    if product.startswith("outputs/closure_stability") and (
        path.endswith(("/scaled_projected_gradient", "/max_scaled_projected_gradient",
                       "/optimizer_status", "/nfev"))):
        return "Optimizer path diagnostic; full-state convergence and non-increasing objective are independently required."
    return None


def absolute_tolerance(product, path):
    if not product.startswith("outputs/closure_stability"):
        return ATOL
    key = path.rsplit("/", 1)[-1]
    if key in ("mass_tg", "bias_median", "max_abs_mass_change_tg") or key.endswith("_bias_median"):
        return 3e-5
    if key == "bias_mean" or key.endswith("_bias_mean"):
        return 1e-6
    if key in ("sigma_tg", "sd", "posterior_sigma_median") or key.endswith(("_sd", "_posterior_sigma_median")):
        return 3e-6
    return ATOL


def walk(a, b, path, diffs, product="", diagnostics=None):
    diagnostics = [] if diagnostics is None else diagnostics
    reason = diagnostic_reason(product, path)
    if reason:
        diagnostics.append(dict(path=path, actual=a, reference=b, changed=a != b, reason=reason))
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if any(t in str(key) for t in SKIP):
                continue
            here = path + "/" + str(key)
            if key not in a or key not in b:
                diffs.append((here, "missing key in output" if key not in a else "missing key in reference"))
            else:
                walk(a[key], b[key], here, diffs, product, diagnostics)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, f"list length {len(a)} vs {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", diffs, product, diagnostics)
    elif isinstance(a, bool) or isinstance(b, bool):
        if type(a) is not type(b) or a != b:
            diffs.append((path, f"flag {a!r} vs {b!r}"))
    elif isinstance(a, int) and isinstance(b, int):
        if a != b:
            diffs.append((path, f"count {a} vs {b}"))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        exact = any(k in path for k in ("coverage_", "p_detect_"))
        equal = a == b or (math.isnan(a) and math.isnan(b))
        if not equal and (exact or not math.isclose(a, b, rel_tol=TOL,
                             abs_tol=absolute_tolerance(product, path))):
            diffs.append((path, f"{a!r} vs {b!r}"))
    elif type(a) is not type(b) or (a is None) != (b is None):
        diffs.append((path, f"type/value {a!r} vs {b!r}"))
    elif isinstance(a, str) and path.endswith(("/case", "/config", "/verdict")) and a != b:
        diffs.append((path, f"identity {a!r} vs {b!r}"))
    # Other strings are descriptive metadata, not numerical outputs.


def load_product(path):
    if path.suffix != ".csv":
        return json.loads(path.read_text())
    rows = list(csv.DictReader(path.open()))
    for row in rows:
        for key, value in row.items():
            if key == "case":
                continue
            row[key] = int(value) if re.fullmatch(r"-?\d+", value) else float(value)
    return rows


def audit_invariants(data):
    """Check convergence/coverage from every draw, independent of the reference."""
    errors = []
    rows = data["rows"]
    if len(rows) != 24 or data["n_fits"] != 1596 or sum(len(r["draws"]) for r in rows) != 1596:
        errors.append("Expected 24 rows and 1596 retained fits")
    for row in rows:
        draws = row["draws"]
        name = f"{row['case']} truth={row['truth_tg']}"
        for i, d in enumerate(draws):
            g = d["after"]["scaled_projected_gradient"]
            if not (math.isfinite(g) and 0 <= g <= 1e-3 and d["converged"] is True):
                errors.append(f"{name} draw {i}: failed full-state convergence")
            before, after = d["before"]["objective"], d["after"]["objective"]
            if not (math.isfinite(before) and math.isfinite(after) and after <= before + 1e-7):
                errors.append(f"{name} draw {i}: invalid/increasing objective")
        for key, expected in (("n_legacy_unconverged", sum(not d["legacy_converged"] for d in draws)),
                              ("n_refitted", sum(d["refitted"] for d in draws)),
                              ("n_refined_unconverged", sum(not d["converged"] for d in draws))):
            if row[key] != expected:
                errors.append(f"{name}: inconsistent {key}")
        for summary_name in ("legacy", "refined", "refit_legacy_failures_only"):
            prefixes = ["legacy_" if summary_name == "legacy" or
                        (summary_name == "refit_legacy_failures_only" and d["legacy_converged"]) else "" for d in draws]
            mass = [d[p+"mass_tg"] for p, d in zip(prefixes, draws)]
            sigma = [d[p+"sigma_tg"] for p, d in zip(prefixes, draws)]
            if not all(math.isfinite(m) and math.isfinite(s) and s > 0 for m, s in zip(mass, sigma)):
                errors.append(f"{name}: invalid mass/uncertainty")
                continue
            residual = [m-row["truth_tg"] for m in mass]
            z = [r/s for r,s in zip(residual, sigma)]
            n = len(draws)
            stats = dict(bias_mean=statistics.mean(residual), bias_median=statistics.median(residual),
                         sd=statistics.stdev(mass), posterior_sigma_median=statistics.median(sigma),
                         coverage_1sig=sum(abs(v)<1 for v in z)/n, coverage_2sig=sum(abs(v)<2 for v in z)/n,
                         p_detect_3sig=sum(m/s>3 for m,s in zip(mass,sigma))/n,
                         p_detect_4p54sig=sum(m/s>4.54 for m,s in zip(mass,sigma))/n)
            for key, value in stats.items():
                if not math.isclose(value, row[summary_name][key], rel_tol=1e-10, abs_tol=1e-12):
                    errors.append(f"{name} {summary_name}: inconsistent {key}")
    return errors



def summary_invariants(audit, summary):
    draws = [d for row in audit["rows"] for d in row["draws"]]
    expected = dict(n_fits=len(draws), n_legacy_unconverged=sum(not d["legacy_converged"] for d in draws),
        n_refitted=sum(d["refitted"] for d in draws), n_refined_unconverged=sum(not d["converged"] for d in draws),
        max_scaled_projected_gradient=max(d["after"]["scaled_projected_gradient"] for d in draws),
        max_objective_increase=max(0., max(d["after"]["objective"]-d["before"]["objective"] for d in draws)),
        max_abs_mass_change_tg=max(abs(d["mass_tg"]-d["legacy_mass_tg"]) for d in draws))
    return [f"summary {key} disagrees with per-draw audit" for key, value in expected.items()
            if not math.isclose(value, summary[key], rel_tol=1e-10, abs_tol=1e-12)]


def provenance(root, product, receipt):
    record = receipt.get("products", {}).get(product)
    if record and (root/product).exists() and record["sha256"] == hashlib.sha256((root/product).read_bytes()).hexdigest():
        return f"regenerated by recorded successful step {record['step']} at {record['finished_utc']}"
    return "archive consistency only; no matching regeneration receipt"


def compare(root=ROOT, reference=REF, receipt=None):
    receipt = {} if receipt is None else receipt
    report = dict(policy="COMPARISON.md", products=[], figures=[])
    for product, ref in PAIRS:
        p, q = root/product, reference/ref
        entry = dict(product=product, provenance=provenance(root, product, receipt), differences=[], diagnostics=[])
        if not p.exists() or not q.exists():
            entry["differences"].append(("", "missing output or reference"))
        else:
            a, b = load_product(p), load_product(q)
            walk(a, b, "", entry["differences"], product, entry["diagnostics"])
            if product == "outputs/closure_stability.json":
                for side, data in (("output", a), ("reference", b)):
                    entry["differences"].extend((side, error) for error in audit_invariants(data))
        if product == "outputs/closure_stability_summary.json" and p.exists() and q.exists():
            for side, base, summary in (("output", root, a), ("reference", reference, b)):
                audit_path = base/"outputs/closure_stability.json"
                if audit_path.exists():
                    entry["differences"].extend((side, error) for error in
                        summary_invariants(json.loads(audit_path.read_text()), summary))
        report["products"].append(entry)
    for f in sorted((reference/"figures").glob("*.png")):
        g = root/"figures"/f.name
        report["figures"].append(dict(path="figures/"+f.name, exists=g.exists(),
            check="existence only, not pixel equality or proof of regeneration"))
    report["failed_products"] = sum(bool(e["differences"]) for e in report["products"])
    report["missing_figures"] = sum(not e["exists"] for e in report["figures"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write complete differences, numerical diagnostics and provenance")
    args = parser.parse_args()
    receipt_path = ROOT/"outputs/reproduction_run.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    report = compare(receipt=receipt)
    for e in report["products"]:
        changed = sum(d["changed"] for d in e["diagnostics"])
        print(f"[{'DIFF' if e['differences'] else 'ok'}] {e['product']}: {len(e['differences'])} scientific differences; {changed} differing explicit diagnostics; {e['provenance']}")
        for path, reason in e["differences"][:10]:
            print("     ", path, reason)
        for d in (d for d in e["diagnostics"] if d["changed"]):
            print("  diagnostic", d["path"], repr(d["actual"]), "vs", repr(d["reference"]))
    for e in report["figures"]:
        print(f"[{'present' if e['exists'] else 'MISSING'}] {e['path']}: {e['check']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2)+"\n")
    print(f"Compared {len(report['products'])} products; {report['failed_products']} failed; {report['missing_figures']} missing figures.")
    return bool(report["failed_products"] or report["missing_figures"])


if __name__ == "__main__":
    raise SystemExit(main())
