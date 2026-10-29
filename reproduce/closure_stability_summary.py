"""Summarize the all-draw convergence audit, including refitting only old failures."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs/closure_stability.json"
OUT = ROOT / "outputs/closure_stability_summary.json"


def stats(draws, truth, failures_only=False):
    masses, sigmas = [], []
    for draw in draws:
        prefix = "legacy_" if failures_only and draw["legacy_converged"] else ""
        masses.append(draw[prefix+"mass_tg"])
        sigmas.append(draw[prefix+"sigma_tg"])
    A, s = np.array(masses), np.array(sigmas)
    z = (A-truth)/s
    return dict(bias_mean=float(np.mean(A-truth)), bias_median=float(np.median(A-truth)),
                sd=float(np.std(A, ddof=1)), posterior_sigma_median=float(np.median(s)),
                coverage_1sig=float(np.mean(np.abs(z)<1)), coverage_2sig=float(np.mean(np.abs(z)<2)),
                p_detect_3sig=float(np.mean(A/s>3)), p_detect_4p54sig=float(np.mean(A/s>4.54)))


def main():
    archive = json.loads(ARCHIVE.read_text())
    rows = archive["rows"]
    metrics = tuple(stats(rows[0]["draws"], rows[0]["truth_tg"]))
    csv_rows = []
    for row in rows:
        row["refit_legacy_failures_only"] = stats(row["draws"], row["truth_tg"], failures_only=True)
        full = stats(row["draws"], row["truth_tg"])
        for metric in metrics:
            assert np.isclose(full[metric], row["refined"][metric], rtol=1e-10, atol=1e-12)
        record = {k: row[k] for k in ("case", "truth_tg", "n_legacy_unconverged", "n_refitted", "n_refined_unconverged")}
        for variant in ("legacy", "refit_legacy_failures_only", "refined"):
            record.update({variant+"_"+key: row[variant][key] for key in metrics})
        csv_rows.append(record)
    all_draws = [d for r in rows for d in r["draws"]]
    summary = dict(n_fits=len(all_draws), n_legacy_unconverged=sum(not d["legacy_converged"] for d in all_draws),
                   n_refitted=sum(d["refitted"] for d in all_draws),
                   n_refined_unconverged=sum(not d["converged"] for d in all_draws),
                   max_scaled_projected_gradient=max(d["after"]["scaled_projected_gradient"] for d in all_draws),
                   max_objective_increase=max(d["after"]["objective"]-d["before"]["objective"] for d in all_draws),
                   max_abs_mass_change_tg=max(abs(d["mass_tg"]-d["legacy_mass_tg"]) for d in all_draws),
                   changes={})
    for name, chosen in (("all_rows", rows), ("final_band", [r for r in rows if r["case"]=="final_band"]),
                         ("final_window", [r for r in rows if r["case"]=="final_window"])):
        summary["changes"][name] = {}
        for variant in ("refit_legacy_failures_only", "refined"):
            summary["changes"][name][variant] = {
                metric: max(abs(r[variant][metric]-r["legacy"][metric]) for r in chosen) for metric in metrics}
    ARCHIVE.write_text(json.dumps(archive, indent=1)+"\n")
    OUT.write_text(json.dumps(summary, indent=1)+"\n")
    with OUT.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    # Enrich the original ensemble in-place without changing any reported statistics.
    primary_path = ROOT / "outputs/closure_final.json"
    replay_path = ARCHIVE.with_suffix(".replay.json")
    # Use masses/statistics from this same replay when available. This avoids
    # pairing newly computed uncertainties with a different platform's masses.
    primary = json.loads((replay_path if replay_path.exists() else primary_path).read_text())
    for row in rows:
        name = row["case"]
        if name.startswith("fixed_database_"):
            group = primary["tests"]["misspecification_final_band"]["fixed_database"]["per_database"][name.removeprefix("fixed_database_")]
        elif name in ("final_band", "final_window", "spectroscopic_flat_window"):
            group = primary["tests"][name]
        else:
            group = primary["tests"]["misspecification_final_band"][name]
        target = group[str(row["truth_tg"])]
        assert np.allclose(target["masses_tg"], [d["legacy_mass_tg"] for d in row["draws"]], rtol=1e-9, atol=1e-11)
        target["posterior_sigmas_tg"] = [d["legacy_sigma_tg"] for d in row["draws"]]
        target["convergence_flags"] = [d["legacy_converged"] for d in row["draws"]]
        target["iterations"] = [d["legacy_iterations"] for d in row["draws"]]
    primary_path.write_text(json.dumps(primary, indent=1)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
