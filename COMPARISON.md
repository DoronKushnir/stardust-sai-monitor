# What the reference comparison establishes

Run `python reproduce/compare_reference.py --json outputs/reference_comparison.json`
after `python make_all.py --fast` or a selected/full run. The command checks 51
JSON/CSV products, including the frozen-trend seasonal statistics, gas budget,
initial closure fits, every initial/refined closure draw, and the closure summary
in both JSON and CSV. It exits nonzero for missing products/references, changed
list lengths or dictionary keys, differing scientific numbers, flags/counts,
coverage/detection fractions, failed closure invariants, or missing figures.
Descriptive metadata strings are not numerical tests; case/configuration/verdict
identifiers are checked. Figure checks establish existence only.

## Regenerated products versus shipped archives

Each `make_all.py` invocation writes a local `outputs/reproduction_run.json`.
It records successful steps, script hashes, commands, times, and SHA-256 hashes
of comparison products rewritten during the step. The comparator calls a product
regenerated only if it has a matching successful-step receipt and its current
hash matches. All other products are labelled **archive consistency only**.
A matching receipt establishes execution of the declared producer, which can
itself read shipped derived inputs; it is not independent reprocessing of raw
measurements. The receipt is ignored by git and is never shipped as evidence of
regeneration on another computer. A new invocation starts a new receipt.

## Numerical policy

The default floating-point tolerance remains relative 0.001 or absolute 1e-8,
whichever is larger. Integer counts, Boolean convergence/refit flags, and
coverage/detection fractions must agree exactly. Equal NaNs/infinities are
accepted in diagnostic outputs; a finite/nonfinite change fails. The convergence
archive separately requires finite positive posterior uncertainties and finite
masses/objectives/returned gradients for every draw.

For `closure_stability` products only, cancellation and optimizer-path differences
are compared with the following absolute tolerances (the default relative rule
still applies). These are below the reported mass-sensitivity precision and are
stated here rather than hidden by a global tolerance increase:

| Quantity | Absolute tolerance, Tg |
|---|---:|
| Refined per-draw mass, median bias, largest individual adjustment | 3e-5 |
| Mean bias | 1e-6 |
| Refined posterior uncertainty, sample SD, median posterior uncertainty | 3e-6 |

All 1596 draws must remain in 24 rows. The comparator reconstructs means,
medians, sample SDs, posterior medians, coverage and detection fractions from the
individual masses/uncertainties, including the initial-failures-only comparison.
These internal identities use 1e-10 relative / 1e-12 absolute tolerance.
Each returned scaled projected gradient must be at most 1e-3, and each objective
must not exceed its initial objective by more than 1e-7. These constraints are
checked independently for output and reference, even if their bytes agree.

The following numerical diagnostics are reported with both values in the JSON
report, and differing values are printed, but they do not require bit identity:

- `marginal_rank.json/summary/n_bit_identical_new_vs_old`. This depends on the
  floating-point library; all ranks, finite/infinite classifications, individual
  thresholds, and relative-difference diagnostics remain checked.
- `closure_stability` scaled projected gradients, optimizer stopping codes, and
  evaluation counts, plus the summary maximum scaled gradient. Different optimizer
  trajectories are allowed only while the scientific comparisons and independent
  convergence/objective/coverage checks pass. They are not silently discarded.
- Wall-clock fields containing `runtime_s`, `seconds`, or `elapsed` are excluded
  from numeric equality, as in the preceding comparator.

## Reconciliation of older diagnostics

The latitude persistence calculation now uses a stable chronological sort.
When decimal-year timestamps tie, the shipped CSV row order is retained. This
makes adjacent-pair counts deterministic across NumPy sort implementations;
these tied observations do not establish a finer time ordering. The refreshed
persistence reference is calculated from that rule. None of its fields is
excluded from comparison.

The three stale alumina reference fields describe comparison with the previously
shipped scalar-optics archive. They are refreshed from the current
`outputs/materials_calibrated/results.json` and the already archived far-infrared
predictions, following the existing producer's quadratic-free ratio calculation.
This is a reconciliation of reference diagnostics, not a new far-infrared forward
run. Scientific prediction fields are unchanged. `reference_reconciliation_rc9.json`
records the exact before/after fields, inputs and hashes.
