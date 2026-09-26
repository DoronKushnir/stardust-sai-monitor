# Examples

Run from the repository root (each script prints what it computes):

| Script | What it shows |
|---|---|
| `slant_od_silica_element.py` | slant optical depth of an injected silica layer at the 8.80 µm element and the heritage channels, for any mass and size distribution (Sect. 2–3 forward model) |
| `threshold_custom_psd.py` | the marginalized 3σ detection threshold of the baseline design for a size distribution of your choice (the calculation of Table 2, one row) |
| `floor_at_element.py` | the measured ACE-FTS residual floor at any 0.1-µm element from the shipped atlas (Appendix E) |

The reproduction scripts in `reproduce/` are the complete worked examples;
`python make_all.py --list` maps them to the paper's figures and tables.
