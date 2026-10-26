"""plot_ace_atlas_element.py -- redraw Fig. 3 (figures/scisat_ace_atlas_874.png) from the archived
fine_boxcar arrays of element_observable_closure.json (both sides in the element observable).
Run from the repo root after the closure script."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
d = json.load(open(ROOT / "outputs/element_observable_closure.json"))
a = d["atlas_test"]; fb = a["fine_boxcar"]; el = np.array(d["setup"]["elements_um"])
c = np.array(fb["centres_um"]); s_fit = a["with_sulfate"]["s_o3_element_observable"]
fig, ax = plt.subplots(figsize=(17 / 2.54, 9.0 / 2.54))
ax.plot(c, fb["y_atlas"], color="black", lw=1.0, label="ACE-FTS atlas, tropical 16--20 km (3967 spectra), $-\\ln\\langle T\\rangle$ per 0.02 $\\mu$m")
ax.plot(c, fb["y_model"], color="red", lw=1.0, label=f"line-resolved model, O$_3$ amplitude {s_fit:.2f} fitted at 9.2 $\\mu$m, same convention")
ax.plot(c, fb["y_silica_1tg"], color="tab:blue", lw=1.0, label="1 Tg silica")
ax.plot(c, fb["y_sulfate"], color="green", ls="--", lw=1.0, label="calibrated sulfate background")
ax.plot(el, a["with_sulfate"]["y_meas"], "o", color="black", ms=5, mfc="none", label="measured 0.1-$\\mu$m elements")
ax.plot(el, a["with_sulfate"]["y_model"], "s", color="red", ms=4, mfc="none", label="modelled 0.1-$\\mu$m elements")
ax.axvspan(8.75, 8.85, color="magenta", alpha=0.15); ax.axvspan(9.15, 9.25, color="green", alpha=0.15)
ax.set_xlabel("wavelength [$\\mu$m]"); ax.set_ylabel("element optical depth $-\\ln\\langle T\\rangle$ (18-km tangent)", fontsize=9)
ax.set_xlim(7.8, 9.3); ax.set_ylim(0, 2.0); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); out = ROOT / "figures/scisat_ace_atlas_874.png"; fig.savefig(out, dpi=200)
print("wrote", out)
