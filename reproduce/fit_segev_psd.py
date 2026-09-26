"""
fit_segev_psd.py — Compare the measured silica dispersion PSDs of
Segev et al. (2026, Fig. 2b) with our single-mode lognormal assumption.

The CSV (data/digitized/Segev2026_Fig2b.csv) holds the cumulative distribution
function (CDF) versus particle diameter [µm] for two dispersed-silica cases,
D300 and D500.  A third case, D300o ("optimal" dispersion, Segev et al. 2026
Fig. 5; CDF_best.xlsx), is the narrowest experimental PSD and our adopted
detection baseline: its 4 µm-truncated lognormal fit is r_med = 268 nm,
sigma_g = 1.31.  We (i) fit each case with a single-mode lognormal CDF (full
and truncated at 4 µm), and (ii) overlay a near-monodisperse reference
(r_med = 250 nm, D_med = 0.5 µm, sigma = 1.05).

Output: figures/segev_psd_fit.png
"""

import sys
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

CSV = "data/digitized/Segev2026_Fig2b.csv"

# Paper's roughly-scanned decile markers (diameter [µm] at 10/30/50/70/90 %)
SCANNED = {
    'D300': [0.302, 0.392, 0.499, 0.700, 8.77],
    'D500': [0.452, 0.582, 0.799, 2.16, 294.0],
}
PCT = np.array([0.10, 0.30, 0.50, 0.70, 0.90])

# Baseline detection-model assumption
DMED_ASSUMED = 0.5     # µm diameter  (r_med = 250 nm)
SIGMA_ASSUMED = 1.05


def load(col_x, col_y):
    raw = np.genfromtxt(CSV, delimiter=',', skip_header=2)
    x, y = raw[:, col_x], raw[:, col_y]
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    o = np.argsort(x)
    return x[o], np.maximum.accumulate(y[o])     # enforce monotone CDF


def load_xlsx_cdf(path):
    """Read an .xlsx (zip of XML, stdlib only) with columns
    [dmin, dmax, fraction, CDF].  Returns (D=dmax [µm], F=CDF), i.e. the
    cumulative fraction with diameter <= the bin's upper edge."""
    import zipfile, re
    import xml.etree.ElementTree as ET
    M = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    z = zipfile.ZipFile(path)
    root = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    def colnum(ref):
        mm = re.match(r'([A-Z]+)(\d+)', ref); col = 0
        for ch in mm.group(1): col = col * 26 + (ord(ch) - 64)
        return col - 1, int(mm.group(2))
    rows = {}
    for c in root.iter(M + 'c'):
        v = c.find(M + 'v')
        if v is None or c.get('t') == 's':     # skip header text cells
            continue
        ci, ri = colnum(c.get('r'))
        rows.setdefault(ri, {})[ci] = float(v.text)
    rr = sorted(r for r in rows if r > 1)      # row 1 is the header
    D = np.array([rows[r][1] for r in rr])     # dmax
    F = np.array([rows[r][3] for r in rr])     # CDF
    o = np.argsort(D)
    return D[o], np.maximum.accumulate(F[o])


def lognormal_cdf(D, Dmed, sigma_g):
    return norm.cdf(np.log(D / Dmed) / np.log(sigma_g))


# D300o = "optimal" dispersion from Segev et al. (2026, their Fig. 5); the
# narrowest experimental PSD and our adopted detection baseline after the
# 4 µm truncation (r_med = 268 nm, sigma_g = 1.31).
cases = {'D300': load(0, 1), 'D500': load(2, 3),
         'D300o': load_xlsx_cdf("data/digitized/Segev2026_CDF_best.xlsx")}
colors = {'D300': 'tab:blue', 'D500': 'tab:red', 'D300o': 'black'}   # round 44 (Doron): D300o green -> black
DCUT = 4.0   # µm — truncation diameter (large particles sediment out fast)

fits, tfits, massrem = {}, {}, {}
print("Lognormal fits (full and truncated at %.0f µm):" % DCUT)
for nm, (D, F) in cases.items():
    F = F / F[-1]
    fits[nm], _ = curve_fit(lognormal_cdf, D, F, p0=[0.5, 2.0], maxfev=20000)
    cdf_cut = np.interp(DCUT, D, F)
    massrem[nm] = 1.0 - cdf_cut                       # volume/mass-weighted CDF
    sel = D <= DCUT
    tfits[nm], _ = curve_fit(lognormal_cdf, D[sel], F[sel] / cdf_cut,
                             p0=[0.5, 1.4], maxfev=20000)
    print(f"  {nm}: full  r_med={fits[nm][0]/2*1000:.0f} nm, sigma_g={fits[nm][1]:.2f}")
    print(f"       trunc r_med={tfits[nm][0]/2*1000:.0f} nm, sigma_g={tfits[nm][1]:.2f}  "
          f"(mass removed = {massrem[nm]*100:.1f}%)")

# ── Figure: truncated-at-4µm CDF + lognormal fits (single panel) ──
# Round 41 (Doron, paper1 Fig. segev_psd_fit): paper-grade styling; the
# sigma = 1.05 reference curve, the "(<4 um)" legend qualifier and the
# "mass removed" inset are dropped (the removed mass fractions are printed
# above and quoted in the caption).
# Round 42 (Doron): only the truncated panel is kept, with the full-panel
# y-label, no title, and the fitted r_med / sigma_psd moved into the legend.
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix", "font.serif": ["STIXGeneral"],
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.8,
})
LABEL = {'D300': 'D300', 'D500': 'D500', 'D300o': 'D300o (optimal)'}
import matplotlib.ticker as mticker
fig, axB = plt.subplots(figsize=(12.0 / 2.54 * 1.15, 9.5 / 2.54 * 1.15))
DgridT = np.logspace(np.log10(0.2), np.log10(DCUT), 300)
for nm, (D, F) in cases.items():
    c = colors[nm]; Dmed, sg = tfits[nm]
    Fn = F / F[-1]; cdf_cut = np.interp(DCUT, D, Fn)
    sel = D <= DCUT
    axB.semilogx(D[sel], Fn[sel] / cdf_cut, 'o', ms=3.0, color=c, alpha=0.55, mew=0,
                 label=f'{LABEL[nm]}, measured')
    axB.semilogx(DgridT, lognormal_cdf(DgridT, Dmed, sg), '-', color=c,
                 label=fr'{LABEL[nm]}, lognormal fit: $r_\mathrm{{med}}={Dmed/2*1000:.0f}$ nm, '
                       fr'$\sigma_\mathrm{{psd}}={sg:.2f}$')
axB.axvline(DCUT, color='0.4', ls='--', lw=0.9)
axB.set_xlabel(r'Particle diameter $D$ [$\mu$m]')
axB.set_ylabel('Mass-weighted cumulative distribution')
axB.set_xlim(0.2, 6); axB.set_ylim(-0.02, 1.02)
axB.grid(True, which='both', alpha=0.25, lw=0.5)
axB.set_xticks([0.2, 0.5, 1, 2, 4, 6])
axB.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
axB.xaxis.set_minor_formatter(mticker.NullFormatter())
axB.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False,
           columnspacing=1.2, handletextpad=0.5)
fig.tight_layout()
for out in (Path('figures/segev_psd_fit.png'), Path('figures/segev_psd_fit.png')):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\nSaved → {out}")
plt.close(fig)
