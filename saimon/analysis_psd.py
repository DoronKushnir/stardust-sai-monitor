"""
analysis_psd.py — Wrana et al. (2021) PSD retrieval figures.

Public functions:
    plot_wrana_fig2a            — left panel of Wrana Fig 2 (good wavelength set)
    plot_wrana_fig2b            — right panel (bad wavelength set, non-uniqueness demo)
    plot_qext_fig3              — Q_ext vs radius at three wavelengths (Wrana Fig 3)
    load_wrana_fig2a_csv        — load digitized Fig 2a CSV (11 sigma series)
    plot_wrana_fig2a_digitized  — overlay model + digitized data for Fig 2a

All functions accept an optional `digitized_ref` argument as a future hook
for overlaying digitised reference data from the published figure.
"""

from __future__ import annotations

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .interfaces import LookupTableConfig, ExtinctionRatioLookupTable

# r_med values annotated in Wrana Fig 2a (nm)
_ANNOTATED_RMED_NM = [50, 60, 160, 240, 370]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _effective_grids(lut: ExtinctionRatioLookupTable):
    """Return (sigma_grid, rmed_grid_m) — always arrays after the table is built."""
    sigma = lut.config.sigma_grid
    rmed  = lut.config.rmed_grid_m
    if sigma is None or rmed is None:
        from .aerosol_mie import _DEFAULT_SIGMA_GRID, _DEFAULT_RMED_GRID_M
        if sigma is None:
            sigma = _DEFAULT_SIGMA_GRID
        if rmed is None:
            rmed = _DEFAULT_RMED_GRID_M
    return np.asarray(sigma, dtype=float), np.asarray(rmed, dtype=float)


def _dot_indices(rmed_nm: np.ndarray, spacing_nm: float = 10.0):
    """Return indices of rmed_nm that are at least spacing_nm apart (linear)."""
    indices = []
    last = -1e9
    for j, r in enumerate(rmed_nm):
        if r - last >= spacing_nm:
            indices.append(j)
            last = r
    return indices


# ---------------------------------------------------------------------------
# Wrana Fig 2a — good wavelength set (449 / 756 / 1544 nm)
# ---------------------------------------------------------------------------

def plot_wrana_fig2a(
    lut: ExtinctionRatioLookupTable,
    ax=None,
    savepath: str = None,
    digitized_ref=None,
):
    """Reproduce the LEFT panel of Wrana et al. (2021) Fig 2.

    x-axis : ratio_x = k_ext(lambda_x)   / k_ext(lambda_ref)
    y-axis : ratio_y = k_ext(lambda_y)   / k_ext(lambda_ref)

    Parameters
    ----------
    lut           : ExtinctionRatioLookupTable from build_extinction_ratio_lookup_table.
    ax            : optional matplotlib Axes; created if None.
    savepath      : if given, saves the figure as PNG.
    digitized_ref : (x_array, y_array) of digitised reference points from the paper;
                    plotted as a red scatter overlay when provided.
                    Pass None for now; this hook is ready for the quantitative check.

    Returns
    -------
    fig, ax
    """
    sigma_grid, rmed_grid_m = _effective_grids(lut)
    rmed_nm = rmed_grid_m * 1e9
    n_sig   = len(sigma_grid)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    colors = cm.viridis(np.linspace(0.05, 0.95, n_sig))

    for i, sigma in enumerate(sigma_grid):
        ax.plot(lut.ratio_x[i], lut.ratio_y[i],
                color=colors[i], lw=1.5, label=f'σ = {sigma:.2f}')
        dot_idx = _dot_indices(rmed_nm, spacing_nm=10.0)
        ax.scatter(lut.ratio_x[i][dot_idx], lut.ratio_y[i][dot_idx],
                   color=colors[i], s=7, zorder=3)

    # Annotated dots on the σ ≈ 1.5 curve
    i_rep = int(np.argmin(np.abs(sigma_grid - 1.5)))
    ann_scatter_x = []
    ann_scatter_y = []
    for r_nm in _ANNOTATED_RMED_NM:
        j = int(np.argmin(np.abs(rmed_nm - r_nm)))
        rx, ry = float(lut.ratio_x[i_rep, j]), float(lut.ratio_y[i_rep, j])
        ann_scatter_x.append(rx)
        ann_scatter_y.append(ry)
        ax.scatter(rx, ry, color='black', s=50, zorder=6)
        ax.annotate(
            f'{r_nm}', (rx, ry),
            textcoords='offset points', xytext=(5, 3),
            fontsize=7, fontweight='bold',
        )
    # Store the annotated points as an axis attribute for test access
    ax._wrana_ann_x = ann_scatter_x
    ax._wrana_ann_y = ann_scatter_y
    ax._wrana_sigma_rep = sigma_grid[i_rep]

    if digitized_ref is not None:
        ref_x, ref_y = digitized_ref
        ax.scatter(ref_x, ref_y, color='red', s=30, zorder=7,
                   label='Wrana (digitized)', marker='x')

    cfg = lut.config
    ax.set_xlabel(
        f'Extinction ratio  ({cfg.lambda_x_m*1e9:.3f} nm / {cfg.lambda_ref_m*1e9:.3f} nm)',
        fontsize=11,
    )
    ax.set_ylabel(
        f'Extinction ratio  ({cfg.lambda_y_m*1e9:.3f} nm / {cfg.lambda_ref_m*1e9:.3f} nm)',
        fontsize=11,
    )
    ax.set_xlim(0.05, 0.55)
    ax.set_ylim(0.3, 3.5)
    ax.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Wrana et al. (2021) Fig 2a  (m = 1.4+0j)', fontsize=11)

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig, ax


# ---------------------------------------------------------------------------
# Wrana Fig 2b — bad wavelength set (384 / 448 / 520 nm)
# ---------------------------------------------------------------------------

def plot_wrana_fig2b(
    lut: ExtinctionRatioLookupTable = None,
    ax=None,
    savepath: str = None,
    digitized_ref=None,
):
    """Reproduce the RIGHT panel of Wrana et al. (2021) Fig 2.

    Uses the BAD wavelength combination 384/448/520 nm.  The broad non-unique
    region visually demonstrates why this set is unusable for size retrieval.

    If lut is None, builds the table with default PSD/Mie settings.
    """
    if lut is None:
        from .aerosol_mie import build_extinction_ratio_lookup_table
        bad_cfg = LookupTableConfig(
            lambda_ref_m=448.511e-9,
            lambda_y_m=384.224e-9,
            lambda_x_m=520.513e-9,
        )
        lut = build_extinction_ratio_lookup_table(config=bad_cfg)

    sigma_grid, rmed_grid_m = _effective_grids(lut)
    rmed_nm = rmed_grid_m * 1e9
    n_sig   = len(sigma_grid)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    colors = cm.viridis(np.linspace(0.05, 0.95, n_sig))

    for i, sigma in enumerate(sigma_grid):
        ax.plot(lut.ratio_x[i], lut.ratio_y[i],
                color=colors[i], lw=1.5, label=f'σ = {sigma:.2f}')
        dot_idx = _dot_indices(rmed_nm, spacing_nm=10.0)
        ax.scatter(lut.ratio_x[i][dot_idx], lut.ratio_y[i][dot_idx],
                   color=colors[i], s=7, zorder=3)

    if digitized_ref is not None:
        ref_x, ref_y = digitized_ref
        ax.scatter(ref_x, ref_y, color='red', s=30, zorder=7,
                   label='Wrana (digitized)', marker='x')

    cfg = lut.config
    ax.set_xlabel(
        f'Extinction ratio  ({cfg.lambda_x_m*1e9:.3f} nm / {cfg.lambda_ref_m*1e9:.3f} nm)',
        fontsize=11,
    )
    ax.set_ylabel(
        f'Extinction ratio  ({cfg.lambda_y_m*1e9:.3f} nm / {cfg.lambda_ref_m*1e9:.3f} nm)',
        fontsize=11,
    )
    ax.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Wrana et al. (2021) Fig 2b — bad wavelength set\n'
                 '(384/448/520 nm, broad non-unique region)', fontsize=11)

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig, ax


# ---------------------------------------------------------------------------
# Wrana Fig 2a — digitized CSV loader
# ---------------------------------------------------------------------------

_DEFAULT_FIG2A_CSV = os.path.join(
    str(__import__('saimon.config', fromlist=['DIGITIZED']).DIGITIZED), 'Wrana2021_Fig2a.csv'
)

# Sigma values corresponding to column-pair indices 0..10 in the CSV
_FIG2A_SIGMA_ORDER = [1.05, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00]

# Colors for σ 1.05 → 2.00 (violet → red)
_FIG2A_SIGMA_COLORS = [
    '#7f00ff', '#4d4dfb', '#1995f2', '#18cde3', '#4cf2ce',
    '#80feb3', '#b2f295', '#e6cd73', '#ff954e', '#ff4d27', '#ff0000',
]


def load_wrana_fig2a_csv(path=None):
    """Load digitized Wrana Fig 2a data from a ragged CSV.

    CSV format:
      Row 0: ``Sigma105,,Sigma110,,Sigma120,...`` — 11 sigma labels (22 cols total)
      Row 1: ``X,Y,X,Y,...`` — column headers
      Rows 2+: (X, Y) pairs per sigma; shorter series have empty trailing cells.

    Returns
    -------
    dict {sigma_value: (x_array, y_array)}
        Keys: 1.05, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00
        Values: float numpy arrays with NaN/empty rows dropped.
    """
    if path is None:
        path = _DEFAULT_FIG2A_CSV

    with open(path, newline='') as f:
        rows = list(csv.reader(f))

    # Skip header rows (row 0: sigma labels, row 1: X/Y headers)
    data_rows = rows[2:]

    # Collect (x, y) lists per sigma column-pair index
    n_sigma = len(_FIG2A_SIGMA_ORDER)
    xs = [[] for _ in range(n_sigma)]
    ys = [[] for _ in range(n_sigma)]

    for row in data_rows:
        for col_idx in range(n_sigma):
            xi = col_idx * 2
            yi = col_idx * 2 + 1
            # Pad row if shorter than required
            xv = row[xi].strip() if xi < len(row) else ''
            yv = row[yi].strip() if yi < len(row) else ''
            if xv and yv:
                try:
                    xs[col_idx].append(float(xv))
                    ys[col_idx].append(float(yv))
                except ValueError:
                    pass  # skip malformed cells

    result = {}
    for col_idx, sigma in enumerate(_FIG2A_SIGMA_ORDER):
        result[sigma] = (np.array(xs[col_idx], dtype=float),
                         np.array(ys[col_idx], dtype=float))
    return result


# ---------------------------------------------------------------------------
# Wrana Fig 2a — model + digitized overlay
# ---------------------------------------------------------------------------

def plot_wrana_fig2a_digitized(
    digitized_data: dict,
    lut=None,
    savepath: str = None,
):
    """Plot model LUT curves and digitized data for Wrana et al. (2021) Fig 2a.

    Parameters
    ----------
    digitized_data : dict output of :func:`load_wrana_fig2a_csv`.
    lut            : ExtinctionRatioLookupTable.  Built internally if None,
                     using BohrenHuffmanMie + 215 K H2SO4 (k forced to 0),
                     11 sigma values, rmed 20–800 nm (120 pts), r-integration
                     1 nm–10 µm (400 pts).
    savepath       : optional PNG output path.

    Returns
    -------
    fig, ax
    """
    from .mie import BohrenHuffmanMie
    from .refractive_index import sulfuric_acid_at_temperature
    from .aerosol_mie import build_extinction_ratio_lookup_table

    sigma_grid_arr = np.array(_FIG2A_SIGMA_ORDER)

    if lut is None:
        # Wrap the 215 K RI so that k is forced to 0 at every wavelength
        _ri_215 = sulfuric_acid_at_temperature(215.0)

        class _ZeroKWrapper:
            """Wraps a RI callable and forces k = 0."""
            def __init__(self, ri):
                self._ri = ri

            def __call__(self, wavelengths):
                m_arr = self._ri(wavelengths)
                return np.array([complex(float(m.real), 0.0) for m in m_arr])

            def __repr__(self):
                return f'ZeroKWrapper({self._ri!r})'

        ri_k0 = _ZeroKWrapper(_ri_215)

        cfg = LookupTableConfig(
            sigma_grid=sigma_grid_arr,
            rmed_grid_m=np.logspace(np.log10(20e-9), np.log10(800e-9), 120),
            r_integration_grid_m=np.logspace(np.log10(1e-9), np.log10(10e-6), 400),
        )
        lut = build_extinction_ratio_lookup_table(
            config=cfg,
            mie_model=BohrenHuffmanMie(),
            refractive_index=ri_k0,
        )

    # --- figure ---
    fig, ax = plt.subplots(figsize=(7, 6))

    lut_sigma = np.asarray(lut.config.sigma_grid, dtype=float)

    for i, sigma in enumerate(sigma_grid_arr):
        color = _FIG2A_SIGMA_COLORS[i]

        # Find matching row in LUT
        lut_idx = int(np.argmin(np.abs(lut_sigma - sigma)))

        # Model curve
        ax.plot(lut.ratio_x[lut_idx], lut.ratio_y[lut_idx],
                color=color, lw=1.5, label=f'σ = {sigma:.2f}')

        # Digitized markers (same color, no label)
        if sigma in digitized_data:
            x_arr, y_arr = digitized_data[sigma]
            if len(x_arr) > 0:
                ax.scatter(x_arr, y_arr, marker='o', s=18, zorder=5,
                           facecolors='none', edgecolors=color, linewidths=0.8)

    ax.set_xlim(0.08, 0.5)
    ax.set_ylim(0.4, 3.0)
    ax.set_xlabel('Extinction ratio (1543.92 nm / 755.979 nm)', fontsize=11)
    ax.set_ylabel('Extinction ratio (448.511 nm / 755.979 nm)', fontsize=11)
    ax.set_title('Wrana et al. (2021) Fig 2a  (215 K, k=0)', fontsize=11)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig, ax


# ---------------------------------------------------------------------------
# Wrana Fig 3 — digitized CSV loader
# ---------------------------------------------------------------------------

_DEFAULT_FIG3_CSV = os.path.join(
    str(__import__('saimon.config', fromlist=['DIGITIZED']).DIGITIZED), 'Wrana2021_Fig3.csv'
)

# Mapping: label → (matplotlib color, exact wavelength in nm)
_FIG3_WAVE_ORDER = [
    ('449',  'red',   448.511),
    ('756',  'green', 755.979),
    ('1544', 'blue',  1543.92),
]


def load_wrana_fig3_csv(path=None):
    """Load digitized Wrana Fig 3 data from a ragged CSV.

    Returns
    -------
    dict with keys '449', '756', '1544'; each value is (r_nm_array, q_ext_array).
    """
    if path is None:
        path = _DEFAULT_FIG3_CSV

    with open(path, newline='') as f:
        rows = list(csv.reader(f))

    # rows[0] = wavelength label row, rows[1] = X,Y header row, rows[2:] = data
    data_rows = rows[2:]

    # cols 0-1 → 1544 nm, cols 2-3 → 756 nm, cols 4-5 → 449 nm
    col_map = [(0, 1, '1544'), (2, 3, '756'), (4, 5, '449')]
    series = {'1544': ([], []), '756': ([], []), '449': ([], [])}

    for row in data_rows:
        row = row + [''] * max(0, 6 - len(row))
        for xi, yi, key in col_map:
            xv = row[xi].strip()
            yv = row[yi].strip()
            if xv and yv:
                series[key][0].append(float(xv))
                series[key][1].append(float(yv))

    return {k: (np.array(v[0]), np.array(v[1])) for k, v in series.items()}


# ---------------------------------------------------------------------------
# Wrana Fig 3 — Q_ext vs radius
# ---------------------------------------------------------------------------

def plot_qext_fig3(
    mie_model=None,
    refractive_index=None,
    savepath: str = None,
    digitized_ref=None,
    dr_nm: float = 1.0,
    show_backend_diff: bool = True,
    diff_per_wavelength: bool = False,
):
    """Reproduce Wrana Fig 3: Q_ext vs radius (1–1000 nm) at 449, 756, 1544 nm.

    Single-particle (monodisperse), no PSD averaging.

    Parameters
    ----------
    mie_model        : MieModelProtocol, default BohrenHuffmanMie().
    refractive_index : RefractiveIndexProtocol, default ConstantRI(1.4, 0).
    savepath         : optional PNG output path.
    digitized_ref    : None, a CSV path string, or dict {'449':(r,q), '756':..., '1544':...}.
                       When provided, overlays digitized points as open circles.
    dr_nm            : Radius step size in nm (default 1.0).  The linear grid
                       np.arange(1, 1000+dr, dr) fully resolves Mie ripples at
                       1 nm; do not use values coarser than 2 nm.
    show_backend_diff : If True, add a second panel showing pairwise absolute
                        differences between available Mie backends (default False).
    diff_per_wavelength : If True, plot per-wavelength diffs in the bottom panel
                          (colored by wavelength).  If False (default), plot the
                          element-wise max over wavelengths (all black).

    Returns
    -------
    fig, ax           when show_backend_diff=False  (backward compatible)
    fig, (ax_top, ax_bot)  when show_backend_diff=True
    """
    from .mie import BohrenHuffmanMie
    from .refractive_index import ConstantRI

    if mie_model is None:
        mie_model = BohrenHuffmanMie()
    if refractive_index is None:
        refractive_index = ConstantRI(1.4, 0.0)

    if isinstance(digitized_ref, str):
        digitized_ref = load_wrana_fig3_csv(digitized_ref)

    # Fine linear grid: 1 nm steps resolve Mie ripples that grow to ~17 nm
    # spacing on a log grid at r ≈ 900 nm (max error ~0.26 on log vs ~0.005 here).
    r_nm = np.arange(1.0, 1000.0 + 1e-9, dr_nm)
    r_m  = r_nm * 1e-9

    # ------------------------------------------------------------------
    # Single-panel path (backward compatible)
    # ------------------------------------------------------------------
    if not show_backend_diff:
        fig, ax = plt.subplots(figsize=(8, 5))

        for label, color, lam_nm in _FIG3_WAVE_ORDER:
            lam_m = lam_nm * 1e-9
            n_val = float(refractive_index(np.array([lam_m]))[0].real)
            m = complex(n_val, 0.0)
            x_param = 2.0 * np.pi * r_m / lam_m
            q_ext, _, _ = mie_model.efficiencies(x_param, m)
            ax.plot(r_nm, q_ext, color=color, lw=1.5,
                    label=f'{label} nm (n = {n_val:.3f}, k = 0)')

            if digitized_ref is not None and label in digitized_ref:
                ref_r, ref_q = digitized_ref[label]
                ax.scatter(ref_r, ref_q, color=color, s=8, zorder=5,
                           marker='o', facecolors='none', linewidths=0.5,
                           label=f'{label} nm (digitized)')

        ax.set_xlabel('Particle radius  r (nm)', fontsize=11)
        ax.set_ylabel('Extinction efficiency  $Q_{ext}$', fontsize=11)
        ax.set_title('Wrana Fig 3: $Q_{ext}$ vs radius', fontsize=11)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 5)
        ax.set_xscale('linear')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if savepath is not None:
            fig.savefig(savepath, dpi=150, bbox_inches='tight')
        return fig, ax

    # ------------------------------------------------------------------
    # Two-panel path: top = main plot, bottom = backend differences
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    # --- Top panel: same as the single-panel, no x-label ---
    for label, color, lam_nm in _FIG3_WAVE_ORDER:
        lam_m = lam_nm * 1e-9
        n_val = float(refractive_index(np.array([lam_m]))[0].real)
        m = complex(n_val, 0.0)
        x_param = 2.0 * np.pi * r_m / lam_m
        q_ext, _, _ = mie_model.efficiencies(x_param, m)
        ax_top.plot(r_nm, q_ext, color=color, lw=1.5,
                    label=f'{label} nm (n = {n_val:.3f}, k = 0)')

        if digitized_ref is not None and label in digitized_ref:
            ref_r, ref_q = digitized_ref[label]
            ax_top.scatter(ref_r, ref_q, color=color, s=8, zorder=5,
                           marker='o', facecolors='none', linewidths=0.5,
                           label=f'{label} nm (digitized)')

    ax_top.set_ylabel('Extinction efficiency  $Q_{ext}$', fontsize=11)
    ax_top.set_title('Wrana Fig 3: $Q_{ext}$ vs radius', fontsize=11)
    ax_top.set_xlim(0, 1000)
    ax_top.set_ylim(0, 5)
    ax_top.set_xscale('linear')
    ax_top.legend(fontsize=9)
    ax_top.grid(True, alpha=0.3)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # --- Bottom panel: compute Q_ext for all available backends ---
    from .mie import BohrenHuffmanMie as _BHMie, MiePythonBackend as _MPB
    from .mie_oxford import OxfordMie as _OxMie

    backends = {'BH': _BHMie(), 'Oxford': _OxMie()}
    miepython_available = True
    try:
        import miepython as _miepython_probe  # noqa: F401
        backends['miepython'] = _MPB()
    except ImportError:
        miepython_available = False

    # Compute Q_ext: qext_data[name][wav_idx] = array of len(r_nm)
    qext_data = {}
    for bname, backend in list(backends.items()):
        qext_data[bname] = []
        failed = False
        for _label, _color, lam_nm in _FIG3_WAVE_ORDER:
            lam_m = lam_nm * 1e-9
            n_val = float(refractive_index(np.array([lam_m]))[0].real)
            m = complex(n_val, 0.0)
            x_param = 2.0 * np.pi * r_m / lam_m
            try:
                q_ext, _, _ = backend.efficiencies(x_param, m)
            except ImportError:
                failed = True
                break
            qext_data[bname].append(q_ext)
        if failed:
            del qext_data[bname]
            del backends[bname]
            if bname == 'miepython':
                miepython_available = False

    # Pairwise differences
    bnames = list(backends.keys())
    linestyles = ['-', '--', ':']
    pair_idx = 0
    for i in range(len(bnames)):
        for j in range(i + 1, len(bnames)):
            a, b = bnames[i], bnames[j]
            ls = linestyles[pair_idx % len(linestyles)]
            pair_label_str = f'|{a}−{b}|'

            if diff_per_wavelength:
                # One curve per wavelength per pair, colored by wavelength
                for wav_idx, (_wlabel, wcolor, _wlam) in enumerate(_FIG3_WAVE_ORDER):
                    diff_arr = np.maximum(
                        np.abs(qext_data[a][wav_idx] - qext_data[b][wav_idx]),
                        1e-16,
                    )
                    ax_bot.plot(r_nm, diff_arr, color=wcolor, ls=ls, lw=1.0,
                                label=f'{pair_label_str} {_wlabel}nm')
            else:
                # Element-wise max over all wavelengths, all black
                diffs = [
                    np.abs(qext_data[a][wi] - qext_data[b][wi])
                    for wi in range(len(_FIG3_WAVE_ORDER))
                ]
                max_diff = np.maximum(np.max(np.stack(diffs, axis=0), axis=0), 1e-16)
                ax_bot.plot(r_nm, max_diff, color='k', ls=ls, lw=1.0,
                            label=pair_label_str)

            pair_idx += 1

    ax_bot.set_yscale('log')
    ax_bot.set_ylabel('|ΔQ_ext|', fontsize=10)
    ax_bot.set_xlabel('Particle radius  r (nm)', fontsize=11)
    ax_bot.set_xlim(0, 1000)
    ax_bot.legend(fontsize=8)
    ax_bot.grid(True, alpha=0.3)

    if not miepython_available:
        ax_bot.text(0.02, 0.05, 'miepython unavailable',
                    transform=ax_bot.transAxes, fontsize=8, color='gray')

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    return fig, (ax_top, ax_bot)


# ---------------------------------------------------------------------------
# Backend cross-check harness (P13)
# ---------------------------------------------------------------------------

_FIG3_LAMS_M = np.array([448.511e-9, 755.979e-9, 1543.92e-9])
_FIG3_R_NM   = np.arange(1.0, 1001.0, 1.0)   # 1 nm grid, same as plot_qext_fig3


def compare_mie_backends(backends=None, refractive_index=None, config=None):
    """Run the full Wrana pipeline with each Mie backend and return max deviations.

    For each backend, using the provided (or default 215 K) refractive index and
    LookupTableConfig:
      - builds the extinction-ratio lookup table (ratio_x, ratio_y);
      - computes Fig 3 Q_ext curves at 449/756/1544 nm on a 1 nm radius grid
        with k forced to zero (matching Wrana Fig 3 convention).

    Returns max deviations relative to BohrenHuffmanMie (the reference backend):
        { backend_name: {'max_delta_ratio_x': float,
                         'max_delta_ratio_y': float,
                         'max_delta_qext':    float} }

    BohrenHuffmanMie is always included in the result with all-zero deviations.

    Parameters
    ----------
    backends         : dict name->MieModelProtocol; default includes
                       BohrenHuffmanMie(), OxfordMie(), and MiePythonBackend()
                       if miepython is importable.
    refractive_index : RefractiveIndexProtocol; default sulfuric_acid_at_temperature(215.0).
    config           : LookupTableConfig; default LookupTableConfig() (full grids).
    """
    from .aerosol_mie import build_extinction_ratio_lookup_table
    from .interfaces import LookupTableConfig
    from .mie import BohrenHuffmanMie
    from .mie_oxford import OxfordMie
    from .refractive_index import sulfuric_acid_at_temperature

    if refractive_index is None:
        refractive_index = sulfuric_acid_at_temperature(215.0)
    if config is None:
        config = LookupTableConfig()
    if backends is None:
        backends = {'BohrenHuffmanMie': BohrenHuffmanMie(),
                    'OxfordMie': OxfordMie()}
        try:
            from .mie import MiePythonBackend
            import miepython  # noqa: F401
            backends['MiePythonBackend'] = MiePythonBackend()
        except ImportError:
            pass

    # Pre-compute m values once (k forced to 0 for Fig 3, same for all backends)
    m_vals_complex = refractive_index(_FIG3_LAMS_M)
    m_k0 = [complex(float(m_vals_complex[i].real), 0.0) for i in range(3)]

    r_m = _FIG3_R_NM * 1e-9

    raw = {}
    for name, backend in backends.items():
        lut = build_extinction_ratio_lookup_table(
            config=config,
            mie_model=backend,
            refractive_index=refractive_index,
        )
        qext = np.zeros((3, len(_FIG3_R_NM)))
        for k_idx, lam_m in enumerate(_FIG3_LAMS_M):
            x_param = 2.0 * np.pi * r_m / lam_m
            qext[k_idx], _, _ = backend.efficiencies(x_param, m_k0[k_idx])
        raw[name] = {'lut': lut, 'qext': qext}

    ref = raw['BohrenHuffmanMie']
    ref_rx = ref['lut'].ratio_x
    ref_ry = ref['lut'].ratio_y
    ref_qe = ref['qext']

    deviations = {}
    for name, data in raw.items():
        if name == 'BohrenHuffmanMie':
            deviations[name] = {'max_delta_ratio_x': 0.0,
                                'max_delta_ratio_y': 0.0,
                                'max_delta_qext':    0.0}
            continue
        lut = data['lut']
        qext = data['qext']
        floor = 0.01
        drx = float(np.max(np.abs((lut.ratio_x - ref_rx) /
                                  np.maximum(np.abs(ref_rx), 1e-30))))
        dry = float(np.max(np.abs((lut.ratio_y - ref_ry) /
                                  np.maximum(np.abs(ref_ry), 1e-30))))
        dqe = float(np.max(np.abs(qext - ref_qe) /
                            np.maximum(np.abs(ref_qe), floor)))
        deviations[name] = {'max_delta_ratio_x': drx,
                            'max_delta_ratio_y': dry,
                            'max_delta_qext':    dqe}

    return deviations
