"""Export verified residual-floor results into Paper 1 (no refitting).

Round 36 (2026-09-25): the PRIMARY archive is the direct 0.1-um-element
analysis (data/residual_floor_w0p1, own manifest); the 0.25-um archive
(data/residual_floor) supplies the comparison column and the comparison
quotes of Appendix app:acefloor.

Called by Papers/paper1/build.sh. Companion to residual_floor_manuscript.py
(the ACE-manuscript exporter), sharing the same archived analysis record in
data/ace_floor/w0p25/. It

  1. verifies the archived result hashes against the analysis manifest,
  2. asserts that every rounded value quoted in Paper 1's Appendix
     app:acefloor text still matches the archived record (so a pipeline
     rerun that changes results fails the paper build loudly),
  3. regenerates Papers/paper1/tables/ace_residual_floors.tex, and
  4. draws the paper-grade floor atlas from atlas.json -> figures/ace_floor_atlas.png,

then records the exporter/manifest/output hashes in
Papers/paper1/ace_floor_export.json.
"""
from pathlib import Path
import json
import shutil

import residual_floor_manuscript as base
from residual_floor_common import SILICA_UM

ROOT = Path(__file__).resolve().parents[2]
DATA = base.DATA
DATA_W01 = ROOT / 'data/ace_floor/w0p1'   # the PRIMARY (0.1-um) archive
PAPER1 = ROOT   # tables/ and figures/ of the repository

MATERIALS = [('Dolomite', 6.4), ('Calcite', 6.9), ('Silica', SILICA_UM),
             ('Dolomite', 11.3), ('Calcite', 11.4), ('Alumina', 12.9)]

CAPTION = (r'Measured residual floors for full $0.1$-$\mu$m elements at '
           r'$19$--$22$~km, using one native tangent spectrum per occultation '
           r'and complete calendar-year holdout. SD and $95\,\%$ full-refit '
           r'year-bootstrap intervals are in $10^{-3}$ residual-ratio OD. The '
           r'five evaluable bands retain all 143 occultations and all 500 '
           r'bootstrap draws. A dash denotes insufficient complete '
           r'predictions, not a zero floor. The last column is the SD of the '
           r'same statistic at the $0.25$-$\mu$m element in which the record '
           r'was first examined (Sect.~\ref{app:acefloor_880}).')


def check(label, actual, expected):
    if actual != expected:
        raise ValueError(f'Appendix quote drifted from archive: {label}: '
                         f'archive {actual} != appendix {expected}')


def verify_appendix_quotes(atlas, silica, sens, reductions):
    """Rounded values of the PRIMARY (0.1-um) analysis quoted in
    sections/appendix_ace_floor.tex."""
    core = {round(r['center_um'], 3): r for r in atlas['19_22']}
    check('silica SD', base.fmt(core[SILICA_UM]['sd']), '1.86')
    check('silica CI', base.ci_text(core[SILICA_UM]['sd_bootstrap_95']),
          '$[1.63, 2.26]$')
    check('silica draws', core[SILICA_UM]['sd_bootstrap_95']['valid_draws'], 500)
    check('6.9 SD', base.fmt(core[6.9]['sd']), '3.21')
    check('11.3 SD', base.fmt(core[11.3]['sd']), '3.13')
    check('11.4 SD', base.fmt(core[11.4]['sd']), '2.99')
    check('12.9 SD', base.fmt(core[12.9]['sd']), '6.28')
    check('8.35 SD', base.fmt(core[8.35]['sd']), '1.52')
    check('8.45 SD', base.fmt(core[8.45]['sd']), '3.79')
    band = [r['sd'] for r in atlas['19_22'] if r.get('sd') is not None
            and 8.0 <= r['center_um'] <= 13.3 and not 9.3 < r['center_um'] < 10.0]
    check('band floor min', f"{1000 * min(band):.1f}", '1.5')
    check('band floor max', f"{1000 * max(band):.1f}", '11.9')
    up = {round(r['center_um'], 3): r for r in atlas['22_25']}
    lo = {round(r['center_um'], 3): r for r in atlas['16_19']}
    check('22-25 silica SD', base.fmt(up[SILICA_UM]['sd']), '1.24')
    check('22-25 silica n', up[SILICA_UM]['n'], 142)
    check('16-19 silica SD', base.fmt(lo[SILICA_UM]['sd']), '6.32')
    check('16-19 silica n', lo[SILICA_UM]['n'], 115)
    check('16-19 silica draws', lo[SILICA_UM]['sd_bootstrap_95']['valid_draws'], 488)
    check('22-25 6.4 SD', base.fmt(up[6.4]['sd']), '5.08')
    check('22-25 6.4 n', up[6.4]['n'], 124)
    # Headline extras quoted in the appendix body.
    check('mean error', f"{silica['mean']:.1e}", '3.9e-06')
    cov = silica['covariance']
    check('linear band SD', base.fmt(cov['linear_band_sd']), '1.86')
    check('diagonal-only SD',
          f"{1000 * cov['diagonal_only_sd']:.3f}", '0.785')
    avg = silica['averaging_fixed_full_element_holdout']
    check('0.08 um SD', base.fmt(avg['0.08']['exact']['sd']), '1.90')
    check('0.04 um SD', base.fmt(avg['0.04']['exact']['sd']), '2.01')
    pairs = silica['matched_pairs']
    check('pairs standard', base.fmt(pairs['standard']['30']['rms']), '1.73')
    check('pairs cross-year', base.fmt(pairs['cross_year']['30']['rms']),
          '2.14')
    check('pairs blue', base.fmt(pairs['split_blue']['30']['rms']), '2.70')
    check('pairs red', base.fmt(pairs['split_red']['30']['rms']), '1.88')
    check('pairs weak', base.fmt(pairs['weak_windows']['30']['rms']), '10.94')
    check('rank3 silica', base.fmt(sens[f'sensitivity_{SILICA_UM:g}um_rank3']['sd']), '1.31')
    check('rank3 paired interval', base.ci_text(reductions[f'{SILICA_UM:g}_rank3']), '$[0.14, 0.78]$')
    check('rank5 silica', base.fmt(sens[f'sensitivity_{SILICA_UM:g}um_rank5']['sd']), '1.64')
    check('guard0 silica', base.fmt(sens[f'sensitivity_{SILICA_UM:g}um_guard0']['sd']), '1.80')
    check('guard8 silica', base.fmt(sens[f'sensitivity_{SILICA_UM:g}um_guard8']['sd']), '1.92')


def verify_comparison_quotes(atlas25, silica25, sens25):
    """Rounded values of the 0.25-um analysis kept for comparison."""
    core = {r['center_um']: r for r in atlas25['19_22']}
    check('0.25 um silica SD', base.fmt(core[SILICA_UM]['sd']), '1.69')
    check('0.25 um silica CI', base.ci_text(core[SILICA_UM]['sd_bootstrap_95']),
          '$[1.49, 2.12]$')
    up = {r['center_um']: r for r in atlas25['22_25']}
    lo = {r['center_um']: r for r in atlas25['16_19']}
    check('0.25 um 22-25 silica SD', base.fmt(up[SILICA_UM]['sd']), '1.28')
    check('0.25 um 16-19 silica SD', base.fmt(lo[SILICA_UM]['sd']), '6.23')
    cov = silica25['covariance']
    check('0.25 um diagonal-only SD',
          f"{1000 * cov['diagonal_only_sd']:.3f}", '0.619')
    avg = silica25['averaging_fixed_full_element_holdout']['0.1']['exact']
    check('0.25 um record, 0.1 averaging', base.fmt(avg['sd']), '1.94')
    check('0.25 um height average', base.fmt(sens25['height_average']['sd']), '1.72')


def paper_atlas_figure(atlas, out):
    """Round 44 (Doron): paper-grade version of the archived floor_atlas.png,
    drawn from atlas.json (the archive figure itself is untouched): black SD
    curve with gray bootstrap shading, red observed-element counts, serif
    fonts at the text size, no title."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.serif': ['STIXGeneral'],
        'font.size': 11, 'axes.labelsize': 11, 'xtick.labelsize': 10,
        'ytick.labelsize': 10, 'legend.fontsize': 9, 'axes.linewidth': 0.8})
    rows = sorted((r for r in atlas['19_22'] if 'n_sample' in r), key=lambda r: r['center_um'])
    x = np.array([r['center_um'] for r in rows])
    est = lambda r, k: r[k] if r['status'] == 'estimated' else np.nan
    y = np.array([est(r, 'sd') for r in rows])
    lo = np.array([r['sd_bootstrap_95']['lower'] if r['status'] == 'estimated' else np.nan for r in rows])
    hi = np.array([r['sd_bootstrap_95']['upper'] if r['status'] == 'estimated' else np.nan for r in rows])
    diag = np.array([r['covariance'].get('diagonal_only_sd', np.nan) if r['status'] == 'estimated' else np.nan for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(12 / 2.54 * 1.5, 8.5 / 2.54 * 1.5), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.1]})
    ax = axes[0]
    ax.fill_between(x, lo, hi, color='0.75', alpha=0.6, lw=0, label='95 % year bootstrap, full refitting')
    ax.plot(x, y, '-', color='black', lw=1.3, label='held-year prediction SD')
    ax.plot(x, diag, ':', color='black', lw=1.1, label='diagonal covariance only')
    for c in [6.4, 6.9, SILICA_UM, 11.3, 11.4, 12.9]:
        ax.axvline(c, color='0.5', alpha=0.6, lw=0.7)
    ax.set(yscale='log', ylabel='Residual-ratio OD scatter')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(True, which='both', alpha=0.25, lw=0.5)
    ax2 = axes[1]
    ax2.plot(x, [r['n'] for r in rows], '-', color='black', lw=1.3, label='predicted complete elements')
    ax2.plot(x, [r['n_full_valid_elements'] for r in rows], ':', color='tab:red', lw=1.4, label='observed complete elements')
    ax2.set(xlabel=r'Wavelength [$\mu$m]', ylabel='Occultations', ylim=(0, 150))
    ax2.legend(loc='lower right', frameon=False)
    ax2.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    for data in (DATA, DATA_W01):
        manifest = json.loads((data / 'manifest.json').read_text())
        skipped = 0
        for name, expected in manifest['outputs'].items():
            if not (data / name).exists():          # the shipped archive omits the bulky
                skipped += 1; continue              # *_pair_bootstrap.csv / *_pairs_*.csv tables
            if base.digest(data / name) != expected:
                raise ValueError('Result hash mismatch: ' + str(data / name))
        print(f'{data.name}: manifest hashes verified for every shipped file ({skipped} bulky files not shipped)')
    read01 = lambda name: json.loads((DATA_W01 / name).read_text())
    atlas = read01('atlas.json')
    silica = read01(f'19_22_{SILICA_UM:g}um.json')
    verify_appendix_quotes(atlas, silica, read01('sensitivity.json'), read01('paired_sd_reductions.json'))
    atlas25 = base.read('atlas.json')
    verify_comparison_quotes(atlas25, base.read(f'19_22_{SILICA_UM:g}um.json'), base.read('sensitivity.json'))

    core = {round(r['center_um'], 3): r for r in atlas['19_22']}
    core25 = {round(r['center_um'], 3): r for r in atlas25['19_22']}
    rows = []
    for material, c in MATERIALS:
        r = core.get(c, {})
        d = core25.get(c, {})
        rows.append([material, f'{c:.2f}' if c == SILICA_UM else f'{c:g}', str(r.get('n', 0)), base.fmt(r.get('sd')),
                     base.ci_text(r['sd_bootstrap_95']) if r.get('sd') is not None else '---',
                     base.fmt(d.get('sd'))])
    body = '\n'.join(' & '.join(r) + r' \\' for r in rows)
    table = (
        '% Generated by residual_floor_paper1.py from the verified\n'
        '% residual-floor analysis record; edit the exporter, not this file.\n'
        r'\begin{table}[t]' + '\n' +
        r'\caption{' + CAPTION + '}\n' +
        r'\label{tab:ace_floors}' + '\n' +
        r'\centering' + '\n' +
        r'\begin{tabular}{llrrrr}' + '\n' + r'\toprule' + '\n' +
        r'Motivating material & Center ($\mu$m) & $n$ & SD & '
        r'$95\,\%$ interval & SD, $0.25~\mu$m \\' + '\n' + r'\midrule' + '\n' +
        body + '\n' + r'\bottomrule' + '\n' +
        r'\end{tabular}' + '\n' + r'\end{table}' + '\n')
    (PAPER1 / 'tables').mkdir(exist_ok=True)
    (PAPER1 / 'tables/ace_residual_floors.tex').write_text(table)

    paper_atlas_figure(atlas, PAPER1 / 'figures/ace_floor_atlas.png')

    outputs = [PAPER1 / 'tables/ace_residual_floors.tex',
               PAPER1 / 'figures/ace_floor_atlas.png']
    record = {'analysis_manifest_sha256': base.digest(DATA_W01 / 'manifest.json'),
              'comparison_0p25um_manifest_sha256': base.digest(DATA / 'manifest.json'),
              'exporter_sha256': base.digest(Path(__file__)),
              'generated_files': {str(p.relative_to(PAPER1)): base.digest(p)
                                  for p in outputs}}
    (PAPER1 / 'ace_floor_export.json').write_text(
        json.dumps(record, indent=2) + '\n')
    print('Paper 1: verified appendix quotes; exported the residual-floor '
          'table and figure.')


if __name__ == '__main__':
    main()
