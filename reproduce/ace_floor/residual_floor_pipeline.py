"""Audited entry point: metadata checks, then the reproducible floor analysis.

Requires a local snapshot of the ACE issues CSV, obtainable with the same
registered ACE access as the spectra. It is never packaged or redistributed.
"""
import argparse
import csv
import io
import json
import zipfile
import subprocess
import sys
from pathlib import Path
from residual_floor_common import ROOT, HERE, OUT, dump, sha, SILICA_UM


def audit(sample, output, issues):
    sample_rows=list(csv.DictReader(Path(sample).open()))
    ids={r['occultation'] for r in sample_rows}
    if len(ids)!=len(sample_rows):raise ValueError('Duplicate sample occultation')
    geo_path=ROOT/'data/ace/v52_subset/occultationlist.csv'
    geo={r['occultation_name']:r for r in csv.DictReader(geo_path.open())}
    for r in sample_rows:
        if int(float(r['decimal_year']))!=int(geo[r['occultation']]['occultation_datetime'][:4]):
            raise ValueError('Calendar-year mismatch for '+r['occultation'])
        if geo[r['occultation']]['occultation_datetime'][:10]<'2004-02-21':
            raise ValueError('Sample contains commissioning observation')
    rows=list(csv.DictReader(io.StringIO(Path(issues).read_text().lstrip())))
    if not rows or 'occultation_name' not in rows[0]:raise ValueError('Invalid ACE issues CSV')
    matches=[{k:r[k] for k in ['occultation_name','data_version','category','status']}
             for r in rows if r['occultation_name'] in ids]
    # Historical version-specific notices are retained as audit metadata.
    # A matching current/unspecified-version notice needs scientific review;
    # the pipeline must not silently declare that observation clean.
    historical={'2.2','3.0','3.5/3.6'}
    unresolved=[r for r in matches if not (r['data_version'] in historical
        and r['category'] in {'n2o','ascii/NaN'})]
    result=dict(sample_size=len(ids),sample_sha256=sha(sample),
        source='https://databace.scisat.ca/validation/data_issues.csv',
        input_path=str(Path(issues).resolve()),input_sha256=sha(issues),
        n_issue_rows=len(rows),matching_notices=matches,
        unresolved_current_or_unspecified_version_notices=unresolved,
        calendar_years_verified=True,commissioning_observations=0,
        interpretation='Historical version-specific notices do not establish a v5.2 problem; no observations excluded by this audit.')
    Path(output).mkdir(parents=True,exist_ok=True)
    dump(Path(output)/'quality_audit.json',result)
    if unresolved:raise ValueError('ACE issue notices need review; see quality_audit.json')
    return result


def finish(output):
    """Summarize model/matching checks, validate outputs, refresh the archive."""
    output=Path(output)
    atlas=json.loads((output/'atlas.json').read_text())
    sensitivity=json.loads((output/'sensitivity.json').read_text())
    reductions=json.loads((output/'paired_sd_reductions.json').read_text())
    primary={r['center_um']:r for r in atlas['19_22']}
    focus=primary[SILICA_UM]
    def value(x):return '—' if x is None else f'{x:.4g}'
    lines=['', '## Model and altitude checks', '',
        '| Center (µm) | Rank 2 SD | Rank 3 SD | Rank 5 SD | Availability-only mask SD | Rank 2 minus rank 3, bootstrap 95% |',
        '|---:|---:|---:|---:|---:|---:|']
    for c in [6.4,6.9,SILICA_UM,11.3,11.4,12.9]:
        if c not in primary:continue
        values=[primary[c].get('sd')]+[sensitivity.get(f'sensitivity_{c:g}um_{m}',{}).get('sd')
             for m in ['rank3','rank5','availability_mask']]
        delta=reductions.get(f'{c:g}_rank3',{})
        lines.append(f"| {c:g} | "+' | '.join(value(x) for x in values)+
             f" | [{value(delta.get('lower'))}, {value(delta.get('upper'))}] |")
    lines += ['', 'Alternatives are sensitivity tests, not a selection of the smallest floor. '
        'Differences use the same year resamples. Consult the JSON counts when '
        'models have different prediction coverage.', '',
        f'| Native-height interval (km) | {SILICA_UM:g} µm SD | Valid / total bootstrap draws |',
        '|---|---:|---:|']
    draws=json.loads((output/'run_config.json').read_text())['bootstrap_draws']
    for altitude,rows in atlas.items():
        row=next(r for r in rows if r['center_um']==SILICA_UM)
        lines.append(f"| {altitude.replace('_','–')} | {value(row['sd'])} | {row['sd_bootstrap_95']['valid_draws']} / {draws} |")
    lines += ['', '| Band (µm) | 16–19 km: SD (n) | 19–22 km: SD (n) | 22–25 km: SD (n) |',
        '|---:|---:|---:|---:|']
    for c in [6.4,6.9,SILICA_UM,11.3,11.4,12.9]:
        cells=[]
        for altitude in ['16_19','19_22','22_25']:
            row=next((r for r in atlas.get(altitude,[]) if r['center_um']==c),{})
            cells.append(f"{value(row.get('sd'))} ({row.get('n',0)})" if row.get('status')=='estimated' else 'unavailable')
        lines.append(f'| {c:g} | '+' | '.join(cells)+' |')
    lines += ['', f"The 19–22 km height-averaged comparison gives SD {value(sensitivity['height_average']['sd'])} OD.",
        '', f'## Spectral averaging and matching at {SILICA_UM:g} µm', '',
        f"The exact 0.25 µm element SD is {value(focus['sd'])} OD. "
        f"The full-covariance linearized SD is {value(focus['covariance']['linear_band_sd'])}, "
        f"compared with {value(focus['covariance']['diagonal_only_sd'])} if off-diagonal terms are discarded.",
        '', '| Width (µm), fixed full-element exclusion | Exact band SD |', '|---:|---:|']
    for width,row in focus['averaging_fixed_full_element_holdout'].items():
        lines.append(f"| {width} | {value(row['exact']['sd'])} |")
    lines += ['', '| Matching rule | RMS(ΔOD/√2), first 30 pairs | Largest proxy distance | 95% resampling range |',
        '|---|---:|---:|---:|']
    for name,curves in focus['matched_pairs'].items():
        row=curves['30'];ci=row['rms_bootstrap_95']
        lines.append(f"| {name.replace('_',' ')} | {value(row['rms'])} | {value(row['max_proxy_distance'])} | [{value(ci['lower'])}, {value(ci['upper'])}] |")
    lines += ['', 'The fixed 30-pair table is a reading aid; all five predeclared pair counts '
        'are retained in the files and figure. Large changes under alternative '
        'proxies can reflect background mismatch as well as correlated errors. '
        'They do not isolate either cause or establish a strict noise bracket.', '',
        'Figures: [wavelength atlas](floor_atlas.png), '
        '[averaging and matching](averaging_and_matching.png).']
    path=output/'RESULTS.md'
    original=path.read_text().split('\n## Model and altitude checks')[0].rstrip()
    path.write_text(original+'\n'+'\n'.join(lines)+'\n')
    # Verify all actual score files against their summary statistics, and the
    # stored covariance against the reported linear variance projection.
    import numpy as np
    from residual_floor_common import response
    checked=0
    for summary_path in sorted(output.glob('*.json')):
        row=json.loads(summary_path.read_text())
        if not isinstance(row,dict) or 'n_sample' not in row:continue
        scores=list(csv.DictReader(summary_path.with_name(summary_path.stem+'_scores.csv').open()))
        errors=np.array([float(r['error_od']) for r in scores if r['error_od']])
        if len(errors)!=row['n']:raise ValueError('Score count mismatch: '+str(summary_path))
        if len(errors)>1 and not np.isclose(errors.std(ddof=1),row['sd'],rtol=1e-12):
            raise ValueError('Score SD mismatch: '+str(summary_path))
        covpath=summary_path.with_name(summary_path.stem+'_covariance.csv')
        if covpath.exists():
            cov=np.loadtxt(covpath,delimiter=',',skiprows=1);w=response(row['center_um'],row['width_um']);w=w[w>0]
            if not np.isclose(np.sqrt(w@cov@w),row['covariance']['linear_band_sd'],rtol=1e-12):
                raise ValueError('Covariance mismatch: '+str(summary_path))
        checked+=1
    manifest_path=output/'manifest.json';manifest=json.loads(manifest_path.read_text())
    for name,digest in manifest['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Source changed since analysis: '+name)
    for name,digest in manifest['outputs'].items():
        if name!='RESULTS.md' and sha(output/name)!=digest:
            raise ValueError('Output changed since analysis: '+name)
    manifest['outputs']['RESULTS.md']=sha(path)
    manifest['verification']=dict(score_tables_checked=checked,full_covariance_projection_checked=True,
        finalizer='residual_floor_pipeline.py',quality_audit_sha256=sha(output/'quality_audit.json'))
    dump(manifest_path,manifest)
    with zipfile.ZipFile(output/'residual_floor_code.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in manifest['source_hashes']:archive.write(ROOT/name,name)
        for path in sorted(output.iterdir()):
            if path.suffix in ['.csv','.json','.md','.png','.txt']:archive.write(path,'results/'+path.name)
    with zipfile.ZipFile(output/'residual_floor_code.zip') as archive:
        if archive.testzip() is not None:raise ValueError('Corrupt archive')
        if any(name.endswith('.npz') or name.endswith('data_issues.csv') for name in archive.namelist()):
            raise ValueError('Raw-data cache leaked into archive')
    print(f'Validated {checked} score tables and archive; report: {output / "RESULTS.md"}')


def main():
    p=argparse.ArgumentParser(description=__doc__,add_help=False)
    p.add_argument('--sample',type=Path,default=OUT/'sample.csv')
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--issues',type=Path,default=ROOT/'data/ace/v52_subset/data_issues.csv')
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--finish-only',action='store_true')
    args,rest=p.parse_known_args()
    if '-h' in rest or '--help' in rest:
        p.print_help();subprocess.run([sys.executable,str(HERE/'residual_floor_analysis.py'),'--help'],check=True);return
    if Path(args.issues).exists(): audit(args.sample,args.output,args.issues)
    else: print(f'NOTE: ACE issues CSV not found at {args.issues}; the metadata audit is skipped (see fetch/README.md)')
    if args.finish_only:
        finish(args.output);return
    if not args.audit_only:
        subprocess.run([sys.executable,str(HERE/'residual_floor_analysis.py'),
                        '--sample',str(args.sample),'--output',str(args.output),*rest],check=True)
        if '--smoke' not in rest:finish(args.output)


if __name__=='__main__':main()
