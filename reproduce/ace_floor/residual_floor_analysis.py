"""Reproduce residual-floor atlas, full-pipeline bootstrap and matching checks.

Run from any directory. All writes are confined to --output (default side study).
No paper source, figure, PDF, or original revision result is changed.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import subprocess
import csv
import platform
import sys
import time
import zipfile
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from residual_floor_common import *


def csv_write(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def interval(values):
    v = np.asarray(values); v = v[np.isfinite(v)]
    return dict(valid_draws=len(v),lower=np.percentile(v,2.5) if len(v) else np.nan,
                upper=np.percentile(v,97.5) if len(v) else np.nan)


def averaging(engine, result, widths):
    answer = {}
    for width in widths:
        w = response(engine.center,width)[engine.cols][engine.target]
        w /= w.sum()
        exact = band_od(result['actual'],w)-band_od(result['pred'],w)
        linear = result['residual']@w
        answer[str(width)] = dict(exact=statistics(exact,result['weights']),
                                  linear=statistics(linear,result['weights']),bins=int((w>0).sum()))
    return answer


def diagnostics(engine, r):
    ok = np.isfinite(r['error']); vec = r['residual'][ok]
    w = engine.w[engine.cols][engine.target]
    if len(vec)<2: return {},None
    cov = np.atleast_2d(np.cov(vec,rowvar=False))
    full = np.sqrt(max(0,w@cov@w)); diagonal = np.sqrt(np.sum(w*w*np.diag(cov)))
    return dict(linear_band_sd=full,diagonal_only_sd=diagonal,
        rms_single_bin_sd=np.sqrt(np.mean(np.diag(cov))),
        correlation_inflation=full/diagonal if diagonal else np.nan),cov


def qualified_stats(engine, r, cfg):
    s = statistics(r['error'],r['weights'])
    active = np.isfinite(r['error']) & (r['weights']>0)
    ny = len(np.unique(engine.gid[active]));s['n_years']=ny
    good = s['n_unique'] >= cfg['minimum_evaluation_occultations'] and ny>=cfg['minimum_evaluation_years']
    s['status'] = 'estimated' if good else 'insufficient_complete_elements'
    return s


def pair_curve(pairs, counts):
    out = {}
    for count in counts:
        p = pairs[:count]
        out[str(count)] = dict(n_pairs=len(p),rms=np.sqrt(np.mean([a['difference']**2 for a in p])) if len(p)==count else np.nan,
            max_proxy_distance=p[-1]['distance'] if p else np.nan,
            median_height_separation=np.median([a['delta_height'] for a in p]) if p else np.nan,
            median_year_separation=np.median([a['delta_year'] for a in p]) if p else np.nan)
    return out


PAIR_VARIANTS = {
    'standard':{}, 'cross_year':{'relation':'cross_year'},
    'same_year':{'relation':'same_year'},
    'same_direction':{'relation':'same_direction'},
    'height_025':{'max_height':.25},
    'latitude_5':{'max_latitude':5.},
    'split_blue':{'variant':'split_blue'}, 'split_red':{'variant':'split_red'},
    'weak_windows':{'variant':'weak'}
}


def run_band(data, center, cfg, out, tag, boots, details=False, **kwargs):
    w = response(center,cfg['width_um'])
    if w is None: return dict(center_um=center,status='outside_ACE_coverage')
    engine = Engine(data,center,cfg['width_um'],cfg['guard_cm_inverse'],**kwargs)
    r = engine.evaluate(); summary = qualified_stats(engine,r,cfg)
    summary.update(center_um=center,width_um=cfg['width_um'],n_sample=len(data['M']),
        n_full_valid_elements=int(np.isfinite(data['M'][:,w>0]).all(1).sum()),
        n_target_bins=int((w>0).sum()),
        fit_bins_range=[int(r['fit_bins'].min()),int(r['fit_bins'].max())])
    covdiag,cov = diagnostics(engine,r); summary['covariance']=covdiag
    if details:
        summary['averaging_fixed_full_element_holdout']=averaging(engine,r,cfg['averaging_widths_um'])
    good = np.isfinite(r['error'])
    summary['background_exceedances']={str(t):dict(count=int(np.sum(np.abs(r['error'][good])>t)),n=int(good.sum())) for t in cfg['diagnostic_thresholds_od']}
    summary['year_means']={str(g):statistics(r['error'][engine.gid==k]) for k,g in enumerate(engine.groups)}
    csv_write(out/f'{tag}_scores.csv',[dict(occultation=o,decimal_year=y,height_km=h,
        error_od=e if np.isfinite(e) else '',fit_bins=int(n)) for o,y,h,e,n in zip(data['occs'],data['years'],data['heights'],r['error'],r['fit_bins'])])
    if details and cov is not None:
        np.savetxt(out/f'{tag}_covariance.csv',cov,delimiter=',',header=','.join(f'{x:g}' for x in GRID[w>0]),comments='')
    target = band_od(data['M'],w)
    variants = PAIR_VARIANTS if details else {'standard':{}}
    pair_results = {}
    for name,options in variants.items():
        p = matches(data,engine.excluded,target,**options)
        pair_results[name] = pair_curve(p,cfg['pair_prefix_counts'])
        if details:
            validation = proxies(data['M'],engine.excluded,'weak')
            scale = np.nanstd(validation,axis=0)
            for a in p:
                i,j = a['i'],a['j']
                a.update(occultation_i=data['occs'][i],occultation_j=data['occs'][j],
                         weak_window_validation_distance=np.linalg.norm((validation[i]-validation[j])/scale))
            csv_write(out/f'{tag}_pairs_{name}.csv',p)
    summary['matched_pairs']=pair_results
    # All draws use the same year multiplicities across bands and alternatives.
    # Draw-specific masks and modes are recomputed, with all copies of each
    # test year removed together. Matching independently reruns in each draw.
    bootrows=[]; pairboot={k:{str(n):[] for n in cfg['pair_prefix_counts']} for k in variants}
    avgboot={str(x):[] for x in cfg['averaging_widths_um']} if details else {}
    for b,weights in enumerate(boots):
        rb = engine.evaluate(weights); s = qualified_stats(engine,rb,cfg)
        bootrows.append(dict(draw=b,sd=s['sd'] if s['status']=='estimated' else np.nan,
            mean=s['mean'],rms=s['rms'],n=s['n'],n_unique=s['n_unique'],n_years=s['n_years']))
        if details:
            a = averaging(engine,rb,cfg['averaging_widths_um'])
            for x in avgboot: avgboot[x].append(a[x]['exact']['sd'] if s['status']=='estimated' else np.nan)
        for name,options in variants.items():
            pairs = matches(data,engine.excluded,target,weights[engine.gid],**options)
            curves = pair_curve(pairs,cfg['pair_prefix_counts'])
            for n in pairboot[name]:pairboot[name][n].append(curves[n]['rms'])
    summary['sd_bootstrap_95']=interval([r['sd'] for r in bootrows])
    summary['mean_bootstrap_95']=interval([r['mean'] if np.isfinite(r['sd']) else np.nan for r in bootrows])
    for name in pairboot:
        for n in pairboot[name]:summary['matched_pairs'][name][n]['rms_bootstrap_95']=interval(pairboot[name][n])
    for x in avgboot:summary['averaging_fixed_full_element_holdout'][x]['sd_bootstrap_95']=interval(avgboot[x])
    csv_write(out/f'{tag}_bootstrap.csv',bootrows)
    # Save replicate pairing statistics too, so intervals are auditable.
    csv_write(out/f'{tag}_pair_bootstrap.csv',[dict(draw=b,variant=k,prefix=int(n),rms=v)
        for k in pairboot for n in pairboot[k] for b,v in enumerate(pairboot[k][n])])
    dump(out/f'{tag}.json',summary)
    return summary


def figures(out, atlas, material):
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,gridspec_kw={'height_ratios':[3,1]})
    ordered=sorted((s for s in atlas if 'n_sample' in s),key=lambda s:s['center_um'])
    x=np.array([s['center_um'] for s in ordered]);ok=np.array([s['status']=='estimated' for s in ordered])
    y=np.array([s['sd'] if s['status']=='estimated' else np.nan for s in ordered])
    axes[0].plot(x,y,'o-',ms=3,label='Held-year prediction SD')
    lo=np.array([s['sd_bootstrap_95']['lower'] if s['status']=='estimated' else np.nan for s in ordered])
    hi=np.array([s['sd_bootstrap_95']['upper'] if s['status']=='estimated' else np.nan for s in ordered])
    axes[0].fill_between(x,lo,hi,alpha=.2,label='95% year bootstrap, full refitting')
    axes[0].plot(x,[s['covariance'].get('diagonal_only_sd',np.nan) if s['status']=='estimated' else np.nan for s in ordered],':',label='Diagonal covariance only')
    for name,bands in material.items():
        for center in bands:
            if 2.2<center<13.3:axes[0].axvline(center,color='gray',alpha=.25,lw=.8)
    axes[0].set(yscale='log',ylabel='Residual-ratio OD scatter',title=f'ACE residuals: {next(r["width_um"] for r in atlas if "width_um" in r):g} µm elements, one native height at 19–22 km')
    axes[0].legend(fontsize=8)
    axes[1].plot(x,[s['n'] for s in ordered],'o-',ms=3,label='Predicted complete elements')
    axes[1].plot(x,[s['n_full_valid_elements'] for s in ordered],':',label='Observed complete elements')
    axes[1].set(xlabel='Wavelength (µm)',ylabel='Occultations',ylim=(0,150));axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'floor_atlas.png',dpi=180);plt.close(fig)
    focus=next(s for s in atlas if s['center_um']==SILICA_UM)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    a=focus['averaging_fixed_full_element_holdout'];xx=np.array([float(x) for x in a]);yy=np.array([a[x]['exact']['sd'] for x in a])
    axes[0].plot(xx,yy*1e3,'o-');axes[0].set(xlabel='Element width (µm)',ylabel='Band OD SD × 1000',title=f'{SILICA_UM:g} µm: fixed 0.25 µm exclusion')
    for k in ['standard','cross_year','same_year','split_blue','split_red','weak_windows']:
        p=focus['matched_pairs'][k];axes[1].plot([int(n) for n in p],[p[n]['rms']*1e3 for n in p],'o-',ms=3,label=k.replace('_',' '))
    axes[1].axhline(focus['sd']*1e3,color='k',ls=':',label='Prediction SD')
    axes[1].set(xlabel='Closest disjoint pairs retained',ylabel='RMS(ΔOD / √2) × 1000',title='Matching assumptions');axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'averaging_and_matching.png',dpi=180);plt.close(fig)


def report(out, results, cfg):
    lines=['# ACE residual-floor results','',
      'Generated by `residual_floor_analysis.py`. Units are residual-ratio slant OD. '
      'These are conditional background-prediction scatters, not an identified pure '
      'trace-gas or instrumental systematic floor. See `../../RESIDUAL_FLOOR.md` for definitions.',
      '', '| Material | Center (µm) | Complete predicted / sample | SD | Full-refit year-bootstrap 95% | Status |',
      '|---|---:|---:|---:|---:|---|']
    atlas=results['19_22'];lookup={s['center_um']:s for s in atlas}
    for material,bands in cfg['material_bands_um'].items():
        for c in bands:
            s=lookup[c]
            if s['status']!='estimated':
                lines.append(f"| {material} | {c:g} | {s.get('n',0)} / {s.get('n_sample','—')} | — | — | {s['status']} |")
            else:
                ci=s['sd_bootstrap_95'];lines.append(f"| {material} | {c:g} | {s['n']} / {s['n_sample']} | {s['sd']:.4g} | [{ci['lower']:.4g}, {ci['upper']:.4g}] | estimated |")
    lines += ['', 'All in-coverage material bands and control bands are repeated at 16–19 and 22–25 km. '
        'The uniform scan is at 19–22 km. Each JSON reports full-element availability, '
        'year means, threshold exceedances, covariance, and matching checks. '
        'Missing target bins are never replaced by a quiet subset.', '',
        f'Intervals are pointwise percentile intervals from {cfg["bootstrap_draws"]} calendar-year resamples '
        '(the actual count is recorded in config/manifest). Resampled observations '
        'remain clustered by original year during validation. Bootstrap copies cannot '
        'match themselves. Greedy matching is nonsmooth; its intervals are descriptive '
        'resampling sensitivity, not guaranteed nominal coverage.', '',
        'The 0.25 µm response averages the public gas-divided residual ratio in '
        'wavelength, then takes −log. It is not the ratio of instrument-binned observed '
        'and gas-only transmissions, which cannot be reconstructed from these data.', '',
        'A common additive error, or an error lying in the fitted background span, '
        'can remain invisible to both methods. Scatter includes real background '
        'variation not captured by the predictor. Thus neither a strict upper nor '
        'a strict lower bound on the mission systematic floor follows.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=OUT/'config.json')
    parser.add_argument('--sample',type=Path,default=OUT/'sample.csv')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--bootstrap',type=int,help='Override draws (use separate output for smoke runs)')
    parser.add_argument('--smoke',action='store_true',help='Only silica, one altitude; use separate output')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.smoke and out==OUT.resolve():parser.error('--smoke requires a separate --output')
    cfg=json.loads(args.config.read_text())
    if args.bootstrap is not None:cfg['bootstrap_draws']=args.bootstrap
    if cfg['bootstrap_draws']<2:parser.error('at least two bootstrap draws required')
    start=time.time()
    check=subprocess.run([sys.executable,str(HERE/'test_residual_floor.py')],capture_output=True,text=True)
    (out/'tests.txt').write_text(check.stdout+check.stderr)
    if check.returncode:raise RuntimeError('Preflight tests failed; see tests.txt')
    print('Tests passed; building matrices from native residual files',flush=True)
    datasets,hashes,selected=build_data(args.sample,[[19,22]] if args.smoke else cfg['altitude_ranges_km'])
    dump(out/'selection.json',selected);dump(out/'run_config.json',cfg)
    results={};material=sorted(set(c for bands in cfg['material_bands_um'].values() for c in bands))
    special=sorted(set(material+cfg['control_bands_um']))
    scan=cfg['scan_centers_um'];centers=sorted(set(special+list(np.arange(scan['start'],scan['stop']+.001,scan['step']))))
    for altitude,data in datasets.items():
        np.savez_compressed(out/f'matrix_{altitude}.npz',**data)
        ng=len(np.unique(np.floor(data['years'])));rng=np.random.default_rng(cfg['seed'])
        boots=rng.multinomial(ng,np.full(ng,1/ng),size=cfg['bootstrap_draws'])
        np.savetxt(out/f'year_multiplicities_{altitude}.csv',boots,fmt='%d',delimiter=',',
                   header=','.join(str(int(x)) for x in np.unique(np.floor(data['years']))),comments='')
        results[altitude]=[]
        for c in ([SILICA_UM] if args.smoke else centers if altitude=='19_22' else special):
            tag=f'{altitude}_{c:g}um'
            print(f'{tag}: {cfg["bootstrap_draws"]} full-pipeline year resamples',flush=True)
            s=run_band(data,c,cfg,out,tag,boots,details=c in special and response(c) is not None)
            results[altitude].append(s)
        if altitude=='19_22':
            sensitivity={}
            variants={'rank3':dict(model='rank3'),'rank5':dict(model='rank5'),
                      'availability_mask':dict(mask='availability')}
            for c in ([SILICA_UM] if args.smoke else [x for x in material if response(x) is not None]):
                for name,options in variants.items():
                    tag=f'sensitivity_{c:g}um_{name}';print(tag,flush=True)
                    sensitivity[tag]=run_band(data,c,cfg,out,tag,boots,**options)
            for guard in ([0] if args.smoke else [0,2,8]):
                conf=dict(cfg,guard_cm_inverse=guard);tag=f'sensitivity_{SILICA_UM:g}um_guard{guard}'
                print(tag,flush=True);sensitivity[tag]=run_band(data,SILICA_UM,conf,out,tag,boots)
            avg=dict(data,M=data['averaged']);print('height-average sensitivity',flush=True)
            sensitivity['height_average']=run_band(avg,SILICA_UM,cfg,out,f'sensitivity_{SILICA_UM:g}um_height_average',boots)
            dump(out/'sensitivity.json',sensitivity)
            # Paired model comparison: the same bootstrap draw in both fits.
            diffs={}
            for c in ([SILICA_UM] if args.smoke else [x for x in material if response(x) is not None]):
                base=list(csv.DictReader((out/f'19_22_{c:g}um_bootstrap.csv').open()))
                for name in variants:
                    alt=list(csv.DictReader((out/f'sensitivity_{c:g}um_{name}_bootstrap.csv').open()))
                    diffs[f'{c:g}_{name}']=interval([float(a['sd'])-float(b['sd']) for a,b in zip(base,alt)])
            dump(out/'paired_sd_reductions.json',diffs)
    dump(out/'atlas.json',results)
    if not args.smoke:figures(out,results['19_22'],cfg['material_bands_um']);report(out,results,cfg)
    sources=list(HERE.glob('residual_floor*.py'))+list(HERE.glob('test_residual_floor.py'))
    sources += [HERE/'revision_common.py',args.sample,args.config]
    if (HERE/'RESIDUAL_FLOOR.md').exists():sources.append(HERE/'RESIDUAL_FLOOR.md')
    manifest=dict(config=cfg,smoke=args.smoke,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
        matplotlib=matplotlib.__version__,platform=platform.platform(),elapsed_seconds=time.time()-start,
        source_hashes={str(Path(p).resolve().relative_to(ROOT)):sha(p) for p in sources},raw_input_hashes=hashes,
        outputs={p.name:sha(p) for p in sorted(out.iterdir()) if p.suffix in ['.csv','.json','.md','.png','.txt'] and p.name!='manifest.json'},
        command=' '.join(sys.argv),raw_data_redistributed=False)
    dump(out/'manifest.json',manifest)
    with zipfile.ZipFile(out/'residual_floor_code.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,str(Path(p).resolve().relative_to(ROOT)))
        for p in sorted(out.iterdir()):
            if p.suffix in ['.csv','.json','.md','.png','.txt']:z.write(p,'results/'+p.name)
    print(f'Finished in {time.time()-start:.1f}s: {out}',flush=True)


if __name__=='__main__':main()
