"""Write the manifest and code archive for a completed residual-floor analysis
whose final bookkeeping step failed (round 43, 2026-09-25: the analysis was
launched with relative --sample/--config paths and the manifest step raised
in Path.relative_to after every result file had been written).  Recomputes
the raw-input hashes exactly as residual_floor_analysis.py does, hashes the
sources and outputs, and writes manifest.json + residual_floor_code.zip; run
residual_floor_pipeline.py --finish-only afterwards for the audit/verification.

Usage: python scripts/residual_floor_finalize.py --output data/residual_floor
       [--sample data/residual_floor/sample.csv] [--config <output>/config.json]
"""
import argparse, json, platform, sys, time, zipfile
from pathlib import Path
import numpy as np, scipy, matplotlib
from residual_floor_common import ROOT, HERE, OUT, dump, sha, build_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--sample', type=Path, default=OUT / 'sample.csv')
    p.add_argument('--config', type=Path, default=None)
    a = p.parse_args()
    out = a.output.resolve(); sample = a.sample.resolve()
    config = (a.config or out / 'config.json').resolve()
    cfg = json.loads((out / 'run_config.json').read_text())
    start = time.time()
    _, hashes, _ = build_data(sample, cfg['altitude_ranges_km'])
    sources = list(HERE.glob('residual_floor*.py')) + list(HERE.glob('test_residual_floor.py'))
    sources += [HERE / 'revision_common.py', sample, config]
    if (HERE / 'RESIDUAL_FLOOR.md').exists(): sources.append(HERE / 'RESIDUAL_FLOOR.md')
    rel = lambda q: str(Path(q).resolve().relative_to(ROOT))
    manifest = dict(config=cfg, smoke=False, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        matplotlib=matplotlib.__version__, platform=platform.platform(), elapsed_seconds=None,
        finalized_by='residual_floor_finalize.py (manifest step re-run after a path error; all result files from the original analysis run)',
        source_hashes={rel(q): sha(q) for q in sources}, raw_input_hashes=hashes,
        outputs={q.name: sha(q) for q in sorted(out.iterdir()) if q.suffix in ['.csv', '.json', '.md', '.png', '.txt'] and q.name != 'manifest.json'},
        command=' '.join(sys.argv), raw_data_redistributed=False)
    dump(out / 'manifest.json', manifest)
    with zipfile.ZipFile(out / 'residual_floor_code.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for q in sources: z.write(q, rel(q))
        for q in sorted(out.iterdir()):
            if q.suffix in ['.csv', '.json', '.md', '.png', '.txt']: z.write(q, 'results/' + q.name)
    print(f'Manifest written in {time.time()-start:.0f}s: {out / "manifest.json"} ({len(manifest["outputs"])} outputs)')


if __name__ == '__main__': main()
