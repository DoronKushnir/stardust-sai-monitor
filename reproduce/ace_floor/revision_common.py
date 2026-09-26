"""Canonical September-2026 revision helpers; imports have no side effects."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]   # repository root
sys.path.insert(0, str(ROOT))
HERE = ROOT / 'reproduce/ace_floor'
OUT = ROOT / 'data/ace/revision20260909'
FIG = ROOT / 'figures'
GRID = np.arange(750., 4546., 2.)
WL = 1e4 / GRID
PRIM = GRID <= 1600
EL = (GRID >= 1128.5) & (GRID <= 1160.8)
CORE = (GRID >= 3300) & (GRID <= 3400)
X1 = (WL >= 8) & (WL <= 8.9)
X2 = (WL >= 10.5) & (WL <= 12.2)
SEED = 20260909


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save_json(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    def serializable(value):
        if isinstance(value, dict):
            return {k: serializable(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [serializable(v) for v in value]
        if isinstance(value, np.ndarray):
            return serializable(value.tolist())
        if isinstance(value, np.generic):
            return serializable(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    (OUT / name).write_text(json.dumps(serializable(obj), indent=2, allow_nan=False)+'\n')


def clean_transmission(raw, grid=GRID):
    """Interpolate only when BOTH bracketing native samples are valid.

    No bridging across saturation, invalid values, or out-of-coverage edges.
    """
    nu, t = raw[:, 0], raw[:, 1]
    order = np.argsort(nu)
    nu, t = nu[order], t[order]
    j = np.searchsorted(nu, grid).clip(1, len(nu)-1)
    valid = ((grid >= nu[0]) & (grid <= nu[-1]) & (t[j-1] > 0)
             & (t[j] > 0) & np.isfinite(t[j-1]) & np.isfinite(t[j]))
    out = np.interp(grid, nu, t)
    out[~valid] = np.nan
    return out


def build_matrix(lo=19., hi=22., source=None, tag=''):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f'matrix_{tag}{lo:g}_{hi:g}.npz'
    if target.exists():
        return dict(np.load(target, allow_pickle=True))
    old = np.load(source or HERE / 'data/fullrange_matrix.npz', allow_pickle=True)
    occs, years, mats, heights = [], [], [], []
    rawhash = hashlib.sha256()
    for occ, yr in zip(old['occs'].astype(str), old['years']):
        rows, zz = [], []
        for f in sorted((ROOT / 'data/ace/v52_subset/residual' / occ).iterdir()):
            try:
                alt = float(f.name.split('.', 1)[1])
            except (ValueError, IndexError):
                continue
            if lo <= alt <= hi:
                rawhash.update(f.name.encode()+f.read_bytes())
                rows.append(clean_transmission(np.loadtxt(f)))
                zz.append(alt)
        if rows:
            # A bin must retain the same tangent spectra as all other bins.
            mats.append(-np.log(np.mean(rows, axis=0)))
            occs.append(occ); years.append(yr); heights.append(zz)
    M = np.array(mats)
    out = dict(M=M, years=np.array(years), occs=np.array(occs),
               heights=np.array(heights, dtype=object),
               sat=np.mean(~np.isfinite(M), axis=0),
               raw_sha256=np.array(rawhash.hexdigest()))
    np.savez_compressed(target, **out)
    return out


def regress(M, hold_element=False, extra=None, min_occ=40):
    b1 = X1 & (~EL if hold_element else np.ones(len(GRID), bool))
    p = [np.nanmean(M[:, b1], axis=1), np.nanmean(M[:, X2], axis=1)]
    if extra is not None:
        p.extend(np.asarray(extra).T)
    P = np.column_stack(p + [np.ones(len(M))])
    C = np.full((P.shape[1], len(GRID)), np.nan)
    goodp = np.isfinite(P).all(axis=1)
    if goodp.sum() < min_occ:
        return P, C
    complete = np.isfinite(M[goodp]).all(axis=0)
    C[:, complete] = np.linalg.lstsq(P[goodp], M[goodp][:, complete], rcond=None)[0]
    for j in np.where(~complete)[0]:
        take = goodp & np.isfinite(M[:, j])
        if take.sum() >= min_occ:
            C[:, j] = np.linalg.lstsq(P[take], M[take, j], rcond=None)[0]
    return P, C


def masks(C, sat):
    strict = (sat <= .01) & np.isfinite(C).all(axis=0) & (np.abs(C[-1]) <= .005)
    core = CORE & (sat <= .01) & np.isfinite(C).all(axis=0) & (np.abs(C[-1]) <= .02)
    return strict, core


def fit_nonnegative(S, y):
    scale = np.linalg.norm(S, axis=0)
    c = nnls(S/scale, y)[0]/scale
    return c


def crossing(w, r):
    hits = []
    for i in range(len(w)-1):
        if np.isfinite(r[i:i+2]).all() and r[i] < 0 <= r[i+1]:
            hits.append(w[i] - r[i]*(w[i+1]-w[i])/(r[i+1]-r[i]))
    return hits[0] if len(hits) == 1 else np.nan


def meteorology(occs, lo=19, hi=22):
    out = []
    for occ in occs:
        f = ROOT / f'data/ace/v52_subset/l2_asc/{occ}v5.2.asc'
        rows = []
        if f.exists():
            for line in f.read_text().splitlines():
                p = line.split()
                try:
                    vals = [float(p[i]) for i in [0, 1, 3, 5, 7]]
                except (ValueError, IndexError):
                    continue
                if lo <= vals[0] <= hi and all(x > 0 for x in vals[1:]):
                    rows.append(vals)
        out.append(np.mean(rows, axis=0) if rows else np.full(5, np.nan))
    return np.array(out)


def bootstrap_indices(years, rng, block=False):
    if not block:
        return rng.integers(len(years), size=len(years))
    groups = np.floor(years).astype(int)
    unique = np.unique(groups)
    return np.concatenate([np.where(groups == g)[0]
                           for g in rng.choice(unique, len(unique), replace=True)])
