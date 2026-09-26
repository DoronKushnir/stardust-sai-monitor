"""Target-excluded ACE residual prediction and matched-pair estimators.

Only sufficient statistics are cached: every bootstrap draw refits every
held-year basis, its fitting mask, amplitudes, and its matching metric/pairs.
"""
from pathlib import Path
import csv
import hashlib
import json
import warnings
import numpy as np
from scipy.special import logsumexp
from revision_common import ROOT, HERE, GRID, WL, PRIM, clean_transmission

OUT = ROOT / 'data/ace_floor/w0p25'   # the 0.25-um archive; the 0.1-um one is data/ace_floor/w0p1


def dump(path, obj):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    Path(path).write_text(json.dumps(clean(obj), indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SILICA_UM = 8.8   # round 43 (paper1): silica reference element centre, moved from 8.74 to 8.80 um


def response(center, width=.25):
    """Flat-wavelength response: exact overlap of resampled 2/cm cells.

    Piecewise-constant *public residual ratio*, not gas-weighted mission T.
    No endpoint clipping: an incompletely covered element is unavailable.
    """
    lo, hi = center-width/2, center+width/2
    if lo < 1e4/(GRID[-1]+1) or hi > 1e4/(GRID[0]-1):
        return None
    w = np.maximum(0, np.minimum(hi, 1e4/(GRID-1)) -
                   np.maximum(lo, 1e4/(GRID+1)))
    return w/w.sum()


def exclusion(w, guard=4.):
    target = GRID[w > 0]
    return (GRID >= target.min()-guard) & (GRID <= target.max()+guard)


def band_od(m, w):
    w = np.asarray(w)
    use = w > 0
    return -logsumexp(-np.asarray(m)[..., use] + np.log(w[use]), axis=-1)


def proxies(M, excluded, variant='standard'):
    windows = [(8., 8.9), (10.5, 12.2)]
    if variant == 'split_blue': windows = [(8., 8.45), (10.5, 11.35)]
    if variant == 'split_red': windows = [(8.45, 8.9), (11.35, 12.2)]
    if variant == 'weak': windows = [(3.85, 4.15), (5.2, 5.5)]
    values = []
    for lo, hi in windows:
        keep = (WL >= lo) & (WL <= hi) & ~excluded
        if keep.sum() < 5:
            values.append(np.full(len(M), np.nan)); continue
        finite = np.isfinite(M[:, keep])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            p = np.nanmean(M[:, keep], axis=1)
        p[finite.mean(axis=1) < .8] = np.nan
        values.append(p)
    return np.column_stack(values)


def statistics(x, weights=None):
    x = np.asarray(x)
    w = np.ones(len(x)) if weights is None else np.asarray(weights)
    ok = np.isfinite(x) & (w > 0)
    x, w = x[ok], w[ok]
    n = w.sum()
    if n < 2: return dict(n=int(n), n_unique=len(x), mean=np.nan, sd=np.nan, rms=np.nan)
    mean = np.average(x, weights=w)
    return dict(n=int(n), n_unique=len(x), mean=mean,
                sd=np.sqrt(np.sum(w*(x-mean)**2)/(n-1)),
                rms=np.sqrt(np.average(x*x, weights=w)))


def build_data(sample, altitude_ranges):
    """Rebuild from raw inputs; deliberately no stale spectral cache reuse."""
    records = list(csv.DictReader(Path(sample).open()))
    occs = np.array([r['occultation'] for r in records])
    years = np.array([float(r['decimal_year']) for r in records])
    geo_path = ROOT/'data/ace/v52_subset/occultationlist.csv'
    geo = {r['occultation_name']:r for r in csv.DictReader(geo_path.open())}
    hashes, selected, datasets = {str(geo_path.relative_to(ROOT)): sha(geo_path)}, [], {}
    for lo, hi in altitude_ranges:
        rows, averages, height, ids = [], [], [], []
        for i, occ in enumerate(occs):
            choices = []
            for path in sorted((ROOT/'data/ace/v52_subset/residual'/occ).iterdir()):
                try: h = float(path.name.split('.', 1)[1].removesuffix('.gz'))
                except (ValueError, IndexError): continue
                if lo <= h <= hi: choices.append((abs(h-(lo+hi)/2), h, path))
            if not choices: continue
            choices.sort()
            spectra = []
            for _, h, path in choices:
                rel = str(path.relative_to(ROOT)); hashes[rel] = sha(path)
                spectra.append(clean_transmission(np.loadtxt(path)))
            rows.append(-np.log(spectra[0]))
            averages.append(-np.log(np.mean(spectra, axis=0)))
            height.append(choices[0][1]); ids.append(i)
            selected.append(dict(occultation=occ, altitude_range=[lo,hi],
                                 native_height=choices[0][1],
                                 selected_file=str(choices[0][2].relative_to(ROOT)),
                                 n_heights_averaged=len(choices)))
        ids = np.array(ids)
        datasets[f'{lo:g}_{hi:g}'] = dict(
            M=np.array(rows), averaged=np.array(averages), heights=np.array(height),
            occs=occs[ids], years=years[ids],
            latitude=np.array([float(geo[o]['latitude']) for o in occs[ids]]),
            longitude=np.array([float(geo[o]['longitude']) for o in occs[ids]]))
    return datasets, hashes, selected


class Engine:
    def __init__(self, data, center, width=.25, guard=4., model='rank2',
                 mask='strict', split='year'):
        self.data, self.center, self.width = data, center, width
        self.w = response(center, width)
        if self.w is None: raise ValueError('outside grid coverage')
        self.excluded = exclusion(self.w, guard)
        self.cols = np.flatnonzero(PRIM | (self.w > 0))
        self.target = self.w[self.cols] > 0
        self.Y = data['M'][:, self.cols]
        self.fit_domain = PRIM[self.cols] & ~self.excluded[self.cols]
        self.groups, self.gid = np.unique(np.floor(data['years']).astype(int)
            if split == 'year' else np.arange(len(self.Y)), return_inverse=True)
        self.mask = mask
        p = proxies(data['M'], self.excluded)
        self.p = p
        features = [p]
        if model == 'rank3':
            keep = (WL >= 5.2) & (WL <= 5.5) & ~self.excluded
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                third = np.nanmean(data['M'][:,keep], axis=1)
            third[np.mean(np.isfinite(data['M'][:,keep]),axis=1)<.8] = np.nan
            features.append(third[:,None])
        if model == 'rank5':
            # This fixed invertible polynomial parameterization is only a
            # numerical preconditioner. Training fit/mask is still held out.
            z = (p-np.nanmean(p,axis=0))/np.nanstd(p,axis=0)
            features.append(np.column_stack([z[:,0]**2,z[:,0]*z[:,1],z[:,1]**2]))
        P = np.column_stack(features + [np.ones(len(p))])
        scale = np.nanstd(P, axis=0); scale[-1] = 1
        scale[~np.isfinite(scale) | (scale == 0)] = 1
        self.P = P/scale
        self.goodp = np.isfinite(P).all(axis=1)
        self.finite = np.isfinite(self.Y)
        v = self.finite & self.goodp[:,None]
        pp = np.nan_to_num(self.P)
        yy = np.nan_to_num(self.Y)
        ng, nj, nk = len(self.groups), len(self.cols), P.shape[1]
        self.gram = np.zeros((ng,nj,nk,nk)); self.rhs = np.zeros((ng,nj,nk))
        self.count = np.zeros((ng,nj)); self.available = np.zeros((ng,nj))
        self.sizes = np.zeros(ng)
        for g in range(ng):
            ix = self.gid == g
            self.gram[g] = np.einsum('ij,ik,il->jkl',v[ix],pp[ix],pp[ix])
            self.rhs[g] = np.einsum('ij,ij,ik->jk',v[ix],yy[ix],pp[ix])
            self.count[g] = v[ix].sum(0)
            self.available[g] = self.finite[ix].sum(0)
            self.sizes[g] = ix.sum()
        self.base = Engine(data,center,width,guard,'rank2',mask,split) if model != 'rank2' else None

    def coefficients(self, weights):
        G = np.einsum('g,gjkl->jkl',weights,self.gram)
        rhs = np.einsum('g,gjk->jk',weights,self.rhs)
        # Hermitian pseudoinverse handles unavailable/singular columns.
        ev = np.linalg.eigvalsh(G)
        good = ((weights > 0) @ self.count >= 40) & (ev[:,0] > 1e-11*np.maximum(ev[:,-1],1))
        C = np.full_like(rhs,np.nan)
        if good.any(): C[good] = np.linalg.solve(G[good],rhs[good,:,None])[...,0]
        denom = weights @ self.sizes
        missing = 1-(weights @ self.available)/denom if denom else np.ones(len(C))
        strict = (missing <= .01000000001) & np.isfinite(C).all(1)
        if self.mask == 'strict': strict &= np.abs(C[:,-1]) <= .005
        return C, strict

    def evaluate(self, weights=None):
        weights = np.ones(len(self.groups)) if weights is None else np.asarray(weights)
        pred = np.full((len(self.Y),self.target.sum()), np.nan)
        counts = np.zeros(len(self.Y),int)
        for g in np.flatnonzero(weights):
            train = weights.copy(); train[g] = 0
            C, fitmask = self.coefficients(train)
            if self.base is not None: _,fitmask = self.base.coefficients(train)
            fitmask &= self.fit_domain & np.isfinite(C).all(1)
            if not np.isfinite(C[self.target]).all(): continue
            for i in np.flatnonzero(self.gid == g):
                fit = fitmask & self.finite[i]
                counts[i] = fit.sum()
                if fit.sum()<80 or not self.finite[i,self.target].all(): continue
                a,_,rank,_ = np.linalg.lstsq(C[fit,:-1],self.Y[i,fit]-C[fit,-1],rcond=None)
                if rank < C.shape[1]-1: continue
                pred[i] = C[self.target,-1]+C[self.target,:-1]@a
        actual = self.Y[:,self.target]
        wt = self.w[self.cols][self.target]
        error = band_od(actual,wt)-band_od(pred,wt)
        return dict(error=error, residual=actual-pred, actual=actual, pred=pred,
                    weights=weights[self.gid], fit_bins=counts)


def matches(data, excluded, target, multiplicity=None, variant='standard',
            relation='all', max_height=.5, max_latitude=None):
    """Greedy disjoint pairs; bootstrap copies have integer capacities.

    No pair ever contains two copies of the same original observation. Metric
    scaling and greedy assignment are recomputed in every bootstrap sample.
    Target values enter only AFTER pair construction; target validity is an
    eligibility requirement (never target error magnitude).
    """
    n = len(target)
    mult = np.ones(n,int) if multiplicity is None else np.asarray(multiplicity,int)
    p = proxies(data['M'],excluded,variant)
    eligible = np.isfinite(p).all(1) & np.isfinite(target) & (mult > 0)
    if eligible.sum()<2: return []
    mu = np.average(p[eligible],axis=0,weights=mult[eligible])
    sd = np.sqrt(np.average((p[eligible]-mu)**2,axis=0,weights=mult[eligible]))
    if np.any(sd<=0): return []
    p = (p-mu)/sd
    i,j = np.triu_indices(n,1)
    ok = eligible[i] & eligible[j] & (np.abs(data['heights'][i]-data['heights'][j])<=max_height)
    groups = np.floor(data['years'])
    if relation == 'cross_year': ok &= groups[i] != groups[j]
    if relation == 'same_year': ok &= groups[i] == groups[j]
    if relation == 'same_direction': ok &= np.array([o[:2] for o in data['occs']])[i] == np.array([o[:2] for o in data['occs']])[j]
    if max_latitude is not None: ok &= np.abs(data['latitude'][i]-data['latitude'][j]) <= max_latitude
    i,j = i[ok],j[ok]
    distance = np.linalg.norm(p[i]-p[j],axis=1)
    order = np.lexsort((j,i,distance))
    capacity = mult.copy(); result = []
    for k in order:
        a,b = i[k],j[k]
        count = min(capacity[a],capacity[b])
        if not count: continue
        capacity[a]-=count; capacity[b]-=count
        for _ in range(count):
            result.append(dict(i=int(a),j=int(b),distance=distance[k],
                difference=(target[a]-target[b])/np.sqrt(2),
                delta_height=abs(data['heights'][a]-data['heights'][b]),
                delta_year=abs(groups[a]-groups[b]),
                delta_latitude=abs(data['latitude'][a]-data['latitude'][b])))
    return result
