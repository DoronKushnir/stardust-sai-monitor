"""Consistent PSD/composition library and conditional sensitivity analysis.

Mie efficiencies are shared across PSD integrations. The grid of native
laboratory members is retained; no ambient-temperature extrapolation is made.
"""
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from revision_common import *
from saimon.refractive_index import sulfuric_acid_at_temperature, TabulatedRI
from saimon.mie import BohrenHuffmanMie
from saimon.psd import LognormalPSD
from saimon.backgrounds import _EMPIRICAL_ALT_KM, _EMPIRICAL_EXT_525
from saimon.geometry import tangent_to_slant_paths
from scipy.interpolate import PchipInterpolator

HRI=ROOT/'external/hitran_ri/hitran_ri/ascii'
MEMBERS=[('B',57,213,'biermann_h2so4/h2so4T213.biermann',2),
 ('B',60,213,'biermann_h2so4/h2so4T213.biermann',3),
 ('B',64,213,'biermann_h2so4/h2so4T213.biermann',4),
 ('B',70,215,'biermann_h2so4/h2so4T215.biermann',0),
 ('B',75,215,'biermann_h2so4/h2so4T215.biermann',1),
 ('B',80,215,'biermann_h2so4/h2so4T215.biermann',2),
 ('L',48,213,'myhre_h2so4/myhreh2so4set6.dat',0),
 ('L',58,233,'myhre_h2so4/myhreh2so4set5.dat',1),
 ('L',65,243,'myhre_h2so4/myhreh2so4set4.dat',2),
 ('L',72,233,'myhre_h2so4/myhreh2so4set3.dat',1),
 ('L',76,213,'myhre_h2so4/myhreh2so4set2.dat',1),
 ('Bt',64,203,'biermann_h2so4/h2so4T203.biermann',3),
 ('Lt',65,223,'myhre_h2so4/myhreh2so4set4.dat',3),
 ('Lt',72,213,'myhre_h2so4/myhreh2so4set3.dat',3)]
PSDS=[(r,s) for r in [40,60,80,100,150] for s in [1.4,1.5,1.6]]+[(270,1.75),(400,1.8)]+[(r,s) for r in [700,1000,1500] for s in [1.6,1.8]]
ALT=np.arange(0.,60001.,500.)
PROFILE=np.maximum(PchipInterpolator(_EMPIRICAL_ALT_KM,_EMPIRICAL_EXT_525['elevated'])(ALT/1000),0)
PROFILE[ALT<16000]=0
PATH=tangent_to_slant_paths(np.array([20500.]),ALT)[0]
BASE_COLUMN=PATH@PROFILE


def batch_qext(x,m):
    """Bohren-Huffman recurrence over a radius array, without a Python loop
    per radius. Same recurrence as sage3.mie._mie_single; validated below.
    """
    x=np.asarray(x);z=m*x
    stop=np.maximum((x+4*x**(1/3)+2).astype(int)+1,3)
    nmax=np.maximum(stop,np.abs(z).astype(int))+15
    D=np.zeros((int(nmax.max())+2,len(x)),complex)
    for n in range(int(nmax.max()),0,-1):
        active=n<=nmax
        D[n-1,active]=n/z[active]-1/(D[n,active]+n/z[active])
    pp=np.cos(x);pc=np.sin(x);cp=-np.sin(x);cc=np.cos(x)
    total=np.zeros(len(x))
    for n in range(1,int(stop.max())+1):
        a=n<=stop;xx=x[a];dn=D[n,a]
        pn=(2*n-1)/xx*pc[a]-pp[a];cn=(2*n-1)/xx*cc[a]-cp[a]
        xi=pn-1j*cn;xim=pc[a]-1j*cc[a]
        ta=dn/m+n/xx;tb=m*dn+n/xx
        an=(ta*pn-pc[a])/(ta*xi-xim);bn=(tb*pn-pc[a])/(tb*xi-xim)
        total[a]+=(2*n+1)*(an+bn).real
        pp[a],pc[a]=pc[a],pn;cp[a],cc[a]=cc[a],cn
    return 2*total/x**2


def member_id(mem): return f'{mem[0]}{mem[1]}T{mem[2]}'
def psd_id(psd): return f'r{psd[0]}s{psd[1]:g}'


def ri_for(mem):
    _,_,T,f,col=mem
    rows=[]
    for line in (HRI/f).read_text().splitlines():
        try: v=[float(x) for x in line.split()]
        except ValueError: continue
        if len(v)>=3+col:rows.append([v[0],v[2+col]])
    rows=np.array(rows); split=np.argmax(np.abs(np.diff(rows[:,0])))+1
    def order(block):
        w=1e7/np.maximum(block[:,0],1e-3);o=np.argsort(w)
        return w[o],block[o,1]
    wr,n=order(rows[:split]);wi,k=order(rows[split:])
    wl=np.unique(np.r_[wr,wi]);wl=wl[(wl>=2050)&(wl<=26000)]
    nn=np.interp(wl,wr,n);kk=np.clip(np.interp(wl,wi,k,left=0,right=np.nan),0,None)
    ok=np.isfinite(kk);vis=np.linspace(300,2040,200)
    vv=sulfuric_acid_at_temperature(215)(vis*1e-9)
    return TabulatedRI(np.r_[vis,wl[ok]],np.r_[vv.real,nn[ok]],np.r_[vv.imag,kk[ok]],T)


def make_library():
    cache=OUT/'optics.npz'
    if cache.exists():return dict(np.load(cache))
    thin=np.r_[GRID[PRIM],GRID[~PRIM][::4]]
    wavelengths=np.r_[525.,603.,1020.,1e7/thin]*1e-9
    # Upper-radius limit increased to 20 um to cover the scanned tail.
    r=np.geomspace(1e-9,20e-6,1000)
    dr=np.r_[np.diff(r)[0]/2,(r[2:]-r[:-2])/2,np.diff(r)[-1]/2]
    weights=np.array([LognormalPSD(rad*1e-9,s,n0_m3=1).dn_dr(r)*np.pi*r*r*dr for rad,s in PSDS])
    model=BohrenHuffmanMie();out={}
    xx=np.geomspace(.0004,250,200)
    for mm in [1.43+0j,1.5+.01j,1.2+.4j]:
        np.testing.assert_allclose(batch_qext(xx,mm),model.efficiencies(xx,mm)[0],rtol=1e-10,atol=1e-12)
    for mem in MEMBERS:
        ri=ri_for(mem);mv=ri(wavelengths);cs=np.empty((len(PSDS),len(wavelengths)))
        for j,lam in enumerate(wavelengths):
            qe=batch_qext(2*np.pi*r/lam,complex(mv[j]))
            cs[:,j]=weights@qe
        for k,psd in enumerate(PSDS):
            key=member_id(mem)+'_'+psd_id(psd)
            # Amplitude denotes local 525-nm extinction in units of the
            # fixed reference profile; slant conversion is varied downstream.
            out[key+'_spec']=np.interp(GRID,thin,cs[k,3:]/cs[k,0]*BASE_COLUMN)
            out[key+'_vis']=cs[k,:3]/cs[k,0]*PROFILE[41]
        print('computed optics',member_id(mem),flush=True)
    np.savez_compressed(cache,**out)
    save_json('optics_provenance.json',dict(radius_grid=[1e-9,20e-6,1000],
       members=MEMBERS,psds=PSDS,files={f:digest(HRI/f) for _,_,_,f,_ in MEMBERS}))
    return out


def shapes(lib,mem,psds):
    return np.column_stack([lib[member_id(mem)+'_'+psd_id(p)+'_spec'] for p in psds])


def composition(M,C,sat,lib,psds,sel=None,family='B',temperature_variant=False,offset=(0,0)):
    if sel is None:sel=np.ones(len(M),bool)
    tau=np.nanmedian(M[sel],axis=0)
    st,co=masks(C,sat);st&=np.isfinite(tau);co&=np.isfinite(tau)
    if co.sum()<5 or st.sum()<50:return np.nan,[],[]
    d=tau-C[-1]-np.array(offset)@C[:2]
    members=[m for m in MEMBERS if m[0]==family]
    if temperature_variant:
        alt={m[1]:m for m in MEMBERS if m[0]==family+'t'}
        members=[alt.get(m[1],m) for m in members]
    residuals=[];rms=[]
    for mem in members:
        S=shapes(lib,mem,psds);c=fit_nonnegative(S[st],d[st]);r=d-S@c
        residuals.append(np.mean(r[co]));rms.append(np.sqrt(np.mean(r[st]**2)))
    return crossing([m[1] for m in members],residuals),residuals,rms


def sensitivity(lib):
    z=build_matrix();M,yrs=z['M'],z['years'];_,C=regress(M)
    epochs={'quiet':(yrs>=2016.5)&(yrs<=2018.5),'post_Hunga':(yrs>=2022.1)&(yrs<=2023.5)}
    variants={'legacy_150_270':[(150,1.6),(270,1.75)],
              'fine_tail':[(60,1.5),(1000,1.8)],
              'fine_coarse_tail':[(60,1.5),(270,1.75),(1000,1.8)]}
    out={}
    fig,ax=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for a,(ename,sel) in zip(ax,epochs.items()):
        out[ename]={}
        for name,psds in variants.items():
            out[ename][name]={}
            for family in ['B','L']:
                cr,res,rm=composition(M,C,z['sat'],lib,psds,sel,family)
                out[ename][name][family]=dict(crossing=cr,core_residual=res,rms=rm)
                if name!='fine_coarse_tail':
                    a.plot([m[1] for m in MEMBERS if m[0]==family],np.array(res)*1e3,
                           'o-' if family=='B' else 's--',ms=3,
                           label=f'{family}: '+('150 + 270 nm' if name.startswith('legacy') else '60 + 1000 nm'))
        a.axhline(0,color='0.4',lw=.7);a.set_title(ename.replace('_',' '));a.grid(alpha=.2)
        a.set_xlabel('H₂SO₄ weight fraction (%)');a.legend(fontsize=8)
    ax[0].set_ylabel('O–H core residual (10⁻³ OD)');fig.tight_layout()
    fig.savefig(FIG/'revision_composition.png',dpi=180)
    # A conditional envelope, not a confidence interval: all scanned models,
    # including poor ones, are retained to reveal sensitivity.
    scan=[]
    for r,s,tail,ts in itertools.product([40,60,80],[1.4,1.5,1.6],[700,1000,1500],[1.6,1.8]):
        psds=[(r,s),(tail,ts)]
        for family in ['B','L']:
            cr=[composition(M,C,z['sat'],lib,psds,sel,family)[0] for sel in epochs.values()]
            scan.append([r,s,tail,ts,family,*cr])
    out['psd_scan']=scan
    out['offset_scenarios']=[]
    for c1,c2 in [(0,0),(-.005,0),(.005,0),(0,-.005),(0,.005)]:
        out['offset_scenarios'].append(dict(offset=[c1,c2],
            crossings={e:{f:composition(M,C,z['sat'],lib,variants['fine_tail'],sel,f,offset=(c1,c2))[0]
                           for f in ['B','L']} for e,sel in epochs.items()}))
    out['temperature_variants']={e:{f:composition(M,C,z['sat'],lib,variants['fine_tail'],sel,f,True)[0]
                                   for f in ['B','L']} for e,sel in epochs.items()}
    save_json('composition.json',out)
    print('composition', {e:out[e] for e in epochs},flush=True)
    # Resample complete spectra by occultation or year; recompute saturation
    # frequencies, intercept and masks. Bootstrap BOTH retained PSD choices.
    rng=np.random.default_rng(SEED);boot={}
    for block in [False,True]:
        samples=[];invalid=0
        for _ in range(500):
            ii=bootstrap_indices(yrs,rng,block);mm=M[ii];yy=yrs[ii]
            _,cc=regress(mm);sat=np.mean(~np.isfinite(mm),axis=0)
            sels=[(yy>=2016.5)&(yy<=2018.5),(yy>=2022.1)&(yy<=2023.5)]
            if min(s.sum() for s in sels)<3:invalid+=1;continue
            row=[]
            for name in ['legacy_150_270','fine_tail']:
                for f in ['B','L']:
                    cr=[composition(mm,cc,sat,lib,variants[name],sel,f)[0] for sel in sels]
                    row.extend([*cr,cr[1]-cr[0]])
            samples.append(row)
        arr=np.array(samples)
        boot['year' if block else 'occultation']=dict(attempted=500,no_epoch=invalid,
             n=len(arr),finite=np.isfinite(arr).sum(0),percentiles=np.nanpercentile(arr,[2.5,16,50,84,97.5],axis=0),
             columns=[f'{p}_{f}_{e}' for p in ['legacy','tail'] for f in ['B','L'] for e in ['quiet','post_Hunga','difference']])
        print('bootstrap',block,boot['year' if block else 'occultation'],flush=True)
    save_json('bootstrap.json',boot)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    lib=make_library();sensitivity(lib)
