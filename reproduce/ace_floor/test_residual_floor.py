"""Numerical, leakage, missingness and resampling tests for the floor pipeline."""
import unittest
import numpy as np
from residual_floor_common import *


def synthetic(seed=42):
    rng=np.random.default_rng(seed);n=96
    p=rng.uniform(.01,.08,(n,2))
    a=np.exp(-((WL-8.5)/1.1)**2)+.1
    b=np.exp(-((WL-11.4)/1.8)**2)+.03
    M=p[:,0,None]*a+p[:,1,None]*b+rng.normal(0,.0003,(n,len(GRID)))
    # A coherent held-band error with a modest year dependence.
    w=response(SILICA_UM)>0
    M[:,w]+=rng.normal(0,.001,(n,1))+np.repeat(rng.normal(0,.0004,8),12)[:,None]
    return dict(M=M,years=np.repeat(np.arange(2004,2012),12)+.5,
        heights=20.5+rng.uniform(-.2,.2,n),latitude=rng.uniform(-20,20,n),
        longitude=rng.uniform(-180,180,n),occs=np.array([f'sr{i:06}' for i in range(n)]))


class FloorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=synthetic()

    def test_response_and_nonlinear_average(self):
        for c in [2.375,6.4,SILICA_UM,11.3,12.9]:
            w=response(c);self.assertAlmostEqual(w.sum(),1)
            self.assertAlmostEqual(band_od(np.full(len(GRID),.17),w),.17)
        self.assertIsNone(response(20.4));self.assertIsNone(response(13.7))
        self.assertIsNone(response(2.2))
        w=np.array([.25,.75]);x=np.array([.1,.2])
        self.assertAlmostEqual(band_od(x,w),-np.log(w@np.exp(-x)))

    def test_native_flags_not_interpolated_across(self):
        raw=np.array([[1,1.],[3,-1],[5,.9],[7,.8]])
        t=clean_transmission(raw,np.array([2.,4.,6.]))
        self.assertTrue(np.isnan(t[:2]).all());self.assertAlmostEqual(t[2],.85)

    def test_weighted_coefficients_match_direct_lstsq(self):
        weights=np.array([0,2,1,1,1,0,2,1])
        for model in ['rank2','rank3','rank5']:
            e=Engine(self.data,SILICA_UM,model=model)
            C,_=e.coefficients(weights)
            ix=np.repeat(np.arange(len(e.Y)),weights[e.gid])
            for j in np.flatnonzero(np.isfinite(C).all(1))[::35]:
                direct=np.linalg.lstsq(e.P[ix],e.Y[ix,j],rcond=None)[0]
                np.testing.assert_allclose(C[j],direct,atol=1e-10,rtol=1e-8)

    def test_all_target_windows_excluded_and_held_year_does_not_leak(self):
        for c in [6.4,6.9,SILICA_UM,11.3,11.4,12.9]:
            e=Engine(self.data,c);r=e.evaluate()
            changed=dict(self.data,M=self.data['M'].copy())
            test=e.gid==0;target=e.w>0
            changed['M'][np.ix_(test,target)]+=.01
            ee=Engine(changed,c);rr=ee.evaluate()
            np.testing.assert_allclose(e.p,ee.p,atol=0,rtol=0)
            np.testing.assert_allclose(r['pred'][test],rr['pred'][test],atol=1e-10)
            np.testing.assert_allclose(rr['error'][test]-r['error'][test],.01,atol=1e-10)
            self.assertFalse(np.any(e.fit_domain & e.excluded[e.cols]))

    def test_bootstrap_duplicates_do_not_leak(self):
        weights=np.array([3,1,1,1,1,1,0,0])
        e=Engine(self.data,SILICA_UM);r=e.evaluate(weights)
        changed=dict(self.data,M=self.data['M'].copy())
        changed['M'][np.ix_(e.gid==0,e.w>0)]+=.02
        rr=Engine(changed,SILICA_UM).evaluate(weights)
        np.testing.assert_allclose(r['pred'][e.gid==0],rr['pred'][e.gid==0],atol=1e-10)
        # Retraining changes errors relative to just weighting precomputed ones.
        base=e.evaluate()
        self.assertGreater(np.nanmax(abs(r['error']-base['error'])),1e-7)

    def test_full_element_missing_is_not_quiet_subset(self):
        data=dict(self.data,M=self.data['M'].copy());target=response(SILICA_UM)>0
        data['M'][0,np.flatnonzero(target)[0]]=np.nan
        r=Engine(data,SILICA_UM).evaluate()
        self.assertTrue(np.isnan(r['error'][0]));self.assertTrue(np.isfinite(r['error'][1:]).all())

    def test_covariance_identity(self):
        r=Engine(self.data,SILICA_UM).evaluate();v=r['residual'];w=response(SILICA_UM);w=w[w>0]
        cov=np.cov(v,rowvar=False)
        self.assertAlmostEqual(np.var(v@w,ddof=1),w@cov@w,places=15)
        self.assertGreater(w@cov@w,np.sum(w*w*np.diag(cov))*2)

    def test_matching_uses_no_target_values_and_no_self_pairs(self):
        d=self.data;w=response(SILICA_UM);ex=exclusion(w);q=band_od(d['M'],w)
        mult=np.repeat([3,1,1,1,1,1,0,0],12)
        for weights in [None,mult]:
            a=matches(d,ex,q,weights)
            b=matches(d,ex,q+np.arange(len(q))*.02,weights)
            self.assertEqual([(x['i'],x['j']) for x in a],[(x['i'],x['j']) for x in b])
            self.assertTrue(all(x['i']!=x['j'] for x in a))
            used=np.bincount([i for x in a for i in (x['i'],x['j'])],minlength=len(q))
            self.assertTrue(np.all(used <= (np.ones(len(q)) if weights is None else weights)))
        different=matches(d,ex,q,relation='cross_year')
        self.assertTrue(all(x['delta_year']>0 for x in different))
        same=matches(d,ex,q,relation='same_year')
        self.assertTrue(all(x['delta_year']==0 for x in same))

    def test_shared_bias_and_correlated_errors_are_unidentifiable(self):
        # Common bias changes neither predictive residuals nor differences.
        d=self.data;e=Engine(d,SILICA_UM);r=e.evaluate()
        altered=dict(d,M=d['M'].copy());altered['M'][:,e.w>0]+=.03
        rr=Engine(altered,SILICA_UM).evaluate()
        np.testing.assert_allclose(rr['error'],r['error'],atol=1e-10)
        # If target error is itself an off-band matching variable, tight
        # matching can drive differences to zero despite large actual errors.
        n=80;z=np.repeat(np.linspace(-.02,.02,n//2),2)
        m=np.zeros((n,len(GRID)));m[:,(WL>=8)&(WL<=8.9)]=z[:,None]
        m[:,(WL>=10.5)&(WL<=12.2)]=(z*z)[:,None]
        dd=dict(M=m,heights=np.ones(n)*20.5,years=np.arange(n),
                latitude=np.zeros(n),occs=np.array([str(i) for i in range(n)]))
        p=matches(dd,e.excluded,z)
        self.assertGreater(z.std(),.01)
        self.assertAlmostEqual(np.sqrt(np.mean([a['difference']**2 for a in p])),0)


if __name__=='__main__':unittest.main(verbosity=2)
