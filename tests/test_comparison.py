"""Scientific failures must not be hidden by portability/provenance handling."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('comparison', ROOT/'reproduce/compare_reference.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

@pytest.mark.parametrize('actual,reference', [
    ({'x': [1]}, {'x': [1, 2]}), ({}, {'missing': 1}),
    ({'flag': False}, {'flag': True}), ({'count': 10}, {'count': 11}),
    ({'value': .2}, {'value': .1}), ({'x': float('nan')}, {'x': 1.}),
    ({'case': 'window'}, {'case': 'band'}),
    ({'coverage_1sig': .70001}, {'coverage_1sig': .7}),
])
def test_scientific_and_structural_changes_fail(actual, reference):
    diffs = []
    C.walk(actual, reference, '', diffs)
    assert diffs


def test_missing_products_fail(tmp_path):
    report = C.compare(tmp_path, tmp_path/'ref')
    assert report['failed_products'] == len(C.PAIRS)


def test_all_revised_products_are_in_standard_command():
    names = {p for p,_ in C.PAIRS}
    assert {'outputs/'+name for name in (
        'ace_null_trend.json', 'gas_budget_conditioning.json', 'closure_final.json',
        'closure_stability.json', 'closure_stability_summary.json',
        'closure_stability_summary.csv')} <= names


def test_portability_diagnostics_are_explicit_and_scoped():
    diffs, diagnostics = [], []
    a, b = {'summary': {'n_bit_identical_new_vs_old': 3}}, {'summary': {'n_bit_identical_new_vs_old': 4}}
    C.walk(a,b,'',diffs,'outputs/marginal_rank.json',diagnostics)
    assert not diffs and len(diagnostics)==1 and diagnostics[0]['changed']
    C.walk(a,b,'',diffs,'outputs/unrelated.json')
    assert diffs


def test_convergence_and_coverage_checked_independently_of_reference():
    a = json.loads((ROOT/'outputs/closure_stability.json').read_text())
    assert not C.audit_invariants(a)
    a['rows'][0]['draws'][0]['after']['scaled_projected_gradient'] = .002
    assert any('convergence' in e for e in C.audit_invariants(a))
    a['rows'][0]['refined']['coverage_1sig'] = .99
    assert any('coverage' in e for e in C.audit_invariants(a))


def test_objective_increase_fails_even_if_reference_is_identical():
    a = json.loads((ROOT/'outputs/closure_stability.json').read_text())
    d=a['rows'][0]['draws'][0]
    d['after']['objective']=d['before']['objective']+1
    assert any('objective' in e for e in C.audit_invariants(a))


def test_portable_closure_precision_does_not_hide_mass_shift():
    diffs=[]
    C.walk(.00012,.00010,'/rows[0]/draws[0]/mass_tg',diffs,'outputs/closure_stability.json')
    assert not diffs
    C.walk(.0002,.0001,'/rows[0]/draws[0]/mass_tg',diffs,'outputs/closure_stability.json')
    assert diffs


def test_receipt_requires_matching_product_hash(tmp_path):
    p=tmp_path/'output.json';p.write_text('{}')
    receipt={'products':{'output.json':{'step':'example','finished_utc':'now',
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}}
    assert 'regenerated' in C.provenance(tmp_path,'output.json',receipt)
    p.write_text('{"x":1}')
    assert 'archive consistency only' in C.provenance(tmp_path,'output.json',receipt)
    assert 'archive consistency only' in C.provenance(tmp_path,'output.json',{})


def test_csv_count_and_summary_values_are_checked(tmp_path):
    p=tmp_path/'summary.csv';p.write_text('case,n_refitted,refined_bias_mean\nband,2,0.0001\n')
    a=C.load_product(p);b=copy.deepcopy(a);b[0]['n_refitted']=3
    diffs=[];C.walk(a,b,'',diffs,'outputs/closure_stability_summary.csv')
    assert diffs


def test_summary_gradient_is_checked_against_draws():
    audit=json.loads((ROOT/'outputs/closure_stability.json').read_text())
    summary=json.loads((ROOT/'outputs/closure_stability_summary.json').read_text())
    assert not C.summary_invariants(audit,summary)
    summary['max_scaled_projected_gradient']=.01
    assert C.summary_invariants(audit,summary)


def test_persistence_reference_uses_stable_chronological_order():
    import csv
    with (ROOT/'outputs/ace_v52/fullspectrum.csv').open() as f:
        rows=list(csv.DictReader(f))
    # Python's stable sorted preserves the input order at tied timestamps.
    mass=[float(r['M_full']) for r in sorted(rows,key=lambda r:float(r['year']))]
    persistence=json.loads((ROOT/'outputs/latitude_declared_level.json').read_text())['persistence']
    mean=sum(mass)/len(mass)
    for k in (1,2):
        for suffix,center in (('',0.),('_mean_subtracted',mean)):
            flags=[m-center>k*persistence['sigma_emp_Tg'] for m in mass]
            expected=sum(x and y for x,y in zip(flags,flags[1:]))
            row=persistence[f'exceed_plus{k}sigma{suffix}']
            assert row['n_consecutive_both']==expected
            assert row['frac_consecutive_both']==expected/(len(mass)-1)
