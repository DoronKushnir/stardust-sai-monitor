"""Regression checks for shared forecasts and the release dependency path."""
import ast
import json
import runpy
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def trend_namespace():
    tree = ast.parse((ROOT / "reproduce/ace_null_trend.py").read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("forecast_var", "apply_rule")]
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "trend_functions", "exec"), namespace)
    return namespace


def test_seasonal_forecast_is_shared():
    ns = trend_namespace()
    ns.update(t=np.full(100, 2016.0), yr=np.full(100, 2016))
    rule = dict(a=0., b_per_decade=0., t_ref=2016.,
                sigma_train=2., cov_ab=[[9., 0.], [0., 1.]])
    _, seasons = ns["apply_rule"](rule, np.full(100, 3.), np.ones(100, bool))
    assert np.isclose(seasons[2016]["seasonal_mean_z"], 3/np.sqrt(4/100+9))
    assert np.isclose(seasons[2016]["forecast_variance"], 9.)


def test_seasonal_covariance_matches_explicit_matrix():
    ns = trend_namespace()
    t = np.array([2016.1, 2016.5, 2016.9])
    ns.update(t=t, yr=np.full(3, 2016))
    cov = np.array([[.03, -.01], [-.01, .04]])
    rule = dict(a=.2, b_per_decade=.1, t_ref=2010.,
                sigma_train=.3, cov_ab=cov)
    X = np.c_[np.ones(3), (t-2010)/10]
    residual = np.array([.1, .2, .5])
    _, seasons = ns["apply_rule"](rule, X @ [.2, .1]+residual, np.ones(3, bool))
    w = np.full(3, 1/3)
    variance = w @ (.3**2*np.eye(3)+X @ cov @ X.T) @ w
    assert np.isclose(seasons[2016]["seasonal_mean_z"], np.mean(residual)/np.sqrt(variance))


def test_table_dependency_order_and_shipped_schema():
    steps = runpy.run_path(str(ROOT/"make_all.py"))["STEPS"]
    keys = [s[0] for s in steps]
    for producer in ("gasb76", "umodel", "fpc", "close76", "stable78", "gconv"):
        assert keys.index(producer) < keys.index("tabD2")
    gas = json.loads((ROOT/"outputs/gas_budget_conditioning.json").read_text())
    assert "joint_round70_prior" in gas["gas_budget_880"]
    assert "['joint_round70_prior']" in (ROOT/"reproduce/element_closure_table.py").read_text()


def test_prediction_variance_partition():
    gas = json.loads((ROOT/"outputs/gas_budget_conditioning.json").read_text())
    f = gas["gas_budget_880"]["final_retrieval_gain"]
    parts = ("photon_term", "discrepancy_term", "triplet", "line_prediction_term")
    assert np.isclose(sum(f[k]**2 for k in parts), f["from_measurement_covariance"]**2, rtol=1e-6, atol=1e-14)
    assert np.isclose(f["sigma_posterior_od"]**2,
                      f["from_measurement_covariance"]**2+f["from_prior"]**2, rtol=1e-6, atol=1e-14)
    assert not np.isclose(f["line_prediction_term"],
                          f["line_term_including_the_elements_own_error"], rtol=.01, atol=0.)


def test_closure_audit_retains_all_draws_and_uncertainties():
    audit = json.loads((ROOT/"outputs/closure_stability.json").read_text())
    assert audit["n_fits"] == 1596
    assert len(audit["rows"]) == 24
    assert sum(len(row["draws"]) for row in audit["rows"]) == 1596
    for row in audit["rows"]:
        draws = row["draws"]
        for prefix, summary_name in (("legacy_", "legacy"), ("", "refined")):
            mass = np.array([d[prefix+"mass_tg"] for d in draws])
            sigma = np.array([d[prefix+"sigma_tg"] for d in draws])
            assert np.all(np.isfinite(sigma) & (sigma > 0))
            z = (mass-row["truth_tg"])/sigma
            summary = row[summary_name]
            assert np.isclose(np.std(mass, ddof=1), summary["sd"])
            assert np.isclose(np.mean(np.abs(z)<1), summary["coverage_1sig"])
            assert np.isclose(np.mean(np.abs(z)<2), summary["coverage_2sig"])
        for d in draws:
            assert d["converged"] == (d["after"]["scaled_projected_gradient"] <= 1e-3)
            assert d["after"]["objective"] <= d["before"]["objective"] + 1e-7
