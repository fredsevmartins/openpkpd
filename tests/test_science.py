import math
import numpy as np
import pytest
from openpkpd import simulate_pk, simulate_multiple_doses, simulate_pd, calculate_nca
from openpkpd.pk import concentration


def test_iv_exponential_and_auc_reference():
    result = simulate_pk(100, 5, 50, duration_h=24, n_points=5)
    assert result["concentration_mg_L"][1] == pytest.approx(2 * math.exp(-0.6))
    assert result["aucinf_mg_h_L"] == pytest.approx(20)


def test_oral_absorption_and_auc_reference():
    result = simulate_pk(100, 5, 50, 0.8, 1.2, 24, 241, "oral")
    assert result["aucinf_mg_h_L"] == pytest.approx(16)
    assert result["concentration_mg_L"][0] == 0
    assert max(result["concentration_mg_L"]) > 0


def test_ka_equal_ke_limit_is_finite_and_correct():
    got = concentration(np.array([0.0, 1.0]), 100, 2, 50, 1, 0.04, "oral")
    assert got[0] == 0
    assert got[1] == pytest.approx(2 * 0.04 * math.exp(-0.04))
    assert np.all(np.isfinite(got))


def test_dose_proportionality():
    c1 = concentration([0, 1, 4], 50, 5, 50)
    c2 = concentration([0, 1, 4], 100, 5, 50)
    assert c2 == pytest.approx(2 * c1)


def test_multiple_iv_dose_superposition():
    res = simulate_multiple_doses(100, 5, 50, 12, 3, points_per_interval=3)
    expected_at_24 = 2 * (math.exp(-2.4) + math.exp(-1.2) + 1)
    i = res["time_h"].index(24.0)
    assert res["concentration_mg_L"][i] == pytest.approx(expected_at_24)
    assert res["dose_times_h"] == [0, 12, 24]


def test_oral_multiple_dose_accumulates():
    res = simulate_multiple_doses(100, 5, 50, 12, 3, ka_per_h=1.2, route="oral", points_per_interval=20)
    assert len(res["time_h"]) > 20
    assert max(res["concentration_mg_L"]) > 0


def test_emax_and_sigmoid_emax():
    linear = simulate_pd([0, 2], 10, 2, baseline=3)
    sigmoid = simulate_pd([0, 2, 20], 10, 2, baseline=3, hill_coefficient=2, model="sigmoid_emax")
    assert linear["effect"] == pytest.approx([3, 8])
    assert sigmoid["effect"] == pytest.approx([3, 8, 3 + 10 * 100 / 101])


def test_nca_metrics_and_terminal_fit():
    t = np.array([0, 1, 2, 3, 4, 5], float)
    c = 2 * np.exp(-0.2 * t)
    res = calculate_nca(t, c, terminal_points=4)
    assert res["cmax_observed_mg_L"] == pytest.approx(2)
    assert res["tmax_observed_h"] == 0
    assert res["lambda_z_per_h"] == pytest.approx(0.2)
    assert res["terminal_half_life_h"] == pytest.approx(math.log(2) / 0.2)
    assert res["auclast_mg_h_L"] == pytest.approx(np.trapezoid(c, t))
    assert res["terminal_diagnostics"]["accepted"]


def test_nca_insufficient_terminal_data_withholds_estimates():
    res = calculate_nca([0, 1, 2], [1, 0.5, 0.0])
    assert res["lambda_z_per_h"] is None
    assert res["aucinf_mg_h_L"] is None
    assert not res["terminal_diagnostics"]["accepted"]


def test_nca_aucinf_allowed_when_extrapolation_is_small():
    t = np.arange(0, 21, dtype=float)
    c = 2 * np.exp(-0.2 * t)
    res = calculate_nca(t, c, terminal_points=5)
    assert res["terminal_diagnostics"]["accepted"]
    assert res["aucinf_mg_h_L"] == pytest.approx(
        res["auclast_mg_h_L"] + c[-1] / 0.2
    )
    assert res["terminal_diagnostics"]["auc_extrapolated_fraction"] <= 0.20


@pytest.mark.parametrize("args", [
    (0, 1, 1), (-1, 1, 1), (1, 0, 1), (1, 1, 0),
])
def test_invalid_pk_parameters_rejected(args):
    with pytest.raises(ValueError):
        simulate_pk(*args)


def test_invalid_route_or_bioavailability_rejected():
    with pytest.raises(ValueError): simulate_pk(1, 1, 1, route="sc")
    with pytest.raises(ValueError): simulate_pk(1, 1, 1, bioavailability=0)
    with pytest.raises(ValueError): simulate_pk(1, 1, 1, route="oral")


def test_invalid_pd_and_nca_inputs_rejected():
    with pytest.raises(ValueError): simulate_pd([-1], 1, 1)
    with pytest.raises(ValueError): simulate_pd([1], 1, 0)
    with pytest.raises(ValueError): calculate_nca([0, 1], [1])
    with pytest.raises(ValueError): calculate_nca([0, 1, 1], [1, 0.5, 0.2])
