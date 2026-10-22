import math
import numpy as np
import pytest
from openpkpd import calculate_nca


def test_log_down_exponential_reference_and_moments():
    t = np.arange(0, 61, 2.)
    c = 2*np.exp(-.2*t)
    r = calculate_nca(t, c, integration_method='linear_up_log_down', route='iv', dose_mg=100, terminal_selection='auto')
    assert r['auclast_mg_h_L'] == pytest.approx(10*(1-math.exp(-12)))
    assert r['aumclast_mg_h2_L'] == pytest.approx(50*(1-13*math.exp(-12)))
    assert r['aucinf_mg_h_L'] == pytest.approx(10)
    assert r['aumcinf_mg_h2_L'] == pytest.approx(50)
    assert r['mrt_h'] == pytest.approx(5)
    assert r['clearance_L_h'] == pytest.approx(10)
    assert r['vz_L'] == pytest.approx(50)
    assert r['vss_L'] == pytest.approx(50)


def test_linear_moments_independent_polynomial():
    # C=2+t: integral C from 0 to 2 = 6; integral t*C = 20/3.
    r = calculate_nca([0, 1, 2], [2, 3, 4])
    assert r['auclast_mg_h_L'] == 6
    assert r['aumclast_mg_h2_L'] == pytest.approx(20/3)
    assert r['lambda_z_per_h'] is None


def test_blq_and_manual_indices():
    t = [0, 1, 2, 3, 4, 5]
    c = np.exp(-np.array(t)).tolist()
    r = calculate_nca(t, c, blq=[False]*5+[True])
    assert r['tlast_h'] == 4
    assert r['terminal_diagnostics']['selected_indices'] == [2, 3, 4]
    z = calculate_nca(t, c, blq=[False]*5+[True], blq_rule='zero')
    assert z['aucinf_mg_h_L'] is None
    h = calculate_nca(t, c, blq=[False]*5+[True], blq_rule='half_lloq', lloq_mg_L=.01)
    assert h['clast_mg_L'] == .005
    with pytest.raises(ValueError):
        calculate_nca(t, c, terminal_selection='manual', terminal_indices=[0, 2, 3])


def test_oral_apparent_and_no_partial_exposure_clearance():
    t = np.arange(61.)
    c = 2*np.exp(-.2*t)
    r = calculate_nca(t, c, route='oral', dose_mg=100, integration_method='linear_up_log_down')
    assert r['apparent_clearance_L_h'] == pytest.approx(10)
    assert r['clearance_L_h'] is None and r['vss_L'] is None
    partial = calculate_nca(t[1:], c[1:], route='iv', dose_mg=100)
    assert partial['clearance_L_h'] is None
    assert partial['aucinf_mg_h_L'] is None


def test_steady_state_interval_and_accumulation():
    r = calculate_nca([12, 13, 14], [4, 3, 2], dose_interval_h=2, interval_start_h=12,
                      steady_state=True, first_interval_auc_mg_h_L=3, route='oral', dose_mg=12)
    assert r['auctau_mg_h_L'] == 6
    assert r['ctrough_mg_L'] == 2
    assert r['cavg_mg_L'] == 3
    assert r['accumulation_ratio'] == 2
    assert r['fluctuation_index'] == pytest.approx(2/3)
    assert r['apparent_clearance_L_h'] == 2
    assert r['aucinf_mg_h_L'] is None
    missing = calculate_nca([0, 1, 2], [4, 3, 2], dose_interval_h=3, steady_state=True)
    assert missing['auctau_mg_h_L'] is None


def test_infusion_mrt_correction():
    # Dose/CL=10, infusion duration 2, ke=.2: MRT observed=5+1.
    t = np.unique(np.r_[np.linspace(0, 2, 2001), np.linspace(2, 80, 7801)])
    c = np.where(t <= 2, 5*(1-np.exp(-.2*t)), 5*(1-np.exp(-.4))*np.exp(-.2*(t-2)))
    r = calculate_nca(t, c, route='infusion', dose_mg=100, infusion_duration_h=2)
    assert r['mrt_h'] == pytest.approx(5, rel=1e-5)
    assert r['vss_L'] == pytest.approx(50, rel=1e-5)


@pytest.mark.parametrize('kwargs', [{'dose_mg':-1}, {'blq_rule':'guess'}, {'integration_method':'bad'},
                                    {'terminal_selection':'manual'}, {'route':'infusion'},
                                    {'terminal_points':2}, {'dose_interval_h':0}])
def test_invalid_nca_options(kwargs):
    with pytest.raises(ValueError):
        calculate_nca([0, 1, 2, 3], [3, 2, 1, .5], **kwargs)


def test_terminal_auto_manual_and_separate_aumc_limit():
    t=np.arange(0.,21)
    c=2*np.exp(-.2*t)
    r=calculate_nca(t,c,route='iv',dose_mg=100,integration_method='linear_up_log_down',
                    terminal_selection='manual',terminal_indices=[16,17,18,19,20])
    assert r['terminal_diagnostics']['selected_indices']==[16,17,18,19,20]
    assert r['auc_extrapolation_percent']==pytest.approx(100*np.exp(-4))
    assert r['aumc_extrapolation_percent']==pytest.approx(500*np.exp(-4))
    assert r['aumcinf_mg_h2_L']==pytest.approx(50)
    # At t=10 AUC extrapolation is 13.5%, AUMC extrapolation is 40.6%.
    shorter=calculate_nca(t[:11],c[:11],route='iv',dose_mg=100,integration_method='linear_up_log_down')
    assert shorter['aucinf_mg_h_L']==pytest.approx(10)
    assert shorter['aumcinf_mg_h2_L'] is None and shorter['vss_L'] is None
    assert shorter['clearance_L_h']==pytest.approx(10)


def test_log_integration_extreme_positive_dynamic_range():
    r=calculate_nca([0,1],[1e200,1e-200],integration_method='linear_up_log_down')
    assert math.isfinite(r['auclast_mg_h_L'])
    assert r['auclast_mg_h_L']==pytest.approx(1e200/(400*math.log(10)))
