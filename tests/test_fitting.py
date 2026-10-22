import math
import numpy as np
import pytest
from openpkpd.fitting import fit_pk_model, compare_pk_models, diagnose_pk_fit


def test_analytical_iv_parameter_recovery():
    t = np.linspace(0,24,25)
    y = 2*np.exp(-.1*t)
    r = fit_pk_model(t,y,100,{'CL':3,'Vc':30})
    assert r['convergence']['success']
    assert r['parameters']['CL'] == pytest.approx(5,rel=1e-6)
    assert r['parameters']['Vc'] == pytest.approx(50,rel=1e-6)
    assert r['aic'] is None # unknown variance likelihood unbounded at zero noise


def test_gaussian_likelihood_and_covariance_independent_reference():
    t = np.linspace(0,24,30)
    y = 2*np.exp(-.1*t)+np.random.default_rng(42).normal(0,.025,len(t))
    r = fit_pk_model(t,y,100,{'CL':4,'Vc':45})
    sse = sum(x*x for x in r['residuals_mg_L'])
    ll = -len(t)/2*(math.log(2*math.pi)+1+math.log(sse/len(t)))
    assert r['statistical_model']['log_likelihood'] == pytest.approx(ll)
    assert r['aic'] == pytest.approx(6-2*ll) # two PK parameters + variance
    assert r['bic'] == pytest.approx(3*math.log(len(t))-2*ll)
    assert r['aicc'] == pytest.approx(r['aic']+24/(len(t)-4))
    # Analytic Jacobian of D/V exp(-CL*t/V).
    cl, v = r['parameters']['CL'],r['parameters']['Vc']
    pred = 100/v*np.exp(-cl*t/v)
    j = np.column_stack((-t/v*pred, pred*(-1/v+cl*t/v**2)))
    cov = np.linalg.inv(j.T@j)*sse/(len(t)-2)
    assert r['covariance'] == pytest.approx(cov,rel=1e-4)
    assert r['confidence_intervals_95']['CL'][0]<cl<r['confidence_intervals_95']['CL'][1]
    assert diagnose_pk_fit(r)['parameter_plausibility']['positive_clearances_volumes_rates']


def test_known_sigma_likelihood():
    t = np.linspace(0,20,20)
    y = 2*np.exp(-.1*t)+.01*np.sin(t)
    r = fit_pk_model(t,y,100,{'CL':4,'Vc':45},weighting='known_sigma',sigma_mg_L=[.05]*len(t))
    ll = -.5*r['objective_weighted_sse']-len(t)*math.log(.05)-len(t)/2*math.log(2*math.pi)
    assert r['aic'] == pytest.approx(4-2*ll)


def test_oral_apparent_recovery_and_flip_flop_warning():
    t = np.linspace(.1,30,50)
    y = .8*100/50*1.2/(1.2-.1)*(np.exp(-.1*t)-np.exp(-1.2*t))
    r = fit_pk_model(t,y,100,{'CL':6,'Vc':60,'ka':1},route='oral')
    assert r['parameters']['CL']==pytest.approx(5/.8,rel=1e-5)
    assert r['parameters']['Vc']==pytest.approx(50/.8,rel=1e-5)
    assert 'apparent' in r['parameter_interpretation']
    assert any('flip-flop' in w for w in r['diagnostics']['warnings'])
    absolute = fit_pk_model(t,y,100,{'CL':4,'Vc':40,'ka':1},route='oral',bioavailability=.8)
    assert absolute['parameters']['CL']==pytest.approx(5,rel=1e-5)


def test_two_compartment_recovery_independent_biexponential():
    t = np.r_[np.linspace(0,3,12),np.linspace(4,80,20)]
    alpha,beta = (.4+math.sqrt(.12))/2,(.4-math.sqrt(.12))/2
    y = 2*(alpha-.1)/(alpha-beta)*np.exp(-alpha*t)+2*(.1-beta)/(alpha-beta)*np.exp(-beta*t)
    r = fit_pk_model(t,y,100,{'CL':4,'Vc':45,'Q':8,'Vp':90},n_compartments=2)
    assert r['parameters'] == pytest.approx({'CL':5,'Vc':50,'Q':10,'Vp':100},rel=1e-4)


def test_comparison_rejects_different_data_and_error_models():
    t = np.linspace(0,24,20)
    y = 2*np.exp(-.1*t)+.01*np.cos(t)
    a = fit_pk_model(t,y,100,{'CL':4,'Vc':40})
    b = fit_pk_model(t,y,100,{'CL':6,'Vc':60})
    c = compare_pk_models([a,b])
    assert c['models'][0]['delta_aic']==pytest.approx(0,abs=1e-6)
    other = fit_pk_model(t,y+.01,100,{'CL':4,'Vc':40})
    with pytest.raises(ValueError): compare_pk_models([a,other])
    relative = fit_pk_model(t,y,100,{'CL':4,'Vc':40},weighting='relative')
    with pytest.raises(ValueError): compare_pk_models([a,relative])


def test_overparameterization_bounds_and_nonconvergence():
    r = fit_pk_model([0,1],[2,1.8],100,{'CL':5,'Vc':50,'Q':10,'Vp':100},n_compartments=2)
    assert r['standard_errors']['CL'] is None
    assert any('Overparameterized' in w for w in r['diagnostics']['warnings'])
    t = np.linspace(0,24,20); y = 2*np.exp(-.1*t)
    bound = fit_pk_model(t,y,100,{'CL':2,'Vc':50},bounds={'CL':[1,3]})
    assert 'CL' in bound['convergence']['parameters_at_bounds']
    assert bound['standard_errors']['CL'] is None
    fail = fit_pk_model(t,y,100,{'CL':1,'Vc':20},max_nfev=1)
    assert not fail['convergence']['success']
    assert fail['aic'] is None


@pytest.mark.parametrize('options',[{'weighting':'bad'},{'weighting':'known_sigma'},
                                    {'bounds':{'CL':[6,10]}},{'bioavailability':.5},
                                    {'sigma_mg_L':[1]*3}])
def test_invalid_fits(options):
    with pytest.raises(ValueError):
        fit_pk_model([0,1,2],[2,1.8,1.6],100,{'CL':5,'Vc':50},**options)
