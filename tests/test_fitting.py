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


@pytest.mark.parametrize('route', ['infusion','oral'])
def test_repeated_dose_parameter_recovery_independent_analytical(route):
    t = np.linspace(.1,36,60)
    y = np.zeros(len(t))
    for dt in [0,12,24]:
        mask = t>=dt
        x = t[mask]-dt
        if route=='infusion':
            y[mask] += 10*(1-np.exp(-.1*np.minimum(x,2)))*np.exp(-.1*np.maximum(x-2,0))
        else:
            x -= .7
            pos=x>=0
            yy=np.zeros(len(x))
            yy[pos]=2*1.2/1.1*(np.exp(-.1*x[pos])-np.exp(-1.2*x[pos]))
            y[mask] += yy
    initial={'CL':4,'Vc':45}
    options={'infusion_duration_h':2} if route=='infusion' else {'bioavailability':1.0}
    if route=='oral': initial.update(ka=1.,lag_time_h=.4)
    r=fit_pk_model(t,y,100,initial,route=route,dose_times_h=[0,12,24],**options)
    assert r['convergence']['success']
    assert r['parameters']['CL']==pytest.approx(5,rel=1e-5)
    assert r['parameters']['Vc']==pytest.approx(50,rel=1e-5)
    if route=='oral':
        assert r['parameters']['lag_time_h']==pytest.approx(.7,rel=1e-5)
        assert r['parameters']['ka']==pytest.approx(1.2,rel=1e-5)


def test_three_compartment_recovery_against_independent_ode():
    from scipy.integrate import solve_ivp
    t = np.r_[0,np.geomspace(.03,300,65)]
    def ode(t,y):
        c,a,b=y
        return [-.75*c+.2*a+.05*b, .4*c-.2*a, .15*c-.05*b]
    y=solve_ivp(ode,[0,300],[100,0,0],t_eval=t,rtol=1e-11,atol=1e-12).y[0]/20
    initial={'CL':3.5,'Vc':18,'Q2':7,'Vp2':36,'Q3':2.5,'Vp3':55}
    r=fit_pk_model(t,y,100,initial,n_compartments=3)
    assert r['convergence']['success']
    assert r['parameters']==pytest.approx({'CL':4,'Vc':20,'Q2':8,'Vp2':40,'Q3':3,'Vp3':60},rel=1e-4)
    assert any('labels' in x for x in r['diagnostics']['warnings'])


def test_near_singular_sensitivity_withholds_inference():
    t=np.array([0.,1e-12,2e-12,3e-12])
    y=2*np.exp(-.1*t)
    r=fit_pk_model(t,y,100,{'CL':5,'Vc':50,'Q':10,'Vp':100},n_compartments=2)
    assert not r['identifiability']['local_inference_available']
    assert r['standard_errors']['CL'] is None
    assert any('identifiable' in x for x in r['diagnostics']['warnings'])


def test_observation_relative_weights_do_not_claim_likelihood():
    t=np.linspace(0,20,20)
    y=2*np.exp(-.1*t)+.01*np.cos(t)
    r=fit_pk_model(t,y,100,{'CL':4,'Vc':45},weighting='relative')
    assert r['statistical_model']['log_likelihood'] is None
    assert r['aic'] is None and r['aicc'] is None and r['bic'] is None
    with pytest.raises(ValueError,match='normalized likelihood'):
        compare_pk_models([r,r])


def test_weakly_determined_extra_compartment_warns():
    t=np.linspace(0,24,30)
    y=2*np.exp(-.1*t)+np.random.default_rng(7).normal(0,.01,len(t))
    r=fit_pk_model(t,y,100,{'CL':4,'Vc':40,'Q':5,'Vp':30},n_compartments=2)
    assert any('poorly determined' in w or 'poorly identifiable' in w for w in r['diagnostics']['warnings'])
