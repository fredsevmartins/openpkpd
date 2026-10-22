import math
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from openpkpd import simulate_pk, simulate_multiple_doses
from openpkpd.models import simulate_compartments


def test_infusion_analytical_boundaries():
    t = [0, .001, 1, 2, 2.001, 5, 12]
    r = simulate_compartments(t, 100, {'CL':5, 'Vc':50}, route='infusion', infusion_duration_h=2)
    expected = [10*(1-math.exp(-.1*min(x,2)))*math.exp(-.1*max(x-2,0)) for x in t]
    assert r['concentration_mg_L'] == pytest.approx(expected)
    assert r['aucinf_mg_h_L'] == 20


def test_biexponential_independent_reference():
    # k10=.1,k12=.2,k21=.1 -> alpha,beta roots of x²-.4x+.01.
    alpha, beta = (.4+math.sqrt(.12))/2, (.4-math.sqrt(.12))/2
    a = 2*(alpha-.1)/(alpha-beta)
    b = 2*(.1-beta)/(alpha-beta)
    t = np.array([0, .1, 1, 5, 20, 100])
    r = simulate_compartments(t, 100, {'CL':5, 'Vc':50, 'Q':10, 'Vp':100}, n_compartments=2)
    assert r['concentration_mg_L'] == pytest.approx(a*np.exp(-alpha*t)+b*np.exp(-beta*t))


@pytest.mark.parametrize('route', ['iv', 'oral', 'infusion'])
def test_three_compartment_against_independent_ode(route):
    # Independent explicit amount equations, including absorption and cumulative elimination.
    p = {'CL':4, 'Vc':20, 'Q2':8, 'Vp2':40, 'Q3':3, 'Vp3':60}
    if route == 'oral':
        p.update(ka=1.2)
    t = np.linspace(0, 24, 61)
    r = simulate_compartments(t, 100, p, n_compartments=3, route=route,
                              bioavailability=.8 if route=='oral' else 1,
                              infusion_duration_h=2 if route=='infusion' else None)
    y0 = [100 if route=='iv' else 0, 0, 0, 80 if route=='oral' else 0, 0]
    def ode(t, y, rate):
        c, p2, p3, gut, _ = y
        return [-.75*c+.2*p2+.05*p3+1.2*gut+rate,
                .4*c-.2*p2, .15*c-.05*p3, -1.2*gut, .2*c]
    if route == 'infusion':
        first = solve_ivp(lambda x,y: ode(x,y,50), [0,2], y0, rtol=1e-10, atol=1e-11, dense_output=True)
        last = solve_ivp(lambda x,y: ode(x,y,0), [2,24], first.y[:,-1], rtol=1e-10, atol=1e-11, dense_output=True)
        reference = np.array([first.sol(x) if x<=2 else last.sol(x) for x in t]).T
    else:
        reference = solve_ivp(lambda x,y: ode(x,y,0), [0,24], y0, t_eval=t, rtol=1e-10, atol=1e-11).y
    assert r['concentration_mg_L'] == pytest.approx(reference[0]/20, rel=1e-7, abs=1e-9)
    assert r['eliminated_amount_mg'] == pytest.approx(reference[4], rel=1e-7, abs=1e-8)
    assert np.min(r['eliminated_amount_mg']) >= -1e-10


def test_oral_equal_rates_and_lag():
    r = simulate_compartments([0,1,2,3,4], 100, {'CL':5,'Vc':50,'ka':.1,'lag_time_h':2}, route='oral', bioavailability=.8)
    assert r['concentration_mg_L'][:3] == [0,0,0]
    assert r['concentration_mg_L'][3] == pytest.approx(1.6*.1*math.exp(-.1))
    assert r['pending_absorption_mg'][:2] == [80,80]
    assert r['unavailable_amount_mg'] == pytest.approx([20]*5)


@pytest.mark.parametrize('n,p', [(1,{'CL':5,'Vc':50}), (2,{'CL':5,'Vc':50,'Q':8,'Vp':40}),
                                 (3,{'CL':5,'Vc':50,'Q2':8,'Vp2':40,'Q3':3,'Vp3':80})])
@pytest.mark.parametrize('route', ['iv','oral','infusion'])
def test_mass_dose_proportionality_and_superposition(n,p,route):
    p = dict(p)
    if route=='oral': p.update(ka=.7,lag_time_h=.5)
    options = dict(n_compartments=n, route=route, bioavailability=.6 if route=='oral' else 1,
                   infusion_duration_h=3 if route=='infusion' else None)
    t = np.linspace(0, 36, 73)
    r = simulate_compartments(t,100,p,dose_times_h=[0,12,24],**options)
    doubled = simulate_compartments(t,200,p,dose_times_h=[0,12,24],**options)
    assert doubled['concentration_mg_L'] == pytest.approx(2*np.array(r['concentration_mg_L']))
    total = np.zeros(len(t))
    for dt in [0,12,24]:
        mask = t>=dt
        total[mask] += simulate_compartments(t[mask]-dt,100,p,**options)['concentration_mg_L']
    assert r['concentration_mg_L'] == pytest.approx(total)
    recovered = np.sum(r['compartment_amounts_mg'],axis=1)+r['depot_amount_mg']+np.array(r['pending_absorption_mg'])+r['eliminated_amount_mg']
    assert recovered == pytest.approx(r['delivered_bioavailable_mg'])


def test_public_extended_wrappers():
    r = simulate_pk(100,5,50,n_compartments=2,q_L_h=10,vp_L=100,route='infusion',infusion_duration_h=2)
    assert r['n_compartments']==2 and r['concentration_mg_L'][0]==0
    m = simulate_multiple_doses(100,5,50,12,3,n_compartments=3,q2_L_h=10,vp2_L=100,q3_L_h=2,vp3_L=20)
    assert m['dose_times_h']==[0,12,24]


@pytest.mark.parametrize('options', [{'n_compartments':4}, {'route':'wrong'}, {'bioavailability':.5},
                                    {'route':'infusion'}, {'infusion_duration_h':2}])
def test_invalid_model_options(options):
    with pytest.raises(ValueError):
        simulate_compartments([0,1],100,{'CL':5,'Vc':50},**options)
