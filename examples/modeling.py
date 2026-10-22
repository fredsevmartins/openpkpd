"""Small reproducible 1C/2C comparison and 3C infusion example; no output files."""
import numpy as np
from openpkpd import (simulate_pk, fit_pk_model, compare_pk_models,
                      generate_pk_plots, calculate_nca)

t = np.linspace(0, 24, 30)
c = 2*np.exp(-.1*t)+np.random.default_rng(7).normal(0,.01,len(t))
fit1 = fit_pk_model(t,c,100,{'CL':4,'Vc':40})
fit2 = fit_pk_model(t,c,100,{'CL':4,'Vc':40,'Q':5,'Vp':30},n_compartments=2)
comparison = compare_pk_models([fit1,fit2])
print('1C parameter estimates:',fit1['parameters'])
print('Comparison (statistical support, not scientific validity):',comparison['models'])
plots = generate_pk_plots(fit1,comparison)
print('Generated PNGs:',[i['name'] for i in plots['images']])
three = simulate_pk(100,5,50,n_compartments=3,q2_L_h=10,vp2_L=100,q3_L_h=3,vp3_L=60,
                    route='infusion',infusion_duration_h=2,duration_h=120,n_points=601)
print('3C infusion total AUCinf:',three['aucinf_mg_h_L'])
nca = calculate_nca(np.arange(61.),2*np.exp(-.2*np.arange(61.)),
                    route='iv',dose_mg=100,integration_method='linear_up_log_down',terminal_selection='auto')
print('Reference NCA CL, Vss:',nca['clearance_L_h'],nca['vss_L'])
