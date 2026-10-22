"""Bounded nonlinear least squares under explicitly defined Gaussian error models."""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t
from .models import simulate_compartments, validate_parameters, positive, integer


def _predict(times, dose, p, n, route, f, duration, doses):
    # Stable analytical fast path for the one-compartment model.
    if n != 1:
        return np.asarray(simulate_compartments(times, dose, p, n_compartments=n, route=route,
                           bioavailability=f, infusion_duration_h=duration, dose_times_h=doses)['concentration_mg_L'])
    ke = p['CL']/p['Vc']
    c = np.zeros(len(times))
    for dt in doses:
        elapsed = times-dt
        if route == 'oral': elapsed -= p.get('lag_time_h',0)
        mask = elapsed >= 0
        x = elapsed[mask]
        if route == 'iv':
            values = dose/p['Vc']*np.exp(-ke*x)
        elif route == 'infusion':
            values = dose/(duration*p['CL'])*(-np.expm1(-ke*np.minimum(x,duration)))*np.exp(-ke*np.maximum(x-duration,0))
        else:
            ka = p['ka']
            delta = ka-ke
            # Factor by the slower exponential to avoid overflow/cancellation.
            shape = (ka*x*np.exp(-ke*x) if abs(delta) < 1e-8*max(ka,ke)
                     else ka/abs(delta)*np.exp(-min(ka,ke)*x)*(-np.expm1(-abs(delta)*x)))
            values = f*dose/p['Vc']*shape
        c[mask] += values
    return c


def diagnose_pk_fit(fit: dict) -> dict:
    """Summarize fit reliability without treating convergence as scientific validity."""
    needed = {'residuals_mg_L','weighted_residuals','predicted_mg_L','observed_mg_L','parameters','convergence','identifiability'}
    if not needed.issubset(fit):
        raise ValueError('a complete fit_pk_model result is required')
    r = np.asarray(fit['residuals_mg_L'],float)
    wr = np.asarray(fit['weighted_residuals'],float)
    if r.ndim != 1 or len(r)<2 or len(wr)!=len(r) or np.any(~np.isfinite(r)) or np.any(~np.isfinite(wr)):
        raise ValueError('finite residual vectors with >=2 values required')
    warnings = list(fit['identifiability']['warnings'])
    if not fit['convergence']['success']:
        warnings.append('Optimizer did not converge; inference and model comparison are withheld.')
    corr = None
    if len(r)>2 and np.std(r[:-1])>0 and np.std(r[1:])>0:
        corr = float(np.corrcoef(r[:-1],r[1:])[0,1])
        if abs(corr)>.5:
            warnings.append('Large lag-one residual correlation suggests time structure or model mismatch.')
    dw = float(np.sum(np.diff(wr)**2)/np.sum(wr**2)) if np.sum(wr**2)>0 else None
    plausible = all(math.isfinite(v) and (v >= 0 if k=='lag_time_h' else v>0) for k,v in fit['parameters'].items())
    if not plausible:
        warnings.append('Nonphysical fitted parameters detected.')
    return {'rmse_mg_L':float(np.sqrt(np.mean(r*r))), 'mean_residual_mg_L':float(np.mean(r)),
            'lag_one_residual_correlation':corr, 'durbin_watson':dw,
            'parameter_plausibility': {'positive_clearances_volumes_rates': plausible,
                                      'note':'Positivity alone is not physiological plausibility; species and study context are required.'},
            'warnings':warnings, 'scientific_validity':'Not established by convergence or information criteria.'}


def fit_pk_model(time_h, concentration_mg_L, dose_mg: float, initial_parameters: dict, *,
                 n_compartments: int = 1, route: str = 'iv', bounds: dict | None = None,
                 weighting: str = 'uniform', sigma_mg_L: list[float] | None = None,
                 bioavailability: float | None = None, infusion_duration_h: float | None = None,
                 dose_times_h: list[float] | None = None, max_nfev: int = 2000) -> dict:
    """Fit concentrations, not log concentrations, by bounded SciPy least_squares.

    uniform: iid Gaussian errors with unknown scale. relative: Gaussian errors with
    fixed shape s_i=observed C_i (all must be positive), common unknown scale.
    known_sigma: independent Gaussian errors with supplied absolute standard deviations.
    Oral F absent means CL/F, Vc/F, Q/F and Vp/F are fitted and labelled apparent.
    F is never estimated jointly with disposition. Optional initial lag is fitted;
    absent lag is fixed at zero. No population or censored-data likelihood is used.
    """
    n = integer('n_compartments',n_compartments,maximum=3)
    max_nfev = integer('max_nfev',max_nfev)
    p0 = validate_parameters(initial_parameters,n,route)
    if 'lag_time_h' not in initial_parameters:
        p0.pop('lag_time_h',None)
    names = list(p0)
    t, y = np.asarray(time_h,float), np.asarray(concentration_mg_L,float)
    if t.ndim!=1 or y.ndim!=1 or len(t)!=len(y) or len(t)<2 or np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)) or np.any(t<0) or np.any(y<0) or np.any(np.diff(t)<=0):
        raise ValueError('at least two ordered finite nonnegative time/concentration pairs required')
    dose = positive('dose_mg',dose_mg)
    apparent = route=='oral' and bioavailability is None
    f = 1.0 if bioavailability is None else positive('bioavailability',bioavailability)
    if f>1 or (route!='oral' and f!=1):
        raise ValueError('F must be <=1 and IV F must equal 1')
    duration = infusion_duration_h
    if route=='infusion':
        if duration is None: raise ValueError('infusion duration required')
        duration = positive('infusion_duration_h',duration)
    elif duration is not None:
        raise ValueError('duration applies only to infusion')
    doses = np.asarray([0.] if dose_times_h is None else dose_times_h,float)
    if doses.ndim!=1 or len(doses)==0 or np.any(~np.isfinite(doses)) or np.any(doses<0) or np.any(np.diff(doses)<=0):
        raise ValueError('dose times must be ordered, finite and nonnegative')
    if weighting=='uniform':
        shape = np.ones(len(y))
    elif weighting=='relative':
        if np.any(y<=0): raise ValueError('relative weighting requires strictly positive observed concentrations')
        shape = y.copy()
    elif weighting=='known_sigma':
        shape = np.asarray(sigma_mg_L,float)
        if shape.shape!=y.shape or np.any(~np.isfinite(shape)) or np.any(shape<=0):
            raise ValueError('known_sigma requires a positive finite sigma per observation')
    else:
        raise ValueError('weighting must be uniform, relative or known_sigma')
    if weighting!='known_sigma' and sigma_mg_L is not None:
        raise ValueError('sigma_mg_L is only used with known_sigma')
    bounds = {} if bounds is None else bounds
    if set(bounds)-set(names): raise ValueError('bounds refer to unknown fitted parameters')
    lower, upper = [], []
    for name in names:
        pair = bounds.get(name,[0 if name=='lag_time_h' else 1e-8, math.inf])
        if len(pair)!=2: raise ValueError('each bound must be [lower,upper]')
        lo, hi = map(float,pair)
        if not math.isfinite(lo) or math.isnan(hi) or lo<0 or (name!='lag_time_h' and lo==0) or hi<=lo or not lo<=p0[name]<=hi:
            raise ValueError('invalid bounds or initial estimate outside bounds')
        lower.append(lo); upper.append(hi)
    x0 = np.array([p0[k] for k in names])
    def predict(x):
        return _predict(t,dose,dict(zip(names,x)),n,route,f,duration,doses)
    def residual(x):
        return (y-predict(x))/shape
    opt = least_squares(residual,x0,bounds=(lower,upper),x_scale='jac',max_nfev=max_nfev,
                        ftol=1e-10,xtol=1e-10,gtol=1e-10)
    predicted = predict(opt.x)
    raw = y-predicted
    wr = raw/shape
    sse = float(wr@wr)
    count, k = len(t), len(names)
    dof = count-k
    # Diagnose a dimensionless Jacobian (relative parameter perturbations).
    scaled_j = opt.jac*np.maximum(np.abs(opt.x),1e-8)
    singular = np.linalg.svd(scaled_j,compute_uv=False)
    rank = int(np.linalg.matrix_rank(scaled_j))
    condition = float(singular[0]/singular[-1]) if len(singular)>=k and singular[-1]>0 else math.inf
    warning = []
    active = [names[i] for i in range(k) if opt.active_mask[i]!=0 or
              np.isclose(opt.x[i],lower[i],rtol=1e-5,atol=1e-9) or
              (math.isfinite(upper[i]) and np.isclose(opt.x[i],upper[i],rtol=1e-5,atol=1e-9))]
    if rank<k or condition>1e8: warning.append('Rank-deficient or ill-conditioned sensitivity matrix; parameters are poorly identifiable.')
    if dof<=0: warning.append('Overparameterized data: no positive residual degrees of freedom.')
    if active: warning.append('Parameters at bounds: '+', '.join(active)+'. Local normal inference withheld.')
    if route=='oral': warning.append('Oral absorption/disposition can have flip-flop or global ambiguity; local full rank does not establish unique identification.')
    if apparent: warning.append('F was not supplied. CL, Vc and peripheral disposition parameters represent apparent values divided by F.')
    if n==3: warning.append('Three-compartment peripheral labels can be exchanged; label uniqueness is not identifiable.')
    reliable = bool(opt.success and dof>0 and rank==k and condition<=1e8 and not active)
    se = {name:None for name in names}
    ci = {name:None for name in names}
    covariance = None
    if reliable:
        # SVD avoids squaring the condition number through direct normal inversion.
        _, sv, vt = np.linalg.svd(opt.jac,full_matrices=False)
        variance = 1.0 if weighting=='known_sigma' else sse/dof
        cov = (vt.T*(variance/sv**2))@vt
        covariance = cov.tolist()
        critical = 1.959963984540054 if weighting=='known_sigma' else float(student_t.ppf(.975,dof))
        for i,name in enumerate(names):
            se[name] = float(np.sqrt(max(cov[i,i],0)))
            ci[name] = [float(opt.x[i]-critical*se[name]),float(opt.x[i]+critical*se[name])]
    likelihood = None
    criteria = {'aic':None,'aicc':None,'bic':None}
    likelihood_k = k + int(weighting!='known_sigma')
    # Zero/noise-free residuals have an unbounded unknown-variance likelihood.
    noise_floor = np.finfo(float).eps*max(float(np.sum((y/shape)**2)),1.)
    if weighting=='known_sigma':
        likelihood = -.5*sse-float(np.sum(np.log(shape)))-count/2*math.log(2*math.pi)
    elif sse>noise_floor:
        likelihood = -count/2*(math.log(2*math.pi)+1+math.log(sse/count))-float(np.sum(np.log(shape)))
    else:
        warning.append('Effectively zero residual variance: unknown-scale Gaussian likelihood and information criteria withheld.')
    if likelihood is not None and opt.success and count>likelihood_k and rank==k:
        aic = 2*likelihood_k-2*likelihood
        criteria = {'aic':aic,'aicc': aic+2*likelihood_k*(likelihood_k+1)/(count-likelihood_k-1) if count>likelihood_k+1 else None,
                    'bic':likelihood_k*math.log(count)-2*likelihood}
    else:
        warning.append('Information criteria unavailable for nonconverged, rank-deficient or saturated fits.')
    identity = {'time':t.tolist(),'observed':y.tolist(),'dose':dose,'route':route,'doses':doses.tolist(),
                'duration':duration,'F':bioavailability,'weighting':weighting,'shape':shape.tolist()}
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()
    result = {'parameters':{name:float(v) for name,v in zip(names,opt.x)},
              'parameter_interpretation':'apparent (CL/F, Vc/F, Q/F, Vp/F)' if apparent else 'absolute disposition with externally specified F' if route=='oral' else 'absolute disposition',
              'parameter_units':{name: 'h' if name=='lag_time_h' else '1/h' if name=='ka' else 'L' if name.startswith('V') else 'L/h' for name in names},
              'standard_errors':se,'confidence_intervals_95':ci,'covariance':covariance,
              'inference_method':'Local linearized conditional Gaussian approximation; not proof of global identifiability; intervals are not clipped to bounds.',
              'time_h':t.tolist(),'observed_mg_L':y.tolist(),'predicted_mg_L':predicted.tolist(),
              'residuals_mg_L':raw.tolist(),'weighted_residuals':wr.tolist(),'objective_weighted_sse':sse,
              'rmse_mg_L':float(np.sqrt(np.mean(raw**2))), 'n_observations':count,'n_parameters':k,
              'residual_degrees_of_freedom':dof, 'n_compartments':n,'route':route,
              'convergence':{'success':bool(opt.success),'status':int(opt.status),'message':opt.message,
                             'nfev':int(opt.nfev),'optimality':float(opt.optimality),'parameters_at_bounds':active},
              'identifiability':{'jacobian_rank':rank,'scaled_condition_number':condition if math.isfinite(condition) else None,
                                 'local_inference_available':reliable,'warnings':warning},
              'statistical_model':{'distribution':'independent Gaussian concentration errors', 'weighting':weighting,
                                    'scale':'known absolute sigma' if weighting=='known_sigma' else 'unknown common scale, maximum-likelihood estimate',
                                    'log_likelihood':likelihood,'likelihood_parameter_count':likelihood_k,
                                    'comparison_fingerprint':fingerprint}, **criteria}
    result['diagnostics'] = diagnose_pk_fit(result)
    return result


def compare_pk_models(fits: list[dict]) -> dict:
    """Compare only fits to identical observations and the same error likelihood.

    Never declares scientific validity. Nonconverged, saturated and singular fits
    retain their diagnostic rows but cannot be ranked by unavailable criteria.
    """
    if len(fits)<2:
        raise ValueError('at least two fits are required')
    fingerprints = [f['statistical_model']['comparison_fingerprint'] for f in fits]
    if len(set(fingerprints))!=1:
        raise ValueError('comparison requires identical observations, dosing and statistical error model')
    rows = []
    for i,f in enumerate(fits):
        rows.append({'model_index':i,'n_compartments':f['n_compartments'],'n_parameters':f['n_parameters'],
                     'aic':f['aic'],'aicc':f['aicc'],'bic':f['bic'],'rmse_mg_L':f['rmse_mg_L'],
                     'converged':f['convergence']['success'],'warnings':f['diagnostics']['warnings']})
    available = [r['aic'] for r in rows if r['aic'] is not None]
    for r in rows:
        r['delta_aic'] = r['aic']-min(available) if available and r['aic'] is not None else None
    return {'models':rows,'comparison_basis':'Same observations, regimen and explicitly specified Gaussian error likelihood.',
            'warnings':['Information criteria compare statistical support only; check residuals, identifiability, study design and physiological plausibility. No model is declared scientifically valid.']}
