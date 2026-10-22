"""Linear mammillary PK solved by matrix exponentials and dose superposition.

Amounts in mg, times in h. Elimination occurs only from the central compartment.
Exact matrix exponentials avoid eigenvector singularities (e.g. Ka equal to Ke).
"""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import expm


def positive(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def integer(name, value, minimum=1, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer)) or not math.isfinite(value) or int(value) != value or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} must be an integer in the supported range')
    return int(value)


def parameter_names(n_compartments, route):
    n = integer('n_compartments', n_compartments, maximum=3)
    if route not in {'iv', 'infusion', 'oral'}:
        raise ValueError('route must be iv, infusion or oral')
    names = ['CL', 'Vc']
    if n == 2:
        names += ['Q', 'Vp']
    elif n == 3:
        names += ['Q2', 'Vp2', 'Q3', 'Vp3']
    if route == 'oral':
        names += ['ka']
    return names


def validate_parameters(parameters, n_compartments, route):
    required = parameter_names(n_compartments, route)
    extra = set(parameters)-set(required)-({'lag_time_h'} if route == 'oral' else set())
    if extra or not set(required).issubset(parameters):
        raise ValueError(f'parameters require {required}; optional oral lag_time_h; unexpected {sorted(extra)}')
    p = {k: positive(k, parameters[k]) for k in required}
    if route == 'oral':
        lag = float(parameters.get('lag_time_h', 0))
        if not math.isfinite(lag) or lag < 0:
            raise ValueError('lag_time_h must be finite and nonnegative')
        p['lag_time_h'] = lag
    return p


def _matrix(p, n, oral):
    a = np.zeros((n+int(oral), n+int(oral)))
    a[0, 0] = -p['CL']/p['Vc']
    exchanges = [('Q', 'Vp')] if n == 2 else [('Q2', 'Vp2'), ('Q3', 'Vp3')] if n == 3 else []
    for j, (q, v) in enumerate(exchanges, 1):
        a[0, 0] -= p[q]/p['Vc']
        a[j, 0] = p[q]/p['Vc']
        a[0, j] = p[q]/p[v]
        a[j, j] = -p[q]/p[v]
    if oral:
        a[0, n] = p['ka']
        a[n, n] = -p['ka']
    return a


def simulate_compartments(time_h, dose_mg, parameters, *, n_compartments=1,
                          route='iv', bioavailability=1.0, infusion_duration_h=None,
                          dose_times_h=None):
    """Return concentration, compartment amounts and a cumulative mass balance.

    A sample exactly at an IV bolus is post-dose. Infusion boundaries are continuous.
    Oral lag leaves bioavailable drug pending until absorption starts. Amounts and
    eliminated mass account for the bioavailable fraction; unavailable mass is separate.
    """
    n = integer('n_compartments', n_compartments, maximum=3)
    p = validate_parameters(parameters, n, route)
    dose = positive('dose_mg', dose_mg)
    f = positive('bioavailability', bioavailability)
    if f > 1 or (route != 'oral' and f != 1):
        raise ValueError('F must be <=1 and must equal 1 for IV routes')
    t = np.asarray(time_h, float)
    if t.ndim != 1 or not len(t) or np.any(~np.isfinite(t)) or np.any(t < 0) or np.any(np.diff(t) <= 0):
        raise ValueError('time_h must be a nonempty strictly increasing finite nonnegative vector')
    doses = np.asarray([0.] if dose_times_h is None else dose_times_h, float)
    if doses.ndim != 1 or not len(doses) or np.any(~np.isfinite(doses)) or np.any(doses < 0) or np.any(np.diff(doses) <= 0):
        raise ValueError('dose_times_h must be finite, nonnegative and strictly increasing')
    if route == 'infusion':
        duration = positive('infusion_duration_h', infusion_duration_h) if infusion_duration_h is not None else None
        if duration is None:
            raise ValueError('infusion_duration_h is required')
    elif infusion_duration_h is not None:
        raise ValueError('infusion_duration_h only applies to infusion')
    oral = route == 'oral'
    a = _matrix(p, n, oral)
    amounts = np.zeros((len(t), len(a)))
    delivered = np.zeros(len(t))
    pending = np.zeros(len(t))
    unavailable = np.zeros(len(t))
    if route == 'infusion':
        augmented = np.zeros((n+1, n+1))
        augmented[:n, :n] = a
        augmented[0, n] = dose/duration
        initial = np.zeros(n+1)
        initial[n] = 1
        at_stop = (expm(augmented*duration) @ initial)[:n]
    else:
        initial = np.zeros(len(a))
        initial[n if oral else 0] = f*dose
    for dosing_time in doses:
        for i in np.flatnonzero(t >= dosing_time):
            elapsed = t[i]-dosing_time
            if route == 'infusion':
                delivered[i] += dose*min(elapsed/duration, 1)
                state = ((expm(augmented*elapsed) @ initial)[:n] if elapsed <= duration
                         else expm(a*(elapsed-duration)) @ at_stop)
            else:
                delivered[i] += f*dose
                unavailable[i] += (1-f)*dose
                lag = p.get('lag_time_h', 0)
                if elapsed < lag:
                    pending[i] += f*dose
                    continue
                state = expm(a*(elapsed-lag)) @ initial
            amounts[i] += state
    eliminated = delivered-amounts.sum(axis=1)-pending
    return {'time_h': t.tolist(), 'concentration_mg_L': (amounts[:, 0]/p['Vc']).tolist(),
            'compartment_amounts_mg': amounts[:, :n].tolist(),
            'depot_amount_mg': amounts[:, n].tolist() if oral else np.zeros(len(t)).tolist(),
            'pending_absorption_mg': pending.tolist(), 'eliminated_amount_mg': eliminated.tolist(),
            'delivered_bioavailable_mg': delivered.tolist(), 'unavailable_amount_mg': unavailable.tolist(),
            'dose_times_h': doses.tolist(), 'parameters': p, 'n_compartments': n,
            'route': route, 'bioavailability': f, 'infusion_duration_h': infusion_duration_h,
            'aucinf_mg_h_L': float(len(doses)*f*dose/p['CL']),
            'cmax_mg_L': float(np.max(amounts[:, 0]/p['Vc'])),
            'tmax_h': float(t[np.argmax(amounts[:, 0])]),
            'peak_source': 'sampled profile; peak may fall between samples',
            'units': {'time': 'h', 'concentration': 'mg/L', 'amount': 'mg', 'clearance': 'L/h', 'volume': 'L', 'ka': '1/h', 'auc': 'mg*h/L'},
            'assumptions': 'Linear mammillary disposition, elimination from central compartment, first-order oral absorption, fixed parameters, equal doses by superposition. AUCinf is total exposure over all scheduled doses, including doses beyond the sample horizon.'}
