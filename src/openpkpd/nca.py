"""Observed NCA with explicit sampling, BLQ and terminal extrapolation rules."""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import linregress


def _integrate(t, c, method):
    """Integrate C and t*C exactly under the selected interval interpolant."""
    auc = aumc = 0.0
    for t0, t1, a, b in zip(t[:-1], t[1:], c[:-1], c[1:]):
        dt = t1 - t0
        if method == 'linear_up_log_down' and 0 < b < a:
            z = math.log(b) - math.log(a)
            area = a * dt * math.expm1(z) / z
            # Stable first moment of exp(z*u), u in [0,1].
            moment = (sum(z**n / (math.factorial(n) * (n + 2)) for n in range(12))
                      if abs(z) < 0.01 else (math.exp(z) * (z - 1) + 1) / z**2)
            first = t0 * area + a * dt**2 * moment
        else:
            area = dt * (a + b) / 2
            first = t0 * area + dt**2 * (a + 2*b) / 6
        auc += area
        aumc += first
    return float(auc), float(aumc)


def calculate_nca(time_h, concentration_mg_L, terminal_points: int = 3, *,
                  integration_method: str = 'linear', route: str | None = None,
                  dose_mg: float | None = None, infusion_duration_h: float | None = None,
                  terminal_selection: str = 'fixed', terminal_indices: list[int] | None = None,
                  blq: list[bool] | None = None, blq_rule: str = 'exclude',
                  lloq_mg_L: float | None = None, dose_interval_h: float | None = None,
                  interval_start_h: float = 0.0, steady_state: bool = False,
                  first_interval_auc_mg_h_L: float | None = None) -> dict:
    """NCA on measured samples; no invented C0, interpolation beyond samples or F.

    BLQ flags refer to original rows: exclude, zero, or half_lloq. Terminal indices
    also refer to original rows. Automatic selection maximizes adjusted log-linear
    R² among declining trailing windows after Cmax (and infusion stop). Thresholds
    are screening rules, not regulatory acceptance criteria. Repeated-dose times
    are relative to the first administered dose; steady state is user-declared.
    """
    t, c = np.asarray(time_h, float), np.asarray(concentration_mg_L, float)
    if t.ndim != 1 or c.ndim != 1 or len(t) != len(c) or len(t) < 2:
        raise ValueError('time and concentration must be equal-length 1D arrays with >=2 points')
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(c)) or np.any(t < 0) or np.any(c < 0) or np.any(np.diff(t) <= 0):
        raise ValueError('finite nonnegative observations and strictly increasing times required')
    if integration_method not in {'linear', 'linear_up_log_down'}:
        raise ValueError('unknown integration_method')
    if route not in {None, 'iv', 'infusion', 'oral'}:
        raise ValueError('route must be iv, infusion or oral')
    for name, value in [('dose_mg', dose_mg), ('infusion_duration_h', infusion_duration_h),
                        ('dose_interval_h', dose_interval_h), ('first_interval_auc_mg_h_L', first_interval_auc_mg_h_L)]:
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be finite and positive')
    if not math.isfinite(interval_start_h) or interval_start_h < 0:
        raise ValueError('interval_start_h must be finite and nonnegative')
    if route == 'infusion' and infusion_duration_h is None:
        raise ValueError('infusion duration required for infusion NCA')
    if route != 'infusion' and infusion_duration_h is not None:
        raise ValueError('infusion duration applies only to infusion')
    if isinstance(terminal_points, bool) or int(terminal_points) != terminal_points or terminal_points < 3:
        raise ValueError('terminal_points must be an integer >=3')
    if terminal_selection not in {'fixed', 'auto', 'manual'}:
        raise ValueError('unknown terminal_selection')
    if terminal_indices is not None and terminal_selection != 'manual':
        raise ValueError('terminal_indices require manual selection')
    if blq_rule not in {'exclude', 'zero', 'half_lloq'}:
        raise ValueError('unknown BLQ rule')
    original = np.arange(len(t))
    flags = np.zeros(len(t), dtype=bool) if blq is None else np.asarray(blq)
    if flags.shape != t.shape or flags.dtype.kind != 'b':
        raise ValueError('blq must contain one boolean per observation')
    c = c.copy()
    if blq_rule == 'exclude':
        t, c, original = t[~flags], c[~flags], original[~flags]
    elif blq_rule == 'zero':
        c[flags] = 0
    else:
        if lloq_mg_L is None or not math.isfinite(lloq_mg_L) or lloq_mg_L <= 0:
            raise ValueError('positive LLOQ required for half_lloq')
        c[flags] = lloq_mg_L / 2
    if len(t) < 2:
        raise ValueError('at least two retained observations required')
    imax = int(np.argmax(c))
    auc, aumc = _integrate(t, c, integration_method)
    warnings = []
    if np.any(flags):
        warnings.append('BLQ handling changes the integration data; excluded rows are bridged by the selected interpolant.')
    diag = {'accepted': False, 'selection': terminal_selection,
            'criteria': {'minimum_points': 3, 'minimum_r_squared': 0.90, 'strictly_declining': True}, 'warnings': []}
    eligible = np.flatnonzero((c > 0) & (np.arange(len(t)) > imax) &
                             (t >= (infusion_duration_h or 0)))
    windows = []
    if terminal_selection == 'manual':
        if not terminal_indices or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in terminal_indices):
            raise ValueError('manual terminal_indices must be integer original row indices')
        if len(set(terminal_indices)) != len(terminal_indices) or terminal_indices != sorted(terminal_indices):
            raise ValueError('terminal_indices must be unique and increasing')
        idx = np.flatnonzero(np.isin(original, terminal_indices))
        if len(idx) != len(terminal_indices) or not set(idx).issubset(set(eligible)):
            raise ValueError('manual terminal points must be retained positive post-peak/post-infusion observations')
        windows = [idx]
    elif terminal_selection == 'auto':
        windows = [eligible[-n:] for n in range(3, len(eligible)+1)]
    elif len(eligible) >= terminal_points:
        windows = [eligible[-int(terminal_points):]]
    candidates = []
    for idx in windows:
        if len(idx) < 3 or np.any(np.diff(c[idx]) >= 0):
            continue
        fit = linregress(t[idx], np.log(c[idx]))
        r2 = float(fit.rvalue**2)
        if fit.slope < 0 and r2 >= 0.90:
            candidates.append((1-(1-r2)*(len(idx)-1)/(len(idx)-2), idx, fit))
    lam = half = aucinf = aumcinf = mrt = None
    auc_pct = aumc_pct = None
    if candidates:
        adjusted, idx, fit = max(candidates, key=lambda x: (x[0], len(x[1])))
        lam = -float(fit.slope)
        half = math.log(2)/lam
        diag.update(accepted=True, n_points=len(idx), selected_times_h=t[idx].tolist(),
                    selected_indices=original[idx].tolist(), r_squared=float(fit.rvalue**2),
                    adjusted_r_squared=float(adjusted), slope_standard_error=float(fit.stderr))
        if c[-1] > 0:
            tail = float(c[-1]/lam)
            tail_m = float(c[-1]*(t[-1]/lam+1/lam**2))
            auc_pct = 100*tail/(auc+tail)
            aumc_pct = 100*tail_m/(aumc+tail_m)
            diag['auc_extrapolated_fraction'] = auc_pct/100
            if auc_pct <= 20:
                aucinf = auc+tail
            else:
                diag['warnings'].append('AUC extrapolation exceeds 20%; AUCinf withheld.')
            if aumc_pct <= 20:
                aumcinf = aumc+tail_m
            else:
                diag['warnings'].append('AUMC extrapolation exceeds 20%; AUMCinf withheld.')
        else:
            diag['warnings'].append('Final concentration is zero; terminal extrapolation withheld.')
    else:
        diag['warnings'].append('No eligible declining terminal window with >=3 points and R² >=0.90.')
    complete_single = bool(t[0] == 0 and dose_interval_h is None)
    if not complete_single:
        warnings.append('Single-dose total-exposure and derived disposition metrics require samples from time zero and no repeated-dose interval.')
        aucinf = aumcinf = None
    cl = clf = vz = vzf = vss = None
    if complete_single and aucinf is not None and aumcinf is not None:
        mrt = aumcinf/aucinf
        if route == 'infusion':
            mrt -= infusion_duration_h/2
        if mrt <= 0:
            warnings.append('Nonpositive corrected MRT; withheld.')
            mrt = None
    if dose_mg is not None and route is not None and aucinf is not None:
        clearance = dose_mg/aucinf
        if route == 'oral':
            clf, vzf = clearance, clearance/lam
        else:
            cl, vz = clearance, clearance/lam
            if mrt is not None:
                vss = cl*mrt
    auctau = trough = cavg = fluctuation = accumulation = None
    if dose_interval_h is not None:
        end = interval_start_h+dose_interval_h
        # Require measured endpoints, avoiding invented interval coverage.
        mask = (t >= interval_start_h) & (t <= end)
        ti, ci = t[mask], c[mask]
        if len(ti) < 2 or ti[0] != interval_start_h or ti[-1] != end:
            warnings.append('AUCtau requires measured samples at both interval boundaries.')
        else:
            auctau, _ = _integrate(ti, ci, integration_method)
            if steady_state:
                trough = float(ci[-1])
                cavg = auctau/dose_interval_h
                fluctuation = float((max(ci)-min(ci))/cavg) if cavg > 0 else None
                if first_interval_auc_mg_h_L is not None:
                    accumulation = auctau/first_interval_auc_mg_h_L
                if dose_mg is not None and auctau > 0:
                    if route == 'oral':
                        clf = dose_mg/auctau
                    elif route in {'iv', 'infusion'}:
                        cl = dose_mg/auctau
            else:
                warnings.append('Steady-state Cavg, Ctrough, fluctuation and accumulation withheld without explicit steady_state=True.')
    elif steady_state or first_interval_auc_mg_h_L is not None:
        raise ValueError('steady-state metrics require dose_interval_h')
    return {'cmax_observed_mg_L': float(c[imax]), 'tmax_observed_h': float(t[imax]),
            'clast_mg_L': float(c[-1]), 'tlast_h': float(t[-1]),
            'auclast_mg_h_L': auc, 'aumclast_mg_h2_L': aumc, 'aucinf_mg_h_L': aucinf,
            'aumcinf_mg_h2_L': aumcinf, 'auc_extrapolation_percent': auc_pct,
            'aumc_extrapolation_percent': aumc_pct, 'lambda_z_per_h': lam,
            'terminal_half_life_h': half, 'mrt_h': mrt, 'clearance_L_h': cl,
            'apparent_clearance_L_h': clf, 'vz_L': vz, 'apparent_vz_L': vzf, 'vss_L': vss,
            'auctau_mg_h_L': auctau, 'ctrough_mg_L': trough, 'cavg_mg_L': cavg,
            'accumulation_ratio': accumulation, 'fluctuation_index': fluctuation,
            'terminal_diagnostics': diag, 'warnings': warnings, 'integration_method': integration_method,
            'blq_rule': blq_rule, 'route': route,
            'parameter_sources': {'observed': ['cmax_observed_mg_L', 'tmax_observed_h', 'clast_mg_L', 'tlast_h', 'ctrough_mg_L'],
                                  'integrated': ['auclast_mg_h_L', 'aumclast_mg_h2_L', 'auctau_mg_h_L'],
                                  'terminal_regression': ['lambda_z_per_h', 'terminal_half_life_h', 'aucinf_mg_h_L', 'aumcinf_mg_h2_L'],
                                  'derived': ['mrt_h', 'clearance_L_h', 'apparent_clearance_L_h', 'vz_L', 'apparent_vz_L', 'vss_L', 'cavg_mg_L', 'fluctuation_index', 'accumulation_ratio']},
            'units': {'time': 'h', 'concentration': 'mg/L', 'auc': 'mg*h/L', 'aumc': 'mg*h^2/L',
                      'lambda_z': '1/h', 'clearance': 'L/h', 'volume': 'L'},
            'assumptions': 'NCA of retained samples; no back-extrapolation to time zero. Oral clearance and volume are apparent (CL/F, Vz/F); oral MRT includes absorption. Interval metrics require declared steady state; Ctrough is measured predose at interval end.'}
