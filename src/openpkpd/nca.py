"""Noncompartmental analysis with explicit terminal-phase diagnostics."""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import linregress


def calculate_nca(time_h, concentration_mg_L, terminal_points: int = 3) -> dict:
    """Calculate observed NCA metrics using linear trapezoids.

    Terminal fit uses the requested final n points after excluding zero concentrations;
    acceptance requires >=3 points, strictly increasing times, strictly declining positive
    concentrations, R² >= 0.90, and extrapolated AUC fraction <=20% for AUCinf.
    """
    t, c = np.asarray(time_h, float), np.asarray(concentration_mg_L, float)
    if t.ndim != 1 or c.ndim != 1 or t.size != c.size or t.size < 2:
        raise ValueError("time_h and concentration_mg_L must be equal-length 1D arrays with at least 2 points")
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(c)) or np.any(t < 0) or np.any(c < 0):
        raise ValueError("times and concentrations must be finite and nonnegative")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time_h must be strictly increasing")
    if isinstance(terminal_points, bool) or int(terminal_points) != terminal_points or terminal_points < 3:
        raise ValueError("terminal_points must be an integer >= 3")
    cmax_i = int(np.argmax(c))
    auc_last = float(np.trapezoid(c, t))
    positive_idx = np.flatnonzero((c > 0) & (np.arange(c.size) > cmax_i))
    diag = {"accepted": False, "criteria": {"minimum_points": 3, "minimum_r_squared": 0.90, "strictly_declining": True}, "warnings": []}
    lam, half, auc_inf, r2 = None, None, None, None
    if len(positive_idx) < terminal_points:
        diag["warnings"].append("Insufficient positive concentrations for terminal estimation.")
    else:
        idx = positive_idx[-int(terminal_points):]
        tt, cc = t[idx], c[idx]
        if np.any(np.diff(cc) >= 0):
            diag["warnings"].append("Selected terminal concentrations are not strictly declining.")
        else:
            fit = linregress(tt, np.log(cc))
            candidate = -float(fit.slope)
            r2 = float(fit.rvalue ** 2)
            diag.update({"n_points": int(len(idx)), "selected_times_h": tt.tolist(), "r_squared": r2})
            if candidate <= 0:
                diag["warnings"].append("Estimated terminal slope is not an elimination phase.")
            elif r2 < 0.90:
                diag["warnings"].append("Terminal fit R² is below 0.90.")
            else:
                lam = candidate
                half = math.log(2) / lam
                extrapolated = float(c[-1] / lam)
                fraction = extrapolated / (auc_last + extrapolated) if auc_last + extrapolated else math.inf
                diag.update({"accepted": True, "auc_extrapolated_fraction": fraction})
                if fraction <= 0.20 and c[-1] > 0:
                    auc_inf = auc_last + extrapolated
                elif c[-1] <= 0:
                    diag["warnings"].append("Final observed concentration is zero; AUCinf withheld because terminal extrapolation is undefined from that endpoint.")
                else:
                    diag["warnings"].append("AUC extrapolated fraction exceeds 20%; AUCinf withheld.")
    if lam is None:
        diag["warnings"].append("Terminal elimination rate and half-life were not estimated.")
    return {"cmax_observed_mg_L": float(c[cmax_i]), "tmax_observed_h": float(t[cmax_i]),
            "auclast_mg_h_L": auc_last, "lambda_z_per_h": lam, "terminal_half_life_h": half,
            "aucinf_mg_h_L": auc_inf, "terminal_diagnostics": diag,
            "integration_method": "Linear trapezoidal rule", "units": {"time": "h", "concentration": "mg/L", "auc": "mg*h/L", "lambda_z": "1/h"}}
