"""Concentration-effect models."""
from __future__ import annotations
import math
import numpy as np


def simulate_pd(concentrations, emax: float, ec50: float, baseline: float = 0.0,
                hill_coefficient: float = 1.0, model: str = "emax") -> dict:
    """Calculate effect from nonnegative concentrations in mg/L.

    E = baseline + Emax*C/(EC50+C), or baseline + Emax*C^h/(EC50^h+C^h).
    """
    c = np.asarray(concentrations, dtype=float)
    params = {"emax": emax, "ec50": ec50, "baseline": baseline, "hill_coefficient": hill_coefficient}
    if np.any(~np.isfinite(c)) or np.any(c < 0):
        raise ValueError("concentrations must be finite and nonnegative")
    if any(not math.isfinite(float(v)) for v in params.values()):
        raise ValueError("all parameters must be finite")
    if emax < 0 or ec50 <= 0 or hill_coefficient <= 0:
        raise ValueError("emax must be >= 0; ec50 and hill_coefficient must be > 0")
    if model not in {"emax", "sigmoid_emax"}:
        raise ValueError("model must be 'emax' or 'sigmoid_emax'")
    h = 1.0 if model == "emax" else float(hill_coefficient)
    # Ratio form avoids overflow in powers at very large C or EC50.
    ratio = np.zeros_like(c)
    positive = c > 0
    log_ratio = h * (np.log(c[positive]) - math.log(ec50))
    ratio[positive] = 1.0 / (1.0 + np.exp(-np.clip(log_ratio, -745, 745)))
    effect = float(baseline) + float(emax) * ratio
    return {"concentration_mg_L": c.tolist(), "effect": effect.tolist(), "model": model,
            "parameters": params, "assumptions": "Instantaneous equilibrium concentration-effect relationship; no delay, tolerance, or effect-site compartment."}
