"""Deterministic one-compartment pharmacokinetic calculations."""
from __future__ import annotations

import math
import numpy as np

ASSUMPTIONS = (
    "One-compartment disposition; linear, time-invariant first-order elimination; "
    "instantaneous IV bolus or first-order oral absorption; constant parameters."
)


def _positive(name: str, value: float) -> float:
    x = float(value)
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be a finite value greater than zero")
    return x


def _grid(duration_h: float, n_points: int) -> np.ndarray:
    duration_h = _positive("duration_h", duration_h)
    if isinstance(n_points, bool) or int(n_points) != n_points or n_points < 2:
        raise ValueError("n_points must be an integer >= 2")
    return np.linspace(0.0, duration_h, int(n_points))


def concentration(time_h, dose_mg: float, clearance_L_h: float, volume_L: float,
                  bioavailability: float = 1.0, ka_per_h: float | None = None,
                  route: str = "iv") -> np.ndarray:
    """Return plasma concentration (mg/L) at time(s) in hours."""
    dose_mg = _positive("dose_mg", dose_mg)
    cl = _positive("clearance_L_h", clearance_L_h)
    v = _positive("volume_L", volume_L)
    f = float(bioavailability)
    if not math.isfinite(f) or not 0 < f <= 1:
        raise ValueError("bioavailability must be in (0, 1]")
    t = np.asarray(time_h, dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t < 0):
        raise ValueError("time_h must contain finite, nonnegative values")
    ke = cl / v
    if route == "iv":
        if ka_per_h is not None:
            raise ValueError("ka_per_h applies only to oral administration")
        return dose_mg / v * np.exp(-ke * t)
    if route != "oral":
        raise ValueError("route must be 'iv' or 'oral'")
    if ka_per_h is None:
        raise ValueError("ka_per_h is required for oral administration")
    ka = _positive("ka_per_h", ka_per_h)
    delta = ka - ke
    if abs(delta) <= 1e-7 * max(ka, ke):
        shape = ka * t * np.exp(-ke * t)
    else:
        shape = ka / delta * (np.exp(-ke * t) - np.exp(-ka * t))
    return f * dose_mg / v * shape


def simulate_pk(dose_mg: float, clearance_L_h: float, volume_L: float,
                bioavailability: float = 1.0, ka_per_h: float | None = None,
                duration_h: float = 24.0, n_points: int = 201,
                route: str = "iv") -> dict:
    """Simulate a single-dose concentration-time profile and analytical summaries."""
    t = _grid(duration_h, n_points)
    c = concentration(t, dose_mg, clearance_L_h, volume_L, bioavailability, ka_per_h, route)
    auc = float(bioavailability * dose_mg / clearance_L_h)
    if route == "iv":
        cmax, tmax = float(c[0]), 0.0
    else:
        ke = float(clearance_L_h) / float(volume_L)
        ka = float(ka_per_h)
        tmax = math.log(ka / ke) / (ka - ke) if not math.isclose(ka, ke, rel_tol=1e-7) else 1.0 / ke
        cmax = float(concentration(tmax, dose_mg, clearance_L_h, volume_L, bioavailability, ka_per_h, route))
    return {
        "time_h": t.tolist(), "concentration_mg_L": c.tolist(),
        "aucinf_mg_h_L": auc, "cmax_mg_L": cmax, "tmax_h": tmax,
        "route": route, "units": {"dose": "mg", "time": "h", "concentration": "mg/L", "clearance": "L/h", "volume": "L", "auc": "mg*h/L"},
        "assumptions": ASSUMPTIONS,
    }


def simulate_multiple_doses(dose_mg: float, clearance_L_h: float, volume_L: float,
                            dose_interval_h: float, n_doses: int,
                            bioavailability: float = 1.0, ka_per_h: float | None = None,
                            points_per_interval: int = 101, route: str = "iv") -> dict:
    """Simulate repeated equal doses using linear superposition."""
    dose_mg = _positive("dose_mg", dose_mg)
    interval = _positive("dose_interval_h", dose_interval_h)
    if isinstance(n_doses, bool) or int(n_doses) != n_doses or n_doses < 1:
        raise ValueError("n_doses must be an integer >= 1")
    if isinstance(points_per_interval, bool) or int(points_per_interval) != points_per_interval or points_per_interval < 2:
        raise ValueError("points_per_interval must be an integer >= 2")
    total = (int(n_doses) - 1) * interval
    t = np.linspace(0, total + interval, (int(n_doses) - 1) * (int(points_per_interval) - 1) + int(points_per_interval))
    c = np.zeros_like(t)
    for dose_time in np.arange(int(n_doses)) * interval:
        elapsed = t - dose_time
        mask = elapsed >= 0
        c[mask] += concentration(elapsed[mask], dose_mg, clearance_L_h, volume_L,
                                 bioavailability, ka_per_h, route)
    return {"time_h": t.tolist(), "concentration_mg_L": c.tolist(),
            "dose_times_h": (np.arange(int(n_doses)) * interval).tolist(),
            "cmax_mg_L": float(np.max(c)), "route": route,
            "units": {"dose": "mg", "time": "h", "concentration": "mg/L"},
            "assumptions": ASSUMPTIONS + " Repeated doses have equal size and are combined by linear superposition."}
