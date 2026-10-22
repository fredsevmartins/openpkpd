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


def _parameters(clearance_L_h, volume_L, n_compartments, ka_per_h, route,
                q_L_h, vp_L, q2_L_h, vp2_L, q3_L_h, vp3_L, lag_time_h):
    from .models import integer
    n = integer("n_compartments", n_compartments, maximum=3)
    p = {"CL": clearance_L_h, "Vc": volume_L}
    if n == 2:
        p.update(Q=q_L_h, Vp=vp_L)
    if n == 3:
        p.update(Q2=q2_L_h, Vp2=vp2_L, Q3=q3_L_h, Vp3=vp3_L)
    used = {2: [q_L_h, vp_L], 3: [q2_L_h, vp2_L, q3_L_h, vp3_L]}.get(n, [])
    if any(x is None for x in used):
        raise ValueError("all intercompartmental clearances and peripheral volumes are required")
    if (n != 2 and any(x is not None for x in [q_L_h, vp_L])) or (n != 3 and any(x is not None for x in [q2_L_h, vp2_L, q3_L_h, vp3_L])):
        raise ValueError("peripheral parameters do not match n_compartments")
    if route == "oral":
        if ka_per_h is None:
            raise ValueError("ka_per_h is required for oral administration")
        p.update(ka=ka_per_h, lag_time_h=lag_time_h)
    elif ka_per_h is not None or lag_time_h != 0:
        raise ValueError("absorption parameters apply only to oral administration")
    return p


def simulate_pk(dose_mg: float, clearance_L_h: float, volume_L: float,
                bioavailability: float = 1.0, ka_per_h: float | None = None,
                duration_h: float = 24.0, n_points: int = 201, route: str = "iv", *,
                n_compartments: int = 1, infusion_duration_h: float | None = None,
                lag_time_h: float = 0.0, q_L_h: float | None = None, vp_L: float | None = None,
                q2_L_h: float | None = None, vp2_L: float | None = None,
                q3_L_h: float | None = None, vp3_L: float | None = None) -> dict:
    """Simulate one, two or three compartments; legacy positional arguments retained."""
    from .models import simulate_compartments
    t = _grid(duration_h, n_points)
    p = _parameters(clearance_L_h, volume_L, n_compartments, ka_per_h, route,
                    q_L_h, vp_L, q2_L_h, vp2_L, q3_L_h, vp3_L, lag_time_h)
    result = simulate_compartments(t, dose_mg, p, n_compartments=n_compartments,
                                  route=route, bioavailability=bioavailability,
                                  infusion_duration_h=infusion_duration_h)
    if n_compartments == 1 and route in {"iv", "oral"}:
        ke = float(clearance_L_h)/float(volume_L)
        if route == "iv":
            tmax = 0.0
        else:
            ka = float(ka_per_h)
            tmax = lag_time_h + (math.log(ka/ke)/(ka-ke) if not math.isclose(ka, ke, rel_tol=1e-7) else 1/ke)
        result.update(tmax_h=tmax, cmax_mg_L=float(concentration(tmax-lag_time_h, dose_mg, clearance_L_h, volume_L, bioavailability, ka_per_h, route)), peak_source="analytical single-dose peak")
    return result


def simulate_multiple_doses(dose_mg: float, clearance_L_h: float, volume_L: float,
                            dose_interval_h: float, n_doses: int,
                            bioavailability: float = 1.0, ka_per_h: float | None = None,
                            points_per_interval: int = 101, route: str = "iv", *,
                            n_compartments: int = 1, infusion_duration_h: float | None = None,
                            lag_time_h: float = 0.0, q_L_h: float | None = None, vp_L: float | None = None,
                            q2_L_h: float | None = None, vp2_L: float | None = None,
                            q3_L_h: float | None = None, vp3_L: float | None = None) -> dict:
    """Repeated equal doses. IV bolus samples at dose times are post-dose."""
    from .models import simulate_compartments, integer
    interval = _positive("dose_interval_h", dose_interval_h)
    count = integer("n_doses", n_doses)
    points = integer("points_per_interval", points_per_interval, minimum=2)
    p = _parameters(clearance_L_h, volume_L, n_compartments, ka_per_h, route,
                    q_L_h, vp_L, q2_L_h, vp2_L, q3_L_h, vp3_L, lag_time_h)
    t = np.linspace(0, count*interval, count*(points-1)+1)
    return simulate_compartments(t, dose_mg, p, n_compartments=n_compartments,
                                 route=route, bioavailability=bioavailability,
                                 infusion_duration_h=infusion_duration_h,
                                 dose_times_h=(np.arange(count)*interval).tolist())
