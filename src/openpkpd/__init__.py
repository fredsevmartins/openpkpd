"""OpenPKPD: deterministic PK/PD models, estimation and diagnostics."""
from .pk import simulate_pk, simulate_multiple_doses, concentration
from .models import simulate_compartments
from .pd import simulate_pd
from .nca import calculate_nca
from .fitting import fit_pk_model, compare_pk_models, diagnose_pk_fit


def generate_pk_plots(*args, **kwargs):
    """Lazily import Matplotlib for portable scientific PNG output."""
    from .plotting import generate_pk_plots as generate
    return generate(*args, **kwargs)


__all__ = ["simulate_pk", "simulate_multiple_doses", "simulate_compartments", "concentration",
           "simulate_pd", "calculate_nca", "fit_pk_model", "compare_pk_models",
           "diagnose_pk_fit", "generate_pk_plots"]
__version__ = "1.0.0"
