# OpenPKPD 1.0

Open-source deterministic pharmacokinetic and pharmacodynamic modeling in Python.
One-, two- and three-compartment mammillary models support IV bolus, finite IV
infusion, oral first-order absorption with optional lag, and repeated equal doses.
The toolkit includes NCA, bounded nonlinear estimation, model diagnostics,
likelihood-based comparison, scientific plots and eight local stdio MCP tools.

## Install and test

Python 3.11+ is required. NumPy, SciPy, Matplotlib and the official MCP Python SDK
stable 2.x line are the runtime dependencies.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pip check
pytest -q
python examples/reproduce.py
python examples/modeling.py
```

On machines with a read-only home directory, set writable graphics caches before
starting Python or the MCP client:

```sh
export MPLCONFIGDIR=/tmp/openpkpd-matplotlib
export XDG_CACHE_HOME=/tmp/openpkpd-cache
```

## Simulate

Legacy 1C calls remain supported:

```python
from openpkpd import simulate_pk, simulate_multiple_doses, simulate_pd

iv = simulate_pk(100, 5, 50)  # mg, L/h, L; AUCinf=20 mg*h/L
oral = simulate_pk(100, 5, 50, bioavailability=0.8, ka_per_h=1.2,
                   route="oral", lag_time_h=0.5)
infusion = simulate_pk(100, 5, 50, route="infusion", infusion_duration_h=2,
                       n_compartments=2, q_L_h=10, vp_L=100)
three = simulate_multiple_doses(100, 5, 50, 12, 6, n_compartments=3,
                                q2_L_h=10, vp2_L=100, q3_L_h=3, vp3_L=60)
pd = simulate_pd([0, 2, 10], emax=80, ec50=2)
```

For arbitrary sampling and dose times, `simulate_compartments(time_h, dose_mg,
parameters, ...)` accepts CL,Vc (1C); CL,Vc,Q,Vp (2C); CL,Vc,Q2,Vp2,Q3,Vp3
(3C). Oral adds ka and optional lag_time_h. Clearances are L/h, volumes L, ka 1/h,
time h, dose mg and concentration mg/L. Returned amount trajectories expose mass
balance, including depot, pending absorption, unavailable drug and elimination.
F must equal 1 for IV routes. Bolus samples at administration times are post-dose.

## Noncompartmental analysis

```python
from openpkpd import calculate_nca

nca = calculate_nca(
    [0, 1, 2, 4, 8, 16, 24, 48],
    [2, 1.81, 1.64, 1.34, 0.899, 0.404, 0.181, 0.0165],
    route="iv", dose_mg=100,
    integration_method="linear_up_log_down", terminal_selection="auto",
)
```

NCA reports observed Cmax/Tmax and Clast/Tlast; AUC and AUMC; terminal slope,
half-life and extrapolation diagnostics; MRT, CL or CL/F, Vz or Vz/F and Vss where
identifiable. Repeated-dose input supports measured AUCtau and declared steady-state
Ctrough, Cavg, accumulation and fluctuation. BLQ policies are explicit: exclude,
zero or half LLOQ, using supplied observation flags. Manual terminal indices refer
to original rows. Unavailable parameters are `None`, with diagnostics.

AUCinf and AUMCinf each require <=20% extrapolation and sampling from time zero.
No missing C0 or interval endpoints are invented. An end-of-interval sample used as
Ctrough must be predose. See [scientific methods](docs/scientific-methods.md) for
all equations, selection rules, units, assumptions and withholding criteria.

## Fit, compare and plot

```python
import numpy as np
from openpkpd import fit_pk_model, compare_pk_models, diagnose_pk_fit, generate_pk_plots

t = np.linspace(0, 24, 30)
c = 2*np.exp(-0.1*t) + np.random.default_rng(7).normal(0, 0.01, t.size)
fit1 = fit_pk_model(t, c, 100, {"CL": 4, "Vc": 40})
fit2 = fit_pk_model(t, c, 100, {"CL": 4, "Vc": 40, "Q": 5, "Vp": 30},
                    n_compartments=2)
comparison = compare_pk_models([fit1, fit2])
diagnostics = diagnose_pk_fit(fit1)
plots = generate_pk_plots(fit1, comparison)  # portable base64 PNG payloads
# Optional: output_directory="/tmp/openpkpd-plots" to also save PNG files.
```

Supply bounds as `{"CL": [0.1, 30]}`. Weighting supports uniform, known absolute
per-observation sigma, and descriptive relative WLS. Only normalized Gaussian
likelihoods receive AIC/AICc/BIC; observation-based relative weights do not.
Oral fitting without externally known F estimates **apparent** CL/F and volumes/F.
Covariance and 95% intervals are conditional local approximations and are withheld
for unreliable fits. Convergence and lowest AIC do not establish scientific validity.

## Local MCP integration

Start `openpkpd-mcp` after installation. A compatible local stdio MCP client launches
this process and manages stdin/stdout. Example configuration, adjusting paths:

```json
{
  "mcpServers": {
    "openpkpd": {
      "command": "/absolute/path/to/openpkpd/.venv/bin/openpkpd-mcp",
      "args": [],
      "env": {
        "MPLCONFIGDIR": "/tmp/openpkpd-matplotlib",
        "XDG_CACHE_HOME": "/tmp/openpkpd-cache"
      }
    }
  }
}
```

The tools are `calculate_nca`, `simulate_pk`, `fit_pk_model`, `compare_pk_models`,
`simulate_multiple_doses`, `simulate_pd`, `generate_pk_plots` and `diagnose_pk_fit`.
Numerical tools return structured results with units and assumptions. Plot output
uses embedded PNG image blocks and structured metadata, without hosted image links.
Existing numerical tool names and positional Python arguments are retained; discovery
now lists eight tools. This is a local stdio interface; remote hosting and native
hosted-assistant integration are not supplied.

## Validation and limitations

Tests use independent analytical and differential-equation references, parameter
recovery from synthetic observations, explicit likelihood/covariance calculations,
invalid-input cases, real MCP subprocesses, and PNG decoding. All original scientific
regressions remain; MCP discovery additionally checks the four new tools.
[Scientific methods](docs/scientific-methods.md) describes exactly what is tested.

Models assume fixed, linear PK, central elimination and first-order absorption.
General multicompartment peaks are sampled estimates. Sparse/noisy terminal phases,
oral flip-flop, unknown F, exchanged peripheral labels and poor sampling can prevent
identification. No censored-data fitting, population modeling, profile-likelihood
intervals or clinical validation is provided. Positivity checks do not replace
physiological review. This software is for research, education and development;
it does not provide individualized clinical dosing recommendations.

Contributions should preserve deterministic calculations, explicit units and
assumptions, input validation, and independent scientific reference tests.
