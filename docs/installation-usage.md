# OpenPKPD: installation and user guide

OpenPKPD is a Python toolkit for pharmacokinetic (PK) and pharmacodynamic (PD)
calculations. This guide covers local installation, use from Python, connection to
Claude Desktop or another MCP host, and the data format expected by each tool.

OpenPKPD provides a local MCP server over standard input/output. The repository
currently distributes a Python package; it does not include a `.mcpb` Claude
extension bundle. In Claude Desktop, connect the installed `openpkpd-mcp` command
as a local MCP server using the configuration below.

All PK quantities should use consistent units: dose in mg, time in hours (h),
concentration in mg/L, clearance in L/h, volume in L, and first-order rates in
1/h. Unless a tool says otherwise, supply finite numeric values. Times must be
nonnegative and strictly increasing. Concentrations and doses must be nonnegative;
model parameters such as clearance, volume, Ka and dose intervals must be positive.

## 1. Install the package

Use Python 3.11 or newer. Open a terminal and run these commands from the repository
root after cloning or downloading the source:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[test]'
python -m pip check
```

On Windows PowerShell, activate the environment with `\.venv\Scripts\Activate.ps1`
and install with `python -m pip install '.[test]'`. If PowerShell blocks script
activation, use `.venv\Scripts\python.exe -m pip install '.[test]'` without
activating it. To use the library only, install `.` instead of `'.[test]'`.

The install command reads the project's dependency declarations and installs NumPy,
SciPy, Matplotlib, the official MCP Python SDK, and the optional pytest test runner.
No database, credentials, network service, or hosted account is needed to run a model.

For the reproducible scientific checks, run:

```sh
python -m pytest -q
python examples/reproduce.py
python examples/modeling.py
```

Confirm the server launcher exists without starting it outside an MCP host:

```sh
test -x .venv/bin/openpkpd-mcp
```

On Windows PowerShell, use `Test-Path .venv\Scripts\openpkpd-mcp.exe`. An MCP
host starts this command and communicates over stdin/stdout; it is not an HTTP
server.

## 2. Connect Claude Desktop

First install the package in a stable virtual environment as described above. Find
the absolute path to its console script:

- macOS or Linux: `<repository>/.venv/bin/openpkpd-mcp`
- Windows: `<repository>\.venv\Scripts\openpkpd-mcp.exe`

In Claude Desktop, open **Settings → Developer → Edit Config**. Add an `openpkpd`
entry under `mcpServers` in the configuration file. Replace the example path with
the actual absolute path on your machine:

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

The cache variables give Matplotlib writable directories in restricted environments.
For a normal desktop account, set them to writable directories. For example, on
Windows they can be set to `%TEMP%\openpkpd-matplotlib` and
`%TEMP%\openpkpd-cache`. The Windows `command` value is a JSON-escaped path such
as `C:\\Users\\YOUR_NAME\\openpkpd\\.venv\\Scripts\\openpkpd-mcp.exe`.

Keep any other entries already present in `mcpServers`. Save the configuration and
restart Claude Desktop. Open its developer or integration panel and check that
`openpkpd` is connected and that it lists all eight tools. If startup fails, confirm
the path exists, that installation completed in that same virtual environment, and
that the account running Claude Desktop can read the project and write to the cache
directories. When Python dependencies change, rerun the install command in that
virtual environment and restart the MCP host.

Other MCP-compatible assistants can connect in the same way: start the absolute
`openpkpd-mcp` executable as a local stdio process. Follow the host's own setup
steps for adding a local MCP server. OpenPKPD does not provide a hosted HTTP endpoint.

## 3. Ask questions in natural language

Once the MCP server is connected, ask for the administration route, dose, model
parameters, sampling period, and desired output. For example:

> Simulate a 100 mg IV infusion over 2 hours with CL 5 L/h and Vc 50 L. Use a
> two-compartment model with Q 10 L/h and Vp 100 L. Report concentrations every
> 2 hours through 24 hours and explain the assumptions.

> Calculate NCA for these observed times and concentrations. Use linear-up/log-down
> integration, select the terminal phase automatically, and report extrapolation
> diagnostics. Times in hours; concentrations in mg/L: ...

> Fit a one-compartment oral model to the following dose and concentration samples.
> The dose was 100 mg, F is known to be 0.8, and initial estimates are CL 5 L/h,
> Vc 50 L, Ka 1.2 1/h. Report residual diagnostics and a concentration plot.

Natural-language requests should include units and distinguish observed values from
assumptions. If the model needs a parameter that was not supplied, provide a
scientifically justified value; OpenPKPD does not infer a dose or study design from
context. For a table, label each column. JSON or a clear time/concentration list is
usually easier to verify than a screenshot of a table.

## 4. Pass observed data

For NCA and parameter estimation, provide numeric paired arrays. Each element at
position `i` in `time_h` corresponds to the concentration at position `i` in
`concentration_mg_L`. Both arrays must have the same length. The observations must
be finite, nonnegative, and ordered by strictly increasing time, with at least two
rows. Do not include a header row, units as text, commas as decimal separators, or
missing values in the numeric arrays. State units separately in your request.

A small NCA data set can be passed like this:

```json
{
  "time_h": [0, 1, 2, 4, 8, 12, 24, 36],
  "concentration_mg_L": [0, 1.2, 1.8, 1.35, 0.72, 0.38, 0.09, 0.022]
}
```

This means there are eight observations: 0 hours maps to 0 mg/L, 1 hour maps to
1.2 mg/L, and so on. These are example values only. A predose concentration and a
post-bolus concentration at the same timestamp cannot both be represented: time
values must be unique. At an IV bolus time, simulation values are post-dose.

### Observed NCA data

The essential fields are `time_h` and `concentration_mg_L`. Useful optional fields
include:

- `integration_method`: `linear` or `linear_up_log_down`.
- `terminal_selection`: `fixed` (default), `auto`, or `manual`.
- `terminal_points`: number of trailing terminal samples for `fixed` (at least 3).
- `terminal_indices`: original zero-based row numbers for `manual` selection, for
  example `[5, 6, 7]`.
- `route`: `iv`, `infusion`, or `oral`; `dose_mg` when asking for dose-derived
  clearance or volume metrics.
- `infusion_duration_h`: required when `route` is `infusion`.
- `blq`: an array of booleans, one per input row, marking below-quantification
  observations. Pair it with `blq_rule`: `exclude` (default), `zero`, or
  `half_lloq`. The `half_lloq` rule also requires a positive `lloq_mg_L`.
- `dose_interval_h`, `interval_start_h`, and `steady_state`: for a measured repeated-
  dose interval. AUCtau requires samples exactly at both interval boundaries. Set
  `steady_state` to true only when the study independently supports that assumption.
  The end-of-interval concentration must be a predose sample before calling it
  Ctrough.
- `first_interval_auc_mg_h_L`: first-dose AUC over a matching interval and regimen,
  if requesting an accumulation ratio.

OpenPKPD applies the BLQ rule you explicitly select. It does not guess which
observations are BLQ from the numbers alone. Excluding BLQ rows joins the remaining
samples under the selected integration rule; zero and half-LLOQ substitutions also
change the integrated curve. These are transparent descriptive methods, not a
censored-data model.

A typical NCA request might ask for Cmax, Tmax, Clast, Tlast, AUClast, AUCinf,
Lambda-z, half-life, AUMC, MRT, and dose-derived parameters where supported. Review
the returned terminal diagnostics and warnings. AUCinf and AUMCinf can be withheld
when terminal extrapolation is too large or the samples do not start at time zero.
Oral dose-derived clearance and volume are apparent CL/F and Vz/F. NCA does not
invent an unobserved predose concentration or interval endpoint.

### Data for fitting

Fitting requires the same paired `time_h` and `concentration_mg_L` arrays, plus:

- `dose_mg`: administered dose in mg.
- `initial_parameters`: starting estimates for every fitted PK parameter.
- `n_compartments`: 1, 2, or 3; defaults to 1.
- `route`: `iv` (default), `infusion`, or `oral`.
- `dose_times_h`: optional ordered times for repeated equal doses. If absent, one
  dose at time zero is used.
- `infusion_duration_h`: required for infusion fits.
- `bounds`: optional mapping from parameter name to `[lower, upper]` bounds in the
  same units as that parameter.
- `weighting`: `uniform` (default), `known_sigma`, or descriptive `relative` weights.
  `known_sigma` requires `sigma_mg_L`, one positive absolute error standard
  deviation per concentration observation.
- `bioavailability`: optional externally known F in `(0, 1]` for oral fits.

Use these parameter names:

| Model | Required starting estimates | Units |
|---|---|---|
| One compartment, IV | `CL`, `Vc` | L/h, L |
| One compartment, oral | `CL`, `Vc`, `ka` | L/h, L, 1/h |
| Two compartment | `CL`, `Vc`, `Q`, `Vp` | L/h, L, L/h, L |
| Three compartment | `CL`, `Vc`, `Q2`, `Vp2`, `Q3`, `Vp3` | L/h, L, L/h, L, L/h, L |
| Oral 2C or 3C | Add `ka` (and optionally `lag_time_h`) | 1/h; lag is h |

Example fit inputs:

```json
{
  "time_h": [0, 0.5, 1, 2, 4, 8, 12, 24],
  "concentration_mg_L": [2.0, 1.91, 1.81, 1.64, 1.34, 0.90, 0.60, 0.18],
  "dose_mg": 100,
  "initial_parameters": {"CL": 5, "Vc": 50},
  "n_compartments": 1,
  "route": "iv",
  "weighting": "uniform"
}
```

The fit is a nonlinear local optimization. Initial values and bounds matter; use
reasonable values for the drug and study. If the input contains only oral
observations and F is omitted, the fitted clearance and volumes are apparent
parameters divided by F. The tool does not assume that absolute F is 1. If F is
known from independent evidence, pass it explicitly. Oral flip-flop ambiguity and
three-compartment peripheral-label ambiguity can remain even when the optimizer
converges.

`known_sigma` uses independent Gaussian errors with the supplied absolute standard
deviations. `relative` weights each observation by its observed concentration; this
is a descriptive weighted objective and is not a normalized likelihood. AIC/AICc/BIC
are therefore withheld for relative-weight fits. Do not compare fits made to
different concentration arrays, sampling times, dosing regimens, F settings, or
likelihood assumptions.

## 5. Use all eight tools

Claude Desktop discovers these tools after connecting the local server:

| Tool | What to provide | What it returns |
|---|---|---|
| `simulate_pk` | Dose, CL, Vc, route, duration and sample count; add Q/Vp or Q2/Vp2/Q3/Vp3 for 2C/3C. Oral requires Ka; infusion requires infusion duration. | Concentration profile, compartment amounts, mass accounting, AUC and peak summary. |
| `simulate_multiple_doses` | Dose, CL, Vc, dose interval, number of doses, route, and optional compartment/absorption parameters. | Superposed profile, dose times, mass accounting and sampled peak. |
| `calculate_nca` | Paired observed arrays and optional integration, terminal, BLQ, route, dose or interval settings. | Observed metrics, integrated metrics, terminal diagnostics and estimable disposition metrics. |
| `fit_pk_model` | Paired data, dose, compartment count, route, initial parameter estimates and optional bounds/weighting. | Estimates, predictions, residuals, convergence, identifiability and conditional uncertainty. |
| `compare_pk_models` | A list of at least two complete fit results from the same observations and statistical model. | AIC/AICc/BIC, delta AIC, RMSE and comparison warnings where available. |
| `diagnose_pk_fit` | One complete result returned by `fit_pk_model`. | RMSE, residual trend checks, Durbin–Watson, basic parameter checks and warnings. |
| `generate_pk_plots` | A simulation or fit result as `profile`, a model comparison result as `comparison`, or both. | Embedded PNGs for concentration, semilog, residual, observed-versus-predicted and comparison plots as applicable. |
| `simulate_pd` | Concentrations, Emax, EC50 and optional baseline, Hill coefficient and model. | Emax or sigmoid Emax effects at the supplied concentrations. |

For a comparison, first fit each candidate model to exactly the same data and
regimen. Then provide the complete fit results to `compare_pk_models`. Pass that
comparison and one fit result to `generate_pk_plots` to create comparison and
observed-versus-predicted figures. A lower AIC is statistical evidence under the
specified likelihood; it does not by itself establish that a model is scientifically
valid.

Plot images are returned as portable PNGs. `generate_pk_plots` can also save files
when called from Python with an `output_directory`. Zeros are omitted from the
semilog panel and are not replaced with artificial positive concentrations. The
current plot tool does not draw confidence bands; review parameter intervals in the
fit result itself. Plots are descriptive and do not replace checking units, input
data, model assumptions, residuals, and identifiability diagnostics.

## 6. Call the Python API directly

You can use the same calculation engines without an MCP host:

```python
import numpy as np
from openpkpd import (
    calculate_nca,
    compare_pk_models,
    fit_pk_model,
    generate_pk_plots,
    simulate_multiple_doses,
    simulate_pk,
)

# Three-compartment oral profile, including a 0.5-hour absorption lag.
profile = simulate_pk(
    dose_mg=200,
    clearance_L_h=5,
    volume_L=50,
    bioavailability=0.8,
    ka_per_h=1.2,
    lag_time_h=0.5,
    route="oral",
    n_compartments=3,
    q2_L_h=8,
    vp2_L=40,
    q3_L_h=3,
    vp3_L=100,
    duration_h=48,
    n_points=241,
)

# Repeated one-compartment IV bolus profile.
repeated = simulate_multiple_doses(
    dose_mg=100,
    clearance_L_h=5,
    volume_L=50,
    dose_interval_h=12,
    n_doses=7,
    points_per_interval=101,
)

# Fit and compare one- and two-compartment IV models to the same observations.
time_h = np.array([0, .5, 1, 2, 4, 8, 12, 24], dtype=float)
observed = np.array([2.0, 1.91, 1.81, 1.64, 1.34, .90, .60, .18])
fit_1c = fit_pk_model(time_h, observed, 100, {"CL": 5, "Vc": 50})
fit_2c = fit_pk_model(
    time_h, observed, 100, {"CL": 4, "Vc": 40, "Q": 5, "Vp": 30},
    n_compartments=2,
)
comparison = compare_pk_models([fit_1c, fit_2c])
figures = generate_pk_plots(fit_1c, comparison)

# NCA input uses the actually observed concentrations, not a model-generated profile.
nca = calculate_nca(
    time_h, observed, route="iv", dose_mg=100,
    integration_method="linear_up_log_down", terminal_selection="auto",
)
print(nca["terminal_diagnostics"])
print(comparison["models"])
print([image["name"] for image in figures["images"]])
```

The general `simulate_compartments` function accepts an explicit `time_h` vector and
`dose_times_h` schedule, useful for custom or overlapping infusion designs. The
`simulate_pk` and `simulate_multiple_doses` functions provide convenient regular
grid interfaces. Python functions return dictionaries containing numeric arrays,
units, assumptions, and warnings or diagnostic fields as applicable.

## 7. Interpret results carefully

A missing (`null` in MCP JSON, `None` in Python) metric means its stated data or
acceptance criteria were not met. Read its warnings and diagnostics before drawing a
conclusion. In particular:

- NCA does not extrapolate back to an unobserved time-zero concentration. AUCinf and
  AUMCinf have separate extrapolation checks.
- An observed terminal slope is not proof that the terminal data reflect elimination;
  oral absorption can produce flip-flop kinetics.
- Steady state is a user-supplied study assumption. One interval cannot establish it.
- Fit confidence intervals are local, conditional approximations. They can be
  unreliable near parameter bounds or under weak identifiability.
- General multicompartment peak values are sampled from the requested profile and
  may miss a peak between samples.
- AIC/BIC compare statistical support only under compatible likelihoods. They do not
  validate the model, identify the correct mechanism, or establish physiological
  plausibility.

Read [scientific methods and validation](scientific-methods.md) for the equations,
BLQ rules, likelihood definitions, assumptions, test references and known limits.
OpenPKPD is for research, education and software development; it has not been
validated for clinical decision-making and does not recommend individual doses.
