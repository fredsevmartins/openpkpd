# Scientific methods and limits

## Scope and units

This is deterministic, individual PK modeling under linear time-invariant disposition,
with elimination from the central compartment. Dose and amounts are mg; time is h;
concentration is mg/L; clearances are L/h; volumes are L; absorption rates are 1/h.
PD remains the original Emax/sigmoid Emax implementation. No clinical validation,
population inference, uncertainty propagation, nonlinear elimination, covariates,
physiological parameter selection or individualized dosing is supplied.

## Noncompartmental analysis

`calculate_nca` keeps observed, integrated, regression-based and derived metrics
separate in `parameter_sources`. Cmax/Tmax and Clast/Tlast refer to retained samples;
Clast is the final retained value, including zero, rather than an invented last
quantifiable point. AUClast and AUMClast start at the first retained sample.

`linear` integrates the piecewise linear concentration interpolant; the first moment
is integrated exactly under that same interpolant (not a trapezoid of t*C).
`linear_up_log_down` uses exponential interpolation only for strictly declining,
positive interval endpoints and linear interpolation everywhere else. Both C and
t*C are integrated exactly under their respective interpolants. For a declining
interval C(t)=C0 exp(k*(t-t0)), AUC=C0*expm1(k*dt)/k. Small-argument expansions
stabilize the first moment. No concentration is back-extrapolated to time zero.

BLQ flags always refer to original observation rows. The caller must provide flags;
concentrations are not automatically classified using LLOQ. Three explicit policies:

- `exclude` (default): remove flagged rows; integration bridges the remaining rows.
- `zero`: replace flagged observations with zero, including terminal observations.
- `half_lloq`: replace flagged observations with half the supplied positive LLOQ.

These are descriptive substitutions, not a censored-data likelihood. Missing/NaN
values are rejected, not inferred. If exclusions remove time zero, total-exposure
and disposition metrics are withheld. BLQ sensitivity should be reviewed explicitly.

Terminal selection supports `fixed` trailing n positive post-peak observations,
`auto` trailing windows maximizing adjusted log-linear R² (ties prefer more points),
and `manual` original row indices. All selections exclude Cmax itself and infusion
samples before infusion stop. A window must have at least three strictly declining
positive concentrations and R² >= 0.90. Selected indices, times, R², adjusted R² and
slope standard error are reported. These screening rules do not prove that the
observed decline is the true elimination phase; oral flip-flop remains possible.

Lambda_z=-slope of log(C) against t and half-life=log(2)/Lambda_z. Extrapolation
uses the last retained observed C, not a fitted endpoint:

- AUC tail = Clast/Lambda_z.
- AUMC tail = Clast*(Tlast/Lambda_z + 1/Lambda_z²).

AUCinf and AUMCinf are independently withheld if their extrapolated fractions exceed
20%. Both are withheld when the endpoint is zero or sampling does not start at zero.
Single-dose totals are withheld for repeated-dose interval input. Terminal slope
estimates remain available, with diagnostics, when total exposure is withheld.

For complete single-dose coverage, MRT=AUMCinf/AUCinf. For a constant finite IV
infusion, subtract half the infusion duration. Oral MRT includes absorption and lag;
it is not interpreted as IV residence time. With a supplied dose:

- IV: CL=Dose/AUCinf, Vz=CL/Lambda_z, Vss=CL*MRT.
- Oral: CL/F=Dose/AUCinf and Vz/F=(CL/F)/Lambda_z. F and Vss are not inferred.

If AUMC extrapolation is unacceptable, MRT and Vss are withheld even when CL is
available. Unknown route or dose also prevents corresponding disposition estimates.
For oral data, the terminal slope can reflect absorption rather than elimination;
Vz/F is a conventional terminal apparent volume, not proof of disposition volume.

Interval calculations require measured samples at `interval_start_h` and exactly
one `dose_interval_h` later. No endpoints are invented. AUCtau is the observed
interval integral. The caller must explicitly declare steady state before Cavg,
Ctrough, fluctuation or accumulation are reported. Cavg=AUCtau/tau;
fluctuation=(observed interval Cmax-observed interval Cmin)/Cavg, as a ratio.
Ctrough is the measured **predose** concentration at the interval end; a post-bolus
endpoint cannot be used as a trough. Sampling should capture the interval maximum.
Accumulation=AUCtau at steady state / supplied first-dose interval AUC, requiring
the same dose, regimen, interval and integration convention. This compatibility is
a caller assertion, not inferable from the denominator alone. Steady-state CL or
CL/F uses Dose/AUCtau. Vss, MRT and terminal total exposure are not inferred from
repeated-dose intervals. Steady state itself is not established from a single interval.

## Compartmental simulation

For central amount Ac and peripheral amounts Ai:

```
dAc/dt = input - (CL + sum(Qi))*Ac/Vc + sum(Qi*Ai/Vi)
dAi/dt = Qi*Ac/Vc - Qi*Ai/Vi
```

One compartment has no peripheral terms. Two use Q,Vp; three use Q2,Vp2,Q3,Vp3.
Oral dosing adds dAg/dt=-ka*Ag and input=ka*Ag, with initial depot F*Dose after
optional lag. Bolus sets Ac(0+)=Dose. Infusion supplies Dose/duration until stop;
an augmented constant-input matrix exponential integrates the on-period exactly,
then the homogeneous matrix propagates the off-period. Overlapping infusions and
repeated doses use linear superposition. `scipy.linalg.expm` avoids singular
eigenvector formulas at repeated rate constants, including Ka=Ke. No ODE time-step
size is chosen by the application.

The engine reports central/peripheral amounts, unabsorbed depot, pending drug during
lag, cumulative eliminated amount, bioavailable delivered amount and unavailable
oral fraction. Eliminated mass is calculated by conservation; independent tests
also integrate an elimination state by differential equations to avoid a tautological
mass check. Floating-point roundoff near zero is possible.

Samples at bolus times are post-dose. Infusion boundaries are continuous. Multiple
dose wrappers simulate through one interval after the final administered dose.
General profiles report **sampled** peaks, which may miss the continuous peak.
Legacy single-dose 1C bolus/oral wrappers retain analytical peak reporting, including
peaks beyond the requested horizon. AUCinf is the analytical total F*sum(Dose)/CL
for the complete scheduled regimen, including doses beyond the observed horizon.

## Parameter estimation and inference

`fit_pk_model` fits all supplied required parameters with `scipy.optimize.least_squares`
and user bounds. Supply CL,Vc; add Q,Vp (2C), or Q2,Vp2,Q3,Vp3 (3C); oral requires ka.
Lag is fitted only if included in the initial estimates; otherwise fixed at zero.
Positive default lower bounds are 1e-8, except lag >=0. Initial estimates and bounds
must be supplied in the physical units above. Nonlinear optimization finds a local
solution; use multiple scientifically informed initial estimates when appropriate.

For oral-only data, omitted `bioavailability` means that CL, Vc, Qi and Vi are fitted
as apparent parameters CL/F, Vc/F, Qi/F and Vi/F. Internally using unit scaling in
this parameterization is not an assumption that absolute F equals one. Externally
known F can be supplied to estimate absolute disposition parameters. F is not jointly
estimated. Oral absorption/elimination may exhibit global flip-flop ambiguity even
when the local Jacobian is full rank. Three-compartment peripheral labels can be
exchanged. A successful optimizer does not resolve these ambiguities.

Residual=observed-predicted. Supported objectives:

- `uniform`: sum of squared residuals, iid Gaussian errors with unknown common sigma.
- `known_sigma`: sum((residual_i/sigma_i)²), independent Gaussian errors with
  externally supplied absolute standard deviations, fixed during fitting.
- `relative`: sum((residual_i/observed_i)²), only positive observations. These
  observed-data weights define descriptive WLS; they do **not** supply a normalized
  statistical likelihood. Information criteria and likelihood comparison are withheld.

For uniform errors, log L = -n/2*[log(2*pi)+1+log(SSE/n)]. For known sigma,
log L = -SSEweighted/2 - sum(log(sigma_i)) - n/2*log(2*pi). AIC=2K-2log L,
BIC=K*log(n)-2log L, AICc=AIC+2K(K+1)/(n-K-1). K includes the estimated variance
for uniform errors and excludes fixed known sigma. AICc requires n>K+1.
Effectively exact noise-free fits have unbounded unknown-scale likelihood; their
information criteria are withheld, not replaced with an arbitrary finite variance.
Nonconverged, saturated and rank-deficient fits also have no reported criteria.

Model comparison enforces matching observations, times, regimen, route, F specification,
weighting and variance shapes by fingerprint. Relative WLS cannot be compared by
information criteria. AIC support is not physiological or scientific validity.
Residual plots, sampling design, assumptions and plausibility must be reviewed.

Covariance uses an SVD of the weighted Jacobian, scaled by SSE/(n-p) for estimated
variance or by 1 for known sigma. Confidence intervals are approximate conditional
95% intervals using Student t (estimated variance) or normal quantiles (known sigma).
They are not clipped to physical bounds. Inference is withheld without convergence,
positive residual degrees of freedom, full rank, dimensionless sensitivity condition
number <=1e8, and an interior solution. For relative WLS, inference is explicitly
conditional on frozen observed weights and is not a substitute for a full error model.
Relative standard errors >=100% and intervals crossing physical limits trigger additional warnings.
No profile likelihood, bootstrap or global identifiability proof is implemented.

Diagnostics report RMSE, mean residual, lag-one correlation, Durbin–Watson and positive
parameters, with warnings. Positivity is only a minimal plausibility check: biological
plausibility requires study/species context. BLQ/censored likelihoods are not supported
by fitting; do not silently treat BLQ substitutions as uncensored measurements.

## Graphics and MCP

Matplotlib produces linear and semilog concentration-time profiles, residuals versus
time, observed-versus-predicted concentration, and separate comparison AIC/RMSE plots.
Multiple-dose profiles mark administration times. Connected samples do not resolve the
instantaneous bolus jump. Zeros are omitted from semilog plots with a warning; no
pseudo-concentration is inserted. PNGs are returned as portable base64 payloads in
Python and as embedded MCP image blocks plus structured metadata. Output files are
optional. Local stdio MCP has eight tools, with SDK adapters separate from the
numerical engine. No remote hosting or native hosted-assistant integration is provided.

## Validation evidence

Run `.venv/bin/python -m pytest -q`. Independent references include exponential IV
bolus, finite infusion on/off solutions, Bateman absorption (including equal rates),
a biexponential 2C solution derived from microconstants, explicit 3C amount/elimination
ODEs integrated independently with tight tolerances, polynomial first moments and
closed-form exponential NCA integrals. Synthetic recovery tests cover one/two/three
compartments, infusion, repeated dosing and oral lag. Additional checks cover mass,
linearity, superposition, BLQ handling, interval coverage, invalid inputs, likelihood,
analytical Jacobian covariance, nonconvergence and singular sensitivity. Real stdio
clients exercise every MCP tool; image tests decode PNGs and verify rendered pixels.

The v1.0 implementation was validated on Python 3.12.14, NumPy 2.5.4, SciPy
1.18.1, Matplotlib 3.11.2, MCP SDK 2.3.0 and pytest 9.1.1: **75 tests passed**.
Both executable examples and `pip check` passed, and a distributable 1.0.0 wheel
was built. Dependency ranges remain declared in pyproject.toml; this evidence does
not claim that every permitted dependency combination or Python version was tested.
