# OpenPKPD

OpenPKPD is a small Python toolkit for deterministic pharmacokinetic (PK) and pharmacodynamic (PD) calculations. It also exposes those calculations as four discoverable tools through the official MCP Python SDK over local stdio.

## Capabilities

- One-compartment IV bolus and first-order oral absorption PK profiles, with analytical AUC∞ and profile Cmax/Tmax.
- Noncompartmental observed Cmax/Tmax, linear-trapezoidal AUClast, and a qualified log-linear terminal fit. Diagnostics state whether terminal parameters were accepted and why; AUC∞ is withheld when extrapolation exceeds 20%.
- Emax and sigmoid Emax concentration-effect calculations.
- Repeated IV or oral doses by linear superposition.

All outputs carry units and model assumptions. Inputs are never inferred or fitted by the server.

## Installation

Python 3.11 or newer is required. From a checkout:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

Core units are dose in mg, clearance in L/h, volume in L, time in h, concentration in mg/L, rate constants in 1/h, and AUC in mg*h/L. Oral bioavailability must be in (0, 1].

## MCP setup

The package uses the official MCP Python SDK stable 2.x line. Start the local stdio server with `openpkpd-mcp` after installation. Example Claude Desktop configuration (use the absolute path to the checkout and adjust the operating system path syntax):

```json
{
  "mcpServers": {
    "openpkpd": {
      "command": "/absolute/path/to/openpkpd/.venv/bin/openpkpd-mcp",
      "args": []
    }
  }
}
```

For other MCP clients, configure a local stdio server with the command above. This project implements local stdio only. It does not provide a remote hosted integration, so native ChatGPT compatibility is not claimed.

The four tool names are `simulate_pk`, `calculate_nca`, `simulate_pd`, and `simulate_multiple_doses`.

Example prompts:

- “Simulate an IV bolus profile for 100 mg, clearance 5 L/h, and volume 50 L for 24 hours.”
- “Calculate observed NCA metrics from these time points and concentrations. Report terminal fit diagnostics.”
- “Calculate sigmoid Emax effects at concentrations 0, 1, 2, and 10 mg/L with Emax 80, EC50 2 mg/L, and Hill coefficient 1.5.”
- “Simulate six oral doses of 100 mg every 12 hours with bioavailability 0.8 and ka 1.2 per hour.”

## Reproducible calculation

With the package installed, run `python examples/reproduce.py`. For an IV bolus of 100 mg with clearance 5 L/h, analytical AUC∞ is `Dose/CL = 20 mg*h/L`, and initial concentration is `Dose/V`. For an oral dose of 100 mg, bioavailability 0.8, and clearance 5 L/h, AUC∞ is `F*Dose/CL = 16 mg*h/L`.

## Assumptions and limitations

The PK model assumes one compartment, linear and time-invariant disposition, and first-order elimination; oral doses use first-order absorption and a fixed bioavailability. Multiple-dose calculations assume equal doses and linear superposition. The PD model assumes instantaneous equilibrium with no effect-site delay, tolerance, or hysteresis. NCA uses observed data with linear trapezoids and a log-linear terminal regression over the final requested positive observations after observed Cmax. Its thresholds (at least three points, strictly declining concentrations, R² ≥ 0.90, and at most 20% AUC extrapolation) are transparent screening criteria, not universal regulatory rules. Sparse, noisy, or otherwise unsuitable data may prevent terminal estimates.

The toolkit does not select or invent parameters, fit population models, account for uncertainty, or provide individualized clinical dosing recommendations. It is educational and computational software, not a substitute for scientific review or clinical judgment.

## Tests

```sh
pytest
```

## Contributing

Contributions should preserve deterministic calculations, explicit units and assumptions, input validation, and independent tests against analytical reference cases. Please include the scientific rationale and a reproducible example for changes to model behavior.

## Scientific disclaimer

OpenPKPD is provided for research, education, and software development. It has not been validated for clinical decision-making. Users are responsible for checking inputs, assumptions, and results for their intended use.
