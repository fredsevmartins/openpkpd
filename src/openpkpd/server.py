"""Local MCP adapters. Numerical engines do not depend on the MCP SDK."""
from __future__ import annotations
from typing import Any
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent, ImageContent
from .pk import simulate_pk, simulate_multiple_doses
from .pd import simulate_pd
from .nca import calculate_nca
from .fitting import fit_pk_model, compare_pk_models, diagnose_pk_fit
from . import generate_pk_plots

server = MCPServer('openpkpd')


@server.tool(name='simulate_pk', structured_output=True)
def simulate_pk_tool(dose_mg: float, clearance_L_h: float, volume_L: float,
                     bioavailability: float = 1.0, ka_per_h: float | None = None,
                     duration_h: float = 24.0, n_points: int = 201, route: str = 'iv',
                     n_compartments: int = 1, infusion_duration_h: float | None = None,
                     lag_time_h: float = 0.0, q_L_h: float | None = None, vp_L: float | None = None,
                     q2_L_h: float | None = None, vp2_L: float | None = None,
                     q3_L_h: float | None = None, vp3_L: float | None = None) -> dict[str, Any]:
    """Simulate mammillary PK (1–3 compartments). Routes: iv, infusion, oral. Units: mg, L/h, L, h, mg/L; ka in 1/h. Two compartments require q/vp; three require q2/vp2/q3/vp3."""
    return simulate_pk(**locals())


@server.tool(name='calculate_nca', structured_output=True)
def calculate_nca_tool(time_h: list[float], concentration_mg_L: list[float], terminal_points: int = 3,
                       integration_method: str = 'linear', route: str | None = None,
                       dose_mg: float | None = None, infusion_duration_h: float | None = None,
                       terminal_selection: str = 'fixed', terminal_indices: list[int] | None = None,
                       blq: list[bool] | None = None, blq_rule: str = 'exclude', lloq_mg_L: float | None = None,
                       dose_interval_h: float | None = None, interval_start_h: float = 0.0,
                       steady_state: bool = False, first_interval_auc_mg_h_L: float | None = None) -> dict[str, Any]:
    """NCA with units and diagnostics. Integration: linear or linear_up_log_down. Terminal selection: fixed/auto/manual (original row indices). BLQ: exclude/zero/half_lloq. Dose and route needed for clearance; oral estimates are apparent. Steady state requires measured interval endpoints; end sample must be predose."""
    return calculate_nca(**locals())


@server.tool(name='simulate_pd', structured_output=True)
def simulate_pd_tool(concentrations_mg_L: list[float], emax: float, ec50_mg_L: float,
                     baseline: float = 0.0, hill_coefficient: float = 1.0, model: str = 'emax') -> dict[str, Any]:
    """Calculate Emax or sigmoid Emax effects; concentrations and EC50 in mg/L."""
    return simulate_pd(concentrations_mg_L,emax,ec50_mg_L,baseline,hill_coefficient,model)


@server.tool(name='simulate_multiple_doses', structured_output=True)
def simulate_multiple_doses_tool(dose_mg: float, clearance_L_h: float, volume_L: float,
                                 dose_interval_h: float, n_doses: int, bioavailability: float = 1.0,
                                 ka_per_h: float | None = None, points_per_interval: int = 101,
                                 route: str = 'iv', n_compartments: int = 1,
                                 infusion_duration_h: float | None = None, lag_time_h: float = 0.0,
                                 q_L_h: float | None = None, vp_L: float | None = None,
                                 q2_L_h: float | None = None, vp2_L: float | None = None,
                                 q3_L_h: float | None = None, vp3_L: float | None = None) -> dict[str, Any]:
    """Repeated equal doses by superposition for 1–3 compartments and iv/infusion/oral routes. Units: mg, L/h, L, h. IV bolus samples at dose times are post-dose, not troughs."""
    return simulate_multiple_doses(**locals())


@server.tool(name='fit_pk_model', structured_output=True)
def fit_pk_model_tool(time_h: list[float], concentration_mg_L: list[float], dose_mg: float,
                      initial_parameters: dict[str,float], n_compartments: int = 1, route: str = 'iv',
                      bounds: dict[str,list[float]] | None = None, weighting: str = 'uniform',
                      sigma_mg_L: list[float] | None = None, bioavailability: float | None = None,
                      infusion_duration_h: float | None = None, dose_times_h: list[float] | None = None,
                      max_nfev: int = 2000) -> dict[str, Any]:
    """Bounded PK fitting. Parameter keys: CL,Vc; add Q,Vp for 2C or Q2,Vp2,Q3,Vp3 for 3C; oral requires ka, optional lag_time_h. Units: L/h, L, 1/h, h. Oral without externally known F estimates apparent CL/F and volumes/F. Gaussian weighting: uniform, relative (positive observed C), known_sigma. Reports residuals, convergence, conditional uncertainty and identifiability warnings."""
    return fit_pk_model(**locals())


@server.tool(name='compare_pk_models', structured_output=True)
def compare_pk_models_tool(fits: list[dict]) -> dict[str, Any]:
    """Compare complete fit_pk_model results only for identical data, dosing and Gaussian error model. AIC/AICc/BIC do not establish scientific validity."""
    return compare_pk_models(fits)


@server.tool(name='diagnose_pk_fit', structured_output=True)
def diagnose_pk_fit_tool(fit: dict) -> dict[str, Any]:
    """Diagnose a complete fit_pk_model result: residual structure, positivity and identifiability. No automatic scientific approval."""
    return diagnose_pk_fit(fit)


@server.tool(name='generate_pk_plots')
def generate_pk_plots_tool(profile: dict | None = None, comparison: dict | None = None,
                           dpi: int = 120) -> CallToolResult:
    """Portable PNG images: linear/semilog concentration, residuals, observed vs predicted and model comparison. Supply simulation/fit and/or comparison results. Images are embedded; no external hosting required."""
    result = generate_pk_plots(profile,comparison,dpi=dpi)
    metadata = {**result,'images':[{'name':i['name'],'mime_type':i['mime_type']} for i in result['images']]}
    content = [TextContent(type='text',text='Scientific plots: '+', '.join(i['name'] for i in result['images']))]
    content += [ImageContent(type='image',data=i['data_base64'],mimeType='image/png') for i in result['images']]
    return CallToolResult(content=content,structured_content=metadata)


def main() -> None:
    server.run(transport='stdio')


if __name__ == '__main__':
    main()
