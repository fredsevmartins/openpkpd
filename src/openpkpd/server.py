"""MCP stdio interface for OpenPKPD scientific functions."""
from __future__ import annotations
import json
from mcp.server.mcpserver import MCPServer
from .pk import simulate_pk, simulate_multiple_doses
from .pd import simulate_pd
from .nca import calculate_nca

server = MCPServer("openpkpd")

@server.tool(name="simulate_pk")
def simulate_pk_tool(dose_mg: float, clearance_L_h: float, volume_L: float,
                     bioavailability: float = 1.0, ka_per_h: float | None = None,
                     duration_h: float = 24.0, n_points: int = 201,
                     route: str = "iv") -> dict:
    """Simulate one-compartment IV bolus or first-order oral PK. Units: mg, L/h, L, h, mg/L."""
    return simulate_pk(dose_mg, clearance_L_h, volume_L, bioavailability, ka_per_h, duration_h, n_points, route)

@server.tool(name="calculate_nca")
def calculate_nca_tool(time_h: list[float], concentration_mg_L: list[float], terminal_points: int = 3) -> dict:
    """Calculate observed NCA and qualified terminal metrics. Inputs use h and mg/L."""
    return calculate_nca(time_h, concentration_mg_L, terminal_points)

@server.tool(name="simulate_pd")
def simulate_pd_tool(concentrations_mg_L: list[float], emax: float, ec50_mg_L: float,
                     baseline: float = 0.0, hill_coefficient: float = 1.0,
                     model: str = "emax") -> dict:
    """Calculate Emax or sigmoid Emax effect over concentrations in mg/L."""
    return simulate_pd(concentrations_mg_L, emax, ec50_mg_L, baseline, hill_coefficient, model)

@server.tool(name="simulate_multiple_doses")
def simulate_multiple_doses_tool(dose_mg: float, clearance_L_h: float, volume_L: float,
                                 dose_interval_h: float, n_doses: int,
                                 bioavailability: float = 1.0, ka_per_h: float | None = None,
                                 points_per_interval: int = 101, route: str = "iv") -> dict:
    """Simulate repeated equal IV or oral doses using linear superposition."""
    return simulate_multiple_doses(dose_mg, clearance_L_h, volume_L, dose_interval_h, n_doses,
                                   bioavailability, ka_per_h, points_per_interval, route)


def main() -> None:
    server.run(transport="stdio")

if __name__ == "__main__":
    main()
