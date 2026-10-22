"""Reproducible calculations using public OpenPKPD functions."""
from openpkpd import simulate_pk, simulate_multiple_doses, simulate_pd

iv = simulate_pk(100, 5, 50, duration_h=24, n_points=5, route="iv")
print("IV AUCinf (expected 20 mg*h/L):", iv["aucinf_mg_h_L"])
print("IV C0 (expected 2 mg/L):", iv["concentration_mg_L"][0])

oral = simulate_pk(100, 5, 50, bioavailability=0.8, ka_per_h=1.2,
                   duration_h=24, n_points=241, route="oral")
print("Oral AUCinf (expected 16 mg*h/L):", oral["aucinf_mg_h_L"])
print("Oral profile Cmax/Tmax:", oral["cmax_mg_L"], oral["tmax_h"])

repeat = simulate_multiple_doses(100, 5, 50, 12, 4, route="iv")
print("Repeated-dose profile points:", len(repeat["time_h"]))
print("Emax at EC50 (expected baseline + Emax/2):",
      simulate_pd([2], emax=10, ec50=2)["effect"][0])
