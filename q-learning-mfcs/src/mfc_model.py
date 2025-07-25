
# With the July 2025 API, compiling the Mojo file creates a standard
# Python module. The legacy `max.mojo.importer` is no longer needed.


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "")
from  odes  import MFCModel # Import the Mojo module compiled from odes.mojo
# 1. Initialize the Mojo model struct
mfc = MFCModel()  # Create an instance of the MFCModel

# 2. Define initial conditions for the state variables [y0]
# C_AC, C_CO2, C_H, X, C_O2, C_OH, C_M, eta_a, eta_c
y0 = [
    1.5,      # C_AC: Start with near-influent concentration
    0.1,      # C_CO2: Small initial amount
    1e-4,     # C_H: Corresponds to pH 4
    0.1,      # X: Initial biomass
    0.3,      # C_O2: Start with near-influent concentration
    1e-7,     # C_OH: Corresponds to pH 7
    0.1,      # C_M: Small initial amount
    0.01,     # eta_a: Small initial anodic overpotential
    -0.01     # eta_c: Small initial cathodic overpotential
]

# 3. Define simulation time span
t_span = [0, 100]  # Simulate for 100 hours
t_eval = np.linspace(t_span[0], t_span[1], 500) # Points to evaluate solution

# 4. Set a constant current density for this example
constant_i_fc = 1.0 # A/m²

# 5. Run the ODE solver, passing the Mojo function as the "fun" argument.
#    The core calculations are now executed by the high-performance Mojo code.
solution = solve_ivp(
    fun=lambda t, y: mfc.mfc_odes(t, y, constant_i_fc),
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method='RK45'
)

# 6. Plot the results using Matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(solution.t, solution.y[0], label='Acetate Concentration (C_AC)', color='b')
ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Concentration (mol/m³)', fontsize=12)
ax.set_title('Mojo-Powered MFC Simulation: Acetate Concentration', fontsize=14)
ax.legend()
ax.grid(True)
plt.show()