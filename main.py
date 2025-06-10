import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import newton

# Sensor parameters: base resistance and Callendar–Van Dusen coefficients
# For RTDs: A, B, C per IEC 60751; for KTY, C=0 (no cubic term)
sensor_params = {
    'PT100':    {'R0': 100.0,  'A': 3.9083e-3,  'B': -5.775e-7,  'C': -4.183e-12},
    'PT1000':   {'R0': 1000.0, 'A': 3.9083e-3,  'B': -5.775e-7,  'C': -4.183e-12},
    'KTY Class A':    {'R0': 1000.0, 'A': 3.84e-3,    'B': -5.00e-7,   'C': 0.0},
    'KTY 1/3B':        {'R0': 1000.0, 'A': 3.84e-3,    'B': -5.00e-7,   'C': 0.0},
    'KTY 1/5B':        {'R0': 1000.0, 'A': 3.84e-3,    'B': -5.00e-7,   'C': 0.0},
    'KTY 1/10B':       {'R0': 1000.0, 'A': 3.84e-3,    'B': -5.00e-7,   'C': 0.0},
}

# Compute resistance from temperature (°C)
def resistance_from_temperature(t, params):
    A, B, C, R0 = params['A'], params['B'], params['C'], params['R0']
    if t >= 0 or C == 0:
        return R0 * (1 + A*t + B*t**2)
    else:
        return R0 * (1 + A*t + B*t**2 + C*(t-100)*t**3)

# Compute temperature from resistance via root finding
def temperature_from_resistance(R, params, initial_guess=0.0):
    def f(t): return resistance_from_temperature(t, params) - R
    try:
        return newton(f, initial_guess, tol=1e-6, maxiter=50)
    except RuntimeError:
        return None

# Streamlit UI
st.set_page_config(page_title="Multi-RTD Calculator", layout="centered")
st.title("RTD & KTY Resistance ↔ Temperature Calculator")

# Sidebar
sensor = st.sidebar.selectbox("Select Sensor Type:", list(sensor_params.keys()))
mode = st.sidebar.radio("Mode:", ["°C → Ω", "Ω → °C"])
params = sensor_params[sensor]

# Input and calculation
if mode == "°C → Ω":
    t = st.number_input("Temperature (°C)", value=0.0, format="%.2f")
    R = resistance_from_temperature(t, params)
    st.write(f"**{sensor}:** At {t:.2f} °C → {R:.3f} Ω")
else:
    R_in = st.number_input("Resistance (Ω)", value=params['R0'], format="%.3f")
    t_est = temperature_from_resistance(R_in, params)
    if t_est is not None:
        st.write(f"**{sensor}:** At {R_in:.3f} Ω → {t_est:.3f} °C")
    else:
        st.error("Calculation did not converge. Check input range.")

st.markdown("---")

# Parameters table for all sensors
df_params = pd.DataFrame([
    {'Sensor': s, **p} for s, p in sensor_params.items()
])[
    ['Sensor', 'R0', 'A', 'B', 'C']
]
st.header("Sensor Coefficients & Base Resistances")
st.table(df_params)

# Explanation
with st.expander("How to adjust coefficients and add sensors"):
    st.write("""
- Each sensor is defined by its base resistance `R0` at 0°C and coefficients `A`, `B`, and `C` in the Callendar–Van Dusen model.
- For KTY sensors (`C=0`), only the quadratic term is used.
- To add new sensors or custom coefficients, modify the `sensor_params` dictionary at the top.
- Ensure coefficients match your sensor datasheet for accurate results.
""")

st.caption("Based on IEC 60751 for Pt RTDs and manufacturer specs for KTY types.")
