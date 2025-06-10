import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import newton

# Callendar-Van Dusen coefficients for platinum (IEC 60751)
A = 3.9083e-3  # Linear coefficient (°C⁻¹)
B = -5.775e-7 # Quadratic coefficient (°C⁻²)
C = -4.183e-12 # Cubic coefficient (°C⁻³), used for t < 0°C

# RTD base resistances
R0_values = {
    'PT100': 100.0,
    'PT1000': 1000.0
}

# Function to compute resistance from temperature (°C)
def resistance_from_temperature(t, R0):
    if t >= 0:
        return R0 * (1 + A*t + B*t**2)
    else:
        return R0 * (1 + A*t + B*t**2 + C*(t-100)*t**3)

# Function to compute temperature from resistance via root finding
def temperature_from_resistance(R, R0, initial_guess=0.0):
    def f(t):
        return resistance_from_temperature(t, R0) - R
    try:
        return newton(f, initial_guess, tol=1e-6, maxiter=50)
    except RuntimeError:
        return None

# Streamlit UI configuration
st.set_page_config(page_title="RTD Calculator", layout="centered")
st.title("RTD Resistance ↔ Temperature Calculator")

# Sidebar controls
sensor = st.sidebar.selectbox("Select RTD Type:", list(R0_values.keys()))
mode = st.sidebar.radio("Calculate:", ["Resistance from Temperature", "Temperature from Resistance"])
R0 = R0_values[sensor]

# Main calculation
if mode == "Resistance from Temperature":
    t = st.number_input("Temperature (°C)", value=0.0, format="%.2f")
    R = resistance_from_temperature(t, R0)
    st.write(f"## Resistance for {sensor} at {t:.2f} °C:")
    st.success(f"{R:.3f} Ω")
else:
    R_input = st.number_input("Resistance (Ω)", value=R0, format="%.3f")
    t_est = temperature_from_resistance(R_input, R0)
    st.write(f"## Temperature for {sensor} at {R_input:.3f} Ω:")
    if t_est is not None:
        st.success(f"{t_est:.3f} °C")
    else:
        st.error("Could not converge to a solution. Please check the resistance value.")

st.markdown("---")

# Coefficients table and explanation
st.header("Calculation Details & Coefficients")
# Display coefficients in a table
coeffs = {
    'Coefficient': ['A', 'B', 'C'],
    'Value': [A, B, C],
    'Unit': ['°C⁻¹', '°C⁻²', '°C⁻³'],
    'Description': [
        'Linear term for all temperatures',
        'Quadratic term for all temperatures',
        'Cubic term, applied only for temperatures below 0°C'
    ]
}
df_coeff = pd.DataFrame(coeffs)
st.table(df_coeff)

# Info about calculation method
with st.expander("How the calculator works"):
    st.write(
        "This app uses the Callendar–Van Dusen equation defined in IEC 60751 to relate resistance and temperature for platinum RTDs. "
        "For `t >= 0°C`, the cubic coefficient `C` is omitted. To find resistance from temperature, the equation is applied directly. "
        "To find temperature from resistance, a Newton-Raphson root-finding method is used to solve for `t` where `R(t) - R_input = 0`.")
    st.info(
        "To use different coefficients (e.g., for custom RTD types), modify the variables `A`, `B`, and `C` at the top of this script. "
        "Ensure your coefficients match your sensor's calibration standard for accurate results.")

st.caption("Powered by IEC 60751 Callendar–Van Dusen formula")
