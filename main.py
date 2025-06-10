import streamlit as st
import numpy as np
from scipy.optimize import newton

# Callendar-Van Dusen coefficients for platinum
A = 3.9083e-3
B = -5.775e-7
C = -4.183e-12

# RTD base resistances
R0_values = {
    'PT100': 100.0,
    'PT1000': 1000.0
}

# Function to compute resistance from temperature (°C)
def resistance_from_temperature(t, R0):
    """
    Callendar-Van Dusen equation for temperatures between -200°C and 0°C:
    R = R0 * (1 + A*t + B*t^2 + C*(t-100)*t^3)
    For t >= 0°C: C term is zero.
    """
    if t >= 0:
        return R0 * (1 + A*t + B*t**2)
    else:
        return R0 * (1 + A*t + B*t**2 + C*(t-100)*t**3)

# Function to compute temperature from resistance via root finding
def temperature_from_resistance(R, R0, initial_guess=0.0):
    # Define f(t) = resistance_from_temperature(t) - R
    def f(t):
        return resistance_from_temperature(t, R0) - R
    # Use Newton-Raphson to find root
    try:
        t_est = newton(f, initial_guess, tol=1e-6, maxiter=50)
        return t_est
    except RuntimeError:
        return None

# Streamlit UI
st.set_page_config(page_title="RTD Calculator", layout="centered")
st.title("RTD Resistance ↔ Temperature Calculator")

# Sidebar configuration
sensor = st.sidebar.selectbox("Select RTD Type:", list(R0_values.keys()))
mode = st.sidebar.radio("Calculate:", ["Resistance from Temperature", "Temperature from Resistance"])

R0 = R0_values[sensor]

if mode == "Resistance from Temperature":
    t = st.number_input("Temperature (°C)", value=0.0, format="%.2f")
    R = resistance_from_temperature(t, R0)
    st.write(f"## Resistance for {sensor} at {t:.2f} °C:")
    st.success(f"{R:.3f} Ω")

else:
    R_input = st.number_input("Resistance (Ω)", value=R0, format="%.3f")
    t_est = temperature_from_resistance(R_input, R0, initial_guess=0.0)
    st.write(f"## Temperature for {sensor} at {R_input:.3f} Ω:")
    if t_est is not None:
        st.success(f"{t_est:.3f} °C")
    else:
        st.error("Could not converge to a solution. Please check the resistance value.")

st.markdown("---")
st.caption("Calculator based on IEC 60751 Callendar-Van Dusen equation")
