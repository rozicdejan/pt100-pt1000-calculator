import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import newton

# ------------------------------------------------------------
# RTD / PTC / KTY calculator
# Special note:
# - PT100 / PT1000 use IEC 60751 Callendar–Van Dusen formula
# - Kollmorgen AKD/AKM KTY83-110 uses datasheet lookup table
#   because it is NOT the same as Pt1000.
# ------------------------------------------------------------

sensor_params = {
    'PT100': {
        'model': 'cvd',
        'R0': 100.0,
        'A': 3.9083e-3,
        'B': -5.775e-7,
        'C': -4.183e-12,
        'note': 'IEC 60751 Pt100, 100 Ω at 0°C'
    },
    'PT1000': {
        'model': 'cvd',
        'R0': 1000.0,
        'A': 3.9083e-3,
        'B': -5.775e-7,
        'C': -4.183e-12,
        'note': 'IEC 60751 Pt1000, 1000 Ω at 0°C'
    },

    # Kollmorgen AKD / AKM motor option:
    # KTY83-110, nominal 1000 Ω at 25°C.
    # This is useful when AKD/motor temperature value is shown as resistance.
    'Kollmorgen AKD Motor KTY83-110 / PTC1000': {
        'model': 'table',
        'note': 'Kollmorgen AKM/AKD motor temp option, KTY83-110, nominal 1000 Ω at 25°C',
        'temperature_table': np.array([
            -55, -50, -40, -30, -20, -10,
              0,  10,  20,  25,  30,  40,
             50,  60,  70,  80,  90, 100,
            110, 120, 125, 130, 140, 150,
            160, 170, 175
        ], dtype=float),
        'resistance_table': np.array([
             500,  525,  577,  632,  691,  754,
             820,  889,  962, 1000, 1039, 1118,
            1202, 1288, 1379, 1472, 1569, 1670,
            1774, 1882, 1937, 1993, 2107, 2225,
            2346, 2471, 2535
        ], dtype=float)
    },

    # Keep your generic KTY entries, but mark them as approximate.
    # These are not as good as using the real KTY83 table above.
    'Generic KTY Approx': {
        'model': 'cvd',
        'R0': 1000.0,
        'A': 3.84e-3,
        'B': -5.00e-7,
        'C': 0.0,
        'note': 'Approximate only. Not recommended for Kollmorgen motor KTY83-110.'
    },
}


def resistance_from_temperature(t, params):
    model = params.get('model', 'cvd')

    if model == 'table':
        temps = params['temperature_table']
        resistances = params['resistance_table']

        if t < temps[0] or t > temps[-1]:
            raise ValueError(
                f"Temperature out of table range: {temps[0]}°C to {temps[-1]}°C"
            )

        return float(np.interp(t, temps, resistances))

    # CVD / polynomial model
    A = params['A']
    B = params['B']
    C = params['C']
    R0 = params['R0']

    if t >= 0 or C == 0:
        return R0 * (1 + A * t + B * t**2)
    else:
        return R0 * (1 + A * t + B * t**2 + C * (t - 100) * t**3)


def temperature_from_resistance(R, params, initial_guess=25.0):
    model = params.get('model', 'cvd')

    if model == 'table':
        temps = params['temperature_table']
        resistances = params['resistance_table']

        if R < resistances[0] or R > resistances[-1]:
            raise ValueError(
                f"Resistance out of table range: {resistances[0]} Ω to {resistances[-1]} Ω"
            )

        return float(np.interp(R, resistances, temps))

    # CVD / polynomial model
    def f(t):
        return resistance_from_temperature(t, params) - R

    try:
        return float(newton(f, initial_guess, tol=1e-6, maxiter=50))
    except RuntimeError:
        return None


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

st.set_page_config(page_title="RTD / PTC / KTY Calculator", layout="centered")
st.title("RTD / PTC / KTY Resistance ↔ Temperature Calculator")

sensor = st.sidebar.selectbox("Select Sensor Type:", list(sensor_params.keys()))
mode = st.sidebar.radio("Mode:", ["°C → Ω", "Ω → °C"])
params = sensor_params[sensor]

st.info(params.get('note', ''))

if mode == "°C → Ω":
    t = st.number_input("Temperature (°C)", value=25.0, format="%.2f")

    try:
        R = resistance_from_temperature(t, params)
        st.success(f"**{sensor}:** {t:.2f} °C → **{R:.3f} Ω**")
    except ValueError as e:
        st.error(str(e))

else:
    default_r = 1000.0
    if params.get('model') == 'cvd':
        default_r = params.get('R0', 1000.0)

    R_in = st.number_input("Resistance (Ω)", value=default_r, format="%.3f")

    try:
        t_est = temperature_from_resistance(R_in, params)

        if t_est is not None:
            st.success(f"**{sensor}:** {R_in:.3f} Ω → **{t_est:.3f} °C**")
        else:
            st.error("Calculation did not converge. Check input range.")

    except ValueError as e:
        st.error(str(e))


st.markdown("---")

# Quick Kollmorgen examples
if sensor == 'Kollmorgen AKD Motor KTY83-110 / PTC1000':
    st.header("Kollmorgen AKD Motor Quick Check")

    examples = [1146.0, 1592.0]
    rows = []

    for r in examples:
        rows.append({
            "Resistance Ω": r,
            "Approx. temperature °C": temperature_from_resistance(r, params)
        })

    st.table(pd.DataFrame(rows))

    st.warning(
        "For Kollmorgen motor temperature, this is normally winding / internal motor sensor temperature, "
        "not outside motor housing temperature. Do not add +30°C automatically unless you are measuring the motor case externally."
    )


# Parameters / table
st.header("Sensor Coefficients / Tables")

rows = []
for name, p in sensor_params.items():
    if p.get('model') == 'cvd':
        rows.append({
            'Sensor': name,
            'Model': 'CVD / polynomial',
            'R0': p.get('R0'),
            'A': p.get('A'),
            'B': p.get('B'),
            'C': p.get('C'),
            'Note': p.get('note', '')
        })
    else:
        rows.append({
            'Sensor': name,
            'Model': 'Lookup table interpolation',
            'R0': '1000 Ω at 25°C nominal',
            'A': '-',
            'B': '-',
            'C': '-',
            'Note': p.get('note', '')
        })

df_params = pd.DataFrame(rows)
st.table(df_params)


with st.expander("Kollmorgen KTY83-110 table used"):
    k = sensor_params['Kollmorgen AKD Motor KTY83-110 / PTC1000']
    df_kty = pd.DataFrame({
        "Temperature °C": k['temperature_table'],
        "Resistance Ω typical": k['resistance_table']
    })
    st.table(df_kty)


with st.expander("Important notes"):
    st.write("""
- PT1000 is **1000 Ω at 0°C**.
- Kollmorgen KTY83-110 / PTC1000 is **about 1000 Ω at 25°C**.
- Do not calculate Kollmorgen KTY83-110 as Pt1000.
- For your values:
  - 1146 Ω is about 43°C.
  - 1592 Ω is about 92°C using the typical KTY83-110 table.
- Because of sensor tolerance, 1592 Ω can still be considered around the 90°C region.
""")

st.caption("PT sensors use IEC 60751. Kollmorgen motor KTY83-110 uses NXP KTY83/110 typical resistance table.")
