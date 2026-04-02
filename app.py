import streamlit as st
import pandas as pd
import math

# --- 1. Load the Database ---
@st.cache_data
def load_data():
    return pd.read_csv('cfd_database.csv')

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: Please make sure 'cfd_database.csv' is in the same folder as this script.")
    st.stop()

# --- 2. The Math Engine (Boundary Layer Physics) ---
def calculate_mesh_blueprint(velocity, length=1.0):
    density = 1.225        # Air at sea level (kg/m^3)
    viscosity = 1.789e-5   # Dynamic viscosity (kg/m*s)
    target_yplus = 1.0
    growth_rate = 1.15
    
    # Calculate Reynolds & Skin Friction (1/7th Power Law)
    reynolds = (density * velocity * length) / viscosity
    cf = 0.026 * math.pow(reynolds, -1/7)
    
    # Calculate First Cell Height
    tau_w = 0.5 * cf * density * math.pow(velocity, 2)
    u_tau = math.sqrt(tau_w / density)
    first_cell_m = (target_yplus * viscosity) / (density * u_tau)
    first_cell_mm = first_cell_m * 1000
    
    # Calculate Total Layers required
    delta = (0.37 * length) / math.pow(reynolds, 0.2)
    try:
        core_math = 1 - ((delta * (1 - growth_rate)) / first_cell_m)
        total_layers = math.ceil(math.log(core_math) / math.log(growth_rate)) if core_math > 0 else 30
    except:
        total_layers = 30
        
    return reynolds, first_cell_mm, total_layers

# --- 3. The Web Interface ---
st.title("Ansys V&V Meshing Engine")
st.write("Target: **NACA 2412 Validation**")
st.divider()

# Section A: Historical Targets
st.subheader("1. Aerodynamic Validation Targets")
selected_flow = st.selectbox("Select Wind-Tunnel Test Speed", df['Flow Velocity'].unique())

target_data = df[df['Flow Velocity'] == selected_flow].iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("Max Lift (Cl_max)", target_data['Max Lift Coefficient'])
col2.metric("Stall Angle", f"{target_data['Stall Angle']}°")
col3.metric("Zero-Lift Drag", target_data['Zero-Lift Drag'])

st.divider()

# Section B: The Meshing Blueprint
st.subheader("2. Generate Ansys Mesh Settings")
v_input = st.number_input("Input Freestream Velocity (m/s)", value=44.0, step=1.0)
c_input = st.number_input("Input Chord Length (m)", value=1.0, step=0.1)

if st.button("Generate Mesh Blueprint"):
    calc_re, calc_dy, calc_layers = calculate_mesh_blueprint(v_input, c_input)
    
    st.success(f"Calculated Flight Reynolds Number: {calc_re:.2e}")
    
    out1, out2 = st.columns(2)
    with out1:
        st.markdown("### Meshing Module Inputs")
        st.write(f"**First Cell Height:** `{calc_dy:.5f} mm`")
        st.write(f"**Growth Rate:** `1.15`")
        st.write(f"**Total Inflation Layers:** `{calc_layers}`")
    with out2:
        st.markdown("### Fluent Solver Inputs")
        st.write(f"**Turbulence Model:** `{target_data['Recommended Turbulence Model']}`")
        st.write(f"**Target y+:** `<= {target_data['Target y+ Value']}`")