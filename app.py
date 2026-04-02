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
def calculate_mesh_blueprint(velocity, length=1.0, target_yplus=1.0, growth_rate=1.15):
    density = 1.225        # Air at sea level (kg/m^3)
    viscosity = 1.789e-5   # Dynamic viscosity (kg/m*s)

    # Reynolds & Skin Friction (1/7th Power Law)
    reynolds = (density * velocity * length) / viscosity
    cf = 0.026 * math.pow(reynolds, -1/7)

    # First Cell Height
    tau_w = 0.5 * cf * density * math.pow(velocity, 2)
    u_tau = math.sqrt(tau_w / density)
    first_cell_m = (target_yplus * viscosity) / (density * u_tau)
    first_cell_mm = first_cell_m * 1000

    # Boundary Layer Thickness (Prandtl 1/7th power law)
    delta = (0.37 * length) / math.pow(reynolds, 0.2)
    delta_mm = delta * 1000

    # Total Inflation Layers
    try:
        core_math = 1 - ((delta * (1 - growth_rate)) / first_cell_m)
        total_layers = math.ceil(math.log(core_math) / math.log(growth_rate)) if core_math > 0 else 30
    except:
        total_layers = 30

    # Estimate actual y+ achieved (for buffer zone check)
    actual_yplus = (density * u_tau * first_cell_m) / viscosity

    return reynolds, first_cell_mm, total_layers, delta_mm, actual_yplus

def get_yplus_zone(yplus):
    if yplus <= 5:
        return "viscous_sublayer"
    elif yplus <= 30:
        return "buffer"
    else:
        return "log_law"

def get_model_recommendation(yplus_mode):
    if yplus_mode == "near_wall":
        return {
            "primary": "k-ω SST",
            "secondary": "Spalart-Allmaras",
            "wall_treatment": "Near-Wall Modeling",
            "note": "Resolves viscous sublayer fully. Best for adverse pressure gradients and stall prediction."
        }
    else:
        return {
            "primary": "Realizable k-ε",
            "secondary": "RSM (Reynolds Stress Model)",
            "wall_treatment": "Standard Wall Function (SWF)",
            "note": "Wall functions bridge to log-law region. Faster solve, suitable for attached flows and bluff bodies."
        }

# --- Extended Airfoil Database (built-in) ---
AIRFOIL_INFO = {
    "NACA 2412": {
        "description": "Classic cambered airfoil — general aviation, moderate lift",
        "recommended_yplus": 1.0,
        "reason": "Adverse pressure gradient near TE; stall separation must be resolved",
        "thickness": "12%",
        "camber": "2%"
    },
    "NACA 0012": {
        "description": "Symmetric airfoil — control surfaces, helicopter blades",
        "recommended_yplus": 1.0,
        "reason": "Clean attached flow but symmetric stall needs near-wall resolution",
        "thickness": "12%",
        "camber": "0%"
    },
    "NACA 4412": {
        "description": "High-camber airfoil — high-lift applications",
        "recommended_yplus": 1.0,
        "reason": "Strong suction-side pressure gradient; separation bubble likely at high AoA",
        "thickness": "12%",
        "camber": "4%"
    },
    "NACA 23012": {
        "description": "Reflex camber — older transport aircraft",
        "recommended_yplus": 1.0,
        "reason": "Complex pressure distribution; near-wall resolution ensures accuracy",
        "thickness": "12%",
        "camber": "2.3%"
    },
    "Flat Plate / Bluff Body": {
        "description": "Surface-mounted obstacle / flat plate — as per Salim & Cheah (2009)",
        "recommended_yplus": 30.0,
        "reason": "Geometry-forced separation; log-law region sufficient (paper validated y⁺=30–60)",
        "thickness": "N/A",
        "camber": "N/A"
    }
}

# --- 3. Web Interface ---
st.set_page_config(page_title="Ansys CFD Meshing Engine", page_icon="✈️", layout="wide")

st.title("✈️ Ansys CFD Meshing Engine")
st.write("**Airfoil Validation & Mesh Blueprint Generator** — Based on Wall y⁺ Strategy (Salim & Cheah, 2009)")
st.divider()

# ── Sidebar: Airfoil Selector ──────────────────────────────────────────
with st.sidebar:
    st.header("🛩️ Airfoil Selection")
    selected_airfoil = st.selectbox("Select Airfoil Geometry", list(AIRFOIL_INFO.keys()))

    info = AIRFOIL_INFO[selected_airfoil]
    st.info(f"**{selected_airfoil}**\n\n{info['description']}")
    st.write(f"- **Thickness:** {info['thickness']}")
    st.write(f"- **Camber:** {info['camber']}")
    st.markdown("---")
    st.markdown("##### 📖 y⁺ Zone Reference")
    st.markdown("""
| Zone | y⁺ Range | Use With |
|---|---|---|
| Viscous sublayer | ≤ 5 | k-ω SST, S-A |
| ⚠️ Buffer layer | 5–30 | **AVOID** |
| Log-law region | 30–60 | k-ε, RSM |
""")

# ── Section A: Aerodynamic Targets ──────────────────────────────────────
st.subheader("1. 📊 Aerodynamic Validation Targets")

airfoil_df = df[df['Geometry Type'] == selected_airfoil] if selected_airfoil in df['Geometry Type'].values else pd.DataFrame()

if not airfoil_df.empty:
    selected_flow = st.selectbox("Select Wind-Tunnel Test Speed", airfoil_df['Flow Velocity'].unique())
    target_data = airfoil_df[airfoil_df['Flow Velocity'] == selected_flow].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Lift (Cl_max)", target_data['Max Lift Coefficient'])
    col2.metric("Stall Angle", f"{target_data['Stall Angle']}°")
    col3.metric("Zero-Lift Drag", target_data['Zero-Lift Drag'])
    col4.metric("Validation Source", target_data['Validation Source'])
else:
    st.warning(f"No wind-tunnel data in CSV for **{selected_airfoil}**. Mesh blueprint can still be generated below using physics-based calculations.")
    target_data = None

st.divider()

# ── Section B: y⁺ Strategy Selector ─────────────────────────────────────
st.subheader("2. 🎯 Wall y⁺ Strategy")

recommended_yplus = info['recommended_yplus']

col_l, col_r = st.columns([1, 1])
with col_l:
    yplus_mode = st.radio(
        "Select Near-Wall Treatment Approach",
        options=["near_wall", "wall_function"],
        format_func=lambda x: "🔬 Near-Wall Modeling  (y⁺ ≈ 1)  — Resolve viscous sublayer"
                               if x == "near_wall"
                               else "⚡ Wall Functions  (y⁺ ≈ 30–60)  — Log-law region only",
        index=0 if recommended_yplus <= 5 else 1
    )

target_yplus = 1.0 if yplus_mode == "near_wall" else 30.0

with col_r:
    rec = get_model_recommendation(yplus_mode)
    if yplus_mode == "near_wall":
        st.success(f"""
**✅ Recommended for {selected_airfoil}**

- **Primary Model:** {rec['primary']}
- **Alternative:** {rec['secondary']}
- **Wall Treatment:** {rec['wall_treatment']}

_{rec['note']}_
""")
    else:
        st.info(f"""
**ℹ️ Wall Function Approach**

- **Primary Model:** {rec['primary']}
- **Alternative:** {rec['secondary']}
- **Wall Treatment:** {rec['wall_treatment']}

_{rec['note']}_
""")

# Warn if user picks wall functions for an airfoil that needs near-wall
if yplus_mode == "wall_function" and recommended_yplus <= 5:
    st.warning(f"⚠️ **Caution:** {selected_airfoil} is recommended to use y⁺ ≈ 1 (near-wall modeling). "
               f"Wall functions may under-predict stall angle and skin friction. "
               f"Reason: {info['reason']}")

st.divider()

# ── Section C: Mesh Blueprint Generator ──────────────────────────────────
st.subheader("3. ⚙️ Generate Ansys Mesh Blueprint")

col_v, col_c, col_gr = st.columns(3)
with col_v:
    v_input = st.number_input("Freestream Velocity (m/s)", value=44.0, step=1.0, min_value=1.0)
with col_c:
    c_input = st.number_input("Chord Length (m)", value=1.0, step=0.1, min_value=0.01)
with col_gr:
    growth_rate = st.number_input("Growth Rate", value=1.15, step=0.01, min_value=1.05, max_value=1.5,
                                   help="Recommended: 1.1 – 1.2. Higher = fewer layers but worse quality.")

if st.button("🚀 Generate Mesh Blueprint", type="primary"):
    calc_re, calc_dy, calc_layers, calc_delta, actual_yplus = calculate_mesh_blueprint(
        v_input, c_input, target_yplus, growth_rate
    )

    zone = get_yplus_zone(actual_yplus)

    st.success(f"**Calculated Reynolds Number: {calc_re:.2e}**")

    # ── Outputs: Two columns ──
    col_mesh, col_solver = st.columns(2)

    with col_mesh:
        st.markdown("### 🔧 Meshing Module (Ansys Meshing)")
        st.write(f"**First Cell Height:** `{calc_dy:.5f} mm`")
        st.write(f"**Growth Rate:** `{growth_rate}`")
        st.write(f"**Total Inflation Layers:** `{calc_layers}`")
        st.write(f"**Boundary Layer Thickness (δ):** `{calc_delta:.3f} mm`")
        st.write(f"**Inflation Algorithm:** `Pre`")
        st.write(f"**Transition Ratio:** `0.272`")

    with col_solver:
        st.markdown("### 💻 Fluent Solver Settings")
        st.write(f"**Primary Turbulence Model:** `{rec['primary']}`")
        st.write(f"**Alternative Model:** `{rec['secondary']}`")
        st.write(f"**Near-Wall Treatment:** `{rec['wall_treatment']}`")
        st.write(f"**Target y⁺:** `{target_yplus}`")
        if target_data is not None:
            st.write(f"**Expected Cl_max:** `{target_data['Max Lift Coefficient']}`")
            st.write(f"**Expected Stall Angle:** `{target_data['Stall Angle']}°`")

    st.divider()

    # ── Mesh Quality Checklist ──
    st.markdown("### ✅ Mesh Quality Checklist")

    checks = []

    # y+ zone check
    if zone == "buffer":
        checks.append(("❌", "y⁺ Zone", f"**BUFFER ZONE DETECTED (y⁺ ≈ {actual_yplus:.1f})** — Neither wall functions nor near-wall modeling is accurate here. Refine mesh (lower y⁺) or coarsen (y⁺>30)."))
    elif zone == "viscous_sublayer":
        checks.append(("✅", "y⁺ Zone", f"Viscous sublayer resolved (y⁺ ≈ {actual_yplus:.1f}) — Near-wall modeling applicable"))
    else:
        checks.append(("✅", "y⁺ Zone", f"Log-law region resolved (y⁺ ≈ {actual_yplus:.1f}) — Wall functions applicable"))

    # Growth rate check
    if 1.1 <= growth_rate <= 1.2:
        checks.append(("✅", "Growth Rate", f"{growth_rate} — Within optimal range (1.1–1.2)"))
    elif growth_rate < 1.1:
        checks.append(("⚠️", "Growth Rate", f"{growth_rate} — Very fine; increases layer count significantly"))
    else:
        checks.append(("⚠️", "Growth Rate", f"{growth_rate} — Above 1.2; may cause mesh quality issues near TE"))

    # Layer count check
    if calc_layers < 15:
        checks.append(("⚠️", "Layer Count", f"{calc_layers} layers — May not fully cover boundary layer. Consider reducing growth rate."))
    elif calc_layers > 50:
        checks.append(("⚠️", "Layer Count", f"{calc_layers} layers — High count; check δ coverage and growth rate"))
    else:
        checks.append(("✅", "Layer Count", f"{calc_layers} layers — Adequate to cover full boundary layer (δ = {calc_delta:.3f} mm)"))

    # Model-yplus compatibility check
    if yplus_mode == "near_wall" and zone == "log_law":
        checks.append(("⚠️", "Model Compatibility", "Near-wall model selected but y⁺ is in log-law range — switch to wall functions or reduce first cell height"))
    elif yplus_mode == "wall_function" and zone == "viscous_sublayer":
        checks.append(("⚠️", "Model Compatibility", "Wall functions selected but y⁺ resolves viscous sublayer — wall functions cease to be valid here (Salim & Cheah, 2009)"))
    else:
        checks.append(("✅", "Model Compatibility", f"{rec['primary']} is consistent with y⁺ = {actual_yplus:.1f}"))

    # Reynolds number check
    if calc_re < 5e4:
        checks.append(("⚠️", "Reynolds Number", f"{calc_re:.2e} — Very low Re; consider laminar solver or transition model (γ-Reθ)"))
    elif calc_re > 1e7:
        checks.append(("⚠️", "Reynolds Number", f"{calc_re:.2e} — High Re; ensure mesh is fine enough and RSM considered"))
    else:
        checks.append(("✅", "Reynolds Number", f"{calc_re:.2e} — Turbulent regime, RANS models applicable"))

    for icon, label, msg in checks:
        st.write(f"{icon} **{label}:** {msg}")

    # ── Fluent Journal Snippet ──
    st.divider()
    st.markdown("### 📋 Fluent TUI Reference Snippet")
    model_tui = "kw-sst" if rec['primary'] == "k-ω SST" else \
                "spalart-allmaras" if "Spalart" in rec['primary'] else \
                "ke-realizable" if "Realizable" in rec['primary'] else "rsm"
    st.code(f"""; ── Fluent TUI Commands ──
/define/models/viscous/{model_tui} yes
/define/boundary-conditions/velocity-inlet inlet yes no {v_input} no 0 no 0
; First cell height: {calc_dy:.5f} mm  →  {calc_dy/1000:.6f} m
; Growth rate: {growth_rate}
; Total inflation layers: {calc_layers}
; Target y+: {target_yplus}
""", language="text")