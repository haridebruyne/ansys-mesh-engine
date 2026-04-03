import streamlit as st
import pandas as pd
import math

# ── 1. Load Database ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('cfd_database.csv')

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'cfd_database.csv' not found. Place it in the same folder.")
    st.stop()

# ── 2. Physics Engines ──────────────────────────────────────────────────

SPEED_OF_SOUND = 340.3   # m/s at sea level, 15 deg C

def get_mach(velocity):
    return velocity / SPEED_OF_SOUND

def get_flow_regime(mach):
    if mach < 0.3:
        return "subsonic"
    elif mach < 0.8:
        return "transonic"
    elif mach < 1.0:
        return "high_transonic"
    else:
        return "supersonic"

def calculate_c_grid_domain(velocity, chord):
    mach   = get_mach(velocity)
    regime = get_flow_regime(mach)

    if regime == "subsonic":
        upstream   = 15.0
        downstream = 20.0
        height     = 20.0
        mesh_type  = "C-mesh (recommended) or O-mesh"
        reason     = ("Pressure disturbance decays as ~1/r in incompressible flow. "
                      "15c upstream gives the inlet negligible influence on the airfoil "
                      "pressure field. 20c downstream fully captures the wake.")
    elif regime == "transonic":
        upstream   = 30.0
        downstream = 40.0
        height     = 30.0
        mesh_type  = "C-mesh with fine wake refinement zone"
        reason     = ("Compressibility increases the effective range of pressure disturbances. "
                      "Shock waves need room to form and dissipate — 30c upstream prevents "
                      "inlet reflection, 40c downstream captures shock-induced wake.")
    elif regime == "high_transonic":
        upstream   = 40.0
        downstream = 50.0
        height     = 35.0
        mesh_type  = "C-mesh with shock-aligned refinement zone"
        reason     = ("Near-sonic: strong normal shocks possible on upper surface. "
                      "Large downstream extent needed to fully capture Mach wave pattern "
                      "before the outlet boundary condition is applied.")
    else:
        upstream   = 10.0
        downstream = 60.0
        height     = 40.0
        mesh_type  = "Structured multi-block with bow-shock standoff region"
        reason     = ("Supersonic: disturbances cannot propagate upstream (Mach cone). "
                      "Small upstream ok, but large downstream needed for bow shock, "
                      "expansion fans, and oblique shock reflections.")

    upstream_m     = upstream   * chord
    downstream_m   = downstream * chord
    height_m       = height     * chord
    total_width_m  = upstream_m + chord + downstream_m
    total_height_m = height_m
    proj_height    = 0.12 * chord
    blockage_pct   = (proj_height / total_height_m) * 100

    return {
        "mach": mach, "regime": regime,
        "upstream_c": upstream, "downstream_c": downstream, "height_c": height,
        "upstream_m": upstream_m, "downstream_m": downstream_m,
        "height_m": height_m, "total_width_m": total_width_m,
        "total_height_m": total_height_m, "blockage_pct": blockage_pct,
        "mesh_type": mesh_type, "reason": reason,
    }

def calculate_square_domain(velocity, chord, bc_type):
    mach   = get_mach(velocity)
    regime = get_flow_regime(mach)
    
    if regime == "subsonic":
        if bc_type == "PVBC":
            A_c = 5.0
            reason = ("Point Vortex BC (PVBC) actively mimics flow at boundaries, allowing "
                      "a much smaller domain (A=5c) with near-zero error (Golmirzaee & Wood, 2024).")
        else:
            A_c = 30.0
            reason = ("Standard Boundaries (Slip/Symmetry). Industry baseline is A=30c. "
                      "Note: Drag error is ~2% unless A=91c is used (Golmirzaee & Wood, 2024).")
    elif regime == "transonic":
        A_c = 50.0
        reason = "Transonic flow requires expanded boundaries (A=50c) to prevent shockwave reflection off walls."
    elif regime == "high_transonic":
        A_c = 100.0
        reason = "High Transonic shockwaves stretch extremely far. Massive domain (A=100c) needed."
    else:
        A_c = 60.0 
        reason = "WARNING: Square grids are highly inefficient for Supersonic flow. A bow-shock aligned grid is strongly recommended."

    A_m = A_c * chord
    total_size_m = 2 * A_m
    proj_height = 0.12 * chord
    blockage_pct = (proj_height / total_size_m) * 100

    return {
        "mach": mach, "regime": regime, "A_c": A_c, "A_m": A_m,
        "total_size_m": total_size_m, "blockage_pct": blockage_pct,
        "reason": reason, "mesh_type": "Square Grid (2A x 2A)"
    }

def calculate_mesh_blueprint(velocity, chord, target_yplus=1.0, growth_rate=1.15):
    density   = 1.225
    viscosity = 1.789e-5
    reynolds      = (density * velocity * chord) / viscosity
    cf            = 0.026 * math.pow(reynolds, -1/7)
    tau_w         = 0.5 * cf * density * math.pow(velocity, 2)
    u_tau         = math.sqrt(tau_w / density)
    first_cell_m  = (target_yplus * viscosity) / (density * u_tau)
    first_cell_mm = first_cell_m * 1000
    delta         = (0.37 * chord) / math.pow(reynolds, 0.2)
    delta_mm      = delta * 1000
    try:
        core_math    = 1 - ((delta * (1 - growth_rate)) / first_cell_m)
        total_layers = math.ceil(math.log(core_math) / math.log(growth_rate)) if core_math > 0 else 30
    except:
        total_layers = 30
    actual_yplus = (density * u_tau * first_cell_m) / viscosity
    return reynolds, first_cell_mm, total_layers, delta_mm, actual_yplus

def get_yplus_zone(yplus):
    if yplus <= 5:    return "viscous_sublayer"
    elif yplus <= 30: return "buffer"
    else:             return "log_law"

def get_model_rec(yplus_mode):
    if yplus_mode == "near_wall":
        return {"primary": "k-omega SST", "secondary": "Spalart-Allmaras",
                "wall_treatment": "Near-Wall Modeling",
                "note": "Resolves viscous sublayer — best for adverse pressure gradients and stall."}
    return {"primary": "Realizable k-epsilon", "secondary": "RSM",
            "wall_treatment": "Standard Wall Function (SWF)",
            "note": "Wall functions bridge to log-law region — faster solve for attached flows."}

# ── 3. Airfoil Database ─────────────────────────────────────────────────
AIRFOIL_INFO = {
    "NACA 2412":              {"description": "General aviation, cambered",  "rec_yplus": 1.0,
                               "reason": "Adverse pressure gradient near TE; stall must be resolved",
                               "thickness": "12%", "camber": "2%"},
    "NACA 0012":              {"description": "Symmetric — control surfaces", "rec_yplus": 1.0,
                               "reason": "Symmetric stall needs near-wall resolution",
                               "thickness": "12%", "camber": "0%"},
    "NACA 4412":              {"description": "High camber — high-lift",     "rec_yplus": 1.0,
                               "reason": "Suction-side separation bubble likely at high AoA",
                               "thickness": "12%", "camber": "4%"},
    "NACA 23012":             {"description": "Reflex camber — transport",   "rec_yplus": 1.0,
                               "reason": "Complex pressure distribution near LE",
                               "thickness": "12%", "camber": "2.3%"},
    "Flat Plate / Bluff Body":{"description": "Flat plate — Salim & Cheah (2009)", "rec_yplus": 30.0,
                               "reason": "Geometry-forced separation — log-law region sufficient",
                               "thickness": "N/A", "camber": "N/A"},
}

REGIME_LABEL = {
    "subsonic": "Subsonic (M < 0.3)",
    "transonic": "Transonic (M 0.3–0.8)",
    "high_transonic": "High transonic (M 0.8–1.0)",
    "supersonic": "Supersonic (M > 1.0)",
}

# ── 4. Page Layout ──────────────────────────────────────────────────────
st.set_page_config(page_title="Ansys CFD Meshing Engine", page_icon="", layout="wide")
st.title("Ansys CFD Meshing Engine")
st.write("Airfoil V&V · Mesh Blueprint · **Mach-aware Domain Sizing**")
st.divider()

with st.sidebar:
    st.header("Airfoil selection")
    selected_airfoil = st.selectbox("Geometry", list(AIRFOIL_INFO.keys()))
    info = AIRFOIL_INFO[selected_airfoil]
    st.info(f"**{selected_airfoil}**\n\n{info['description']}\n\n"
            f"Thickness: {info['thickness']}  |  Camber: {info['camber']}")
    st.markdown("---")
    st.markdown("##### y+ zone reference")
    st.markdown("""
| Zone | y+ | Model |
|---|---|---|
| Viscous sublayer | <= 5 | k-w SST, S-A |
| AVOID buffer | 5-30 | neither |
| Log-law | 30-60 | k-e, RSM |
""")

# ── Section 1: Validation Targets ───────────────────────────────────────
st.subheader("1. Aerodynamic validation targets")
airfoil_df  = df[df['Geometry Type'] == selected_airfoil] if selected_airfoil in df['Geometry Type'].values else pd.DataFrame()
target_data = None

if not airfoil_df.empty:
    selected_flow = st.selectbox("Wind-tunnel test speed", airfoil_df['Flow Velocity / Mach Number'].unique())
    target_data   = airfoil_df[airfoil_df['Flow Velocity / Mach Number'] == selected_flow].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cl_max", target_data['Max Lift Coefficient'])
    c2.metric("Stall angle", f"{target_data['Stall Angle']} deg")
    c3.metric("Zero-lift drag", target_data['Zero-Lift Drag'])
    c4.metric("Source", target_data['Validation Source'])
else:
    st.warning(f"No wind-tunnel data in CSV for **{selected_airfoil}**. Domain + mesh still generated below.")

st.divider()

# ── Section 2: Flow Conditions ────────────────────────────────────────────
st.subheader("2. Flow conditions")
col_v, col_c, col_alt = st.columns(3)
with col_v:
    v_input = st.number_input("Freestream velocity (m/s)", value=44.0, step=1.0, min_value=1.0)
with col_c:
    c_input = st.number_input("Chord length (m)", value=1.0, step=0.1, min_value=0.01)
with col_alt:
    altitude_label = st.selectbox("Altitude", [
        "Sea level (1.225 kg/m3)",
        "1000 m (1.112 kg/m3)",
        "5000 m (0.736 kg/m3)",
        "10000 m (0.413 kg/m3)",
    ])

mach_val = get_mach(v_input)
regime   = get_flow_regime(mach_val)

m1, m2, m3 = st.columns(3)
m1.metric("Mach number", f"{mach_val:.4f}")
m2.metric("Flow regime", REGIME_LABEL[regime])
m3.metric("Speed of sound (sea level)", f"{SPEED_OF_SOUND} m/s")

st.divider()

# ── Section 3: Domain Sizing ─────────────────────────────────────────────
st.subheader("3. CFD domain sizing")
st.write("Select your domain topology strategy to generate the SpaceClaim dimensions.")

domain_strategy = st.radio("Domain Topology", 
                           ["Standard C-Grid (Industry Standard)", "Square Grid (Academic / Golmirzaee 2024)"], 
                           horizontal=True)

# Variables to hold domain bounds for the TUI output later
tui_domain_text = ""

if "C-Grid" in domain_strategy:
    domain = calculate_c_grid_domain(v_input, c_input)
    
    da, db, dc, dd, de = st.columns(5)
    da.metric("Upstream", f"{domain['upstream_c']:.0f}c = {domain['upstream_m']:.2f} m")
    db.metric("Downstream", f"{domain['downstream_c']:.0f}c = {domain['downstream_m']:.2f} m")
    dc.metric("Height", f"{domain['height_c']:.0f}c = {domain['height_m']:.2f} m")
    dd.metric("Total Width", f"{domain['total_width_m']:.2f} m")
    de.metric("Blockage Ratio", f"{domain['blockage_pct']:.3f}%")
    
    with st.expander("Domain details, boundary conditions & physics reasoning", expanded=True):
        st.info(domain['reason'])
        st.write(f"**Mesh topology:** `{domain['mesh_type']}`")
        
    tui_domain_text = (f"; Upstream extent    : {domain['upstream_m']:.3f} m  ({domain['upstream_c']:.0f} x chord)\n"
                       f"; Downstream extent  : {domain['downstream_m']:.3f} m  ({domain['downstream_c']:.0f} x chord)\n"
                       f"; Domain height      : {domain['height_m']:.3f} m  ({domain['height_c']:.0f} x chord)\n"
                       f"; Total width        : {domain['total_width_m']:.3f} m")

else:
    bc_type = st.selectbox("Boundary Condition Type", ["Standard Boundaries (Slip/Symmetry)", "Point Vortex BC (PVBC)"])
    if regime != "subsonic" and bc_type == "PVBC":
        st.warning("PVBC was strictly validated for incompressible subsonic flow. Proceed with caution.")
        
    sq_domain = calculate_square_domain(v_input, c_input, bc_type)
    
    da, db, dc, dd = st.columns(4)
    da.metric("Parameter A", f"{sq_domain['A_c']:.0f}c = {sq_domain['A_m']:.2f} m")
    db.metric("Total Width", f"{sq_domain['total_size_m']:.2f} m")
    dc.metric("Total Height", f"{sq_domain['total_size_m']:.2f} m")
    dd.metric("Blockage Ratio", f"{sq_domain['blockage_pct']:.3f}%")
    
    with st.expander("Square Domain Details & Physics", expanded=True):
        st.info(sq_domain['reason'])
        st.write(f"**Mesh topology:** `{sq_domain['mesh_type']}`")
        
        if bc_type == "Standard Boundaries (Slip/Symmetry)":
            st.markdown("##### Drag Error Correction Formula")
            st.write("If using Standard Boundaries, the digital walls induce an artificial pressure drag. Calculate the true infinite-domain drag using:")
            st.latex(r"C_{d,\infty} \approx C_{d} - 0.0205 \frac{C_{l}^{2}}{A}")
            
    tui_domain_text = (f"; Square Parameter A : {sq_domain['A_m']:.3f} m  ({sq_domain['A_c']:.0f} x chord)\n"
                       f"; Total Size         : {sq_domain['total_size_m']:.3f} m x {sq_domain['total_size_m']:.3f} m\n"
                       f"; Airfoil LE at      : (0, 0) relative to domain center")

st.divider()

# ── Section 4: y+ Strategy ───────────────────────────────────────────────
st.subheader("4. Wall y+ strategy")
col_yl, col_yr = st.columns(2)
with col_yl:
    yplus_mode = st.radio(
        "Near-wall treatment approach",
        options=["near_wall", "wall_function"],
        format_func=lambda x: (
            "Near-wall modeling  (y+ = 1)  — resolve viscous sublayer"
            if x == "near_wall"
            else "Wall functions  (y+ = 30-60)  — log-law region only"
        ),
        index=0 if info['rec_yplus'] <= 5 else 1,
    )
target_yplus = 1.0 if yplus_mode == "near_wall" else 30.0
rec = get_model_rec(yplus_mode)
with col_yr:
    box_fn = st.success if yplus_mode == "near_wall" else st.info
    box_fn(f"**Primary model:** {rec['primary']}\n\n"
           f"**Alternative:** {rec['secondary']}\n\n"
           f"**Wall treatment:** {rec['wall_treatment']}\n\n"
           f"_{rec['note']}_")
if yplus_mode == "wall_function" and info['rec_yplus'] <= 5:
    st.warning(f"Caution: {selected_airfoil} is recommended y+ = 1. Reason: {info['reason']}")

st.divider()

# ── Section 5: Mesh Blueprint ────────────────────────────────────────────
st.subheader("5. Generate mesh blueprint")
col_gr, col_btn = st.columns([2, 1])
with col_gr:
    growth_rate = st.number_input("Growth rate", value=1.15, step=0.01,
                                  min_value=1.05, max_value=1.5,
                                  help="Optimal: 1.1-1.2. Above 1.2 risks poor quality near TE.")
with col_btn:
    st.write(""); st.write("")
    generate = st.button("Generate full blueprint", type="primary", use_container_width=True)

if generate:
    calc_re, calc_dy, calc_layers, calc_delta, actual_yplus = calculate_mesh_blueprint(
        v_input, c_input, target_yplus, growth_rate
    )
    zone = get_yplus_zone(actual_yplus)

    st.success(f"Reynolds: {calc_re:.2e}  |  Mach: {mach_val:.4f}  |  Regime: {REGIME_LABEL[regime]}")

    col_mesh, col_solver = st.columns(2)

    with col_mesh:
        st.markdown("#### Meshing module (Ansys Meshing)")
        st.write(f"**First cell height:** `{calc_dy:.5f} mm`  ({calc_dy/1000:.7f} m)")
        st.write(f"**Growth rate:** `{growth_rate}`")
        st.write(f"**Total inflation layers:** `{calc_layers}`")
        st.write(f"**BL thickness delta:** `{calc_delta:.3f} mm`")

    with col_solver:
        st.markdown("#### Fluent solver settings")
        st.write(f"**Turbulence model:** `{rec['primary']}`")
        st.write(f"**Target y+:** `{target_yplus}`")
        if regime == "subsonic":
            st.write("**Solver:** `Pressure-Based`")
        else:
            st.write("**Solver:** `Density-Based`")

    st.divider()
    st.markdown("#### Fluent TUI reference snippet")
    model_tui  = ("kw-sst" if "SST" in rec['primary']
                  else "spalart-allmaras" if "Spalart" in rec['primary']
                  else "ke-realizable" if "Realizable" in rec['primary']
                  else "rsm")
    solver_tui = "pressure-based" if regime == "subsonic" else "density-based"
    
    st.code(
        f"; ── Fluent TUI commands ──────────────────────────────\n"
        f"; Solver\n"
        f"/define/models/solver/{solver_tui} yes\n\n"
        f"; Turbulence model\n"
        f"/define/models/viscous/{model_tui} yes\n\n"
        f"; Velocity inlet\n"
        f"/define/boundary-conditions/velocity-inlet inlet yes no {v_input} no 0 no 0\n\n"
        f"; ── Ansys Meshing inflation inputs ─────────────────\n"
        f"; First cell height  : {calc_dy:.5f} mm\n"
        f"; Growth rate        : {growth_rate}\n"
        f"; Total layers       : {calc_layers}\n\n"
        f"; ── Domain geometry (SpaceClaim / DesignModeler) ───\n"
        f"{tui_domain_text}\n",
        language="text"
    )