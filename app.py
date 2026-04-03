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

def calculate_domain(velocity, chord):
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
    st.markdown("---")
    st.markdown("##### Domain multipliers by Mach")
    st.markdown("""
| Regime | Up | Down | Height |
|---|---|---|---|
| Subsonic | 15c | 20c | 20c |
| Transonic | 30c | 40c | 30c |
| High trans | 40c | 50c | 35c |
| Supersonic | 10c | 60c | 40c |
""")

# ── Section 1: Validation Targets ───────────────────────────────────────
st.subheader("1. Aerodynamic validation targets")
airfoil_df  = df[df['Geometry Type'] == selected_airfoil] if selected_airfoil in df['Geometry Type'].values else pd.DataFrame()
target_data = None

if not airfoil_df.empty:
    selected_flow = st.selectbox("Wind-tunnel test speed", airfoil_df['Flow Velocity'].unique())
    target_data   = airfoil_df[airfoil_df['Flow Velocity'] == selected_flow].iloc[0]
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

if regime == "supersonic":
    st.error("Supersonic flow detected. Use Density-Based solver + Roe-FDS flux. Enable energy equation. Ideal Gas density.")
elif regime in ("transonic", "high_transonic"):
    st.warning("Transonic flow. Switch to Density-Based solver. Enable energy equation + compressibility corrections.")

st.divider()

# ── Section 3: Domain Sizing ─────────────────────────────────────────────
st.subheader("3. CFD domain sizing")
st.write("Domain extents are automatically calculated from **Mach number** and **chord length** "
         "using regime-specific multipliers from AIAA/Ansys best-practice guidelines.")

domain = calculate_domain(v_input, c_input)

da, db, dc, dd, de = st.columns(5)
da.metric("Upstream", f"{domain['upstream_c']:.0f}c = {domain['upstream_m']:.2f} m")
db.metric("Downstream", f"{domain['downstream_c']:.0f}c = {domain['downstream_m']:.2f} m")
dc.metric("Height", f"{domain['height_c']:.0f}c = {domain['height_m']:.2f} m")
dd.metric("Total width", f"{domain['total_width_m']:.2f} m")
de.metric("Blockage ratio", f"{domain['blockage_pct']:.3f}%")

with st.expander("Domain details, boundary conditions & physics reasoning", expanded=True):
    exp_l, exp_r = st.columns(2)

    with exp_l:
        st.markdown("##### Domain box")
        st.write(f"**Width:** `{domain['total_width_m']:.3f} m`  "
                 f"({domain['upstream_c']:.0f}c upstream + 1c airfoil + {domain['downstream_c']:.0f}c downstream)")
        st.write(f"**Height:** `{domain['total_height_m']:.3f} m`  ({domain['height_c']:.0f}c)")
        st.write(f"**Airfoil leading edge at:** x = `{domain['upstream_m']:.3f} m` from inlet")
        st.write(f"**Mesh topology:** `{domain['mesh_type']}`")

        st.markdown("##### Blockage check")
        if domain['blockage_pct'] > 5.0:
            st.error(f"Blockage {domain['blockage_pct']:.2f}% exceeds 5% limit. "
                     f"Increase height to at least {(0.12 * c_input / 0.05):.2f} m.")
        elif domain['blockage_pct'] > 2.0:
            st.warning(f"Blockage {domain['blockage_pct']:.2f}% — acceptable for preliminary work.")
        else:
            st.success(f"Blockage {domain['blockage_pct']:.3f}% — well within 1% limit.")

        st.markdown("##### Boundary conditions")
        st.write("**Left (inlet):** Velocity Inlet — U, TI = 0.1%, L = 0.07c")
        st.write("**Right (outlet):** Pressure Outlet — gauge P = 0 Pa")
        st.write("**Top / bottom:** Symmetry (subsonic) or Pressure Far-Field (trans/supersonic)")
        st.write("**Airfoil surface:** No-slip Wall with inflation mesh")

    with exp_r:
        st.markdown("##### Why this domain size?")
        st.info(domain['reason'])

        st.markdown("##### Regime-specific solver settings")
        if regime == "subsonic":
            st.success("Solver: Pressure-Based  |  Coupling: SIMPLE/Coupled\n\n"
                       "Density: Constant  |  Energy eq: OFF\n\n"
                       "Inlet TI: 0.1%  |  Operating P: 101325 Pa")
        elif regime == "transonic":
            st.warning("Solver: Density-Based  |  Formulation: Implicit\n\n"
                       "Density: Ideal Gas  |  Energy eq: ON\n\n"
                       "Compressibility correction: ON  |  CFL: 5-10")
        elif regime == "high_transonic":
            st.warning("Solver: Density-Based  |  Flux: Roe-FDS\n\n"
                       "Density: Ideal Gas  |  Energy eq: ON\n\n"
                       "Shock refinement region required near LE upper surface")
        else:
            st.error("Solver: Density-Based  |  Flux: Roe-FDS\n\n"
                     "Density: Ideal Gas  |  Energy eq: ON\n\n"
                     "CFL: 1-3 (start very conservative)  |  Bow-shock standoff mesh required")

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

    col_mesh, col_solver, col_dom = st.columns(3)

    with col_mesh:
        st.markdown("#### Meshing module (Ansys Meshing)")
        st.write(f"**First cell height:** `{calc_dy:.5f} mm`  ({calc_dy/1000:.7f} m)")
        st.write(f"**Growth rate:** `{growth_rate}`")
        st.write(f"**Total inflation layers:** `{calc_layers}`")
        st.write(f"**BL thickness delta:** `{calc_delta:.3f} mm`")
        st.write(f"**Inflation algorithm:** `Pre`")
        st.write(f"**Transition ratio:** `0.272`")
        st.write(f"**Max face size (far-field):** `{c_input * 0.5:.3f} m`")
        st.write(f"**Min face size (TE):** `{c_input * 0.001:.5f} m`")

    with col_solver:
        st.markdown("#### Fluent solver settings")
        st.write(f"**Turbulence model:** `{rec['primary']}`")
        st.write(f"**Wall treatment:** `{rec['wall_treatment']}`")
        st.write(f"**Target y+:** `{target_yplus}`")
        if regime == "subsonic":
            st.write("**Solver:** `Pressure-Based`")
            st.write("**Pressure discretisation:** `Second Order`")
            st.write("**Momentum:** `Second Order Upwind`")
        else:
            st.write("**Solver:** `Density-Based`")
            st.write("**Flux type:** `Roe-FDS`")
            st.write("**Energy equation:** `ON`")
        if target_data is not None:
            st.write(f"**Expected Cl_max:** `{target_data['Max Lift Coefficient']}`")
            st.write(f"**Expected stall angle:** `{target_data['Stall Angle']} deg`")

    with col_dom:
        st.markdown("#### Domain summary")
        st.write(f"**Upstream:** `{domain['upstream_m']:.3f} m` ({domain['upstream_c']:.0f}c)")
        st.write(f"**Downstream:** `{domain['downstream_m']:.3f} m` ({domain['downstream_c']:.0f}c)")
        st.write(f"**Height:** `{domain['height_m']:.3f} m` ({domain['height_c']:.0f}c)")
        st.write(f"**Total width:** `{domain['total_width_m']:.3f} m`")
        st.write(f"**Topology:** `{domain['mesh_type'].split('(')[0].strip()}`")
        st.write(f"**Blockage:** `{domain['blockage_pct']:.3f}%`")

    st.divider()
    st.markdown("#### Mesh quality checklist")
    checks = []

    if zone == "buffer":
        checks.append(("FAIL", "y+ zone",
                        f"BUFFER ZONE (y+ = {actual_yplus:.1f}). Max 20% error in Cf. "
                        "Refine below y+=5 or coarsen above y+=30."))
    elif zone == "viscous_sublayer":
        checks.append(("PASS", "y+ zone", f"Viscous sublayer resolved (y+ = {actual_yplus:.1f})."))
    else:
        checks.append(("PASS", "y+ zone", f"Log-law region (y+ = {actual_yplus:.1f})."))

    checks.append(("PASS" if 1.1 <= growth_rate <= 1.2 else "WARN", "Growth rate",
                   f"{growth_rate} — {'optimal' if 1.1 <= growth_rate <= 1.2 else 'outside 1.1-1.2 range'}."))

    if calc_layers < 15:
        checks.append(("WARN", "Layer count", f"{calc_layers} — low. May not cover full BL."))
    elif calc_layers > 60:
        checks.append(("WARN", "Layer count", f"{calc_layers} — very high. Check growth rate."))
    else:
        checks.append(("PASS", "Layer count", f"{calc_layers} layers — covers delta = {calc_delta:.2f} mm."))

    if yplus_mode == "near_wall" and zone == "log_law":
        checks.append(("WARN", "Model compatibility", "Near-wall model but y+ in log-law range."))
    elif yplus_mode == "wall_function" and zone == "viscous_sublayer":
        checks.append(("WARN", "Model compatibility", "Wall function but y+ in viscous sublayer — invalid."))
    else:
        checks.append(("PASS", "Model compatibility", f"{rec['primary']} consistent with y+ = {actual_yplus:.1f}."))

    if domain['blockage_pct'] > 5.0:
        checks.append(("FAIL", "Blockage", f"{domain['blockage_pct']:.2f}% — too high. Increase domain height."))
    elif domain['blockage_pct'] > 2.0:
        checks.append(("WARN", "Blockage", f"{domain['blockage_pct']:.2f}% — acceptable."))
    else:
        checks.append(("PASS", "Blockage", f"{domain['blockage_pct']:.3f}% — excellent."))

    if regime in ("transonic", "high_transonic", "supersonic"):
        checks.append(("WARN", "Compressibility",
                        f"Mach {mach_val:.3f} — verify density-based solver, ideal gas, energy eq ON."))
    else:
        checks.append(("PASS", "Compressibility", f"Mach {mach_val:.4f} — incompressible assumption valid."))

    if calc_re < 5e4:
        checks.append(("WARN", "Reynolds", f"{calc_re:.2e} — consider laminar or transition model."))
    elif calc_re > 1e7:
        checks.append(("WARN", "Reynolds", f"{calc_re:.2e} — high Re, check mesh density."))
    else:
        checks.append(("PASS", "Reynolds", f"{calc_re:.2e} — RANS turbulence models applicable."))

    icon_map = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    for status, label, msg in checks:
        st.write(f"{icon_map[status]} **{label}:** {msg}")

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
        f"; First cell height  : {calc_dy:.5f} mm  = {calc_dy/1000:.7f} m\n"
        f"; Growth rate        : {growth_rate}\n"
        f"; Total layers       : {calc_layers}\n"
        f"; Target y+          : {target_yplus}\n\n"
        f"; ── Domain geometry (SpaceClaim / DesignModeler) ───\n"
        f"; Upstream extent    : {domain['upstream_m']:.3f} m  ({domain['upstream_c']:.0f} x chord)\n"
        f"; Downstream extent  : {domain['downstream_m']:.3f} m  ({domain['downstream_c']:.0f} x chord)\n"
        f"; Domain height      : {domain['height_m']:.3f} m  ({domain['height_c']:.0f} x chord)\n"
        f"; Total width        : {domain['total_width_m']:.3f} m\n"
        f"; Airfoil LE at x    : {domain['upstream_m']:.3f} m from inlet\n",
        language="text"
    )