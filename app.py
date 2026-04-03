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
    st.error("Error: 'cfd_database.csv' not found. Place it in the same folder as app.py.")
    st.stop()

# ── 2. Physics Engines ──────────────────────────────────────────────────

SPEED_OF_SOUND = 340.3   # m/s at sea level, 15 deg C
GAMMA = 1.4

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

# ── C-Grid domain (unchanged) ──────────────────────────────────────────
def calculate_c_grid_domain(velocity, chord):
    mach   = get_mach(velocity)
    regime = get_flow_regime(mach)

    if regime == "subsonic":
        upstream, downstream, height = 15.0, 20.0, 20.0
        mesh_type = "C-mesh (recommended) or O-mesh"
        reason    = ("Pressure disturbance decays as ~1/r in incompressible flow. "
                     "15c upstream gives negligible inlet influence. "
                     "20c downstream fully captures the wake.")
    elif regime == "transonic":
        upstream, downstream, height = 30.0, 40.0, 30.0
        mesh_type = "C-mesh with fine wake refinement zone"
        reason    = ("Compressibility increases the range of pressure disturbances. "
                     "Shock waves need room to form and dissipate. "
                     "30c upstream prevents inlet reflection; 40c captures shock-induced wake.")
    elif regime == "high_transonic":
        upstream, downstream, height = 40.0, 50.0, 35.0
        mesh_type = "C-mesh with shock-aligned refinement zone"
        reason    = ("Near-sonic: strong normal shocks possible on upper surface. "
                     "Large downstream needed to fully capture Mach wave pattern "
                     "before the outlet boundary condition.")
    else:
        upstream, downstream, height = 10.0, 60.0, 40.0
        mesh_type = "Structured multi-block with bow-shock standoff region"
        reason    = ("Supersonic: disturbances cannot propagate upstream (Mach cone). "
                     "Small upstream ok. Large downstream for bow shock, "
                     "expansion fans, and oblique shock reflections.")

    upstream_m   = upstream   * chord
    downstream_m = downstream * chord
    height_m     = height     * chord
    blockage_pct = (0.12 * chord / height_m) * 100

    return {
        "mach": mach, "regime": regime,
        "upstream_c": upstream, "downstream_c": downstream, "height_c": height,
        "upstream_m": upstream_m, "downstream_m": downstream_m,
        "height_m": height_m, "total_width_m": upstream_m + chord + downstream_m,
        "total_height_m": height_m, "blockage_pct": blockage_pct,
        "mesh_type": mesh_type, "reason": reason,
    }

# ── Square domain: compressible physics ───────────────────────────────
def prandtl_glauert_factor(mach):
    """β = sqrt(1 - M²)  — Prandtl-Glauert compressibility correction.
    As M→1, β→0 meaning disturbances propagate infinitely far:
    the domain must grow to compensate."""
    return math.sqrt(max(1.0 - mach**2, 1e-6))

def mach_cone_half_angle_deg(mach):
    """Mach cone half-angle µ = arcsin(1/M).
    The domain height must exceed the cone spread at the downstream extent."""
    if mach <= 1.0:
        return 90.0
    return math.degrees(math.asin(1.0 / mach))

def calculate_square_domain(velocity, chord, bc_type):
    """
    Square domain sizing for both incompressible (Golmirzaee & Wood 2024)
    and compressible flow (Prandtl-Glauert scaling + Jameson transonic practice).

    Subsonic incompressible (M < 0.3):
      - PVBC:              A = 5c   (Golmirzaee & Wood 2024 — near-zero error)
      - Standard BC:       A = 30c  (industry baseline; drag error ~2% unless A=91c)

    Subsonic compressible (0.3 ≤ M < 0.8):
      Prandtl-Glauert scaling: A_comp = A_incomp / β
      where β = sqrt(1 - M²). This corrects for the increased reach of
      pressure disturbances as compressibility grows.
      - PVBC base:       A_incomp = 5c  → A_comp = 5c / β
      - Standard base:   A_incomp = 30c → A_comp = 30c / β
      (Capped at 100c for standard BC to stay physically reasonable)

    High transonic (0.8 ≤ M < 1.0):
      β → 0 makes Prandtl-Glauert singular near M=1.
      Use empirical rule: A = 80–100c.
      Physical reason: shock waves from the airfoil surface can reach
      the domain boundary and reflect back, corrupting the solution
      unless the domain is very large.

    Supersonic (M ≥ 1.0):
      Square grids are inefficient — bow shock and Mach cone geometry
      means the domain must extend at least:
        Height ≥ downstream_extent × tan(µ)
      where µ = arcsin(1/M) is the Mach cone half-angle.
      Minimum A = 60c recommended (square domain very inefficient here).
    """
    mach   = get_mach(velocity)
    regime = get_flow_regime(mach)
    beta   = prandtl_glauert_factor(mach)
    mu_deg = mach_cone_half_angle_deg(mach)

    if regime == "subsonic":
        # ── Incompressible: Golmirzaee & Wood (2024) ──────────────────
        if bc_type == "PVBC":
            A_c = 5.0
            formula_used = "A = 5c  (Golmirzaee & Wood, 2024 — incompressible PVBC)"
            reason = ("Point Vortex BC actively mimics the far-field vortex induced by "
                      "the airfoil, allowing a 5c domain with near-zero error. "
                      "Valid only for incompressible subsonic flow (M < 0.3).")
        else:
            A_c = 30.0
            formula_used = "A = 30c  (industry standard — incompressible)"
            reason = ("Standard Boundaries (Slip/Symmetry). Industry baseline A=30c. "
                      "Note: Drag error ~2% unless A=91c is used "
                      "(Golmirzaee & Wood, 2024). For high-accuracy drag, use PVBC.")

    elif regime == "transonic":
        # ── Compressible: Prandtl-Glauert scaling ─────────────────────
        # A_comp = A_incomp / β  where β = sqrt(1 - M²)
        if bc_type == "PVBC":
            A_incomp = 5.0
            A_c      = min(A_incomp / beta, 80.0)   # cap at 80c
            formula_used = (f"A = A_incomp / β = {A_incomp}c / {beta:.4f} = {A_c:.1f}c"
                            f"  (Prandtl-Glauert compressibility correction)")
            reason = ("PVBC with Prandtl-Glauert scaling: compressibility stretches the "
                      "pressure field by 1/β. The 5c incompressible base is scaled up to "
                      "account for the wider reach of disturbances at this Mach number. "
                      "Note: PVBC was derived for incompressible flow — use with caution "
                      "above M=0.5 and verify with grid independence test.")
        else:
            A_incomp = 30.0
            A_c      = min(A_incomp / beta, 100.0)  # cap at 100c
            formula_used = (f"A = A_incomp / β = {A_incomp}c / {beta:.4f} = {A_c:.1f}c"
                            f"  (Prandtl-Glauert compressibility correction)")
            reason = ("Prandtl-Glauert compressibility correction scales the incompressible "
                      "A=30c baseline by 1/β = 1/√(1−M²). As M increases toward 0.8, β→0.6, "
                      "so the required domain grows to ~50c. This accounts for the longer "
                      "reach of pressure disturbances in compressible subsonic flow "
                      "(Jameson & Vassberg, AIAA 2008-0145).")

    elif regime == "high_transonic":
        # ── Near-sonic: P-G singular, use empirical rule ──────────────
        # β → 0 near M=1 makes P-G break down (critical Mach singularity).
        # Jameson AIAA 2008 used domains ~80–100c for M=0.8–0.85 transonic wings.
        A_c = 100.0
        formula_used = "A = 100c  (empirical — P-G singular near M=1)"
        reason = ("Near M=1.0, Prandtl-Glauert factor β = √(1−M²) → 0, making "
                  "the correction singular. Instead, use an empirically determined "
                  "large domain of 100c. Strong normal shocks on the airfoil upper "
                  "surface require vast downstream extent to prevent shock reflection "
                  "off the outlet boundary. Reference: Jameson & Vassberg AIAA 2008-0145 "
                  "used structured grids ~80–100c for transonic M=0.8–0.85 wings.")
        if bc_type == "PVBC":
            st.warning("PVBC is strictly validated for incompressible subsonic flow only. "
                       "At this Mach number, standard boundaries with a large domain are required.")

    else:
        # ── Supersonic: Mach cone geometry ────────────────────────────
        # The Mach cone half-angle µ = arcsin(1/M).
        # For a square domain of side A, the cone from the trailing edge
        # must not hit the domain walls before the outlet.
        # Minimum height: A ≥ 60c regardless. Very inefficient — C-grid strongly preferred.
        A_c = 60.0
        formula_used = (f"A = 60c  (Mach cone µ = {mu_deg:.1f}°; square domain very inefficient)")
        reason = (f"Supersonic flow: Mach cone half-angle µ = arcsin(1/M) = {mu_deg:.1f}°. "
                  "The bow shock and expansion fans spread laterally at this angle. "
                  "A square domain of 60c provides minimal containment, but a structured "
                  "C-mesh or multi-block grid aligned with the shock is STRONGLY preferred. "
                  "Square grids waste cells in the upstream corners where nothing happens.")
        if bc_type == "PVBC":
            st.error("PVBC is not valid for supersonic flow. Use standard pressure far-field BC.")

    A_m          = A_c * chord
    total_size_m = 2.0 * A_m
    blockage_pct = (0.12 * chord / total_size_m) * 100

    return {
        "mach": mach, "regime": regime, "beta": beta, "mu_deg": mu_deg,
        "A_c": A_c, "A_m": A_m, "total_size_m": total_size_m,
        "blockage_pct": blockage_pct, "reason": reason,
        "formula_used": formula_used,
        "mesh_type": "Square Grid (2A × 2A)",
    }

# ── Mesh blueprint ──────────────────────────────────────────────────────
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

# ── Airfoil database ────────────────────────────────────────────────────
AIRFOIL_INFO = {
    "NACA 2412":               {"description": "General aviation, cambered",  "rec_yplus": 1.0,
                                "reason": "Adverse pressure gradient near TE; stall must be resolved",
                                "thickness": "12%", "camber": "2%"},
    "NACA 0012":               {"description": "Symmetric — control surfaces", "rec_yplus": 1.0,
                                "reason": "Symmetric stall needs near-wall resolution",
                                "thickness": "12%", "camber": "0%"},
    "NACA 4412":               {"description": "High camber — high-lift",      "rec_yplus": 1.0,
                                "reason": "Suction-side separation bubble likely at high AoA",
                                "thickness": "12%", "camber": "4%"},
    "NACA 23012":              {"description": "Reflex camber — transport",    "rec_yplus": 1.0,
                                "reason": "Complex pressure distribution near LE",
                                "thickness": "12%", "camber": "2.3%"},
    "Flat Plate / Bluff Body": {"description": "Flat plate — Salim & Cheah (2009)", "rec_yplus": 30.0,
                                "reason": "Geometry-forced separation — log-law region sufficient",
                                "thickness": "N/A", "camber": "N/A"},
}

REGIME_LABEL = {
    "subsonic":       "Subsonic (M < 0.3)",
    "transonic":      "Transonic (M 0.3-0.8)",
    "high_transonic": "High transonic (M 0.8-1.0)",
    "supersonic":     "Supersonic (M > 1.0)",
}

# ── 4. Page Layout ──────────────────────────────────────────────────────
st.set_page_config(page_title="Ansys CFD Meshing Engine", page_icon="", layout="wide")
st.title("Ansys CFD Meshing Engine")
st.write("Airfoil V&V · Mesh Blueprint · **Mach-aware Domain Sizing** · Compressible Square Grid")
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
    st.markdown("##### Square domain formula guide")
    st.markdown("""
| Regime | Formula | Source |
|---|---|---|
| M < 0.3 PVBC | A = 5c | Golmirzaee 2024 |
| M < 0.3 std | A = 30c | Industry std |
| M 0.3-0.8 | A = base/β | Prandtl-Glauert |
| M 0.8-1.0 | A = 100c | Empirical |
| M > 1.0 | A = 60c | Mach cone |
""")

# ── Section 1: Validation Targets ───────────────────────────────────────
st.subheader("1. Aerodynamic validation targets")
airfoil_df  = df[df['Geometry Type'] == selected_airfoil] if selected_airfoil in df['Geometry Type'].values else pd.DataFrame()
target_data = None

if not airfoil_df.empty:
    flow_col = 'Flow Velocity / Mach Number' if 'Flow Velocity / Mach Number' in df.columns else 'Flow Velocity'
    selected_flow = st.selectbox("Wind-tunnel test speed", airfoil_df[flow_col].unique())
    target_data   = airfoil_df[airfoil_df[flow_col] == selected_flow].iloc[0]
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
        "Sea level (1.225 kg/m3)", "1000 m (1.112 kg/m3)",
        "5000 m (0.736 kg/m3)", "10000 m (0.413 kg/m3)",
    ])

mach_val = get_mach(v_input)
regime   = get_flow_regime(mach_val)
beta_val = prandtl_glauert_factor(mach_val)
mu_val   = mach_cone_half_angle_deg(mach_val)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Mach number", f"{mach_val:.4f}")
m2.metric("Flow regime", REGIME_LABEL[regime])
m3.metric("P-G factor beta", f"{beta_val:.4f}", help="beta = sqrt(1-M^2). Prandtl-Glauert compressibility factor.")
m4.metric("Mach cone angle mu", f"{mu_val:.1f} deg" if mach_val > 1 else "N/A (subsonic)")

if regime == "supersonic":
    st.error("Supersonic flow detected. Use Density-Based solver + Roe-FDS flux. Enable energy equation. Ideal Gas density.")
elif regime in ("transonic", "high_transonic"):
    st.warning("Transonic flow. Switch to Density-Based solver. Enable energy equation + compressibility corrections.")

st.divider()

# ── Section 3: Domain Sizing ─────────────────────────────────────────────
st.subheader("3. CFD domain sizing")
st.write("Select domain topology. **Square grid** now uses Prandtl-Glauert compressibility scaling for M > 0.3.")

domain_strategy = st.radio("Domain topology", 
                           ["Standard C-Grid (Industry Standard)",
                            "Square Grid (Academic / Compressible Golmirzaee 2024)"],
                           horizontal=True)

tui_domain_text = ""

if "C-Grid" in domain_strategy:
    domain = calculate_c_grid_domain(v_input, c_input)
    da, db, dc, dd, de = st.columns(5)
    da.metric("Upstream",     f"{domain['upstream_c']:.0f}c = {domain['upstream_m']:.2f} m")
    db.metric("Downstream",   f"{domain['downstream_c']:.0f}c = {domain['downstream_m']:.2f} m")
    dc.metric("Height",       f"{domain['height_c']:.0f}c = {domain['height_m']:.2f} m")
    dd.metric("Total width",  f"{domain['total_width_m']:.2f} m")
    de.metric("Blockage",     f"{domain['blockage_pct']:.3f}%")

    with st.expander("Domain details, boundary conditions & physics reasoning", expanded=True):
        exp_l, exp_r = st.columns(2)
        with exp_l:
            st.markdown("##### Domain box")
            st.write(f"**Width:** `{domain['total_width_m']:.3f} m`  "
                     f"({domain['upstream_c']:.0f}c + 1c + {domain['downstream_c']:.0f}c)")
            st.write(f"**Height:** `{domain['total_height_m']:.3f} m`  ({domain['height_c']:.0f}c)")
            st.write(f"**Airfoil LE at:** x = `{domain['upstream_m']:.3f} m` from inlet")
            st.write(f"**Mesh topology:** `{domain['mesh_type']}`")
            st.markdown("##### Boundary conditions")
            st.write("**Left (inlet):** Velocity Inlet — U, TI=0.1%, L=0.07c")
            st.write("**Right (outlet):** Pressure Outlet — gauge P=0 Pa")
            st.write("**Top/bottom:** Symmetry (subsonic) | Pressure Far-Field (trans/supersonic)")
            st.write("**Airfoil:** No-slip Wall with inflation mesh")
        with exp_r:
            st.markdown("##### Why this size?")
            st.info(domain['reason'])

    tui_domain_text = (f"; Upstream  : {domain['upstream_m']:.3f} m  ({domain['upstream_c']:.0f}c)\n"
                       f"; Downstream: {domain['downstream_m']:.3f} m  ({domain['downstream_c']:.0f}c)\n"
                       f"; Height    : {domain['height_m']:.3f} m  ({domain['height_c']:.0f}c)\n"
                       f"; Airfoil LE: x = {domain['upstream_m']:.3f} m from inlet")

else:
    # Square grid with full compressible formulas
    bc_type = st.selectbox("Boundary condition type",
                           ["Standard Boundaries (Slip/Symmetry)", "Point Vortex BC (PVBC)"])

    sq = calculate_square_domain(v_input, c_input, bc_type)

    da, db, dc, dd, de = st.columns(5)
    da.metric("Parameter A",   f"{sq['A_c']:.1f}c = {sq['A_m']:.2f} m")
    db.metric("Total width",   f"{sq['total_size_m']:.2f} m")
    dc.metric("Total height",  f"{sq['total_size_m']:.2f} m")
    dd.metric("Blockage",      f"{sq['blockage_pct']:.3f}%")
    de.metric("P-G factor β",  f"{sq['beta']:.4f}")

    with st.expander("Square domain physics, formulas & boundary conditions", expanded=True):
        exp_l, exp_r = st.columns(2)
        with exp_l:
            st.markdown("##### Formula applied")
            st.code(sq['formula_used'], language="text")
            st.markdown("##### Compressibility physics")
            if regime == "subsonic":
                st.markdown(r"""
**Incompressible (M < 0.3) — Golmirzaee & Wood (2024):**
- PVBC:    A = 5c  → near-zero error
- Std BC:  A = 30c → ~2% drag error (need A=91c for <0.1%)
""")
            elif regime == "transonic":
                st.markdown(r"""
**Compressible Prandtl-Glauert scaling (0.3 ≤ M < 0.8):**

$$A_{comp} = \frac{A_{incomp}}{\beta}, \quad \beta = \sqrt{1 - M^2}$$

As M increases, β decreases, and the pressure disturbance 
field stretches by factor 1/β in the streamwise direction.
The domain must grow proportionally to avoid boundary 
contamination of the solution.
""")
                st.metric("1/β scaling factor", f"{1/sq['beta']:.3f}x", 
                          help="Domain is this many times larger than the incompressible case")
            elif regime == "high_transonic":
                st.markdown(r"""
**Near-sonic empirical rule (0.8 ≤ M < 1.0):**

P-G factor β → 0 as M → 1 (critical Mach singularity).
Prandtl-Glauert breaks down. Use empirical A = 100c.

Physical basis: Jameson & Vassberg (AIAA 2008-0145) used
structured grids ~80-100c for M=0.8-0.85 transonic wings.
""")
            else:
                st.markdown(f"""
**Supersonic Mach cone (M > 1.0):**

Mach cone half-angle: µ = arcsin(1/M) = **{sq['mu_deg']:.1f}°**

Square grid A = 60c is a minimum. The C-grid/multi-block
approach is STRONGLY preferred for supersonic flows.
""")

            st.markdown("##### Boundary conditions")
            if bc_type == "Standard Boundaries (Slip/Symmetry)":
                st.write("**All 4 outer edges:** Slip wall / Symmetry")
                st.write("**Airfoil:** No-slip Wall")
                if regime == "subsonic":
                    st.markdown("##### Drag correction formula")
                    st.write("Standard boundaries create artificial drag. Correct using:")
                    st.latex(r"C_{d,\infty} \approx C_d - 0.0205 \frac{C_l^2}{A}")
                elif regime == "transonic":
                    st.warning("For compressible flow, use **Pressure Far-Field** BC instead of slip walls. "
                               "Slip walls reflect shockwaves and corrupt the solution.")
            else:
                st.write("**All 4 outer edges:** Point Vortex BC (PVBC)")
                st.write("**Airfoil:** No-slip Wall")
                if regime != "subsonic":
                    st.warning("PVBC was derived for incompressible flow. For M > 0.3, "
                               "compressibility effects reduce its accuracy — verify with grid independence test.")

        with exp_r:
            st.markdown("##### Why this domain size?")
            st.info(sq['reason'])
            st.markdown("##### Regime-specific solver")
            if regime == "subsonic":
                st.success("Solver: Pressure-Based | Density: Constant\nEnergy: OFF | Coupling: SIMPLE")
            elif regime == "transonic":
                st.warning(f"Solver: Density-Based | Density: Ideal Gas\n"
                           f"Energy: ON | Flux: Roe-FDS | CFL: 5-10\n"
                           f"P-G β = {sq['beta']:.4f} → domain scaled x{1/sq['beta']:.2f}")
            elif regime == "high_transonic":
                st.warning("Solver: Density-Based | Density: Ideal Gas\n"
                           "Energy: ON | Shock refinement required near LE\n"
                           "P-G singular — empirical 100c domain used")
            else:
                st.error(f"Solver: Density-Based | Flux: Roe-FDS\n"
                         f"Energy: ON | CFL: 1-3\n"
                         f"Mach cone µ = {sq['mu_deg']:.1f}° — C-grid strongly preferred")

    tui_domain_text = (f"; Square domain Parameter A: {sq['A_m']:.3f} m  ({sq['A_c']:.1f}c)\n"
                       f"; Total size: {sq['total_size_m']:.3f} m x {sq['total_size_m']:.3f} m\n"
                       f"; P-G factor beta: {sq['beta']:.4f}  (1/beta = {1/sq['beta']:.3f})\n"
                       f"; Formula applied: {sq['formula_used']}\n"
                       f"; Airfoil center at domain origin (0, 0)")

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
            else "Wall functions  (y+ = 30-60)  — log-law region only"),
        index=0 if info['rec_yplus'] <= 5 else 1,
    )
target_yplus = 1.0 if yplus_mode == "near_wall" else 30.0
rec = get_model_rec(yplus_mode)
with col_yr:
    box_fn = st.success if yplus_mode == "near_wall" else st.info
    box_fn(f"**Primary:** {rec['primary']}  |  **Alt:** {rec['secondary']}\n\n"
           f"**Wall treatment:** {rec['wall_treatment']}\n\n_{rec['note']}_")
if yplus_mode == "wall_function" and info['rec_yplus'] <= 5:
    st.warning(f"Caution: {selected_airfoil} recommended y+=1. Reason: {info['reason']}")

st.divider()

# ── Section 5: Mesh Blueprint ────────────────────────────────────────────
st.subheader("5. Generate mesh blueprint")
col_gr, col_btn = st.columns([2, 1])
with col_gr:
    growth_rate = st.number_input("Growth rate", value=1.15, step=0.01, min_value=1.05, max_value=1.5)
with col_btn:
    st.write(""); st.write("")
    generate = st.button("Generate full blueprint", type="primary", use_container_width=True)

if generate:
    calc_re, calc_dy, calc_layers, calc_delta, actual_yplus = calculate_mesh_blueprint(
        v_input, c_input, target_yplus, growth_rate)
    zone = get_yplus_zone(actual_yplus)

    st.success(f"Re: {calc_re:.2e}  |  M: {mach_val:.4f}  |  β: {beta_val:.4f}  |  {REGIME_LABEL[regime]}")

    col_m, col_s, col_d = st.columns(3)
    with col_m:
        st.markdown("#### Meshing module")
        st.write(f"**First cell height:** `{calc_dy:.5f} mm`")
        st.write(f"**Growth rate:** `{growth_rate}`")
        st.write(f"**Total inflation layers:** `{calc_layers}`")
        st.write(f"**BL thickness delta:** `{calc_delta:.3f} mm`")
        st.write(f"**Inflation algorithm:** `Pre`")
        st.write(f"**Transition ratio:** `0.272`")
        st.write(f"**Max face size (far-field):** `{c_input*0.5:.3f} m`")
        st.write(f"**Min face size (TE):** `{c_input*0.001:.5f} m`")
    with col_s:
        st.markdown("#### Fluent solver")
        st.write(f"**Turbulence model:** `{rec['primary']}`")
        st.write(f"**Wall treatment:** `{rec['wall_treatment']}`")
        st.write(f"**Target y+:** `{target_yplus}`")
        if regime == "subsonic":
            st.write("**Solver:** `Pressure-Based`")
            st.write("**Pressure:** `Second Order`")
            st.write("**Momentum:** `Second Order Upwind`")
        else:
            st.write("**Solver:** `Density-Based`")
            st.write("**Flux type:** `Roe-FDS`")
            st.write("**Energy equation:** `ON`")
        if target_data is not None:
            st.write(f"**Expected Cl_max:** `{target_data['Max Lift Coefficient']}`")
            st.write(f"**Expected stall angle:** `{target_data['Stall Angle']} deg`")
    with col_d:
        st.markdown("#### Domain summary")
        if "C-Grid" in domain_strategy:
            st.write(f"**Upstream:** `{domain['upstream_m']:.3f} m` ({domain['upstream_c']:.0f}c)")
            st.write(f"**Downstream:** `{domain['downstream_m']:.3f} m` ({domain['downstream_c']:.0f}c)")
            st.write(f"**Height:** `{domain['height_m']:.3f} m` ({domain['height_c']:.0f}c)")
            st.write(f"**Blockage:** `{domain['blockage_pct']:.3f}%`")
        else:
            st.write(f"**Parameter A:** `{sq['A_m']:.3f} m` ({sq['A_c']:.1f}c)")
            st.write(f"**Total size:** `{sq['total_size_m']:.3f}m x {sq['total_size_m']:.3f}m`")
            st.write(f"**P-G factor β:** `{sq['beta']:.4f}`")
            st.write(f"**Blockage:** `{sq['blockage_pct']:.3f}%`")

    st.divider()
    st.markdown("#### Mesh quality checklist")
    checks = []
    if zone == "buffer":
        checks.append(("FAIL", "y+ zone",
                       f"BUFFER ZONE (y+ = {actual_yplus:.1f}). Max 20% Cf error. Refine or coarsen."))
    elif zone == "viscous_sublayer":
        checks.append(("PASS", "y+ zone", f"Viscous sublayer resolved (y+ = {actual_yplus:.1f})."))
    else:
        checks.append(("PASS", "y+ zone", f"Log-law region (y+ = {actual_yplus:.1f})."))
    checks.append(("PASS" if 1.1 <= growth_rate <= 1.2 else "WARN", "Growth rate",
                   f"{growth_rate} — {'optimal' if 1.1 <= growth_rate <= 1.2 else 'outside 1.1-1.2'}."))
    if calc_layers < 15:
        checks.append(("WARN", "Layer count", f"{calc_layers} — low. May not cover full BL."))
    elif calc_layers > 60:
        checks.append(("WARN", "Layer count", f"{calc_layers} — high. Check growth rate."))
    else:
        checks.append(("PASS", "Layer count", f"{calc_layers} layers — covers delta={calc_delta:.2f} mm."))
    if yplus_mode == "near_wall" and zone == "log_law":
        checks.append(("WARN", "Model compat.", "Near-wall model but y+ in log-law range."))
    elif yplus_mode == "wall_function" and zone == "viscous_sublayer":
        checks.append(("WARN", "Model compat.", "Wall function but y+ in viscous sublayer — invalid."))
    else:
        checks.append(("PASS", "Model compat.", f"{rec['primary']} consistent with y+={actual_yplus:.1f}."))
    if regime in ("transonic", "high_transonic", "supersonic"):
        checks.append(("WARN", "Compressibility",
                       f"M={mach_val:.3f}, β={beta_val:.4f}. Density-based solver, ideal gas, energy ON."))
    else:
        checks.append(("PASS", "Compressibility", f"M={mach_val:.4f} — incompressible valid."))
    if calc_re < 5e4:
        checks.append(("WARN", "Reynolds", f"{calc_re:.2e} — consider laminar/transition model."))
    elif calc_re > 1e7:
        checks.append(("WARN", "Reynolds", f"{calc_re:.2e} — high Re, check mesh density."))
    else:
        checks.append(("PASS", "Reynolds", f"{calc_re:.2e} — RANS applicable."))

    icon_map = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    for status, label, msg in checks:
        st.write(f"{icon_map[status]} **{label}:** {msg}")

    st.divider()
    st.markdown("#### Fluent TUI reference snippet")
    model_tui  = ("kw-sst" if "SST" in rec['primary']
                  else "spalart-allmaras" if "Spalart" in rec['primary']
                  else "ke-realizable" if "Realizable" in rec['primary'] else "rsm")
    solver_tui = "pressure-based" if regime == "subsonic" else "density-based"
    st.code(
        f"; ── Fluent TUI commands ──────────────────────────────\n"
        f"/define/models/solver/{solver_tui} yes\n"
        f"/define/models/viscous/{model_tui} yes\n"
        f"/define/boundary-conditions/velocity-inlet inlet yes no {v_input} no 0 no 0\n\n"
        f"; ── Meshing inflation inputs ─────────────────────────\n"
        f"; First cell height  : {calc_dy:.5f} mm  ({calc_dy/1000:.7f} m)\n"
        f"; Growth rate        : {growth_rate}\n"
        f"; Total layers       : {calc_layers}\n"
        f"; Target y+          : {target_yplus}\n\n"
        f"; ── Domain geometry (SpaceClaim / DesignModeler) ─────\n"
        f"{tui_domain_text}\n",
        language="text"
    )
