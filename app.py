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

SPEED_OF_SOUND = 340.3
GAMMA = 1.4

def get_mach(velocity):
    return velocity / SPEED_OF_SOUND

def get_flow_regime(mach):
    if mach < 0.3:   return "subsonic"
    elif mach < 0.8: return "transonic"
    elif mach < 1.0: return "high_transonic"
    else:            return "supersonic"

def prandtl_glauert_factor(mach):
    """β = sqrt(1 - M²). Compressibility correction factor.
    As M→1, β→0 meaning disturbances propagate much further."""
    return math.sqrt(max(1.0 - mach**2, 1e-6))

def mach_cone_half_angle_deg(mach):
    """µ = arcsin(1/M). Mach cone half-angle in degrees.
    For subsonic M≤1 returns 90 (no cone — spherical propagation)."""
    if mach <= 1.0:
        return 90.0
    return math.degrees(math.asin(1.0 / mach))

# ── C-Grid domain: full physics-based calculation ──────────────────────
def calculate_c_grid_domain(velocity, chord):
    """
    C-Grid domain sizing with proper physics for all four Mach regimes.

    ── Subsonic (M < 0.3) ─────────────────────────────────────────────
    Incompressible potential flow. Pressure disturbance from the airfoil
    decays as ~1/r². Industry standard (NACA TN 1135 / ESDU 76024):
      upstream   = 15c
      downstream = 20c
      height     = 20c

    ── Transonic (0.3 ≤ M < 0.8) ─────────────────────────────────────
    Compressible subsonic: Prandtl-Glauert correction scales the reach of
    pressure disturbances by factor 1/β where β = √(1−M²).
    The incompressible base multipliers are scaled accordingly:
      upstream   = min(15c / β,  30c)   ← capped to prevent runaway near M→0.8
      downstream = min(20c / β,  50c)
      height     = min(20c / β,  40c)
    Reference: Prandtl (1928), Glauert (1928) linearised compressibility theory.
    At M=0.5: β=0.866 → ×1.15 scaling. At M=0.75: β=0.661 → ×1.51 scaling.

    ── High transonic (0.8 ≤ M < 1.0) ────────────────────────────────
    P-G factor β→0 as M→1 (critical Mach singularity — P-G breaks down).
    Use empirically validated values from Jameson & Vassberg (AIAA 2008-0145)
    who used structured grids for M=0.79–0.85 transonic wing simulations:
      upstream   = 50c  (strong upstream influence from LE shock)
      downstream = 60c  (large shock-induced wake)
      height     = 40c  (normal shock lateral spread)

    ── Supersonic (M ≥ 1.0) ───────────────────────────────────────────
    Disturbances cannot propagate upstream (confined to Mach cone).
    Mach cone half-angle: µ = arcsin(1/M).
    The domain height must contain the cone spread over the downstream length:
      upstream   = 10c              (no upstream influence — Mach cone)
      downstream = 60c              (bow shock + expansion fans + oblique reflections)
      height     = max(40c, downstream_m × tan(µ))
                                    (cone spread at downstream extent)
    At M=1.5: µ=41.8°, tan(µ)=0.894 → 60c×0.894=53.6c ≈ max → height=53.6c
    At M=2.0: µ=30.0°, tan(µ)=0.577 → 60c×0.577=34.6c → height=40c (floor)
    At M=3.0: µ=19.5°, tan(µ)=0.354 → 60c×0.354=21.2c → height=40c (floor)
    """
    mach   = get_mach(velocity)
    regime = get_flow_regime(mach)
    beta   = prandtl_glauert_factor(mach)
    mu_deg = mach_cone_half_angle_deg(mach)

    if regime == "subsonic":
        upstream   = 15.0
        downstream = 20.0
        height     = 20.0
        mesh_type  = "C-mesh or O-mesh"
        bc_top     = "Symmetry"
        reason     = (
            "Incompressible potential flow. Pressure disturbance decays as ~1/r². "
            "15c upstream gives negligible inlet influence on the airfoil pressure "
            "field. 20c downstream fully captures the viscous wake. "
            "Source: NACA TN 1135 / ESDU 76024 industry standard."
        )
        formula_detail = "Upstream=15c, Downstream=20c, Height=20c  (incompressible standard)"

    elif regime == "transonic":
        upstream   = min(15.0 / beta, 30.0)
        downstream = min(20.0 / beta, 50.0)
        height     = min(20.0 / beta, 40.0)
        mesh_type  = "C-mesh with wake refinement zone"
        bc_top     = "Pressure Far-Field"
        reason     = (
            f"Compressible subsonic. Prandtl-Glauert correction: domain scales by "
            f"1/β = 1/√(1−M²) = 1/{beta:.4f} = {1/beta:.3f}×. "
            f"Base multipliers (15c, 20c, 20c) are scaled up proportionally. "
            f"Shock waves need room to form. Slip walls replaced by Pressure Far-Field BC "
            f"to avoid shock reflection off boundaries."
        )
        formula_detail = (
            f"β = √(1−M²) = √(1−{mach:.4f}²) = {beta:.4f}\n"
            f"Upstream   = min(15c / {beta:.4f}, 30c) = {upstream:.2f}c\n"
            f"Downstream = min(20c / {beta:.4f}, 50c) = {downstream:.2f}c\n"
            f"Height     = min(20c / {beta:.4f}, 40c) = {height:.2f}c"
        )

    elif regime == "high_transonic":
        upstream   = 50.0
        downstream = 60.0
        height     = 40.0
        mesh_type  = "C-mesh with shock-aligned refinement + bow-shock standoff"
        bc_top     = "Pressure Far-Field"
        reason     = (
            f"Near M=1.0: Prandtl-Glauert factor β = {beta:.4f} → 0. "
            f"P-G linearisation breaks down at the critical Mach singularity. "
            f"Empirical values from Jameson & Vassberg (AIAA 2008-0145) "
            f"for structured grids at M=0.79–0.85: 50c upstream, 60c downstream, 40c height. "
            f"Strong normal shocks on upper surface require large domain to prevent "
            f"shock reflection off boundaries."
        )
        formula_detail = (
            f"P-G factor β = {beta:.4f} (→0 near M=1, formula singular)\n"
            f"Upstream   = 50c  (empirical — Jameson & Vassberg AIAA 2008-0145)\n"
            f"Downstream = 60c  (shock-induced wake extent)\n"
            f"Height     = 40c  (normal shock lateral spread)"
        )

    else:
        mu_rad          = math.radians(mu_deg)
        downstream      = 60.0
        height_from_cone = downstream * math.tan(mu_rad)
        height           = max(40.0, height_from_cone)
        upstream         = 10.0
        mesh_type        = "Structured multi-block with bow-shock standoff"
        bc_top           = "Pressure Far-Field"
        reason           = (
            f"Supersonic: Mach cone half-angle µ = arcsin(1/M) = arcsin(1/{mach:.4f}) = {mu_deg:.2f}°. "
            f"Upstream extent = 10c only — disturbances cannot propagate upstream past the Mach cone. "
            f"Domain height must contain the Mach wave spreading laterally over the downstream length: "
            f"H ≥ downstream × tan(µ) = {downstream:.0f}c × tan({mu_deg:.1f}°) = {height_from_cone:.2f}c → "
            f"height = max(40c floor, {height_from_cone:.2f}c) = {height:.2f}c."
        )
        formula_detail = (
            f"µ = arcsin(1/M) = arcsin(1/{mach:.4f}) = {mu_deg:.2f}°\n"
            f"Upstream   = 10c  (no upstream propagation — Mach cone)\n"
            f"Downstream = 60c  (bow shock + expansion fans)\n"
            f"H_cone     = downstream × tan(µ) = {downstream:.0f}c × {math.tan(mu_rad):.4f} = {height_from_cone:.2f}c\n"
            f"Height     = max(40c, {height_from_cone:.2f}c) = {height:.2f}c"
        )

    upstream_m   = upstream   * chord
    downstream_m = downstream * chord
    height_m     = height     * chord
    blockage_pct = (0.12 * chord / height_m) * 100

    return {
        "mach": mach, "regime": regime, "beta": beta, "mu_deg": mu_deg,
        "upstream_c": upstream, "downstream_c": downstream, "height_c": height,
        "upstream_m": upstream_m, "downstream_m": downstream_m,
        "height_m": height_m, "total_width_m": upstream_m + chord + downstream_m,
        "total_height_m": height_m, "blockage_pct": blockage_pct,
        "mesh_type": mesh_type, "bc_top": bc_top,
        "reason": reason, "formula_detail": formula_detail,
    }

# ── Square domain ──────────────────────────────────────────────────────
def calculate_square_domain(velocity, chord, bc_type):
    mach   = get_mach(velocity)
    regime = get_flow_regime(mach)
    beta   = prandtl_glauert_factor(mach)
    mu_deg = mach_cone_half_angle_deg(mach)

    if regime == "subsonic":
        if bc_type == "PVBC":
            A_c = 5.0
            formula_used = "A = 5c  (Golmirzaee & Wood, 2024 — incompressible PVBC)"
            reason = ("PVBC mimics far-field vortex, allowing 5c domain with near-zero error. "
                      "Valid only for incompressible subsonic flow (M < 0.3).")
        else:
            A_c = 30.0
            formula_used = "A = 30c  (industry standard — incompressible)"
            reason = ("Standard Boundaries A=30c. Drag error ~2% unless A=91c "
                      "(Golmirzaee & Wood, 2024).")
    elif regime == "transonic":
        if bc_type == "PVBC":
            A_c = min(5.0 / beta, 80.0)
            formula_used = f"A = 5c / β = 5c / {beta:.4f} = {A_c:.1f}c  (P-G scaled PVBC)"
            reason = "PVBC with P-G scaling. Use with caution above M=0.5 — PVBC derived for incompressible flow."
        else:
            A_c = min(30.0 / beta, 100.0)
            formula_used = f"A = 30c / β = 30c / {beta:.4f} = {A_c:.1f}c  (P-G scaled standard)"
            reason = f"Prandtl-Glauert scaling: A = 30c / β = 30c / {beta:.4f} = {A_c:.1f}c."
    elif regime == "high_transonic":
        A_c = 100.0
        formula_used = "A = 100c  (empirical — P-G singular near M=1)"
        reason = "P-G singular near M=1. Empirical 100c based on Jameson & Vassberg AIAA 2008-0145."
        if bc_type == "PVBC":
            st.warning("PVBC not valid at high transonic Mach numbers.")
    else:
        A_c = 60.0
        formula_used = f"A = 60c  (Mach cone µ = {mu_deg:.1f}°; square domain very inefficient)"
        reason = f"Supersonic: Mach cone µ = {mu_deg:.1f}°. Square domain very inefficient — use C-grid."
        if bc_type == "PVBC":
            st.error("PVBC not valid for supersonic flow.")

    A_m = A_c * chord
    total_size_m = 2.0 * A_m
    blockage_pct = (0.12 * chord / total_size_m) * 100
    return {
        "mach": mach, "regime": regime, "beta": beta, "mu_deg": mu_deg,
        "A_c": A_c, "A_m": A_m, "total_size_m": total_size_m,
        "blockage_pct": blockage_pct, "reason": reason, "formula_used": formula_used,
        "mesh_type": "Square Grid (2A × 2A)",
    }

# ── Airfoil max-thickness lookup (fraction of chord) ───────────────────
# Used for blockage calculation instead of hardcoded 0.12.
# NACA 4-digit: 4th+3rd digits = thickness %. Flat plate → near-zero, use 0.
AIRFOIL_THICKNESS = {
    "NACA 2412":               0.12,
    "NACA 0012":               0.12,
    "NACA 4412":               0.12,
    "NACA 23012":              0.12,
    "Flat Plate / Bluff Body": 0.01,   # negligible thickness; use 1% as conservative minimum
}

def get_airfoil_thickness(airfoil_name):
    """Return max thickness as fraction of chord for blockage calculation."""
    return AIRFOIL_THICKNESS.get(airfoil_name, 0.12)

# ── Mesh blueprint ──────────────────────────────────────────────────────
def calculate_mesh_blueprint(velocity, chord, target_yplus=1.0, growth_rate=1.15):
    density, viscosity = 1.225, 1.789e-5
    reynolds      = (density * velocity * chord) / viscosity
    cf            = 0.026 * math.pow(reynolds, -1/7)
    tau_w         = 0.5 * cf * density * math.pow(velocity, 2)
    u_tau         = math.sqrt(tau_w / density)
    first_cell_m  = (target_yplus * viscosity) / (density * u_tau)
    first_cell_mm = first_cell_m * 1000

    # ── BL thickness: Prandtl 1/5-power law ───────────────────────────
    # δ/c = 0.37 · Re_c^(−1/5)  →  δ = 0.37 · c · Re^(−0.2)
    delta         = 0.37 * chord * math.pow(reynolds, -0.2)
    delta_mm      = delta * 1000

    # ── Layer count via geometric series inversion ─────────────────────
    # Total BL height = Δy · (r^N − 1) / (r − 1) = δ
    # r^N = 1 + δ·(r−1)/Δy
    # N   = log(1 + δ·(r−1)/Δy) / log(r)
    # Equivalent form used: 1 − δ·(1−r)/Δy  ≡  1 + δ·(r−1)/Δy  ✓
    geometric_argument = 1.0 + (delta * (growth_rate - 1.0)) / first_cell_m
    if geometric_argument > 1.0:
        total_layers = math.ceil(math.log(geometric_argument) / math.log(growth_rate))
    else:
        # geometric_argument ≤ 1 → log undefined or negative.
        # This means δ·(r−1)/Δy ≥ 1, which is physically impossible for
        # a well-posed grid (BL thicker than a single inflated cell at this growth rate).
        # Surface the problem to the user rather than silently falling back.
        total_layers = None   # sentinel — handled in UI below

    actual_yplus = (density * u_tau * first_cell_m) / viscosity
    return reynolds, first_cell_mm, total_layers, delta_mm, actual_yplus, u_tau

# ── FIX 3: y+ zone — add outer-layer upper bound ──────────────────────
def get_yplus_zone(yplus):
    """
    Classify y+ into the standard three near-wall zones plus outer layer.

    Zone boundaries (Wilcox 2006 / ANSYS Fluent Theory Guide):
      y⁺ ≤ 5    → viscous sublayer  (resolved with near-wall model)
      5 < y⁺ ≤ 30 → buffer layer   (AVOID — neither model valid here)
      30 < y⁺ ≤ 300 → log-law region (wall functions valid)
      y⁺ > 300  → outer layer      (wall functions break down)

    Note: for fully-resolved LES/DNS the sublayer target is y⁺ ≈ 1.
    The threshold of 5 is a practical upper limit for near-wall RANS.
    """
    if yplus <= 5:
        return "viscous_sublayer"
    elif yplus <= 30:
        return "buffer"
    elif yplus <= 300:
        return "log_law"
    else:
        return "outer_layer"

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

# ── Page Layout ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Ansys CFD Meshing Engine", page_icon="", layout="wide")
st.title("Ansys CFD Meshing Engine")
st.write("Airfoil V&V · Mesh Blueprint · **Mach-aware Domain Sizing** · Compressible C-Grid + Square Grid")
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
| Viscous sublayer | ≤ 5 | k-w SST, S-A |
| AVOID buffer | 5–30 | neither |
| Log-law | 30–300 | k-e, RSM |
| Outer layer | > 300 | wall fn. invalid |
""")
    st.markdown("---")
    st.markdown("##### C-grid domain formulas")
    st.markdown("""
| Regime | Up | Down | Height |
|---|---|---|---|
| M<0.3 | 15c | 20c | 20c |
| M 0.3-0.8 | 15c/β | 20c/β | 20c/β |
| M 0.8-1.0 | 50c | 60c | 40c |
| M>1.0 | 10c | 60c | f(µ) |
""")
    st.markdown("##### Square grid formulas")
    st.markdown("""
| Regime | Formula |
|---|---|
| M<0.3 PVBC | A=5c |
| M<0.3 std | A=30c |
| M 0.3-0.8 | A=base/β |
| M 0.8-1.0 | A=100c |
| M>1.0 | A=60c |
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
m3.metric("P-G factor β", f"{beta_val:.4f}", help="β=√(1-M²). Domain scales by 1/β in transonic.")
m4.metric("Mach cone µ", f"{mu_val:.1f}°" if mach_val > 1 else "N/A (subsonic)")

if regime == "supersonic":
    st.error("Supersonic: use Density-Based solver + Roe-FDS + Ideal Gas + energy equation ON.")
elif regime in ("transonic", "high_transonic"):
    st.warning("Transonic: use Density-Based solver + energy equation + compressibility corrections.")

st.divider()

# ── Section 3: Domain Sizing ─────────────────────────────────────────────
st.subheader("3. CFD domain sizing")
st.write("C-Grid now uses **Prandtl-Glauert scaling** (transonic) and **Mach cone geometry** (supersonic).")

domain_strategy = st.radio("Domain topology",
                           ["Standard C-Grid (Physics-based, all Mach)",
                            "Square Grid (Academic / Compressible Golmirzaee 2024)"],
                           horizontal=True)

# FIX 2: use actual airfoil thickness for blockage
airfoil_t_frac = get_airfoil_thickness(selected_airfoil)
tui_domain_text = ""

if "C-Grid" in domain_strategy:
    domain = calculate_c_grid_domain(v_input, c_input)

    # Override blockage with airfoil-specific thickness
    domain["blockage_pct"] = (airfoil_t_frac * c_input / domain["height_m"]) * 100

    da, db, dc, dd, de = st.columns(5)
    da.metric("Upstream",    f"{domain['upstream_c']:.1f}c = {domain['upstream_m']:.2f} m")
    db.metric("Downstream",  f"{domain['downstream_c']:.1f}c = {domain['downstream_m']:.2f} m")
    dc.metric("Height",      f"{domain['height_c']:.1f}c = {domain['height_m']:.2f} m")
    dd.metric("Total width", f"{domain['total_width_m']:.2f} m")
    de.metric("Blockage",    f"{domain['blockage_pct']:.3f}%")

    with st.expander("Domain formula, boundary conditions & physics reasoning", expanded=True):
        exp_l, exp_r = st.columns(2)
        with exp_l:
            st.markdown("##### Computed domain box")
            st.write(f"**Width:** `{domain['total_width_m']:.3f} m`  "
                     f"({domain['upstream_c']:.1f}c + 1c + {domain['downstream_c']:.1f}c)")
            st.write(f"**Height:** `{domain['total_height_m']:.3f} m`  ({domain['height_c']:.1f}c)")
            st.write(f"**Airfoil LE at:** x = `{domain['upstream_m']:.3f} m` from inlet")
            st.write(f"**Mesh topology:** `{domain['mesh_type']}`")
            st.write(f"**Blockage t/c used:** `{airfoil_t_frac*100:.0f}%`  ({selected_airfoil})")

            st.markdown("##### Formula applied")
            st.code(domain['formula_detail'], language="text")

            st.markdown("##### Boundary conditions")
            st.write(f"**Left (inlet):** Velocity Inlet — U∞, TI=0.1%, L=0.07c")
            st.write(f"**Right (outlet):** Pressure Outlet — gauge P=0 Pa")
            st.write(f"**Top/bottom:** `{domain['bc_top']}`")
            st.write(f"**Airfoil surface:** No-slip Wall with inflation mesh")

            if domain['blockage_pct'] > 5.0:
                st.error(f"Blockage {domain['blockage_pct']:.2f}% > 5%! Increase domain height.")
            elif domain['blockage_pct'] > 2.0:
                st.warning(f"Blockage {domain['blockage_pct']:.2f}% — acceptable for preliminary work.")
            else:
                st.success(f"Blockage {domain['blockage_pct']:.3f}% — excellent (< 1%).")

        with exp_r:
            st.markdown("##### Physics reasoning")
            st.info(domain['reason'])

            st.markdown("##### Compressibility parameters")
            st.write(f"**β = √(1−M²):** `{domain['beta']:.4f}`")
            if mach_val > 1:
                st.write(f"**Mach cone µ = arcsin(1/M):** `{domain['mu_deg']:.2f}°`")
                st.write(f"**tan(µ):** `{math.tan(math.radians(domain['mu_deg'])):.4f}`")
            elif regime == "transonic":
                st.write(f"**1/β (domain scale factor):** `{1/domain['beta']:.4f}×`")

            st.markdown("##### Solver recommendation")
            if regime == "subsonic":
                st.success("Solver: Pressure-Based | Density: Constant\nEnergy: OFF | Coupling: SIMPLE")
            elif regime == "transonic":
                st.warning(f"Solver: Density-Based | Density: Ideal Gas\n"
                           f"Energy: ON | Flux: Roe-FDS | CFL: 5-10\n"
                           f"β={domain['beta']:.4f} → domain scaled ×{1/domain['beta']:.2f}")
            elif regime == "high_transonic":
                st.warning("Solver: Density-Based | Density: Ideal Gas\n"
                           "Energy: ON | Shock refinement required near LE\n"
                           "P-G singular — empirical Jameson 2008 values used")
            else:
                st.error(f"Solver: Density-Based | Flux: Roe-FDS\n"
                         f"Energy: ON | CFL: 1-3 (conservative)\n"
                         f"Mach cone µ={domain['mu_deg']:.1f}° → height={domain['height_c']:.1f}c")

    tui_domain_text = (
        f"; Upstream  : {domain['upstream_m']:.3f} m  ({domain['upstream_c']:.1f}c)\n"
        f"; Downstream: {domain['downstream_m']:.3f} m  ({domain['downstream_c']:.1f}c)\n"
        f"; Height    : {domain['height_m']:.3f} m  ({domain['height_c']:.1f}c)\n"
        f"; beta (P-G): {domain['beta']:.4f}  |  Mach cone mu: {domain['mu_deg']:.2f} deg\n"
        f"; Formula   : {domain['formula_detail'].splitlines()[0]}\n"
        f"; Airfoil LE: x = {domain['upstream_m']:.3f} m from inlet"
    )

else:
    bc_type = st.selectbox("Boundary condition type",
                           ["Standard Boundaries (Slip/Symmetry)", "Point Vortex BC (PVBC)"])
    sq = calculate_square_domain(v_input, c_input, bc_type)

    # FIX 2: airfoil-specific blockage for square domain
    sq["blockage_pct"] = (airfoil_t_frac * c_input / sq["total_size_m"]) * 100

    da, db, dc, dd, de = st.columns(5)
    da.metric("Parameter A",  f"{sq['A_c']:.1f}c = {sq['A_m']:.2f} m")
    db.metric("Total width",  f"{sq['total_size_m']:.2f} m")
    dc.metric("Total height", f"{sq['total_size_m']:.2f} m")
    dd.metric("Blockage",     f"{sq['blockage_pct']:.3f}%")
    de.metric("P-G factor β", f"{sq['beta']:.4f}")

    with st.expander("Square domain physics, formulas & boundary conditions", expanded=True):
        exp_l, exp_r = st.columns(2)
        with exp_l:
            st.markdown("##### Formula applied")
            st.code(sq['formula_used'], language="text")
            if regime == "transonic":
                st.metric("1/β scaling factor", f"{1/sq['beta']:.3f}x")
            st.markdown("##### Boundary conditions")
            if bc_type == "Standard Boundaries (Slip/Symmetry)":
                st.write("**All 4 outer edges:** Slip wall / Symmetry")
                if regime == "subsonic":
                    st.latex(r"C_{d,\infty} \approx C_d - 0.0205 \frac{C_l^2}{A}")
                elif regime == "transonic":
                    st.warning("Use Pressure Far-Field BC — slip walls reflect shocks.")
            else:
                st.write("**All 4 outer edges:** Point Vortex BC (PVBC)")
                if regime != "subsonic":
                    st.warning("PVBC derived for incompressible flow. Verify with GI test above M=0.3.")
        with exp_r:
            st.markdown("##### Why this size?")
            st.info(sq['reason'])

    tui_domain_text = (
        f"; Square Parameter A: {sq['A_m']:.3f} m  ({sq['A_c']:.1f}c)\n"
        f"; Total size : {sq['total_size_m']:.3f} m x {sq['total_size_m']:.3f} m\n"
        f"; P-G beta   : {sq['beta']:.4f}  (1/beta = {1/sq['beta']:.3f})\n"
        f"; Formula    : {sq['formula_used']}\n"
        f"; Airfoil center at domain origin (0, 0)"
    )

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
    (st.success if yplus_mode == "near_wall" else st.info)(
        f"**Primary:** {rec['primary']}  |  **Alt:** {rec['secondary']}\n\n"
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
    result = calculate_mesh_blueprint(v_input, c_input, target_yplus, growth_rate)
    calc_re, calc_dy, calc_layers, calc_delta, actual_yplus, u_tau = result

    # ── FIX 1: surface the degenerate layer-count case ─────────────────
    if calc_layers is None:
        st.error(
            f"**Layer count calculation failed — degenerate grid geometry.**\n\n"
            f"The boundary layer thickness δ = {calc_delta:.4f} mm is too large relative to "
            f"the first cell height Δy = {calc_dy:.5f} mm at growth rate r = {growth_rate:.2f}. "
            f"The geometric series argument (1 + δ·(r−1)/Δy) is ≤ 1, making log undefined.\n\n"
            f"**Fix:** increase growth rate (try 1.2–1.3) or reduce chord / velocity "
            f"to increase Reynolds number and shrink δ relative to Δy."
        )
        st.stop()

    zone = get_yplus_zone(actual_yplus)

    st.success(f"Re: {calc_re:.2e}  |  M: {mach_val:.4f}  |  β: {beta_val:.4f}  |  {REGIME_LABEL[regime]}")

    col_m, col_s, col_d = st.columns(3)
    with col_m:
        st.markdown("#### Meshing module")
        st.write(f"**First cell height:** `{calc_dy:.5f} mm`")
        st.write(f"**Growth rate:** `{growth_rate}`")
        st.write(f"**Total inflation layers:** `{calc_layers}`")
        st.write(f"**BL thickness δ:** `{calc_delta:.3f} mm`")
        st.write(f"**u_τ (friction velocity):** `{u_tau:.4f} m/s`")
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
            st.write(f"**Upstream:** `{domain['upstream_m']:.3f} m` ({domain['upstream_c']:.1f}c)")
            st.write(f"**Downstream:** `{domain['downstream_m']:.3f} m` ({domain['downstream_c']:.1f}c)")
            st.write(f"**Height:** `{domain['height_m']:.3f} m` ({domain['height_c']:.1f}c)")
            st.write(f"**β:** `{domain['beta']:.4f}`  |  **Blockage:** `{domain['blockage_pct']:.3f}%`")
        else:
            st.write(f"**A:** `{sq['A_m']:.3f} m` ({sq['A_c']:.1f}c)")
            st.write(f"**Total:** `{sq['total_size_m']:.3f}m × {sq['total_size_m']:.3f}m`")
            st.write(f"**β:** `{sq['beta']:.4f}`  |  **Blockage:** `{sq['blockage_pct']:.3f}%`")

    st.divider()
    st.markdown("#### Mesh quality checklist")
    checks = []

    # ── FIX 3: full four-zone y+ check ────────────────────────────────
    if zone == "buffer":
        checks.append(("FAIL","y+ zone",
                        f"BUFFER ZONE (y+={actual_yplus:.1f}). Max 20% Cf error. "
                        f"Adjust first cell height — target y+<5 or y+=30–300."))
    elif zone == "outer_layer":
        checks.append(("FAIL","y+ zone",
                        f"OUTER LAYER (y+={actual_yplus:.1f} > 300). Wall functions are invalid here. "
                        f"Reduce first cell height to bring y+ below 300."))
    elif zone == "viscous_sublayer":
        checks.append(("PASS","y+ zone", f"Viscous sublayer resolved (y+={actual_yplus:.1f} ≤ 5)."))
    else:
        checks.append(("PASS","y+ zone", f"Log-law region (y+={actual_yplus:.1f}, 30–300)."))

    checks.append(("PASS" if 1.1<=growth_rate<=1.2 else "WARN","Growth rate",
                   f"{growth_rate} — {'optimal (1.1-1.2)' if 1.1<=growth_rate<=1.2 else 'outside 1.1-1.2 range'}."))

    if calc_layers < 15:
        checks.append(("WARN","Layer count",f"{calc_layers} — low. May not cover full BL."))
    elif calc_layers > 60:
        checks.append(("WARN","Layer count",f"{calc_layers} — high. Check growth rate."))
    else:
        checks.append(("PASS","Layer count",f"{calc_layers} layers — covers δ={calc_delta:.2f}mm."))

    if yplus_mode=="near_wall" and zone in ("log_law","outer_layer"):
        checks.append(("WARN","Model compat.","Near-wall model selected but y+ is NOT in viscous sublayer."))
    elif yplus_mode=="wall_function" and zone=="viscous_sublayer":
        checks.append(("WARN","Model compat.","Wall function selected but y+ is in viscous sublayer — invalid."))
    elif yplus_mode=="wall_function" and zone=="outer_layer":
        checks.append(("FAIL","Model compat.","Wall function selected but y+ > 300 — outer layer, functions invalid."))
    else:
        checks.append(("PASS","Model compat.",f"{rec['primary']} consistent with y+={actual_yplus:.1f}."))

    if regime in ("transonic","high_transonic","supersonic"):
        checks.append(("WARN","Compressibility",
                       f"M={mach_val:.3f}, β={beta_val:.4f}. Density-based, ideal gas, energy ON."))
    else:
        checks.append(("PASS","Compressibility",f"M={mach_val:.4f} — incompressible valid."))

    if calc_re < 5e4:
        checks.append(("WARN","Reynolds",f"{calc_re:.2e} — consider laminar/transition model."))
    elif calc_re > 1e7:
        checks.append(("WARN","Reynolds",f"{calc_re:.2e} — high Re, check mesh density."))
    else:
        checks.append(("PASS","Reynolds",f"{calc_re:.2e} — RANS applicable."))

    icon_map = {"PASS":"✅","WARN":"⚠️","FAIL":"❌"}
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
        f"; Target y+          : {target_yplus}\n"
        f"; BL thickness δ     : {calc_delta:.4f} mm\n"
        f"; Friction velocity  : {u_tau:.4f} m/s\n\n"
        f"; ── Domain geometry (SpaceClaim / DesignModeler) ─────\n"
        f"{tui_domain_text}\n",
        language="text"
    )
