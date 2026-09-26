"""CRUCIBLE — a real-time 3D MLS-MPM physics sandbox (Taichi).

MLS-MPM (Moving Least Squares Material Point Method), the practical/stable
variant used by the canonical Taichi mpm88/mpm99 examples, extended from 2D to
3D. Each particle carries a ``material_id`` selecting its constitutive class
(weakly-compressible liquid, neo-Hookean elastic, Drucker-Prager granular),
rest density (encoded as per-particle mass), stiffness and colour. Scenes that
use heat (lava, melt) also diffuse a per-particle temperature through the grid
and transform materials at temperature thresholds.

Run interactively (opens a GGUI window):
    python src/mpm_fluid.py [--scene pool|empty|drop|jelly|sand|lava|melt|...]

Run a headless check (no window; asserts no NaN, bounded velocity, and the
scene's own success criterion):
    python src/mpm_fluid.py --headless --scene drop --frames 600
"""

import argparse
import sys

import numpy as np
import taichi as ti

# Taichi must be initialized before any field is allocated. Do an early, minimal
# scan of argv for backend + scene so module-level fields below can be sized.
def _early_flag(name, default):
    # Accepts both "--flag value" and "--flag=value". Malformed input falls back
    # to the default; argparse in main() then reports it properly.
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


_arch = _early_flag("--arch", "cuda")
SCENE = _early_flag("--scene", "pool")   # re-read (and validated) in main()
ti.init(arch=getattr(ti, _arch), default_fp=ti.f32)

# ----------------------------------------------------------------------------
# Simulation parameters
# ----------------------------------------------------------------------------
DIM = 3
N_GRID = int(_early_flag("--grid", 64))   # scratch grid resolution per axis
DX = 1.0 / N_GRID
INV_DX = float(N_GRID)

# A coarse sim grid is fine; render quality is decoupled (later phase). Particle
# count is sized for ~8 particles/cell over the filled region — MLS-MPM needs
# that density or the mass field is sparse/noisy and interfaces turn to mush.
# The jelly/sand scenes fill a small region, so they need far fewer particles.
# Overridable with --particles (fewer = faster, coarser surface).
N_SIM = int(_early_flag("--particles",
                        0 if SCENE == "empty" else
                        150_000 if SCENE in ("jelly", "sand") else
                        400_000 if SCENE == "pool" else 800_000))
# A pool of initially-inactive particles reserved for the interactive spawn
# brush ("add material"). They cost nothing until activated (skipped in the
# substep). Total allocation = scene particles + pool. The "empty" scene seeds
# nothing, so give it a large pool to build from.
N_POOL = int(_early_flag("--pool", 700_000 if SCENE == "empty" else 250_000))
N_PARTICLES = N_SIM + N_POOL

# Timestep is sized per scene to the *stiffest material present*: stability is
# bounded by the elastic sound speed c=sqrt(E/rho), dt < dx/c. Pure fluids
# (E=400, c~20) tolerate a much larger dt than stiff solids (sand/ice/stone),
# so they need far fewer substeps for the same wall-clock sim-time per frame
# (dt*SUBSTEPS is held ~constant, ~2.5e-3, to keep the visual speed identical).
def scene_timestep(scene):
    if scene in ("pool", "empty"):
        return 2.5e-4, 9      # sandbox baseline. dt is jelly-stable (E=4e3);
        #                       substeps = sim-time/frame (motion liveliness),
        #                       overridden live by the GUI quality slider.
    if scene in ("drop", "layers", "mixed"):
        return 3.5e-4, 7      # weakly-compressible fluids only
    if scene == "jelly":
        return 3.0e-4, 8      # soft elastic (hardening 0.3)
    if scene == "sand":
        return 1.8e-4, 14     # granular, E=1e4
    if scene == "lava":
        return 2.3e-4, 11     # stone (E=8e3, c~55) is the binding material
    if scene == "melt":
        return 2.0e-4, 13     # ice (E=4e3, c~66) is the binding material
    return 1.0e-4, 25


DT, SUBSTEPS = scene_timestep(SCENE)
# Heat is only meaningful for scenes whose materials transform. Compiling the
# whole temperature scatter/diffuse/gather/transform out of the substep when
# it's unused removes several grid passes + per-particle work.
HEAT_SCENES = ("lava", "melt")
HEAT_ON = SCENE in HEAT_SCENES

P_VOL = (DX * 0.5) ** DIM        # one particle nominally fills 1/8 of a cell
GRAVITY = 20.0   # stronger-than-earth fall so motion doesn't feel floaty in the
#                  sim's slow-motion timescale (was 9.8)
BOUND = 3                        # wall thickness in grid cells

# --- Material table ---------------------------------------------------------
# Each material belongs to a *class* with its own constitutive model:
#   LIQUID  — weakly-compressible EOS pressure (volume ratio J). Rest density is
#             encoded as per-particle mass, so density layering is emergent.
#   ELASTIC — neo-Hookean solid using the full deformation gradient F. Wobbles
#             and bounces; high stiffness ≈ rigid.
#   GRANULAR — Drucker-Prager elastoplastic (sand). Same Hencky elasticity as a
#              solid, but every step the deformation is projected onto a friction
#              yield cone, so it piles and slumps at an angle of repose.
CLS_LIQUID, CLS_ELASTIC, CLS_GRANULAR = 0, 1, 2

# Material ids. The first four drive the Phase 2/3 demos; the rest are the
# Phase 4 heat-driven set that transforms into one another by temperature.
(LIGHT, HEAVY, JELLY, SAND, WATER, ICE, STEAM, LAVA, STONE,
 WATER_C, WATER_P) = range(11)
NUM_MAT = 11
# WATER_C / WATER_P are cosmetic water variants (identical physics to WATER,
# just a different colour) so the sandbox can have several colours of water.
#                LIGHT      HEAVY      JELLY      SAND       WATER
#                ICE        STEAM      LAVA       STONE      WATER_C     WATER_P
MAT_CLASS_PY = [CLS_LIQUID, CLS_LIQUID, CLS_ELASTIC, CLS_GRANULAR, CLS_LIQUID,
                CLS_ELASTIC, CLS_LIQUID, CLS_LIQUID, CLS_ELASTIC,
                CLS_LIQUID, CLS_LIQUID]
MAT_RHO_PY = [1.0, 6.0, 1.0, 1.6, 1.0,
              0.92, 0.3, 2.5, 2.6, 1.0, 1.0]           # rest densities
# For LIQUID, E is the bulk EOS stiffness; for ELASTIC/GRANULAR, E is Young's.
# Stone/ice kept only as stiff as stability needs: their sound speed sets the
# substep count for the lava/melt scenes, so softening them (while still clearly
# rigid solids) directly buys frames. See scene_timestep().
# Water bulk stiffness raised 400->1200: softer water is compressible and
# oozes like batter; stiffer water holds volume and flows/splashes like water.
MAT_E_PY = [400.0, 400.0, 4.0e3, 1.0e4, 1200.0,
            4.0e3, 200.0, 400.0, 8.0e3, 1200.0, 1200.0]
MAT_NU_PY = [0.0, 0.0, 0.2, 0.3, 0.0,
             0.2, 0.0, 0.0, 0.2, 0.0, 0.0]             # Poisson ratio (solids only)
MAT_COLOR_PY = [[0.25, 0.55, 0.95],                    # light = blue
                [0.95, 0.45, 0.20],                    # heavy = orange
                [0.35, 0.85, 0.45],                    # jelly = green
                [0.85, 0.72, 0.38],                    # sand  = tan
                [0.25, 0.55, 0.95],                    # water = blue
                [0.70, 0.85, 0.95],                    # ice   = pale cyan
                [0.80, 0.80, 0.85],                    # steam = light grey
                [0.95, 0.35, 0.10],                    # lava  = red-orange
                [0.35, 0.33, 0.36],                    # stone = dark grey
                [0.20, 0.85, 0.80],                    # water_c = teal
                [0.60, 0.40, 0.95]]                    # water_p = violet
# Initial temperature per material (deg-C-like scale; ambient ~20).
MAT_TEMP0_PY = [20.0, 20.0, 20.0, 20.0, 20.0,
                -10.0, 110.0, 1200.0, 20.0, 20.0, 20.0]

# Elastic hardening factor (softens the jelly so it visibly wobbles, per mpm99).
# Sand keeps full stiffness (hardening folded into its Lamé params via H=1).
ELASTIC_H = 0.3

# Drucker-Prager friction: angle of repose for the granular class.
SAND_FRICTION_DEG = 45.0
_sp = np.sin(np.radians(SAND_FRICTION_DEG))
SAND_ALPHA = float(np.sqrt(2.0 / 3.0) * 2.0 * _sp / (3.0 - _sp))

# --- Heat + transformations (Phase 4) ---------------------------------------
# Grid heat diffusion strength per substep (dimensionless; stable while
# HEAT_DIFF * 2*DIM < 1). Kept modest so heat stays *local* — a tiny lava ball
# should flash-boil the water it touches, not slowly cook the whole pool.
HEAT_DIFF = 0.10
# Radiative cooling: every substep, relax temperature toward ambient. Without a
# heat sink, diffusion alone just conserves enthalpy and the system equilibrates
# to one hot average (so everything boils). Cooling lets the lava actually shed
# its heat and solidify, and lets transient steam re-condense. The rate is tuned
# so adjacent water flash-boils before the heat (and the lava) cools away, while
# distant water never gets hot enough.
AMBIENT_T = 20.0
COOL_RATE = 0.0020
# Transition thresholds (material, condition) -> new material. Contact effects
# (e.g. lava+water -> stone+steam) are emergent from diffusion, not scripted.
T_MELT_ICE = 0.0       # ICE  -> WATER above this
T_FREEZE_WATER = -5.0  # WATER -> ICE below this
T_BOIL_WATER = 100.0   # WATER -> STEAM above this
T_CONDENSE_STEAM = 95.0  # STEAM -> WATER below this
T_SOLIDIFY_LAVA = 600.0  # LAVA -> STONE below this
T_MELT_STONE = 1100.0  # STONE -> LAVA above this

# ----------------------------------------------------------------------------
# Fields
# ----------------------------------------------------------------------------
x = ti.Vector.field(DIM, ti.f32, N_PARTICLES)   # position
v = ti.Vector.field(DIM, ti.f32, N_PARTICLES)   # velocity
C = ti.Matrix.field(DIM, DIM, ti.f32, N_PARTICLES)  # affine velocity (APIC/MLS-MPM)
J = ti.field(ti.f32, N_PARTICLES)               # volume ratio for the LIQUID class
F = ti.Matrix.field(DIM, DIM, ti.f32, N_PARTICLES)  # deformation gradient (ELASTIC class)
mass = ti.field(ti.f32, N_PARTICLES)            # per-particle mass (= P_VOL * rest density)
mat = ti.field(ti.i32, N_PARTICLES)             # material id
temp = ti.field(ti.f32, N_PARTICLES)            # per-particle temperature
active = ti.field(ti.i32, N_PARTICLES)          # 1 = simulated/drawn, 0 = free pool

grid_v = ti.Vector.field(DIM, ti.f32, (N_GRID,) * DIM)
grid_v0 = ti.Vector.field(DIM, ti.f32, (N_GRID,) * DIM)  # velocity BEFORE forces
grid_f = ti.Vector.field(DIM, ti.f32, (N_GRID,) * DIM)   # internal (stress) force
grid_m = ti.field(ti.f32, (N_GRID,) * DIM)
grid_T = ti.field(ti.f32, (N_GRID,) * DIM)      # mass-weighted grid temperature
grid_Tn = ti.field(ti.f32, (N_GRID,) * DIM)     # diffusion scratch buffer

# Container shape for the walls: 0 = box (full domain), 1 = round (cylinder).
container_shape = ti.field(ti.i32, ())
CONTAINER_R = 0.46                              # round-container radius (in [0,1])

# FLIP/PIC blend for liquids: high fraction keeps kinetic energy (low numerical
# viscosity -> slippery, splashy water, not viscous "batter"). Solids stay PIC.
# It's a runtime field so a scene can pre-settle with heavy damping (low FLIP,
# starts dead-flat) then raise it for slippery interaction.
FLIP_LIQUID = 0.92
flip_ratio = ti.field(ti.f32, ())
floor_sticky = ti.field(ti.i32, ())   # 1 = no-slip floor (basal friction for sand)

colors = ti.Vector.field(3, ti.f32, N_PARTICLES)  # per-particle render color

# Material properties as device fields, indexed by material id.
mat_class = ti.field(ti.i32, NUM_MAT)
mat_rho = ti.field(ti.f32, NUM_MAT)
mat_E = ti.field(ti.f32, NUM_MAT)
mat_mu = ti.field(ti.f32, NUM_MAT)    # Lamé mu (elastic, hardening baked in)
mat_la = ti.field(ti.f32, NUM_MAT)    # Lamé lambda (elastic, hardening baked in)
mat_temp0 = ti.field(ti.f32, NUM_MAT)  # initial temperature
mat_color = ti.Vector.field(3, ti.f32, NUM_MAT)


def load_materials():
    for m in range(NUM_MAT):
        mat_class[m] = MAT_CLASS_PY[m]
        mat_rho[m] = MAT_RHO_PY[m]
        mat_E[m] = MAT_E_PY[m]
        e, nu = MAT_E_PY[m], MAT_NU_PY[m]
        # Lamé parameters from Young's modulus + Poisson ratio. The jelly is
        # softened by the hardening factor so it visibly wobbles; sand keeps full
        # stiffness. (Unused by the liquid class.)
        h = ELASTIC_H if MAT_CLASS_PY[m] == CLS_ELASTIC else 1.0
        mat_mu[m] = h * e / (2.0 * (1.0 + nu))
        mat_la[m] = h * e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mat_temp0[m] = MAT_TEMP0_PY[m]
        mat_color[m] = ti.Vector(MAT_COLOR_PY[m])


@ti.func
def hencky_kirchhoff_stress(Fp, mu, la):
    # Kirchhoff stress of a Hencky (log-strain) elastic solid:
    #   tau = U * (2*mu*eps + la*tr(eps) I) * U^T,  eps_i = log(sigma_i).
    # Used by the granular class (the elastic part of its elastoplastic model).
    U, sig, V = ti.svd(Fp)
    eps = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1]), ti.log(sig[2, 2])])
    tr = eps.sum()
    center = ti.Matrix.zero(ti.f32, DIM, DIM)
    for d in ti.static(range(DIM)):
        center[d, d] = 2.0 * mu * eps[d] + la * tr
    return U @ center @ U.transpose()


@ti.func
def drucker_prager_project(Fp, mu, la):
    # Return-mapping: project the trial deformation onto the Drucker-Prager
    # friction cone so the material yields (slumps) under shear it can't support.
    U, sig, V = ti.svd(Fp)
    eps = ti.Vector([ti.log(ti.max(sig[0, 0], 1e-6)),
                     ti.log(ti.max(sig[1, 1], 1e-6)),
                     ti.log(ti.max(sig[2, 2], 1e-6))])
    tr = eps.sum()
    eps_hat = eps - ti.Vector([tr, tr, tr]) / DIM
    eps_hat_norm = eps_hat.norm()
    new_eps = eps
    if tr > 0.0:
        # Volumetric expansion: cohesionless sand can't take tension -> tip.
        new_eps = ti.Vector.zero(ti.f32, DIM)
    elif eps_hat_norm > 1e-12:
        dgamma = (eps_hat_norm
                  + (DIM * la + 2.0 * mu) / (2.0 * mu) * tr * SAND_ALPHA)
        if dgamma > 0.0:
            new_eps = eps - (dgamma / eps_hat_norm) * eps_hat
    new_sig = ti.Matrix.zero(ti.f32, DIM, DIM)
    for d in ti.static(range(DIM)):
        new_sig[d, d] = ti.exp(new_eps[d])
    return U @ new_sig @ V.transpose()


@ti.kernel
def substep():
    # --- reset scratch grid -------------------------------------------------
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector.zero(ti.f32, DIM)
        grid_v0[I] = ti.Vector.zero(ti.f32, DIM)
        grid_f[I] = ti.Vector.zero(ti.f32, DIM)
        grid_m[I] = 0.0
        if ti.static(HEAT_ON):
            grid_T[I] = 0.0

    # --- P2G: scatter mass + momentum (atomic via +=) -----------------------
    for p in x:
        if active[p] == 0:
            continue
        base = (x[p] * INV_DX - 0.5).cast(int)
        base = ti.min(ti.max(base, 0), N_GRID - 3)   # clamp: a NaN/escaped
        #   particle must never index the grid out of bounds (hard GPU crash)
        fx = x[p] * INV_DX - base.cast(ti.f32)
        # quadratic B-spline weights
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        m_p = mass[p]
        cls = mat_class[mat[p]]
        # MLS-MPM internal-force term: affine = (-dt*p_vol*4*inv_dx^2) * Cauchy
        # stress  +  m * C. The stress is mass-independent (force/volume).
        stress = ti.Matrix.zero(ti.f32, DIM, DIM)
        if cls == CLS_LIQUID:
            # weakly-compressible EOS. Cauchy stress (J-1)*E so that the shared
            # MLS prefactor below reproduces the verified Phase-1/2 liquid term.
            stress = (J[p] - 1.0) * mat_E[mat[p]] * ti.Matrix.identity(ti.f32, DIM)
        elif cls == CLS_ELASTIC:  # neo-Hookean (mpm99 form)
            Fp = F[p]
            U, sig, V = ti.svd(Fp)
            Jd = Fp.determinant()
            mu, la = mat_mu[mat[p]], mat_la[mat[p]]
            stress = (2.0 * mu * (Fp - U @ V.transpose()) @ Fp.transpose()
                      + ti.Matrix.identity(ti.f32, DIM) * la * Jd * (Jd - 1.0))
        else:  # CLS_GRANULAR — Hencky elastic stress (yield handled in G2P)
            stress = hencky_kirchhoff_stress(F[p], mat_mu[mat[p]], mat_la[mat[p]])
        # Stress is now applied as a separate grid FORCE (not folded into the
        # momentum) so a FLIP/PIC blend is correct: the grid velocity *change*
        # then carries the force impulse. APIC affine (m*C) stays in momentum.
        sforce = (-P_VOL * 4.0 * INV_DX * INV_DX) * stress
        aff = m_p * C[p]
        for offset in ti.static(ti.grouped(ti.ndrange(*([3] * DIM)))):
            dpos = (offset.cast(ti.f32) - fx) * DX
            weight = 1.0
            for d in ti.static(range(DIM)):
                weight *= w[offset[d]][d]
            grid_v[base + offset] += weight * (m_p * v[p] + aff @ dpos)
            grid_f[base + offset] += weight * (sforce @ dpos)
            grid_m[base + offset] += weight * m_p
            if ti.static(HEAT_ON):
                grid_T[base + offset] += weight * m_p * temp[p]   # mass-weighted heat

    # --- grid update: momentum -> velocity, gravity, walls ------------------
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0.0:
            v_old = grid_v[I] / grid_m[I]            # APIC velocity, pre-force
            grid_v0[I] = v_old                       # FLIP reference
            if ti.static(HEAT_ON):
                grid_T[I] /= grid_m[I]   # mass-weighted temperature at the node
            v_new = v_old + DT * (grid_f[I] / grid_m[I])   # internal stress force
            v_new[1] -= DT * GRAVITY
            # floor + ceiling (always): zero inward normal component (free-slip)
            if I[1] < BOUND and v_new[1] < 0.0:
                v_new[1] = 0.0
            if I[1] > N_GRID - BOUND and v_new[1] > 0.0:
                v_new[1] = 0.0
            if container_shape[None] == 0:
                # box: x/z side walls
                for d in ti.static((0, 2)):
                    if I[d] < BOUND and v_new[d] < 0.0:
                        v_new[d] = 0.0
                    if I[d] > N_GRID - BOUND and v_new[d] > 0.0:
                        v_new[d] = 0.0
            else:
                # round: cylindrical side wall — zero outward radial velocity
                gx = (I[0] + 0.5) * DX - 0.5
                gz = (I[2] + 0.5) * DX - 0.5
                rr = ti.sqrt(gx * gx + gz * gz)
                if rr > CONTAINER_R and rr > 1e-6:
                    nx, nz = gx / rr, gz / rr
                    vn = v_new[0] * nx + v_new[2] * nz
                    if vn > 0.0:
                        v_new[0] -= vn * nx
                        v_new[2] -= vn * nz
            # Optional no-slip floor: granular material grips the base (basal
            # friction) instead of sliding freely, so a pile can actually lock.
            if floor_sticky[None] == 1 and I[1] < BOUND:
                v_new = ti.Vector.zero(ti.f32, DIM)
            grid_v[I] = v_new

    # --- heat diffusion: explicit Laplacian over occupied cells -------------
    # Only cells with mass participate; empty cells are treated as insulating,
    # so heat flows through the material, not through "air".
    if ti.static(HEAT_ON):
        for I in ti.grouped(grid_m):
            grid_Tn[I] = grid_T[I]
            if grid_m[I] > 0.0:
                lap = 0.0
                for d in ti.static(range(DIM)):
                    for s in ti.static((-1, 1)):
                        Jn = I + ti.Vector.unit(DIM, d) * s
                        if Jn[d] >= 0 and Jn[d] < N_GRID and grid_m[Jn] > 0.0:
                            lap += grid_T[Jn] - grid_T[I]
                grid_Tn[I] = (grid_T[I] + HEAT_DIFF * lap
                              - COOL_RATE * (grid_T[I] - AMBIENT_T))
        for I in ti.grouped(grid_m):
            grid_T[I] = grid_Tn[I]

    # --- G2P: gather velocity + affine + heat, advect, transform -----------
    for p in x:
        if active[p] == 0:
            continue
        base = (x[p] * INV_DX - 0.5).cast(int)
        base = ti.min(ti.max(base, 0), N_GRID - 3)   # clamp: a NaN/escaped
        #   particle must never index the grid out of bounds (hard GPU crash)
        fx = x[p] * INV_DX - base.cast(ti.f32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        cls = mat_class[mat[p]]
        vp_old = v[p]                      # FLIP needs the particle's old velocity
        v_pic = ti.Vector.zero(ti.f32, DIM)
        v_flip = vp_old                    # accumulates grid velocity *changes*
        new_C = ti.Matrix.zero(ti.f32, DIM, DIM)
        new_T = 0.0
        for offset in ti.static(ti.grouped(ti.ndrange(*([3] * DIM)))):
            dpos = (offset.cast(ti.f32) - fx) * DX
            weight = 1.0
            for d in ti.static(range(DIM)):
                weight *= w[offset[d]][d]
            g_vn = grid_v[base + offset]        # velocity after forces
            g_vo = grid_v0[base + offset]       # velocity before forces
            v_pic += weight * g_vn
            v_flip += weight * (g_vn - g_vo)
            new_C += 4.0 * INV_DX * weight * g_vn.outer_product(dpos)
            if ti.static(HEAT_ON):
                new_T += weight * grid_T[base + offset]
        # FLIP/PIC blend: liquids keep most of their energy (slippery water);
        # solids/granular stay PIC for stability.
        flip = flip_ratio[None] if cls == CLS_LIQUID else 0.0
        new_v = (1.0 - flip) * v_pic + flip * v_flip
        v[p] = new_v
        C[p] = new_C
        if ti.static(HEAT_ON):
            temp[p] = new_T
        if cls == CLS_LIQUID:
            # advance volume ratio by the velocity divergence (trace of C),
            # clamped so brush-spawning into already-full fluid can't drive J
            # toward 0 -> a runaway EOS pressure spike that explodes (the
            # weakly-compressible stress is (J-1)*E, so bounding J bounds it).
            J[p] = ti.min(ti.max(J[p] * (1.0 + DT * new_C.trace()), 0.6), 1.4)
        else:
            # elastic predictor: evolve the deformation gradient
            F_trial = (ti.Matrix.identity(ti.f32, DIM) + DT * new_C) @ F[p]
            if cls == CLS_GRANULAR:
                # plastic corrector: project onto the Drucker-Prager yield cone
                F_trial = drucker_prager_project(F_trial, mat_mu[mat[p]], mat_la[mat[p]])
            F[p] = F_trial
        x[p] += DT * new_v

        # --- safety net: a blow-up (bad brush swipe, stiff contact) must never
        # produce a NaN/escaped particle. That would index the grid OOB or feed
        # NaN vertices to GGUI -> a hard crash (esp. when switching to the
        # particle/surface renderers). Reset runaway state and keep x in-box.
        bad = 0
        for d in ti.static(range(DIM)):
            if ti.math.isnan(new_v[d]) or ti.abs(new_v[d]) > 1.0e4:
                bad = 1
        if bad == 1:
            v[p] = ti.Vector.zero(ti.f32, DIM)
            C[p] = ti.Matrix.zero(ti.f32, DIM, DIM)
            F[p] = ti.Matrix.identity(ti.f32, DIM)
            J[p] = 1.0
        for d in ti.static(range(DIM)):
            x[p][d] = ti.min(ti.max(x[p][d], 2.0 * DX), 1.0 - 2.0 * DX)
        if container_shape[None] == 1:        # round: keep inside the cylinder
            gx = x[p][0] - 0.5
            gz = x[p][2] - 0.5
            rr = ti.sqrt(gx * gx + gz * gz)
            if rr > CONTAINER_R:
                s = CONTAINER_R / rr
                x[p][0] = 0.5 + gx * s
                x[p][2] = 0.5 + gz * s

        # --- transformations: temperature-threshold "chemistry" ------------
        if ti.static(HEAT_ON):
            m = mat[p]
            if m == ICE and new_T > T_MELT_ICE:
                set_material(p, WATER)
            elif m == WATER and new_T > T_BOIL_WATER:
                set_material(p, STEAM)
            elif m == WATER and new_T < T_FREEZE_WATER:
                set_material(p, ICE)
            elif m == STEAM and new_T < T_CONDENSE_STEAM:
                set_material(p, WATER)
            elif m == LAVA and new_T < T_SOLIDIFY_LAVA:
                set_material(p, STONE)
            elif m == STONE and new_T > T_MELT_STONE:
                set_material(p, LAVA)


@ti.func
def _init_particle(p, m):
    v[p] = ti.Vector.zero(ti.f32, DIM)
    J[p] = 1.0
    F[p] = ti.Matrix.identity(ti.f32, DIM)
    C[p] = ti.Matrix.zero(ti.f32, DIM, DIM)
    mat[p] = m
    mass[p] = P_VOL * mat_rho[m]
    temp[p] = mat_temp0[m]
    colors[p] = mat_color[m]
    active[p] = 1


@ti.kernel
def park_pool():
    # Mark every particle inactive and park it off-domain. Scene seeding then
    # re-activates the ones it places; the rest stay as the spawn pool.
    for p in x:
        active[p] = 0
        x[p] = ti.Vector([-9.0, -9.0, -9.0])
        v[p] = ti.Vector.zero(ti.f32, DIM)
        C[p] = ti.Matrix.zero(ti.f32, DIM, DIM)


spawn_count = ti.field(ti.i32, ())


@ti.kernel
def spawn_brush(center: ti.types.vector(DIM, ti.f32), radius: ti.f32,
                m: ti.i32, budget: ti.i32, vel: ti.types.vector(DIM, ti.f32)):
    # "Add material": pull up to `budget` particles from the free pool and place
    # them, jittered, inside the brush sphere as material m, with initial
    # velocity `vel` (the cursor drag velocity -> you can throw/spray material).
    # Occupancy gate: skip cells that are already (nearly) full so material
    # can't be jammed into packed fluid -> no over-compression explosion. You
    # can still pour onto the surface (the empty cells above it accept it).
    occ = 0.6 * DX * DX * DX                  # ~full-cell mass for rho=1 fluid
    spawn_count[None] = 0
    for p in x:
        if active[p] == 0:
            r = ti.Vector([ti.random() * 2.0 - 1.0, ti.random() * 2.0 - 1.0,
                           ti.random() * 2.0 - 1.0]) * radius
            pos = center + r
            cell = ti.min(ti.max((pos * INV_DX).cast(int), 0), N_GRID - 1)
            if grid_m[cell] < occ:            # only place into free space
                idx = ti.atomic_add(spawn_count[None], 1)
                if idx < budget:
                    x[p] = pos
                    _init_particle(p, m)
                    v[p] = vel


@ti.kernel
def delete_brush(center: ti.types.vector(DIM, ti.f32), radius: ti.f32):
    # "Delete material": return active particles inside the sphere to the pool.
    for p in x:
        if active[p] == 1 and (x[p] - center).norm() < radius:
            active[p] = 0
            x[p] = ti.Vector([-9.0, -9.0, -9.0])
            v[p] = ti.Vector.zero(ti.f32, DIM)


@ti.kernel
def force_brush(center: ti.types.vector(DIM, ti.f32), radius: ti.f32,
                fvec: ti.types.vector(DIM, ti.f32)):
    # "Force/wind": add a velocity impulse to particles in the sphere, falling
    # off to zero at the edge. `fvec` is the cursor drag velocity -> swipe to
    # push fluid around, stir it, or knock piles over.
    for p in x:
        if active[p] == 1:
            d = (x[p] - center).norm()
            if d < radius:
                v[p] += fvec * (1.0 - d / radius)


@ti.kernel
def count_active() -> ti.i32:
    n = 0
    for p in x:
        if active[p] == 1:
            n += 1
    return n


@ti.func
def set_material(p, m):
    # Transform a particle into a new material in place: swap id/mass/colour and
    # reset its constitutive state to rest. Temperature is preserved (heat is
    # what drove the change). Used by the transformation rules.
    mat[p] = m
    mass[p] = P_VOL * mat_rho[m]
    colors[p] = mat_color[m]
    J[p] = 1.0
    F[p] = ti.Matrix.identity(ti.f32, DIM)


@ti.kernel
def seed_block(start: ti.i32, count: ti.i32, m: ti.i32,
               lo: ti.types.vector(DIM, ti.f32), hi: ti.types.vector(DIM, ti.f32)):
    for i in range(count):
        p = start + i
        for d in ti.static(range(DIM)):
            x[p][d] = lo[d] + ti.random() * (hi[d] - lo[d])
        _init_particle(p, m)


@ti.kernel
def seed_sphere(start: ti.i32, count: ti.i32, m: ti.i32,
                center: ti.types.vector(DIM, ti.f32), radius: ti.f32):
    for i in range(count):
        p = start + i
        # rejection-sample a uniform point inside the sphere
        pos = center
        done = False
        while not done:
            for d in ti.static(range(DIM)):
                pos[d] = center[d] + (ti.random() * 2.0 - 1.0) * radius
            if (pos - center).norm() <= radius:
                done = True
        x[p] = pos
        _init_particle(p, m)


# Boundary between the lower (light) and upper (heavy) layers, in domain units.
INTERFACE_Y = 0.46
N_LIGHT = N_SIM // 2
RENDER_MODE = "fluid"   # "fluid" | "surface" | "particles"; overridden by --render


@ti.kernel
def perturb_interface(amp: ti.f32, band: ti.f32):
    # Seed a single-mode Rayleigh-Taylor perturbation: push the heavy fluid
    # down in the centre and let light rise at the edges. A flat interface is
    # a (meta)stable equilibrium that random particle noise won't reliably tip,
    # so this coherent kick is what turns "stratified" into "overturning".
    for p in x:
        dy = x[p][1] - INTERFACE_Y
        if ti.abs(dy) < band:
            cx = ti.cos(3.1415926 * (x[p][0] - 0.5))
            cz = ti.cos(3.1415926 * (x[p][2] - 0.5))
            falloff = 1.0 - ti.abs(dy) / band
            v[p][1] -= amp * cx * cz * falloff


def init_scene():
    load_materials()
    floor_sticky[None] = 1 if SCENE == "sand" else 0
    container_shape[None] = 0          # box by default (GUI can switch to round)
    park_pool()   # all particles inactive+parked; seeding re-activates scene ones
    if SCENE == "empty":
        # Nothing seeded — a blank tank. Build it yourself with the add brush
        # (the whole particle budget sits in the free pool).
        pass
    elif SCENE == "pool":
        # A calm, settled tank of WATER at rest. The default sandbox: a clean
        # water surface to sculpt, heat, push, or drop things into. Seed it
        # FLUSH to the walls and shallow so the low-viscosity (FLIP) water has
        # nothing to collapse into -> it starts calm, not as a splash.
        seed_block(0, N_SIM, WATER,
                   ti.Vector([0.055, 0.05, 0.055]), ti.Vector([0.945, 0.35, 0.945]))
    elif SCENE == "jelly":
        # A cube of elastic jelly dropped onto the floor. It should squish on
        # impact, then wobble and bounce while holding together as a blob —
        # NOT spread out flat like a liquid.
        seed_block(0, N_SIM, JELLY,
                   ti.Vector([0.35, 0.45, 0.35]),
                   ti.Vector([0.65, 0.75, 0.65]))
    elif SCENE == "sand":
        # A tall, narrow column of sand. It should collapse and slump outward
        # into a low, wide pile (angle of repose) — wider and shorter than it
        # started, but NOT flowing flat like a liquid nor holding rigid.
        seed_block(0, N_SIM, SAND,
                   ti.Vector([0.38, 0.05, 0.38]),
                   ti.Vector([0.62, 0.55, 0.62]))
    elif SCENE == "lava":
        # A ball of molten LAVA (1200 deg) dropped into a cold WATER pool. With
        # no scripted contact rules, heat diffusion alone drives the marquee
        # reaction: the lava cools below its freezing point and turns to STONE,
        # while the water it touches boils past 100 deg and turns to STEAM that
        # rises. (pillar 3: emergence from physics, not rules.)
        import math
        # The ball starts resting in the water surface (not high above it), so
        # it transfers heat while still molten — otherwise radiative cooling
        # spends most of its heat during the fall before it ever touches water.
        ball_c, ball_r = (0.5, 0.50, 0.5), 0.15
        pool_lo, pool_hi = (0.10, 0.05, 0.10), (0.90, 0.42, 0.90)
        pool_vol = (pool_hi[0] - pool_lo[0]) * (pool_hi[1] - pool_lo[1]) * (pool_hi[2] - pool_lo[2])
        ball_vol = 4.0 / 3.0 * math.pi * ball_r ** 3
        n_ball = int(N_SIM * ball_vol / (pool_vol + ball_vol))
        seed_block(0, N_SIM - n_ball, WATER, ti.Vector(pool_lo), ti.Vector(pool_hi))
        seed_sphere(N_SIM - n_ball, n_ball, LAVA, ti.Vector(ball_c), ball_r)
    elif SCENE == "melt":
        # A cold ICE block dropped onto a warm WATER pool: the ice warms past
        # its melting point by diffusion and turns to liquid WATER.
        seed_block(0, N_SIM * 3 // 4, WATER,
                   ti.Vector([0.10, 0.05, 0.10]), ti.Vector([0.90, 0.40, 0.90]))
        seed_block(N_SIM * 3 // 4, N_SIM - N_SIM * 3 // 4, ICE,
                   ti.Vector([0.38, 0.45, 0.38]), ti.Vector([0.62, 0.69, 0.62]))
    elif SCENE == "drop":
        # A pool of light fluid at rest with a compact ball of heavy fluid
        # released above it. Grid-resolved and dynamic (no metastable trap), so
        # this is the clean test of buoyancy: the heavy ball should plunge in
        # and sink to the floor while the light fluid is displaced upward.
        import math
        ball_c = (0.5, 0.72, 0.5)
        ball_r = 0.14
        pool_lo, pool_hi = (0.10, 0.05, 0.10), (0.90, 0.45, 0.90)
        pool_vol = (pool_hi[0] - pool_lo[0]) * (pool_hi[1] - pool_lo[1]) * (pool_hi[2] - pool_lo[2])
        ball_vol = 4.0 / 3.0 * math.pi * ball_r ** 3
        n_ball = int(N_SIM * ball_vol / (pool_vol + ball_vol))
        n_pool = N_SIM - n_ball
        seed_block(0, n_pool, LIGHT, ti.Vector(pool_lo), ti.Vector(pool_hi))
        seed_sphere(n_pool, n_ball, HEAVY, ti.Vector(ball_c), ball_r)
    elif SCENE == "layers":
        # Heavy sits on top of light. NOTE: a flat interface is *metastable* in
        # single-grid pure-APIC MPM — the shared grid velocity at interface
        # nodes locks the layers and APIC's numerical viscosity damps the
        # Rayleigh-Taylor instability, so this does NOT spontaneously overturn
        # even with a seeded perturbation. Kept as a documented limitation; a
        # FLIP/PIC velocity blend (lower viscosity) is the path to real plumes.
        # Use "drop" for a robust density-settling demo.
        seed_block(0, N_LIGHT, LIGHT,
                   ti.Vector([0.15, 0.08, 0.15]),
                   ti.Vector([0.85, INTERFACE_Y, 0.85]))
        seed_block(N_LIGHT, N_SIM - N_LIGHT, HEAVY,
                   ti.Vector([0.15, INTERFACE_Y, 0.15]),
                   ti.Vector([0.85, 0.84, 0.85]))
        perturb_interface(amp=1.2, band=0.10)
    else:
        # Both fluids intermixed in one region. NOTE: in single-grid MPM,
        # co-located materials share one grid velocity and cannot demix at the
        # sub-cell scale — this scene stays mixed by design, and is kept only to
        # demonstrate that limitation. Use "layers" for density separation.
        region_lo = ti.Vector([0.15, 0.10, 0.15])
        region_hi = ti.Vector([0.85, 0.80, 0.85])
        seed_block(0, N_LIGHT, LIGHT, region_lo, region_hi)
        seed_block(N_LIGHT, N_SIM - N_LIGHT, HEAVY, region_lo, region_hi)

    # Pre-settle the pool with HEAVY damping (low FLIP) so it opens dead-flat
    # instead of ringing from the seed's release transient; then raise FLIP for
    # slippery interaction.
    if SCENE == "pool":
        flip_ratio[None] = 0.3
        for _ in range(90):
            for _ in range(SUBSTEPS):
                substep()
    flip_ratio[None] = FLIP_LIQUID


# ----------------------------------------------------------------------------
# Diagnostics for the headless stability gate
# ----------------------------------------------------------------------------
max_speed = ti.field(ti.f32, ())
nan_count = ti.field(ti.i32, ())
sum_y = ti.field(ti.f32, NUM_MAT)   # sum of heights per material
count_m = ti.field(ti.i32, NUM_MAT)  # particle count per material
min_y = ti.field(ti.f32, ())        # lowest / highest particle (vertical extent)
max_y = ti.field(ti.f32, ())
min_x = ti.field(ti.f32, ())        # horizontal extent (for slump/spread tests)
max_x = ti.field(ti.f32, ())
min_T = ti.field(ti.f32, ())        # temperature range (heat scenes)
max_T = ti.field(ti.f32, ())


@ti.kernel
def diagnostics():
    max_speed[None] = 0.0
    nan_count[None] = 0
    min_y[None] = 1.0
    max_y[None] = 0.0
    min_x[None] = 1.0
    max_x[None] = 0.0
    min_T[None] = 1.0e9
    max_T[None] = -1.0e9
    for m in range(NUM_MAT):
        sum_y[m] = 0.0
        count_m[m] = 0
    for p in x:
        if active[p] == 0:
            continue
        s = v[p].norm()
        ti.atomic_max(max_speed[None], s)
        ti.atomic_min(min_y[None], x[p][1])
        ti.atomic_max(max_y[None], x[p][1])
        ti.atomic_min(min_x[None], x[p][0])
        ti.atomic_max(max_x[None], x[p][0])
        bad = 0
        for d in ti.static(range(DIM)):
            if ti.math.isnan(x[p][d]) or ti.math.isnan(v[p][d]):
                bad = 1
        nan_count[None] += bad
        sum_y[mat[p]] += x[p][1]
        count_m[mat[p]] += 1
        ti.atomic_min(min_T[None], temp[p])
        ti.atomic_max(max_T[None], temp[p])


def mean_heights():
    sy = sum_y.to_numpy()
    cm = count_m.to_numpy()
    return [float(sy[m] / cm[m]) if cm[m] > 0 else float("nan")
            for m in range(NUM_MAT)]


MAT_NAMES = ["light", "heavy", "jelly", "sand", "water",
             "ice", "steam", "lava", "stone", "water_c", "water_p"]
assert len(MAT_NAMES) == NUM_MAT


def counts():
    return count_m.to_numpy()


def run_headless(frames: int) -> int:
    init_scene()
    print(f"Headless run: scene={SCENE}, {N_PARTICLES} particles, {N_GRID}^3 grid, "
          f"{frames} frames x {SUBSTEPS} substeps (dt={DT})")
    light_y = heavy_y = extent = float("nan")
    peak_steam = 0
    sample_every = 20 if SCENE in ("lava", "melt") else 50
    for f in range(frames):
        for _ in range(SUBSTEPS):
            substep()
        if f % sample_every == 0 or f == frames - 1:
            diagnostics()
            ms = max_speed[None]
            nc = nan_count[None]
            mh = mean_heights()
            light_y, heavy_y = mh[LIGHT], mh[HEAVY]
            extent = max_y[None] - min_y[None]
            h_extent = max_x[None] - min_x[None]
            if SCENE == "jelly":
                print(f"  frame {f:4d}  max|v|={ms:7.3f}  "
                      f"y-extent={extent:.4f}  jelly_mean_y={mh[JELLY]:.4f}  NaNs={nc}")
            elif SCENE == "sand":
                print(f"  frame {f:4d}  max|v|={ms:7.3f}  "
                      f"h-extent={h_extent:.4f}  y-extent={extent:.4f}  NaNs={nc}")
            elif SCENE in ("pool", "empty"):
                print(f"  frame {f:4d}  max|v|={ms:7.3f}  "
                      f"water_mean_y={mh[WATER]:.4f}  NaNs={nc}")
            elif SCENE in ("lava", "melt"):
                cm = counts()
                peak_steam = max(peak_steam, int(cm[STEAM]))
                pops = "  ".join(f"{MAT_NAMES[m]}={int(cm[m])}"
                                 for m in (WATER, ICE, STEAM, LAVA, STONE) if cm[m] > 0)
                print(f"  frame {f:4d}  max|v|={ms:6.2f}  "
                      f"T=[{min_T[None]:6.1f},{max_T[None]:7.1f}]  {pops}  NaNs={nc}")
            else:
                print(f"  frame {f:4d}  max|v|={ms:7.3f}  "
                      f"mean_y light={light_y:.4f} heavy={heavy_y:.4f}  NaNs={nc}")
            if nc > 0:
                print("  !! NaNs detected — simulation blew up")
                return 1
            if ms > 1e3:
                print("  !! velocity exploded (>1e3) — unstable")
                return 1

    # Per-scene success criteria.
    cm = counts()
    if SCENE == "lava":
        # Heat diffusion alone (no contact rules) drives the reaction: lava ->
        # stone (cooled), and water it touches flash-boils to steam. The steam
        # is transient — it condenses back as the system cools — so we grade on
        # the PEAK steam produced, plus stone remaining at the end.
        if cm[STONE] > 0 and peak_steam > 0:
            print(f"PASS: heat-driven reaction — lava solidified to STONE "
                  f"({int(cm[STONE])}) and water flash-boiled to STEAM "
                  f"(peak {peak_steam}, now {int(cm[STEAM])} as it re-condenses).")
            return 0
        print(f"FAIL: expected stone + (peak) steam from lava+water "
              f"(stone={int(cm[STONE])}, peak_steam={peak_steam}).")
        return 1
    if SCENE == "melt":
        # The ice block must (largely) melt into liquid water.
        n_ice0 = N_SIM - N_SIM * 3 // 4
        if cm[ICE] < n_ice0 * 0.5:
            print(f"PASS: ice melted to water "
                  f"(ice {n_ice0} -> {int(cm[ICE])}, water now {int(cm[WATER])}).")
            return 0
        print(f"FAIL: ice did not melt (ice {n_ice0} -> {int(cm[ICE])}).")
        return 1
    if SCENE == "sand":
        # A granular column slumps: it must spread horizontally well beyond its
        # initial 0.24 width AND lose height from its initial 0.50 — yet keep a
        # real pile (not flow flat to the walls like a liquid).
        if 0.30 < h_extent < 0.70 and extent < 0.40 and ms < 0.15:
            print(f"PASS: stable, sand slumped into a settling pile "
                  f"(h-extent={h_extent:.4f} from 0.24, y-extent={extent:.4f} "
                  f"from 0.50, max|v|={ms:.3f}).")
            return 0
        print(f"FAIL: sand did not form a settling pile "
              f"(h-extent={h_extent:.4f}, y-extent={extent:.4f}, max|v|={ms:.3f}).")
        return 1
    if SCENE == "jelly":
        # An elastic blob holds its shape; a liquid of the same volume would
        # flatten into a thin puddle. Require the jelly to keep real vertical
        # extent (it has settled but not spread flat).
        if extent > 0.12:
            print(f"PASS: stable, and the jelly held together as a blob "
                  f"(y-extent={extent:.4f} > 0.12 — did not flatten).")
            return 0
        print(f"FAIL: jelly flattened like a liquid (y-extent={extent:.4f}).")
        return 1
    if SCENE in ("layers", "drop"):
        # The heavy fluid must end up clearly below the light fluid (real gap,
        # not numerical noise).
        gap = light_y - heavy_y
        if gap > 0.02:
            print(f"PASS: stable, and the heavy fluid settled below the light "
                  f"(light_y={light_y:.4f} - heavy_y={heavy_y:.4f} = {gap:+.4f}).")
            return 0
        print(f"FAIL: density layering did not occur "
              f"(light_y={light_y:.4f}, heavy_y={heavy_y:.4f}, gap={gap:+.4f}).")
        return 1
    if SCENE in ("pool", "empty"):
        print(f"PASS: stable ({SCENE} scene), max|v|={ms:.3f}.")
        return 0
    # mixed scene — graded on stability only (cannot demix; see init_scene).
    print(f"PASS (mixed scene): stable; fluids remain mixed as expected "
          f"(light_y={light_y:.4f}, heavy_y={heavy_y:.4f}).")
    return 0


@ti.func
def _glow(base, T):
    # Temperature -> emission tint so hot material visibly glows white-hot.
    g = ti.max(0.0, ti.min(1.0, (T - 300.0) / 900.0))
    return base * (1.0 - g) + ti.Vector([1.0, 0.95, 0.7]) * g


@ti.kernel
def update_render_colors():
    # Refresh per-particle render colour from material + temperature. Needed
    # every frame in heat scenes (glow tracks temperature) and after a load.
    for p in x:
        colors[p] = _glow(mat_color[mat[p]], temp[p])


# ============================================================================
# Phase 5 — surface rendering
# ----------------------------------------------------------------------------
# A coarse particle sim can render as a smooth surface: splat particles to a
# density field on a regular lattice, then extract an isosurface mesh. This
# "hides the grid" (design pillar 5) and lifts the look past the particle-blob
# stage. We use *marching tetrahedra* (each cube split into 6 tets) rather than
# marching cubes: its triangulation follows from first principles (no 256-entry
# magic table), and with gradient normals + two-sided lighting the result is
# robust to triangle-winding mistakes.
# ============================================================================
# Render lattice is DECOUPLED from the sim grid (design pillar 5): a finer
# render grid gives smooth surfaces / ripples without terracing, while the sim
# stays coarse. Default 96 (vs 64 sim); --render-grid trades quality for speed.
NR = int(_early_flag("--render-grid", 96))
rho_r = ti.field(ti.f32, (NR, NR, NR))       # splatted number density
col_r = ti.Vector.field(3, ti.f32, (NR, NR, NR))  # density-weighted colour
tmp_r = ti.field(ti.f32, (NR, NR, NR))       # density-weighted temperature
grad_r = ti.Vector.field(3, ti.f32, (NR, NR, NR))  # density gradient (for normals)
rho_s = ti.field(ti.f32, (NR, NR, NR))       # smoothing scratch buffer

# GGUI re-uploads the ENTIRE vertex buffer every frame regardless of
# vertex_count, so this must be sized as tight as the mesh actually needs — an
# oversized buffer (e.g. 6M) silently costs >100 ms/frame of pointless upload.
# Surface triangle count scales with the lattice's cross-sectional area (~NR^2).
MAX_VERTS = NR * NR * 110
surf_pos = ti.Vector.field(3, ti.f32, MAX_VERTS)
surf_nrm = ti.Vector.field(3, ti.f32, MAX_VERTS)
surf_col = ti.Vector.field(3, ti.f32, MAX_VERTS)
surf_count = ti.field(ti.i32, ())
# The isosurface level is set per frame as a FRACTION of the peak splatted
# density (rho_max), so it is independent of render resolution and particle
# count — no manual retuning when those change.
RHO_ISO_FRAC = 0.40
iso_level = ti.field(ti.f32, ())
rho_max = ti.field(ti.f32, ())

# --- Phase 6 interaction state ----------------------------------------------
# Clipping/cross-section plane: hide everything on one side of an axis-aligned
# plane so you can see inside the volume (the core "3D placement is awkward" fix).
clip_on = ti.field(ti.i32, ())     # 0/1
clip_axis = ti.field(ti.i32, ())   # 0=x 1=y 2=z
clip_pos = ti.field(ti.f32, ())    # plane position in [0,1]
rpos = ti.Vector.field(3, ti.f32, N_PARTICLES)  # particle render positions (clipped)
cam_pos = ti.Vector.field(3, ti.f32, ())  # camera position, for fresnel sheen


@ti.func
def _clipped(p_world):
    # True if the point is on the hidden side of the clip plane.
    return clip_on[None] == 1 and p_world[clip_axis[None]] > clip_pos[None]


@ti.kernel
def update_particle_render():
    # Per-particle render colour, plus push clipped particles off-screen so the
    # cross-section reads (GGUI draws the whole field, so we relocate, not cull).
    for p in x:
        colors[p] = _glow(mat_color[mat[p]], temp[p])
        hidden = active[p] == 0 or _clipped(x[p])
        rpos[p] = ti.Vector([-9.0, -9.0, -9.0]) if hidden else x[p]


@ti.kernel
def heat_brush(center: ti.types.vector(DIM, ti.f32), radius: ti.f32, delta: ti.f32):
    # Interactive heat tool: add (or remove) heat to particles within a sphere,
    # so you can melt ice / boil water / freeze things by hand.
    for p in x:
        if active[p] == 1 and (x[p] - center).norm() < radius:
            temp[p] += delta

# Cube corner offsets (bit = x + 2y + 4z) and a 6-tetrahedron decomposition that
# shares the main diagonal 0--7, tiling the cube exactly.
_CORNER = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
           (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
_TETS = [(0, 7, 3, 1), (0, 7, 1, 5), (0, 7, 5, 4),
         (0, 7, 4, 6), (0, 7, 6, 2), (0, 7, 2, 3)]


@ti.kernel
def splat_render():
    for I in ti.grouped(rho_r):
        rho_r[I] = 0.0
        col_r[I] = ti.Vector.zero(ti.f32, 3)
        tmp_r[I] = 0.0
    for p in x:
        if active[p] == 0:
            continue
        fx = x[p] * NR
        base = ti.floor(fx, ti.i32)
        fr = fx - base.cast(ti.f32)
        for off in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
            node = base + off
            inb = True
            for d in ti.static(range(DIM)):
                if node[d] < 0 or node[d] >= NR:
                    inb = False
            if inb:
                w = 1.0
                for d in ti.static(range(DIM)):
                    w *= fr[d] if off[d] == 1 else 1.0 - fr[d]
                rho_r[node] += w
                col_r[node] += w * mat_color[mat[p]]
                tmp_r[node] += w * temp[p]
    for I in ti.grouped(rho_r):
        if rho_r[I] > 0.0:
            col_r[I] /= rho_r[I]
            tmp_r[I] /= rho_r[I]


@ti.kernel
def smooth_density():
    # One box-blur pass on the density field only (not colour): rounds off the
    # grid-scale bumpiness of the isosurface for a smoother fluid look.
    for I in ti.grouped(rho_r):
        s = rho_r[I]
        cnt = 1.0
        for d in ti.static(range(DIM)):
            for off in ti.static((-1, 1)):
                Jn = I + ti.Vector.unit(DIM, d) * off
                if Jn[d] >= 0 and Jn[d] < NR:
                    s += rho_r[Jn]
                    cnt += 1.0
        rho_s[I] = s / cnt
    for I in ti.grouped(rho_r):
        rho_r[I] = rho_s[I]


@ti.kernel
def compute_gradient():
    for I in ti.grouped(rho_r):
        g = ti.Vector.zero(ti.f32, 3)
        for d in ti.static(range(DIM)):
            a = I + ti.Vector.unit(DIM, d)
            b = I - ti.Vector.unit(DIM, d)
            hi = rho_r[a] if a[d] < NR else rho_r[I]
            lo = rho_r[b] if b[d] >= 0 else rho_r[I]
            g[d] = 0.5 * (hi - lo)
        grad_r[I] = g


@ti.func
def _wv(slot, R):
    # Write one vertex (packed as rows pos / normal / colour of a 3x3 matrix).
    if slot < MAX_VERTS:
        surf_pos[slot] = ti.Vector([R[0, 0], R[0, 1], R[0, 2]])
        surf_nrm[slot] = ti.Vector([R[1, 0], R[1, 1], R[1, 2]])
        surf_col[slot] = ti.Vector([R[2, 0], R[2, 1], R[2, 2]])


@ti.kernel
def marching_tets():
    surf_count[None] = 0
    for I in ti.grouped(ti.ndrange(NR - 1, NR - 1, NR - 1)):
        # cross-section: skip cells on the hidden side of the clip plane
        cell_world = (I.cast(ti.f32) + 0.5) / NR
        if not _clipped(cell_world):
            _march_cell(I)


@ti.func
def _march_cell(I):
        # gather the 8 cube corners
        cv = ti.Vector.zero(ti.f32, 8)
        cpos = ti.Matrix.zero(ti.f32, 8, 3)
        ccol = ti.Matrix.zero(ti.f32, 8, 3)
        cnrm = ti.Matrix.zero(ti.f32, 8, 3)
        for c in ti.static(range(8)):
            o = ti.Vector(_CORNER[c])
            node = I + o
            cv[c] = rho_r[node]
            colv = _glow(col_r[node], tmp_r[node])
            for d in ti.static(range(3)):
                cpos[c, d] = float(node[d])
                ccol[c, d] = colv[d]
                cnrm[c, d] = grad_r[node][d]
        for t in ti.static(range(6)):
            tet = _TETS[t]
            _emit_tet(tet[0], tet[1], tet[2], tet[3], cv, cpos, ccol, cnrm)


@ti.func
def _corner(idx, cpos):
    return ti.Vector([cpos[idx, 0], cpos[idx, 1], cpos[idx, 2]])


@ti.func
def _ccol(idx, ccol):
    return ti.Vector([ccol[idx, 0], ccol[idx, 1], ccol[idx, 2]])


@ti.func
def _cnrm(idx, cnrm):
    return ti.Vector([cnrm[idx, 0], cnrm[idx, 1], cnrm[idx, 2]])


@ti.func
def edge_vertex(m, n, cv, cpos, ccol, cnrm):
    # Isosurface crossing on edge m--n, packed as a 3x3 matrix (rows: world
    # position / outward normal / colour). Order of m,n does not affect the point.
    va, vb = cv[m], cv[n]
    t = (iso_level[None] - va) / (vb - va)
    pos = (_corner(m, cpos) + t * (_corner(n, cpos) - _corner(m, cpos))) / NR
    nrm = _cnrm(m, cnrm) + t * (_cnrm(n, cnrm) - _cnrm(m, cnrm))
    ln = nrm.norm()
    nrm = -nrm / ln if ln > 1e-6 else ti.Vector([0.0, 1.0, 0.0])  # outward = -grad
    col = _ccol(m, ccol) + t * (_ccol(n, ccol) - _ccol(m, ccol))
    # Fresnel sheen: surfaces seen at a grazing angle brighten toward a pale
    # sky colour, the way real water reflects the sky at its edges. Cheap, but
    # it reads as wet/translucent far more than flat Lambert shading.
    view = cam_pos[None] - pos
    vl = view.norm()
    if vl > 1e-6:
        ndotv = ti.abs(nrm.dot(view / vl))
        fres = (1.0 - ndotv) ** 4
        col = col * (1.0 - 0.7 * fres) + ti.Vector([0.75, 0.85, 1.0]) * (0.7 * fres)
    R = ti.Matrix.zero(ti.f32, 3, 3)
    for d in ti.static(range(3)):
        R[0, d] = pos[d]
        R[1, d] = nrm[d]
        R[2, d] = col[d]
    return R


@ti.func
def _emit_tet(a, b, c, d, cv, cpos, ccol, cnrm):
    # Classify the 4 tet corners; emit triangle(s) on the inside/outside
    # boundary. CRITICAL: the mesh is non-indexed, so each triangle's 3 vertices
    # must be CONSECUTIVE in the buffer. We reserve a whole triangle's slots with
    # one atomic_add and fill them in order — otherwise parallel tets interleave
    # vertices and GGUI stitches triangles across the whole volume.
    idx = ti.Vector([a, b, c, d])
    ins = ti.Vector([0, 0, 0, 0])
    n_in = 0
    for k in ti.static(range(4)):
        if cv[idx[k]] >= iso_level[None]:
            ins[k] = 1
            n_in += 1
    if n_in == 1 or n_in == 3:
        # one corner on its own side -> a single triangle of its three edges
        lone = 0
        want = 1 if n_in == 1 else 0
        for k in ti.static(range(4)):
            if ins[k] == want:
                lone = k
        base = ti.atomic_add(surf_count[None], 3)
        j = 0
        for k in ti.static(range(4)):
            if k != lone:
                _wv(base + j, edge_vertex(idx[lone], idx[k], cv, cpos, ccol, cnrm))
                j += 1
    elif n_in == 2:
        # two-vs-two -> a quad (two triangles) across the four crossing edges
        i0, i1, o0, o1 = -1, -1, -1, -1
        for k in ti.static(range(4)):
            if ins[k] == 1:
                if i0 < 0:
                    i0 = idx[k]
                else:
                    i1 = idx[k]
            else:
                if o0 < 0:
                    o0 = idx[k]
                else:
                    o1 = idx[k]
        # quad P(i0,o0) P(i0,o1) P(i1,o1) P(i1,o0) -> tris (0,1,2) and (0,2,3)
        A = edge_vertex(i0, o0, cv, cpos, ccol, cnrm)
        B = edge_vertex(i0, o1, cv, cpos, ccol, cnrm)
        Cc = edge_vertex(i1, o1, cv, cpos, ccol, cnrm)
        D = edge_vertex(i1, o0, cv, cpos, ccol, cnrm)
        base = ti.atomic_add(surf_count[None], 6)
        _wv(base + 0, A)
        _wv(base + 1, B)
        _wv(base + 2, Cc)
        _wv(base + 3, A)
        _wv(base + 4, Cc)
        _wv(base + 5, D)


@ti.kernel
def compute_rho_max():
    rho_max[None] = 0.0
    for I in ti.grouped(rho_r):
        ti.atomic_max(rho_max[None], rho_r[I])


# ============================================================================
# Screen-space fluid rendering (the technique dli/fluid, Splash, etc. use)
# ----------------------------------------------------------------------------
# Instead of building a mesh, splat each particle as a sphere sprite into a
# screen depth buffer, smooth the depth in 2D, reconstruct normals from the
# depth gradient, then shade with diffuse + specular + fresnel + thickness-based
# transparency. Cost is O(screen pixels), not O(triangles), and you get
# refraction/translucency for free. Output is written to an image and blitted.
# ============================================================================
SS_W, SS_H = 1280, 800
SS_FOV = 45.0                                 # vertical field of view (deg)
SS_NEAR = 0.02
SS_RADIUS = 0.021                             # world-space sprite radius (larger
#                                               -> sprites overlap -> smoother,
#                                               more liquid-looking surface)

ss_depth = ti.field(ti.f32, (SS_W, SS_H))     # nearest surface depth per pixel
ss_depth2 = ti.field(ti.f32, (SS_W, SS_H))    # blur scratch
ss_w = ti.field(ti.f32, (SS_W, SS_H))         # accumulated coverage (~thickness)
ss_colw = ti.Vector.field(3, ti.f32, (SS_W, SS_H))  # coverage-weighted colour
ss_img = ti.Vector.field(3, ti.f32, (SS_W, SS_H))   # final shaded image
blur_n = ti.field(ti.i32, ())   # bilateral blur passes (render-quality knob)
blur_n[None] = 4
view_m = ti.Matrix.field(4, 4, ti.f32, ())    # world -> view (camera) matrix
light_v = ti.Vector.field(3, ti.f32, ())      # light direction in view space
cam_s = ti.Vector.field(3, ti.f32, ())        # camera basis in WORLD space:
cam_u = ti.Vector.field(3, ti.f32, ())        #   right / up / forward. Used to
cam_f = ti.Vector.field(3, ti.f32, ())        #   turn view dirs back into world
sun_w = ti.Vector.field(3, ti.f32, ())        # sun direction, WORLD space (sky)

# World sun direction shared by the surface light and the sky (so the specular
# highlight on the fluid and the sky's sun disc line up).
_SUN = np.array([0.4, 0.9, 0.5], np.float32)
_SUN /= np.linalg.norm(_SUN)


def _set_view(eye, center, up):
    eye = np.asarray(eye, np.float32)
    f = np.asarray(center, np.float32) - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, np.asarray(up, np.float32))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.array([
        [s[0], s[1], s[2], -s.dot(eye)],
        [u[0], u[1], u[2], -u.dot(eye)],
        [-f[0], -f[1], -f[2], f.dot(eye)],
        [0, 0, 0, 1]], np.float32)
    view_m.from_numpy(m)
    cam_s[None] = ti.Vector([float(s[0]), float(s[1]), float(s[2])])
    cam_u[None] = ti.Vector([float(u[0]), float(u[1]), float(u[2])])
    cam_f[None] = ti.Vector([float(f[0]), float(f[1]), float(f[2])])
    # the sun, expressed in view space (rotation only) for surface lighting
    lv = m[:3, :3] @ _SUN
    light_v[None] = ti.Vector([float(lv[0]), float(lv[1]), float(lv[2])])
    sun_w[None] = ti.Vector([float(_SUN[0]), float(_SUN[1]), float(_SUN[2])])


@ti.func
def _view_pos(i, j, depth, t, aspect):
    ndcx = (i + 0.5) / SS_W * 2.0 - 1.0
    ndcy = (j + 0.5) / SS_H * 2.0 - 1.0
    return ti.Vector([ndcx * t * aspect * depth, ndcy * t * depth, -depth])


@ti.kernel
def ss_clear():
    for i, j in ss_depth:
        ss_depth[i, j] = 1e9
        ss_w[i, j] = 0.0
        ss_colw[i, j] = ti.Vector.zero(ti.f32, 3)


@ti.kernel
def ss_splat(t: ti.f32, aspect: ti.f32):
    vm = view_m[None]
    for p in x:
        if active[p] == 1 and not _clipped(x[p]):
            wp = x[p]
            vx = vm[0, 0] * wp[0] + vm[0, 1] * wp[1] + vm[0, 2] * wp[2] + vm[0, 3]
            vy = vm[1, 0] * wp[0] + vm[1, 1] * wp[1] + vm[1, 2] * wp[2] + vm[1, 3]
            vz = vm[2, 0] * wp[0] + vm[2, 1] * wp[1] + vm[2, 2] * wp[2] + vm[2, 3]
            depth = -vz
            if depth > SS_NEAR:
                sx = ((vx / (depth * t * aspect)) * 0.5 + 0.5) * SS_W
                sy = ((vy / (depth * t)) * 0.5 + 0.5) * SS_H
                rpx = ti.min(ti.max(SS_RADIUS * SS_H / (2.0 * t * depth), 1.0), 24.0)
                col = colors[p]
                i0 = ti.max(int(sx - rpx), 0)
                i1 = ti.min(int(sx + rpx) + 1, SS_W)
                j0 = ti.max(int(sy - rpx), 0)
                j1 = ti.min(int(sy + rpx) + 1, SS_H)
                for i in range(i0, i1):
                    for j in range(j0, j1):
                        dx = i + 0.5 - sx
                        dy = j + 0.5 - sy
                        dd = (dx * dx + dy * dy) / (rpx * rpx)
                        if dd < 1.0:
                            bulge = ti.sqrt(1.0 - dd)
                            ti.atomic_min(ss_depth[i, j], depth - SS_RADIUS * bulge)
                            ss_w[i, j] += bulge
                            ss_colw[i, j] += col * bulge


@ti.kernel
def ss_blur():
    # Bilateral, Gaussian-weighted depth blur: averages only neighbours within a
    # depth-relative tolerance so a surface smooths out (kills the splat-sprite
    # "cottage cheese") while silhouettes between separate blobs stay crisp.
    for i, j in ss_depth:
        d = ss_depth[i, j]
        if d < 1e8:
            tol = ti.max(0.02, d * 0.06)      # widen tolerance with distance
            s = 0.0
            wsum = 0.0
            for di in range(-3, 4):
                for dj in range(-3, 4):
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < SS_W and 0 <= jj < SS_H:
                        dn = ss_depth[ii, jj]
                        if dn < 1e8 and ti.abs(dn - d) < tol:
                            w = ti.exp(-(di * di + dj * dj) * 0.18)
                            s += dn * w
                            wsum += w
            ss_depth2[i, j] = s / ti.max(wsum, 1e-6)
        else:
            ss_depth2[i, j] = d
    for i, j in ss_depth:
        ss_depth[i, j] = ss_depth2[i, j]


@ti.func
def _view_to_world_dir(d):
    # camera looks down -z, so view +z maps to world -forward.
    return d[0] * cam_s[None] + d[1] * cam_u[None] - d[2] * cam_f[None]


@ti.func
def sky_color(dir):
    # A cheap procedural environment used for both the background and for the
    # fluid's reflections: a zenith->horizon sky gradient, a warm ground
    # hemisphere below, and a sun disc with a soft halo.
    up = ti.max(dir[1], -1.0)
    zenith = ti.Vector([0.16, 0.33, 0.62])
    horizon = ti.Vector([0.74, 0.84, 0.96])
    ground = ti.Vector([0.15, 0.14, 0.17])
    col = horizon
    if up >= 0.0:
        k = ti.pow(1.0 - up, 2.5)            # thin bright band at the horizon
        col = zenith * (1.0 - k) + horizon * k
    else:
        g = ti.min(-up * 3.0, 1.0)
        col = horizon * (1.0 - g) + ground * g
    sd = ti.max(dir.dot(sun_w[None]), 0.0)
    col += ti.Vector([1.0, 0.97, 0.90]) * (ti.pow(sd, 1800.0) * 9.0)   # disc
    col += ti.Vector([1.0, 0.90, 0.76]) * (ti.pow(sd, 12.0) * 0.22)    # halo
    return col


@ti.kernel
def ss_shade(t: ti.f32, aspect: ti.f32):
    L = light_v[None]
    for i, j in ss_depth:
        d = ss_depth[i, j]
        # world-space ray through this pixel -> the sky behind/through the fluid
        rw = _view_to_world_dir(_view_pos(i, j, 1.0, t, aspect)).normalized()
        skyc = sky_color(rw)
        if d > 1e8:
            ss_img[i, j] = ti.min(skyc, ti.Vector([1.0, 1.0, 1.0]))
        else:
            P = _view_pos(i, j, d, t, aspect)
            # reconstruct normal from depth gradient (use the closer neighbour
            # on each axis to avoid smearing across silhouettes)
            dxp = ss_depth[ti.min(i + 1, SS_W - 1), j]
            dxm = ss_depth[ti.max(i - 1, 0), j]
            dyp = ss_depth[i, ti.min(j + 1, SS_H - 1)]
            dym = ss_depth[i, ti.max(j - 1, 0)]
            ddx = _view_pos(i + 1, j, dxp, t, aspect) - P
            if ti.abs(dxm - d) < ti.abs(dxp - d):
                ddx = P - _view_pos(i - 1, j, dxm, t, aspect)
            ddy = _view_pos(i, j + 1, dyp, t, aspect) - P
            if ti.abs(dym - d) < ti.abs(dyp - d):
                ddy = P - _view_pos(i, j - 1, dym, t, aspect)
            N = ddx.cross(ddy)
            nl = N.norm()
            N = N / nl if nl > 1e-9 else ti.Vector([0.0, 0.0, 1.0])
            if N[2] < 0.0:
                N = -N
            V = -P.normalized()
            diff = ti.max(N.dot(L), 0.0)
            H = (L + V).normalized()
            spec = ti.pow(ti.max(N.dot(H), 0.0), 60.0)   # broader, softer sheen
            # Schlick fresnel (water f0 ~ 0.04) -> strong grazing reflection
            fres = 0.04 + 0.96 * ti.pow(1.0 - ti.max(N.dot(V), 0.0), 5.0)
            base = ss_colw[i, j] / ti.max(ss_w[i, j], 1e-4)
            thick = ss_w[i, j]
            Nw = _view_to_world_dir(N).normalized()
            # --- translucent water: see the (refracted) background through it,
            # deepening to the water tint with thickness (Beer-Lambert-ish) ----
            refr_dir = (rw + Nw * 0.18).normalized()     # background, bent by N
            bg_refr = sky_color(refr_dir)
            depth_t = 1.0 - ti.exp(-thick * 0.16)        # 0 thin/clear .. 1 deep
            transmitted = bg_refr * (1.0 - depth_t) + base * depth_t
            transmitted *= (0.75 + 0.25 * diff)          # gentle body shading
            # --- specular reflection of the environment, by Schlick fresnel ---
            I = -V
            Rw = _view_to_world_dir(I - 2.0 * I.dot(N) * N).normalized()
            refl = sky_color(Rw)
            out = transmitted * (1.0 - fres) + refl * fres
            out += ti.Vector([1.0, 1.0, 1.0]) * (spec * 0.7)
            ss_img[i, j] = ti.min(out, ti.Vector([1.0, 1.0, 1.0]))


@ti.kernel
def draw_ring(sx: ti.f32, sy: ti.f32, rpix: ti.f32,
              col: ti.types.vector(3, ti.f32)):
    # Draw a thin ring outline into the shaded fluid image — used for the brush
    # cursor and for emitter/drain markers (so you can see where you're aiming).
    half = rpix + 3.0
    for i, j in ss_img:
        dx = i + 0.5 - sx
        dy = j + 0.5 - sy
        if ti.abs(dx) < half and ti.abs(dy) < half:
            dist = ti.sqrt(dx * dx + dy * dy)
            if ti.abs(dist - rpix) < 1.5:
                ss_img[i, j] = col


def render_screenspace(canvas, camera):
    cp = camera.curr_position
    cl = camera.curr_lookat
    cu = camera.curr_up
    _set_view([cp[0], cp[1], cp[2]], [cl[0], cl[1], cl[2]], [cu[0], cu[1], cu[2]])
    # ss_splat clips on x[p] directly, so update_particle_render() isn't
    # needed here. Colours are static (set at spawn/transform) except in heat
    # scenes, where the glow follows temperature every frame.
    if HEAT_ON:
        update_render_colors()
    t = float(np.tan(np.radians(SS_FOV) * 0.5))
    aspect = SS_W / SS_H
    ss_clear()
    ss_splat(t, aspect)
    for _ in range(blur_n[None]):   # bilateral passes -> smooth surface (quality)
        ss_blur()
    ss_shade(t, aspect)
    canvas.set_image(ss_img)


def build_surface() -> int:
    splat_render()
    for _ in range(2):            # blur passes -> smoother, less blocky
        smooth_density()
    compute_rho_max()
    iso_level[None] = max(RHO_ISO_FRAC * rho_max[None], 1e-4)
    compute_gradient()
    marching_tets()
    return min(int(surf_count[None]), MAX_VERTS)


# ----------------------------------------------------------------------------
# Save / load scenes (Phase 6)
# ----------------------------------------------------------------------------
def save_scene(path: str):
    np.savez(path,
             x=x.to_numpy(), v=v.to_numpy(), C=C.to_numpy(), F=F.to_numpy(),
             J=J.to_numpy(), mass=mass.to_numpy(), mat=mat.to_numpy(),
             temp=temp.to_numpy(), active=active.to_numpy(),
             n=N_PARTICLES, n_sim=N_SIM, n_pool=N_POOL, scene=SCENE)
    print(f"  saved scene -> {path} ({N_PARTICLES} particles)")


def load_scene(path: str) -> bool:
    import os
    if not os.path.exists(path):
        print(f"  !! load skipped: {path} not found (save a scene first)")
        return False
    d = np.load(path)
    if int(d["n"]) != N_PARTICLES:
        hint = (f"--particles {int(d['n_sim'])} --pool {int(d['n_pool'])}"
                if "n_sim" in d else f"a total of {int(d['n'])} particles")
        print(f"  !! load skipped: file has {int(d['n'])} particles, "
              f"running with {N_PARTICLES}. Restart with {hint}.")
        return False
    x.from_numpy(d["x"]); v.from_numpy(d["v"]); C.from_numpy(d["C"])
    F.from_numpy(d["F"]); J.from_numpy(d["J"]); mass.from_numpy(d["mass"])
    mat.from_numpy(d["mat"]); temp.from_numpy(d["temp"])
    if "active" in d:
        active.from_numpy(d["active"])
    update_render_colors()   # colours aren't saved; derive them from mat/temp
    print(f"  loaded scene <- {path}")
    return True


def run_saveload_test(frames: int) -> int:
    # Headless round-trip: simulate, save, perturb, load, verify exact restore.
    init_scene()
    for _ in range(frames):
        for _ in range(SUBSTEPS):
            substep()
    before = x.to_numpy().copy()
    import os
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "crucible_saveload_test.npz")
    save_scene(tmp)
    for _ in range(50):           # perturb the live state
        substep()
    moved = float(np.abs(x.to_numpy() - before).max())
    ok = load_scene(tmp)
    after = x.to_numpy()
    restored = float(np.abs(after - before).max())
    os.remove(tmp)
    print(f"  perturbation moved particles by {moved:.4f}; after reload "
          f"max diff = {restored:.2e}")
    if ok and restored < 1e-6 and moved > 1e-4:
        print("PASS: scene saved and restored exactly.")
        return 0
    print("FAIL: save/load did not round-trip.")
    return 1


@ti.kernel
def surface_nan_check(n: ti.i32) -> ti.i32:
    bad = 0
    for i in range(n):
        for d in ti.static(range(3)):
            if ti.math.isnan(surf_pos[i][d]) or ti.math.isnan(surf_nrm[i][d]):
                bad += 1
    return bad


def run_surface_selftest(frames: int) -> int:
    # Structural verification of the Phase 5 mesh pipeline (no window): advance
    # the scene, build the isosurface, and confirm it produces a sane,
    # finite triangle mesh. Visual quality still needs a human eye / screenshot.
    init_scene()
    print(f"Surface self-test: scene={SCENE}, iso_frac={RHO_ISO_FRAC}, lattice={NR}^3")
    for f in range(frames):
        for _ in range(SUBSTEPS):
            substep()
    nv = build_surface()
    nan = surface_nan_check(nv)
    tris = nv // 3
    print(f"  after {frames} frames: {nv} vertices ({tris} triangles), NaNs={nan}, "
          f"capped={'yes' if surf_count[None] > MAX_VERTS else 'no'}")
    if nv == 0:
        print("FAIL: surface extraction produced no geometry (check iso level).")
        return 1
    if nan > 0:
        print("FAIL: surface mesh contains NaN vertices/normals.")
        return 1
    print("PASS: surface pipeline produced a finite, non-empty mesh.")
    return 0


def _render_scene(scene, camera, render_mode):
    cp = camera.curr_position
    cam_pos[None] = ti.Vector([cp[0], cp[1], cp[2]])
    scene.ambient_light((0.30, 0.32, 0.38))
    scene.point_light(pos=(1.6, 2.2, 1.6), color=(1.0, 0.98, 0.92))
    scene.point_light(pos=(-0.6, 1.2, -0.4), color=(0.30, 0.38, 0.55))
    if render_mode == "surface":
        nv = build_surface()              # respects the clip plane internally
        if nv > 0:
            scene.mesh(surf_pos, normals=surf_nrm, per_vertex_color=surf_col,
                       vertex_count=nv, two_sided=True)
    else:
        update_particle_render()          # tint + clip (relocates clipped pts)
        scene.particles(rpos, radius=0.006, per_vertex_color=colors)


def run_gui(frames: int, snapshot: str = ""):
    # Interactive sandbox. With `snapshot`, runs offscreen for `frames` frames
    # and saves the last one (control panel included) to that PNG instead.
    import time
    init_scene()
    clip_axis[None] = 1          # default cross-section along Y
    clip_pos[None] = 0.5
    # the window must match the screen-space buffers blitted by set_image
    window = ti.ui.Window("CRUCIBLE", (SS_W, SS_H), vsync=False,  # uncapped
                          show_window=not snapshot)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(1.7, 1.1, 1.7)
    camera.lookat(0.5, 0.35, 0.5)
    camera.up(0.0, 1.0, 0.0)

    render_mode = RENDER_MODE
    paused = False
    clip = False
    brush_r = 0.10
    container_mode = 0           # 0 = box, 1 = round (live container shape)
    quality = 1                  # 0 fast / 1 balanced / 2 pretty (substeps+blur)
    # quality -> (substep multiplier, render blur passes). Substeps scale the
    # scene's own count (sized to its stiffest material), so "balanced" runs
    # exactly SUBSTEPS. More substeps = more sim-time/frame = livelier motion
    # but lower FPS; dt itself never changes, so stability is unaffected.
    QUALITY = [(0.67, 2), (1.0, 4), (1.45, 5)]
    flip_val = FLIP_LIQUID       # live water slipperiness (FLIP ratio)
    throw_gain = 25.0            # cursor drag -> spawn velocity (add tool)
    force_gain = 40.0            # cursor drag -> push impulse  (force tool)
    emit_rate = 600.0            # particles/frame per emitter
    heat_rate = 60.0             # degrees/frame added by the heat tool (<0 cools)
    fallback_h = 0.5             # y-plane used only when the ray hits nothing
    tool = 0                     # 0=add 1=delete 2=force 3=emitter 4=drain 5=heat
    spawn_idx = 0
    SPAWN_MATS = [WATER, WATER_C, WATER_P, JELLY]
    SPAWN_NAMES = ["water blue", "water teal", "water violet", "jelly"]
    TOOL_NAMES = ["add", "delete", "force", "emitter", "drain"]
    if HEAT_ON:                  # heat only does anything where it's simulated
        TOOL_NAMES.append("heat")
    # cursor/marker colours per tool
    TOOL_COL = [(0.3, 0.8, 1.0), (1.0, 0.2, 0.2), (0.6, 1.0, 0.6),
                (0.2, 1.0, 0.4), (0.8, 0.4, 1.0), (1.0, 0.55, 0.1)]
    TOOL_KEYS = [str(k + 1) for k in range(len(TOOL_NAMES))]
    emitters = []                # list of [pos(np3), radius, mat]
    drains = []                  # list of [pos(np3), radius]
    save_path = "scene.npz"
    last_t, fps = time.perf_counter(), 0.0

    # --- orbit camera state (turntable around `target`) --------------------
    target = np.array([0.5, 0.35, 0.5], np.float32)
    _off = np.array([1.7, 1.1, 1.7], np.float32) - target
    cam_dist = float(np.linalg.norm(_off))
    cam_yaw = float(np.arctan2(_off[2], _off[0]))
    cam_pitch = float(np.arcsin(np.clip(_off[1] / cam_dist, -1.0, 1.0)))
    prev_cursor = None           # for orbit/pan drag deltas
    prev_hit = None              # for throw/force drag velocity
    lmb_prev = False             # for click-edge detection (emitter/drain place)
    WORLD_UP = np.array([0.0, 1.0, 0.0], np.float32)

    def reset():
        init_scene()
        emitters.clear()
        drains.clear()

    frame = 0
    quit_requested = False
    while window.running and not quit_requested and (frames <= 0 or frame < frames):
        # --- discrete key presses ------------------------------------------
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                paused = not paused
            elif e.key == 'r':
                reset()
            elif e.key == 'm':
                order = ["fluid", "surface", "particles"]
                render_mode = order[(order.index(render_mode) + 1) % 3]
            elif e.key == 'c':
                clip = not clip
            elif e.key in TOOL_KEYS:
                tool = int(e.key) - 1
            elif e.key == ti.ui.TAB:
                spawn_idx = (spawn_idx + 1) % len(SPAWN_MATS)
            elif e.key == 'x':
                emitters.clear(); drains.clear()
            elif e.key == '[':
                brush_r = max(0.02, brush_r - 0.02)
            elif e.key == ']':
                brush_r = min(0.30, brush_r + 0.02)
            elif e.key == ti.ui.ESCAPE:
                quit_requested = True

        # --- camera: RMB drag = orbit, MMB drag = pan, =/- (or Up/Down) zoom
        cx, cy = window.get_cursor_pos()
        if window.is_pressed(ti.ui.UP):
            cam_dist *= 0.97
        if window.is_pressed(ti.ui.DOWN):
            cam_dist *= 1.03
        cam_dist = float(np.clip(cam_dist, 0.4, 6.0))
        orbiting = window.is_pressed(ti.ui.RMB)
        panning = window.is_pressed(ti.ui.MMB)
        cp, sp = np.cos(cam_pitch), np.sin(cam_pitch)
        cyw, syw = np.cos(cam_yaw), np.sin(cam_yaw)
        view_dir = np.array([cp * cyw, sp, cp * syw], np.float32)  # target->eye
        if (orbiting or panning) and prev_cursor is not None:
            dx = cx - prev_cursor[0]
            dy = cy - prev_cursor[1]
            if orbiting:
                cam_yaw -= dx * 3.0
                cam_pitch = float(np.clip(cam_pitch + dy * 3.0, -1.45, 1.45))
            else:  # pan the pivot in the screen plane
                fwd = -view_dir
                rgt = np.cross(fwd, WORLD_UP); rgt /= np.linalg.norm(rgt)
                up = np.cross(rgt, fwd)
                target = target - (rgt * dx + up * dy) * cam_dist
            cp, sp = np.cos(cam_pitch), np.sin(cam_pitch)
            cyw, syw = np.cos(cam_yaw), np.sin(cam_yaw)
            view_dir = np.array([cp * cyw, sp, cp * syw], np.float32)
        prev_cursor = (cx, cy)

        eye = target + cam_dist * view_dir
        camera.position(float(eye[0]), float(eye[1]), float(eye[2]))
        camera.lookat(float(target[0]), float(target[1]), float(target[2]))
        camera.up(0.0, 1.0, 0.0)

        # --- control panel --------------------------------------------------
        gui = window.get_gui()
        with gui.sub_window("CRUCIBLE", 0.0, 0.0, 0.30, 0.95):
            gui.text(f"scene: {SCENE}   sim {N_SIM} + pool {N_POOL}")
            gui.text(f"{fps:5.1f} FPS   {'PAUSED' if paused else 'running'}")
            paused = gui.checkbox("pause (space)", paused)
            if gui.button("reset scene (r)"):
                reset()
            _modes = ["fluid", "surface", "particles"]
            ridx = gui.slider_int("render 0fluid 1surf 2pts (m)",
                                  _modes.index(render_mode), 0, 2)
            render_mode = _modes[ridx]
            gui.text("--- camera ---")
            gui.text("RMB drag orbit   MMB drag pan")
            cam_dist = gui.slider_float("zoom (Up/Down)", cam_dist, 0.4, 6.0)
            gui.text("--- water ---")
            quality = gui.slider_int("quality 0fast 1med 2pretty", quality, 0, 2)
            flip_val = gui.slider_float("slipperiness (low=viscous)",
                                        flip_val, 0.50, 0.97)
            container_mode = gui.slider_int("container 0=box 1=round",
                                            container_mode, 0, 1)
            gui.text("--- cross-section (c) ---")
            clip = gui.checkbox("clip plane on", clip)
            ax = gui.slider_int("axis 0=x 1=y 2=z", clip_axis[None], 0, 2)
            clip_axis[None] = ax
            clip_pos[None] = gui.slider_float("position", clip_pos[None], 0.0, 1.0)
            gui.text(f"--- tool (keys 1-{len(TOOL_NAMES)}) ---")
            tool = gui.slider_int(f"tool = {TOOL_NAMES[tool]}", tool, 0,
                                  len(TOOL_NAMES) - 1)
            # Tool-specific control comes first so it is always visible:
            if tool in (0, 3):     # add / emitter both need a material
                gui.text("material palette:")
                for k, nm in enumerate(SPAWN_NAMES):
                    label = (">> " if k == spawn_idx else "   ") + nm
                    if gui.button(label):
                        spawn_idx = k
                if tool == 0:
                    gui.text(f"ADD {SPAWN_NAMES[spawn_idx]} - drag to throw")
                    throw_gain = gui.slider_float("throw strength", throw_gain,
                                                  0.0, 80.0)
                else:
                    gui.text(f"EMITTER pours {SPAWN_NAMES[spawn_idx]} - click to place")
                    emit_rate = gui.slider_float("emit rate", emit_rate, 50.0, 2000.0)
                    gui.text(f"  emitters: {len(emitters)}   (x = clear all)")
            elif tool == 2:
                gui.text("  FORCE - swipe to push fluid")
                force_gain = gui.slider_float("force strength", force_gain,
                                              0.0, 120.0)
            elif tool == 4:
                gui.text("  DRAIN - click to place a sink")
                gui.text(f"  drains: {len(drains)}   (x = clear all)")
            elif tool == 5:
                gui.text("  HEAT - hold to heat (negative rate cools)")
                heat_rate = gui.slider_float("heat rate", heat_rate,
                                             -200.0, 200.0)
            else:
                gui.text("  DELETE removes material in the sphere")
            brush_r = gui.slider_float("brush radius [ ]", brush_r, 0.02, 0.3)
            gui.text("LMB in view: use tool (depth-picked)")
            gui.text("--- scene file ---")
            if gui.button("save scene"):
                save_scene(save_path)
            if gui.button("load scene"):
                load_scene(save_path)
        clip_on[None] = 1 if clip else 0
        flip_ratio[None] = flip_val      # live water slipperiness from the slider
        container_shape[None] = container_mode
        sub_mult, blur_n[None] = QUALITY[quality]   # quality -> substeps + blur
        n_sub = max(1, round(SUBSTEPS * sub_mult))

        # --- simulate + run placed emitters/drains --------------------------
        if not paused:
            for _ in range(n_sub):
                substep()
            for ep, er, em_mat in emitters:
                spawn_brush(ti.Vector([float(ep[0]), float(ep[1]), float(ep[2])]),
                            er, em_mat, int(emit_rate),
                            ti.Vector([0.0, -1.5, 0.0]))
            for dp, dr_ in drains:
                delete_brush(ti.Vector([float(dp[0]), float(dp[1]),
                                        float(dp[2])]), dr_)

        # --- world-space camera basis (for picking + projecting overlays) ---
        fwd = (target - eye); fwd /= np.linalg.norm(fwd)
        rgt = np.cross(fwd, WORLD_UP); rgt /= np.linalg.norm(rgt)
        upc = np.cross(rgt, fwd)
        tt = float(np.tan(np.radians(SS_FOV) * 0.5))
        asp = SS_W / SS_H

        def pick(px, py):
            # depth-pick the actual surface under the cursor (fluid mode leaves
            # view-space depth in ss_depth); fall back to the air-spawn plane.
            if render_mode == "fluid":
                ii = min(max(int(px * SS_W), 0), SS_W - 1)
                jj = min(max(int(py * SS_H), 0), SS_H - 1)
                dd = ss_depth[ii, jj]
                if dd < 1e8:
                    vx = (px * 2 - 1) * tt * asp * dd
                    vy = (py * 2 - 1) * tt * dd
                    return eye + vx * rgt + vy * upc + dd * fwd
            rd = fwd + (px * 2 - 1) * tt * asp * rgt + (py * 2 - 1) * tt * upc
            rd /= np.linalg.norm(rd)
            if abs(rd[1]) > 1e-4:
                kk = (fallback_h - eye[1]) / rd[1]
                if kk > 0:
                    return eye + kk * rd
            return None

        # --- aim + act ------------------------------------------------------
        over_view = cx > 0.31 and not (orbiting or panning)
        hover = pick(cx, cy) if over_view else None
        lmb = window.is_pressed(ti.ui.LMB)
        lmb_edge = lmb and not lmb_prev
        if lmb and over_view and hover is not None:
            drag = (hover - prev_hit) if prev_hit is not None \
                else np.zeros(3, np.float32)
            bc = ti.Vector([float(hover[0]), float(hover[1]), float(hover[2])])
            if tool == 0:                    # add material (drag to throw)
                vel = drag * throw_gain
                spawn_brush(bc, brush_r, SPAWN_MATS[spawn_idx], 2000,
                            ti.Vector([float(vel[0]), float(vel[1]),
                                       float(vel[2])]))
            elif tool == 1:
                delete_brush(bc, brush_r)
            elif tool == 2:
                f = drag * force_gain
                force_brush(bc, brush_r,
                            ti.Vector([float(f[0]), float(f[1]), float(f[2])]))
            elif tool == 3 and lmb_edge:     # place emitter (one per click)
                emitters.append([hover.copy(), brush_r, SPAWN_MATS[spawn_idx]])
            elif tool == 4 and lmb_edge:     # place drain
                drains.append([hover.copy(), brush_r])
            elif tool == 5:                  # heat / cool
                heat_brush(bc, brush_r, heat_rate)
            prev_hit = hover
        else:
            prev_hit = None
        lmb_prev = lmb

        # --- render ---------------------------------------------------------
        scene.set_camera(camera)
        if render_mode == "fluid":
            render_screenspace(canvas, camera)
            # overlays: emitter/drain markers + the live brush cursor ring

            def screen_ring(p, world_r, col):
                rel = p - eye
                dd = float(np.dot(rel, fwd))
                if dd <= SS_NEAR:
                    return
                sx = (float(np.dot(rel, rgt)) / (dd * tt * asp) * 0.5 + 0.5) * SS_W
                sy = (float(np.dot(rel, upc)) / (dd * tt) * 0.5 + 0.5) * SS_H
                rpix = world_r / (dd * tt) * (SS_H * 0.5)
                draw_ring(sx, sy, max(rpix, 3.0),
                          ti.Vector([col[0], col[1], col[2]]))
            for ep, er, _m in emitters:
                screen_ring(ep, er, TOOL_COL[3])     # emitter colour
            for dp, dr_ in drains:
                screen_ring(dp, dr_, TOOL_COL[4])     # drain colour
            if hover is not None:
                screen_ring(hover, brush_r, TOOL_COL[tool])
            canvas.set_image(ss_img)         # re-upload with overlays drawn in
        else:
            _render_scene(scene, camera, render_mode)
            canvas.set_background_color((0.05, 0.05, 0.08))
            canvas.scene(scene)
        if snapshot and frame + 1 >= frames:
            window.save_image(snapshot)
            print(f"  saved {snapshot}")
        elif not snapshot:
            window.show()

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
        last_t = now
        frame += 1


def run_screenshot(frames: int, path: str):
    # Render `frames` steps then save a PNG, so the (otherwise headless) build
    # can be visually inspected. Uses an offscreen-capable GGUI window.
    init_scene()
    window = ti.ui.Window(f"CRUCIBLE — {SCENE}", (SS_W, SS_H),
                          vsync=False, show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(1.6, 1.1, 1.6)
    camera.lookat(0.5, 0.35, 0.5)
    camera.up(0.0, 1.0, 0.0)
    for _ in range(max(frames, 1)):
        for _ in range(SUBSTEPS):
            substep()
    if RENDER_MODE == "fluid":
        scene.set_camera(camera)              # so curr_position is populated
        render_screenspace(canvas, camera)
    else:
        scene.set_camera(camera)
        _render_scene(scene, camera, RENDER_MODE)
        canvas.set_background_color((0.05, 0.05, 0.08))
        canvas.scene(scene)
    window.save_image(path)
    print(f"  saved {path} (mode={RENDER_MODE}, clip={'on' if clip_on[None] else 'off'})")


def run_spawn_test(path: str) -> int:
    # Verify the spawn/delete brush: start from an empty box, "add" a ball of
    # material, let it fall, "delete" part of it, and confirm the active count
    # tracks. Also saves a screenshot to eyeball.
    load_materials()
    park_pool()
    floor_sticky[None] = 0
    n0 = count_active()
    for _ in range(40):
        spawn_brush(ti.Vector([0.5, 0.72, 0.5]), 0.13, JELLY, 4000,
                    ti.Vector([0.0, 0.0, 0.0]))
    n1 = count_active()
    for _ in range(120):
        for _ in range(SUBSTEPS):
            substep()
    n2 = count_active()
    delete_brush(ti.Vector([0.5, 0.10, 0.5]), 0.15)
    n3 = count_active()
    print(f"spawn-test: empty={n0}, after add={n1}, after settle={n2}, "
          f"after delete={n3}")

    res = (1024, 768)
    window = ti.ui.Window("spawn", res, vsync=False, show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(1.7, 1.1, 1.7); camera.lookat(0.5, 0.3, 0.5); camera.up(0, 1, 0)
    scene.set_camera(camera)
    _render_scene(scene, camera, "surface")
    canvas.set_background_color((0.05, 0.05, 0.08))
    canvas.scene(scene)
    window.save_image(path)
    print(f"  saved {path}")
    if n0 == 0 and n1 > 100000 and n3 < n2:
        print("PASS: spawn added material and delete removed it.")
        return 0
    print("FAIL: spawn/delete counts unexpected.")
    return 1


def run_panel_shot(path: str):
    # Render one frame of the real GUI (control panel in add mode) to a PNG, to
    # verify the panel layout — including the material picker — is fully visible.
    run_gui(1, snapshot=path)


def run_bench(frames: int):
    import time
    init_scene()
    print(f"Benchmark: scene={SCENE}, {N_PARTICLES} particles, {N_GRID}^3 grid, "
          f"{SUBSTEPS} substeps/frame")

    def timeit(fn, n, warmup=3):
        for _ in range(warmup):
            fn()
        ti.sync()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        ti.sync()
        return (time.perf_counter() - t0) / n * 1e3  # ms per call

    sub_ms = timeit(substep, 200)
    splat_ms = timeit(splat_render, 50)
    smooth_ms = timeit(smooth_density, 50)
    grad_ms = timeit(compute_gradient, 50)
    march_ms = timeit(marching_tets, 50)
    surf_ms = timeit(build_surface, 50)
    frame_sub = sub_ms * SUBSTEPS
    print(f"  substep            : {sub_ms:7.3f} ms  x{SUBSTEPS} = {frame_sub:8.2f} ms/frame")
    print(f"  build_surface total: {surf_ms:7.3f} ms/frame")
    print(f"    - splat_render   : {splat_ms:7.3f} ms")
    print(f"    - smooth_density : {smooth_ms:7.3f} ms")
    print(f"    - compute_grad   : {grad_ms:7.3f} ms")
    print(f"    - marching_tets  : {march_ms:7.3f} ms")
    surf_frame = frame_sub + surf_ms
    part_frame = frame_sub
    print(f"  => surface render  : {surf_frame:8.2f} ms/frame  (~{1000/surf_frame:5.1f} FPS, sim only)")
    print(f"  => particle render : {part_frame:8.2f} ms/frame  (~{1000/part_frame:5.1f} FPS, sim only)")


def run_render_bench(frames: int):
    # Measure the *full* per-frame render cost (including the GGUI mesh
    # upload+draw that run_bench's kernel-only timing misses), surface vs
    # particles, using an offscreen window.
    import time
    init_scene()
    for _ in range(120):                      # settle into a representative state
        for _ in range(SUBSTEPS):
            substep()
    window = ti.ui.Window("bench", (1280, 800), vsync=False, show_window=False)
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(1.7, 1.1, 1.7)
    camera.lookat(0.5, 0.35, 0.5)
    camera.up(0.0, 1.0, 0.0)
    scene.set_camera(camera)

    def draw(mode):
        if mode == "fluid":
            render_screenspace(canvas, camera)
        else:
            _render_scene(scene, camera, mode)
            canvas.set_background_color((0.05, 0.05, 0.08))
            canvas.scene(scene)
        window.save_image("._renderbench.png")   # forces the frame to complete

    def timeit(fn, n=30):
        for _ in range(3):
            fn()
        ti.sync()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        ti.sync()
        return (time.perf_counter() - t0) / n * 1e3

    # Pure screen-space kernel cost (no save_image), the honest fluid number.
    fluid_kernels = timeit(lambda: render_screenspace(canvas, camera))
    surf_ms = timeit(lambda: draw("surface"))
    part_ms = timeit(lambda: draw("particles"))
    fluid_ms = timeit(lambda: draw("fluid"))
    import os
    if os.path.exists("._renderbench.png"):
        os.remove("._renderbench.png")
    print(f"Render bench: scene={SCENE}, {N_PARTICLES} particles")
    print(f"  fluid (screen-space) kernels only : {fluid_kernels:7.2f} ms  "
          f"(~{1000/fluid_kernels:5.1f} FPS)")
    print(f"  [with save_image overhead] surface={surf_ms:.1f}  "
          f"particles={part_ms:.1f}  fluid={fluid_ms:.1f} ms")


def main():
    global SCENE, RENDER_MODE, RHO_ISO_FRAC
    parser = argparse.ArgumentParser(description="CRUCIBLE MLS-MPM physics sandbox")
    parser.add_argument("--headless", action="store_true",
                        help="run without a window; assert stability")
    parser.add_argument("--frames", type=int, default=0,
                        help="frame count (0 = run forever in GUI mode)")
    parser.add_argument("--arch", default="cuda", choices=["cuda", "vulkan", "cpu"])
    parser.add_argument("--render", default="fluid",
                        choices=["fluid", "surface", "particles"],
                        help="fluid: screen-space refractive water (default); "
                             "surface: isosurface mesh; particles: raw cloud")
    parser.add_argument("--surface-test", action="store_true",
                        help="headless structural check of the surface mesh pipeline")
    parser.add_argument("--bench", action="store_true",
                        help="time substep + surface stages and report ms/frame")
    parser.add_argument("--render-bench", action="store_true",
                        help="time full per-frame render (incl. GGUI draw), surface vs particles")
    parser.add_argument("--dt", type=float, default=0.0,
                        help="override substep size (0 = scene default)")
    parser.add_argument("--substeps", type=int, default=0,
                        help="override substeps per frame (0 = scene default)")
    parser.add_argument("--particles", type=int, default=N_SIM,
                        help="scene particle count (fewer = faster, coarser); sizes fields at startup")
    parser.add_argument("--pool", type=int, default=N_POOL,
                        help="free particles reserved for the add/emitter tools; sizes fields at startup")
    parser.add_argument("--grid", type=int, default=N_GRID,
                        help="sim grid resolution per axis (sizes fields at startup)")
    parser.add_argument("--render-grid", type=int, default=NR,
                        help="render lattice resolution (higher = smoother surface, slower)")
    parser.add_argument("--save-test", action="store_true",
                        help="headless save/load round-trip check")
    parser.add_argument("--spawn-test", action="store_true",
                        help="verify the spawn/delete material brush + screenshot")
    parser.add_argument("--panel-shot", action="store_true",
                        help="render the control panel (add mode) to shots/panel.png")
    parser.add_argument("--clip-axis", type=int, default=-1,
                        help="cross-section axis for screenshots (-1=off, 0=x,1=y,2=z)")
    parser.add_argument("--clip-pos", type=float, default=0.5,
                        help="cross-section plane position in [0,1]")
    parser.add_argument("--screenshot", default="",
                        help="render N frames then save a PNG to this path and exit")
    parser.add_argument("--iso", type=float, default=RHO_ISO_FRAC,
                        help="isosurface level as a fraction of peak density (0-1)")
    parser.add_argument("--scene", default="pool",
                        choices=["pool", "empty", "drop", "layers", "mixed",
                                 "jelly", "sand", "lava", "melt"],
                        help="pool: a calm water tank (default sandbox); "
                             "empty: a blank tank to build in with the add brush; "
                             "drop: heavy ball sinks through a light pool; "
                             "layers: heavy-over-light interface; "
                             "mixed: demo of why MPM can't demix; "
                             "jelly: an elastic cube wobbles and bounces; "
                             "sand: a granular column slumps into a pile; "
                             "lava: lava+water -> stone+steam by heat; "
                             "melt: a cold ice block melts to water")
    args = parser.parse_args()
    # Taichi was already initialized at import time (see top of file) so that
    # module-level fields could be allocated; args.arch is honored there.
    global DT, SUBSTEPS, HEAT_ON
    SCENE = args.scene
    RENDER_MODE = args.render
    RHO_ISO_FRAC = args.iso
    DT, SUBSTEPS = scene_timestep(SCENE)
    HEAT_ON = SCENE in HEAT_SCENES
    if args.dt > 0.0:
        DT = args.dt
    if args.substeps > 0:
        SUBSTEPS = args.substeps

    if args.save_test:
        raise SystemExit(run_saveload_test(args.frames if args.frames > 0 else 100))
    if args.spawn_test:
        raise SystemExit(run_spawn_test("shots/spawn.png"))
    if args.panel_shot:
        run_panel_shot("shots/panel.png")
        raise SystemExit(0)
    if args.bench:
        run_bench(args.frames if args.frames > 0 else 200)
        raise SystemExit(0)
    if args.render_bench:
        run_render_bench(args.frames if args.frames > 0 else 30)
        raise SystemExit(0)
    if args.screenshot:
        if args.clip_axis >= 0:
            clip_on[None] = 1
            clip_axis[None] = args.clip_axis
            clip_pos[None] = args.clip_pos
        run_screenshot(args.frames if args.frames > 0 else 120, args.screenshot)
        raise SystemExit(0)
    if args.surface_test:
        frames = args.frames if args.frames > 0 else 120
        raise SystemExit(run_surface_selftest(frames))
    if args.headless:
        frames = args.frames if args.frames > 0 else 600
        raise SystemExit(run_headless(frames))
    else:
        run_gui(args.frames)


if __name__ == "__main__":
    main()
