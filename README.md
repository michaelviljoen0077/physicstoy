# CRUCIBLE

A real-time 3D physics sandbox built on **MLS-MPM** (GPU, via Taichi) — a
physically-grounded cousin of *The Powder Toy* where a curated set of materials
flow, pile, deform, and transform into one another through a shared heat field.

See [`spec.md`](spec.md) for the full design (if present); this README tracks
what actually runs.

## Status

| Phase | Goal | State |
|---|---|---|
| 0 — Setup | Taichi env + CUDA on the RTX 4080 | ✅ done |
| 1 — One fluid, stable | Single weakly-compressible water in a box | ✅ done — settles, no blow-ups |
| 2 — First slice | Two fluids: colour + density | ✅ done — heavy fluid settles below light |
| 3 — Material classes | Granular (sand) + elastic (jelly) | ✅ done — jelly wobbles/bounces, sand piles/slumps |
| 4 — Heat + transformations | Per-particle temperature, grid diffusion, transitions | ✅ done — lava+water→stone+steam, ice→water |
| 5 — Beautiful rendering | Isosurface mesh, lighting, temperature→emission | ✅ done — smooth lit surfaces, glowing lava |
| 6 — Interaction & polish | Control panel, clipping plane, save/load, heat brush | ✅ done — interactive toy with cross-section + save/load |

## Environment

> **Important:** Taichi 1.7.4 ships no wheels for Python 3.12+ (and none for the
> 3.14 on this machine). Use **Python 3.9–3.11**. This repo is set up against
> Python 3.9.13 in `.venv/`.

```powershell
# from the project root
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
# Interactive GGUI window (orbit camera: hold right mouse + WASD)
.\.venv\Scripts\python.exe src\mpm_fluid.py                 # default: drop scene

# Headless gate — no window; asserts no NaN, bounded velocity, correct layering
.\.venv\Scripts\python.exe src\mpm_fluid.py --headless --frames 600

# Scenes:
.\.venv\Scripts\python.exe src\mpm_fluid.py --scene drop    # heavy ball sinks through light pool
.\.venv\Scripts\python.exe src\mpm_fluid.py --scene jelly   # elastic cube wobbles and bounces
.\.venv\Scripts\python.exe src\mpm_fluid.py --scene sand    # granular column slumps into a pile
.\.venv\Scripts\python.exe src\mpm_fluid.py --scene lava    # lava+water -> stone+steam (heat)
.\.venv\Scripts\python.exe src\mpm_fluid.py --scene melt    # cold ice block melts to water
.\.venv\Scripts\python.exe src\mpm_fluid.py --scene layers  # heavy-over-light interface (see note)
.\.venv\Scripts\python.exe src\mpm_fluid.py --scene mixed   # why MPM can't demix co-located fluids
```

`--arch cpu` falls back to CPU if needed; default is `cuda`.

### Interactive controls (GGUI window)

A control panel (top-left) plus keyboard shortcuts:

| Key | Action | Panel |
|---|---|---|
| `space` | pause / resume | pause checkbox |
| `r` | reset scene | reset button |
| `m` | toggle surface / particle render | checkbox |
| `c` | toggle clip plane | clip checkbox + axis/position sliders |
| right-drag + WASD | orbit / fly camera | — |

The **cross-section plane** (clip) hides everything past an axis-aligned plane so
you can see *inside* the volume — the practical answer to "placing/seeing things
in a 3D box is awkward." **Save/load** writes the full particle state to
`scene.npz`.

The **brush tool** (panel) is **mouse-driven** — left-click in the view to apply
it where the cursor ray meets the "brush height" plane (slider). Three modes:
- **heat** — add/remove heat (melt ice, boil water, freeze); acts only in
  lava/melt scenes where heat is simulated.
- **add** — spawn a chosen material (water/ice/lava/sand/jelly/stone) by pulling
  from a free particle **pool** (`--pool N`, default 250k).
- **delete** — return material in the brush back to the pool.

```powershell
# headless save/load round-trip check; cross-section screenshots:
.\.venv\Scripts\python.exe src\mpm_fluid.py --save-test
.\.venv\Scripts\python.exe src\mpm_fluid.py --screenshot cut.png --scene lava --clip-axis 2 --clip-pos 0.5
```

### Performance

Substeps are sized per scene to the **stiffest material's sound-speed CFL**
(`dt < dx/√(E/ρ)`), holding `dt × substeps` ≈ constant so the visual speed is
unchanged. Pure fluids (soft EOS) need ~7 substeps; stiff solids (sand, ice,
stone) need more. Heat is compiled out of the substep for scenes that don't use
it. Approximate sim cost (RTX 4080, 64³ grid, surface render):

| Scene | particles | substeps | ~FPS |
|---|---|---|---|
| drop  | 800k | 7  | ~60 |
| jelly | 150k | 8  | ~90 |
| sand  | 150k | 14 | ~54 |
| lava  | 800k | 11 | ~30 |

Knobs: `--particles N` (fewer = faster, coarser), `--substeps N`, `--dt F`,
`--render particles` (skips the mesh), `--bench` (per-stage timing).

**Surface render gotcha (fixed):** GGUI re-uploads the *entire* mesh vertex
buffer every frame, ignoring `vertex_count`. An oversized buffer
(`MAX_VERTS = 6M`) silently cost **>130 ms/frame** uploading 216 MB to draw a
~280k-vertex mesh — the real reason surface mode felt slow. Sizing the buffer to
what the mesh needs (`~NR²·300`) dropped surface-vs-particle overhead from
**132 ms → 10 ms**. Use `--render-bench` to measure full per-frame render cost.

**Where the time goes:** `--bench` shows `substep` is ~94% of frame cost. It is
**per-particle bound**: a controlled A/B at fixed particle count found grid
resolution (64³/96³/128³) makes *no* difference to substep cost, so it is not
grid-bandwidth or atomic-contention limited. That also means a **finer grid does
not help** — it leaves substep cost unchanged but shrinks `dx`, tightening the
CFL and forcing *more* substeps (a net loss). The real levers are therefore
**substep count** (sized to each material's sound-speed CFL) and **particle
count** (`--particles`).

Reduced precision (f16 "rounding") is *not* useful here: the cost is compute,
not bandwidth, and particle masses (~5e-7) are subnormal in f16 — every
mass-weighted grid value would underflow without a full rescale.

> Caveat: these micro-benchmarks vary ~1.5× run-to-run from GPU boost-clock
> state, so treat single-run numbers as approximate.

### Rendering

```powershell
# Smooth isosurface mesh (default) vs raw particle cloud
.\.venv\Scripts\python.exe src\mpm_fluid.py --scene lava --render surface
.\.venv\Scripts\python.exe src\mpm_fluid.py --scene lava --render particles
.\.venv\Scripts\python.exe src\mpm_fluid.py --scene drop --iso 4.0     # tune surface tightness

# Save a PNG (no interactive window) to inspect the look
.\.venv\Scripts\python.exe src\mpm_fluid.py --screenshot shot.png --scene lava --frames 8
# Structural self-test of the mesh pipeline (no window)
.\.venv\Scripts\python.exe src\mpm_fluid.py --surface-test --scene drop
```

## What Phase 2 does

Each particle carries a `material_id` selecting its rest density (encoded as
per-particle **mass**), colour, and stiffness. Density-driven layering is then
emergent — nothing scripted.

The default **drop** scene releases a ball of heavy (orange) fluid above a pool
of light (blue) fluid. It plunges in, mushrooms downward, and **settles below
the light fluid** — verified by the headless gate (heavy mean-height ends clearly
below light, velocities bounded, no NaNs). ~800k particles keep the filled region
at ~8 particles/cell, MLS-MPM's sweet spot.

## What Phase 3 does

Two new **material classes** join the liquid, selected per particle:

- **Elastic (jelly)** — tracks the full deformation gradient `F` with a
  neo-Hookean stress. The `jelly` scene drops a cube that squishes on impact,
  **bounces**, wobbles, and holds together as a blob (verified: it keeps real
  vertical extent instead of flattening like a liquid).
- **Granular (sand)** — Hencky elasticity with a **Drucker-Prager** yield
  return-mapping each step, so it piles and slumps at an angle of repose. The
  `sand` scene collapses a tall column into a low, wide, settling pile. A
  **no-slip (sticky) floor** gives the pile basal friction so it locks instead
  of sliding flat — without that, pure free-slip walls let the pile creep
  outward forever.

## What Phase 4 does

Every particle carries a **temperature**, scattered to the grid alongside
momentum, **diffused** across occupied cells each substep (empty cells are
insulating "air"), and gathered back. A gentle **radiative cooling** relaxes
temperature toward ambient so the system can actually shed heat. Transformations
are pure temperature thresholds applied in the particle update:

```
ice  --warm-->  water  --boil-->  steam        (and the reverse on cooling)
lava --cool-->  stone  --very hot--> lava
```

The marquee reaction — **lava + water → stone + steam** — is *not* a scripted
contact rule. A molten lava ball quenches in a cold pool: heat diffusion alone
flash-boils the water it touches into steam while the lava cools below its
freezing point and turns to stone; the steam then condenses back as everything
returns to ambient. (Pillar 3: emergence from physics, not rules.) Hot material
glows toward white-hot in the GGUI preview (a taste of the Phase-5 emission).

## What Phase 5 does — rendering

Three render modes (`--render`, or `m` to cycle in the GUI):

- **`fluid` (default)** — **screen-space fluid rendering**, the technique used by
  GPU water demos like *dli/fluid* and *Splash*. Each particle is splatted as a
  sphere sprite into a screen **depth buffer**; the depth is blurred in 2D,
  normals are reconstructed from the depth gradient, and the surface is shaded
  with diffuse + specular + **fresnel** + **thickness-based transparency** (you
  can see submerged material *through* the water). Cost is O(pixels), ~16 ms/frame
  for ~1M particles — faster *and* better-looking than meshing, with no per-frame
  mesh upload. Implemented in Taichi kernels, blitted with `canvas.set_image`.
- **`surface`** — marching-tetrahedra **isosurface mesh** (gradient normals,
  per-material colour, temperature→emission, fresnel sheen). Decoupled render
  lattice (`--render-grid`, default 96) and a grid-independent iso level
  (fraction of peak density). Good for opaque solids.
- **`particles`** — the raw particle cloud (`m` to toggle; handy for debugging).

> Two bugs worth remembering, both fixed: a non-indexed mesh needs each
> triangle's 3 vertices written *consecutively* (reserve them with one
> `atomic_add`), or parallel cells interleave vertices into volume-spanning
> triangles; and the iso level must sit in dense fluid, not the sparse fringe.

### Known limitation (honest note)
- **`layers`** (a *flat* heavy-over-light interface) does **not** spontaneously
  overturn into Rayleigh-Taylor plumes. In single-grid pure-APIC MPM the shared
  grid velocity at interface nodes locks the layers and APIC's numerical
  viscosity damps the instability; the inverted state is metastable. The path to
  real flat-interface plumes is a FLIP/PIC velocity blend (lower viscosity) —
  deferred so as not to trade away Phase-1/2 stability.
- **`mixed`** (two fluids intermixed at the sub-cell scale) cannot demix at all:
  co-located particles of different materials share one grid velocity and advect
  identically. This is a fundamental property of single-grid MPM, kept as a demo.

## Phase 1 (still runnable via git history / earlier scenes)

The single-fluid sloshing test that proved the core is stable: a blob falls,
splashes, and settles with no blow-ups.
