# CRUCIBLE — Project Spec v0.1

A real-time 3D physics sandbox: a beautiful, physically-grounded cousin of *The
Powder Toy*, where a **curated** set of materials flow, deform, pile, mix, and
**transform into one another through a shared heat field** — all simulated with
MLS-MPM on the GPU. ~15 materials simulated *properly*, not 500 shallow ones.

## Design pillars
1. **Physical realism over breadth** — ~12–18 deeply-simulated materials.
2. **Temperature is the bridge** — one per-particle heat field drives physics
   (viscosity, melting, stiffness) *and* transformations.
3. **Emergence from physics, not scripted rules** — buoyancy, layering, plumes,
   piling fall out of mass + momentum + gravity; nothing fragile to balance.
4. **Vertical-slice discipline** — stable core first, beauty second, breadth
   last. Never start phase N+1 until phase N is stable.
5. **Decouple sim resolution from render quality** — a coarse sim grid renders
   as smooth, glowing, translucent beauty.

## Technical approach
- **Method:** MLS-MPM (fast, stable practical variant of MPM).
- **Compute:** Taichi (Python-embedded GPU DSL) → CUDA on an RTX 4080 (16 GB).
- **Render:** Taichi GGUI particles early; screen-space fluids + custom shaders
  later.

### Per-particle data
`position`, `velocity`, `C` (affine velocity, APIC/MLS-MPM), `F` (deformation
gradient for elastic/plastic), `mass`, `volume`, `material_id`, `temperature`,
`color`/`dye`, `Jp` (plastic accumulator).

### Background grid (scratch, reset every step)
`grid_mass`, `grid_momentum/velocity`, `grid_temperature`. Target 128³ grid;
particle budget 100k → 1M+.

### Substep loop
1. **P2G** scatter mass/momentum/heat to grid (atomics).
2. **Grid update** gravity, stress, walls, diffuse temperature.
3. **G2P** gather velocity/heat back.
4. **Particle update** advect; update F/plasticity/temperature; apply
   transformations.

### Constitutive models
- Fluid (water, oil, lava, mercury): weakly-compressible EOS pressure.
- Granular (sand, ash, snow): Drucker–Prager plasticity.
- Elastic (jelly, ice, rigid): neo-Hookean.

### Transformations (curated table, mostly temperature thresholds + contact)
Ice+warm→Water · Water+hot→Steam · Lava+cold/Water→Stone(+Steam) ·
Wood+hot→Fire+Ash · Oil+hot→Fire · Sand+very hot→Glass.

## Roadmap
| Phase | Goal | Success criterion |
|---|---|---|
| 0 Setup | env + render static cloud | toolchain runs on 4080 |
| 1 One fluid | single weakly-compressible water, gravity, walls | sloshes & settles, no blow-ups |
| 2 ⭐ First slice | two fluids: colour + density | denser settles below; Rayleigh-Taylor plumes; no blow-ups |
| 3 Material classes | granular (sand) + elastic (jelly) | ✅ done — jelly wobbles/bounces; sand piles/slumps |
| 4 Heat + transforms | per-particle temp, grid diffusion, first transitions | ✅ done — lava+water→stone+steam (emergent), ice→water |
| 5 Rendering | screen-space fluids, refraction, temp→emission, lighting | ✅ done — marching-tets isosurface, lighting, temp→emission glow |
| 6 Interaction | picker, emitters/brush, clipping plane, camera, save/load | ✅ done — control panel, cross-section clip, save/load, heat brush, pause/reset |

**Phase gate rule:** never start N+1 until N is stable. Over-scope is the
historical project-killer.

## Curated material set (~15)
Water, Ice, Steam, Lava, Stone, Sand, Glass, Oil, Fire, Wood, Ash, Mud, Jelly,
Mercury/molten metal, Snow.

## Definition of done — v0.1
The Phase 2 first slice is make-or-break: two coloured fluids of different
density dropped into a box, settling, swirling, layering, and throwing
Rayleigh-Taylor plumes — stable, interactive, on the 4080.
