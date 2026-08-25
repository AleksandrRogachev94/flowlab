# Fluid Simulation Project — Implementation Plan (Final)

_Revised August 2026. Supersedes the earlier Python + JS two-language plan._

---

## 1. Goal

Build real, working understanding of the math and numerical methods behind
incompressible fluid simulation, and turn that understanding into two
concrete artifacts:

1. **A first-principles reference implementation** (JavaScript, CPU, coarse
   grid) — proof that the physics and discretization are genuinely
   understood, not copied.
2. **A real-time, shareable, tunable browser demo** (JavaScript + WebGPU) —
   the thing that actually gets shown to other people.

Constraints driving every decision below: keep the total time investment
lean (weeks, not months), prioritize understanding over exhaustive coverage,
and make sure nothing gets built twice.

---

## 2. Language decision: all JavaScript

**Python is dropped.** The original plan justified a Python/NumPy reference
implementation partly on "full float64 precision — closest to correct."
**That rationale was wrong:** JavaScript's `Number` is IEEE-754
double-precision by default. Float64 is what you get in JS unless you
deliberately opt into `Float32Array`. So the precision half of the
CPU-vs-GPU comparison (§8) survives the language change untouched.

What Python would actually have bought:

| Python advantage                     | Verdict                                                                                                                           |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| `scipy.sparse.linalg.cg` in one line | **Counterproductive.** Calling a black-box CG is _less_ "genuinely understood, not copied" than writing ~30 lines of CG yourself. |
| NumPy vectorized stencil ops         | Real convenience, but explicit JS loops make every neighbor access visible — arguably better for learning.                        |
| matplotlib / Jupyter inspect loop    | Replaced by a canvas heatmap utility (§5, Step 0) that gets reused in Phase 2 anyway.                                             |

Costs accepted: stencil loops are more verbose in JS than in NumPy, and
there's no `ndarray` ecosystem worth leaning on. Both acceptable at the
grid sizes involved.

**What this preserves:** the §8 comparison still isolates the same
variables (solver, iteration count, precision), now without a
language-translation step confounding the results. And the Phase 2 claim
of "a translation pass, not a redesign" becomes _more_ honest — JS → WGSL
is still a genuine structural jump (storage buffers, workgroups,
constrained control flow), so that exercise isn't lost, it's just done
once instead of twice.

**WebGPU viability — verified, not assumed.** As of August 2026 WebGPU
ships enabled-by-default in Chrome/Edge (since 113), Firefox (Windows
141+, Apple Silicon macOS 145+), and Safari (macOS Tahoe 26 / iOS 26).
Remaining gaps: Firefox on Linux and Android, and Apple platforms gated on
OS version. For a shareable desktop demo this is now a mainstream target,
not a bleeding-edge gamble. Feature-detect `navigator.gpu` and show a
graceful "unsupported browser" message rather than building a WebGL
fallback — a fallback doubles the shader work for a shrinking minority.

---

## 3. What the demo simulates

One solver, three presets, because they share almost all their machinery:

| Preset                              | What it shows                                                                                                           | New ingredient vs. base solver                      |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Kármán vortex street** (headline) | Flow around an obstacle in a channel; steady flow → paired vortices → alternating shedding as Reynolds number increases | Interior solid cells (obstacle mask)                |
| **Rising buoyant plume**            | Smoke/heat rising and billowing                                                                                         | Buoyancy force + advected temperature/density field |
| **Free draw**                       | User paints obstacles and injects dye with the mouse                                                                    | Same solid-cell mechanism, interactive              |

**Why this over smoke alone:** Reynolds number is a _real physical
parameter_ producing qualitatively different, well-documented flow regimes
as you tune it — a far more meaningful playground slider than purely
artistic knobs. It doubles as a rigorous validation case (vortex shedding
frequency / Strouhal number), on top of the lid-driven-cavity benchmark
(Ghia et al.) already familiar from Barba's course. And it costs almost
nothing extra: obstacle cells are a direct, small extension of the
solid-wall boundary conditions already required reading (Ch. 5).

**Why not water/liquid:** free-surface tracking (level sets or particles)
is a genuinely different subsystem, not a small addition — deliberately
deferred (§7).

---

## 4. Physics and math foundation (already built)

This part is _done_; every implementation choice below traces back to it.

- **Vector calculus**: gradient, divergence, curl, Laplacian, and the
  Jacobian as the unifying object each derives from (trace = divergence;
  antisymmetric part = curl).
- **Differential identities**: curl(∇f)≡0 and ∇·(∇×u⃗)≡0 — why subtracting
  a pressure gradient never introduces spurious rotation. Helmholtz
  decomposition — the structural guarantee that pressure projection always
  works.
- **Boundary integrals**: divergence theorem, Stokes' theorem — the direct
  justification for solid-wall boundary conditions and for deriving the
  momentum equation (Appendix B).
- **The Poisson pressure equation**: derived by taking the divergence of
  the momentum equation and using ∇·u⃗=0. **Pressure is a Lagrange
  multiplier, not a state variable** — in the incompressible limit there
  is no ∂p/∂t and no equation of state; p is instantaneously whatever it
  must be to keep ∇·u⃗=0. "Solve for p" and "enforce incompressibility"
  are the same task described two ways.
- **Why the solve is global**: incompressibility implies infinite sound
  speed. Advection is hyperbolic (finite domain of dependence → explicit
  per-cell update); the pressure equation is elliptic (boundary conditions
  felt instantly everywhere → global coupled solve, every frame, forever).
- **Electrostatic analogy**: ∇²φ = −ρ/ε₀ and ∇²p = ∇·u⃗\* are the same
  equation. u⃗ ↔ E, p ↔ φ, and the "charge" sourcing pressure is the
  divergence defect of the tentative velocity field. Gauss's law is the
  same physics one derivation step earlier (first-order in E, second-order
  once E = −∇φ is substituted).
- **Finite differences & Taylor series**: truncation error, order of
  accuracy. Note: the MAC divergence stencil is **second-order**, not
  first — it's a centered difference with half-width Δx/2, and symmetric
  stencils cancel the even-power Taylor terms regardless of stencil width.
- **Time integration**: forward Euler vs. RK2/RK4; splitting error is
  unavoidably first-order regardless of per-term accuracy.
- **Numerical stability**: von Neumann analysis — why forward Euler +
  central-difference advection is unconditionally unstable, while forward
  Euler + upwind is conditionally stable under CFL. Verified numerically.
- **Numerical diffusion**: modified-equation analysis — upwind schemes
  secretly solve advection _plus_ implicit diffusion; why numerical
  methods look viscous with zero physical viscosity.
- **The material derivative**: Dq/Dt = ∂q/∂t + u⃗·∇q — the bridge between
  Lagrangian physics and Eulerian implementation; the exact justification
  for splitting into advect → forces → project.
- **The MAC (staggered) grid**: pressure at cell centers, velocity at cell
  faces. Fixes the checkerboard null-space problem of collocated grids
  (verified numerically), while achieving full second-order accuracy in
  both directions simultaneously.
- **Semi-Lagrangian advection**: why it's unconditionally stable, and its
  real costs (interpolation dissipation, non-conservation, boundary
  handling), honestly weighed against finite-difference alternatives.
- **Assembling Ap = b**: one equation per fluid cell → one matrix row;
  columns are neighbor couplings. A is sparse (≤5 nonzeros/row in 2D) and
  **symmetric**, because the coupling between two cells comes from the one
  shared face — which is exactly the precondition CG requires.

---

## 5. Reading plan

Barba's 12 Steps: **done.** Used for hands-on numerical intuition and
stability lessons, not as the final solver design.

| Chapter                             | Verdict   | Why                                                                                                     |
| ----------------------------------- | --------- | ------------------------------------------------------------------------------------------------------- |
| 1. Equations of Fluids              | Fast skim | Already understood                                                                                      |
| 2. Overview of Numerical Simulation | **Read**  | Formalizes splitting + introduces MAC grid                                                              |
| 3. Advection Algorithms             | **Read**  | Semi-Lagrangian in full: RK2 backtrace, interpolation, boundary handling, MacCormack/BFECC upgrade path |
| 4. Level Set Geometry               | Skip      | Water-only; deferred                                                                                    |
| 5. Making Fluids Incompressible     | **Read**  | The actual pressure solve — least already-covered chapter                                               |
| 6. Smoke                            | **Read**  | Buoyancy + advected temperature/density — short, direct payoff                                          |
| 7. Particle Methods                 | Skip      | FLIP machinery; deferred with water                                                                     |
| 8. Water                            | Skip      | Deferred (Phase 3)                                                                                      |
| 9. Fire                             | Skip      | Out of scope                                                                                            |
| 10. Viscous Fluids                  | Skip      | Numerical dissipation already exceeds real air viscosity at target scale                                |
| 11. Turbulence                      | **Read**  | Vorticity confinement — main visual-quality lever, cheap                                                |
| 12–15                               | Skip      | Niche                                                                                                   |

**Supplementary reading, with explicit scope limits:**

- **GPU Gems Ch. 38 (Harris, 2004)** — take the _math_ (operator
  composition S = P∘F∘D∘A; concrete Jacobi iteration counts: 40–80 for
  pressure, error visibly bad below ~20; Neumann pressure BC via ghost-cell
  mirroring). **Do not take the architecture.** It's Cg fragment programs,
  `samplerRECT`, and copy-to-texture hacks that exist only because compute
  shaders didn't. It also uses a _collocated_ grid and lists staggered as an
  unadopted "extension" — the opposite of this project's choice.
- **Stam, "Real-Time Fluid Dynamics for Games" (GDC 2003)** — the origin of
  the stable-fluids splitting.
- **kishimisu/WebGPU-Fluid-Simulation** — the actual architectural
  reference for Phase 2 (WGSL compute shaders, closest match to this stack).
- **Müller, "How to write an Eulerian fluid simulator with 200 lines"** —
  worth understanding as _the same math in disguise_. His per-cell update
  (`d = u[i+1]-u[i]+v[j+1]-v[j]`, then push each face by ±d/4) is exactly
  one Gauss-Seidel sweep of Bridson's Poisson equation with pressure never
  materialized: solving `4p = -d` for a single cell against zeroed
  neighbors gives Δp = −d/4, and applying Bridson's velocity update on the
  four faces reproduces his four lines identically. It fuses "solve" and
  "subtract gradient" into one pass. **Caveat:** the flat `/4` holds only
  for interior cells with four fluid neighbors — the general rule is
  divide by the number of non-solid neighbors, matching Bridson's
  coefficient rule. Same convergence rate as GS, and same
  non-parallelizability, so it needs the same red-black reordering on GPU.
- https://www.youtube.com/watch?v=Q78wvrQ9xsU&t=1818s

---

## 6. Implementation phases

### Phase 0 — Barba's 12 Steps ✅ Done

### Phase 1 — JS CPU reference implementation (coarse grid, 64×64 → 128×128)

**Data structures.** MAC grid: `p` at cell centers, `u`/`v` at cell faces,
all flat `Float64Array` with manual indexing. A `label` array per cell
(FLUID / AIR / SOLID) drives every boundary decision, exactly as in
Bridson's pseudocode.

**Per-frame algorithm** (this is the whole thing; everything else is detail):

1. **Advect** velocity (and temperature/dye) — semi-Lagrangian, RK2
   backtrace, bilinear interpolation
2. **Apply forces** — buoyancy from the temperature field, vorticity
   confinement
3. **Compute divergence** of the resulting velocity
4. **Solve the Poisson equation** for pressure, with label-driven boundary
   edits: air neighbor → drop the term (p = 0); solid neighbor → drop the
   term _and_ decrement the diagonal, adding the (u_face − u_solid)
   correction to the rhs. **Diagonal coefficient = number of non-solid
   neighbors.**
5. **Subtract the pressure gradient**

**Build order — strictly incremental, each step verified before the next:**

- **Step 0: the debug heatmap.** Before any solver code, write a
  canvas utility that renders any scalar field as a color map. This is the
  `plt.imshow` replacement, it's how every subsequent step gets verified,
  and it gets reused verbatim in Phase 2 where WGSL has no breakpoints.
  Building it first is not a detour.
- **Step 1: projection in a plain box.** No obstacles, no buoyancy, no
  advection subtleties. Seed a divergent velocity field, project it, and
  check that the divergence field goes to ~0. If this isn't right, nothing
  downstream can be.
- **Step 2: semi-Lagrangian advection.** Add dye, confirm it moves
  sensibly and doesn't blow up at large timesteps.
- **Step 3: solid obstacle cells.** Unlocks the Kármán preset. Verify the
  diagonal-coefficient rule by hand on a small grid near a wall.
- **Step 4: buoyancy + advected temperature.** Unlocks the plume preset.
- **Step 5: vorticity confinement.** Visual-quality pass — last, because
  it's a knob on an already-correct solver, not a correctness feature.
  **Reconsider after Step 6** — it exists to replace detail that advection
  destroyed, and Step 6 destroys much less of it.
- **Step 6 (added, done): MacCormack advection.** The plan's original gap.
  Two extra semi-Lagrangian passes estimate the first pass's own error and
  subtract it, cancelling the leading _dissipative_ term; a limiter clamping
  each result to its donor stencil keeps it unconditionally stable. Measured
  against plain semi-Lagrangian on the same seed: cone peak after one
  revolution 0.30 -> 0.81 at 64^2, kinetic energy kept at t=10 2% -> 9%,
  enstrophy 0.3% -> 1.8%, for ~2.5x the advection cost (~20% of a frame,
  since the pressure solve dominates). It sits in the same slot as FLIP
  below — attacking the _cause_ of numerical dissipation — but is a 60-line
  change rather than a new subsystem, which is why deferring it with FLIP
  was the wrong call. Not conservative: the limiter clips the undershoots
  that would balance the overshoots, so dye MASS grows a few percent per
  revolution (0.4% without the limiter, at the price of negative dye).

**Poisson solver, in this order:**

1. **Jacobi first.** Simplest to verify by hand on a tiny grid, and it's
   what the GPU port will use — so debugging it now pays twice.
2. **Then hand-written conjugate gradient.** ~30 lines, matrix-free
   (the matvec is just the stencil; `A` is never materialized). This is the
   "closest to correct" reference solver, and writing it rather than calling
   it is the point. CG's iteration count scales like O(√N) vs. Gauss-Seidel's
   O(N) as resolution grows.

**Validation ladder:**

1. Eyeball — does dye swirl plausibly, does post-projection divergence
   drop to ~0
2. **Lid-driven cavity vs. Ghia et al.** tabulated velocity profiles — the
   real correctness gate
3. **Strouhal number** from shedding frequency behind the obstacle, once
   Step 3 exists

### Phase 2 — WebGPU port

Same five steps, each its own WGSL compute shader, chained with ping-pong
storage buffers (two per field; a shader can't safely read and write the
same buffer):

- `advect.wgsl`, `forces.wgsl`, `divergence.wgsl`, `pressure.wgsl`,
  `subtractGradient.wgsl`

Fields as storage buffers: `u`, `v`, `p`, `divergence`, `temperature`,
`dye`, `obstacleMask`. Workgroup size 8×8 as a starting point, tuned later.
The pressure step is the only one that loops N times per frame — start at
40–80 iterations per the GPU Gems figure, then tune down against the frame
budget and watch for the visible-error threshold around 20.

**Solver:** Jacobi or **red-black Gauss-Seidel** — parallelizable via
checkerboard passes. Plain sequential GS does not parallelize (its update
chain is inherently ordered), and CG needs a global dot-product reduction
every iteration, which fights the ping-pong architecture. This is the
deliberate accuracy-for-throughput trade that Phase 3 measures.

**UI:** sliders for Reynolds number, buoyancy strength, vorticity
confinement strength (driving uniform buffers); mouse-to-grid mapping for
obstacle painting and dye injection.

**Expect debugging friction.** No breakpoints inside WGSL — verification
means reading buffers back and rendering them through the Step 0 heatmap.
Budget for this rather than being surprised by it.

**No WASM/C:** the real-time solver is entirely GPU-parallel; there's no
meaningful CPU bottleneck for WASM to accelerate.

### Phase 3 — optional, deferred

FLIP + level sets for a splashy liquid mode. Explicitly not core scope (§7).

---

## 7. Deliberate scope cuts, with reasoning

- **Python — dropped.** See §2. Its stated float64 advantage didn't exist,
  and its main real advantage (library solvers) works against the project's
  own "understood, not copied" goal.
- **WebGL fallback — dropped.** WebGPU support is now broad enough (§2)
  that a fallback would double the shader work for a shrinking minority.
  Feature-detect and message instead.
- **Water / free surfaces (Ch. 4, 7, 8) — deferred.** Needs an entirely
  separate subsystem (level sets or particles) to track a moving
  fluid/air boundary — a real scope increase, not a small tweak.
- **Real viscosity (Ch. 10) — dropped.** Real air viscosity is tiny;
  semi-Lagrangian's own numerical dissipation already exceeds it at
  graphics scale. Standard industry practice, not a reluctant shortcut.
  (Worth knowing: implicit viscous diffusion is _also_ a Poisson-type
  solve — pressure isn't the only one. Just not needed here.)
- **FLIP over vorticity confinement — not now.** FLIP is more physically
  grounded (reduces the cause of numerical dissipation rather than patching
  the symptom), but particle-grid scatter/gather is a fundamentally harder
  parallel problem than ping-ponged grid passes. Its biggest payoff
  (splashes, topology change) is a liquid feature anyway — best paired with
  Phase 3, where the complexity buys two improvements for one cost.
- **Higher-order splitting (Strang) — not needed.** Splitting error is
  first-order regardless, but small relative to the approximations already
  accepted (dropped viscosity, semi-Lagrangian dissipation, large
  interactive timesteps). Not the dominant error source.
- **Fire, shallow water, ocean, vortex methods, solid coupling — out of
  scope.** Different phenomena or specializations with no bearing on the
  project's goals.

---

## 8. The comparison (the point of building both versions)

Run identical initial conditions through both implementations:

- **JS CPU reference**: hand-written conjugate gradient, run to tight
  convergence, float64 — closest to "correct."
- **WebGPU real-time**: Jacobi/red-black with a bounded per-frame
  iteration count, float32 — trading accuracy for a guaranteed frame budget.

Expected, checkable differences:

- The GPU version should show visibly more numerical dissipation (softer,
  less swirly detail) with vorticity confinement disabled
- Timing comparison (CPU CG vs. GPU parallel relaxation) gives a concrete
  performance number
- Both should agree on **bulk flow behavior** — vortex shedding onset, wake
  shape, Strouhal number — if both are correct. This agreement is the real
  test of whether the port was done right.

Because both sides are now JavaScript and both start from float64, any
divergence between them is attributable to the two variables actually
under study (solver convergence and float32 precision) rather than to a
language or library difference.
