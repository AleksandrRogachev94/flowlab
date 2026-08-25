# The WebGPU port

Two pieces of the step run on the device: the **pressure solve** (§1-§5) and
**advection and dye** (§8). Everything else is still CPU. This document covers
why those pieces were chosen, how they are put together, and the one thing
that went wrong.

---

## 1. The seam

`PressureSolver` (`src/core/pressureSolver.ts`) is the only interface between
the two worlds:

```ts
solve(g, p, div, label, scale, iterations, omega, tol): number | Promise<number>
```

The pressure solve is the right first port because it is a **pure function of
`(div, label, scale)` that returns `p`**. Nothing else in `step()` can tell
which implementation ran. That is what makes the CPU/GPU comparison a
controlled experiment rather than a comparison of two whole programs.

Three implementations sit behind it, and pressing **G** cycles them in this
order on purpose:

| solver      | ordering      | precision | changes vs. the previous one   |
| ----------- | ------------- | --------- | ------------------------------ |
| `cpu-sor`   | lexicographic | f64       | — (the reference)              |
| `cpu-rbsor` | red-black     | f64       | the sweep **ordering**         |
| `gpu-rbsor` | red-black     | f32       | the **hardware and precision** |

One variable per step. When the GPU picture looks wrong, the middle entry is
what tells you whether the algorithm or the WGSL is at fault. Without it you
only have one comparison and it confounds both.

`solve` may return a promise because WebGPU has **no synchronous readback** —
getting a buffer off the device is `await buffer.mapAsync()`, full stop. That
single fact is why `Simulation.step()` is `async`. On the CPU path no promise
is ever created; `await` on a plain number costs one microtask tick.

---

## 2. WebGPU in one page

Six objects, and that is genuinely all of them for a compute-only port:

- **Adapter / Device** — the driver handle. `navigator.gpu.requestAdapter()`
  then `adapter.requestDevice()`. Everything else is created from the device.
- **Buffer** — a flat block of device memory. Created with a `usage` mask that
  must list every way you intend to use it. `MAP_READ` cannot be combined with
  `STORAGE`, which is why results are copied into a second, read-only buffer
  instead of being mapped in place.
- **Shader module** — the compiled WGSL. Ours is `redBlack.wgsl`, imported as
  a string via Vite's `?raw`.
- **Pipeline** — a shader module plus an entry point. `layout: 'auto'` lets
  WebGPU infer the binding layout from the WGSL, which is one less thing to
  keep in sync by hand.
- **Bind group** — which actual buffers fill the shader's `@binding` slots.
- **Command encoder** — records `dispatchWorkgroups` calls into a command
  buffer, which `queue.submit()` hands to the GPU.

The mental model that matters: **`submit()` does not run anything.** It queues
work and returns immediately. The GPU only finishes when something forces it
to — here, the `mapAsync` readback. That is why the profiler reports `wait`
rather than "compute time", and why a separate timestamp query is needed to
learn what the device really spent.

### The dispatch grid

`@compute @workgroup_size(8, 8)` means every workgroup is 64 threads laid out
8×8. `dispatchWorkgroups(ceil(nx/8), ceil(ny/8))` covers the grid. There is no
such thing as a partial workgroup, so the last row and column overshoot and
**every WGSL kernel needs a bounds guard as its first statement**:

```wgsl
if (i >= params.nx || j >= params.ny) { return; }
```

### Uniforms are padded

A uniform struct must be a multiple of 16 bytes. `redBlack.wgsl` declares
`pad0/pad1/pad2` explicitly rather than trusting the layout rules to round up
silently — a mismatch between the TypeScript `ArrayBuffer` and the WGSL struct
produces a plausible-looking wrong answer, not an error.

---

## 3. Why red-black, and not plain Gauss-Seidel

The CPU sweep is fast **because** it reads partially-updated neighbours: cell
`(i,j)` uses the already-new `p[i-1,j]` written earlier in the same sweep.
That is a serial dependency chain across the whole grid. On a GPU, where
thousands of threads run in an order nobody controls, "already updated" has no
meaning at all.

Colouring the grid like a checkerboard fixes it exactly, with no
approximation. In the 5-point stencil a cell's four neighbours are **always**
the opposite colour:

```
     B          colour = (i + j) & 1
  B  R  B
     B
```

So a pass over the red cells reads only black values and writes only red ones.
Nothing a red thread reads can be changed by another red thread. One sweep =
one red dispatch then one black dispatch, and the black pass sees the
brand-new red values — still genuine Gauss-Seidel, not Jacobi.

Two details that make this cheap:

- **One buffer, `read_write`, no ping-pong.** The colouring is what makes
  reading the array you write safe here.
- **All 192 dispatches go into one command encoder and one submit.** WebGPU
  orders dispatches within a compute pass and inserts the memory barriers
  between them, so "red sees the previous black" is guaranteed with no
  explicit synchronisation, and the loop costs one CPU→GPU handoff, not 192.

---

## 4. The omega trap

This is the bug that made the velocity field oscillate, and it is worth
understanding because it has nothing to do with WebGPU.

`optimalOmega()` computes the classical SOR relaxation factor, ~1.97 at these
grid sizes. That number minimises the **asymptotic** convergence rate — the
factor the error shrinks by _once the iteration has settled_, which for SOR on
an N-wide grid takes O(N) sweeps. The browser runs 96 sweeps on a 640-wide
grid. It never gets there.

In that regime what matters is the **transient**, and SOR near ω = 2 has a
violent one: it _amplifies_ the residual for tens of sweeps before decay sets
in. Stopping inside that hump is worse than never over-relaxing at all.

Measured on karman 240×180, 36 sweeps, residual RMS after 150 steps:

| ω             | 1.97   | 1.80   | 1.70   | 1.60       | 1.50   |
| ------------- | ------ | ------ | ------ | ---------- | ------ |
| lexicographic | 9.0e-2 | 7.9e-3 | 1.1e-2 | **8.4e-3** | 1.2e-2 |
| red-black     | 1.8e+0 | 5.6e-2 | 4.4e-2 | **3.8e-2** | 3.8e-2 |

Both orderings are hurt, but **red-black is hurt 48×**, and that is what
turned a numerical detail into a visible instability. The reason is the same
property that made red-black parallelisable: a lexicographic sweep carries
information across the whole domain in one pass, because of the very
dependency chain the GPU cannot have. Red-black moves information exactly one
cell per half-sweep. It has no long-range transport to hide the transient
behind.

So `main.ts` passes **ω = 1.6** — near the floor for _both_ orderings, which
is what makes it the right shared value: **G** must change the solver and
nothing else. `defaultParams` keeps `optimalOmega()` for headless reference
runs, where the tolerance really does iterate to convergence and the
asymptotic rate really is what matters.

The general lesson: **with a budget of M sweeps on an N-wide grid, if M ≪ N
you are not a solver, you are a smoother.** Smoothers want ω near 1, not near 2. (This is also exactly the setting multigrid is built for — see §6.)

### Should ω depend on the grid size?

`optimalOmega()` did, and that is what makes the question worth answering
rather than assuming. Measured: red-black, sweep budget `0.15 × N` as the code
actually uses, residual divided by the incoming `u*` divergence so the numbers
compare across resolutions.

| grid    | sweeps | ω=1.2     | ω=1.4     | ω=1.6     | ω=1.8 |
| ------- | ------ | --------- | --------- | --------- | ----- |
| 160×120 | 24     | 0.037     | 0.033     | **0.030** | 0.033 |
| 240×180 | 36     | 0.061     | **0.060** | 0.062     | 0.067 |
| 320×240 | 48     | 0.074     | 0.072     | **0.070** | 0.097 |
| 400×300 | 60     | **0.122** | 0.125     | 0.123     | 0.147 |

**It does not drift.** The optimum sits at 1.4–1.6 everywhere, and the curve is
flat to within 5% across 1.2–1.6 at every size. A constant is correct.

The reason is not a coincidence, and it is worth holding onto: **the budget
already scales with the grid.** `PRESSURE_ITERS = 0.15 × N`, so `M/N` is
constant, so "how far into the transient do we stop" is constant, so the best ω
is constant. Make the budget a fixed number of sweeps instead and ω would have
to fall as the grid grew — the two knobs are coupled, and only one of them
needs to carry the N-dependence.

What the table _also_ shows is the wall this is heading for: the best
achievable residual degrades steadily with resolution — 0.030 → 0.122 for 2.5×
the grid, at 2.5× the sweep count and ~6× the work. That is the O(N) sweep
count in `PRESSURE_ITERS` being outrun by the problem. No choice of ω fixes it;
§6 does.

Precision, by contrast, turned out to be a non-issue: running the same
red-black solver over f32 storage on the CPU reproduces the f64 result to
every printed digit, and the on-device result agrees with the CPU f64 solver
to 1.7e-7 on a peak of 1.4e-1 — four orders below the solver's own residual.

---

## 5. What the numbers say

Medians over 60 frames, Apple M-series, karman scene. "CPU solve" is the
lexicographic SOR this replaces.

| grid    | sweeps | CPU solve | upload | wait | device | speedup |
| ------- | ------ | --------- | ------ | ---- | ------ | ------- |
| 320×180 | 48     | 28.6      | 0.08   | 3.5  | 2.9    | ~8×     |
| 480×270 | 72     | 96.5      | 0.20   | 5.5  | 4.8    | ~17×    |
| 640×360 | 96     | 228.8     | 0.32   | 7.7  | 6.9    | ~28×    |

Two things to read out of that:

1. **`wait − device` is 0.6–0.8 ms at every size.** The round trip is cheap;
   the fear that readback would eat the win was unfounded.
2. **The speedup grows with resolution**, because SOR needs O(N) sweeps and
   the GPU absorbs the extra sweeps far better than the extra cells.

What it does **not** do is make the frame fast, and the profiler (press **P**)
says why. Porting one phase caps the gain at Amdahl's law: at 320×180 the
pressure solve was 40% of the step, so removing it entirely can only give
1.7×. Advection and dye are the other 58%.

If the pressure `device` time looks higher than you expected, the kernel is
**memory-bandwidth bound, not compute bound**. Each dispatch streams `p`,
`div` and `label` — about 4 MB per dispatch at 640×480, ~800 MB per frame
across 192 dispatches. At ~100 GB/s that is most of the measured time. Faster
arithmetic would buy nothing; fewer sweeps or fewer bytes per sweep would.

---

## 6. Getting to 1024 and beyond: which solver

SOR needs **O(N) sweeps** because it moves information one cell per sweep, and
each sweep costs O(N²) cells. Total work per solve is O(N³). Going 640 → 1920
is 3× the width and therefore **~27× the work** — the GPU does not have 27× in
hand. The solver has to change, not just the hardware.

The two candidates, honestly compared:

### Conjugate gradient (PCG)

- **Iterations:** O(√N) unpreconditioned — roughly 30–45 at N = 1024, versus
  ~150 for SOR. Real, but still grows.
- **CPU cost to write:** genuinely small. ~30 lines, matrix-free (the matvec
  _is_ the existing stencil; `A` is never built). PLAN.md already wants this as
  the reference solver, and it is the better first move for that reason.
- **GPU cost to write:** this is where it gets awkward. Every iteration needs
  **two global dot products**, and a dot product is a tree reduction — extra
  kernels, and a scalar that the next step depends on. You can keep the scalars
  in a buffer and never read back, but you need ~5 shaders (matvec, three
  axpys, reduction) plus a scalar-update kernel, and the ordering is fiddly.
- **The good preconditioners do not port.** MIC(0), the standard choice in
  Bridson, is a sequential triangular solve — the exact thing a GPU cannot do,
  for the exact reason plain Gauss-Seidel could not. On the GPU you are left
  with Jacobi preconditioning, which barely dents the iteration count.

### Geometric multigrid

- **Iterations:** **O(1)**. Three to five V-cycles at any resolution. This is
  the property that actually solves the 1920 problem rather than deferring it.
- **Why it works:** relaxation kills high-frequency error fast and smooth error
  slowly. Multigrid solves the smooth part on a coarse grid where "one cell" is
  a much longer distance, then interpolates the correction back. Every level
  costs a quarter of the one above, so a whole V-cycle is ~4/3 of one fine
  sweep.
- **Cost to write:** two new shaders — `restrict.wgsl` (fine residual → coarse
  RHS, a 4-cell average) and `prolong.wgsl` (coarse correction → fine,
  bilinear). Each is ~20 lines and far simpler than the solve. Plus host-side
  level management: `log2(N)` sets of buffers, allocated once.
- **The smoother is already written.** Red-black GS at ω ≈ 1 is _the_ standard
  multigrid smoother, which is exactly the regime §4 pushed us into. The shader
  in this repo is a multigrid component as it stands.
- **Where the bugs will be:** coarsening the `label` field. A cylinder and a
  channel wall have to be represented sensibly on a 32×24 grid, and Neumann
  boundaries on coarse levels are the classic source of a V-cycle that quietly
  stops converging. Budget the debugging there, not in the shaders.

### Recommendation

**Multigrid**, and the reasoning is not that it is easier — per line of code it
is not — but that PCG buys a _smaller exponent_ while multigrid buys the
_right_ one. At 1920 that is the difference between "slower but fine" and
"done".

Two things make it much less daunting than it sounds here: the smoother
already exists and is tested, and the CPU mirror pattern (`pressureRedBlack.ts`)
works just as well for a V-cycle — write it in TypeScript first, verify the
residual really drops ~10× per cycle on a small grid, and only then port it.
That is the same ladder that caught the ω bug in twenty minutes.

Write the ~30-line CPU CG anyway. It costs an afternoon, PLAN.md §8 wants it as
the "closest to correct" reference, and having a solver that converges to
machine precision is what lets you tell a multigrid bug from a multigrid
tuning problem.

## 7. Order of work

1. ~~**Advection and dye to the device.**~~ Done — §8.
2. **Keep the fields resident.** Three round trips per frame is two more than
   necessary; see the end of §8 for what has to move first.
3. **CPU conjugate gradient**, as the converged reference (PLAN.md §8).
4. **Multigrid**, CPU first, then the two extra shaders.

---

## 8. Step 2: advection and dye

`advect.wgsl` is the counterpart of `core/advect.ts`, behind an `Advector`
interface built on the same terms as `PressureSolver`: one CPU implementation,
one GPU implementation, swappable at runtime, and `Simulation.step()` cannot
tell which one ran.

It is one interface with **two methods**, `velocity()` and `dye()`, because
the two calls sit on opposite sides of the pressure solve — velocity advects
`u^n`, dye rides the projected `u^{n+1}`. They are otherwise the same kernel,
which is why one object owns both.

### One binding layout, six entry points

Every pass has the same shape — read a carrier velocity, read a source field,
write one output — so the bindings never change. Only the entry point and the
bind group do:

| binding      | role                                                 |
| ------------ | ---------------------------------------------------- |
| `carU, carV` | the **carrier**: the velocity the backtrace follows  |
| `src`        | the field **sampled** at the backtraced point        |
| `orig`       | the field before the step; only `correct_*` reads it |
| `dst`        | the output — never aliases any of the above          |
| `label`      | Fluid/Air/Solid, widened to `u32` on upload          |

That is what pays for an explicit `GPUBindGroupLayout` instead of the
`layout: 'auto'` §2 recommends. Auto-layouts are distinct objects even when
identical, so a bind group built for one pipeline is rejected by the next —
and the whole point here is six pipelines sharing one layout and six bind
groups. Self-advection binds `carU = src = uIn`; the correction binds
`src = uHat`, `orig = uIn`. Nothing else moves.

### MacCormack in two dispatches, not three

The CPU does forward, reverse, then correction. The reverse pass's output is
read **only at its own index** by the correction, so the two fuse into one
kernel with no intermediate buffer:

```
qBar      = A_rev(qHat)             traced forward from this cell
corrected = qHat + (q^n - qBar)/2
dst       = clamp(corrected)        to the forward pass's donor stencil
```

Same arithmetic, one less pass over memory — and this kernel is
memory-bandwidth bound, so the pass it saves is the thing that costs.

### Three dye channels in one dispatch

The backtrace is the expensive part and it does not depend on the channel. The
CPU advects the channels one after another and pays for it three times; the
shader backtraces once and gathers three values, so three channels cost barely
more than one. The channels share a single buffer laid out end to end, which
is also what keeps the binding count at six.

### What the numbers say

Same machine, same scene (karman 640×480, MacCormack, GPU pressure solver in
both columns), from the in-app profiler — press **P**. Only the advector
changed between the two runs:

| phase      | CPU advector | GPU advector | speedup |
| ---------- | ------------ | ------------ | ------- |
| advect     | 83.4         | **5.7**      | ~15×    |
| dye        | 135.8        | **3.5**      | ~39×    |
| step (all) | 243.1        | **33.8**     | ~7×     |
| frame      | 250.8        | **48.9**     | ~5×     |
| fps        | 4.0          | **20.5**     |         |

Those two runs are headless Chrome, whose CPU side is roughly half the speed
of a windowed browser — in a window the same two phases read 44.4 and 66.5 ms.
The RATIO is what transfers between the two; re-measure the absolutes.

Two things worth reading out of it:

1. **Dye gains more than velocity** (39× vs 15×), and that is the
   one-dispatch, three-channel kernel rather than anything about the hardware.
2. **Most of the remaining 5.7 ms is not the GPU.** A microbenchmark of the
   phase alone — f64 → f32 conversion, upload, dispatch, readback, nothing
   else — is 1.8 ms for velocity and 2.8 ms for dye at this size, of which
   0.5–0.7 ms is upload. The rest of the profiler's `advect` row is host work
   that never moved: `maxFaceSpeed()` scans every face, and the `dye` row
   still carries the decay and the source stamp.

**The pressure solve is 59% of the step again**, which is the position §6 was
written for.

### The three round trips, and what removes them

`u` and `v` are uploaded twice per step, because `subtractGradient` runs on the
host in between and the device's copy is stale by the time dye advects. Only
`label` is genuinely shared: nothing between the two calls can change it, so
`dye()` reuses the buffer `velocity()` filled. That is an ordering contract
written into the `Advector` interface, not a hopeful assumption.

Removing the rest is not a matter of tightening the plumbing. The uploads
exist because `computeDivergence`, `subtractGradient`, `applyOutflow` and the
dye source still run on the host — each is a ~20-line kernel, and porting them
is what lets the fields stay resident and collapses three stalls into one. The
shaders and bind groups are unchanged by that, which is why this was worth
doing first.

### The correctness gate

`advect.gputest.ts` runs the shader on real hardware and diffs both schemes
against `CpuAdvector`: 1.4e-7 relative on `u`, 1.2e-6 on dye. It exists
because the MAC grid has three different extents — `(nx+1)*ny`, `nx*(ny+1)`,
`nx*ny`, each with its own half-cell offset and its own bounds guard — and a
swapped stride produces a picture that still flows, just half a cell wrong.
`npm run test:gpu` runs it, serialised: two GPU test files cannot share the
harness's fixed debug port.
