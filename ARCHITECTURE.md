# Architecture conventions

Tooling and code-shape decisions. Deliberately short — this records rules
that are expensive to change later, not a directory tree planned in advance.
Structure grows as needed; see `PLAN.md` for the actual roadmap.

## Stack

- **Vite + TypeScript, no framework.** The UI is a canvas and a handful of
  sliders; a component tree buys nothing and would put simulation state on
  the wrong clock. Vite is here mainly for Phase 2: `import s from './x.wgsl?raw'`
  works with zero config.
- **Tests: `node --test`, no test framework.** Node 24 strips TS types
  natively, so `.test.ts` files run directly. Zero dependencies.
- **Total dependencies: TypeScript, Vite.** Plus `lil-gui` when sliders
  arrive, and `@webgpu/types` at Phase 2.

## Rule 1 — `src/core/` never touches the DOM

No `document`, no canvas, no imports from `src/viz/`. The solver takes
arrays in and writes arrays out.

Why: the validation ladder (PLAN.md §6 — lid-driven cavity vs. Ghia,
Strouhal number) then runs headlessly under `npm test` instead of needing a
browser and a screenshot. Correctness checks you can run in one second get
run; ones needing a browser don't.

## Rule 2 — kernels: one pass per function, inputs and outputs separate

Every per-cell loop is its own exported function taking input arrays and
writing to a _distinct_ output array:

```ts
computeDivergence(u, v, /* out */ div, g);
jacobiSweep(pIn, div, /* out */ pOut, g); // caller swaps pIn <-> pOut
subtractGradient(uIn, vIn, p, /* out */ uOut, vOut, g);
```

Never `for (...) u[i] = f(u[i-1], u[i+1])`.

Why, in order of how soon it bites:

1. **Testable.** A single pass over a hand-seeded field has an answer you
   can compute on paper. Buried inside `project()`, it doesn't.
2. **Debuggable.** The input survives the pass, so you can heatmap before
   and after side by side. In-place mutation destroys the evidence.
3. **Portable.** A WGSL compute shader cannot safely read and write one
   storage buffer — threads run in nondeterministic order, so a thread
   reading its neighbor can't know whether that neighbor is updated yet.
   That's what ping-pong buffers are for. CPU code already in this shape
   ports mechanically; code that isn't gets rewritten, and then PLAN.md §8
   is comparing two different programs instead of two solvers.

**The pressure solver is the deliberate exception.** Gauss-Seidel reads
partially-updated values and writes `p` in place — that is not a bug, it is
why it converges 2x faster than Jacobi, and it makes the sweep *less* code
(one array, no ping-pong, no swap). Jacobi is skipped entirely: it is both
slower and more machinery.

Consequences worth planning around:

- Adding SOR is one line: `p = (1-w)*p_old + w*p_gs`, with w ~ 1.9 at 64x64.
  That changes the convergence class from O(n^2) to O(n) iterations.
- Plain GS -> red-black GS (the Phase 2 GPU solver) is a small change: add a
  color parameter, run two passes. A cell's four neighbours in the 5-point
  stencil are always the opposite colour, so each half-sweep is fully
  parallel while staying in place.

Every *other* kernel keeps strict read/write separation.

Cost: extra buffers and a swap dance. Allocate once at startup; at 64x64
the memory is irrelevant.

## Rule 3 — allocate once, never in the frame loop

Every field buffer is created at startup and reused. No allocation inside
`step()`, including no temporary arrays inside kernels. GC pauses in a
realtime loop show up as visible stutter, and they are tedious to find later.

## Rule 4 — grid indexing lives in one file

`src/core/grid.ts` owns the array sizes and the `idxP` / `idxU` / `idxV`
helpers. Nothing else computes a flat index by hand. The staggered layout
means `u` has stride `nx + 1` while `v` has stride `nx`; one file with one
test is the difference between an obvious bug and a subtle one.

## Rule 5 — `Float64Array` via an injected constructor

The grid allocator takes the typed-array constructor as a parameter.
PLAN.md §8 compares CPU float64 against GPU float32 while _also_ changing
the solver; injecting the constructor makes float32-on-CPU a one-line run,
which separates the precision variable from the convergence variable.

## Deployment

`vite build` -> `dist/`, published to GitHub Pages by CI. `base` in
`vite.config.ts` must match the repo name or every asset 404s on the
project URL. Pages is HTTPS, which WebGPU requires (secure context only).
