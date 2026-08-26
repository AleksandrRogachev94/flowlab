# Architecture conventions

Tooling and code-shape decisions. Deliberately short — this records rules
that are expensive to change later, not a directory tree planned in advance.
Structure grows as needed; see `PLAN.md` for the actual roadmap.

## Stack

- **Vite + TypeScript, no framework.** The UI is a full-bleed canvas with a
  scene switcher and one settings panel over it (`src/ui/controls.ts`, built
  from plain DOM); a component tree buys nothing and would put simulation
  state on the wrong clock. Vite is here mainly for Phase 2:
  `import s from './x.wgsl?raw'` works with zero config.
- **Tests: `node --test`, no test framework.** Node 24 strips TS types
  natively, so `.test.ts` files run directly. Zero dependencies.
  `*.gputest.ts` is the second tier — see Rule 7 — run by `npm run test:gpu`
  and kept out of `npm test` so the fast suite stays a one-second run with no
  browser.
- **Total dependencies: TypeScript, Vite, `@webgpu/types`.** The settings
  panel is ~250 lines of DOM built from option lists, which is less than
  wiring a widget library to the same lists would have been.

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
why it converges 2x faster than Jacobi, and it makes the sweep _less_ code
(one array, no ping-pong, no swap). Jacobi is skipped entirely: it is both
slower and more machinery.

Consequences worth planning around:

- Adding SOR is one line: `p = (1-w)*p_old + w*p_gs`, with w ~ 1.9 at 64x64.
  That changes the convergence class from O(n^2) to O(n) iterations.
- Plain GS -> red-black GS (the Phase 2 GPU solver) is a small change: add a
  color parameter, run two passes. A cell's four neighbours in the 5-point
  stencil are always the opposite colour, so each half-sweep is fully
  parallel while staying in place.

Both of those landed, and the second one landed as predicted — the WGSL sweep
in `src/gpu/redBlack.wgsl` is the CPU sweep with a colour test added. Note
what the colouring bought beyond parallelism: it also removed the
four-direction sweep cycling, because there is no ordering left to be biased
by.

Every _other_ kernel keeps strict read/write separation.

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

## Rule 5 — one seam between CPU and GPU: the solver interface

`PressureSolver` in `src/core/pressureSolver.ts` is the _only_ place the two
implementations meet. Five of them exist, and the UI reaches them through an
ENGINE (`G`) that picks advector and solver family together:

| engine | solver      | ordering      | precision | where |
| ------ | ----------- | ------------- | --------- | ----- |
| CPU    | `cpu-mg`    | multigrid     | float64   | CPU   |
| CPU    | `cpu-rbsor` | red-black     | float64   | CPU   |
| CPU    | `cpu-sor`   | lexicographic | float64   | CPU   |
| GPU    | `gpu-mg`    | multigrid     | float32   | WGSL  |
| GPU    | `gpu-rbsor` | red-black     | float32   | WGSL  |

The engine gate is not cosmetic. A GPU advector against a CPU solver (or the
reverse) pays a full round trip per frame that neither pure configuration
does, so its timing describes the seam and not either implementation. Within
one engine the choice is still one variable at a time — ordering, then
algorithm — and both engines offer the same two rungs, which is what keeps
CPU-vs-GPU controlled.

`cpu-rbsor` looks redundant and is the most useful row in the table. Without
it, a wrong GPU picture has three candidate causes at once — ordering,
precision, plumbing — and no way to separate them. With it, "CPU red-black
disagrees too" means the algorithm, and "only the GPU disagrees" means the
shader or the bindings.

**`Simulation.step()` is async because of this, and only because of this.**
WebGPU has no synchronous readback: getting `p` off the device is
`await buffer.mapAsync()` and there is no alternative. On the CPU path the
solver returns a plain number and the `await` resolves in a microtask, so
nothing observable changes.

The ordering is not free, and the price is not where you would look for it:
red-black needs a much lower SOR `omega` than the classical optimum under a
short sweep budget. See `docs/WEBGPU.md` §4 for the numbers — that mistake is
what a wrong GPU picture looked like the first time.

## Rule 6 — measure before porting, with the same instrumentation

`src/core/profiler.ts` is a flat, always-on, sequential phase timer. The
in-browser overlay (`P`) and the headless benchmark (`npm run bench`) both
read it, so the two report the same numbers rather than two different
definitions of "a frame".

This is not optional ceremony. The first measurement contradicted the
assumption the port was based on: at 320x180 the pressure solve was 40% of the
step, not most of it, which caps a pressure-only port at 1.7x however fast the
GPU is. Advection plus dye — 58% — turned out to be the bigger target. The
pressure share does grow with resolution (40% at 320x180, 51% at 480x270, 58%
at 640x360), which is what still made it the right kernel to do first.

## Rule 7 — WGSL is tested against its CPU twin, on real hardware

`src/gpu/chromeHarness.ts` launches headless Chrome, evaluates an expression
in it over the DevTools Protocol, and hands the result back to `node --test`.
Zero dependencies — Node 24 has `fetch` and `WebSocket` globally.

Every shader gets a `*.gputest.ts` that runs it on the GPU and diffs it
against the CPU implementation of the same kernel. That is the only thing that
catches plumbing bugs — a swapped binding, a workgroup count that leaves the
last rows untouched, uniform padding that disagrees with the struct — all of
which produce a plausible-looking field and are invisible from TypeScript.

One trap worth remembering: **the page must be served over
`http://127.0.0.1`**. WebGPU is gated on a secure context, and `about:blank`
reports `isSecureContext === false`, where `navigator.gpu` is simply absent —
indistinguishable from "this machine has no WebGPU".

## Rule 8 — `Float64Array` via an injected constructor

The grid allocator takes the typed-array constructor as a parameter.
PLAN.md §8 compares CPU float64 against GPU float32 while _also_ changing
the solver; injecting the constructor makes float32-on-CPU a one-line run,
which separates the precision variable from the convergence variable.

## Rule 9 — the browser changes structure at frame boundaries, never in a handler

A GPU step spends most of its duration `await`ing a readback. Anything that
resets or reallocates fields from an event handler therefore lands INSIDE a
step: the device's answer, computed from the old scene or into the old
buffers, is written over the new state the moment the readback resolves. That
is what made switching scene or dye leave two scenes composited on screen.

So `src/main.ts` sets `pendingRestart` / `pendingRebuild` and drains them at
the top of `frame()`, between a completed step and the next one. The same
rule is what makes it safe to `destroy()` GPU buffers on a resolution change:
no submit is in flight there.

Related, and the reason the old scene/tracer confusion is gone: a scene owns
its list of meaningful dyes (`src/scenes/catalog.ts`). Tracers and flows are
not freely combinable — a seeded tracer washes out of a channel with an
outlet, and a re-stamped one needs an inflow to be stamped into — so the flow
carries the list instead of a compatibility table sitting between two
independent cycles.

## Deployment

`vite build` -> `dist/`, published to GitHub Pages by CI. `base` in
`vite.config.ts` must match the repo name or every asset 404s on the
project URL. Pages is HTTPS, which WebGPU requires (secure context only).
