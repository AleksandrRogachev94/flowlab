import { CpuAdvector } from './core/advector.ts';
import { CpuMultigridSolver } from './core/pressureMultigrid.ts';
import {
  cpuPressureSolver,
  cpuRedBlackSolver,
  type PressureSolver,
} from './core/pressureSolver.ts';
import { Profiler } from './core/profiler.ts';
import { Simulation, type DyeSeed, type DyeSource, type SceneSpec } from './core/simulation.ts';
import { describeGpu, initGpu } from './gpu/device.ts';
import { GpuAdvector } from './gpu/advectGpu.ts';
import { GpuMultigridSolver } from './gpu/multigridGpu.ts';
import { GpuPressureSolver } from './gpu/pressureGpu.ts';
import { karmanChannel } from './scenes/karman.ts';
import { wallJet } from './scenes/emitters.ts';
import { openRight } from './scenes/obstacles.ts';
import {
  addDyeMono,
  addDyeStripes,
  addDyeTriad,
  addVortexCluster,
  addVortexPair,
  stripeInflow,
} from './scenes/testFields.ts';
import { FieldView, VIEWS } from './viz/fieldView.ts';
import { PerfOverlay } from './viz/perfOverlay.ts';

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const stage = document.querySelector<HTMLDivElement>('#stage')!;
const ctx = canvas.getContext('2d')!;
const readout = document.querySelector<HTMLPreElement>('#readout')!;

/**
 * One screen-shaped grid for every scene: h = 1/ny, so the domain is always
 * exactly one unit tall and nx/ny units wide. Scenes used to carry their own
 * dimensions, but a shared domain beat resizing per scene; karman keeps its
 * long wake with a SMALLER cylinder rather than a wider box — what matters is
 * diameters downstream, and 4:3 gives it less room than 16:9 did.
 *
 * Cost grows faster than the cell count, since SOR's sweep count also scales
 * with N. Measured per step with `npm run bench` (macCormack, fixed budget):
 *
 *   grid       total    advect   pressure   dye    everything else
 *   320x180    70.6      16.1      28.6     25.1        0.9
 *   480x270   190.7      35.3      96.5     57.1        1.9
 *   640x360   395.6      62.8     228.8    100.8        3.3
 *
 * What sets the ceiling has MOVED, and that is the part to keep in mind when
 * changing NX/NY. Those are all-CPU numbers, and both of the big columns have
 * since gone to the device: advect and dye now cost ~5.7 and ~3.5 ms at
 * 640x480 (docs/WEBGPU.md §8), and the pressure solve is ~60% of the step
 * again. The next resolution step is therefore a better SOLVER, not another
 * shader — see docs/WEBGPU.md §6.
 */
const NX = 1920;
const NY = 1080;

/**
 * A FIXED sweep budget, not a convergence tolerance — the real-time half of
 * PLAN.md §8, where a guaranteed frame budget is worth more than a converged
 * solve. `defaultParams` keeps the strict relative tolerance for headless
 * reference runs, which is where accuracy is the point.
 *
 * Two measurements decide it. The physics is indifferent to solver effort over
 * a ~100x range: St and wake amplitude are unchanged from tol 5e-3 (median 196
 * sweeps) down to 20 sweeps. But a tolerance produces a VARIABLE sweep count —
 * on karman at 480x270 it asked for 24 sweeps on the easiest steady-state
 * frame and 96 on the hardest — and that 4x swing in frame time is the stutter.
 * A cap alone cannot fix it, since it only truncates the top of the range.
 *
 * Spending the budget every frame instead: 1.3x frame-time spread at 480x270
 * (1.1x at 320x180), identical St, and a residual that settles to the same
 * 5e-2 the tolerance was reaching. It is also marginally FASTER at the median
 * (128 vs 136 ms), because the residual check it no longer runs cost about one
 * sweep in every twelve.
 *
 * Scaled with the grid because SOR needs O(N) sweeps for a fixed error
 * reduction, so any constant is tuned to one resolution. Well above the floor
 * where under-solving compounds (a flat 10 sweeps diverged to NaN at 240x135).
 *
 * Re-measured under MacCormack, since it hands the solver a sharper u* and
 * does raise demand by ~a third (median 144 vs 108 sweeps at 240x135). The
 * physics is unmoved, which is what actually decides this: St 0.2045 / 0.2053 /
 * 0.2055 and wake amplitude 0.675 / 0.680 / 0.683 at 36 / 72 / 150 sweeps —
 * 0.5% across a 4x range, for 2.1x the frame time. The budget stays.
 */
const PRESSURE_ITERS = Math.round(0.15 * Math.max(NX, NY));
const PRESSURE_TOL = 0;

/**
 * Relaxation for the FIXED-BUDGET path, and deliberately NOT optimalOmega().
 *
 * optimalOmega minimises the ASYMPTOTIC rate, which SOR only reaches after
 * O(N) sweeps. This runs 0.15*N of them, so what matters instead is the
 * TRANSIENT — and SOR near omega = 2 amplifies the residual for tens of sweeps
 * before it decays. Stopping inside that hump is worse than not over-relaxing
 * at all, and red-black suffers it far worse than lexicographic does: at
 * optimalOmega it left 48x the divergence, enough to make the velocity field
 * visibly oscillate. Numbers, and why red-black is the sensitive one, in
 * docs/WEBGPU.md §4.
 *
 * CONSTANT, not grid-dependent, and that is a consequence of PRESSURE_ITERS
 * scaling with N: holding sweeps/N fixed holds the stopping point in the
 * transient fixed too, so the best omega does not move. Measured 1.4-1.6 at
 * every size from 160x120 to 400x300, flat to within 5% over 1.2-1.6. If the
 * sweep budget ever became a constant instead, this would have to drop as the
 * grid grew.
 *
 * 1.6 is near the floor for BOTH orderings, which is what makes it the right
 * shared value: G must change the solver and nothing else, or the comparison
 * stops being controlled. defaultParams keeps optimalOmega for headless
 * reference runs, where the tolerance really does iterate to convergence.
 */
const PRESSURE_OMEGA = 1.6;

/**
 * 'dipole'  self-propelling pair — the clearest check that advection moves
 *           things the right way at all.
 * 'cluster' interacting vortices that stretch filaments and merge — the scene
 *           for comparing schemes. Deterministic, so two runs are comparable.
 * 'jet'     steady inflow through part of the left wall, dye injected at the
 *           nozzle. A confined jet entrains, so the box recirculates and a
 *           permanent source would saturate every cell; `decay` keeps it
 *           legible. Tune it as a DISTANCE: k = speed / fadeLength, so 0.5 at
 *           speed 1 fades dye to 1/e after two domain widths of travel.
 * 'karman'  flow past a cylinder — the headline scene.
 */
interface SceneDef {
  name: string;
  /** Scene injects its own dye, so entering it defaults the tracer to 'none'
   *  rather than layering a preset underneath. Pressing T still overrides. */
  ownDye?: boolean;
  /** The whole left edge is inflow, so a tracer can be topped up there — see
   *  stripeInflow. False for wallJet, whose left edge is mostly wall. */
  inflow?: boolean;
  decay?: number;
  build: () => SceneSpec;
}

const SCENES: SceneDef[] = [
  { name: 'karman', ownDye: true, inflow: true, build: () => karmanChannel(sim.g) },
  { name: 'dipole', build: () => ({ seed: addVortexPair }) },
  { name: 'cluster', build: () => ({ seed: addVortexCluster }) },
  {
    name: 'jet',
    ownDye: true,
    decay: 0.5,
    // openRight() is not optional: wallJet is pure inflow, so without an outlet
    // the all-Neumann system is inconsistent and never converges.
    build: () => {
      const jet = wallJet();
      return { labels: openRight(), seed: jet.seed, dyeSource: jet.source };
    },
  },
];

/**
 * What the dye is seeded with, independent of the velocity scene — the two
 * lists combine freely, since dye is passive.
 *
 * 'triad' three colours that mix where they interleave, so sub-cell filaments
 *         show up as colours that were never seeded.
 * 'mono'  one white disk. The control: same advection, no mixing signal, so
 *         it isolates how much of the picture the colour is really carrying.
 * 'stripes' many thin horizontal bands. Fat bands show where fluid CAME FROM;
 *         thin ones show how much it has been STRETCHED, since every band edge
 *         is a material line and the local spacing is the strain field.
 * 'none'  empty, for scenes that inject their own dye.
 */
interface Tracer {
  name: string;
  /** Written once, at reset. */
  seed?: DyeSeed;
  /** Re-stamped every step, so the tracer keeps arriving instead of washing
   *  out through an outlet. Only meaningful where the whole left edge is
   *  inflow — see SceneDef.inflow and stripeInflow. */
  source?: DyeSource;
}

const TRACERS: Tracer[] = [
  { name: 'triad', seed: addDyeTriad },
  { name: 'mono', seed: addDyeMono },
  { name: 'stripes', seed: addDyeStripes, source: stripeInflow() },
  { name: 'none' },
];

/**
 * MacCormack by default: on the same seed it keeps ~4x the kinetic energy and
 * ~6x the enstrophy of plain semi-Lagrangian at t = 10, and a dye peak of 0.95
 * against 0.56 (128x128 cluster). 'A' flips it live — the schemes share all
 * state, so the switch shows up immediately in how fast filaments blur.
 *
 * It is NOT free here: 130 -> 191 ms/step on karman at this grid, because the
 * fixed 72-sweep budget leaves advection a real share of the frame. Worth it
 * for the picture; if the target is frame rate rather than detail, dropping to
 * 320x180 with MacCormack beats 480x270 without it.
 */
const sim = new Simulation(NX, NY, {
  scheme: 'macCormack',
  pressureIters: PRESSURE_ITERS,
  tol: PRESSURE_TOL,
  omega: PRESSURE_OMEGA,
});
const fieldView = new FieldView(sim.g);
const overlay = new PerfOverlay(stage);

/**
 * G cycles all of these, and the ORDER is the point: each step changes one
 * variable. CPU-lexicographic -> CPU-red-black is the sweep ordering;
 * -> CPU-multigrid is the algorithm; each CPU rung then has its GPU twin
 * (hardware + precision together). When a GPU picture looks wrong, the CPU
 * rung with the same algorithm says whether the algorithm or the WGSL is at
 * fault. See docs/WEBGPU.md §1 and §9.
 *
 * The GPU entries append themselves when the device is ready, and the sim
 * switches to gpu-mg then — it is the fastest AND best-converged solver here
 * (docs/WEBGPU.md §9). Not awaited at startup: the first frame should not
 * wait on a driver, and there is a correct thing to run in the meantime.
 */
const SOLVERS: PressureSolver[] = [cpuPressureSolver, cpuRedBlackSolver, new CpuMultigridSolver()];
let gpuSolvers: (GpuPressureSolver | GpuMultigridSolver)[] = [];
let gpuName = '';

void initGpu().then((ctx) => {
  if (!ctx) return;
  gpuSolvers = [new GpuPressureSolver(ctx, sim.g), new GpuMultigridSolver(ctx, sim.g)];
  gpuName = describeGpu(ctx);
  SOLVERS.push(...gpuSolvers);
  sim.solver = gpuSolvers[1];
  // Not on the G cycle: G exists to compare pressure SOLVERS, and swapping
  // advection underneath it would stop that being one variable at a time.
  // Advection has no algorithmic choice to make here — the GPU kernel is the
  // same scheme A already toggles — so it simply takes over when available.
  sim.advector = new GpuAdvector(ctx, sim.g);
});

/**
 * Drops the GPU solvers on any device-side failure, once. A GPU error is
 * usually permanent (a lost device, an out-of-memory), so retrying it every
 * frame turns one console line into thousands. They are always last in
 * SOLVERS, so truncating also stops G cycling back onto them.
 */
function dropGpu(why: string): void {
  if (gpuSolvers.length === 0) return;
  console.error(`[gpu] falling back to the CPU: ${why}`);
  SOLVERS.length -= gpuSolvers.length;
  gpuSolvers = [];
  sim.solver = cpuPressureSolver;
  sim.advector = new CpuAdvector(sim.g);
}

function toggleSolver(): void {
  sim.solver = SOLVERS[(SOLVERS.indexOf(sim.solver) + 1) % SOLVERS.length];
}

const TRACER_NONE = TRACERS.findIndex((t) => t.name === 'none');

let sceneIndex = 0;
let viewIndex = 0;
let tracerIndex = TRACER_NONE;

/**
 * Largest canvas that fits the stage at the grid's aspect, at DEVICE pixel
 * resolution — the backing store is CSS size times dpr, so the arrows and the
 * cylinder stay crisp on a retina display instead of being upscaled.
 */
function fitCanvas(): void {
  const aspect = sim.g.nx / sim.g.ny;
  const box = stage.getBoundingClientRect();
  const w = Math.max(320, Math.min(box.width, box.height * aspect));
  const h = w / aspect;
  const dpr = window.devicePixelRatio || 1;

  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
}

/**
 * Which source keeps stamping dye, the scene's or the tracer's. The one place
 * the two can disagree, so it is worth a table rather than a nested ternary:
 *
 *   tracer 'none'          -> the scene's own source (karman's RGB inlet bands)
 *   tracer + inflow scene  -> the tracer's source, if it has one
 *   tracer, no inflow      -> none; the tracer is seeded once and left to run
 *
 * An explicit tracer OWNS the dye: karman stamps its RGB bands every step, and
 * left running it composites a second, different tracer on top of the chosen
 * one. And only a full-width inflow can top a tracer up — stripeInflow would
 * otherwise pin dye against a stagnant wall.
 */
function dyeSourceFor(def: SceneDef, spec: SceneSpec, tracer: Tracer): DyeSource | undefined {
  if (!tracer.seed) return spec.dyeSource;
  return def.inflow ? tracer.source : undefined;
}

function restart(): void {
  const def = SCENES[sceneIndex];
  const spec = def.build();
  sim.params.dyeDecay = def.decay ?? 0;
  // The selected tracer always applies, so the readout can never claim one the
  // sim is not running; `?? spec.dye` keeps a scene's own seed when it is
  // 'none'. `ownDye` only picks the DEFAULT on entry, in nextScene.
  const tracer = TRACERS[tracerIndex];
  sim.reset({ ...spec, dye: tracer.seed ?? spec.dye, dyeSource: dyeSourceFor(def, spec, tracer) });
}
function toggleScheme(): void {
  sim.params.scheme = sim.params.scheme === 'macCormack' ? 'semiLagrangian' : 'macCormack';
}
function toggleView(): void {
  viewIndex = (viewIndex + 1) % VIEWS.length;
}
function nextScene(): void {
  sceneIndex = (sceneIndex + 1) % SCENES.length;
  if (SCENES[sceneIndex].ownDye) tracerIndex = TRACER_NONE;
  restart();
}
// Reseeding is the only way to change tracer: dye already in the domain was
// advected under the old seed and cannot be reinterpreted as the new one.
function nextTracer(): void {
  tracerIndex = (tracerIndex + 1) % TRACERS.length;
  restart();
}

/**
 * One table, because a button and its key are the same action. Two parallel
 * lists is how you end up with a button that has no shortcut.
 */
const ACTIONS: { id: string; key: string; run: () => void }[] = [
  { id: 'restart', key: 'r', run: restart },
  { id: 'toggleView', key: 'd', run: toggleView },
  { id: 'nextScene', key: 's', run: nextScene },
  { id: 'nextTracer', key: 't', run: nextTracer },
  { id: 'toggleScheme', key: 'a', run: toggleScheme },
  { id: 'toggleSolver', key: 'g', run: toggleSolver },
  { id: 'togglePerf', key: 'p', run: () => overlay.toggle() },
];

for (const a of ACTIONS) {
  document.querySelector<HTMLButtonElement>(`#${a.id}`)!.addEventListener('click', a.run);
}
window.addEventListener('keydown', (e) => {
  // Cmd+R, Cmd+S and Cmd+P are all browser shortcuts that collide with these.
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  ACTIONS.find((a) => a.key === e.key.toLowerCase())?.run();
});
window.addEventListener('resize', fitCanvas);

fitCanvas();
restart();

/**
 * The one cost sim.perf cannot see: `step()` owns its own phase breakdown, but
 * the draw happens out here. Reusing Profiler rather than keeping a hand-rolled
 * average means the smoothing, the peak tracking and the units all come from
 * the same place as every other number on the panel.
 *
 * begin() sits AFTER the step, not at the top of the frame — a Profiler
 * measures from begin() to the first mark(), so starting it any earlier would
 * quietly bill the whole step to 'draw'.
 */
const drawPerf = new Profiler();

async function frame(): Promise<void> {
  try {
    await sim.step();
  } catch (e) {
    dropGpu(e instanceof Error ? e.message : String(e));
    await sim.step();
  }
  drawPerf.begin();
  const { maxSpeed, divMax, divRms } = fieldView.draw(ctx, sim, VIEWS[viewIndex]);
  drawPerf.mark('draw');

  readout.textContent =
    `scene ${SCENES[sceneIndex].name} (S)   view ${VIEWS[viewIndex]} (D)   ` +
    `dye ${TRACERS[tracerIndex].name} (T)   ` +
    `advect ${sim.params.scheme}/${sim.advector.name} (A)   ` +
    `solver ${sim.solver.name} (G)   perf (P)   grid ${sim.g.nx}x${sim.g.ny}\n` +
    `t ${sim.time.toFixed(2)}   max speed ${maxSpeed.toFixed(3)}   ` +
    `dt ${sim.dt.toExponential(2)} (CFL ${sim.cfl.toFixed(2)})   ` +
    // Sweeps on the SOR solvers, V-cycles on the multigrid ones.
    `solve ${sim.iters} its   ` +
    `div rms ${(100 * divRms).toFixed(3)}% of u/h (worst cell ±${divMax.toExponential(1)})`;

  // Pure wiring: everything here is already measured and already smoothed by
  // whoever owns it, and the panel decides how it reads.
  const onGpu = gpuSolvers.find((s) => s === sim.solver) ?? null;
  overlay.update({
    phases: sim.perf,
    totals: drawPerf,
    solver: sim.solver.name,
    adapter: gpuName,
    gpu: onGpu ? { ...onGpu.timings, hasDeviceTime: onGpu.hasDeviceTime } : undefined,
  });

  requestAnimationFrame(() => void frame());
}

requestAnimationFrame(() => void frame());
