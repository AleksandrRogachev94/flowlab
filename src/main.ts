import { CpuAdvector } from './core/advector.ts';
import { CpuMultigridSolver } from './core/pressureMultigrid.ts';
import {
  cpuPressureSolver,
  cpuRedBlackSolver,
  type PressureSolver,
} from './core/pressureSolver.ts';
import { Profiler } from './core/profiler.ts';
import { Simulation } from './core/simulation.ts';
import { GpuAdvector } from './gpu/advectGpu.ts';
import { describeGpu, initGpu, type GpuContext } from './gpu/device.ts';
import { GpuMultigridSolver } from './gpu/multigridGpu.ts';
import { GpuPressureSolver } from './gpu/pressureGpu.ts';
import { GpuStepper } from './gpu/stepGpu.ts';
import { dyeOf, SCENES, sceneSpec, type Scene } from './scenes/catalog.ts';
import {
  Controls,
  type Engine,
  type Option,
  type Quality,
  type UiKey,
  type UiState,
} from './ui/controls.ts';
import { DyeRenderer } from './viz/dyeGpu.ts';
import { divergenceStats, FieldView, VIEWS, type View } from './viz/fieldView.ts';
import { PerfOverlay } from './viz/perfOverlay.ts';

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const dyeCanvas = document.querySelector<HTMLCanvasElement>('#dye')!;
const ui = document.querySelector<HTMLDivElement>('#ui')!;
const ctx = canvas.getContext('2d')!;

/* ------------------------------------------------------------------ tables */

/**
 * Grid rows per quality setting. ROWS, not a cell count: the domain is one unit
 * tall by construction (Simulation sets h = 1/ny), so ny is the physical
 * resolution and nx follows from the window's aspect ratio. Picking rows this
 * way is what lets the simulation fill the viewport exactly at any window
 * shape — see gridFor().
 *
 * Cost grows faster than the cell count on the SOR options (their sweep count
 * scales with N too), and roughly with it on multigrid. The measured all-CPU
 * baseline that set these, per step at 4:3 (npm run bench, macCormack):
 *
 *   grid       total    advect   pressure   dye
 *   320x180     70.6     16.1      28.6     25.1
 *   480x270    190.7     35.3      96.5     57.1
 *   640x360    395.6     62.8     228.8    100.8
 *
 * Most of that has since moved to the device. Measured on the fused GPU path,
 * an M4 Air at a ~1.8:1 window: medium (0.66M cells) 60 fps, high (1.31M) 30
 * fps — dead linear in the cell count at ~25 ms per million, which is what
 * puts 'very high' (2.1M) near 20 fps.
 *
 * Medium stays the default because 60 fps reads better than 20 and the
 * difference in what you can SEE at rest is small. It is NOT small in the
 * wake: the dye there is stretched past the cell scale and numerical diffusion
 * takes it, so filaments dim as they get interesting. Resolution is the only
 * real lever on that — there is no viscosity to turn down — which is what
 * 'very high' is for.
 */
const QUALITIES: { value: Quality; label: string; rows: number }[] = [
  { value: 'low', label: 'Low', rows: 400 },
  { value: 'medium', label: 'Medium', rows: 600 },
  { value: 'high', label: 'High', rows: 850 },
  // 1080 rows is 1920x1080 on a 16:9 window — the one round number a viewer
  // already has a feel for.
  { value: 'ultra', label: 'Very high', rows: 1080 },
];

/**
 * The resolution dropdown, labelled with the grid each setting would actually
 * produce in THIS window — "Medium" alone says nothing, and the number it
 * stands for changes with the window anyway, so a static label would be
 * misleading rather than merely vague. Regenerated on every rebuild, which is
 * also every resize.
 */
function qualityOptions(): Option[] {
  return QUALITIES.map((q) => {
    const { nx, ny } = gridFor(q.value);
    const millions = (nx * ny) / 1e6;
    return { value: q.value, label: `${q.label} — ${nx}x${ny} (${millions.toFixed(2)}M cells)` };
  });
}

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
 * 5e-2 the tolerance was reaching. It is also marginally FASTER at the median,
 * because the residual check it no longer runs cost about one sweep in twelve.
 *
 * Scaled with the grid because SOR needs O(N) sweeps for a fixed error
 * reduction, so any constant is tuned to one resolution. Well above the floor
 * where under-solving compounds (a flat 10 sweeps diverged to NaN at 240x135).
 *
 * These two only reach the SOR solvers, which live under Advanced now that
 * multigrid is the default on both engines. They still decide what those
 * options do, and they are still what the ladder is compared against.
 */
const pressureIters = (nx: number, ny: number): number => Math.round(0.15 * Math.max(nx, ny));

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
 * CONSTANT, not grid-dependent, and that is a consequence of the sweep budget
 * scaling with N: holding sweeps/N fixed holds the stopping point in the
 * transient fixed too, so the best omega does not move. Measured 1.4-1.6 at
 * every size from 160x120 to 400x300, flat to within 5% over 1.2-1.6.
 *
 * 1.6 is near the floor for BOTH orderings, which is what makes it the right
 * shared value: switching solver must change the solver and nothing else, or
 * the comparison stops being controlled.
 */
const PRESSURE_OMEGA = 1.6;

const VIEW_LABELS: Record<View, string> = {
  dye: 'Dye',
  vorticity: 'Vorticity (spin)',
  speed: 'Speed',
  divergence: 'Divergence (debug)',
};

/** Built once and shared by the dropdown and the shortcut that cycles it, so
 *  the two cannot disagree about the order or the contents. */
const VIEW_OPTIONS: Option[] = VIEWS.map((v) => ({ value: v, label: VIEW_LABELS[v] }));
const SCENE_OPTIONS = SCENES.map((s) => ({ value: s.id, label: s.label, blurb: s.blurb }));

const SCHEMES: Option[] = [
  { value: 'macCormack', label: 'MacCormack (sharp)' },
  { value: 'semiLagrangian', label: 'Semi-Lagrangian (blurry)' },
];

const ENGINES: Option[] = [
  { value: 'gpu', label: 'GPU' },
  { value: 'cpu', label: 'CPU' },
];

/**
 * The pressure solvers reachable on each engine — the comparison ladder of
 * docs/WEBGPU.md §1, now gated by engine rather than cycled through as one
 * flat list.
 *
 * That gating is the point. Running a GPU advector against a CPU solver, or
 * the reverse, measures neither: every frame pays a full round trip that
 * neither pure configuration does, so the number belongs to the seam and not
 * to either implementation. Within one engine the choice is a real
 * one-variable-at-a-time question — ordering, then algorithm — and both
 * engines offer the same two rungs, so GPU-vs-CPU stays controlled too.
 */
const SOLVERS: Record<Engine, Option[]> = {
  gpu: [
    { value: 'gpu-mg', label: 'Multigrid (fused step)' },
    { value: 'gpu-rbsor', label: 'Red-black SOR' },
  ],
  cpu: [
    { value: 'cpu-mg', label: 'Multigrid' },
    { value: 'cpu-rbsor', label: 'Red-black SOR' },
    { value: 'cpu-sor', label: 'SOR (lexicographic)' },
  ],
};

/* ------------------------------------------------------------------- state */

const state: UiState = {
  scene: SCENES[0].id,
  dye: SCENES[0].dyes[0].id,
  view: 'dye',
  arrows: false,
  engine: 'gpu',
  solver: 'gpu-mg',
  /**
   * MacCormack by default: on the same seed it keeps ~4x the kinetic energy and
   * ~6x the enstrophy of plain semi-Lagrangian at t = 10, and a dye peak of
   * 0.95 against 0.56 (128x128 cluster). The schemes share all state, so the
   * switch shows up immediately in how fast filaments blur.
   */
  scheme: 'macCormack',
  quality: 'medium',
  perf: false,
  diagnostics: false,
};

/**
 * Everything that depends on the GRID, and therefore has to be rebuilt when
 * the resolution changes. Grouped rather than left as eight module variables
 * so that "the grid changed" is one assignment and nothing can be left behind
 * pointing at the old size.
 */
interface App {
  sim: Simulation;
  view: FieldView;
  cpuAdvector: CpuAdvector;
  /** id -> instance, for whatever SOLVERS offers on the live engine. */
  solvers: Map<string, PressureSolver>;
  gpu: GpuBundle | null;
}

/** The device-side objects. All of them are sized to one grid. */
interface GpuBundle {
  advector: GpuAdvector;
  mg: GpuMultigridSolver;
  rbsor: GpuPressureSolver;
  stepper: GpuStepper;
  /** Draws the dye view from the advector's resident buffer — the reason the
   *  dye never has to come back to the host. Null only if the canvas would not
   *  give up a 'webgpu' context, which is not survivable on this path. */
  dyeView: DyeRenderer;
}

let app: App;
let gpuCtx: GpuContext | null = null;
let gpuName = 'no adapter';

/**
 * Multigrid keeps no per-grid state a caller can see — it rebuilds its level
 * stack from whatever grid the first solve hands it — so one instance survives
 * every rebuild. The GPU objects cannot: their buffers ARE the size.
 */
const cpuMg = new CpuMultigridSolver();

/**
 * Structural changes are QUEUED, never applied inside an event handler, and
 * this is a correctness requirement rather than tidiness.
 *
 * A GPU step is `await`ing a readback for most of its duration. A click that
 * resets the fields in the middle of that window is undone the moment the
 * readback lands: the device's answer, computed from the PREVIOUS scene, is
 * written straight over the fresh one. That is exactly what made switching
 * scene or dye leave two scenes composited on screen. Draining these flags at
 * the top of frame(), between a completed step and the next one, is the whole
 * fix — and it also means a rebuild never frees a buffer a pending submit is
 * still reading.
 */
let pendingRebuild = false;
let pendingRestart = false;
/** Engine/solver changes, deferred for the reason above plus the one in
 *  applyEngineDeferred. */
let pendingEngine = false;

/** Stepping is held off until the GPU probe resolves — see boot(). */
let ready = false;

/* ------------------------------------------------------------------- chrome */

/**
 * Built before the simulation, deliberately: `rebuild()` calls back into
 * `controls` to relabel the resolution dropdown, and a `const` referenced
 * before its initialiser is a ReferenceError rather than an undefined. Nothing
 * here needs a grid — the option lists are functions of `state` and the
 * window — so the ordering costs nothing.
 */
const overlay = new PerfOverlay(ui);

const dyeOptions = (): Option[] =>
  sceneById(state.scene).dyes.map((d) => ({ value: d.id, label: d.label }));

const controls = new Controls(
  ui,
  state,
  {
    scenes: SCENE_OPTIONS,
    dyes: dyeOptions(),
    views: VIEW_OPTIONS,
    engines: ENGINES,
    solvers: SOLVERS[state.engine],
    schemes: SCHEMES,
    qualities: qualityOptions(),
  },
  set,
  () => {
    pendingRestart = true;
  },
);

/* ------------------------------------------------------------------- build */

/**
 * How much finer than the velocity grid the dye is stored — Simulation's
 * `dyeScale`.
 *
 * Pinned to the DISPLAY rather than set as a fixed multiplier, and that is the
 * whole design of the knob. Rendering more dye cells than there are pixels to
 * show them in is work nobody can see, so `floor` of the ratio is the natural
 * cap and it makes the setting self-limiting: at 'very high' on most screens
 * the grid already matches the display and this returns 1, changing nothing.
 * At 'medium' on a retina laptop it returns 2, which is where the win is —
 * that is exactly the case where the old path was upscaling a 600-row picture
 * to an 1800-pixel canvas and calling the blur numerical diffusion.
 *
 * Capped at 2 because the carrier velocity is bilinearly interpolated, so it
 * is effectively band-limited at the velocity grid: past ~2x the dye starts
 * resolving the interpolation's own kinks rather than the flow, and the extra
 * cells buy artifacts. Whole numbers only, so the two grids cover exactly the
 * same rectangle.
 */
function dyeScaleFor(ny: number): number {
  const rows = window.innerHeight * (window.devicePixelRatio || 1);
  return Math.max(1, Math.min(2, Math.floor(rows / ny)));
}

/** 0 is "never" — the residual is a diagnostic, so it is not worth an
 *  O(cells) host pass per 4 steps while the readout that shows it is off. */
function residualEveryFor(): number {
  return state.diagnostics ? 4 : 0;
}

/** Rows from the quality setting, columns from the window: the grid is the
 *  same shape as the viewport, so the canvas needs no letterbox. */
function gridFor(quality: Quality): { nx: number; ny: number } {
  const ny = QUALITIES.find((q) => q.value === quality)!.rows;
  const aspect = window.innerWidth / Math.max(1, window.innerHeight);
  return { nx: Math.max(64, Math.round(ny * aspect)), ny };
}

/**
 * Sizes the backing store, which is NOT simply CSS pixels times dpr.
 *
 * The heatmap's source is one pixel per cell, upscaled by the blit, so a
 * backing store much larger than the grid buys no detail — it only makes the
 * per-frame drawImage bigger, and at 1600x900 that draw already cost 11 ms of
 * a 48 ms frame. Capping it near the grid's own resolution is most of that
 * cost back. The cap sits slightly above 1:1 so the arrows and the obstacle
 * staircase still get a little oversampling to land on.
 */
function fitCanvas(): void {
  const cssW = window.innerWidth;
  const cssH = window.innerHeight;
  const dpr = window.devicePixelRatio || 1;
  const scale = Math.min(dpr, Math.max(1, (1.25 * app.sim.g.nx) / cssW));
  canvas.width = Math.round(cssW * scale);
  canvas.height = Math.round(cssH * scale);
}

function buildGpu(ctx: GpuContext, sim: Simulation): GpuBundle {
  const advector = new GpuAdvector(ctx, sim.g, sim.dyeG);
  const mg = new GpuMultigridSolver(ctx, sim.g);
  return {
    advector,
    mg,
    rbsor: new GpuPressureSolver(ctx, sim.g),
    stepper: new GpuStepper(ctx, sim.g, advector, mg),
    dyeView: new DyeRenderer(ctx, dyeCanvas, sim.dyeG, advector.dyeIn),
  };
}

function destroyGpu(bundle: GpuBundle): void {
  bundle.dyeView.destroy();
  bundle.stepper.destroy();
  bundle.advector.destroy();
  bundle.mg.destroy();
  bundle.rbsor.destroy();
}

/** Everything grid-shaped, from scratch. Also the only path that resizes the
 *  canvas, so the picture and the domain cannot disagree. */
function rebuild(): void {
  if (app?.gpu) destroyGpu(app.gpu);

  const { nx, ny } = gridFor(state.quality);
  /**
   * Float32 fields in the browser — the precision study's other arm, and at
   * this grid a straight bandwidth win: every remaining host loop (CFL scan,
   * residual, the draw's stats) moves half the bytes, and the GPU staging
   * copies become same-type memcpys instead of million-element f64 -> f32
   * conversions. The headless tests and benchmarks keep the Float64Array
   * default, so the CPU reference stays the reference.
   */
  const sim = new Simulation(
    nx,
    ny,
    {
      scheme: state.scheme,
      pressureIters: pressureIters(nx, ny),
      tol: 0, // spend the budget; see pressureIters
      omega: PRESSURE_OMEGA,
      // A diagnostic, and the last O(cells) host loop in the frame — see
      // SimulationParams.residualEvery. The headless default stays 1.
      residualEvery: residualEveryFor(),
    },
    Float32Array,
    dyeScaleFor(ny),
  );

  app = {
    sim,
    view: new FieldView(sim.g, sim.dyeG),
    cpuAdvector: new CpuAdvector(sim.g),
    solvers: new Map(),
    gpu: gpuCtx ? buildGpu(gpuCtx, sim) : null,
  };

  app.solvers.set('cpu-sor', cpuPressureSolver);
  app.solvers.set('cpu-rbsor', cpuRedBlackSolver);
  app.solvers.set('cpu-mg', cpuMg);
  if (app.gpu) {
    app.solvers.set('gpu-mg', app.gpu.mg);
    app.solvers.set('gpu-rbsor', app.gpu.rbsor);
  }

  fitCanvas();
  // The labels quote the live window's grid, so they stay right only until the
  // next resize — and a resize comes through here.
  controls.setQualities(qualityOptions());
  applyEngine();
  restart();
}

/**
 * Points the simulation at one engine's advector and solver, together.
 *
 * The fused whole-step path (gpu/stepGpu.ts) is active exactly while the
 * selected solver is the multigrid it embeds — that solver IS the stepper's
 * V-cycle, so the two cannot be chosen separately. Every other combination
 * falls back to the phase-wise seams, which is what keeps the ladder's
 * comparisons honest: the fused path is a rung of its own, not a modifier.
 *
 * Re-entering the fused path must invalidate. While another solver was
 * selected the host advanced u, v and dye on its own and the device's resident
 * copies went stale; without this the first fused frame back would advect a
 * field from however many frames ago.
 */
function applyEngine(): void {
  const { sim, gpu } = app;
  const onGpu = state.engine === 'gpu' && gpu !== null;

  sim.advector = onGpu ? gpu.advector : app.cpuAdvector;
  sim.solver = app.solvers.get(state.solver) ?? cpuMg;
  sim.stepper = onGpu && sim.solver === gpu.mg ? gpu.stepper : null;
  sim.stepper?.invalidate();
}

/**
 * applyEngine plus the one thing that has to happen at a frame boundary rather
 * than inside an event handler: bringing the dye home.
 *
 * While the fused path drives, the DEVICE holds the authoritative dye — the
 * host's copy is whatever reset() last wrote (see GpuStep.readDye). Anything
 * else taking over would advect that stale field and the picture would jump
 * back in time. The readback is async, which is why the engine and solver
 * widgets queue this through pendingEngine instead of calling applyEngine
 * directly the way they used to.
 */
async function applyEngineDeferred(): Promise<void> {
  const wasFused = app.sim.stepper !== null;
  const { gpu } = app;
  applyEngine();
  if (wasFused && app.sim.stepper === null && gpu) {
    await gpu.stepper.readDye(app.sim.f.dye);
  }
}

function sceneById(id: string): Scene {
  return SCENES.find((s) => s.id === id) ?? SCENES[0];
}

/**
 * Back to t = 0 with the current (scene, dye) pair. Reseeding is the only way
 * to change either: dye already in the domain was advected under the old seed
 * and cannot be reinterpreted as the new one, and velocity carries the old
 * geometry's imprint.
 */
function restart(): void {
  const scene = sceneById(state.scene);
  app.sim.params.dyeDecay = scene.decay ?? 0;
  app.sim.reset(sceneSpec(scene, dyeOf(scene, state.dye), app.sim.g));
  app.view.invalidateSolids();
}

/**
 * The single place state changes. Every widget and every key funnels through
 * it, so a control and its shortcut cannot end up doing different things —
 * which is what two parallel tables of buttons and keys used to guarantee.
 */
function set(key: UiKey, value: string | boolean): void {
  if (state[key] === value) return;
  // The one cast in the file, and it is the DOM boundary: `view`, `engine` and
  // `quality` are narrow string unions, while a <select> yields a plain
  // string. Sound because every widget is built from an Option[] taken from
  // the typed tables above, so the only values reachable here are members —
  // a runtime guard would check what the option lists already guarantee.
  (state as Record<UiKey, string | boolean>)[key] = value;
  apply(key);
  controls.sync();
}

function apply(key: UiKey): void {
  switch (key) {
    case 'scene':
      // The new scene's DEFAULT tracer, always — never the one carried over
      // from the scene before. Carrying it over was the first version, on the
      // theory that a tracer is a preference; in use it just means a scene
      // opens on something other than the look it was tuned for, which for a
      // demo is the wrong trade. Re-picking is one click.
      state.dye = sceneById(state.scene).dyes[0].id;
      controls.setDyes(dyeOptions());
      pendingRestart = true;
      break;
    case 'dye':
      // A dye view with no dye is a black screen, and it looks like a bug
      // rather than a choice. Vorticity is the view that needs no tracer.
      if (state.dye === 'none' && state.view === 'dye') state.view = 'vorticity';
      pendingRestart = true;
      break;
    case 'engine':
      if (!SOLVERS[state.engine].some((s) => s.value === state.solver)) {
        state.solver = SOLVERS[state.engine][0].value;
      }
      controls.setSolvers(SOLVERS[state.engine]);
      pendingEngine = true;
      break;
    case 'solver':
      pendingEngine = true;
      break;
    case 'scheme':
      // Read fresh every step, so this takes effect on the next one and the
      // two schemes can be compared on identical state.
      app.sim.params.scheme = state.scheme;
      break;
    case 'quality':
      pendingRebuild = true;
      break;
    case 'perf':
      overlay.setVisible(state.perf);
      break;
    case 'diagnostics':
      // Turning the readout on is what starts the residual being computed at
      // all; leaving it off skips both that and the stats pass in the frame.
      app.sim.params.residualEvery = residualEveryFor();
      break;
    default:
      // view and arrows are read by the draw every frame; there is nothing to
      // install.
      break;
  }
}

/** Next entry in a list, wrapping — what the single-key shortcuts do. */
function cycle(key: UiKey, options: Option[]): void {
  const i = options.findIndex((o) => o.value === state[key]);
  set(key, options[(i + 1) % options.length].value);
}

const KEYS: Record<string, () => void> = {
  r: () => {
    pendingRestart = true;
  },
  s: () => cycle('scene', SCENE_OPTIONS),
  d: () => cycle('view', VIEW_OPTIONS),
  t: () => cycle('dye', dyeOptions()),
  a: () => cycle('scheme', SCHEMES),
  g: () => set('engine', state.engine === 'gpu' ? 'cpu' : 'gpu'),
  v: () => set('arrows', !state.arrows),
  p: () => set('perf', !state.perf),
  c: () => controls.togglePanel(),
};

window.addEventListener('keydown', (e) => {
  // Cmd+R, Cmd+S and Cmd+P are all browser shortcuts that collide with these.
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  // A key pressed while a <select> has focus belongs to the select.
  if (e.target instanceof HTMLSelectElement) return;
  KEYS[e.key.toLowerCase()]?.();
});

/**
 * The grid is derived from the window, so a resize changes the simulation and
 * not just the picture — hence a rebuild rather than a re-fit, and hence the
 * debounce: a drag fires this continuously, and rebuilding per event would
 * allocate a grid and a set of GPU buffers per pixel of drag.
 */
let resizeTimer = 0;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => {
    pendingRebuild = true;
  }, 250);
});

/**
 * Drops the GPU on any device-side failure, once. A GPU error is usually
 * permanent (a lost device, an out-of-memory), so retrying it every frame
 * turns one console line into thousands.
 */
function dropGpu(why: string): void {
  if (!gpuCtx) return;
  console.error(`[gpu] falling back to the CPU: ${why}`);
  // Guarded, unlike the rebuild path: this runs mid-frame with a submit that
  // just failed, and a lost device makes destroy() itself throw on some
  // drivers. Leaking the buffers of a device that is already gone is fine;
  // taking the render loop down with it is not.
  try {
    if (app.gpu) destroyGpu(app.gpu);
  } catch {
    /* the device is gone; there is nothing left to free */
  }
  app.gpu = null;
  gpuCtx = null;
  gpuName = 'no adapter';
  app.solvers.delete('gpu-mg');
  app.solvers.delete('gpu-rbsor');
  controls.setGpuAvailable(false);
  set('engine', 'cpu');
  // set() only QUEUES the engine change, and this runs mid-frame with a failed
  // submit behind it: the caller's immediate retry would otherwise still be
  // holding the destroyed stepper. Nothing is lost by doing it here instead —
  // applyEngineDeferred's readback exists to rescue the dye from a working
  // device, and there is no longer one.
  applyEngine();
  pendingEngine = false;
}

/* -------------------------------------------------------------------- frame */

/**
 * Which GPU object's timings the panel should show, if any.
 *
 * When the fused stepper is driving, the solver's own timings are dead —
 * solve() never runs — so the stepper's take their place. Written as four
 * plain checks rather than nested conditionals with a cast: the question is
 * "which of these three objects is doing the work", and identity comparison
 * answers it without asserting anything to the type system.
 */
function gpuTimings(): GpuStepper | GpuMultigridSolver | GpuPressureSolver | null {
  const { sim, gpu } = app;
  if (!gpu) return null;
  if (sim.stepper) return gpu.stepper;
  if (sim.solver === gpu.mg) return gpu.mg;
  if (sim.solver === gpu.rbsor) return gpu.rbsor;
  return null;
}

/**
 * The one cost sim.perf cannot see: `step()` owns its own phase breakdown, but
 * the draw happens out here. Reusing Profiler rather than keeping a hand-rolled
 * average means the smoothing, the peak tracking and the units all come from
 * the same place as every other number on the panel.
 */
const drawPerf = new Profiler();

function statusLine(maxSpeed: number, divMax: number, divRms: number): string {
  const { sim } = app;
  const fps = overlay.frameMs > 0 ? 1000 / overlay.frameMs : 0;
  // Line one is for everyone: what am I looking at, how big is it, how fast is
  // it going. No units anyone has to have studied.
  const head =
    `${sceneById(state.scene).label}   ${sim.g.nx}x${sim.g.ny} grid   ` +
    `${sim.solver.name} on ${state.engine === 'gpu' ? gpuName : 'the CPU'}   ` +
    `${fps.toFixed(0)} fps`;
  if (!state.diagnostics) return head;

  // Line two is the physics: the numbers that say whether to believe the
  // picture. div rms is the projection residual as a fraction of the largest
  // divergence one cell could carry, so it is comparable across resolutions.
  return (
    `${head}\n` +
    `t ${sim.time.toFixed(2)}   max speed ${maxSpeed.toFixed(3)}   ` +
    `dt ${sim.dt.toExponential(2)} (CFL ${sim.cfl.toFixed(2)})   ` +
    // Sweeps on the SOR solvers, V-cycles on the multigrid ones.
    `solve ${sim.iters} its   ` +
    `div rms ${(100 * divRms).toFixed(3)}% of u/h (worst cell ±${divMax.toExponential(1)})`
  );
}

async function frame(): Promise<void> {
  // Structural changes land HERE and nowhere else — see pendingRebuild.
  if (pendingRebuild) {
    pendingRebuild = false;
    pendingRestart = false;
    rebuild();
  }
  if (pendingRestart) {
    pendingRestart = false;
    restart();
  }
  if (pendingEngine) {
    pendingEngine = false;
    await applyEngineDeferred();
  }

  if (ready) {
    try {
      await app.sim.step();
    } catch (e) {
      dropGpu(e instanceof Error ? e.message : String(e));
      await app.sim.step();
    }
  }

  drawPerf.begin();
  // The dye view comes off the device whenever the fused path is driving,
  // which is the only time the device holds the authoritative dye. The 2D
  // canvas on top is then cleared and carries only the solids and the arrows.
  const dyeOnDevice = state.view === 'dye' && app.sim.stepper !== null && app.gpu !== null;
  dyeCanvas.hidden = !dyeOnDevice;
  if (dyeOnDevice) app.gpu!.dyeView.draw();
  const maxSpeed = app.view.draw(ctx, app.sim, state.view, state.arrows, dyeOnDevice);
  // Only when something reads it: a full pass over every cell, and the status
  // line drops it otherwise. Simulation stops refreshing `div` at all in that
  // case — see residualEvery below.
  const { divMax, divRms } = state.diagnostics
    ? divergenceStats(app.sim, maxSpeed)
    : { divMax: 0, divRms: 0 };
  drawPerf.mark('draw');

  controls.setHud(statusLine(maxSpeed, divMax, divRms));

  // Pure wiring: everything here is already measured and already smoothed by
  // whoever owns it, and the panel decides how it reads.
  const onGpu = gpuTimings();
  overlay.update({
    phases: app.sim.perf,
    totals: drawPerf,
    solver: app.sim.solver.name,
    adapter: gpuName,
    gpu: onGpu
      ? {
          ...onGpu.timings,
          hasDeviceTime: onGpu.hasDeviceTime,
          what: onGpu === app.gpu?.stepper ? 'whole step' : 'pressure solve',
        }
      : undefined,
  });

  requestAnimationFrame(() => void frame());
}

/* --------------------------------------------------------------------- boot */

/**
 * The adapter is awaited BEFORE the first step, which is a reversal: the old
 * loop started on the CPU and swapped the device in when it arrived. That was
 * right when the browser ran a 320x180 grid. Full-screen it is not — a single
 * CPU multigrid step at 600 rows is a visible freeze, and the reward for it is
 * one frame of a scene that has not moved yet.
 *
 * So: build and DRAW immediately (the seeded scene is worth looking at), and
 * let `ready` gate the stepping until we know what we are stepping on.
 */
async function boot(): Promise<void> {
  rebuild();
  requestAnimationFrame(() => void frame());

  gpuCtx = await initGpu();
  if (gpuCtx) {
    gpuName = describeGpu(gpuCtx);
  } else {
    controls.setGpuAvailable(false);
    state.engine = 'cpu';
    state.solver = 'cpu-mg';
    controls.setSolvers(SOLVERS.cpu);
    // One thread of JavaScript cannot carry 600 rows at anything like frame
    // rate, and a demo that opens at 2 fps reads as broken rather than as
    // honest. The picture is smaller; it still moves.
    state.quality = 'low';
    controls.sync();
  }
  // Either way the grid-shaped objects were built without a device, or for the
  // wrong resolution. One rebuild covers both.
  pendingRebuild = true;
  ready = true;
}

controls.togglePanel();
overlay.setVisible(state.perf);
void boot();
