import { Simulation, type DyeSeed, type SceneSpec } from './core/simulation.ts';
import { karmanChannel } from './scenes/karman.ts';
import { wallJet } from './scenes/emitters.ts';
import { openRight } from './scenes/obstacles.ts';
import { addDyeMono, addDyeTriad, addVortexCluster, addVortexPair } from './scenes/testFields.ts';
import { FieldView, VIEWS } from './viz/fieldView.ts';

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const stage = document.querySelector<HTMLDivElement>('#stage')!;
const ctx = canvas.getContext('2d')!;
const readout = document.querySelector<HTMLPreElement>('#readout')!;

/**
 * One 16:9 grid for every scene (1 unit tall, 16/9 wide). Scenes used to carry
 * their own dimensions, but resizing the window per scene was worse than a
 * shared screen-shaped domain; karman keeps its long wake with a SMALLER
 * cylinder rather than a wider box — what matters is diameters downstream.
 *
 * Cost grows faster than the cell count, since SOR's sweep count also scales
 * with N: a solve is ~17 ms/step at 240x135, 41 at 320x180, 136 at 480x270.
 * 320x180 is roughly the real-time ceiling on the CPU; going higher is a
 * solver problem (PLAN.md's CG step, then WebGPU), not a tuning one.
 */
const NX = 480;
const NY = 270;

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
 * reduction, so any constant is tuned to one resolution — measured demand ran
 * 0.10-0.15 * N across 240x135, 320x180 and 480x270. 0.15 * N sits at the top
 * of that, well above the floor where under-solving compounds (a flat 10
 * sweeps diverged to NaN at 240x135).
 */
const PRESSURE_ITERS = Math.round(0.15 * Math.max(NX, NY));
const PRESSURE_TOL = 0;

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
  decay?: number;
  build: () => SceneSpec;
}

const SCENES: SceneDef[] = [
  { name: 'karman', ownDye: true, build: () => karmanChannel(sim.g) },
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
 * 'none'  empty, for scenes that inject their own dye.
 */
const TRACERS: { name: string; seed?: DyeSeed }[] = [
  { name: 'triad', seed: addDyeTriad },
  { name: 'mono', seed: addDyeMono },
  { name: 'none' },
];

const sim = new Simulation(NX, NY, { pressureIters: PRESSURE_ITERS, tol: PRESSURE_TOL });
const fieldView = new FieldView(sim.g);

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

function restart(): void {
  const def = SCENES[sceneIndex];
  const spec = def.build();
  sim.params.dyeDecay = def.decay ?? 0;
  // The selected tracer always applies, so the readout can never claim one the
  // sim is not running; `?? spec.dye` keeps a scene's own seed when it is
  // 'none'. `ownDye` only picks the DEFAULT on entry, in nextScene.
  sim.reset({ ...spec, dye: TRACERS[tracerIndex].seed ?? spec.dye });
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

const onClick = (id: string, run: () => void): void =>
  document.querySelector<HTMLButtonElement>(`#${id}`)!.addEventListener('click', run);
onClick('restart', restart);
onClick('toggleView', toggleView);
onClick('nextScene', nextScene);
onClick('nextTracer', nextTracer);

window.addEventListener('keydown', (e) => {
  if (e.key === 'r' || e.key === 'R') restart();
  if (e.key === 'd' || e.key === 'D') toggleView();
  if (e.key === 's' || e.key === 'S') nextScene();
  if (e.key === 't' || e.key === 'T') nextTracer();
});
window.addEventListener('resize', fitCanvas);

fitCanvas();
restart();

function frame(): void {
  sim.step();
  const { maxSpeed, divMax } = fieldView.draw(ctx, sim, VIEWS[viewIndex]);

  readout.textContent =
    `scene ${SCENES[sceneIndex].name} (S)   view ${VIEWS[viewIndex]} (D)   ` +
    `dye ${TRACERS[tracerIndex].name} (T)   grid ${sim.g.nx}x${sim.g.ny}\n` +
    `t ${sim.time.toFixed(2)}   max speed ${maxSpeed.toFixed(3)}   ` +
    `dt ${sim.dt.toExponential(2)} (CFL ${sim.cfl.toFixed(2)})   ` +
    `SOR ${sim.iters} sweeps   ` +
    `div residual ±${divMax.toExponential(1)}`;

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
