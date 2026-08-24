import { Simulation, type DyeSeed, type DyeSource, type Seed } from './core/simulation.ts';
import { wallJet } from './scenes/emitters.ts';
import { addDyeMono, addDyeTriad, addVortexCluster, addVortexPair } from './scenes/testFields.ts';
import { FieldView, VIEWS } from './viz/fieldView.ts';

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const ctx = canvas.getContext('2d')!;
const readout = document.querySelector<HTMLPreElement>('#readout')!;

/**
 * 'dipole'  self-propelling pair — the clearest check that advection moves
 *           things the right way at all.
 * 'cluster' interacting vortices that stretch filaments and merge — the scene
 *           for comparing schemes. Deterministic, so two runs are comparable.
 * 'jet'     steady inflow through part of the left wall, dye injected at the
 *           nozzle — the first scene that is DRIVEN rather than left to decay.
 *           A confined jet entrains, so the box recirculates and a permanent
 *           source would saturate every cell; `decay` is what keeps it legible.
 *           Tune it as a DISTANCE: k = speed / fadeLength, so 0.5 at speed 1
 *           fades dye to 1/e after two domain widths of travel. That is
 *           resolution-independent — see SimulationParams.dyeDecay.
 */
const jet = wallJet();
const SCENES: { name: string; seed: Seed; source?: DyeSource; decay?: number }[] = [
  { name: 'dipole', seed: addVortexPair },
  { name: 'cluster', seed: addVortexCluster },
  { name: 'jet', seed: jet.seed, source: jet.source, decay: 0.5 },
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

const sim = new Simulation(128);
const fieldView = new FieldView(sim.g);

let sceneIndex = 0;
let viewIndex = 0;
let tracerIndex = 0;

function restart(): void {
  const scene = SCENES[sceneIndex];
  sim.params.dyeDecay = scene.decay ?? 0;
  sim.reset(scene.seed, TRACERS[tracerIndex].seed, scene.source);
}
function toggleView(): void {
  viewIndex = (viewIndex + 1) % VIEWS.length;
}
function nextScene(): void {
  sceneIndex = (sceneIndex + 1) % SCENES.length;
  // A scene with its own dye source seeds its own dye; layering a tracer
  // preset underneath it would only show up outside the source's band.
  if (SCENES[sceneIndex].source) {
    tracerIndex = TRACERS.findIndex((t) => t.name === 'none');
  }
  restart();
}
// Reseeding is the only way to change tracer: dye already in the domain was
// advected under the old seed and cannot be reinterpreted as the new one.
function nextTracer(): void {
  tracerIndex = (tracerIndex + 1) % TRACERS.length;
  restart();
}
restart();

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

function frame(): void {
  sim.step();
  const { maxSpeed, divMax } = fieldView.draw(ctx, sim, VIEWS[viewIndex]);

  readout.textContent =
    `scene: ${SCENES[sceneIndex].name} (S) | view: ${VIEWS[viewIndex]} (D) | ` +
    `dye: ${TRACERS[tracerIndex].name} (T)\n` +
    `t = ${sim.time.toFixed(2)} | max speed: ${maxSpeed.toFixed(4)} | ` +
    `dt = ${sim.dt.toExponential(2)} (CFL ${sim.cfl.toFixed(2)}) | ` +
    `SOR sweeps: ${sim.iters}/${sim.params.pressureIters} | ` +
    `div residual: ±${divMax.toExponential(1)}`;

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
