import { Simulation, type Seed } from './core/simulation.ts';
import { addVortexCluster, addVortexPair } from './scenes/testFields.ts';
import { FieldView, VIEWS } from './viz/fieldView.ts';

const canvas = document.querySelector<HTMLCanvasElement>('#view')!;
const ctx = canvas.getContext('2d')!;
const readout = document.querySelector<HTMLPreElement>('#readout')!;

/**
 * 'dipole'  self-propelling pair — the clearest check that advection moves
 *           things the right way at all.
 * 'cluster' interacting vortices that stretch filaments and merge — the scene
 *           for comparing schemes. Deterministic, so two runs are comparable.
 */
const SCENES: { name: string; seed: Seed }[] = [
  { name: 'dipole', seed: addVortexPair },
  { name: 'cluster', seed: addVortexCluster },
];

const sim = new Simulation(128);
const fieldView = new FieldView(sim.g);

let sceneIndex = 0;
let viewIndex = 0;

function restart(): void {
  sim.reset(SCENES[sceneIndex].seed);
}
function toggleView(): void {
  viewIndex = (viewIndex + 1) % VIEWS.length;
}
function nextScene(): void {
  sceneIndex = (sceneIndex + 1) % SCENES.length;
  restart();
}
restart();

const onClick = (id: string, run: () => void): void =>
  document.querySelector<HTMLButtonElement>(`#${id}`)!.addEventListener('click', run);
onClick('restart', restart);
onClick('toggleView', toggleView);
onClick('nextScene', nextScene);

window.addEventListener('keydown', (e) => {
  if (e.key === 'r' || e.key === 'R') restart();
  if (e.key === 'd' || e.key === 'D') toggleView();
  if (e.key === 's' || e.key === 'S') nextScene();
});

function frame(): void {
  sim.step();
  const { maxSpeed, divMax } = fieldView.draw(ctx, sim, VIEWS[viewIndex]);

  readout.textContent =
    `scene: ${SCENES[sceneIndex].name} (S) | view: ${VIEWS[viewIndex]} (D) | ` +
    `t = ${sim.time.toFixed(2)} | max speed: ${maxSpeed.toFixed(4)}\n` +
    `dt = ${sim.dt.toExponential(2)} (CFL ${sim.cfl.toFixed(2)}) | ` +
    `SOR sweeps: ${sim.iters}/${sim.params.pressureIters} | ` +
    `div residual: ±${divMax.toExponential(1)}`;

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
