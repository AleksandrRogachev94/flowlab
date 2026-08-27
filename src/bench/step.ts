/**
 * Headless phase timing for one karman step, at the browser's real settings.
 *
 * Exists because "the pressure solve is the slow part" was an assumption, and
 * porting the wrong kernel to WebGPU is a week spent for nothing. Run it
 * BEFORE and AFTER any optimisation:
 *
 *   node src/bench/step.ts [nx] [ny] [steps]
 *
 * NOTE the solver: this leaves Simulation's default, which is SOR, while the
 * browser's CPU engine runs cpu-mg. The two therefore print very different
 * step shapes — pressure is ~58% here and ~1% there — so a number from this
 * file is comparable to another number from this file, not to the overlay.
 * The browser's own baseline is tabulated in main.ts above QUALITIES.
 *
 * Node cannot run WebGPU, so this measures the CPU path only — which is
 * exactly what it is for. It is the baseline the GPU solver has to beat, and
 * the same Profiler drives the in-browser overlay, so the two are directly
 * comparable rather than two different measurements of "a frame".
 */
import { Simulation } from '../core/simulation.ts';
import { karmanChannel } from '../scenes/karman.ts';

const nx = Number(process.argv[2] ?? 320);
const ny = Number(process.argv[3] ?? 180);
const steps = Number(process.argv[4] ?? 60);

// Mirrors main.ts: a fixed sweep budget, not a tolerance.
const sim = new Simulation(nx, ny, {
  scheme: 'macCormack',
  pressureIters: Math.round(0.15 * Math.max(nx, ny)),
  omega: 1.6, // see PRESSURE_OMEGA in main.ts — a fixed budget wants a lower omega
  tol: 0,
});
sim.reset(karmanChannel(sim.g));

// Warm-up: the first steps run in the interpreter before the JIT has tiered
// them up, and they are 5-10x slower. Averaging them in would flatter whatever
// runs later and understate whatever runs first.
for (let n = 0; n < 15; n++) await sim.step();
sim.perf.reset();

const t0 = performance.now();
for (let n = 0; n < steps; n++) await sim.step();
const wall = performance.now() - t0;

const total = sim.perf.totalEma();
console.log(`\n${nx}x${ny}  macCormack  ${sim.params.pressureIters} sweeps  ${steps} steps`);
console.log(
  `wall ${(wall / steps).toFixed(1)} ms/step  (${((1000 * steps) / wall).toFixed(1)} steps/s)\n`,
);
for (let k = 0; k < sim.perf.count; k++) {
  const ms = sim.perf.ema[k];
  const pct = (100 * ms) / total;
  console.log(
    `  ${sim.perf.labels[k].padEnd(10)} ${ms.toFixed(2).padStart(7)} ms  ${pct.toFixed(1).padStart(5)}%  ` +
      `${'#'.repeat(Math.round(pct / 2))}`,
  );
}
console.log(`  ${'TOTAL'.padEnd(10)} ${total.toFixed(2).padStart(7)} ms`);
