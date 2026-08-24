import { test } from 'node:test';
import assert from 'node:assert/strict';
import { isSolidCell } from '../core/boundaries.ts';
import { Cell, idxP, idxU, idxV } from '../core/grid.ts';
import { Simulation } from '../core/simulation.ts';
import { karmanChannel } from './karman.ts';

/** In-domain AND labeled Solid — the same rule commitLabels enforces. */
const solid = (sim: Simulation, i: number, j: number): boolean =>
  isSolidCell(sim.g, sim.f.label, i, j);

function run(steps: number): Simulation {
  // Stubbier than the real preset, but the same code path.
  const sim = new Simulation(96, 48, { pressureIters: 400 });
  sim.reset(karmanChannel(sim.g, { diameter: 0.25, cx: 0.5 }));
  for (let n = 0; n < steps; n++) sim.step();
  return sim;
}

test('the domain is one unit tall and nx/ny wide', () => {
  const sim = new Simulation(96, 48);
  assert.ok(Math.abs(sim.g.ny * sim.g.h - 1) < 1e-15, 'height should be exactly 1');
  assert.ok(Math.abs(sim.g.nx * sim.g.h - 2) < 1e-15, '96x48 should be 2 units wide');
});

test('the cylinder rasterizes ASYMMETRICALLY, or the wake never sheds', () => {
  // The quarter-cell offset. Nothing else in the scene is asymmetric, so if
  // the staircase is not either, there is nothing for the instability to
  // amplify. Note a HALF-cell offset would pass no better than zero: cell
  // centres sit at (j+0.5)h, so both land symmetrically.
  const sim = new Simulation(96, 48);
  sim.reset(karmanChannel(sim.g));

  let above = 0;
  let below = 0;
  for (let j = 0; j < sim.g.ny; j++) {
    for (let i = 0; i < sim.g.nx; i++) {
      if (sim.f.label[idxP(sim.g, i, j)] !== Cell.Solid) continue;
      if ((j + 0.5) * sim.g.h > 0.5) above += 1;
      else below += 1;
    }
  }
  assert.ok(above > 0 && below > 0, 'disk did not straddle the axis');
  assert.notEqual(above, below, 'the staircase is symmetric — no shedding seed');
});

test('no flow passes through the cylinder', () => {
  const sim = run(40);
  for (let j = 0; j < sim.g.ny; j++) {
    for (let i = 0; i <= sim.g.nx; i++) {
      if (!solid(sim, i - 1, j) && !solid(sim, i, j)) continue;
      assert.equal(sim.f.u[idxU(sim.g, i, j)], 0, `u[${i},${j}] leaked through the solid`);
    }
  }
  for (let j = 0; j <= sim.g.ny; j++) {
    for (let i = 0; i < sim.g.nx; i++) {
      if (!solid(sim, i, j - 1) && !solid(sim, i, j)) continue;
      assert.equal(sim.f.v[idxV(sim.g, i, j)], 0, `v[${i},${j}] leaked through the solid`);
    }
  }
});

test('the projection still clears divergence from the fluid around an obstacle', () => {
  const sim = run(60);
  assert.ok(sim.f.u.every(Number.isFinite), 'u went non-finite');

  // RMS over FLUID cells, matching what solvePressure's tolerance is stated
  // against. The Air outlet is SUPPOSED to be divergent — it is where mass
  // leaves — and a solid's divergence is whatever its frozen faces sum to.
  //
  // RMS and not max-norm, for the same reason the solver's stopping test uses
  // RMS: around a staircased obstacle the worst single cell sits at a corner
  // and runs ~10x the RMS (measured: 5.5e-2 max against 5.5e-3 rms at 96x48),
  // so a max-norm assertion would be testing one corner, not the projection.
  let sumSq = 0;
  let count = 0;
  for (let k = 0; k < sim.div.length; k++) {
    if (sim.f.label[k] !== Cell.Fluid) continue;
    sumSq += sim.div[k] * sim.div[k];
    count += 1;
  }
  assert.ok(Math.sqrt(sumSq / count) < 1e-2, `fluid-cell residual rms ${Math.sqrt(sumSq / count)}`);
  assert.ok(sim.iters < 400, `solve pinned at the cap (${sim.iters})`);
});
