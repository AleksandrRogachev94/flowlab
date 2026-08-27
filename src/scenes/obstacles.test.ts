import { test } from 'node:test';
import assert from 'node:assert/strict';
import { isSolidCell } from '../core/boundaries.ts';
import { Cell, idxP, idxU, idxV } from '../core/grid.ts';
import { Simulation } from '../core/simulation.ts';
import { karmanChannel } from './karman.ts';
import { paintSolid } from './obstacles.ts';

/** A running channel flow, so the brush lands in moving fluid rather than in a
 *  field that happens to be zero where it paints. */
async function running(steps = 40): Promise<Simulation> {
  const sim = new Simulation(96, 48, { pressureIters: 400 });
  sim.reset(karmanChannel(sim.g, { diameter: 0.25, cx: 0.5 }));
  for (let n = 0; n < steps; n++) await sim.step();
  return sim;
}

test('paintSolid only converts Fluid, so it cannot close the outlet', async () => {
  const sim = await running(1);
  const airBefore = [...sim.f.label].filter((l) => l === Cell.Air).length;
  assert.ok(airBefore > 0, 'the channel should have an Air outlet to protect');

  // A stroke straight across the outlet column and the cylinder both.
  const w = sim.g.nx * sim.g.h;
  paintSolid(sim.g, sim.f.label, 0.2, 0.5, w, 0.5, 0.1);

  assert.equal(
    [...sim.f.label].filter((l) => l === Cell.Air).length,
    airBefore,
    'the brush ate Air cells — the outlet, and with it the only consistent BC',
  );
});

test('an obstacle painted into a running flow does not leak', async () => {
  const sim = await running();
  // Well downstream of the cylinder, in the wake, where the fluid is moving.
  const before = sim.f.u[idxU(sim.g, Math.round(0.9 / sim.g.h), sim.g.ny >> 1)];
  assert.ok(Math.abs(before) > 0.1, `expected moving fluid to paint into, got ${before}`);

  assert.ok(paintSolid(sim.g, sim.f.label, 0.85, 0.35, 0.95, 0.65, 0.03), 'painted nothing');
  sim.commitLabelEdits();

  for (let n = 0; n < 20; n++) await sim.step();

  const solid = (i: number, j: number): boolean => isSolidCell(sim.g, sim.f.label, i, j);
  for (let j = 0; j < sim.g.ny; j++) {
    for (let i = 0; i <= sim.g.nx; i++) {
      if (!solid(i - 1, j) && !solid(i, j)) continue;
      assert.equal(sim.f.u[idxU(sim.g, i, j)], 0, `u[${i},${j}] leaked through painted solid`);
    }
  }
  for (let j = 0; j <= sim.g.ny; j++) {
    for (let i = 0; i < sim.g.nx; i++) {
      if (!solid(i, j - 1) && !solid(i, j)) continue;
      assert.equal(sim.f.v[idxV(sim.g, i, j)], 0, `v[${i},${j}] leaked through painted solid`);
    }
  }
  assert.ok(sim.f.u.every(Number.isFinite), 'u went non-finite after painting');
});

test('the projection still converges around a painted obstacle', async () => {
  const sim = await running();
  paintSolid(sim.g, sim.f.label, 0.85, 0.35, 0.95, 0.65, 0.03);
  sim.commitLabelEdits();
  for (let n = 0; n < 40; n++) await sim.step();

  let sumSq = 0;
  let count = 0;
  for (let k = 0; k < sim.div.length; k++) {
    if (sim.f.label[k] !== Cell.Fluid) continue;
    sumSq += sim.div[k] * sim.div[k];
    count += 1;
  }
  const rms = Math.sqrt(sumSq / count);
  assert.ok(rms < 1e-2, `fluid-cell residual rms ${rms} around the painted body`);
});

test('a stroke sweeps a capsule, not a dotted line of disks', () => {
  const sim = new Simulation(96, 48);
  sim.reset(karmanChannel(sim.g));
  const row = sim.g.ny >> 1;
  paintSolid(sim.g, sim.f.label, 0.3, 0.25, 0.9, 0.25, 0.02);
  // Every cell along the painted span, at the stroke's height, must be solid:
  // a gap means the brush stamped endpoints instead of sweeping between them.
  const j = Math.floor(0.25 / sim.g.h);
  for (let i = Math.ceil(0.32 / sim.g.h); i < Math.floor(0.88 / sim.g.h); i++) {
    assert.equal(sim.f.label[idxP(sim.g, i, j)], Cell.Solid, `gap in the stroke at i=${i}`);
  }
  assert.notEqual(sim.f.label[idxP(sim.g, i0(sim), row)], Cell.Solid, 'stroke bled off its line');
});

/** A column well left of the stroke's start. */
function i0(sim: Simulation): number {
  return Math.floor(0.1 / sim.g.h);
}
