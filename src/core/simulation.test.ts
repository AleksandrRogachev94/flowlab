import { test } from 'node:test';
import assert from 'node:assert/strict';
import { optimalOmega, Simulation } from './simulation.ts';
import { addVortexCluster, addVortexPair } from '../scenes/testFields.ts';

test('a long run stays finite and keeps removing divergence', () => {
  const sim = new Simulation(32, 32, { pressureIters: 40 });
  sim.reset({ seed: addVortexPair });

  for (let n = 0; n < 200; n++) sim.step();

  assert.ok(sim.f.u.every(Number.isFinite), 'u went non-finite');
  assert.ok(sim.f.v.every(Number.isFinite), 'v went non-finite');
  let worst = 0;
  for (const d of sim.div) worst = Math.max(worst, Math.abs(d));
  assert.ok(worst < 1e-2, `residual after 200 steps = ${worst}`);
  assert.ok(sim.time > 0, 'clock did not advance');
});

test('dt respects the CFL target and the dtMax cap', () => {
  const sim = new Simulation(32, 32, { cflTarget: 1.0, dtMax: 1 / 30 });
  sim.reset({ seed: addVortexPair });

  for (let n = 0; n < 50; n++) {
    sim.step();
    assert.ok(sim.dt <= 1 / 30 + 1e-12, `dt ${sim.dt} exceeded dtMax`);
    // The cap can hold dt BELOW the CFL target, but never above it.
    assert.ok(sim.cfl <= 1.0 + 1e-6, `CFL ${sim.cfl} exceeded target`);
  }
});

test('reset makes a run reproducible — required for comparing schemes', () => {
  const a = new Simulation(24, 24, { pressureIters: 20 });
  const b = new Simulation(24, 24, { pressureIters: 20 });
  a.reset({ seed: addVortexCluster });
  b.reset({ seed: addVortexCluster });

  for (let n = 0; n < 25; n++) {
    a.step();
    b.step();
  }
  for (let k = 0; k < a.f.u.length; k++) {
    assert.equal(a.f.u[k], b.f.u[k], `u[${k}] diverged between identical runs`);
  }
  assert.equal(a.time, b.time);
});

test('dye decay is exponential in TIME, not per step', () => {
  // Zero velocity: advection is the identity at cell centres, so the only
  // thing acting on dye is the decay — and the answer is exact.
  const sim = new Simulation(16, 16, { dyeDecay: 2 });
  sim.reset({ dye: (g, dye) => dye[0].fill(1) });

  for (let n = 0; n < 30; n++) sim.step();

  const expected = Math.exp(-2 * sim.time);
  assert.ok(
    Math.abs(sim.f.dye[0][0] - expected) < 1e-12,
    `dye ${sim.f.dye[0][0]} after t=${sim.time}, expected ${expected}`,
  );
});

test('optimalOmega reduces to 2/(1+sin(pi/n)) on a square grid', () => {
  for (const n of [16, 64, 256]) {
    assert.ok(Math.abs(optimalOmega(n, n) - 2 / (1 + Math.sin(Math.PI / n))) < 1e-12);
  }
  // A channel is not square, and the aspect ratio genuinely moves the optimum.
  assert.ok(optimalOmega(512, 128) > optimalOmega(128, 128));
});
