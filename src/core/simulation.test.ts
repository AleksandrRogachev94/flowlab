import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Simulation } from './simulation.ts';
import { addVortexCluster, addVortexPair } from '../scenes/testFields.ts';

test('a long run stays finite and keeps removing divergence', () => {
  const sim = new Simulation(32, { pressureIters: 40 });
  sim.reset(addVortexPair);

  for (let n = 0; n < 200; n++) sim.step();

  assert.ok(sim.f.u.every(Number.isFinite), 'u went non-finite');
  assert.ok(sim.f.v.every(Number.isFinite), 'v went non-finite');
  let worst = 0;
  for (const d of sim.div) worst = Math.max(worst, Math.abs(d));
  assert.ok(worst < 1e-2, `residual after 200 steps = ${worst}`);
  assert.ok(sim.time > 0, 'clock did not advance');
});

test('dt respects the CFL target and the dtMax cap', () => {
  const sim = new Simulation(32, { cflTarget: 1.0, dtMax: 1 / 30 });
  sim.reset(addVortexPair);

  for (let n = 0; n < 50; n++) {
    sim.step();
    assert.ok(sim.dt <= 1 / 30 + 1e-12, `dt ${sim.dt} exceeded dtMax`);
    // The cap can hold dt BELOW the CFL target, but never above it.
    assert.ok(sim.cfl <= 1.0 + 1e-6, `CFL ${sim.cfl} exceeded target`);
  }
});

test('reset makes a run reproducible — required for comparing schemes', () => {
  const a = new Simulation(24, { pressureIters: 20 });
  const b = new Simulation(24, { pressureIters: 20 });
  a.reset(addVortexCluster);
  b.reset(addVortexCluster);

  for (let n = 0; n < 25; n++) {
    a.step();
    b.step();
  }
  for (let k = 0; k < a.f.u.length; k++) {
    assert.equal(a.f.u[k], b.f.u[k], `u[${k}] diverged between identical runs`);
  }
  assert.equal(a.time, b.time);
});
