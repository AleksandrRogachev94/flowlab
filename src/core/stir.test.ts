import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Cell, idxP, idxU, idxV } from './grid.ts';
import { Simulation } from './simulation.ts';
import { applyStir } from './stir.ts';
import { karmanChannel } from '../scenes/karman.ts';

const stir = { x: 0.9, y: 0.5, r: 0.06, dx: 0.8, dy: -0.4 };

test('a stir pushes the fluid along the drag, and only near the brush', () => {
  const sim = new Simulation(96, 48);
  sim.reset();
  applyStir(sim.g, sim.f.u, sim.f.v, sim.f.label, stir);

  // At the brush centre the increment is the full drag; the sign is the drag's.
  const i = Math.round(stir.x / sim.g.h);
  const j = Math.round(stir.y / sim.g.h - 0.5);
  assert.ok(sim.f.u[idxU(sim.g, i, j)] > 0.5 * stir.dx, 'no push at the centre');
  assert.ok(sim.f.v[idxV(sim.g, i, j)] < 0.5 * stir.dy, 'v did not follow the drag');

  // Four radii away it is numerically nothing: a stir is local or it is a wind.
  const far = Math.round((stir.x - 4 * stir.r) / sim.g.h);
  assert.ok(Math.abs(sim.f.u[idxU(sim.g, far, j)]) < 1e-3, 'the brush reached too far');
});

test('a stir never edits prescribed boundary data', () => {
  const sim = new Simulation(96, 48);
  // A channel: u[0,j] is the inlet, and dragging over it must not rewrite the
  // free stream. The brush is placed deliberately ON the inlet.
  sim.reset(karmanChannel(sim.g));
  const before = [...Array(sim.g.ny).keys()].map((j) => sim.f.u[idxU(sim.g, 0, j)]);
  applyStir(sim.g, sim.f.u, sim.f.v, sim.f.label, { ...stir, x: 0, y: 0.5 });
  for (let j = 0; j < sim.g.ny; j++) {
    assert.equal(sim.f.u[idxU(sim.g, 0, j)], before[j], `inlet u[0,${j}] was stirred`);
  }
});

test('a stir does not push through a solid wall', () => {
  const sim = new Simulation(96, 48);
  sim.reset(karmanChannel(sim.g));
  // Centred on the cylinder, which sits at cx 0.45.
  applyStir(sim.g, sim.f.u, sim.f.v, sim.f.label, { ...stir, x: 0.45, y: 0.5 });

  for (let j = 0; j < sim.g.ny; j++) {
    for (let i = 0; i <= sim.g.nx; i++) {
      const l = (a: number): boolean =>
        a >= 0 && a < sim.g.nx && sim.f.label[idxP(sim.g, a, j)] === Cell.Solid;
      if (l(i - 1) || l(i)) assert.equal(sim.f.u[idxU(sim.g, i, j)], 0, `u[${i},${j}] stirred`);
    }
  }
});

test('the projection absorbs a stir: no divergence is left behind', async () => {
  const sim = new Simulation(96, 48, { pressureIters: 400 });
  sim.reset(karmanChannel(sim.g));
  for (let n = 0; n < 20; n++) await sim.step();

  const rms = (): number => {
    let sum = 0;
    let count = 0;
    for (let k = 0; k < sim.div.length; k++) {
      if (sim.f.label[k] !== Cell.Fluid) continue;
      sum += sim.div[k] * sim.div[k];
      count += 1;
    }
    return Math.sqrt(sum / count);
  };
  const baseline = rms();

  sim.stir = stir;
  await sim.step();
  assert.equal(sim.stir, null, 'step() did not consume the stir');

  // RECOVERY is the property, not the residual on the stirred step itself.
  // A hard shove is a large spike of incoming divergence and the solve stops
  // on a budget, so that one step ends less converged than a quiet one —
  // measured at 1.7e-2 against a 4.8e-3 baseline, with the sweep count pinned
  // at its cap. That is the solver running out of iterations, not the stir
  // leaving anything permanent, and the two look identical for exactly one
  // frame. What separates them is whether it comes back, so that is what is
  // asserted: leave the divergence in the field and it would compound.
  for (let n = 0; n < 3; n++) await sim.step();
  assert.ok(
    rms() < 2 * baseline,
    `stir left divergence behind: ${rms()} against a ${baseline} baseline`,
  );
  assert.ok(sim.f.u.every(Number.isFinite), 'u went non-finite after a stir');
});

test('a stir actually moves the fluid — energy goes up', async () => {
  const sim = new Simulation(96, 48, { pressureIters: 400 });
  sim.reset();
  // An indexed loop, not reduce: f.u is FieldArray, a union of Float32Array
  // and Float64Array, and the union's reduce overloads do not unify.
  const energy = (): number => {
    let sum = 0;
    for (let k = 0; k < sim.f.u.length; k++) sum += sim.f.u[k] * sim.f.u[k];
    return sum;
  };
  await sim.step();
  const before = energy();
  sim.stir = stir;
  await sim.step();
  assert.ok(energy() > before + 0.01, `stir added no energy (${before} -> ${energy()})`);
});
