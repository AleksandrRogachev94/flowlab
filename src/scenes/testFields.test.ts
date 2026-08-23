import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGrid, createFields, idxU, idxV } from '../core/grid.ts';
import { computeDivergence } from '../core/divergence.ts';
import {
  addRotational,
  addGradient,
  addVortexPair,
  addVortexCluster,
  addDyeTriad,
  addDyeMono,
} from './testFields.ts';

const N = 32;
const setup = () => {
  const g = createGrid(N, N, 1 / N); // unit square
  const f = createFields(g, Float64Array);
  const div = new Float64Array(f.p.length);
  return { g, f, div };
};

test('curl(psi) is divergence-free to MACHINE precision, not just O(h^2)', () => {
  const { g, f, div } = setup();
  addRotational(g, f.u, f.v);
  computeDivergence(g, f.u, f.v, div);

  let worst = 0;
  for (const d of div) worst = Math.max(worst, Math.abs(d));
  assert.ok(worst < 1e-9, `max |div| = ${worst}, expected machine-zero`);
});

test('grad(phi) is genuinely divergent (otherwise the test is vacuous)', () => {
  const { g, f, div } = setup();
  addGradient(g, f.u, f.v);
  computeDivergence(g, f.u, f.v, div);

  let worst = 0;
  for (const d of div) worst = Math.max(worst, Math.abs(d));
  assert.ok(worst > 1, `max |div| = ${worst}, expected a strongly divergent field`);
});

test('both fields satisfy compatibility: total divergence sums to ~0', () => {
  const { g, f, div } = setup();
  addRotational(g, f.u, f.v);
  addGradient(g, f.u, f.v);
  computeDivergence(g, f.u, f.v, div);

  let sum = 0;
  for (const d of div) sum += d;
  assert.ok(Math.abs(sum) < 1e-9, `total divergence = ${sum}, expected ~0`);
});

test('normal velocity vanishes on all four walls', () => {
  const { g, f } = setup();
  addRotational(g, f.u, f.v);
  addGradient(g, f.u, f.v);

  for (let j = 0; j < g.ny; j++) {
    assert.ok(Math.abs(f.u[idxU(g, 0, j)]) < 1e-12, `left wall leaks at j=${j}`);
    assert.ok(Math.abs(f.u[idxU(g, g.nx, j)]) < 1e-12, `right wall leaks at j=${j}`);
  }
  for (let i = 0; i < g.nx; i++) {
    assert.ok(Math.abs(f.v[idxV(g, i, 0)]) < 1e-12, `bottom wall leaks at i=${i}`);
    assert.ok(Math.abs(f.v[idxV(g, i, g.ny)]) < 1e-12, `top wall leaks at i=${i}`);
  }
});

test('addVortexPair is divergence-free to MACHINE precision too', () => {
  const { g, f, div } = setup();
  addVortexPair(g, f.u, f.v);
  computeDivergence(g, f.u, f.v, div);

  let worst = 0;
  let total = 0;
  for (const d of div) {
    worst = Math.max(worst, Math.abs(d));
    total += d;
  }
  assert.ok(worst < 1e-9, `max |div| = ${worst}, expected machine-zero`);
  // Compatibility telescopes to exactly 0 for any psi, so the all-Neumann
  // pressure solve stays solvable even though the blobs do not vanish
  // identically at the walls.
  assert.ok(Math.abs(total) < 1e-9, `total divergence = ${total}, expected ~0`);
});

test('addVortexPair leaks only negligibly through the walls', () => {
  const { g, f } = setup();
  addVortexPair(g, f.u, f.v);

  // Gaussian tails, not an exact zero like addRotational's sin() — so the
  // tolerance is 1e-5, not 1e-12. Measured max is ~8e-7 at N=64.
  let worst = 0;
  for (let j = 0; j < g.ny; j++) {
    worst = Math.max(worst, Math.abs(f.u[idxU(g, 0, j)]), Math.abs(f.u[idxU(g, g.nx, j)]));
  }
  for (let i = 0; i < g.nx; i++) {
    worst = Math.max(worst, Math.abs(f.v[idxV(g, i, 0)]), Math.abs(f.v[idxV(g, i, g.ny)]));
  }
  assert.ok(worst < 1e-5, `max wall-normal velocity = ${worst}`);
});

test('addVortexPair amp is calibrated to mean peak speed', () => {
  const { g, f } = setup();
  addVortexPair(g, f.u, f.v, 1);

  let peak = 0;
  for (const x of f.u) peak = Math.max(peak, Math.abs(x));
  for (const x of f.v) peak = Math.max(peak, Math.abs(x));
  // The prefactor divides out sqrt(2)/sigma*exp(-1/2); discretization keeps
  // it a little under 1, so this pins the calibration without over-fitting.
  assert.ok(peak > 0.9 && peak < 1.05, `amp=1 gave peak speed ${peak}, expected ~1`);
});

test('curl and grad do not cancel each other out', () => {
  const { g, f } = setup();
  addRotational(g, f.u, f.v);
  addGradient(g, f.u, f.v);

  let maxU = 0;
  for (const x of f.u) maxU = Math.max(maxU, Math.abs(x));
  assert.ok(maxU > 1, `max |u| = ${maxU}; matching frequencies would cancel to 0`);
});

test('addVortexCluster is divergence-free and reproducible', () => {
  const { g, f, div } = setup();
  addVortexCluster(g, f.u, f.v);
  computeDivergence(g, f.u, f.v, div);

  let worst = 0;
  for (const d of div) worst = Math.max(worst, Math.abs(d));
  assert.ok(worst < 1e-9, `max |div| = ${worst}, expected machine-zero`);

  // Comparing two advection schemes is only meaningful from an identical
  // initial condition, so the seeding must not depend on Math.random().
  const b = createFields(g, Float64Array);
  addVortexCluster(g, b.u, b.v);
  for (let k = 0; k < f.u.length; k++) {
    assert.equal(f.u[k], b.u[k], `u[${k}] differs between identical seeds`);
  }
});

test('addVortexCluster leaks only negligibly through the walls', () => {
  const { g, f } = setup();
  addVortexCluster(g, f.u, f.v);

  let worst = 0;
  for (let j = 0; j < g.ny; j++) {
    worst = Math.max(worst, Math.abs(f.u[idxU(g, 0, j)]), Math.abs(f.u[idxU(g, g.nx, j)]));
  }
  for (let i = 0; i < g.nx; i++) {
    worst = Math.max(worst, Math.abs(f.v[idxV(g, i, 0)]), Math.abs(f.v[idxV(g, i, g.ny)]));
  }
  assert.ok(worst < 1e-4, `max wall-normal velocity = ${worst}`);
});

test('addDyeTriad seeds every channel, in a different place', () => {
  const { g, f } = setup();
  addDyeTriad(g, f.dye);

  // Each channel must carry a real disk. A loop that seeded only dye[0], or
  // stacked all three disks at the same centre, renders as plausible-looking
  // dye and is easy to miss by eye.
  const peaks = f.dye.map((c) => c.indexOf(Math.max(...c)));
  for (let c = 0; c < f.dye.length; c++) {
    assert.ok(Math.max(...f.dye[c]) > 0.99, `channel ${c} has no fully saturated cell`);
    for (let other = 0; other < c; other++) {
      assert.notEqual(peaks[c], peaks[other], `channels ${c} and ${other} peak in the same cell`);
    }
  }
});

test('addDyeTriad disks overlap, and stay in [0, 1] where they do', () => {
  const { g, f } = setup();
  addDyeTriad(g, f.dye);

  // The overlaps are the whole point of the triad — they are what show mixing.
  // But addDyeDisk ADDs, so a triad packed too tightly would push a channel
  // past 1 and clip against the fixed display range instead of blending.
  let mixed = 0;
  for (let k = 0; k < f.p.length; k++) {
    if (f.dye.filter((c) => c[k] > 0.01).length >= 2) mixed++;
    for (const c of f.dye) assert.ok(c[k] >= 0 && c[k] <= 1, `cell ${k} = ${c[k]} outside [0,1]`);
  }
  assert.ok(mixed > 0, 'no cell carries two dyes: the disks never overlap');
});

test('addDyeMono fills every channel identically, so it renders as grey', () => {
  const { g, f } = setup();
  addDyeMono(g, f.dye);

  // The whole premise of the mono tracer: three equal channels composite to
  // pure greyscale. Any per-channel difference here would tint the picture and
  // fake exactly the mixing signal the triad is supposed to have a monopoly on.
  for (let k = 0; k < f.p.length; k++) {
    assert.equal(f.dye[1][k], f.dye[0][k], `channel 1 differs at cell ${k}`);
    assert.equal(f.dye[2][k], f.dye[0][k], `channel 2 differs at cell ${k}`);
  }
  assert.ok(Math.max(...f.dye[0]) > 0.99, 'no fully saturated cell');
});
