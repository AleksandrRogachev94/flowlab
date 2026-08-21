import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGrid, createFields, idxU, idxV } from '../core/grid.ts';
import { computeDivergence } from '../core/divergence.ts';
import { addRotational, addGradient } from './testFields.ts';

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

test('curl and grad do not cancel each other out', () => {
  const { g, f } = setup();
  addRotational(g, f.u, f.v);
  addGradient(g, f.u, f.v);

  let maxU = 0;
  for (const x of f.u) maxU = Math.max(maxU, Math.abs(x));
  assert.ok(maxU > 1, `max |u| = ${maxU}; matching frequencies would cancel to 0`);
});
