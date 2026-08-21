import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGrid, createFields, idxU, idxV } from './grid.ts';
import { computeDivergence } from './divergence.ts';

const H = 0.5; // deliberately not 1.0, so a missing "/h" can't hide

test('uniform flow is divergence-free', () => {
  const g = createGrid(4, 4, H);
  const f = createFields(g, Float64Array);
  f.u.fill(1.0); // constant rightward flow, v stays 0

  computeDivergence(g, f.u, f.v, f.p);
  for (let k = 0; k < f.p.length; k++) {
    assert.equal(f.p[k], 0, `cell ${k} should be divergence-free`);
  }
});

test('u growing linearly in x gives divergence 1 in EVERY cell', () => {
  const g = createGrid(4, 4, H);
  const f = createFields(g, Float64Array);
  // u[i,j] = i*h  =>  (u[i+1,j] - u[i,j]) / h  ==  h/h  ==  1
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i <= g.nx; i++) f.u[idxU(g, i, j)] = i * g.h;
  }

  computeDivergence(g, f.u, f.v, f.p);
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i < g.nx; i++) {
      assert.ok(
        Math.abs(f.p[i + j * g.nx] - 1) < 1e-12,
        `cell (${i},${j}) = ${f.p[i + j * g.nx]}, expected 1`,
      );
    }
  }
});

test('v growing linearly in y gives divergence 1 in EVERY cell', () => {
  const g = createGrid(4, 4, H);
  const f = createFields(g, Float64Array);
  // exercises idxV's stride independently of idxU's
  for (let j = 0; j <= g.ny; j++) {
    for (let i = 0; i < g.nx; i++) f.v[idxV(g, i, j)] = j * g.h;
  }

  computeDivergence(g, f.u, f.v, f.p);
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i < g.nx; i++) {
      assert.ok(
        Math.abs(f.p[i + j * g.nx] - 1) < 1e-12,
        `cell (${i},${j}) = ${f.p[i + j * g.nx]}, expected 1`,
      );
    }
  }
});
