import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGrid, createFields, idxP, idxU, idxV, Cell } from './grid.ts';

test('array lengths match the MAC layout', () => {
  const g = createGrid(4, 4, 1);
  const f = createFields(g, Float64Array);
  assert.equal(f.p.length, 16); // nx * ny
  assert.equal(f.u.length, 20); // (nx + 1) * ny
  assert.equal(f.v.length, 20); // nx * (ny + 1)
  assert.equal(f.label.length, 16);
});

test('fields default to all-Fluid', () => {
  const g = createGrid(4, 4, 1);
  const f = createFields(g, Float64Array);
  assert.ok(f.label.every((v) => v === Cell.Fluid));
});

// Hand-derived on a 4x4 grid for interior cell (2,2):
//   left face:   idxU(2,2) = 2 + 2*5 = 12
//   right face:  idxU(3,2) = 3 + 2*5 = 13
//   bottom face: idxV(2,2) = 2 + 2*4 = 10
//   top face:    idxV(2,3) = 2 + 3*4 = 14
test('face indices around one interior cell match the diagram by hand', () => {
  const g = createGrid(4, 4, 1);
  assert.equal(idxP(g, 2, 2), 10);
  assert.equal(idxU(g, 2, 2), 12);
  assert.equal(idxU(g, 3, 2), 13);
  assert.equal(idxV(g, 2, 2), 10);
  assert.equal(idxV(g, 2, 3), 14);
});
