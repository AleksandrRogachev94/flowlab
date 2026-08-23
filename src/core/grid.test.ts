import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGrid, createFields, idxP, idxU, idxV, sampleP, Cell, DYE_CHANNELS } from './grid.ts';

const H = 0.5; // deliberately not 1.0, so a missing "/h" can't hide

test('array lengths match the MAC layout', () => {
  const g = createGrid(4, 4, 1);
  const f = createFields(g, Float64Array);
  assert.equal(f.p.length, 16); // nx * ny
  assert.equal(f.u.length, 20); // (nx + 1) * ny
  assert.equal(f.v.length, 20); // nx * (ny + 1)
  assert.equal(f.label.length, 16);
  assert.equal(f.dye.length, DYE_CHANNELS);
  for (const c of f.dye) assert.equal(c.length, 16); // cell-centered, like p
});

test('dye channels are independent buffers, not aliases of one', () => {
  const g = createGrid(4, 4, 1);
  const f = createFields(g, Float64Array);

  // A factory that returned the same array DYE_CHANNELS times would advect
  // correctly and render as pure grey forever — a confusing way to discover
  // an allocation bug.
  f.dye[0][5] = 1;
  assert.equal(f.dye[1][5], 0);
  assert.equal(f.dye[2][5], 0);
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

/** A distinct value per cell, so any wrong neighbour is a wrong answer. */
function ramp(g: ReturnType<typeof createGrid>): Float64Array {
  const q = new Float64Array(g.nx * g.ny);
  for (let j = 0; j < g.ny; j++) for (let i = 0; i < g.nx; i++) q[idxP(g, i, j)] = i + 10 * j;
  return q;
}

test('sampleP is exact at cell centers', () => {
  const g = createGrid(4, 4, H);
  const q = ramp(g);

  // The tightest check on the half-cell offset: at a cell's OWN center the
  // interpolation weights must collapse onto that single cell. Dropping the
  // -0.5 puts the sample halfway between two cells instead, which averages
  // neighbours everywhere and reads downstream as dye drifting diagonally.
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i < g.nx; i++) {
      const got = sampleP(g, q, (i + 0.5) * H, (j + 0.5) * H);
      assert.ok(Math.abs(got - (i + 10 * j)) < 1e-12, `center (${i},${j}) sampled ${got}`);
    }
  }
});

test('sampleP averages neighbours halfway between centers', () => {
  const g = createGrid(4, 4, H);
  const q = ramp(g);

  // Midway between (1,1) and (2,1): the mean of 11 and 12. Fixes the x offset
  // in one direction only, so it also pins down the SIGN of the shift.
  assert.ok(Math.abs(sampleP(g, q, 2 * H, 1.5 * H) - 11.5) < 1e-12);
  // ... and between (1,1) and (1,2): the mean of 11 and 21.
  assert.ok(Math.abs(sampleP(g, q, 1.5 * H, 2 * H) - 16) < 1e-12);
});

test('sampleP clamps outside the domain instead of extrapolating', () => {
  const g = createGrid(4, 4, H);
  const q = ramp(g);

  // Backtraces routinely land outside a closed box near the walls. Clamping
  // holds the edge value; extrapolating would invent dye out past the wall and
  // break the bound that keeps semi-Lagrangian stable.
  assert.equal(sampleP(g, q, -5 * H, -5 * H), 0); // corner cell (0,0)
  assert.equal(sampleP(g, q, 9 * H, 9 * H), 33); // corner cell (3,3) = 3 + 30
});
