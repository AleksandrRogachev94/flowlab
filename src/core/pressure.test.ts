import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Cell, createFields, createGrid, idxP } from './grid.ts';
import { gaussSeidelSweep } from './pressure.ts';

/**
 * The diagonal-coefficient rule, hand-derived (PLAN.md §6, Step 3):
 *
 *   count * p[i,j] - sum(p over FLUID neighbours) = -scale * div[i,j]
 *   count = number of NON-SOLID neighbours
 *
 * Setup is chosen so one sweep gives a closed-form answer. Divergence is zero
 * everywhere except the probe cell, so every cell the forward sweep visits
 * BEFORE it stays at p = 0 (pGS = (0 - 0)/count = 0), and every cell after it
 * has not been touched yet. The probe's neighbour sum is therefore exactly 0
 * and only `count` is left to check.
 */
const SCALE = 7; // not 1, so a dropped `scale` cannot hide
const DIV = 3; // ditto
const PROBE = { i: 2, j: 1 };
const NEIGHBOUR = { i: 1, j: 1 }; // the probe's LEFT neighbour

function sweepOnce(neighbourLabel: Cell): Float64Array {
  const g = createGrid(4, 4, 0.25);
  const f = createFields(g, Float64Array);
  const div = new Float64Array(f.p.length);

  f.label[idxP(g, NEIGHBOUR.i, NEIGHBOUR.j)] = neighbourLabel;
  div[idxP(g, PROBE.i, PROBE.j)] = DIV;
  gaussSeidelSweep(g, f.p, div, f.label, SCALE, 1.0);

  return f.p as Float64Array;
}

test('an interior FLUID neighbour gives the full count of 4', () => {
  const p = sweepOnce(Cell.Fluid);
  assert.ok(
    Math.abs(p[2 + 1 * 4] - (-SCALE * DIV) / 4) < 1e-15,
    `expected -scale*div/4, got ${p[2 + 1 * 4]}`,
  );
});

test('a SOLID neighbour drops OUT of the count: /3, not /4', () => {
  // This is the whole rule. A solid contributes no equation and no coupling,
  // so it is not counted — the same reason an out-of-domain cell is not.
  const p = sweepOnce(Cell.Solid);
  assert.ok(
    Math.abs(p[2 + 1 * 4] - (-SCALE * DIV) / 3) < 1e-15,
    `expected -scale*div/3, got ${p[2 + 1 * 4]}`,
  );
});

test('an AIR neighbour STAYS in the count, contributing p = 0', () => {
  // The distinction that makes Air an outflow rather than a wall: it is a
  // Dirichlet value, so it couples (count += 1) but adds nothing to the sum.
  // Getting this wrong turns an open outlet back into a solid boundary.
  const p = sweepOnce(Cell.Air);
  assert.ok(
    Math.abs(p[2 + 1 * 4] - (-SCALE * DIV) / 4) < 1e-15,
    `expected -scale*div/4, got ${p[2 + 1 * 4]}`,
  );
});

test('the sweep never writes an Air cell — that is what pins p = 0 there', () => {
  const g = createGrid(4, 4, 0.25);
  const f = createFields(g, Float64Array);
  const div = new Float64Array(f.p.length);

  const air = idxP(g, 1, 1);
  f.label[air] = Cell.Air;
  div.fill(DIV); // including the Air cell, which must be ignored anyway

  gaussSeidelSweep(g, f.p, div, f.label, SCALE, 1.0);

  assert.equal(f.p[air], 0, 'Air cell was solved instead of held at the Dirichlet value');
});

test('a corner cell counts 2 neighbours — out-of-domain is solid', () => {
  const g = createGrid(4, 4, 0.25);
  const f = createFields(g, Float64Array);
  const div = new Float64Array(f.p.length);
  div[idxP(g, 0, 0)] = DIV; // the first cell the forward sweep visits

  gaussSeidelSweep(g, f.p, div, f.label, SCALE, 1.0);

  assert.ok(
    Math.abs(f.p[idxP(g, 0, 0)] - (-SCALE * DIV) / 2) < 1e-15,
    `expected -scale*div/2, got ${f.p[idxP(g, 0, 0)]}`,
  );
});
