import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Cell, createFields, createGrid, idxP } from './grid.ts';
import { gaussSeidelSweep, rmsRemainingDivergence, solvePressure } from './pressure.ts';
import { solvePressureRedBlack } from './pressureRedBlack.ts';

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

/**
 * The GPU port's correctness gate, run on the CPU.
 *
 * Red-black and lexicographic sweeps solve the SAME linear system — they only
 * disagree about the order to visit rows in — so run past convergence they
 * must land on the same p. If this ever fails, the checkerboard's premise
 * (every 5-point neighbour is the opposite colour) has been broken, and no
 * amount of WebGPU debugging would find it.
 *
 * Deliberately at omega = 1: SOR's relaxation factor changes the convergence
 * PATH, and near the stability edge two orderings can be many sweeps apart
 * without either being wrong. Plain Gauss-Seidel isolates the question.
 */
test('red-black and lexicographic sweeps converge to the SAME pressure', () => {
  const nx = 40;
  const ny = 32;
  const g = createGrid(nx, ny, 1 / ny);

  // Well-posed and not near-singular: an Air column pins p, an interior solid
  // exercises the Neumann branch, and a rough RHS excites every mode.
  const label = new Uint8Array(nx * ny);
  for (let j = 0; j < ny; j++) label[idxP(g, nx - 1, j)] = Cell.Air;
  for (let j = 12; j < 20; j++) for (let i = 10; i < 16; i++) label[idxP(g, i, j)] = Cell.Solid;

  let seed = 12345;
  const div = new Float64Array(nx * ny);
  for (let k = 0; k < div.length; k++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    div[k] = (seed / 0x7fffffff) * 2 - 1;
  }
  const scale = 0.02;

  const lex = new Float64Array(nx * ny);
  const rb = new Float64Array(nx * ny);
  solvePressure(g, lex, div, label, scale, 4000, 1.0, 0);
  solvePressureRedBlack(g, rb, div, label, scale, 4000, 1.0, 0);

  let pMax = 0;
  let diff = 0;
  for (let k = 0; k < lex.length; k++) {
    pMax = Math.max(pMax, Math.abs(lex[k]));
    diff = Math.max(diff, Math.abs(lex[k] - rb[k]));
  }
  assert.ok(pMax > 1e-3, 'test is vacuous if p is ~0');
  // Loose because both sides still carry their own iteration error at 4000
  // sweeps; the point is agreement to 4 digits, not to machine precision.
  assert.ok(diff < 1e-4 * pMax, `orderings disagree: max |diff| ${diff} vs |p|max ${pMax}`);
});

/**
 * Both orderings must actually SOLVE it, not merely agree with each other.
 *
 * An Air RING, not a single Air column: with only one Dirichlet cell the box
 * is nearly all-Neumann, its smoothest mode decays like (1 - c/n^2) per sweep,
 * and plain Gauss-Seidel needs tens of thousands of sweeps to clear it. That
 * would be a test of conditioning, not of the sweep. A full Dirichlet border
 * makes the same solver converge in ~2000.
 */
test('red-black drives the residual down like a solver, not a smoother', () => {
  const n = 32;
  const g = createGrid(n, n, 1 / n);
  const label = new Uint8Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      if (i === 0 || j === 0 || i === n - 1 || j === n - 1) label[idxP(g, i, j)] = Cell.Air;
    }
  }
  const div = new Float64Array(n * n).fill(1);
  const scale = 0.05;

  const p = new Float64Array(n * n);
  solvePressureRedBlack(g, p, div, label, scale, 2000, 1.0, 0);
  const r = rmsRemainingDivergence(g, p, div, label, scale);
  assert.ok(r < 1e-6, `residual should collapse, got ${r}`);
});
