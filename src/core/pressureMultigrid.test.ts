import test from 'node:test';
import assert from 'node:assert/strict';
import { Cell, createGrid, idxP } from './grid.ts';
import { fluidDivRms, rmsRemainingDivergence, solvePressure } from './pressure.ts';
import { CpuMultigridSolver } from './pressureMultigrid.ts';

/**
 * The claim under test is docs/WEBGPU.md §6's O(1) property: the V-cycle
 * contracts the residual by a factor that does NOT depend on the grid size —
 * measured per cycle by running the solver one cycle at a time. The two
 * regimes are asserted separately because they are genuinely different
 * (docs/WEBGPU.md §9): away from irregular boundaries the cycle is textbook
 * (~0.06); next to Air and Solid boundaries the re-discretized coarse
 * operators are only first-order right, and the rate degrades to ~0.5. The
 * lexicographic SOR solver is the converged reference, per the same ladder
 * that debugged red-black.
 */

const SCALE = 0.02;

/** Deterministic rough RHS, so every frequency is present, not only the
 *  smooth ones a V-cycle finds easy. */
function roughRhs(n: number, seed = 12345): Float64Array {
  const div = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    div[k] = (seed / 0x7fffffff) * 2 - 1;
  }
  return div;
}

/** Air column to pin p, an interior solid block to exercise the Neumann
 *  branch and the label coarsening — the same shape as the gputest fixture. */
function boundaryFixture(nx: number, ny: number): { label: Uint8Array; div: Float64Array } {
  const g = createGrid(nx, ny, 1 / ny);
  const label = new Uint8Array(nx * ny);
  for (let j = 0; j < ny; j++) label[idxP(g, nx - 1, j)] = Cell.Air;
  for (let j = ny >> 2; j < ny >> 1; j++) {
    for (let i = nx >> 2; i < (nx * 3) >> 3; i++) label[idxP(g, i, j)] = Cell.Solid;
  }
  return { label, div: roughRhs(nx * ny) };
}

/** Residual contraction factor of each of `cycles` consecutive V-cycles. */
function contractionFactors(
  nx: number,
  ny: number,
  label: Uint8Array,
  div: Float64Array,
  cycles: number,
): number[] {
  const g = createGrid(nx, ny, 1 / ny);
  const p = new Float64Array(nx * ny);
  const solver = new CpuMultigridSolver(1);
  let prev = fluidDivRms(div, label);
  const factors: number[] = [];
  for (let c = 0; c < cycles; c++) {
    // cycles = 1 and tol = 0: each call is exactly one V-cycle, warm-started
    // on the last — the per-cycle contraction laid bare.
    solver.solve(g, p, div, label, SCALE, 0, 0, 0);
    const res = rmsRemainingDivergence(g, p, div, label, SCALE);
    factors.push(res / prev);
    prev = res;
  }
  return factors;
}

test('interior contraction is ~0.06 per cycle and grid-independent', () => {
  for (const [nx, ny] of [
    [32, 32],
    [64, 48],
    [128, 96],
  ]) {
    // All-fluid closed box: walls only, via outside-as-solid — the regime the
    // O(1) claim holds cleanly in. Zero-mean RHS, because a closed box only
    // admits a divergence a velocity field can actually have (the system is
    // singular but consistent; the residual is gauge-free so it is what gets
    // asserted).
    const div = roughRhs(nx * ny, 999);
    const mean = div.reduce((a, b) => a + b, 0) / div.length;
    for (let k = 0; k < div.length; k++) div[k] -= mean;
    const factors = contractionFactors(nx, ny, new Uint8Array(nx * ny), div, 5);
    // Measured 0.016-0.08 across sizes; 0.15 is the alarm threshold. A broken
    // transfer operator stalls near 1, an SOR in disguise degrades with nx.
    for (const f of factors) {
      assert.ok(f < 0.15, `interior contraction ${factors.join(', ')} at ${nx}x${ny}`);
    }
  }
});

test('irregular boundaries degrade contraction to ~0.5 but never stall it', () => {
  for (const [nx, ny] of [
    [32, 32],
    [64, 48],
    [128, 96],
  ]) {
    const { label, div } = boundaryFixture(nx, ny);
    const factors = contractionFactors(nx, ny, label, div, 6);
    // Measured ~0.45-0.55 asymptotically — the known first-order boundary
    // limit of re-discretized coarse operators (docs/WEBGPU.md §9), asserted
    // here so an accidental fix OR regression shows up as a test failure.
    for (const f of factors) {
      assert.ok(f < 0.7, `boundary contraction ${factors.join(', ')} at ${nx}x${ny}`);
    }
    const total = factors.reduce((a, b) => a * b, 1);
    assert.ok(total < 1e-2, `6 cycles only reduced the residual by ${total} at ${nx}x${ny}`);
  }
});

test('multigrid converges to the same pressure as the SOR reference', () => {
  const nx = 64;
  const ny = 48;
  const g = createGrid(nx, ny, 1 / ny);
  const { label, div } = boundaryFixture(nx, ny);

  const expected = new Float64Array(nx * ny);
  solvePressure(g, expected, div, label, SCALE, 20000, 1.9, 1e-12);

  const p = new Float64Array(nx * ny);
  new CpuMultigridSolver(40).solve(g, p, div, label, SCALE, 0, 0, 1e-12);

  // The Air column makes the solution unique, so p itself is comparable —
  // no gauge constant to mod out.
  let pMax = 0;
  let diff = 0;
  for (let k = 0; k < expected.length; k++) {
    pMax = Math.max(pMax, Math.abs(expected[k]));
    diff = Math.max(diff, Math.abs(expected[k] - p[k]));
  }
  assert.ok(pMax > 1e-3, 'test is vacuous if p is ~0');
  assert.ok(diff < 1e-8 * pMax, `max diff ${diff} vs |p|max ${pMax}`);
});
