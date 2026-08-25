import { Cell, createGrid, idxP, isSolidOrOutside, type FieldArray, type Grid } from './grid.ts';
import { fluidDivRms, rmsRemainingDivergence } from './pressure.ts';
import type { PressureSolver } from './pressureSolver.ts';
import { redBlackSweep } from './pressureRedBlack.ts';

/**
 * Geometric multigrid V-cycle on the CPU: the f64 mirror of gpu/multigrid.wgsl,
 * playing the same role pressureRedBlack.ts plays for redBlack.wgsl — the rung
 * of the debugging ladder where the ALGORITHM can be checked without a browser.
 * The why-multigrid case is docs/WEBGPU.md §6; the short version:
 *
 * Relaxation kills the high-frequency half of the error in a few sweeps and
 * then stalls, because a 5-point stencil moves information one cell per sweep
 * and the remaining error is SMOOTH — visible only over long distances. The
 * fix is not more sweeps but a change of scale: restrict the residual to a
 * grid of half the resolution, where yesterday's smooth error is today's
 * high-frequency error and the same cheap smoother works on it, and recurse.
 * Each level costs a quarter of the one above, so the whole V-cycle is ~4/3
 * of one fine-grid sweep-pair — and a few cycles reach a fixed accuracy at ANY
 * resolution, which is the O(1) property SOR's O(N) sweep count lacks.
 *
 * THE ONE ALGEBRAIC TRICK to read before the code: every level solves
 *
 *     count * x - sum(x over non-solid nbrs) = b        (per FLUID cell)
 *
 * with b in "code units", i.e. already multiplied by the level's h^2 — on the
 * fine grid b = -scale * div, exactly the RHS the SOR solvers use. That form
 * makes both grid-transfer operators h-free:
 *
 *   - RESTRICTION: the coarse b is the SUM of the four children's residuals.
 *     (Full-weighting is the average of child residuals r_code/h^2 in
 *     continuous units; multiplying back by the coarse (2h)^2 turns the
 *     average into a plain sum. The 4s cancel — nothing to tune, nothing to
 *     get wrong per level.)
 *   - The SMOOTHER is the existing redBlackSweep with scale = -1, div = b:
 *     (sum - scale*div)/count becomes (sum + b)/count, which is the equation
 *     above. Reuse rather than a near-copy, so the smoother multigrid runs is
 *     the one pressure.test.ts already pins down.
 *
 * Coarse levels solve for a CORRECTION e (starting from 0), not a pressure;
 * prolongation adds the interpolated e back into the finer level's solution.
 */

/** Pre/post smoothing sweeps per level, each one red+black pair. The standard
 *  V(2,2); one sweep is only marginally worse but this makes the contraction
 *  comfortably budget-insensitive. Shared with the GPU implementation. */
export const MG_PRE_SWEEPS = 2;
export const MG_POST_SWEEPS = 2;
/** Sweep pairs on the coarsest (<= 4x4-ish) level, in lieu of a direct solve.
 *  A dozen cells converge to roundoff well inside this. */
export const MG_COARSE_SWEEPS = 8;
/**
 * Plain Gauss-Seidel, no over-relaxation. Inside multigrid the sweeps are a
 * SMOOTHER, not a solver: their job is only the high-frequency half of the
 * error, which omega = 1 damps fastest. Over-relaxing chases the smooth modes
 * the coarse grid is about to handle anyway — see docs/WEBGPU.md §4's "you are
 * not a solver, you are a smoother".
 */
export const MG_OMEGA = 1.0;
/** V-cycles per solve. Fixed for the same frame-pacing reason as the SOR sweep
 *  budget in main.ts; 3 with a warm start lands well below the residual the
 *  fixed SOR budget was reaching (measured in pressureMultigrid.test.ts). */
export const MG_CYCLES = 3;

/** Coarsest useful level: below ~4 cells a side there is nothing left to
 *  coarsen and the sweeps ARE a direct solve. */
const COARSEST = 4;

/** Finest first. ceil so odd dimensions coarsen without dropping their last
 *  row/column — those cells simply have fewer than four children. */
export function levelSizes(nx: number, ny: number): { nx: number; ny: number }[] {
  const sizes = [{ nx, ny }];
  while (sizes[sizes.length - 1].nx > COARSEST || sizes[sizes.length - 1].ny > COARSEST) {
    const prev = sizes[sizes.length - 1];
    sizes.push({ nx: Math.ceil(prev.nx / 2), ny: Math.ceil(prev.ny / 2) });
  }
  return sizes;
}

/**
 * One coarse label from up to four fine ones, priority Air > Fluid > Solid.
 *
 * The order is chosen by what each mistake costs. Air anywhere in the children
 * must win so the Dirichlet anchor survives coarsening — lose it and a coarse
 * level near an outlet turns all-Neumann and its correction drifts. Fluid must
 * beat Solid or a one-cell wall GROWS on every level and can seal a channel
 * the fine grid keeps open, which stalls the whole cycle; letting thin walls
 * instead LEAK on coarse levels only degrades the convergence rate near them,
 * and the fine level still enforces the true boundary. This rule is the part
 * docs/WEBGPU.md §6 predicted the bugs would live in.
 */
export function coarsenLabels(f: Grid, fine: Uint8Array, c: Grid, coarse: Uint8Array): void {
  for (let J = 0; J < c.ny; J++) {
    for (let I = 0; I < c.nx; I++) {
      let air = false;
      let fluid = false;
      for (let dj = 0; dj < 2; dj++) {
        for (let di = 0; di < 2; di++) {
          const i = 2 * I + di;
          const j = 2 * J + dj;
          if (i >= f.nx || j >= f.ny) continue;
          const l = fine[idxP(f, i, j)];
          air ||= l === Cell.Air;
          fluid ||= l === Cell.Fluid;
        }
      }
      coarse[idxP(c, I, J)] = air ? Cell.Air : fluid ? Cell.Fluid : Cell.Solid;
    }
  }
}

/**
 * r = b - (count*x - sum), the code-unit residual of the equation above, and 0
 * in every non-fluid or walled-in cell — restriction sums r blindly, so the
 * zeros are what keep non-fluid children out of the coarse RHS.
 */
function computeResidual(
  g: Grid,
  x: FieldArray,
  b: FieldArray,
  label: Uint8Array,
  r: FieldArray,
): void {
  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i < g.nx; i++) {
      const k = idxP(g, i, j);
      r[k] = 0;
      if (label[k] !== Cell.Fluid) continue;
      let count = 0;
      let sum = 0;
      if (!isSolidOrOutside(g, label, i + 1, j)) {
        count += 1;
        sum += x[k + 1];
      }
      if (!isSolidOrOutside(g, label, i - 1, j)) {
        count += 1;
        sum += x[k - 1];
      }
      if (!isSolidOrOutside(g, label, i, j + 1)) {
        count += 1;
        sum += x[k + g.nx];
      }
      if (!isSolidOrOutside(g, label, i, j - 1)) {
        count += 1;
        sum += x[k - g.nx];
      }
      if (count === 0) continue;
      r[k] = b[k] - (count * x[k] - sum);
    }
  }
}

/** Coarse RHS = sum of child residuals (see the header for why a plain sum),
 *  and the coarse correction zeroed in the same pass — e's initial guess is
 *  always 0, since it estimates an error that restriction just measured. */
function restrictResidual(
  f: Grid,
  r: FieldArray,
  c: Grid,
  bCoarse: FieldArray,
  eCoarse: FieldArray,
): void {
  for (let J = 0; J < c.ny; J++) {
    for (let I = 0; I < c.nx; I++) {
      let sum = 0;
      for (let dj = 0; dj < 2; dj++) {
        for (let di = 0; di < 2; di++) {
          const i = 2 * I + di;
          const j = 2 * J + dj;
          if (i < f.nx && j < f.ny) sum += r[idxP(f, i, j)];
        }
      }
      const K = idxP(c, I, J);
      bCoarse[K] = sum;
      eCoarse[K] = 0;
    }
  }
}

/**
 * x += bilinear(e), fluid cells only. A fine cell's centre sits a quarter-cell
 * off its parent's, so the interpolation weights are 3/4 : 1/4 per axis —
 * 9/16 parent, 3/16 each side neighbour, 1/16 diagonal. Indices clamp at the
 * domain edge, which extends e constantly outward — the discrete Neumann
 * condition, matching what the walls impose on p itself.
 *
 * Non-fluid coarse neighbours contribute their stored e: 0 for Air (the
 * correct Dirichlet value) and 0 for Solid (restrict zeroed it), which merely
 * dilutes the correction beside a wall instead of special-casing it. Cheap,
 * standard, and the fine smoother repairs the difference.
 */
function prolongAdd(f: Grid, x: FieldArray, label: Uint8Array, c: Grid, e: FieldArray): void {
  for (let j = 0; j < f.ny; j++) {
    const J = j >> 1;
    const J2 = j & 1 ? Math.min(J + 1, c.ny - 1) : Math.max(J - 1, 0);
    for (let i = 0; i < f.nx; i++) {
      const k = idxP(f, i, j);
      if (label[k] !== Cell.Fluid) continue;
      const I = i >> 1;
      const I2 = i & 1 ? Math.min(I + 1, c.nx - 1) : Math.max(I - 1, 0);
      x[k] +=
        0.5625 * e[idxP(c, I, J)] +
        0.1875 * (e[idxP(c, I2, J)] + e[idxP(c, I, J2)]) +
        0.0625 * e[idxP(c, I2, J2)];
    }
  }
}

/** One level's storage. `x` is the pressure on the finest level and the
 *  correction e below it — same equation, different meaning of the unknown. */
interface Level {
  g: Grid;
  label: Uint8Array;
  x: Float64Array;
  b: Float64Array;
  r: Float64Array;
}

export class CpuMultigridSolver implements PressureSolver {
  readonly name = 'cpu-mg';

  /** Finest first. Lazily (re)built from the grid the first solve sees, so the
   *  solver can sit in a module-level list before the Simulation exists. */
  private levels: Level[] = [];

  private readonly cycles: number;

  // Not a parameter property: node --test runs this file with type stripping,
  // which cannot rewrite those (the GPU files never meet node, so theirs are
  // fine).
  constructor(cycles = MG_CYCLES) {
    this.cycles = cycles;
  }

  private ensureLevels(g: Grid): void {
    if (this.levels.length > 0 && this.levels[0].g.nx === g.nx && this.levels[0].g.ny === g.ny) {
      return;
    }
    this.levels = levelSizes(g.nx, g.ny).map((s, l) => {
      const n = s.nx * s.ny;
      return {
        g: createGrid(s.nx, s.ny, g.h * 2 ** l),
        label: new Uint8Array(n),
        x: new Float64Array(n),
        b: new Float64Array(n),
        r: new Float64Array(n),
      };
    });
  }

  /** `x` rather than levels[l].x because the finest level smooths the caller's
   *  p in place — the warm start comes in and the answer goes out through it. */
  private vcycle(l: number, x: FieldArray): void {
    const { g, label, b, r } = this.levels[l];
    if (l === this.levels.length - 1) {
      for (let s = 0; s < MG_COARSE_SWEEPS; s++) {
        redBlackSweep(g, x, b, label, -1, MG_OMEGA, 0);
        redBlackSweep(g, x, b, label, -1, MG_OMEGA, 1);
      }
      return;
    }
    for (let s = 0; s < MG_PRE_SWEEPS; s++) {
      redBlackSweep(g, x, b, label, -1, MG_OMEGA, 0);
      redBlackSweep(g, x, b, label, -1, MG_OMEGA, 1);
    }
    computeResidual(g, x, b, label, r);
    const next = this.levels[l + 1];
    restrictResidual(g, r, next.g, next.b, next.x);
    this.vcycle(l + 1, next.x);
    prolongAdd(g, x, label, next.g, next.x);
    for (let s = 0; s < MG_POST_SWEEPS; s++) {
      redBlackSweep(g, x, b, label, -1, MG_OMEGA, 0);
      redBlackSweep(g, x, b, label, -1, MG_OMEGA, 1);
    }
  }

  /**
   * `iterations` and `omega` are accepted for interface compatibility and
   * IGNORED: the sweep budget and its omega are SOR tuning (main.ts sizes both
   * to SOR's O(N) transient), while a V-cycle's effort is the fixed
   * `cycles` and its smoother wants MG_OMEGA — see that constant. Returns
   * V-cycles used, so sim.iters reads in cycles on this solver.
   */
  solve(
    g: Grid,
    p: FieldArray,
    div: FieldArray,
    label: Uint8Array,
    scale: number,
    iterations: number,
    omega: number,
    tol: number,
  ): number {
    const divRms = fluidDivRms(div, label);
    if (divRms === 0) {
      // Same guard as every other solver: p = 0 is the exact answer, and a
      // warm-started leftover would re-inject divergence.
      p.fill(0);
      return 0;
    }

    this.ensureLevels(g);
    const levels = this.levels;
    // The finest label is the caller's array; coarser ones are rebuilt every
    // solve. O(N/3) total — cheap enough that caching against label changes
    // is not worth owning a staleness bug.
    levels[0].label = label;
    for (let l = 1; l < levels.length; l++) {
      coarsenLabels(levels[l - 1].g, levels[l - 1].label, levels[l].g, levels[l].label);
    }
    for (let k = 0; k < levels[0].b.length; k++) levels[0].b[k] = -scale * div[k];

    const target = tol > 0 ? tol * divRms : -1;
    for (let c = 0; c < this.cycles; c++) {
      this.vcycle(0, p);
      if (target >= 0 && rmsRemainingDivergence(g, p, div, label, scale) <= target) return c + 1;
    }
    return this.cycles;
  }
}
