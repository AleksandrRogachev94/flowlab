import { Cell, idxP, isSolidOrOutside, type FieldArray, type Grid } from './grid.ts';
import { fluidDivRms, rmsRemainingDivergence } from './pressure.ts';

/**
 * Red-black SOR on the CPU: a line-for-line mirror of gpu/redBlack.wgsl.
 *
 * It exists to make the GPU port debuggable. When the device produces a wrong
 * picture there are three candidate causes — the ORDERING (red-black vs
 * lexicographic), the PRECISION (f32 vs f64), and the PLUMBING (buffers, bind
 * groups, dispatch) — and with only a CPU-lexicographic and a GPU-red-black
 * solver you cannot tell them apart. This is the third corner of that square:
 * same ordering as the GPU, same float64 as the reference. If it agrees with
 * the lexicographic solver but the GPU does not, the bug is plumbing.
 *
 * It is also the honest place to CHECK the claim the shader comment makes —
 * that red-black converges as fast as lexicographic at the optimal omega —
 * rather than assert it, since running the shader needs a browser and running
 * this does not.
 */

/**
 * One half-sweep over cells of a single colour, colour = (i + j) & 1.
 *
 * In place on `p`, and safely so: in the 5-point stencil every neighbour of a
 * colour-c cell is colour 1-c, so a half-sweep reads nothing it writes. That
 * is what makes the same code correct on thousands of GPU threads in
 * arbitrary order — the property is the colouring's, not the loop's.
 *
 * Two half-sweeps (0 then 1) make one full Gauss-Seidel sweep: the second sees
 * the first's brand-new values, which is where GS beats Jacobi.
 */
export function redBlackSweep(
  g: Grid,
  p: FieldArray,
  div: FieldArray,
  label: Uint8Array,
  scale: number,
  omega: number,
  color: 0 | 1,
): void {
  for (let j = 0; j < g.ny; j += 1) {
    // Start at the first cell of this colour in the row and step by 2, rather
    // than testing every cell. Same set, half the loop iterations.
    for (let i = (j & 1) ^ color; i < g.nx; i += 2) {
      const k = idxP(g, i, j);
      if (label[k] !== Cell.Fluid) continue;

      let count = 0;
      let sum = 0;
      if (!isSolidOrOutside(g, label, i + 1, j)) {
        count += 1;
        sum += p[k + 1];
      }
      if (!isSolidOrOutside(g, label, i - 1, j)) {
        count += 1;
        sum += p[k - 1];
      }
      if (!isSolidOrOutside(g, label, i, j + 1)) {
        count += 1;
        sum += p[k + g.nx];
      }
      if (!isSolidOrOutside(g, label, i, j - 1)) {
        count += 1;
        sum += p[k - g.nx];
      }
      if (count === 0) continue;

      const pGS = (sum - scale * div[k]) / count;
      p[k] = (1 - omega) * p[k] + omega * pGS;
    }
  }
}

/** Multiple of 2, so a check only lands on a completed red+black pair. */
const CHECK_EVERY = 12;

/**
 * Same contract as solvePressure(), same warm start, same relative tolerance.
 * The only difference is the traversal, so a diff between the two is a
 * statement about ORDERING and nothing else.
 *
 * Note what is absent: the four-direction cycling. A lexicographic sweep
 * carries information downstream of its own ordering faster than upstream, and
 * cycling F/R x F/R is what keeps that bias from breaking symmetric scenes.
 * Red-black updates every cell of a colour from the same data, so there is no
 * preferred direction to correct for.
 */
export function solvePressureRedBlack(
  g: Grid,
  p: FieldArray,
  div: FieldArray,
  label: Uint8Array,
  scale: number,
  iterations: number,
  omega = 1.0,
  tol = 1e-3,
): number {
  const divRms = fluidDivRms(div, label);
  if (divRms === 0) {
    p.fill(0);
    return 0;
  }

  const target = tol > 0 ? tol * divRms : -1;
  for (let k = 0; k < iterations; k++) {
    redBlackSweep(g, p, div, label, scale, omega, 0);
    redBlackSweep(g, p, div, label, scale, omega, 1);
    if (target >= 0 && k % CHECK_EVERY === CHECK_EVERY - 1) {
      if (rmsRemainingDivergence(g, p, div, label, scale) <= target) return k + 1;
    }
  }
  return iterations;
}
