import { idxP, isSolid, type FieldArray, type Grid } from './grid.ts';

/**
 * One Gauss-Seidel / SOR sweep of the pressure Poisson equation.
 *
 *   count * p[i,j] - sum(p over FLUID neighbours) = -scale * div[i,j]
 *
 * with count = number of NON-SOLID neighbours, scale = rho * h^2 / dt.
 *
 * In-place on `p` by design: reading partially-updated neighbours is exactly
 * what makes GS converge 2x faster than Jacobi (ARCHITECTURE.md Rule 2).
 *
 * TODO(you):
 *   for each cell (i,j):
 *     skip if the cell itself is solid
 *     count = 0, sum = 0
 *     for each of the 4 neighbours:
 *       solid (or outside domain) -> contributes nothing, not counted
 *       air                       -> counted, but p = 0 so adds nothing
 *       fluid                     -> counted, and add its p to sum
 *     if count === 0 continue            // fully enclosed, nothing to solve
 *     pGS = (sum - scale * div[i,j]) / count
 *     p[i,j] = (1 - omega) * p[i,j] + omega * pGS
 *
 * @param p     cell-centred pressure. Read AND written in the same pass.
 * @param div   divergence of the tentative velocity u*, cell-centred.
 * @param label per-cell Fluid/Air/Solid. Solid neighbours impose dp/dn = 0
 *              (Neumann); Air neighbours impose p = 0 (Dirichlet).
 * @param scale rho * h^2 / dt. Converts a divergence into a pressure — it is
 *              what makes the RHS of the linear system dimensionally a
 *              pressure rather than a rate of volume change.
 * @param omega SOR relaxation factor. 1 = plain Gauss-Seidel. Between 1 and 2
 *              it over-relaxes (steps past the GS answer) and converges much
 *              faster; >= 2 diverges unconditionally. Optimal for an n x n
 *              box is 2 / (1 + sin(PI / n)) ~ 1.907 at n = 64, worth roughly
 *              40x fewer sweeps than omega = 1.
 *
 * (Air never occurs in Step 1's closed box; it arrives with free surfaces.)
 */
export function gaussSeidelSweep(
  g: Grid,
  p: FieldArray,
  div: FieldArray,
  label: Uint8Array,
  scale: number,
  omega = 1.0,
  reverseX = false,
  reverseY = false,
): void {
  const iStep = reverseX ? -1 : 1;
  const iFrom = reverseX ? g.nx - 1 : 0;
  const iTo = reverseX ? -1 : g.nx;
  const jStep = reverseY ? -1 : 1;
  const jFrom = reverseY ? g.ny - 1 : 0;
  const jTo = reverseY ? -1 : g.ny;

  for (let j = jFrom; j !== jTo; j += jStep) {
    for (let i = iFrom; i !== iTo; i += iStep) {
      if (isSolid(g, label, i, j)) {
        continue;
      }
      let count = 0;
      let sum = 0;

      if (!isSolid(g, label, i + 1, j)) {
        count += 1;
        sum += p[idxP(g, i + 1, j)];
      }
      if (!isSolid(g, label, i, j + 1)) {
        count += 1;
        sum += p[idxP(g, i, j + 1)];
      }
      if (!isSolid(g, label, i - 1, j)) {
        count += 1;
        sum += p[idxP(g, i - 1, j)];
      }
      if (!isSolid(g, label, i, j - 1)) {
        count += 1;
        sum += p[idxP(g, i, j - 1)];
      }

      if (count === 0) {
        continue;
      }
      const pGS = (sum - scale * div[idxP(g, i, j)]) / count;
      p[idxP(g, i, j)] = (1 - omega) * p[idxP(g, i, j)] + omega * pGS;
    }
  }
}

/**
 * RMS of the divergence that would remain if subtractGradient ran with `p`.
 * An identity, not an estimate: subtracting the gradient changes a cell's
 * divergence by exactly (count*p - sum)/scale.
 *
 * RMS and not max: a max-norm test is gated by the single slowest cell,
 * usually one stubborn corner, so the sweep count swings wildly between
 * near-identical frames. Measured at N=256, mean sweeps 234 (max) vs 40 (RMS).
 */
export function rmsRemainingDivergence(
  g: Grid,
  p: FieldArray,
  div: FieldArray,
  label: Uint8Array,
  scale: number,
): number {
  let sumSq = 0;
  let n = 0;

  for (let j = 0; j < g.ny; j += 1) {
    for (let i = 0; i < g.nx; i += 1) {
      if (isSolid(g, label, i, j)) continue;
      let count = 0;
      let sum = 0;
      if (!isSolid(g, label, i + 1, j)) {
        count += 1;
        sum += p[idxP(g, i + 1, j)];
      }
      if (!isSolid(g, label, i, j + 1)) {
        count += 1;
        sum += p[idxP(g, i, j + 1)];
      }
      if (!isSolid(g, label, i - 1, j)) {
        count += 1;
        sum += p[idxP(g, i - 1, j)];
      }
      if (!isSolid(g, label, i, j - 1)) {
        count += 1;
        sum += p[idxP(g, i, j - 1)];
      }
      if (count === 0) continue;

      const k = idxP(g, i, j);
      const r = div[k] + (count * p[k] - sum) / scale;
      sumSq += r * r;
      n += 1;
    }
  }
  return n > 0 ? Math.sqrt(sumSq / n) : 0;
}

/** Multiple of 4, so a check only lands on a completed direction cycle. */
const CHECK_EVERY = 12;

/**
 * Runs up to `iterations` sweeps, cycling all four sweep directions, stopping
 * once the projection would remove `tol` of the incoming divergence (RMS).
 * Returns the sweeps used.
 *
 * A one-directional sweep propagates information downstream of its ordering
 * much faster than upstream; re-injected every frame and amplified by
 * advection, that bias visibly breaks a symmetric scene. All four directions
 * are needed — alternating X alone leaves Y biased, so the same scene rotated
 * 90 degrees still drifts. Relative asymmetry after 200 frames at N=64:
 * forward-only 8.6e-5, alternate-X 2.9e-6 (X) but 6.9e-5 (Y), all four 1.7e-6
 * on both. A reversed sweep costs the same as a forward one.
 *
 * `p` is NOT zeroed first — warm-starting from the previous frame is a real
 * speedup once this runs every frame.
 */
export function solvePressure(
  g: Grid,
  p: FieldArray,
  div: FieldArray,
  label: Uint8Array,
  scale: number,
  iterations: number,
  omega = 1.0,
  tol = 1e-3,
): number {
  let sumSq = 0;
  for (const d of div) sumSq += d * d;
  const divRms = Math.sqrt(sumSq / div.length);
  if (divRms === 0) return 0; // already divergence-free; any p works

  // tol <= 0 means "always run the full cap", so skip the checks entirely —
  // they cost about a sweep each and could never pass.
  const target = tol > 0 ? tol * divRms : -1;
  for (let k = 0; k < iterations; k++) {
    // k = 0,1,2,3 -> (F,F) (T,F) (F,T) (T,T), then repeat.
    gaussSeidelSweep(g, p, div, label, scale, omega, k % 2 === 1, k % 4 >= 2);
    if (target >= 0 && k % CHECK_EVERY === CHECK_EVERY - 1) {
      if (rmsRemainingDivergence(g, p, div, label, scale) <= target) return k + 1;
    }
  }
  return iterations;
}
