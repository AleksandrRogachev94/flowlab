import { Cell, idxP, isSolidOrOutside, type FieldArray, type Grid } from './grid.ts';

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
 * Only FLUID cells get an equation. Solid cells have no pressure at all, and
 * Air cells ARE the Dirichlet condition p = 0 — skipping them is what pins
 * their value, since nothing else ever writes p there.
 *
 * The neighbour loop needs no Air branch, and that is not an oversight: an Air
 * neighbour is non-solid, so it is counted in `count`, and its stored p is 0,
 * so it adds nothing to `sum`. That is exactly Bridson's row (diagonal = number
 * of non-solid neighbours, off-diagonal -1 for FLUID neighbours only). The
 * invariant it rests on — p is 0 in every Air cell — is established by
 * commitLabels() in boundaries.ts and never disturbed afterwards.
 *
 * @param p     cell-centred pressure. Read AND written in the same pass.
 * @param div   divergence of the tentative velocity u*, cell-centred.
 * @param label per-cell Fluid/Air/Solid. Solid neighbours impose dp/dn = 0
 *              (Neumann); Air neighbours impose p = 0 (Dirichlet). At least
 *              one Air cell makes the system nonsingular — an all-Neumann box
 *              is solvable only when total boundary flux is exactly zero, and
 *              its p drifts by an arbitrary constant.
 * @param scale rho * h^2 / dt. Converts a divergence into a pressure — it is
 *              what makes the RHS of the linear system dimensionally a
 *              pressure rather than a rate of volume change.
 * @param omega SOR relaxation factor. 1 = plain Gauss-Seidel. Between 1 and 2
 *              it over-relaxes (steps past the GS answer) and converges much
 *              faster; >= 2 diverges unconditionally. Optimal for an n x n
 *              box is 2 / (1 + sin(PI / n)) ~ 1.907 at n = 64, worth roughly
 *              40x fewer sweeps than omega = 1.
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
      // Not isSolidOrOutside(): (i,j) is in-domain by construction here, and this must
      // skip Air as well as Solid.
      if (label[idxP(g, i, j)] !== Cell.Fluid) {
        continue;
      }
      let count = 0;
      let sum = 0;

      if (!isSolidOrOutside(g, label, i + 1, j)) {
        count += 1;
        sum += p[idxP(g, i + 1, j)];
      }
      if (!isSolidOrOutside(g, label, i, j + 1)) {
        count += 1;
        sum += p[idxP(g, i, j + 1)];
      }
      if (!isSolidOrOutside(g, label, i - 1, j)) {
        count += 1;
        sum += p[idxP(g, i - 1, j)];
      }
      if (!isSolidOrOutside(g, label, i, j - 1)) {
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
      if (label[idxP(g, i, j)] !== Cell.Fluid) continue;
      let count = 0;
      let sum = 0;
      if (!isSolidOrOutside(g, label, i + 1, j)) {
        count += 1;
        sum += p[idxP(g, i + 1, j)];
      }
      if (!isSolidOrOutside(g, label, i, j + 1)) {
        count += 1;
        sum += p[idxP(g, i, j + 1)];
      }
      if (!isSolidOrOutside(g, label, i - 1, j)) {
        count += 1;
        sum += p[idxP(g, i - 1, j)];
      }
      if (!isSolidOrOutside(g, label, i, j - 1)) {
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
  // FLUID cells only: divergence inside a solid is stale face data, and an Air
  // outlet is SUPPOSED to be divergent — either would inflate this denominator
  // and silently loosen a tolerance stated relative to the real divergence.
  let sumSq = 0;
  let nFluid = 0;
  for (let k = 0; k < div.length; k++) {
    if (label[k] !== Cell.Fluid) continue;
    sumSq += div[k] * div[k];
    nFluid += 1;
  }
  const divRms = nFluid > 0 ? Math.sqrt(sumSq / nFluid) : 0;
  if (divRms === 0) {
    // Already divergence-free, and p = 0 is the exact solution. NOT "any p":
    // subtractGradient applies p unconditionally, so returning a warm-started
    // leftover here would re-inject divergence into a clean field.
    p.fill(0);
    return 0;
  }

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
