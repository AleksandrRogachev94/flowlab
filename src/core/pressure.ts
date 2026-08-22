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
): void {
  for (let j = 0; j < g.ny; j += 1) {
    for (let i = 0; i < g.nx; i += 1) {
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
 * Runs `iterations` sweeps. `p` is NOT zeroed first — warm-starting from the
 * previous frame's pressure is a real speedup once this runs every frame.
 */
export function solvePressure(
  g: Grid,
  p: FieldArray,
  div: FieldArray,
  label: Uint8Array,
  scale: number,
  iterations: number,
  omega = 1.0,
): void {
  for (let k = 0; k < iterations; k++) {
    gaussSeidelSweep(g, p, div, label, scale, omega);
  }
}
