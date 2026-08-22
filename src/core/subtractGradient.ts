import { idxP, idxU, idxV, isSolid, type FieldArray, type Grid } from './grid.ts';

/**
 * Applies u^{n+1} = u* - (dt / (rho * h)) * grad(p), making the field
 * divergence-free.
 *
 *   u[i,j] -= gradScale * (p[i,j] - p[i-1,j])
 *   v[i,j] -= gradScale * (p[i,j] - p[i,j-1])
 *
 * In-place on u/v, and this is NOT a Rule 2 violation: each face reads only
 * its OWN u value (neighbours are read from `p`, a different array). Rule 2
 * forbids reading NEIGHBOURS of the array you write — a per-element update
 * is safe here and safe on the GPU, where each thread owns one element.
 *
 * Only interior faces are updated. A face touching a solid keeps its
 * prescribed velocity (zero for static walls), which is the u.n = 0
 * boundary condition:
 *   u: i in [1, nx-1], j in [0, ny-1] — the OTHER dimension is unbounded,
 *      since u has no wall condition tied to j
 *   v: i in [0, nx-1], j in [1, ny-1] — symmetric, unbounded in i
 * plus, for interior obstacles: skip a face if EITHER adjacent cell is
 * solid, via isSolid() (currently only true outside the domain, so this is
 * a no-op until Step 3 paints obstacles — but the check costs nothing now
 * and removes a whole bug class later).
 *
 * Those bounds are deliberately identical to addGradient() in
 * scenes/testFields.ts — same discrete operator, so seeding a pure gradient
 * and projecting should return ~0 to solver tolerance, not just to O(h^2).
 *
 * @param p         cell-centred pressure from solvePressure().
 * @param u, v      face velocities. Updated in place.
 * @param label     per-cell Fluid/Air/Solid. A face touching a solid is
 *                  skipped entirely, which IS the u.n = 0 wall condition.
 * @param gradScale dt / (rho * h). Converts a pressure difference into a
 *                  velocity change — the reciprocal partner of pressure.ts's
 *                  `scale`. The two cancel: the projected velocity is
 *                  identical for any rho and dt (see chat derivation), which
 *                  is why the tests can just use rho = dt = 1.
 */
export function subtractGradient(
  g: Grid,
  p: FieldArray,
  u: FieldArray,
  v: FieldArray,
  label: Uint8Array,
  gradScale: number,
): void {
  for (let j = 0; j < g.ny; j += 1) {
    for (let i = 1; i < g.nx; i += 1) {
      if (isSolid(g, label, i - 1, j) || isSolid(g, label, i, j)) continue;
      u[idxU(g, i, j)] -= gradScale * (p[idxP(g, i, j)] - p[idxP(g, i - 1, j)]);
    }
  }

  for (let j = 1; j < g.ny; j += 1) {
    for (let i = 0; i < g.nx; i += 1) {
      if (isSolid(g, label, i, j - 1) || isSolid(g, label, i, j)) continue;
      v[idxV(g, i, j)] -= gradScale * (p[idxP(g, i, j)] - p[idxP(g, i, j - 1)]);
    }
  }
}
