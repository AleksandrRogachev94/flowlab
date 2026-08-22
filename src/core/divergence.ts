import { idxP, idxU, idxV, type FieldArray, type Grid } from './grid.ts';

/**
 * Purely local: reads only the four faces of each cell, so it needs no
 * labels.
 *
 * INVARIANT this relies on: a face bordering a solid must already store that
 * solid's velocity (zero for static walls). Bridson's RHS correction —
 * substituting u_solid at solid faces — is then a no-op, which is why it is
 * absent here.
 *
 * Advection (Step 2) is what can break this: it must not write to solid
 * faces, and whatever sits there gets sampled during backtrace, so the value
 * has to stay correct rather than merely untouched. Newly-painted obstacles
 * (Step 3) need their faces reset for the same reason.
 */
export const computeDivergence = (g: Grid, u: FieldArray, v: FieldArray, out: FieldArray) => {
  const coef = 1 / g.h;
  for (let j = 0; j < g.ny; j += 1) {
    for (let i = 0; i < g.nx; i += 1) {
      out[idxP(g, i, j)] =
        coef * (u[idxU(g, i + 1, j)] - u[idxU(g, i, j)] + v[idxV(g, i, j + 1)] - v[idxV(g, i, j)]);
    }
  }
};
