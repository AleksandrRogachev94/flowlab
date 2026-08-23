import { idxP, idxU, idxV, type FieldArray, type Grid } from './grid.ts';

/**
 * w = dv/dx - du/dy at CORNER (i*h, j*h). The corner is vorticity's natural
 * home on a MAC grid: u[i,j], u[i,j-1], v[i,j], v[i-1,j] form a closed
 * circulation loop around it, so both differences are centred there with the
 * tightest stencil. Valid for i in [1, nx-1], j in [1, ny-1].
 */
function curlAtCorner(g: Grid, u: FieldArray, v: FieldArray, i: number, j: number): number {
  return (
    (v[idxV(g, i, j)] - v[idxV(g, i - 1, j)]) / g.h -
    (u[idxU(g, i, j)] - u[idxU(g, i, j - 1)]) / g.h
  );
}

/**
 * Vorticity at CELL CENTERS — averaged from the four surrounding corners,
 * clamped at walls — so it reuses the nx*ny heatmap. Pair with a diverging
 * colormap and symmetric normalization: the sign distinguishes
 * counter-rotating vortices, so zero must land on the neutral midpoint.
 *
 * The averaging costs half a cell of smoothing; invisible at N >= 128.
 */
export function computeVorticity(g: Grid, u: FieldArray, v: FieldArray, out: FieldArray): void {
  const hiI = g.nx - 1;
  const hiJ = g.ny - 1;

  for (let j = 0; j < g.ny; j++) {
    const j0 = Math.min(Math.max(j, 1), hiJ);
    const j1 = Math.min(Math.max(j + 1, 1), hiJ);
    for (let i = 0; i < g.nx; i++) {
      const i0 = Math.min(Math.max(i, 1), hiI);
      const i1 = Math.min(Math.max(i + 1, 1), hiI);
      out[idxP(g, i, j)] =
        0.25 *
        (curlAtCorner(g, u, v, i0, j0) +
          curlAtCorner(g, u, v, i1, j0) +
          curlAtCorner(g, u, v, i0, j1) +
          curlAtCorner(g, u, v, i1, j1));
    }
  }
}
