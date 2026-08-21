import { idxP, idxU, idxV, type FieldArray, type Grid } from './grid.ts';

export const computeDivergence = (g: Grid, u: FieldArray, v: FieldArray, out: FieldArray) => {
  const coef = 1 / g.h;
  for (let j = 0; j < g.ny; j += 1) {
    for (let i = 0; i < g.nx; i += 1) {
      out[idxP(g, i, j)] =
        coef * (u[idxU(g, i + 1, j)] - u[idxU(g, i, j)] + v[idxV(g, i, j + 1)] - v[idxV(g, i, j)]);
    }
  }
};
