/**
 * Test fields for verifying projection: u* = curl(psi) + grad(phi).
 * Projection must kill the gradient part and leave the curl part alone.
 *
 * Both are built from DISCRETE differences of a potential rather than
 * sampled analytic derivatives, so div(curl(psi)) cancels exactly (1e-16)
 * instead of to O(h^2). Tolerances stay tight enough to blame the solver.
 *
 * Assumes the unit square (h = 1/nx, nx === ny). Both potentials give zero
 * normal velocity at the walls, so total divergence sums to 0 as the
 * pressure solve requires. Frequencies differ on purpose: same frequency
 * makes curl and grad exact negatives that cancel to nothing.
 *
 * Both functions ADD into u/v — zero the arrays first for a fresh field.
 */

import { idxU, idxV, type FieldArray, type Grid } from '../core/grid.ts';

const PI = Math.PI;

/** Divergence-free part: u = curl(psi), psi = sin(pi x) sin(pi y) at corners. */
export function addRotational(g: Grid, u: FieldArray, v: FieldArray, amp = 1): void {
  const inv = amp / g.h;
  const psi = (i: number, j: number) =>
    Math.sin(PI * i * g.h) * Math.sin(PI * j * g.h);

  for (let j = 0; j < g.ny; j++)
    for (let i = 0; i <= g.nx; i++)
      u[idxU(g, i, j)] += inv * (psi(i, j + 1) - psi(i, j));

  for (let j = 0; j <= g.ny; j++)
    for (let i = 0; i < g.nx; i++)
      v[idxV(g, i, j)] += -inv * (psi(i + 1, j) - psi(i, j));
}

/**
 * Pure gradient part: u = grad(phi), phi = cos(2pi x) cos(2pi y) at centers.
 * Boundary faces are skipped, which is the Neumann condition dphi/dn = 0.
 */
export function addGradient(g: Grid, u: FieldArray, v: FieldArray, amp = 1): void {
  const inv = amp / g.h;
  const phi = (i: number, j: number) =>
    Math.cos(2 * PI * (i + 0.5) * g.h) * Math.cos(2 * PI * (j + 0.5) * g.h);

  for (let j = 0; j < g.ny; j++)
    for (let i = 1; i < g.nx; i++)
      u[idxU(g, i, j)] += inv * (phi(i, j) - phi(i - 1, j));

  for (let j = 1; j < g.ny; j++)
    for (let i = 0; i < g.nx; i++)
      v[idxV(g, i, j)] += inv * (phi(i, j) - phi(i, j - 1));
}
