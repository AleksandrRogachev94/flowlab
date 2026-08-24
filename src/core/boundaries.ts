/**
 * Label-driven housekeeping: the two passes that keep the kernels' stated
 * invariants true once `label` stops being uniformly Fluid.
 *
 * Neither is a solver step. They exist because every kernel in core/ was
 * written against assumptions ("a face bordering a solid already stores that
 * solid's velocity", "p is 0 in every Air cell") that nothing enforces on its
 * own. Concentrating them here means a scene, or a mouse painting obstacles,
 * never has to remember the list.
 */

import { Cell, idxP, idxU, idxV, type FieldArray, type Fields, type Grid } from './grid.ts';

/**
 * In-domain AND labeled Solid — the counterpart to grid.ts's
 * isSolidOrOutside(), which reports true off-grid as well.
 *
 * That difference is the whole point: out-of-domain-as-solid is what gives the
 * box its walls in the solver, but the outer faces carry PRESCRIBED boundary
 * data — a jet inflow, a free-stream velocity — and zeroing them would
 * silently delete the scene.
 */
export function isSolidCell(g: Grid, label: Uint8Array, i: number, j: number): boolean {
  return i >= 0 && j >= 0 && i < g.nx && j < g.ny && label[idxP(g, i, j)] === Cell.Solid;
}

/**
 * Restores every label-dependent invariant. Call after ANY edit to `label` —
 * scene setup, or a painted obstacle — and never in the frame loop otherwise.
 *
 * Three invariants, one place:
 *
 * 1. p = 0 in Air cells. gaussSeidelSweep never writes them, so whatever is
 *    there is the Dirichlet value it hands to their fluid neighbours. Zeroing
 *    Solid cells too is cosmetic (their p is never read) but keeps the
 *    pressure heatmap free of stale values.
 * 2. Every face touching a Solid stores the solid's velocity — zero, since
 *    obstacles here are static. computeDivergence depends on this: it reads
 *    faces with no label lookup, which is only correct because Bridson's
 *    u_solid correction to the RHS is already baked into the stored value.
 * 3. No dye inside a Solid. advectScalar copies solid cells through
 *    untouched, so dye stamped there before the obstacle existed would sit
 *    frozen forever AND be sampled by neighbouring fluid cells' backtraces.
 */
export function commitLabels(g: Grid, f: Fields): void {
  const { p, u, v, label, dye } = f;

  for (let k = 0; k < label.length; k++) {
    if (label[k] === Cell.Fluid) continue;
    p[k] = 0;
    if (label[k] === Cell.Solid) for (const c of dye) c[k] = 0;
  }

  for (let j = 0; j < g.ny; j++) {
    for (let i = 0; i <= g.nx; i++) {
      if (isSolidCell(g, label, i - 1, j) || isSolidCell(g, label, i, j)) u[idxU(g, i, j)] = 0;
    }
  }
  for (let j = 0; j <= g.ny; j++) {
    for (let i = 0; i < g.nx; i++) {
      if (isSolidCell(g, label, i, j - 1) || isSolidCell(g, label, i, j)) v[idxV(g, i, j)] = 0;
    }
  }
}

/**
 * Zero-gradient (du/dn = 0) extrapolation onto the domain-boundary face of any
 * Air cell sitting on the edge — i.e. an open outlet.
 *
 * The pressure solve skips Air cells and never needs this; ADVECTION does.
 * That outermost face is written by neither subtractGradient (its loops stop
 * one short) nor advectVelocity (out-of-domain counts as solid), so left alone
 * it sits at its seeded value forever — a stationary wall one bilinear sample
 * away from the outlet, which every backtrace near the exit blends against.
 *
 * Runs after subtractGradient, so it extrapolates the PROJECTED velocity.
 * A no-op on a closed box. Covers all four edges even though only openRight()
 * exists today — an open top (buoyant plume) and painted Air will need them.
 */
export function applyOutflow(g: Grid, u: FieldArray, v: FieldArray, label: Uint8Array): void {
  for (let j = 0; j < g.ny; j++) {
    if (label[idxP(g, 0, j)] === Cell.Air) u[idxU(g, 0, j)] = u[idxU(g, 1, j)];
    if (label[idxP(g, g.nx - 1, j)] === Cell.Air) u[idxU(g, g.nx, j)] = u[idxU(g, g.nx - 1, j)];
  }
  for (let i = 0; i < g.nx; i++) {
    if (label[idxP(g, i, 0)] === Cell.Air) v[idxV(g, i, 0)] = v[idxV(g, i, 1)];
    if (label[idxP(g, i, g.ny - 1)] === Cell.Air) v[idxV(g, i, g.ny)] = v[idxV(g, i, g.ny - 1)];
  }
}
