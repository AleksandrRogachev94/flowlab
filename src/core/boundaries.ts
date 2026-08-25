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
 * The open-boundary pass. Two jobs on every edge Air cell — i.e. an outlet:
 * a NO-BACKFLOW clamp on the outlet face, then zero-gradient (du/dn = 0)
 * extrapolation of that value onto the outermost face.
 *
 * THE EXTRAPOLATION. The pressure solve skips Air cells and never needs it;
 * ADVECTION does. The outermost face is written by neither subtractGradient
 * (its loops stop one short) nor advectVelocity (out-of-domain counts as
 * solid), so left alone it sits at its seeded value forever — a stationary
 * wall one bilinear sample away from the outlet, which every backtrace near
 * the exit blends against.
 *
 * THE CLAMP, and why an outlet cannot be left to take whatever the projection
 * gives it. p = 0 in the Air column is a Dirichlet condition, so the outlet
 * face comes out of subtractGradient as
 *
 *     u_outlet = u* + gradScale * p_fluid
 *
 * and a vortex core is a pressure MINIMUM. When one drifts into the exit,
 * p_fluid goes far enough negative to reverse the face: the outlet starts
 * taking fluid IN. That is not merely untidy, it is an energy source. The
 * kinetic-energy budget of the domain carries a boundary term
 *
 *     dE/dt = -∮ (½|u|² + p) (u·n) dS
 *
 * which for p = 0 is -∮ ½|u|² (u·n). While the flow leaves (u·n > 0) that is
 * negative and energy drains out, as an outlet should. The instant any part of
 * the boundary reverses it flips SIGN and pumps energy in, with nothing to
 * limit it: the inflowing velocity is whatever the zero-gradient rule copied
 * outward, so the boundary feeds the interior its own values back. The loop
 * compounds — measured at 512x384 the reversal deepens monotonically
 * (-0.03 -> -0.23 -> -0.39 over t = 3..4), and at 1024x768 it runs away to
 * 17x the free-stream speed with dye flooding back in through the exit.
 *
 * Clamping the NORMAL component to the outflow direction is enough to fix the
 * sign of that integral, and is the standard backflow stabilisation. It costs
 * a little divergence in the one cell layer behind the outlet — the clamp
 * happens after the projection, so the face no longer matches the p that was
 * solved for — which the next step's solve removes. That is a real trade and
 * the cheap side of it: the alternative is an unbounded energy source.
 *
 * It is NOT the reflecting outlet that openRight() warns about. A prescribed
 * outflow profile pins every face every step; this leaves the exit free
 * whenever the flow is actually leaving, and only refuses the reversal.
 *
 * Runs after subtractGradient, so it sees the PROJECTED velocity. A no-op on
 * a closed box. Covers all four edges even though only openRight() exists
 * today — an open top (buoyant plume) and painted Air will need them.
 */
export function applyOutflow(g: Grid, u: FieldArray, v: FieldArray, label: Uint8Array): void {
  // Each edge clamps toward ITS OWN outward normal, hence min on the low
  // edges and max on the high ones. The clamped value is what gets copied
  // outward, so the ghost face can never disagree with the outlet face.
  for (let j = 0; j < g.ny; j++) {
    if (label[idxP(g, 0, j)] === Cell.Air) {
      const out = Math.min(u[idxU(g, 1, j)], 0);
      u[idxU(g, 1, j)] = out;
      u[idxU(g, 0, j)] = out;
    }
    if (label[idxP(g, g.nx - 1, j)] === Cell.Air) {
      const out = Math.max(u[idxU(g, g.nx - 1, j)], 0);
      u[idxU(g, g.nx - 1, j)] = out;
      u[idxU(g, g.nx, j)] = out;
    }
  }
  for (let i = 0; i < g.nx; i++) {
    if (label[idxP(g, i, 0)] === Cell.Air) {
      const out = Math.min(v[idxV(g, i, 1)], 0);
      v[idxV(g, i, 1)] = out;
      v[idxV(g, i, 0)] = out;
    }
    if (label[idxP(g, i, g.ny - 1)] === Cell.Air) {
      const out = Math.max(v[idxV(g, i, g.ny - 1)], 0);
      v[idxV(g, i, g.ny - 1)] = out;
      v[idxV(g, i, g.ny)] = out;
    }
  }
}
