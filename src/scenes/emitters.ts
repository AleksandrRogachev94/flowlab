/**
 * A jet entering through the left wall, plus the dye that rides it.
 *
 * No solver changes needed: a wall face like u[0,j] is already prescribed
 * boundary data — advectVelocity copies it through, subtractGradient skips
 * it, and computeDivergence reads it into the pressure RHS (which is
 * Bridson's u_solid correction, already folded in). Write it once at reset
 * and it holds for the whole run.
 *
 * The catch is the flux budget, and it is the SCENE's to settle, not this
 * file's. An all-Neumann box is solvable only if total boundary flux is zero;
 * net inflow makes the system inconsistent, and SOR then burns every sweep
 * without crossing its residual floor while p drifts by a growing constant.
 *
 * This emitter is therefore pure inflow, and every scene using it must pair it
 * with an outlet — `openRight()` from obstacles.ts, an Air column pinned at
 * p = 0. An earlier version balanced the books itself by pushing the same flux
 * back out across the opposite wall; that kept the solve consistent but
 * PRESCRIBED the exit profile, which makes the boundary reflect anything that
 * reaches it. An open boundary lets the flow choose its own way out.
 */

import { idxP, idxU, type Grid } from '../core/grid.ts';
import type { DyeSource, Seed } from '../core/simulation.ts';

/** smoothstep, clamped. */
function ramp(t: number): number {
  const c = Math.min(Math.max(t, 0), 1);
  return c * c * (3 - 2 * c);
}

export interface WallJetOptions {
  /** Peak inflow speed, normal to the wall. */
  speed: number;
  /** Emitting band, in WORLD units — like the disk radii in testFields.ts. */
  y0: number;
  y1: number;
  /**
   * Smoothstep ramp at each end of the band, in WORLD units (clamped to half
   * the band). Not cells: a cell-width ramp shrinks as N grows, which drifts
   * the inlet flux 20% between N=64 and N=512 and makes two resolutions a
   * different scene. 0 gives a top-hat, which quantizes the band to whole
   * cells instead — also N-dependent, and it seeds the shear-layer
   * instability ~3x harder by putting the velocity jump on one face.
   */
  taper: number;
  /** Dye source thickness, in cells inward from the wall. */
  depthCells: number;
  /** Per-channel value the dye source holds. */
  colour: readonly [number, number, number];
}

export const defaultWallJet: WallJetOptions = {
  speed: 1,
  y0: 0.42,
  y1: 0.58,
  taper: 0.02,
  depthCells: 3,
  colour: [1, 1, 1], // smoke; the RGB triad is a diffusion probe, not a look
};

/** Seed and source come from one factory so they cannot disagree on the band. */
export function wallJet(options: Partial<WallJetOptions> = {}): {
  seed: Seed;
  source: DyeSource;
} {
  const { speed, y0, y1, taper, depthCells, colour } = { ...defaultWallJet, ...options };
  const t = Math.min(Math.max(taper, 0), 0.5 * (y1 - y0));

  // u[0,j] and the centre of cell (i,j) both sit at y = (j + 0.5)h, so
  // velocity and dye share this profile with no half-cell correction.
  const weight = (g: Grid, j: number): number => {
    const y = (j + 0.5) * g.h;
    if (t <= 0) return y >= y0 && y <= y1 ? 1 : 0;
    return ramp((y - y0) / t) * ramp((y1 - y) / t);
  };

  // ADDs, like the seeds in testFields.ts, so a jet composes with a scene.
  // v is untouched: v[0,j] is an interior face the projection owns.
  const seed: Seed = (g, u) => {
    for (let j = 0; j < g.ny; j++) u[idxU(g, 0, j)] += speed * weight(g, j);
  };

  // Plain assignment, and no taper on the dye: weighting it toward `colour`
  // instead converges to the same value within a few steps anyway (measured:
  // 0.7% difference in the whole field after 600 steps), so the weight was
  // complexity with no effect. Overwriting is what makes a source a Dirichlet
  // condition on dye, and keeps it in [0, 1] with no clamp.
  const source: DyeSource = (g, dye) => {
    const depth = Math.min(depthCells, g.nx);
    for (let j = 0; j < g.ny; j++) {
      if (weight(g, j) <= 0) continue;
      for (let i = 0; i < depth; i++) {
        const k = idxP(g, i, j);
        for (let c = 0; c < dye.length; c++) dye[c][k] = colour[c];
      }
    }
  };

  return { seed, source };
}
