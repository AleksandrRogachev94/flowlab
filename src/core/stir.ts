import { idxU, idxV, type FieldArray, type Grid } from './grid.ts';
import { isSolidCell } from './boundaries.ts';

/**
 * One frame's worth of stirring — the mouse dragging through the fluid.
 *
 * A DISPLACEMENT and not a force, and that choice is what makes the gesture
 * behave the same on a 60 fps machine and a 20 fps one.
 *
 * The obvious formulation is a force held for the frame, `u += dt * F`, which
 * is how a sustained push (buoyancy, gravity) has to be written: the same F
 * over the same wall-clock time must produce the same motion however many
 * steps it was split into. A drag is not that. Its input is not a force the
 * user holds, it is a DISTANCE the pointer travelled, and the frames only
 * decide how finely that distance was sampled. Halve the frame rate and each
 * frame's delta doubles, so a stroke of a given length delivers the same total
 * impulse either way — the rate cancels out on its own, with no dt in the
 * expression at all. Multiplying by dt as well would undo that and make a
 * slow machine stir weakly.
 */
export interface Stir {
  /** Brush centre, in world units. */
  x: number;
  y: number;
  /** Gaussian falloff radius, in world units. */
  r: number;
  /** The velocity increment at the centre — pointer displacement times gain. */
  dx: number;
  dy: number;
}

/**
 * Adds the brush's velocity bump to u and v, before the projection.
 *
 * BEFORE, and that ordering is the whole mechanism rather than a detail. The
 * bump as written is divergent — it pushes fluid into a region without taking
 * any out — and the pressure solve is precisely what turns that into motion
 * the fluid can actually make: push here and the projection is what makes it
 * come back somewhere else. Applied after subtractGradient it would instead be
 * a divergence the next step inherits, which is a leak.
 *
 * The bounds are subtractGradient's, exactly: interior faces only, and nothing
 * touching a solid. Both halves matter. Writing a domain-edge face would
 * overwrite prescribed boundary data — u[0, j] is the channel's inlet, and
 * stirring it would let the user quietly edit the free stream. Writing a
 * solid's face would break the invariant every other kernel rests on, that a
 * face bordering a solid stores the solid's velocity.
 */
export function applyStir(g: Grid, u: FieldArray, v: FieldArray, label: Uint8Array, s: Stir): void {
  const invR2 = 1 / (s.r * s.r);
  // 3 radii out the Gaussian is 1e-4 of its peak; past that the write is below
  // anything visible and the loop is the whole grid for nothing.
  const reach = 3 * s.r;
  const c0 = Math.floor((s.x - reach) / g.h);
  const c1 = Math.ceil((s.x + reach) / g.h);
  const r0 = Math.floor((s.y - reach) / g.h);
  const r1 = Math.ceil((s.y + reach) / g.h);
  const clamp = (a: number, lo: number, hi: number): number => Math.min(Math.max(a, lo), hi);

  // The two face grids have DIFFERENT interior ranges, and collapsing them to
  // one shared box silently drops a row: u's low edge is i = 1 but its rows
  // run from j = 0, and v is the transpose of that. Sharing the tighter box
  // left the bottom row of u and the left column of v unstirrable, and — worse
  // than the missing row — made the host disagree with gpu/project.wgsl's
  // kernel, which had the ranges right. Two engines differing at a wall is the
  // kind of thing that surfaces as an unreproducible bug months later.
  for (let j = clamp(r0, 0, g.ny - 1); j <= clamp(r1, 0, g.ny - 1); j++) {
    const dy = (j + 0.5) * g.h - s.y;
    for (let i = clamp(c0, 1, g.nx - 1); i <= clamp(c1, 1, g.nx - 1); i++) {
      if (isSolidCell(g, label, i - 1, j) || isSolidCell(g, label, i, j)) continue;
      const dx = i * g.h - s.x;
      u[idxU(g, i, j)] += s.dx * Math.exp(-(dx * dx + dy * dy) * invR2);
    }
  }
  for (let j = clamp(r0, 1, g.ny - 1); j <= clamp(r1, 1, g.ny - 1); j++) {
    const dy = j * g.h - s.y;
    for (let i = clamp(c0, 0, g.nx - 1); i <= clamp(c1, 0, g.nx - 1); i++) {
      if (isSolidCell(g, label, i, j - 1) || isSolidCell(g, label, i, j)) continue;
      const dx = (i + 0.5) * g.h - s.x;
      v[idxV(g, i, j)] += s.dy * Math.exp(-(dx * dx + dy * dy) * invR2);
    }
  }
}
