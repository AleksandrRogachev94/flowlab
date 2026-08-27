/**
 * Kármán vortex street: uniform flow past a circular cylinder. The headline
 * scene (PLAN.md §3).
 *
 * The domain is one unit tall by construction (Simulation sets h = 1/ny), so
 * every length here reads as a fraction of the channel height. The benchmark's
 * standard sizing rules: blockage D/H <= ~0.15, >= 5D upstream, >= 15D
 * downstream. Wake length is measured in DIAMETERS, so a longer street wants a
 * SMALLER cylinder rather than a wider box — but a smaller one is also more
 * staircase than circle. Measured across that trade, D = 0.11 gave the
 * textbook-closest St (0.195 against 0.180 at D = 0.08), and that is the value
 * to set if a number is what you want.
 *
 * The DEFAULT is 0.14, and it is a legibility choice rather than a fluid one.
 * The smoke rake runs 22 lines across the height, so a line every 0.045; at
 * D = 0.11 only about two and a half of them ever cross the body, which puts
 * the split that generates the entire street at the smallest visible scale in
 * the frame. 0.14 puts three across it, stays inside the blockage guideline,
 * and costs roughly 12D of downstream for 9.5D — still three or four vortex
 * pairs before the outlet. Going much past 0.15 is where the walls start
 * setting the shedding frequency instead of the body.
 *
 * WHAT SETS THE REYNOLDS NUMBER: nothing here. There is no viscosity term, so
 * the effective Re is whatever semi-Lagrangian dissipation makes it — set by
 * RESOLUTION, not by any knob. Expect a plausible street; do not expect a
 * trustworthy Strouhal number until real viscosity exists.
 *
 * Geometry and velocity only. Dye is the catalog's business: a scene is a
 * flow, and which tracer rides it is the viewer's choice. What this file owes
 * the catalog is `centreY` — the stagnation streamline's height, which is
 * where the two-tone rake splits its colours (see scenes/catalog.ts).
 */

import { type Grid } from '../core/grid.ts';
import type { SceneSpec, Seed } from '../core/simulation.ts';
import { allLabels, openRight, solidDisk } from './obstacles.ts';

export interface KarmanOptions {
  /** Free-stream speed. */
  speed: number;
  /** Cylinder diameter, in world units (the channel is 1 tall). */
  diameter: number;
  /** Cylinder centre x, in world units. */
  cx: number;
}

export const defaultKarman: KarmanOptions = {
  speed: 1,
  diameter: 0.14,
  cx: 0.45,
};

/**
 * Cylinder centre height: a QUARTER cell off the channel axis, and the
 * fraction matters.
 *
 * Shedding is an instability: it amplifies an asymmetry, it does not create
 * one. Everything else here is symmetric about the axis — inflow, walls, and
 * the sweep-direction cycling in solvePressure, which was made symmetric on
 * purpose — so a symmetric wake stays symmetric until roundoff grows, which
 * can take longer than anyone watches.
 *
 * Half a cell would NOT do it. The disk is rasterized by cell centres at
 * y = (j + 0.5)h, so a disk on the axis and one shifted by exactly h/2 both
 * land symmetrically: one straddling a row, the other centred on it. A
 * quarter cell makes the top and bottom staircases genuinely different.
 *
 * Takes the grid because this is the one length here that cannot be
 * resolution-independent. Exported because it is also the STAGNATION
 * STREAMLINE's height: the free stream divides here, everything above the line
 * feeds the upper shear layer and everything below feeds the lower one, and
 * that is exactly where the two-tone rake has to change colour for the two
 * tints to mean "which layer shed this vortex".
 */
export function centreY(g: Grid): number {
  return 0.5 + 0.25 * g.h;
}

/** Cylinder, outlet and free stream. No dye — see the file comment. */
export function karmanChannel(g: Grid, options: Partial<KarmanOptions> = {}): SceneSpec {
  const { speed, diameter, cx } = { ...defaultKarman, ...options };

  /**
   * Uniform through-flow everywhere, walls included — an exact steady solution
   * of the discrete system (emitters.test asserts that), so at t = 0 the only
   * source of divergence is the cylinder's own zeroed faces. That is an
   * impulsive start, the standard way this benchmark begins. u[0,j] is never
   * written again, so it stands as the prescribed inlet for the whole run.
   */
  const seed: Seed = (_g, u) => u.fill(speed);

  return {
    labels: allLabels(openRight(), solidDisk(cx, centreY(g), 0.5 * diameter)),
    seed,
  };
}
