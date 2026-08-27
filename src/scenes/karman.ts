/**
 * Kármán vortex street: uniform flow past a circular cylinder. The headline
 * scene (PLAN.md §3).
 *
 * The domain is one unit tall by construction (Simulation sets h = 1/ny), so
 * every length here reads as a fraction of the channel height. The benchmark's
 * standard sizing rules: blockage D/H <= ~0.15, >= 5D upstream, >= 15D
 * downstream. D = 0.11 on a 16:9 grid gives blockage 0.11, 4D upstream and 12D
 * downstream. Wake length is measured in DIAMETERS, so a longer street wants a
 * SMALLER cylinder rather than a wider box — but a smaller one is also more
 * staircase than circle. Measured across that trade, D = 0.11 gave the
 * textbook-closest St (0.195 against 0.180 at D = 0.08).
 *
 * WHAT SETS THE REYNOLDS NUMBER: nothing here. There is no viscosity term, so
 * the effective Re is whatever semi-Lagrangian dissipation makes it — set by
 * RESOLUTION, not by any knob. Expect a plausible street; do not expect a
 * trustworthy Strouhal number until real viscosity exists.
 *
 * Geometry+velocity and dye are two separate exports. They share the cylinder
 * sizing through `defaultKarman`, but nothing else: a scene is a flow, and
 * which tracer rides it is the viewer's choice (see scenes/catalog.ts).
 */

import { inDyeCells, makeDyePatch } from '../core/dye.ts';
import { type Grid } from '../core/grid.ts';
import type { DyeSource, SceneSpec, Seed } from '../core/simulation.ts';
import { allLabels, openRight, solidDisk } from './obstacles.ts';

export interface KarmanOptions {
  /** Free-stream speed. */
  speed: number;
  /** Cylinder diameter, in world units (the channel is 1 tall). */
  diameter: number;
  /** Cylinder centre x, in world units. */
  cx: number;
  /** Dye source thickness, in cells inward from the inlet. */
  depthCells: number;
}

export const defaultKarman: KarmanOptions = {
  speed: 1,
  diameter: 0.11,
  cx: 0.45,
  depthCells: 2,
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
 * resolution-independent. The dye bands read it too, so they stay centred on
 * the body at any resolution.
 */
function centreY(g: Grid): number {
  return 0.5 + 0.25 * g.h;
}

/** Cylinder, outlet and free stream. No dye — see karmanBands. */
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

/**
 * Three horizontal dye bands at the inlet, centred on the cylinder —
 * streaklines. Each enters straight and the roll-up shears them past each
 * other, so alternating vortices read as alternating colour instead of a grey
 * blur.
 *
 * The band edges line up with the BODY, not with equal thirds. The middle
 * band is exactly one diameter wide, so it is precisely the stream tube the
 * cylinder splits — the fluid that becomes both shear layers and both
 * families of vortex cores. The outer bands are half a diameter each and
 * carry fluid that passes above or below without ever touching it. Under
 * equal thirds every colour was a MIXTURE of the two populations, which is
 * what made the middle band read as merely "thin".
 *
 * It stays thin either way, and that is physics, not a defect: the flux
 * between two streamlines is conserved, so a material band's width goes as
 * 1/speed. The stagnation tube necessarily narrows as it accelerates around
 * the shoulder, then gets wound into the cores. Total span is 2D — much
 * wider and the outer bands sail past the wake without entering it.
 *
 * EVERY ROW of the inlet columns is written, including the clean fluid outside
 * the bands, and skipping those rows was a real bug rather than an
 * optimisation. The whole left edge is inflow here, so "clean fluid arrives
 * outside the bands" is a boundary condition and has to be imposed like one.
 * Leaving those cells unwritten instead made them free: the backtrace at i = 0
 * clamps to the domain edge, so an inlet cell effectively copies itself, and
 * whatever dye reached it by the shedding's vertical velocity and by numerical
 * diffusion was then RETAINED and re-copied every step. Measured on the wake:
 * the dyed fraction of the inlet column climbed 26% -> 56% over 90 s with no
 * sign of stopping, and total dye in the domain grew with it — a dyed strip
 * spreading along the inlet, feeding the whole channel. That is what made the
 * bands look like they were stretching further and further vertically.
 */
export function karmanBands(g: Grid, options: Partial<KarmanOptions> = {}): DyeSource {
  const { diameter, depthCells } = { ...defaultKarman, ...options };
  const cy = centreY(g);

  // Coverage 1 on every row, including the clean fluid outside the bands —
  // that is the paragraph above, expressed as data.
  return (dg, vg) =>
    makeDyePatch(0, 0, Math.min(inDyeCells(depthCells, vg, dg), dg.nx), dg.ny, (_i, j, rgb) => {
      const d = (j + 0.5) * dg.h - cy;
      const inBand = Math.abs(d) < diameter;
      // Below / the body's own tube / above, and -1 for the clean fluid
      // outside — which selects no channel, so all three get 0. Assumes the
      // three RGB channels.
      const band = inBand ? (Math.abs(d) < 0.5 * diameter ? 1 : d < 0 ? 0 : 2) : -1;
      for (let c = 0; c < rgb.length; c++) rgb[c] = c === band ? 1 : 0;
      return 1;
    });
}
