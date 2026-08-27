/**
 * A row of rods across the inlet — a turbulence grid, and the classic way a
 * wind tunnel makes turbulence out of a smooth stream.
 *
 * WHY THIS ONE WORKS WHERE A SHEAR LAYER DID NOT. A free mixing layer is
 * CONVECTIVELY unstable: it amplifies disturbances as they pass but generates
 * none, so with no time-dependent forcing at the inlet it sits there flat, and
 * this project has no force stage to shake it with. A bluff body's wake is
 * ABSOLUTELY unstable — it oscillates on its own, out of nothing, which is the
 * entire reason the vortex street scene works from an impulsive start. Every
 * rod here is one of those. There is no forcing to arrange because each rod
 * is its own oscillator.
 *
 * WHAT THE PICTURE IS ABOUT, and it is a different thing from the single
 * cylinder. Near the grid the wakes are separate and the vortices are all
 * roughly one size, set by the rod diameter. Downstream they interact, and in
 * TWO DIMENSIONS interacting vortices merge: like-signed neighbours orbit and
 * coalesce into one larger vortex, again and again. So the eddies get BIGGER
 * with distance rather than breaking up into smaller ones. That is the inverse
 * cascade, the headline structural fact about two-dimensional turbulence and
 * the exact opposite of what a three-dimensional flow does — and this scene
 * shows it happening across the frame, in one shot, with a length scale that
 * you can watch grow from left to right.
 *
 * It is also the scene where the dye earns the most. The two-tone rake splits
 * at mid-height, so the colours report large-scale STIRRING: growing tongues
 * of each colour reaching across the centreline is mixing being done by
 * structures far larger than anything the grid injected.
 */

import { type Grid } from '../core/grid.ts';
import type { LabelSeed, SceneSpec, Seed } from '../core/simulation.ts';
import { allLabels, openRight, solidDisk } from './obstacles.ts';

export interface TurbulenceGridOptions {
  /** Free-stream speed. */
  speed: number;
  /**
   * Rod count across the height. A COUNT and not a density, unlike the vortex
   * cluster's blob count: the grid spans the channel height, which is always
   * exactly 1, so the same number is the same physical grid at any aspect
   * ratio. Widening the window gives more DOWNSTREAM to watch the cascade in,
   * which is what the extra room should buy.
   */
  count: number;
  /** Rod diameter, in world units. */
  diameter: number;
  /** The row's x position, in world units. */
  cx: number;
}

/**
 * Solidity — the blocked fraction, diameter/spacing — is what actually sets
 * how hard a grid trips the flow, and 8 rods of 0.045 give 0.36. Wind tunnel
 * practice is 0.3 to 0.4 and the ends of that range are ends for real reasons:
 * below ~0.3 the wakes never merge and the picture stays eight tidy separate
 * streets, while above ~0.45 the jets between the rods are strong enough to
 * coalesce into a few large channels instead, which is a blockage effect
 * rather than turbulence.
 *
 * The rods have to stay big enough to RESOLVE as well. At 0.045 a rod is 38
 * cells across on the high preset and 9 on the low one; below about 6 the
 * staircase is coarse enough that the shedding frequency starts being a
 * property of the rasterization.
 */
export const defaultTurbulenceGrid: TurbulenceGridOptions = {
  speed: 1,
  count: 8,
  diameter: 0.045,
  cx: 0.28,
};

export function turbulenceGridChannel(
  g: Grid,
  options: Partial<TurbulenceGridOptions> = {},
): SceneSpec {
  const { speed, count, diameter, cx } = { ...defaultTurbulenceGrid, ...options };

  const rods: LabelSeed[] = [];
  for (let k = 0; k < count; k++) {
    /**
     * A quarter cell off the lattice, ALTERNATING in sign — karman.ts's
     * centreY trick, applied per rod, and the alternation is the part worth
     * having.
     *
     * Each rod needs its own asymmetry or its wake has nothing to amplify. A
     * single shared offset would give every rod an identical staircase, so
     * all eight would start shedding in phase and the near field would be one
     * flat wall of vortices rather than eight independent wakes. Alternating
     * puts neighbours in antiphase, which is also what a real grid settles
     * into.
     */
    const cy = (k + 0.5) / count + (k % 2 ? 0.25 : -0.25) * g.h;
    rods.push(solidDisk(cx, cy, 0.5 * diameter));
  }

  // Uniform through-flow, impulsively started, and the inlet column stands as
  // the prescribed free stream for the whole run — karman.ts's seed, unchanged.
  const seed: Seed = (_gg, u) => u.fill(speed);

  return { labels: allLabels(openRight(), ...rods), seed };
}
