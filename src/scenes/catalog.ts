/**
 * The demo's menu: every scene the UI offers, and the dye choices that make
 * sense inside each one.
 *
 * This file exists because scene and tracer are NOT freely combinable, and
 * pretending otherwise is what made the old two-list arrangement confusing.
 * A tracer that is only SEEDED washes straight out of a channel with an
 * outlet; a tracer that is re-stamped every step needs a full-width inflow to
 * be stamped into. So the flow decides which tracers are meaningful, and the
 * scene carries that list — one dropdown whose contents follow the other,
 * rather than two independent cycles with a compatibility table between them
 * (which is exactly what main.ts used to hold).
 *
 * Scenes here supply geometry and velocity ONLY. Dye is always the DyeOption's
 * business, so there is exactly one place a tracer can come from and no way
 * for two of them to end up composited on top of each other.
 */

import type { Grid } from '../core/grid.ts';
import type { DyeSeed, DyeSource, SceneSpec } from '../core/simulation.ts';
import { wallJet } from './emitters.ts';
import { karmanBands, karmanChannel } from './karman.ts';
import { openRight } from './obstacles.ts';
import {
  addDyeMono,
  addDyeStripes,
  addDyeTriad,
  addVortexCluster,
  addVortexPair,
  stripeInflow,
} from './testFields.ts';

export interface DyeOption {
  id: string;
  /** Shown in the dropdown. Plain words: most viewers are not fluid people. */
  label: string;
  /** Written once, at reset. */
  seed?: DyeSeed;
  /**
   * Re-stamped every step, so the tracer keeps arriving instead of washing out
   * through an outlet. A factory rather than a DyeSource, because some sources
   * are placed against the geometry and so need the grid — karman's bands are
   * centred on the cylinder, which sits a quarter cell off the axis.
   */
  source?: (g: Grid) => DyeSource;
}

export interface Scene {
  id: string;
  label: string;
  /** One sentence, for someone who has never heard of a vortex street. */
  blurb: string;
  /** Geometry and velocity. Never dye — see the file comment. */
  build: (g: Grid) => SceneSpec;
  /** First entry is what the scene opens with. */
  dyes: DyeOption[];
  /**
   * Dye fade, per second of sim time. Only recirculating scenes need it: a
   * permanent source in a closed box saturates every cell otherwise. Tune it
   * as a DISTANCE — k = speed / fadeLength.
   */
  decay?: number;
}

/** Domain width in world units; the height is always 1 (h = 1/ny). */
const widthOf = (g: Grid): number => g.nx * g.h;

const none: DyeOption = { id: 'none', label: 'None' };

// Seeded once: for closed boxes, where the fluid recirculates forever and
// nothing ever leaves.
const triad: DyeOption = { id: 'triad', label: 'Three blobs', seed: addDyeTriad };
const blob: DyeOption = { id: 'mono', label: 'Single blob', seed: addDyeMono };
const stripes: DyeOption = { id: 'stripes', label: 'Stripes', seed: addDyeStripes };

/**
 * Streaklines released along the whole inlet — the wind tunnel's smoke wire,
 * and the reason it is the default on the channel scenes: a solid band of dye
 * renders as a flat slab whatever the flow does to it, while a thin line is a
 * MATERIAL line, so its every fold is the strain field made visible. It is
 * also what makes the dye view as sharp as the vorticity view, which is the
 * comparison worth passing.
 *
 * Line SPACING is tied to the domain height, not to a cell count, so the
 * picture has the same ~20 lines in it at every resolution — the eye's limit
 * is the number of lines it can follow, and that does not change with the
 * grid. Floored at 8 cells because below that the strain thins a band past
 * the grid within a few diameters and the lines alias into moire.
 *
 * Two colourings, and they are two different instruments — see stripeAt. Cool
 * white for the photograph; the hue ramp when the question is WHICH line.
 */
/**
 * Line COUNT and line THICKNESS, and they had to be separated before either
 * could be set. Under the old raised-cosine profile one exponent controlled
 * both, so thinning the lines also faded them and cutting the count to keep
 * them distinct emptied the frame — a dozen soft ribbons read as neither a
 * photograph nor a texture. ribbonAt's top hat makes them independent: 22
 * lines for density, 0.4 of the period bright, so the gaps are real black and
 * every line has an actual edge.
 *
 * Spacing is tied to the domain height rather than to a cell count, so the
 * picture holds the same ~22 lines at every resolution — the eye's limit is
 * how many lines it can follow, and that does not change with the grid.
 * Floored at 8 cells of period because below that the strain thins a ribbon
 * past the grid within a few diameters and the lines alias into moire.
 *
 * They span the full height on purpose. The cylinder is only 0.11 of the
 * channel, so most lines never come near it and stay dead straight — and that
 * is the CONTROL, the thing that says the free stream is undisturbed and
 * everything happening in the middle is the body's doing.
 */
const LINES = 22;
const DUTY = 0.4;
const linePeriod = (g: Grid): number => Math.max(8, Math.round(g.ny / LINES));
const SMOKE = [0.85, 0.95, 1] as const;

const smokeIn: DyeOption = {
  id: 'smoke-lines',
  label: 'Smoke lines',
  source: (g) => stripeInflow({ periodCells: linePeriod(g), duty: DUTY, tint: SMOKE }),
};
const stripesIn: DyeOption = {
  id: 'stripes',
  label: 'Colour lines',
  source: (g) => stripeInflow({ periodCells: linePeriod(g), duty: DUTY }),
};

// Seed and source come from one factory so they cannot disagree on the band.
const jet = wallJet();

export const SCENES: Scene[] = [
  {
    id: 'karman',
    label: 'Vortex street',
    blurb: 'Flow past a cylinder, shedding vortices left and right in turn.',
    build: (g) => karmanChannel(g),
    dyes: [
      smokeIn,
      stripesIn,
      { id: 'bands', label: 'Inlet bands', source: (g) => karmanBands(g) },
      none,
    ],
  },
  {
    id: 'cluster',
    label: 'Vortex cluster',
    blurb: 'Vortices orbit, stretch each other into filaments, and merge.',
    // Vortex COUNT scales with the box, blob size does not. The blobs are
    // sized against the height (sigma is in world units and the domain is one
    // unit tall), so on a wide screen a fixed 14 of them leaves most of the
    // frame still — the density is what has to be held constant, not the
    // number. Sign still alternates, so net circulation stays ~0.
    build: (g) => ({
      seed: (gg, u, v) => addVortexCluster(gg, u, v, 1, Math.round(14 * widthOf(g))),
    }),
    dyes: [stripes, triad, blob, none],
  },
  {
    id: 'dipole',
    label: 'Dipole',
    blurb: 'A counter-rotating pair, each vortex carried by the other one’s flow.',
    build: () => ({ seed: addVortexPair }),
    dyes: [stripes, triad, blob, none],
  },
  {
    id: 'jet',
    label: 'Wall jet',
    blurb: 'A nozzle in the left wall; the confined jet entrains and recirculates.',
    // openRight() is not optional: wallJet is pure inflow, so without an outlet
    // the all-Neumann system is inconsistent and never converges.
    build: () => ({ labels: openRight(), seed: jet.seed }),
    dyes: [{ id: 'smoke', label: 'Smoke', source: () => jet.source }, none],
    decay: 0.5,
  },
];

/** The scene's own dye choice, by id, falling back to its default. */
export function dyeOf(scene: Scene, id: string): DyeOption {
  return scene.dyes.find((d) => d.id === id) ?? scene.dyes[0];
}

/** Everything reset() needs for one (scene, dye) pair — the whole combination
 *  rule, in one place, with no flags. */
export function sceneSpec(scene: Scene, dye: DyeOption, g: Grid): SceneSpec {
  return { ...scene.build(g), dye: dye.seed, dyeSource: dye.source?.(g) };
}
