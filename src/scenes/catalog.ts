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
import { airfoilChannel, leadingEdgeY } from './airfoil.ts';
import { wallJet } from './emitters.ts';
import { centreY as karmanCentreY, karmanChannel } from './karman.ts';
import { openRight } from './obstacles.ts';
import { turbulenceGridChannel } from './turbulenceGrid.ts';
import {
  addDyeMono,
  addDyeStripes,
  addDyeTriad,
  addVortexCluster,
  addVortexPair,
  linePeriodCells,
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
   * are placed against the geometry and so need the grid — the two-tone rake
   * splits its colours at the cylinder's centreline or the splitter plate's,
   * both of which sit a quarter cell off the axis.
   */
  source?: (g: Grid) => DyeSource;
}

export interface Scene {
  id: string;
  label: string;
  /** One sentence, for someone who has never heard of a vortex street. */
  blurb: string;
  /**
   * One experiment to try with the brush, named by its OUTCOME rather than by
   * the gesture — ui/controls.ts's GESTURE constant already says shift-drag
   * draws and a plain drag stirs, once, ahead of every scene's hint, so this
   * only has to say what to try and what should happen.
   *
   * The outcome matters because a blank invitation to draw is not a question:
   * left to themselves people scribble for five seconds and stop. Told that a
   * bar behind the cylinder will WEAKEN the shedding, they are running an
   * experiment with a prediction to check, and the flow answers.
   *
   * The two — the stroke and its outcome — had better match, and that is worth
   * restating even with the gesture said elsewhere: a hint whose outcome does
   * not happen is worse than no hint, since the viewer concludes the brush is
   * broken rather than that they aimed badly.
   */
  hint?: string;
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
 * Streaklines released along the whole inlet — the wind tunnel's smoke rake,
 * and the default on every channel scene: a solid band of dye renders as a
 * flat slab whatever the flow does to it, while a thin line is a MATERIAL
 * line, so its every fold is the strain field made visible. It is also what
 * makes the dye view as sharp as the vorticity view, which is the comparison
 * worth passing.
 *
 * Line COUNT and line THICKNESS, and they had to be separated before either
 * could be set. Under the old raised-cosine profile one exponent controlled
 * both, so thinning the lines also faded them and cutting the count to keep
 * them distinct emptied the frame — a dozen soft ribbons read as neither a
 * photograph nor a texture. ribbonAt's top hat makes them independent: 22
 * lines for density, 0.4 of the period bright, so the gaps are real black and
 * every line has an actual edge.
 *
 * Spacing is tied to the domain height rather than to a cell count, so the
 * picture holds the same 22 lines at every resolution — see linePeriodCells,
 * which carries that argument and the floor that goes with it.
 *
 * They span the full height on purpose. The cylinder is 0.14 of the channel
 * and the wing not much more, so most lines never come near the body and stay
 * dead straight — and that is the CONTROL, the thing that says the free stream
 * is undisturbed and everything happening in the middle is the body's doing.
 */
const LINES = 22;
const DUTY = 0.4;
const linePeriod = (g: Grid): number => linePeriodCells(g, LINES);
const SMOKE = [0.85, 0.95, 1] as const;

/**
 * The two-tone pair, and the colours are picked as much for how they MIX as
 * for how they look apart.
 *
 * Dye is composited straight to RGB with no colormap and no normalization (see
 * viz/dye.wgsl), so a tint is a literal screen colour and two tints that
 * interleave below the cell scale average to their mean. Warm amber and cool
 * cyan average to something close to white, which is the brightest thing the
 * picture can show — so mixing announces itself rather than muddying, and the
 * same frame carries both "which side did this come from" and "where has the
 * scheme stopped resolving the interface". A pair of neighbouring hues would
 * average to a third hue and lose that reading entirely.
 *
 * Neither is a saturated primary. One-hot RGB was what the flat inlet bands
 * used, and full-gamut red/green/blue on black reads as a chart rather than as
 * smoke; pulling each tint off its axis costs nothing in separability and
 * buys a picture that looks lit.
 */
const WARM = [1, 0.55, 0.14] as const;
const COOL = [0.18, 0.74, 1] as const;

/**
 * The rake, split in two at a streamline the scene chooses. testFields.ts's
 * StripeOptions.tint carries the argument for the split itself.
 *
 * The split is SNAPPED to the middle of a gap, and that is not a detail. Left
 * where the scene asks for it, the dividing line lands wherever it lands —
 * usually part way through a ribbon, which then comes out warm on its lower
 * half and cool on its upper one. One bi-coloured line among twenty single
 * ones reads as a rendering fault rather than as information, and it is worse
 * than it sounds: that ribbon is the one nearest the body, so it is the line
 * the eye follows. Rounding to the nearest gap centre moves the boundary by at
 * most half a period — 0.02 of the channel height, well under the thing being
 * marked — and every line comes out one colour.
 *
 * `splitOf` is a function of the grid because no scene's dividing line is a
 * constant: the cylinder's is quarter-cell offset, the wing's moves with the
 * angle of attack.
 *
 * There is deliberately no brightness taper. One was tried — full strength
 * over the body, easing to a floor at the walls, on the theory that a wing
 * occupying a fifth of the height leaves too much of the frame at full
 * brightness doing nothing. In the picture it earns nothing: the flow already
 * decides where the eye goes, because straight lines are quiet and folded ones
 * are not, and dimming the quiet ones only makes the frame darker. The lines
 * stay uniform and the flow does the emphasis.
 */
const twoToneIn = (splitOf: (g: Grid) => number): DyeOption => ({
  id: 'two-tone',
  label: 'Two-tone smoke',
  source: (g) => {
    const period = linePeriod(g) * g.h;
    const yc = (Math.round(splitOf(g) / period - 0.5) + 0.5) * period;
    return stripeInflow({
      periodCells: linePeriod(g),
      duty: DUTY,
      tint: (y) => (y < yc ? WARM : COOL),
    });
  },
});

const smokeIn: DyeOption = {
  id: 'smoke-lines',
  label: 'White smoke',
  source: (g) => stripeInflow({ periodCells: linePeriod(g), duty: DUTY, tint: SMOKE }),
};
const stripesIn: DyeOption = {
  id: 'stripes',
  label: 'Color lines',
  source: (g) => stripeInflow({ periodCells: linePeriod(g), duty: DUTY }),
};

// Seed and source come from one factory so they cannot disagree on the band.
const jet = wallJet();

export const SCENES: Scene[] = [
  /**
   * FIRST, so it is what the demo opens on. Every other scene is one body and
   * its wake — legible, and quiet over most of the frame until you know what
   * to look at. This one is moving everywhere from the first second, which is
   * what a landing view has to be.
   */
  {
    id: 'grid',
    label: 'Turbulence grid',
    blurb: 'A row of rods trips the stream; the small wakes merge into ever larger eddies.',
    hint: 'Try a wall with a gap in it — the churn funnels into a jet.',
    build: (g) => turbulenceGridChannel(g),
    /**
     * The HUE RAMP leads here, and it is the one scene where it beats the two
     * tints on their own terms rather than only on looks.
     *
     * Two colours answer "which half of the channel did this come from", which
     * is the right question when there are exactly two populations — a
     * cylinder's upper and lower shear layer, a wing's suction and pressure
     * side. This scene has eight sources, and what is worth watching is the
     * length scale GROWING with distance as vortices merge. The ramp labels
     * every line by the height it entered at, so that growth is legible
     * directly: a magenta filament wound into a green eddy is transport across
     * a third of the channel by a structure far larger than the rods that made
     * it, and it says so in one glance without needing an interface to watch.
     *
     * Two-tone stays one entry down, because the large-scale interface it
     * draws is the single clearest statement of the same fact.
     */
    dyes: [stripesIn, twoToneIn(() => 0.5), smokeIn, none],
  },
  {
    id: 'jet',
    label: 'Wall jet',
    blurb: 'A nozzle in the left wall; the confined jet entrains and recirculates.',
    hint: 'The emptiest scene — good one to build in.',
    // openRight() is not optional: wallJet is pure inflow, so without an outlet
    // the all-Neumann system is inconsistent and never converges.
    build: () => ({ labels: openRight(), seed: jet.seed }),
    dyes: [{ id: 'smoke', label: 'Smoke', source: () => jet.source }, none],
    decay: 0.5,
  },
  {
    id: 'karman',
    label: 'Vortex street',
    blurb: 'Flow past a cylinder, shedding vortices left and right in turn.',
    hint: 'Try a long bar back along the wake — the street weakens and stretches.',
    build: (g) => karmanChannel(g),
    // Two tints lead on the bodies: there are exactly two populations of fluid
    // and the wake sheds alternately from each, so the colours arrive one per
    // vortex. See twoToneIn.
    dyes: [twoToneIn(karmanCentreY), stripesIn, smokeIn, none],
  },
  {
    id: 'airfoil',
    label: 'Wing section',
    blurb: 'A cambered wing past its stalling angle; the flow lets go of the upper surface.',
    hint: 'Try a flap on the trailing edge, or a wall in front of the nose.',
    build: (g) => airfoilChannel(g),
    dyes: [stripesIn, twoToneIn(() => leadingEdgeY()), smokeIn, none],
  },
  {
    id: 'cluster',
    label: 'Vortex cluster',
    blurb: 'Vortices orbit, stretch each other into filaments, and merge.',
    hint: 'Try a wall through the middle and watch the vortices work around it.',
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
    hint: "Try a wall in the pair's path — it splits and rebounds.",
    build: () => ({ seed: addVortexPair }),
    dyes: [stripes, triad, blob, none],
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
