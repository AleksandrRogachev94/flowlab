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

import type { Grid, PerChannel } from '../core/grid.ts';
import type { DyeSeed, DyeSource, SceneSpec } from '../core/simulation.ts';
import type { DyePalette } from '../viz/colormaps.ts';
import { airfoilChannel, leadingEdgeY } from './airfoil.ts';
import { wallJet } from './emitters.ts';
import { draught, hotVent } from './plume.ts';
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
   *
   * A triple sets it per channel, which is what a scene with a temperature in
   * it needs: heat cools and soot does not.
   */
  decay?: number | PerChannel;
  /**
   * Buoyancy weight on dye channel 0 — see core/buoyancy.ts. Omitted on every
   * scene whose dye is a passive tracer, which is all of them but one.
   *
   * A scene setting this is declaring that its channel 0 is a TEMPERATURE, and
   * that is the one place the meaning lives: nothing in core/ knows what a
   * channel is for, so this, `decay` and `palette` have to agree with each
   * other and with the emitter that fills them.
   */
  buoyancy?: number;
  /** How those channels become a colour. Defaults to straight RGB — see
   *  viz/colormaps.ts. */
  palette?: DyePalette;
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

/** The two-flame variant's vents, shared by its seed and its source for the
 *  same reason `jet` is one object: two lists that must agree. */
const TWIN = [0.36, 0.64];

export const SCENES: Scene[] = [
  /**
   * The one scene where the dye is not passive: channel 0 is a TEMPERATURE, and
   * `buoyancy` is what makes it lift the fluid carrying it (core/buoyancy.ts).
   *
   * WHY NOT "FIRE", which is what this was called and what it plainly looks
   * like. There is no combustion here — no fuel, no reaction, no flame sheet.
   * There is a strip of floor held hot and a body force proportional to
   * temperature, and what that produces is a THERMAL PLUME: the same flow over
   * a candle, a radiator, a chimney and a cumulus cloud. A real fire's visible
   * shape IS its plume, so the picture is honest and the blurb can say so; the
   * label should not claim chemistry the solver has never heard of. It also
   * puts the scene in the same register as its neighbours, which are named for
   * flows ("vortex street", "wall jet") and not for objects.
   *
   * THE ONLY SCENE WITH NO GEOMETRY AND NO IMPOSED FLOW, and that is the point
   * of it rather than an omission. Every other scene here is a body in a
   * stream — the flow is arranged at an inlet and the interest is in what the
   * body does to it. This one has no inlet and no obstacle: the column, its
   * flapping, the vortex rings peeling off it, the roll under the ceiling and
   * the return flow down the walls are ALL the projection's answer to one body
   * force. Putting a cylinder in front of it would only hand the eye something
   * familiar to look at instead.
   *
   * The one velocity it is given is scenes/plume.ts's `draught`, and it is a
   * TIE-BREAK rather than a flow: a few percent of plume speed, at box scale,
   * written once. A perfectly mirror-symmetric box has a mirror-symmetric
   * answer, and that answer is a straight column under one frozen mushroom cap
   * — correct, and not a thing that happens in any room. That file argues the
   * case; the short version is that suppressing the instability by exact
   * symmetry is the artificial choice, not breaking it.
   *
   * An earlier version did exactly that, and it was worse in a way worth
   * recording: a plume flaps, so it engages a fixed body on one side and then
   * the other and never simply hits it, which reads as a misplaced obstacle
   * rather than as an experiment. The brush is the better answer — the hint
   * asks for a lid, and a lid you drew yourself is an experiment where a lid
   * that shipped with the scene is furniture.
   *
   * THE TWO FADE RATES have to be read together with the vent:
   *
   *   decay [1.5, 0.1, 0]   the heat fades over ~a quarter of the box height at
   *                         plume speed (k = speed / fadeLength), which is what
   *                         keeps the bright core a FLAME shape rather than a hot column
   *                         reaching the ceiling. The smoke lasts fifteen times
   *                         as long, so it survives the whole rise, the roll
   *                         under the ceiling and the return down the walls —
   *                         and that gap is the whole reason dyeDecay had to
   *                         become per-channel. Set the two equal and the
   *                         second channel carries no information the first one
   *                         does not.
   *
   *                         The smoke rate is also what decides how much of the
   *                         FRAME carries anything: fade it as fast as the heat
   *                         and the scene is a thread on black, which is a
   *                         picture of the plume and not of the room it is
   *                         stirring. Slow enough and the box fills with the
   *                         2D inverse cascade the turbulence grid scene is
   *                         about, for free, out of the same one force.
   *   no outlet             convection needs no net boundary flux; see
   *                         scenes/plume.ts.
   */
  {
    id: 'plume',
    label: 'Thermal plume',
    blurb:
      'Hot gas weighs less than the air around it, so it rises and drags the smoke with it — the plume over every fire.',
    hint: 'Try a lid above the plume — it spreads underneath and rolls back down.',
    // The one-time symmetry break, and the only velocity this scene is ever
    // given — scenes/plume.ts says why a perfectly symmetric box produces a
    // frozen mushroom instead of a flickering column.
    build: () => ({ seed: draught() }),
    dyes: [
      { id: 'flame', label: 'One flame', source: () => hotVent() },
      {
        id: 'twin',
        label: 'Twin flames',
        source: () => hotVent({ centres: TWIN, halfWidth: 0.13, taper: 0.05 }),
      },
    ],
    decay: [1.5, 0.1, 0],
    buoyancy: 4,
    palette: 'fire',
  },
  /**
   * SECOND, and the pair at the top of this list is deliberate. Both fill the
   * whole frame with motion, which is what a landing view has to do — every
   * scene below is one body and its wake, legible but quiet over most of the
   * picture until you know where to look. They fill it by opposite routes,
   * and that is the argument for showing them together: the plume has no
   * geometry and makes its structure out of one force, this one has eight rods
   * and makes its structure out of them. The order between them is the order
   * of stuff on the screen — nothing, then rods.
   *
   * The rest run outward from there: one body (vortex street), one body shaped
   * on purpose (wing section), no body at all but a wall (wall jet), and then
   * the two that are really velocity fields rather than experiments.
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
