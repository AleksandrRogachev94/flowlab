/**
 * A hot vent in the floor, and the tie-break that lets the column flicker —
 * everything the thermal-plume scene is built on.
 *
 * It is an ordinary DyeSource, and that is the point: core/buoyancy.ts made a
 * dye channel into a temperature, so "a fire" is a rectangle of prescribed
 * dye exactly like the wind tunnel's smoke rake, and every mechanism it needs
 * (Dirichlet at the wall, applied at the end of each step, uploaded once,
 * consumed by one kernel on either engine) already existed.
 *
 * CHANNEL 0 IS HEAT, CHANNEL 1 IS SMOKE. Nothing enforces that — the meaning
 * lives entirely in the scene's `buoyancy` weights and its palette — but the
 * two have to agree, so it is stated once here and once in scenes/catalog.ts.
 *
 * NO OUTLET, and none is wanted. Convection needs no net flux through the
 * boundary: the rising plume is paid for by the same fluid sinking somewhere
 * else, which is exactly what the projection arranges in a closed box. What
 * would otherwise fill the room is smoke, and the fade handles that (see the
 * scene's `decay`) — a fire vents its heat by radiating, not by leaking it out
 * of the picture.
 */

import { makeDyePatch } from '../core/dye.ts';
import type { DyeSource, Seed } from '../core/simulation.ts';
import { ramp } from './emitters.ts';
import { addCurlOfStream } from './testFields.ts';

export interface HotVentOptions {
  /**
   * Vent centres, as fractions of the domain WIDTH. Fractions and not world
   * units because the domain is one unit tall and however many wide the window
   * is — a flame quoted at x = 0.9 would be off-screen on a square grid and
   * off-centre on every other.
   */
  centres: readonly number[];
  /**
   * Half-width of each bed, in WORLD units. World units and not cells, for the
   * reason WallJetOptions.taper gives: a cell-quoted profile is a different
   * scene at every resolution.
   */
  halfWidth: number;
  /**
   * Width of the bed's soft shoulder, also in WORLD units: the vent is at FULL
   * strength across `halfWidth - taper` either side of a centre and eases to
   * nothing over the remaining `taper`.
   *
   * A PLATEAU, and this used to be a raised cosine with no flat top. The
   * change is about the shape of the PLUME, not the shape of the bed.
   * Buoyant force is proportional to temperature (core/buoyancy.ts), so a
   * profile that reaches full strength only on the centreline barely lifts its
   * own shoulders; they get entrained sideways into the middle instead, and
   * what leaves the vent is a thread a few cells across however wide the bed
   * is. A plateau lifts its whole width at once, and the column comes out the
   * width of the fire.
   *
   * The shoulder still has to be SOFT, and soft over a fixed world distance: a
   * heat step on a single face is a Rayleigh-Taylor front that rolls up
   * immediately, so a hard-edged vent takes its structure from the
   * discretization rather than from the flow, differently at every resolution.
   *
   * What the cosine was really buying was a bed the palette could draw a
   * gradient across, and the vertical ramp below still buys that — the bed
   * fades out through its own top, which is the edge the eye actually reads.
   */
  taper: number;
  /**
   * Height of the burning bed, in WORLD units up from the floor. World and not
   * cells, unlike the wall jet's inlet depth, and the difference is not
   * cosmetic: this source's job is to SUPPLY the plume, and how much it
   * supplies per second scales with its area. A depth quoted in cells shrinks
   * with h, so the same scene would starve at high resolution and flood at low
   * — the plume was a thread at 600 rows on a three-cell vent.
   */
  height: number;
  /** Channel 0 at the vent — the temperature the flame is held at. */
  heat: number;
  /** Channel 1 at the vent — soot per unit of burnt fuel. */
  smoke: number;
}

export const defaultHotVent: HotVentOptions = {
  centres: [0.5],
  halfWidth: 0.17,
  taper: 0.06,
  height: 0.03,
  heat: 1,
  smoke: 1,
};

export function hotVent(options: Partial<HotVentOptions> = {}): DyeSource {
  const { centres, halfWidth, taper, height, heat, smoke } = {
    ...defaultHotVent,
    ...options,
  };

  // The patch spans the FULL width and carries the profile in its coverage
  // plane, rather than being one tight rectangle per vent. Coverage 0 leaves a
  // cell alone (core/dye.ts), so the empty floor between two flames costs
  // nothing but a few kilobytes of zeros — and a DyePatch is one rectangle, so
  // the alternative was a list of patches and a loop in three places.
  return (dg, g) => {
    const width = g.nx * g.h;
    // At least two rows, so a coarse grid still has a bed the backtrace cannot
    // step over in one dt.
    const depth = Math.min(Math.max(2, Math.round(height / dg.h)), dg.ny);
    return makeDyePatch(0, 0, dg.nx, depth, (i, j, rgb) => {
      const x = (i + 0.5) * dg.h;
      const y = (j + 0.5) * dg.h;
      let cover = 0;
      // ramp() clamps, so this is a flat top out to `halfWidth - taper` and a
      // smoothstep shoulder over the rest. See HotVentOptions.taper.
      for (const c of centres) {
        const d = Math.abs(x - c * width);
        if (d < halfWidth) cover = Math.max(cover, ramp((halfWidth - d) / taper));
      }
      // Hottest at the floor and easing out through the top of the bed, so the
      // source hands the column over to the flow instead of stopping dead.
      cover *= ramp((height - y) / height);
      if (cover <= 0) return 0;
      // COVERAGE carries the shape, not the value: the apply is a lerp toward
      // `rgb`, so a soft edge eases the bed in without ever writing a dark
      // value over smoke that has drifted down beside it. A tapered VALUE at
      // full coverage would scrub that away every step — and, worse, would
      // erase the top of the bed, which is where the plume is trying to leave.
      rgb[0] = heat;
      rgb[1] = smoke;
      return cover;
    });
  };
}

/**
 * The one-time kick that lets the flame flicker.
 *
 * WHY IT IS NEEDED AT ALL. The box starts at rest, all four walls are
 * symmetric and the vent is centred, so the scene is exactly mirror-symmetric
 * about the vertical centreline — and the discrete equations preserve that
 * symmetry. What comes out is the textbook STARTING PLUME: a straight column
 * under one symmetric mushroom cap, holding that pose forever. It is a correct
 * answer to the problem as posed, and it is not what a fire looks like. Real
 * flames flicker at 1-3 Hz because the buoyant column is unstable, and an
 * instability can only amplify something that is already there.
 *
 * ONE KICK IS ENOUGH, which is the same argument scenes/turbulenceGrid.ts
 * makes about bluff bodies: a plume over a heat source is ABSOLUTELY unstable,
 * it oscillates on its own once started, so this only has to break the tie.
 * That is why it is initial data and not forcing — nothing is re-imposed, and
 * everything you then watch is the flow's own.
 *
 * SMOOTH AND AT BOX SCALE, not per-cell noise, which is the obvious choice and
 * the wrong one twice over: white noise is almost all divergence, so the first
 * projection deletes most of it, and what survives sits at the grid scale
 * where advection smooths it away within a few steps. A stream function gives
 * a divergence-free field the solve leaves alone (testFields.ts's
 * addCurlOfStream), and at this scale it is a slow draught across the room —
 * which is also the honest physical reason a candle in a real room never burns
 * straight.
 *
 * The two modes are one ODD and one EVEN about the centreline. That pairing is
 * the whole trick: either alone is still a symmetry, and the plume would
 * happily keep it.
 */
export function draught(amp = 0.004): Seed {
  return (g, u, v) => {
    const w = g.nx * g.h;
    // Peak speed lands near amp * pi * 1.6, i.e. ~2% of plume speed: far too
    // small to see as a draught, and ~1e5 times larger than the rounding the
    // instability would otherwise have to grow from.
    addCurlOfStream(
      g,
      u,
      v,
      (i, j) => {
        const x = (i * g.h) / w;
        // Vanishes on all four walls whatever the aspect ratio, so the seed
        // has no wall-normal velocity to be projected away.
        const across = Math.sin(2 * Math.PI * x) + 0.6 * Math.sin(3 * Math.PI * x);
        return Math.sin(Math.PI * j * g.h) * across;
      },
      amp,
    );
  };
}
